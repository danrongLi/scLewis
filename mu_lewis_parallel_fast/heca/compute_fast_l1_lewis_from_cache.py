from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.model_selection import StratifiedShuffleSplit
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

def load_split(obs_names: np.ndarray, labels: np.ndarray, split_csv: str, seed: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, str]:
    if split_csv:
        p = Path(split_csv).expanduser()
        if p.exists():
            df = pd.read_csv(p)
            if {'obs_name', 'split'}.issubset(df.columns):
                obs_to_idx = pd.Series(np.arange(len(obs_names), dtype=int), index=pd.Index(obs_names.astype(str)))
                idx = df['obs_name'].astype(str).map(obs_to_idx)
                if not idx.isna().any():
                    arr = idx.astype(int).to_numpy()
                    split = df['split'].astype(str).to_numpy()
                    train = arr[split == 'train']
                    test = arr[split == 'test']
                    if len(train) and len(test):
                        return (train, test, str(p.resolve()))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (train, test) = next(splitter.split(np.zeros(len(labels)), labels))
    return (train, test, 'recreated_stratified_split')

def compute_q_exact_gemm(A: np.ndarray, cf, block_size: int) -> np.ndarray:
    d = A.shape[1]
    Minv = cho_solve(cf, np.eye(d, dtype=np.float64), check_finite=False)
    n = A.shape[0]
    q = np.empty(n, dtype=np.float64)
    bs = n if block_size <= 0 else block_size
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        B = A[start:stop]
        BM = B @ Minv
        q[start:stop] = np.einsum('ij,ij->i', BM, B, optimize=True)
    return q

def compute_q_exact_solve(A: np.ndarray, cf, block_size: int) -> np.ndarray:
    n = A.shape[0]
    q = np.empty(n, dtype=np.float64)
    bs = n if block_size <= 0 else block_size
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        B = A[start:stop]
        solved = cho_solve(cf, B.T, check_finite=False).T
        q[start:stop] = np.einsum('ij,ij->i', B, solved, optimize=True)
    return q

def compute_q_jl(A: np.ndarray, cf, jl_dim: int, rng: np.random.Generator, block_size: int) -> np.ndarray:
    if jl_dim <= 0:
        raise ValueError('jl_dim must be positive')
    d = A.shape[1]
    R = rng.choice(np.array([-1.0, 1.0]), size=(d, jl_dim))
    R /= math.sqrt(float(jl_dim))
    L = np.tril(cf[0])
    H = solve_triangular(L.T, R, lower=False, check_finite=False, overwrite_b=False)
    n = A.shape[0]
    q = np.empty(n, dtype=np.float64)
    bs = n if block_size <= 0 else block_size
    for start in range(0, n, bs):
        stop = min(start + bs, n)
        Z = A[start:stop] @ H
        q[start:stop] = np.einsum('ij,ij->i', Z, Z, optimize=True)
    return q

def approximate_l1_lewis_fast(A: np.ndarray, n_iter: int, ridge: float, damping: float, block_size: int, backend: str, jl_dim: int, random_seed: int, early_stop_log_tol: float, min_iter: int) -> Tuple[np.ndarray, pd.DataFrame]:
    A = np.asarray(A, dtype=np.float64)
    (n, d) = A.shape
    w = np.full(n, d / max(n, 1), dtype=np.float64)
    trace_ata = float(np.sum(A * A))
    ridge_scale = ridge * max(trace_ata / max(d, 1), 1e-12)
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, float]] = []
    for it in range(int(n_iter)):
        iter_t0 = time.perf_counter()
        inv_w = 1.0 / np.maximum(w, 1e-12)
        gram_t0 = time.perf_counter()
        M = A.T @ (A * inv_w[:, None])
        M.flat[::d + 1] += ridge_scale
        gram_sec = time.perf_counter() - gram_t0
        chol_t0 = time.perf_counter()
        cf = cho_factor(M, lower=True, check_finite=False)
        chol_sec = time.perf_counter() - chol_t0
        q_t0 = time.perf_counter()
        if backend == 'exact_gemm':
            q = compute_q_exact_gemm(A, cf, block_size)
        elif backend == 'exact_solve':
            q = compute_q_exact_solve(A, cf, block_size)
        elif backend == 'jl':
            q = compute_q_jl(A, cf, jl_dim, rng, block_size)
        else:
            raise ValueError(f'Unknown backend: {backend}')
        q_sec = time.perf_counter() - q_t0
        w_new = np.sqrt(np.maximum(q, 1e-30))
        w_new *= d / max(float(w_new.sum()), 1e-30)
        log_update = np.log(np.maximum(w_new, 1e-300)) - np.log(np.maximum(w, 1e-300))
        max_abs_log_update = float(np.max(np.abs(log_update)))
        median_abs_log_update = float(np.median(np.abs(log_update)))
        w = float(damping) * w + (1.0 - float(damping)) * w_new
        w *= d / max(float(w.sum()), 1e-30)
        total_sec = time.perf_counter() - iter_t0
        rows.append({'iteration': int(it + 1), 'gram_sec': float(gram_sec), 'cholesky_sec': float(chol_sec), 'leverage_sec': float(q_sec), 'iteration_total_sec': float(total_sec), 'max_abs_log_update': max_abs_log_update, 'median_abs_log_update': median_abs_log_update, 'weight_sum': float(w.sum()), 'weight_min': float(w.min()), 'weight_max': float(w.max())})
        print(f'[Lewis] iter={it + 1}/{n_iter} total={total_sec:.3f}s gram={gram_sec:.3f}s leverage={q_sec:.3f}s max|log update|={max_abs_log_update:.4g}', flush=True)
        if early_stop_log_tol > 0 and it + 1 >= int(min_iter) and (max_abs_log_update <= float(early_stop_log_tol)):
            print(f'[Lewis] early stop at iteration {it + 1}: max|log update| <= {early_stop_log_tol}', flush=True)
            break
    w *= d / max(float(w.sum()), 1e-30)
    return (w, pd.DataFrame(rows))

def main() -> None:
    ap = argparse.ArgumentParser(description='Compute fast approximate p=1 Lewis weights from a shared SVD cache. Default 7 undamped iterations follow the Cohen-Peng fixed-point update.')
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--split_csv', default='')
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--n_iter', type=int, default=7)
    ap.add_argument('--ridge', type=float, default=1e-06)
    ap.add_argument('--damping', type=float, default=0.0, help="0 follows the paper's direct fixed-point update; old code used 0.5.")
    ap.add_argument('--block_size', type=int, default=0, help='0 means all rows in one BLAS block')
    ap.add_argument('--backend', choices=['exact_gemm', 'exact_solve', 'jl'], default='exact_gemm')
    ap.add_argument('--jl_dim', type=int, default=32)
    ap.add_argument('--random_seed', type=int, default=1729)
    ap.add_argument('--early_stop_log_tol', type=float, default=0.0)
    ap.add_argument('--min_iter', type=int, default=3)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    cache_t0 = time.perf_counter()
    cache = Path(args.cache_dir).expanduser().resolve()
    X_all = np.load(cache / 'X_svd_all.npy', mmap_mode='r')
    labels = np.load(cache / 'labels.npy', allow_pickle=True).astype(str)
    obs_names = np.load(cache / 'obs_names.npy', allow_pickle=True).astype(str)
    meta = json.loads((cache / 'cache_meta.json').read_text())
    cache_load_sec = time.perf_counter() - cache_t0
    (train_idx, test_idx, split_source) = load_split(obs_names, labels, args.split_csv, args.seed, args.test_size)
    design_t0 = time.perf_counter()
    X_train = np.asarray(X_all[train_idx], dtype=np.float64)
    A = np.empty((len(train_idx), X_train.shape[1] + 1), dtype=np.float64)
    A[:, :-1] = X_train
    A[:, -1] = 1.0
    design_sec = time.perf_counter() - design_t0
    print(f'[config] seed={args.seed} A={A.shape} backend={args.backend} iterations={args.n_iter} damping={args.damping} threads={args.threads}', flush=True)
    lewis_t0 = time.perf_counter()
    if threadpool_limits is None:
        (weights, iter_df) = approximate_l1_lewis_fast(A, args.n_iter, args.ridge, args.damping, args.block_size, args.backend, args.jl_dim, args.random_seed + args.seed, args.early_stop_log_tol, args.min_iter)
    else:
        with threadpool_limits(limits=args.threads):
            (weights, iter_df) = approximate_l1_lewis_fast(A, args.n_iter, args.ridge, args.damping, args.block_size, args.backend, args.jl_dim, args.random_seed + args.seed, args.early_stop_log_tol, args.min_iter)
    lewis_sec = time.perf_counter() - lewis_t0
    n = len(weights)
    d = A.shape[1]
    prob = weights / max(float(weights.sum()), 1e-300)
    effective_n = 1.0 / max(float(np.sum(prob * prob)), 1e-300)
    np.savez_compressed(out / f"lewis_weights_seed{args.seed}_pc{meta['n_pcs']}.npz", lewis_weights=weights, train_idx=np.asarray(train_idx, dtype=np.int64), test_idx=np.asarray(test_idx, dtype=np.int64), train_obs_names=obs_names[train_idx].astype(object), seed=np.asarray([args.seed], dtype=np.int64), n_pcs=np.asarray([meta['n_pcs']], dtype=np.int64))
    iter_df.to_csv(out / 'lewis_iterations.csv', index=False)
    runtime_row = {'dataset_name': meta.get('dataset_name', 'hECA'), 'seed': int(args.seed), 'n_pcs': int(meta['n_pcs']), 'repeat': 1, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_features_lewis': int(d), 'lewis_iter_requested': int(args.n_iter), 'lewis_iter_actual': int(len(iter_df)), 'lewis_ridge': float(args.ridge), 'lewis_damping': float(args.damping), 'lewis_block_size': int(args.block_size), 'lewis_backend': args.backend, 'lewis_jl_dim': int(args.jl_dim if args.backend == 'jl' else 0), 'cache_load_sec': float(cache_load_sec), 'design_build_sec': float(design_sec), 'lewis_runtime_sec': float(lewis_sec), 'lewis_runtime_sec_median_over_repeats': float(lewis_sec), 'lewis_runtime_sec_mean_over_repeats': float(lewis_sec), 'total_runtime_sec': float(time.perf_counter() - total_t0), 'split_source': split_source, 'weights_sum': float(weights.sum()), 'weights_min': float(weights.min()), 'weights_max': float(weights.max()), 'probability_tv_from_uniform': float(0.5 * np.sum(np.abs(prob - 1.0 / max(n, 1)))), 'lewis_effective_n': float(effective_n), 'lewis_effective_n_ratio': float(effective_n / max(n, 1)), 'paper_p1_worst_case_starting_factor_after_T_exact_updates': float(n ** 0.5 ** max(len(iter_df), 1))}
    pd.DataFrame([runtime_row]).to_csv(out / 'lewis_runtime_repeats.csv', index=False)
    pd.DataFrame([runtime_row]).to_csv(out / 'lewis_diagnostics_timed.csv', index=False)
    (out / 'run_config.json').write_text(json.dumps({**vars(args), 'cache_meta': meta}, indent=2, default=str))
    print(json.dumps(runtime_row, indent=2, default=str), flush=True)
    weights_path = out / f"lewis_weights_seed{args.seed}_pc{meta['n_pcs']}.npz"
    print(f'[output] {weights_path}', flush=True)
if __name__ == '__main__':
    main()
