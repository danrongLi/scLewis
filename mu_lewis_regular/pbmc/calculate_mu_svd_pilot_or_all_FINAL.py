from __future__ import annotations
import argparse
import gc
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from validate_pbmc_branch import require_preprocessed, file_sha256, array_sha256
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
try:
    import anndata as ad
except Exception as exc:
    raise RuntimeError('This script requires anndata to read .h5ad files.') from exc
try:
    import psutil
except Exception:
    psutil = None

def now() -> float:
    return time.time()

def rss_mb() -> float:
    if psutil is None:
        return float('nan')
    return float(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

def as_bool01(x: object) -> bool:
    s = str(x).strip().lower()
    return s in {'1', 'true', 't', 'yes', 'y', 'on'}

def parse_csv_floats(x: str) -> List[float]:
    return [float(v.strip()) for v in str(x).split(',') if v.strip()]

def parse_csv_strings(x: str) -> List[str]:
    return [v.strip() for v in str(x).split(',') if v.strip()]

def append_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False)

def read_h5ad(path: str):
    if hasattr(ad, 'read_h5ad'):
        return ad.read_h5ad(path)
    return ad.read(path)

def filter_labels(adata, label_key: str, min_class_count: int):
    if label_key not in adata.obs:
        raise KeyError(f'label_key={label_key!r} not found in adata.obs')
    y = adata.obs[label_key]
    mask = y.notna().to_numpy()
    y_str = y.astype(str).to_numpy()
    bad = np.isin(y_str, ['', 'nan', 'NaN', 'None', 'NA', 'null'])
    mask = mask & ~bad
    adata = adata[mask].copy()
    y_str = adata.obs[label_key].astype(str).to_numpy()
    counts = pd.Series(y_str).value_counts()
    keep_classes = set(counts[counts >= int(min_class_count)].index.astype(str))
    keep_mask = np.array([str(v) in keep_classes for v in y_str], dtype=bool)
    adata = adata[keep_mask].copy()
    y_str = adata.obs[label_key].astype(str).to_numpy()
    return (adata, y_str)

def split_train_test(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    if test_size <= 0:
        return (np.arange(n), np.array([], dtype=int))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (train_idx, test_idx) = next(splitter.split(np.zeros(n), y))
    return (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))

def scale_mu_features(X_train, mu_centering: str):
    if mu_centering not in {'centered', 'uncentered'}:
        raise ValueError('mu_centering must be explicitly centered or uncentered')
    X = X_train.astype(np.float64, copy=True)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    if mu_centering == 'centered':
        X_scaled = X_scaled.toarray() if sparse.issparse(X_scaled) else np.asarray(X_scaled)
        X_scaled -= np.asarray(scaler.mean_ / scaler.scale_)[None, :]
    return (X_scaled, scaler)

def fit_svd_embedding(X_train, n_pcs: int, seed: int, mu_centering: str):
    t0 = now()
    (X_scaled, scaler) = scale_mu_features(X_train, mu_centering)
    n_eff = int(min(n_pcs, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1))
    if n_eff <= 0:
        raise ValueError(f'Invalid effective SVD dimension {n_eff}')
    svd = TruncatedSVD(n_components=n_eff, algorithm='randomized', random_state=seed)
    A_train = svd.fit_transform(X_scaled).astype(np.float32, copy=False)
    info = {'embedding': 'svd', 'embedding_note': 'training_variance_scale_then_' + mu_centering + '_TruncatedSVD', 'mu_centering': mu_centering, 'input_state': 'preprocessed', 'normalization_backend': 'preprocessed', 'normalization_applied': False, 'log1p_applied': False, 'hvg_selection_applied': False, 'svd_embedding_storage_dtype': 'float32', 'lp_design_dtype': 'float64', 'variance_scaling_dtype': 'float64', 'n_pcs': n_eff, 'n_pcs_requested': int(n_pcs), 'n_pcs_effective': n_eff, 'svd_explained_variance_ratio_sum': float(np.sum(svd.explained_variance_ratio_)), 'embedding_runtime_sec': float(now() - t0)}
    return (A_train, info)

def prepare_mu_design(adata, args):
    require_preprocessed(adata, args.label_key, args.input_state)
    y_all = adata.obs[args.label_key].astype(str).to_numpy()
    if pd.Series(y_all).value_counts().min() < args.min_class_count:
        raise ValueError('Input contains classes below min_class_count; refusing to silently change benchmark cells')
    (train_idx, test_idx) = split_train_test(y_all, args.test_size, args.seed)
    y_train = y_all[train_idx]
    genes = list(map(str, adata.var_names))
    (A_train, embed_info) = fit_svd_embedding(adata.X[train_idx], args.n_pcs, args.seed, args.mu_centering)
    mu_seed = args.seed + 9973 * max(int(embed_info['n_pcs_effective']), 1)
    mu_idx = sample_mu_rows(y_train, args.mu_rows, seed=mu_seed, mode=args.mu_row_sampling, min_pos=args.mu_min_pos)
    A_sub = np.asarray(A_train[mu_idx], dtype=np.float64)
    if args.add_intercept:
        A_sub = add_intercept_column(A_sub)
    return {'A_sub': A_sub, 'y_sub': y_train[mu_idx], 'y_all': y_all, 'y_train': y_train, 'train_idx': train_idx, 'test_idx': test_idx, 'mu_idx': mu_idx, 'genes': genes, 'embed_info': embed_info, 'obs_names': adata.obs_names.astype(str).to_numpy()}

def sample_mu_rows(y_train: np.ndarray, max_rows: int, seed: int, mode: str, min_pos: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_train)
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=int)
    (classes, counts) = np.unique(y_train, return_counts=True)
    if mode == 'uniform':
        return np.sort(rng.choice(n, size=max_rows, replace=False))
    if mode == 'balanced':
        per_class = max(1, max_rows // max(len(classes), 1))
        selected: List[np.ndarray] = []
        for cls in classes:
            idx = np.where(y_train == cls)[0]
            take = min(per_class, len(idx))
            if take > 0:
                selected.append(rng.choice(idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=int)
        if len(out) < max_rows:
            remaining = np.setdiff1d(np.arange(n), out, assume_unique=False)
            extra = rng.choice(remaining, size=min(max_rows - len(out), len(remaining)), replace=False)
            out = np.concatenate([out, extra])
        if len(out) > max_rows:
            out = rng.choice(out, size=max_rows, replace=False)
        return np.sort(out.astype(int))
    if mode == 'stratified':
        selected = []
        for (cls, count) in zip(classes, counts):
            cls_idx = np.where(y_train == cls)[0]
            take = max(1, int(round(max_rows * int(count) / n)))
            take = min(take, len(cls_idx))
            selected.append(rng.choice(cls_idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=int)
        if len(out) > max_rows:
            out = rng.choice(out, size=max_rows, replace=False)
        elif len(out) < max_rows:
            remaining = np.setdiff1d(np.arange(n), out, assume_unique=False)
            extra = rng.choice(remaining, size=min(max_rows - len(out), len(remaining)), replace=False)
            out = np.concatenate([out, extra])
        return np.sort(out.astype(int))
    if mode == 'stratified_minpos':
        selected_parts: List[np.ndarray] = []
        eligible_classes = []
        for (cls, count) in zip(classes, counts):
            if int(count) >= min_pos:
                eligible_classes.append(cls)
                cls_idx = np.where(y_train == cls)[0]
                selected_parts.append(rng.choice(cls_idx, size=min_pos, replace=False))
        selected = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
        if len(selected) > max_rows:
            warnings.warn(f'min_pos reserve needs {len(selected)} rows > max_rows={max_rows}; falling back to balanced sampling. Some classes may be skipped.')
            return sample_mu_rows(y_train, max_rows, seed=seed, mode='balanced', min_pos=min_pos)
        remaining = np.setdiff1d(np.arange(n), selected, assume_unique=False)
        extra_n = min(max_rows - len(selected), len(remaining))
        if extra_n > 0:
            extra = rng.choice(remaining, size=extra_n, replace=False)
            selected = np.concatenate([selected, extra])
        return np.sort(selected.astype(int))
    raise ValueError(f'Unknown --mu_row_sampling {mode!r}')

def select_mu_classes(train_counts: pd.Series, mu_classes_arg: str) -> List[str]:
    all_classes = list(train_counts.index.astype(str))
    arg = str(mu_classes_arg).strip()
    if arg == '' or arg.lower() == 'all':
        return all_classes
    low = arg.lower()
    if low.startswith('top:'):
        n = int(arg.split(':', 1)[1])
        return all_classes[:max(0, n)]
    if low.startswith('top') and low[3:].isdigit():
        n = int(low[3:])
        return all_classes[:max(0, n)]
    if arg.isdigit():
        n = int(arg)
        return all_classes[:max(0, n)]
    requested = parse_csv_strings(arg)
    available = set(all_classes)
    missing = [c for c in requested if c not in available]
    if missing:
        suggestions = []
        for miss in missing:
            ml = miss.lower()
            hits = [c for c in all_classes if ml in c.lower() or c.lower() in ml]
            suggestions.append({'requested': miss, 'possible_matches': hits[:10]})
        raise ValueError(f'Some requested --mu_classes labels were not found exactly. Missing={missing}. Suggestions={suggestions}. First available labels={all_classes[:20]}')
    return requested

def add_intercept_column(A: np.ndarray) -> np.ndarray:
    return np.hstack([A, np.ones((A.shape[0], 1), dtype=A.dtype)])

def compute_mu_lp_scipy(A: np.ndarray, y_pm: np.ndarray, time_limit: Optional[float]) -> Dict[str, object]:
    from scipy.optimize import linprog
    A = np.asarray(A, dtype=np.float64)
    y_pm = np.asarray(y_pm, dtype=np.float64)
    B = A * y_pm[:, None]
    (n, d) = B.shape
    c = np.zeros(d + 2 * n, dtype=np.float64)
    c[d:d + n] = -1.0
    A_eq = sparse.hstack([sparse.csr_matrix(B), -sparse.eye(n, format='csr'), sparse.eye(n, format='csr')], format='csr')
    b_eq = np.zeros(n, dtype=np.float64)
    A_ub = sparse.csr_matrix((np.ones(n, dtype=np.float64), (np.zeros(n, dtype=int), np.arange(d + n, d + 2 * n))), shape=(1, d + 2 * n))
    b_ub = np.array([1.0], dtype=np.float64)
    bounds = [(None, None)] * d + [(0, None)] * n + [(0, None)] * n
    options: Dict[str, object] = {}
    if time_limit is not None and time_limit > 0:
        options['time_limit'] = float(time_limit)
    t0 = now()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options=options)
    elapsed = now() - t0
    out: Dict[str, object] = {'mu': np.nan, 'lp_status': int(res.status), 'lp_success': bool(res.success), 'lp_message': str(res.message), 'lp_runtime_sec': float(elapsed), 'lp_objective': np.nan, 'z_neg_l1_constraint': 1.0}
    if res.status == 3:
        out.update({'mu': float('inf'), 'lp_objective': float('inf')})
    elif res.success:
        mu = -float(res.fun)
        out.update({'mu': mu, 'lp_objective': mu})
        try:
            x = np.asarray(res.x, dtype=np.float64)
            out['z_pos_l1'] = float(np.sum(x[d:d + n]))
            out['z_neg_l1'] = float(np.sum(x[d + n:d + 2 * n]))
        except Exception:
            pass
    return out

def compute_mu_lp_gurobi(A: np.ndarray, y_pm: np.ndarray, time_limit: Optional[float], threads: int, mem_limit_gb: float, output_flag: bool, env) -> Dict[str, object]:
    t0 = now()
    out: Dict[str, object] = {'mu': np.nan, 'lp_status': np.nan, 'lp_success': False, 'lp_message': '', 'lp_runtime_sec': np.nan, 'lp_objective': np.nan, 'z_neg_l1_constraint': 1.0}
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        out.update({'lp_message': f'gurobipy not available: {exc}', 'lp_runtime_sec': now() - t0})
        return out
    model = None
    try:
        A = np.asarray(A, dtype=np.float64)
        y_pm = np.asarray(y_pm, dtype=np.float64)
        B = sparse.csr_matrix(A * y_pm[:, None])
        (n, d) = B.shape
        model = gp.Model('mu_direct_lp', env=env)
        model.Params.OutputFlag = 1 if output_flag else 0
        if time_limit is not None and time_limit > 0:
            model.Params.TimeLimit = float(time_limit)
        if threads and threads > 0:
            model.Params.Threads = int(threads)
        if mem_limit_gb and mem_limit_gb > 0:
            model.Params.MemLimit = float(mem_limit_gb)
        beta = model.addMVar(d, lb=-GRB.INFINITY, name='beta')
        z_pos = model.addMVar(n, lb=0.0, name='z_pos')
        z_neg = model.addMVar(n, lb=0.0, name='z_neg')
        model.addConstr(B @ beta - z_pos + z_neg == np.zeros(n), name='signed_margin_decomp')
        model.addConstr(z_neg.sum() <= 1.0, name='negative_l1_le_1')
        model.setObjective(z_pos.sum(), GRB.MAXIMIZE)
        model.optimize()
        out['lp_status'] = int(model.Status)
        out['lp_runtime_sec'] = float(now() - t0)
        out['lp_message'] = str(model.Status)
        if model.Status == GRB.UNBOUNDED:
            out.update({'mu': float('inf'), 'lp_objective': float('inf'), 'lp_success': False})
        elif model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT} and model.SolCount > 0:
            mu = float(model.ObjVal)
            out.update({'mu': mu, 'lp_objective': mu, 'lp_success': bool(model.Status == GRB.OPTIMAL), 'z_pos_l1': float(np.sum(z_pos.X)), 'z_neg_l1': float(np.sum(z_neg.X))})
    except Exception as exc:
        out.update({'lp_message': f'gurobi mu error: {exc}', 'lp_runtime_sec': now() - t0})
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
    return out

def compute_mu_lp(A: np.ndarray, y_pm: np.ndarray, args: argparse.Namespace, env=None) -> Dict[str, object]:
    if args.mu_solver == 'scipy':
        out = compute_mu_lp_scipy(A, y_pm, time_limit=args.mu_lp_time_limit)
    elif args.mu_solver == 'gurobi':
        if env is None:
            raise ValueError('Gurobi requires the explicit shared environment for this seed')
        out = compute_mu_lp_gurobi(A, y_pm, time_limit=args.mu_lp_time_limit, threads=args.gurobi_threads, mem_limit_gb=args.gurobi_mem_limit_gb, output_flag=args.gurobi_output, env=env)
    else:
        raise ValueError(f'Unknown --mu_solver {args.mu_solver!r}')
    out['mu_solver'] = args.mu_solver
    return out

def finite_mu_array(df: pd.DataFrame) -> np.ndarray:
    vals = pd.to_numeric(df['mu'], errors='coerce').to_numpy(dtype=np.float64)
    return vals[np.isfinite(vals)]

def summarize_mu(mu_df: pd.DataFrame, skipped_df: pd.DataFrame) -> pd.DataFrame:
    if mu_df.empty:
        return pd.DataFrame()
    group_cols = ['dataset_name', 'seed', 'embedding', 'embedding_note', 'n_pcs_requested', 'n_pcs_effective', 'mu_solver', 'mu_rows_requested', 'mu_rows_actual', 'mu_row_sampling', 'intercept_added']
    rows: List[Dict[str, object]] = []
    for (key, g) in mu_df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        finite = finite_mu_array(g)
        mu_values = pd.to_numeric(g['mu'], errors='coerce').to_numpy(dtype=np.float64)
        succeeded = g['lp_success'].astype(str).str.lower().isin(['true', '1']).to_numpy() & np.isfinite(mu_values) & (mu_values > 0)
        skip_count = int(len(skipped_df)) if skipped_df is not None and (not skipped_df.empty) else 0
        rows.append({**base, 'n_classes_mu_attempted': int(len(g)), 'n_classes_mu_success': int(succeeded.sum()), 'n_classes_mu_failed': int(len(g) - succeeded.sum()), 'n_classes_skipped': skip_count, 'n_features_mu': int(pd.to_numeric(g['n_features_mu'], errors='coerce').iloc[0]), 'n_train': int(pd.to_numeric(g['n_train'], errors='coerce').iloc[0]), 'mean_mu': float(np.mean(finite)) if len(finite) else np.nan, 'median_mu': float(np.median(finite)) if len(finite) else np.nan, 'q90_mu': float(np.quantile(finite, 0.9)) if len(finite) else np.nan, 'q95_mu': float(np.quantile(finite, 0.95)) if len(finite) else np.nan, 'max_mu': float(np.max(finite)) if len(finite) else np.nan, 'mean_mu_squared': float(np.mean(finite ** 2)) if len(finite) else np.nan})
    return pd.DataFrame(rows)

def mu_statistic(summary_row: pd.Series, stat_name: str) -> float:
    stat_name = str(stat_name).lower()
    col_map = {'mean': 'mean_mu', 'median': 'median_mu', 'q90': 'q90_mu', 'q95': 'q95_mu', 'max': 'max_mu'}
    if stat_name not in col_map:
        raise ValueError(f'Unknown mu stat {stat_name!r}; use mean,median,q90,q95,max')
    return float(summary_row[col_map[stat_name]])

def make_calibrated_sizes(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    stats = parse_csv_strings(args.mu_calibrated_stats)
    epsilons = parse_csv_floats(args.mu_calibrated_epsilons)
    alphas = parse_csv_floats(args.mu_calibrated_alphas)
    for (_, r) in summary.iterrows():
        d_eff = int(r['n_features_mu'])
        n_train = int(r['n_train'])
        max_cells = n_train
        if args.mu_calibrated_max_ratio > 0:
            max_cells = min(max_cells, max(1, int(round(args.mu_calibrated_max_ratio * n_train))))
        min_cells = max(1, int(args.mu_calibrated_min_cells))
        for stat in stats:
            mu_val = mu_statistic(r, stat)
            if not np.isfinite(mu_val):
                continue
            for eps in epsilons:
                for alpha in alphas:
                    proxy = float(alpha) * float(d_eff) * float(mu_val) ** 2 / float(eps) ** 2
                    m = int(math.ceil(proxy))
                    m = max(min_cells, min(max_cells, m))
                    rows.append({**{c: r[c] for c in summary.columns if c not in {'mean_mu', 'median_mu', 'q90_mu', 'q95_mu', 'max_mu', 'mean_mu_squared'}}, 'mu_calibration_stat': stat, 'mu_calibration_value': float(mu_val), 'mu_calibration_epsilon': float(eps), 'mu_calibration_alpha': float(alpha), 'mu_calibration_proxy_size': float(proxy), 'coreset_size_requested': int(m), 'coreset_ratio': float(m / max(n_train, 1)), 'coreset_size_capped': bool(m != int(math.ceil(proxy)))})
    return pd.DataFrame(rows)

def run(args: argparse.Namespace) -> None:
    args.dataset_sha256 = file_sha256(args.dataset)
    args.normalization_backend = 'preprocessed'
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mu_lp_time_limit <= 0:
        args.mu_lp_time_limit = None
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    t_all = now()
    print(f'[load] {args.dataset}', flush=True)
    adata = read_h5ad(args.dataset)
    prepared = prepare_mu_design(adata, args)
    (A_sub, y_sub) = (prepared['A_sub'], prepared['y_sub'])
    (y_all, y_train) = (prepared['y_all'], prepared['y_train'])
    (train_idx, test_idx, mu_idx) = (prepared['train_idx'], prepared['test_idx'], prepared['mu_idx'])
    (genes, embed_info) = (prepared['genes'], prepared['embed_info'])
    n_test = len(test_idx)
    mu_design_sha256 = array_sha256(A_sub)
    np.savez_compressed(out_dir / 'mu_design_provenance.npz', A_sub=A_sub, y_sub=np.asarray(y_sub, str), train_idx=train_idx, test_idx=test_idx, mu_idx=mu_idx, obs_names=np.asarray(prepared['obs_names'], str), genes=np.asarray(genes, str))
    del adata
    gc.collect()
    try:
        range_basis_dim = int(np.linalg.matrix_rank(A_sub))
    except Exception:
        range_basis_dim = int(A_sub.shape[1])
    subset_counts = pd.Series(y_sub).value_counts()
    subset_rows = []
    train_counts = pd.Series(y_train).value_counts()
    for (cls, cnt) in subset_counts.items():
        subset_rows.append({'dataset_name': args.dataset_name, 'seed': args.seed, **embed_info, 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(mu_idx)), 'row_sampling': args.mu_row_sampling, 'mu_row_sampling': args.mu_row_sampling, 'class_label': str(cls), 'class_count_in_mu_subset': int(cnt), 'class_fraction_in_mu_subset': float(cnt / max(len(mu_idx), 1)), 'class_count_in_train': int(train_counts.get(cls, 0)), 'class_fraction_in_train': float(train_counts.get(cls, 0) / max(len(y_train), 1))})
    pd.DataFrame(subset_rows).to_csv(out_dir / 'mu_subset_counts.csv', index=False)
    prep_row = {'dataset_name': args.dataset_name, 'seed': args.seed, **embed_info, 'n_cells_after_label_filter': int(len(y_all)), 'n_train': int(len(y_train)), 'n_test': int(n_test), 'n_classes_train': int(pd.Series(y_train).nunique()), 'mu_classes_arg': str(args.mu_classes), 'n_hvgs': int(len(genes)), 'target_sum': None, 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(mu_idx)), 'row_sampling': args.mu_row_sampling, 'mu_row_sampling': args.mu_row_sampling, 'intercept_added': bool(args.add_intercept), 'n_features_mu': int(A_sub.shape[1]), 'memory_rss_mb': rss_mb()}
    pd.DataFrame([prep_row]).to_csv(out_dir / 'preprocessing_summary.csv', index=False)
    preprocessing_runtime_sec = now() - t_all
    all_train_classes = list(train_counts.index.astype(str))
    mu_classes = select_mu_classes(train_counts, args.mu_classes)
    pd.DataFrame({'selected_class_label': mu_classes, 'selected_class_rank_by_train_count': [all_train_classes.index(c) + 1 for c in mu_classes], 'selected_class_train_count': [int(train_counts.get(c, 0)) for c in mu_classes]}).to_csv(out_dir / 'mu_selected_classes.csv', index=False)
    print(f'[mu] computing one-vs-rest mu for {len(mu_classes)} selected classes out of {len(all_train_classes)} train classes; mu_classes={args.mu_classes!r}; subset rows={len(mu_idx):,}; features={A_sub.shape[1]}; solver={args.mu_solver}; sampling={args.mu_row_sampling}', flush=True)
    by_class_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    env = None
    session_info = {'session_policy': 'not_applicable_scipy', 'attempts': 0, 'license_wait_sec': 0.0, 'environment_startup_sec': 0.0}
    if args.mu_solver == 'gurobi':
        import gurobipy as gp
        from pbmc_gurobi_session import start_gurobi_environment
        (env, session_info) = start_gurobi_environment(gp)
    try:
        (out_dir / 'gurobi_session.json').write_text(json.dumps(session_info, indent=2) + '\n')
        for (class_i, cls) in enumerate(mu_classes, start=1):
            y_pm = np.where(y_sub == cls, 1.0, -1.0)
            pos = int(np.sum(y_pm == 1))
            neg = int(np.sum(y_pm == -1))
            base = {'dataset_name': args.dataset_name, 'seed': args.seed, **embed_info, 'mu_solver': args.mu_solver, 'mu_source': 'calculate_mu_svd_pilot_or_all_final', 'gurobi_session_policy': session_info['session_policy'], 'lp_formulation': 'direct_ratio_lp_max_pos_l1_given_neg_l1_le_1', 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(mu_idx)), 'row_sampling': args.mu_row_sampling, 'mu_row_sampling': args.mu_row_sampling, 'intercept_added': bool(args.add_intercept), 'class_label': str(cls), 'positive_label': str(cls), 'pos_count_mu_subset': pos, 'neg_count_mu_subset': neg, 'pos_fraction_mu_subset': float(pos / max(len(mu_idx), 1)), 'pos_count_train': int(train_counts.get(cls, 0)), 'pos_fraction_train': float(train_counts.get(cls, 0) / max(len(y_train), 1)), 'n_features_mu': int(A_sub.shape[1]), 'range_basis_dim': int(range_basis_dim), 'n_train': int(len(y_train))}
            if pos < args.mu_min_pos or neg < args.mu_min_neg:
                skipped_rows.append({**base, 'skip_reason': f'pos={pos}, neg={neg}, need pos>={args.mu_min_pos}, neg>={args.mu_min_neg}'})
                continue
            print(f'[mu] {class_i}/{len(mu_classes)} {cls!r}: pos={pos}, neg={neg}', flush=True)
            result = compute_mu_lp(A_sub, y_pm, args, env=env)
            mu_val = result.get('mu', np.nan)
            if isinstance(mu_val, float) and math.isinf(mu_val):
                mu_squared = float('inf')
            elif pd.isna(mu_val):
                mu_squared = np.nan
            else:
                mu_squared = float(mu_val) ** 2
            row = {**base, 'mu': mu_val, 'mu_squared': mu_squared, 'lp_status': result.get('lp_status', np.nan), 'lp_success': result.get('lp_success', False), 'lp_message': result.get('lp_message', ''), 'lp_runtime_sec': result.get('lp_runtime_sec', np.nan), 'lp_objective': result.get('lp_objective', np.nan), 'z_pos_l1': result.get('z_pos_l1', np.nan), 'z_neg_l1': result.get('z_neg_l1', np.nan), 'z_neg_l1_constraint': result.get('z_neg_l1_constraint', np.nan), 'peak_rss_mb': rss_mb(), 'memory_rss_mb': rss_mb()}
            by_class_rows.append(row)
            append_rows_csv(out_dir / 'mu_by_class.csv', [row])
            gc.collect()
    finally:
        if env is not None:
            env.dispose()
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(out_dir / 'mu_skipped_classes.csv', index=False)
    else:
        pd.DataFrame(columns=['class_label', 'skip_reason']).to_csv(out_dir / 'mu_skipped_classes.csv', index=False)
    mu_df = pd.DataFrame(by_class_rows)
    mu_df['dataset_sha256'] = args.dataset_sha256
    mu_df['mu_design_sha256'] = mu_design_sha256
    mu_df['input_state'] = args.input_state
    mu_df['normalization_backend'] = 'preprocessed'
    mu_df['mu_centering'] = args.mu_centering
    skipped_df = pd.DataFrame(skipped_rows)
    if not mu_df.empty:
        mu_df.to_csv(out_dir / 'mu_by_class.csv', index=False)
        summary = summarize_mu(mu_df, skipped_df)
        summary.to_csv(out_dir / 'mu_summary.csv', index=False)
        calibrated = make_calibrated_sizes(summary, args)
        calibrated.to_csv(out_dir / 'mu_calibrated_coreset_sizes.csv', index=False)
    else:
        pd.DataFrame().to_csv(out_dir / 'mu_by_class.csv', index=False)
        pd.DataFrame().to_csv(out_dir / 'mu_summary.csv', index=False)
        pd.DataFrame().to_csv(out_dir / 'mu_calibrated_coreset_sizes.csv', index=False)
    lp_seconds = float(pd.to_numeric(mu_df.get('lp_runtime_sec', pd.Series(dtype=float)), errors='coerce').sum())
    pd.DataFrame([{'dataset_name': args.dataset_name, 'seed': args.seed, 'mu_centering': args.mu_centering, 'total_runtime_sec': now() - t_all - session_info['license_wait_sec'], 'sum_per_class_lp_runtime_sec': lp_seconds, 'gurobi_session_policy': session_info['session_policy'], 'license_wait_sec': session_info['license_wait_sec'], 'environment_startup_sec': session_info['environment_startup_sec'], 'preprocessing_runtime_sec': preprocessing_runtime_sec}]).to_csv(out_dir / 'mu_walltime.csv', index=False)
    print('\n[done] wrote:', flush=True)
    for name in ['mu_by_class.csv', 'mu_summary.csv', 'mu_calibrated_coreset_sizes.csv', 'mu_subset_counts.csv', 'mu_selected_classes.csv', 'mu_skipped_classes.csv', 'preprocessing_summary.csv', 'run_config.json']:
        print(f'  {out_dir / name}', flush=True)
    print(f'[done] total runtime: {(now() - t_all) / 60:.2f} min', flush=True)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Compute pilot/top-k/all one-vs-rest mu after train-only SVD embedding.')
    p.add_argument('--dataset', required=True, help='Input .h5ad file')
    p.add_argument('--label_key', required=True, help='adata.obs column with cell-type labels')
    p.add_argument('--dataset_name', default='dataset')
    p.add_argument('--input-state', '--input_state', dest='input_state', choices=['preprocessed'], required=True)
    p.add_argument('--mu-centering', '--mu_centering', dest='mu_centering', choices=['centered', 'uncentered'], required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--min_class_count', type=int, default=20)
    p.add_argument('--target_sum', type=float, default=10000.0, help='Unused compatibility option; supplied expression is preserved')
    p.add_argument('--n_hvgs', type=int, default=2000, help='Unused compatibility option; all supplied genes are retained')
    p.add_argument('--n_pcs', type=int, default=100)
    p.add_argument('--mu_classes', default='all', help='Classes to compute: all, topK/top:K, integer K, one exact label, or comma-separated exact labels')
    p.add_argument('--mu_rows', type=int, default=5000, help='Rows used for the mu LP subset; 0 or >=n_train uses all train rows')
    p.add_argument('--mu_row_sampling', choices=['stratified_minpos', 'uniform', 'stratified', 'balanced'], default='stratified_minpos', help='Use stratified_minpos for all-class mu; uniform matches the old run but may skip rare classes.')
    p.add_argument('--mu_min_pos', type=int, default=5)
    p.add_argument('--mu_min_neg', type=int, default=5)
    p.add_argument('--add_intercept', action='store_true', help='Append a constant column to the SVD embedding before the mu LP')
    p.add_argument('--mu_solver', choices=['gurobi', 'scipy'], default='gurobi')
    p.add_argument('--mu_lp_time_limit', type=float, default=0.0, help='Seconds per LP; 0 means no explicit limit')
    p.add_argument('--gurobi_threads', type=int, default=1)
    p.add_argument('--gurobi_mem_limit_gb', type=float, default=100.0)
    p.add_argument('--gurobi_output', action='store_true')
    p.add_argument('--mu_calibrated_stats', default='q95,max', help='mean,median,q90,q95,max')
    p.add_argument('--mu_calibrated_epsilons', default='0.5,0.3,0.2,0.1')
    p.add_argument('--mu_calibrated_alphas', default='0.25,0.5,1,2,4')
    p.add_argument('--mu_calibrated_min_cells', type=int, default=100)
    p.add_argument('--mu_calibrated_max_ratio', type=float, default=0.5)
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)
if __name__ == '__main__':
    main()
