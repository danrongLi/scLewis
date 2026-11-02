import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'internal'))


def main():
    parser = argparse.ArgumentParser(description='Original prediction-path Lewis weights with 20 iterations and damping 0.5.')
    parser.add_argument('--cache_dir', default=str(ROOT / 'caches' / 'lewis_svd'))
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--ridge', type=float, default=1e-6)
    parser.add_argument('--block_size', type=int, default=50000)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(args.threads)
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    from threadpoolctl import threadpool_limits
    from corlewis_ovr_theory_benchmark import approximate_l1_lewis_weights
    from sclewis_selected_methods_benchmark import make_design_with_intercept
    cache = Path(args.cache_dir)
    X = np.load(cache / 'X_svd_all.npy', mmap_mode='r')
    labels = np.load(cache / 'labels.npy', allow_pickle=True).astype(str)
    names = np.load(cache / 'obs_names.npy', allow_pickle=True).astype(str)
    meta = json.loads((cache / 'cache_meta.json').read_text())
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train, test = next(splitter.split(np.zeros(len(labels)), labels))
    A = make_design_with_intercept(X[train])
    start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        weights = approximate_l1_lewis_weights(A, n_iter=args.n_iter, ridge=args.ridge, damping=args.damping, block_size=args.block_size, verbose=False)
    elapsed = time.perf_counter() - start
    out = Path(args.output_dir) if args.output_dir else ROOT / 'results' / 'slow_lewis' / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'lewis_weights_seed{args.seed}_pc{meta["n_pcs"]}.npz', lewis_weights=weights, train_idx=train, test_idx=test, train_obs_names=names[train], seed=np.asarray([args.seed]), n_pcs=np.asarray([meta['n_pcs']]))
    pd.DataFrame([{'seed': args.seed, 'n_pcs': meta['n_pcs'], 'lewis_iter_requested': args.n_iter, 'lewis_damping': args.damping, 'lewis_runtime_sec': elapsed, 'split_source': 'recreated_stratified_split_original_prediction_order'}]).to_csv(out / 'slow_lewis_runtime.csv', index=False)
    (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
