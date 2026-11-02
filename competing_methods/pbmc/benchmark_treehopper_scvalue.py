from __future__ import annotations
import argparse, sys, numpy as np
from pathlib import Path
import corlewis_official_baseline_common as C

def add_args(ap):
    ap.add_argument('--treehopper_max_partition_size', type=int, default=1000)
    ap.add_argument('--scvalue_repo', default='external/scvalue')
    return ap

def selector(data, m, seed, args):
    np.random.seed(seed)
    repo = Path(args.scvalue_repo)
    if repo.exists():
        sys.path.insert(0, str(repo / 'reproducibility'))
    try:
        from baselines.treehopper.hoppers import treehopper, PCATreePartition
        backend = 'scvalue_reproducibility_baselines.treehopper.hoppers'
    except Exception:
        from treehopper import treehopper, PCATreePartition
        backend = 'treehopper package'
    X = np.asarray(data['X_train'], dtype=np.float64)
    th = treehopper(X, partition=PCATreePartition, max_partition_size=int(args.treehopper_max_partition_size))
    sketch = th.hop(int(m))
    path = getattr(th, 'path', None)
    idx = np.asarray(path[:int(m)] if path is not None and len(path) >= int(m) else sketch, dtype=int).ravel()
    return (idx, {'official_backend': backend, 'selection_backend': 'official_treehopper', 'max_partition_size': args.treehopper_max_partition_size})

def main():
    ap = add_args(C.add_common_args(argparse.ArgumentParser()))
    args = ap.parse_args()
    C.run_official_baseline(args, 'treehopper_scvalue', selector, {'method_family': 'treehopper', 'official_package': 'scValue reproducibility vendored TreeHopper or treehopper', 'official_call': 'treehopper(X_pca, partition=PCATreePartition, max_partition_size=1000); th.hop(m)', 'supervision': 'unsupervised sketch; supervised LR evaluation'})
if __name__ == '__main__':
    main()
