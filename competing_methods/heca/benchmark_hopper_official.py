from __future__ import annotations
import argparse, numpy as np
import corlewis_official_baseline_common as C

def selector(data, m, seed, args):
    np.random.seed(seed)
    from treehopper import hopper
    X = np.asarray(data['X_train'], dtype=np.float64)
    h = hopper(X)
    sketch = h.hop(int(m))
    path = getattr(h, 'path', None)
    idx = np.asarray(path[:int(m)] if path is not None and len(path) >= int(m) else sketch, dtype=int).ravel()
    return (idx, {'official_backend': 'treehopper.hopper', 'selection_backend': 'official_hopper_exact'})

def main():
    ap = C.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    C.run_official_baseline(args, 'hopper_official', selector, {'method_family': 'hopper', 'official_package': 'bendemeo/hopper treehopper package', 'official_call': 'hopper(X_pca); hopper.hop(m)', 'supervision': 'unsupervised sketch; supervised LR evaluation'})
if __name__ == '__main__':
    main()
