from __future__ import annotations
import argparse, numpy as np
import corlewis_official_baseline_common as C

def selector(data, m, seed, args):
    np.random.seed(seed)
    from geosketch import gs
    X = data['X_train']
    idx = gs(np.asarray(X, dtype=np.float64), int(m), replace=False)
    return (np.asarray(idx, dtype=int).ravel(), {'official_backend': 'geosketch.gs', 'selection_backend': 'official_geosketch', 'replace': False})

def main():
    ap = C.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    C.run_official_baseline(args, 'geosketch_global', selector, {'method_family': 'geosketch', 'official_package': 'geosketch', 'official_call': 'from geosketch import gs; gs(X_pca, m, replace=False)', 'supervision': 'unsupervised sketch; supervised LR evaluation'})
if __name__ == '__main__':
    main()
