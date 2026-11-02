from __future__ import annotations
import argparse, numpy as np
import corlewis_official_baseline_common as C

def add_args(ap):
    ap.add_argument('--scvalue_n_trees', type=int, default=100)
    ap.add_argument('--scvalue_strategy', default='FB')
    return ap

def selector(data, m, seed, args):
    from scvalue import SCValue
    a = data['adata_train_pca'].copy()
    scv = SCValue(adata=a, sketch_size=int(m), use_rep='X_pca', cell_type_key=args.label_key, n_trees=int(args.scvalue_n_trees), type_weighting='inv_cv', strategy=args.scvalue_strategy, write_dv=False, seed=int(seed))
    sub = scv.value()
    obs = np.asarray(sub.obs_names if sub is not None else scv.adata_sub.obs_names).astype(str)
    idx = np.asarray([int(x) for x in obs], dtype=int)
    return (idx, {'official_backend': 'scvalue.SCValue', 'selection_backend': 'official_scvalue', 'type_weighting': 'inv_cv', 'strategy': args.scvalue_strategy, 'n_trees': args.scvalue_n_trees})

def main():
    ap = add_args(C.add_common_args(argparse.ArgumentParser()))
    args = ap.parse_args()
    C.run_official_baseline(args, 'scvalue_default', selector, {'method_family': 'scvalue', 'official_package': 'scvalue', 'official_call': "SCValue(..., type_weighting='inv_cv', strategy='FB', n_trees=100)", 'supervision': 'supervised sketch uses cell_type labels; supervised LR evaluation'})
if __name__ == '__main__':
    main()
