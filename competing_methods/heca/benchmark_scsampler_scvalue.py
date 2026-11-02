from __future__ import annotations
import argparse, numpy as np
import corlewis_official_baseline_common as C

def add_args(ap):
    ap.add_argument('--scsampler_random_split', type=int, default=4)
    return ap

def selector(data, m, seed, args):
    np.random.seed(seed)
    from scsampler import scsampler
    a = data['adata_train_pca'].copy()
    sub = scsampler(a, n_obs=int(m), copy=True, random_split=int(args.scsampler_random_split))
    idx = np.asarray([int(x) for x in np.asarray(sub.obs_names).astype(str)], dtype=int)
    return (idx, {'official_backend': 'scsampler.scsampler', 'selection_backend': 'official_scsampler_anndata', 'random_split': args.scsampler_random_split, 'official_call': 'scsampler(adata, n_obs=m, copy=True, random_split=4)'})

def main():
    ap = add_args(C.add_common_args(argparse.ArgumentParser()))
    args = ap.parse_args()
    C.run_official_baseline(args, 'scsampler_scvalue', selector, {'method_family': 'scsampler', 'official_package': 'scsampler', 'official_call': 'scsampler(AnnData, n_obs=m, copy=True, random_split=4)', 'supervision': 'unsupervised sketch; supervised LR evaluation'})
if __name__ == '__main__':
    main()
