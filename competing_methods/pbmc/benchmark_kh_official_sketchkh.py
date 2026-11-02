from __future__ import annotations
import argparse, numpy as np
import corlewis_official_baseline_common as C

def add_args(ap):
    ap.add_argument('--kh_gamma', default='1.0')
    ap.add_argument('--kh_n_jobs', type=int, default=1)
    return ap

def selector(data, m, seed, args):
    import anndata as ad
    from sketchKH import sketch
    X = np.asarray(data['X_train'], dtype=np.float32)
    a = ad.AnnData(X=X)
    a.obs_names = [str(i) for i in range(X.shape[0])]
    a.obs['sample_set'] = 'global'
    gamma = float(args.kh_gamma)
    (idx, _sub) = sketch(a, sample_set_key='sample_set', gamma=gamma, num_subsamples=int(m), frequency_seed=int(seed), n_jobs=int(args.kh_n_jobs))
    if isinstance(idx, (tuple, list)) and len(idx) == 1:
        idx = idx[0]
    return (np.asarray(idx, dtype=int).ravel(), {'official_backend': 'sketchKH.sketch', 'selection_backend': 'official_sketchkh', 'gamma': gamma, 'n_jobs': args.kh_n_jobs})

def main():
    ap = add_args(C.add_common_args(argparse.ArgumentParser()))
    args = ap.parse_args()
    C.run_official_baseline(args, 'kernel_herding_official', selector, {'method_family': 'kernel_herding', 'official_package': 'SketchKH/sketchKH', 'official_call': "sketchKH.sketch(AnnData, sample_set_key='sample_set', gamma=1.0, num_subsamples=m)", 'supervision': 'unsupervised sketch; supervised LR evaluation'})
if __name__ == '__main__':
    main()
