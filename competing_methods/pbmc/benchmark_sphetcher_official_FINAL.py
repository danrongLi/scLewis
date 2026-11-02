from __future__ import annotations
import argparse, subprocess, numpy as np, pandas as pd
from pathlib import Path
import corlewis_official_baseline_common as C

def add_args(ap):
    ap.add_argument('--sphetcher_bin', default='sphetcher')
    ap.add_argument('--sphetcher_mode', default='global', choices=['global', 'labeled'])
    ap.add_argument('--sphetcher_l_min', type=int, default=1)
    ap.add_argument('--sphetcher_distance_metric', default='cosine', choices=['cosine', 'euclidean', 'manhattan', 'spearman'], help='Metadata/method-name tag for the compiled Sphetcher binary. The official binary has the metric hard-coded in src/sphetcherTime.C.')
    return ap

def selector(data, m, seed, args):
    outdir = Path(args.output_dir)
    tmp = C.ensure_dir(outdir / 'tmp')
    X = np.asarray(data['X_train'], dtype=np.float32)
    matrix_path = tmp / 'sphetcher_matrix.csv'
    if not matrix_path.exists():
        pd.DataFrame(X).to_csv(matrix_path, header=False, index=False)
    out_path = tmp / f'sphetcher_indicator_m{m}_seed{seed}_{args.sphetcher_mode}_{args.sphetcher_distance_metric}.csv'
    stdout = tmp / f'sphetcher_m{m}_seed{seed}_{args.sphetcher_mode}_{args.sphetcher_distance_metric}.stdout.txt'
    if args.sphetcher_mode == 'global':
        cmd = [args.sphetcher_bin, str(matrix_path), str(int(m)), str(out_path)]
    else:
        y = np.asarray(data['y_train'], dtype=int) + 1
        lab_path = tmp / 'sphetcher_labels.csv'
        if not lab_path.exists():
            pd.DataFrame(y).to_csv(lab_path, header=False, index=False)
        cmd = [args.sphetcher_bin, str(matrix_path), str(int(m)), str(lab_path), str(int(args.sphetcher_l_min)), str(out_path)]
    with open(stdout, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Sphetcher failed returncode={proc.returncode}; see {stdout}')
    ind = pd.read_csv(out_path, header=None).iloc[:, 0].to_numpy()
    idx = np.flatnonzero(ind.astype(float) > 0)
    return (idx, {'official_backend': 'Sphetcher binary', 'selection_backend': 'official_sphetcher', 'sphetcher_mode': args.sphetcher_mode, 'sphetcher_distance_metric': args.sphetcher_distance_metric, 'sphetcher_cmd': ' '.join(cmd), 'sphetcher_stdout': str(stdout)})

def main():
    ap = add_args(C.add_common_args(argparse.ArgumentParser()))
    args = ap.parse_args()
    metric = str(args.sphetcher_distance_metric).lower()
    prefix = 'sphetcher' if metric == 'cosine' else f'sphetcher_{metric}'
    method_name = f'{prefix}_{args.sphetcher_mode}'
    C.run_official_baseline(args, method_name, selector, {'method_family': 'sphetcher', 'official_package': 'canzarlab/Sphetcher', 'official_call': 'sphetcher expression_matrix.csv sketch_size sketch_output.csv', 'sphetcher_distance_metric': metric, 'supervision': 'unsupervised global sketch by default; supervised LR evaluation'})
if __name__ == '__main__':
    main()
