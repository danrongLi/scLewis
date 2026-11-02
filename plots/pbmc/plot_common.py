import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.name
BLEND = 'mu_blend_eta0p5' if DATASET == 'pbmc' else 'mu_blend_eta0p1_weightnone'
RELATIVE = 'relative_regularized_objective_gap_vs_full_lr'
OBJECTIVE = 'objective_full_train_objective_total'
METRICS = [(RELATIVE, 'Relative train loss'), ('trueclass_nll_macro_test', 'Test loss'), ('balanced_accuracy', 'Balanced accuracy')]
STYLE = {
    'freqmu_alpha1_beta0': ('scLewis-Freq', '#BFDDEE', '-', 'o'),
    'freqmu_alpha0_beta1': ('scLewis-μ', '#5AA9D6', '-', 'o'),
    BLEND: ('scLewis-μBlend', '#2C7FB8', '--', 's'),
    'random_global': ('Uniform', '#A6A6A6', '-', 'o'),
    'geosketch_global': ('GeoSketch', '#9467BD', '-', '^'),
    'sphetcher_euclidean_global': ('Sphetcher', '#C49A6C', '-', '^'),
    'scvalue_default': ('scValue', '#FC8D59', '-', 'D'),
    'treehopper_scvalue': ('TreeHopper', '#74C476', '-', 'v'),
    'hopper_official': ('Hopper', '#006D2C', '-', 'X'),
    'scsampler_scvalue': ('scSampler', '#B79F00', '-', 'v'),
    'kernel_herding_official': ('Kernel Herding', '#E7298A', '-', 'P'),
    'full_lr_pca': ('Full LR', '#000000', '--', None),
}
METHODS = list(STYLE)[:-1]
KEYS = ['method', 'seed', 'ratio']
COMPONENTS = ['base_sec', 'lewis_sec', 'mu_sec', 'total_sec']
plt.rcParams.update({'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 8, 'savefig.dpi': 350, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'hatch.linewidth': 1.05})


def read_csvs(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame['_input_file'] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def identity(frame):
    frame = frame.copy()
    if 'ratio' not in frame:
        frame['ratio'] = frame['coreset_ratio_target']
    if DATASET == 'heca':
        frame['method'] = frame.method.replace({'mu_blend_eta0p1': BLEND})
    frame = frame.loc[frame.method.isin(STYLE)].copy()
    if frame.empty:
        raise ValueError('No supported canonical method IDs. Use Euclidean Sphetcher, default scValue, and the dataset-specific Blend eta.')
    for column in ['seed', 'ratio']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if frame[KEYS].isna().any().any():
        raise ValueError('Missing method, seed or ratio identity')
    if not np.allclose(frame.seed, frame.seed.astype(int)):
        raise ValueError('Seeds must be integers')
    frame['seed'] = frame.seed.astype(int)
    frame = frame.loc[frame.method.eq('full_lr_pca') | frame.ratio.between(.02 - 1e-10, .60 + 1e-10)].copy()
    return frame


def unique_values(frame, columns):
    for key, group in frame.groupby(KEYS, dropna=False):
        if len(group) > 1:
            for column in columns:
                values = pd.to_numeric(group[column], errors='raise').to_numpy(float)
                if not np.allclose(values, values[0], rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise ValueError(f'Conflicting duplicate rows for {key}, {column}; select one canonical run per identity')
    return frame.drop_duplicates(KEYS).copy()


def full_lr_reference(frame, args):
    reference_source = identity(read_csvs(args.full_lr_reference)) if args.full_lr_reference else frame
    full = reference_source.loc[reference_source.method.eq('full_lr_pca')].copy()
    columns = [column for column in [OBJECTIVE, 'trueclass_nll_macro_test', 'balanced_accuracy'] if column in full]
    try:
        full = unique_values(full, columns)
    except ValueError as exc:
        raise ValueError('Conflicting Full LR references; supply --full-lr-reference with the selected-run or standalone FullLR.py output files') from exc
    if full.empty or full.seed.duplicated().any() or not full.ratio.eq(1).all():
        raise ValueError('Provide exactly one Full LR reference at ratio 1 per seed')
    return full


def relative_from_reference(frame, full):
    if OBJECTIVE not in frame or OBJECTIVE not in full:
        raise ValueError(f'Computing relative train loss requires {OBJECTIVE}')
    full = unique_values(full, [OBJECTIVE])
    if full.seed.duplicated().any():
        raise ValueError('Ambiguous source-run Full LR references; pass separate metrics.csv files or use --relative-loss-reference selected')
    denominator = frame.seed.map(full.set_index('seed')[OBJECTIVE])
    if denominator.isna().any() or not (denominator > 0).all():
        raise ValueError('Every prediction needs a positive same-seed Full LR objective from its requested reference')
    return (frame[OBJECTIVE] - denominator).abs() / denominator.abs()


def prediction_values(raw, args):
    frame = identity(raw)
    full = full_lr_reference(frame, args)
    for metric in ['trueclass_nll_macro_test', 'balanced_accuracy']:
        if metric not in frame:
            raise ValueError(f'Missing prediction metric: {metric}')
    if RELATIVE not in frame:
        frame[RELATIVE] = np.nan
    frame['relative_loss_source'] = 'saved_regularized_relative_loss'
    if args.relative_loss_reference in ['selected', 'legacy-global-baselines']:
        frame[RELATIVE] = relative_from_reference(frame, full)
        frame['relative_loss_source'] = 'explicit_selected_full_lr_reference'
        if args.relative_loss_reference == 'legacy-global-baselines':
            if DATASET != 'heca' or len(full) != 5:
                raise ValueError('legacy-global-baselines is HECA-only and requires five selected Full LR seed references')
            global_reference = float(full[OBJECTIVE].mean())
            baselines = frame.method.isin(METHODS[4:])
            frame.loc[baselines, RELATIVE] = (frame.loc[baselines, OBJECTIVE] - global_reference).abs() / abs(global_reference)
            frame.loc[baselines, 'relative_loss_source'] = 'explicit_legacy_global_baseline_reference'
    else:
        for _, group in frame.groupby('_input_file', dropna=False):
            missing = group[RELATIVE].isna()
            if not missing.any():
                continue
            own_full = group.loc[group.method.eq('full_lr_pca')]
            calculated = relative_from_reference(group.loc[missing], own_full)
            generic = 'relative_objective_gap_vs_full_lr'
            saved = pd.to_numeric(group.loc[missing, generic], errors='raise') if generic in group else pd.Series(np.nan, index=calculated.index)
            finite = saved.notna()
            if finite.any() and not np.allclose(saved.loc[finite], calculated.loc[finite], rtol=1e-10, atol=1e-12):
                raise ValueError('Saved source-run relative gap differs from the regularized objective definition; provide the explicit regularized metric or choose --relative-loss-reference selected')
            frame.loc[calculated.index, RELATIVE] = saved.where(finite, calculated)
            frame.loc[calculated.index, 'relative_loss_source'] = np.where(finite, 'saved_source_run_relative_gap', 'computed_source_run_relative_gap')
    full = full.copy()
    full[RELATIVE] = 0.
    full['relative_loss_source'] = 'full_lr_reference_zero'
    frame = pd.concat([frame.loc[~frame.method.eq('full_lr_pca')], full], ignore_index=True, sort=False)
    frame = unique_values(frame, [m for m,_ in METRICS] + ([OBJECTIVE] if OBJECTIVE in frame else []))
    for metric, _ in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors='raise')
        if not np.isfinite(frame[metric]).all():
            raise ValueError(f'Missing or nonfinite prediction values for {metric}')
    return frame


def timing_values(paths, kind):
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        if 'seed' not in frame:
            found = re.findall(r'seed[_-]?(\d+)', str(path))
            if not found or len(set(found)) != 1:
                raise ValueError(f'Missing unambiguous seed in {path}')
            frame['seed'] = int(found[0])
        column = 'lp_pool_wall_sec' if kind == 'mu' else 'lewis_runtime_sec'
        if column not in frame and kind == 'lewis':
            column = 'lewis_runtime_sec_median_over_repeats'
        if column not in frame:
            raise ValueError(f'Missing measured {column} in {path}')
        for seed, group in frame.groupby('seed'):
            values = pd.to_numeric(group[column], errors='raise')
            if not np.isfinite(values).all() or not (values > 0).all():
                raise ValueError(f'Invalid {kind} timing in {path}')
            if kind == 'mu' and len(values) != 1:
                raise ValueError(f'Expected one LP-stage timing per seed in {path}')
            rows.append((int(seed), float(values.median())))
    table = pd.DataFrame(rows, columns=['seed', 'seconds'])
    if table.seed.duplicated().any():
        raise ValueError(f'Multiple {kind} timing files for one seed')
    return table.set_index('seed').seconds


def runtime_values(raw, args):
    if args.runtime:
        frame = identity(read_csvs([args.runtime]))
    elif args.metrics:
        if not args.lewis_timings or not args.mu_timings:
            raise ValueError('Fresh metrics require --runtime or both --lewis-timings and --mu-timings; use --skip-runtime for prediction plots only')
        frame = identity(raw)
        frame['base_sec'] = pd.to_numeric(frame.runtime_sec, errors='raise')
        lewis = timing_values(args.lewis_timings, 'lewis')
        mu = timing_values(args.mu_timings, 'mu')
        frame['lewis_sec'] = np.where(frame.method.isin(METHODS[:3]), frame.seed.map(lewis), 0.)
        frame['mu_sec'] = np.where(frame.method.isin(METHODS[1:3]), frame.seed.map(mu), 0.)
        frame['total_sec'] = frame.base_sec + frame.lewis_sec + frame.mu_sec
    else:
        frame = identity(read_csvs([ROOT / 'published_results/runtime.csv']))
    alternatives = {'base_sec': ['runtime_base_component_sec', 'runtime_base_lr_fit_sec'], 'lewis_sec': ['runtime_lewis_component_sec'], 'mu_sec': ['runtime_mu_lp_component_sec', 'mu_lp_sec'], 'total_sec': ['runtime_total_sec', 'runtime_lewis_lp_lrfit_sec']}
    for target, choices in alternatives.items():
        for column in choices:
            if column in frame:
                frame[target] = frame[column]
                break
        if target not in frame:
            raise ValueError(f'Missing runtime component {target}')
        frame[target] = pd.to_numeric(frame[target], errors='raise')
    if args.full_lr_reference:
        references = identity(read_csvs(args.full_lr_reference))
        references = references.loc[references.method.eq('full_lr_pca')].copy()
        if 'runtime_sec' not in references:
            raise ValueError('--full-lr-reference files need runtime_sec for runtime plots')
        references['base_sec'] = pd.to_numeric(references.runtime_sec, errors='raise')
        references['lewis_sec'] = 0.
        references['mu_sec'] = 0.
        references['total_sec'] = references.base_sec
        frame = pd.concat([frame.loc[~frame.method.eq('full_lr_pca')], references], ignore_index=True, sort=False)
    frame = unique_values(frame, COMPONENTS)
    if not np.isfinite(frame[COMPONENTS]).all().all() or (frame[COMPONENTS] < 0).any().any() or not frame.total_sec.gt(0).all():
        raise ValueError('Runtime components must be complete, finite and nonnegative, with positive totals')
    if not np.allclose(frame.total_sec, frame.base_sec + frame.lewis_sec + frame.mu_sec, rtol=1e-12, atol=1e-12):
        raise ValueError('Runtime totals do not equal their measured component sums')
    if not frame.loc[frame.method.isin(METHODS[:3]), 'lewis_sec'].gt(0).all() or not frame.loc[frame.method.isin(METHODS[1:3]), 'mu_sec'].gt(0).all():
        raise ValueError('Missing accelerated Lewis or LP-stage timing for scLewis')
    if not frame.loc[~frame.method.isin(METHODS[:3]), ['lewis_sec', 'mu_sec']].eq(0).all().all() or not frame.loc[frame.method.eq(METHODS[0]), 'mu_sec'].eq(0).all():
        raise ValueError('Runtime precomputation assigned to an inapplicable method')
    reference = frame.loc[frame.method.eq('full_lr_pca')]
    if reference.empty or reference.seed.duplicated().any() or not reference.ratio.eq(1).all():
        raise ValueError('Provide one Full LR runtime reference at ratio 1 per seed')
    if not set(frame.seed).issubset(set(reference.seed)):
        raise ValueError('Every runtime seed needs a Full LR runtime reference')
    return frame


def summary(frame, metric):
    return frame.groupby(['method', 'ratio'])[metric].agg(mean='mean', sd='std', n_seed='count').reset_index().fillna({'sd': 0.})


def save(fig, path, args, bounds=None, preserve_axis20=False):
    if preserve_axis20:
        fig.canvas.draw()
        old = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', bbox_inches='tight', pad_inches=.1)
        match = re.search(rb'/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*\]', buffer.getvalue())
        x0, y0, x1, y1 = map(float, match.groups())
        bounds = Bbox.from_bounds(old.x0, old.y0, (x1-x0)/72, (y1-y0)/72)
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
    outputs = []
    for extension in args.formats.split(','):
        file = path.with_suffix('.' + extension)
        fig.savefig(file, dpi=args.dpi, bbox_inches=bounds if bounds is not None else 'tight', pad_inches=0 if bounds is not None else .1)
        outputs.append(file.name)
    plt.close(fig)
    return outputs


def line_plot(frame, metric, ylabel, path, args):
    agg = summary(frame.loc[frame.method.isin(METHODS)], metric)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for method in METHODS:
        group = agg.loc[agg.method.eq(method)].sort_values('ratio')
        if group.empty:
            continue
        label, color, linestyle, marker = STYLE[method]
        x, y, sd = group.ratio.to_numpy(float)*100, group['mean'].to_numpy(float), group.sd.to_numpy(float)
        ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=5, label=label, zorder=3)
        if np.max(sd) > 0:
            ax.fill_between(x, y-sd, y+sd, color=color, alpha=.14, linewidth=0, zorder=2)
    reference = frame.loc[frame.method.eq('full_lr_pca'), metric]
    if len(reference):
        y, sd = reference.mean(), reference.std(ddof=1) if len(reference)>1 else 0.
        ax.axhline(y, color='#000000', linestyle='--', linewidth=1.7, zorder=1)
        if sd > 0:
            ax.axhspan(y-sd, y+sd, color='#000000', alpha=.10, linewidth=0, zorder=0)
    ratios = agg.ratio.to_numpy()*100
    if not len(ratios):
        raise ValueError(f'No sketch rows for {metric}')
    lo, hi = ratios.min(), min(60., ratios.max())
    ax.set_xlim(max(0, lo-max(.6, (hi-lo)*.035)), min(62., hi+max(.6, (hi-lo)*.035)))
    ax.set_xticks(np.arange(0, 60.1, 10))
    ax.set_xlabel('Sketch size (% training cells)', fontsize=20, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=8)
    ax.tick_params(axis='both', labelsize=11.5)
    if metric == RELATIVE:
        ax.set_ylim(bottom=0)
    if metric == 'total_sec':
        ax.set_yscale('log')
    agg.to_csv(path.with_name(path.name+'_summary').with_suffix('.csv'), index=False)
    return save(fig, path, args, preserve_axis20=True)


def box_plot(frame, metric, ylabel, ratio, path, args):
    selected = frame.loc[np.isclose(frame.ratio, ratio)]
    methods = [m for m in METHODS if selected.method.eq(m).any()]
    if not methods:
        raise ValueError(f'No prediction rows at {ratio}')
    rows = [selected.loc[selected.method.eq(m), metric].to_numpy(float) for m in methods]
    fig, ax = plt.subplots(figsize=(max(7.8, .66*len(rows)), 4.8))
    boxes = ax.boxplot(rows, patch_artist=True, showfliers=False)
    for box, method in zip(boxes['boxes'], methods):
        color = STYLE[method][1]
        box.set(facecolor=color, edgecolor=color, alpha=.47, linewidth=1.2)
    for median in boxes['medians']:
        median.set(color='black', linewidth=1.25)
    for i, (values, method) in enumerate(zip(rows, methods), 1):
        jitter = np.linspace(-.055, .055, len(values)) if len(values)>1 else np.array([0.])
        ax.scatter(np.full(len(values), i)+jitter, values, color=STYLE[method][1], marker=STYLE[method][3], s=22, edgecolor='black', linewidth=.25, zorder=3)
    ax.set_xlim(.55, len(rows)+.45)
    ax.set_xticks(range(1, len(methods)+1))
    ax.set_xticklabels([STYLE[m][0] for m in methods], rotation=55, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=11.5)
    ax.set_ylabel(f'{ylabel} at {int(round(ratio*100))}%', fontsize=18, labelpad=8)
    if metric == RELATIVE:
        ax.set_ylim(bottom=0)
    return save(fig, path, args)


def darken(color, factor):
    return tuple(value*factor for value in to_rgb(color))


def runtime_bar(frame, ratio):
    selected = frame.loc[np.isclose(frame.ratio, ratio)]
    agg = selected.groupby('method')[COMPONENTS].mean()
    methods = [method for method in METHODS if method in agg.index]
    if not methods:
        raise ValueError(f'No runtime rows at {ratio}')
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    width = .72
    totals = []
    for i, method in enumerate(methods):
        color = STYLE[method][1]
        row = agg.loc[method]
        totals.append(row.total_sec)
        bottom = 0.
        for column, hatch in [('base_sec', None), ('lewis_sec', '////'), ('mu_sec', 'xxxx')]:
            height = row[column]
            if height <= 0:
                continue
            ax.bar(i, height, width=width, bottom=bottom, color=color, edgecolor=darken(color,.68), linewidth=.95, alpha=.47, zorder=3)
            if hatch:
                ax.bar(i, height, width=width, bottom=bottom, facecolor='none', edgecolor=darken(color,.42), linewidth=0., hatch=hatch, zorder=4)
                ax.hlines(bottom, i-width/2, i+width/2, colors=darken(color,.34), linewidth=.95, zorder=6)
            bottom += height
    reference = frame.loc[frame.method.eq('full_lr_pca'), 'total_sec']
    if len(reference):
        value = reference.mean()
        ax.hlines(value, -.55, len(methods)-.45, colors='#000000', linestyles='--', linewidth=2., zorder=4)
        totals.append(value)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([STYLE[m][0] for m in methods], rotation=55, ha='right', fontsize=13)
    ax.set_ylabel(f'Runtime at {round(ratio*100)}% (s)', fontsize=18, labelpad=8)
    ax.tick_params(axis='y', labelsize=11.5)
    ax.set_yscale('log')
    ax.set_ylim(max(min(totals)/1.8, 1e-3), max(totals)*2.1)
    ax.set_xlim(-.55, len(methods)-.45)
    return fig


def legend(path, args, components=False):
    if components:
        handles = [Patch(facecolor='#A6A6A6', edgecolor=darken('#A6A6A6',.68), alpha=.47, linewidth=.95, label='Sampling + LR fitting'), Patch(facecolor='white', edgecolor='black', linewidth=.75, hatch='////', label='Lewis weights'), Patch(facecolor='white', edgecolor='black', linewidth=.75, hatch='xxxx', label='μ LP stage'), Line2D([0],[0],color='black',linestyle='--',linewidth=2.1,label='Full LR')]
        fig, ax = plt.subplots(figsize=(6.2,.42))
        ax.axis('off')
        item = ax.legend(handles=handles, loc='center', ncol=4, frameon=False, handlelength=1.55, handletextpad=.30, columnspacing=.85, borderaxespad=0, borderpad=0, labelspacing=.05, fontsize=8.6)
        fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    else:
        handles = [Line2D([0],[0],color=c,linestyle=s,marker=m,linewidth=2,markersize=5,label=l) for l,c,s,m in STYLE.values()]
        fig = plt.figure(figsize=(1,1))
        item = fig.legend(handles=handles, loc='center', ncol=7, frameon=False, handlelength=2.4, handletextpad=.45, columnspacing=.80, labelspacing=.20, borderaxespad=0, borderpad=0)
    fig.canvas.draw()
    bounds = item.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.02)
    return save(fig,path,args,bounds=bounds)


def main(mode):
    parser = argparse.ArgumentParser(description=f'Reproduce {DATASET.upper()} paper {mode} for relative train loss, test loss, balanced accuracy and accelerated runtime.')
    parser.add_argument('--metrics', type=Path, nargs='+')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--full-lr-reference', type=Path, nargs='+', help='Select Full LR predictive and runtime reference rows from selected-run or standalone FullLR.py metrics.csv files')
    parser.add_argument('--relative-loss-reference', choices=['source-run', 'selected', 'legacy-global-baselines'], default='source-run', help='source-run preserves saved relative metrics or derives them from each input file; selected recomputes against same-seed --full-lr-reference (canonical figure policy); legacy-global-baselines uses the five-reference mean for HECA official baselines (historical table policy)')
    parser.add_argument('--lewis-timings', type=Path, nargs='+')
    parser.add_argument('--mu-timings', type=Path, nargs='+')
    parser.add_argument('--skip-runtime', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'plots/generated'/mode)
    parser.add_argument('--formats', default='pdf,png')
    parser.add_argument('--dpi', type=int, default=350)
    args = parser.parse_args()
    if set(args.formats.split(',')) - {'pdf','png','svg'}:
        parser.error('--formats must be a comma-separated subset of pdf,png,svg')
    raw = read_csvs(args.metrics or [ROOT/'published_results/metrics.csv'])
    predictions = prediction_values(raw,args)
    runtimes = None if args.skip_runtime else runtime_values(raw,args)
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}; choose a new --output directory')
    args.output.mkdir(parents=True)
    outputs = []
    predictions[KEYS+[m for m,_ in METRICS]+['relative_loss_source']].to_csv(args.output/'predictive_values.csv',index=False)
    if runtimes is not None:
        runtimes[KEYS+COMPONENTS].to_csv(args.output/'runtime_values.csv',index=False)
    if mode == 'boxplots':
        for ratio in [.02,.06]:
            for metric,ylabel in METRICS:
                outputs += box_plot(predictions,metric,ylabel,ratio,args.output/f'{metric}_box_{round(ratio*100):02d}pct',args)
        if runtimes is not None:
            figures = [(runtime_bar(runtimes,ratio),ratio) for ratio in [.02,.06]]
            for fig,_ in figures:
                fig.canvas.draw()
            bounds = Bbox.union([fig.get_tightbbox(fig.canvas.get_renderer()) for fig,_ in figures]).padded(.1)
            for fig,ratio in figures:
                outputs += save(fig,args.output/f'runtime_bar_{round(ratio*100):02d}pct',args,bounds=bounds)
            outputs += legend(args.output/'legend_runtime_components',args,components=True)
    else:
        for metric,ylabel in METRICS:
            outputs += line_plot(predictions,metric,ylabel,args.output/metric,args)
        if runtimes is not None:
            outputs += line_plot(runtimes,'total_sec','Runtime (s)',args.output/'runtime',args)
        outputs += legend(args.output/'legend_methods',args)
    paths = list(args.metrics or [ROOT/'published_results/metrics.csv'])
    if runtimes is not None:
        paths += [args.runtime] if args.runtime else list(args.lewis_timings or [])+list(args.mu_timings or []) if args.metrics else [ROOT/'published_results/runtime.csv']
    paths += list(args.full_lr_reference or [])
    report = {'full_lr_reference_files':[str(p) for p in args.full_lr_reference or []],'relative_loss_reference_policy':args.relative_loss_reference,'relative_loss_sources':predictions.relative_loss_source.value_counts().to_dict(),'dataset':DATASET,'mode':mode,'predictive_rows':len(predictions),'runtime_rows':len(runtimes) if runtimes is not None else 0,'metrics':dict(METRICS),'runtime':'Accelerated measured components; LP stage only; sum of separate jobs, not end-to-end runtime','blend_eta':.5 if DATASET=='pbmc' else .1,'statistics':'Mean and sample standard deviation over available seeds; no imputation','seed_counts':predictions.groupby(['method','ratio']).seed.nunique().reset_index(name='n_seed').to_dict('records'),'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'outputs':outputs}
    (args.output/'render_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'plots':len(outputs),'predictive_rows':len(predictions),'runtime_rows':report['runtime_rows']}))
