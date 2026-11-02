from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from validate_pbmc_branch import require_preprocessed
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, log_loss, roc_auc_score, precision_score, recall_score, matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
try:
    import anndata as ad
except Exception:
    ad = None
try:
    import scanpy as sc
except Exception:
    sc = None
try:
    import psutil
except Exception:
    psutil = None
try:
    import celltypist
except Exception:
    celltypist = None
LOGGER = logging.getLogger('pilot_corLR_corLewis')

def setup_logging(verbose: bool=False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    LOGGER.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def now() -> float:
    return time.perf_counter()

def rss_mb() -> float:
    if psutil is None:
        return float('nan')
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        return float('nan')

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]

def parse_csv_strs(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(',') if x.strip()]

def ratio_allowed(value: float, spec: str, tol: float=0.0001) -> bool:
    if spec is None or str(spec).strip() == '' or str(spec).strip().lower() in {'all', 'any', 'none'}:
        return True
    try:
        vals = parse_csv_floats(spec)
    except Exception:
        return True
    if not vals:
        return True
    try:
        v = float(value)
    except Exception:
        return False
    return any((abs(v - float(x)) <= tol for x in vals))

def safe_auc_metrics(y_true: np.ndarray, proba: np.ndarray, prefix: str='original_softmax') -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        y = np.asarray(y_true, dtype=int)
        P = np.asarray(proba, dtype=float)
        labels = list(range(P.shape[1]))
        auc = float(roc_auc_score(y, P, labels=labels, multi_class='ovr', average='macro'))
        one_minus = float(max(1.0 - auc, 1e-12))
        out[f'roc_auc_ovr_macro_{prefix}'] = auc
        out[f'one_minus_auc_ovr_macro_{prefix}'] = one_minus
        out[f'log10_one_minus_auc_ovr_macro_{prefix}'] = float(np.log10(one_minus))
    except Exception:
        out[f'roc_auc_ovr_macro_{prefix}'] = np.nan
        out[f'one_minus_auc_ovr_macro_{prefix}'] = np.nan
        out[f'log10_one_minus_auc_ovr_macro_{prefix}'] = np.nan
    return out

def float_tag(x: float) -> str:
    s = f'{float(x):g}'
    return s.replace('-', 'm').replace('.', 'p')

def float_from_tag(text: str) -> float:
    return float(str(text).replace('p', '.').replace('m', '-'))

def expand_benchmark_methods(raw_methods: Sequence[str], args: argparse.Namespace) -> List[str]:
    mixes = parse_csv_floats(getattr(args, 'lewis_uniform_mixes', '0.05'))
    etas = parse_csv_floats(getattr(args, 'mu_blend_etas', '0.25,0.5'))
    out: List[str] = []
    seen = set()
    for method in raw_methods:
        m = str(method).strip()
        expanded: List[str]
        if m == 'corLewis_global_grid':
            expanded = [f'corLewis_global_mix{float_tag(x)}' for x in mixes]
        elif m == 'corLewis_equal_grid':
            expanded = [f'corLewis_equal_mix{float_tag(x)}' for x in mixes]
        elif m == 'corLewis_mu_grid':
            expanded = [f'corLewis_mu_mix{float_tag(x)}' for x in mixes]
        elif m == 'corLewis_mu_blend_grid':
            expanded = [f'corLewis_mu_blend_eta{float_tag(e)}_mix{float_tag(x)}' for e in etas for x in mixes]
        else:
            expanded = [m]
        for e in expanded:
            if e and e not in seen:
                out.append(e)
                seen.add(e)
    return out

def parse_method_variant(method: str, args: argparse.Namespace) -> Dict[str, object]:
    m = str(method)
    cfg: Dict[str, object] = {'method': m, 'family': m, 'lewis_uniform_mix': float(getattr(args, 'lewis_uniform_mix', 0.05)), 'mu_blend_eta': float(getattr(args, 'mu_blend_eta', 0.25)), 'mu_power': float(getattr(args, 'mu_power', 2.0)), 'mu_cap': float(getattr(args, 'mu_cap', 10.0)), 'mu_floor': float(getattr(args, 'mu_floor', 1.0))}
    if m.startswith('corLewis_global_mix'):
        cfg['family'] = 'corLewis_global'
        cfg['lewis_uniform_mix'] = float_from_tag(m.split('corLewis_global_mix', 1)[1])
    elif m == 'corLewis_global':
        cfg['family'] = 'corLewis_global'
    elif m.startswith('corLewis_equal_mix'):
        cfg['family'] = 'corLewis_equal'
        cfg['lewis_uniform_mix'] = float_from_tag(m.split('corLewis_equal_mix', 1)[1])
    elif m.startswith('corLewis_mu_mix'):
        cfg['family'] = 'corLewis_mu'
        cfg['lewis_uniform_mix'] = float_from_tag(m.split('corLewis_mu_mix', 1)[1])
    elif m.startswith('corLewis_mu_blend_eta'):
        mm = re.match('corLewis_mu_blend_eta([^_]+)_mix(.+)$', m)
        if not mm:
            raise ValueError(f'Cannot parse CorLewis blend method name: {m}')
        cfg['family'] = 'corLewis_mu_blend'
        cfg['mu_blend_eta'] = float_from_tag(mm.group(1))
        cfg['lewis_uniform_mix'] = float_from_tag(mm.group(2))
    elif m == 'corLewis_mu_blend':
        cfg['family'] = 'corLewis_mu_blend'
    return cfg

def method_needs_lewis(method: str) -> bool:
    return str(method).startswith('corLewis')

def safe_name(text: object, max_len: int=160) -> str:
    import re
    out = re.sub('[^A-Za-z0-9_.-]+', '_', str(text)).strip('_')
    return (out or 'unnamed')[:max_len]

def stable_int_hash(text: str, modulo: int=1000000000) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16) % modulo

def append_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    new_df = pd.DataFrame(list(rows))
    if not path.exists():
        new_df.to_csv(path, index=False)
        return
    old_cols = list(pd.read_csv(path, nrows=0).columns)
    cols = old_cols[:]
    added = False
    for col in new_df.columns:
        if col not in cols:
            cols.append(col)
            added = True
    if added:
        old_df = pd.read_csv(path)
        old_df = old_df.reindex(columns=cols)
        new_df = new_df.reindex(columns=cols)
        pd.concat([old_df, new_df], ignore_index=True).to_csv(path, index=False)
    else:
        new_df = new_df.reindex(columns=old_cols)
        new_df.to_csv(path, mode='a', header=False, index=False)

def dense_float64(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)

def dense_float32(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)

@contextlib.contextmanager
def timed():
    start = now()
    yield (lambda : now() - start)

def read_h5ad(path: str):
    if ad is None:
        raise RuntimeError('anndata is required. Install with: pip install anndata')
    LOGGER.info('Reading %s', path)
    adata = ad.read_h5ad(path)
    if adata.obs_names.has_duplicates:
        adata.obs_names_make_unique()
    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
    return adata

def normalize_total_log1p_inplace(adata, target_sum: float) -> None:
    raise RuntimeError('Normalization/log1p is disabled for the preprocessed PBMC branch')

def maybe_stratified_downsample_adata(adata, label_key: str, max_cells: int, seed: int):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    LOGGER.info('Pilot max_cells=%d: stratified downsampling from %d cells', max_cells, adata.n_obs)
    y = adata.obs[label_key].astype(str).values
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_cells, random_state=seed)
    (keep_idx, _) = next(splitter.split(np.zeros(len(y)), y))
    return adata[keep_idx].copy()

def fit_github_svd_full_matrix(X, n_components: int, seed: int) -> Tuple[np.ndarray, TruncatedSVD, float]:
    t0 = now()
    n_eff = min(int(n_components), X.shape[0] - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_eff, algorithm='randomized', random_state=seed)
    X_svd = svd.fit_transform(X).astype(np.float64, copy=False)
    return (X_svd, svd, now() - t0)

def _fallback_downsample_each(y_train: Sequence[object], n_cells: int, seed: int) -> np.ndarray:
    y = pd.Series(np.asarray(y_train).astype(str))
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    for cls in sorted(y.unique()):
        idx = np.flatnonzero(y.values == cls)
        take = min(len(idx), max(0, int(n_cells)))
        if take > 0:
            selected.append(rng.choice(idx, size=take, replace=False))
    if not selected:
        return np.array([], dtype=int)
    out = np.concatenate(selected).astype(int)
    rng.shuffle(out)
    return out

def github_corLR_indices(adata_train, y_train: pd.Series, label_key: str, target_size: int, n_labels_full: int, random_state: int, use_celltypist_downsample: bool) -> Tuple[np.ndarray, Dict[str, object]]:
    n_per_label = int(target_size / max(n_labels_full, 1))
    used_celltypist = False
    if use_celltypist_downsample and celltypist is not None:
        try:
            sampled = celltypist.samples.downsample_adata(adata=adata_train, mode='each', n_cells=n_per_label, by=label_key, random_state=random_state, return_index=True)
            idx = np.asarray(sampled, dtype=int)
            used_celltypist = True
        except Exception as exc:
            LOGGER.warning('CellTypist downsample_adata failed; using fallback. Error: %s', exc)
            idx = _fallback_downsample_each(y_train, n_per_label, random_state)
    else:
        idx = _fallback_downsample_each(y_train, n_per_label, random_state)
    return (idx, {'quota_rule': 'github_equal_per_label', 'n_per_label_requested': int(n_per_label), 'used_celltypist_downsample_adata': bool(used_celltypist)})

def approximate_l1_lewis_weights(A: np.ndarray, n_iter: int=10, ridge: float=1e-06, damping: float=0.5, block_size: int=50000, verbose: bool=False) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    (n, d) = A.shape
    w = np.ones(n, dtype=np.float64) * (d / max(n, 1))
    trace_ata = float(np.sum(A * A))
    ridge_scale = ridge * max(trace_ata / max(d, 1), 1e-12)
    for it in range(int(n_iter)):
        inv_w = 1.0 / np.maximum(w, 1e-12)
        M = A.T @ (A * inv_w[:, None])
        M.flat[::d + 1] += ridge_scale
        try:
            cf = cho_factor(M, lower=True, check_finite=False)
            q = np.empty(n, dtype=np.float64)
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                B = A[start:stop]
                solved = cho_solve(cf, B.T, check_finite=False).T
                q[start:stop] = np.sum(B * solved, axis=1)
        except Exception:
            Minv = np.linalg.pinv(M)
            q = np.einsum('ij,jk,ik->i', A, Minv, A)
        w_new = np.sqrt(np.maximum(q, 1e-30))
        w_new *= d / max(w_new.sum(), 1e-30)
        w = damping * w + (1.0 - damping) * w_new
        w *= d / max(w.sum(), 1e-30)
        if verbose:
            max_rel = np.max(np.abs(w_new - w) / np.maximum(w, 1e-12))
            LOGGER.info('Lewis iter %02d: sum=%.4f max_rel=%.3e', it + 1, w.sum(), max_rel)
    w *= d / max(w.sum(), 1e-30)
    return w

def lewis_fixed_point_diagnostics(A: np.ndarray, w: np.ndarray, ridge: float=1e-06, block_size: int=50000) -> Dict[str, float]:
    try:
        A = np.asarray(A, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        (n, d) = A.shape
        inv_w = 1.0 / np.maximum(w, 1e-12)
        trace_ata = float(np.sum(A * A))
        ridge_scale = ridge * max(trace_ata / max(d, 1), 1e-12)
        M = A.T @ (A * inv_w[:, None])
        M.flat[::d + 1] += ridge_scale
        try:
            cf = cho_factor(M, lower=True, check_finite=False)
            q = np.empty(n, dtype=np.float64)
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                B = A[start:stop]
                solved = cho_solve(cf, B.T, check_finite=False).T
                q[start:stop] = np.sum(B * solved, axis=1)
        except Exception:
            Minv = np.linalg.pinv(M)
            q = np.einsum('ij,jk,ik->i', A, Minv, A)
        w_target = np.sqrt(np.maximum(q, 1e-30))
        w_target *= d / max(float(w_target.sum()), 1e-30)
        ratio = w_target / np.maximum(w, 1e-30)
        log_ratio = np.log(np.maximum(ratio, 1e-300))
        return {'lewis_fixed_point_ratio_min': float(np.min(ratio)), 'lewis_fixed_point_ratio_q01': float(np.quantile(ratio, 0.01)), 'lewis_fixed_point_ratio_median': float(np.median(ratio)), 'lewis_fixed_point_ratio_q99': float(np.quantile(ratio, 0.99)), 'lewis_fixed_point_ratio_max': float(np.max(ratio)), 'lewis_fixed_point_log_ratio_abs_max': float(np.max(np.abs(log_ratio))), 'lewis_fixed_point_log_ratio_abs_median': float(np.median(np.abs(log_ratio)))}
    except Exception as exc:
        return {'lewis_fixed_point_ratio_min': np.nan, 'lewis_fixed_point_ratio_q01': np.nan, 'lewis_fixed_point_ratio_median': np.nan, 'lewis_fixed_point_ratio_q99': np.nan, 'lewis_fixed_point_ratio_max': np.nan, 'lewis_fixed_point_log_ratio_abs_max': np.nan, 'lewis_fixed_point_log_ratio_abs_median': np.nan}

def lewis_diagnostic_row(w: np.ndarray, base: Mapping[str, object]) -> Dict[str, object]:
    q = np.asarray(w, dtype=float) / max(float(np.sum(w)), 1e-300)
    n = len(q)
    row = dict(base)
    row.update({'lewis_sum': float(np.sum(w)), 'lewis_prob_min': float(np.min(q)), 'lewis_prob_max': float(np.max(q)), 'lewis_prob_mean': float(np.mean(q)), 'lewis_prob_std': float(np.std(q)), 'lewis_prob_cv': float(np.std(q) / max(np.mean(q), 1e-300)), 'lewis_max_over_uniform': float(np.max(q) * n), 'lewis_min_over_uniform': float(np.min(q) * n), 'lewis_effective_n': float(1.0 / max(np.sum(q * q), 1e-300)), 'lewis_effective_n_ratio': float(1.0 / max(np.sum(q * q), 1e-300) / max(n, 1))})
    return row

def random_global_indices(n_train: int, target_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(n_train)
    m = max(1, min(int(target_size), n))
    return rng.choice(np.arange(n), size=m, replace=False).astype(int)

def sample_global_by_lewis(lewis_weights: np.ndarray, target_size: int, seed: int, uniform_mix: float, return_inverse_probability_weights: bool=False) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    w = np.asarray(lewis_weights, dtype=np.float64)
    n = len(w)
    m = max(1, min(int(target_size), n))
    rng = np.random.default_rng(seed)
    if n == 0:
        return (np.array([], dtype=int), None, {'global_sampling_note': 'empty_input'})
    if not np.isfinite(w).all() or w.sum() <= 0:
        p = np.ones(n, dtype=np.float64) / n
    else:
        p_lewis = w / max(float(w.sum()), 1e-300)
        p_uniform = np.ones(n, dtype=np.float64) / n
        mix = float(np.clip(float(uniform_mix), 0.0, 1.0))
        p = (1.0 - mix) * p_lewis + mix * p_uniform
        p = np.maximum(p, 0.0)
        p /= max(float(p.sum()), 1e-300)
    if m >= n:
        idx = np.arange(n, dtype=int)
    else:
        idx = rng.choice(np.arange(n), size=m, replace=False, p=p).astype(int)
    sample_weight = None
    if return_inverse_probability_weights:
        sample_weight = 1.0 / np.maximum(m * p[idx], 1e-300)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    info = {'global_sampling_note': 'global_lewis_uniform_mixture_without_replacement', 'global_sampling_prob_min': float(np.min(p)), 'global_sampling_prob_max': float(np.max(p)), 'global_sampling_prob_effective_n': float(1.0 / max(float(np.sum(p * p)), 1e-300)), 'sample_weight_used': bool(sample_weight is not None), 'sample_weight_sum': float(np.sum(sample_weight)) if sample_weight is not None else np.nan, 'sample_weight_min': float(np.min(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan, 'sample_weight_max': float(np.max(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan, 'sample_weight_mean': float(np.mean(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan}
    return (idx, sample_weight, info)

def sample_global_by_lewis_class_floor(lewis_weights: np.ndarray, y_train: Sequence[object], target_size: int, seed: int, uniform_mix: float, class_floor_mix: float, return_inverse_probability_weights: bool=True, weight_clip_quantile: float=1.0, rescale_weight_sum_to_n: bool=True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    w = np.asarray(lewis_weights, dtype=np.float64)
    y = np.asarray(y_train).astype(str)
    n = len(w)
    m = max(1, min(int(target_size), n))
    rng = np.random.default_rng(seed)
    if n == 0:
        return (np.array([], dtype=int), None, {'global_sampling_note': 'empty_input'})
    if not np.isfinite(w).all() or float(w.sum()) <= 0:
        p_lewis = np.ones(n, dtype=np.float64) / n
    else:
        p_lewis = np.maximum(w, 0.0)
        p_lewis /= max(float(p_lewis.sum()), 1e-300)
    p_uniform = np.ones(n, dtype=np.float64) / n
    counts = pd.Series(y).value_counts()
    n_classes = max(int(len(counts)), 1)
    count_lookup = counts.to_dict()
    p_class = np.array([1.0 / (n_classes * max(int(count_lookup.get(lbl, 1)), 1)) for lbl in y], dtype=np.float64)
    p_class = np.maximum(p_class, 0.0)
    p_class /= max(float(p_class.sum()), 1e-300)
    c_uniform = float(np.clip(float(uniform_mix), 0.0, 1.0))
    c_class = float(np.clip(float(class_floor_mix), 0.0, 1.0))
    if c_uniform + c_class > 0.95:
        scale = 0.95 / max(c_uniform + c_class, 1e-300)
        c_uniform *= scale
        c_class *= scale
    c_lewis = max(0.0, 1.0 - c_uniform - c_class)
    p = c_lewis * p_lewis + c_class * p_class + c_uniform * p_uniform
    p = np.maximum(p, 0.0)
    p /= max(float(p.sum()), 1e-300)
    if m >= n:
        idx = np.arange(n, dtype=int)
    else:
        idx = rng.choice(np.arange(n), size=m, replace=False, p=p).astype(int)
    sample_weight = None
    if return_inverse_probability_weights:
        sample_weight = 1.0 / np.maximum(m * p[idx], 1e-300)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        q = float(weight_clip_quantile)
        if np.isfinite(q) and q > 0.0 and (q < 1.0) and (len(sample_weight) > 0):
            cap = float(np.quantile(sample_weight, q))
            sample_weight = np.minimum(sample_weight, cap)
        if bool(rescale_weight_sum_to_n) and len(sample_weight) > 0 and (float(sample_weight.sum()) > 0):
            sample_weight = sample_weight * (float(n) / float(sample_weight.sum()))
    info = {'global_sampling_note': 'global_lewis_classfloor_uniform_mixture_without_replacement', 'global_sampling_prob_min': float(np.min(p)), 'global_sampling_prob_max': float(np.max(p)), 'global_sampling_prob_effective_n': float(1.0 / max(float(np.sum(p * p)), 1e-300)), 'lewis_uniform_mix_used': float(c_uniform), 'theory_class_floor_mix_used': float(c_class), 'theory_lewis_mix_used': float(c_lewis), 'theory_weight_clip_quantile_used': float(weight_clip_quantile), 'sample_weight_used': bool(sample_weight is not None), 'sample_weight_sum': float(np.sum(sample_weight)) if sample_weight is not None else np.nan, 'sample_weight_min': float(np.min(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan, 'sample_weight_max': float(np.max(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan, 'sample_weight_mean': float(np.mean(sample_weight)) if sample_weight is not None and len(sample_weight) else np.nan}
    return (idx, sample_weight, info)

def mu_summary_value(mu_df: pd.DataFrame, stat: str) -> float:
    vals = pd.to_numeric(mu_df.get('mu', pd.Series(dtype=float)), errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    s = str(stat).lower()
    if s in {'max', 'maximum'}:
        return float(np.max(vals))
    if s in {'mean', 'avg'}:
        return float(np.mean(vals))
    if s in {'median', 'q50'}:
        return float(np.quantile(vals, 0.5))
    m = re.match('q([0-9]+(?:p[0-9]+)?|0?\\.[0-9]+)$', s)
    if m:
        raw = m.group(1).replace('p', '.')
        q = float(raw)
        if q > 1.0:
            q /= 100.0
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(vals, q))
    raise ValueError(f'Unknown mu summary stat: {stat!r}')

def theory_global_size_choices(mu_df: pd.DataFrame, n_features: int, n_train: int, stats: Sequence[str], epsilons: Sequence[float], alphas: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stat in stats:
        mu_val = mu_summary_value(mu_df, stat)
        for eps in epsilons:
            for alpha in alphas:
                size = int(math.ceil(float(alpha) * int(n_features) * float(mu_val) ** 2 / max(float(eps) ** 2, 1e-300)))
                size = max(1, min(size, int(n_train)))
                rows.append({'theory_mu_stat': str(stat), 'theory_mu_value': float(mu_val), 'theory_epsilon': float(eps), 'theory_alpha': float(alpha), 'theory_d_eff': int(n_features), 'n_train': int(n_train), 'theory_coreset_size': int(size), 'theory_coreset_ratio': float(size / max(int(n_train), 1))})
    return pd.DataFrame(rows)

def theory_method_name(stat: str, epsilon: float, alpha: float, mix: float) -> str:
    return f'corLewis_theory_{safe_name(stat)}_eps{float_tag(epsilon)}_alpha{float_tag(alpha)}_mix{float_tag(mix)}'

def theory_classfloor_method_name(stat: str, epsilon: float, alpha: float, mix: float, class_floor_mix: float) -> str:
    return f'corLewis_theory_classfloor_{safe_name(stat)}_eps{float_tag(epsilon)}_alpha{float_tag(alpha)}_mix{float_tag(mix)}_class{float_tag(class_floor_mix)}'

def sample_within_labels_by_lewis(y_train: Sequence[object], quotas: Mapping[str, int], lewis_weights: np.ndarray, seed: int, uniform_mix: float) -> np.ndarray:
    y = np.asarray(y_train).astype(str)
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    for cls in sorted(quotas.keys()):
        cls_idx = np.flatnonzero(y == str(cls))
        quota = min(int(quotas[cls]), len(cls_idx))
        if quota <= 0:
            continue
        if quota >= len(cls_idx):
            selected.append(cls_idx)
            continue
        ww = np.asarray(lewis_weights[cls_idx], dtype=np.float64)
        if not np.isfinite(ww).all() or ww.sum() <= 0:
            p = np.ones(len(cls_idx), dtype=np.float64) / len(cls_idx)
        else:
            p_lewis = ww / ww.sum()
            p_uniform = np.ones(len(cls_idx), dtype=np.float64) / len(cls_idx)
            p = (1.0 - uniform_mix) * p_lewis + uniform_mix * p_uniform
            p = np.maximum(p, 0)
            p /= max(p.sum(), 1e-300)
        selected.append(rng.choice(cls_idx, size=quota, replace=False, p=p))
    if not selected:
        return np.array([], dtype=int)
    out = np.concatenate(selected).astype(int)
    rng.shuffle(out)
    return out

def equal_label_quotas(y_train: Sequence[object], target_size: int, n_labels_full: int) -> Dict[str, int]:
    y = pd.Series(np.asarray(y_train).astype(str))
    n_per_label = int(target_size / max(n_labels_full, 1))
    counts = y.value_counts().to_dict()
    return {str(cls): int(min(count, n_per_label)) for (cls, count) in counts.items()}

def sample_mu_subset_indices(y_train: Sequence[object], max_rows: int, seed: int, mode: str) -> np.ndarray:
    y = np.asarray(y_train).astype(str)
    n = len(y)
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    (classes, counts) = np.unique(y, return_counts=True)
    mode = str(mode).lower()
    if mode == 'uniform':
        return np.sort(rng.choice(n, size=max_rows, replace=False)).astype(int)
    if mode == 'balanced':
        per_class = max(1, max_rows // max(len(classes), 1))
        selected = []
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            take = min(per_class, len(idx))
            if take > 0:
                selected.append(rng.choice(idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=int)
        if len(out) < max_rows:
            remaining = np.setdiff1d(np.arange(n), out, assume_unique=False)
            if len(remaining):
                extra = rng.choice(remaining, size=min(max_rows - len(out), len(remaining)), replace=False)
                out = np.concatenate([out, extra])
        if len(out) > max_rows:
            out = rng.choice(out, size=max_rows, replace=False)
        return np.sort(out).astype(int)
    if mode == 'stratified':
        selected = []
        for (cls, count) in zip(classes, counts):
            idx = np.flatnonzero(y == cls)
            take = max(1, int(round(max_rows * count / max(n, 1))))
            take = min(take, len(idx))
            selected.append(rng.choice(idx, size=take, replace=False))
        out = np.concatenate(selected) if selected else np.array([], dtype=int)
        if len(out) > max_rows:
            out = rng.choice(out, size=max_rows, replace=False)
        elif len(out) < max_rows:
            remaining = np.setdiff1d(np.arange(n), out, assume_unique=False)
            if len(remaining):
                extra = rng.choice(remaining, size=min(max_rows - len(out), len(remaining)), replace=False)
                out = np.concatenate([out, extra])
        return np.sort(out).astype(int)
    raise ValueError(f'Unknown mu row sampling mode: {mode!r}')

def choose_mu_classes_from_train(y_train: Sequence[object], classes_arg: str, min_pos: int) -> List[str]:
    y_all = pd.Series(np.asarray(y_train).astype(str))
    counts_train = y_all.value_counts()
    eligible = [str(c) for (c, cnt) in counts_train.items() if int(cnt) >= int(min_pos) and int(cnt) < len(y_all)]
    if str(classes_arg).lower() == 'all':
        return sorted(eligible)
    try:
        k = int(classes_arg)
        return [str(c) for c in counts_train.index if str(c) in eligible][:k]
    except ValueError:
        requested = [x.strip() for x in str(classes_arg).split(',') if x.strip()]
        return [x for x in requested if x in eligible]

def ovr_balanced_mu_subset_indices(y_train: Sequence[object], cls: str, seed: int, max_pos: int, neg_multiplier: float, max_neg: int, min_neg: int) -> np.ndarray:
    y = np.asarray(y_train).astype(str)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == str(cls))
    neg_idx = np.flatnonzero(y != str(cls))
    if len(pos_idx) == 0 or len(neg_idx) < int(min_neg):
        return np.array([], dtype=int)
    if int(max_pos) > 0:
        n_pos = min(len(pos_idx), int(max_pos))
    else:
        n_pos = len(pos_idx)
    pos_take = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos < len(pos_idx) else pos_idx
    desired_neg = int(math.ceil(float(neg_multiplier) * max(1, n_pos)))
    if int(max_neg) > 0:
        desired_neg = min(desired_neg, int(max_neg))
    desired_neg = max(int(min_neg), desired_neg)
    n_neg = min(len(neg_idx), desired_neg)
    neg_take = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg < len(neg_idx) else neg_idx
    out = np.concatenate([pos_take, neg_take]).astype(int)
    rng.shuffle(out)
    return np.sort(out)

def choose_mu_classes(y_subset: Sequence[object], y_train: Sequence[object], classes_arg: str, min_pos: int) -> List[str]:
    y_sub = pd.Series(np.asarray(y_subset).astype(str))
    y_all = pd.Series(np.asarray(y_train).astype(str))
    counts_sub = y_sub.value_counts()
    counts_train = y_all.value_counts()
    eligible = [str(c) for (c, cnt) in counts_sub.items() if cnt >= int(min_pos) and cnt < len(y_sub)]
    if str(classes_arg).lower() == 'all':
        return sorted(eligible)
    try:
        k = int(classes_arg)
        return [str(c) for c in counts_train.index if str(c) in eligible][:k]
    except ValueError:
        requested = [x.strip() for x in str(classes_arg).split(',') if x.strip()]
        return [x for x in requested if x in eligible]

def compute_mu_lp_gurobi_projection(A: np.ndarray, y_pm: np.ndarray, time_limit: Optional[float]=None, threads: int=1, mem_limit_gb: float=100.0, output_flag: bool=False, tolerance: float=1e-12) -> Dict[str, object]:
    t0 = now()
    out: Dict[str, object] = {'mu': np.nan, 'lp_status': np.nan, 'lp_success': False, 'lp_message': '', 'lp_runtime_sec': np.nan, 'lp_objective': np.nan}
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        out.update({'lp_message': f'gurobipy not available: {exc}', 'lp_runtime_sec': now() - t0})
        return out
    try:
        A = np.asarray(A, dtype=np.float64)
        y_pm = np.asarray(y_pm, dtype=np.float64)
        B = A * y_pm[:, None]
        (n, _) = B.shape
        (Q, _) = np.linalg.qr(B, mode='reduced')
        M = np.eye(n, dtype=np.float64) - Q @ Q.T
        if tolerance and tolerance > 0:
            M[np.abs(M) < tolerance] = 0.0
        M_sparse = sparse.csr_matrix(M)
        model = gp.Model('mu_projection_lp')
        model.Params.OutputFlag = 1 if output_flag else 0
        if time_limit is not None and time_limit > 0:
            model.Params.TimeLimit = float(time_limit)
        if threads and threads > 0:
            model.Params.Threads = int(threads)
        if mem_limit_gb and mem_limit_gb > 0:
            model.Params.MemLimit = float(mem_limit_gb)
        z_plus = model.addMVar(n, lb=0.0, name='z_plus')
        z_minus = model.addMVar(n, lb=0.0, name='z_minus')
        z = z_plus - z_minus
        model.addConstr(z_plus.sum() + z_minus.sum() <= 1.0, name='l1_norm')
        model.addConstr(M_sparse @ z == np.zeros(n), name='range_constraint')
        model.setObjective(z_plus.sum() - z_minus.sum(), GRB.MAXIMIZE)
        model.optimize()
        out['lp_status'] = int(model.Status)
        out['lp_runtime_sec'] = now() - t0
        out['lp_message'] = str(model.Status)
        if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and model.SolCount > 0:
            z_val = np.asarray(z_plus.X - z_minus.X, dtype=np.float64)
            pos = float(np.sum(np.maximum(z_val, 0.0)))
            neg = float(np.sum(np.maximum(-z_val, 0.0)))
            if neg <= 1e-12 and pos > 1e-12:
                mu = float('inf')
            elif neg <= 1e-12:
                mu = np.nan
            else:
                mu = pos / neg
            out.update({'mu': mu, 'lp_success': bool(model.Status == GRB.OPTIMAL), 'lp_objective': float(model.ObjVal) if model.SolCount > 0 else np.nan, 'z_pos_l1': pos, 'z_neg_l1': neg, 'range_basis_dim': int(Q.shape[1])})
        model.dispose()
        try:
            gp.disposeDefaultEnv()
        except Exception:
            pass
    except Exception as exc:
        out.update({'lp_message': f'gurobi mu error: {exc}', 'lp_runtime_sec': now() - t0})
    return out

def compute_mu_same_embedding(X_train_pca: np.ndarray, y_train: Sequence[object], classes: Sequence[str], seed: int, n_pcs: int, args: argparse.Namespace) -> pd.DataFrame:
    y = np.asarray(y_train).astype(str)
    intercept_added = bool(args.mu_compute_intercept)
    mode = str(args.mu_row_sampling).lower()
    if mode == 'ovr_balanced':
        chosen = choose_mu_classes_from_train(y, args.mu_classes, min_pos=args.mu_min_pos)
        rows: List[Dict[str, object]] = []
        LOGGER.info('Computing fast OvR-balanced same-embedding mu: classes=%d intercept=%s max_pos=%d neg_mult=%.3g max_neg=%d', len(chosen), intercept_added, int(args.mu_ovr_max_pos), float(args.mu_ovr_neg_multiplier), int(args.mu_ovr_max_neg))
        for (ii, cls) in enumerate(chosen, start=1):
            idx = ovr_balanced_mu_subset_indices(y, cls=str(cls), seed=seed + 13579 + int(n_pcs) + stable_int_hash(str(cls), modulo=1000000), max_pos=int(args.mu_ovr_max_pos), neg_multiplier=float(args.mu_ovr_neg_multiplier), max_neg=int(args.mu_ovr_max_neg), min_neg=int(args.mu_min_neg))
            if len(idx) == 0:
                continue
            X_sub = np.asarray(X_train_pca[idx], dtype=np.float64)
            y_sub = y[idx]
            if intercept_added:
                X_lp = np.hstack([X_sub, np.ones((X_sub.shape[0], 1), dtype=X_sub.dtype)])
            else:
                X_lp = X_sub
            y_pm = np.where(y_sub == str(cls), 1.0, -1.0)
            pos = int(np.sum(y_pm == 1.0))
            neg = int(np.sum(y_pm == -1.0))
            if pos < int(args.mu_min_pos) or neg < int(args.mu_min_neg):
                continue
            LOGGER.info('[mu same embedding fast] %d/%d class=%s rows=%d pos=%d neg=%d', ii, len(chosen), cls, len(idx), pos, neg)
            res = compute_mu_lp_gurobi_projection(X_lp, y_pm, time_limit=args.mu_lp_time_limit if args.mu_lp_time_limit > 0 else None, threads=int(args.gurobi_threads), mem_limit_gb=float(args.gurobi_mem_limit_gb), output_flag=bool(args.gurobi_output))
            mu_val = res.get('mu', np.nan)
            row = {'dataset_name': args.dataset_name, 'seed': int(seed), 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'mu_solver': 'gurobi_projection_same_embedding', 'mu_source': 'computed_same_embedding_current_run_ovr_balanced_fast', 'intercept_added': bool(intercept_added), 'class_label': str(cls), 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(idx)), 'mu_row_sampling': str(args.mu_row_sampling), 'mu_ovr_max_pos': int(args.mu_ovr_max_pos), 'mu_ovr_neg_multiplier': float(args.mu_ovr_neg_multiplier), 'mu_ovr_max_neg': int(args.mu_ovr_max_neg), 'pos_count_mu_subset': pos, 'neg_count_mu_subset': neg, 'pos_count_train': int(np.sum(y == str(cls))), 'pos_fraction_train': float(np.mean(y == str(cls))), 'n_features_mu': int(X_lp.shape[1]), 'mu': mu_val, 'mu_squared': float(mu_val) ** 2 if np.isfinite(mu_val) else mu_val, 'memory_rss_mb': rss_mb()}
            row.update(res)
            rows.append(row)
        found = {str(r['class_label']) for r in rows}
        for cls in classes:
            if str(cls) not in found:
                rows.append({'dataset_name': args.dataset_name, 'seed': int(seed), 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'mu_solver': 'gurobi_projection_same_embedding', 'mu_source': 'computed_same_embedding_current_run_missing_class_fallback_one', 'intercept_added': bool(intercept_added), 'class_label': str(cls), 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': 0, 'mu_row_sampling': str(args.mu_row_sampling), 'pos_count_train': int(np.sum(y == str(cls))), 'pos_fraction_train': float(np.mean(y == str(cls))), 'n_features_mu': int(X_train_pca.shape[1] + (1 if intercept_added else 0)), 'mu': 1.0, 'mu_squared': 1.0, 'lp_success': False, 'lp_message': 'fallback_mu_1_missing_or_insufficient_subset_class'})
        return pd.DataFrame(rows)
    idx = sample_mu_subset_indices(y, max_rows=int(args.mu_rows), seed=seed + 13579 + int(n_pcs), mode=args.mu_row_sampling)
    X_sub = np.asarray(X_train_pca[idx], dtype=np.float64)
    y_sub = y[idx]
    if intercept_added:
        X_lp = np.hstack([X_sub, np.ones((X_sub.shape[0], 1), dtype=X_sub.dtype)])
    else:
        X_lp = X_sub
    chosen = choose_mu_classes(y_sub, y, args.mu_classes, min_pos=args.mu_min_pos)
    rows: List[Dict[str, object]] = []
    LOGGER.info('Computing same-embedding OvR mu: rows=%d classes=%d intercept=%s', len(idx), len(chosen), intercept_added)
    for (ii, cls) in enumerate(chosen, start=1):
        y_pm = np.where(y_sub == str(cls), 1.0, -1.0)
        pos = int(np.sum(y_pm == 1.0))
        neg = int(np.sum(y_pm == -1.0))
        if pos < int(args.mu_min_pos) or neg < int(args.mu_min_neg):
            continue
        LOGGER.info('[mu same embedding] %d/%d class=%s pos=%d neg=%d', ii, len(chosen), cls, pos, neg)
        res = compute_mu_lp_gurobi_projection(X_lp, y_pm, time_limit=args.mu_lp_time_limit if args.mu_lp_time_limit > 0 else None, threads=int(args.gurobi_threads), mem_limit_gb=float(args.gurobi_mem_limit_gb), output_flag=bool(args.gurobi_output))
        mu_val = res.get('mu', np.nan)
        row = {'dataset_name': args.dataset_name, 'seed': int(seed), 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'mu_solver': 'gurobi_projection_same_embedding', 'mu_source': 'computed_same_embedding_current_run', 'intercept_added': bool(intercept_added), 'class_label': str(cls), 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(idx)), 'mu_row_sampling': str(args.mu_row_sampling), 'pos_count_mu_subset': pos, 'neg_count_mu_subset': neg, 'pos_count_train': int(np.sum(y == str(cls))), 'pos_fraction_train': float(np.mean(y == str(cls))), 'n_features_mu': int(X_lp.shape[1]), 'mu': mu_val, 'mu_squared': float(mu_val) ** 2 if np.isfinite(mu_val) else mu_val, 'memory_rss_mb': rss_mb()}
        row.update(res)
        rows.append(row)
    found = {str(r['class_label']) for r in rows}
    for cls in classes:
        if str(cls) not in found:
            rows.append({'dataset_name': args.dataset_name, 'seed': int(seed), 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'mu_solver': 'gurobi_projection_same_embedding', 'mu_source': 'computed_same_embedding_current_run_missing_class_fallback_one', 'intercept_added': bool(intercept_added), 'class_label': str(cls), 'mu_rows_requested': int(args.mu_rows), 'mu_rows_actual': int(len(idx)), 'mu_row_sampling': str(args.mu_row_sampling), 'pos_count_train': int(np.sum(y == str(cls))), 'pos_fraction_train': float(np.mean(y == str(cls))), 'n_features_mu': int(X_lp.shape[1]), 'mu': 1.0, 'mu_squared': 1.0, 'lp_success': False, 'lp_message': 'fallback_mu_1_missing_or_insufficient_subset_class'})
    return pd.DataFrame(rows)

def load_mu_by_class(path: Optional[str], classes: Sequence[str], seed: int, n_pcs: int, prefer_intercept_added: Optional[bool], solver: str) -> pd.DataFrame:
    classes = [str(c) for c in classes]
    if not path:
        return pd.DataFrame({'class_label': classes, 'mu_raw': np.ones(len(classes)), 'mu': np.ones(len(classes)), 'mu_source': 'all_one_no_mu_file', 'mu_is_infinite': False, 'mu_is_missing': False, 'mu_allocation_note': 'all_one_no_mu_file'})
    p = Path(path)
    if not p.exists():
        LOGGER.warning('mu_by_class file not found: %s; using mu=1 for all classes', p)
        return pd.DataFrame({'class_label': classes, 'mu_raw': np.ones(len(classes)), 'mu': np.ones(len(classes)), 'mu_source': 'all_one_missing_mu_file', 'mu_is_infinite': False, 'mu_is_missing': False, 'mu_allocation_note': 'all_one_missing_mu_file'})
    df = pd.read_csv(p)
    if 'class_label' not in df.columns or 'mu' not in df.columns:
        LOGGER.warning('mu_by_class file lacks class_label/mu columns; using mu=1')
        return pd.DataFrame({'class_label': classes, 'mu_raw': np.ones(len(classes)), 'mu': np.ones(len(classes)), 'mu_source': 'all_one_bad_mu_file', 'mu_is_infinite': False, 'mu_is_missing': False, 'mu_allocation_note': 'all_one_bad_mu_file'})
    sub = df.copy()
    if 'n_pcs' in sub.columns:
        sub = sub[pd.to_numeric(sub['n_pcs'], errors='coerce') == int(n_pcs)]
    if 'seed' in sub.columns:
        seed_sub = sub[pd.to_numeric(sub['seed'], errors='coerce') == int(seed)]
        if not seed_sub.empty:
            sub = seed_sub
    if prefer_intercept_added is not None and 'intercept_added' in sub.columns:
        mask = sub['intercept_added'].astype(str).str.lower().isin({'true', '1', 'yes'})
        sub2 = sub[mask == bool(prefer_intercept_added)]
        if not sub2.empty:
            sub = sub2
    if solver and solver.lower() not in {'', 'any', 'first'} and ('mu_solver' in sub.columns):
        sub2 = sub[sub['mu_solver'].astype(str) == solver]
        if not sub2.empty:
            sub = sub2
    sub['class_label'] = sub['class_label'].astype(str)
    sub['mu'] = pd.to_numeric(sub['mu'], errors='coerce')
    rows = []
    for (cls, g) in sub.groupby('class_label', dropna=False):
        vals = pd.to_numeric(g['mu'], errors='coerce')
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            mu_val = float(np.median(finite))
        elif np.isposinf(vals).any():
            mu_val = float('inf')
        else:
            mu_val = np.nan
        rows.append({'class_label': str(cls), 'mu_raw': mu_val})
    agg = pd.DataFrame(rows)
    out = pd.DataFrame({'class_label': classes}).merge(agg, on='class_label', how='left')
    raw = pd.to_numeric(out['mu_raw'], errors='coerce')
    out['mu_raw'] = raw
    out['mu_is_infinite'] = np.isposinf(raw)
    out['mu_is_missing'] = raw.isna() | np.isneginf(raw)
    out['mu'] = raw
    out['mu_source'] = str(p)
    out['mu_allocation_note'] = np.where(out['mu_is_infinite'], 'positive_infinite_mu_to_be_capped', np.where(out['mu_is_missing'], 'missing_mu_to_be_filled_with_1', 'finite_mu'))
    return out

def mu_label_quotas(y_train: Sequence[object], mu_df: pd.DataFrame, target_size: int, mu_power: float, mu_floor: float, mu_cap: float, min_per_class: int) -> Tuple[Dict[str, int], pd.DataFrame]:
    y = pd.Series(np.asarray(y_train).astype(str))
    counts = y.value_counts().rename_axis('class_label').reset_index(name='train_class_count')
    counts['class_label'] = counts['class_label'].astype(str)
    keep_cols = [c for c in ['class_label', 'mu', 'mu_raw', 'mu_source', 'mu_is_infinite', 'mu_is_missing', 'mu_allocation_note'] if c in mu_df.columns]
    m = counts.merge(mu_df[keep_cols], on='class_label', how='left')
    raw_mu = pd.to_numeric(m.get('mu_raw', m.get('mu', pd.Series(np.nan, index=m.index))), errors='coerce')
    m['mu_raw'] = raw_mu
    m['mu_is_infinite'] = np.isposinf(raw_mu)
    m['mu_is_missing'] = raw_mu.isna() | np.isneginf(raw_mu)
    mu_eff = raw_mu.copy()
    mu_eff = mu_eff.mask(np.isposinf(mu_eff), float(mu_cap))
    mu_eff = mu_eff.mask(np.isneginf(mu_eff), np.nan)
    mu_eff = mu_eff.fillna(1.0)
    clipped = mu_eff.clip(lower=float(mu_floor), upper=float(mu_cap))
    m['mu'] = mu_eff
    m['mu_clipped'] = clipped
    m['mu_allocation_note'] = np.where(m['mu_is_infinite'], 'positive_infinite_mu_capped_to_mu_cap', np.where(m['mu_is_missing'], 'missing_mu_filled_with_1', 'finite_mu'))
    m['allocation_score'] = np.power(clipped, float(mu_power))
    m.loc[~np.isfinite(m['allocation_score']), 'allocation_score'] = 1.0
    m.loc[m['allocation_score'] <= 0, 'allocation_score'] = 1.0
    K = len(m)
    target_size = int(min(max(target_size, 1), int(m['train_class_count'].sum())))
    if K == 0:
        return ({}, m)
    min_eff = int(max(0, min_per_class))
    if min_eff * K > target_size:
        min_eff = max(0, target_size // K)
    quotas = np.minimum(m['train_class_count'].to_numpy(dtype=int), min_eff).astype(int)
    remaining = target_size - int(quotas.sum())
    capacity = m['train_class_count'].to_numpy(dtype=int) - quotas
    scores = m['allocation_score'].to_numpy(dtype=float)
    while remaining > 0 and np.any(capacity > 0):
        active = capacity > 0
        active_scores = scores[active]
        if active_scores.sum() <= 0 or not np.isfinite(active_scores).all():
            active_scores = np.ones(active.sum(), dtype=float)
        raw = remaining * active_scores / active_scores.sum()
        add = np.floor(raw).astype(int)
        if add.sum() == 0:
            order = np.argsort(-(raw - np.floor(raw)))
            add[order[:remaining]] = 1
        add = np.minimum(add, capacity[active])
        if add.sum() == 0:
            break
        quotas[np.flatnonzero(active)] += add
        capacity = m['train_class_count'].to_numpy(dtype=int) - quotas
        remaining = target_size - int(quotas.sum())
    m['target_class_quota'] = quotas.astype(int)
    m['actual_class_quota'] = quotas.astype(int)
    m['target_total_size'] = target_size
    m['actual_total_size'] = int(quotas.sum())
    quota_dict = dict(zip(m['class_label'].astype(str), m['actual_class_quota'].astype(int)))
    return (quota_dict, m)

def blended_mu_label_quotas(y_train: Sequence[object], mu_df: pd.DataFrame, target_size: int, n_labels_full: int, eta: float, mu_power: float, mu_floor: float, mu_cap: float, min_per_class: int) -> Tuple[Dict[str, int], pd.DataFrame]:
    eta = float(np.clip(float(eta), 0.0, 1.0))
    equal = equal_label_quotas(y_train, target_size, n_labels_full)
    (mu_quota, m) = mu_label_quotas(y_train=y_train, mu_df=mu_df, target_size=target_size, mu_power=mu_power, mu_floor=mu_floor, mu_cap=mu_cap, min_per_class=min_per_class)
    if m.empty:
        return (equal, m)
    m = m.copy()
    m['equal_class_quota'] = m['class_label'].astype(str).map(equal).fillna(0).astype(int)
    m['pure_mu_class_quota'] = m['class_label'].astype(str).map(mu_quota).fillna(0).astype(int)
    raw = (1.0 - eta) * m['equal_class_quota'].to_numpy(dtype=float) + eta * m['pure_mu_class_quota'].to_numpy(dtype=float)
    counts = m['train_class_count'].to_numpy(dtype=int)
    quotas = np.minimum(np.floor(raw).astype(int), counts)
    target_size = int(min(max(target_size, 1), int(counts.sum())))
    remaining = target_size - int(quotas.sum())
    capacity = counts - quotas
    frac = raw - np.floor(raw)
    score = pd.to_numeric(m.get('allocation_score', pd.Series(1.0, index=m.index)), errors='coerce').fillna(1.0).to_numpy(float)
    while remaining > 0 and np.any(capacity > 0):
        active_idx = np.flatnonzero(capacity > 0)
        order = active_idx[np.lexsort((-score[active_idx], -frac[active_idx]))]
        if len(order) == 0:
            break
        take_n = min(remaining, len(order))
        quotas[order[:take_n]] += 1
        capacity = counts - quotas
        remaining = target_size - int(quotas.sum())
    over = int(quotas.sum()) - target_size
    while over > 0 and np.any(quotas > 0):
        active_idx = np.flatnonzero(quotas > 0)
        order = active_idx[np.lexsort((score[active_idx], -quotas[active_idx]))]
        take_n = min(over, len(order))
        quotas[order[:take_n]] -= 1
        over = int(quotas.sum()) - target_size
    m['mu_blend_eta'] = eta
    m['target_class_quota'] = quotas.astype(int)
    m['actual_class_quota'] = quotas.astype(int)
    m['target_total_size'] = target_size
    m['actual_total_size'] = int(quotas.sum())
    m['quota_rule_detail'] = 'blend_equal_and_mu_powered'
    quota_dict = dict(zip(m['class_label'].astype(str), m['actual_class_quota'].astype(int)))
    return (quota_dict, m)

def biological_recall_metrics(y_true_encoded: np.ndarray, y_pred_encoded: np.ndarray, y_train_full: Sequence[object], classes: Sequence[str], bottom_fraction: float=0.25, worst_k: int=10) -> Dict[str, object]:
    y_true = np.asarray(y_true_encoded, dtype=int)
    y_pred = np.asarray(y_pred_encoded, dtype=int)
    class_names = list(map(str, classes))
    train_counts = pd.Series(np.asarray(y_train_full).astype(str)).value_counts()
    rows = []
    for (j, cls) in enumerate(class_names):
        mask = y_true == j
        recall = float(np.mean(y_pred[mask] == j)) if mask.sum() > 0 else np.nan
        rows.append({'class_label': cls, 'class_encoded': j, 'train_count': int(train_counts.get(cls, 0)), 'recall': recall})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df = df.sort_values(['train_count', 'class_label'])
    k_bottom = max(1, int(math.ceil(float(bottom_fraction) * len(df))))
    bottom = df.head(k_bottom)
    recall_vals = pd.to_numeric(df['recall'], errors='coerce').dropna().to_numpy(float)
    worst = np.sort(recall_vals)[:max(1, min(int(worst_k), len(recall_vals)))] if len(recall_vals) else np.array([], dtype=float)
    return {'bottom25_recall_mean': float(bottom['recall'].mean()), 'bottom25_recall_median': float(bottom['recall'].median()), 'bottom25_zero_recall_classes': int((bottom['recall'].fillna(0.0) <= 0).sum()), 'worst10_recall_mean': float(np.mean(worst)) if len(worst) else np.nan, 'worst10_recall_min': float(np.min(worst)) if len(worst) else np.nan, 'zero_recall_classes': int((df['recall'].fillna(0.0) <= 0).sum()), 'n_bottom25_classes': int(len(bottom)), 'n_eval_classes': int(len(df))}

def selected_cell_sample_rows(base: Mapping[str, object], obs_names: Sequence[str], y_train: Sequence[object], idx: np.ndarray, X_train_pca: np.ndarray, lewis_weights: Optional[np.ndarray], mu_lookup: Optional[Mapping[str, float]], max_rows: int, seed: int) -> List[Dict[str, object]]:
    if max_rows <= 0 or len(idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx, dtype=int)
    if len(idx) > max_rows:
        keep = rng.choice(idx, size=max_rows, replace=False)
    else:
        keep = idx
    y_arr = np.asarray(y_train).astype(str)
    obs_arr = np.asarray(obs_names).astype(str)
    rows: List[Dict[str, object]] = []
    q = None
    if lewis_weights is not None:
        w = np.asarray(lewis_weights, dtype=float)
        q = w / max(float(np.nansum(w)), 1e-300)
    for i in keep:
        cls = str(y_arr[i])
        row = dict(base)
        row.update({'obs_name': str(obs_arr[i]), 'class_label': cls, 'train_index': int(i), 'svd1': float(X_train_pca[i, 0]) if X_train_pca.shape[1] > 0 else np.nan, 'svd2': float(X_train_pca[i, 1]) if X_train_pca.shape[1] > 1 else np.nan, 'mu': float(mu_lookup.get(cls, np.nan)) if mu_lookup is not None else np.nan, 'lewis_weight': float(lewis_weights[i]) if lewis_weights is not None else np.nan, 'lewis_prob_over_uniform': float(q[i] * len(q)) if q is not None else np.nan})
        rows.append(row)
    return rows

def fit_logistic_regression_github(X_train_pca: np.ndarray, y_train_encoded: np.ndarray, C: float, solver: str, max_iter: int, n_jobs: int, sample_weight: Optional[np.ndarray]=None) -> LogisticRegression:
    clf = LogisticRegression(C=float(C), multi_class='ovr', solver=solver, max_iter=int(max_iter), n_jobs=int(n_jobs))
    if sample_weight is None:
        clf.fit(np.asarray(X_train_pca, dtype=np.float64), y_train_encoded)
    else:
        clf.fit(np.asarray(X_train_pca, dtype=np.float64), y_train_encoded, sample_weight=np.asarray(sample_weight, dtype=np.float64))
    return clf

def _infer_n_classes_full(clf: LogisticRegression, y_test_encoded: Optional[np.ndarray]=None, y_train_full_encoded: Optional[np.ndarray]=None, y_core_encoded: Optional[np.ndarray]=None, n_classes_full: Optional[int]=None) -> int:
    if n_classes_full is not None and int(n_classes_full) > 0:
        return int(n_classes_full)
    max_label = -1
    for arr in (getattr(clf, 'classes_', None), y_test_encoded, y_train_full_encoded, y_core_encoded):
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size:
            try:
                max_label = max(max_label, int(np.nanmax(a.astype(int))))
            except Exception:
                pass
    return int(max_label + 1) if max_label >= 0 else int(np.asarray(clf.coef_).shape[0])

def align_clf_coef_intercept_to_global_classes(clf: LogisticRegression, n_classes_full: int, missing_intercept: float=-40.0, missing_coef_value: float=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef = np.asarray(clf.coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    present_classes = np.asarray(getattr(clf, 'classes_', np.arange(coef.shape[0])), dtype=int)
    n_features = int(coef.shape[1]) if coef.ndim == 2 else 0
    B = np.full((int(n_classes_full), n_features), float(missing_coef_value), dtype=np.float64)
    b = np.full(int(n_classes_full), float(missing_intercept), dtype=np.float64)
    present_mask = np.zeros(int(n_classes_full), dtype=bool)
    if coef.shape[0] == 1 and present_classes.size == 2:
        (c0, c1) = (int(present_classes[0]), int(present_classes[1]))
        if 0 <= c1 < n_classes_full:
            B[c1, :] = coef[0]
            b[c1] = intercept[0]
            present_mask[c1] = True
        if 0 <= c0 < n_classes_full:
            B[c0, :] = -coef[0]
            b[c0] = -intercept[0]
            present_mask[c0] = True
    else:
        rows = min(coef.shape[0], present_classes.size)
        for row in range(rows):
            c = int(present_classes[row])
            if 0 <= c < n_classes_full:
                B[c, :] = coef[row]
                b[c] = intercept[row]
                present_mask[c] = True
    return (B, b, present_mask)

def expand_coef_to_gene_space(clf: LogisticRegression, svd: TruncatedSVD) -> Tuple[np.ndarray, np.ndarray]:
    expanded_coef = np.asarray(clf.coef_, dtype=np.float64).dot(np.asarray(svd.components_, dtype=np.float64))
    expanded_intercept = np.asarray(clf.intercept_, dtype=np.float64).copy()
    return (expanded_coef, expanded_intercept)

def expand_coef_to_gene_space_aligned(clf: LogisticRegression, svd: TruncatedSVD, n_classes_full: int, missing_intercept: float=-40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (B_pca, b, present_mask) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=missing_intercept)
    expanded_coef = B_pca.dot(np.asarray(svd.components_, dtype=np.float64))
    return (expanded_coef, b, present_mask)

def decision_scores_original_space(X_original, expanded_coef: np.ndarray, expanded_intercept: np.ndarray) -> np.ndarray:
    scores = X_original.dot(expanded_coef.T)
    if sparse.issparse(scores):
        scores = scores.toarray()
    scores = np.asarray(scores, dtype=np.float64)
    scores += expanded_intercept[None, :]
    return scores

def decision_scores_pca_space(X_pca: np.ndarray, clf: LogisticRegression) -> np.ndarray:
    scores = np.asarray(X_pca, dtype=np.float64).dot(np.asarray(clf.coef_, dtype=np.float64).T)
    scores += np.asarray(clf.intercept_, dtype=np.float64)[None, :]
    return scores

def decision_scores_pca_space_aligned(X_pca: np.ndarray, clf: LogisticRegression, n_classes_full: int, missing_intercept: float=-40.0) -> np.ndarray:
    (B, b, _) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=missing_intercept)
    scores = np.asarray(X_pca, dtype=np.float64).dot(B.T)
    scores += b[None, :]
    return scores

def softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    z = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)

def sigmoid_ovr_probs_from_scores(scores: np.ndarray) -> np.ndarray:
    P = 1.0 / (1.0 + np.exp(-np.clip(scores, -40, 40)))
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)

def custom_ovr_log_loss_weighted(true_labels: np.ndarray, predicted_probs: np.ndarray) -> float:
    y = np.asarray(true_labels, dtype=int)
    P = np.clip(np.asarray(predicted_probs, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    (n_samples, n_classes) = P.shape
    class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-15)
    total_loss = 0.0
    for c in range(n_classes):
        y_binary = (y == c).astype(float)
        class_loss = -np.sum(y_binary * np.log(P[:, c]) + (1.0 - y_binary) * np.log(1.0 - P[:, c]))
        total_loss += class_weights[c] * class_loss
    return float(total_loss / max(n_samples, 1))

def custom_ovr_log_loss_unweighted(true_labels: np.ndarray, predicted_probs: np.ndarray) -> float:
    y = np.asarray(true_labels, dtype=int)
    P = np.clip(np.asarray(predicted_probs, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    (n_samples, n_classes) = P.shape
    total_loss = 0.0
    for c in range(n_classes):
        y_binary = (y == c).astype(float)
        total_loss += -np.sum(y_binary * np.log(P[:, c]) + (1.0 - y_binary) * np.log(1.0 - P[:, c]))
    return float(total_loss / max(n_samples, 1))

def ovr_logistic_objective_from_scores(scores: np.ndarray, y_encoded: np.ndarray, C: float, coef: Optional[np.ndarray]=None) -> Dict[str, float]:
    y = np.asarray(y_encoded, dtype=int)
    (n, K) = scores.shape
    unreg = 0.0
    for c in range(K):
        y_pm = np.where(y == c, 1.0, -1.0)
        unreg += float(np.logaddexp(0.0, -y_pm * scores[:, c]).sum())
    reg = 0.0
    if coef is not None:
        reg = 0.5 / float(C) * float(np.sum(np.asarray(coef, dtype=np.float64) ** 2))
    return {'objective_unregularized': float(unreg), 'objective_regularization': float(reg), 'objective_total': float(unreg + reg), 'objective_per_cell': float((unreg + reg) / max(n, 1)), 'objective_unregularized_per_cell': float(unreg / max(n, 1))}

def ovr_logistic_objective_from_scores_weighted(scores: np.ndarray, y_encoded: np.ndarray, C: float, coef: Optional[np.ndarray]=None, sample_weight: Optional[np.ndarray]=None) -> Dict[str, float]:
    y = np.asarray(y_encoded, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    (n, K) = scores.shape
    if sample_weight is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.shape[0] != n:
            raise ValueError(f'sample_weight length {sw.shape[0]} does not match n={n}')
    sw = np.where(np.isfinite(sw), sw, 0.0)
    unreg = 0.0
    for c in range(K):
        y_pm = np.where(y == c, 1.0, -1.0)
        losses = np.logaddexp(0.0, -y_pm * scores[:, c])
        unreg += float(np.sum(sw * losses))
    reg = 0.0
    if coef is not None:
        reg = 0.5 / float(C) * float(np.sum(np.asarray(coef, dtype=np.float64) ** 2))
    sw_sum = float(np.sum(sw))
    return {'objective_unregularized': float(unreg), 'objective_regularization': float(reg), 'objective_total': float(unreg + reg), 'objective_per_weight': float((unreg + reg) / max(sw_sum, 1e-300)), 'objective_unregularized_per_weight': float(unreg / max(sw_sum, 1e-300)), 'sample_weight_sum': float(sw_sum)}

def weighted_loss_preservation_diagnostics(full_clf: LogisticRegression, core_clf: LogisticRegression, X_train_full_pca: np.ndarray, y_train_full_encoded: np.ndarray, X_core_pca: np.ndarray, y_core_encoded: np.ndarray, sample_weight_core: Optional[np.ndarray], C: float, n_classes_full: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    sw = None if sample_weight_core is None else np.asarray(sample_weight_core, dtype=np.float64)
    for (tag, clf) in (('full_beta', full_clf), ('core_beta', core_clf)):
        (B, b, _) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
        full_scores = np.asarray(X_train_full_pca, dtype=np.float64).dot(B.T) + b[None, :]
        core_scores = np.asarray(X_core_pca, dtype=np.float64).dot(B.T) + b[None, :]
        full_obj = ovr_logistic_objective_from_scores(full_scores, y_train_full_encoded, C=C, coef=B)
        core_w_obj = ovr_logistic_objective_from_scores_weighted(core_scores, y_core_encoded, C=C, coef=B, sample_weight=sw)
        for component in ('unregularized', 'total'):
            fval = float(full_obj[f'objective_{component}'])
            cval = float(core_w_obj[f'objective_{component}'])
            out[f'loss_preservation_{tag}_full_{component}'] = fval
            out[f'loss_preservation_{tag}_coreset_weighted_{component}'] = cval
            out[f'loss_preservation_{tag}_relerr_{component}'] = float(abs(cval - fval) / max(abs(fval), 1e-300))
            out[f'loss_preservation_{tag}_signed_relerr_{component}'] = float((cval - fval) / max(abs(fval), 1e-300))
    return out

def evaluate_model(clf: LogisticRegression, svd: TruncatedSVD, X_test_original, X_test_pca: np.ndarray, y_test_encoded: np.ndarray, X_train_full_pca: np.ndarray, y_train_full_encoded: np.ndarray, X_core_pca: np.ndarray, y_core_encoded: np.ndarray, C: float, n_classes_full: Optional[int]=None) -> Dict[str, object]:
    n_classes_full = _infer_n_classes_full(clf, y_test_encoded=y_test_encoded, y_train_full_encoded=y_train_full_encoded, y_core_encoded=y_core_encoded, n_classes_full=n_classes_full)
    (expanded_coef, expanded_intercept, present_mask) = expand_coef_to_gene_space_aligned(clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    scores_test_original = decision_scores_original_space(X_test_original, expanded_coef, expanded_intercept)
    probs_test_softmax = softmax_from_scores(scores_test_original)
    pred_test = np.argmax(probs_test_softmax, axis=1)
    scores_test_pca = decision_scores_pca_space_aligned(X_test_pca, clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    probs_test_ovr = sigmoid_ovr_probs_from_scores(scores_test_pca)
    scores_train_full = decision_scores_pca_space_aligned(X_train_full_pca, clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    probs_train_full = sigmoid_ovr_probs_from_scores(scores_train_full)
    scores_core = decision_scores_pca_space_aligned(X_core_pca, clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    probs_core = sigmoid_ovr_probs_from_scores(scores_core)
    (B_aligned, _, _) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    out: Dict[str, object] = {'accuracy': float(accuracy_score(y_test_encoded, pred_test)), 'balanced_accuracy': float(balanced_accuracy_score(y_test_encoded, pred_test)), 'macro_f1': float(f1_score(y_test_encoded, pred_test, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_test_encoded, pred_test, average='weighted', zero_division=0)), 'micro_f1': float(f1_score(y_test_encoded, pred_test, average='micro', zero_division=0)), 'macro_precision': float(precision_score(y_test_encoded, pred_test, average='macro', zero_division=0)), 'macro_recall': float(recall_score(y_test_encoded, pred_test, average='macro', zero_division=0)), 'weighted_precision': float(precision_score(y_test_encoded, pred_test, average='weighted', zero_division=0)), 'weighted_recall': float(recall_score(y_test_encoded, pred_test, average='weighted', zero_division=0)), 'micro_precision': float(precision_score(y_test_encoded, pred_test, average='micro', zero_division=0)), 'micro_recall': float(recall_score(y_test_encoded, pred_test, average='micro', zero_division=0)), 'matthews_corrcoef': float(matthews_corrcoef(y_test_encoded, pred_test)), 'cohen_kappa': float(cohen_kappa_score(y_test_encoded, pred_test)), 'ovr_log_loss_weighted_test_original_softmax': custom_ovr_log_loss_weighted(y_test_encoded, probs_test_softmax), 'ovr_log_loss_unweighted_test_original_softmax': custom_ovr_log_loss_unweighted(y_test_encoded, probs_test_softmax), 'ovr_log_loss_weighted_test_pca': custom_ovr_log_loss_weighted(y_test_encoded, probs_test_ovr), 'ovr_log_loss_unweighted_test_pca': custom_ovr_log_loss_unweighted(y_test_encoded, probs_test_ovr), 'ovr_log_loss_weighted_train_full_pca': custom_ovr_log_loss_weighted(y_train_full_encoded, probs_train_full), 'ovr_log_loss_unweighted_train_full_pca': custom_ovr_log_loss_unweighted(y_train_full_encoded, probs_train_full), 'ovr_log_loss_weighted_train_coreset_pca': custom_ovr_log_loss_weighted(y_core_encoded, probs_core), 'ovr_log_loss_unweighted_train_coreset_pca': custom_ovr_log_loss_unweighted(y_core_encoded, probs_core), 'test_prediction_argmax_source': 'original_space_expanded_coef_softmax', 'test_proba_source': 'original_space_expanded_coef_softmax'}
    try:
        out['multiclass_log_loss_test_original_softmax'] = float(log_loss(y_test_encoded, probs_test_softmax, labels=list(range(probs_test_softmax.shape[1]))))
    except Exception:
        out['multiclass_log_loss_test_original_softmax'] = np.nan
    out.update(safe_auc_metrics(y_test_encoded, probs_test_softmax, prefix='original_softmax'))
    full_obj = ovr_logistic_objective_from_scores(scores_train_full, y_train_full_encoded, C=C, coef=B_aligned)
    core_obj = ovr_logistic_objective_from_scores(scores_core, y_core_encoded, C=C, coef=B_aligned)
    for (k, v) in full_obj.items():
        out[f'objective_full_train_{k}'] = v
    for (k, v) in core_obj.items():
        out[f'objective_coreset_train_{k}'] = v
    out['model_classes_present_count'] = int(present_mask.sum())
    out['model_classes_missing_count'] = int(n_classes_full - present_mask.sum())
    out['pred_test_encoded'] = pred_test
    out['probs_test_softmax'] = probs_test_softmax
    out['expanded_coef'] = expanded_coef
    out['expanded_intercept'] = expanded_intercept
    return out

def coefficient_summary_metrics(clf: LogisticRegression, full_clf: LogisticRegression, svd: TruncatedSVD, ks: Sequence[int]=(20, 50, 100)) -> Dict[str, float]:
    n_classes_full = _infer_n_classes_full(full_clf)
    (B, _, present_mask) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (R, _, ref_present_mask) = align_clf_coef_intercept_to_global_classes(full_clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (Bg, _, _) = expand_coef_to_gene_space_aligned(clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (Rg, _, _) = expand_coef_to_gene_space_aligned(full_clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    valid_mask = present_mask & ref_present_mask
    out: Dict[str, float] = {'beta_compared_classes': int(valid_mask.sum()), 'beta_missing_classes': int(n_classes_full - valid_mask.sum())}
    for (space, X, Y) in [('pca', B, R), ('gene', Bg, Rg)]:
        cosines: List[float] = []
        rels: List[float] = []
        for j in np.flatnonzero(valid_mask):
            b = X[j]
            r = Y[j]
            denom = np.linalg.norm(b) * np.linalg.norm(r)
            if denom > 0:
                cosines.append(float(np.dot(b, r) / denom))
            rn = np.linalg.norm(r)
            if rn > 0:
                rels.append(float(np.linalg.norm(b - r) / rn))
        out[f'beta_cosine_{space}_mean'] = float(np.mean(cosines)) if cosines else np.nan
        out[f'beta_cosine_{space}_median'] = float(np.median(cosines)) if cosines else np.nan
        out[f'beta_relative_l2_{space}_mean'] = float(np.mean(rels)) if rels else np.nan
    d = Bg.shape[1]
    for k in ks:
        kk = min(int(k), d)
        vals = []
        for j in np.flatnonzero(valid_mask):
            top_b = set(np.argsort(-Bg[j])[:kk])
            top_r = set(np.argsort(-Rg[j])[:kk])
            vals.append(len(top_b & top_r) / max(kk, 1))
        out[f'marker_overlap_at_{k}_mean'] = float(np.mean(vals)) if vals else np.nan
    return out

def beta_comparison_rows(clf: LogisticRegression, full_clf: LogisticRegression, svd: TruncatedSVD, classes: Sequence[str], base: Mapping[str, object], ks: Sequence[int]=(20, 50, 100)) -> List[Dict[str, object]]:
    n_classes_full = len(classes)
    (B, bi, present_mask) = align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (R, ri, ref_present_mask) = align_clf_coef_intercept_to_global_classes(full_clf, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (Bg, _, _) = expand_coef_to_gene_space_aligned(clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (Rg, _, _) = expand_coef_to_gene_space_aligned(full_clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    rows: List[Dict[str, object]] = []
    for (j, cls) in enumerate(classes):
        row: Dict[str, object] = dict(base)
        row['class_label'] = str(cls)
        row['model_class_present'] = bool(present_mask[j])
        if not (present_mask[j] and ref_present_mask[j]):
            for space in ('pca', 'gene'):
                row[f'beta_cosine_{space}'] = np.nan
                row[f'beta_relative_l2_{space}'] = np.nan
                row[f'beta_core_norm_{space}'] = np.nan
                row[f'beta_opt_norm_{space}'] = float(np.linalg.norm(R[j] if space == 'pca' else Rg[j]))
            row['intercept_core'] = np.nan
            row['intercept_opt'] = float(ri[j]) if j < len(ri) else np.nan
            row['intercept_abs_diff'] = np.nan
            for k in ks:
                row[f'marker_overlap_at_{k}_class'] = np.nan
            rows.append(row)
            continue
        for (space, b_vec, r_vec) in [('pca', B[j], R[j]), ('gene', Bg[j], Rg[j])]:
            denom = np.linalg.norm(b_vec) * np.linalg.norm(r_vec)
            row[f'beta_cosine_{space}'] = float(np.dot(b_vec, r_vec) / denom) if denom > 0 else np.nan
            rn = np.linalg.norm(r_vec)
            row[f'beta_relative_l2_{space}'] = float(np.linalg.norm(b_vec - r_vec) / rn) if rn > 0 else np.nan
            row[f'beta_core_norm_{space}'] = float(np.linalg.norm(b_vec))
            row[f'beta_opt_norm_{space}'] = float(np.linalg.norm(r_vec))
        row['intercept_core'] = float(bi[j])
        row['intercept_opt'] = float(ri[j])
        row['intercept_abs_diff'] = float(abs(bi[j] - ri[j]))
        for k in ks:
            kk = min(int(k), Bg.shape[1])
            top_b = set(np.argsort(-Bg[j])[:kk])
            top_r = set(np.argsort(-Rg[j])[:kk])
            row[f'marker_overlap_at_{k}_class'] = float(len(top_b & top_r) / max(kk, 1))
        rows.append(row)
    return rows

def marker_gene_rows(clf: LogisticRegression, full_clf: LogisticRegression, svd: TruncatedSVD, classes: Sequence[str], gene_names: Sequence[str], base: Mapping[str, object], top_k: int) -> List[Dict[str, object]]:
    if top_k <= 0:
        return []
    n_classes_full = len(classes)
    (Bg, _, present_mask) = expand_coef_to_gene_space_aligned(clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    (Rg, _, ref_present_mask) = expand_coef_to_gene_space_aligned(full_clf, svd, n_classes_full=n_classes_full, missing_intercept=-40.0)
    genes = list(map(str, gene_names))
    rows: List[Dict[str, object]] = []
    kk = min(int(top_k), Bg.shape[1])
    for (j, cls) in enumerate(classes):
        if not (present_mask[j] and ref_present_mask[j]):
            continue
        top_core = list(np.argsort(-Bg[j])[:kk])
        top_opt = list(np.argsort(-Rg[j])[:kk])
        union = list(dict.fromkeys(top_opt + top_core))
        rank_core = {idx: r + 1 for (r, idx) in enumerate(top_core)}
        rank_opt = {idx: r + 1 for (r, idx) in enumerate(top_opt)}
        for idx in union:
            row = dict(base)
            row.update({'class_label': str(cls), 'model_class_present': True, 'gene': genes[idx] if idx < len(genes) else f'gene_{idx}', 'gene_index': int(idx), 'beta_core_gene_coef': float(Bg[j, idx]), 'beta_opt_gene_coef': float(Rg[j, idx]), 'abs_diff_gene_coef': float(abs(Bg[j, idx] - Rg[j, idx])), 'rank_in_core_top_positive': rank_core.get(idx, np.nan), 'rank_in_opt_top_positive': rank_opt.get(idx, np.nan), 'in_core_top_k': idx in rank_core, 'in_opt_top_k': idx in rank_opt, 'marker_top_k': int(kk)})
            rows.append(row)
    return rows

def class_metric_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], base: Mapping[str, object], y_train_full: Sequence[object], y_core: Sequence[object], mu_lookup: Optional[Mapping[str, float]]=None, quota_lookup: Optional[Mapping[str, int]]=None) -> List[Dict[str, object]]:
    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    full_counts = pd.Series(np.asarray(y_train_full).astype(str)).value_counts()
    core_counts = pd.Series(np.asarray(y_core).astype(str)).value_counts()
    rows: List[Dict[str, object]] = []
    for (c, cls) in enumerate(classes):
        vals = report.get(str(c), {})
        row = dict(base)
        row.update({'class_label': str(cls), 'class_encoded': int(c), 'precision': vals.get('precision', np.nan), 'recall': vals.get('recall', np.nan), 'f1': vals.get('f1-score', np.nan), 'support': vals.get('support', np.nan), 'n_train_full': int(full_counts.get(str(cls), 0)), 'n_train_coreset': int(core_counts.get(str(cls), 0)), 'train_full_fraction': float(full_counts.get(str(cls), 0) / max(len(y_train_full), 1)), 'train_coreset_fraction': float(core_counts.get(str(cls), 0) / max(len(y_core), 1))})
        if mu_lookup is not None:
            row['mu'] = float(mu_lookup.get(str(cls), np.nan))
        if quota_lookup is not None:
            row['target_class_quota'] = int(quota_lookup.get(str(cls), 0))
        rows.append(row)
    return rows

def composition_rows(y_train_full: Sequence[object], idx: np.ndarray, classes: Sequence[str], base: Mapping[str, object], mu_lookup: Optional[Mapping[str, float]]=None, quota_lookup: Optional[Mapping[str, int]]=None) -> List[Dict[str, object]]:
    y_full = np.asarray(y_train_full).astype(str)
    y_core = y_full[np.asarray(idx, dtype=int)] if len(idx) else np.array([], dtype=str)
    n_full = len(y_full)
    n_core = len(y_core)
    rows: List[Dict[str, object]] = []
    for cls in classes:
        cls = str(cls)
        row = dict(base)
        full_count = int(np.sum(y_full == cls))
        core_count = int(np.sum(y_core == cls))
        row.update({'class_label': cls, 'train_class_count': full_count, 'train_class_fraction': float(full_count / max(n_full, 1)), 'coreset_class_count': core_count, 'coreset_class_fraction': float(core_count / max(n_core, 1)), 'coreset_over_train_class_fraction': float(core_count / max(full_count, 1))})
        if mu_lookup is not None:
            row['mu'] = float(mu_lookup.get(cls, np.nan))
        if quota_lookup is not None:
            row['target_class_quota'] = int(quota_lookup.get(cls, 0))
        rows.append(row)
    return rows

def make_feature_adata_for_celltypist(X_features: np.ndarray, obs: pd.DataFrame, label_key: str, feature_prefix: str='SVD'):
    if ad is None:
        raise RuntimeError('anndata is required for CellTypist SVD baselines')
    X = np.asarray(X_features, dtype=np.float32)
    obs_copy = obs.copy()
    obs_copy[label_key] = obs_copy[label_key].astype(str)
    var = pd.DataFrame(index=[f'{feature_prefix}{j + 1:03d}' for j in range(X.shape[1])])
    return ad.AnnData(X=X, obs=obs_copy, var=var)

def make_feature_matrix_payload_for_celltypist(X_features: np.ndarray, obs: pd.DataFrame, label_key: str, feature_prefix: str='SVD') -> Dict[str, object]:
    X = np.asarray(X_features, dtype=np.float32)
    labels = obs[label_key].astype(str).to_numpy()
    genes = np.array([f'{feature_prefix}{j + 1:03d}' for j in range(X.shape[1])], dtype=object)
    return {'kind': 'matrix', 'X': X, 'labels': labels, 'genes': genes}

def is_celltypist_matrix_payload(obj: object) -> bool:
    return isinstance(obj, dict) and obj.get('kind') == 'matrix'

def align_probability_frame(prob_df: pd.DataFrame, classes: Sequence[str]) -> np.ndarray:
    class_names = [str(c) for c in classes]
    P = np.zeros((prob_df.shape[0], len(class_names)), dtype=np.float64)
    col_lookup = {str(c): c for c in prob_df.columns}
    for (j, cls) in enumerate(class_names):
        if cls in col_lookup:
            P[:, j] = pd.to_numeric(prob_df[col_lookup[cls]], errors='coerce').fillna(0.0).values
    row_sum = P.sum(axis=1, keepdims=True)
    bad = row_sum.ravel() <= 0
    if np.any(bad):
        P[bad, :] = 1.0 / len(class_names)
        row_sum = P.sum(axis=1, keepdims=True)
    return P / np.maximum(row_sum, 1e-300)

def run_celltypist_matrix_prediction(model, X_test: np.ndarray, genes: Sequence[str], classes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    clf = getattr(model, 'classifier', None)
    scaler = getattr(model, 'scaler', None)
    if clf is None or scaler is None:
        raise RuntimeError('Unexpected CellTypist model: missing classifier or scaler')
    model_features = getattr(clf, 'features', None)
    if model_features is None:
        model_features = getattr(model, 'features', None)
    if model_features is None:
        model_features = genes
    model_features = [str(g) for g in np.asarray(model_features).ravel()]
    gene_lookup = {str(g): i for (i, g) in enumerate(genes)}
    missing = [g for g in model_features if g not in gene_lookup]
    if missing:
        raise RuntimeError(f'CellTypist matrix prediction missing {len(missing)} model features in query matrix; first={missing[:5]}')
    idx = np.array([gene_lookup[g] for g in model_features], dtype=int)
    Xq = np.asarray(X_test, dtype=np.float32)[:, idx]
    Xs = scaler.transform(Xq)
    try:
        Xs = np.asarray(Xs)
        Xs[Xs > 10] = 10
    except Exception:
        pass
    decision = clf.decision_function(Xs)
    clf_classes = np.asarray(clf.classes_).astype(str)
    if np.ndim(decision) == 1:
        decision_2d = np.vstack([-decision, decision]).T
    else:
        decision_2d = np.asarray(decision, dtype=float)
    pred_idx = np.argmax(decision_2d, axis=1)
    y_pred_str = clf_classes[pred_idx]
    prob_raw = 1.0 / (1.0 + np.exp(-np.clip(decision_2d, -50, 50)))
    class_names = [str(c) for c in classes]
    P = np.zeros((prob_raw.shape[0], len(class_names)), dtype=np.float64)
    lookup = {str(c): j for (j, c) in enumerate(class_names)}
    for (k, cls) in enumerate(clf_classes):
        if str(cls) in lookup and k < prob_raw.shape[1]:
            P[:, lookup[str(cls)]] = prob_raw[:, k]
    row_sum = P.sum(axis=1, keepdims=True)
    bad = row_sum.ravel() <= 0
    if np.any(bad):
        P[bad, :] = 1.0 / len(class_names)
        row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sum, 1e-300)
    return (y_pred_str, P)

def run_celltypist_baseline(adata_train, adata_test, label_key: str, method: str, classes: Sequence[str], label_encoder: LabelEncoder, n_jobs: int, majority_voting: bool, celltypist_C: float, celltypist_solver: str, celltypist_max_iter: int, celltypist_sgd_alpha: float, celltypist_sgd_alpha_from_C: bool, celltypist_sgd_C: float, celltypist_sgd_batch_number: int, celltypist_sgd_batch_size: int, celltypist_sgd_epochs: int, celltypist_top_genes: int, celltypist_with_mean: bool, celltypist_check_expression: bool) -> Dict[str, object]:
    if celltypist is None:
        raise RuntimeError('celltypist is not installed')
    method = method.strip().lower()
    matrix_mode = is_celltypist_matrix_payload(adata_train)
    if matrix_mode:
        train_X = np.asarray(adata_train['X'], dtype=np.float32)
        test_X = np.asarray(adata_test['X'], dtype=np.float32)
        train_labels = np.asarray(adata_train['labels']).astype(str)
        train_genes = np.asarray(adata_train['genes']).astype(str)
        test_genes = np.asarray(adata_test['genes']).astype(str)
        train_n_obs = int(train_X.shape[0])
        kwargs: Dict[str, object] = {'labels': train_labels, 'genes': train_genes, 'n_jobs': n_jobs, 'with_mean': bool(celltypist_with_mean), 'check_expression': False}
        if majority_voting:
            LOGGER.warning('CellTypist majority voting is disabled for SVD/array baseline; using per-cell predictions only.')
            majority_voting = False
    else:
        train_n_obs = int(adata_train.n_obs)
        kwargs = {'labels': label_key, 'n_jobs': n_jobs, 'with_mean': bool(celltypist_with_mean), 'check_expression': bool(celltypist_check_expression)}

    def add_traditional_solver_options(d: Dict[str, object]) -> None:
        d['use_SGD'] = False
        d['C'] = float(celltypist_C)
        if str(celltypist_solver).lower() not in {'', 'none', 'auto'}:
            d['solver'] = str(celltypist_solver)
        if int(celltypist_max_iter) > 0:
            d['max_iter'] = int(celltypist_max_iter)
    if celltypist_sgd_alpha_from_C:
        effective_sgd_alpha = 1.0 / (float(celltypist_sgd_C) * max(int(train_n_obs), 1))
    else:
        effective_sgd_alpha = float(celltypist_sgd_alpha)
    top_match = re.search('top_(?:genes|features)([0-9]+)', method)
    top_n = int(top_match.group(1)) if top_match else int(celltypist_top_genes)
    if method in {'celltypist', 'default', 'celltypist_default', 'traditional', 'global'}:
        add_traditional_solver_options(kwargs)
        method_name = 'CellTypist_default'
    elif method in {'celltypist_top_genes', 'top_genes', 'celltypist_top_features', 'top_features'} or re.match('^(celltypist_)?top_(genes|features)[0-9]+$', method):
        add_traditional_solver_options(kwargs)
        kwargs.update({'feature_selection': True, 'top_genes': int(top_n)})
        method_name = f'CellTypist_top_features{int(top_n)}'
    elif method in {'celltypist_mini_batch_sgd', 'celltypist_sgd', 'sgd', 'mini_batch_sgd', 'celltypist_minibatch_sgd', 'celltypist_sgd_minibatch', 'sgd_minibatch'}:
        kwargs.update({'use_SGD': True, 'mini_batch': True, 'alpha': float(effective_sgd_alpha), 'batch_number': int(celltypist_sgd_batch_number), 'batch_size': int(celltypist_sgd_batch_size), 'epochs': int(celltypist_sgd_epochs), 'balance_cell_type': False, 'feature_selection': False})
        method_name = 'CellTypist_mini_batch_SGD'
    elif method in {'celltypist_mini_batch_sgd_top_genes', 'celltypist_mini_batch_sgd_top_features', 'mini_batch_sgd_top_genes', 'mini_batch_sgd_top_features'} or re.match('^(celltypist_)?(mini_batch_sgd|sgd|minibatch_sgd).*top_(genes|features)[0-9]+$', method):
        kwargs.update({'use_SGD': True, 'mini_batch': True, 'alpha': float(effective_sgd_alpha), 'batch_number': int(celltypist_sgd_batch_number), 'batch_size': int(celltypist_sgd_batch_size), 'epochs': int(celltypist_sgd_epochs), 'balance_cell_type': False, 'feature_selection': True, 'top_genes': int(top_n)})
        method_name = f'CellTypist_mini_batch_SGD_top_features{int(top_n)}'
    elif method in {'celltypist_mini_batch_sgd_balanced', 'celltypist_sgd_balanced', 'sgd_balanced', 'mini_batch_sgd_balanced', 'celltypist_sgd_minibatch_balanced', 'celltypist_sgd_balanced'}:
        kwargs.update({'use_SGD': True, 'mini_batch': True, 'alpha': float(effective_sgd_alpha), 'batch_number': int(celltypist_sgd_batch_number), 'batch_size': int(celltypist_sgd_batch_size), 'epochs': int(celltypist_sgd_epochs), 'balance_cell_type': True, 'feature_selection': False})
        method_name = 'CellTypist_mini_batch_SGD_balanced'
    else:
        raise ValueError(f'Unknown CellTypist method: {method}')
    train_start = now()
    if matrix_mode:
        model = celltypist.train(train_X, **kwargs)
    else:
        model = celltypist.train(adata_train, **kwargs)
    train_runtime = now() - train_start
    pred_start = now()
    if matrix_mode:
        (y_pred_str, P) = run_celltypist_matrix_prediction(model, test_X, test_genes, classes)
        pred_col = 'manual_matrix_best_match'
    else:
        preds = celltypist.annotate(adata_test, model=model, majority_voting=majority_voting)
        pred_df = preds.predicted_labels
        pred_col = 'majority_voting' if majority_voting and 'majority_voting' in pred_df.columns else 'predicted_labels'
        y_pred_str = pred_df[pred_col].astype(str).values
        P = align_probability_frame(preds.probability_matrix, classes)
    prediction_runtime = now() - pred_start
    y_pred_encoded = label_encoder.transform(y_pred_str)
    return {'method_name': method_name, 'y_pred_encoded': y_pred_encoded, 'proba': P, 'prediction_column': pred_col, 'model_fit_runtime_sec': float(train_runtime), 'prediction_runtime_sec': float(prediction_runtime), 'runtime_sec': float(train_runtime + prediction_runtime), 'celltypist_train_kwargs_json': json.dumps(kwargs, sort_keys=True, default=str), 'celltypist_sgd_alpha_effective': float(effective_sgd_alpha), 'celltypist_sgd_alpha_from_C': bool(celltypist_sgd_alpha_from_C), 'celltypist_sgd_C_for_alpha': float(celltypist_sgd_C)}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='CorLR/CorLewis benchmark with μ theory/global and practical class-allocation variants.')
    p.add_argument('--dataset', required=True, help='Input .h5ad file')
    p.add_argument('--dataset_name', default='dataset')
    p.add_argument('--input-state', '--input_state', dest='input_state', choices=['preprocessed'], required=True)
    p.add_argument('--label_key', default='cell_type')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--seeds', default='42')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--target_sum', type=float, default=10000.0)
    p.add_argument('--max_cells', type=int, default=0, help='Optional stratified cell cap for quick pilot debugging; 0 disables')
    p.add_argument('--n_pcs', default='100')
    p.add_argument('--svd_random_state', type=int, default=42)
    p.add_argument('--github_random_state', type=int, default=42, help='Random state used by add_pca coreset sampling')
    p.add_argument('--use_celltypist_downsample', action='store_true', help='Use celltypist.samples.downsample_adata for exact corLR sampling when installed')
    p.add_argument('--ratios', default='0.02,0.06,0.10,0.20')
    p.add_argument('--methods', default='corLR,random_global,corLewis_global_grid,corLewis_equal_grid,corLewis_mu_grid,corLewis_mu_blend_grid', help='Comma-separated practical methods. Supports grid shorthands: corLewis_global_grid, corLewis_equal_grid, corLewis_mu_grid, corLewis_mu_blend_grid.')
    p.add_argument('--lewis_uniform_mixes', default='0.05,0.25,0.50', help='Comma-separated Lewis/uniform floor values used by grid methods.')
    p.add_argument('--mu_blend_etas', default='0.25,0.50', help='Comma-separated eta values for CorLewis-mu-blend grid.')
    p.add_argument('--run_theory_global_size_methods', action='store_true', help='Run Corollary-9-facing global Lewis methods whose total size is computed from q95/max μ and ε, independent of --ratios.')
    p.add_argument('--theory_mu_stats', default='q95,max', help='Comma-separated μ summaries for theory/global size methods: q95,max,mean,q90,etc.')
    p.add_argument('--theory_epsilons', default='0.5,0.3,0.2', help='Comma-separated epsilon values for m = alpha*d*mu^2/epsilon^2.')
    p.add_argument('--theory_alphas', default='1', help='Comma-separated alpha multipliers for m = alpha*d*mu^2/epsilon^2.')
    p.add_argument('--theory_max_ratio', type=float, default=1.0, help='Skip theory/global and theory/class-floor rows whose requested size exceeds this fraction of n_train. Use 1.0 to disable.')
    p.add_argument('--theory_min_ratio', type=float, default=0.0, help='Skip theory/global and theory/class-floor rows whose requested size is below this fraction of n_train. Use 0.0 to disable.')
    p.add_argument('--theory_lewis_uniform_mix', type=float, default=0.05, help='Lewis/uniform floor for theory/global size methods.')
    p.add_argument('--theory_use_sample_weight', action='store_true', default=True, help='Use inverse-probability sample weights for theory/global size methods.')
    p.add_argument('--theory_no_sample_weight', dest='theory_use_sample_weight', action='store_false', help='Disable sample weights for theory/global size methods.')
    p.add_argument('--run_theory_classfloor_methods', action='store_true', help='Run loss-friendly theory/global bridge methods: μ chooses total size, but sampling uses Lewis + class-balanced floor + uniform floor.')
    p.add_argument('--theory_classfloor_mixes', default='0.25', help='Comma-separated class-balanced mixture weights for theory class-floor methods.')
    p.add_argument('--theory_weight_clip_quantile', type=float, default=1.0, help='Optional quantile cap for inverse-probability weights in theory class-floor methods. 1.0 disables clipping.')
    p.add_argument('--global_lewis_ratio_use_sample_weight', action='store_true', help='Also use inverse-probability weights for fixed-ratio corLewis_global. Default is unweighted practical baseline.')
    p.add_argument('--mu_blend_eta', type=float, default=0.25, help='Eta used when method is exactly corLewis_mu_blend.')
    p.add_argument('--C', type=float, default=0.1)
    p.add_argument('--solver', default='lbfgs', choices=['lbfgs', 'saga', 'liblinear', 'newton-cg', 'sag'])
    p.add_argument('--max_iter', type=int, default=200)
    p.add_argument('--n_jobs', type=int, default=-1)
    p.add_argument('--mu_by_class', default='', help='Optional mu_by_class.csv from the existing pipeline')
    p.add_argument('--mu_solver', default='gurobi', help='Preferred mu solver when mu_by_class has multiple solvers; use any to ignore')
    p.add_argument('--mu_intercept', choices=['any', 'yes', 'no'], default='any')
    p.add_argument('--mu_power', type=float, default=2.0)
    p.add_argument('--mu_floor', type=float, default=1.0)
    p.add_argument('--mu_cap', type=float, default=10.0)
    p.add_argument('--mu_min_per_class', type=int, default=5)
    p.add_argument('--compute_mu_same_embedding', action='store_true', help='Compute OvR mu on the current train split and current global TruncatedSVD features, then use it for corLewis_mu.')
    p.add_argument('--mu_only', action='store_true', help='Only compute/save same-embedding mu and preprocessing/split metadata, then exit before model benchmarks.')
    p.add_argument('--mu_rows', type=int, default=5000, help='Rows for same-embedding mu LP diagnostic; 0 uses all train rows.')
    p.add_argument('--mu_classes', default='all', help='Classes for same-embedding mu: all, an integer count, or comma-separated labels.')
    p.add_argument('--mu_row_sampling', choices=['uniform', 'stratified', 'balanced', 'ovr_balanced'], default='ovr_balanced')
    p.add_argument('--mu_ovr_max_pos', type=int, default=80, help='For mu_row_sampling=ovr_balanced, cap positives per OvR LP; <=0 uses all positives.')
    p.add_argument('--mu_ovr_neg_multiplier', type=float, default=3.0, help='For mu_row_sampling=ovr_balanced, negatives sampled per positive.')
    p.add_argument('--mu_ovr_max_neg', type=int, default=240, help='For mu_row_sampling=ovr_balanced, cap negatives per OvR LP; <=0 disables cap.')
    p.add_argument('--mu_min_pos', type=int, default=5)
    p.add_argument('--mu_min_neg', type=int, default=5)
    p.add_argument('--mu_compute_intercept', action='store_true', help='Add intercept column when computing same-embedding mu. Default is no intercept, matching the old mu table setup.')
    p.add_argument('--mu_lp_time_limit', type=float, default=0.0)
    p.add_argument('--gurobi_threads', type=int, default=1)
    p.add_argument('--gurobi_mem_limit_gb', type=float, default=96.0)
    p.add_argument('--gurobi_output', action='store_true')
    p.add_argument('--lewis_iter', type=int, default=10)
    p.add_argument('--lewis_ridge', type=float, default=1e-06)
    p.add_argument('--lewis_damping', type=float, default=0.5)
    p.add_argument('--lewis_uniform_mix', type=float, default=0.05)
    p.add_argument('--lewis_block_size', type=int, default=50000)
    p.add_argument('--run_celltypist', action='store_true')
    p.add_argument('--celltypist_methods', default='celltypist_mini_batch_sgd,celltypist_mini_batch_sgd_balanced,celltypist_mini_batch_sgd_top_genes50', help='Comma-separated CellTypist baselines. Defaults are three SVD-input mini-batch SGD variants.')
    p.add_argument('--celltypist_input_space', choices=['svd', 'original', 'both'], default='svd', help='Fairness mode for CellTypist baselines. svd gives CellTypist the same n_pcs features used by CorLewis; original uses normalized genes; both runs both.')
    p.add_argument('--celltypist_majority_voting', action='store_true')
    p.add_argument('--celltypist_C', type=float, default=1.0, help='C for traditional/default CellTypist logistic classifier')
    p.add_argument('--celltypist_solver', default='auto', help='auto/none lets CellTypist choose; otherwise pass sklearn solver')
    p.add_argument('--celltypist_max_iter', type=int, default=0, help='0 lets CellTypist use its data-size default')
    p.add_argument('--celltypist_sgd_alpha', type=float, default=0.0001)
    p.add_argument('--celltypist_sgd_alpha_from_C', action='store_true', help='For mini-batch SGD CellTypist, set alpha=1/(celltypist_sgd_C*n_train), matching the old CorLR hECA script.')
    p.add_argument('--celltypist_sgd_C', type=float, default=0.0001, help='C used only when --celltypist_sgd_alpha_from_C is set; old hECA script used C=0.0001.')
    p.add_argument('--celltypist_sgd_batch_number', type=int, default=100)
    p.add_argument('--celltypist_sgd_batch_size', type=int, default=1000)
    p.add_argument('--celltypist_sgd_epochs', type=int, default=10)
    p.add_argument('--celltypist_top_genes', type=int, default=50)
    p.add_argument('--celltypist_no_with_mean', action='store_true', help='Pass with_mean=False to celltypist.train to reduce memory on sparse inputs')
    p.add_argument('--celltypist_no_check_expression', action='store_true', help='Pass check_expression=False to celltypist.train')
    p.add_argument('--save_beta_npz', action='store_true')
    p.add_argument('--save_marker_gene_rows', action='store_true')
    p.add_argument('--marker_top_k', type=int, default=50)
    p.add_argument('--marker_gene_methods', default='corLR,corLewis_mu,corLewis_equal')
    p.add_argument('--marker_gene_ratios', default='0.06,0.10,0.20,0.40', help='Only write marker-gene rows for methods whose observed ratio is close to these values. Use all to disable filtering.')
    p.add_argument('--save_predictions', action='store_true')
    p.add_argument('--save_selected_cell_sample', action='store_true', help='Save a capped sample of selected cells per method/ratio for SVD/UMAP overlay diagnostics.')
    p.add_argument('--selected_cell_sample_max', type=int, default=2000, help='Max selected cells saved per method/ratio when --save_selected_cell_sample is set.')
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'splits')
    ensure_dir(out_dir / 'beta_npz')
    paths = {'metrics': out_dir / 'metrics.csv', 'class_metrics': out_dir / 'class_metrics.csv', 'beta_comparison': out_dir / 'beta_comparison.csv', 'marker_gene_comparison': out_dir / 'marker_gene_comparison.csv', 'coreset_composition': out_dir / 'coreset_composition.csv', 'mu_allocation': out_dir / 'mu_allocation_by_class.csv', 'lewis_diagnostics': out_dir / 'lewis_diagnostics.csv', 'predictions': out_dir / 'predictions.csv', 'selected_cells': out_dir / 'selected_cells_sample.csv', 'preprocessing': out_dir / 'preprocessing_summary.csv'}
    if args.overwrite:
        for pth in paths.values():
            if pth.exists():
                pth.unlink()
        for pth in (out_dir / 'splits').glob('*.csv'):
            pth.unlink()
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    adata = read_h5ad(args.dataset)
    if args.label_key not in adata.obs:
        raise KeyError(f'label_key={args.label_key!r} not found in adata.obs')
    adata.obs[args.label_key] = adata.obs[args.label_key].astype(str)
    keep = adata.obs[args.label_key].notna() & ~adata.obs[args.label_key].astype(str).isin(['nan', 'None', 'NA'])
    adata = adata[keep.values].copy()
    adata = maybe_stratified_downsample_adata(adata, args.label_key, args.max_cells, seed=parse_csv_ints(args.seeds)[0])
    LOGGER.info('After label filtering/pilot cap: %s', adata.shape)
    require_preprocessed(adata, args.label_key, args.input_state)
    full_labels = adata.obs[args.label_key].astype(str)
    full_label_counts = full_labels.value_counts()
    n_labels_full = int(len(full_label_counts))
    gene_names = list(map(str, adata.var_names))
    seeds = parse_csv_ints(args.seeds)
    n_pcs_values = parse_csv_ints(args.n_pcs)
    ratios = parse_csv_floats(args.ratios)
    methods = expand_benchmark_methods(parse_csv_strs(args.methods), args)
    marker_methods = set(expand_benchmark_methods(parse_csv_strs(args.marker_gene_methods), args))
    LOGGER.info('Expanded benchmark methods: %s', methods)
    for seed in seeds:
        set_all_seeds(seed)
        y_all = adata.obs[args.label_key].astype(str).values
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        (train_idx, test_idx) = next(splitter.split(np.zeros(len(y_all)), y_all))
        split_df = pd.DataFrame({'obs_name': adata.obs_names.astype(str), 'split': 'unused'})
        split_df.loc[train_idx, 'split'] = 'train'
        split_df.loc[test_idx, 'split'] = 'test'
        split_df.to_csv(out_dir / 'splits' / f'split_seed_{seed}.csv', index=False)
        adata_train = adata[train_idx].copy()
        adata_test = adata[test_idx].copy()
        y_train_str = adata_train.obs[args.label_key].astype(str)
        y_test_str = adata_test.obs[args.label_key].astype(str)
        le = LabelEncoder()
        le.fit(np.unique(np.concatenate([y_train_str.values, y_test_str.values])))
        classes = list(map(str, le.classes_))
        y_train_encoded = le.transform(y_train_str.values)
        y_test_encoded = le.transform(y_test_str.values)
        for n_pcs in n_pcs_values:
            LOGGER.info('Seed=%s n_pcs=%s: fitting full-matrix TruncatedSVD', seed, n_pcs)
            (X_svd_all, svd, embedding_runtime) = fit_github_svd_full_matrix(adata.X, n_pcs, seed=args.svd_random_state)
            X_train_pca = X_svd_all[train_idx]
            X_test_pca = X_svd_all[test_idx]
            X_test_original = adata_test.X
            append_rows_csv(paths['preprocessing'], [{'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'embedding_note': 'Matches add_pca: TruncatedSVD fitted on full normalized matrix before split', 'n_cells_total': int(adata.n_obs), 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_genes': int(adata.n_vars), 'n_classes': int(len(classes)), 'embedding_runtime_sec': float(embedding_runtime), 'rss_mb': rss_mb()}])
            if args.mu_only:
                if not args.compute_mu_same_embedding:
                    raise ValueError('--mu_only requires --compute_mu_same_embedding')
                LOGGER.info('MU_ONLY mode: computing same-embedding mu and skipping all model benchmarks.')
                mu_df_full = compute_mu_same_embedding(X_train_pca=X_train_pca, y_train=y_train_str.values, classes=classes, seed=seed, n_pcs=n_pcs, args=args)
                append_rows_csv(out_dir / 'mu_by_class_same_embedding.csv', mu_df_full.to_dict(orient='records'))
                mu_df_full[['class_label', 'mu', 'mu_source']].to_csv(out_dir / f'mu_used_seed{seed}_pc{n_pcs}.csv', index=False)
                del X_svd_all, X_train_pca, X_test_pca, svd
                gc.collect()
                continue
            LOGGER.info('Training full LR reference beta_opt')
            t0 = now()
            full_clf = fit_logistic_regression_github(X_train_pca, y_train_encoded, args.C, args.solver, args.max_iter, args.n_jobs)
            full_fit_time = now() - t0
            full_pred_start = now()
            full_eval = evaluate_model(full_clf, svd, X_test_original, X_test_pca, y_test_encoded, X_train_pca, y_train_encoded, X_train_pca, y_train_encoded, C=args.C)
            full_prediction_time = now() - full_pred_start
            full_eval.update(biological_recall_metrics(y_test_encoded, np.asarray(full_eval['pred_test_encoded'], dtype=int), y_train_str.values, classes))
            full_obj_ref = float(full_eval['objective_full_train_objective_total'])
            full_base = {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'method': 'full_lr_pca', 'method_family': 'reference', 'coreset_ratio_target': 1.0, 'coreset_size_target': int(len(train_idx)), 'coreset_size_observed': int(len(train_idx)), 'coreset_ratio_observed': 1.0, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_classes': int(len(classes)), 'C': float(args.C), 'solver': args.solver, 'max_iter': int(args.max_iter), 'runtime_sec': float(embedding_runtime + full_fit_time + full_prediction_time), 'model_fit_runtime_sec': float(full_fit_time), 'prediction_runtime_sec': float(full_prediction_time), 'sampling_runtime_sec': 0.0, 'embedding_runtime_sec': float(embedding_runtime), 'lewis_runtime_sec': 0.0, 'rss_mb': rss_mb()}
            full_row = dict(full_base)
            for (k, v) in full_eval.items():
                if k not in {'pred_test_encoded', 'probs_test_softmax', 'expanded_coef', 'expanded_intercept'}:
                    full_row[k] = v
            full_row.update({'relative_objective_gap_vs_full_lr': 0.0, 'beta_cosine_pca_mean': 1.0, 'beta_cosine_gene_mean': 1.0, 'beta_relative_l2_pca_mean': 0.0, 'beta_relative_l2_gene_mean': 0.0, 'marker_overlap_at_20_mean': 1.0, 'marker_overlap_at_50_mean': 1.0, 'marker_overlap_at_100_mean': 1.0})
            append_rows_csv(paths['metrics'], [full_row])
            append_rows_csv(paths['class_metrics'], class_metric_rows(y_test_encoded, np.asarray(full_eval['pred_test_encoded'], dtype=int), classes, full_base, y_train_str.values, y_train_str.values))
            append_rows_csv(paths['beta_comparison'], beta_comparison_rows(full_clf, full_clf, svd, classes, full_base))
            if args.save_beta_npz:
                np.savez_compressed(out_dir / 'beta_npz' / f'seed{seed}_pc{n_pcs}_full_lr_pca.npz', classes=np.asarray(classes, dtype=object), beta_opt_pca=full_clf.coef_, intercept_opt=full_clf.intercept_, beta_core_pca=full_clf.coef_, intercept_core=full_clf.intercept_, beta_opt_gene=expand_coef_to_gene_space(full_clf, svd)[0], beta_core_gene=expand_coef_to_gene_space(full_clf, svd)[0])
            if args.run_celltypist:
                ct_spaces = [args.celltypist_input_space] if args.celltypist_input_space != 'both' else ['svd', 'original']
                for ct_space in ct_spaces:
                    if ct_space == 'svd':
                        adata_train_ct = make_feature_matrix_payload_for_celltypist(X_train_pca, adata_train.obs.copy(), args.label_key, feature_prefix='SVD')
                        adata_test_ct = make_feature_matrix_payload_for_celltypist(X_test_pca, adata_test.obs.copy(), args.label_key, feature_prefix='SVD')
                        ct_embedding = f'celltypist_svd{int(n_pcs)}_features'
                        ct_check_expression = False
                    else:
                        adata_train_ct = adata_train
                        adata_test_ct = adata_test
                        ct_embedding = 'celltypist_original_normalized_genes'
                        ct_check_expression = not args.celltypist_no_check_expression
                    for ct_method in parse_csv_strs(args.celltypist_methods):
                        LOGGER.info('Running CellTypist baseline: input=%s method=%s', ct_space, ct_method)
                        try:
                            ct = run_celltypist_baseline(adata_train_ct, adata_test_ct, args.label_key, ct_method, classes, le, n_jobs=args.n_jobs, majority_voting=args.celltypist_majority_voting, celltypist_C=args.celltypist_C, celltypist_solver=args.celltypist_solver, celltypist_max_iter=args.celltypist_max_iter, celltypist_sgd_alpha=args.celltypist_sgd_alpha, celltypist_sgd_alpha_from_C=args.celltypist_sgd_alpha_from_C, celltypist_sgd_C=args.celltypist_sgd_C, celltypist_sgd_batch_number=args.celltypist_sgd_batch_number, celltypist_sgd_batch_size=args.celltypist_sgd_batch_size, celltypist_sgd_epochs=args.celltypist_sgd_epochs, celltypist_top_genes=args.celltypist_top_genes, celltypist_with_mean=not args.celltypist_no_with_mean, celltypist_check_expression=ct_check_expression)
                            ct_time = float(ct['runtime_sec'])
                            y_pred_ct = np.asarray(ct['y_pred_encoded'], dtype=int)
                            Pct = np.asarray(ct['proba'], dtype=float)
                            raw_method_name = str(ct['method_name'])
                            if ct_space == 'svd':
                                ct_method_name = raw_method_name.replace('CellTypist_', 'CellTypist_SVD_', 1)
                            else:
                                ct_method_name = raw_method_name.replace('CellTypist_', 'CellTypist_original_', 1)
                            ct_row = {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': ct_embedding, 'method': ct_method_name, 'method_family': 'reference', 'method_base_family': 'celltypist', 'celltypist_input_space': ct_space, 'coreset_ratio_target': np.nan, 'coreset_size_target': np.nan, 'coreset_size_observed': np.nan, 'coreset_ratio_observed': np.nan, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_classes': int(len(classes)), 'runtime_sec': float(ct_time), 'model_fit_runtime_sec': float(ct['model_fit_runtime_sec']), 'prediction_runtime_sec': float(ct['prediction_runtime_sec']), 'sampling_runtime_sec': 0.0, 'embedding_runtime_sec': 0.0, 'lewis_runtime_sec': 0.0, 'accuracy': float(accuracy_score(y_test_encoded, y_pred_ct)), 'balanced_accuracy': float(balanced_accuracy_score(y_test_encoded, y_pred_ct)), 'macro_f1': float(f1_score(y_test_encoded, y_pred_ct, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_test_encoded, y_pred_ct, average='weighted', zero_division=0)), 'micro_f1': float(f1_score(y_test_encoded, y_pred_ct, average='micro', zero_division=0)), 'macro_precision': float(precision_score(y_test_encoded, y_pred_ct, average='macro', zero_division=0)), 'macro_recall': float(recall_score(y_test_encoded, y_pred_ct, average='macro', zero_division=0)), 'weighted_precision': float(precision_score(y_test_encoded, y_pred_ct, average='weighted', zero_division=0)), 'weighted_recall': float(recall_score(y_test_encoded, y_pred_ct, average='weighted', zero_division=0)), 'matthews_corrcoef': float(matthews_corrcoef(y_test_encoded, y_pred_ct)), 'cohen_kappa': float(cohen_kappa_score(y_test_encoded, y_pred_ct)), **biological_recall_metrics(y_test_encoded, y_pred_ct, y_train_str.values, classes), 'ovr_log_loss_weighted_test_original_softmax': custom_ovr_log_loss_weighted(y_test_encoded, Pct), 'ovr_log_loss_unweighted_test_original_softmax': custom_ovr_log_loss_unweighted(y_test_encoded, Pct), 'multiclass_log_loss_test_original_softmax': float(log_loss(y_test_encoded, Pct, labels=list(range(len(classes))))), **safe_auc_metrics(y_test_encoded, Pct, prefix='original_softmax'), 'prediction_column': ct['prediction_column'], 'celltypist_train_kwargs_json': ct.get('celltypist_train_kwargs_json', ''), 'celltypist_sgd_alpha_effective': ct.get('celltypist_sgd_alpha_effective', np.nan), 'celltypist_sgd_alpha_from_C': ct.get('celltypist_sgd_alpha_from_C', np.nan), 'celltypist_sgd_C_for_alpha': ct.get('celltypist_sgd_C_for_alpha', np.nan), 'rss_mb': rss_mb()}
                            append_rows_csv(paths['metrics'], [ct_row])
                            append_rows_csv(paths['class_metrics'], class_metric_rows(y_test_encoded, y_pred_ct, classes, ct_row, y_train_str.values, []))
                        except Exception as exc:
                            LOGGER.warning('CellTypist baseline input=%s method=%s skipped/error: %s', ct_space, ct_method, exc)
            lewis_weights: Optional[np.ndarray] = None
            lewis_runtime = 0.0
            if any((method_needs_lewis(m) for m in methods)) or bool(args.run_theory_global_size_methods) or bool(args.run_theory_classfloor_methods):
                LOGGER.info('Computing l1-Lewis weights on [SVD features, intercept]')
                A_lewis = np.hstack([X_train_pca, np.ones((X_train_pca.shape[0], 1), dtype=X_train_pca.dtype)])
                t0 = now()
                lewis_weights = approximate_l1_lewis_weights(A_lewis, n_iter=args.lewis_iter, ridge=args.lewis_ridge, damping=args.lewis_damping, block_size=args.lewis_block_size, verbose=args.verbose)
                lewis_runtime = now() - t0
                _lew_diag = lewis_diagnostic_row(lewis_weights, {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'lewis_runtime_sec': float(lewis_runtime), 'lewis_iter': int(args.lewis_iter), 'lewis_uniform_mix': float(args.lewis_uniform_mix), 'n_train': int(len(train_idx)), 'n_features_lewis': int(A_lewis.shape[1])})
                _lew_diag.update(lewis_fixed_point_diagnostics(A_lewis, lewis_weights, ridge=args.lewis_ridge, block_size=args.lewis_block_size))
                append_rows_csv(paths['lewis_diagnostics'], [_lew_diag])
            mu_intercept: Optional[bool]
            if args.mu_intercept == 'yes':
                mu_intercept = True
            elif args.mu_intercept == 'no':
                mu_intercept = False
            else:
                mu_intercept = None
            if args.compute_mu_same_embedding:
                mu_df_full = compute_mu_same_embedding(X_train_pca=X_train_pca, y_train=y_train_str.values, classes=classes, seed=seed, n_pcs=n_pcs, args=args)
                append_rows_csv(out_dir / 'mu_by_class_same_embedding.csv', mu_df_full.to_dict(orient='records'))
                mu_df = mu_df_full[['class_label', 'mu', 'mu_source']].copy()
            else:
                mu_df = load_mu_by_class(args.mu_by_class or None, classes, seed=seed, n_pcs=n_pcs, prefer_intercept_added=mu_intercept, solver=args.mu_solver)
            mu_lookup = dict(zip(mu_df['class_label'].astype(str), pd.to_numeric(mu_df['mu'], errors='coerce')))
            mu_df.to_csv(out_dir / f'mu_used_seed{seed}_pc{n_pcs}.csv', index=False)
            theory_choices = pd.DataFrame()
            if args.run_theory_global_size_methods or args.run_theory_classfloor_methods:
                theory_choices = theory_global_size_choices(mu_df=mu_df, n_features=int(X_train_pca.shape[1]), n_train=int(len(train_idx)), stats=parse_csv_strs(args.theory_mu_stats), epsilons=parse_csv_floats(args.theory_epsilons), alphas=parse_csv_floats(args.theory_alphas))
                if not theory_choices.empty:
                    before_n = len(theory_choices)
                    if float(args.theory_min_ratio) > 0:
                        theory_choices = theory_choices[theory_choices['theory_coreset_ratio'] >= float(args.theory_min_ratio)].copy()
                    if float(args.theory_max_ratio) < 1.0:
                        theory_choices = theory_choices[theory_choices['theory_coreset_ratio'] <= float(args.theory_max_ratio)].copy()
                    after_n = len(theory_choices)
                    if after_n < before_n:
                        LOGGER.info('Filtered theory size choices by ratio range [%.4g, %.4g]: kept %d/%d rows', float(args.theory_min_ratio), float(args.theory_max_ratio), after_n, before_n)
                theory_choices['dataset_name'] = args.dataset_name
                theory_choices['seed'] = int(seed)
                theory_choices['n_pcs'] = int(n_pcs)
                theory_choices['embedding'] = 'github_add_pca_truncated_svd_global'
                append_rows_csv(out_dir / 'theory_global_size_choices.csv', theory_choices.to_dict(orient='records'))
            for ratio in ratios:
                target_size = int(float(ratio) * len(train_idx))
                target_size = max(1, min(target_size, len(train_idx)))
                for method in methods:
                    method = method.strip()
                    if not method:
                        continue
                    LOGGER.info('Method=%s ratio=%.4g target_size=%d', method, ratio, target_size)
                    t_sample = now()
                    quota_df = pd.DataFrame()
                    quota_lookup: Dict[str, int] = {}
                    sample_weight_core: Optional[np.ndarray] = None
                    sampling_note = ''
                    cfg = parse_method_variant(method, args)
                    family = str(cfg['family'])
                    lewis_mix = float(cfg['lewis_uniform_mix'])
                    mu_power = float(cfg['mu_power'])
                    mu_cap = float(cfg['mu_cap'])
                    mu_floor = float(cfg['mu_floor'])
                    mu_blend_eta = float(cfg['mu_blend_eta'])
                    if family == 'random_global':
                        idx = random_global_indices(n_train=len(y_train_str), target_size=target_size, seed=seed + stable_int_hash(f'{method}_{ratio}'))
                        quota_lookup = {}
                        extra = {'quota_rule': 'global_uniform_random_no_label_balance', 'n_per_label_requested': np.nan, 'used_celltypist_downsample_adata': False, 'lewis_uniform_mix_used': np.nan, 'mu_blend_eta': np.nan, 'mu_power_used': np.nan, 'mu_cap_used': np.nan}
                        sampling_note = 'Global uniform random subset without label balancing; deliberately weak random baseline.'
                    elif family == 'corLR':
                        (idx, extra) = github_corLR_indices(adata_train=adata_train, y_train=y_train_str, label_key=args.label_key, target_size=target_size, n_labels_full=n_labels_full, random_state=args.github_random_state, use_celltypist_downsample=args.use_celltypist_downsample)
                        quotas = equal_label_quotas(y_train_str.values, target_size, n_labels_full)
                        quota_lookup = quotas
                        extra.update({'lewis_uniform_mix_used': np.nan, 'mu_blend_eta': np.nan, 'mu_power_used': np.nan, 'mu_cap_used': np.nan})
                        sampling_note = 'Exact add_pca equal-per-label target; sample up to n_per_label per class.'
                    elif family == 'corLewis_global':
                        if lewis_weights is None:
                            raise RuntimeError('Lewis weights were not computed')
                        (idx, sample_weight_core, sw_info) = sample_global_by_lewis(lewis_weights=lewis_weights, target_size=target_size, seed=seed + stable_int_hash(f'{method}_{ratio}'), uniform_mix=lewis_mix, return_inverse_probability_weights=bool(args.global_lewis_ratio_use_sample_weight))
                        quota_lookup = {}
                        extra = {'quota_rule': 'global_lewis_fixed_ratio', 'n_per_label_requested': np.nan, 'used_celltypist_downsample_adata': False, 'lewis_uniform_mix_used': lewis_mix, 'mu_blend_eta': np.nan, 'mu_power_used': np.nan, 'mu_cap_used': np.nan, **sw_info}
                        sampling_note = f'Fixed ratio global Lewis sampling from all training cells; uniform_mix={lewis_mix:g}; sample_weight={bool(sample_weight_core is not None)}.'
                    elif family == 'corLewis_equal':
                        if lewis_weights is None:
                            raise RuntimeError('Lewis weights were not computed')
                        quotas = equal_label_quotas(y_train_str.values, target_size, n_labels_full)
                        quota_lookup = quotas
                        idx = sample_within_labels_by_lewis(y_train_str.values, quotas, lewis_weights, seed=seed + stable_int_hash(f'{method}_{ratio}'), uniform_mix=lewis_mix)
                        extra = {'quota_rule': 'github_equal_per_label_lewis_within_label', 'n_per_label_requested': int(target_size / max(n_labels_full, 1)), 'used_celltypist_downsample_adata': False, 'lewis_uniform_mix_used': lewis_mix, 'mu_blend_eta': np.nan, 'mu_power_used': np.nan, 'mu_cap_used': np.nan}
                        sampling_note = f'Equal per-label quota, Lewis-biased within each label; uniform_mix={lewis_mix:g}.'
                    elif family == 'corLewis_mu':
                        if lewis_weights is None:
                            raise RuntimeError('Lewis weights were not computed')
                        (quotas, quota_df) = mu_label_quotas(y_train_str.values, mu_df, target_size=target_size, mu_power=mu_power, mu_floor=mu_floor, mu_cap=mu_cap, min_per_class=args.mu_min_per_class)
                        quota_lookup = quotas
                        idx = sample_within_labels_by_lewis(y_train_str.values, quotas, lewis_weights, seed=seed + stable_int_hash(f'{method}_{ratio}'), uniform_mix=lewis_mix)
                        extra = {'quota_rule': 'mu_power_allocation_lewis_within_label', 'n_per_label_requested': np.nan, 'used_celltypist_downsample_adata': False, 'lewis_uniform_mix_used': lewis_mix, 'mu_blend_eta': np.nan, 'mu_power_used': mu_power, 'mu_cap_used': mu_cap}
                        sampling_note = f'Per-label quota proportional to clipped mu_c^power, Lewis-biased within each label; power={mu_power:g}, uniform_mix={lewis_mix:g}.'
                    elif family == 'corLewis_mu_blend':
                        if lewis_weights is None:
                            raise RuntimeError('Lewis weights were not computed')
                        (quotas, quota_df) = blended_mu_label_quotas(y_train=y_train_str.values, mu_df=mu_df, target_size=target_size, n_labels_full=n_labels_full, eta=mu_blend_eta, mu_power=mu_power, mu_floor=mu_floor, mu_cap=mu_cap, min_per_class=args.mu_min_per_class)
                        quota_lookup = quotas
                        idx = sample_within_labels_by_lewis(y_train_str.values, quotas, lewis_weights, seed=seed + stable_int_hash(f'{method}_{ratio}'), uniform_mix=lewis_mix)
                        extra = {'quota_rule': 'blend_equal_and_mu_powered_lewis_within_label', 'n_per_label_requested': np.nan, 'used_celltypist_downsample_adata': False, 'lewis_uniform_mix_used': lewis_mix, 'mu_blend_eta': mu_blend_eta, 'mu_power_used': mu_power, 'mu_cap_used': mu_cap}
                        sampling_note = f'Blended equal and mu-powered quota, Lewis-biased within label; eta={mu_blend_eta:g}, power={mu_power:g}, uniform_mix={lewis_mix:g}.'
                    else:
                        raise ValueError(f'Unknown method {method!r}; use corLR, corLewis_global(_mix*), corLewis_equal(_mix*), corLewis_mu(_mix*), corLewis_mu_blend(_eta*_mix*), random_global')
                    sampling_runtime = now() - t_sample
                    idx = np.asarray(idx, dtype=int)
                    if len(idx) == 0:
                        LOGGER.warning('Skipping method=%s ratio=%s because sampled index set is empty', method, ratio)
                        continue
                    X_core_pca = X_train_pca[idx]
                    y_core_str = y_train_str.values[idx]
                    y_core_encoded = le.transform(y_core_str)
                    t0 = now()
                    clf = fit_logistic_regression_github(X_core_pca, y_core_encoded, args.C, args.solver, args.max_iter, args.n_jobs, sample_weight=sample_weight_core)
                    fit_runtime = now() - t0
                    pred_start = now()
                    eval_out = evaluate_model(clf, svd, X_test_original, X_test_pca, y_test_encoded, X_train_pca, y_train_encoded, X_core_pca, y_core_encoded, C=args.C)
                    prediction_runtime = now() - pred_start
                    eval_out.update(biological_recall_metrics(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), y_train_str.values, classes))
                    pres_diag = {}
                    if sample_weight_core is not None:
                        pres_diag = weighted_loss_preservation_diagnostics(full_clf=full_clf, core_clf=clf, X_train_full_pca=X_train_pca, y_train_full_encoded=y_train_encoded, X_core_pca=X_core_pca, y_core_encoded=y_core_encoded, sample_weight_core=sample_weight_core, C=args.C, n_classes_full=len(classes))
                    beta_summary = coefficient_summary_metrics(clf, full_clf, svd)
                    obj = float(eval_out['objective_full_train_objective_total'])
                    rel_gap = (obj - full_obj_ref) / max(abs(full_obj_ref), 1e-300)
                    observed_size = int(len(idx))
                    observed_ratio = float(observed_size / max(len(train_idx), 1))
                    base = {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'method': method, 'method_family': 'coreset', 'method_base_family': family, 'sampling_note': sampling_note, 'coreset_ratio_target': float(ratio), 'coreset_size_target': int(target_size), 'coreset_size_observed': observed_size, 'coreset_ratio_observed': observed_ratio, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_classes': int(len(classes)), 'C': float(args.C), 'solver': args.solver, 'max_iter': int(args.max_iter), 'runtime_sec': float(embedding_runtime + lewis_runtime + sampling_runtime + fit_runtime + prediction_runtime), 'model_fit_runtime_sec': float(fit_runtime), 'prediction_runtime_sec': float(prediction_runtime), 'sampling_runtime_sec': float(sampling_runtime), 'embedding_runtime_sec': float(embedding_runtime), 'lewis_runtime_sec': float(lewis_runtime if method.startswith('corLewis') else 0.0), 'rss_mb': rss_mb(), **extra}
                    row = dict(base)
                    for (k, v) in eval_out.items():
                        if k not in {'pred_test_encoded', 'probs_test_softmax', 'expanded_coef', 'expanded_intercept'}:
                            row[k] = v
                    row.update(beta_summary)
                    if sample_weight_core is not None and 'pres_diag' in locals():
                        row.update(pres_diag)
                    row['relative_objective_gap_vs_full_lr'] = float(rel_gap)
                    append_rows_csv(paths['metrics'], [row])
                    append_rows_csv(paths['class_metrics'], class_metric_rows(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), classes, base, y_train_str.values, y_core_str, mu_lookup=mu_lookup, quota_lookup=quota_lookup))
                    append_rows_csv(paths['coreset_composition'], composition_rows(y_train_str.values, idx, classes, base, mu_lookup=mu_lookup, quota_lookup=quota_lookup))
                    append_rows_csv(paths['beta_comparison'], beta_comparison_rows(clf, full_clf, svd, classes, base))
                    if args.save_marker_gene_rows and method in marker_methods and ratio_allowed(observed_ratio, args.marker_gene_ratios):
                        append_rows_csv(paths['marker_gene_comparison'], marker_gene_rows(clf, full_clf, svd, classes, gene_names, base, top_k=args.marker_top_k))
                    if family in {'corLewis_mu', 'corLewis_mu_blend'}:
                        if quota_df.empty:
                            if family == 'corLewis_mu_blend':
                                (_, quota_df) = blended_mu_label_quotas(y_train=y_train_str.values, mu_df=mu_df, target_size=target_size, n_labels_full=n_labels_full, eta=mu_blend_eta, mu_power=mu_power, mu_floor=mu_floor, mu_cap=mu_cap, min_per_class=args.mu_min_per_class)
                            else:
                                (_, quota_df) = mu_label_quotas(y_train_str.values, mu_df, target_size=target_size, mu_power=mu_power, mu_floor=mu_floor, mu_cap=mu_cap, min_per_class=args.mu_min_per_class)
                        q = quota_df.copy()
                        for (k, v) in base.items():
                            q[k] = v
                        append_rows_csv(paths['mu_allocation'], q.to_dict(orient='records'))
                    if args.save_selected_cell_sample:
                        append_rows_csv(paths['selected_cells'], selected_cell_sample_rows(base=base, obs_names=adata_train.obs_names.astype(str), y_train=y_train_str.values, idx=idx, X_train_pca=X_train_pca, lewis_weights=lewis_weights if method_needs_lewis(method) else None, mu_lookup=mu_lookup, max_rows=args.selected_cell_sample_max, seed=seed + stable_int_hash(f'selected_{method}_{ratio}')))
                    if args.save_predictions:
                        pred_labels = le.inverse_transform(np.asarray(eval_out['pred_test_encoded'], dtype=int))
                        append_rows_csv(paths['predictions'], [{'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'method': method, 'coreset_ratio_target': float(ratio), 'obs_name': str(obs), 'true_label': str(true), 'pred_label': str(pred)} for (obs, true, pred) in zip(adata_test.obs_names.astype(str), y_test_str.values, pred_labels)])
                    if args.save_beta_npz:
                        (beta_opt_gene, intercept_opt) = expand_coef_to_gene_space(full_clf, svd)
                        (beta_core_gene, intercept_core) = expand_coef_to_gene_space(clf, svd)
                        np.savez_compressed(out_dir / 'beta_npz' / f'seed{seed}_pc{n_pcs}_{safe_name(method)}_ratio{int(round(ratio * 1000))}.npz', classes=np.asarray(classes, dtype=object), beta_opt_pca=full_clf.coef_, intercept_opt=full_clf.intercept_, beta_core_pca=clf.coef_, intercept_core=clf.intercept_, beta_opt_gene=beta_opt_gene, beta_core_gene=beta_core_gene, gene_names=np.asarray(gene_names, dtype=object))
                    gc.collect()
            if args.run_theory_global_size_methods and (not theory_choices.empty):
                if lewis_weights is None:
                    raise RuntimeError('Theory/global size methods require Lewis weights')
                for (_, choice) in theory_choices.iterrows():
                    stat = str(choice['theory_mu_stat'])
                    eps = float(choice['theory_epsilon'])
                    alpha = float(choice['theory_alpha'])
                    theory_size = int(choice['theory_coreset_size'])
                    theory_ratio = float(choice['theory_coreset_ratio'])
                    theory_mu_val = float(choice['theory_mu_value'])
                    mix = float(args.theory_lewis_uniform_mix)
                    method = theory_method_name(stat, eps, alpha, mix)
                    LOGGER.info('Theory/global method=%s target_size=%d ratio=%.4g mu=%s eps=%.4g alpha=%.4g', method, theory_size, theory_ratio, stat, eps, alpha)
                    t_sample = now()
                    (idx, sample_weight_core, sw_info) = sample_global_by_lewis(lewis_weights=lewis_weights, target_size=theory_size, seed=seed + stable_int_hash(f'{method}_{theory_size}'), uniform_mix=mix, return_inverse_probability_weights=bool(args.theory_use_sample_weight))
                    sampling_runtime = now() - t_sample
                    idx = np.asarray(idx, dtype=int)
                    if len(idx) == 0:
                        LOGGER.warning('Skipping theory/global method=%s because sampled index set is empty', method)
                        continue
                    X_core_pca = X_train_pca[idx]
                    y_core_str = y_train_str.values[idx]
                    y_core_encoded = le.transform(y_core_str)
                    t0 = now()
                    clf = fit_logistic_regression_github(X_core_pca, y_core_encoded, args.C, args.solver, args.max_iter, args.n_jobs, sample_weight=sample_weight_core)
                    fit_runtime = now() - t0
                    pred_start = now()
                    eval_out = evaluate_model(clf, svd, X_test_original, X_test_pca, y_test_encoded, X_train_pca, y_train_encoded, X_core_pca, y_core_encoded, C=args.C)
                    prediction_runtime = now() - pred_start
                    eval_out.update(biological_recall_metrics(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), y_train_str.values, classes))
                    pres_diag = {}
                    if sample_weight_core is not None:
                        pres_diag = weighted_loss_preservation_diagnostics(full_clf=full_clf, core_clf=clf, X_train_full_pca=X_train_pca, y_train_full_encoded=y_train_encoded, X_core_pca=X_core_pca, y_core_encoded=y_core_encoded, sample_weight_core=sample_weight_core, C=args.C, n_classes_full=len(classes))
                    beta_summary = coefficient_summary_metrics(clf, full_clf, svd)
                    obj = float(eval_out['objective_full_train_objective_total'])
                    rel_gap = (obj - full_obj_ref) / max(abs(full_obj_ref), 1e-300)
                    observed_size = int(len(idx))
                    observed_ratio = float(observed_size / max(len(train_idx), 1))
                    sampling_note = f'Theory/global CorLewis: total size alpha*d*mu^2/eps^2 with mu_{stat}={theory_mu_val:.4g}, epsilon={eps:g}, alpha={alpha:g}; global Lewis sampling, uniform_mix={mix:g}; sample_weight={bool(sample_weight_core is not None)}.'
                    base = {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'method': method, 'method_family': 'coreset', 'method_base_family': 'corLewis_theory_global_size', 'sampling_note': sampling_note, 'coreset_ratio_target': theory_ratio, 'coreset_size_target': int(theory_size), 'coreset_size_observed': observed_size, 'coreset_ratio_observed': observed_ratio, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_classes': int(len(classes)), 'C': float(args.C), 'solver': args.solver, 'max_iter': int(args.max_iter), 'runtime_sec': float(embedding_runtime + lewis_runtime + sampling_runtime + fit_runtime + prediction_runtime), 'model_fit_runtime_sec': float(fit_runtime), 'prediction_runtime_sec': float(prediction_runtime), 'sampling_runtime_sec': float(sampling_runtime), 'embedding_runtime_sec': float(embedding_runtime), 'lewis_runtime_sec': float(lewis_runtime), 'rss_mb': rss_mb(), 'quota_rule': 'theory_global_mu_size_global_lewis', 'lewis_uniform_mix_used': mix, 'theory_mu_stat': stat, 'theory_mu_value': theory_mu_val, 'theory_epsilon': eps, 'theory_alpha': alpha, 'theory_d_eff': int(choice['theory_d_eff']), 'theory_use_sample_weight': bool(args.theory_use_sample_weight), **sw_info}
                    row = dict(base)
                    for (k, v) in eval_out.items():
                        if k not in {'pred_test_encoded', 'probs_test_softmax', 'expanded_coef', 'expanded_intercept'}:
                            row[k] = v
                    row.update(beta_summary)
                    if sample_weight_core is not None and 'pres_diag' in locals():
                        row.update(pres_diag)
                    row['relative_objective_gap_vs_full_lr'] = float(rel_gap)
                    append_rows_csv(paths['metrics'], [row])
                    append_rows_csv(paths['class_metrics'], class_metric_rows(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), classes, base, y_train_str.values, y_core_str, mu_lookup=mu_lookup, quota_lookup=None))
                    append_rows_csv(paths['coreset_composition'], composition_rows(y_train_str.values, idx, classes, base, mu_lookup=mu_lookup, quota_lookup=None))
                    append_rows_csv(paths['beta_comparison'], beta_comparison_rows(clf, full_clf, svd, classes, base))
                    if args.save_marker_gene_rows and method in marker_methods and ratio_allowed(observed_ratio, args.marker_gene_ratios):
                        append_rows_csv(paths['marker_gene_comparison'], marker_gene_rows(clf, full_clf, svd, classes, gene_names, base, top_k=args.marker_top_k))
                    if args.save_selected_cell_sample:
                        append_rows_csv(paths['selected_cells'], selected_cell_sample_rows(base=base, obs_names=adata_train.obs_names.astype(str), y_train=y_train_str.values, idx=idx, X_train_pca=X_train_pca, lewis_weights=lewis_weights, mu_lookup=mu_lookup, max_rows=args.selected_cell_sample_max, seed=seed + stable_int_hash(f'selected_{method}')))
                    if args.save_predictions:
                        pred_labels = le.inverse_transform(np.asarray(eval_out['pred_test_encoded'], dtype=int))
                        append_rows_csv(paths['predictions'], [{'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'method': method, 'coreset_ratio_target': float(theory_ratio), 'obs_name': str(obs), 'true_label': str(true), 'pred_label': str(pred)} for (obs, true, pred) in zip(adata_test.obs_names.astype(str), y_test_str.values, pred_labels)])
                    if args.save_beta_npz:
                        (beta_opt_gene, intercept_opt) = expand_coef_to_gene_space(full_clf, svd)
                        (beta_core_gene, intercept_core) = expand_coef_to_gene_space(clf, svd)
                        np.savez_compressed(out_dir / 'beta_npz' / f'seed{seed}_pc{n_pcs}_{safe_name(method)}.npz', classes=np.asarray(classes, dtype=object), beta_opt_pca=full_clf.coef_, intercept_opt=full_clf.intercept_, beta_core_pca=clf.coef_, intercept_core=clf.intercept_, beta_opt_gene=beta_opt_gene, beta_core_gene=beta_core_gene, gene_names=np.asarray(gene_names, dtype=object))
                    gc.collect()
            if args.run_theory_classfloor_methods and (not theory_choices.empty):
                if lewis_weights is None:
                    raise RuntimeError('Theory/class-floor methods require Lewis weights')
                class_floor_mixes = parse_csv_floats(args.theory_classfloor_mixes)
                for (_, choice) in theory_choices.iterrows():
                    stat = str(choice['theory_mu_stat'])
                    eps = float(choice['theory_epsilon'])
                    alpha = float(choice['theory_alpha'])
                    theory_size = int(choice['theory_coreset_size'])
                    theory_ratio = float(choice['theory_coreset_ratio'])
                    theory_mu_val = float(choice['theory_mu_value'])
                    mix = float(args.theory_lewis_uniform_mix)
                    for class_floor_mix in class_floor_mixes:
                        method = theory_classfloor_method_name(stat, eps, alpha, mix, float(class_floor_mix))
                        LOGGER.info('Theory/classfloor method=%s target_size=%d ratio=%.4g mu=%s eps=%.4g alpha=%.4g class_floor=%.4g', method, theory_size, theory_ratio, stat, eps, alpha, class_floor_mix)
                        t_sample = now()
                        (idx, sample_weight_core, sw_info) = sample_global_by_lewis_class_floor(lewis_weights=lewis_weights, y_train=y_train_str.values, target_size=theory_size, seed=seed + stable_int_hash(f'{method}_{theory_size}'), uniform_mix=mix, class_floor_mix=float(class_floor_mix), return_inverse_probability_weights=bool(args.theory_use_sample_weight), weight_clip_quantile=float(args.theory_weight_clip_quantile), rescale_weight_sum_to_n=True)
                        sampling_runtime = now() - t_sample
                        idx = np.asarray(idx, dtype=int)
                        if len(idx) == 0:
                            LOGGER.warning('Skipping theory/classfloor method=%s because sampled index set is empty', method)
                            continue
                        X_core_pca = X_train_pca[idx]
                        y_core_str = y_train_str.values[idx]
                        y_core_encoded = le.transform(y_core_str)
                        t0 = now()
                        clf = fit_logistic_regression_github(X_core_pca, y_core_encoded, args.C, args.solver, args.max_iter, args.n_jobs, sample_weight=sample_weight_core)
                        fit_runtime = now() - t0
                        pred_start = now()
                        eval_out = evaluate_model(clf, svd, X_test_original, X_test_pca, y_test_encoded, X_train_pca, y_train_encoded, X_core_pca, y_core_encoded, C=args.C)
                        prediction_runtime = now() - pred_start
                        eval_out.update(biological_recall_metrics(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), y_train_str.values, classes))
                        pres_diag = {}
                        if sample_weight_core is not None:
                            pres_diag = weighted_loss_preservation_diagnostics(full_clf=full_clf, core_clf=clf, X_train_full_pca=X_train_pca, y_train_full_encoded=y_train_encoded, X_core_pca=X_core_pca, y_core_encoded=y_core_encoded, sample_weight_core=sample_weight_core, C=args.C, n_classes_full=len(classes))
                        beta_summary = coefficient_summary_metrics(clf, full_clf, svd)
                        obj = float(eval_out['objective_full_train_objective_total'])
                        rel_gap = (obj - full_obj_ref) / max(abs(full_obj_ref), 1e-300)
                        observed_size = int(len(idx))
                        observed_ratio = float(observed_size / max(len(train_idx), 1))
                        sampling_note = f'Theory/classfloor CorLewis: total size alpha*d*mu^2/eps^2 with mu_{stat}={theory_mu_val:.4g}, epsilon={eps:g}, alpha={alpha:g}; global p=(Lewis + class floor + uniform), uniform_mix={mix:g}, class_floor_mix={float(class_floor_mix):g}; sample_weight={bool(sample_weight_core is not None)}, weight_clip_q={float(args.theory_weight_clip_quantile):g}.'
                        base = {'dataset_name': args.dataset_name, 'seed': seed, 'n_pcs': int(n_pcs), 'embedding': 'github_add_pca_truncated_svd_global', 'method': method, 'method_family': 'coreset', 'method_base_family': 'corLewis_theory_classfloor_size', 'sampling_note': sampling_note, 'coreset_ratio_target': theory_ratio, 'coreset_size_target': int(theory_size), 'coreset_size_observed': observed_size, 'coreset_ratio_observed': observed_ratio, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'n_classes': int(len(classes)), 'C': float(args.C), 'solver': args.solver, 'max_iter': int(args.max_iter), 'runtime_sec': float(embedding_runtime + lewis_runtime + sampling_runtime + fit_runtime + prediction_runtime), 'model_fit_runtime_sec': float(fit_runtime), 'prediction_runtime_sec': float(prediction_runtime), 'sampling_runtime_sec': float(sampling_runtime), 'embedding_runtime_sec': float(embedding_runtime), 'lewis_runtime_sec': float(lewis_runtime), 'rss_mb': rss_mb(), 'quota_rule': 'theory_global_mu_size_global_lewis_class_floor', 'lewis_uniform_mix_used': mix, 'theory_class_floor_mix_used': float(class_floor_mix), 'theory_mu_stat': stat, 'theory_mu_value': theory_mu_val, 'theory_epsilon': eps, 'theory_alpha': alpha, 'theory_d_eff': int(choice['theory_d_eff']), 'theory_use_sample_weight': bool(args.theory_use_sample_weight), **sw_info}
                        row = dict(base)
                        for (k, v) in eval_out.items():
                            if k not in {'pred_test_encoded', 'probs_test_softmax', 'expanded_coef', 'expanded_intercept'}:
                                row[k] = v
                        row.update(beta_summary)
                        if sample_weight_core is not None:
                            row.update(pres_diag)
                        row['relative_objective_gap_vs_full_lr'] = float(rel_gap)
                        append_rows_csv(paths['metrics'], [row])
                        append_rows_csv(paths['class_metrics'], class_metric_rows(y_test_encoded, np.asarray(eval_out['pred_test_encoded'], dtype=int), classes, base, y_train_str.values, y_core_str, mu_lookup=mu_lookup, quota_lookup=None))
                        append_rows_csv(paths['coreset_composition'], composition_rows(y_train_str.values, idx, classes, base, mu_lookup=mu_lookup, quota_lookup=None))
                        append_rows_csv(paths['beta_comparison'], beta_comparison_rows(clf, full_clf, svd, classes, base))
                        if args.save_marker_gene_rows and method in marker_methods and ratio_allowed(observed_ratio, args.marker_gene_ratios):
                            append_rows_csv(paths['marker_gene_comparison'], marker_gene_rows(clf, full_clf, svd, classes, gene_names, base, top_k=args.marker_top_k))
                        if args.save_selected_cell_sample:
                            append_rows_csv(paths['selected_cells'], selected_cell_sample_rows(base=base, obs_names=adata_train.obs_names.astype(str), y_train=y_train_str.values, idx=idx, X_train_pca=X_train_pca, lewis_weights=lewis_weights, mu_lookup=mu_lookup, max_rows=args.selected_cell_sample_max, seed=seed + stable_int_hash(f'selected_{method}')))
                        gc.collect()
            del X_svd_all, X_train_pca, X_test_pca, full_clf, svd
            gc.collect()
    LOGGER.info('Done. Outputs written to %s', out_dir)
    LOGGER.info('Key files: metrics.csv, class_metrics.csv, coreset_composition.csv, beta_comparison.csv, mu_allocation_by_class.csv')
if __name__ == '__main__':
    main()
