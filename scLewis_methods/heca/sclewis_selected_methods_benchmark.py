from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
try:
    import celltypist
except Exception:
    celltypist = None
LOGGER = logging.getLogger('sclewis_selected_methods')

def setup_logging(verbose: bool=False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def now() -> float:
    return time.perf_counter()

def load_base_module(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Base script not found: {p}. Put corlewis_ovr_theory_benchmark.py in PROJECT_DIR or pass --base_script.')
    spec = importlib.util.spec_from_file_location('corlewis_base', str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not import base script from {p}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['corlewis_base'] = mod
    spec.loader.exec_module(mod)
    return mod

def parse_list(s: str) -> List[str]:
    return [x.strip() for x in re.split('[,;\\s]+', str(s).strip()) if x.strip()]

def parse_floats(s: str) -> List[float]:
    return [float(x) for x in parse_list(s)]

def tag_float(x: float) -> str:
    return f'{float(x):g}'.replace('-', 'm').replace('.', 'p')

def make_design_with_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.asarray(X, dtype=np.float64), np.ones((X.shape[0], 1), dtype=np.float64)])

def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def append_same_schema_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Optional[Sequence[str]]=None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if columns is not None:
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[list(columns)]
    if path.exists():
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

def largest_remainder_counts(raw: Mapping[str, float], total: int, caps: Mapping[str, int], min_per_class: int=0) -> Dict[str, int]:
    labels = list(raw.keys())
    score = np.asarray([max(float(raw[k]), 0.0) for k in labels], dtype=np.float64)
    caps_arr = np.asarray([int(caps.get(k, total)) for k in labels], dtype=int)
    if score.sum() <= 0:
        score = np.ones(len(labels), dtype=np.float64)
    score = score / score.sum()
    target = score * int(total)
    floor = np.floor(target).astype(int)
    min_arr = np.asarray([min(int(min_per_class), c) for c in caps_arr], dtype=int)
    counts = np.maximum(floor, min_arr)
    counts = np.minimum(counts, caps_arr)
    if counts.sum() > total:
        removable = counts - np.minimum(min_arr, counts)
        order = np.argsort(target - floor)
        for j in order:
            while counts.sum() > total and removable[j] > 0:
                counts[j] -= 1
                removable[j] -= 1
        if counts.sum() > total:
            order = np.argsort(counts)[::-1]
            for j in order:
                while counts.sum() > total and counts[j] > 0:
                    counts[j] -= 1
        return {k: int(v) for (k, v) in zip(labels, counts)}
    leftover = int(total - counts.sum())
    remainder = target - floor
    order = np.argsort(-remainder, kind='mergesort')
    while leftover > 0:
        progressed = False
        for j in order:
            if leftover <= 0:
                break
            if counts[j] < caps_arr[j]:
                counts[j] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return {k: int(v) for (k, v) in zip(labels, counts)}

def frequency_mu_quotas(y_train: Sequence[str], mu_df: pd.DataFrame, target_size: int, alpha: float, beta: float, min_per_class: int, mu_floor: float=1.0, mu_cap: float=10.0, freq_floor: float=1e-12) -> Tuple[Dict[str, int], pd.DataFrame]:
    y = np.asarray(y_train).astype(str)
    counts = pd.Series(y).value_counts().astype(int)
    n = len(y)
    mu_lookup = dict(zip(mu_df['class_label'].astype(str), pd.to_numeric(mu_df['mu'], errors='coerce')))
    raw: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []
    for (cls, cnt) in counts.items():
        cls = str(cls)
        freq = max(float(cnt) / max(n, 1), float(freq_floor))
        mu = float(mu_lookup.get(cls, np.nan))
        mu_used = float(np.clip(mu if np.isfinite(mu) else mu_floor, mu_floor, mu_cap))
        raw[cls] = freq ** float(alpha) * mu_used ** float(beta)
    quotas = largest_remainder_counts(raw, int(target_size), caps=counts.to_dict(), min_per_class=min_per_class)
    total_score = max(sum(raw.values()), 1e-300)
    for (cls, cnt) in counts.items():
        cls = str(cls)
        freq = max(float(cnt) / max(n, 1), float(freq_floor))
        mu = float(mu_lookup.get(cls, np.nan))
        mu_used = float(np.clip(mu if np.isfinite(mu) else mu_floor, mu_floor, mu_cap))
        rows.append({'class_label': cls, 'class_count': int(cnt), 'frequency': float(cnt) / max(n, 1), 'mu': mu, 'mu_used': mu_used, 'quota_score': float(raw[cls]), 'quota_prob': float(raw[cls] / total_score), 'quota_target': int(quotas[cls]), 'selected_count': np.nan, 'alpha_freq': float(alpha), 'beta_mu': float(beta)})
    return (quotas, pd.DataFrame(rows))

def fallback_mu_blend_quotas(y_train: Sequence[str], mu_df: pd.DataFrame, target_size: int, eta: float, min_per_class: int, mu_power: float=2.0) -> Tuple[Dict[str, int], pd.DataFrame]:
    y = np.asarray(y_train).astype(str)
    counts = pd.Series(y).value_counts().astype(int)
    labels = sorted(counts.index.astype(str).tolist())
    caps = counts.to_dict()
    equal_raw = {c: 1.0 for c in labels}
    mu_lookup = dict(zip(mu_df['class_label'].astype(str), pd.to_numeric(mu_df['mu'], errors='coerce')))
    mu_raw = {c: max(float(mu_lookup.get(c, 1.0)), 1.0) ** float(mu_power) for c in labels}
    equal_sum = sum(equal_raw.values())
    mu_sum = sum(mu_raw.values())
    raw = {c: (1.0 - float(eta)) * (equal_raw[c] / equal_sum) + float(eta) * (mu_raw[c] / mu_sum) for c in labels}
    quotas = largest_remainder_counts(raw, int(target_size), caps=caps, min_per_class=min_per_class)
    rows = []
    for c in labels:
        rows.append({'class_label': c, 'class_count': int(caps[c]), 'frequency': float(caps[c] / len(y)), 'mu': float(mu_lookup.get(c, np.nan)), 'quota_score': float(raw[c]), 'quota_prob': float(raw[c] / max(sum(raw.values()), 1e-300)), 'quota_target': int(quotas[c]), 'selected_count': np.nan, 'mu_blend_eta': float(eta)})
    return (quotas, pd.DataFrame(rows))

def load_existing_split(adata, split_csv: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not split_csv:
        return None
    p = Path(split_csv)
    if not p.exists():
        return None
    split = pd.read_csv(p)
    if not {'obs_name', 'split'}.issubset(split.columns):
        return None
    obs_to_idx = pd.Series(np.arange(adata.n_obs), index=adata.obs_names.astype(str))
    idx = split['obs_name'].astype(str).map(obs_to_idx)
    if idx.isna().any():
        return None
    idx = idx.astype(int).to_numpy()
    lab = split['split'].astype(str).to_numpy()
    return (idx[lab == 'train'], idx[lab == 'test'])

def write_split(adata, train_idx: np.ndarray, test_idx: np.ndarray, out_path: Path) -> None:
    split = pd.DataFrame({'obs_name': adata.obs_names.astype(str), 'split': 'unused'})
    split.loc[train_idx, 'split'] = 'train'
    split.loc[test_idx, 'split'] = 'test'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_path, index=False)

def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, probs: Optional[np.ndarray], classes: Sequence[str]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out: Dict[str, float] = {'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))}
    if probs is not None:
        P = np.asarray(probs, dtype=float)
        P = np.clip(P, 1e-15, 1.0 - 1e-15)
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
        K = min(len(classes), P.shape[1])
        y_safe = np.clip(y_true, 0, K - 1)
        try:
            class_counts = np.bincount(y_safe, minlength=K).astype(np.float64)
            class_weights = 1.0 / (class_counts + 1e-15)
            weighted_total = 0.0
            unweighted_total = 0.0
            for c in range(K):
                y_binary = (y_safe == c).astype(float)
                class_loss = -np.sum(y_binary * np.log(P[:, c]) + (1.0 - y_binary) * np.log(1.0 - P[:, c]))
                weighted_total += class_weights[c] * class_loss
                unweighted_total += class_loss
            out['ovr_log_loss_weighted_test_original_softmax'] = float(weighted_total / max(len(y_safe), 1))
            out['ovr_log_loss_unweighted_test_original_softmax'] = float(unweighted_total / max(len(y_safe), 1))
        except Exception:
            out['ovr_log_loss_weighted_test_original_softmax'] = np.nan
            out['ovr_log_loss_unweighted_test_original_softmax'] = np.nan
        try:
            out['multiclass_log_loss_test_original_softmax'] = float(log_loss(y_safe, P[:, :K], labels=list(range(K))))
            out['trueclass_nll_micro_test'] = out['multiclass_log_loss_test_original_softmax']
        except Exception:
            out['multiclass_log_loss_test_original_softmax'] = np.nan
            out['trueclass_nll_micro_test'] = np.nan
        try:
            out['roc_auc_ovr_macro_original_softmax'] = float(roc_auc_score(y_safe, P[:, :K], multi_class='ovr', average='macro', labels=list(range(K))))
            out['log10_one_minus_auc_ovr_macro_original_softmax'] = float(np.log10(max(1.0 - out['roc_auc_ovr_macro_original_softmax'], 1e-15)))
        except Exception:
            out['roc_auc_ovr_macro_original_softmax'] = np.nan
            out['log10_one_minus_auc_ovr_macro_original_softmax'] = np.nan
    return out

def class_balanced_loss_metrics(eval_like: Mapping[str, object], y_test_encoded: np.ndarray, y_train_labels: Sequence[str], classes: Sequence[str], mu_df: pd.DataFrame) -> Dict[str, float]:
    P = np.asarray(eval_like.get('probs_test_softmax'), dtype=float)
    y = np.asarray(y_test_encoded, dtype=int)
    out: Dict[str, float] = {}
    if P.ndim != 2 or len(y) != P.shape[0]:
        return out
    eps = 1e-15
    P = np.clip(P, eps, 1.0)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
    y_safe = np.clip(y, 0, P.shape[1] - 1)
    nll = -np.log(P[np.arange(len(y_safe)), y_safe])
    K = min(len(classes), P.shape[1])
    class_nll = np.full(K, np.nan, dtype=float)
    for c in range(K):
        mask = y_safe == c
        if mask.any():
            class_nll[c] = float(np.mean(nll[mask]))
    valid = np.isfinite(class_nll)
    out['trueclass_nll_macro_test'] = float(np.nanmean(class_nll)) if valid.any() else np.nan
    out['trueclass_nll_median_class_test'] = float(np.nanmedian(class_nll)) if valid.any() else np.nan
    out['trueclass_nll_worst_class_test'] = float(np.nanmax(class_nll)) if valid.any() else np.nan
    train_counts = pd.Series(np.asarray(y_train_labels).astype(str)).value_counts()
    counts = np.array([train_counts.get(str(cls), 0) for cls in classes[:K]], dtype=float)
    if (counts > 0).any():
        q25 = np.quantile(counts[counts > 0], 0.25)
        q10 = np.quantile(counts[counts > 0], 0.1)
        q05 = np.quantile(counts[counts > 0], 0.05)
        bottom25 = (counts <= q25) & (counts > 0)
        bottom10 = (counts <= q10) & (counts > 0)
        bottom5 = (counts <= q05) & (counts > 0)
        out['trueclass_nll_bottom25_macro_test'] = float(np.nanmean(class_nll[bottom25])) if np.isfinite(class_nll[bottom25]).any() else np.nan
        out['trueclass_nll_bottom10_macro_test'] = float(np.nanmean(class_nll[bottom10])) if np.isfinite(class_nll[bottom10]).any() else np.nan
        out['trueclass_nll_bottom5_macro_test'] = float(np.nanmean(class_nll[bottom5])) if np.isfinite(class_nll[bottom5]).any() else np.nan
    lookup = dict(zip(mu_df['class_label'].astype(str), pd.to_numeric(mu_df['mu'], errors='coerce')))
    mu = np.array([lookup.get(str(cls), np.nan) for cls in classes[:K]], dtype=float)
    if np.isfinite(mu).any():
        finite_mu = mu[np.isfinite(mu)]
        q90 = np.nanquantile(finite_mu, 0.9)
        q95 = np.nanquantile(finite_mu, 0.95)
        q10 = np.nanquantile(finite_mu, 0.1)
        high90 = np.isfinite(mu) & (mu >= q90) & valid
        high95 = np.isfinite(mu) & (mu >= q95) & valid
        low10 = np.isfinite(mu) & (mu <= q10) & valid
        out['trueclass_nll_highmu_q90_macro_test'] = float(np.nanmean(class_nll[high90])) if np.isfinite(class_nll[high90]).any() else np.nan
        out['trueclass_nll_highmu_q95_macro_test'] = float(np.nanmean(class_nll[high95])) if np.isfinite(class_nll[high95]).any() else np.nan
        out['trueclass_nll_lowmu_q10_macro_test'] = float(np.nanmean(class_nll[low10])) if np.isfinite(class_nll[low10]).any() else np.nan
        w = np.square(np.where(np.isfinite(mu), mu, np.nanmedian(finite_mu)))
        w = np.where(valid, w, 0.0)
        out['trueclass_nll_mu2_weighted_macro_test'] = float(np.nansum(w * class_nll) / np.sum(w)) if w.sum() > 0 else np.nan
    return out

def class_metric_rows(y_true: np.ndarray, y_pred: np.ndarray, y_train_full: Sequence[str], y_core: Sequence[str], classes: Sequence[str], mu_df: pd.DataFrame, info: Mapping[str, object]) -> List[Dict[str, object]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    train_counts = pd.Series(np.asarray(y_train_full).astype(str)).value_counts()
    core_counts = pd.Series(np.asarray(y_core).astype(str)).value_counts()
    mu_lookup = dict(zip(mu_df['class_label'].astype(str), pd.to_numeric(mu_df['mu'], errors='coerce')))
    rows = []
    for (c, cls) in enumerate(classes):
        cls = str(cls)
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        support = int(np.sum(y_true == c))
        pred_count = int(np.sum(y_pred == c))
        recall = tp / support if support else np.nan
        precision = tp / pred_count if pred_count else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall > 0) else np.nan
        rows.append({**info, 'class_label': cls, 'class_encoded': int(c), 'support': support, 'pred_count': pred_count, 'true_positive': tp, 'precision': precision, 'recall': recall, 'f1': f1, 'n_train_full': int(train_counts.get(cls, 0)), 'n_train_coreset': int(core_counts.get(cls, 0)), 'mu': float(mu_lookup.get(cls, np.nan))})
    return rows

def add_subset_recall_metrics(row: Dict[str, object], class_rows: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(class_rows)
    if df.empty:
        return
    counts = pd.to_numeric(df['n_train_full'], errors='coerce')
    recalls = pd.to_numeric(df['recall'], errors='coerce')
    if (counts > 0).any():
        for (q, name) in [(0.05, 'bottom5'), (0.1, 'bottom10'), (0.25, 'bottom25'), (0.75, 'top25_abundance')]:
            if name == 'top25_abundance':
                cut = counts[counts > 0].quantile(q)
                mask = counts >= cut
            else:
                cut = counts[counts > 0].quantile(q)
                mask = (counts <= cut) & (counts > 0)
            row[f'{name}_recall_mean'] = float(recalls[mask].mean()) if mask.any() else np.nan
    mu = pd.to_numeric(df['mu'], errors='coerce')
    if mu.notna().any():
        q90 = mu.quantile(0.9)
        q95 = mu.quantile(0.95)
        q10 = mu.quantile(0.1)
        row['highmu_q90_recall_mean'] = float(recalls[mu >= q90].mean())
        row['highmu_q95_recall_mean'] = float(recalls[mu >= q95].mean())
        row['lowmu_q10_recall_mean'] = float(recalls[mu <= q10].mean())

def align_betas(base, clf, n_classes: int, svd) -> Tuple[np.ndarray, np.ndarray]:
    (B_svd, b, present) = base.align_clf_coef_intercept_to_global_classes(clf, n_classes_full=n_classes, missing_intercept=-40.0)
    B_gene = np.asarray(B_svd, dtype=np.float64).dot(np.asarray(svd.components_, dtype=np.float64))
    return (B_svd, B_gene)

def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(num, den, out=np.full(len(num), np.nan, dtype=float), where=den > 0)

def row_rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(b, axis=1)
    return np.divide(np.linalg.norm(a - b, axis=1), den, out=np.full(len(den), np.nan, dtype=float), where=den > 0)

def class_beta_rows(method: str, ratio: float, classes: Sequence[str], B_svd: np.ndarray, B_gene: np.ndarray, full_svd: np.ndarray, full_gene: np.ndarray, genes: Sequence[str], info: Mapping[str, object], ks=(20, 50, 100)) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    cos_svd = row_cosine(B_svd, full_svd)
    cos_gene = row_cosine(B_gene, full_gene)
    rel_svd = row_rel_l2(B_svd, full_svd)
    rel_gene = row_rel_l2(B_gene, full_gene)
    beta_rows = []
    marker_rows = []
    genes = np.asarray(genes).astype(str)
    for (i, cls) in enumerate(classes):
        row = {**info, 'method': method, 'coreset_ratio_target': ratio, 'class_label': str(cls), 'class_encoded': int(i), 'beta_cosine_pca': float(cos_svd[i]), 'beta_cosine_gene': float(cos_gene[i]), 'beta_relative_l2_pca': float(rel_svd[i]), 'beta_relative_l2_gene': float(rel_gene[i])}
        for k in ks:
            kk = min(int(k), B_gene.shape[1])
            top_m = np.argsort(-np.abs(B_gene[i]))[:kk]
            top_f = np.argsort(-np.abs(full_gene[i]))[:kk]
            inter = len(set(top_m.tolist()) & set(top_f.tolist()))
            row[f'marker_overlap_at_{kk}_class'] = inter / max(kk, 1)
        beta_rows.append(row)
        kk = min(50, B_gene.shape[1])
        rank_m = {int(g): r + 1 for (r, g) in enumerate(np.argsort(-np.abs(B_gene[i]))[:kk])}
        rank_f = {int(g): r + 1 for (r, g) in enumerate(np.argsort(-np.abs(full_gene[i]))[:kk])}
        union = sorted(set(rank_m) | set(rank_f), key=lambda g: min(rank_m.get(g, 10 ** 9), rank_f.get(g, 10 ** 9)))
        for g in union:
            marker_rows.append({**info, 'method': method, 'coreset_ratio_target': ratio, 'class_label': str(cls), 'class_encoded': int(i), 'gene': str(genes[g]), 'gene_index': int(g), 'top_k': kk, 'in_core_top_k': g in rank_m, 'in_opt_top_k': g in rank_f, 'core_rank': rank_m.get(g, np.nan), 'opt_rank': rank_f.get(g, np.nan), 'core_coef_gene': float(B_gene[i, g]), 'opt_coef_gene': float(full_gene[i, g])})
    return (beta_rows, marker_rows)

def selected_cell_rows(adata, global_indices: np.ndarray, method: str, ratio: float, labels: Sequence[str], info: Mapping[str, object]) -> List[Dict[str, object]]:
    rows = []
    has_umap = 'X_umap' in getattr(adata, 'obsm', {})
    umap = adata.obsm['X_umap'] if has_umap else None
    obs_names = np.asarray(adata.obs_names.astype(str))
    labels = np.asarray(labels).astype(str)
    for (local_order, idx) in enumerate(global_indices):
        rec = {**info, 'method': method, 'coreset_ratio_target': ratio, 'selected_order': int(local_order), 'obs_name': str(obs_names[idx]), 'global_obs_index': int(idx), 'class_label': str(labels[idx])}
        if has_umap and umap is not None and (umap.shape[1] >= 2):
            rec['umap1'] = float(umap[idx, 0])
            rec['umap2'] = float(umap[idx, 1])
        rows.append(rec)
    return rows

def celltypist_train_kwargs(method: str, labels: np.ndarray, genes: np.ndarray, n_jobs: int, alpha_from_C: bool, sgd_C: float, sgd_alpha: float, batch_number: int, batch_size: int, epochs: int, with_mean: bool) -> Tuple[str, Dict[str, object]]:
    method_l = method.strip().lower()
    effective_alpha = 1.0 / (float(sgd_C) * max(int(len(labels)), 1)) if alpha_from_C else float(sgd_alpha)
    base = {'labels': labels, 'genes': genes, 'n_jobs': int(n_jobs), 'with_mean': bool(with_mean), 'check_expression': False}
    if method_l in {'celltypist_mini_batch_sgd', 'celltypist_sgd', 'sgd', 'mini_batch_sgd'}:
        base.update({'use_SGD': True, 'mini_batch': True, 'alpha': float(effective_alpha), 'batch_number': int(batch_number), 'batch_size': int(batch_size), 'epochs': int(epochs), 'balance_cell_type': False, 'feature_selection': False})
        return ('CellTypist_SVD_mini_batch_SGD', base)
    if method_l in {'celltypist_mini_batch_sgd_balanced', 'celltypist_sgd_balanced', 'sgd_balanced', 'mini_batch_sgd_balanced'}:
        base.update({'use_SGD': True, 'mini_batch': True, 'alpha': float(effective_alpha), 'batch_number': int(batch_number), 'batch_size': int(batch_size), 'epochs': int(epochs), 'balance_cell_type': True, 'feature_selection': False})
        return ('CellTypist_SVD_mini_batch_SGD_balanced', base)
    raise ValueError(f'Unsupported CellTypist method: {method}')

def predict_celltypist_matrix(model, X_test: np.ndarray, genes: Sequence[str], classes: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    clf = getattr(model, 'classifier', None)
    scaler = getattr(model, 'scaler', None)
    if clf is None or scaler is None:
        raise RuntimeError('Unexpected CellTypist model missing classifier/scaler')
    model_features = getattr(clf, 'features', None)
    if model_features is None:
        model_features = getattr(model, 'features', None)
    if model_features is None:
        model_features = genes
    model_features = [str(g) for g in np.asarray(model_features).ravel()]
    gene_lookup = {str(g): i for (i, g) in enumerate(genes)}
    idx = np.array([gene_lookup[g] for g in model_features if g in gene_lookup], dtype=int)
    if len(idx) != len(model_features):
        missing = [g for g in model_features if g not in gene_lookup]
        raise RuntimeError(f'CellTypist missing model features in query: {missing[:5]}')
    Xq = np.asarray(X_test, dtype=np.float32)[:, idx]
    Xs = scaler.transform(Xq)
    Xs = np.asarray(Xs)
    Xs[Xs > 10] = 10
    decision = clf.decision_function(Xs)
    clf_classes = np.asarray(clf.classes_).astype(str)
    if np.ndim(decision) == 1:
        decision_2d = np.vstack([-decision, decision]).T
    else:
        decision_2d = np.asarray(decision, dtype=float)
    pred_idx = np.argmax(decision_2d, axis=1)
    y_pred_str = clf_classes[pred_idx]
    prob_raw = 1.0 / (1.0 + np.exp(-np.clip(decision_2d, -50, 50)))
    P = np.zeros((prob_raw.shape[0], len(classes)), dtype=np.float64)
    lookup = {str(c): j for (j, c) in enumerate(classes)}
    for (k, cls) in enumerate(clf_classes):
        if str(cls) in lookup and k < prob_raw.shape[1]:
            P[:, lookup[str(cls)]] = prob_raw[:, k]
    row_sum = P.sum(axis=1, keepdims=True)
    bad = row_sum.ravel() <= 0
    if np.any(bad):
        P[bad, :] = 1.0 / len(classes)
        row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sum, 1e-300)
    return (y_pred_str, P)

def get_model_features(model, clf, input_genes: Sequence[str], n_coef_features: int) -> List[str]:
    for cand in [getattr(clf, 'features', None), getattr(model, 'features', None), getattr(clf, 'feature_names_in_', None), getattr(model, 'genes', None)]:
        if cand is None:
            continue
        arr = np.asarray(cand).ravel().astype(str).tolist()
        if len(arr) == n_coef_features:
            return arr
    input_genes = list(map(str, input_genes))
    if n_coef_features == len(input_genes):
        return input_genes
    return input_genes[:n_coef_features]

def align_coef_to_global_classes(coef: np.ndarray, intercept: np.ndarray, coef_classes: Sequence[object], global_classes: Sequence[str], missing_intercept: float=-40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).ravel()
    coef_classes = np.asarray(coef_classes).astype(str)
    global_classes = np.asarray(global_classes).astype(str)
    B = np.zeros((len(global_classes), coef.shape[1]), dtype=np.float64)
    b = np.full(len(global_classes), float(missing_intercept), dtype=np.float64)
    present = np.zeros(len(global_classes), dtype=bool)
    lookup = {c: i for (i, c) in enumerate(global_classes)}
    if coef.shape[0] == 1 and len(coef_classes) == 2:
        (c0, c1) = (str(coef_classes[0]), str(coef_classes[1]))
        if c1 in lookup:
            j = lookup[c1]
            B[j] = coef[0]
            b[j] = intercept[0] if len(intercept) else 0.0
            present[j] = True
        if c0 in lookup:
            j = lookup[c0]
            B[j] = -coef[0]
            b[j] = -intercept[0] if len(intercept) else 0.0
            present[j] = True
    else:
        for (r, cls) in enumerate(coef_classes[:coef.shape[0]]):
            cls = str(cls)
            if cls in lookup:
                j = lookup[cls]
                B[j] = coef[r]
                if r < len(intercept):
                    b[j] = intercept[r]
                present[j] = True
    return (B, b, present)

def extract_celltypist_coef_svd_raw(model, input_genes: Sequence[str], global_classes: Sequence[str], n_features_total: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    clf = getattr(model, 'classifier', None)
    scaler = getattr(model, 'scaler', None)
    if clf is None:
        raise RuntimeError('CellTypist model missing classifier')
    coef_scaled = np.asarray(clf.coef_, dtype=np.float64)
    intercept_scaled = np.asarray(clf.intercept_, dtype=np.float64).ravel()
    clf_classes = np.asarray(clf.classes_).astype(str)
    model_features = get_model_features(model, clf, input_genes, coef_scaled.shape[1])
    if scaler is not None and hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
        scale = np.asarray(scaler.scale_, dtype=np.float64).ravel()
        mean = np.asarray(scaler.mean_, dtype=np.float64).ravel()
        if len(scale) != coef_scaled.shape[1]:
            scale = np.ones(coef_scaled.shape[1])
            mean = np.zeros(coef_scaled.shape[1])
    else:
        scale = np.ones(coef_scaled.shape[1])
        mean = np.zeros(coef_scaled.shape[1])
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef_raw_selected = coef_scaled / scale[None, :]
    intercept_raw_selected = intercept_scaled - coef_raw_selected.dot(mean)
    lookup = {str(g): i for (i, g) in enumerate(input_genes)}
    coef_raw_total = np.zeros((coef_raw_selected.shape[0], int(n_features_total)), dtype=np.float64)
    for (j, feat) in enumerate(model_features):
        if str(feat) in lookup:
            coef_raw_total[:, lookup[str(feat)]] = coef_raw_selected[:, j]
    (B_svd, b_svd, present) = align_coef_to_global_classes(coef_raw_total, intercept_raw_selected, clf_classes, global_classes)
    return (B_svd, b_svd, present, model_features)

def main() -> None:
    ap = argparse.ArgumentParser(description='Final selected scLewis/CellTypist benchmark driver.')
    ap.add_argument('--base_script', default=str(Path(__file__).resolve().with_name('corlewis_ovr_theory_benchmark.py')))
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--dataset_name', default='dataset')
    ap.add_argument('--label_key', default='cell_type')
    ap.add_argument('--mu_by_class', required=True, help='mu_by_class.csv from the dataset-specific μ calculation.')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--reference_result_dir', default='', help='Optional prior result folder to reuse split_seed_<seed>.csv')
    ap.add_argument('--split_csv', default='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--n_pcs', type=int, default=100)
    ap.add_argument('--target_sum', type=float, default=10000.0)
    ap.add_argument('--svd_random_state', type=int, default=42)
    ap.add_argument('--ratios', default='0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30,0.40,0.50,0.60')
    ap.add_argument('--methods', default='random_global,freqmu_alpha1_beta0,freqmu_alpha0_beta1,mu_blend_eta0p1,mu_blend_eta0p25')
    ap.add_argument('--run_celltypist', action='store_true')
    ap.add_argument('--celltypist_methods', default='celltypist_mini_batch_sgd,celltypist_mini_batch_sgd_balanced')
    ap.add_argument('--celltypist_sgd_C', type=float, default=0.0001)
    ap.add_argument('--celltypist_sgd_alpha', type=float, default=0.0001)
    ap.add_argument('--celltypist_sgd_alpha_from_C', action='store_true')
    ap.add_argument('--celltypist_sgd_batch_number', type=int, default=100)
    ap.add_argument('--celltypist_sgd_batch_size', type=int, default=1000)
    ap.add_argument('--celltypist_sgd_epochs', type=int, default=10)
    ap.add_argument('--celltypist_with_mean', action='store_true', default=True)
    ap.add_argument('--min_per_class', type=int, default=5)
    ap.add_argument('--lewis_uniform_mix', type=float, default=0.05)
    ap.add_argument('--lewis_iter', type=int, default=20)
    ap.add_argument('--lewis_damping', type=float, default=0.5)
    ap.add_argument('--lewis_ridge', type=float, default=1e-06)
    ap.add_argument('--lewis_block_size', type=int, default=50000)
    ap.add_argument('--C', type=float, default=0.1)
    ap.add_argument('--solver', default='lbfgs')
    ap.add_argument('--max_iter', type=int, default=500)
    ap.add_argument('--n_jobs', type=int, default=16)
    ap.add_argument('--max_selected_cells_per_method_ratio', type=int, default=0, help='0 means save all selected cells')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    setup_logging(args.verbose)
    out = Path(args.output_dir)
    if args.overwrite and out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    base = load_base_module(args.base_script)
    LOGGER.info('Loading h5ad: %s', args.dataset)
    adata = base.read_h5ad(args.dataset)
    if args.label_key not in adata.obs:
        raise KeyError(f'label_key {args.label_key!r} not found')
    y_all = adata.obs[args.label_key].astype(str).values
    LOGGER.info('Normalizing/log1p')
    base.normalize_total_log1p_inplace(adata, target_sum=args.target_sum)
    split_csv = args.split_csv
    if not split_csv and args.reference_result_dir:
        cand = Path(args.reference_result_dir) / 'splits' / f'split_seed_{args.seed}.csv'
        if cand.exists():
            split_csv = str(cand)
    split = load_existing_split(adata, split_csv)
    if split is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        (train_idx, test_idx) = next(splitter.split(np.zeros(len(y_all)), y_all))
    else:
        (train_idx, test_idx) = split
    write_split(adata, train_idx, test_idx, out / 'splits' / f'split_seed_{args.seed}.csv')
    y_train_str = y_all[train_idx]
    y_test_str = y_all[test_idx]
    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_str, y_test_str])))
    classes = list(map(str, le.classes_))
    y_train_encoded = le.transform(y_train_str)
    y_test_encoded = le.transform(y_test_str)
    LOGGER.info('Fitting SVD n_pcs=%d', args.n_pcs)
    (X_svd_all, svd, embed_runtime) = base.fit_github_svd_full_matrix(adata.X, args.n_pcs, seed=args.svd_random_state)
    X_train = X_svd_all[train_idx]
    X_test = X_svd_all[test_idx]
    X_test_original = adata[test_idx].X
    genes = np.asarray(adata.var_names.astype(str))
    svd_feature_names = np.array([f'SVD{j + 1:03d}' for j in range(X_train.shape[1])], dtype=object)
    LOGGER.info('Loading mu_by_class')
    mu_df = base.load_mu_by_class(args.mu_by_class, classes, seed=args.seed, n_pcs=args.n_pcs, prefer_intercept_added=None, solver='any')
    mu_df['class_label'] = mu_df['class_label'].astype(str)
    mu_df['mu'] = pd.to_numeric(mu_df['mu'], errors='coerce')
    metrics_rows: List[Dict[str, object]] = []
    class_rows_all: List[Dict[str, object]] = []
    beta_rows_all: List[Dict[str, object]] = []
    marker_rows_all: List[Dict[str, object]] = []
    alloc_rows_all: List[Dict[str, object]] = []
    provenance: List[Dict[str, object]] = []
    LOGGER.info('Training full LR reference')
    t0 = now()
    full_clf = base.fit_logistic_regression_github(X_train, y_train_encoded, args.C, args.solver, args.max_iter, args.n_jobs)
    full_fit_runtime = now() - t0
    full_eval = base.evaluate_model(full_clf, svd, X_test_original, X_test, y_test_encoded, X_train, y_train_encoded, X_train, y_train_encoded, C=args.C, n_classes_full=len(classes))
    full_obj = float(full_eval.get('objective_full_train_objective_total', np.nan))
    full_pred = np.asarray(full_eval['pred_test_encoded'], dtype=int)
    full_row = {k: v for (k, v) in full_eval.items() if not isinstance(v, (np.ndarray, list))}
    full_row.update(class_balanced_loss_metrics(full_eval, y_test_encoded, y_train_str, classes, mu_df))
    full_class_rows = class_metric_rows(y_test_encoded, full_pred, y_train_str, y_train_str, classes, mu_df, {'method': 'full_lr_pca', 'seed': args.seed, 'n_pcs': args.n_pcs, 'coreset_ratio_target': 1.0, 'coreset_size_observed': len(train_idx), 'coreset_ratio_observed': 1.0})
    full_row.update(base.biological_recall_metrics(y_test_encoded, full_pred, y_train_str, classes))
    add_subset_recall_metrics(full_row, full_class_rows)
    full_row.update({'dataset_name': args.dataset_name, 'method': 'full_lr_pca', 'method_family': 'reference', 'seed': args.seed, 'n_pcs': args.n_pcs, 'n_train': len(train_idx), 'n_test': len(test_idx), 'n_classes': len(classes), 'coreset_ratio_target': 1.0, 'coreset_size_observed': len(train_idx), 'coreset_ratio_observed': 1.0, 'fit_runtime_sec': full_fit_runtime, 'runtime_sec': full_fit_runtime, 'relative_objective_gap_vs_full_lr': 0.0, 'objective_gap_is_posthoc': False, 'objective_gap_note': 'native Full LR reference objective'})
    metrics_rows.append(full_row)
    class_rows_all.extend(full_class_rows)
    (full_svd, full_gene) = align_betas(base, full_clf, len(classes), svd)
    (br, mr) = class_beta_rows('full_lr_pca', 1.0, classes, full_svd, full_gene, full_svd, full_gene, genes, {'dataset_name': args.dataset_name, 'seed': args.seed, 'n_pcs': args.n_pcs})
    beta_rows_all.extend(br)
    marker_rows_all.extend(mr)
    provenance.append({'method': 'full_lr_pca', 'backend': 'sklearn.LogisticRegression', 'parameters': json.dumps({'C': args.C, 'solver': args.solver, 'max_iter': args.max_iter})})
    methods = parse_list(args.methods)
    ratios = parse_floats(args.ratios)
    lewis_weights = None
    if methods:
        LOGGER.info('Computing Lewis weights')
        A = make_design_with_intercept(X_train)
        lewis_weights = base.approximate_l1_lewis_weights(A, n_iter=args.lewis_iter, ridge=args.lewis_ridge, damping=args.lewis_damping, block_size=args.lewis_block_size, verbose=False)
        diag = base.lewis_diagnostic_row(lewis_weights, {'method': 'selected_sclewis', 'seed': args.seed, 'n_pcs': args.n_pcs})
        try:
            diag.update(base.lewis_fixed_point_diagnostics(A, lewis_weights, ridge=args.lewis_ridge, block_size=args.lewis_block_size))
        except Exception as exc:
            LOGGER.warning('Lewis fixed point diagnostic failed: %s', exc)
        write_df(out / 'lewis_diagnostics.csv', pd.DataFrame([diag]))
    else:
        LOGGER.info('No scLewis methods requested; skipping Lewis-weight computation and selected-method ratio loop.')
    n_train = len(y_train_str)
    rng = np.random.default_rng(args.seed)
    selected_columns = ['dataset_name', 'seed', 'n_pcs', 'method', 'coreset_ratio_target', 'selected_order', 'obs_name', 'global_obs_index', 'class_label', 'umap1', 'umap2']
    for ratio in ratios:
        target = max(1, min(int(round(ratio * n_train)), n_train))
        for method in methods:
            LOGGER.info('ratio=%s target=%s method=%s', ratio, target, method)
            t_sample = now()
            alloc_rows: List[Dict[str, object]] = []
            if method == 'random_global':
                idx_local = base.random_global_indices(n_train, target, seed=args.seed + base.stable_int_hash(f'{method}_{ratio}'))
                provenance.append({'method': method, 'backend': 'numpy/sklearn utility', 'parameters': json.dumps({'rule': 'uniform random without replacement'})})
            elif method.startswith('freqmu_'):
                m = re.search('alpha(?P<a>[0-9p]+)_beta(?P<b>[0-9p]+)', method)
                if not m:
                    raise ValueError(f'Could not parse freqmu method: {method}')
                alpha = float(m.group('a').replace('p', '.'))
                beta = float(m.group('b').replace('p', '.'))
                (quotas, qdf) = frequency_mu_quotas(y_train_str, mu_df, target, alpha, beta, min_per_class=args.min_per_class)
                idx_local = base.sample_within_labels_by_lewis(y_train_str, quotas, lewis_weights, seed=args.seed + base.stable_int_hash(f'{method}_{ratio}'), uniform_mix=args.lewis_uniform_mix)
                qdf['dataset_name'] = args.dataset_name
                qdf['method'] = method
                qdf['seed'] = args.seed
                qdf['n_pcs'] = args.n_pcs
                qdf['coreset_ratio_target'] = ratio
                alloc_rows = qdf.to_dict('records')
                provenance.append({'method': method, 'backend': 'scLewis frequency-mu quota + l1 Lewis within class', 'parameters': json.dumps({'alpha': alpha, 'beta': beta, 'lewis_uniform_mix': args.lewis_uniform_mix})})
            elif method.startswith('mu_blend_eta'):
                eta_match = re.search('eta(?P<e>[0-9p]+)', method)
                eta = float(eta_match.group('e').replace('p', '.')) if eta_match else 0.25
                if hasattr(base, 'blended_mu_label_quotas'):
                    (quotas, qdf) = base.blended_mu_label_quotas(y_train=y_train_str, mu_df=mu_df, target_size=target, n_labels_full=len(classes), eta=eta, mu_power=2.0, mu_floor=1.0, mu_cap=10.0, min_per_class=args.min_per_class)
                else:
                    (quotas, qdf) = fallback_mu_blend_quotas(y_train_str, mu_df, target, eta=eta, min_per_class=args.min_per_class)
                idx_local = base.sample_within_labels_by_lewis(y_train_str, quotas, lewis_weights, seed=args.seed + base.stable_int_hash(f'{method}_{ratio}'), uniform_mix=args.lewis_uniform_mix)
                qdf['dataset_name'] = args.dataset_name
                qdf['method'] = method
                qdf['seed'] = args.seed
                qdf['n_pcs'] = args.n_pcs
                qdf['coreset_ratio_target'] = ratio
                alloc_rows = qdf.to_dict('records')
                provenance.append({'method': method, 'backend': 'scLewis mu-blend quota + l1 Lewis within class', 'parameters': json.dumps({'eta': eta, 'mu_power': 2.0, 'lewis_uniform_mix': args.lewis_uniform_mix})})
            else:
                raise ValueError(f'Unknown method: {method}')
            sampling_runtime = now() - t_sample
            global_idx = train_idx[idx_local]
            X_core = X_train[idx_local]
            y_core = y_train_encoded[idx_local]
            y_core_str = y_train_str[idx_local]
            t_fit = now()
            clf = base.fit_logistic_regression_github(X_core, y_core, args.C, args.solver, args.max_iter, args.n_jobs)
            fit_runtime = now() - t_fit
            eval_out = base.evaluate_model(clf, svd, X_test_original, X_test, y_test_encoded, X_train, y_train_encoded, X_core, y_core, C=args.C, n_classes_full=len(classes))
            pred = np.asarray(eval_out['pred_test_encoded'], dtype=int)
            row = {k: v for (k, v) in eval_out.items() if not isinstance(v, (np.ndarray, list))}
            row.update(class_balanced_loss_metrics(eval_out, y_test_encoded, y_train_str, classes, mu_df))
            cm_rows = class_metric_rows(y_test_encoded, pred, y_train_str, y_core_str, classes, mu_df, {'dataset_name': args.dataset_name, 'method': method, 'seed': args.seed, 'n_pcs': args.n_pcs, 'coreset_ratio_target': ratio, 'coreset_size_observed': len(idx_local), 'coreset_ratio_observed': len(idx_local) / n_train})
            row.update(base.biological_recall_metrics(y_test_encoded, pred, y_train_str, classes))
            add_subset_recall_metrics(row, cm_rows)
            obj = float(row.get('objective_full_train_objective_total', np.nan))
            row.update({'dataset_name': args.dataset_name, 'method': method, 'method_family': 'selected_sclewis', 'seed': args.seed, 'n_pcs': args.n_pcs, 'n_train': n_train, 'n_test': len(test_idx), 'n_classes': len(classes), 'coreset_ratio_target': ratio, 'coreset_size_target': target, 'coreset_size_observed': len(idx_local), 'coreset_ratio_observed': len(idx_local) / n_train, 'sampling_runtime_sec': sampling_runtime, 'fit_runtime_sec': fit_runtime, 'runtime_sec': sampling_runtime + fit_runtime, 'relative_objective_gap_vs_full_lr': (obj - full_obj) / max(abs(full_obj), 1e-300) if np.isfinite(obj) else np.nan, 'objective_gap_is_posthoc': False, 'objective_gap_note': 'native sklearn OvR LR objective'})
            try:
                row.update(base.coefficient_summary_metrics(clf, full_clf, svd, ks=(20, 50, 100)))
            except Exception as exc:
                LOGGER.warning('aggregate coefficient metrics failed: %s', exc)
            metrics_rows.append(row)
            class_rows_all.extend(cm_rows)
            if alloc_rows:
                sel_counts = pd.Series(y_core_str).value_counts().to_dict()
                for r in alloc_rows:
                    cls = str(r.get('class_label'))
                    r['selected_count'] = int(sel_counts.get(cls, 0))
                alloc_rows_all.extend(alloc_rows)
            else:
                sel_counts = pd.Series(y_core_str).value_counts().to_dict()
                train_counts = pd.Series(y_train_str).value_counts().to_dict()
                mu_lookup = dict(zip(mu_df['class_label'], mu_df['mu']))
                for cls in classes:
                    alloc_rows_all.append({'dataset_name': args.dataset_name, 'method': method, 'seed': args.seed, 'n_pcs': args.n_pcs, 'coreset_ratio_target': ratio, 'class_label': cls, 'class_count': int(train_counts.get(cls, 0)), 'frequency': float(train_counts.get(cls, 0) / n_train), 'mu': float(mu_lookup.get(cls, np.nan)), 'selected_count': int(sel_counts.get(cls, 0))})
            if args.max_selected_cells_per_method_ratio and len(global_idx) > args.max_selected_cells_per_method_ratio:
                keep = rng.choice(np.arange(len(global_idx)), size=args.max_selected_cells_per_method_ratio, replace=False)
                save_global = global_idx[keep]
            else:
                save_global = global_idx
            append_same_schema_csv(out / 'selected_cells.csv', selected_cell_rows(adata, save_global, method, ratio, y_all, {'dataset_name': args.dataset_name, 'seed': args.seed, 'n_pcs': args.n_pcs}), columns=selected_columns)
            (B_svd, B_gene) = align_betas(base, clf, len(classes), svd)
            (br, mr) = class_beta_rows(method, ratio, classes, B_svd, B_gene, full_svd, full_gene, genes, {'dataset_name': args.dataset_name, 'seed': args.seed, 'n_pcs': args.n_pcs})
            beta_rows_all.extend(br)
            marker_rows_all.extend(mr)
    if args.run_celltypist:
        if celltypist is None:
            raise RuntimeError('--run_celltypist requested but celltypist is not installed')
        labels = y_train_str.astype(str)
        for meth in parse_list(args.celltypist_methods):
            (method_name, kwargs) = celltypist_train_kwargs(meth, labels, svd_feature_names, args.n_jobs, args.celltypist_sgd_alpha_from_C, args.celltypist_sgd_C, args.celltypist_sgd_alpha, args.celltypist_sgd_batch_number, args.celltypist_sgd_batch_size, args.celltypist_sgd_epochs, args.celltypist_with_mean)
            LOGGER.info('Training CellTypist baseline: %s', method_name)
            t0 = now()
            model = celltypist.train(np.asarray(X_train, dtype=np.float32), **kwargs)
            fit_runtime = now() - t0
            t1 = now()
            (pred_str, P) = predict_celltypist_matrix(model, X_test, svd_feature_names, classes)
            pred_enc = le.transform(pred_str.astype(str))
            pred_runtime = now() - t1
            row = metrics_from_predictions(y_test_encoded, pred_enc, P, classes)
            eval_like = {'probs_test_softmax': P}
            row.update(class_balanced_loss_metrics(eval_like, y_test_encoded, y_train_str, classes, mu_df))
            cm_rows = class_metric_rows(y_test_encoded, pred_enc, y_train_str, y_train_str, classes, mu_df, {'dataset_name': args.dataset_name, 'method': method_name, 'seed': args.seed, 'n_pcs': args.n_pcs, 'coreset_ratio_target': np.nan, 'coreset_size_observed': len(train_idx), 'coreset_ratio_observed': 1.0})
            add_subset_recall_metrics(row, cm_rows)
            row.update({'dataset_name': args.dataset_name, 'method': method_name, 'method_family': 'celltypist_full_reference', 'seed': args.seed, 'n_pcs': args.n_pcs, 'n_train': len(train_idx), 'n_test': len(test_idx), 'n_classes': len(classes), 'coreset_ratio_target': np.nan, 'coreset_size_target': len(train_idx), 'coreset_size_observed': len(train_idx), 'coreset_ratio_observed': 1.0, 'fit_runtime_sec': fit_runtime, 'prediction_runtime_sec': pred_runtime, 'runtime_sec': fit_runtime + pred_runtime, 'relative_objective_gap_vs_full_lr': np.nan, 'relative_objective_gap_vs_full_lr_posthoc_total': np.nan, 'relative_objective_gap_vs_full_lr_posthoc_unregularized': np.nan, 'objective_gap_is_posthoc': True, 'objective_gap_note': 'CellTypist evaluated post hoc under the scLewis/FullLR OvR logistic objective; not CellTypist native training objective.', 'celltypist_train_kwargs_json': json.dumps(kwargs, sort_keys=True, default=str)})
            try:
                (B_ct_svd, b_ct_svd, present_ct, model_features) = extract_celltypist_coef_svd_raw(model, svd_feature_names, classes, X_train.shape[1])
                scores_train_ct = np.asarray(X_train, dtype=np.float64).dot(B_ct_svd.T) + b_ct_svd[None, :]
                ct_obj = base.ovr_logistic_objective_from_scores(scores_train_ct, y_train_encoded, C=args.C, coef=B_ct_svd)
                for (k, v) in ct_obj.items():
                    row[f'objective_full_train_{k}'] = v
                ct_total = float(ct_obj.get('objective_total', np.nan))
                ct_unreg = float(ct_obj.get('objective_unregularized', np.nan))
                full_unreg = float(full_eval.get('objective_full_train_objective_unregularized', np.nan))
                if np.isfinite(ct_total) and np.isfinite(full_obj):
                    row['relative_objective_gap_vs_full_lr_posthoc_total'] = (ct_total - full_obj) / max(abs(full_obj), 1e-300)
                    row['relative_objective_gap_vs_full_lr'] = row['relative_objective_gap_vs_full_lr_posthoc_total']
                if np.isfinite(ct_unreg) and np.isfinite(full_unreg):
                    row['relative_objective_gap_vs_full_lr_posthoc_unregularized'] = (ct_unreg - full_unreg) / max(abs(full_unreg), 1e-300)
                row['celltypist_coef_features_count'] = int(len(model_features))
                row['celltypist_classes_present_in_objective_count'] = int(np.asarray(present_ct, dtype=bool).sum())
                row['celltypist_classes_missing_in_objective_count'] = int(len(classes) - np.asarray(present_ct, dtype=bool).sum())
                B_ct_gene = B_ct_svd.dot(np.asarray(svd.components_, dtype=np.float64))
                (br, mr) = class_beta_rows(method_name, np.nan, classes, B_ct_svd, B_ct_gene, full_svd, full_gene, genes, {'dataset_name': args.dataset_name, 'seed': args.seed, 'n_pcs': args.n_pcs})
                beta_rows_all.extend(br)
                marker_rows_all.extend(mr)
            except Exception as exc:
                LOGGER.warning('CellTypist coefficient extraction/post-hoc objective failed for %s: %s', method_name, exc)
            metrics_rows.append(row)
            class_rows_all.extend(cm_rows)
            provenance.append({'method': method_name, 'backend': 'official celltypist.train on SVD matrix features + post-hoc FullLR OvR objective evaluation', 'parameters': json.dumps(kwargs, sort_keys=True, default=str)})
    LOGGER.info('Writing final CSVs')
    write_df(out / 'metrics.csv', pd.DataFrame(metrics_rows))
    write_df(out / 'class_metrics.csv', pd.DataFrame(class_rows_all))
    write_df(out / 'beta_comparison.csv', pd.DataFrame(beta_rows_all))
    write_df(out / 'marker_gene_comparison.csv', pd.DataFrame(marker_rows_all))
    write_df(out / 'mu_allocation_by_class.csv', pd.DataFrame(alloc_rows_all))
    prov = pd.DataFrame(provenance).drop_duplicates(subset=['method', 'backend', 'parameters'], keep='first')
    write_df(out / 'method_provenance.csv', prov)
    LOGGER.info('Done. Output=%s', out)
if __name__ == '__main__':
    main()
