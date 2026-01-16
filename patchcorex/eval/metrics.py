from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _binary_clf_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(scores)[::-1]
    scores = scores[order]
    labels = labels[order]
    pos = labels == 1
    neg = labels == 0
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    return fp, tp


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fp, tp = _binary_clf_curve(scores, labels)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    return fpr, tpr


def auc_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    return float(np.trapz(precision, recall))


def image_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    fpr, tpr = roc_curve(scores, labels)
    return {
        "auroc": auc_from_curve(fpr, tpr),
        "ap": average_precision(scores, labels),
    }


def industrial_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> Dict[str, float]:
    if threshold is None:
        threshold = float(np.median(scores))
    preds = scores >= threshold
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    fp_per_1k = fpr * 1000.0
    return {
        "fpr": float(fpr),
        "fnr": float(fnr),
        "fp_per_1k": float(fp_per_1k),
    }


def pixel_metrics(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> Dict[str, float]:
    scores = maps.reshape(-1).astype(np.float32)
    labels = (masks.reshape(-1) > 0.5).astype(np.int64)
    fpr, tpr = roc_curve(scores, labels)
    auroc = auc_from_curve(fpr, tpr)
    ap = average_precision(scores, labels)

    limit = float(fpr_limit)
    if limit <= 0:
        aupro = 0.0
    else:
        idx = fpr <= limit
        if idx.sum() < 2:
            aupro = 0.0
        else:
            fpr_l = fpr[idx]
            tpr_l = tpr[idx]
            if fpr_l[-1] < limit:
                tpr_limit = float(np.interp(limit, fpr, tpr))
                fpr_l = np.append(fpr_l, limit)
                tpr_l = np.append(tpr_l, tpr_limit)
            aupro = float(np.trapz(tpr_l, fpr_l) / limit)

    return {
        "pixel_auroc": float(auroc),
        "pixel_ap": float(ap),
        "pixel_aupro": float(aupro),
    }
