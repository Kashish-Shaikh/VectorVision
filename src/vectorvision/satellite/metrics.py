"""Evaluation metrics. Pure numpy so they can be tested without a GPU."""
from __future__ import annotations

import numpy as np


def per_tile_stats(pred: np.ndarray, truth: np.ndarray, n_cls: int):
    """Per-tile intersection, prediction count and truth count for every class.

    Keeping these per tile (rather than one global total) is what makes the
    bootstrap confidence intervals possible.
    """
    N = len(pred)
    inter = np.zeros((N, n_cls), np.int64)
    pcnt = np.zeros((N, n_cls), np.int64)
    tcnt = np.zeros((N, n_cls), np.int64)
    for c in range(n_cls):
        p, t = pred == c, truth == c
        inter[:, c] = (p & t).reshape(N, -1).sum(1)
        pcnt[:, c] = p.reshape(N, -1).sum(1)
        tcnt[:, c] = t.reshape(N, -1).sum(1)
    return inter, pcnt, tcnt


def scores(inter, pcnt, tcnt):
    I, P, T = inter.sum(0).astype(float), pcnt.sum(0).astype(float), tcnt.sum(0).astype(float)
    iou = I / np.maximum(P + T - I, 1)
    prec = I / np.maximum(P, 1)
    rec = I / np.maximum(T, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return iou, prec, rec, f1


def confusion(pred, truth, n_cls):
    idx = truth.reshape(-1).astype(np.int64) * n_cls + pred.reshape(-1)
    return np.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def bootstrap_ci(inter, pcnt, tcnt, water: int, n_boot: int = 1000, seed: int = 42):
    """95% intervals for mean IoU and water IoU by resampling whole tiles."""
    rng = np.random.default_rng(seed)
    N = len(inter)
    m, w = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, N, N)
        iou, *_ = scores(inter[ix], pcnt[ix], tcnt[ix])
        m[b], w[b] = iou.mean(), iou[water]
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"mean_iou": q(m), "water_iou": q(w)}


def threshold_sweep(water_prob: np.ndarray, water_truth: np.ndarray, thresholds=None):
    """Precision / recall / IoU for water at different decision thresholds."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.1, 0.95, 0.05), 2)
    t = water_truth.astype(bool)
    out = []
    for th in thresholds:
        p = water_prob >= th
        tp = float((p & t).sum()); fp = float((p & ~t).sum()); fn = float((~p & t).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out.append({"threshold": float(th), "precision": prec, "recall": rec,
                    "iou": tp / max(tp + fp + fn, 1), "f1": 2 * prec * rec / max(prec + rec, 1e-12)})
    return out


def tile_detection(pred_water: np.ndarray, truth_water: np.ndarray, min_px: int):
    """Does this tile contain a water body? The satellite analogue of the drone's
    frame-level question. A tile is positive if it has >= min_px water pixels."""
    N = len(pred_water)
    p = pred_water.reshape(N, -1).sum(1) >= min_px
    t = truth_water.reshape(N, -1).sum(1) >= min_px
    tp, tn = int((p & t).sum()), int((~p & ~t).sum())
    fp, fn = int((p & ~t).sum()), int((~p & t).sum())
    acc = (tp + tn) / max(N, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"tiles": N, "positive_tiles": int(t.sum()), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "accuracy_ci95": wilson(tp + tn, N)}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]
