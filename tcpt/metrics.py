from __future__ import annotations

import numpy as np


def macro_f1(y_true, y_pred, n_classes: int):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    return float(np.mean(f1s))


def binary_prf1(y_true, prob, threshold=0.5):
    y_pred = np.asarray(prob) >= threshold
    y_true = np.asarray(y_true) >= 0.5
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)


def binary_f1(y_true, prob, threshold=0.5):
    return binary_prf1(y_true, prob, threshold)[2]


def best_binary_threshold(y_true, prob, lo=0.10, hi=0.90, steps=81):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(lo, hi, steps):
        f1 = binary_f1(y_true, prob, float(t))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1



def _safe_prf1_counts(tp: int, fp: int, fn: int):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return float(p), float(r), float(f1)


def match_events_with_tolerance(true_times, pred_times, tolerance_sec: float):
    """One-to-one greedy event matching within a time tolerance.

    Events are matched by the smallest absolute timing error first so one prediction
    cannot satisfy multiple ground-truth events. Returns precision/recall/F1 plus the
    absolute errors of matched pairs.
    """
    true_times = np.sort(np.asarray(true_times, dtype=np.float64))
    pred_times = np.sort(np.asarray(pred_times, dtype=np.float64))
    if len(true_times) == 0 and len(pred_times) == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "matched_abs_errors_sec": np.asarray([], dtype=np.float64),
        }

    candidates = []
    for ti, t in enumerate(true_times):
        for pi, p in enumerate(pred_times):
            err = abs(float(p - t))
            if err <= tolerance_sec:
                candidates.append((err, ti, pi))
    candidates.sort(key=lambda z: z[0])

    used_t, used_p, errors = set(), set(), []
    for err, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        errors.append(err)

    tp = len(errors)
    fp = len(pred_times) - tp
    fn = len(true_times) - tp
    precision, recall, f1 = _safe_prf1_counts(tp, fp, fn)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "matched_abs_errors_sec": np.asarray(errors, dtype=np.float64),
    }


def summarize_abs_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {"mae": float("nan"), "median": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mae": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "n": int(len(errors)),
    }
