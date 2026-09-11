from __future__ import annotations

from pathlib import Path
import re
import zipfile
import tarfile

import numpy as np
import pandas as pd


TABLE_EXTS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table: {path}")


def normalized_name(v) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(v).lower())


def find_time_col(df: pd.DataFrame):
    exact = {
        "time", "times", "sec", "secs", "second", "seconds", "timestamp",
        "elapsed", "elapsedtime", "elapsedseconds", "timesec", "timeseconds",
        "runtime", "duration", "minutes", "minute", "min",
    }
    for c in df.columns:
        n = normalized_name(c)
        if n in exact:
            return c
    # Prefer names containing time but avoid obvious duration summaries.
    for c in df.columns:
        n = normalized_name(c)
        if "time" in n and "total" not in n:
            return c
    return None


def time_to_seconds(s: pd.Series, name: str = "") -> np.ndarray:
    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
        n = normalized_name(name)
        if "min" in n and "sec" not in n:
            x = x * 60.0
        elif "hour" in n or n.endswith("hr"):
            x = x * 3600.0
        return x.astype(np.float32)
    # timedelta-like strings
    td = pd.to_timedelta(s, errors="coerce")
    if td.notna().sum() >= max(3, len(s) // 2):
        return td.dt.total_seconds().to_numpy(np.float32)
    # fallback numeric conversion
    return pd.to_numeric(s, errors="coerce").to_numpy(np.float32)


def smooth(y: np.ndarray, win: int = 9) -> np.ndarray:
    y = np.asarray(y, np.float32)
    if len(y) < 3:
        return y.copy()
    win = min(max(3, int(win)), len(y) if len(y) % 2 else len(y) - 1)
    if win < 3:
        return y.copy()
    k = np.ones(win, np.float32) / win
    yp = np.pad(y, (win // 2, win // 2), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)].astype(np.float32)


def robust_progress(signal: np.ndarray, monotonic: bool = True) -> np.ndarray:
    s = smooth(np.asarray(signal, np.float32), 9)
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return np.zeros_like(s)
    lo, hi = np.percentile(finite, [2, 98])
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(s)
    p = np.clip((s - lo) / (hi - lo), 0, 1)
    if monotonic:
        p = np.maximum.accumulate(p)
    return p.astype(np.float32)


def weak_stage_targets(progress: np.ndarray, time_s: np.ndarray | None = None):
    """Create kinetics-aware WEAK process-stage targets.

    V5.2 deliberately avoids fixed progress cuts such as 0.20/0.72/0.90 as the
    primary definition of process stages.  Instead it uses the *rate of change* of
    the progress trajectory with sustained-run rules:

      INITIAL      : before sustained observable change;
      ONSET        : sustained change has begun but is not yet strongly active;
      ACTIVE       : high process-rate region;
      STABILISING  : rate has fallen after the active region;
      STABLE       : sustained low-rate plateau at sufficiently advanced progress.

    The labels are still adapter-derived weak labels, not author/manual chemical GT.
    Full-trajectory information is used only to construct offline targets, never as
    an inference feature.
    """
    p = np.clip(np.asarray(progress, np.float32).reshape(-1), 0, 1)
    n = len(p)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), p, 0, 0
    if n < 5:
        stage = np.minimum((p * 5).astype(np.int64), 4)
        return stage, np.zeros(n, np.float32), p, 0, n - 1

    ps = np.maximum.accumulate(smooth(p, max(5, min(15, (n // 30) * 2 + 5))))

    if time_s is not None and len(time_s) == n:
        t = np.asarray(time_s, np.float64).reshape(-1)
        # Defensive repair for non-finite/non-increasing source timestamps.
        if not np.all(np.isfinite(t)):
            t = np.arange(n, dtype=np.float64)
        t = np.maximum.accumulate(t)
        d = np.diff(t)
        pos = d[d > 1e-9]
        fallback_dt = float(np.median(pos)) if len(pos) else 1.0
        for i in range(1, n):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + fallback_dt
    else:
        t = np.arange(n, dtype=np.float64)
        fallback_dt = 1.0

    rate = np.gradient(ps.astype(np.float64), t, edge_order=1)
    rate = smooth(np.maximum(rate, 0.0).astype(np.float32), max(5, min(13, (n // 40) * 2 + 5))).astype(np.float64)
    finite_rate = rate[np.isfinite(rate)]
    if len(finite_rate) == 0:
        finite_rate = np.asarray([0.0])

    # Robust scale estimates: baseline noise from low-rate region and signal from
    # upper-rate quantiles.  This makes thresholds comparable across seconds/minutes.
    low = finite_rate[finite_rate <= np.percentile(finite_rate, 35)]
    if len(low) == 0:
        low = finite_rate
    low_med = float(np.median(low))
    mad = float(1.4826 * np.median(np.abs(low - low_med))) if len(low) else 0.0
    peak = float(np.percentile(finite_rate, 95))
    dynamic = max(peak - low_med, 1e-8)

    onset_thr = low_med + max(3.0 * mad, 0.08 * dynamic)
    active_thr = low_med + max(4.0 * mad, 0.28 * dynamic)
    stable_thr = low_med + max(2.0 * mad, 0.05 * dynamic)

    sustain = max(2, min(12, int(round(0.02 * n))))
    stable_sustain = max(3, min(24, int(round(0.05 * n))))

    def first_sustained(mask: np.ndarray, start: int, run: int):
        count = 0
        begin = None
        for i in range(max(0, int(start)), n):
            if bool(mask[i]):
                if count == 0:
                    begin = i
                count += 1
                if count >= run:
                    return int(begin)
            else:
                count = 0
                begin = None
        return None

    onset_mask = (rate >= onset_thr) & (ps >= 0.01)
    onset_idx = first_sustained(onset_mask, 0, sustain)
    if onset_idx is None:
        # Last-resort weak onset based on robust progress departure.
        cand = np.where(ps >= 0.03)[0]
        onset_idx = int(cand[0]) if len(cand) else 0

    active_mask = rate >= active_thr
    active_idx = first_sustained(active_mask, onset_idx, sustain)
    if active_idx is None:
        # Keep an explicit ONSET region when the trajectory is gradual.
        active_idx = min(n - 1, onset_idx + max(sustain, int(0.04 * n)))
    # Weak labels must have enough temporal support to be learnable.  Prevent a
    # one- or two-sample ONSET class caused by a sharp derivative threshold crossing.
    min_onset_width = max(sustain, int(round(0.05 * n)))
    active_idx = max(active_idx, onset_idx + min_onset_width) if onset_idx < n - 1 else onset_idx
    active_idx = min(active_idx, n - 1)

    peak_idx = int(np.argmax(rate[max(active_idx, 0):]) + max(active_idx, 0)) if active_idx < n else n - 1
    stabilising_mask = (rate < active_thr) & (ps >= max(0.45, float(ps[active_idx]) if active_idx < n else 0.45))
    stabilising_idx = first_sustained(stabilising_mask, peak_idx, sustain)
    if stabilising_idx is None:
        cand = np.where((np.arange(n) > active_idx) & (ps >= 0.70))[0]
        stabilising_idx = int(cand[0]) if len(cand) else max(active_idx + 1, int(0.75 * n))
    stabilising_idx = min(max(stabilising_idx, active_idx + 1), n - 1)

    stable_mask = (rate <= stable_thr) & (ps >= 0.75)
    endpoint_idx = first_sustained(stable_mask, stabilising_idx, stable_sustain)
    if endpoint_idx is None:
        cand = np.where((np.arange(n) > stabilising_idx) & (ps >= 0.93))[0]
        endpoint_idx = int(cand[0]) if len(cand) else n - 1
    endpoint_idx = min(max(endpoint_idx, stabilising_idx + 1), n - 1)
    # Likewise keep a meaningful STABILISING interval before STABLE.
    min_stabilising_width = max(stable_sustain, int(round(0.08 * n)))
    if endpoint_idx - stabilising_idx < min_stabilising_width:
        stabilising_idx = max(active_idx + 1, endpoint_idx - min_stabilising_width)

    stage = np.zeros(n, np.int64)
    stage[onset_idx:active_idx] = 1
    stage[active_idx:stabilising_idx] = 2
    stage[stabilising_idx:endpoint_idx] = 3
    stage[endpoint_idx:] = 4

    # Transition targets are localized around the four stage boundaries.  Gaussian
    # width is defined in real seconds when timestamps are available, which is much
    # less sensitive to irregular sampling than a fixed number of frames.
    transition = np.zeros(n, np.float32)
    span_sec = max(float(t[-1] - t[0]), fallback_dt)
    sigma_sec = max(2.0 * fallback_dt, 0.008 * span_sec)
    for c in sorted(set([onset_idx, active_idx, stabilising_idx, endpoint_idx])):
        tc = float(t[c])
        g = np.exp(-0.5 * ((t - tc) / max(sigma_sec, 1e-6)) ** 2)
        transition = np.maximum(transition, g.astype(np.float32))

    return stage, np.clip(transition, 0, 1).astype(np.float32), ps.astype(np.float32), int(onset_idx), int(endpoint_idx)


def change_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    scale = np.nanstd(x, axis=0) + 1e-6
    z = (x - x[0:1]) / scale
    return np.linalg.norm(z, axis=1).astype(np.float32)


def recursive_extract(root: Path) -> int:
    count = 0
    changed = True
    while changed:
        changed = False
        for p in list(root.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            marker = p.with_suffix(p.suffix + ".extracted")
            if marker.exists():
                continue
            try:
                if ext == ".zip":
                    target = p.parent / p.stem
                    target.mkdir(exist_ok=True)
                    with zipfile.ZipFile(p) as zf:
                        zf.extractall(target)
                    marker.write_text("ok")
                    count += 1
                    changed = True
                elif ext in {".tar", ".gz", ".tgz", ".bz2", ".xz"} and tarfile.is_tarfile(p):
                    target = p.parent / p.name.split(".")[0]
                    target.mkdir(exist_ok=True)
                    with tarfile.open(p) as tf:
                        tf.extractall(target)
                    marker.write_text("ok")
                    count += 1
                    changed = True
            except Exception:
                # Leave unsupported/compressed scientific files untouched.
                pass
    return count
