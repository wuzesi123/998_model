from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ForecastExperiment:
    experiment_id: str
    source_group: str
    family: str
    x: np.ndarray
    time: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def save_forecast_experiment(exp: ForecastExperiment, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment_id=np.array(exp.experiment_id),
        source_group=np.array(exp.source_group),
        family=np.array(exp.family),
        x=np.asarray(exp.x, np.float32),
        time=np.asarray(exp.time, np.float32),
        feature_names=np.asarray(exp.feature_names, dtype=object),
        target_names=np.asarray(exp.target_names, dtype=object),
    )
    return path


def load_forecast_experiment(path: str | Path) -> ForecastExperiment:
    z = np.load(path, allow_pickle=True)
    return ForecastExperiment(
        experiment_id=str(z["experiment_id"].item()),
        source_group=str(z["source_group"].item()),
        family=str(z["family"].item()),
        x=z["x"].astype(np.float32),
        time=z["time"].astype(np.float32),
        feature_names=[str(v) for v in z["feature_names"].tolist()],
        target_names=[str(v) for v in z["target_names"].tolist()],
    )


def list_forecast_experiments(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.forecast.npz"))


def split_by_source_group(paths: list[Path], seed: int = 7, ratios=(0.70, 0.15, 0.15)):
    groups: dict[str, list[Path]] = {}
    for p in paths:
        e = load_forecast_experiment(p)
        groups.setdefault(e.source_group, []).append(Path(p))
    keys = list(groups)
    if len(keys) < 3:
        raise ValueError(f"Need >=3 source groups for leakage-safe train/val/test split; found {len(keys)}")
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    trk = set(keys[:n_train])
    vak = set(keys[n_train:n_train+n_val])
    tek = set(keys[n_train+n_val:])

    def collect(sel):
        out = []
        for k in keys:
            if k in sel:
                out.extend(groups[k])
        return out

    info = {
        "mode": "source_group",
        "n_source_groups": len(keys),
        "train_groups": sorted(trk),
        "val_groups": sorted(vak),
        "test_groups": sorted(tek),
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }
    return collect(trk), collect(vak), collect(tek), info


def validate_schema(paths: Iterable[str | Path]) -> tuple[list[str], list[str]]:
    paths = list(paths)
    if not paths:
        raise ValueError("No forecast experiments")
    first = load_forecast_experiment(paths[0])
    fn = first.feature_names
    tn = first.target_names
    errors = []
    for p in paths[1:]:
        e = load_forecast_experiment(p)
        if e.feature_names != fn or e.target_names != tn or e.x.shape[1] != len(fn):
            errors.append({"experiment": e.experiment_id, "feature_names": e.feature_names, "target_names": e.target_names})
    if errors:
        raise ValueError(f"Forecast feature schema mismatch. First={fn}/{tn}; mismatches={errors[:5]}")
    return list(fn), list(tn)


def _positive_median_dt(time: np.ndarray) -> float:
    d = np.diff(np.asarray(time, np.float64))
    d = d[np.isfinite(d) & (d > 1e-9)]
    return float(np.median(d)) if len(d) else 1.0


def estimate_training_horizons(paths: list[Path]) -> dict:
    spans, dts = [], []
    for p in paths:
        e = load_forecast_experiment(p)
        if len(e.time) < 2:
            continue
        span = float(e.time[-1] - e.time[0])
        if np.isfinite(span) and span > 0:
            spans.append(span)
        dts.append(_positive_median_dt(e.time))
    if not spans:
        raise ValueError("Cannot estimate forecast horizons: no positive experiment spans")
    med_span = float(np.median(spans))
    med_dt = float(np.median(dts)) if dts else 1.0
    fractions = [0.005, 0.02, 0.05]
    names = ["short", "medium", "long"]
    secs = []
    for frac in fractions:
        sec = max(med_dt, med_span * frac)
        sec = min(sec, med_span * 0.25)
        secs.append(float(sec))
    # Keep distinct horizons even for coarse sampling.
    for i in range(1, len(secs)):
        secs[i] = max(secs[i], secs[i-1] + med_dt)
    return {
        "training_median_span_sec": med_span,
        "training_median_dt_sec": med_dt,
        "fractions": fractions,
        "horizons": {n: s for n, s in zip(names, secs)},
    }


def compute_input_normalization(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for p in paths:
        e = load_forecast_experiment(p)
        chunks.append(e.x.astype(np.float64))
    x = np.concatenate(chunks, axis=0)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_target_scale(paths: list[Path], target_indices: list[int]) -> dict:
    chunks = []
    for p in paths:
        e = load_forecast_experiment(p)
        chunks.append(e.x[:, target_indices].astype(np.float64))
    y = np.concatenate(chunks, axis=0)
    mean = np.nanmean(y, axis=0)
    std = np.nanstd(y, axis=0)
    p05 = np.nanpercentile(y, 5, axis=0)
    p95 = np.nanpercentile(y, 95, axis=0)
    robust_range = p95 - p05
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    robust_range = np.where(np.isfinite(robust_range) & (robust_range > 1e-6), robust_range, std)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "robust_range": robust_range.astype(np.float32),
        "p05": p05.astype(np.float32),
        "p95": p95.astype(np.float32),
    }


def _sample_recent_indices(end_idx: int, length: int) -> np.ndarray:
    length = max(int(length), 1)
    start = max(0, int(end_idx) - length + 1)
    ids = np.arange(start, int(end_idx) + 1, dtype=np.int64)
    if len(ids) < length:
        pad = np.full(length - len(ids), ids[0] if len(ids) else 0, np.int64)
        ids = np.concatenate([pad, ids])
    return ids


def _sample_log_history_indices(time: np.ndarray, end_idx: int, length: int) -> np.ndarray:
    length = max(int(length), 1)
    prefix = np.asarray(time[: end_idx + 1], np.float64)
    if len(prefix) <= 1:
        return np.zeros(length, np.int64)
    t0, t1 = float(prefix[0]), float(prefix[-1])
    elapsed = max(t1 - t0, 0.0)
    med_dt = _positive_median_dt(prefix)
    if elapsed <= med_dt:
        return _sample_recent_indices(end_idx, length)
    if length == 1:
        return np.asarray([end_idx], np.int64)
    lookbacks = np.geomspace(max(med_dt, 1e-9), max(elapsed, med_dt), num=length-1)
    targets = np.concatenate([t1 - lookbacks[::-1], np.asarray([t1])])
    ids = np.searchsorted(prefix, targets, side="right") - 1
    ids = np.clip(ids, 0, end_idx).astype(np.int64)
    ids[-1] = end_idx
    return ids


def _sample_global_indices(time: np.ndarray, end_idx: int, length: int) -> np.ndarray:
    length = max(int(length), 1)
    prefix = np.asarray(time[: end_idx + 1], np.float64)
    if len(prefix) <= 1:
        return np.zeros(length, np.int64)
    targets = np.linspace(float(prefix[0]), float(prefix[-1]), length)
    ids = np.searchsorted(prefix, targets, side="right") - 1
    ids = np.clip(ids, 0, end_idx).astype(np.int64)
    ids[-1] = end_idx
    return ids


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        horizon_sec: float,
        short_len: int = 8,
        medium_len: int = 12,
        global_len: int = 16,
        stride: int = 1,
        input_mean: np.ndarray | None = None,
        input_std: np.ndarray | None = None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ):
        self.experiments = [load_forecast_experiment(p) for p in paths]
        self.feature_names, self.target_names = validate_schema(paths)
        self.target_indices = [self.feature_names.index(n) for n in self.target_names]
        self.horizon_sec = float(horizon_sec)
        self.short_len = int(short_len)
        self.medium_len = int(medium_len)
        self.global_len = int(global_len)
        self.stride = max(int(stride), 1)
        self.input_mean = np.zeros(len(self.feature_names), np.float32) if input_mean is None else np.asarray(input_mean, np.float32)
        self.input_std = np.ones(len(self.feature_names), np.float32) if input_std is None else np.asarray(input_std, np.float32)
        self.target_mean = np.zeros(len(self.target_indices), np.float32) if target_mean is None else np.asarray(target_mean, np.float32)
        self.target_std = np.ones(len(self.target_indices), np.float32) if target_std is None else np.asarray(target_std, np.float32)
        self.index: list[tuple[int, int, int]] = []
        for ei, e in enumerate(self.experiments):
            t = np.asarray(e.time, np.float64)
            for i in range(0, len(t)-1, self.stride):
                j = int(np.searchsorted(t, float(t[i]) + self.horizon_sec, side="left"))
                if j < len(t) and j > i:
                    self.index.append((ei, i, j))
        if not self.index:
            raise ValueError(f"No forecast samples at horizon={self.horizon_sec:.3f}s")

    def __len__(self):
        return len(self.index)

    def _norm_x(self, x):
        return (x - self.input_mean[None, :]) / self.input_std[None, :]

    def __getitem__(self, idx):
        ei, i, j = self.index[idx]
        e = self.experiments[ei]
        sids = _sample_recent_indices(i, self.short_len)
        mids = _sample_log_history_indices(e.time, i, self.medium_len)
        gids = _sample_global_indices(e.time, i, self.global_len)
        sx = self._norm_x(e.x[sids])
        mx = self._norm_x(e.x[mids])
        gx = self._norm_x(e.x[gids])
        # Time encoding gets elapsed time only; no future timestamp is exposed to the model.
        elapsed = e.time - e.time[0]
        y_raw = e.x[j, self.target_indices].astype(np.float32)
        y = (y_raw - self.target_mean) / self.target_std
        current_raw = e.x[i, self.target_indices].astype(np.float32)
        return {
            "short_x": torch.from_numpy(sx.astype(np.float32)),
            "medium_x": torch.from_numpy(mx.astype(np.float32)),
            "global_x": torch.from_numpy(gx.astype(np.float32)),
            "short_time": torch.from_numpy(elapsed[sids].astype(np.float32)),
            "medium_time": torch.from_numpy(elapsed[mids].astype(np.float32)),
            "global_time": torch.from_numpy(elapsed[gids].astype(np.float32)),
            "target": torch.from_numpy(y.astype(np.float32)),
            "target_raw": torch.from_numpy(y_raw),
            "current_raw": torch.from_numpy(current_raw),
            "experiment_index": torch.tensor(ei, dtype=torch.long),
            "current_index": torch.tensor(i, dtype=torch.long),
            "future_index": torch.tensor(j, dtype=torch.long),
        }


def dataset_audit(paths: list[Path]) -> dict:
    exps = [load_forecast_experiment(p) for p in paths]
    families: dict[str, int] = {}
    spans = []
    for e in exps:
        families[e.family] = families.get(e.family, 0) + 1
        spans.append(float(e.time[-1] - e.time[0]) if len(e.time) > 1 else 0.0)
    return {
        "n_experiments": len(exps),
        "n_source_groups": len({e.source_group for e in exps}),
        "families": families,
        "span_sec": {
            "median": float(np.median(spans)) if spans else None,
            "min": float(np.min(spans)) if spans else None,
            "max": float(np.max(spans)) if spans else None,
        },
        "feature_names": exps[0].feature_names if exps else [],
        "target_names": exps[0].target_names if exps else [],
    }
