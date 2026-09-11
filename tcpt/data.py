from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


STAGES = ["INITIAL", "ONSET", "ACTIVE", "STABILISING", "STABLE"]


@dataclass
class Experiment:
    experiment_id: str
    x: np.ndarray
    stage: np.ndarray
    transition: np.ndarray
    progress: np.ndarray
    time: np.ndarray
    feature_names: list[str]


def save_experiment(exp: Experiment, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment_id=np.array(exp.experiment_id),
        x=exp.x.astype(np.float32),
        stage=exp.stage.astype(np.int64),
        transition=np.clip(exp.transition, 0, 1).astype(np.float32),
        progress=exp.progress.astype(np.float32),
        time=exp.time.astype(np.float32),
        feature_names=np.array(exp.feature_names, dtype=object),
    )


def load_experiment(path: str | Path) -> Experiment:
    z = np.load(path, allow_pickle=True)
    return Experiment(
        experiment_id=str(z["experiment_id"].item()),
        x=z["x"].astype(np.float32),
        stage=z["stage"].astype(np.int64),
        transition=np.clip(z["transition"].astype(np.float32), 0, 1),
        progress=z["progress"].astype(np.float32),
        time=z["time"].astype(np.float32),
        feature_names=[str(v) for v in z["feature_names"].tolist()],
    )


def list_experiments(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.npz"))


def split_by_experiment(paths: list[Path], seed: int = 7, ratios=(0.7, 0.15, 0.15)):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = len(paths)
    n_train = max(1, int(n * ratios[0])) if n >= 3 else max(1, n - 2)
    n_val = max(1, int(n * ratios[1])) if n >= 3 else int(n >= 2)
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1 if n >= 2 else 0
    return paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:]


def _meta_for_npz(path: str | Path) -> dict:
    p = Path(path).with_suffix(".meta.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def source_group_for_path(path: str | Path) -> str:
    """Return a provenance group used to keep related representations in one split.

    Kineticolor V5.2.2 writes meta.source_group. Other datasets safely fall back to
    experiment_id / filename, so this remains backward compatible.
    """
    p = Path(path)
    meta = _meta_for_npz(p)
    inner = meta.get("meta", {}) if isinstance(meta.get("meta", {}), dict) else {}
    group = inner.get("source_group") or meta.get("source_group")
    if group:
        return str(group)
    exp_id = meta.get("experiment_id")
    if exp_id:
        return str(exp_id)
    return p.stem


def split_by_source_group(paths: list[Path], seed: int = 7, ratios=(0.7, 0.15, 0.15), return_info: bool = False):
    """Group-aware train/val/test split.

    All prepared experiments sharing the same `source_group` sidecar metadata are kept
    together. This prevents RGB/Lab/alternate trajectory representations of one source
    reaction from leaking across train/validation/test.
    """
    paths = list(paths)
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(source_group_for_path(p), []).append(Path(p))

    keys = list(groups)
    if len(keys) < 3:
        tr, va, te = split_by_experiment(paths, seed=seed, ratios=ratios)
        info = {
            "mode": "experiment_fallback",
            "n_source_groups": len(keys),
            "groups": {k: [x.name for x in v] for k, v in groups.items()},
        }
        return (tr, va, te, info) if return_info else (tr, va, te)

    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1]))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    def collect(selected):
        out = []
        for k in keys:
            if k in selected:
                out.extend(groups[k])
        return out

    tr, va, te = collect(train_keys), collect(val_keys), collect(test_keys)
    info = {
        "mode": "source_group",
        "n_source_groups": len(keys),
        "train_groups": sorted(train_keys),
        "val_groups": sorted(val_keys),
        "test_groups": sorted(test_keys),
        "group_sizes": {k: len(v) for k, v in groups.items()},
    }
    return (tr, va, te, info) if return_info else (tr, va, te)


def _transition_centres(y: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    y = np.asarray(y, np.float32)
    above = y >= threshold
    centres: list[int] = []
    i = 0
    while i < len(y):
        if not above[i]:
            i += 1
            continue
        j = i + 1
        while j < len(y) and above[j]:
            j += 1
        centres.append(i + int(np.argmax(y[i:j])))
        i = j
    if not centres and len(y) and float(np.max(y)) >= 0.20:
        centres = [int(np.argmax(y))]
    return np.asarray(centres, dtype=np.int64)


def _distance_targets(exp: Experiment, horizon_samples: int, horizon_sec: float | None = None):
    centres = _transition_centres(exp.transition)
    n = len(exp.transition)
    dist_norm = np.ones(n, dtype=np.float32)
    dist_sec = np.full(n, np.nan, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)

    if len(centres) == 0:
        return dist_norm, dist_sec, mask, centres, float(horizon_sec or 1.0)

    time = np.asarray(exp.time, np.float64)
    pos_dt = np.diff(time)
    pos_dt = pos_dt[np.isfinite(pos_dt) & (pos_dt > 0)]
    median_dt = float(np.median(pos_dt)) if len(pos_dt) else 1.0
    h_sec = float(horizon_sec) if horizon_sec is not None and horizon_sec > 0 else max(float(horizon_samples) * median_dt, median_dt)

    c_times = time[centres]
    for i, ti in enumerate(time):
        k = int(np.argmin(np.abs(c_times - ti)))
        # Signed definition: current time - true boundary time.
        ds = float(ti - c_times[k])
        if abs(ds) <= h_sec:
            mask[i] = 1.0
            dist_sec[i] = ds
            dist_norm[i] = float(np.clip(ds / h_sec, -1.0, 1.0))
        else:
            dist_norm[i] = float(np.sign(ds)) if ds != 0 else 0.0
    return dist_norm, dist_sec, mask, centres, h_sec


def _sample_recent_indices(end_idx: int, length: int) -> np.ndarray:
    """Return the most recent `length` causal sample indices, padding at the left."""
    length = max(int(length), 1)
    end_idx = max(int(end_idx), 0)
    start = max(0, end_idx - length + 1)
    ids = np.arange(start, end_idx + 1, dtype=np.int64)
    if len(ids) < length:
        pad = np.full(length - len(ids), ids[0] if len(ids) else 0, dtype=np.int64)
        ids = np.concatenate([pad, ids])
    return ids


def _sample_log_history_indices(time: np.ndarray, end_idx: int, length: int) -> np.ndarray:
    """Scale-adaptive causal sampling over all history using logarithmic lookback.

    The window automatically expands with elapsed experiment time.  It therefore
    works for both second-scale and hour-scale reactions without knowing the future
    reaction duration.
    """
    length = max(int(length), 1)
    prefix = np.asarray(time[: end_idx + 1], np.float64)
    if len(prefix) <= 1 or length == 1:
        return np.full(length, max(end_idx, 0), dtype=np.int64)

    t0 = float(prefix[0])
    t1 = float(prefix[-1])
    elapsed = max(t1 - t0, 0.0)
    pos_dt = np.diff(prefix)
    pos_dt = pos_dt[np.isfinite(pos_dt) & (pos_dt > 0)]
    median_dt = float(np.median(pos_dt)) if len(pos_dt) else max(elapsed / max(len(prefix) - 1, 1), 1.0)

    if elapsed <= max(median_dt, 1e-9):
        return _sample_recent_indices(end_idx, length)

    # length-1 historical targets plus the current sample.  Geometric spacing
    # gives dense recent context while still reaching back to experiment start.
    lookbacks = np.geomspace(max(median_dt, 1e-9), max(elapsed, median_dt), num=max(length - 1, 1))
    target_times = t1 - lookbacks[::-1]
    target_times = np.concatenate([target_times, np.asarray([t1], np.float64)])
    ids = np.searchsorted(prefix, target_times, side="right") - 1
    ids = np.clip(ids, 0, end_idx).astype(np.int64)
    ids[-1] = end_idx
    return ids


def _sample_indices_by_time(time: np.ndarray, end_idx: int, length: int, span_sec: float | None, global_history: bool = False):
    """Causally sample a fixed number of tokens using real physical time."""
    length = max(int(length), 1)
    prefix = np.asarray(time[: end_idx + 1], np.float64)
    if len(prefix) == 0:
        return np.zeros(length, dtype=np.int64)
    t0 = float(prefix[0])
    t1 = float(prefix[-1])
    if global_history or span_sec is None or span_sec <= 0:
        start_t = t0
    else:
        start_t = max(t0, t1 - float(span_sec))
    targets = np.linspace(start_t, t1, length, dtype=np.float64)
    ids = np.searchsorted(prefix, targets, side="right") - 1
    ids = np.clip(ids, 0, end_idx).astype(np.int64)
    ids[-1] = end_idx
    return ids


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        experiment_paths: Iterable[str | Path],
        short_len: int = 8,
        medium_len: int = 12,
        long_len: int = 16,
        long_span: int = 64,  # retained for CLI/checkpoint compatibility; not used for physical-time sampling
        stride: int = 1,
        normalize_mean: np.ndarray | None = None,
        normalize_std: np.ndarray | None = None,
        transition_horizon_samples: int = 12,
        transition_horizon_sec: float | None = None,
        short_span_sec: float | None = 0.0,
        medium_span_sec: float | None = 0.0,
    ):
        self.short_len = int(short_len)
        self.medium_len = int(medium_len)
        self.long_len = int(long_len)
        self.long_span = int(long_span)
        self.stride = max(int(stride), 1)
        self.short_span_sec = None if short_span_sec is None else float(short_span_sec)
        self.medium_span_sec = None if medium_span_sec is None else float(medium_span_sec)
        self.transition_horizon_samples = max(int(transition_horizon_samples), 1)
        self.transition_horizon_sec = None if transition_horizon_sec is None else float(transition_horizon_sec)
        self.experiments = [load_experiment(p) for p in experiment_paths]
        self.index: list[tuple[int, int]] = []
        self.transition_aux = []

        for ei, exp in enumerate(self.experiments):
            self.transition_aux.append(_distance_targets(exp, self.transition_horizon_samples, self.transition_horizon_sec))
            # Padding-by-repetition allows INITIAL-stage windows while staying causal.
            for t in range(0, len(exp.x), self.stride):
                self.index.append((ei, t))

        if not self.experiments:
            raise ValueError("No experiments supplied")
        self.feature_names = list(self.experiments[0].feature_names)
        expected_dim = int(self.experiments[0].x.shape[1])
        schema_errors = []
        for exp in self.experiments:
            if int(exp.x.shape[1]) != expected_dim or list(exp.feature_names) != self.feature_names:
                schema_errors.append({
                    "experiment_id": exp.experiment_id,
                    "dim": int(exp.x.shape[1]),
                    "feature_names": list(exp.feature_names),
                })
        if schema_errors:
            preview = schema_errors[:5]
            raise ValueError(
                "Feature schema mismatch across prepared experiments. Re-run dataset preparation with the V5.2.2 adapter. "
                f"Expected dim={expected_dim}, names={self.feature_names}; mismatches={preview}"
            )

        if normalize_mean is None or normalize_std is None:
            all_x = np.concatenate([e.x for e in self.experiments], axis=0)
            self.mean = all_x.mean(0).astype(np.float32)
            self.std = (all_x.std(0) + 1e-6).astype(np.float32)
        else:
            self.mean = np.asarray(normalize_mean, dtype=np.float32)
            self.std = np.asarray(normalize_std, dtype=np.float32)

    def __len__(self):
        return len(self.index)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def sampling_weights(self, boost: float = 3.0) -> torch.Tensor:
        weights = np.ones(len(self.index), dtype=np.float64)
        for i, (ei, t) in enumerate(self.index):
            _, _, near_mask, _, _ = self.transition_aux[ei]
            heat = float(self.experiments[ei].transition[t])
            if near_mask[t] > 0:
                weights[i] += boost * (0.35 + 0.65 * heat)
        return torch.as_tensor(weights, dtype=torch.double)

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        e = self.experiments[ei]
        dist_norm, dist_sec, dist_mask, _, h_sec = self.transition_aux[ei]

        # V5.2.2 default: scale-adaptive causal sampling with strict schema checks.
        # short<=0  -> most recent samples (local detail)
        # medium<=0 -> logarithmic lookback over all observed history
        # global     -> uniform sparse history from experiment start to now
        if self.short_span_sec is None or self.short_span_sec <= 0:
            short_ids = _sample_recent_indices(t, self.short_len)
        else:
            short_ids = _sample_indices_by_time(e.time, t, self.short_len, self.short_span_sec, global_history=False)

        if self.medium_span_sec is None or self.medium_span_sec <= 0:
            medium_ids = _sample_log_history_indices(e.time, t, self.medium_len)
        else:
            medium_ids = _sample_indices_by_time(e.time, t, self.medium_len, self.medium_span_sec, global_history=False)

        global_ids = _sample_indices_by_time(e.time, t, self.long_len, None, global_history=True)

        short = self._norm(e.x[short_ids])
        medium = self._norm(e.x[medium_ids])
        long = self._norm(e.x[global_ids])
        elapsed = np.asarray(e.time - e.time[0], np.float32)

        target_dist_sec = float(dist_sec[t]) if np.isfinite(dist_sec[t]) else 0.0
        return {
            "short_x": torch.from_numpy(short.astype(np.float32)),
            "medium_x": torch.from_numpy(medium.astype(np.float32)),
            "long_x": torch.from_numpy(long.astype(np.float32)),
            "short_time": torch.from_numpy(elapsed[short_ids].astype(np.float32)),
            "medium_time": torch.from_numpy(elapsed[medium_ids].astype(np.float32)),
            "long_time": torch.from_numpy(elapsed[global_ids].astype(np.float32)),
            "stage": torch.tensor(int(e.stage[t]), dtype=torch.long),
            "transition": torch.tensor(float(e.transition[t]), dtype=torch.float32),
            "transition_distance_norm": torch.tensor(float(dist_norm[t]), dtype=torch.float32),
            "transition_distance_sec": torch.tensor(target_dist_sec, dtype=torch.float32),
            "transition_distance_mask": torch.tensor(float(dist_mask[t]), dtype=torch.float32),
            "transition_distance_scale_sec": torch.tensor(float(h_sec), dtype=torch.float32),
            "progress": torch.tensor(float(e.progress[t]), dtype=torch.float32),
            "time": torch.tensor(float(e.time[t]), dtype=torch.float32),
            "experiment_id": e.experiment_id,
        }


def save_split_json(path: str | Path, train, val, test, extra: dict | None = None):
    data = {
        "train": [str(Path(p).name) for p in train],
        "val": [str(Path(p).name) for p in val],
        "test": [str(Path(p).name) for p in test],
    }
    if extra:
        data["split_provenance"] = extra
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
