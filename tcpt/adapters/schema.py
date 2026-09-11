from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import re

import numpy as np

from tcpt.data import Experiment, save_experiment


@dataclass
class AdapterCapabilities:
    temporal: bool
    frame_labels: bool = False
    author_stage_gt: bool = False
    raw_video: bool = False
    notes: str = ""


@dataclass
class CanonicalExperiment:
    """Dataset-neutral representation of one reaction/process run.

    Important: `stage`, `transition`, and `progress` may be author ground truth,
    derived labels, or weak labels. `label_quality` and `meta` must preserve that
    provenance so downstream reports do not silently call derived labels GT.
    """

    experiment_id: str
    source: str
    time_s: np.ndarray
    signals: dict[str, np.ndarray]
    stage: np.ndarray | None = None
    transition: np.ndarray | None = None
    progress: np.ndarray | None = None
    onset_time_s: float | None = None
    endpoint_time_s: float | None = None
    label_quality: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self, min_rows: int = 8) -> "CanonicalExperiment":
        t = np.asarray(self.time_s, dtype=np.float32).reshape(-1)
        if len(t) < min_rows:
            raise ValueError(f"{self.experiment_id}: only {len(t)} rows")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"{self.experiment_id}: non-finite timestamps")

        arrays: dict[str, np.ndarray] = {}
        for name, arr in self.signals.items():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 1:
                a = a[:, None]
            if a.ndim != 2 or a.shape[0] != len(t):
                raise ValueError(f"{self.experiment_id}: signal {name!r} shape {a.shape} != T={len(t)}")
            arrays[name] = a

        if not arrays:
            raise ValueError(f"{self.experiment_id}: no signals")

        order = np.argsort(t, kind="stable")
        t = t[order]
        # Keep first occurrence for duplicate timestamps to avoid zero dt loops.
        keep = np.ones(len(t), dtype=bool)
        if len(t) > 1:
            keep[1:] = np.diff(t) > 1e-9
        order = order[keep]
        t = np.asarray(self.time_s, np.float32).reshape(-1)[order]

        clean_signals: dict[str, np.ndarray] = {}
        for name, arr in arrays.items():
            a = arr[order]
            if not np.all(np.isfinite(a)):
                med = np.nanmedian(np.where(np.isfinite(a), a, np.nan), axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                a = np.where(np.isfinite(a), a, med)
            clean_signals[name] = a.astype(np.float32)

        def _take_optional(x, dtype):
            if x is None:
                return None
            a = np.asarray(x, dtype=dtype).reshape(-1)
            if len(a) != len(self.time_s):
                raise ValueError(f"{self.experiment_id}: label length mismatch")
            return a[order]

        self.time_s = t.astype(np.float32)
        self.signals = clean_signals
        self.stage = _take_optional(self.stage, np.int64)
        self.transition = _take_optional(self.transition, np.float32)
        self.progress = _take_optional(self.progress, np.float32)
        return self

    @property
    def reaction_duration_s(self) -> float | None:
        if self.onset_time_s is None or self.endpoint_time_s is None:
            return None
        d = float(self.endpoint_time_s) - float(self.onset_time_s)
        return d if math.isfinite(d) and d > 0 else None

    def flat_feature_dict(self, add_time_features: bool = True) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for group, arr in self.signals.items():
            a = np.asarray(arr, np.float32)
            if a.ndim == 1:
                a = a[:, None]
            for j in range(a.shape[1]):
                name = group if a.shape[1] == 1 else f"{group}_{j}"
                out[name] = a[:, j].astype(np.float32)

        if add_time_features:
            t = self.time_s.astype(np.float32)
            dt = np.diff(t, prepend=t[0]) if len(t) else np.zeros(0, np.float32)
            if len(dt) > 1 and dt[0] <= 0:
                positive = dt[1:][dt[1:] > 0]
                dt[0] = float(np.median(positive)) if len(positive) else 0.0
            elapsed = t - t[0]
            # Both are causal. No true reaction duration is used as a feature.
            out["time_delta_s"] = dt.astype(np.float32)
            out["time_log_elapsed"] = np.log1p(np.maximum(elapsed, 0)).astype(np.float32)
        return out

    def to_tcpt(self, feature_space: list[str] | None = None, add_time_features: bool = True) -> Experiment:
        self.validate()
        flat = self.flat_feature_dict(add_time_features=add_time_features)
        names = list(flat.keys()) if feature_space is None else list(feature_space)
        cols = []
        missing = []
        for name in names:
            if name in flat:
                cols.append(flat[name][:, None])
                missing.append(np.zeros((len(self.time_s), 1), np.float32))
            else:
                cols.append(np.zeros((len(self.time_s), 1), np.float32))
                missing.append(np.ones((len(self.time_s), 1), np.float32))

        x = np.concatenate(cols, axis=1).astype(np.float32)
        # If mixing heterogeneous sources, explicit missing indicators prevent zero-fill ambiguity.
        if feature_space is not None:
            miss = np.concatenate(missing, axis=1)
            x = np.concatenate([x, miss], axis=1)
            names = names + [f"missing::{n}" for n in names]

        n = len(self.time_s)
        if self.stage is None:
            raise ValueError(f"{self.experiment_id}: no stage labels (not suitable for TCPT stage training)")
        transition = np.zeros(n, np.float32) if self.transition is None else np.clip(self.transition, 0, 1).astype(np.float32)
        progress = np.zeros(n, np.float32) if self.progress is None else np.clip(self.progress, 0, 1).astype(np.float32)
        return Experiment(
            experiment_id=self.experiment_id,
            x=x,
            stage=self.stage.astype(np.int64),
            transition=transition,
            progress=progress,
            time=self.time_s.astype(np.float32),
            feature_names=names,
        )

    def save_tcpt(self, out_dir: str | Path, feature_space: list[str] | None = None, add_time_features: bool = True) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.experiment_id)[:150]
        npz = out / f"{safe}.npz"
        save_experiment(self.to_tcpt(feature_space=feature_space, add_time_features=add_time_features), npz)
        sidecar = npz.with_suffix(".meta.json")
        payload = {
            "experiment_id": self.experiment_id,
            "source": self.source,
            "label_quality": self.label_quality,
            "onset_time_s": self.onset_time_s,
            "endpoint_time_s": self.endpoint_time_s,
            "reaction_duration_s": self.reaction_duration_s,
            "meta": self.meta,
        }
        sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return npz


def union_feature_space(experiments: list[CanonicalExperiment], add_time_features: bool = True) -> list[str]:
    seen: dict[str, None] = {}
    for exp in experiments:
        for name in exp.flat_feature_dict(add_time_features=add_time_features):
            seen.setdefault(name, None)
    return list(seen.keys())
