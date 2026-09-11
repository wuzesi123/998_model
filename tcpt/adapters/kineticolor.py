from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .base import BaseReactionAdapter
from .schema import AdapterCapabilities, CanonicalExperiment
from .utils import TABLE_EXTS, read_table, find_time_col, time_to_seconds, change_signal, robust_progress, weak_stage_targets, smooth


KINETICOLOR_CANONICAL_FEATURES = [
    "process_relative_magnitude",
    "process_activity_rate",
    "process_acceleration",
    "process_cumulative_activity",
    "process_local_slope",
    "process_local_variance",
    "process_stability",
]


class KineticolorAdapter(BaseReactionAdapter):
    dataset_name = "kineticolor"
    capabilities = AdapterCapabilities(
        temporal=True,
        frame_labels=False,
        author_stage_gt=False,
        raw_video=False,
        notes=(
            "Machine-readable reaction trajectories. TCPT stage/transition labels are weak/derived, not author GT. "
            "V5.2.2 maps RGB/Lab/arbitrary numeric trajectories into one causal canonical process-feature schema."
        ),
    )

    def __init__(self, root: str | Path, min_rows: int = 48):
        super().__init__(root)
        self.min_rows = int(min_rows)

    def discover(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [p for p in sorted(self.root.rglob("*")) if p.is_file() and p.suffix.lower() in TABLE_EXTS]

    @staticmethod
    def _group_col(df: pd.DataFrame):
        priority = ["well", "wellid", "sample", "sampleid", "reaction", "reactionid", "experiment", "experimentid", "vial", "vialid"]
        by_norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
        for k in priority:
            if k in by_norm:
                c = by_norm[k]
                nunique = df[c].nunique(dropna=True)
                if 1 < nunique <= max(4, len(df) // 6):
                    return c
        return None

    @staticmethod
    def _channel_groups(columns):
        """Find prefixes like A1_R/A1_G/A1_B or sample.L/sample.a/sample.b."""
        groups: dict[str, dict[str, str]] = {}
        for c in columns:
            s = str(c)
            m = re.match(r"^(.*?)[_.\- ]+(r|g|b|l|a|b\*|l\*|a\*)$", s, flags=re.I)
            if not m:
                continue
            prefix = m.group(1).strip()
            ch = m.group(2).lower().replace("*", "")
            groups.setdefault(prefix, {})[ch] = c
        valid = {}
        for prefix, mp in groups.items():
            if {"r", "g", "b"}.issubset(mp) or {"l", "a", "b"}.issubset(mp):
                valid[prefix] = mp
        return valid

    @staticmethod
    def _source_group_token(text: str) -> str:
        """Collapse representation-only suffixes so RGB/Lab views of the same source stay together."""
        s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
        s = re.sub(r"(?i)(?:_)?(?:rgb|lab|cie_lab|cielab|colour|color)$", "", s).strip("_")
        return s or "series"

    @staticmethod
    def _activity_progress(x: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Offline weak target from direction-agnostic cumulative trajectory activity.

        This may use the full recorded trajectory because it is an annotation/weak-target
        generator, not an inference feature.  V5.2.2 separately constructs strictly causal
        model-input features below.
        """
        xx = np.asarray(x, np.float32)
        if xx.ndim == 1:
            xx = xx[:, None]
        tt = np.asarray(time_s, np.float64).reshape(-1)
        n = len(xx)
        if n == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)

        med = np.nanmedian(xx, axis=0)
        scale = np.nanstd(xx, axis=0) + 1e-6
        z = (xx - med) / scale
        win = max(5, min(15, (n // 40) * 2 + 5))
        z_s = np.column_stack([smooth(z[:, j], win) for j in range(z.shape[1])]).astype(np.float64)

        if len(tt) != n or not np.all(np.isfinite(tt)):
            tt = np.arange(n, dtype=np.float64)
        tt = np.maximum.accumulate(tt)
        dtt = np.diff(tt)
        pos = dtt[dtt > 1e-9]
        fallback_dt = float(np.median(pos)) if len(pos) else 1.0
        dtt = np.where(dtt > 1e-9, dtt, fallback_dt)

        speed = np.zeros(n, dtype=np.float64)
        if n > 1:
            speed[1:] = np.linalg.norm(np.diff(z_s, axis=0), axis=1) / dtt
        speed = np.maximum(smooth(speed.astype(np.float32), win).astype(np.float64), 0.0)

        finite = speed[np.isfinite(speed)]
        if len(finite) == 0:
            return robust_progress(change_signal(xx), monotonic=True), np.zeros(n, np.float32)
        low = finite[finite <= np.percentile(finite, 35)]
        if len(low) == 0:
            low = finite
        low_med = float(np.median(low))
        mad = float(1.4826 * np.median(np.abs(low - low_med))) if len(low) else 0.0
        noise_floor = low_med + 2.0 * mad
        activity = np.maximum(speed - noise_floor, 0.0)

        p95 = float(np.percentile(activity, 95)) if len(activity) else 0.0
        if p95 > 1e-12:
            activity[activity < 0.05 * p95] = 0.0

        cumulative = np.zeros(n, dtype=np.float64)
        if n > 1:
            inc = 0.5 * (activity[1:] + activity[:-1]) * dtt
            cumulative[1:] = np.cumsum(np.maximum(inc, 0.0))

        if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 1e-8:
            progress = robust_progress(change_signal(xx), monotonic=True)
        else:
            progress = robust_progress(cumulative.astype(np.float32), monotonic=True)
        return progress.astype(np.float32), activity.astype(np.float32)

    @staticmethod
    def _canonical_process_features(x: np.ndarray, time_s: np.ndarray) -> dict[str, np.ndarray]:
        """Map any numeric trajectory into a fixed, strictly causal process-feature schema.

        No full-reaction min/max, endpoint, final duration, centred smoothing, or future sample
        is used.  RGB, Lab and single/multichannel numeric series therefore have the same model
        semantics and can safely share one TCPT feature space.
        """
        xx = np.asarray(x, np.float64)
        if xx.ndim == 1:
            xx = xx[:, None]
        tt = np.asarray(time_s, np.float64).reshape(-1)
        n = len(xx)
        if n == 0:
            return {k: np.zeros(0, np.float32) for k in KINETICOLOR_CANONICAL_FEATURES}

        if len(tt) != n or not np.all(np.isfinite(tt)):
            tt = np.arange(n, dtype=np.float64)
        tt = np.maximum.accumulate(tt)
        dt = np.diff(tt, prepend=tt[0])
        pos = dt[dt > 1e-9]
        fallback_dt = float(np.median(pos)) if len(pos) else 1.0
        dt = np.where(dt > 1e-9, dt, fallback_dt)

        x0 = xx[0].copy()
        baseline_scale = float(np.sqrt(np.mean(np.square(x0))))
        if not np.isfinite(baseline_scale) or baseline_scale < 1e-3:
            baseline_scale = 1.0

        rel = np.linalg.norm(xx - x0[None, :], axis=1) / baseline_scale
        rel = np.log1p(np.maximum(rel, 0.0))

        raw_speed = np.zeros(n, dtype=np.float64)
        if n > 1:
            raw_speed[1:] = np.linalg.norm(np.diff(xx, axis=0), axis=1) / dt[1:] / baseline_scale

        # Causal EMA only; unlike centred moving averages, this never sees a future point.
        speed = np.zeros(n, dtype=np.float64)
        alpha = 0.25
        for i in range(1, n):
            speed[i] = alpha * raw_speed[i] + (1.0 - alpha) * speed[i - 1]

        acceleration = np.zeros(n, dtype=np.float64)
        local_slope = np.zeros(n, dtype=np.float64)
        if n > 1:
            acceleration[1:] = np.diff(speed) / dt[1:]
            local_slope[1:] = np.diff(rel) / dt[1:]

        cumulative = np.zeros(n, dtype=np.float64)
        if n > 1:
            cumulative[1:] = np.cumsum(np.maximum(speed[1:] * dt[1:], 0.0))
        cumulative = np.log1p(np.maximum(cumulative, 0.0))

        local_var = np.zeros(n, dtype=np.float64)
        window = 8
        for i in range(n):
            j = max(0, i - window + 1)
            local_var[i] = float(np.var(rel[j:i + 1])) if i > j else 0.0

        stability = 1.0 / (1.0 + np.maximum(speed, 0.0) + np.sqrt(np.maximum(local_var, 0.0)))

        vals = [rel, speed, acceleration, cumulative, local_slope, local_var, stability]
        out: dict[str, np.ndarray] = {}
        for name, arr in zip(KINETICOLOR_CANONICAL_FEATURES, vals):
            a = np.asarray(arr, np.float32)
            a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
            out[name] = a
        return out

    def _from_group(self, df: pd.DataFrame, path: Path, suffix: str = "") -> list[CanonicalExperiment]:
        time_col = find_time_col(df)
        if time_col is not None:
            t = time_to_seconds(df[time_col], str(time_col))
            work = df.drop(columns=[time_col])
        else:
            t = np.arange(len(df), dtype=np.float32)
            work = df.copy()

        good_t = np.isfinite(t)
        t = t[good_t]
        work = work.loc[good_t].reset_index(drop=True)
        if len(t) < self.min_rows:
            return []

        numeric = work.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return []

        out: list[CanonicalExperiment] = []
        groups = self._channel_groups(numeric.columns)
        rel = str(path.relative_to(self.root))
        base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)

        def build(exp_id: str, arr: np.ndarray, original_feature_names: list[str], meta_extra: dict, source_group_token: str):
            arr = np.asarray(arr, np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            valid = np.isfinite(arr).any(axis=1)
            if valid.sum() < self.min_rows:
                return
            tt = t[valid]
            xx = arr[valid]
            med = np.nanmedian(xx, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            xx = np.where(np.isfinite(xx), xx, med)

            # Weak annotation targets may use the whole trajectory offline.
            prog, _ = self._activity_progress(xx, tt)
            stage, trans, prog, oi, ei = weak_stage_targets(prog, tt)

            # Model inputs are fixed-schema and strictly causal.
            signals = self._canonical_process_features(xx, tt)
            source_group = f"{rel}::{suffix}::{self._source_group_token(source_group_token)}"

            out.append(CanonicalExperiment(
                experiment_id=exp_id,
                source="kineticolor",
                time_s=tt,
                signals=signals,
                stage=stage,
                transition=trans,
                progress=prog,
                onset_time_s=float(tt[oi]),
                endpoint_time_s=float(tt[ei]),
                label_quality="weak_trajectory_derived",
                meta={
                    "source_file": rel,
                    "source_group": source_group,
                    "original_features": original_feature_names,
                    "canonical_features": KINETICOLOR_CANONICAL_FEATURES,
                    "input_feature_method": "v522_causal_schema_invariant_process_features",
                    "weak_label_method": "v521_direction_agnostic_activity_integral+sustained_rate_boundaries",
                    "warning": (
                        "Stage/transition/progress labels were derived by this adapter from camera-derived trajectories; "
                        "they are not author-provided GT. Canonical model features are causal; weak targets are offline annotations."
                    ),
                    **meta_extra,
                },
            ).validate(min_rows=self.min_rows))

        if groups:
            for prefix, mp in groups.items():
                if {"r", "g", "b"}.issubset(mp):
                    cols = [mp["r"], mp["g"], mp["b"]]
                    names = ["R", "G", "B"]
                else:
                    cols = [mp["l"], mp["a"], mp["b"]]
                    names = ["L", "a", "b"]
                token = self._source_group_token(prefix)
                build(
                    f"{base_id}{suffix}__{re.sub(r'[^A-Za-z0-9_.-]+','_',prefix)}",
                    numeric[cols].to_numpy(), names,
                    {"series_prefix": prefix, "representation": "RGB_or_Lab"},
                    token,
                )
            return out

        cols = [c for c in numeric.columns if not re.search(r"(^|_)(index|frame|id)$", str(c).lower())]
        if 1 <= len(cols) <= 12:
            token = self._source_group_token(Path(rel).stem + suffix)
            build(f"{base_id}{suffix}", numeric[cols].to_numpy(), [str(c) for c in cols], {}, token)
            return out

        for c in cols:
            s = pd.to_numeric(numeric[c], errors="coerce").to_numpy(np.float32)
            if np.isfinite(s).sum() < self.min_rows:
                continue
            token = self._source_group_token(str(c))
            build(
                f"{base_id}{suffix}__{re.sub(r'[^A-Za-z0-9_.-]+','_',str(c))}",
                s[:, None], [str(c)], {"series_column": str(c)}, token,
            )
        return out

    def convert(self) -> list[CanonicalExperiment]:
        all_exps: list[CanonicalExperiment] = []
        for path in self.discover():
            try:
                df = read_table(path)
                if len(df) < self.min_rows:
                    continue
                gcol = self._group_col(df)
                if gcol is None:
                    all_exps.extend(self._from_group(df, path))
                else:
                    for key, grp in df.groupby(gcol, dropna=True, sort=False):
                        if len(grp) < self.min_rows:
                            continue
                        suffix = f"__{gcol}_{key}"
                        all_exps.extend(self._from_group(grp.drop(columns=[gcol]), path, suffix=suffix))
            except Exception as e:
                print(f"[KineticolorAdapter] skip {path}: {type(e).__name__}: {e}")
        return all_exps
