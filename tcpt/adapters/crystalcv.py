from __future__ import annotations

from pathlib import Path
import json
import pickle
import re

import cv2
import numpy as np
import pandas as pd

from .base import BaseReactionAdapter
from .schema import AdapterCapabilities, CanonicalExperiment
from .utils import TABLE_EXTS, VIDEO_EXTS, read_table, find_time_col, time_to_seconds, robust_progress, weak_stage_targets, smooth


class CrystalCVAdapter(BaseReactionAdapter):
    dataset_name = "crystalcv"
    capabilities = AdapterCapabilities(
        temporal=True,
        frame_labels=False,
        author_stage_gt=False,
        raw_video=True,
        notes="Uses CrystalCV tracking tables when present; can fall back to causal visual features from videos. Process-stage labels are derived/weak.",
    )

    def __init__(self, root: str | Path, min_rows: int = 48, video_hz: float = 2.0):
        super().__init__(root)
        self.min_rows = int(min_rows)
        self.video_hz = float(video_hz)

    def discover(self) -> list[Path]:
        if not self.root.exists():
            return []
        allowed = TABLE_EXTS | VIDEO_EXTS | {".json", ".npz", ".npy", ".pkl", ".pickle"}
        return [p for p in sorted(self.root.rglob("*")) if p.is_file() and p.suffix.lower() in allowed]

    def _table_to_experiment(self, df: pd.DataFrame, path: Path) -> CanonicalExperiment | None:
        if len(df) < self.min_rows:
            return None
        time_col = find_time_col(df)
        if time_col is not None:
            t = time_to_seconds(df[time_col], str(time_col))
        else:
            frame_col = next((c for c in df.columns if "frame" in str(c).lower()), None)
            if frame_col is not None:
                t = pd.to_numeric(df[frame_col], errors="coerce").to_numpy(np.float32)
            else:
                t = np.arange(len(df), dtype=np.float32)

        # Try long-form crystal tracking: aggregate per time point.
        id_col = next((c for c in df.columns if re.search(r"crystal.*id|track.*id|object.*id", str(c), re.I)), None)
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return None

        if id_col is not None and time_col is not None:
            tmp = df.copy()
            tmp["__time_s"] = t
            groups = []
            for tv, g in tmp.groupby("__time_s", sort=True):
                row = {"time": float(tv), "crystal_count": float(g[id_col].nunique(dropna=True))}
                for c in numeric.columns:
                    n = re.sub(r"[^a-z0-9]", "", str(c).lower())
                    if c == time_col or c == id_col:
                        continue
                    if any(k in n for k in ["area", "size", "length", "width", "growth", "aspect", "extent"]):
                        vals = pd.to_numeric(g[c], errors="coerce")
                        row[f"mean_{c}"] = float(vals.mean())
                        if "area" in n or "size" in n:
                            row[f"sum_{c}"] = float(vals.sum())
                groups.append(row)
            agg = pd.DataFrame(groups)
            return self._aggregate_tracking_table(agg, path)

        return self._aggregate_tracking_table(df.assign(__time_s=t), path)

    def _aggregate_tracking_table(self, df: pd.DataFrame, path: Path) -> CanonicalExperiment | None:
        tcol = "__time_s" if "__time_s" in df.columns else find_time_col(df)
        if tcol is None:
            t = np.arange(len(df), dtype=np.float32)
        else:
            t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(np.float32)
        numeric = df.select_dtypes(include=[np.number]).copy()
        candidates = []
        for c in numeric.columns:
            if c == tcol:
                continue
            n = re.sub(r"[^a-z0-9]", "", str(c).lower())
            if any(k in n for k in ["area", "size", "length", "width", "growth", "aspect", "extent", "count", "nucle", "radius", "volume"]):
                candidates.append(c)
        if not candidates:
            candidates = [c for c in numeric.columns if c != tcol][:12]
        if not candidates:
            return None

        x = numeric[candidates].to_numpy(np.float32)
        good = np.isfinite(t) & np.isfinite(x).any(axis=1)
        t, x = t[good], x[good]
        if len(t) < self.min_rows:
            return None
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0)
        x = np.where(np.isfinite(x), x, med)

        # Prefer total/sum crystal area as progress driver, then size/area, then first feature.
        driver_idx = 0
        scores = []
        for j, c in enumerate(candidates):
            n = str(c).lower()
            score = (3 if "sum_" in n and "area" in n else 0) + (2 if "area" in n or "size" in n else 0) + (1 if "count" in n else 0)
            scores.append(score)
        driver_idx = int(np.argmax(scores))
        driver = x[:, driver_idx]
        # Crystal growth is usually increasing; robust envelope suppresses segmentation jitter.
        p = robust_progress(driver, monotonic=True)
        stage, trans, prog, oi, ei = weak_stage_targets(p, t)

        signals = {re.sub(r"[^A-Za-z0-9_.-]+", "_", str(c)): x[:, j] for j, c in enumerate(candidates)}
        signals["crystal_progress_driver"] = driver.astype(np.float32)
        if len(t) > 2:
            progress_rate = np.gradient(smooth(prog, 5), t, edge_order=1).astype(np.float32)
            signals["progress_rate"] = progress_rate
            signals["progress_acceleration"] = np.gradient(smooth(progress_rate, 5), t, edge_order=1).astype(np.float32)

        rel = str(path.relative_to(self.root))
        exp_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
        return CanonicalExperiment(
            experiment_id=exp_id,
            source="crystalcv_tracking",
            time_s=t,
            signals=signals,
            stage=stage,
            transition=trans,
            progress=prog,
            onset_time_s=float(t[oi]),
            endpoint_time_s=float(t[ei]),
            label_quality="derived_from_crystal_tracking",
            meta={
                "source_file": rel,
                "driver_feature": str(candidates[driver_idx]),
                "weak_label_method": "v52_kinetics_rate_sustained_boundaries",
                "warning": "Reaction-stage labels are TCPT adapter-derived from CrystalCV tracking trajectories, not author stage GT.",
            },
        ).validate(min_rows=self.min_rows)

    def _video_to_experiment(self, path: Path) -> CanonicalExperiment | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        step = max(1, int(round(fps / max(self.video_hz, 0.1))))
        feats, times = [], []
        prev_gray = None
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step:
                frame_idx += 1
                continue
            small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
            means = lab.reshape(-1, 3).mean(0)
            stds = lab.reshape(-1, 3).std(0)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float((edges > 0).mean())
            if prev_gray is None:
                motion = 0.0
            else:
                motion = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
            prev_gray = gray
            sat_mean = float(hsv[..., 1].mean() / 255.0)
            bright_std = float(gray.std() / 255.0)
            feats.append([*means.tolist(), *stds.tolist(), edge_density, motion, sat_mean, bright_std])
            times.append(frame_idx / fps)
            frame_idx += 1
        cap.release()
        if len(feats) < self.min_rows:
            return None
        x = np.asarray(feats, np.float32)
        t = np.asarray(times, np.float32)
        # For video-only fallback, use a combined visual-change trajectory for weak pretraining.
        z = (x - x[0:1]) / (x.std(0, keepdims=True) + 1e-6)
        driver = np.linalg.norm(z, axis=1)
        p = robust_progress(driver, monotonic=True)
        stage, trans, prog, oi, ei = weak_stage_targets(p, t)
        names = ["Lab_L_mean", "Lab_a_mean", "Lab_b_mean", "Lab_L_std", "Lab_a_std", "Lab_b_std", "edge_density", "motion", "saturation_mean", "brightness_std"]
        signals = {names[j]: x[:, j] for j in range(x.shape[1])}
        rel = str(path.relative_to(self.root))
        exp_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
        return CanonicalExperiment(
            experiment_id=exp_id,
            source="crystalcv_video",
            time_s=t,
            signals=signals,
            stage=stage,
            transition=trans,
            progress=prog,
            onset_time_s=float(t[oi]),
            endpoint_time_s=float(t[ei]),
            label_quality="weak_visual_change_derived",
            meta={"source_file": rel, "sample_hz": self.video_hz, "warning": "Video fallback uses weak visual-change labels, not author stage GT."},
        ).validate(min_rows=self.min_rows)

    def _load_structured(self, path: Path):
        ext = path.suffix.lower()
        if ext in TABLE_EXTS:
            return read_table(path)
        if ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                # Common pattern: dict of rows/columns
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return None
        if ext == ".npz":
            z = np.load(path, allow_pickle=True)
            cols = {k: np.asarray(z[k]).reshape(-1) for k in z.files if np.asarray(z[k]).ndim <= 2 and np.asarray(z[k]).size > 1}
            lengths = [len(v) for v in cols.values()]
            if lengths:
                n = max(set(lengths), key=lengths.count)
                cols = {k: v for k, v in cols.items() if len(v) == n and v.ndim == 1}
                if cols:
                    return pd.DataFrame(cols)
        if ext in {".pkl", ".pickle"}:
            # Only use this on trusted public CrystalCV files.
            obj = pd.read_pickle(path)
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return None
        return None

    def convert(self) -> list[CanonicalExperiment]:
        out: list[CanonicalExperiment] = []
        structured_seen = 0
        for path in self.discover():
            try:
                if path.suffix.lower() in VIDEO_EXTS:
                    continue
                df = self._load_structured(path)
                if isinstance(df, pd.DataFrame):
                    exp = self._table_to_experiment(df, path)
                    if exp is not None:
                        out.append(exp)
                        structured_seen += 1
            except Exception as e:
                print(f"[CrystalCVAdapter] skip {path}: {type(e).__name__}: {e}")

        # Prefer official tracking outputs; videos are fallback when no usable tracking table exists.
        if structured_seen == 0:
            for path in self.discover():
                if path.suffix.lower() not in VIDEO_EXTS:
                    continue
                try:
                    exp = self._video_to_experiment(path)
                    if exp is not None:
                        out.append(exp)
                except Exception as e:
                    print(f"[CrystalCVAdapter] video skip {path}: {type(e).__name__}: {e}")
        return out
