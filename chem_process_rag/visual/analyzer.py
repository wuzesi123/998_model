from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .features import FEATURE_NAMES, extract_frame_features


def _slope(y: np.ndarray, t: np.ndarray) -> float:
    """Least-squares slope y(t), robust to short/degenerate sequences."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    t = np.asarray(t, dtype=np.float32).reshape(-1)

    if len(y) < 3 or len(t) != len(y) or t[-1] <= t[0]:
        return 0.0

    x = t - t.mean()
    yy = y - y.mean()
    den = float((x * x).sum())
    if den <= 1e-12:
        return 0.0
    return float((x * yy).sum() / den)


def _trend_stats(arr: np.ndarray, t: np.ndarray, recent_n: int = 8) -> dict:
    """Summarise *causal* history up to the current frame.

    The helper deliberately reports both a whole-history trend and a recent trend.
    This lets the semantic layer distinguish, for example:

        overall solid evidence increased, but recent slope ~= 0

    from:

        solid evidence is still actively increasing.

    No future frames or final-reaction metadata are used.
    """
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    t = np.asarray(t, dtype=np.float32).reshape(-1)

    n = len(arr)
    if n == 0:
        return {
            "early_mean": 0.0,
            "late_mean": 0.0,
            "slope": 0.0,
            "recent_slope": 0.0,
            "late_to_early_ratio": 1.0,
            "delta": 0.0,
        }

    if n == 1:
        value = float(arr[0])
        return {
            "early_mean": value,
            "late_mean": value,
            "slope": 0.0,
            "recent_slope": 0.0,
            "late_to_early_ratio": 1.0,
            "delta": 0.0,
        }

    # For very short clips use what is available. For normal clips compare
    # the first and last recent_n observations.
    k = min(n, max(2, int(recent_n)))
    early = arr[:k]
    late = arr[-k:]

    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    overall_slope = float(_slope(arr, t))
    recent_slope = float(_slope(late, t[-k:]))

    # Ratios are diagnostic only. Near-zero early values otherwise create
    # meaningless 1e3-1e6 x ratios, so use a floor and cap the display value.
    denom = max(abs(early_mean), 1e-4)
    ratio = float(np.clip(late_mean / denom, -100.0, 100.0))

    return {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "slope": overall_slope,
        "recent_slope": recent_slope,
        "late_to_early_ratio": ratio,
        "delta": float(late_mean - early_mean),
    }


class VisualProcessAnalyzer:
    """Camera-only visual-process analyser.

    Deterministic OpenCV features are always available. A trained TCPT
    checkpoint is optional; an absent checkpoint never causes a random neural
    model to contribute to the chemical interpretation.
    """

    def __init__(
        self,
        sample_every_sec: float = 1.0,
        checkpoint: str | None = None,
        device: str | None = None,
    ):
        self.sample_every_sec = float(sample_every_sec)
        self.checkpoint = checkpoint
        self.device = device

    def analyze_video(self, video_path: str | Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        step = max(1, int(round(fps * self.sample_every_sec)))

        rows: list[np.ndarray] = []
        times: list[float] = []
        prev = None
        i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if i % step == 0:
                vec, prev = extract_frame_features(frame, prev)
                rows.append(vec)
                times.append(i / fps)
            i += 1

        cap.release()

        if len(rows) < 2:
            raise RuntimeError("Video too short after sampling")

        x = np.stack(rows).astype(np.float32, copy=False)
        t = np.asarray(times, dtype=np.float32)

        recent = min(len(t), max(3, min(len(t), 8)))
        sl = slice(len(t) - recent, len(t))

        # Lab-space distance from the first observed frame. This is a visual
        # trajectory measure only; it is not a chemical conversion estimate.
        color = np.linalg.norm(x[:, 3:6] - x[0, 3:6], axis=1)

        recent_color_slope = abs(_slope(color[sl], t[sl]))
        recent_motion = float(x[sl, 6].mean())
        stability = float(
            np.clip(
                1.0 - (recent_color_slope * 10.0 + recent_motion * 3.0),
                0.0,
                1.0,
            )
        )

        summary = {
            "duration_sec": float(t[-1]),
            "samples": int(len(t)),
            "color_change_magnitude": float(color[-1]),
            "color_change_rate": float(_slope(color[sl], t[sl])),
            "motion_activity": float(x[sl, 6].mean()),
            "turbidity_proxy": float(x[sl, 9].mean()),
            "solid_proxy": float(x[sl, 10].mean()),
            "phase_boundary_proxy": float(x[sl, 11].mean()),
            "stability": stability,
            "brightness_change": float(x[-1, 2] - x[0, 2]),
        }

        # Causal trend summaries used by the semantic layer. These are far more
        # informative than a single current scalar: they capture both how much
        # the observable changed over the clip and whether that change continues
        # in the most recent window.
        summary["trends"] = {
            "color_change": _trend_stats(color, t, recent_n=recent),
            "motion": _trend_stats(x[:, 6], t, recent_n=recent),
            "edge_density": _trend_stats(x[:, 7], t, recent_n=recent),
            "local_contrast": _trend_stats(x[:, 8], t, recent_n=recent),
            "turbidity": _trend_stats(x[:, 9], t, recent_n=recent),
            "solid": _trend_stats(x[:, 10], t, recent_n=recent),
            "phase_boundary": _trend_stats(x[:, 11], t, recent_n=recent),
            "brightness_std": _trend_stats(x[:, 12], t, recent_n=recent),
        }

        tcpt = self._optional_tcpt(x, t)

        return {
            "feature_names": FEATURE_NAMES,
            "timestamps": t.tolist(),
            "features": x.tolist(),
            "summary": summary,
            "tcpt": tcpt,
        }

    def _optional_tcpt(self, x: np.ndarray, t: np.ndarray) -> dict:
        if not self.checkpoint or not Path(self.checkpoint).exists():
            return {
                "enabled": False,
                "reason": (
                    "no checkpoint supplied; semantic reasoning uses "
                    "deterministic visual evidence only"
                ),
            }

        try:
            import torch

            from .tcpt import MultiScaleTCPT

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(self.checkpoint, map_location=dev)

            model = MultiScaleTCPT(x.shape[1])
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            model.to(dev).eval()

            def take(idxs: np.ndarray):
                xx = torch.tensor(
                    x[idxs], dtype=torch.float32, device=dev
                )[None]
                tt = torch.tensor(
                    t[idxs] - t[idxs][0], dtype=torch.float32, device=dev
                )[None]
                return xx, tt

            n = len(t)
            short_idx = np.arange(max(0, n - 8), n)
            medium_idx = np.unique(
                np.clip(
                    np.round(
                        n - 1 - np.geomspace(1, max(n - 1, 1), 12)
                    ).astype(int),
                    0,
                    n - 1,
                )
            )
            global_idx = np.unique(
                np.linspace(0, n - 1, min(16, n)).round().astype(int)
            )

            with torch.no_grad():
                out = model(
                    *take(short_idx),
                    *take(medium_idx),
                    *take(global_idx),
                )

            return {
                "enabled": True,
                "embedding": out["embedding"][0].cpu().tolist(),
                "state_logits": out["state_logits"][0].cpu().tolist(),
                "device": dev,
            }

        except Exception as exc:
            return {
                "enabled": False,
                "reason": f"checkpoint load/inference failed: {exc}",
            }
