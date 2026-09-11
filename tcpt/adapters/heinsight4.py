from __future__ import annotations

from pathlib import Path
import csv
import re

import cv2
import numpy as np

from .base import BaseReactionAdapter
from .schema import AdapterCapabilities, CanonicalExperiment
from .utils import VIDEO_EXTS, robust_progress, weak_stage_targets


class HeinSight4Adapter(BaseReactionAdapter):
    dataset_name = "heinsight4"
    capabilities = AdapterCapabilities(
        temporal=False,
        frame_labels=True,
        author_stage_gt=False,
        raw_video=False,
        notes="The public release is primarily manually annotated phase images/boxes. Use it for visual encoder/phase pretraining, not as reaction-stage GT.",
    )

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root: str | Path, allow_filename_sequences: bool = False, min_sequence_frames: int = 48):
        super().__init__(root)
        self.allow_filename_sequences = bool(allow_filename_sequences)
        self.min_sequence_frames = int(min_sequence_frames)

    def discover(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [p for p in sorted(self.root.rglob("*")) if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS | VIDEO_EXTS | {".txt", ".yaml", ".yml"}]

    def _find_images(self):
        return [p for p in self.discover() if p.suffix.lower() in self.IMAGE_EXTS]

    def _label_path(self, image: Path) -> Path | None:
        # YOLO convention: .../images/train/x.jpg -> .../labels/train/x.txt
        parts = list(image.parts)
        for i, part in enumerate(parts):
            if part.lower() == "images":
                cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
                if cand.exists():
                    return cand
        cand = image.with_suffix(".txt")
        return cand if cand.exists() else None

    def static_manifest(self, out_csv: str | Path) -> dict:
        images = self._find_images()
        rows = []
        class_counts: dict[int, int] = {}
        for img in images:
            lp = self._label_path(img)
            labels = []
            if lp is not None:
                try:
                    for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                        tok = line.strip().split()
                        if tok:
                            c = int(float(tok[0]))
                            labels.append(c)
                            class_counts[c] = class_counts.get(c, 0) + 1
                except Exception:
                    pass
            rows.append({
                "image": str(img.relative_to(self.root)),
                "label_file": str(lp.relative_to(self.root)) if lp else "",
                "classes": ";".join(map(str, labels)),
                "n_boxes": len(labels),
            })
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "label_file", "classes", "n_boxes"])
            w.writeheader(); w.writerows(rows)
        return {"n_images": len(images), "class_box_counts": class_counts, "manifest": str(out)}

    def _sequence_groups(self):
        groups: dict[str, list[tuple[int, Path]]] = {}
        for p in self._find_images():
            m = re.match(r"^(.*?)(\d{3,})$", p.stem)
            if not m:
                continue
            prefix, idx = m.group(1), int(m.group(2))
            key = str(p.parent.relative_to(self.root)) + "::" + prefix
            groups.setdefault(key, []).append((idx, p))
        return {k: sorted(v) for k, v in groups.items() if len(v) >= self.min_sequence_frames}

    def _image_sequence_to_exp(self, key: str, seq: list[tuple[int, Path]]) -> CanonicalExperiment | None:
        feats, times = [], []
        prev = None
        for order_idx, (frame_no, p) in enumerate(seq):
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            means = lab.reshape(-1, 3).mean(0)
            stds = lab.reshape(-1, 3).std(0)
            edge = float((cv2.Canny(gray, 50, 150) > 0).mean())
            motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(gray, prev)) / 255.0)
            prev = gray
            # Phase box fractions by class, if labels are available.
            counts = np.zeros(5, np.float32)
            lp = self._label_path(p)
            if lp:
                for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    tok = line.split()
                    if tok:
                        c = int(float(tok[0]))
                        if 0 <= c < 5:
                            counts[c] += 1
            if counts.sum() > 0:
                counts /= counts.sum()
            feats.append([*means.tolist(), *stds.tolist(), edge, motion, *counts.tolist()])
            times.append(float(order_idx))
        if len(feats) < self.min_sequence_frames:
            return None
        x = np.asarray(feats, np.float32)
        t = np.asarray(times, np.float32)
        z = (x - x[0:1]) / (x.std(0, keepdims=True) + 1e-6)
        p = robust_progress(np.linalg.norm(z, axis=1), monotonic=True)
        stage, trans, prog, oi, ei = weak_stage_targets(p, t)
        names = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std", "edge", "motion", "phase0", "phase1", "phase2", "phase3", "phase4"]
        return CanonicalExperiment(
            experiment_id=re.sub(r"[^A-Za-z0-9_.-]+", "_", key),
            source="heinsight4_filename_sequence",
            time_s=t,
            signals={names[j]: x[:, j] for j in range(x.shape[1])},
            stage=stage,
            transition=trans,
            progress=prog,
            onset_time_s=float(t[oi]),
            endpoint_time_s=float(t[ei]),
            label_quality="weak_filename_sequence_derived",
            meta={
                "warning": "Sequence order was inferred from filenames and stage labels are weak. Do not use as primary temporal benchmark unless provenance confirms frames came from the same ordered video.",
                "sequence_key": key,
            },
        ).validate(min_rows=self.min_sequence_frames)

    def convert(self) -> list[CanonicalExperiment]:
        # Public HeinSight4 image annotations are valid static phase supervision, not temporal stage GT.
        if not self.allow_filename_sequences:
            return []
        out = []
        for key, seq in self._sequence_groups().items():
            try:
                exp = self._image_sequence_to_exp(key, seq)
                if exp is not None:
                    out.append(exp)
            except Exception as e:
                print(f"[HeinSight4Adapter] skip {key}: {type(e).__name__}: {e}")
        return out
