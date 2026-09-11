from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import yaml

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _find_dir(root: Path, candidates: list[str]) -> Path | None:
    lows = [c.lower() for c in candidates]
    for p in root.rglob('*'):
        if p.is_dir() and p.name.lower() in lows:
            if (p / 'images').is_dir():
                return p
    return None


def _find_split_images(root: Path, split_names: list[str]) -> Path | None:
    split = _find_dir(root, split_names)
    if split:
        return split / 'images'
    # Alternative: images/train etc.
    for images_dir in root.rglob('images'):
        if not images_dir.is_dir():
            continue
        for name in split_names:
            p = images_dir / name
            if p.is_dir():
                return p
    return None


def discover_yolo_dataset(root: str | Path) -> dict:
    root = Path(root)
    train_images = _find_split_images(root, ['train', 'training'])
    heldout_images = _find_split_images(root, ['val', 'valid', 'validation', 'test'])
    if not train_images or not heldout_images:
        # Use an existing data.yaml if it resolves cleanly.
        for yml in list(root.rglob('data.yaml')) + list(root.rglob('data.yml')):
            try:
                d = yaml.safe_load(yml.read_text(encoding='utf-8')) or {}
                if 'train' in d and ('val' in d or 'test' in d):
                    base = Path(d.get('path') or yml.parent)
                    if not base.is_absolute():
                        base = (yml.parent / base).resolve()
                    tr = Path(d['train'])
                    va = Path(d.get('val') or d.get('test'))
                    if not tr.is_absolute(): tr = (base / tr).resolve()
                    if not va.is_absolute(): va = (base / va).resolve()
                    if tr.exists() and va.exists():
                        return {'train_images': tr, 'heldout_images': va, 'source_yaml': yml, 'names': d.get('names')}
            except Exception:
                continue
        raise RuntimeError(f'Could not discover YOLO train/heldout images under {root}')

    names = None
    for yml in list(root.rglob('data.yaml')) + list(root.rglob('data.yml')):
        try:
            d = yaml.safe_load(yml.read_text(encoding='utf-8')) or {}
            if d.get('names'):
                names = d['names']
                break
        except Exception:
            pass
    return {'train_images': train_images, 'heldout_images': heldout_images, 'source_yaml': None, 'names': names}


def _images(dir_or_file: Path) -> list[Path]:
    if dir_or_file.is_file() and dir_or_file.suffix.lower() == '.txt':
        out = []
        for line in dir_or_file.read_text(encoding='utf-8').splitlines():
            p = Path(line.strip())
            if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p.resolve())
        return out
    return sorted(p.resolve() for p in dir_or_file.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _group_id(path: Path) -> str:
    s = path.stem.lower()
    # Strip common frame/image suffixes and trailing frame numbers.
    s = re.sub(r'(?:[_-](?:frame|img|image))?[_-]?\d{3,}$', '', s)
    s = re.sub(r'[_-]\d{1,2}$', '', s)
    return s or path.stem.lower()


def build_leakage_reduced_yaml(dataset_root: str | Path, work_dir: str | Path, seed: int = 7, val_fraction: float = 0.12) -> dict:
    info = discover_yolo_dataset(dataset_root)
    tr_imgs = _images(info['train_images'])
    held_imgs = _images(info['heldout_images'])
    if len(tr_imgs) < 50 or len(held_imgs) < 10:
        raise RuntimeError(f'Unexpectedly small HeinSight dataset: train={len(tr_imgs)} heldout={len(held_imgs)}')

    groups: dict[str, list[Path]] = {}
    for p in tr_imgs:
        groups.setdefault(_group_id(p), []).append(p)
    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) >= 3:
        n_val_groups = max(1, int(round(len(keys) * val_fraction)))
        val_keys = set(keys[:n_val_groups])
        train_sub = [p for k in keys if k not in val_keys for p in groups[k]]
        val_sub = [p for k in keys if k in val_keys for p in groups[k]]
        split_mode = 'filename_group_heuristic'
    else:
        shuffled = tr_imgs[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_sub, train_sub = shuffled[:n_val], shuffled[n_val:]
        split_mode = 'image_random_fallback'

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    train_txt = work / 'train_images.txt'
    val_txt = work / 'val_images.txt'
    test_txt = work / 'test_images.txt'
    train_txt.write_text('\n'.join(str(p) for p in train_sub) + '\n', encoding='utf-8')
    val_txt.write_text('\n'.join(str(p) for p in val_sub) + '\n', encoding='utf-8')
    test_txt.write_text('\n'.join(str(p) for p in held_imgs) + '\n', encoding='utf-8')

    names = info.get('names')
    if not names:
        # HeinSight4 chemical phase class order from the public dataset.
        names = {0: 'Empty', 1: 'Residue', 2: 'Solid', 3: 'Homogeneous liquid', 4: 'Heterogeneous liquid'}

    data_yaml = work / 'heinsight4_nightly.yaml'
    data_yaml.write_text(yaml.safe_dump({
        'train': str(train_txt),
        'val': str(val_txt),
        'test': str(test_txt),
        'names': names,
    }, sort_keys=False), encoding='utf-8')

    audit = {
        'split_mode': split_mode,
        'n_official_train_images': len(tr_imgs),
        'n_train': len(train_sub),
        'n_internal_val': len(val_sub),
        'n_official_heldout_test': len(held_imgs),
        'n_filename_groups': len(groups),
        'yaml': str(data_yaml),
    }
    (work / 'split_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return {'yaml': data_yaml, 'audit': audit}


def _metrics_dict(metrics) -> dict:
    out = {}
    try:
        rd = getattr(metrics, 'results_dict', {}) or {}
        out.update({str(k): float(v) for k, v in rd.items() if isinstance(v, (int, float))})
    except Exception:
        pass
    for name, attr in [('map50_95', 'box.map'), ('map50', 'box.map50'), ('map75', 'box.map75')]:
        try:
            obj = metrics
            for part in attr.split('.'):
                obj = getattr(obj, part)
            out[name] = float(obj)
        except Exception:
            pass
    return out


def train_and_test_heinsight(
    dataset_root: str | Path,
    out_dir: str | Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    seed: int = 7,
    device: str = '0',
    pretrained: str = 'yolov8n.pt',
) -> dict:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError('ultralytics is required. Run: pip install -r requirements-public.txt') from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prep = build_leakage_reduced_yaml(dataset_root, out / 'dataset_split', seed=seed)
    print('\n=== HEINSIGHT4 VISUAL PHASE DETECTOR ===')
    print(json.dumps(prep['audit'], indent=2))

    model = YOLO(pretrained)
    train_result = model.train(
        data=str(prep['yaml']),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device=device,
        seed=int(seed),
        project=str(out),
        name='heinsight4_visual',
        exist_ok=True,
        amp=True,
        patience=12,
        cache=False,
        verbose=True,
    )

    best = out / 'heinsight4_visual' / 'weights' / 'best.pt'
    if not best.exists():
        # Ultralytics may normalize project path.
        candidates = list(out.rglob('best.pt'))
        if candidates:
            best = candidates[0]
    eval_model = YOLO(str(best if best.exists() else pretrained))
    test_metrics = eval_model.val(data=str(prep['yaml']), split='test', imgsz=int(imgsz), batch=int(batch), device=device, workers=int(workers), verbose=True)
    result = {
        'task': 'manual_phase_object_detection',
        'dataset': 'HeinSight4.0 chemical dataset v2',
        'split_audit': prep['audit'],
        'checkpoint': str(best) if best.exists() else None,
        'heldout_metrics': _metrics_dict(test_metrics),
        'note': 'The official held-out split is used only for final evaluation; an internal validation subset is carved from official train.',
    }
    (out / 'HEINSIGHT4_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
