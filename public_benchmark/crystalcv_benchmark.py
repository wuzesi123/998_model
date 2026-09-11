from __future__ import annotations

"""CrystalCV V2 benchmark.

Key validity changes from V1:
- relative horizon = fraction of each experiment's own duration;
- balanced sampling caps each experiment so long runs cannot dominate training;
- primary metric is macro-per-experiment NMAE, not pooled-frame NMAE;
- primary target is one consistent observable: log(total visible crystal area).
"""

import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tcpt.forecast_data import (
    load_forecast_experiment,
    list_forecast_experiments,
    validate_schema,
    _sample_recent_indices,
    _sample_log_history_indices,
    _sample_global_indices,
)
from tcpt.forecast_models import MLPForecaster, GRUForecaster, TCPTForecaster


HORIZONS = [
    ("very_short_0p5pct", 0.005),
    ("short_2pct", 0.02),
    ("medium_5pct", 0.05),
    ("long_10pct", 0.10),
    ("very_long_20pct", 0.20),
]


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _family_stratified_split(paths: list[Path], seed: int = 7, ratios=(0.70, 0.15, 0.15)):
    by_family: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        e = load_forecast_experiment(p)
        by_family[e.family or "unknown"].append(p)

    rng = random.Random(seed)
    train, val, test = [], [], []
    split_families = {}
    for fam, items in sorted(by_family.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n >= 7:
            ntr = max(1, int(round(n * ratios[0])))
            nv = max(1, int(round(n * ratios[1])))
            if ntr + nv >= n:
                ntr, nv = n - 2, 1
            tr, va, te = items[:ntr], items[ntr:ntr + nv], items[ntr + nv:]
        elif n >= 3:
            # Keep all families represented where possible.
            tr, va, te = items[:-2], items[-2:-1], items[-1:]
        else:
            # Tiny families cannot be stratified safely; keep them in train rather than
            # allowing one rare family to become the entire test set.
            tr, va, te = items, [], []
        train += tr; val += va; test += te
        split_families[fam] = {
            "total": n,
            "train": [load_forecast_experiment(p).experiment_id for p in tr],
            "val": [load_forecast_experiment(p).experiment_id for p in va],
            "test": [load_forecast_experiment(p).experiment_id for p in te],
        }

    # If small-family handling left val/test too small, top them up deterministically
    # from train while preserving experiment-level separation.
    rng.shuffle(train)
    min_holdout = max(3, int(round(len(paths) * 0.10)))
    while len(val) < min_holdout and len(train) > min_holdout * 2:
        val.append(train.pop())
    while len(test) < min_holdout and len(train) > min_holdout * 2:
        test.append(train.pop())

    info = {
        "mode": "physical_experiment_family_stratified",
        "seed": seed,
        "n_physical_experiments": len(paths),
        "train_groups": [load_forecast_experiment(p).experiment_id for p in train],
        "val_groups": [load_forecast_experiment(p).experiment_id for p in val],
        "test_groups": [load_forecast_experiment(p).experiment_id for p in test],
        "families": split_families,
    }
    return sorted(train), sorted(val), sorted(test), info


def _balanced_input_normalization(paths: list[Path], max_points_per_experiment: int = 1024):
    chunks = []
    for p in paths:
        e = load_forecast_experiment(p)
        n = len(e.x)
        ids = np.linspace(0, n - 1, min(n, max_points_per_experiment)).round().astype(int)
        chunks.append(e.x[ids].astype(np.float64))
    x = np.concatenate(chunks, axis=0)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _balanced_target_scale(paths: list[Path], target_indices: list[int], max_points_per_experiment: int = 1024):
    chunks = []
    for p in paths:
        e = load_forecast_experiment(p)
        n = len(e.x)
        ids = np.linspace(0, n - 1, min(n, max_points_per_experiment)).round().astype(int)
        chunks.append(e.x[ids][:, target_indices].astype(np.float64))
    y = np.concatenate(chunks, axis=0)
    p05 = np.nanpercentile(y, 5, axis=0)
    p95 = np.nanpercentile(y, 95, axis=0)
    rr = p95 - p05
    std = np.nanstd(y, axis=0)
    rr = np.where(np.isfinite(rr) & (rr > 1e-6), rr, np.where(std > 1e-6, std, 1.0))
    return {"robust_range": rr.astype(np.float32), "p05": p05.tolist(), "p95": p95.tolist()}


def _evenly_cap_pairs(pairs: list[tuple[int, int]], cap: int | None):
    if cap is None or len(pairs) <= cap:
        return pairs
    ids = np.linspace(0, len(pairs) - 1, cap).round().astype(int)
    return [pairs[i] for i in ids]


class RelativeResidualForecastDataset(Dataset):
    def __init__(
        self,
        paths,
        horizon_fraction,
        input_mean,
        input_std,
        delta_mean,
        delta_std,
        short_len=8,
        medium_len=12,
        global_len=16,
        max_samples_per_experiment=512,
    ):
        self.experiments = [load_forecast_experiment(p) for p in paths]
        self.feature_names, self.target_names = validate_schema(paths)
        self.target_indices = [self.feature_names.index(n) for n in self.target_names]
        self.horizon_fraction = float(horizon_fraction)
        self.input_mean = np.asarray(input_mean, np.float32)
        self.input_std = np.asarray(input_std, np.float32)
        self.delta_mean = np.asarray(delta_mean, np.float32)
        self.delta_std = np.asarray(delta_std, np.float32)
        self.short_len = int(short_len); self.medium_len = int(medium_len); self.global_len = int(global_len)
        self.index = []
        self.horizon_sec_by_experiment = {}

        for ei, e in enumerate(self.experiments):
            t = np.asarray(e.time, np.float64)
            span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            if span <= 0:
                continue
            hsec = max(1e-9, span * self.horizon_fraction)
            self.horizon_sec_by_experiment[e.experiment_id] = hsec
            pairs = []
            for i in range(len(t) - 1):
                j = int(np.searchsorted(t, float(t[i]) + hsec, side="left"))
                if j < len(t) and j > i:
                    pairs.append((i, j))
            pairs = _evenly_cap_pairs(pairs, max_samples_per_experiment)
            self.index.extend((ei, i, j) for i, j in pairs)
        if not self.index:
            raise ValueError(f"No forecast samples at fraction={horizon_fraction}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ei, i, j = self.index[idx]
        e = self.experiments[ei]
        sids = _sample_recent_indices(i, self.short_len)
        mids = _sample_log_history_indices(e.time, i, self.medium_len)
        gids = _sample_global_indices(e.time, i, self.global_len)
        norm = lambda x: (x - self.input_mean[None, :]) / self.input_std[None, :]
        span = max(float(e.time[-1] - e.time[0]), 1e-9)
        rel_time = (e.time - e.time[0]) / span
        cur = e.x[i, self.target_indices].astype(np.float32)
        future = e.x[j, self.target_indices].astype(np.float32)
        delta = future - cur
        dn = (delta - self.delta_mean) / self.delta_std
        return {
            "short_x": torch.from_numpy(norm(e.x[sids]).astype(np.float32)),
            "medium_x": torch.from_numpy(norm(e.x[mids]).astype(np.float32)),
            "global_x": torch.from_numpy(norm(e.x[gids]).astype(np.float32)),
            "short_time": torch.from_numpy(rel_time[sids].astype(np.float32)),
            "medium_time": torch.from_numpy(rel_time[mids].astype(np.float32)),
            "global_time": torch.from_numpy(rel_time[gids].astype(np.float32)),
            "delta_target": torch.from_numpy(dn.astype(np.float32)),
            "current_raw": torch.from_numpy(cur),
            "future_raw": torch.from_numpy(future),
            "experiment_index": torch.tensor(ei, dtype=torch.long),
        }


def _delta_scale(paths, horizon_fraction, target_indices, max_pairs_per_experiment=512):
    vals = []
    for p in paths:
        e = load_forecast_experiment(p)
        t = np.asarray(e.time, np.float64)
        span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if span <= 0:
            continue
        hsec = span * float(horizon_fraction)
        pairs = []
        for i in range(len(t) - 1):
            j = int(np.searchsorted(t, t[i] + hsec, side="left"))
            if j < len(t) and j > i:
                pairs.append((i, j))
        for i, j in _evenly_cap_pairs(pairs, max_pairs_per_experiment):
            vals.append(e.x[j, target_indices].astype(np.float64) - e.x[i, target_indices].astype(np.float64))
    d = np.asarray(vals, np.float64)
    mean = np.nanmean(d, axis=0)
    std = np.nanstd(d, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _metrics(y, p, groups, robust_range, target_names):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64); groups = np.asarray(groups, np.int64)
    rr = np.maximum(np.asarray(robust_range, np.float64), 1e-8)
    err = p - y
    mae = np.mean(np.abs(err), axis=0)
    nmae = mae / rr
    per_target = {}
    for j, name in enumerate(target_names):
        denom = np.sum((y[:, j] - np.mean(y[:, j])) ** 2)
        r2 = None if denom <= 1e-12 else float(1 - np.sum(err[:, j] ** 2) / denom)
        per_target[name] = {"mae": float(mae[j]), "nmae": float(nmae[j]), "r2": r2}

    per_exp = {}
    exp_scores = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        g_mae = np.mean(np.abs(err[m]), axis=0)
        score = float(np.mean(g_mae / rr))
        per_exp[str(g)] = score
        exp_scores.append(score)
    return {
        "pooled_nmae": float(np.mean(nmae)),
        "macro_experiment_nmae": float(np.mean(exp_scores)),
        "median_experiment_nmae": float(np.median(exp_scores)),
        "n_experiments": int(len(exp_scores)),
        "per_target": per_target,
        "per_experiment_index_nmae": per_exp,
    }


def _loader_arrays(loader):
    ys, cs, gs = [], [], []
    for b in loader:
        ys.append(b["future_raw"].numpy()); cs.append(b["current_raw"].numpy()); gs.append(b["experiment_index"].numpy())
    return np.concatenate(ys), np.concatenate(cs), np.concatenate(gs)


def _linear_trend_predictions(ds: RelativeResidualForecastDataset):
    ys, ps, gs = [], [], []
    for ei, i, j in ds.index:
        e = ds.experiments[ei]
        ids = _sample_recent_indices(i, min(ds.short_len, i + 1))
        t = np.asarray(e.time[ids], np.float64)
        target = np.asarray(e.x[ids][:, ds.target_indices], np.float64)
        dt = float(e.time[j] - e.time[i])
        pred = e.x[i, ds.target_indices].astype(np.float64).copy()
        if len(np.unique(t)) >= 2:
            tt = t - t.mean(); denom = float(np.sum(tt ** 2))
            if denom > 1e-12:
                slopes = np.sum(tt[:, None] * (target - target.mean(0, keepdims=True)), axis=0) / denom
                pred = pred + slopes * dt
        ys.append(e.x[j, ds.target_indices]); ps.append(pred); gs.append(ei)
    return np.asarray(ys, np.float32), np.asarray(ps, np.float32), np.asarray(gs, np.int64)


def _collate_device(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def _predict_model(model, loader, device, delta_mean, delta_std):
    model.eval(); ys = []; ps = []; gs = []
    with torch.no_grad():
        for b in loader:
            bd = _collate_device(b, device)
            pred_n = model(
                short_x=bd["short_x"], medium_x=bd["medium_x"], global_x=bd["global_x"],
                short_time=bd["short_time"], medium_time=bd["medium_time"], global_time=bd["global_time"],
            )
            delta = pred_n.detach().cpu().numpy() * delta_std[None, :] + delta_mean[None, :]
            ps.append(b["current_raw"].numpy() + delta)
            ys.append(b["future_raw"].numpy())
            gs.append(b["experiment_index"].numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)


def _train_model(model, train_loader, val_loader, device, delta_mean, delta_std, robust_range, target_names, epochs, lr, patience, save_path=None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = float("inf"); best = None; best_epoch = 0; stale = 0; hist = []
    for ep in range(1, int(epochs) + 1):
        model.train(); losses = []; t0 = time.time()
        for b in train_loader:
            b = _collate_device(b, device); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(
                    short_x=b["short_x"], medium_x=b["medium_x"], global_x=b["global_x"],
                    short_time=b["short_time"], medium_time=b["medium_time"], global_time=b["global_time"],
                )
                loss = loss_fn(pred, b["delta_target"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        vy, vp, vg = _predict_model(model, val_loader, device, delta_mean, delta_std)
        score = _metrics(vy, vp, vg, robust_range, target_names)["macro_experiment_nmae"]
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_macro_experiment_nmae": score, "sec": time.time() - t0})
        print(f"    ep={ep:02d} loss={np.mean(losses):.4f} valMacroNMAE={score:.4f}")
        if score < best_score - 1e-5:
            best_score = score; best_epoch = ep; stale = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": best, "best_val_macro_nmae": best_score, "best_epoch": best_epoch}, save_path)
        else:
            stale += 1
            if stale >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return {"best_val_macro_experiment_nmae": best_score, "best_epoch": best_epoch, "history": hist}


def _fit_ridge(train_loader, input_dim, short_len, output_dim, lam=1e-2):
    d = input_dim * short_len + 1
    xtx = np.zeros((d, d)); xty = np.zeros((d, output_dim))
    for b in train_loader:
        x = b["short_x"].numpy().reshape(len(b["short_x"]), -1).astype(np.float64)
        y = b["delta_target"].numpy().astype(np.float64)
        xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        xtx += xb.T @ xb; xty += xb.T @ y
    reg = np.eye(d) * lam; reg[-1, -1] = 0
    return np.linalg.solve(xtx + reg, xty)


def _ridge_predict(loader, w, delta_mean, delta_std):
    ys = []; ps = []; gs = []
    for b in loader:
        x = b["short_x"].numpy().reshape(len(b["short_x"]), -1).astype(np.float64)
        xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        dn = xb @ w; delta = dn * delta_std[None, :] + delta_mean[None, :]
        ps.append(b["current_raw"].numpy() + delta); ys.append(b["future_raw"].numpy()); gs.append(b["experiment_index"].numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)


def _add_skill(methods: dict):
    lv = methods["LastValue"]["macro_experiment_nmae"]
    lv_exp = methods["LastValue"]["per_experiment_index_nmae"]
    for name, m in methods.items():
        x = m["macro_experiment_nmae"]
        m["skill_vs_last_value"] = float(1 - x / lv) if lv > 1e-12 else None
        if name == "LastValue":
            m["experiment_win_rate_vs_last_value"] = 0.0
            continue
        wins = []
        for g, score in m["per_experiment_index_nmae"].items():
            if g in lv_exp:
                wins.append(float(score < lv_exp[g]))
        m["experiment_win_rate_vs_last_value"] = float(np.mean(wins)) if wins else None


def run_crystalcv_benchmark(
    prepared_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 30,
    batch: int = 512,
    workers: int = 4,
    seed: int = 7,
    d_model: int = 192,
    layers: int = 3,
    heads: int = 6,
    max_train_samples_per_experiment: int = 512,
    max_eval_samples_per_experiment: int = 1024,
) -> dict:
    seed_all(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = list_forecast_experiments(Path(prepared_dir))
    if len(paths) < 20:
        raise RuntimeError(f"Need >=20 physical CrystalCV experiments for V2 benchmark, found {len(paths)}")
    fn, tn = validate_schema(paths)
    if tn != ["crystal_log_area"]:
        raise RuntimeError(f"CrystalCV V2 expects primary target ['crystal_log_area']; found {tn}")
    target_idx = [fn.index(n) for n in tn]
    tr, va, te, split = _family_stratified_split(paths, seed=seed)
    if min(len(va), len(te)) < 3:
        raise RuntimeError(f"Holdout too small after split: train/val/test={len(tr)}/{len(va)}/{len(te)}")

    inp_mean, inp_std = _balanced_input_normalization(tr)
    target_scale = _balanced_target_scale(tr, target_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    report = {
        "version": "ChemProcessRAG_CrystalCV_Benchmark_V2",
        "dataset": "CrystalCV",
        "task": "leakage_free_relative_horizon_residual_forecasting",
        "primary_target": "future log(total visible crystal area)",
        "primary_metric": "macro_experiment_nmae",
        "split": split,
        "feature_names": fn,
        "target_names": tn,
        "device": str(device),
        "sampling_policy": {
            "relative_horizons": [f for _, f in HORIZONS],
            "max_train_samples_per_experiment": max_train_samples_per_experiment,
            "max_eval_samples_per_experiment": max_eval_samples_per_experiment,
            "time_encoding": "relative_elapsed_fraction_0_to_1",
        },
        "results": {},
    }
    csv_rows = []
    print("\n=== CRYSTALCV V2 TEMPORAL BENCHMARK ===")
    print(f"physical experiments train/val/test = {len(tr)}/{len(va)}/{len(te)} | device={device}")

    for hname, frac in HORIZONS:
        print(f"\n--- {hname}: {frac*100:.1f}% of each experiment duration ---")
        dm, ds = _delta_scale(tr, frac, target_idx, max_pairs_per_experiment=max_train_samples_per_experiment)
        common = dict(
            horizon_fraction=frac,
            input_mean=inp_mean, input_std=inp_std, delta_mean=dm, delta_std=ds,
            short_len=8, medium_len=12, global_len=16,
        )
        train_ds = RelativeResidualForecastDataset(tr, **common, max_samples_per_experiment=max_train_samples_per_experiment)
        val_ds = RelativeResidualForecastDataset(va, **common, max_samples_per_experiment=max_eval_samples_per_experiment)
        test_ds = RelativeResidualForecastDataset(te, **common, max_samples_per_experiment=max_eval_samples_per_experiment)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
        train_order = DataLoader(train_ds, batch_size=batch, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

        y, cur, g = _loader_arrays(test_loader)
        methods = {"LastValue": _metrics(y, cur, g, target_scale["robust_range"], tn)}

        # TrainMean on the absolute target, balanced across train experiments.
        train_target_means = []
        for p in tr:
            e = load_forecast_experiment(p)
            train_target_means.append(np.mean(e.x[:, target_idx], axis=0))
        future_mean = np.mean(train_target_means, axis=0)
        methods["TrainMean"] = _metrics(y, np.broadcast_to(future_mean, y.shape), g, target_scale["robust_range"], tn)

        yl, pl, gl = _linear_trend_predictions(test_ds)
        methods["LinearTrend"] = _metrics(yl, pl, gl, target_scale["robust_range"], tn)
        w = _fit_ridge(train_order, len(fn), 8, len(tn))
        yr, pr, gr = _ridge_predict(test_loader, w, dm, ds)
        methods["RidgeResidual"] = _metrics(yr, pr, gr, target_scale["robust_range"], tn)

        specs = [
            ("MLPResidual", MLPForecaster(len(fn), 8, len(tn), hidden=256), 1e-3, min(epochs, 20), 5),
            ("GRUResidual", GRUForecaster(len(fn), len(tn), hidden=192, layers=2), 6e-4, min(epochs, 24), 5),
            ("TCPTResidual", TCPTForecaster(len(fn), len(tn), d_model=d_model, nhead=heads, layers=layers, ff=d_model * 4, dropout=.1), 3e-4, epochs, 7),
        ]
        train_meta = {}
        for name, model, lr, eps, pat in specs:
            print("  training", name)
            ck = out / "checkpoints" / f"{hname}_{name}.pt"
            tm = _train_model(model, train_loader, val_loader, device, dm, ds, target_scale["robust_range"], tn, eps, lr, pat, ck)
            yt, pt, gt = _predict_model(model, test_loader, device, dm, ds)
            methods[name] = _metrics(yt, pt, gt, target_scale["robust_range"], tn)
            train_meta[name] = tm
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _add_skill(methods)
        ranking = sorted(methods, key=lambda n: methods[n]["macro_experiment_nmae"])
        print("  ranking:", " | ".join(f"{n}={methods[n]['macro_experiment_nmae']:.4f}" for n in ranking))

        test_h = list(test_ds.horizon_sec_by_experiment.values())
        block = {
            "horizon_fraction": frac,
            "horizon_sec_test": {
                "min": float(np.min(test_h)), "median": float(np.median(test_h)), "max": float(np.max(test_h)),
            },
            "n_train_samples": len(train_ds), "n_val_samples": len(val_ds), "n_test_samples": len(test_ds),
            "n_train_experiments": len(train_ds.experiments), "n_val_experiments": len(val_ds.experiments), "n_test_experiments": len(test_ds.experiments),
            "models": methods, "ranking": ranking, "training": train_meta,
        }
        report["results"][hname] = block
        for name, m in methods.items():
            csv_rows.append({
                "horizon": hname,
                "horizon_fraction": frac,
                "model": name,
                "macro_experiment_nmae": m["macro_experiment_nmae"],
                "pooled_nmae": m["pooled_nmae"],
                "skill_vs_last_value": m.get("skill_vs_last_value"),
                "experiment_win_rate_vs_last_value": m.get("experiment_win_rate_vs_last_value"),
            })

    (out / "CRYSTALCV_TEMPORAL_RESULT_V2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "CRYSTALCV_TEMPORAL_RESULT_V2.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["horizon", "horizon_fraction", "model", "macro_experiment_nmae", "pooled_nmae", "skill_vs_last_value", "experiment_win_rate_vs_last_value"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(csv_rows)
    return report
