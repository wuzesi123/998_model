from __future__ import annotations

"""CrystalCV V2 preparation.

Goals:
1) discover trajectory tables directly instead of depending on the legacy adapter;
2) merge raw/processed copies into one physical experiment;
3) infer/verify ambiguous time units conservatively and record the audit;
4) use one consistent observable target: future log(total crystal area).

No weak stage/progress/transition labels are created here.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tcpt.adapters.utils import TABLE_EXTS, find_time_col, normalized_name, read_table
from tcpt.forecast_data import ForecastExperiment, save_forecast_experiment


CRYSTAL_FEATURES = [
    "crystal_log_area",
    "crystal_log_count",
    "crystal_area_velocity",
    "crystal_activity",
    "time_delta_s",
    "time_log_elapsed",
    "count_available",
]
CRYSTAL_TARGETS = ["crystal_log_area"]

# Names observed in the public release/repository, e.g. A_230528_0.2mmh,
# FRC3_231013_0.2mmh. We keep the final match because some paths contain a
# parent/source run and then the concrete tracked run.
_PHYS_RE = re.compile(
    r"(?P<kind>A|FRC[1-9])[_\-](?P<date>\d{6})(?:[_\-](?P<rate>\d+(?:\.\d+)?mmh))?",
    re.I,
)
_DATE_RE = re.compile(r"(?<!\d)(?P<date>\d{6})(?!\d)")
_RATE_RE = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s*mm\s*[/_\-]?\s*h|(?P<rate2>\d+(?:\.\d+)?)mmh", re.I)

_SKIP_NAME_TOKENS = {
    "figure", "figures", "heatmap", "summary", "nucleation_frequency",
    "rsd", "accuracy", "acc_test", "manual_verification", "si_acc",
}


def _safe(a) -> np.ndarray:
    a = np.asarray(a, np.float64).reshape(-1)
    return np.where(np.isfinite(a), a, np.nan)


def _physical_id(text: str) -> str | None:
    clean = text.replace("02mmh", "0.2mmh")
    hits = list(_PHYS_RE.finditer(clean))
    if hits:
        m = hits[-1]
        kind = m.group("kind").upper()
        date = m.group("date")
        rate = (m.group("rate") or "unknown").lower()
        return f"{kind}_{date}_{rate}"

    # Conservative fallback: only use a standalone date when the path otherwise
    # looks like a CrystalCV experiment. This avoids treating figure/summary tables
    # as experiments.
    low = clean.lower()
    if not any(k in low.replace('\\','/') for k in ["crystal", "tracking", "bulk", "frc", "mapb", "cspb", "fapb", "pre_processed", "/ma/", "/cs/"]):
        return None
    dates = list(_DATE_RE.finditer(clean))
    if not dates:
        return None
    date = dates[-1].group("date")
    r = list(_RATE_RE.finditer(clean))
    rate = "unknown"
    if r:
        mm = r[-1]
        rate = f"{mm.group('rate') or mm.group('rate2')}mmh"
    return f"EXP_{date}_{rate}".lower()


def _material_from_path(text: str) -> str:
    s = text.lower().replace("\\", "/")
    if "fapbi3" in s or "fa_pbi3" in s:
        return "FAPbI3"
    if "cspbbr3" in s or re.search(r"(^|/)cs(/|$)", s):
        return "CsPbBr3"
    if "mapbbr3" in s or re.search(r"(^|/)ma(/|$)", s):
        return "MAPbBr3"
    return "unknown"


def _explicit_time_factor(name: str) -> tuple[float | None, str | None]:
    n = normalized_name(name)
    if any(tok in n for tok in ["millisecond", "msec"]) or ("time" in n and n.endswith("ms")):
        return 1e-3, "milliseconds"
    if any(tok in n for tok in ["second", "seconds", "timesec", "timeseconds"]) or n in {"sec", "secs", "s"} or ("time" in n and n.endswith("s")):
        return 1.0, "seconds"
    if any(tok in n for tok in ["minute", "minutes"]) or n in {"min", "mins"} or ("time" in n and n.endswith("min")):
        return 60.0, "minutes"
    if any(tok in n for tok in ["hour", "hours"]) or n in {"hr", "hrs", "h"} or ("time" in n and (n.endswith("h") or n.endswith("hr"))):
        return 3600.0, "hours"
    return None, None


def _time_candidate_score(raw: np.ndarray, factor: float, path_text: str, columns: Iterable[str]) -> float:
    x = np.asarray(raw, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return -1e9
    x = np.sort(np.unique(x))
    if len(x) < 3:
        return -1e9
    t = x * factor
    d = np.diff(t)
    d = d[d > 1e-12]
    if not len(d):
        return -1e9
    med_dt = float(np.median(d))
    span = float(t[-1] - t[0])

    # Static crystallization tracking is usually sampled on a seconds-to-minutes
    # scale and lasts minutes-to-days, not milliseconds. Use broad bounds so the
    # inference is conservative rather than tuned to one file.
    score = 0.0
    if 0.25 <= med_dt <= 120.0:
        score += 4.0
    elif 0.05 <= med_dt <= 600.0:
        score += 1.5
    else:
        score -= 4.0 * abs(math.log10(max(med_dt, 1e-12) / 5.0))

    if 5 * 60 <= span <= 7 * 24 * 3600:
        score += 3.0
    elif 60 <= span <= 30 * 24 * 3600:
        score += 1.0
    else:
        score -= 2.0

    low = path_text.lower()
    col_low = " ".join(map(str, columns)).lower()
    raw_d = np.diff(np.sort(np.unique(raw[np.isfinite(raw)])))
    raw_d = raw_d[raw_d > 1e-12]
    raw_med = float(np.median(raw_d)) if len(raw_d) else 0.0
    raw_span = float(np.nanmax(raw) - np.nanmin(raw))

    # Existing CrystalCV tracking_data tables in the release commonly have ~10 s
    # steps, so an ambiguous raw step already in the seconds range strongly favors s.
    if "tracking_data" in low and 0.5 <= raw_med <= 120:
        score += 3.0 if factor == 1.0 else -1.0

    # Critical repair for tables like the observed A_230326 file: Time is unlabeled,
    # values span ~40 with ~0.001 increments, while growth columns are in mm/h and the
    # experiment itself is specified in mm/h. That pattern is much more consistent
    # with hours than with 40 seconds.
    mmh_context = ("mmh" in low) or ("mm_h" in col_low) or ("mm/h" in col_low)
    if mmh_context and raw_med < 0.02 and 1.0 <= raw_span <= 200.0:
        if factor == 3600.0:
            score += 5.0
        elif factor == 60.0:
            score += 0.5
        else:
            score -= 5.0

    return float(score)


def _convert_time_to_seconds(series: pd.Series, name: str, path_text: str, columns: Iterable[str]) -> tuple[np.ndarray, dict]:
    # Timedelta-like/object timestamps are deterministic.
    if not pd.api.types.is_numeric_dtype(series):
        td = pd.to_timedelta(series, errors="coerce")
        if td.notna().sum() >= max(3, len(series) // 2):
            t = td.dt.total_seconds().to_numpy(np.float64)
            return t, {"unit": "timedelta", "factor_to_seconds": 1.0, "method": "timedelta", "confidence": "high"}

    raw = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    factor, unit = _explicit_time_factor(name)
    if factor is not None:
        return raw * factor, {"unit": unit, "factor_to_seconds": factor, "method": "column_name", "confidence": "high"}

    scores = {f: _time_candidate_score(raw, f, path_text, columns) for f in (1.0, 60.0, 3600.0)}
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_factor, best_score = ordered[0]
    margin = best_score - ordered[1][1]
    unit_name = {1.0: "seconds_inferred", 60.0: "minutes_inferred", 3600.0: "hours_inferred"}[best_factor]
    confidence = "high" if margin >= 4 else "medium" if margin >= 2 else "low"
    return raw * best_factor, {
        "unit": unit_name,
        "factor_to_seconds": best_factor,
        "method": "plausibility_audit",
        "confidence": confidence,
        "score_margin": float(margin),
        "candidate_scores": {str(k): float(v) for k, v in scores.items()},
    }


def _is_area_column(name: str) -> bool:
    n = normalized_name(name)
    if not any(tok in n for tok in ["area", "size"]):
        return False
    if any(tok in n for tok in ["aspect", "ratio", "growth", "rate", "velocity", "change", "delta", "error"]):
        return False
    return True


def _is_count_column(name: str) -> bool:
    n = normalized_name(name)
    return "count" in n or "nucleationcount" in n or n in {"ncrystals", "crystals"}


def _choose_time_column(df: pd.DataFrame) -> str | None:
    c = find_time_col(df)
    if c is not None:
        return c
    for col in df.columns:
        n = normalized_name(col)
        if n in {"frame", "frameid", "framenumber", "index"}:
            return col
    return None


def _collapse_table(df: pd.DataFrame, path: Path, min_rows: int) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if len(df) < min_rows:
        return None

    time_col = _choose_time_column(df)
    if time_col is None:
        return None

    t, time_audit = _convert_time_to_seconds(df[time_col], str(time_col), str(path), df.columns)
    good_t = np.isfinite(t)
    if good_t.sum() < min_rows:
        return None

    work = df.loc[good_t].copy()
    work["__time_s"] = t[good_t]

    # Reject clearly unusable/ambiguous time axes instead of silently creating a
    # benchmark with impossible milliseconds-long crystallizations.
    if time_audit.get("confidence") == "low":
        return None

    area_cols = [c for c in work.columns if c != "__time_s" and _is_area_column(str(c))]
    count_cols = [c for c in work.columns if c != "__time_s" and _is_count_column(str(c))]
    if not area_cols:
        return None

    # Long-form tracking may contain multiple rows per time point. In wide-form data
    # each row is already one time point and area columns correspond to crystals or a
    # total-area signal. Both cases are collapsed to one observable trajectory.
    area_num = work[area_cols].apply(pd.to_numeric, errors="coerce")

    # Prefer explicit sum/total area columns when available; otherwise sum independent
    # area channels. This mirrors the paper's use of tracked crystal areas while keeping
    # the target definition consistent across files.
    total_cols = [c for c in area_cols if any(tok in normalized_name(c) for tok in ["sumarea", "totalarea"])]
    if total_cols:
        area_series = area_num[total_cols].median(axis=1, skipna=True)
        area_source = [str(c) for c in total_cols]
    elif len(area_cols) == 1:
        area_series = area_num.iloc[:, 0]
        area_source = [str(area_cols[0])]
    else:
        # Avoid double-counting obvious duplicate representations of the same area.
        preferred = [c for c in area_cols if "mm2" in normalized_name(c)] or area_cols
        area_series = area_num[preferred].clip(lower=0).sum(axis=1, min_count=1)
        area_source = [str(c) for c in preferred]

    if count_cols:
        count_num = work[count_cols].apply(pd.to_numeric, errors="coerce")
        count_series = count_num.median(axis=1, skipna=True)
        count_available = True
        count_source = [str(c) for c in count_cols]
    elif len(area_cols) > 1 and not total_cols:
        # A wide table with one area channel per tracked crystal lets us derive the
        # number of currently visible crystals causally from non-zero area channels.
        preferred = [c for c in area_cols if "mm2" in normalized_name(c)] or area_cols
        count_series = area_num[preferred].gt(0).sum(axis=1).astype(float)
        count_available = True
        count_source = ["derived_nonzero_area_channels"]
    else:
        count_series = pd.Series(np.zeros(len(work), dtype=float), index=work.index)
        count_available = False
        count_source = []

    tmp = pd.DataFrame({
        "time_s": pd.to_numeric(work["__time_s"], errors="coerce"),
        "area": pd.to_numeric(area_series, errors="coerce"),
        "count": pd.to_numeric(count_series, errors="coerce"),
    })
    tmp = tmp[np.isfinite(tmp["time_s"])].copy()
    if len(tmp) < min_rows:
        return None

    # Aggregate duplicate timestamps. Area/count represent the state at the time point,
    # so median is safer than summing duplicate rows.
    tmp = tmp.groupby("time_s", as_index=False).median(numeric_only=True).sort_values("time_s")
    if len(tmp) < min_rows:
        return None

    t = tmp["time_s"].to_numpy(np.float64)
    area = tmp["area"].to_numpy(np.float64)
    count = tmp["count"].to_numpy(np.float64)

    finite_area = np.isfinite(area)
    if finite_area.sum() < min_rows:
        return None
    area = pd.Series(area).interpolate(limit_direction="both").fillna(0).to_numpy(np.float64)
    count = pd.Series(count).interpolate(limit_direction="both").fillna(0).to_numpy(np.float64)
    area = np.maximum(area, 0.0)
    count = np.maximum(count, 0.0)

    # Shift to zero elapsed time and defensively enforce strict monotonicity.
    t = t - t[0]
    d = np.diff(t)
    pos = d[d > 1e-9]
    if len(pos) < max(2, len(t) // 4):
        return None
    med_dt = float(np.median(pos))
    keep = np.r_[True, d > 1e-9]
    t, area, count = t[keep], area[keep], count[keep]
    if len(t) < min_rows:
        return None

    log_area = np.log1p(area)
    log_count = np.log1p(count)
    vel = np.zeros_like(log_area)
    if len(t) > 1:
        dt = np.diff(t)
        vel[1:] = np.divide(np.diff(log_area), dt, out=np.zeros(len(dt)), where=dt > 1e-9)
    activity = np.abs(vel)
    dt_feat = np.r_[med_dt, np.diff(t)]
    dt_feat = np.where(dt_feat > 1e-9, dt_feat, med_dt)
    log_elapsed = np.log1p(t)
    count_flag = np.full(len(t), 1.0 if count_available else 0.0, np.float64)

    x = np.column_stack([
        log_area,
        log_count,
        vel,
        activity,
        dt_feat,
        log_elapsed,
        count_flag,
    ]).astype(np.float32)

    audit = {
        "time_column": str(time_col),
        "time_audit": time_audit,
        "median_dt_sec": med_dt,
        "span_sec": float(t[-1] - t[0]),
        "area_source": area_source[:32],
        "count_source": count_source[:32],
        "count_available": bool(count_available),
        "n_rows_raw": int(len(df)),
        "n_samples": int(len(t)),
    }
    return x, t.astype(np.float32), audit


def _representation_priority(path: Path, audit: dict) -> float:
    s = str(path).lower().replace("\\", "/")
    score = 0.0
    # The authors describe postprocessing as removing merged/bad crystals and smoothing
    # growth data, so prefer those trajectories when they are valid.
    if "pre_processed" in s or "preprocessed" in s:
        score += 40
    if "all_tracked" in s:
        score += 25
    if "ma_processed" in s or "cs_processed" in s or "postprocess" in s or "processed" in s:
        score += 20
    if "tracking_data" in s:
        score += 12
    if "raw-images" in s:
        score += 4
    conf = audit.get("time_audit", {}).get("confidence")
    score += {"high": 12, "medium": 5, "low": -20}.get(conf, 0)
    score += min(10.0, math.log10(max(audit.get("n_samples", 1), 1)) * 3.0)
    if audit.get("count_available"):
        score += 3
    return float(score)


def _iter_tables(roots: Iterable[str | Path]):
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TABLE_EXTS:
                continue
            low = p.name.lower()
            if any(tok in low for tok in _SKIP_NAME_TOKENS):
                continue
            # Avoid re-reading exact same resolved path when roots overlap.
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            yield p


def prepare_crystalcv(
    raw_roots: str | Path | Iterable[str | Path],
    out_dir: str | Path,
    min_rows: int = 48,
    min_physical_experiments: int = 30,
) -> dict:
    roots = [raw_roots] if isinstance(raw_roots, (str, Path)) else list(raw_roots)
    roots = [Path(r).resolve() for r in roots if Path(r).exists()]
    if not roots:
        raise FileNotFoundError("No CrystalCV roots exist")

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.forecast.npz"):
        p.unlink()

    candidates: dict[str, list[dict]] = {}
    skipped: list[dict] = []
    scanned = 0
    accepted_tables = 0

    for path in _iter_tables(roots):
        scanned += 1
        pid = _physical_id(str(path))
        if pid is None:
            continue
        try:
            df = read_table(path)
            packed = _collapse_table(df, path, min_rows=min_rows)
            if packed is None:
                skipped.append({"path": str(path), "physical_experiment_id": pid, "reason": "not_a_valid_trajectory_or_ambiguous_time"})
                continue
            x, t, audit = packed
            accepted_tables += 1
            material = _material_from_path(str(path))
            candidates.setdefault(pid, []).append({
                "path": path,
                "x": x,
                "t": t,
                "audit": audit,
                "material": material,
                "priority": _representation_priority(path, audit),
            })
        except Exception as e:
            skipped.append({"path": str(path), "physical_experiment_id": pid, "reason": f"{type(e).__name__}: {e}"})

    experiments = []
    duplicate_groups = {}
    unit_counter = Counter()
    material_counter = Counter()

    for pid, items in sorted(candidates.items()):
        items.sort(key=lambda z: (z["priority"], len(z["t"])), reverse=True)
        best = items[0]
        duplicate_groups[pid] = [str(v["path"]) for v in items]
        material = best["material"]
        material_counter[material] += 1
        unit_counter[best["audit"]["time_audit"]["unit"]] += 1

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", pid)[:140]
        save_forecast_experiment(
            ForecastExperiment(
                experiment_id=pid,
                source_group=pid,
                family=material,
                x=best["x"],
                time=best["t"],
                feature_names=list(CRYSTAL_FEATURES),
                target_names=list(CRYSTAL_TARGETS),
            ),
            out / f"{safe}.forecast.npz",
        )
        experiments.append({
            "physical_experiment_id": pid,
            "material": material,
            "selected_source": str(best["path"]),
            "representation_priority": float(best["priority"]),
            "n_alternate_representations_dropped": len(items) - 1,
            **best["audit"],
        })

    spans = np.asarray([e["span_sec"] for e in experiments], np.float64)
    dts = np.asarray([e["median_dt_sec"] for e in experiments], np.float64)
    manifest = {
        "version": "ChemProcessRAG_CrystalCV_V2",
        "dataset": "CrystalCV",
        "task": "future_observable_forecasting",
        "scientific_rule": "Observed tracking trajectories only; no adapter-derived stage/progress/transition labels.",
        "primary_target": "future log(total visible crystal area)",
        "source_roots": [str(r) for r in roots],
        "n_tables_scanned": int(scanned),
        "n_valid_trajectory_tables": int(accepted_tables),
        "n_physical_experiments": int(len(experiments)),
        "n_duplicate_representations_dropped": int(sum(max(0, len(v) - 1) for v in duplicate_groups.values())),
        "reported_case_study_experiments_in_paper": 129,
        "coverage_vs_reported_129": float(len(experiments) / 129.0),
        "feature_names": CRYSTAL_FEATURES,
        "target_names": CRYSTAL_TARGETS,
        "materials": dict(material_counter),
        "time_unit_audit": dict(unit_counter),
        "duration_sec": {
            "min": float(np.min(spans)) if len(spans) else None,
            "median": float(np.median(spans)) if len(spans) else None,
            "max": float(np.max(spans)) if len(spans) else None,
        },
        "median_dt_sec": {
            "min": float(np.min(dts)) if len(dts) else None,
            "median": float(np.median(dts)) if len(dts) else None,
            "max": float(np.max(dts)) if len(dts) else None,
        },
        "experiments": experiments,
        "duplicate_groups": {k: v for k, v in duplicate_groups.items() if len(v) > 1},
        "skipped": skipped[:1000],
    }
    manifest_path = out / "CRYSTALCV_PREPARE_MANIFEST_V2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== CRYSTALCV V2 PREPARE AUDIT ===")
    print("roots:", *[str(r) for r in roots], sep="\n  - ")
    print(f"tables scanned={scanned} | valid trajectories={accepted_tables}")
    print(f"physical experiments={len(experiments)} | duplicate reps dropped={manifest['n_duplicate_representations_dropped']}")
    print("materials:", dict(material_counter))
    print("time units:", dict(unit_counter))
    if len(spans):
        print(f"duration sec min/median/max = {np.min(spans):.1f} / {np.median(spans):.1f} / {np.max(spans):.1f}")
        print(f"median dt sec min/median/max = {np.min(dts):.4g} / {np.median(dts):.4g} / {np.max(dts):.4g}")

    if len(experiments) < int(min_physical_experiments):
        raise RuntimeError(
            f"CrystalCV V2 prepared only {len(experiments)} physical experiments. "
            f"The paper reports 129 experiments in the bulk case study; refusing to train a misleading benchmark. "
            f"Inspect {manifest_path}."
        )
    return manifest
