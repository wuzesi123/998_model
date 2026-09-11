# CrystalCV V2 correction

This overwrite fixes the validity problems revealed by the first overnight run.

## What changed

1. **Full trajectory discovery**
   - The runner now uses both the CrystalCV source-data record (Zenodo 19140131) and the archived software/repository record (Zenodo 20655010).
   - The latter is important because the public CrystalCV repository documents author-provided `Pre_Processed` tracking data.
   - Tables are scanned directly; the legacy weak-label adapter is not used.

2. **Physical experiment deduplication**
   - raw / tracking / processed / `all_tracked` representations are grouped by canonical physical experiment ID.
   - exactly one validated representation is selected per physical run.
   - postprocessed/Pre_Processed trajectories are preferred when valid.

3. **Time-unit audit**
   - explicit sec/min/hour column names are honored.
   - ambiguous numeric `Time` columns are scored under seconds/minutes/hours.
   - impossible millisecond-like crystallization timelines are rejected or corrected.
   - the previously observed pattern (`~0..40`, tiny dt, mm/h context) resolves to hours.
   - every selected experiment records unit, conversion factor, confidence, median dt and duration.

4. **No misleading tiny benchmark**
   - preparation refuses to train if fewer than 30 physical experiments are recovered by default.
   - the manifest is still written so discovery problems can be inspected.

5. **One consistent primary target**
   - ranking target = future `log(total visible crystal area)`.
   - count is retained as an input/diagnostic when available.
   - growth rate is no longer mixed with area/count into one NMAE.

6. **Relative horizons**
   - 0.5%, 2%, 5%, 10%, 20% are computed from *each experiment's own duration*.
   - short experiments therefore do not disappear from long-horizon evaluation.

7. **Experiment-balanced training/evaluation**
   - train/eval windows are capped per physical experiment.
   - primary metric = macro experiment NMAE: calculate each held-out experiment first, then average.
   - pooled frame NMAE is retained only as a secondary diagnostic.
   - report also gives TCPT experiment win-rate versus LastValue.

8. **Context runner bug**
   - missing `import random` was fixed. This does not create context labels; if explicit image->context mappings are absent, the context benchmark still correctly remains audit-only/skipped.

## Fastest rerun

If HeinSight is already finished, run only:

```bash
python RUN_CRYSTALCV_V2.py
```

The previous 6.7 GB source download is reused if its manifest/files are still present. The script additionally downloads the archived CrystalCV software/repository record once.

Outputs:

```text
results/public_nightly/crystalcv/prepared_v2/CRYSTALCV_PREPARE_MANIFEST_V2.json
results/public_nightly/crystalcv/benchmark_v2/CRYSTALCV_TEMPORAL_RESULT_V2.json
results/public_nightly/crystalcv/benchmark_v2/CRYSTALCV_TEMPORAL_RESULT_V2.csv
```

Before trusting training results, inspect the prepare audit. You want substantially more than the previous 13 physical experiments, sensible durations and sampling intervals, and no suspicious 40-second multi-hour crystallizations.

## Full overnight rerun

You can also run:

```bash
python RUN_PUBLIC_OVERNIGHT.py
```

It uses the new CrystalCV V2 paths/stage names, so the old V1 CrystalCV result is not silently reused.
