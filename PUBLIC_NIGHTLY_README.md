# ChemProcessRAG Public Nightly V1

This runner is designed for one unattended overnight run on an RTX 3080 Ti-class machine.

## One command

```bash
python RUN_PUBLIC_OVERNIGHT.py
```

The runner is resumable. Re-running the same command reuses downloaded data and completed result JSON files. Use `--force` only when you intentionally want to retrain.

## What it downloads and evaluates

### 1. HeinSight4.0 chemical dataset v2 — visual layer

- Zenodo record: `15605098`
- Manual bounding-box labels for five physical phase classes.
- The runner downloads only the chemical dataset, not the multi-GB vessel dataset/model.
- It carves an internal validation subset from the official training set and preserves the official held-out split for final evaluation.
- It trains a small Ultralytics YOLO model from pretrained weights.
- Output: held-out mAP50, mAP50-95 and checkpoint.

### 2. CrystalCV — temporal layer

- Zenodo source record: `19140131`
- The adapter uses observed crystal tracking measurements, not the old weak 5-stage labels.
- Before splitting, filenames such as `all_tracked_A_230528_0.2mmh` and `MA_processed_all_A_230528_0.2mmh` are canonicalized to one physical experiment ID and duplicate representations are dropped.
- Forecasting uses five horizons: 0.5%, 2%, 5%, 10% and 20% of the **training-set** median experiment duration.
- Models predict the **residual change** from the current observation.
- Baselines/models:
  - LastValue
  - LinearTrend
  - TrainMean
  - RidgeResidual
  - MLPResidual
  - GRUResidual
  - TCPTResidual
- Main score: NMAE (lower is better) and skill vs LastValue (positive is better).

### 3. 2026 context-aware chemical state dataset — context/RAG external audit

- Zenodo record: `17436705`
- The runner first looks for a machine-readable image-to-context mapping.
- It **does not** manufacture context from class names or directory names.
- If enough explicit context is available, it runs:
  - VisualOnly
  - VisualPlusContext
  - VisualPlusContextPlusRAG
- If the public archive does not expose enough explicit per-image context, the quantitative fusion benchmark is marked `SKIPPED`, with the reason written to JSON. This is intentional, not an error.
- RAG uses the project's causal-filtered knowledge index. Ground-truth visual labels are never included in RAG queries.

## Important outputs

After the run, open:

```text
results/public_nightly/PUBLIC_NIGHTLY_SUMMARY.json
results/public_nightly/PUBLIC_NIGHTLY_SUMMARY.csv
results/public_nightly/RUN_PUBLIC_OVERNIGHT.log
```

Detailed outputs:

```text
results/public_nightly/heinsight4/HEINSIGHT4_RESULT.json
results/public_nightly/crystalcv/prepared/CRYSTALCV_PREPARE_MANIFEST.json
results/public_nightly/crystalcv/benchmark/CRYSTALCV_TEMPORAL_RESULT.json
results/public_nightly/crystalcv/benchmark/CRYSTALCV_TEMPORAL_RESULT.csv
results/public_nightly/context_aware/CONTEXT_RAG_AUDIT.json
results/public_nightly/context_aware/CONTEXT_FUSION_RESULT.json
```

## RTX 3080 Ti defaults

The public runner defaults to:

- YOLO: 640 px, batch 24, 60 epochs, AMP.
- CrystalCV: batch 512, FP16 autocast on CUDA, TCPT `d_model=192`, 3 layers, 6 heads, up to 30 epochs with early stopping.
- PyTorch TF32 paths are enabled where supported.

You can shorten the run:

```bash
python RUN_PUBLIC_OVERNIGHT.py --yolo-epochs 20 --tcpt-epochs 12
```

## Scientific interpretation

A successful run does **not** automatically prove a universal chemical reasoning model.

- HeinSight4 tests manual visual phase detection.
- CrystalCV tests temporal forecasting on crystallization measurements.
- The context/RAG stage tests multimodal context only when the archive supports a leakage-safe context mapping.

The point of this runner is to establish clean, reproducible public-data evidence before training a larger unified model.
