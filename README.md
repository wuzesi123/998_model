# ChemProcessRAG V1

A fresh project for **causal chemical-process understanding from video**.

It implements:

- universal visual process evidence from reaction video;
- optional short/medium/global TCPT temporal encoder;
- `ReactionContext` / `context_builder` for reactants, reagents, solvents, catalysts, conditions and procedure;
- `ORDRAG` for local Open Reaction Database files plus a lightweight local chemistry knowledge index;
- a causal RAG policy that excludes ORD reaction outcomes by default;
- `semantic_reasoner` that keeps visual facts separate from chemical hypotheses;
- CLI, local web UI, demo, ORD download/index scripts, and optional TCPT training.

## 1. Fastest possible test

Python 3.10+ recommended for the core project.

```bash
pip install -r requirements.txt
python RUN_DEMO.py
```

This generates `outputs/demo_reaction.avi`, analyzes it, retrieves chemistry knowledge, and writes `outputs/demo_result.json`.

## 2. Analyze a real reaction video

Copy/edit `examples/reaction_context.json`, then:

```bash
python RUN_ANALYZE.py --video your_video.mp4 --context examples/reaction_context.json
```

No neural checkpoint is required. Without one, the program deliberately uses deterministic visual evidence rather than random TCPT predictions.

## 3. Local web interface

```bash
pip install -r requirements-web.txt
streamlit run RUN_WEB.py
```

## 4. Add Open Reaction Database data

Official ORD uses the `ord-schema` package and published `ord_dataset-*` files. Install:

```bash
pip install -r requirements-ord.txt
```

Download a known published dataset ID:

```bash
python scripts/FETCH_ORD_DATASET.py ord_dataset-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python scripts/INDEX_ORD.py
```

You can also place official `.parquet`, `.pb.gz`, `.pb`, `.pbtxt`, or protobuf JSON files in `data/ord/`.

The current ORD schema has structured reaction inputs, roles, setup and conditions. This project indexes only causal fields by default and deliberately excludes outcome/yield/product result fields from RAG inference.

## 5. Optional trained TCPT

Install PyTorch:

```bash
pip install -r requirements-tcpt.txt
```

Training NPZ format:

- `features`: `[T,D]`
- `timestamps`: `[T]` seconds
- `labels`: `[T]` integer observable-state labels

Then:

```bash
python TRAIN_TCPT.py --glob "data/training/*.npz" --epochs 20
python RUN_ANALYZE.py --video your_video.mp4 --context examples/reaction_context.json --checkpoint checkpoints/tcpt_observable.pt
```

## ReactionContext example

```json
{
  "experiment_id": "run_001",
  "participants": [
    {"name": "compound A", "role": "reactant", "smiles": "..."},
    {"name": "solvent B", "role": "solvent"},
    {"name": "catalyst C", "role": "catalyst"}
  ],
  "conditions": {"temperature_c": 40, "stirring_rpm": 500},
  "procedure": {"experiment_type": "crystallization", "current_step": "cooling"}
}
```

## Important scientific boundary

`solid_proxy`, `turbidity_proxy`, `phase_boundary_proxy`, etc. are **observable visual proxies**, not chemical ground truth. The semantic layer uses context/RAG to form hypotheses and reports uncertainty. It should not be evaluated as a chemical-identity system without manually/author-validated semantic labels.

---

## 中文说明 / One-click entry

完整中文架构、原理、公开数据结果与操作步骤见：`README_中文.md`。

推荐入口：

```bash
python RUN_ONE_CLICK.py --dry-run   # 先检查
python RUN_ONE_CLICK.py             # 一键下载公开数据并运行完整 benchmark
```
