# 项目检查报告（2026-09-11）

## 已执行检查

- `python -m compileall -q .`：PASS
- `pytest -q`：PASS，2/2 tests passed
- `python RUN_SELF_TEST.py`：PASS
- `python RUN_DEMO.py`：PASS
- `python RUN_ONE_CLICK.py --dry-run`：PASS
- `python RUN_CRYSTALCV_V2.py --help`：PASS
- `python RUN_PUBLIC_OVERNIGHT.py --help`：PASS

## 已确认的关键链路

1. Visual analyzer 可以读取视频并生成可观测特征与 causal trend summary。
2. ReactionContext schema / builder 正常。
3. ORDRAG 能读取 seed chemistry knowledge 并返回文档。
4. SemanticReasoner 能生成结构化报告。
5. CrystalCV V2 downloader / prepare / benchmark 入口完整。
6. HeinSight4 downloader / YOLO train-test 入口完整。
7. Context/RAG benchmark 已包含 `random` import，旧版 `name 'random' is not defined` 代码缺陷在当前整理代码中已修复。
8. 新增 `RUN_ONE_CLICK.py` 可统一进行 full / crystalcv / selftest / demo 四种模式。

## 未在打包环境重新执行的部分

完整 `RUN_PUBLIC_OVERNIGHT.py` 会重新下载数 GB 公开数据并进行较长 GPU 训练，因此未在打包环境重复执行全量训练。当前 `reference_results/` 保存了已完成 V2 run 的 JSON/CSV 结果供核对。

## 已知研究限制（不是程序错误）

- CrystalCV V2 当前主 benchmark 是 future observable forecasting，不是最终 current-stage recognition。
- 当前 TCPT 已稳定优于 LastValue，但没有稳定超过 GRU/MLP/Ridge。
- Context-aware 公共数据没有足够明确的 per-image machine-readable context mapping，因此 quantitative +Context/+RAG 仍为 audit-only。
- Current-stage ground truth 仍需要重新定义为客观、可重复的过程阶段。
