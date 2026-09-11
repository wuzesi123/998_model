# ChemProcessRAG / TCPT 中文使用说明

> 当前版本整理日期：2026-09-11  
> 本文档按当前项目真实代码整理。重点是：**化学实验视频 → 可观测视觉证据 → 因果多尺度 TCPT → Reaction Context / RAG → Semantic Reasoner**。

---

## 1. 项目到底要解决什么问题

最终目标不是单纯预测“未来晶体面积”，而是让摄像头持续观察一个化学实验，并在任意时刻 `t` 只利用 `0...t` 的信息回答：

- 当前处于什么阶段（Current Stage）
- 当前过程进度大约是多少（Current Progress）
- 当前正在发生什么可见事件（Active Event）
- 是否正在接近阶段转变或终点（Transition / Endpoint Status）
- 判断置信度以及依据（Confidence + Evidence）

核心约束是 **causal（因果）**：不能偷看未来帧、最终产物、最终收率或测试实验未来结果。

当前公开数据 benchmark 中，CrystalCV V2 暂时把“未来可观测量预测”作为 **时序能力辅助验证**，它证明历史轨迹有信息，但它不是最终主任务。后续主线应重新回到 `current-stage / progress / event recognition`。

---

## 2. 当前完整架构

```text
实验视频 / 实时摄像头
        │
        ▼
┌───────────────────────────────┐
│  A. Visual Perception         │
│  可观测视觉特征               │
│  RGB/Lab、亮度、运动、边缘、   │
│  局部对比度、浑浊、固体、相界面 │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  B. Causal Multi-scale TCPT   │
│                               │
│  Short  : 最近窗口            │
│  Medium : 中期变化            │
│  Global : 从实验开始的稀疏历史 │
│                               │
│  每个分支 = 时间编码 +         │
│  causal Transformer encoder   │
│              │                │
│       三分支融合 Fusion        │
└───────────────────────────────┘
        │
        ├─────────────┐
        │             │
        ▼             ▼
┌─────────────┐  ┌────────────────┐
│Reaction     │  │ ORD / Knowledge │
│Context      │  │ RAG             │
│物质/条件/步骤│  │ 化学知识检索     │
└─────────────┘  └────────────────┘
        │             │
        └──────┬──────┘
               ▼
┌───────────────────────────────┐
│ C. Semantic Reasoner          │
│                               │
│ process_type                  │
│ temporal_state                │
│ active_event                  │
│ confidence                    │
│ hypotheses                    │
│ evidence / provenance         │
│ expected_next_events          │
│ uncertainty_notes             │
└───────────────────────────────┘
```

---

## 3. A层：Visual Perception 的原理

入口代码主要在：

```text
chem_process_rag/visual/features.py
chem_process_rag/visual/analyzer.py
```

系统首先把视频按设定时间间隔采样，然后对每个采样帧计算可观测特征。

当前特征强调“肉眼/摄像头真正能看到什么”，例如：

- 颜色与 Lab 颜色变化
- brightness / brightness std
- motion activity
- edge density
- local contrast
- turbidity proxy
- solid proxy
- phase-boundary proxy

这里非常重要：`solid_proxy`、`turbidity_proxy` 等只是 **视觉代理量**，不能直接宣称是真实化学浓度、转化率或物质身份。

`analyzer.py` 还会计算：

- `early_mean`
- `late_mean`
- `slope`
- `recent_slope`
- `delta`
- `late_to_early_ratio`

因此语义层不仅知道“现在固体 proxy 是多少”，还知道“过去一直在增加，最近是否还在增加”。

### 已有公开数据成绩

HeinSight4.0 chemical dataset v2 官方 held-out test：

- Precision ≈ **95.6%**
- Recall ≈ **93.6%**
- mAP50 ≈ **96.5%**
- mAP50-95 ≈ **93.0%**

因此当前视觉层是项目里证据最扎实的一部分。

---

## 4. B层：TCPT 因果多尺度时序层

当前项目里有两套“用途不同但思想一致”的 TCPT，请不要混淆。

### 4.1 实时语义管线版 TCPT

文件：

```text
chem_process_rag/visual/tcpt.py
```

类：

```text
MultiScaleTCPT
```

它包含三个分支：

```text
short branch
medium branch
global branch
```

每个 branch 大致为：

```text
输入特征
  ↓
Linear projection
  +
Time Encoding
  ↓
Transformer Encoder
  ↓
取当前/最后 token 表示
```

三个分支最终拼接：

```text
short embedding
      +
medium embedding
      +
global embedding
      ↓
Fusion MLP
      ↓
128-d temporal embedding
      ↓
optional observable-state head
```

#### 为什么要三个时间尺度？

**Short**：回答“刚才发生了什么”。例如最近几十秒突然变浑浊、晶体快速出现。

**Medium**：回答“最近一段时间趋势是什么”。例如晶体最近几分钟持续增长还是已经减速。

**Global**：回答“从实验开始到现在整体经历了什么”。用于区分“刚开始稳定”和“反应结束后稳定”。

只看当前帧时，这两种稳定状态可能很像；全局历史可以把它们区分开。

### 4.2 公开 benchmark / 研究版 TCPT

文件：

```text
tcpt/model.py
tcpt/forecast_models.py
```

核心类：

```text
TCPTv4
TCPTForecaster
```

虽然历史类名保留 `TCPTv4` 以兼容旧 checkpoint，但代码已经包含三尺度 short / medium / global causal encoders。

其关键结构为：

```text
input LayerNorm
      ↓
Linear + GELU + LayerNorm
      ↓
Continuous Time Encoding
      ↓
┌────────┬────────┬────────┐
│ Short  │ Medium │ Global │
│ causal │ causal │ causal │
│ encoder│ encoder│ encoder│
└────────┴────────┴────────┘
      ↓
concat
      ↓
Fusion
      ↓
shared temporal representation
```

`TCPTv4` 当前拥有多个任务头：

- `stage_head`：阶段分类 logits
- `transition_heat_head`：是否接近 transition
- `transition_distance_head`：归一化 transition 距离
- `progress_head`：0~1 progress
- `embedding`：融合后的时序表示

这其实已经与最终目标“当前阶段 + 当前进度 + transition”高度吻合，只是现阶段可信公开 benchmark 还没有给这些头提供可靠的 current-stage ground truth。

### Continuous Time Encoding 为什么重要？

普通 Transformer 如果只按 token 顺序处理，会把：

```text
相邻两帧间隔 1 秒
```

和：

```text
相邻两帧间隔 60 秒
```

看成相邻两个 token。

当前代码显式编码真实 elapsed time，并同时编码：

1. 从实验开始到该 token 的 elapsed time；
2. 该 token 距离当前时刻的 age。

因此模型知道真实时间间隔，而不只是数组位置。

---

## 5. TCPT 输出是否“人能看懂”

TCPT 内部最核心的输出是一个 latent temporal embedding，本身不是自然语言解释。

推荐理解为两层：

```text
TCPT 内部：
short / medium / global representation
→ fused temporal embedding

人类可读层：
stage / progress / active event /
transition / confidence / evidence
```

最终的人类解释由 task heads + Semantic Reasoner 完成。

不要把 attention weight 直接当作“模型解释”。更可靠的解释应绑定到真实可观测趋势，例如：

```text
solid proxy ↑
turbidity proxy ↑
crystal area 持续增加
recent slope 开始下降
```

然后解释：

```text
当前判断：ACTIVE_CRYSTAL_GROWTH
依据：中期持续增长，但最近增长速度开始下降
```

---

## 6. Reaction Context

代码：

```text
chem_process_rag/context/schema.py
chem_process_rag/context/context_builder.py
```

ReactionContext 可以记录：

### Participants

- name
- role
- SMILES
- InChI
- amount / unit
- concentration / unit

### Conditions

- temperature
- pressure
- pH
- stirring RPM
- atmosphere
- illumination

### Procedure

- experiment type
- current step
- objective
- notes

它解决的问题是：**同样的视觉现象，在不同实验上下文中可能对应不同化学过程。**

例如看到“新固体出现”，仅靠视觉只能说：

```text
new solid / heterogeneous phase appeared
```

如果 Context 明确是冷却结晶，则更支持 crystallization；如果 Context 是两种盐溶液混合，则可能更支持 precipitation。

---

## 7. ORD / RAG 层

代码：

```text
chem_process_rag/rag/ord_rag.py
chem_process_rag/rag/text_index.py
```

RAG 输入：

```text
Reaction Context
+
当前视觉摘要
```

输出：与当前实验最相关的化学知识记录。

### 最重要的防泄漏规则

实时 t 时刻推理时可以使用：

- 物质名称/结构
- 已知实验条件
- 实验开始前已知步骤
- 一般物性/反应知识
- t 时刻以前的观察

不能使用测试目标的未来信息，例如：

- held-out 实验最终 outcome
- 最终 yield
- 如果 product 是预测目标，则不能检索该实验真实 product
- t 之后的 observations

因此 RAG 是“提供背景知识”，不是“偷偷查答案”。

---

## 8. Semantic Reasoner

代码：

```text
chem_process_rag/reasoning/semantic_reasoner.py
```

当前结构刻意把几个概念分开：

- `process_type`：这是什么类型的过程
- `temporal_state`：当前时间状态
- `active_event`：现在正在发生什么
- `confidence`
- `hypotheses`
- `retrieved_knowledge`
- `evidence`
- `expected_next_events`
- `uncertainty_notes`

这避免了旧版本把“crystallization（过程类型）”和“active/stable（时间阶段）”混成同一个 `current_state`。

---

## 9. 当前公开数据 benchmark

### HeinSight4

任务：当前画面中的化学相/容器内容视觉检测。

结果已经比较强：约 93~96% 级别的检测表现。

### CrystalCV V2

当前任务：**future observable forecasting**，即根据历史预测未来 `log(total visible crystal area)`。

这不是最终 current-stage 任务，而是用于验证 TCPT 是否真的从历史中学习到过程演化。

数据清洗后：

- 扫描表：226
- 有效 trajectory tables：222
- 独立 physical experiments：**125**
- 去除重复表示：97
- 论文报告 case-study experiments：129
- 覆盖约：**96.9%**

划分：

```text
88 train / 19 validation / 18 test
```

TCPT 相比 LastValue baseline 的结果大致：

| Horizon | TCPT macro-exp NMAE | 相比 LastValue skill | TCPT胜过LastValue的测试实验 |
|---|---:|---:|---:|
| 0.5% | 0.0278 | +7.5% | 18/18 |
| 2% | 0.0484 | +11.3% | 17/18 |
| 5% | 0.0784 | +17.4% | 18/18 |
| 10% | 0.1278 | +15.0% | 15/18 |
| 20% | 0.2028 | +13.0% | 16/18 |

解释：**时序信息确实有价值**，而且不是少数长实验把平均值拉出来。

但当前 TCPT 还没有稳定超过 GRU / MLP / Ridge，因此尚不能宣称 TCPT 的多尺度结构已经证明最优。

### Context-aware + RAG

目前工程链路已经存在，但公开数据里没有找到足够明确的 per-image machine-readable context mapping，所以当前保持 audit-only，而不是人工制造 context 得分。

---

## 10. 当前最重要的问题

1. **最终任务曾暂时偏离**：CrystalCV 当前主要验证未来 observable forecasting，而最终目标是“判断现在反应到哪一步”。
2. **Current-stage Ground Truth 还不够可靠**：以前简单把轨迹切成 INITIAL/ACTIVE/STABLE 等 weak labels，科学依据不足。
3. **TCPT 目前没有稳定超过 GRU/MLP**：已证明“时间历史有用”，但未证明“三尺度 TCPT 是最佳结构”。
4. **Context/RAG 缺公开定量 GT**：工程已经通，但缺泄漏安全的公开 image→context 对应数据。
5. **跨化学过程不足**：CrystalCV 主要是 crystallization，后续应扩展 precipitation、dissolution、phase separation、color-change kinetics 等。
6. **CrystalCV material metadata 当前没有完全恢复**：现有 V2 benchmark 的 material 字段可能为 unknown，因此真正的跨材料泛化测试仍需补 metadata。

---

## 11. 下一阶段推荐主任务

主任务改成：

```text
输入：X(0:t)

输出：
Current Stage
Current Progress
Active Event
Transition / Endpoint Status
Confidence
Evidence
```

CrystalCV 可以先建立结晶专用的客观状态：

```text
PRE_VISIBLE_CHANGE
        ↓
NUCLEATION_ONSET
        ↓
ACTIVE_CRYSTAL_GROWTH
        ↓
GROWTH_DECELERATION
        ↓
VISUAL_PLATEAU
```

但阶段边界不能凭主观感觉硬切，应该从作者可观测量和可重复规则中定义，例如 crystal area、count、growth rate、persistent derivative、plateau criterion。

---

# 12. 如何运行

## 12.1 推荐环境

- Windows 10/11 或 Linux
- Python **3.10+**
- NVIDIA GPU 推荐；项目也支持 CPU，但训练会慢很多
- 完整公开数据建议至少 **20 GB 可用空间**

你目前如果使用 RTX 3080 Ti，完整 benchmark 建议保留现有可正常识别 CUDA 的 PyTorch 环境，不要随便覆盖 GPU 版 PyTorch。

---

## 12.2 最简单：一键完整运行

在项目目录打开终端：

```bash
python RUN_ONE_CLICK.py
```

它会：

1. 检查 Python
2. 检查磁盘空间
3. 检查 PyTorch / CUDA / GPU
4. 检查关键项目文件
5. 调用完整公开 benchmark
6. 自动安装缺失的普通 Python 依赖
7. 自动从 Zenodo 下载数据
8. 自动解压
9. 自动训练 HeinSight visual detector
10. 自动准备 CrystalCV V2 physical-experiment 数据
11. 自动训练/评估 baselines + TCPT
12. 对 context-aware 数据做 leakage-safe audit
13. 输出 JSON / CSV 汇总

当前自动数据来源包括：

```text
HeinSight4 chemical dataset v2       Zenodo 15605098
CrystalCV source data                Zenodo 19140131
CrystalCV software / Pre_Processed   Zenodo 20655010
Context-aware 2026 dataset           Zenodo 17436705
```

下载过的数据会复用，不会每次重新下载；已有完成结果也会复用，除非显式 `--force`。

---

## 12.3 只运行 CrystalCV V2

如果 HeinSight 已经跑过，只想测试时序：

```bash
python RUN_ONE_CLICK.py --mode crystalcv
```

等价核心入口：

```bash
python RUN_CRYSTALCV_V2.py
```

指定训练 epoch：

```bash
python RUN_ONE_CLICK.py --mode crystalcv --epochs 30
```

---

## 12.4 工程自测

```bash
python RUN_ONE_CLICK.py --mode selftest
```

或：

```bash
python RUN_SELF_TEST.py
```

它会测试核心 pipeline 是否能够：

```text
视觉分析
→ Reaction Context
→ RAG
→ Semantic Reasoner
→ JSON result
```

---

## 12.5 Demo

```bash
python RUN_ONE_CLICK.py --mode demo
```

会生成一个 demo reaction video 并跑完整基础语义链路。

结果：

```text
outputs/demo_reaction.avi
outputs/demo_result.json
```

没有 TCPT checkpoint 时，程序不会让随机神经网络参与语义判断，而会明确退回 deterministic visual evidence。

---

## 12.6 只检查、不下载

```bash
python RUN_ONE_CLICK.py --dry-run
```

适合第一次解压项目后先确认入口、Python、磁盘和关键文件没有问题。

---

# 13. 完整公开 benchmark 输出位置

```text
results/public_nightly/
│
├─ PUBLIC_NIGHTLY_SUMMARY.json
├─ PUBLIC_NIGHTLY_SUMMARY.csv
├─ RUN_PUBLIC_OVERNIGHT.log
│
├─ heinsight4/
│  ├─ HEINSIGHT4_RESULT.json
│  └─ heinsight4_visual/
│
├─ crystalcv/
│  ├─ prepared_v2/
│  │  └─ CRYSTALCV_PREPARE_MANIFEST_V2.json
│  └─ benchmark_v2/
│     ├─ CRYSTALCV_TEMPORAL_RESULT_V2.json
│     ├─ CRYSTALCV_TEMPORAL_RESULT_V2.csv
│     └─ checkpoints/
│
└─ context_aware/
   ├─ CONTEXT_RAG_AUDIT.json
   └─ CONTEXT_FUSION_RESULT.json   （仅条件满足时）
```

---

# 14. 项目目录说明

```text
chem_process_rag/
├─ visual/       视频特征、实时 TCPT、visual analyzer
├─ context/      ReactionContext schema / builder
├─ rag/          ORD + local chemistry retrieval
├─ reasoning/    Semantic Reasoner
└─ pipeline.py   主语义链路

public_benchmark/
├─ downloads.py              Zenodo自动下载/解压
├─ heinsight_yolo.py         HeinSight视觉检测
├─ crystalcv_prepare.py      CrystalCV V2清洗/去重/时间审计
├─ crystalcv_benchmark.py    CrystalCV时序benchmark
└─ context_rag_benchmark.py  Context/RAG泄漏安全审计

tcpt/
├─ model.py                  研究版多尺度TCPT
├─ forecast_models.py        MLP/GRU/TCPT forecasters
├─ forecast_data.py          forecast数据窗口
├─ transition_decoder.py     transition解码
├─ metrics.py                指标
└─ adapters/                 公共数据适配器
```

---

# 15. 已完成的工程检查

本整理版本已经实际执行：

```text
python -m compileall -q .        PASS
pytest -q                       2 passed
python RUN_SELF_TEST.py         PASS
python RUN_DEMO.py              PASS
python RUN_ONE_CLICK.py --dry-run PASS
```

注意：完整 public benchmark 会下载数 GB 数据并进行 GPU 训练，所以没有在打包环境里重新下载全套数据；但其代码路径来自已经跑出当前 HeinSight / CrystalCV V2 结果的版本。

---

# 16. 一句话说明当前研究状态

**视觉层已经在公开化学数据上达到较强结果；TCPT 已经证明真实历史轨迹对过程预测有价值；Context/RAG/Reasoner 工程链路已打通。下一步的核心研究不是继续把“未来晶体面积预测”当主任务，而是建立可靠的 current-stage / progress / event ground truth，让 TCPT 真正回答“这个化学反应现在进行到哪里了”。**
