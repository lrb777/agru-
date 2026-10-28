# AGRU 神经网络大盘择时策略

基于广发证券《神经网络择时与截面叠加的 ETF 绝对收益策略》研报复现的深度学习大盘择时系统。

## 项目背景

使用与截面选股相同的 **AGRU** 模型（Attention + GRU），通过"个股样本 + 市场标签"的训练方式，使模型具备大盘择时能力。每日将全市场个股预测聚合为三分类择时信号（0=空仓 / 1=半仓 / 2=满仓），实现绝对收益。

**研报目标基线**（中证全指，2020–2026/8，日频调仓，开盘价，单边万五）：

| 指标 | 目标值 | 复现值 |
|---|---|---|
| 年化收益率 | 15.86% | — |
| 夏普比率 | 1.18 | — |
| 最大回撤 | -12.66% | — |
| 择时 IC | 15.81% | 10.54% |

> 复现值基于 7 折滚动训练（2020–2026/8）样本外信号计算，IC 口径为 `spearman(score, 次日收益率)`。

## 环境要求

- Python 3.12+
- PyTorch 2.x（当前为 CPU 版）
- DolphinDB Python API (`dolphindb` 3.0+)
- pandas, numpy, scipy, scikit-learn

```bash
pip install torch dolphindb pandas numpy scipy scikit-learn pyarrow
```

## 数据源

本项目从本地 DolphinDB 数据库（`10.65.0.75:8902`）获取数据，不依赖外部数据接口。

| 数据 | DolphinDB 库表 | 说明 |
|---|---|---|
| 指数日频行情 | `dfs://d0_day.jy_qt_indexquote` join `jy_secumain` | 中证全指(000985) OHLCV，2010–2026/8 |
| 预计算因子 | `dfs://d2_day.factors_std` | 260 个因子（alpha158 + cicc + custom），长格式存储，pool=`filtered`，2015–2026/8 |

**因子覆盖范围**（260 个）：
- `alpha158_*`：Qlib Alpha158 因子集（kbar/price/rolling/volume/vol_price/reg/updown/rsqr 等）
- `cicc_*`：中金因子（动量/波动/流动性/资金流/相关性等）
- `custom_*`：自定义规模因子

项目精选 40 个因子用于训练，完整列表见 `config.py` 的 `SELECTED_FEATURES`。

## 项目结构

```
agru择时/
├── config.py              # 全局配置：DolphinDB连接/模型超参/滚动区间/特征清单/回测参数
├── run.py                 # 主入口（discover / fetch / preprocess / train / backtest / run / smoke）
├── extract_stats.py       # 提取标准化统计量和阈值（推理前一次性运行）
├── run_daily_predict.py   # 每日择时信号预测（实时拉取因子 → 模型推理 → 输出信号）
├── 复现框架.md            # 复现框架文档（策略介绍/参数/计划/验收基线）
├── README.md              # 本文件
│
├── src/                   # 核心代码
│   ├── __init__.py        # 包导出
│   ├── model.py           # AGRU 模型（Linear→Tanh→GRU→Attn→Mul/Sum/Concat→Linear）
│   ├── data_loader.py     # DolphinDB 数据接口（指数行情/因子拉取/本地缓存）
│   ├── features.py        # 特征管理（标准化/清洗）
│   ├── dataset.py         # 数据集构建（TimingDataStore / 市场标签 / 样本构建）
│   ├── train.py           # 滚动训练流水线（8+2+1逐年滚动 / 信号聚合 / TSRankIC早停）
│   ├── backtest.py        # 回测系统（日频调仓 / 开盘价 / 手续费 / IC统计）
│   └── metrics.py         # 评估指标（TSRankIC / 信号准确性 / 回测指标）
│
├── data/                  # 数据缓存（parquet，自动生成，已 gitignore）
│   ├── index_daily.parquet           # 中证全指日频行情
│   ├── factors_2015.parquet          # 逐年因子数据（每文件 ~1–2GB）
│   ├── ...
│   ├── factors_2026.parquet
│   └── factors_{year}_slim.parquet   # 精简版（仅40特征，float32，~100–200MB）
│
├── checkpoints/           # 模型权重 + 统计量（训练/提取后自动保存）
│   ├── model_{year}.pt               # 模型权重（7折：2020–2026）
│   ├── stats_{year}.pkl              # 标准化统计量 + 阈值（推理用）
│   └── signals_{year}.csv            # 各折测试集信号
│
├── results/               # 回测结果 + 每日预测
│   ├── signals.csv                   # 样本外信号（2020–2026/8，1553天）
│   ├── daily_predict.csv             # 每日预测结果（追加模式）
│   ├── daily_result.csv              # 回测逐日净值
│   └── train_*.log                   # 训练日志
│
├── run_real_test.py       # 真实数据快速测试脚本（子样本+少量epoch）
├── scripts_discover*.py   # DolphinDB 数据库探查脚本（开发调试用）
├── test_factor_fetch.py   # 因子拉取测试
├── test_fetch_year.py     # 单年拉取测试
└── fix_fetch.py           # 数据修复工具
```

## 运行顺序

### 第 1 步：验证代码可跑通（合成数据，无需 DolphinDB）

```bash
python run.py smoke
```

用 50 只合成股票 × 10 个随机特征验证全流程：特征构建 → 滚动训练 → 信号生成 → 回测 → 指标输出。约 30 秒完成。

### 第 2 步：探查 DolphinDB 数据库

```bash
python run.py discover
```

连接 DolphinDB，列出可用的 260 个因子名称和数据库结构。确认数据源可用。

### 第 3 步：拉取数据（首次运行，约 30–60 分钟）

```bash
python run.py fetch
```

从 DolphinDB 逐年拉取：
- 中证全指日频行情 → `data/index_daily.parquet`
- 预计算因子（260 个，pool=filtered）→ `data/factors_{year}.parquet`（每年约 1–2GB）

已缓存的年份会自动跳过。数据范围：2015–2026/8（因子数据从 2015 年起）。

> **注意**：`fetch_all` 在合并所有年份时可能 OOM（20+ GiB），但各年份 parquet 已保存成功，不影响后续使用。训练时按折单独加载。

### 第 3.5 步：预处理（精简 parquet，加速训练）

```bash
python run.py preprocess
```

将每年 parquet 精简为仅 40 个选定特征 + float32，文件从 ~1–2GB 降至 ~100–200MB。训练时优先读取 slim 版本。

### 第 4 步：快速验证（真实数据子样本，约 4 分钟）

```bash
python run_real_test.py
```

使用 500 只随机采样股票、2022–2025 数据、3 个 epoch 快速验证真实数据全流程。确认数据加载、模型训练、信号生成、回测链路正常。

### 第 5 步：完整训练与回测

```bash
python run.py run
```

全量数据、7 个滚动折（2020–2026）、100 epoch（早停 patience=10），输出对标研报的完整回测结果。结果保存到 `results/signals.csv` 和 `results/daily_result.csv`。

> **注意**：CPU 环境下完整训练耗时极长（每个滚动折数小时），建议使用 GPU 或先缩减 `config.py` 中的 `ROLLING_YEARS`、`TRAIN["max_epochs"]`、`TRAIN["batch_size"]` 进行调试。

### 第 6 步：每日择时预测

#### 6.1 提取统计量（一次性，每次重训后需重新运行）

```bash
python extract_stats.py --year 2026
```

从训练数据重建标准化统计量（mean/std）和验证集 score 分位数阈值（t_low/t_high），保存到 `checkpoints/stats_2026.pkl`。推理时依赖此文件。

#### 6.2 每日预测

```bash
python run_daily_predict.py                    # 预测今日信号
python run_daily_predict.py --date 2026-09-01  # 指定预测日期
python run_daily_predict.py --year 2025        # 指定模型年份
```

**时序逻辑**：

```
T日收盘 → DolphinDB因子可用 → 拉取[T-19, T]全市场因子 → 标准化 → 模型推理 → 聚合 → 阈值映射 → 信号
                                                                                    ↓
                                                                          T+1开盘前调仓
```

实时从 DolphinDB 拉取最近 45 天因子数据（无需更新本地 parquet），每只股票构建 20 日特征窗口，模型推理后全市场概率均值聚合为当日 score，按阈值映射为 0/1/2 信号。结果追加到 `results/daily_predict.csv`。

**输出格式**：

| 字段 | 说明 |
|---|---|
| date | 预测日期（T+1） |
| signal | 0=空仓 / 1=半仓 / 2=满仓 |
| score | 连续择时打分（满仓概率 − 空仓概率的均值） |
| position | 建议仓位（0.0 / 0.5 / 1.0） |
| model_year | 使用的模型年份 |
| n_stocks | 参与预测的股票数 |
| data_end_date | 因子数据截止日期（T） |
| predict_time | 预测时间戳 |

## 核心配置说明

所有参数集中在 `config.py`，关键项如下：

### DolphinDB 连接

```python
DOLPHINDB = {"host": "10.65.0.75", "port": 8902, "username": "admin", "password": "123456"}
```

### 模型结构（对齐研报图 1）

```python
MODEL = {"hidden_dim": 64, "num_gru_layers": 2, "num_classes": 3, "dropout": 0.3, "activation": "tanh"}
```

AGRU 数据流：`Linear→Tanh → GRU(2层×64) → Attention → Mul/Sum/Concat融合 → Dropout(0.3) → Linear → 三分类`

### 训练配置

```python
TRAIN = {"batch_size": 12000, "lr": 1e-3, "max_epochs": 100, "patience": 10, "val_metric": "tsrankic"}
```

优化器 Adam，验证集 TSRankIC 早停。

### 滚动训练区间

```python
ROLLING_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
```

每年滚动：8 年训练 + 2 年验证 + 1 年测试（因子数据从 2015 年起，前几折训练集不足 8 年时自动截断到 2015）。数据截止 2026/8/31。

### 标签与仓位

```python
LABEL = {"forecast_horizon": 1, "class_ratios": [0.3194, 0.4159, 0.2647]}
BACKTEST = {"trade_price": "open", "commission_rate": 0.0005, "position_map": {0: 0.0, 1: 0.5, 2: 1.0}}
```

标签为中证全指未来 1 日收益三分类（按训练集分位数切分）；仓位映射为 0→空仓 / 1→半仓 / 2→满仓（待波动率校验后定案）；手续费单边万五。

## 训练方案详解

### "全新训练方式"实现

研报的核心创新在于训练方式——网络结构和特征与截面选股完全相同，但标签从"个股未来收益"替换为"市场未来收益"：

1. **样本**：每日、每只股票一个样本（股票 i 的过去 20 日特征序列），样本量约为 交易日 × 股票数
2. **标签**：同一交易日内所有股票共享同一市场标签——中证全指未来 1 日收益的三分类（0/1/2）
3. **训练**：模型从个股特征中学习大盘方向
4. **聚合**：每日将全市场个股预测的概率均值聚合为单一择时信号，按验证集分位数阈值映射为 0/1/2

### AGRU 模型结构

```
输入 (B, 20, 40)              # batch × 时序窗口 × 特征数
  ↓
Linear(40→64) + Tanh           # 特征编码
  ↓
GRU(2层, hidden=64)            # 时序编码
  ↓
┌─────────────────────────────┐
│  gru_last (B,64)  [旁路]     │
│  Attention(gru_seq) (B,64)  │
└─────────────────────────────┘
  ↓
融合: [Mul, Sum, Concat] (B, 256)
  ↓
Dropout(0.3) → Linear(256→64) → Tanh → Linear(64→3)
  ↓
三分类输出 (0/1/2)
```

### 评估指标

- **TSRankIC**：每日预测得分与次日收益率的 Spearman 秩相关（研报目标 15.81%，复现值 10.54%）
- **信号准确性**：按信号分组统计出现比例、次日收益率均值、胜率（对标研报表 4）
- **回测指标**：总收益率、年化收益率、年化波动率、夏普比率、最大回撤

> **注意**：`backtest.py` 中 IC 计算有已知 bug——使用了当日收益（`close.pct_change()`）而非次日收益，导致回测输出的 IC 值不正确。正确的 IC 应在 `signals.csv` 的 score 与次日收益率之间计算。

## 验收基线

### 基线 A：策略表现

| 指标 | 研报目标值 |
|---|---|
| 总收益率 | 131.87% |
| 年化收益率 | 15.86% |
| 年化波动率 | 13.44% |
| 夏普比率 | 1.18 |
| 最大回撤 | -12.66%（2024/1/26 – 2024/2/6） |
| 择时 IC | 15.81% |

### 基线 B：信号准确性

| 信号 | 出现比例 | 收益率均值 | 胜率 |
|---|---|---|---|
| 0 | 31.94% | -0.19% | 40.67% |
| 1 | 41.59% | 0.03% | 47.78% |
| 2 | 26.47% | 0.29% | 62.47% |

### 基线 C：稳健性测试

| 测试项 | 研报目标值 |
|---|---|
| 周频信号 | 夏普 0.71 |
| 周频调仓 | 夏普 0.35 |
| 日频信号 + 月频调仓 | 年化 4.68%，夏普 0.42 |
| 早盘 60 分钟 TWAP | 年化 13.15%，夏普 1.08 |
| 沪深 300 | 年化 13.43%，夏普 1.10 |
| 中证 1000 | 年化 17.51%，夏普 1.02 |

## 训练结果

7 折滚动训练全部完成（2020–2026/8），样本外信号共 1553 天。

| 滚动折 | 训练集 | 验证集 | 测试集 | 验证集 IC | 测试集 IC |
|---|---|---|---|---|---|
| 2020 | 2015–2017 | 2018–2019 | 2020 | 0.0796 | 0.1485 |
| 2021 | 2015–2018 | 2019–2020 | 2021 | 0.1626 | 0.1428 |
| 2022 | 2015–2019 | 2020–2021 | 2022 | 0.1374 | 0.0753 |
| 2023 | 2015–2020 | 2021–2022 | 2023 | 0.0950 | 0.1074 |
| 2024 | 2015–2021 | 2022–2023 | 2024 | 0.0735 | 0.0977 |
| 2025 | 2015–2022 | 2023–2024 | 2025 | 0.0849 | 0.1302 |
| 2026 | 2016–2023 | 2024–2025 | 2026 | 0.1037 | 0.0806 |

整体样本外 IC = **10.54%**（研报目标 15.81%）。信号分布：0→515 天（33%），1→643 天（41%），2→395 天（25%），与研报目标比例 32%/42%/26% 基本吻合。

## 已知问题与待办

- [x] 因子数据从 2015 年起，研报需 2010 年起，前几折训练集不足 8 年（已自动截断到 2015）
- [x] CPU 训练速度慢（已用 GPU 完成全部 7 折训练）
- [x] 每日推理脚本（`run_daily_predict.py` + `extract_stats.py` 已实现）
- [ ] `backtest.py` 的 IC 计算有 bug（用了当日收益而非次日收益），需修复
- [ ] 仓位规则（0/1/2 → 空/半/满）待波动率校验后定案
- [ ] 信号聚合规则与阈值方案待调优（当前为概率均值 + 分位数阈值）
- [ ] 研报第二、三部分（截面 ETF 轮动、叠加策略）暂未复现
- [ ] Attention 实现细节、Dropout 插入位置需消融验证
- [ ] 定期重训机制（当前需手动运行 fetch → train → extract_stats）

## 复现范围

本期仅复现研报第一部分「神经网络择时（深度学习择时策略）」。第二部分「截面 ETF 轮动」与第三部分「择时+截面叠加的 ETF 绝对收益策略」暂不纳入。

## 参考

- 研报：广发证券《神经网络择时与截面叠加的 ETF 绝对收益策略》（安宁宁、陈原文、王小康）
- 关联报告：《基于 AGRU 因子聚合的 ETF 轮动策略》（截面特征来源）
