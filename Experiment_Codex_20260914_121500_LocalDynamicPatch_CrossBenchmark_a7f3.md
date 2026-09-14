# Local Dynamic Patch 跨 horizon / 跨 benchmark 实验报告

日期：2026-09-14

## 1. 实验目标与模型定义

本轮以 Local Dynamic Patch 为主模型，验证 Weather、Electricity 和 M4 的跨 horizon / 跨 benchmark 表现。

Local Dynamic Patch 在本项目中定义为 `TimeMixerAdaptivePatch` 的 patch-only 路由配置：

- patch sizes：4、8、16、32
- patch routing：learned
- scale routing：uniform
- patch / scale router hidden size：64 / 64
- adaptive patch alpha：0.1
- patch / scale balance weight：0.001 / 0.001
- seed：2021

## 2. 数据与运行范围

使用项目官方数据源 `thuml/Time-Series-Library` 的 Weather、Electricity 和 M4 文件，数据落在项目 `dataset/` 目录。M4 额外补齐了 `submission-Naive2.csv`，用于标准 OWA 评估。

运行范围：

| Benchmark | 配置 |
|---|---|
| Weather | seq_len=96，pred_len=96/192/336/720 |
| Electricity | seq_len=96，pred_len=96/192/336/720 |
| M4 | Monthly、Yearly、Quarterly、Daily、Weekly、Hourly |

Weather 与 Electricity 按 horizon 分 4 个波次并行：GPU 0 运行 Weather，GPU 1 运行 Electricity。M4 按两个频率一波并行完成 6 个配置。所有模型 ID 均包含 benchmark、horizon / pattern、seed 和唯一 run tag，未覆盖此前结果。

## 3. 与 TimeMixer 对齐的超参数

### Weather / Electricity

- e_layers=3，down_sampling_layers=3，down_sampling_window=2
- learning_rate=0.01，train_epochs=20，patience=10
- d_model=16，d_ff=32
- batch_size：Weather=128，Electricity=32
- label_len=48

### M4

- e_layers=4，down_sampling_layers=1，down_sampling_window=2
- learning_rate=0.01，train_epochs=50，patience=20
- d_model=32，batch_size=128，loss=SMAPE
- seq_len=2×pred_len，label_len=pred_len（由 TimeMixer 的 M4 实验入口设置）
- d_ff：Monthly/Yearly/Weekly/Hourly=32，Quarterly=64，Daily=16

## 4. Weather / Electricity 结果

指标来自各训练日志最终 test 输出。

| Benchmark | Horizon | GPU | MSE | MAE |
|---|---:|---:|---:|---:|
| Weather | 96 | 0 | 0.161316 | 0.208628 |
| Weather | 192 | 0 | 0.209543 | 0.252824 |
| Weather | 336 | 0 | 0.263927 | 0.293343 |
| Weather | 720 | 0 | 0.346860 | 0.347737 |
| Electricity | 96 | 1 | 0.158061 | 0.251481 |
| Electricity | 192 | 1 | 0.173174 | 0.268499 |
| Electricity | 336 | 1 | 0.188787 | 0.281022 |
| Electricity | 720 | 1 | 0.226744 | 0.313535 |

## 5. M4 结果

6 个 M4 频率均已生成完整预测文件：

| Pattern | Horizon | Test shape |
|---|---:|---:|
| Monthly | 18 | (48000, 18, 1) |
| Yearly | 6 | (23000, 6, 1) |
| Quarterly | 8 | (24000, 8, 1) |
| Daily | 14 | (4227, 14, 1) |
| Weekly | 13 | (359, 13, 1) |
| Hourly | 48 | (414, 48, 1) |

按照项目 M4 标准汇总程序计算 SMAPE、MAPE、MASE 和 OWA；Others 为 Weekly、Daily、Hourly 的标准合并组。

| Group | SMAPE | MAPE | MASE | OWA |
|---|---:|---:|---:|---:|
| Yearly | 13.292 | 16.106 | 2.986 | 0.782 |
| Quarterly | 10.111 | 11.524 | 1.190 | 0.893 |
| Monthly | 12.749 | 14.906 | 0.946 | 0.887 |
| Others | 4.963 | 6.407 | 3.328 | 1.047 |
| Average | 11.851 | 13.945 | 1.593 | 0.853 |

## 6. M4 兼容性修复说明

M4 的训练序列长度不一致。在当前 NumPy 环境中，原始读取器将变长序列强制构造成规则 ndarray 会报错，因此将 M4 清洗后的序列保留为 list；M4 汇总器同步改为支持变长训练序列，并在 SMAPE/MAPE 计算处显式转换为 ndarray。该修复不改变模型结构或训练超参数。

第一次 M4 启动因该读取器问题退出，随后仅使用新 run tag 重跑失败的 6 个 M4 配置；Weather/Electricity 的 8 个已完成任务没有重复运行。M4 预测文件生成后，再补齐官方 Naive2 基准并单独完成标准汇总。

## 7. 变更文件与日志

代码变更：

- `scripts/long_term_forecast/LocalDynamicPatch_CrossBenchmark.sh`
- `data_provider/data_loader.py`
- `utils/m4_summary.py`

主要日志：

- `experiment_logs/LDP_Cross_weather_{96,192,336,720}_gpu0_ldp_cross_20260914_102500.log`
- `experiment_logs/LDP_Cross_electricity_{96,192,336,720}_gpu1_ldp_cross_20260914_102500.log`
- `experiment_logs/LDP_Cross_m4_{Monthly,Yearly,Quarterly,Daily,Weekly,Hourly}_gpu{0/1}_ldp_m4fix_20260914_123000.log`
- `m4_results/TimeMixerAdaptivePatch/{Monthly,Yearly,Quarterly,Daily,Weekly,Hourly}_forecast.csv`

## 8. 验证状态

- 远程 Python：`/home/ouyangguanghong/miniconda3/envs/tslib/bin/python`
- Torch：2.5.1+cu121；CUDA 可用，设备数 2
- 两块 RTX 4090 D 在最终检查时均空闲
- runner `bash -n` 通过
- 修改后的 M4 读取器和汇总器 `py_compile` 通过
- Weather/Electricity 4 个 horizon 及 M4 6 个 pattern 均完成预测输出

数据源：[Time-Series-Library 数据集](https://huggingface.co/datasets/thuml/Time-Series-Library)

