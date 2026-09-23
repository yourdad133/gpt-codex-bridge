# Weather：TimeMixer 与 TimeMixerGTR 对比

## 请求与配置

在 Weather 数据集上对比 TimeMixer baseline 与输入级 GTR 增强版。按用户要求运行 3 组配对实验，随机种子为 2021、2022、2023；每组两模型使用相同 seed。

本次按 Weather 脚本采用 `seq_len=96`、`pred_len=96`，因为请求限定为两模型各重复三次、共六次运行。其余共同设置为：`features=M`、`enc_in=dec_in=c_out=21`、`e_layers=3`、`d_layers=1`、`factor=3`、`d_model=16`、`d_ff=32`、`batch_size=128`、`learning_rate=0.01`、`train_epochs=20`、`patience=10`、3 层 avg 下采样且窗口为 2。两模型均使用相同的 GTR 辅助默认参数（period_len=24、gate_init=0.1、dropout=0.1）；baseline 的 `gtr_cycle_len=24` 被 TimeMixer 忽略，TimeMixerGTR 使用 `gtr_cycle_len=144`。

## 测试集结果

| 重复 | Seed | TimeMixer MSE | TimeMixer MAE | TimeMixerGTR MSE | TimeMixerGTR MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 2021 | 0.162347 | 0.209557 | 0.164536 | 0.231836 |
| 2 | 2022 | 0.162648 | 0.209097 | 0.160852 | 0.227710 |
| 3 | 2023 | 0.162078 | 0.208918 | 0.222869 | 0.270282 |
| 均值 ± 样本标准差 | — | 0.162358 ± 0.000285 | 0.209191 ± 0.000330 | 0.182753 ± 0.034791 | 0.243276 ± 0.023479 |

测试数据形状为 10,444 个窗口，每个窗口 96 个时间点、21 个变量。GTR 的第三个 seed 指标明显偏高，因此三次均值受该次结果影响；这里保留全部单次结果供比较。

## 代码与运行文件

- `Time-Series-Library/models/TimeMixerGTR.py`：小时标记按 `(weekday * 24 + hour) % gtr_cycle_len` 定位，支持用户指定的 144 点（六天）周期。
- `Time-Series-Library/experiment_logs/weather_timemixer_gtr_c144_repeats_20260923.sh`：六次运行脚本。
- `Time-Series-Library/experiment_logs/TimeMixerGTR_Weather_pred96_20260923_142147.log`：完整训练与测试日志，包含六条精确 MSE/MAE。

运行前确认远程 `tslib` 环境与 CUDA 可用、两张 GPU 可见；六次实验均完成并在日志中复核了指标。
