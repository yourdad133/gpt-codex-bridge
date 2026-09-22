# Codex 实验回复归档

日期：2026-09-22

## 请求摘要

在 Time-Series-Library 项目中，按照 TimeMixer 原论文配置，在 ETTh1 数据集上完成 long-term forecasting 复现，覆盖预测长度 96、192、336、720。

## 最终结果

已在远程 `tslib` 环境完成 12 次成功实验：4 个 prediction horizon × 3 个独立随机种子（2021、2022、2023）。论文配置下的三 seed 均值如下：

| Horizon | MSE | MAE |
|---:|---:|---:|
| 96 | 0.377340 | 0.397182 |
| 192 | 0.436168 | 0.431958 |
| 336 | 0.480726 | 0.451701 |
| 720 | 0.506334 | 0.481687 |
| 四 horizon 平均 | **0.450142** | **0.440632** |

论文统一配置结果为四 horizon 平均 MSE 0.447、MAE 0.440；本次结果与论文接近。

## 配置与验证

- `seq_len=96`，`label_len=0`，`pred_len={96,192,336,720}`
- `e_layers=2`，`d_model=16`，`d_ff=32`
- `down_sampling_layers=3`，窗口 2，平均池化
- Adam，初始学习率 0.01，MSE，batch size 128，10 epochs
- 远程 Torch 2.5.1+cu121，CUDA 可用，2 张 RTX 4090 D
- 12 个成功日志均含最终 MSE/MAE，12 个最佳 checkpoint 均成功加载并完成测试
- 项目代码未因本次复现实验修改；新增项目报告为 `TimeMixer_ETTh1_Paper_Reproduction_20260922.md`

GPU 0 当时被其他任务占用，因此成功任务将物理 GPU 1 映射为进程内 `cuda:0`；模型和实验超参数未改变。首次直接选择 `--gpu 1` 的尝试因项目现有设备重映射问题失败，未计入统计。

## 参考链接

- [TimeMixer 原论文](https://arxiv.org/abs/2405.14616)
- [官方 ETTh1 配置脚本](https://github.com/kwuking/TimeMixer/blob/main/scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh)
