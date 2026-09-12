# ETTm1 TimeMixerAdaptivePatch 实验执行归档

- 时间：2026-09-12
- 请求：在物理 GPU 0 上执行 baseline，在物理 GPU 1 上执行 joint；两组均使用 `SEQ_LEN=96`、`PRED_LEN=96`，并确保实验结果不重名。
- 前置检查：已按项目约定读取 `AGENTS.md`，并确认远程项目目录、tslib Python、两张 GPU 与 PyTorch CUDA 均可用。

## 执行结果

| 实验 | 设备 | 训练状态 | 最佳验证损失 | 测试 MSE | 测试 MAE |
|---|---|---|---:|---:|---:|
| baseline | 物理 GPU 0 | 完成 10 个 epoch，退出码 0 | 0.3910310 | 0.3266225457 | 0.3644873500 |
| joint | 物理 GPU 1 | 第 6 个 epoch 正常早停，最佳为第 3 个 epoch，退出码 0 | 0.3869392 | 0.3190336227 | 0.3598976433 |

joint 相对 baseline：

- MSE 绝对降低 0.0075889230，相对降低 2.3235%。
- MAE 绝对降低 0.0045897067，相对降低 1.2592%。

## 独立命名与产物

- baseline 标识：`ETTm1_96_96_TimeMixer` / `TimeMixer`
- joint 标识：`ETTm1_96_96_TMAdaptive_joint` / `TMAdaptive_joint`
- 两组 checkpoint、results、test_results 目录均不同。
- 两组 `checkpoint.pth`、`metrics.npy`、测试可视化目录及独立日志均已核验存在。
- baseline 日志：`experiment_logs/ETTm1_96_96_baseline_gpu0_20260912_162252.log`
- joint 日志：`experiment_logs/ETTm1_96_96_joint_gpu1_20260912_162252.log`

## 改动与验证

- 本次未修改项目源代码或实验脚本。
- 仅生成训练检查点、测试指标、可视化结果与运行日志。
- 两组测试均未出现 NaN、Inf 或 CUDA 错误；DTW 未启用。
- joint 路由统计为有限值，未触发路由塌缩中断。

任务依据：[TimeMixer AdaptivePatch Subtask 5](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
