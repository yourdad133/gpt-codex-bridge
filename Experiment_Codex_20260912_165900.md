# ETTm1 uniform 与 joint_nobalance 并行实验归档

- 时间：2026-09-12
- 请求：使用 `SEQ_LEN=96`、`PRED_LEN=96` 并行执行 `uniform` 与 `joint_nobalance`。
- 设备分配：`uniform` 使用物理 GPU 0，`joint_nobalance` 使用物理 GPU 1。
- 前置检查：已按项目约定读取 `AGENTS.md`，并确认项目目录、tslib Python、两张 GPU 与 PyTorch CUDA 均正常。
- 输出标签分别为 `TMAdaptive_uniform` 和 `TMAdaptive_joint_nobalance`，未覆盖已有结果。

## 执行结果

| 实验 | 状态 | 最佳 epoch | 最佳验证损失 | 测试 MSE | 测试 MAE |
|---|---|---:|---:|---:|---:|
| uniform | 第 6 个 epoch 正常早停，退出码 0 | 3 | 0.3936367 | 0.3212362826 | 0.3592185378 |
| joint_nobalance | 第 6 个 epoch 正常早停，退出码 0 | 3 | 0.3914342 | 0.3197555840 | 0.3596812487 |

`joint_nobalance` 相对 `uniform`：

- MSE 绝对降低 0.0014806986，相对降低 0.4609%。
- MAE 绝对增加 0.0004627109，相对增加 0.1288%。

## 产物与验证

- uniform 日志：`experiment_logs/ETTm1_96_96_uniform_gpu0_20260912_165145.log`
- joint_nobalance 日志：`experiment_logs/ETTm1_96_96_joint_nobalance_gpu1_20260912_165145.log`
- 两组 `checkpoint.pth`、`metrics.npy` 和测试输出目录均已核验存在。
- 两组退出码均为 0；未发现 Traceback、RuntimeError、CUDA 错误、显存不足或 NaN。
- 路由统计均为有限值；DTW 未启用。

## 文件改动

- 本次未修改项目源代码或实验脚本。
- 仅生成训练检查点、测试指标、可视化结果和独立运行日志。

任务依据：[TimeMixer AdaptivePatch Subtask 5](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
