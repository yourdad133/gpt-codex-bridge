# ETTm1 fixed4 与 fixed8 并行实验归档

- 时间：2026-09-12
- 请求：使用 `SEQ_LEN=96`、`PRED_LEN=96` 并行执行 `fixed4` 与 `fixed8`。
- 设备分配：`fixed4` 使用物理 GPU 0，`fixed8` 使用物理 GPU 1。
- 前置检查：已按项目约定读取 `AGENTS.md`，并确认项目目录、tslib Python、两张 GPU 与 PyTorch CUDA 均正常。
- 输出标签分别为 `TMAdaptive_fixed4` 和 `TMAdaptive_fixed8`，未覆盖已有结果。

## 执行结果

| 实验 | 状态 | 最佳 epoch | 最佳验证损失 | 测试 MSE | 测试 MAE |
|---|---|---:|---:|---:|---:|
| fixed4 | 第 4 个 epoch 正常早停，退出码 0 | 1 | 0.3934828 | 0.3327924311 | 0.3665492535 |
| fixed8 | 第 7 个 epoch 正常早停，退出码 0 | 4 | 0.3931341 | 0.3204535842 | 0.3585510850 |

`fixed8` 相对 `fixed4`：

- MSE 绝对降低 0.0123388469，相对降低 3.7077%。
- MAE 绝对降低 0.0079981685，相对降低 2.1820%。

## 产物与验证

- fixed4 日志：`experiment_logs/ETTm1_96_96_fixed4_gpu0_20260912_170306.log`
- fixed8 日志：`experiment_logs/ETTm1_96_96_fixed8_gpu1_20260912_170306.log`
- 两组 `checkpoint.pth`、`metrics.npy` 和测试输出目录均已核验存在。
- 两组退出码均为 0；未发现 Traceback、RuntimeError、CUDA 错误、显存不足或 NaN。
- 日志中的固定路由单选提示是 fixed 模式的预期行为；DTW 未启用。

## 文件改动

- 本次未修改项目源代码或实验脚本。
- 仅生成训练检查点、测试指标、可视化结果和独立运行日志。

任务依据：[TimeMixer AdaptivePatch Subtask 5](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
