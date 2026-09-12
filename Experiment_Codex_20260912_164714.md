# ETTm1 patch_only 与 scale_only 并行实验归档

- 时间：2026-09-12
- 请求：沿用 `SEQ_LEN=96`、`PRED_LEN=96`，并行执行 `patch_only` 与 `scale_only` 对比实验。
- 设备分配：`patch_only` 使用物理 GPU 0，`scale_only` 使用物理 GPU 1。
- 前置检查：已按项目约定重新读取 `AGENTS.md`，并确认项目环境、Python、双 GPU 与 PyTorch CUDA 可用。
- 两组结果分别使用 `TMAdaptive_patch_only` 与 `TMAdaptive_scale_only` 标识，不会互相覆盖。

## 最终结果

| 实验 | 状态 | 最佳 epoch | 最佳验证损失 | 测试 MSE | 测试 MAE |
|---|---|---:|---:|---:|---:|
| patch_only | 第 6 个 epoch 正常早停，退出码 0 | 3 | 0.3963407 | 0.3187181354 | 0.3606434166 |
| scale_only | 第 6 个 epoch 正常早停，退出码 0 | 3 | 0.3924313 | 0.3249004483 | 0.3624021411 |

以 `scale_only` 为参照，`patch_only`：

- MSE 绝对降低 0.0061823130，相对降低 1.9028%。
- MAE 绝对降低 0.0017587245，相对降低 0.4853%。

## 产物与验证

- `patch_only` 日志：`experiment_logs/ETTm1_96_96_patch_only_gpu0_20260912_164027.log`
- `scale_only` 日志：`experiment_logs/ETTm1_96_96_scale_only_gpu1_20260912_164027.log`
- 两组独立的 `checkpoint.pth`、`metrics.npy` 和测试可视化目录均已核验存在。
- 日志未发现 Traceback、RuntimeError、CUDA 错误、显存不足或 NaN。
- 两组 DTW 均未启用。

## 改动说明

- 本次未修改项目源代码或实验脚本。
- 新增内容仅为训练检查点、指标、测试可视化结果和独立运行日志。

任务依据：[TimeMixer AdaptivePatch Subtask 5](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
