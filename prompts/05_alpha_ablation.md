# Stage 05 — alpha_base 小规模消融与验证集选择

## 研究目标

在进入大规模实验前回答两个问题：

1. Fixed EMA 本身是否可用？
2. Scale-Aware EMA 相比 fixed EMA 是否有额外信号？

并为后续正式实验选择合理的 alpha_base。

## 数据与模型

- dataset: ETTh1
- task: long-term forecasting
- seq_len=96
- pred_len=96
- seed=2021
- 训练配置与已复现 TimeMixer baseline 保持一致
- 正式 epoch 数恢复为 baseline 配置

## Candidate grid

分别对 `ema` 与 `scale_aware_ema` 使用相同候选：

`alpha_base ∈ {0.02, 0.05, 0.10, 0.15, 0.20}`

### 重要：禁止 test-set tuning

每个候选：
- 可以训练并记录 validation；
- **候选选择只根据 validation loss**；
- 不允许因为某个候选 test MSE 更低而选择它。

如果当前训练框架自动在训练结束打印 test，仍然必须在报告中明确：选择依据是 validation。

Fixed EMA 与 Scale-Aware EMA 可以各自选出自己的最佳 alpha_base，以保证消融公平。

## 需要记录的 Scale-Aware alpha

对每个 alpha_base，把每个 scale 的实际：

`alpha_k = 1-(1-alpha_base)^(w^k)`

写入表中。

## 最终对比

至少包含：

- original moving average baseline；
- best fixed EMA；
- best Scale-Aware EMA。

报告：
- best validation loss；
- 对应 test MSE / MAE（只在选择完成后用于报告）；
- 参数量；
- 训练时间。

不要在本阶段扩大到其他数据集。

## 输出

写：

`results/05_alpha_ablation.md`

以及：

`results/data/05_alpha_ablation.csv`

报告结尾必须明确给出：

- `Selected fixed-EMA alpha_base = ...`
- `Selected scale-aware-EMA alpha_base = ...`

这些值将冻结用于 Stage 06。

完成后停止，不执行 Stage 06。
