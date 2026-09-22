# Stage 07 — 多数据集 Generalization Pilot（先小规模，不直接铺满）

## 目标

在投入大量 GPU 前，检查 Scale-Aware EMA 的效果是否只局限于 ETTh1。

使用 Stage 05 已冻结的 Scale-Aware EMA alpha_base，不针对新数据集重新用 test 调参。

## Pilot datasets

优先检查当前 TSLib 已准备好的：

- ETTm1
- Weather
- Electricity

若其中某数据集当前环境未准备好，记录缺失，不要随意从未知来源改数据。

## Pilot 设置

每个数据集：

- pred_len=96；
- seed=2021；
- original moving-average TimeMixer；
- fixed EMA；
- Scale-Aware EMA。

模型与训练参数优先遵循 TimeMixer 对该数据集的官方/仓库 baseline 配置，而不是强行复制 ETTh1 参数。

## 记录

每个 dataset × method：

- validation loss；
- test MSE；
- test MAE；
- runtime；
- parameter count；
- Scale-Aware 实际 alpha_k。

## 决策用途

本阶段不是最终 benchmark，只判断：

- Scale-Aware EMA 是否出现跨数据集的正向或至少非灾难性趋势；
- 是否有某类数据集明显不适合；
- 是否值得进入 Stage 08 的完整 multi-horizon × multi-seed 实验。

## 输出

写：

`results/07_generalization_pilot.md`

和：

`results/data/07_generalization_pilot.csv`

不要根据 pilot test 指标重新调整 alpha_base。

完成后停止，不执行 Stage 08。
