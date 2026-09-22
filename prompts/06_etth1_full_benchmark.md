# Stage 06 — ETTh1 正式主实验：4 horizons × 3 seeds

## 前置条件

Stage 05 已冻结：
- fixed EMA alpha_base；
- Scale-Aware EMA alpha_base。

本阶段不得继续根据 test 调 alpha。

## 模型

A. Original TimeMixer moving-average baseline  
B. TimeMixer + fixed EMA  
C. TimeMixer + Scale-Aware EMA

## 数据与预测长度

ETTh1:

`pred_len ∈ {96, 192, 336, 720}`

seeds:

`{2021, 2022, 2023}`

每个新增方法共 12 次正式实验。

### Baseline 是否重跑

此前已经完成原 TimeMixer 的 4 horizons × 3 seeds。

只有在以下全部一致时可以复用：
- dataset split；
- TSLib baseline code；
- model hyperparameters；
- training hyperparameters；
- seed；
- environment 足够一致。

如果存在影响可比性的代码/配置变化，则重跑 baseline。报告必须说明决定和理由。

## 公平性

除 decomposition 与其 alpha 外，其余保持完全一致：

- epochs；
- optimizer；
- lr；
- batch size；
- architecture；
- seed；
- data loader；
- early stopping；
- loss。

## 输出指标

每个 run：
- MSE；
- MAE；
- best validation loss；
- training time；
- status。

每个 horizon：
- 3-seed mean；
- std；
- 相对 baseline 的 MSE change %；
- 相对 baseline 的 MAE change %。

再给：
- 4 horizons 的平均 mean MSE / MAE；
- 参数量；
- 如果方便，平均训练时间。

不要只报告“最好 seed”。

## 统计检查

如果 3-seed 数据足够，做基础配对比较即可，但不要夸大统计显著性。保留每个 seed 的原始数据。

## 输出

写：

`results/06_etth1_full_benchmark.md`

和机器可读：

`results/data/06_etth1_full_benchmark.csv`

报告中明确回答：

1. EMA 相比 moving average 是否有稳定变化？
2. Scale-Aware EMA 相比 fixed EMA 是否在多数 horizon/seed 上保持一致方向？
3. 是否存在只在某一个 horizon 提升、其他 horizon 退化的情况？
4. 当前证据是否值得进入多数据集验证？

只做基于结果的描述，不修改方法来迎合结果。

完成后停止，不执行 Stage 07。
