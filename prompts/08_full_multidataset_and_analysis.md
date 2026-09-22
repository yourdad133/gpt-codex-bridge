# Stage 08 — 完整多数据集实验与 Scale-Aware EMA 分析

## 前置条件

仅当 Stage 07 表明值得继续时执行。

## 目标

将已经冻结的方法扩展到多数据集、多 horizon、多 seed，形成可用于论文实验章节的最终证据。

具体数据集以 Stage 07 可用且合理的数据集为准，优先：

- ETTh1（已有 Stage 06，可直接复用）
- ETTm1
- Weather
- Electricity

预测长度优先：

`{96,192,336,720}`

seeds：

`{2021,2022,2023}`

至少比较：
- Original TimeMixer moving average
- Fixed EMA
- Scale-Aware EMA

## 必须额外分析

### A. Scale coefficient table

对每个使用的 down_sampling 配置给出：
- scale index；
- effective interval ratio；
- alpha_k。

### B. Efficiency

比较：
- parameter count；
- training time；
- inference time（可稳定测量时）；
- GPU memory（可稳定测量时）。

### C. Robustness

报告 mean ± std，不挑最好 seed。

### D. Failure cases

主动找出：
- Scale-Aware EMA 退化的数据集/horizon；
- 退化幅度；
- 可能与序列频率、趋势变化速度有关的可验证解释。

解释必须区分“观察到的事实”和“假设”。

### E. 可视化素材

在 TSLib 实验输出目录生成适合论文使用的数据/图表：
- 每个 scale 的 alpha；
- baseline vs EMA vs Scale-Aware EMA 的 MSE/MAE；
- 典型 forecasting case（只选预先规定或固定 index，避免 cherry-picking）。

## 输出

写：

`results/08_full_multidataset_and_analysis.md`

以及至少：
- `results/data/08_full_metrics.csv`
- `results/data/08_scale_alphas.csv`

报告最后形成：
- 当前方法能支持的结论；
- 当前方法不能支持的结论；
- 是否有必要进入下一研究阶段：Scale + Instance Adaptive EMA。

不要在本阶段直接实现 instance-adaptive EMA。
