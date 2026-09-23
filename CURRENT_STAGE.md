# Current Stage

Stage 00–06.5 已审核通过。

Stage 06.5 结论：

- 单独改变 k1/k2/k3 都没有发现明确的单尺度伤害；
- P3（只增大 k3 alpha）反而在 3/3 seeds 的 MSE 上优于 F0；
- FULL02 同时改变 k1/k2/k3 后平均显著退化，并表现出明显 seed sensitivity；
- 因此当前最可能的问题是 **scale interaction / non-additivity**，而不是某个单独 scale 的 alpha 增大。

当前只允许执行：

`prompts/06_6_pairwise_interaction_diagnostic.md`

继续使用：

- branch: `codex/saema-scale-diagnostic`
- SHA: `785f43d66cc96d42a62980c7fd27d9c764a11bb7`

新增 P12 / P13 / P23 三组 pairwise vectors，每组 seeds 2021/2022/2023，共 9 个新 runs。

不得进入 Stage 07，不得实现 learnable alpha 或 instance-adaptive。

完成后写：

- `results/06_6_pairwise_interaction_diagnostic.md`
- `results/data/06_6_pairwise_raw.csv`
- `results/data/06_6_interactions.csv`

完成后停止。
