# Stage 06.6 — Pairwise Scale Interaction Diagnostic

## 前置结论

必须完整读取 Stage 06 与 Stage 06.5 结果，重点：

- `results/06_etth1_full_benchmark.md`
- `results/06_5_scale_sensitivity_diagnostic.md`
- `results/data/06_5_scale_sensitivity_raw.csv`
- `results/data/06_5_scale_sensitivity_summary.csv`

Stage 06.5 已得到：

- F0 = [0.02,0.02,0.02,0.02]
- P1 = [0.02,0.0396,0.02,0.02]，mean ΔMSE = +0.000162244
- P2 = [0.02,0.02,0.07763184,0.02]，mean ΔMSE = -0.000437796
- P3 = [0.02,0.02,0.02,0.14923698]，mean ΔMSE = -0.000810097，3/3 seeds MSE 改善
- FULL02 = [0.02,0.0396,0.07763184,0.14923698]，mean ΔMSE = +0.007778426

关键现象：

> 单尺度 perturbation 没有解释 FULL02 的明显退化，说明主要问题可能来自尺度之间的非加性交互，而不是某一个尺度单独有害。

本阶段只定位 pairwise interaction，不设计新最终方法。

---

# 1. Git / source boundary

继续使用 Stage 06.5 独立诊断分支：

- `codex/saema-scale-diagnostic`
- Stage 06.5 source commit: `785f43d66cc96d42a62980c7fd27d9c764a11bb7`

开始前：

- HEAD 必须为上述 SHA；
- `git status --short` 必须为空；
- `codex/saema-v1 @ 91735a83...` 保持冻结不动。

正常情况下本阶段不应再修改模型源码，因为 `scale_vector_ema` 已经足够表达所有组合。

---

# 2. 实验设计

固定：

- dataset: ETTh1
- seq_len=96
- pred_len=192
- seeds={2021,2022,2023}
- 10 epochs
- 其余训练配置完全复用 Stage 06.5

已有并复用：

- F0
- P1
- P2
- P3
- FULL02

新增三个 pairwise vectors：

### P12
`[0.02,0.0396,0.07763184,0.02]`

### P13
`[0.02,0.0396,0.02,0.14923698]`

### P23
`[0.02,0.02,0.07763184,0.14923698]`

每个 3 seeds，共 **9 个新 runs**。

不得新增其他 alpha 或根据中途结果改 vector。

---

# 3. 主分析：pairwise interaction

定义相对 F0 的 seed-level effect：

`E1 = MSE(P1)-MSE(F0)`
`E2 = MSE(P2)-MSE(F0)`
`E3 = MSE(P3)-MSE(F0)`

pairwise interaction：

`I12 = MSE(P12)-MSE(P1)-MSE(P2)+MSE(F0)`

`I13 = MSE(P13)-MSE(P1)-MSE(P3)+MSE(F0)`

`I23 = MSE(P23)-MSE(P2)-MSE(P3)+MSE(F0)`

解释：

- `Iij > 0`：两个 scale 同时改变时出现额外负向交互；
- `Iij < 0`：两个 scale 组合有额外正向协同；
- 接近 0：近似可加。

必须对每个 seed 单独算 interaction，再汇总 mean ± sample std。

MAE 同样计算。

---

# 4. 三阶 interaction

利用已有 FULL02 计算三阶项：

`I123 = FULL02 - P12 - P13 - P23 + P1 + P2 + P3 - F0`

对 MSE / MAE 都算。

这一步用来区分：

- FULL02 的退化主要已经能由某个 pairwise interaction 解释；
- 还是只有三个尺度同时变化时才出现额外三阶交互。

---

# 5. 必须回答的问题

1. 哪个 pair 的 mean MSE interaction 最大、最稳定？
2. 是否有某个 pair 在 3/3 seeds 都是正 interaction？
3. seed 2021 的 FULL02 异常能否被某个 pairwise interaction 解释？
4. P3 单独改善，但与 P1/P2 组合后是否转为有害？
5. FULL02 的总 effect 可被 main effects + pairwise effects 解释到什么程度？
6. 是否存在明显三阶 interaction？

---

# 6. 决策规则

### CASE A — 一个 pair 明显负向且稳定
例如 I13 或 I23 在 3/3 seeds 为正且幅度明显。

结论：
- 主要问题是特定尺度组合冲突；
- 下一步优先考虑 constrained / partially adaptive scale design，而不是所有尺度独立学习。

### CASE B — 多个 pair 都负向
结论：
- 多尺度同时改变 alpha 的耦合普遍有害；
- 下一步优先做 learnable scale vector，但需要 regularization / bounded deviation，初始化 F0。

### CASE C — pairwise interaction 不大，但 I123 很大
结论：
- 三尺度联合改变才触发问题；
- 下一步需要 jointly learnable scale vector，而不是逐尺度独立启发式。

### CASE D — interactions seed-unstable
结论：
- 仍不足以进入复杂 adaptive 模块；
- 下一步扩大 horizon / seed diagnostic，而不是直接 learnable/instance-adaptive。

---

# 7. Tests / integrity

开始训练前：

- 重新运行 Stage 06.5 的全部 86 tests；
- 确认 legacy regression 继续 PASS。

所有 9 个新 run：

- checkpoint finite；
- result arrays finite；
- shape 正确；
- 记录 best validation / epoch / test MSE / MAE；
- 不修改源码。

---

# 8. 输出

写入：

- `results/06_6_pairwise_interaction_diagnostic.md`
- `results/data/06_6_pairwise_raw.csv`
- `results/data/06_6_interactions.csv`

`06_6_interactions.csv` 至少包含：

```text
metric,
interaction,
seed,
value
```

并提供 mean/std 汇总。

报告最终必须给：

- Status
- 9/9 new runs complete
- strongest pairwise interaction
- pairwise direction
- three-way interaction direction
- seed stability
- diagnostic case A/B/C/D
- recommended next research step
- source code changed YES/NO

完成后 push 回 bridge 并停止。
