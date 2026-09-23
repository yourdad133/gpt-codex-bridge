# Stage 06.5 — Scale Sensitivity Diagnostic：定位 hand-crafted SAEMA 的负贡献来自哪个尺度

## 前置条件

必须完整读取 Stage 00–06 的全部结果，重点包括：

- `results/05_alpha_ablation.md`
- `results/06_etth1_full_benchmark.md`
- `results/data/06_trackA_selected_models_summary.csv`
- `results/data/06_trackB_same_alpha_mechanism.csv`
- `results/data/06_paired_deltas.csv`

Stage 06 已得到：

- MovingAvg four-horizon mean MSE = 0.450142
- Fixed EMA alpha=0.02 four-horizon mean MSE = 0.444800
- SAEMA alpha=0.05 four-horizon mean MSE = 0.451193
- same-alpha alpha=0.05：SAEMA 仅 4/12 cells MSE 更低，0/4 horizons mean MSE 更低
- mechanism direction = CONSISTENT_NEGATIVE

因此本阶段不继续做多数据集扩展，而是回答：

> hand-crafted scale-aware schedule 中，究竟是哪一个/哪些尺度的 alpha 增大导致性能退化？

---

# 1. Git 隔离

冻结现有正式分支：

- `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d`

不得在该分支继续提交诊断代码。

从该 SHA 新建独立 sibling worktree / branch：

- branch: `codex/saema-scale-diagnostic`

所有 Stage 06.5 诊断代码只允许存在于该分支。

原 `codex/saema-v1`、baseline-freeze 和原 dirty worktree均保持不动。

---

# 2. Diagnostic mode

为了隔离每个 scale 的作用，新增仅用于诊断的显式 per-scale static alpha 配置。

建议新增 decomposition mode：

`scale_vector_ema`

并通过一个明确配置传入 4 个 scale alpha，例如：

`--ema_scale_alphas 0.02,0.0396,0.02,0.02`

要求：

- 只用于 `channel_independence=1`；
- alpha 全部为固定常数，不可学习；
- 无新增 trainable parameter；
- 无 persistent state_dict key；
- 不改变 EMA recurrence；
- 不改变 TimeMixer 其他结构；
- legacy moving_avg / ema / scale_aware_ema 行为必须保持 bitwise / regression compatible；
- vector length 必须等于 `down_sampling_layers + 1`；
- 每个 alpha 必须 finite 且位于 (0,1)。

如果能在不修改核心模块的情况下通过测试 harness 注入 per-scale alpha，也可以采用，但必须保证行为清楚、可复现、可审计。优先选择最小、明确、可测试的实现。

---

# 3. 核心诊断基准

以 Stage 05/06 证明有效的 Fixed EMA：

`alpha = 0.02`

作为 reference vector：

`F0 = [0.02, 0.02, 0.02, 0.02]`

根据原 scale-aware 公式：

`alpha_k = 1-(1-0.02)^(2^k)`

得到：

- k0 = 0.020000
- k1 = 0.039600
- k2 = 0.07763184
- k3 = 0.14923698（日志可显示更多精度）

构造 one-scale-at-a-time perturbations：

### F0 — Fixed reference
`[0.02, 0.02, 0.02, 0.02]`

### P1 — Only scale 1 changed
`[0.02, 0.0396, 0.02, 0.02]`

### P2 — Only scale 2 changed
`[0.02, 0.02, 0.07763184, 0.02]`

### P3 — Only scale 3 changed
`[0.02, 0.02, 0.02, 0.14923698]`

### FULL02 — Full hand-crafted schedule at base 0.02
`[0.02, 0.0396, 0.07763184, 0.14923698]`

其中 F0 可复用 Stage 06 的 Fixed EMA alpha=0.02 结果。

P1/P2/P3 是本阶段最重要的因果式局部消融：每次只改变一个 coarse scale。

FULL02 用于检查三个 perturbation 的组合效应是否近似可加或存在交互。

---

# 4. 实验规模：先做 diagnostic pilot，不铺满 4×3

第一轮只使用：

- dataset: ETTh1
- seq_len=96
- pred_len=192
- seeds = {2021, 2022, 2023}
- 10 epochs
- 其余配置完全沿用 Stage 06

选择 pred_len=192 的原因：

- Stage 06 Track B 在 H=192 的 same-alpha 负效应最明显；
- 三 seed 全部 Fixed 0.05 优于 SAEMA 0.05；
- 是定位机制问题的高信号 horizon。

运行：

- P1 × 3 seeds
- P2 × 3 seeds
- P3 × 3 seeds
- FULL02 × 3 seeds

共 **12 个新 runs**。

F0 × 3 seeds 直接复用 Stage 06 Fixed EMA alpha=0.02 / H=192。

不得因为结果中途出现某个趋势而修改 alpha vector。

---

# 5. Primary diagnostic metric

对每个 perturbation X：

`delta_X(seed) = MSE_X(seed) - MSE_F0(seed)`

以及：

`mean_delta_X = mean_seed(delta_X(seed))`

解释：

- `mean_delta < 0`：仅改变该 scale 后性能改善；
- `mean_delta > 0`：仅改变该 scale 后性能退化。

同时计算 MAE delta。

主报告必须给：

| Config | Vector | Mean MSE ± std | ΔMSE vs F0 | Mean MAE ± std | ΔMAE vs F0 | MSE wins vs F0 |
|---|---|---:|---:|---:|---:|---:|

并保留每 seed paired delta。

---

# 6. Diagnostic interpretation

重点回答：

1. P1 是否主要造成退化？
2. P2 是否主要造成退化？
3. P3 是否主要造成退化？
4. 是否只有最粗尺度 k=3 明显有害？
5. 是否 coarse scales 越粗，增大 alpha 的伤害越大？
6. FULL02 的退化是否大致等于 P1/P2/P3 的组合，还是存在明显交互？

不要把 3-seed pilot 描述成广泛结论。

---

# 7. Optional reverse-direction probe

只有在 P1/P2/P3 中至少一个 scale 明显显示：

`mean_delta_MSE > 0`

才允许对最有害的那个 scale 做一个非常小的反向 probe。

例如若 k3 最有害：

Reference:

`[0.02,0.02,0.02,0.02]`

额外测试两个更小 alpha：

- `[0.02,0.02,0.02,0.01]`
- `[0.02,0.02,0.02,0.005]`

只使用：

- pred_len=192
- seed=2021

最多 **2 个额外 runs**。

目的不是调出最佳 test，而是判断：

> 最粗尺度的合理方向是否可能是 alpha 减小，而不是增大。

这些 probe 只能作为方向性诊断，不得据此冻结正式模型超参数。

如果 P1/P2/P3 没有清晰单尺度伤害，则跳过 reverse probe。

---

# 8. Tests

若新增 `scale_vector_ema`：

必须新增/扩展测试，至少覆盖：

- vector length validation；
- alpha boundary validation；
- per-scale alpha exact usage；
- no parameters；
- no state_dict key；
- CPU/CUDA finite；
- backward finite；
- moving_avg regression unchanged；
- fixed EMA regression unchanged；
- original scale_aware_ema regression unchanged；
- legacy old checkpoint strict-load 仍通过。

Stage 03 的 74 tests 必须继续通过。

如果修改了 `EMADecomp.py`，Stage 02 tests 也必须全部通过。

---

# 9. Source-code policy

诊断 branch 可以为了 explicit vector mode 做最小代码扩展，但：

- 不得修改已冻结 `codex/saema-v1`；
- 不得改变已有三种 decomposition 的语义；
- 不得把 vector alpha 设为 learnable；
- 不得实现 instance-adaptive；
- 不得做 per-channel alpha；
- 不得开始新的最终方法设计。

这个阶段只负责“定位问题”。

---

# 10. Required outputs

写入 bridge：

- `results/06_5_scale_sensitivity_diagnostic.md`
- `results/data/06_5_scale_sensitivity_raw.csv`
- `results/data/06_5_scale_sensitivity_summary.csv`

如果执行 reverse probe，再写：

- `results/data/06_5_reverse_probe.csv`

Raw CSV 至少包含：

```text
config,
alpha_vector,
pred_len,
seed,
best_validation_loss,
best_epoch,
test_mse,
test_mae,
finite,
checkpoint_path,
git_sha,
reused,
source_stage
```

---

# 11. Final decision rule

最终必须把结果归入以下之一：

### CASE A — 单个 coarse scale 明显有害
例如 P3 在 3/3 seeds 都恶化。

结论：
- hand-crafted monotonic-increasing schedule 的问题已定位；
- 下一阶段优先做 learnable per-scale alpha；
- 暂不做 instance-adaptive。

### CASE B — 多个 coarse scales 都有害，且越粗越差
结论：
- alpha 随 scale 单调增大的先验方向很可能错误；
- 下一阶段优先验证 unconstrained learnable per-scale alpha；
- 可考虑初始化为 fixed EMA 0.02。

### CASE C — 单尺度 perturbation都不明显，但 FULL02 明显恶化
结论：
- 主要是尺度间交互问题；
- 下一阶段考虑 jointly learnable scale vector，而不是逐尺度 deterministic rule。

### CASE D — pilot 结果混乱 / seed 不稳定
结论：
- 当前证据不足以继续设计复杂 adaptive 模块；
- 先扩 diagnostic seeds/horizon，不进入 instance-adaptive。

---

# 12. 本阶段明确不做

- 不进入 ETTm1 / Weather / Electricity；
- 不做 Stage 07；
- 不做 instance-adaptive；
- 不做 learnable alpha；
- 不做 channel-adaptive；
- 不做大范围 alpha search；
- 不根据 test 结果优化最终方法。

完成并 push bridge 结果后停止。
