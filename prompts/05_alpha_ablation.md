# Stage 05 — ETTh1 alpha 消融：同 alpha 配对比较 + Validation-Only 参数选择

## 前置条件

必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `results/03_integrate_timemixer.md`
- `results/04_smoke_training.md`

Stage 04 必须为 PASS。

当前 TSLib 正式实验代码基线：

- branch/worktree: `codex/saema-v1`
- SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

开始前要求：

- HEAD == 上述 SHA；
- `git status --short` 为空；
- 不修改 Stage 01 冻结的 SAEMA 数学定义；
- 不修改模型源码、DataLoader、optimizer、loss 或 seed logic。

如果当前状态与 Stage 04 不一致且无法解释，停止并报告。

---

# 1. 研究目标

Stage 05 是第一个真正用于方法选择的实验阶段。

要回答两个不同问题：

### Q1 — 同一个 alpha 下，Scale-Aware 换算是否带来额外信号？

对每一个相同的 `ema_alpha`：

`Fixed EMA(alpha) vs Scale-Aware EMA(alpha_base=alpha)`

比较 **validation loss**。

这用于隔离：

> “EMA → Scale-Aware EMA”

而不是让两种方法使用不同 alpha 后再比较。

### Q2 — 如果允许两种方法各自通过 validation 调一个 alpha，最佳配置分别是什么？

分别从完全相同的候选集合中选择：

- best Fixed EMA alpha；
- best Scale-Aware EMA alpha。

这两个值一旦选定，将被冻结用于 Stage 06，不允许在 Stage 06 根据 test / horizon / seed 再调整。

---

# 2. 实验配置

只使用：

- dataset: ETTh1
- task: long_term_forecast
- features: M
- seq_len: 96
- label_len: 0
- pred_len: 96
- seed: 2021
- e_layers: 2
- enc_in: 7
- c_out: 7
- d_model: 16
- d_ff: 32
- down_sampling_layers: 3
- down_sampling_window: 2
- down_sampling_method: avg
- channel_independence: 1
- moving_avg: 25
- use_norm: 1
- learning_rate: 0.01
- optimizer: existing Adam path
- loss: MSE
- batch_size: 128
- num_workers: 10
- train_epochs: **10**
- patience: 10
- AMP: off
- itr: 1

这是 Stage 00 已复现的正式 baseline 训练长度，不再使用 Stage 04 的 2 epochs。

除 decomposition / ema_alpha 外，所有 EMA candidate 配置必须完全一致。

---

# 3. Alpha candidate grid

Fixed EMA 和 Scale-Aware EMA **使用完全相同的候选集合**：

`ema_alpha ∈ {0.02, 0.05, 0.10, 0.15, 0.20}`

总共：

- 5 个 Fixed EMA runs；
- 5 个 Scale-Aware EMA runs。

另外跑 1 次 integrated-code `moving_avg` 10-epoch control。

总计计划：**11 个正式 Stage 05 runs**。

不得在看到结果后临时增加 0.08、0.12、0.25 等候选。

如果所有候选都表现异常，先完成并报告本阶段，不要擅自扩 grid；由 ChatGPT 审核后决定是否需要 Stage 05b。

---

# 4. Scale-Aware alpha table

在运行前就把以下理论 alpha 表写入报告，不能从结果反推：

| alpha_base | k=0 | k=1 | k=2 | k=3 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.020000 | 0.039600 | 0.077632 | 0.149237 |
| 0.05 | 0.050000 | 0.097500 | 0.185494 | 0.336580 |
| 0.10 | 0.100000 | 0.190000 | 0.343900 | 0.569533 |
| 0.15 | 0.150000 | 0.277500 | 0.477994 | 0.727509 |
| 0.20 | 0.200000 | 0.360000 | 0.590400 | 0.832228 |

运行日志中的实际 alpha 必须与该表一致（允许正常浮点显示差异）。

---

# 5. 核心规则：参数选择只能使用 Validation

这是 Stage 05 最重要的实验纪律。

## Primary selection metric

对每一个 run，定义：

`best_validation_loss = 训练过程中所有 epoch validation loss 的最小值`

即保存 best checkpoint 所依据的 validation loss。

Fixed EMA 的选择：

`selected_fixed_alpha = argmin_alpha(best_validation_loss_fixed(alpha))`

Scale-Aware EMA 的选择：

`selected_saema_alpha = argmin_alpha(best_validation_loss_saema(alpha))`

### Tie rule

若两个候选的 best validation loss 在 `1e-6` 内相同：

- 选择较小的 alpha；
- 并在报告中标记发生 tie。

不得使用以下信息打破 tie：

- test MSE；
- test MAE；
- Stage 04 smoke；
- 肉眼预测图；
- 训练时间。

---

# 6. Test-set leakage 控制

当前 TSLib runner 在训练过程中可能计算/打印 runner test loss，并在训练结束后调用 full test。

**这些 test 信息不得用于 alpha 选择。**

为了形成可审计证据，必须按以下两阶段顺序做汇总：

## Phase A — Validation selection

完成所有 10 个 EMA candidate 训练后：

1. 只解析每个 run 的：
   - epoch validation losses；
   - best validation loss；
   - best epoch；
   - checkpoint path；
2. 生成：
   `results/data/05_alpha_validation_selection.csv`
3. 用一个确定性的 selection script 或等价程序，仅从该 CSV 的 validation 列计算两个 selected alpha；
4. 把选择结果写入：
   `results/data/05_selected_alphas.json`

JSON 至少包含：

```json
{
  "selection_metric": "best_validation_loss",
  "fixed_ema_alpha": ...,
  "scale_aware_ema_alpha": ...,
  "candidate_grid": [0.02, 0.05, 0.10, 0.15, 0.20],
  "dataset": "ETTh1",
  "pred_len": 96,
  "seed": 2021
}
```

**先生成并固定这个 selection artifact，再整理 test 指标。**

如果方便，对 `05_selected_alphas.json` 记录 SHA-256 到 Markdown，证明后面的 test 汇总没有修改选择。

## Phase B — Test reporting

只有 Phase A selection artifact 固定之后：

- 报告 moving_avg control 的 test MSE/MAE；
- 报告 selected Fixed EMA 的 test MSE/MAE；
- 报告 selected SAEMA 的 test MSE/MAE。

对于**未被选中的 alpha candidate**：

- Markdown 主报告不要展示其 test MSE/MAE 排名；
- `05_alpha_validation_selection.csv` 不放 test columns；
- 即使原始训练 log 中 runner 自动打印了 test，也明确写明未用于 selection。

如果需要保留所有原始 runner logs，可以保留，但不要据此重新选择 alpha。

---

# 7. 同 alpha 的配对分析

对每一个 alpha，必须计算：

`delta_val(alpha) = best_val_saema(alpha) - best_val_fixed(alpha)`

解释：

- `delta_val < 0`：该 alpha 下 SAEMA validation 更低；
- `delta_val > 0`：该 alpha 下 Fixed EMA validation 更低。

报告一个 paired validation 表：

| alpha | Fixed EMA best val | SAEMA best val | SAEMA - Fixed |
|---:|---:|---:|---:|

并统计：

- SAEMA validation 更低的 alpha 个数 / 5；
- Fixed EMA validation 更低的 alpha 个数 / 5；
- exact/tolerance tie 个数。

这个表比“各自最佳 alpha 的 test 指标”更直接回答 Scale-Aware 机制是否有一致信号。

不要对 5 个点做夸大的显著性结论。

---

# 8. Moving-average control

在同一个 Stage 05 环境 / integrated SHA 下，跑一次：

`decomp_method=moving_avg`

完整 10 epochs，seed=2021。

目的：

- 给 validation 曲线一个 contemporaneous control；
- 检查 Stage 00 baseline 在当前冻结代码上的复现一致性；
- 不是 alpha selection 的候选。

记录：

- epoch-wise validation；
- best validation；
- best epoch；
- final test MSE/MAE。

同时与 Stage 00 的 pred_len=96 / seed=2021：

- MSE `0.376218`
- MAE `0.399333`

做数值对照。

如果存在明显差异，不要调整 EMA；先在报告解释环境/训练差异并标记。

---

# 9. Run ordering and artifact identity

建议固定顺序：

1. moving_avg control
2. ema alpha=0.02
3. scale_aware_ema alpha=0.02
4. ema alpha=0.05
5. scale_aware_ema alpha=0.05
6. ema alpha=0.10
7. scale_aware_ema alpha=0.10
8. ema alpha=0.15
9. scale_aware_ema alpha=0.15
10. ema alpha=0.20
11. scale_aware_ema alpha=0.20

这样每个 alpha 的两个方法相邻，便于资源条件审计。

所有 run 必须有唯一：

- model_id；
- des；
- log；
- checkpoint；
- result directory。

identity 至少编码：

- Stage05；
- method；
- alpha；
- seed；
- pred_len。

不能覆盖 Stage 00 或 Stage 04 artifacts。

---

# 10. GPU / environment

11 个 run 尽量使用同一物理 GPU。

开始前记录：

- GPU index；
- GPU model；
- 当前 memory / utilization；
- Python / PyTorch / CUDA；
- git SHA。

如果中途因其他任务不得不切 GPU：

- 记录在哪一个 run 发生；
- 不需要重跑只因 GPU 型号相同且配置一致；
- runtime 比较需标记资源变化；
- metric selection 仍只基于 validation。

不要让 GPU runtime 成为 alpha selection 标准。

---

# 11. 每个 candidate 记录

至少记录：

- method；
- ema_alpha；
- theoretical alpha table；
- seed；
- full command；
- epoch 1..10 train loss；
- epoch 1..10 validation loss；
- best validation loss；
- best epoch；
- checkpoint path；
- checkpoint successfully saved/reloaded；
- finite / NaN status；
- wall-clock；
- git SHA。

Test metric 在 Phase A selection CSV 中禁止出现。

---

# 12. Source-code policy

Stage 05 正常情况下 **不修改任何 TSLib source code**。

允许：

- 新增 Stage 05 run script；
- 新增结果解析/selection script；
- 新增日志。

如果发现真实实现 bug：

- 停止 alpha ablation；
- 不要边修 bug 边继续剩余候选；
- 写 BLOCKED/FAIL；
- 由 ChatGPT 审核后决定是否重启 Stage 05。

不要为了得到更好的 validation/test 修改模型或训练配置。

---

# 13. Required structured outputs

必须写：

### A. 主报告

`results/05_alpha_ablation.md`

### B. Validation-only candidate table

`results/data/05_alpha_validation_selection.csv`

至少列：

```text
method,
ema_alpha,
best_validation_loss,
best_epoch,
seed,
pred_len,
checkpoint_path,
finite,
wall_clock_seconds,
git_sha
```

**不得有 test_mse/test_mae 列。**

### C. Frozen selection

`results/data/05_selected_alphas.json`

### D. Final selected-model test summary

`results/data/05_selected_models_test.csv`

只包含：

- moving_avg control；
- selected fixed EMA；
- selected SAEMA。

列：

```text
method,
ema_alpha,
best_validation_loss,
best_epoch,
test_mse,
test_mae,
seed,
pred_len,
git_sha
```

---

# 14. Markdown 报告结构

至少包括：

## Environment
## Fixed experimental configuration
## Candidate grid
## Theoretical SAEMA alpha table
## Run completion / finite check
## Validation-only candidate results
## Same-alpha paired validation analysis
## Frozen alpha selection
## Selection artifact SHA
## Moving-average control
## Selected-model test report
## Runtime
## Source / Git status
## Final

---

# 15. Stage 05 Final 必须回答

明确填写：

- `Status: PASS | FAIL | BLOCKED`
- 11/11 runs completed: YES/NO
- selection used validation only: YES/NO
- test metrics used for tuning: YES/NO（正确答案必须是 NO）
- moving_avg control consistent with prior baseline: YES/NO/PARTIAL
- same-alpha SAEMA lower validation in: `X/5`
- selected fixed EMA alpha: `...`
- selected scale-aware EMA alpha: `...`
- selected alpha frozen for Stage 06: YES/NO
- source code changed: YES/NO
- ready for Stage 06: YES/NO

同时写出：

> The selected alpha values were chosen exclusively from ETTh1, pred_len=96, seed=2021 validation performance and will not be retuned on Stage 06 test results, other horizons, or other seeds.

Stage 06 后续会检验这些超参数能否泛化到其他 horizons 和 seeds。

完成并 push bridge 结果后停止。

**不要执行 Stage 06。**
