# Stage 06 — ETTh1 正式主实验：跨 Horizon / Seed 泛化 + 同 Alpha 机制对照

## 前置条件

必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `results/03_integrate_timemixer.md`
- `results/04_smoke_training.md`
- `results/05_alpha_ablation.md`
- `results/data/05_alpha_validation_selection.csv`
- `results/data/05_selected_alphas.json`
- `results/data/05_selected_models_test.csv`

Stage 05 必须为 PASS。

当前冻结代码：

- branch/worktree: `codex/saema-v1`
- SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

Stage 05 已冻结：

- selected Fixed EMA alpha = **0.02**
- selected Scale-Aware EMA alpha = **0.05**

这些 alpha **不得**根据 Stage 06 的 test、horizon 或 seed 重新调整。

---

# 1. 为什么 Stage 06 采用双轨设计

Stage 05 的 same-alpha validation 配对结果是：

- SAEMA lower validation: 1/5
- Fixed EMA lower validation: 4/5

因此目前不能把“各自最佳配置”的差异直接解释为 scale-aware 机制本身的贡献，因为：

- Fixed EMA 的 selected alpha = 0.02
- SAEMA 的 selected alpha = 0.05

所以 Stage 06 必须同时回答两个问题。

## Track A — Tuned-model performance

比较：

1. Moving Average baseline
2. Fixed EMA @ alpha=0.02
3. Scale-Aware EMA @ alpha=0.05

目的：

> 在 Stage 05 各自仅由 validation 选出的超参数冻结后，两个 EMA variant 能否跨 horizon / seed 泛化？

这里允许两个方法使用各自 selected alpha，因为这是“模型最终配置”的公平比较。

但 Track A **不能单独用于声称 scale-aware 机制本身优于 fixed EMA**。

## Track B — Mechanism-controlled comparison

比较同一个 base alpha：

1. Fixed EMA @ alpha=0.05
2. Scale-Aware EMA @ alpha=0.05

目的：

> 在完全相同的 alpha_base 下，仅打开/关闭 scale-aware coefficient conversion 后，结果如何变化？

这个 Track 才是 Stage 06 中隔离 scale-aware mechanism 的主要证据。

选择 0.05 是因为它是 Stage 05 通过 validation-only procedure 冻结得到的 SAEMA operating point，不是根据 Stage 06 结果选择。

---

# 2. 数据与实验矩阵

Dataset：

`ETTh1`

Prediction horizons：

`{96, 192, 336, 720}`

Seeds：

`{2021, 2022, 2023}`

共 12 个 horizon × seed cells。

## Config A — Moving Average baseline

`decomp_method=moving_avg`
`moving_avg=25`

12 cells。

## Config B — Selected Fixed EMA

`decomp_method=ema`
`ema_alpha=0.02`

12 cells。

## Config C — Selected SAEMA

`decomp_method=scale_aware_ema`
`ema_alpha=0.05`

12 cells。

对应 SAEMA scale coefficients：

- k0 = 0.050000
- k1 = 0.097500
- k2 = 0.18549375
- k3 = 0.3365795687

## Config D — Same-alpha Fixed EMA control

`decomp_method=ema`
`ema_alpha=0.05`

12 cells。

Config C vs D 是机制控制对照。

---

# 3. Baseline 与 Stage 05 已有结果的复用原则

## 3.1 Moving Average baseline

Stage 00 已有完整：

- 4 horizons × 3 seeds = 12 runs

并且后续已经证明：

- Stage 03 integrated moving_avg 与 baseline-freeze 数值路径 bitwise exact；
- Stage 05 moving_avg / horizon96 / seed2021 精确复现 Stage 00 指标到记录精度。

因此，如果逐项核对以下内容全部一致，可以**复用 Stage 00 的 12 个 moving_avg metrics**：

- dataset / split；
- seq_len / label_len；
- each pred_len；
- architecture；
- training epochs；
- lr / optimizer / loss；
- batch size；
- seed；
- deterministic DataLoader logic；
- moving_avg=25；
- downsampling；
- channel_independence；
- normalization。

报告必须写出这项核对。

如果发现任何会影响数值可比性的差异，则不要部分复用，改为重跑 12 个 moving_avg cells。

若复用 baseline：

- 不把旧 baseline wall-clock 与当前 EMA runtime 做效率结论；
- baseline runtime 可留空/NA。

## 3.2 Stage 05 overlap reuse

Stage 05 已经在同一冻结 SHA、正式 10 epoch 配置下跑过以下 ETTh1 / horizon96 / seed2021：

- Fixed EMA alpha=0.02
- SAEMA alpha=0.05
- Fixed EMA alpha=0.05

如果完整配置逐项一致，可直接复用这 3 个 cell：

- B: pred96 seed2021
- C: pred96 seed2021
- D: pred96 seed2021

不要为了形式重复计算。

复用时必须记录：

- source Stage 05 log；
- checkpoint；
- metrics；
- best validation；
- git SHA；
- config-equivalence verdict。

因此，若所有复用条件成立：

- baseline A: 12 reused
- B/C/D: 各 1 reused
- new Stage 06 runs = **33**

---

# 4. 固定训练配置

除 method / alpha / pred_len / seed 外，保持 Stage 00 baseline：

- task: long_term_forecast
- data: ETTh1
- features: M
- seq_len=96
- label_len=0
- e_layers=2
- enc_in=7
- c_out=7
- d_model=16
- d_ff=32
- down_sampling_layers=3
- down_sampling_window=2
- down_sampling_method=avg
- channel_independence=1
- use_norm=1
- learning_rate=0.01
- optimizer=existing Adam path
- loss=MSE
- train_epochs=10
- patience=10
- batch_size=128
- num_workers=10
- AMP=off
- itr=1

不得针对 horizon / seed / method 改训练超参数。

---

# 5. Artifact identity

所有新 run 必须有唯一：

- model_id
- des
- log
- checkpoint
- result directory

identity 至少编码：

- Stage06
- config label A/B/C/D 或 method
- alpha（适用时）
- pred_len
- seed

不得覆盖 Stage 00 / 04 / 05 artifacts。

---

# 6. GPU / 并行策略

优先保证配对公平，而不是最快完成。

可以使用两张相同 RTX 4090 D，但必须遵守：

- 同一个 paired comparison 中的两个方法尽量处于相近资源条件；
- 如果并行，明确记录 physical GPU；
- 相同 seed 不要求强制同 GPU，但 GPU 型号/软件栈必须一致；
- runtime 只有在资源条件可比时才解释；
- metrics 不因 GPU runtime 资源差异做筛选。

如果后台任务造成严重资源竞争：

- 记录；
- 不使用 wall-clock 做模型优劣结论。

---

# 7. 每个 run 必须记录

对于每个 method × horizon × seed：

- method/config label；
- ema_alpha；
- pred_len；
- seed；
- git SHA；
- full command；
- physical GPU；
- epoch-wise train loss；
- epoch-wise validation loss；
- best validation loss；
- best epoch；
- checkpoint saved；
- checkpoint reloaded；
- final test MSE；
- final test MAE；
- finite status；
- wall-clock；
- reused/new；
- reused source（适用时）。

不得只保存三 seed mean；必须保留每一个 seed 原始数据。

---

# 8. Structured raw table

写：

`results/data/06_etth1_full_benchmark.csv`

至少包含：

```text
config,
method,
ema_alpha,
pred_len,
seed,
best_validation_loss,
best_epoch,
test_mse,
test_mae,
finite,
checkpoint_path,
wall_clock_seconds,
gpu,
reused,
source_stage,
git_sha
```

其中 config：

- A_moving_avg
- B_fixed_selected_a0p02
- C_saema_selected_a0p05
- D_fixed_control_a0p05

最终应有：

`4 configs × 4 horizons × 3 seeds = 48 rows`

即使 A 大部分来自 Stage 00、部分 B/C/D 来自 Stage 05，也都统一整理进同一个 CSV，并通过 `reused/source_stage` 标记来源。

---

# 9. Track A 汇总：最终配置泛化

比较：

- A moving_avg
- B Fixed EMA alpha=0.02
- C SAEMA alpha=0.05

## 9.1 每个 horizon

对 3 seeds 分别计算：

- mean MSE；
- sample std MSE；
- mean MAE；
- sample std MAE。

并计算相对 baseline A：

`relative_mse_change_% = 100 * (method_mean_mse - baseline_mean_mse) / baseline_mean_mse`

`relative_mae_change_% = 100 * (method_mean_mae - baseline_mean_mae) / baseline_mean_mae`

负值表示误差降低。

## 9.2 12-cell paired signs

对同一个 horizon+seed：

- B - A MSE / MAE；
- C - A MSE / MAE；
- C - B MSE / MAE。

统计：

- lower / higher / tie cell counts；
- mean paired delta；
- median paired delta；
- mean relative % delta。

注意：

`C vs B` 在 Track A 中包含 alpha 不同这一混杂，因此只能描述“冻结后的模型配置差异”，不能称为纯 scale-aware mechanism effect。

## 9.3 Four-horizon aggregate

对每个 horizon 的 3-seed mean 再做简单四-horizon平均：

- average MSE；
- average MAE。

同时报告 12 cell pooled descriptive mean 可以，但必须标清定义，不要和“four-horizon mean of means”混淆。

---

# 10. Track B 汇总：Scale-Aware 机制控制实验

核心比较：

- D Fixed EMA alpha=0.05
- C SAEMA alpha=0.05

这是 Stage 06 最重要的机制归因分析。

对每个 horizon：

- 3-seed mean ± sample std；
- `C - D` MSE；
- `C - D` MAE；
- relative MSE change %；
- relative MAE change %。

对全部 12 paired cells：

- SAEMA lower MSE count / 12；
- Fixed lower MSE count / 12；
- tie count；
- SAEMA lower MAE count / 12；
- mean paired MSE delta；
- median paired MSE delta；
- mean relative MSE delta %；
- 对 MAE 同样计算。

### Interpretation rule

只有当 C vs D 在多个 horizons / seeds 上呈现较一致方向时，才可以描述：

> scale-aware conversion shows a consistent empirical benefit under the shared alpha=0.05 condition.

如果结果混合，必须如实写：

> no consistent mechanism-level advantage was observed under the shared-alpha control.

不要因为 Track A 的 SAEMA selected config 更好，就覆盖 Track B 的负面/混合结果。

---

# 11. Statistics

样本量很小且 12 cells 跨不同 horizons，不要求做夸大的显著性检验。

优先报告：

- mean；
- sample std；
- median paired delta；
- sign counts；
- relative change。

可以附加 exploratory paired test，但若做：

- 必须标为 exploratory；
- 不作为主要结论；
- 说明 horizon heterogeneity；
- 不因 p-value 改写描述性结果。

不要只报告 best seed。

---

# 12. Validation stability

除了 test 指标，汇总：

- 各 method/config 的 best validation loss；
- best epoch 分布；
- 是否出现训练不稳定 / NaN / checkpoint failure。

特别检查：

- Stage 05 selected alpha 是否在 seed 2022/2023 仍表现合理；
- 不允许因 seed 2022/2023 validation 不理想而重新选 alpha。

---

# 13. Source-code policy

Stage 06 正常情况下不得修改 TSLib source。

如果发现真实 bug：

1. 立即停止未完成 runs；
2. 标记 BLOCKED；
3. 不要一边修 bug 一边继续；
4. 把 bug、受影响 run、建议修复写入报告；
5. 等 ChatGPT 审核。

不得为了提升 benchmark 指标修改方法或配置。

---

# 14. Required outputs

必须写：

### 主报告
`results/06_etth1_full_benchmark.md`

### Raw 48-cell table
`results/data/06_etth1_full_benchmark.csv`

### Track A summary
`results/data/06_trackA_selected_models_summary.csv`

### Track B mechanism control
`results/data/06_trackB_same_alpha_mechanism.csv`

推荐另外保存：

`results/data/06_paired_deltas.csv`

用于后续画图与论文表格。

---

# 15. Markdown 报告结构

至少包括：

## Environment / Git
## Frozen hyperparameters from Stage 05
## Reuse audit
## Experiment matrix
## Run completion / finite check
## Raw result integrity
## Track A — Selected-model performance
## Track A — Per-horizon mean ± std
## Track A — 12-cell paired analysis
## Track B — Same-alpha mechanism control
## Track B — Per-horizon mean ± std
## Track B — 12-cell paired mechanism analysis
## Validation stability
## Runtime caveat
## Source status
## Interpretation boundaries
## Final

---

# 16. Final 必须回答

- `Status: PASS | FAIL | BLOCKED`
- 48/48 matrix complete: YES/NO
- newly executed runs: N
- reused Stage 00 baseline cells: N
- reused Stage 05 cells: N
- source code changed: YES/NO

### Track A
- Fixed EMA selected config lower MSE than baseline in: X/12 cells
- SAEMA selected config lower MSE than baseline in: X/12 cells
- SAEMA selected config lower MSE than Fixed selected config in: X/12 cells
- four-horizon mean MSE/MAE for A/B/C

### Track B
- same-alpha SAEMA lower MSE than Fixed EMA in: X/12 cells
- same-alpha SAEMA lower MAE than Fixed EMA in: X/12 cells
- horizons with lower mean MSE under SAEMA: X/4
- mechanism-level direction: CONSISTENT_POSITIVE / MIXED / CONSISTENT_NEGATIVE

### Decision
- evidence supports proceeding to multi-dataset pilot: YES/NO
- alpha retuned in Stage 06: **NO**
- ready for Stage 07: YES/NO

报告最后必须明确区分：

> Track A evaluates the frozen, validation-selected model configurations and includes different alpha values.

> Track B holds alpha_base fixed at 0.05 and is the primary Stage 06 evidence for the scale-aware mechanism itself.

完成并 push bridge 结果后停止。

**不要执行 Stage 07。**
