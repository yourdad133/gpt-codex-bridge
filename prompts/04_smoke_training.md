# Stage 04 — ETTh1 三种 decomposition 的完整训练链路 Smoke Test

## 前置条件

必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `results/03_integrate_timemixer.md`

Stage 03 必须为 PASS。

当前 TSLib 集成基线：

- branch/worktree: `codex/saema-v1`
- Stage 03 final SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

开始前要求：

- HEAD == 上述 SHA；
- `git status --short` 为空；
- Stage 03 的 74 个 tests 仍可通过，至少在正式 smoke 前重跑一次。

如果状态不一致且无法解释，停止并报告。

---

# 1. 目标

确认以下三种 decomposition 都能完整走通：

`train -> validation -> best checkpoint save -> best checkpoint reload -> test -> MSE/MAE output`

三种方法：

A. original `moving_avg`  
B. fixed `ema`  
C. `scale_aware_ema`

本阶段是**训练管线 smoke test**，不是正式 benchmark。

本阶段的 MSE/MAE：

- 可以记录；
- 只能用于确认数值有限、流程正确；
- **禁止据此判断哪种方法更好**；
- **禁止据此调整 alpha**；
- 不作为 Stage 05 alpha 选择依据。

---

# 2. 固定实验配置

使用此前已经复现的 ETTh1 TimeMixer baseline 核心配置：

- task: `long_term_forecast`
- dataset: ETTh1
- features: M
- seq_len: 96
- label_len: 0
- pred_len: 96
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
- optimizer: existing Adam path
- learning_rate: 0.01
- loss: MSE
- batch_size: 128
- seed: 2021
- num_workers: 10
- itr: 1
- AMP: 与已复现 baseline 一致，保持关闭
- patience: 10
- train_epochs: **2**

除 decomposition 外，三次实验配置必须一致。

Stage 04 不允许为了某个 EMA 运行成功而单独改变：

- batch size；
- learning rate；
- epoch；
- seed；
- architecture；
- data split；
- optimizer；
- scheduler；
- early stopping；
- normalization；
- GPU 数；
- num_workers。

---

# 3. EMA 配置

严格使用已冻结 CLI：

`--ema_alpha 0.10`

不要使用 `--ema_alpha_base`。

### Fixed EMA

```text
--decomp_method ema
--ema_alpha 0.10
```

所有 scale 使用 0.10。

### Scale-Aware EMA

```text
--decomp_method scale_aware_ema
--ema_alpha 0.10
```

对于：

- w=2
- down_sampling_layers=3

启动日志应出现约：

- k=0: 0.100000
- k=1: 0.190000
- k=2: 0.343900
- k=3: 0.569533

### Moving Average

显式写：

```text
--decomp_method moving_avg
--moving_avg 25
```

即使这些是默认值，也建议 Stage 04 命令显式指定，保证三次运行的审计记录清楚。

---

# 4. GPU 与运行隔离

开始三次实验前检查 GPU。

要求：

- 三种方法尽量使用同一物理 GPU；
- 若使用 `CUDA_VISIBLE_DEVICES=<physical_gpu>`，进程内部使用 `--gpu 0`；
- 不再使用 Stage 00 曾失败的“只暴露一张 GPU 但内部指定 cuda:1”组合；
- 记录物理 GPU index、型号和进程内映射。

如果运行期间该 GPU 被其他重负载任务占用，停止或换到同一条件的空闲 GPU，记录原因；不要让三个方法处于明显不同的资源竞争环境。

---

# 5. Artifact 隔离

三个 smoke run 必须使用不同的：

- `model_id`
- `des`
- log filename
- checkpoint setting/directory
- test/result directory

建议 identity 明确包含：

- Stage04Smoke
- method
- alpha（EMA mode）
- seed
- pred_len

例如语义上：

`ETTh1_96_96_Stage04Smoke_SAEMA_a0p10_seed2021`

不要覆盖：

- Stage 00 baseline checkpoints；
- Stage 00 logs；
- Stage 03 smoke artifacts；
- 后续 Stage 05 正式 alpha ablation artifacts。

---

# 6. 三个实际实验

按固定顺序运行，便于审计：

1. `moving_avg`
2. `ema`
3. `scale_aware_ema`

三者均运行完整 2 epochs，不能只跑 batch smoke。

每个 run 必须成功完成：

1. train loader creation；
2. 两个 training epochs；
3. 每 epoch validation；
4. checkpoint save；
5. best checkpoint reload；
6. full test set inference；
7. final MSE / MAE output。

---

# 7. 每个 run 必须记录

## Configuration

- full command；
- git SHA；
- decomposition method；
- ema_alpha；
- scale-aware actual alpha table（适用时）；
- seed；
- GPU mapping；
- Python / torch / CUDA。

## Training

- epoch 1 train loss；
- epoch 1 validation loss；
- epoch 2 train loss；
- epoch 2 validation loss；
- test loss（若 runner 输出）；
- learning rate 变化（若日志中有）；
- 是否发生 early stopping；
- 是否存在 NaN/Inf。

## Checkpoint

- checkpoint directory；
- `checkpoint.pth` 是否存在；
- file size；
- best epoch（可从日志可靠确定时记录）；
- best checkpoint 是否成功 reload；
- reload 后 test 是否成功。

## Test

- test output shape（能可靠获得时）；
- final MSE；
- final MAE；
- MSE/MAE 是否 finite。

## Runtime

- start/end timestamp；
- wall-clock duration；
- peak GPU memory（若能低成本可靠获取）；
- GPU resource note。

---

# 8. 健康检查规则

本阶段 PASS 的条件不是“EMA 指标更好”。

每种方法只需满足：

- 训练正常下降/至少数值稳定；
- validation 数值 finite；
- checkpoint 成功保存；
- checkpoint 成功 reload；
- full test 成功；
- MSE/MAE finite；
- 无 shape/device/runtime error；
- 无 NaN/Inf。

即使：

`EMA MSE > moving_avg MSE`

或者：

`SAEMA MSE > EMA MSE`

Stage 04 仍然可以 PASS。

**绝不能因为 smoke 指标不好就在本阶段改 alpha。**

---

# 9. Stage 03 regression protection

训练前执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_ema_decomp.py \
  tests/test_timemixer_ema_integration.py
```

预期：

`74 passed`

训练后如果没有源码修改，不必重复全部 suite；如任何源码因真实 bug 被修改：

1. 必须单独 commit；
2. 报告 bug 根因；
3. 重跑全部 74 tests；
4. 重做受影响 smoke run；
5. 明确说明是否改变 Frozen v1 语义。

不得为了提升 MSE/MAE 修改实现。

---

# 10. Source-code policy

正常情况下 Stage 04 **不应修改模型源码**。

允许：

- 新增非侵入式 smoke command script；
- 新增日志；
- 新增结果汇总文件。

若不需要脚本，优先直接使用明确记录的 CLI 命令。

不要修改：

- EMA 数学；
- TimeMixer architecture；
- optimizer/loss；
- DataLoader；
- baseline seed logic。

---

# 11. 结构化 CSV

除 Markdown 报告外，写入：

`results/data/04_smoke_training.csv`

至少包含列：

```text
method,
ema_alpha,
pred_len,
seed,
epochs,
train_loss_epoch1,
val_loss_epoch1,
train_loss_epoch2,
val_loss_epoch2,
test_mse,
test_mae,
checkpoint_saved,
checkpoint_reloaded,
finite,
wall_clock_seconds,
git_sha
```

对于 moving_avg，`ema_alpha` 留空或 NA，不要填成有实际意义的 0.10。

Scale-Aware 的各尺度 alpha 可以在 Markdown 里单独表格记录，不必塞入单列 CSV。

---

# 12. 输出

写入 bridge：

`results/04_smoke_training.md`

和：

`results/data/04_smoke_training.csv`

Markdown 必须包含：

## Environment
## Pre-training regression tests
## Commands
## Moving Average Smoke
## Fixed EMA Smoke
## Scale-Aware EMA Smoke
## Checkpoint Verification
## Finite / NaN Check
## Runtime
## Non-comparative Metrics Table
## Git / Source Changes
## Final

Final 必须明确：

- `Status: PASS | FAIL | BLOCKED`
- moving_avg full pipeline: PASS/FAIL
- fixed EMA full pipeline: PASS/FAIL
- scale-aware EMA full pipeline: PASS/FAIL
- source code changed during Stage 04: YES/NO
- ready for Stage 05 alpha ablation: YES/NO

在 Non-comparative Metrics Table 附近必须写明：

> Stage 04 metrics are smoke-test observations only and were not used to select a method or tune alpha.

完成并 push bridge 结果后停止。

**不要执行 Stage 05。**
