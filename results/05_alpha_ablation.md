# Stage 05 — ETTh1 alpha 消融：同 alpha 配对比较与 Validation-Only 选择

Status: **PASS**

- TSLib branch: `codex/saema-v1`
- TSLib SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`
- Bridge branch at start: `main`
- Bridge HEAD at start: `c99c980b1dcc27146abbf9a0c3f878df36d57a67`
- Formal runs completed: **11/11**
- Stage 06: **NOT EXECUTED**

本阶段严格执行 `CURRENT_STAGE.md` 指向的 `prompts/05_alpha_ablation.md`。所有 alpha 候选均在运行前固定；参数选择只使用 best validation loss。先生成 validation-only CSV，再由只读取该 CSV 的独立程序生成并冻结 selected-alpha JSON，之后才解析三种 selected/control 模型的 final test 指标。

## Environment

- Remote host: `ps`
- Worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- Python: `/home/ouyangguanghong/miniconda3/envs/tslib/bin/python` (`3.11.16`)
- PyTorch: `2.5.1+cu121`
- CUDA runtime: `12.1`
- NVIDIA driver: `570.211.01`
- GPU: 2 × NVIDIA GeForce RTX 4090 D, 24564 MiB each
- Training GPU: physical GPU 0 exposed through `CUDA_VISIBLE_DEVICES=0`; process argument `--gpu 0` therefore used `cuda:0`
- Run window: 2026-09-22 16:47:34–17:06:21 CST
- All 11 runs stayed on the same physical GPU 0.

两张 GPU 在本阶段开始前均已有同类后台 `main.py` 任务，各占约 744 MiB；GPU 0 的 per-run start utilization snapshots 为 33%–95%。没有停止、修改或迁移这些既有任务。该并发负载可能影响 wall-clock，因此 runtime 仅作审计记录，不参与 alpha 选择。

开始前门禁全部通过：

```text
HOST=ps
PROJECT=/home/ouyangguanghong/projects/Time-Series-Library-saema-v1
PYTHON=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python
TORCH=2.5.1+cu121
CUDA_AVAILABLE=True
CUDA_COUNT=2
branch=codex/saema-v1
HEAD=91735a83b27ce1bf766df76a953cf86ef1e7500d
git status --short=<empty>
```

## Fixed experimental configuration

| Item | Value |
|---|---|
| Task / data | `long_term_forecast`, `ETTh1`, `features=M` |
| Sequence | `seq_len=96`, `label_len=0`, `pred_len=96` |
| Seed | `2021` |
| Model | `TimeMixer`, `e_layers=2`, `enc_in=7`, `c_out=7`, `d_model=16`, `d_ff=32` |
| Multiscale | `down_sampling_layers=3`, `down_sampling_window=2`, `down_sampling_method=avg` |
| Channel / norm | `channel_independence=1`, `use_norm=1` |
| Moving-average control | `moving_avg=25` |
| Optimization | existing Adam path, MSE, learning rate `0.01`, `lradj=type1` |
| Training | `train_epochs=10`, `patience=10`, `batch_size=128`, `num_workers=10`, `itr=1` |
| AMP | off |
| Checkpoint root | `./checkpoints` in the frozen worktree |
| Dataset root | `/home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/` |

除 `decomp_method` 与 EMA candidate 的 `ema_alpha` 外，十个 EMA candidate 的配置完全相同。moving_avg control 使用同一环境、训练长度、seed、数据划分和模型配置。

## Candidate grid and run identity

Fixed EMA 与 Scale-Aware EMA 使用同一预注册集合：

`{0.02, 0.05, 0.10, 0.15, 0.20}`

没有在观察中间结果后新增、删除或替换候选。运行顺序固定如下：

| Order | Method | alpha | `model_id` suffix | `des` | Log |
|---:|---|---:|---|---|---|
| 01 | moving_avg | — | `Stage05Alpha_movingavg_seed2021` | `Stage05Alpha_MovingAvg_Seed2021` | `Stage05Alpha_01_movingavg_ETTh1_96_seed2021.log` |
| 02 | ema | 0.02 | `Stage05Alpha_ema_a0p02_seed2021` | `Stage05Alpha_EMA_a0p02_Seed2021` | `Stage05Alpha_02_ema_a0p02_ETTh1_96_seed2021.log` |
| 03 | scale_aware_ema | 0.02 | `Stage05Alpha_saema_a0p02_seed2021` | `Stage05Alpha_SAEMA_a0p02_Seed2021` | `Stage05Alpha_03_saema_a0p02_ETTh1_96_seed2021.log` |
| 04 | ema | 0.05 | `Stage05Alpha_ema_a0p05_seed2021` | `Stage05Alpha_EMA_a0p05_Seed2021` | `Stage05Alpha_04_ema_a0p05_ETTh1_96_seed2021.log` |
| 05 | scale_aware_ema | 0.05 | `Stage05Alpha_saema_a0p05_seed2021` | `Stage05Alpha_SAEMA_a0p05_Seed2021` | `Stage05Alpha_05_saema_a0p05_ETTh1_96_seed2021.log` |
| 06 | ema | 0.10 | `Stage05Alpha_ema_a0p10_seed2021` | `Stage05Alpha_EMA_a0p10_Seed2021` | `Stage05Alpha_06_ema_a0p10_ETTh1_96_seed2021.log` |
| 07 | scale_aware_ema | 0.10 | `Stage05Alpha_saema_a0p10_seed2021` | `Stage05Alpha_SAEMA_a0p10_Seed2021` | `Stage05Alpha_07_saema_a0p10_ETTh1_96_seed2021.log` |
| 08 | ema | 0.15 | `Stage05Alpha_ema_a0p15_seed2021` | `Stage05Alpha_EMA_a0p15_Seed2021` | `Stage05Alpha_08_ema_a0p15_ETTh1_96_seed2021.log` |
| 09 | scale_aware_ema | 0.15 | `Stage05Alpha_saema_a0p15_seed2021` | `Stage05Alpha_SAEMA_a0p15_Seed2021` | `Stage05Alpha_09_saema_a0p15_ETTh1_96_seed2021.log` |
| 10 | ema | 0.20 | `Stage05Alpha_ema_a0p20_seed2021` | `Stage05Alpha_EMA_a0p20_Seed2021` | `Stage05Alpha_10_ema_a0p20_ETTh1_96_seed2021.log` |
| 11 | scale_aware_ema | 0.20 | `Stage05Alpha_saema_a0p20_seed2021` | `Stage05Alpha_SAEMA_a0p20_Seed2021` | `Stage05Alpha_11_saema_a0p20_ETTh1_96_seed2021.log` |

日志根目录：

`/home/ouyangguanghong/projects/Time-Series-Library-saema-v1/experiment_logs/`

每个日志开头通过 shell trace 保存了完整展开命令。共同命令为：

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f 'STAGE05_WALL_SECONDS=%e' \
  /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path /home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id "ETTh1_96_96_<model_id suffix>" \
  --model TimeMixer --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 \
  --e_layers 2 --enc_in 7 --c_out 7 --d_model 16 --d_ff 32 \
  --learning_rate 0.01 --train_epochs 10 --patience 10 --batch_size 128 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --channel_independence 1 --decomp_method <method> [--ema_alpha <alpha>] \
  --moving_avg 25 --use_norm 1 --loss MSE --lradj type1 \
  --seed 2021 --gpu 0 --num_workers 10 \
  --des <des> --itr 1 --checkpoints ./checkpoints
```

上表给出每个 run 对 `<model_id suffix>`、`<method>`、`<alpha>`、`<des>` 的完整替换值；moving_avg 不传 `--ema_alpha`。因此 11 个命令和 artifact identity 均唯一，未覆盖 Stage 00 或 Stage 04 产物。

## Theoretical SAEMA alpha table

该表在运行前由 Stage 05 prompt 固定。运行日志中的 model-level alpha summary 与下表一致（允许正常浮点显示截断）：

| alpha_base | k=0 | k=1 | k=2 | k=3 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.020000 | 0.039600 | 0.077632 | 0.149237 |
| 0.05 | 0.050000 | 0.097500 | 0.185494 | 0.336580 |
| 0.10 | 0.100000 | 0.190000 | 0.343900 | 0.569533 |
| 0.15 | 0.150000 | 0.277500 | 0.477994 | 0.727509 |
| 0.20 | 0.200000 | 0.360000 | 0.590400 | 0.832228 |

Fixed EMA 日志则分别确认四个尺度均使用对应的同一 alpha。

## Run completion / finite check

- 11/11 runs exit code `0`；每个 run 均完成 epoch 1–10。
- 11/11 best checkpoints 存在，size 范围 655650–655906 bytes。
- runner 在 `train()` 末尾以默认 strict 行为重新加载 best checkpoint 后完成 full test；因此 11/11 runner reload 成功。
- 另以 CPU `torch.load(..., weights_only=True)` 读取 11/11 checkpoint，所有 tensor 均 finite。
- 11/11 `metrics.npy`、`pred.npy`、`true.npy` 均存在且 finite。
- 11/11 prediction/target shape 均为 `(2785, 96, 7)`。
- 所有 epoch train/validation loss 均 finite；没有 NaN/Inf、shape、device 或 runtime 错误。
- 每个 run 的 Git SHA 均为 `91735a83b27ce1bf766df76a953cf86ef1e7500d`。

### Epoch-wise train and validation trajectories

列表均按 epoch 1→10 排列；这些 validation 数值是 Phase A selection 的唯一性能输入。

| Run | Train loss, epochs 1–10 | Validation loss, epochs 1–10 | Best val / epoch |
|---|---|---|---:|
| moving_avg | 0.3916414, 0.3551973, 0.3501988, 0.3420170, 0.3337008, 0.3249417, 0.3217637, 0.3226453, 0.3208002, 0.3218563 | 0.7061212, 0.7199184, 0.7119444, 0.6873904, 0.6978562, 0.6909162, 0.6861665, 0.6890509, 0.6898097, 0.6903111 | 0.6861665 / 7 |
| ema 0.02 | 0.3909950, 0.3557024, 0.3497669, 0.3443899, 0.3369319, 0.3282897, 0.3250580, 0.3256183, 0.3244907, 0.3251291 | 0.7063309, 0.7040843, 0.7044353, 0.6844822, 0.6888519, 0.6918528, 0.6880117, 0.6911138, 0.6916886, 0.6917694 | 0.6844822 / 4 |
| scale_aware_ema 0.02 | 0.3914396, 0.3553651, 0.3494455, 0.3435533, 0.3351888, 0.3269827, 0.3235876, 0.3241272, 0.3224107, 0.3236878 | 0.7062598, 0.7014674, 0.7157342, 0.6891475, 0.6999907, 0.7031454, 0.6992528, 0.6986738, 0.7014196, 0.7007875 | 0.6891475 / 4 |
| ema 0.05 | 0.3915902, 0.3545638, 0.3478015, 0.3427280, 0.3353417, 0.3271839, 0.3236112, 0.3243675, 0.3228741, 0.3241699 | 0.7077508, 0.7050400, 0.7143857, 0.6907149, 0.6975959, 0.6944079, 0.6923429, 0.6946993, 0.6953146, 0.6954064 | 0.6907149 / 4 |
| scale_aware_ema 0.05 | 0.3915532, 0.3534892, 0.3482834, 0.3418943, 0.3341810, 0.3261515, 0.3233773, 0.3236595, 0.3224062, 0.3233783 | 0.7093427, 0.7301116, 0.7158952, 0.6875293, 0.6981254, 0.6996637, 0.6961871, 0.6963285, 0.6982264, 0.6985732 | 0.6875293 / 4 |
| ema 0.10 | 0.3917070, 0.3532233, 0.3482243, 0.3430632, 0.3351209, 0.3270805, 0.3241711, 0.3244893, 0.3231839, 0.3243309 | 0.7104271, 0.7164249, 0.7210445, 0.6893156, 0.7033125, 0.7026549, 0.6994442, 0.7010906, 0.7013811, 0.7018206 | 0.6893156 / 4 |
| scale_aware_ema 0.10 | 0.3910709, 0.3518818, 0.3466034, 0.3384492, 0.3298895, 0.3215986, 0.3182807, 0.3189779, 0.3168803, 0.3182320 | 0.7057169, 0.7201904, 0.7169077, 0.6959320, 0.7061671, 0.7091260, 0.7056475, 0.7083230, 0.7089520, 0.7097611 | 0.6959320 / 4 |
| ema 0.15 | 0.3914259, 0.3523308, 0.3471032, 0.3403820, 0.3336819, 0.3258301, 0.3229816, 0.3235814, 0.3219586, 0.3231322 | 0.7117513, 0.7148824, 0.7169357, 0.6948961, 0.7037464, 0.7057142, 0.7007396, 0.7024962, 0.7030084, 0.7034268 | 0.6948961 / 4 |
| scale_aware_ema 0.15 | 0.3907594, 0.3514335, 0.3452698, 0.3379401, 0.3288925, 0.3207365, 0.3175967, 0.3180241, 0.3162395, 0.3173891 | 0.7041303, 0.7253859, 0.7155999, 0.6963616, 0.7093795, 0.7127985, 0.7101237, 0.7124173, 0.7134761, 0.7137235 | 0.6963616 / 4 |
| ema 0.20 | 0.3912203, 0.3512979, 0.3457165, 0.3397212, 0.3324050, 0.3244611, 0.3216197, 0.3223638, 0.3207164, 0.3219546 | 0.7103001, 0.7164851, 0.7152714, 0.6934689, 0.7031778, 0.7074068, 0.7013830, 0.7036816, 0.7044821, 0.7048288 | 0.6934689 / 4 |
| scale_aware_ema 0.20 | 0.3905427, 0.3514995, 0.3447755, 0.3371253, 0.3290007, 0.3208942, 0.3177281, 0.3180509, 0.3165639, 0.3177572 | 0.7031753, 0.7188237, 0.7159921, 0.6940304, 0.7032530, 0.7088296, 0.7041209, 0.7075516, 0.7083927, 0.7085657 | 0.6940304 / 4 |

## Validation-only candidate results

`results/data/05_alpha_validation_selection.csv` 在任何 test 汇总之前生成。它恰有 10 行、覆盖两种方法的相同五点 grid，列为：

```text
method,ema_alpha,best_validation_loss,best_epoch,seed,pred_len,
checkpoint_path,finite,wall_clock_seconds,git_sha
```

该 CSV **没有** `test_mse` 或 `test_mae` 列。

| Method | alpha | Best validation loss | Best epoch | Finite | Wall-clock (s) |
|---|---:|---:|---:|---|---:|
| ema | 0.02 | **0.6844822** | 4 | yes | 93.88 |
| scale_aware_ema | 0.02 | 0.6891475 | 4 | yes | 101.03 |
| ema | 0.05 | 0.6907149 | 4 | yes | 90.26 |
| scale_aware_ema | 0.05 | **0.6875293** | 4 | yes | 95.05 |
| ema | 0.10 | 0.6893156 | 4 | yes | 100.76 |
| scale_aware_ema | 0.10 | 0.6959320 | 4 | yes | 99.68 |
| ema | 0.15 | 0.6948961 | 4 | yes | 95.02 |
| scale_aware_ema | 0.15 | 0.6963616 | 4 | yes | 91.66 |
| ema | 0.20 | 0.6934689 | 4 | yes | 91.61 |
| scale_aware_ema | 0.20 | 0.6940304 | 4 | yes | 86.60 |

粗体只表示各方法内部、由 validation 得到的选择；不是 test 排名。

## Same-alpha paired validation analysis

定义 `delta_val = best_val_saema - best_val_fixed`。负数表示相同 alpha 下 SAEMA validation 更低。

| alpha | Fixed EMA best val | SAEMA best val | SAEMA - Fixed | Lower validation |
|---:|---:|---:|---:|---|
| 0.02 | 0.6844822 | 0.6891475 | +0.0046653 | Fixed EMA |
| 0.05 | 0.6907149 | 0.6875293 | -0.0031856 | SAEMA |
| 0.10 | 0.6893156 | 0.6959320 | +0.0066164 | Fixed EMA |
| 0.15 | 0.6948961 | 0.6963616 | +0.0014655 | Fixed EMA |
| 0.20 | 0.6934689 | 0.6940304 | +0.0005615 | Fixed EMA |

- SAEMA validation lower: **1/5**
- Fixed EMA validation lower: **4/5**
- Exact/tolerance ties (`1e-6`): **0/5**

描述性结论：在这五个同-alpha 配对点上，Scale-Aware 换算只在 alpha 0.05 显示更低 validation；其余四点 Fixed EMA 更低。因此本阶段没有观察到跨 grid 一致的 SAEMA validation 优势。该结论仅针对 ETTh1、pred_len 96、seed 2021 的五个预注册点，不作显著性或广泛泛化声明。

## Frozen alpha selection

Phase A 的顺序与隔离如下：

1. 仅从 10 个 candidate 日志解析 epoch train/validation、best validation、best epoch、checkpoint、finite、wall-clock 与 Git SHA，生成 `05_alpha_validation_selection.csv`。
2. 第二个独立 selection 程序只打开该 CSV；它断言不存在 test 列、两种方法均恰有相同五点 grid，然后按 `best_validation_loss` 与 `1e-6` tie rule 选择 alpha。
3. 使用 exclusive-create 写出 `05_selected_alphas.json`，随后立即设置为 mode `0444` 并计算 SHA-256。
4. JSON 冻结后才运行 Phase B test parser；该 parser 从冻结 JSON 读取两个 selected alpha，只打开 control 与两个 selected candidate 的日志。

选择结果：

| Method | Selected alpha | Best validation loss | Best epoch | Tie |
|---|---:|---:|---:|---|
| Fixed EMA | **0.02** | 0.6844822 | 4 | no |
| Scale-Aware EMA | **0.05** | 0.6875293 | 4 | no |

没有使用 test MSE/MAE、Stage 04 smoke、图形、训练时间或 runtime 打破选择或 tie。

## Selection artifact SHA

- Frozen artifact: `results/data/05_selected_alphas.json`
- File mode immediately after generation: `0444`
- SHA-256: `8d24db3196126c02146117e9602e31e04682988651c4b593c4086c9c1628473e`
- Source validation CSV SHA-256: `82468e60469920b119d1fcfe7856b2c220aaf766b96644e1a13765748ec5ef7a2`

> The selected alpha values were chosen exclusively from ETTh1, pred_len=96, seed=2021 validation performance and will not be retuned on Stage 06 test results, other horizons, or other seeds.

## Moving-average control

- Best validation loss: `0.6861665` at epoch 7
- Final test MSE: `0.3762175738811493`
- Final test MAE: `0.3993326425552368`
- Stage 00 reference MSE / MAE: `0.376218 / 0.399333`
- Difference from Stage 00: MSE `-0.0000004261`, MAE `-0.0000003574`
- Consistency verdict: **YES** — the values reproduce Stage 00 to the precision shown there.

## Selected-model test report

该表在 selection JSON 冻结后生成，只包含 prompt 允许的三行。未被选择的 alpha 的 test MSE/MAE 不在本报告中展示、排序或解释。

| Method | alpha | Best validation loss / epoch | Test MSE | Test MAE |
|---|---:|---:|---:|---:|
| moving_avg control | — | 0.6861665 / 7 | 0.3762175738811493 | 0.3993326425552368 |
| selected Fixed EMA | 0.02 | 0.6844822 / 4 | 0.3768504559993744 | 0.3971256613731384 |
| selected Scale-Aware EMA | 0.05 | 0.6875293 / 4 | 0.3738960325717926 | 0.39574533700942993 |

这些 test 数值是冻结选择后的报告结果，不会反向改变 alpha。selected SAEMA 在本次 test 上低于 control 的 MSE/MAE；selected Fixed EMA 的 MSE 略高于 control、MAE 较低。由于选择依据只有单个 validation setting，本阶段不把这些 test 观察扩展为跨 seed/horizon 结论。

## Runtime

| Group | Runs | Total seconds | Mean seconds | Min–max seconds |
|---|---:|---:|---:|---:|
| moving_avg | 1 | 65.60 | 65.60 | 65.60–65.60 |
| Fixed EMA | 5 | 471.53 | 94.31 | 90.26–100.76 |
| Scale-Aware EMA | 5 | 474.02 | 94.80 | 86.60–101.03 |
| All | 11 | **1011.15** | 91.92 | 65.60–101.03 |

上述为每个命令自身的 `/usr/bin/time` wall-clock；因同机已有 GPU 任务，不用于方法效率结论或 alpha selection。

## Structured outputs

| Artifact | Rows / purpose | SHA-256 |
|---|---|---|
| `results/data/05_alpha_validation_selection.csv` | 10 candidate rows; validation only; no test columns | `82468e60469920b119d1fcfe7856b2c220aaf766b96644e1a13765748ec5ef7a2` |
| `results/data/05_selected_alphas.json` | frozen validation-only selection | `8d24db3196126c02146117e9602e31e04682988651c4b593c4086c9c1628473e` |
| `results/data/05_selected_models_test.csv` | control + two selected models only | `6483f6e04d09ec04346694d54e9cf1d5781e0e12cd2299f45ce51f1cd86f63ab4` |

`05_alpha_validation_selection.csv` 保存每个 candidate 的完整绝对 checkpoint path。原始日志、checkpoints 和 array artifacts 保留在 TSLib worktree 的 ignored experiment directories，不提交到 bridge。

## Source / Git status

- TSLib final branch: `codex/saema-v1`
- TSLib final SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`
- Final `git status --short`: empty
- Final `git diff --check`: PASS
- Model source, `run.py`, DataLoader, optimizer, loss, seed logic and tests changed in Stage 05: **NO**
- Only ignored runtime artifacts were added under `experiment_logs/`, `checkpoints/`, `results/` and `test_results/`.
- Bridge files added by Stage 05: this report and the three structured outputs above, plus the required final-response archive.
- Stage 06 was not launched.

## Final

- `Status: PASS`
- 11/11 runs completed: **YES**
- selection used validation only: **YES**
- test metrics used for tuning: **NO**
- moving_avg control consistent with prior baseline: **YES**
- same-alpha SAEMA lower validation in: **1/5**
- selected fixed EMA alpha: **0.02**
- selected scale-aware EMA alpha: **0.05**
- selected alpha frozen for Stage 06: **YES**
- source code changed: **NO**
- ready for Stage 06: **YES**
- Stage 06 executed: **NO**

Stage 05 is complete. Execution stops here as required.
