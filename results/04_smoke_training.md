# Stage 04 — ETTh1 三种 decomposition 的完整训练链路 Smoke Test

Status: **PASS**

TSLib branch: `codex/saema-v1`  
TSLib SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`  
Bridge branch at start: `main`  
Bridge SHA at start: `2853bca26f493adc6db6e135b4e52d3717c36910`

本阶段严格执行 `CURRENT_STAGE.md` 指向的 `prompts/04_smoke_training.md`，只做 ETTh1、`seq_len=96`、`pred_len=96`、seed 2021 的三组 2-epoch 完整训练链路。Stage 05 未执行。

## Environment

- Remote host: `ps`
- TSLib worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- Python: `/home/ouyangguanghong/miniconda3/envs/tslib/bin/python`
- PyTorch: `2.5.1+cu121`
- CUDA available: `True`
- Visible GPU count in the pre-training check: `2`
- GPU: 2 × NVIDIA GeForce RTX 4090 D, driver `570.211.01`, 24564 MiB each
- Training GPU mapping: physical GPU 0 exposed as `CUDA_VISIBLE_DEVICES=0`; process argument `--gpu 0` therefore used `cuda:0`.
- GPU 1 remained occupied by another task. All three runs stayed on the same physical GPU 0. GPU 0 had a residual 744 MiB process with intermittent utilization; the initial snapshots were recorded in each log. No GPU switch occurred between methods.
- Data root: `/home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/`. The implementation worktree does not contain a second dataset copy, so the already audited ETTh1 data directory was passed explicitly. This did not modify source code or the data split.
- AMP: disabled. `num_workers=10`, batch size `128`, Adam, learning rate `0.01`, `lradj=type1`, patience `10`, and all other baseline settings were held constant.

## Pre-training regression tests

The required final command was:

```bash
cd /home/ouyangguanghong/projects/Time-Series-Library-saema-v1
source /home/ouyangguanghong/miniconda3/etc/profile.d/conda.sh
conda activate tslib
export CUDA_VISIBLE_DEVICES=0,1
TIMEMIXER_BASELINE_REGRESSION_PAYLOAD=/tmp/timemixer_stage03_regression_payload.pt \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q -rs \
  tests/test_ema_decomp.py \
  tests/test_timemixer_ema_integration.py
```

Result: **74 passed in 2.59s**. Two PyTorch `torch.load` FutureWarnings were emitted; no test failed. The first check collected 74 tests but reported `73 passed, 1 skipped` because the Stage 03 cross-worktree regression payload is intentionally supplied through an environment variable and is not stored in the repository. A temporary payload was generated from the baseline-freeze worktree, the complete suite was rerun, and the temporary file was removed. No model source or test source was changed.

## Commands

All three commands used the same baseline configuration and the same physical GPU. The only experimental differences were decomposition mode, the required EMA alpha argument, and unique artifact identities.

### Moving average

```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path /home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_Stage04Smoke_movingavg_seed2021 \
  --model TimeMixer --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 \
  --e_layers 2 --enc_in 7 --c_out 7 --d_model 16 --d_ff 32 \
  --learning_rate 0.01 --train_epochs 2 --patience 10 --batch_size 128 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --decomp_method moving_avg --moving_avg 25 --use_norm 1 --loss MSE \
  --seed 2021 --gpu 0 --num_workers 10 \
  --des Stage04Smoke_MovingAvg_Seed2021 --itr 1 --checkpoints ./checkpoints \
  2>&1 | tee experiment_logs/Stage04Smoke_moving_avg_ETTh1_96_seed2021.log
```

### Fixed EMA

```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path /home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_Stage04Smoke_ema_a0p10_seed2021 \
  --model TimeMixer --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 \
  --e_layers 2 --enc_in 7 --c_out 7 --d_model 16 --d_ff 32 \
  --learning_rate 0.01 --train_epochs 2 --patience 10 --batch_size 128 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --decomp_method ema --ema_alpha 0.10 --moving_avg 25 --use_norm 1 --loss MSE \
  --seed 2021 --gpu 0 --num_workers 10 \
  --des Stage04Smoke_EMA_a0p10_Seed2021 --itr 1 --checkpoints ./checkpoints \
  2>&1 | tee experiment_logs/Stage04Smoke_ema_a0p10_ETTh1_96_seed2021.log
```

### Scale-Aware EMA

```bash
CUDA_VISIBLE_DEVICES=0 /home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path /home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_Stage04Smoke_scaleaware_a0p10_seed2021 \
  --model TimeMixer --data ETTh1 --features M \
  --seq_len 96 --label_len 0 --pred_len 96 \
  --e_layers 2 --enc_in 7 --c_out 7 --d_model 16 --d_ff 32 \
  --learning_rate 0.01 --train_epochs 2 --patience 10 --batch_size 128 \
  --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 \
  --decomp_method scale_aware_ema --ema_alpha 0.10 --moving_avg 25 --use_norm 1 --loss MSE \
  --seed 2021 --gpu 0 --num_workers 10 \
  --des Stage04Smoke_SAEMA_a0p10_Seed2021 --itr 1 --checkpoints ./checkpoints \
  2>&1 | tee experiment_logs/Stage04Smoke_scale_aware_ema_a0p10_ETTh1_96_seed2021.log
```

The first moving-average launch was aborted before training because the dedicated log directory had not yet been created and the relative worktree data path triggered an unavailable Hugging Face fallback. The successful rerun created `experiment_logs/` first and used the explicit audited data root above. This did not affect any baseline artifact.

## Moving Average Smoke

- Pipeline: **PASS**
- Configuration: `decomp_method=moving_avg`, `moving_avg=25`, no meaningful EMA alpha
- Epoch 1: train `0.3916414`, validation `0.7061211`, runner test loss `0.4002173`
- Epoch 2: train `0.3551973`, validation `0.7199184`, runner test loss `0.3857801`
- Early stopping: not triggered; both epochs completed, counter `1/10` after epoch 2
- Best epoch: 1; best checkpoint saved after validation
- Full test output shape: `(2785, 96, 7)` for both predictions and targets
- Final full-test MSE: `0.39903831481933594`
- Final full-test MAE: `0.4057489037513733`
- Log: `experiment_logs/Stage04Smoke_moving_avg_ETTh1_96_seed2021.log` (`3395` bytes)

## Fixed EMA Smoke

- Pipeline: **PASS**
- Configuration: `decomp_method=ema`, `ema_alpha=0.10`
- Actual alpha table: `k=0..3 -> [0.10, 0.10, 0.10, 0.10]`
- Epoch 1: train `0.3917070`, validation `0.7104271`, runner test loss `0.3951078`
- Epoch 2: train `0.3532233`, validation `0.7164249`, runner test loss `0.3822091`
- Early stopping: not triggered; both epochs completed, counter `1/10` after epoch 2
- Best epoch: 1; best checkpoint saved after validation
- Best checkpoint was reloaded by the existing `Exp_Long_Term_Forecast.train()` path before the runner's full test call; an additional strict CPU reload check also passed with empty missing/unexpected key lists.
- Full test output shape: `(2785, 96, 7)` for both predictions and targets
- Final full-test MSE: `0.39403727650642395`
- Final full-test MAE: `0.4015093445777893`
- Log: `experiment_logs/Stage04Smoke_ema_a0p10_ETTh1_96_seed2021.log` (`3722` bytes)

## Scale-Aware EMA Smoke

- Pipeline: **PASS**
- Configuration: `decomp_method=scale_aware_ema`, `ema_alpha=0.10`
- Actual alpha table for `w=2`, `down_sampling_layers=3`: `k=0..3 -> [0.100000, 0.190000, 0.343900, 0.569533]`
- Epoch 1: train `0.3910709`, validation `0.7057169`, runner test loss `0.3953349`
- Epoch 2: train `0.3518818`, validation `0.7201904`, runner test loss `0.3793044`
- Early stopping: not triggered; both epochs completed, counter `1/10` after epoch 2
- Best epoch: 1; best checkpoint saved after validation
- Best checkpoint was reloaded by the existing `Exp_Long_Term_Forecast.train()` path before the runner's full test call; an additional strict CPU reload check also passed with empty missing/unexpected key lists.
- Full test output shape: `(2785, 96, 7)` for both predictions and targets
- Final full-test MSE: `0.3943324387073517`
- Final full-test MAE: `0.4033014178276062`
- Log: `experiment_logs/Stage04Smoke_scale_aware_ema_a0p10_ETTh1_96_seed2021.log` (`3769` bytes)

## Checkpoint Verification

The three full setting paths were distinct and did not overlap any Stage 00 or Stage 03 artifact:

| Method | Checkpoint | Size | Saved | Strict reload | Result arrays |
|---|---|---:|---|---|---|
| moving_avg | `checkpoints/long_term_forecast_ETTh1_96_96_Stage04Smoke_movingavg_seed2021_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Stage04Smoke_MovingAvg_Seed2021_0/checkpoint.pth` | 655906 B | PASS | PASS | `metrics.npy`, `pred.npy`, `true.npy` |
| ema | `checkpoints/long_term_forecast_ETTh1_96_96_Stage04Smoke_ema_a0p10_seed2021_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Stage04Smoke_EMA_a0p10_Seed2021_0/checkpoint.pth` | 655650 B | PASS | PASS | `metrics.npy`, `pred.npy`, `true.npy` |
| scale_aware_ema | `checkpoints/long_term_forecast_ETTh1_96_96_Stage04Smoke_scaleaware_a0p10_seed2021_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Stage04Smoke_SAEMA_a0p10_Seed2021_0/checkpoint.pth` | 655650 B | PASS | PASS | `metrics.npy`, `pred.npy`, `true.npy` |

Each `metrics.npy`, `pred.npy`, and `true.npy` was present. Each prediction/target array had shape `(2785, 96, 7)`. The strict reload verification instantiated the corresponding mode and reported `missing=[]`, `unexpected=[]`, and finite state tensors for all three checkpoints.

## Finite / NaN Check

All validation losses, runner test losses, final MSE/MAE values, checkpoint tensors, `metrics.npy`, `pred.npy`, and `true.npy` were finite. No NaN/Inf, shape, device, or runtime error occurred in any successful run.

## Runtime

Times are from the dedicated log file birth/modify timestamps, CST on 2026-09-22:

| Method | Start | End | Wall-clock seconds |
|---|---|---|---:|
| moving_avg | 15:52:53.806638 | 15:53:12.616357 | 18.81 |
| ema | 15:54:46.092932 | 15:55:10.787878 | 24.69 |
| scale_aware_ema | 15:56:04.726947 | 15:56:28.384855 | 23.66 |

The logs contain the GPU snapshots, full argument printout, model-level EMA summary where applicable, train/validation/test-loss lines, checkpoint-save message, full-test shape, and final metrics.

## Non-comparative Metrics Table

| Method | EMA alpha | Epoch 1 train / val | Epoch 2 train / val | Final MSE | Final MAE | Finite |
|---|---:|---:|---:|---:|---:|---|
| moving_avg | — | 0.3916414 / 0.7061211 | 0.3551973 / 0.7199184 | 0.3990383148 | 0.4057489038 | yes |
| ema | 0.10 | 0.3917070 / 0.7104271 | 0.3532233 / 0.7164249 | 0.3940372765 | 0.4015093446 | yes |
| scale_aware_ema | 0.10 | 0.3910709 / 0.7057169 | 0.3518818 / 0.7201904 | 0.3943324387 | 0.4033014178 | yes |

> Stage 04 metrics are smoke-test observations only and were not used to select a method or tune alpha.

No metric was used to compare methods, change alpha, or select a later experiment.

## Git / Source Changes

- TSLib remained at `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d`.
- `git status --short` was empty after the runs; generated logs/checkpoints/results are ignored artifacts.
- `git diff --name-status 91735a83b27ce1bf766df76a953cf86ef1e7500d -- models/TimeMixer.py run.py layers/EMADecomp.py tests/test_ema_decomp.py tests/test_timemixer_ema_integration.py` was empty.
- No model source, runner source, DataLoader, optimizer, loss, seed logic, or test source was modified in Stage 04.
- Bridge files added by this stage: `results/04_smoke_training.md`, `results/data/04_smoke_training.csv`, and the execution archive at repository root.

## Final

- `Status: PASS`
- moving_avg full pipeline: **PASS**
- fixed EMA full pipeline: **PASS**
- scale-aware EMA full pipeline: **PASS**
- source code changed during Stage 04: **NO**
- ready for Stage 05 alpha ablation: **YES**
- Stage 05 alpha ablation: **NOT EXECUTED**

Per the workflow instruction, execution stops here.
