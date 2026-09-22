# Stage 03 — TimeMixer EMA / Scale-Aware EMA 集成

Status: **PASS**

Bridge source HEAD at start: `ae36fa3bcba7c5fa8c697033efd13b0bc8447bc5`

TSLib Stage 03 start SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`

TSLib baseline-freeze SHA: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`

TSLib final SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

Formal ETTh1 training/evaluation: **not run**

Alpha search: **not run**

## 1. Git and worktree boundary

All TSLib source changes were made only in:

- worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- branch: `codex/saema-v1`

The start worktree was clean at the required EMA implementation SHA. The final ancestry check passed:

`462875291aa7eb5963e9ba81dc5ab86c6b8f5230` is an ancestor of `91735a83b27ce1bf766df76a953cf86ef1e7500d`.

Commits added in Stage 03:

- `91735a83b27ce1bf766df76a953cf86ef1e7500d` — `feat: integrate scale-aware EMA into TimeMixer`

`git diff --stat 8f3a6b951360053ec1afd5f747f11beb7195d30d..HEAD`:

```text
 models/TimeMixer.py                     |  48 +++++++-
 run.py                                  |   2 +
 tests/test_timemixer_ema_integration.py | 187 ++++++++++++++++++++++++++++++++
 3 files changed, 235 insertions(+), 2 deletions(-)
```

Final target worktree status:

```text
## codex/saema-v1
```

The original dirty `main` worktree was not modified, and `codex/saema-v1-baseline` remained clean. Stage 02 files `layers/EMADecomp.py` and `tests/test_ema_decomp.py` were not changed.

## 2. Integration

### CLI

- `--decomp_method` remains defaulted to `moving_avg`.
- TimeMixer supports `moving_avg`, `dft_decomp`, `ema`, and `scale_aware_ema`.
- Added exactly `--ema_alpha`, with default `0.10`.
- No `--ema_alpha_base` was added.
- Legacy modes do not construct or validate the EMA module, so their behavior is independent of EMA alpha validity.
- The parser remains compatible with other models; TimeMixer rejects unsupported decomposition values in its existing selector path.

### PDM selector and per-scale call

`models/TimeMixer.py` imports the Stage 02 `EMADecomposition` and adds only two explicit EMA selector branches in `PastDecomposableMixing`.

- `ema` uses the same `ema_alpha` at every scale.
- `scale_aware_ema` uses the Stage 01/02 frozen coefficient schedule.
- The call uses `enumerate(x_list)` and passes the real `scale_index`.
- `moving_avg) and `dft_decomp` continue to call `self.decompsition(x)` through their legacy paths.
- `moving_avg`, DFT, seasonal/trend mixing, normalization, FMM, projection, prediction head, loss, and data split code were not refactored.

### channel_independence guard

For `ema` and `scale_aware_ema` with `channel_independence=0`, model construction raises:

```text
SAEMA v1 does not support channel_independence=0: this mode has an
additional moving-average preprocess before embedding, so the EMA
combination is outside the SAEMA v1 experiment definition rather than
a runtime shape bug.
```

The guard is applied before PDM construction. `moving_avg + channel_independence=0` and `dft_decomp + channel_independence=0` both still instantiate successfully. Additional checks with legacy alpha values `0.0`, `nan`, and `inf` also passed for both legacy methods.

### Logging

EMA construction prints one model-level summary, not one line per PDM block or batch. Example:

```text
TimeMixer EMA decomposition: decomp_method=scale_aware_ema, ema_alpha=0.1, down_sampling_window=2, down_sampling_layers=3, channel_independence=1, scales=[{'scale_index': 0, 'r_k': 1, 'alpha_k': 0.1}, {'scale_index': 1, 'r_k': 2, 'alpha_k': 0.19}, {'scale_index': 2, 'r_k': 4, 'alpha_k': 0.3439}, {'scale_index': 3, 'r_k': 8, 'alpha_k': 0.569533}]
```

## 3. Regression and checkpoint compatibility

The common model configuration was the ETTh1 baseline architecture: `seq_len=96`, `pred_len=96`, `d_model=16`, `d_ff=32`, `e_layers=2`, `down_sampling_layers=3`, `down_sampling_window=2`, `down_sampling_method=avg`, `moving_avg=25`, and `channel_independence=1`.

### Parameter and state-dict checks

For all four decomposition modes:

| mode | total parameters | trainable parameters | state-dict keys |
|---|---:|---:|---:|
| moving_avg | 75,497 | 75,497 | 81 |
| dft_decomp | 75,497 | 75,497 | 81 |
| ema | 75,497 | 75,497 | 81 |
| scale_aware_ema | 75,497 | 75,497 | 81 |

All state-dict key sequences were equal. The Stage 02 EMA module has no parameters and an empty state dict, so no EMA coefficient key was added.

### Old ETTh1 checkpoint

The Stage 00 ETTh1 `pred_len=96 / seed=2021` baseline checkpoint was loaded in the integrated `moving_avg` model with:

```python
model.load_state_dict(state_dict, strict=True)
```

The load passed with empty `missing_keys` and `unexpected_keys`. The original checkpoint was not overwritten.

### Cross-worktree moving_avg numerical regression

A deterministic CPU harness was run in:

- baseline-freeze worktree at `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`;
- integrated worktree at `91735a83b27ce1bf766df76a953cf86ef1e7500d`.

The harness used the same configuration, fixed seeds, synthetic input, `eval()`, and `torch.set_num_threads(1)`. The baseline state dict and output were loaded into the integrated model for comparison.

Results:

- state-dict keys: exact match;
- 81 parameter tensors after strict load: exact match;
- first PDM output: exact match at all four scales, shapes `(14,96,16)`, `(14,48,16)`, `(14,24,16)`, `(14,12,16)`;
- final output shape: `(2,96,7)`;
- final output: bitwise exact;
- PDM max absolute difference: `0.0`;
- PDM max relative difference: `0.0`;
- final output max absolute difference: `0.0`;
- final output max relative difference: `0.0`.

This proves that selecting `moving_avg` after the integration preserves the baseline numerical path.

## 4. Tests

### Environment

The required remote checks passed before the test and smoke runs:

```text
HOST=ps
PROJECT=/home/ouyangguanghong/projects/Time-Series-Library-saema-v1
PYTHON=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python
TORCH=2.5.1+cu121
CUDA_AVAILABLE=True
CUDA_COUNT=2
GPU=2 x NVIDIA GeForce RTX 4090 D, driver 570.211.01
```

### Stage 02 regression suite

Command:

```bash
cd /home/ouyangguanghong/projects/Time-Series-Library-saema-v1
source /home/ouyangguanghong/miniconda3/etc/profile.d/conda.sh
conda activate tslib
export CUDA_VISIBLE_DEVICES=0,1
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_ema_decomp.py
```

Result:

```text
65 passed in 1.24s
```

### Stage 03 integration tests

Added `tests/test_timemixer_ema_integration.py` with 9 tests covering:

- all four mode instantiations;
- EMA guard and legacy CI=0 behavior;
- CPU shape/finite forward;
- fixed and scale-aware alpha schedules;
- one-time logging;
- parameter/state-dict invariants;
- old checkpoint strict load;
- cross-worktree moving_avg regression;
- CUDA finite forward for all modes.

Final combined command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_ema_decomp.py tests/test_timemixer_ema_integration.py
```

Result:

```text
74 passed in 3.78s
```

The run emitted two PyTorch `torch.load` FutureWarnings from the compatibility/regression tests; they did not affect results. `git diff --check` passed.

Intermediate test-only failures were resolved without changing the required model behavior:

1. The first CUDA integration assertion compared `cuda:0` with an unindexed `cuda` device; the test was corrected to compare with the input device.
2. The first regression-test version compared randomly initialized integrated parameters before loading the baseline state; the test was corrected to strict-load first, then compare parameters and output. The independent cross-worktree harness was exact before and after this test correction.

## 5. Extremely short ETTh1 data smoke

No formal epoch runner, checkpoint save, benchmark evaluation, or alpha search was executed.

The remote smoke harness used the real ETTh1 DataLoader from:

`/home/ouyangguanghong/projects/Time-Series-Library/dataset/ETT-small/ETTh1.csv`

with:

- `seq_len=96`, `pred_len=96`, `label_len=0`;
- baseline model width/depth and multiscale settings;
- `channel_independence=1`;
- seed `2021`;
- `batch_size=4`, `num_workers=0`;
- GPU `cuda:0`;
- 2 train batches and 1 validation batch per mode;
- MSE loss, backward, finite-gradient check, and one Adam optimizer step per train batch.

Results:

```text
SMOKE mode=moving_avg train_batches=2 val_batches=1 train_shape=(4, 96, 7) val_shape=(4, 96, 7) train_loss_finite=True val_loss_finite=True backward_finite=True optimizer_step=True
SMOKE mode=ema train_batches=2 val_batches=1 train_shape=(4, 96, 7) val_shape=(4, 96, 7) train_loss_finite=True val_loss_finite=True backward_finite=True optimizer_step=True
SMOKE mode=scale_aware_ema train_batches=2 val_batches=1 train_shape=(4, 96, 7) val_shape=(4, 96, 7) train_loss_finite=True val_loss_finite=True backward_finite=True optimizer_step=True
```

One initial harness attempt stopped before model execution because its temporary `SimpleNamespace` omitted the existing runner field `augmentation_ratio`. Adding `augmentation_ratio=0` reproduced the intended runner contract; no source code or formal artifact was changed by that failed attempt.

Smoke losses are not reported as research or benchmark metrics.

## 6. Final

- Status: **PASS**
- legacy moving_avg behavior unchanged: **YES** — bitwise-equal cross-worktree output and PDM intermediates
- old checkpoint compatible: **YES** — strict load passed
- EMA adds trainable params: **NO**
- ready for Stage 04: **YES** (handoff-ready only; Stage 04 was not executed)

Stage 03 is complete. Per the workflow instruction, execution stops here and does not enter Stage 04.

