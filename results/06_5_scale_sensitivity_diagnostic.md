# Stage 06.5 — Scale Sensitivity Diagnostic

**Status: PASS — diagnostic pilot complete.** Twelve new runs completed; three F0 Fixed EMA reference rows were reused from Stage 06. Stage 07 was not started.

## Git isolation and implementation

- Frozen branch `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d` was verified unchanged.
- Created independent worktree/branch `codex/saema-scale-diagnostic` from that exact SHA: `/home/ouyangguanghong/projects/Time-Series-Library/checkpoints/.codex-worktrees/saema-scale-diagnostic`.
- Diagnostic source commit: `785f43d66cc96d42a62980c7fd27d9c764a11bb7`; worktree clean.
- Only source/test files changed: `layers/EMADecomp.py`, `models/TimeMixer.py`, `run.py`, `tests/test_scale_vector_ema.py`.
- Added only explicit fixed per-scale `scale_vector_ema` via `--ema_scale_alphas`; vector length must be 4, values finite and in (0,1), and it requires `channel_independence=1`. It adds no trainable parameter or persistent state key. No learnable, instance-adaptive, or channel-adaptive alpha was implemented.
- Existing moving_avg, ema, and scale_aware_ema paths passed bitwise/regression checks and legacy strict checkpoint loading.

## Verification and run protocol

Tests ran in the remote project `tslib` Python environment: **86 passed in 1.49 s** (including Stage 02/03 regression suites; two non-failing torch.load FutureWarnings). CPU/CUDA forward/backward finiteness, vector validation/exact use, no-parameter/no-state invariants, and legacy regressions passed.

The pre-run remote gate confirmed host `ps`, the diagnostic worktree directory, the configured tslib Python, PyTorch 2.5.1+cu121, and CUDA availability on two RTX 4090s. All 12 new runs exited successfully after 10 epochs with Stage 06 settings. Their 12 checkpoints were independently loaded on CPU; all 81 tensors per checkpoint were finite.

## Conditions and results

Dataset ETTh1, seq_len=96, pred_len=192, seeds 2021/2022/2023, 10 epochs. F0 = Fixed EMA `[0.02,0.02,0.02,0.02]`. Perturbations: P1 `[0.02,0.0396,0.02,0.02]`; P2 `[0.02,0.02,0.07763184,0.02]`; P3 `[0.02,0.02,0.02,0.14923698]`; FULL02 `[0.02,0.0396,0.07763184,0.14923698]`. Remaining settings followed Stage 06 (features=M, label_len=0, TimeMixer e_layers=2, 7 input/output channels, d_model=16, d_ff=32, lr=0.01, batch=128, patience=10, Adam/MSE/type1 schedule, 3 average downsampling layers/window 2, channel_independence=1, use_norm=1, moving_avg=25, 10 workers).

Mean and sample SD are across seeds. Delta is the mean paired `X − F0`; negative favors the perturbation.

| Config | Alpha vector | Mean MSE ± sample SD | ΔMSE vs F0 | Mean MAE ± sample SD | ΔMAE vs F0 | MSE wins vs F0 |
|---|---|---:|---:|---:|---:|---:|
| F0 | [0.02,0.02,0.02,0.02] | 0.426907837 ± 0.002883279 | +0.000000000 | 0.428518236 ± 0.001438881 | +0.000000000 | — |
| P1 | [0.02,0.0396,0.02,0.02] | 0.427070081 ± 0.002527383 | +0.000162244 | 0.429184854 ± 0.001181597 | +0.000666618 | 1/3 |
| P2 | [0.02,0.02,0.07763184,0.02] | 0.426470041 ± 0.001962819 | -0.000437796 | 0.428981731 ± 0.001386908 | +0.000463496 | 2/3 |
| P3 | [0.02,0.02,0.02,0.14923698] | 0.426097741 ± 0.002861969 | -0.000810097 | 0.428313583 ± 0.001142081 | -0.000204653 | 3/3 |
| FULL02 | [0.02,0.0396,0.07763184,0.14923698] | 0.434686263 ± 0.013771171 | +0.007778426 | 0.430552473 ± 0.002910348 | +0.002034237 | 2/3 |

### Per-seed paired deltas

| Config | Seed | ΔMSE vs F0 | ΔMAE vs F0 |
|---|---:|---:|---:|
| P1 | 2021 | +0.000070542 | +0.001126111 |
| P1 | 2022 | -0.000473499 | +0.000719666 |
| P1 | 2023 | +0.000889689 | +0.000154078 |
| P2 | 2021 | +0.000963598 | +0.000596344 |
| P2 | 2022 | -0.001375675 | +0.000579417 |
| P2 | 2023 | -0.000901312 | +0.000214726 |
| P3 | 2021 | -0.000459850 | +0.000163972 |
| P3 | 2022 | -0.000739008 | -0.000357419 |
| P3 | 2023 | -0.001231432 | -0.000420511 |
| FULL02 | 2021 | +0.025715709 | +0.006643713 |
| FULL02 | 2022 | -0.000266820 | +0.000694096 |
| FULL02 | 2023 | -0.002113611 | -0.001235098 |

## Interpretation and decision

- **P1 / k=1:** mean ΔMSE +0.000162244, 1/3 MSE wins; small and mixed, not clear harm.
- **P2 / k=2:** mean ΔMSE −0.000437796 (2/3 wins), while mean ΔMAE is +0.000463496; seed 2021 is adverse, seeds 2022/2023 improve.
- **P3 / k=3:** mean ΔMSE −0.000810097 (3/3 wins), mean ΔMAE −0.000204653. The coarsest-scale increase did not show harm in this pilot.
- **FULL02:** mean ΔMSE +0.007778426 and mean ΔMAE +0.002034237, while 2/3 seeds have lower MSE. Seed 2021 contributes +0.025715709 paired ΔMSE; seeds 2022/2023 have negative MSE deltas. The FULL02 MSE sample SD is 0.013771171. Its average MSE delta exceeds the sum of single-scale mean deltas (−0.001085649), showing a non-additive and seed-sensitive combined result.
- No individual scale showed clear harm, so the optional reverse-direction probe was skipped. No test metric was used to tune alpha.

**Decision: CASE D — pilot evidence is seed-unstable and insufficient for a more complex adaptive module.** This is descriptive evidence for one dataset, one horizon, and three seeds, not a broad generalization claim. Work stops here; no extra datasets/horizons or Stage 07 were run.

## Required outputs

- `results/06_5_scale_sensitivity_diagnostic.md`
- `results/data/06_5_scale_sensitivity_raw.csv`
- `results/data/06_5_scale_sensitivity_summary.csv`

The raw CSV contains all required provenance fields, paired deltas, checkpoint paths, and F0 reuse/source-stage flags. Reverse-probe CSV was not required because that probe was skipped.
