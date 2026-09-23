# Codex experiment archive — 2026-09-23

## Request and outcome
Completed only Stage 06.5 scale sensitivity diagnostics. All 12 P1/P2/P3/FULL02 runs completed; Fixed EMA 0.02 F0 rows were reused from Stage 06. Paired mean ΔMSE: P1 +0.000162244, P2 −0.000437796, P3 −0.000810097, FULL02 +0.007778426. FULL02 was seed-unstable; classified CASE D. No single-scale harm was clear, so reverse probes were skipped. Stage 07 was not run.

## Changed files
Diagnostic branch `codex/saema-scale-diagnostic` at `785f43d66cc96d42a62980c7fd27d9c764a11bb7`, based on frozen `91735a83b27ce1bf766df76a953cf86ef1e7500d`:
- `layers/EMADecomp.py`
- `models/TimeMixer.py`
- `run.py`
- `tests/test_scale_vector_ema.py`

Bridge outputs:
- `results/06_5_scale_sensitivity_diagnostic.md`
- `results/data/06_5_scale_sensitivity_raw.csv`
- `results/data/06_5_scale_sensitivity_summary.csv`

## Verification
Remote tslib environment and CUDA gate passed; PyTorch 2.5.1+cu121, two CUDA devices.
86 tests passed; 12/12 trainings completed; 12/12 checkpoints had finite tensors.
Frozen `codex/saema-v1` remained at the requested SHA; diagnostic worktree clean.
This archive includes no credentials or connection secrets.

## Links
- [Stage 06.5 report](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/06_5_scale_sensitivity_diagnostic.md)
- [Raw CSV](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/data/06_5_scale_sensitivity_raw.csv)
- [Summary CSV](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/data/06_5_scale_sensitivity_summary.csv)
