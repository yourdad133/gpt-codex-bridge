# Codex reply archive — Stage 01 method specification

Time: 2026-09-22 13:18:06 +08:00

## Request summary

Re-read the bridge repository entry documents, Stage 00 audit, and the Stage 01 prompt; execute only Stage 01 for Scale-Aware EMA. Deliver the mathematical definition, engineering interface, `channel_independence` v1 boundary, and the pre-Stage-02 baseline-freeze Git procedure. Do not modify Time-Series-Library source code, implement EMA, run experiments, or begin Stage 02.

## Final result

Stage 01 is complete with status PASS. The frozen specification is in [`results/01_method_spec.md`](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/01_method_spec.md).

Key decisions:

- Support the `ema` and `scale_aware_ema` ablation modes while retaining `moving_avg` as the default.
- Use `alpha_k = 1 - (1 - alpha_base)^(w^k)` for Scale-Aware EMA.
- Keep alpha fixed, shared within each scale, and local to each forward pass.
- Restrict formal v1 EMA experiments to `channel_independence=1`; reject EMA with `channel_independence=0` explicitly to avoid silently changing or mixing the extra preprocess decomposition.
- Freeze the baseline in an isolated worktree with only the audited seed and DataLoader RNG changes, leaving the existing dirty worktree untouched.

## Changed files

- `results/01_method_spec.md` — Stage 01 deliverable.
- `Experiment_Codex_20260922_131806.md` — this reply archive.

No Time-Series-Library source file was changed.

## Verification

- Confirmed `CURRENT_STAGE.md` points only to `prompts/01_method_spec.md`.
- Confirmed Stage 00 status is PASS and re-read the audited TimeMixer decomposition call chain.
- Recomputed the required coefficient table for `alpha_base = 0.05, 0.10, 0.20` and scales `r = 1, 2, 4, 8`.
- Performed a read-only check of the relevant TimeMixer, CLI, and moving-average code locations.
- Checked the bridge deliverable for Git whitespace errors and scope.
- Did not run training, evaluation, inference, preprocessing, CUDA, or Python experiments.
- Did not execute Stage 02 or any Git cleanup/freeze operation in Time-Series-Library.

## Links

- [Bridge repository](https://github.com/yourdad133/gpt-codex-bridge)
- [Stage 01 result](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/01_method_spec.md)
