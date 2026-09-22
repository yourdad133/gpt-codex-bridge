# Codex Execution Archive — 2026-09-22

## Request

在 `codex/saema-v1` worktree 上严格执行 bridge 的 Stage 03：将 Stage 02 EMA / Scale-Aware EMA 接入 TimeMixer PDM，完成 CLI、guard、logging、baseline-freeze 跨 worktree 回归、旧 checkpoint strict-load、65 个 EMA 单测、新集成测试和极短 ETTh1 smoke；写回 Stage 03 报告并 push，停止，不执行 Stage 04。

## Result

Stage 03 **PASS**。

- TSLib worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- Branch: `codex/saema-v1`
- Start SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`
- Final SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`
- Commit: `feat: integrate scale-aware EMA into TimeMixer`
- Legacy moving_avg regression: bitwise exact
- Old ETTh1 checkpoint strict=True load: passed
- Tests: 74 passed
- ETTh1 smoke: moving_avg, ema, scale_aware_ema each passed 2 train batches + 1 validation batch with finite forward/loss/gradients and optimizer steps
- Formal training and alpha search: not run
- Stage 04: not executed

## Changed files

- `models/TimeMixer.py`
- `run.py`
- `tests/test_timemixer_ema_integration.py`

The original dirty `main` worktree and baseline-freeze worktree were preserved.

## Evidence

Full report:

https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/03_integrate_timemixer.md

Remote runtime used `tslib`, PyTorch `2.5.1+cu121`, CUDA available with 2 × NVIDIA GeForce RTX 4090 D.


