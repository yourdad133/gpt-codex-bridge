# Experiment Codex Archive — 2026-09-22 16:02:39 CST

## Request summary

Executed the bridge repository's current Stage 04 only after rereading `README.md`, `CURRENT_STAGE.md`, the Stage 00–03 result reports, and `prompts/04_smoke_training.md`. Used TSLib `codex/saema-v1` at SHA `91735a83b27ce1bf766df76a953cf86ef1e7500d`, reran the complete 74-test Stage 02/03 regression suite, and ran the three required ETTh1 smoke trainings with identical baseline settings.

## Final result

Stage 04: **PASS**.

- moving_avg: full 2-epoch train/validation/checkpoint/reload/full-test pipeline passed; MSE `0.39903831481933594`, MAE `0.4057489037513733`.
- fixed EMA (`ema_alpha=0.10`): full pipeline passed; MSE `0.39403727650642395`, MAE `0.4015093445777893`.
- Scale-Aware EMA (`ema_alpha=0.10`): full pipeline passed; MSE `0.3943324387073517`, MAE `0.4033014178276062`.
- All output values and saved arrays were finite; test shape was `(2785, 96, 7)`.
- Metrics were recorded only as smoke-test health observations; they were not used for method comparison or alpha tuning.
- No TSLib source code was changed. Stage 05 was not executed.

## Changed bridge files

- `results/04_smoke_training.md`
- `results/data/04_smoke_training.csv`
- this archive file

## Validation

- Regression suite: `74 passed in 2.59s`.
- TSLib branch/SHA remained `codex/saema-v1` / `91735a83b27ce1bf766df76a953cf86ef1e7500d`.
- Three unique model IDs, descriptions, logs, checkpoint setting paths, test-result paths, and result paths were used.
- Final strict checkpoint reload checks passed with empty missing/unexpected key lists for all three modes.

## Links

- [Stage 04 report](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/04_smoke_training.md)
- [Stage 04 CSV](https://github.com/yourdad133/gpt-codex-bridge/blob/main/results/data/04_smoke_training.csv)
