# Current Stage

Stage 00、01、02、03、04、05 已审核通过。

当前只允许执行：

`prompts/06_etth1_full_benchmark.md`

执行前必须完整读取 Stage 00–05 的 results，以及：

- `results/data/05_alpha_validation_selection.csv`
- `results/data/05_selected_alphas.json`
- `results/data/05_selected_models_test.csv`
- `prompts/06_etth1_full_benchmark.md`

当前冻结代码：

- branch: `codex/saema-v1`
- SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

Stage 05 冻结超参数：

- Fixed EMA: `ema_alpha=0.02`
- Scale-Aware EMA: `ema_alpha=0.05`

Stage 06 使用双轨：

1. **Track A — selected-model performance**
   - moving_avg
   - Fixed EMA 0.02
   - SAEMA 0.05

2. **Track B — same-alpha mechanism control**
   - Fixed EMA 0.05
   - SAEMA 0.05

ETTh1 horizons：

`{96,192,336,720}`

seeds：

`{2021,2022,2023}`

最终统一形成 4 configs × 4 horizons × 3 seeds = 48-cell matrix。允许按 Stage 06 prompt 严格审计后复用 Stage 00 baseline cells 和 Stage 05 完全重合 cells。

不得根据 Stage 06 结果重新调 alpha。

完成后写：

- `results/06_etth1_full_benchmark.md`
- `results/data/06_etth1_full_benchmark.csv`
- `results/data/06_trackA_selected_models_summary.csv`
- `results/data/06_trackB_same_alpha_mechanism.csv`
- 推荐：`results/data/06_paired_deltas.csv`

完成后停止，不要自动执行 Stage 07。
