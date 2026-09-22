# Current Stage

Stage 00、01、02、03 已审核通过。

当前只允许执行：

`prompts/04_smoke_training.md`

执行前必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `results/03_integrate_timemixer.md`
- `prompts/04_smoke_training.md`

当前 TSLib 集成基线：

- branch: `codex/saema-v1`
- SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

Stage 04 只做 ETTh1 / pred_len=96 / seed=2021 的三种 decomposition **2-epoch 完整训练链路 smoke**：

- moving_avg
- ema, `--ema_alpha 0.10`
- scale_aware_ema, `--ema_alpha 0.10`

必须走完整 train → validation → checkpoint → reload → full test，但这些指标只用于健康检查，**不得用于选方法或调 alpha**。

完成后写：

- `results/04_smoke_training.md`
- `results/data/04_smoke_training.csv`

完成后停止，不要自动执行 Stage 05。
