# Current Stage

Stage 00、01、02、03、04 已审核通过。

当前只允许执行：

`prompts/05_alpha_ablation.md`

执行前必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `results/03_integrate_timemixer.md`
- `results/04_smoke_training.md`
- `prompts/05_alpha_ablation.md`

当前 TSLib 代码基线：

- branch: `codex/saema-v1`
- SHA: `91735a83b27ce1bf766df76a953cf86ef1e7500d`

Stage 05 是正式 alpha 消融阶段：

- ETTh1
- seq_len=96
- pred_len=96
- seed=2021
- 10 epochs
- moving_avg control × 1
- fixed EMA × 5 alpha
- Scale-Aware EMA × 5 alpha

候选：

`{0.02, 0.05, 0.10, 0.15, 0.20}`

**alpha 只能依据 validation 选择。必须先冻结 validation-only selection artifact，再整理 selected models 的 test 结果。**

完成后写：

- `results/05_alpha_ablation.md`
- `results/data/05_alpha_validation_selection.csv`
- `results/data/05_selected_alphas.json`
- `results/data/05_selected_models_test.csv`

完成后停止，不要自动执行 Stage 06。
