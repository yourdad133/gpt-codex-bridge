# Current Stage

Stage 00–06 已审核通过。

Stage 06 结论：

- Fixed EMA alpha=0.02 相比 Moving Average 有稳定正向信号；
- hand-crafted Scale-Aware EMA 在 same-alpha control 中为 CONSISTENT_NEGATIVE；
- Stage 07 multi-dataset pilot 暂停。

当前只允许执行：

`prompts/06_5_scale_sensitivity_diagnostic.md`

目标是定位 hand-crafted scale-aware schedule 的负贡献来自哪个尺度。

冻结正式实现：

- `codex/saema-v1 @ 91735a83b27ce1bf766df76a953cf86ef1e7500d`

诊断必须在独立 branch/worktree：

- `codex/saema-scale-diagnostic`

进行。

第一轮只做 ETTh1 / pred_len=192 / seeds 2021,2022,2023 的 one-scale-at-a-time perturbation，不做多数据集、不做 learnable alpha、不做 instance-adaptive。

完成后写：

- `results/06_5_scale_sensitivity_diagnostic.md`
- `results/data/06_5_scale_sensitivity_raw.csv`
- `results/data/06_5_scale_sensitivity_summary.csv`

如满足 prompt 条件并执行 reverse probe，再写：

- `results/data/06_5_reverse_probe.csv`

完成后停止。
