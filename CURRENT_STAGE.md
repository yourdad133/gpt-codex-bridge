# Current Stage

Stage 00、01、02 已审核通过。

当前只允许执行：

`prompts/03_integrate_timemixer.md`

执行前必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`
- `prompts/03_integrate_timemixer.md`

当前实现基线：

- baseline-freeze SHA: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- EMA module SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`

Stage 03 只允许在 `codex/saema-v1` worktree 中进行 TimeMixer 集成、CLI、guard、logging、regression/integration tests 和极短数据 smoke。

**不得进行正式训练或 alpha 搜索。**

执行完成后写：

`results/03_integrate_timemixer.md`

完成后停止，不要自动执行 Stage 04。
