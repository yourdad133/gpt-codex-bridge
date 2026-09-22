# Current Stage

Stage 00、Stage 01 已审核通过。

当前只允许执行：

`prompts/02_implement_ema_module.md`

执行前必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `prompts/02_implement_ema_module.md`

Stage 02 分为两个串行部分：

1. 按 Stage 01 冻结方案建立安全、可追溯的 baseline worktree / baseline-freeze commit；
2. 从 baseline-freeze 派生 SAEMA worktree，只实现独立 EMA decomposition 模块及单元测试。

**不要接入 TimeMixer PDM，不要改 EMA CLI，不要跑正式训练。**

执行完成后，把结果写入：

`results/02_implement_ema_module.md`

完成后停止，不要自动执行 Stage 03。
