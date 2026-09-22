# Stage 00 — Baseline Audit / 冻结现有 TimeMixer baseline

## 目标

在开始任何 Scale-Aware EMA 代码修改之前，对当前已经复现成功的 TimeMixer / TSLib 环境做一次完整审计，确定后续修改点和可比较的 baseline。

**本阶段禁止修改 Time-Series-Library 源代码，禁止重新设计模型。**

## 任务

### A. 定位并记录 TSLib 工作区

找到此前用于 TimeMixer ETTh1 复现的 Time-Series-Library 工作目录，并记录：

- 绝对路径；
- git remote；
- 当前 branch；
- 当前 commit SHA；
- `git status --short`；
- Python / PyTorch / CUDA 版本；
- GPU 型号；
- 当前 conda/venv 环境名称（如果能确定）。

如果工作区存在未提交修改，不要清理、不要 reset，先完整记录。

### B. 核对已完成的 baseline

读取此前的复现实验报告和日志。已知 bridge 仓库中有：

`Experiment_Codex_20260922_102200.md`

核对 ETTh1 TimeMixer long-term forecasting 的：

- pred_len = 96, 192, 336, 720；
- seeds = 2021, 2022, 2023；
- seq_len / label_len；
- d_model / d_ff / e_layers；
- down_sampling_layers；
- down_sampling_window；
- down_sampling_method；
- learning rate / epochs / batch size；
- MSE / MAE；
- checkpoint 与日志位置。

除非核对结果缺失，否则本阶段不要重新跑 12 次训练。

### C. 审计 TimeMixer decomposition 调用链

至少检查并在报告中给出实际文件路径、类/函数名和关键代码位置：

- `models/TimeMixer.py`；
- TimeMixer 使用的 moving-average decomposition 实现；
- DFT decomposition（如果当前版本存在）；
- `run.py` 或当前 CLI 参数入口；
- long-term forecast 实验入口；
- ETTh1 TimeMixer 脚本。

回答以下问题：

1. 当前 `decomp_method` 支持哪些值？
2. moving average 的 kernel/window 参数来自哪里？
3. decomposition 在 PDM 中具体在哪一层调用？
4. multi-scale `x_list` 有多少个尺度？
5. 每个尺度的时间长度如何由 `down_sampling_layers` 与 `down_sampling_window` 决定？
6. 所有尺度是否共享同一个 decomposition 模块/参数？
7. seasonal 与 trend 后续分别经过哪些 mixing 路径？
8. 输入张量在 decomposition 处的实际 shape 约定是什么（例如 [B,T,C]）？
9. 如果新增 EMA decomposition，最小侵入式修改点在哪里？

### D. 给出后续修改风险

只做分析，不改代码。至少检查：

- 与 DFT 分解分支的兼容风险；
- 与 `channel_independence` 的关系；
- 不同尺度 sequence length 边界；
- AMP / dtype / device；
- checkpoint 向后兼容；
- 默认配置必须保持原 TimeMixer 行为。

## 输出

把结果写入 bridge 仓库：

`results/00_baseline_audit.md`

报告必须以以下字段开头：

- `Status: PASS | FAIL | BLOCKED`
- `TSLib commit:`
- `TSLib branch:`
- `Baseline reproducible: YES | NO | PARTIAL`

最后写：

### Recommended edit points

只列推荐修改位置，不要实际修改。

## 完成条件

结果文件已提交并 push 到：

https://github.com/yourdad133/gpt-codex-bridge

然后停止。**不要执行 Stage 01。**
