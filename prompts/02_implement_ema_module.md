# Stage 02 — 冻结可追溯 baseline + 实现独立 EMA decomposition 与单元测试

## 前置条件

必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`

严格遵守其中：

- `Frozen v1 specification`
- `Stage 02 baseline-freeze procedure`

若当前 TSLib 状态与 Stage 00/01 记录出现无法解释的变化，停止并报告，不要继续。

## 本阶段目标

本阶段包含两个串行子阶段：

### Stage 02A — Baseline Freeze

在**不修改、不清理、不 stash、不 commit 原 dirty worktree**的前提下，按 Stage 01 冻结的 procedure 建立：

- `codex/saema-v1-baseline`：只包含 upstream audited SHA + baseline 复现实验真正依赖的 seed/DataLoader RNG 差异；
- `codex/saema-v1`：从 baseline-freeze commit 派生，用于后续 SAEMA 开发。

### Stage 02B — EMA Module

只在 `codex/saema-v1` 实现独立的：

- fixed EMA decomposition；
- Scale-Aware EMA decomposition；
- alpha 换算 helper；
- 单元测试。

**本阶段绝对不要把 EMA 接入 TimeMixer PDM / forward，也不要新增 EMA CLI，也不要修改正式实验脚本。**

---

# Stage 02A — Baseline Freeze

## A1. 原 dirty worktree 安全检查

按照 Stage 01 的 `Stage 02 baseline-freeze procedure` 执行。

至少记录：

- 原 worktree path；
- HEAD / branch；
- `git status --short`；
- tracked staged / unstaged diff 摘要；
- untracked manifest；
- `git worktree list`。

如果 audited upstream SHA 不再是：

`4e938a1767106324dd753b2a44832bf870a0252e`

或者原 dirty tree 相对 Stage 00 出现无法解释的新改动：

**Status=BLOCKED，停止。**

禁止在原 dirty worktree 执行：

- `git clean`
- `git reset`
- `git stash`
- `git commit`
- checkout/switch
- 删除或覆盖未跟踪文件

## A2. 私有恢复备份

按 Stage 01 方案，在 bridge/TSLib 仓库之外保存恢复材料：

- tracked diff/patch；
- staged diff；
- untracked manifest；
- 必要的 untracked 文件副本或校验信息。

这些恢复材料**不要提交进 bridge 或 TSLib 仓库**。

报告中只说明备份位置与验证结果，不暴露敏感本地配置内容。

## A3. 建立 baseline worktree

从 audited SHA 建立 sibling worktree：

branch：

`codex/saema-v1-baseline`

初始 HEAD 必须等于：

`4e938a1767106324dd753b2a44832bf870a0252e`

新 worktree 初始必须 clean。

## A4. 只移植 baseline 所需可复现性差异

严格按 Stage 01 结果，只允许移植：

### `run.py`

只保留与 deterministic reproduction 直接相关的：

- seed CLI/default；
- Python random seed；
- NumPy seed；
- PyTorch CPU seed；
- CUDA seed；
- 必要的 deterministic seed initialization 调用。

不得带入 AdaptivePatch 或其他模型的 CLI 参数。

### `data_provider/data_factory.py`

只保留与 deterministic DataLoader 直接相关的：

- worker RNG 初始化；
- 由 `args.seed` 初始化的 `torch.Generator`；
- DataLoader 使用该 generator / worker init 的必要改动。

明确排除：

- router auxiliary loss；
- AdaptivePatch；
- M4 修复；
- README 删除；
- mode-only 改动；
- 本地 Codex 配置；
- 旧报告；
- 无关 shell script 改动；
- 其他未跟踪实验文件。

## A5. 验证 baseline snapshot

必须：

- `git diff --check`；
- 人工/程序化核对最终 diff；
- 最终相对 audited upstream SHA 的实质修改只应出现在：
  - `run.py`
  - `data_provider/data_factory.py`
- 做轻量 deterministic RNG/DataLoader 验证；
- 不跑正式训练；
- 不覆盖已有 baseline artifacts。

验证通过后创建 commit：

`baseline-freeze: preserve deterministic seed and dataloader rng`

记录：

- baseline-freeze commit SHA；
- 精确的非-upstream diff 清单。

如果 remote 未明确允许，不要 push TSLib branch；本阶段只要求把所有 SHA 和证据写回 bridge 仓库。

## A6. 建立 SAEMA 实现 worktree

从 baseline-freeze SHA 建立 sibling worktree / branch：

`codex/saema-v1`

所有后续 Stage 02B 修改只能发生在该 worktree。

---

# Stage 02B — 独立 EMA 模块实现

## B1. 文件边界

按照 Stage 01 冻结设计，优先新建：

`layers/EMADecomp.py`

若项目测试目录不存在，可新增一个最小、清晰的测试文件，例如：

`tests/test_ema_decomp.py`

但不要为了测试重构项目。

本阶段 SAEMA implementation branch **不得修改**：

- `models/TimeMixer.py`
- `run.py`
- 正式 TimeMixer shell scripts
- PDM / FMM / mixing 代码

注意：implementation branch 会继承 baseline-freeze commit 中已经存在的 `run.py` seed 变化，这不算 Stage 02B 新修改；报告要区分“继承的 baseline diff”和“本阶段新增 diff”。

## B2. Frozen API

实现接口应遵循 Stage 01：

```python
class EMADecomposition(nn.Module):
    def __init__(
        self,
        alpha_base: float,
        down_sampling_window: int,
        mode: str,  # exactly 'ema' or 'scale_aware_ema'
    ):
        ...

    def alpha_for_scale(self, scale_index: int) -> float:
        ...

    def forward(
        self,
        x: torch.Tensor,       # [batch_like, time, feature]
        scale_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...                    # (seasonal, trend)
```

若由于项目 Python 风格需要轻微调整类型标注可以调整，但语义不能改变。

## B3. 数学行为

初始化：

`trend[:, 0, :] = x[:, 0, :]`

对 `t >= 1`：

`trend_t = alpha_k * x_t + (1 - alpha_k) * trend_{t-1}`

`seasonal = x - trend`

### fixed EMA

`mode='ema'`

所有尺度：

`alpha_k = alpha_base`

### Scale-Aware EMA

`mode='scale_aware_ema'`

`r_k = down_sampling_window ** scale_index`

`alpha_k = 1 - (1-alpha_base) ** r_k`

数值计算优先使用 Stage 01 冻结的稳定形式：

`alpha_k = -expm1(r_k * log1p(-alpha_base))`

其数学定义仍以上述闭式公式为准。

## B4. 参数与状态约束

必须满足：

- alpha 不得是 `nn.Parameter`；
- 不得增加 persistent state-dict key；
- 不保存跨 batch EMA state；
- forward 不使用 numpy；
- 不 `.detach()`；
- 不用 `torch.no_grad()` 包住正常 forward；
- 不做破坏 autograd 的 inplace 操作；
- 输入输出 shape 完全相同；
- 返回顺序固定为 `(seasonal, trend)`。

## B5. 输入校验

至少验证：

- `alpha_base` finite 且严格在 `(0,1)`；
- `down_sampling_window` 为整数且 >=1；
- mode 只能是 `ema` / `scale_aware_ema`；
- `scale_index` 为非负整数；
- x 必须是 3D floating tensor；
- T=0 清晰报错；
- T=1 正常工作。

## B6. dtype / device / autograd

要求：

- float32 / float64 保持合理精度；
- fp16 / bf16 时，递推 accumulator 使用 fp32，再将输出转回输入 dtype；
- 输出 device 与 input 相同；
- backward 对 input 产生 finite gradient；
- 不创建错误的 CPU 中间状态。

---

# 单元测试

必须至少覆盖以下内容。

## C1. 数学正确性

1. 输出 shape == 输入 shape；
2. `seasonal + trend ≈ x`；
3. 常数序列 → trend 为常数、seasonal≈0；
4. fixed EMA 小样本与手算一致；
5. Scale-Aware alpha 与公式一致；
6. `w>1` 时 alpha 随 scale 单调增大；
7. `w=1` 时各尺度 alpha 相同；
8. `scale_index=0` 时 scale-aware alpha == alpha_base；
9. `down_sampling_layers=0` 对应的 k=0 语义可用。

## C2. 边界/错误

10. alpha=0；
11. alpha=1；
12. alpha<0 / >1；
13. NaN / Inf alpha；
14. 非法 mode；
15. 非法 w；
16. 负 scale_index；
17. 空序列；
18. length=1；
19. 非浮点输入；
20. 非 3D 输入。

非法输入必须产生明确、可理解的异常。

## C3. device / dtype / gradient

21. CPU float32；
22. CPU float64；
23. CUDA float32（CUDA 可用时）；
24. CUDA fp16（支持时）；
25. CUDA bf16（支持时）；
26. backward finite gradient；
27. 输出 dtype 与输入一致；
28. 输出 device 与输入一致；
29. state_dict 不包含 EMA coefficient 的 persistent key；
30. 模块 `parameters()` 为空。

## C4. Regression properties

31. 同一输入 + 同一配置，多次 forward deterministic；
32. `ema` 模式不同 scale_index 得到相同结果；
33. `scale_aware_ema` 在 scale_index 不同时实际使用不同 alpha；
34. 不修改输入 tensor 内容。

---

# 性能 smoke

使用与 ETTh1 PDM 接近的 tensor 尺寸做粗略 forward timing，例如：

- batch_like 可模拟 `128 * 7`；
- T 至少测试 96；
- D=16；
- float32；
- CPU；
- CUDA 可用时 GPU。

只用于发现数量级异常，例如 Python 实现极端缓慢。

必须记录：

- tensor shape；
- device；
- warmup；
- 重复次数；
- 平均/中位或简单总耗时。

**不要为了优化耗时牺牲数学正确性或 autograd。**
本阶段不把 timing 当作论文 efficiency 结论。

---

# 本阶段禁止

- 不接入 TimeMixer PDM；
- 不改 `models/TimeMixer.py`；
- 不新增 `--ema_alpha`；
- 不新增/修改正式训练脚本；
- 不跑 ETTh1 正式训练；
- 不做 alpha grid search；
- 不加 instance adaptive；
- 不加 channel adaptive；
- 不把 alpha 变成可学习参数；
- 不修改 moving-average / DFT implementation。

---

# Commit 边界

建议在 `codex/saema-v1` 中让 Stage 02B 的 implementation commit 只包含：

- `layers/EMADecomp.py`
- EMA 单元测试文件

建议 commit message：

`feat: add scale-aware EMA decomposition module`

如果测试需要极小的测试配置文件，可一并提交并在报告解释。

---

# 输出

写入 bridge：

`results/02_implement_ema_module.md`

必须包含：

## Baseline Freeze
- 原 dirty worktree 是否保持未修改；
- recovery backup 是否建立并验证；
- baseline worktree path；
- baseline branch；
- audited upstream SHA；
- baseline-freeze SHA；
- baseline-freeze 相对 upstream 的 changed files + diff summary；
- deterministic check；
- 被明确排除的 unrelated changes。

## EMA Implementation
- implementation worktree path；
- branch；
- parent baseline-freeze SHA；
- implementation commit SHA；
- 本阶段新增 changed files；
- 核心 API；
- alpha 计算方法；
- parameter/state_dict 检查。

## Tests
- 完整测试命令；
- pass/fail 数；
- CPU/CUDA/dtype/backward 结果；
- 失败测试及修复历史。

## Performance Smoke
- shape/device；
- timing；
- 是否发现数量级异常。

## Git
- `git status --short`；
- `git diff --check`；
- `git diff --stat baseline-freeze..implementation`；
- implementation commit 相对 baseline-freeze 的实际文件列表。

## Final
- `Status: PASS | FAIL | BLOCKED`
- 是否满足 Frozen v1 specification；
- 是否可以进入 Stage 03。

如任何 baseline-freeze 安全步骤、测试或 diff boundary 无法满足：

**停止，写 FAIL/BLOCKED，不进入 Stage 03。**

完成结果文件并 push 回 bridge 后停止。
