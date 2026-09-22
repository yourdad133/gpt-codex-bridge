# Stage 01 — Scale-Aware EMA 数学与工程设计冻结

## 前置条件

只有在 `results/00_baseline_audit.md` 为 PASS 或明确可继续时执行。

本阶段仍然 **不修改 TSLib 源代码**。目标是先把数学定义和接口冻结，避免边写代码边改变研究假设。

Stage 00 已确认：
- 当前 baseline 结果可信，但 TSLib 工作树不是 commit-clean；
- 三 seed 可复现性依赖 `run.py` 与 DataLoader 的未提交 seed/RNG 改动；
- baseline 主实验使用 `channel_independence=1`；
- 当 `channel_independence=0` 时，`Model.preprocess` 还存在一层独立的 moving-average decomposition。

因此本阶段还必须冻结 **baseline snapshot 策略** 与 **channel_independence v1 边界**，以便 Stage 02 安全实现。

## 研究问题

TimeMixer 在不同 down-sampling scale 上做 decomposition，但原始 moving-average decomposition 的机制在尺度间并未显式根据采样间隔调整。

第一版只验证：

> 当不同 scale 的有效采样间隔不同，是否应该使用 scale-aware EMA smoothing coefficient？

暂不引入 instance / channel / time-dependent 自适应。

## 必须采用的第一版定义

EMA 定义：

`trend_t = alpha_k * x_t + (1 - alpha_k) * trend_{t-1}`

`seasonal_t = x_t - trend_t`

初始化：

`trend_0 = x_0`

设基础尺度的 EMA 系数为：

`alpha_base in (0,1)`

如果 TimeMixer 的 `down_sampling_window = w`，第 k 个尺度相对基础尺度的采样间隔倍率：

`r_k = w^k`

采用：

`alpha_k = 1 - (1 - alpha_base)^(r_k)`

其中 k=0 是原始分辨率。

这等价于假设一个固定连续时间 smoothing time constant，并根据不同 scale 的实际采样间隔换算 EMA coefficient。

## 本阶段需要完成

### A. 验证数学定义

解释并验证：

- 为什么 `alpha_k` 会随 scale 增大；
- 为什么不是简单手工设置 `alpha_0, alpha_1, ...`；
- `alpha_base` 很小时与很大时的行为；
- 当 `w=2`、`down_sampling_layers=3` 时，给出 alpha_base = 0.05 / 0.10 / 0.20 的各尺度数值表；
- 明确说明该换算保持的是“连续时间意义下相同的 EMA time constant / decay horizon”，而不是人为规定“粗尺度必须更追随当前点”。

### B. 冻结第一版方法边界

明确第一版：

- alpha 不可学习；
- alpha 不依赖 instance；
- alpha 不依赖 channel；
- alpha 不随时间 t 变化；
- 所有 channel 在同一 scale 共用同一个 alpha_k；
- 只替换 PDM decomposition，不改变 PDM 的 seasonal bottom-up mixing 与 trend top-down mixing；
- 不改变 FMM；
- 不改变 loss；
- 不改变数据划分；
- **正式 v1 主实验固定 `channel_independence=1`，与已复现 baseline 保持一致。**

对于 `channel_independence=0`：
- 本阶段必须明确工程行为；
- v1 不得悄悄把 `Model.preprocess` 的额外 moving-average split 也替换成 EMA；
- 优先方案是保留原 preprocess 行为并清楚记录，或在 EMA 模式下显式声明该组合暂不属于 v1 实验边界；
- 不允许在没有实验设计的情况下同时改变两处分解。

### C. 设计两种 EMA 作为消融

后续代码必须同时支持：

1. `ema`
   - 所有尺度都使用同一个 `alpha_base`。
2. `scale_aware_ema`
   - 第 k 个尺度使用上述 `alpha_k`。

这样后续可以区分：

- Moving Average → EMA 的收益；
- EMA → Scale-Aware EMA 的额外收益。

### D. 工程接口设计

根据 Stage 00 实际代码，提出最小修改方案，包括：

- 新模块建议文件；
- 类名 / 方法签名；
- 如何传入 scale index；
- 如何传入 `down_sampling_window`；
- CLI 参数名；
- `decomp_method` 如何扩展；
- 默认值如何保证原 baseline 完全不变；
- EMA alpha 如何记录到实验日志；
- checkpoint 是否需要新增参数（第一版应尽量无可训练参数）；
- 如何确保原 `moving_avg` 与 `dft_decomp` 分支代码路径不被重构式修改。

不要写伪造的文件位置，必须基于 Stage 00 的真实审计。

### E. 数值稳定性设计

说明：

- alpha 边界校验；
- dtype / device 处理；
- 是否允许 fp16/bfloat16；
- EMA recursion 是否保持 autograd；
- 空序列 / 长度 1；
- NaN/Inf 检查建议。

### F. 冻结 Stage 02 前的 baseline snapshot 策略

Stage 00 已发现当前 TSLib 工作树有大量未提交内容。**本阶段不得执行 git clean/reset/stash/commit 等写操作**，但必须在报告中给出 Stage 02 应遵循的安全方案。

方案必须满足：

1. 不丢失当前任何未提交工作；
2. 保留已复现实验真正依赖的 seed / DataLoader RNG 改动；
3. Scale-Aware EMA 的实现有明确、可追溯的起点；
4. unrelated AdaptivePatch / M4 等改动不能被误认为 SAEMA 的贡献；
5. 后续 baseline 与 EMA 使用同一实验基础代码；
6. 能明确列出“baseline snapshot 中包含哪些非 upstream 改动”。

请给出推荐的 branch / commit 组织方式，并说明为什么这样最安全。不要在 Stage 01 实际执行。

## 输出

写入：

`results/01_method_spec.md`

最后必须给出两段：

### Frozen v1 specification

把后续实现必须遵守的数学公式、配置项、默认行为浓缩成一份不可含糊的规格。

### Stage 02 baseline-freeze procedure

给出 Codex 在真正改源码前必须执行的、安全且可回滚的 git 步骤。该步骤不得丢失当前 dirty-tree 内容。

完成后 push 到 bridge 仓库并停止。不要执行 Stage 02。
