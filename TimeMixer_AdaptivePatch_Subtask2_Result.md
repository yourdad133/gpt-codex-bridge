# TimeMixerAdaptivePatch：子任务二结果

- **子任务**：实现独立的 Local Dynamic Patch 模块
- **目标仓库**：yourdad133/gpt-codex-bridge
- **实现状态**：完成并通过 CPU smoke / backward 验证
- **范围控制**：未修改 TimeMixer baseline、exp、run.py 或训练脚本；未开始子任务三；未使用 GPU（本模块验证不需要 GPU）。
- **任务规范**：[TimeMixer_AdaptivePatch_Subtask2.md](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask2.md)

## 1. 实际修改文件

新增：

- layers/AdaptivePatchRouter.py

未新增 utils/router_utils.py，因为当前模块的路由、统计和辅助损失可以保持在一个独立文件内。

现有 TSLib baseline 文件没有被本子任务覆盖或重置。当前工作区中原有的未提交修改保持原样；本子任务没有 commit 或 push 当前 TSLib 仓库。

## 2. 模块结构

### PatchCandidateEncoder

对一个候选 patch size p 独立建模：

1. 对时间轴右侧 padding，使长度可被 p 整除；
2. 将 [B_eff,T,D] 重排为 [B_eff,N_p,p,D]；
3. 用独立的 Linear(p, 1) 在每个 patch 内做时间投影；
4. 用特征 MLP 生成 patch enhancement；
5. repeat-interleave 回到时间轴并 crop，输出 Delta_p: [B_eff,T,D]。

每个 patch size 都有独立的 temporal projection，不共享不同 p 的时间投影参数。

### LocalAdaptivePatch

输入和输出均为：

[B_eff,T,D] -> [B_eff,T,D]

其中 B_eff 可以是普通 batch，也可以是 channel-independence 下的 B*C。模块不需要知道原始的 B 与 C。

默认候选 patch sizes：(4, 8, 16, 32)。对长度为 T 的输入，仅启用满足 p <= T 的候选；输出仍保留完整 K_total 个 logits，非法候选的概率严格置零。

### MultiScaleAdaptivePatch

接收多尺度张量列表，为每个尺度建立独立的 LocalAdaptivePatch。各尺度模块不共享参数，也不导入 TimeMixer，便于后续接入原始 PDM。

## 3. 路由与残差公式

局部变化量：

δ₀=0，δₜ=|Hₜ-Hₜ₋₁|。

路由输入：

Gₜ=[Hₜ;δₜ] ∈ R^(2D)。

路由器输出 K_total 个 logits，并对非法候选做 mask。学习模式使用温度 τ>0 的 softmax：

πₜ,ₖ=softmax(sₜ,ₖ / τ)。

候选增强融合：

Δₜ=Σₖ πₜ,ₖ Δₜ,ₖ。

最终使用残差形式：

H_out=H+βΔ，

其中 beta 是可学习标量，初始化为 0.1。当 beta=0 时，模块严格退化为恒等映射，不会破坏原始 TimeMixer 表示。

提供 routing_mode='uniform'，该模式忽略 learned logits，仅在当前有效候选上均匀分配概率。

## 4. 有效候选过滤

| T | active patch sizes |
|---:|:---|
| 96 | 4, 8, 16, 32 |
| 48 | 4, 8, 16, 32 |
| 24 | 4, 8, 16 |
| 12 | 4, 8 |
| 95 | 4, 8, 16, 32 |
| 13 | 4, 8 |

对无效候选，输出概率为零；有效候选概率重新归一化，避免 padding 或无效分支参与路由统计。

## 5. 辅助损失与统计

实现了：

- patch balance loss：有效候选上的 KL(q || uniform)；
- patch entropy：有效候选概率熵；
- get_last_stats()：返回 detached 的 active sizes、mean patch probabilities、entropy、beta；
- 当有效候选数为 1 时，balance loss 稳定为零；所有统计均保持有限值。

辅助损失通过 get_auxiliary_losses() 暴露为可反向传播的张量，后续可在训练入口组合为：

 total_loss = prediction_loss + lambda_balance * patch_balance_loss + lambda_entropy * patch_entropy

本子任务没有修改训练入口。

## 6. 验证结果

### 形状、概率和有限性

| T | output shape | max probability-sum error | balance loss | entropy |
|---:|:---|---:|---:|---:|
| 96 | (14, 96, 16) | 1.19e-7 | 0.01471 | 1.36027 |
| 48 | (14, 48, 16) | 1.19e-7 | 0.01413 | 1.36189 |
| 24 | (14, 24, 16) | 1.19e-7 | 0.00675 | 1.08308 |
| 12 | (14, 12, 16) | 0 | 0.00754 | 0.68090 |
| 95 | (14, 95, 16) | 1.19e-7 | 0.01599 | 1.35811 |
| 13 | (14, 13, 16) | 0 | 0.00619 | 0.68298 |

验证通过：

- import smoke test；
- T=96/48/24/12 及非整除长度 T=95/13；
- invalid candidate probability 为零；
- uniform routing；
- beta=0 identity，最大差值为 0.0；
- balance / entropy 有限；
- backward，主要参数梯度非 None 且有限；
- multiscale 独立模块及形状保持；
- No NaN/Inf；
- CPU smoke test。

反向传播示例梯度范数：

- beta：0.00319
- p=4 temporal projection：0.000128
- p=4 feature projection：0.000187
- router first layer：0.14510
- router final layer：0.07661

参数量：

| 模块 | 参数量 |
|:---|---:|
| PatchCandidateEncoder，D=16，p=4 | 549 |
| LocalAdaptivePatch，D=16，默认候选 | 2,837 |
| MultiScaleAdaptivePatch，4 个尺度 | 11,348 |

## 7. 与 channel independence 的兼容性

在 channel-independence=1 时，TimeMixer 通常会把 [B,T,C] 变为等价的 [B*C,T,1] 或在 embedding 后形成 [B*C,T,D]。因此本模块将第一维解释为有效样本维 B_eff，在每个变量通道的时间序列内部独立执行局部 patch 路由，不跨变量混合，也不错误地把 B*C 当作原始 batch 做 scale routing。

## 8. 已知限制与后续接入点

1. 当前 patch candidate 是固定候选集合上的动态选择，不学习离散 patch 边界；
2. 没有使用 RL、Gumbel、FFT、diffusion 或 causal module；
3. 右 padding 会使边界 patch 包含补零位置，输出阶段通过 crop 恢复原长度；
4. MultiScaleAdaptivePatch 只负责独立多尺度增强和辅助损失汇总，不负责 TimeMixer PDM、future mixing 或 prediction head；
5. 子任务三需要在 TimeMixer 的 embedding/PDM 数据流中接入该模块，并在训练入口合并 prediction loss 与 router auxiliary loss。

## 9. 子任务二结论

独立的 Local Dynamic Patch 模块已经完成，接口为：

[B_eff,T,D] -> [B_eff,T,D]

它满足有效候选过滤、局部动态路由、uniform baseline、残差恒等初始化、辅助损失和多尺度独立封装要求，可作为后续 TimeMixerAdaptivePatch 的增强模块接入原始 PDM。

本文件只记录子任务二结果；截至本结果生成时，没有开始实现子任务三。
