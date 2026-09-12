# TimeMixerAdaptivePatch 子任务 2/5：实现独立 Local Dynamic Patch 模块

> 前置审查：[`TimeMixer_AdaptivePatch_Review.md`](./TimeMixer_AdaptivePatch_Review.md)
>
> 当前阶段只实现独立 Adaptive Patch Layer，不接入 TimeMixer，不修改训练框架，不进行 ETTm1 正式训练。

## 目标

实现并严格验证：

```text
[B_eff, T, D]
        ↓
Local Adaptive Patch
        ↓
[B_eff, T, D]
```

其中在 `channel_independence=1` 时：

```text
B_eff = B * C
```

模块本身完全不需要知道原始 `B` 和 `C`，只将第一维视为有效 batch。

子任务 1 已确认，TimeMixer 的 PDM 与预测层依赖固定时间长度，因此本模块必须是 **shape-preserving residual enhancement**，不能改变送入 PDM 的时间维长度。

---

## 1. 本阶段允许修改的范围

优先只新增：

```text
layers/AdaptivePatchRouter.py
```

如果当前项目已有测试目录并适合增加单元测试，可以新增：

```text
tests/test_adaptive_patch_router.py
```

否则可以使用临时 Python smoke test，不要为了测试重构项目测试体系。

本阶段不要修改：

```text
models/TimeMixer.py
models/TimeMixerAdaptivePatch.py
exp/exp_long_term_forecasting.py
exp/exp_basic.py
run.py
任何 ETTm1 shell script
```

子任务 1 已发现当前工作区存在若干 file mode 改动，并有用户自己的：

```text
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_paper.sh
```

这些都不要修改、删除、reset 或覆盖。

禁止：

```bash
git reset --hard
git clean
git checkout .
```

不要 commit，不要 push。

---

## 2. 模块总体设计

在：

```text
layers/AdaptivePatchRouter.py
```

中建议实现：

```python
PatchCandidateEncoder
LocalAdaptivePatch
```

如确有必要，可以增加：

```python
MultiScaleAdaptivePatch
```

但 MultiScale wrapper 不能依赖 `models.TimeMixer`。

最重要的单尺度接口：

```text
Input:
H [B_eff, T, D]

Output:
H_out [B_eff, T, D]
```

必须严格满足：

```text
H_out.shape == H.shape
```

---

## 3. Candidate Patch Sizes

默认候选：

```text
[4, 8, 16, 32]
```

但不能 hard-code，应通过 constructor 传入，例如：

```python
patch_sizes=(4, 8, 16, 32)
```

运行时根据当前 `T` 自动过滤，仅激活：

```text
p <= T
```

ETTm1 后续实际会出现：

```text
T0 = 96 → [4,8,16,32]
T1 = 48 → [4,8,16,32]
T2 = 24 → [4,8,16]
T3 = 12 → [4,8]
```

Router 不要因不同 `T` 动态创建新的 `nn.Linear`。

推荐始终输出完整候选集合：

```text
K_total = len(patch_sizes)
logits [B_eff, T, K_total]
```

再对无效 patch：

```text
p > T
```

进行 mask。

learned routing 中，无效位置的 logits 应被 mask 到极小值，使对应概率为 0。

uniform routing 中，只对 active patch sizes 均匀分配概率。

例如：

```text
T=12
patch_sizes=[4,8,16,32]

p=4  → 0.5
p=8  → 0.5
p=16 → 0
p=32 → 0
```

---

## 4. PatchCandidateEncoder

对于每个 patch size `p`，实现真正具备 temporal receptive field 的操作。

输入：

```text
H [B_eff, T, D]
```

计算：

```text
N = ceil(T / p)
```

若 `T % p != 0`，在时间维末尾 padding。

形成：

```text
[B_eff, N, p, D]
```

V1 推荐采用简单、稳定、可解释的实现：

```text
[B_eff,N,p,D]
       ↓
permute
       ↓
[B_eff,N,D,p]
       ↓
learnable temporal projection
Linear(p → 1)
       ↓
[B_eff,N,D]
       ↓
small feature projection / MLP
       ↓
[B_eff,N,D]
       ↓
repeat_interleave(p)
       ↓
[B_eff,N*p,D]
       ↓
crop
       ↓
[B_eff,T,D]
```

每个 patch size 必须有独立 temporal projection 参数，例如：

```text
p=4  → Linear(4,1)
p=8  → Linear(8,1)
p=16 → Linear(16,1)
p=32 → Linear(32,1)
```

这样不同 patch size 才真正拥有不同 temporal receptive field。

禁止只使用：

```python
Linear(D, D)
```

然后把它描述为不同 patch size。

---

## 5. Candidate Encoder 输出 enhancement，而不是替代原表示

为了后续让新模型能严格退化回原始 TimeMixer，本阶段不要让 candidate 直接成为完整新表示。

每个 candidate 输出：

```text
Delta_p [B_eff, T, D]
```

其语义是：

> patch size `p` 对原始 representation 提供的 temporal context enhancement。

不要在最终残差外再强制使用会改变 identity 的 LayerNorm。

尤其不要出现 `beta=0` 时仍改变原始 `H` 的操作。

---

## 6. Local Patch Router

必须实现 **local routing**。

不是：

```text
一个 sequence 只选择一个 patch size
```

而是：

```text
不同时间位置可以具有不同 patch granularity preference
```

输入：

```text
H [B_eff,T,D]
```

先计算 temporal variation：

```text
delta_t = abs(H_t - H_(t-1))
```

第一位置：

```text
delta_0 = 0
```

因此：

```text
delta [B_eff,T,D]
```

Router 输入：

```text
concat(H, delta, dim=-1)
```

shape：

```text
[B_eff,T,2D]
```

推荐 Router：

```text
Linear(2D, hidden)
GELU
Dropout
Linear(hidden, K_total)
```

得到：

```text
patch_logits [B_eff,T,K_total]
```

---

## 7. Learned Routing

默认使用 softmax：

```text
pi(t,k) = softmax(logits(t,k) / temperature)
```

得到：

```text
patch_probs [B_eff,T,K_total]
```

必须满足：

```text
sum_k pi[b,t,k] ≈ 1
```

无效 patch size 概率必须为 0。

`temperature` 必须大于 0。

如果传入：

```text
temperature <= 0
```

应明确报错，不能默默产生 NaN。

本阶段不要实现：

```text
reinforcement learning
hard discrete sampling
arbitrary learned boundary
Gumbel-Softmax
```

先把 soft routing 做稳定。

---

## 8. Uniform Routing

必须支持：

```text
routing_mode="learned"
```

以及：

```text
routing_mode="uniform"
```

uniform 模式用于后续消融实验。

例如 active candidates 为：

```text
[4,8,16]
```

则：

```text
pi4  = 1/3
pi8  = 1/3
pi16 = 1/3
pi32 = 0
```

uniform 模式不能依赖 Router logits。

---

## 9. Dynamic Patch Fusion

假设 candidates 产生：

```text
Delta_4
Delta_8
Delta_16
Delta_32
```

每个 shape：

```text
[B_eff,T,D]
```

stack 后：

```text
Delta [B_eff,T,K,D]
```

Router：

```text
pi [B_eff,T,K]
```

计算：

```text
Delta_dynamic[b,t,:]
=
sum_k pi[b,t,k] * Delta_k[b,t,:]
```

最终：

```text
Delta_dynamic [B_eff,T,D]
```

实现时仔细检查 broadcasting。

所有 `permute / reshape / unsqueeze` 都要在注释中写明前后 shape。

---

## 10. Residual Enhancement

最终定义：

```text
H_out = H + beta * Delta_dynamic
```

其中 `beta` 是 learnable scalar。

默认初始化：

```text
beta = 0.1
```

例如：

```python
self.beta = nn.Parameter(torch.tensor(0.1))
```

本阶段不要强制 beta 为正。

最关键要求：

当：

```text
beta = 0
```

时必须严格满足：

```text
H_out == H
```

这个性质以后用于验证：

```text
TimeMixerAdaptivePatch = TimeMixer + Adaptive Component
```

---

## 11. Patch Balance Regularization

Dynamic Router 很可能 collapse 到某一个 patch size。

计算 active patch 的平均使用率：

```text
q_k = mean_{B_eff,T}(pi[:,:,k])
```

然后定义：

```text
L_patch_balance = KL(q || Uniform)
```

可用稳定形式：

```text
sum_k q_k * log(q_k * K_active)
```

需要 epsilon 避免数值问题。

只对 active patch sizes 计算。

无效 patch 不参与 KL。

特殊情况：

```text
K_active = 1
```

则：

```text
balance_loss = 0
```

不能出现：

```text
NaN
Inf
log(0)
除零
```

本阶段只计算 raw balance loss。

不要在 layer 内乘未来的：

```text
patch_balance_weight
```

全局权重以后由模型/训练框架负责。

---

## 12. Router Statistics

`LocalAdaptivePatch` 至少要能返回或查询：

```text
active_patch_sizes
mean_patch_probs
patch_entropy
beta
```

例如：

```python
{
    "active_patch_sizes": [4, 8, 16],
    "mean_patch_probs": [0.21, 0.46, 0.33],
    "patch_entropy": ...,
    "beta": ...,
}
```

所有用于日志的 tensor 必须 `detach()`，不要保留完整 autograd graph。

Router entropy：

```text
H(pi) = -sum pi * log(pi)
```

再在：

```text
B_eff,T
```

维度取平均。

自然对数即可。

若：

```text
K_active = 1
```

则 entropy = 0。

---

## 13. MultiScaleAdaptivePatch（可选但推荐）

如果实现：

```python
MultiScaleAdaptivePatch
```

输入：

```text
[
 H0 [B_eff,T0,D],
 H1 [B_eff,T1,D],
 ...
]
```

输出：

```text
[
 H0' [B_eff,T0,D],
 H1' [B_eff,T1,D],
 ...
]
```

一个关键要求：

**不同 TimeMixer scale 不要默认共享 LocalAdaptivePatch 参数。**

未来应当是：

```text
scale 0
→ 独立 Patch Router + Patch Candidate Encoders

scale 1
→ 独立 Patch Router + Patch Candidate Encoders

...
```

原因是不同 temporal scale 的语义和局部复杂度不同。

推荐：

```python
nn.ModuleList(
    LocalAdaptivePatch(...)
    for each scale
)
```

不要实例化一个 `LocalAdaptivePatch` 再重复用于所有 scale。

本阶段 wrapper 不能 import：

```text
models.TimeMixer
```

它仍然应该是独立 layer。

---

## 14. ETTm1 对应重点测试

子任务 1 已确认：

```text
seq_len = 96
down_sampling_layers = 3
down_sampling_window = 2
d_model = 16
```

因此必须测试：

```text
T=96
T=48
T=24
T=12
```

例如使用：

```text
B_eff = 14
D = 16
```

对应典型：

```text
B=2
C=7
```

CI 模式。

验证：

```text
[14,96,16] → [14,96,16]
[14,48,16] → [14,48,16]
[14,24,16] → [14,24,16]
[14,12,16] → [14,12,16]
```

---

## 15. 非整除长度测试

模块必须通用。

测试：

```text
T=95
```

输入：

```text
[2,95,16]
```

输出必须：

```text
[2,95,16]
```

检查 padding + crop 正确。

再测试：

```text
T=13
```

active patch sizes 应为：

```text
[4,8]
```

输出仍为：

```text
[B_eff,13,D]
```

---

## 16. Probability Tests

learned routing 下验证：

```text
patch_probs.shape == [B_eff,T,K_total]
```

检查：

```text
sum(patch_probs, dim=-1)
```

与 1 的最大误差。

目标应满足 `allclose`。

同时：

```text
p > T
```

的 candidate probability 必须为 0。

---

## 17. Uniform Tests

分别测试：

### T=96

预期：

```text
[0.25,0.25,0.25,0.25]
```

### T=24

预期：

```text
[1/3,1/3,1/3,0]
```

### T=12

预期：

```text
[0.5,0.5,0,0]
```

允许正常浮点误差。

---

## 18. Identity Test

这是本阶段最重要的验收之一。

将：

```text
beta = 0
```

输入随机：

```text
H
```

检查：

```text
H_out == H
```

或者：

```text
max_abs_diff
```

接近 machine precision。

如果 `beta=0` 时仍改变 H，必须修复。

---

## 19. Backward Test

使用：

```text
routing_mode=learned
```

执行类似：

```python
H_out = module(H)

loss = (
    H_out.pow(2).mean()
    + balance_loss
)

loss.backward()
```

确认以下参数全部有有限梯度：

```text
Patch Candidate temporal projection
Patch Candidate feature projection
Patch Router first Linear
Patch Router final Linear
beta
```

要求：

```text
grad is not None
isfinite(grad)
```

不能存在 NaN / Inf。

额外报告各主要模块 gradient norm，确认 Router 真正进入计算图。

---

## 20. 输入合法性测试

至少处理以下情况。

### T < min(patch_sizes)

不能生成空 stack。

可以直接抛清晰异常：

```text
ValueError: No valid patch size for sequence length T=...
```

### 非法 patch size

检查：

```text
patch size <= 0
duplicate patch sizes
```

建议：

- patch size 必须是正整数；
- duplicate 可初始化时去重排序，或明确报错；
- 选一种一致策略并在报告中说明。

### 非法 temperature

```text
temperature <= 0
```

必须明确报错。

---

## 21. 代码质量要求

必须清楚写 shape 注释，例如：

```python
# x: [B_eff, T, D]
# patches: [B_eff, N, p, D]
# candidate: [B_eff, T, D]
# logits: [B_eff, T, K]
# probs: [B_eff, T, K]
```

不要 hard-code：

```text
7
16
96
48
24
12
```

这些数字只能出现在测试里。

模型实现不能依赖：

```text
ETTm1
CUDA
specific batch size
specific number of variables
```

必须能在 CPU 上完成 smoke test。

使用 reshape/permute 时注意 contiguous 问题。

优先使用：

```python
reshape
```

或显式：

```python
contiguous()
```

不要留下 debug print。

---

## 22. 暂时不要处理 x_mark 风险

子任务 1 发现 baseline 中：

```python
x_mark.repeat(N, 1, 1)
```

可能与：

```text
[B,C,T] → [B*C,T,1]
```

的顺序不一致。

这个问题只记录，**本阶段禁止修改**。

原因是本任务只验证 Adaptive Patch Layer，避免同时改变 baseline 与新结构。

---

## 23. 本阶段明确禁止

不要实现：

```text
Scale Router
Joint Scale/Patch Router
TimeMixerAdaptivePatch Model
训练 auxiliary loss integration
ETTm1 training script
Gumbel
RL
arbitrary learned patch boundary
FFT router
diffusion
causal module
MoE
```

不要跑完整训练。

---

## 24. 最终验收标准

以下全部必须 PASS 才算子任务 2 完成：

```text
[ ] import PASS

[ ] T=96 shape PASS
[ ] T=48 shape PASS
[ ] T=24 shape PASS
[ ] T=12 shape PASS

[ ] T=95 non-divisible PASS
[ ] T=13 non-divisible PASS

[ ] learned probability sum=1 PASS
[ ] invalid patch probability=0 PASS

[ ] uniform T=96 PASS
[ ] uniform T=24 PASS
[ ] uniform T=12 PASS

[ ] beta=0 exact identity PASS

[ ] balance loss finite PASS
[ ] entropy finite PASS

[ ] backward PASS
[ ] Patch Encoder grad PASS
[ ] Patch Router grad PASS
[ ] beta grad PASS

[ ] no NaN PASS
[ ] no Inf PASS

[ ] CPU smoke test PASS
```

如果任何测试 FAIL，先修复，再结束本任务。

---

## 25. 完成后必须写出的报告

不要只回答“已完成”。

最终报告必须包含：

### 1. 文件变化

```text
新增：
修改：
```

### 2. 类结构

说明：

```text
PatchCandidateEncoder
LocalAdaptivePatch
MultiScaleAdaptivePatch（若实现）
```

各自负责什么。

### 3. 数学定义

写出：

```text
Delta_p
pi(t,k)
Delta_dynamic
H_out
L_patch_balance
```

### 4. Tensor Shapes

展示：

```text
[B_eff,T,D]
→ patches
→ candidate
→ router
→ fusion
→ [B_eff,T,D]
```

### 5. Active Patch Filtering

展示：

```text
T=96
T=48
T=24
T=12
```

分别启用哪些 patch。

### 6. 测试表

逐项给出：

```text
PASS / FAIL
```

以及关键数值，例如：

```text
probability max sum error
identity max abs diff
balance loss
entropy
gradient norm
```

### 7. 参数量

报告：

```text
LocalAdaptivePatch parameter count
```

最好拆分：

```text
Patch Encoders
Router
beta
```

### 8. 已知限制

至少说明当前版本：

```text
使用固定候选 patch 集合
patch boundary 从固定网格起点切分
不是 arbitrary segmentation
最终恢复原始 T
```

不要把当前实现描述成 fully arbitrary adaptive segmentation。

### 9. Git

最后执行：

```bash
git status
git diff --stat
git diff -- layers/AdaptivePatchRouter.py
```

报告结果。

---

## 最终结论

最后只给出一个明确结论：

> 子任务 2 是否全部通过，Local Dynamic Patch 是否已经具备进入子任务 3、接入 TimeMixer 的条件？

如果存在阻塞问题，明确列出。

完成后停止。

不要继续执行子任务 3。
不要 commit。
不要 push。
