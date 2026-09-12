# TimeMixerAdaptivePatch 子任务 3/5：接入 TimeMixer + Scale Router + Joint Routing

> 前置审查：[`TimeMixer_AdaptivePatch_Review.md`](./TimeMixer_AdaptivePatch_Review.md)
>
> 子任务 2 指令：[`TimeMixer_AdaptivePatch_Subtask2.md`](./TimeMixer_AdaptivePatch_Subtask2.md)
>
> 子任务 2 结果：[`TimeMixer_AdaptivePatch_Subtask2_Result.md`](./TimeMixer_AdaptivePatch_Subtask2_Result.md)

## 本阶段目标

子任务 2 已完成并验证独立的 Local Dynamic Patch：

```text
[B_eff, T, D]
        ↓
LocalAdaptivePatch
        ↓
[B_eff, T, D]
```

并确认：

- `PatchCandidateEncoder` 正常；
- learned / uniform patch routing 正常；
- invalid patch probability 为 0；
- `beta=0` 时严格恒等；
- `MultiScaleAdaptivePatch` 对每个 scale 使用独立参数；
- CPU forward / backward / gradient 均通过；
- 当前 patch 模块不会改变时间长度。

现在开始子任务 3。

本阶段只完成：

```text
TimeMixer
+
Local Dynamic Patch
+
Scale Router
+
Factorized Joint Scale/Patch Routing
```

即创建新的模型：

```text
models/TimeMixerAdaptivePatch.py
```

本阶段 **不要修改训练 loss、不要增加 run.py CLI、不要创建正式 ETTm1 训练脚本、不要进行完整训练**。

---

# 1. 开始前先重新确认当前工作区

执行：

```bash
git status
git diff --stat
```

确认子任务 2 实际存在：

```text
layers/AdaptivePatchRouter.py
```

并实际阅读该文件，不要只依赖结果报告。

重点确认真实 API：

```text
PatchCandidateEncoder
LocalAdaptivePatch
MultiScaleAdaptivePatch
get_auxiliary_losses()
get_last_stats()
```

以及实际 constructor 参数名称。

如果实际 API 和子任务 2 结果报告存在轻微差异，以 **当前工作区真实代码** 为准。

不要重写已经通过测试的 Adaptive Patch 模块。

除非发现阻塞模型接入的明确 bug，否则本阶段应尽量只新增模型文件。

仍然禁止：

```bash
git reset --hard
git clean
git checkout .
```

不要覆盖用户已有改动。

不要 commit。
不要 push。

---

# 2. 本阶段允许修改的文件

主要新增：

```text
models/TimeMixerAdaptivePatch.py
```

如确实为了修复子任务 2 中的明确接口 bug，可以最小修改：

```text
layers/AdaptivePatchRouter.py
```

但如果修改它：

1. 必须说明原因；
2. 必须重新运行子任务 2 的关键测试；
3. 不允许为了“代码更漂亮”进行无关重构。

本阶段不要修改：

```text
models/TimeMixer.py
exp/exp_long_term_forecasting.py
exp/exp_basic.py
run.py
任何 ETTm1 shell script
```

特别是原始：

```text
models/TimeMixer.py
```

必须保持 baseline 行为不变。

---

# 3. 模型实现策略：优先继承原始 TimeMixer

子任务 1 已确认推荐：

```python
from models.TimeMixer import Model as BaseTimeMixerModel

class Model(BaseTimeMixerModel):
    ...
```

优先采用这个方案。

目标是：

```text
TimeMixerAdaptivePatch
=
原始 TimeMixer backbone
+
Adaptive Patch Enhancement
+
Scale Router
```

而不是复制并重写整个 TimeMixer。

应尽量复用原始：

```text
normalization
embedding
PastDecomposableMixing
future_multi_mixing
projection
predict_layers
```

如果为了在 embedding 和 PDM 之间插入新模块，必须 override `forecast()`，允许复制原始 `forecast()` 中必要的一小段控制逻辑。

但不要复制整个 `models/TimeMixer.py`。

对于原始私有函数：

```text
__multi_scale_process_inputs
```

请先检查本地实现。

优先选择最少重复、最稳定的方式复用。

如果直接调用 name-mangled private method 会造成明显可维护性问题，可以在新模型文件中复制这一小段 multi-scale input processing 逻辑，并清楚注明：

```text
Copied from baseline TimeMixer input preprocessing;
kept behavior identical intentionally.
```

不要因此修改 baseline。

---

# 4. 目标数据流

最终 forecast 数据流必须是：

```text
x_enc [B,L,C]
   │
   ▼
Original TimeMixer multi-scale processing
   │
   ├── X0 [B,T0,C]
   ├── X1 [B,T1,C]
   ├── X2 [B,T2,C]
   └── ...
   │
   ▼
Original per-scale normalization
   │
   ▼
Original channel-independence reshape
   │
   ▼
Original embedding
   │
   ├── H0 [B_eff,T0,D]
   ├── H1 [B_eff,T1,D]
   ├── H2 [B_eff,T2,D]
   └── ...
   │
   ├────────────────────────────┐
   │                            │
   ▼                            ▼
Local Adaptive Patch       Scale Router
per scale                  from original H_m
   │                            │
   ▼                            ▼
Hpatch_m                  alpha_m
   │                            │
   └──────────────┬─────────────┘
                  ▼
      Scale-gated patch enhancement

Hfinal_m = H_m + alpha_m * (Hpatch_m - H_m)

                  │
                  ▼
Original TimeMixer PDM × e_layers
                  │
                  ▼
Original future_multi_mixing
                  │
                  ▼
Prediction [B,pred_len,C]
```

关键约束：

```text
H_m
Hpatch_m
Hfinal_m
```

三者 shape 必须完全一致：

```text
[B_eff,T_m,D]
```

---

# 5. 为什么 Scale Router 要作用在 enhancement 上

子任务 2 已实现：

```text
Hpatch_m
=
H_m + beta_m * Delta_dynamic_m
```

因此定义：

```text
R_m
=
Hpatch_m - H_m
```

就是当前 scale 的 Adaptive Patch enhancement。

Scale Router 不应直接执行：

```text
alpha_m * H_m
```

因为这会衰减原始 TimeMixer representation。

正确设计：

```text
Hfinal_m
=
H_m + alpha_m * R_m
```

代入子任务 2 的 patch routing：

```text
R_m(t)
=
beta_m * sum_k pi(m,t,k) * Delta(m,t,k)
```

因此：

```text
Hfinal_m(t)
=
H_m(t)
+
beta_m * sum_k [alpha_m * pi(m,t,k)] * Delta(m,t,k)
```

这就自然得到联合权重：

```text
P(m,t,k | X)
=
alpha_m * pi(m,t,k)
```

所以本阶段 **不需要重新实现 Patch Router**。

Scale Router 只需要对已有 Adaptive Patch enhancement 做 gating。

---

# 6. Factorized Joint Routing 的数学定义

必须在代码注释和最终报告中明确写出：

```text
Patch Router:
pi(m,t,k)
=
P(k | m,t,X)
```

其中：

```text
pi shape per scale:
[B_eff,T_m,K]
```

Scale Router：

```text
alpha_m
=
P(m | X)
```

其中：

```text
alpha shape:
[B_eff,S]
```

S 为：

```text
S = down_sampling_layers + 1
```

联合路由：

```text
P(m,t,k | X)
=
P(m | X)
P(k | m,t,X)
=
alpha_m * pi(m,t,k)
```

明确称为：

```text
factorized joint scale-patch routing
```

不要把 Scale Router 和 Patch Router 实现成两个完全无关、无法在数学上解释的模块。

---

# 7. Scale Router 输入

对于 embedding 后、尚未经过 Adaptive Patch 的原始：

```text
H_m [B_eff,T_m,D]
```

进行 temporal mean pooling：

```text
g_m
=
mean(H_m, dim=1)
```

得到：

```text
g_m [B_eff,D]
```

V1 请使用 **原始 H_m** 来计算 Scale Router，而不是 `Hpatch_m`。

原因：

```text
alpha_m = P(m | X)
```

应该由该 scale 的原始输入 representation 决定；

而：

```text
pi(m,t,k)
```

负责条件于 scale 和 local region 决定 patch granularity。

这样联合概率解释更干净：

```text
P(m,t,k|X)
=
P(m|X) P(k|m,t,X)
```

并避免 Scale Router 与 patch enhancement 形成不必要的循环依赖。

---

# 8. Scale Router 网络结构

建议在：

```text
models/TimeMixerAdaptivePatch.py
```

中实现一个小模块，例如：

```python
class ScaleRouter(nn.Module):
    ...
```

对于每个 scale：

```text
g_m [B_eff,D]
```

计算一个标量 score。

推荐 V1 使用每个 scale 独立的 scoring MLP：

```text
Linear(D, hidden)
GELU
Dropout
Linear(hidden, 1)
```

即内部：

```python
nn.ModuleList([...])
```

长度：

```text
S = down_sampling_layers + 1
```

输出拼接：

```text
scale_logits [B_eff,S]
```

然后：

```text
alpha
=
softmax(scale_logits / scale_temperature, dim=-1)
```

必须满足：

```text
sum_m alpha[b,m] ≈ 1
```

`scale_temperature` 必须：

```text
> 0
```

否则初始化时直接报清晰异常。

---

# 9. Scale Routing 模式

Scale Router 必须支持：

```text
scale_routing="learned"
```

和：

```text
scale_routing="uniform"
```

## learned

正常：

```text
alpha = softmax(scale_logits / tau)
```

## uniform

完全忽略 scale logits：

```text
alpha_m = 1 / S
```

对所有 scale 均匀分配。

这个接口后面用于消融：

```text
Dynamic Patch only:
patch_routing=learned
scale_routing=uniform

Full Joint:
patch_routing=learned
scale_routing=learned
```

---

# 10. Patch Routing 模式如何传递

子任务 2 已经实现：

```text
routing_mode="learned"
routing_mode="uniform"
```

新模型中不要重新实现。

通过 configs 读取：

```text
patch_routing
```

并传递给：

```text
MultiScaleAdaptivePatch
```

本阶段暂时不修改 `run.py`。

因此所有新配置都必须使用安全默认值：

```python
getattr(configs, "...", default)
```

保证旧的 TimeMixer configs 也能构造新模型进行 smoke test。

---

# 11. 本阶段约定的新模型配置默认值

暂时在模型内部用 `getattr` 支持以下未来 CLI 名称。

推荐统一采用：

```text
patch_sizes              default [4,8,16,32]
patch_router_hidden       default 64
router_temperature        default 1.0
patch_routing             default "learned"

scale_router_hidden       default 64
scale_temperature         default 1.0
scale_routing             default "learned"

adaptive_patch_alpha      default 0.1
```

其中：

```text
adaptive_patch_alpha
```

用于初始化子任务 2 的 `beta`（如果实际 LocalAdaptivePatch constructor 支持相应参数）。

如果子任务 2 实际 constructor 参数名称不同：

- 不要粗暴修改已验证模块；
- 在新模型构造时做正确映射；
- 最终报告中列出映射关系。

本阶段不要增加 argparse。

CLI 接入留给子任务 4。

---

# 12. MultiScaleAdaptivePatch 的使用原则

子任务 2 已实现：

```text
MultiScaleAdaptivePatch
```

并确认：

```text
每个 scale 有独立 LocalAdaptivePatch 参数
```

优先直接复用这个 wrapper。

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
 Hpatch0 [B_eff,T0,D],
 Hpatch1 [B_eff,T1,D],
 ...
]
```

然后计算：

```text
R_m = Hpatch_m - H_m
```

最后：

```text
Hfinal_m
=
H_m
+
alpha[:,m].view(B_eff,1,1) * R_m
```

特别注意 broadcasting shape。

必须明确注释：

```text
alpha[:,m]: [B_eff]
→ [B_eff,1,1]
```

---

# 13. Channel Independence 的正确解释

子任务 1 已确认，当：

```text
channel_independence = 1
```

TimeMixer embedding 后：

```text
[B,T,C]
→
[B*C,T,D]
```

因此本模型中的：

```text
B_eff = B*C
```

Patch Router：

```text
pi(m,t,k)
```

按每个 variable stream 独立计算。

Scale Router：

```text
alpha [B_eff,S]
```

也按每个 variable stream 独立计算。

因此同一个原始 sample 中，不同变量允许得到不同的：

```text
scale preference
patch preference
```

这是设计的一部分，不是 bug。

本阶段不要自行把：

```text
[B*C,...]
```

reshape 成：

```text
[B,C,...]
```

再做跨变量 scale routing。

当前版本不做 cross-variable routing。

---

# 14. 不要修复 baseline 的 x_mark.repeat 风险

子任务 1 已发现原始 TimeMixer 可能存在：

```python
x_mark.repeat(N, 1, 1)
```

与：

```text
[B,C,T] → [B*C,T,1]
```

顺序潜在不一致的问题。

本阶段：

**不要修复。**

原因：

我们当前需要保持：

```text
TimeMixerAdaptivePatch
vs
TimeMixer
```

baseline preprocessing 完全一致。

否则无法判断性能变化来自 Adaptive Patch，还是来自 baseline bug fix。

把该问题继续记录到 Known Limitations 即可。

---

# 15. Scale Balance Loss

为了防止 Scale Router collapse：

计算：

```text
q_m
=
mean_{B_eff}(alpha[:,m])
```

得到：

```text
q [S]
```

定义：

```text
L_scale_balance
=
KL(q || Uniform)
```

稳定形式例如：

```text
sum_m q_m * log(q_m * S)
```

加入 epsilon。

如果：

```text
S = 1
```

则：

```text
L_scale_balance = 0
```

不要出现 NaN/Inf。

本阶段只计算 raw loss。

不要乘最终：

```text
scale_balance_weight
```

这个权重留到子任务 4 的训练框架。

---

# 16. 关于 Patch Entropy：重要修正

子任务 2 结果报告中给出了未来示例：

```text
total_loss
=
prediction_loss
+ lambda_balance * patch_balance_loss
+ lambda_entropy * patch_entropy
```

本阶段 **不要采用这个公式**。

原因：

如果优化器最小化：

```text
+ lambda_entropy * entropy
```

且：

```text
lambda_entropy > 0
```

会倾向于降低 entropy，反而可能鼓励 Router 变得更尖锐甚至 collapse。

因此当前设计规定：

```text
patch_entropy
scale_entropy
```

只作为诊断统计量。

防 collapse 的训练辅助项目前只保留：

```text
patch balance KL
scale balance KL
```

如果未来要显式加入 entropy regularization，必须重新明确符号与目标，不在本阶段处理。

---

# 17. 模型级 Auxiliary Loss API

本阶段不要修改训练入口，但模型必须准备好接口。

建议新模型提供：

```python
def get_router_aux_losses(self):
    ...
```

返回类似：

```python
{
    "patch_balance": differentiable_tensor,
    "scale_balance": differentiable_tensor,
}
```

其中：

```text
patch_balance
```

来自所有 scale 的 Patch Router balance loss 聚合。

如果 `MultiScaleAdaptivePatch` 已经实现了 aggregate auxiliary loss，直接复用。

推荐对多个 scale：

```text
patch_balance
=
mean(scale_patch_balance_losses)
```

而不是 sum。

原因：

避免 `down_sampling_layers` 改变时辅助损失量级线性变化。

但先检查子任务 2 实际 API；如果它已有合理聚合方式，不要重复计算。

`scale_balance` 为本任务新增。

此函数返回的 loss 必须保持计算图，可用于下一阶段 backward。

不要 detach。

---

# 18. Router Statistics API

新模型必须提供：

```python
def get_router_stats(self):
    ...
```

返回 detached stats。

至少包含：

```text
scale_probs
scale_entropy
patch stats for each scale
beta for each scale
```

建议结构：

```python
{
    "scale_probs": ...,
    "scale_entropy": ...,
    "patch": [
        {
            "active_patch_sizes": ...,
            "mean_patch_probs": ...,
            "patch_entropy": ...,
            "beta": ...,
        },
        ...
    ]
}
```

其中：

```text
scale_probs
```

建议返回 batch 平均：

```text
mean_{B_eff}(alpha)
```

所有统计：

```text
detach
```

不要保存整批大 tensor 到长期属性中导致显存泄漏。

如需要保存最近一次 alpha / pi 用于统计，应保证旧 graph 不被保留。

---

# 19. Scale Entropy

计算每个 effective sample：

```text
H(alpha_b)
=
-sum_m alpha_b,m log(alpha_b,m)
```

再对：

```text
B_eff
```

取平均。

如果：

```text
S=1
```

entropy = 0。

Scale entropy 只用于监控，不加入当前 aux loss。

---

# 20. 建议抽出模型内部 helper

为了方便测试，建议在新模型中有一个清晰 helper，例如：

```python
def _apply_adaptive_patch_and_scale_routing(self, enc_out_list):
    ...
```

输入：

```text
List[H_m]
```

输出：

```text
List[Hfinal_m]
```

并在内部更新：

```text
scale router stats
scale balance loss
patch stats
```

这样可以在不经过完整 data preprocessing 的情况下直接测试：

```text
H → Patch → Scale Gate → Hfinal
```

不要为了测试暴露大量无关内部状态。

---

# 21. Forecast 接入位置

Adaptive 模块必须插入在：

```text
embedding 后
PDM 前
```

即原始：

```text
enc_out_list = embedding(...)

for i in range(self.layer):
    enc_out_list = self.pdm_blocks[i](enc_out_list)
```

修改为概念上：

```text
enc_out_list = embedding(...)

enc_out_list = adaptive_patch_and_scale_routing(enc_out_list)

for i in range(self.layer):
    enc_out_list = self.pdm_blocks[i](enc_out_list)
```

不要放到：

```text
raw x_enc 之前
normalization 之前
PDM 之后
future prediction 之后
```

V1 就固定在 embedding → PDM 之间。

---

# 22. Baseline Degeneration Property

本阶段最重要的性质：

如果所有 LocalAdaptivePatch 的：

```text
beta = 0
```

则：

```text
Hpatch_m = H_m
```

于是：

```text
R_m = 0
```

不论 Scale Router alpha 是什么：

```text
Hfinal_m = H_m
```

因此整个模型必须退化成原始 TimeMixer。

这个性质必须通过数值测试验证。

---

# 23. Baseline Degeneration Test

创建：

```text
baseline = TimeMixer.Model(configs)
adaptive = TimeMixerAdaptivePatch.Model(configs)
```

把 baseline 的共有参数同步到 adaptive。

推荐：

```python
missing, unexpected = adaptive.load_state_dict(
    baseline.state_dict(),
    strict=False,
)
```

检查：

```text
missing keys
```

应该只对应 Adaptive Patch / Scale Router 新参数。

不要忽略异常的 baseline missing/unexpected keys。

然后：

1. `baseline.eval()`；
2. `adaptive.eval()`；
3. 将所有 LocalAdaptivePatch 的 beta 设为 0；
4. 使用完全相同的 `x_enc / x_mark_enc / x_dec / x_mark_dec`；
5. 在 `torch.no_grad()` 下比较输出。

要求：

```text
output_baseline.shape == output_adaptive.shape
```

并报告：

```text
max_abs_diff
mean_abs_diff
```

在相同共有权重、eval mode、beta=0 条件下：

目标应接近浮点数值误差。

如果差异明显：

先查明原因，不允许把它当作正常现象跳过。

---

# 24. ETTm1 典型 Shape Test

子任务 1 已确认：

```text
seq_len=96
enc_in=7
d_model=16
down_sampling_layers=3
down_sampling_window=2
channel_independence=1
```

典型：

```text
B=2
C=7
B_eff=14
```

embedding 后应得到：

```text
H0 [14,96,16]
H1 [14,48,16]
H2 [14,24,16]
H3 [14,12,16]
```

Adaptive 后必须仍是：

```text
[14,96,16]
[14,48,16]
[14,24,16]
[14,12,16]
```

Scale Router：

```text
alpha [14,4]
```

完整 forecast：

若：

```text
pred_len=96
```

则：

```text
output [2,96,7]
```

---

# 25. Scale Probability Tests

learned mode：

检查：

```text
alpha.shape == [B_eff,S]
```

并验证：

```text
sum(alpha, dim=-1) ≈ 1
```

报告：

```text
max probability sum error
```

检查：

```text
isfinite(alpha)
```

uniform mode：

当 S=4：

```text
alpha = [0.25,0.25,0.25,0.25]
```

允许正常浮点误差。

---

# 26. Joint Gating Numerical Test

直接对随机 embedding list：

```text
H_m
```

执行 Patch + Scale routing。

检查每个 scale：

```text
R_m
=
Hpatch_m - H_m
```

和：

```text
Hfinal_m - H_m
```

是否满足：

```text
Hfinal_m - H_m
≈
alpha_m * R_m
```

报告：

```text
max joint gating identity error
```

这个测试用于确认 Joint Routing 不是只写在注释里，而是确实作用到 computation graph。

---

# 27. Gradient Test

本阶段要验证完整新模型结构有梯度。

使用 learned patch + learned scale routing。

至少执行：

```python
output = model(...)

aux = model.get_router_aux_losses()

loss = (
    output.pow(2).mean()
    + aux["patch_balance"]
    + aux["scale_balance"]
)

loss.backward()
```

这只是 smoke/backward test，不代表最终训练权重。

检查：

```text
Patch Candidate Encoder grad != None
Patch Router grad != None
Local beta grad != None
Scale Router grad != None
Baseline TimeMixer parameters grad != None
```

所有关键 gradient：

```text
finite
```

报告代表性 gradient norm。

特别确认：

```text
Scale Router final Linear
```

有非零梯度。

如果 Scale Router 没有梯度，说明 gating 没真正参与预测路径或 aux loss，必须修复。

---

# 28. Patch Router 和 Scale Router 的职责不能混淆

最终设计应满足：

```text
Patch Router
负责：
在 scale m 内部、时间位置 t 选择 patch granularity k

Scale Router
负责：
对 effective sample/variable stream 选择各 temporal scale 的重要性
```

不要让 Scale Router 输出：

```text
[B_eff,T,K]
```

不要让 Patch Router 输出：

```text
[B_eff,S]
```

两个层次保持清楚。

---

# 29. 不要在本阶段增加第二套 beta

子任务 2 已经有每个 LocalAdaptivePatch 的 learnable：

```text
beta_m
```

因此本阶段不要再额外加入：

```text
global adaptive beta
joint beta
scale beta
```

当前有效 enhancement：

```text
alpha_m * beta_m * Delta_dynamic_m
```

已经足够。

避免参数语义重复。

未来如实验需要全局强度参数再讨论。

---

# 30. Router Stats 不能保留计算图

模型可以为了日志保存最近一次：

```text
mean scale probs
entropy
patch stats
```

但用于日志的缓存必须 detached。

不要把完整：

```text
alpha [B_eff,S]
patch_probs [B_eff,T,K]
```

带 graph 长期挂在 model 属性上。

如果为了计算本次 aux loss 需要临时 tensor，可以保存到当前 forward 后可取的位置，但注意下一次 forward 覆盖，不能累积历史 graph。

---

# 31. 模型注册检查

子任务 1 已确认当前 TSLib 会自动扫描：

```text
models/*.py
```

因此只要新增：

```text
models/TimeMixerAdaptivePatch.py
```

并包含：

```python
class Model(...):
    ...
```

理论上即可被发现。

本阶段不要修改：

```text
exp/exp_basic.py
```

除非重新检查后发现当前工作区实际状态已经变化。

只做一个 import / discovery smoke test。

---

# 32. 任务类型范围

当前论文原型重点是：

```text
long_term_forecast
```

不要为了：

```text
classification
anomaly_detection
imputation
```

重构模型。

如果继承 BaseTimeMixer 后这些任务仍沿用原始行为，可以保留。

本阶段只要求：

```text
long_term_forecast
```

完整通过。

不要为了兼容所有 task 扩大工作范围。

---

# 33. 本阶段不进行训练框架接入

不要修改：

```text
exp/exp_long_term_forecasting.py
```

即使现在已经有：

```text
patch balance
scale balance
```

也不要加入 training loss。

本阶段只要求模型暴露可微 auxiliary losses。

正式：

```text
prediction_loss
+
lambda_patch * patch_balance
+
lambda_scale * scale_balance
```

在子任务 4 再接入。

---

# 34. 本阶段禁止事项

不要实现：

```text
Gumbel
RL
learnable arbitrary boundary
FFT router
frequency branch
diffusion
causal discovery
cross-variable attention
MoE
top-k hard routing
```

不要修改：

```text
baseline TimeMixer
baseline ETTm1 script
baseline preprocessing
```

不要开始：

```text
ETTm1 10 epoch training
96/192/336/720 完整 benchmark
hyperparameter search
```

本任务的目标只是：

```text
architecture correctness
```

---

# 35. 参数量统计

完成后报告：

```text
Original TimeMixer parameter count
TimeMixerAdaptivePatch parameter count
```

再计算：

```text
absolute increase
percentage increase
```

并尽量拆分新增参数：

```text
Adaptive Patch modules
Scale Router
```

不要把 baseline 参数也算成新增模块参数。

---

# 36. 本阶段必须通过的测试

以下全部完成后，子任务 3 才算通过：

```text
[ ] models/TimeMixerAdaptivePatch.py import PASS

[ ] baseline TimeMixer 未被修改

[ ] model construction PASS

[ ] embedding-scale shapes PASS
    T=96 / 48 / 24 / 12

[ ] Adaptive Patch shape preservation PASS

[ ] learned scale alpha shape PASS

[ ] learned scale probability sum=1 PASS

[ ] uniform scale routing PASS

[ ] scale balance loss finite PASS

[ ] scale entropy finite PASS

[ ] factorized joint gating numerical identity PASS

[ ] full forecast forward PASS
    [2,96,7] -> [2,96,7]

[ ] beta=0 baseline degeneration PASS

[ ] baseline/adaptive shared-state loading checked

[ ] backward PASS

[ ] Patch Encoder gradient PASS

[ ] Patch Router gradient PASS

[ ] beta gradient PASS

[ ] Scale Router gradient PASS

[ ] baseline TimeMixer backbone gradient PASS

[ ] no NaN PASS

[ ] no Inf PASS

[ ] CPU smoke test PASS

[ ] automatic model discovery/import PASS
```

如果任何关键测试 FAIL：

先修复，再结束本任务。

---

# 37. 特别关注的数值报告

最终报告至少给出以下数值：

```text
scale probability max sum error

joint gating identity max error

baseline degeneration max abs diff
baseline degeneration mean abs diff

scale balance loss
scale entropy

Patch Router representative grad norm
Scale Router representative grad norm
beta representative grad norm

Original TimeMixer parameter count
Adaptive parameter count
parameter increase percentage
```

不要只写“PASS”。

---

# 38. 最终报告格式

完成后不要只回复“已实现”。

请按以下结构生成完整结果报告。

## 1. 文件变化

列出：

```text
新增：
修改：
```

如果 `AdaptivePatchRouter.py` 被修改，说明为什么。

## 2. Model Inheritance

说明如何继承 / 复用原 TimeMixer。

如果复制了少量 baseline preprocessing，说明具体复制哪一部分以及原因。

## 3. 完整 Architecture

给出：

```text
Input
→ TimeMixer multi-scale
→ embedding
→ Adaptive Patch
→ Scale Router
→ Joint Gating
→ PDM
→ future mixing
→ Forecast
```

## 4. Mathematical Definition

明确写：

```text
pi(m,t,k)
alpha(m)
P(m,t,k)=alpha(m)pi(m,t,k)
Hfinal_m
```

## 5. Tensor Shapes

至少给 ETTm1：

```text
B=2
C=7
D=16
T=[96,48,24,12]
```

对应 shape。

## 6. Scale Router

说明：

```text
input
pooling
MLP
softmax
uniform mode
```

## 7. Auxiliary Losses

说明：

```text
patch balance
scale balance
```

以及为什么 entropy 当前只做统计、不加入 loss。

## 8. Router Statistics

给出一次实际 smoke forward 的：

```text
mean scale probs
scale entropy
各 scale patch probs
patch entropy
beta
```

## 9. Baseline Degeneration Test

报告：

```text
shared-state load result
missing keys
unexpected keys
max_abs_diff
mean_abs_diff
```

## 10. Gradient Test

逐项报告：

```text
Patch Encoder
Patch Router
beta
Scale Router
baseline backbone
```

是否有有限梯度和代表性 grad norm。

## 11. Parameter Count

报告 baseline 与 adaptive。

## 12. Test Table

将第 36 节所有项目逐项：

```text
PASS / FAIL
```

## 13. Known Limitations

至少保留：

```text
patch size 来自固定候选集合
patch boundary 仍是固定网格
不是 arbitrary segmentation
Dynamic Patch 最终恢复原 T
Scale Router 当前按 B_eff 独立运行
CI 模式不做跨变量 routing
x_mark baseline 风险未修复
未接训练辅助损失
未开始正式 benchmark
```

## 14. Git

最后执行：

```bash
git status
git diff --stat
git diff -- models/TimeMixerAdaptivePatch.py
git diff -- layers/AdaptivePatchRouter.py
```

如果 `AdaptivePatchRouter.py` 未修改，明确说明。

---

# 39. 最终结论

最后只给一个明确判断：

> 子任务 3 是否全部通过，`TimeMixerAdaptivePatch` 是否已经完成架构级接入并具备进入子任务 4（训练框架、CLI、auxiliary loss integration、router logging）的条件？

如果答案是否，列出阻塞项。

完成后停止。

不要继续执行子任务 4。
不要 commit。
不要 push。
