# TimeMixerAdaptivePatch 子任务 4/5：训练框架接入、Auxiliary Loss、CLI 与 Router 监控

> 前置结果：
>
> - `TimeMixer_AdaptivePatch_Review.md`：子任务 1，完成 TimeMixer 结构审查。
> - `TimeMixer_AdaptivePatch_Subtask2_Result.md`：子任务 2，完成独立 Local Dynamic Patch。
> - `Experiment_Codex_20260912_143210.md`：子任务 3，完成 `TimeMixerAdaptivePatch`、Scale Router 与 factorized joint routing 接入。
>
> 子任务 3 已验证：完整 forecast、joint gating 恒等式、`beta=0` baseline degeneration、梯度、CPU/CUDA smoke test 均通过。因此本阶段**不要再修改核心架构设计**。

---

# 0. 本阶段目标

本阶段只完成：

```text
TimeMixerAdaptivePatch
        ↓
TSLib training pipeline
        ↓
prediction loss
+ patch balance regularization
+ scale balance regularization
        ↓
optimizer
```

同时补齐：

```text
CLI 参数
Router statistics logging
消融实验开关
AMP / 非 AMP 训练兼容
baseline compatibility
最小真实训练 smoke test
```

本阶段**不做完整 ETTm1 benchmark，不做四个 pred_len 全量训练，不做超参数搜索**。

---

# 1. 开始前先读取实际实现，不要按旧设计猜

首先读取：

```text
AGENTS.md
layers/AdaptivePatchRouter.py
models/TimeMixerAdaptivePatch.py
models/TimeMixer.py
exp/exp_long_term_forecasting.py
run.py
```

重点确认当前 `TimeMixerAdaptivePatch` 实际暴露的接口：

```python
get_router_aux_losses()
get_router_stats()
```

以及实际使用的 config 字段名称。

子任务 3 的归档显示当前模型已经实现：

```text
patch_balance
scale_balance
```

并且 entropy 只是诊断统计。

**后续必须沿用实际实现，不要重新造一套重复接口。**

---

# 2. 严格保护现有架构和 baseline

本阶段禁止改变：

```text
Local Dynamic Patch 的数学定义
PatchCandidateEncoder
Patch Router
Scale Router
Joint Routing
Hfinal_m = H_m + alpha_m * (Hpatch_m - H_m)
P(m,t,k|X) = alpha_m * pi(m,t,k)
```

除非发现明确 bug，否则不要重构：

```text
models/TimeMixerAdaptivePatch.py
layers/AdaptivePatchRouter.py
```

尤其不要修改：

```text
models/TimeMixer.py
```

原始 TimeMixer 必须继续作为 untouched baseline。

---

# 3. Git index 当前损坏：禁止顺手“修 Git”

子任务 3 已确认：

```text
普通 Git index 当前损坏
常规 git status / git diff 会报 index error
```

本阶段：

**不要修改、删除、重建或 reset `.git/index`。**

禁止为了让 Git 命令正常而执行：

```bash
git reset --hard
git reset
git checkout .
git clean
git read-tree
git update-index --really-refresh
```

也不要覆盖用户已有工作区修改。

如果需要检查修改范围：

- 优先沿用子任务 3 已采用的“临时只读 index”方法；
- 或直接对本阶段明确修改的文件进行内容检查；
- 如果无法安全得到普通 `git diff`，在报告中如实说明。

本任务不要 commit，不要 push TSLib 工作区。

---

# 4. Auxiliary Loss 的正确数学形式

训练目标定义为：

```text
L_total
=
L_pred
+
lambda_patch * L_patch_balance
+
lambda_scale * L_scale_balance
```

其中：

```text
L_pred = MSE(prediction, target)
```

当前模型已有：

```text
L_patch_balance
L_scale_balance
```

二者都是防止 routing 过早 collapse 的 balance regularization。

默认建议：

```text
lambda_patch = 0.001
lambda_scale = 0.001
```

通过 CLI 控制，不允许 hard-code 到 layer 内部。

---

# 5. 非常重要：不要把正 entropy loss 加入优化目标

子任务 2 曾在说明中举过类似：

```text
+ lambda_entropy * entropy
```

本阶段**不要这样做**。

原因：训练是最小化 loss，如果直接：

```text
+ positive_lambda * entropy
```

会鼓励更低 entropy，使 Router 更尖锐，可能反而促进 collapse。

因此当前阶段：

```text
patch entropy
scale entropy
```

**只用于 logging / diagnosis，不参与 total loss。**

如果未来要做 entropy regularization，必须重新明确符号和研究目的；本任务不做。

---

# 6. 训练入口：只修改训练分支

检查真实：

```text
exp/exp_long_term_forecasting.py
```

找到 AMP 与非 AMP 两条训练路径。

当前 baseline 逻辑大致是：

```python
prediction_loss = criterion(outputs, batch_y)
```

对于支持 router aux loss 的模型，训练阶段改成等价逻辑：

```python
prediction_loss = criterion(outputs, batch_y)

patch_balance = ...
scale_balance = ...

total_loss = (
    prediction_loss
    + patch_balance_weight * patch_balance
    + scale_balance_weight * scale_balance
)
```

但必须根据当前实际模型接口实现，不要照抄伪代码。

要求：

```text
AMP 分支       使用 total_loss backward
非 AMP 分支    使用 total_loss backward
```

---

# 7. Validation / Test 不得加入 Router regularization

验证、early stopping 和最终测试指标必须仍然只看 forecasting prediction。

也就是说：

```text
validation loss = MSE(prediction, target)
test metrics    = 原 TSLib forecasting metrics
```

不要改成：

```text
validation loss = prediction + router regularization
```

否则会破坏：

```text
early stopping
baseline fairness
MSE/MAE 的含义
```

因此 auxiliary regularization **只参与训练优化**。

---

# 8. 其他模型必须完全兼容

不要写死：

```python
if model_name == "TimeMixerAdaptivePatch":
```

除非当前框架确实没有更安全方式。

优先基于能力检测，例如检查模型是否支持：

```python
get_router_aux_losses
```

但注意 DataParallel 情况，见下一节。

对于：

```text
TimeMixer
TimesNet
DLinear
其他普通模型
```

如果没有 router auxiliary loss：

```text
total_loss == prediction_loss
```

原训练行为必须保持不变。

---

# 9. DataParallel 是本阶段必须认真验证的风险点

不要简单假设：

```python
self.model.module.get_router_aux_losses()
```

一定正确。

原因：`nn.DataParallel` 的 forward 会在多个 replica 上执行；如果 `get_router_aux_losses()` 依赖 forward 时写入的模块属性，那么这些属性可能只存在于临时 replica，forward 后读取 base module 可能拿不到当前 batch 的正确 loss。

因此请先检查当前：

```text
LocalAdaptivePatch
TimeMixerAdaptivePatch
```

的 aux loss 是如何保存/计算的。

必须明确回答：

> 当前 `get_router_aux_losses()` 在 `nn.DataParallel` 下是否能返回本次 forward 的真实可微 auxiliary loss？

### 如果答案是可以

用实际 smoke test 证明，而不是只凭代码猜测。

### 如果答案是不可以

不要伪装支持。

优先采用最小、明确的兼容方案，例如：

- 为 `TimeMixerAdaptivePatch.forward` 增加一个**仅训练时可选**的 `return_router_aux=False` 参数；
- 普通 validation/test 调用仍返回 prediction tensor；
- training 对支持该能力的模型可以请求 prediction + aux tensors；
- 在 DataParallel gather 后正确 reduce auxiliary tensors。

这是示例思路，不要求机械照做；请根据当前代码选择最小且正确的方案。

**不要为了 DataParallel 大规模重构所有 TSLib 模型接口。**

如果本环境无法可靠完成多 GPU验证：

- 至少保证单 GPU 训练正确；
- 明确报告 DataParallel 的验证状态和限制；
- 不要声称未经验证的兼容性。

---

# 10. CLI 参数：只暴露当前实现真正使用的参数

检查：

```text
models/TimeMixerAdaptivePatch.py
layers/AdaptivePatchRouter.py
```

把实际需要的 config 参数加入 `run.py`。

至少应覆盖以下研究控制项；如果实际代码字段名不同，以当前实现为准，并统一命名。

建议参数：

```text
--patch_sizes 4 8 16 32

--patch_routing learned
--scale_routing learned

--patch_router_hidden 64
--scale_router_hidden 64

--router_temperature 1.0
--scale_temperature 1.0

--adaptive_patch_alpha 0.1

--patch_balance_weight 0.001
--scale_balance_weight 0.001
```

如果当前实现还有真正使用的：

```text
dropout
router type
```

可以暴露。

不要增加当前模型根本不用的“装饰参数”。

---

# 11. `patch_sizes` 的 argparse 必须正确

目标命令：

```bash
--patch_sizes 4 8 16 32
```

在 Python 中必须得到：

```python
[4, 8, 16, 32]
```

建议使用：

```python
nargs="+"
type=int
```

或项目中等价风格。

必须测试：

```text
--patch_sizes 16
```

能够得到单元素候选。

---

# 12. 消融开关必须通过 CLI 完成

代码必须支持以下配置，无需重新修改模型源码。

## A. Full Joint Routing

```text
patch_routing=learned
scale_routing=learned
```

## B. Dynamic Patch Only

```text
patch_routing=learned
scale_routing=uniform
```

## C. Scale Routing Only

```text
patch_routing=uniform
scale_routing=learned
```

## D. Uniform Multi-Patch

```text
patch_routing=uniform
scale_routing=uniform
patch_sizes=4 8 16 32
```

## E. Single Fixed Patch

例如：

```text
patch_routing=uniform
scale_routing=uniform
patch_sizes=16
```

注意：这里的 `Single Fixed Patch` 是 **Adaptive 模型内部的 control**，不是原始 TimeMixer baseline。

---

# 13. Uniform 模式下 auxiliary loss 行为必须合理

检查：

```text
patch_routing=uniform
scale_routing=uniform
```

时：

```text
patch balance
scale balance
```

应当理论上为 0 或数值接近 0。

不要让 uniform 模式产生虚假的 collapse penalty。

测试并报告实际数值。

---

# 14. Router Logging

训练时需要能够观察 Router 是否工作、是否 collapse。

至少输出：

```text
mean scale probabilities
scale entropy

mean patch probabilities per scale
patch entropy per scale
beta per scale
```

使用现有：

```python
get_router_stats()
```

不要另造重复统计逻辑。

所有日志数据必须 detached，不能保存 graph。

---

# 15. Logging 频率要克制

不要每个 batch 打印完整 Router 信息。

优先：

```text
每个 epoch 结束时打印一次
```

或提供：

```text
--router_log_interval
```

但如果增加该参数会让本任务复杂化，epoch-level logging 就够。

日志应简洁，例如：

```text
Router Stats | epoch=1
scale_probs=[0.24,0.28,0.25,0.23]
scale_entropy=1.38
scale0 patch_probs=[...]
scale1 patch_probs=[...]
...
```

不要输出巨大 tensor。

---

# 16. Router Collapse 诊断

增加轻量诊断即可，不要干扰训练。

例如：

```text
if max(mean_scale_prob) > 0.98:
    warning
```

或者：

```text
if max(mean_patch_prob) > 0.98:
    warning
```

这只是 warning：

- 不停止训练；
- 不自动改 temperature；
- 不自动增加 regularization；
- 不动态修改超参数。

如果实现 collapse warning 会显著增加代码复杂度，可以只在最终报告中通过 stats 人工检查，本阶段优先保证训练逻辑正确。

---

# 17. Training log 必须区分三类 loss

训练日志至少能区分：

```text
prediction_loss
patch_balance_loss
scale_balance_loss
total_loss
```

不要只打印一个总 loss，让后面无法判断 regularization 是否失控。

至少在 smoke test / epoch summary 中报告这些值。

---

# 18. 权重为 0 的退化测试

设置：

```text
patch_balance_weight = 0
scale_balance_weight = 0
```

必须验证：

```text
total_loss == prediction_loss
```

允许 machine precision 浮点误差。

这是训练接口正确性的基础测试。

---

# 19. Auxiliary loss 必须参与梯度

使用：

```text
patch_balance_weight > 0
scale_balance_weight > 0
```

做一个最小 backward。

至少确认：

```text
Patch Router grad finite
Scale Router grad finite
Patch Encoder grad finite
beta grad finite
baseline backbone grad finite
```

同时检查：

```text
prediction loss finite
patch balance finite
scale balance finite
total loss finite
```

---

# 20. Optimizer step 必须真的更新 Router

不能只验证 `grad != None`。

至少挑一个：

```text
Patch Router parameter
Scale Router parameter
```

记录：

```text
before optimizer.step()
after optimizer.step()
```

确认参数确实发生变化。

报告：

```text
max abs parameter update
```

或者等价数值。

---

# 21. 非 AMP smoke test

使用真实训练路径或最接近真实训练路径的方法，完成至少：

```text
forward
prediction loss
auxiliary loss
total loss
backward
optimizer.step
```

要求：

```text
PASS
finite
no NaN
no Inf
```

不要只调用模型裸 forward 就称为“training smoke test”。

---

# 22. AMP smoke test

当前子任务 3 归档显示远程环境：

```text
PyTorch 2.5.1+cu121
CUDA available
2 GPUs visible
```

如果该环境仍然可用，测试 TSLib 当前 AMP 分支至少一个最小 step。

验证：

```text
autocast
GradScaler
total_loss
backward
optimizer step
```

均正常。

如果本次运行环境 GPU 不可用：

明确报告 AMP 未实测，不要伪造结果。

---

# 23. 使用真实 ETTm1 数据做 tiny smoke test

如果当前项目已有：

```text
./dataset/ETT-small/ETTm1.csv
```

则优先从真实 dataloader 取少量 batch。

不要跑完整 benchmark。

推荐：

```text
pred_len = 96
2~5 training batches
1 validation batch
```

或等价的极小训练过程。

目标只验证：

```text
data loader
model
loss
backward
optimizer
validation
```

整条链路。

如果数据集不存在：

使用 synthetic smoke test，并明确说明未完成真实 ETTm1 loader 测试。

---

# 24. Validation smoke test

训练若干 step 后执行至少一个 validation batch。

确认 validation 使用：

```text
prediction MSE only
```

并报告：

```text
validation prediction loss
```

不得加入 router auxiliary loss。

---

# 25. Baseline compatibility test

至少测试：

```text
TimeMixer
TimesNet
TimeMixerAdaptivePatch
```

目标不是正式训练，而是确认训练框架修改后：

```text
普通模型不会因为不存在 get_router_aux_losses 而报错
```

对于普通模型必须满足：

```text
total_loss == prediction_loss
```

至少完成 model construction + one train-style loss path。

---

# 26. 自动模型发现不能被破坏

子任务 1 已确认当前 TSLib 自动扫描：

```text
models/*.py
```

因此：

```text
--model TimeMixerAdaptivePatch
```

应该继续自动发现。

不要为了训练接入又增加一套手工 `model_dict`。

---

# 27. 不要修 baseline `x_mark.repeat` 风险

前面审查发现 TimeMixer baseline 中：

```text
x_mark.repeat(N, ...)
```

可能存在 CI 顺序问题。

本阶段仍然不要修。

原因：

```text
TimeMixer vs TimeMixerAdaptivePatch
```

目前必须共享同一 baseline 行为，避免实验变量混杂。

单独记录该风险即可。

---

# 28. 不要开始正式 benchmark

本阶段禁止：

```text
pred_len 96/192/336/720 全部训练
多 seed 完整训练
大量 grid search
寻找最好 patch size
大规模 temperature sweep
正式论文表格
```

这些属于子任务 5 及之后的正式实验阶段。

---

# 29. 本阶段推荐修改文件范围

预计主要修改：

```text
run.py
exp/exp_long_term_forecasting.py
```

如 DataParallel 正确实现确实需要，可以对：

```text
models/TimeMixerAdaptivePatch.py
```

做**最小接口级修改**。

如果不是必要，不要修改：

```text
layers/AdaptivePatchRouter.py
models/TimeMixer.py
```

不要为了代码风格做无关重构。

---

# 30. 必须执行的 CLI parsing tests

至少验证：

```text
--patch_sizes 4 8 16 32
```

解析正确。

验证：

```text
--patch_sizes 16
```

解析正确。

验证：

```text
--patch_routing learned
--scale_routing learned
```

模型构造成功。

验证：

```text
--patch_routing uniform
--scale_routing uniform
```

模型构造成功。

---

# 31. 必须执行的数值验收

至少报告：

```text
prediction_loss
patch_balance_raw
scale_balance_raw
weighted_patch_balance
weighted_scale_balance
total_loss
```

并验证：

```text
total_loss
≈
prediction_loss
+ weighted_patch_balance
+ weighted_scale_balance
```

报告最大绝对误差。

---

# 32. Router stats 验收

至少一个 learned-routing batch 后报告：

```text
scale_probs
scale_entropy
patch_probs per scale
patch_entropy per scale
beta per scale
```

检查：

```text
all finite
```

并确认概率归一化仍然正确。

不要仅报告一个 Python dict 对象地址。

---

# 33. Uniform 消融验收

使用：

```text
patch_routing=uniform
scale_routing=uniform
```

确认：

```text
scale probabilities = 1 / num_scales
```

patch probabilities 对每个 scale 的 active candidates 均匀。

同时报告：

```text
patch_balance ≈ 0
scale_balance ≈ 0
```

---

# 34. Single patch control 验收

使用：

```text
patch_sizes=16
patch_routing=uniform
```

注意 TimeMixer 最低尺度可能长度小于 16。

当前 LocalAdaptivePatch 的规则是：

```text
仅启用 p <= T
```

因此必须检查单 patch control 在所有 TimeMixer scales 上是否可构造。

ETTm1 当前最低尺度：

```text
T3 = 12
```

所以：

```text
patch_sizes=[16]
```

会导致最低尺度没有有效 candidate。

**这是本阶段必须处理清楚的实验设计问题。**

不要悄悄绕过。

优先方案之一：

- single fixed patch control 只允许选择不大于最短 scale 的 patch，例如 `patch_size=4` 或 `8`；

或者：

- 明确设计 per-scale fallback，但这会改变原定义，需要充分说明，不建议在本阶段引入。

因此本阶段至少验证：

```text
patch_sizes=4
patch_sizes=8
```

可以作为全尺度 single-patch control。

对于：

```text
patch_sizes=16
patch_sizes=32
```

如果模型按设计报 `No valid patch size`，这是合理行为；请明确记录，而不是把它标为模型 bug。

---

# 35. 一个需要明确记录的实验事实

ETTm1 当前尺度：

```text
96 / 48 / 24 / 12
```

因此全尺度统一候选集合必须至少包含一个：

```text
p <= 12
```

默认：

```text
[4,8,16,32]
```

是合法的，因为 scale 3 仍有 `[4,8]`。

后续正式固定 patch baseline 不能简单把所有 `{4,8,16,32}` 都当成完全等价的全尺度 single-patch 配置。

这个限制必须进入最终报告，供子任务 5 设计公平消融。

---

# 36. 不要增加新的研究变量

本阶段禁止新增：

```text
Gumbel
RL
learnable arbitrary boundary
FFT router
MoE
cross-variable routing
causal module
diffusion
new attention block
new normalization
```

如果训练表现不好，本阶段也不要“顺便改模型”。

当前任务是验证训练工程链路，而不是追指标。

---

# 37. 最终 PASS Checklist

以下项目必须逐项报告：

```text
[ ] CLI: patch_sizes multi-value PASS
[ ] CLI: patch_sizes single-value PASS
[ ] CLI: learned routing PASS
[ ] CLI: uniform routing PASS

[ ] non-AMP training step PASS
[ ] AMP training step PASS / clearly NOT TESTED

[ ] prediction loss finite PASS
[ ] patch balance finite PASS
[ ] scale balance finite PASS
[ ] total loss finite PASS

[ ] total loss identity formula PASS
[ ] aux weights=0 => total==prediction PASS

[ ] optimizer updates Patch Router PASS
[ ] optimizer updates Scale Router PASS

[ ] learned router stats finite PASS
[ ] uniform router stats correct PASS
[ ] uniform patch balance≈0 PASS
[ ] uniform scale balance≈0 PASS

[ ] validation uses prediction loss only PASS

[ ] TimeMixer baseline train-style path PASS
[ ] TimesNet baseline train-style path PASS
[ ] TimeMixerAdaptivePatch train-style path PASS

[ ] real ETTm1 tiny smoke PASS / clearly NOT TESTED

[ ] no NaN PASS
[ ] no Inf PASS

[ ] baseline TimeMixer source untouched PASS
[ ] no full benchmark performed PASS
```

DataParallel 单独写：

```text
[ ] DataParallel aux-loss correctness PASS
```

如果不能证明，就写：

```text
NOT VERIFIED
```

不要猜测。

---

# 38. 完成后报告格式

完成后不要只说“实现成功”。

必须给出以下结构。

## A. Modified Files

列出：

```text
新增：
修改：
```

并说明每个修改的目的。

## B. Training Objective

写出最终实际代码对应的：

```text
L_total
=
L_pred
+
lambda_patch L_patch
+
lambda_scale L_scale
```

以及默认权重。

## C. CLI

列出所有新增参数：

```text
name
type
default
meaning
```

不要列出没有实际被模型使用的参数。

## D. AMP / Non-AMP

分别说明修改位置及测试结果。

## E. DataParallel

明确回答：

> 当前 auxiliary loss 在 DataParallel 下是否是真实来自各 replica 的本 batch loss？

给证据或写 NOT VERIFIED。

## F. Ablation Configuration

给出：

```text
Full
Patch only
Scale only
Uniform
Single-patch legal controls
```

对应参数。

## G. Numerical Smoke Test

报告：

```text
prediction_loss
patch_balance
scale_balance
total_loss
formula error
```

## H. Gradient / Optimizer

报告：

```text
Patch Router grad norm
Scale Router grad norm
Patch Router parameter update
Scale Router parameter update
```

## I. Router Statistics

展示一个真实 learned batch 和一个 uniform batch 的简洁结果。

## J. Baseline Compatibility

逐项说明：

```text
TimeMixer
TimesNet
TimeMixerAdaptivePatch
```

是否通过 train-style smoke。

## K. ETTm1 Tiny Smoke

说明是否使用了真实 ETTm1 数据、跑了多少 batch，以及结果。

## L. Known Limitations

至少包括：

```text
固定候选 patch 集合
固定 patch 网格
动态 patch 最终恢复原 T
CI 模式不跨变量 routing
x_mark.repeat baseline 风险未修
Git index 仍损坏且本任务未修复
single-patch 受最短 scale 长度约束
```

## M. Workspace Safety

说明：

```text
未 reset
未 clean
未 checkout 覆盖用户修改
未 commit
未 push
```

如果普通 Git index 仍损坏，如实报告。

---

# 39. 最终结论

最后只给出明确判断：

> 当前 `TimeMixerAdaptivePatch` 是否已经完成训练框架接入，并具备进入子任务 5（ETTm1 正式 baseline / ablation 实验准备与最终验收）的条件？

如果答案是否：

列出阻塞项。

完成后停止。

**不要继续执行子任务 5。**

不要进行完整训练。
不要 commit。
不要 push。
