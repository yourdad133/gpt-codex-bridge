# TimeMixerAdaptivePatch 子任务 5/5：ETTm1 实验配置、消融方案与最终工程验收

> 前置结果：
>
> - 子任务 1：TimeMixer 结构审查完成
> - 子任务 2：Local Dynamic Patch 完成并通过 shape / probability / backward / identity 验证
> - 子任务 3：TimeMixerAdaptivePatch + Scale Router + factorized joint routing 完成
> - 子任务 4：training loss、CLI、AMP/非 AMP、DataParallel、router logging 与真实 ETTm1 单 batch smoke test 完成
>
> 本阶段是正式 benchmark 之前的最后一个工程任务。
>
> **目标不是进行大规模超参数搜索，也不是把四个 pred_len 全部完整训练完，而是把正式实验需要的脚本、消融配置、公平性检查、复杂度统计和最终 sanity checks 全部准备好，并明确判断代码是否已经达到“可以开始正式论文实验”的状态。**

---

# 1. 当前已确认事实

子任务 4 已确认：

```text
L_total
=
L_pred
+ lambda_patch * L_patch_balance
+ lambda_scale * L_scale_balance
```

默认：

```text
lambda_patch = 0.001
lambda_scale = 0.001
```

并已验证：

- 非 AMP 训练入口 PASS；
- AMP 训练入口 PASS；
- 两卡 DataParallel auxiliary loss PASS；
- patch router / scale router / patch encoder / beta / backbone 均有 finite gradient；
- learned/uniform routing PASS；
- baseline TimeMixer / TimesNet compatibility PASS；
- validation/test 不加入 router auxiliary loss；
- entropy 只作为诊断统计，不作为正权重最小化项。

因此本阶段不要重新设计 loss。

---

# 2. 非常重要的实验约束：固定 Patch 对照

当前 ETTm1 配置：

```text
seq_len = 96
down_sampling_layers = 3
down_sampling_window = 2
```

对应尺度长度：

```text
T0 = 96
T1 = 48
T2 = 24
T3 = 12
```

当前 LocalAdaptivePatch 的规则是：

```text
patch_size <= 当前 scale 的 T
```

因此：

```text
patch=4  → 所有 scale 合法
patch=8  → 所有 scale 合法
patch=16 → T3=12 非法
patch=32 → T2/T3 非法
```

子任务 4 已实际验证：

```text
patch_sizes=[16]
```

在最短 scale 会按设计报：

```text
No valid patch size
```

所以本阶段 **禁止** 直接把：

```text
patch=4 / 8 / 16 / 32
```

都当作“所有尺度统一 fixed patch”进行对照。

V1 正式 fixed-patch control 只要求：

```text
patch=4
patch=8
```

两种全尺度合法设置。

此外可以设计一个可选的 **scale-aware fixed control**，例如：

```text
scale0: 32
scale1: 32
scale2: 16
scale3: 8
```

但仅当当前代码能通过很小修改清晰支持时才实现。

如果实现 scale-aware fixed control 会迫使模型重构或扩大本阶段范围，则不要实现，只在最终报告中把它列为后续实验扩展。

不要为了支持 patch=16/32 全尺度 fixed control 而改变 LocalAdaptivePatch 的有效候选规则。

---

# 3. 本阶段允许修改的主要文件

预计可以新增：

```text
scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh
```

必要时可以再增加一个专门的 ablation 脚本，例如：

```text
scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1_ablation.sh
```

如果一个脚本能够清晰组织所有配置，则不要为了拆分而拆分。

可以新增研究说明文件，例如：

```text
TimeMixerAdaptivePatch_NOTES.md
```

如果确有明确 bug 才允许小范围修改：

```text
models/TimeMixerAdaptivePatch.py
layers/AdaptivePatchRouter.py
exp/exp_long_term_forecasting.py
run.py
```

但本阶段原则是：

> **冻结核心模型设计，只做实验准备和 bug fix，不继续扩展 architecture。**

禁止修改：

```text
models/TimeMixer.py
```

除非发现会导致 baseline 本身无法运行的现存致命问题；即使发现，也先报告，不要自行修改 baseline。

---

# 4. Git index 风险

前序任务确认普通 Git index 当前损坏。

本阶段：

**不要尝试修复 Git index。**

禁止：

```bash
git reset --hard
git clean
git checkout .
rm .git/index
任何会重建/覆盖用户工作区的 Git 操作
```

如果普通：

```bash
git status
git diff
```

仍报 index error，应继续采用前序任务使用的安全、只读检查方式。

不要让“Git index 损坏”阻塞模型实验准备。

不要 commit。
不要 push。

---

# 5. 读取并锁定 ETTm1 baseline

找到当前正式 tracked baseline：

```text
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
```

以它作为公平比较基准。

重新确认并记录：

```text
seq_len
label_len
pred_len
e_layers
d_model
d_ff
batch_size
learning_rate
train_epochs
patience
down_sampling_layers
down_sampling_window
down_sampling_method
channel_independence
moving_avg
itr
```

已有未跟踪文件：

```text
TimeMixer_ETTm1_paper.sh
```

不要把它自动替换成 baseline。

除非明确说明，否则本阶段实验脚本应优先继承 tracked `TimeMixer_ETTm1.sh` 的训练设置。

---

# 6. 创建正式 Adaptive ETTm1 脚本

创建：

```text
scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh
```

总体原则：

除了：

```text
--model TimeMixerAdaptivePatch
```

和 Adaptive Patch / Router 专属参数外，其余 baseline 超参数尽量与 `TimeMixer_ETTm1.sh` 一致。

默认 Full Model 推荐显式写出：

```text
--patch_sizes 4 8 16 32
--patch_routing learned
--scale_routing learned
--patch_router_type softmax
--patch_router_hidden 64
--router_temperature 1.0
--scale_temperature 1.0
--adaptive_patch_alpha 0.1
--patch_balance_weight 0.001
--scale_balance_weight 0.001
```

如果实际 CLI 参数名与上述略有不同，以子任务 4 已实现的真实参数名为准，不能凭记忆重复添加新参数。

---

# 7. 正式需要准备的实验组

必须保证以下实验都可以仅通过 CLI / shell 配置完成，不重新改 Python 代码。

## A. Baseline

```text
Model: TimeMixer
```

这是最终主 baseline。

---

## B. Uniform Multi-Patch

```text
Model: TimeMixerAdaptivePatch
patch_sizes = [4,8,16,32]
patch_routing = uniform
scale_routing = uniform
```

目的：

> 判断“只是增加多 patch representation”本身是否有效，而不依赖 learned routing。

---

## C. Local Dynamic Patch Only

```text
patch_routing = learned
scale_routing = uniform
```

目的：

> 单独验证 local adaptive patch routing 的贡献。

---

## D. Scale Router Only

```text
patch_routing = uniform
scale_routing = learned
```

目的：

> 单独验证 learned scale importance 的贡献。

注意这里仍然存在 multi-patch branch，只是 patch routing 为 uniform。

最终报告中不要把它错误描述为“完全没有 patch module”。

---

## E. Full Joint Router

```text
patch_routing = learned
scale_routing = learned
```

这才是完整：

```text
P(m,t,k|X)
=
P(m|X) P(k|m,t,X)
```

---

## F1. Fixed Patch = 4

```text
patch_sizes = [4]
patch_routing = uniform
scale_routing = uniform
```

---

## F2. Fixed Patch = 8

```text
patch_sizes = [8]
patch_routing = uniform
scale_routing = uniform
```

注意：

```text
patch=16 / patch=32
```

不属于当前“所有尺度统一 fixed patch”的合法实验，不要生成会失败的正式命令。

---

## G. No Balance Regularization

Full Joint Router，但：

```text
patch_balance_weight = 0
scale_balance_weight = 0
```

目的：

> 判断 balance regularization 是否真的必要，以及 learned router 是否会 collapse。

这一组对论文解释很有价值，建议准备配置。

---

# 8. 不要一次执行全部正式 benchmark

本阶段不要直接运行：

```text
7 个实验 × 4 个 pred_len × 多 itr
```

这样的完整矩阵。

只做：

1. shell/CLI 正确性检查；
2. model construction；
3. forward；
4. 必要的小规模 smoke；
5. 最多选择 `pred_len=96` 做极短训练确认各实验组能真正启动。

正式长时间 benchmark 留到子任务 5 完成之后再开始。

---

# 9. 四个 pred_len 的 forward sanity check

至少验证：

```text
pred_len = 96
pred_len = 192
pred_len = 336
pred_len = 720
```

每一种只需要：

```text
model construction + real/synthetic batch forward
```

无需完整训练。

ETTm1 `seq_len=96` 情况下，确认输出分别为：

```text
[B,96,C]
[B,192,C]
[B,336,C]
[B,720,C]
```

检查 Dynamic Patch 是否意外 hard-code `pred_len=96`。

---

# 10. Seq Length 通用性 sanity check

如果当前 TimeMixer 配置允许，至少额外测试：

```text
seq_len=192
```

保持：

```text
down_sampling_window=2
down_sampling_layers=3
```

对应理论尺度：

```text
192 / 96 / 48 / 24
```

只做 construction + forward。

目的：

确认 Adaptive Patch / Scale Router 没有偷偷 hard-code：

```text
96 / 48 / 24 / 12
```

如果 baseline TimeMixer 本身对该临时配置存在其它限制，应明确区分：

```text
baseline restriction
vs
Adaptive Patch bug
```

---

# 11. Patch Edge Cases 最终复核

至少复核：

```text
patch_sizes=[4]
patch_sizes=[8]
patch_sizes=[4,8,16,32]
```

并确认：

```text
T=12 scale
```

对完整候选集合仍只激活：

```text
[4,8]
```

无效 patch probability 必须严格为 0。

同时明确验证：

```text
patch_sizes=[16]
```

确实会在 T=12 scale 抛出预期异常。

这个 FAIL 是设计约束，不应误判为实现 bug。

---

# 12. Uniform Router 正确性最终检查

在模型级别验证：

```text
scale_routing=uniform
```

四尺度时：

```text
scale probs = [0.25,0.25,0.25,0.25]
```

对于：

```text
patch_sizes=[4,8,16,32]
```

检查：

```text
T=96 → [0.25,0.25,0.25,0.25]
T=24 → [1/3,1/3,1/3,0]
T=12 → [0.5,0.5,0,0]
```

允许正常浮点误差。

---

# 13. Learned Router 正确性最终检查

至少做若干 optimizer steps 后确认：

```text
Patch Router parameters changed
Scale Router parameters changed
```

并验证：

```text
sum_k pi(m,t,k) = 1
sum_m alpha_m = 1
```

所有值 finite。

比较不同：

```text
sample
variable stream
time position
scale
```

的 patch probabilities。

目标不是要求训练几步后就出现巨大差异，而是确认：

> router 的输出确实依赖输入，而不是所有位置由代码 bug 强制成完全相同概率。

如果初始化时接近 uniform，这是正常的。

---

# 14. Router Collapse 检查

使用 Full Joint Router 做一段很短的真实训练或若干 optimizer steps。

分别记录：

```text
mean scale probs
scale entropy
mean patch probs per scale
patch entropy per scale
```

再用：

```text
patch_balance_weight=0
scale_balance_weight=0
```

做同级别短 smoke。

本阶段不要求得出统计显著结论，只需要验证：

- balance loss 开关有效；
- weight=0 时 prediction training 正常；
- router stats 可观察；
- 没有立刻出现 NaN/Inf；
- collapse warning 不会中断训练。

---

# 15. Baseline Degeneration 最终复核

再次验证：

当 adaptive contribution 被关闭时，例如：

```text
所有 LocalAdaptivePatch beta = 0
```

并同步 baseline 共有参数后：

```text
TimeMixerAdaptivePatch output
≈
TimeMixer output
```

之前子任务 3 已得到：

```text
max abs diff = 0.0
```

本阶段只做一次回归验证，防止子任务 4 的 training-interface 修改意外破坏这一性质。

目标仍应接近：

```text
0
```

如果不再成立，必须先定位原因再进入正式 benchmark。

---

# 16. Baseline Regression

必须再次验证原始：

```text
TimeMixer
```

能够正常：

```text
construction
forward
train-style loss
optimizer step
```

如果方便，再保留子任务 4 已有的：

```text
TimesNet
```

compatibility smoke。

目标是确保新增 CLI 和训练辅助损失逻辑没有改变普通模型行为。

普通模型：

```text
patch aux = 0
scale aux = 0
```

且总 loss 与原 prediction loss 一致。

---

# 17. 参数量最终统计

重新报告：

```text
TimeMixer total parameters
TimeMixerAdaptivePatch total parameters
absolute increase
percentage increase
```

前序结果约为：

```text
TimeMixer              75,497
TimeMixerAdaptivePatch 98,561
increase               23,064
increase               ≈30.55%
```

本阶段重新用当前最终代码统计一次。

如果结果变化，必须解释为什么。

进一步拆分：

```text
Adaptive Patch parameters
Scale Router parameters
```

---

# 18. Runtime Overhead

进行一个简单但规范的 engineering benchmark。

比较：

```text
TimeMixer
vs
TimeMixerAdaptivePatch Full Joint Router
```

使用同样：

```text
batch size
seq_len
pred_len
dtype
device
```

建议：

```text
warmup 若干次
然后重复 forward 多次
同步 CUDA 后统计平均时间
```

如果测试 GPU，必须使用：

```python
torch.cuda.synchronize()
```

避免异步 CUDA 导致错误计时。

至少报告：

```text
mean forward time
relative slowdown
```

如果方便，可以再报告峰值显存：

```text
max_memory_allocated
```

但不要伪造 FLOPs。

如果没有可靠 FLOP 工具：

明确写：

```text
FLOPs not measured
```

不要用参数量代替 FLOPs。

---

# 19. Router 可解释性样例

用真实 ETTm1 batch 做一次 learned model forward（可在少量训练 step 后）。

至少输出一个简洁示例：

```text
Scale probabilities

scale0: ...
scale1: ...
scale2: ...
scale3: ...
```

每个 scale 再输出：

```text
mean patch probabilities
```

例如：

```text
scale0:
p4  ...
p8  ...
p16 ...
p32 ...
```

对于 scale3：

```text
p16 = 0
p32 = 0
```

必须保持。

不要只打印一个全局平均然后声称模型实现了 local routing。

如果当前 stats API 只提供 scale-level mean，那么额外在测试代码中查看少量：

```text
pi[b,t,:]
```

以确认不同时间位置可以不同。

不要求把这些 debug 输出长期保留在训练代码中。

---

# 20. 实验命名规范

正式 shell 脚本中的 `model_id` / experiment name 必须能区分不同消融组。

例如：

```text
ETTm1_96_96_TimeMixer
ETTm1_96_96_TMAdaptive_uniform
ETTm1_96_96_TMAdaptive_patch_only
ETTm1_96_96_TMAdaptive_scale_only
ETTm1_96_96_TMAdaptive_joint
ETTm1_96_96_TMAdaptive_fixed4
ETTm1_96_96_TMAdaptive_fixed8
ETTm1_96_96_TMAdaptive_joint_nobalance
```

具体格式遵循当前 TSLib shell 风格。

不能让不同实验覆盖到同一个 checkpoint/results 路径。

---

# 21. Seed / itr 公平性

检查当前 TSLib：

```text
random seed
itr
```

实际怎么控制。

不要擅自重构 seed 机制。

最终报告必须说明：

- baseline 与 Adaptive 是否使用相同 seed policy；
- `itr` 的含义；
- 正式论文实验建议至少几次重复。

本阶段不要求把所有重复实验跑完。

如果现有脚本 `itr=1`，不要无依据改成其它数字；可以在 NOTES 中建议正式结果后续使用多次重复。

---

# 22. Metrics 公平性

确认正式 ETTm1 long-term forecast 使用的最终指标仍是 TSLib 原始：

```text
MSE
MAE
```

以及项目真实 test pipeline 中输出的其它指标（若有）。

Router auxiliary loss：

```text
不得计入最终 forecasting metrics
```

validation early stopping 也仍以原 prediction loss 为准。

必须在最终报告再次确认。

---

# 23. 创建研究说明文档

建议新增：

```text
TimeMixerAdaptivePatch_NOTES.md
```

至少包含以下内容。

## Motivation

不同 temporal scale 与不同 local regions 可能适合不同 patch granularity。

## Base Architecture

```text
TimeMixer multiscale
→ embedding
→ Local Adaptive Patch
→ Scale Router
→ PDM
→ future mixing
```

## Local Patch Routing

```text
pi(m,t,k) = P(k | m,t,X)
```

## Scale Routing

```text
alpha_m = P(m | X)
```

## Factorized Joint Routing

```text
P(m,t,k|X)
=
alpha_m * pi(m,t,k)
```

## Training Objective

```text
L_total
=
L_pred
+ lambda_patch * L_patch_balance
+ lambda_scale * L_scale_balance
```

## Ablations

列出：

```text
A Baseline TimeMixer
B Uniform Multi-Patch
C Dynamic Patch Only
D Scale Router Only
E Full Joint Router
F1 Fixed Patch 4
F2 Fixed Patch 8
G Joint Router without balance regularization
```

## Current Limitations

必须诚实写明：

1. patch size 来自离散候选集合；
2. patch boundary 仍基于固定网格，不是 arbitrary learned segmentation；
3. dynamic patch 最终恢复为原始 `T_m`；
4. PDM 仍是 fixed-length architecture；
5. channel-independence 模式下 routing 是每个 variable stream 独立进行；
6. 当前不做跨变量 joint routing；
7. patch=16/32 无法作为所有尺度统一 fixed patch control，因为最短 scale 是 T=12；
8. baseline `x_mark.repeat` 潜在风险仍未在本研究变量中修改。

不要把模型描述成：

```text
fully arbitrary adaptive segmentation
```

因为当前并不是。

---

# 24. 不要修复 baseline x_mark.repeat 风险

前序审查发现：

```python
x_mark.repeat(N, 1, 1)
```

可能与 CI reshape 顺序存在错位风险。

本阶段依然：

**不要修改 baseline。**

原因：

正式比较必须先保持：

```text
TimeMixer
vs
TimeMixerAdaptivePatch
```

共享同一 baseline 数据处理行为。

把这一点记录到 Known Limitations / Future Cleanup 即可。

---

# 25. 本阶段禁止扩展的新研究点

不要加入：

```text
Gumbel routing
reinforcement learning
learned arbitrary patch boundary
FFT router
diffusion
causal discovery
cross-variable graph
MoE
new attention block
新的 loss family
```

本阶段不再扩大论文变量。

---

# 26. 最终 Test Matrix

最终报告必须逐项给出 PASS / FAIL：

```text
[ ] Python compile/import

[ ] TimeMixer baseline construction
[ ] TimeMixer baseline forward
[ ] TimeMixer baseline train-style optimizer step

[ ] TimeMixerAdaptivePatch construction
[ ] pred_len=96 forward
[ ] pred_len=192 forward
[ ] pred_len=336 forward
[ ] pred_len=720 forward

[ ] seq_len=192 sanity forward（若 baseline 支持）

[ ] patch_sizes=[4]
[ ] patch_sizes=[8]
[ ] patch_sizes=[4,8,16,32]
[ ] patch_sizes=[16] expected failure at T=12 correctly recognized

[ ] uniform patch router
[ ] uniform scale router
[ ] learned patch router
[ ] learned scale router

[ ] patch probability sums to 1
[ ] scale probability sums to 1
[ ] invalid patch probability = 0

[ ] patch router parameters update
[ ] scale router parameters update

[ ] auxiliary loss finite
[ ] zero balance weights work
[ ] AMP smoke
[ ] non-AMP smoke
[ ] DataParallel regression smoke（如果环境仍为双卡）

[ ] beta=0 baseline degeneration
[ ] no NaN
[ ] no Inf

[ ] parameter count
[ ] runtime benchmark

[ ] all ablation commands parse
[ ] experiment names do not collide
```

若某项不适用，要写：

```text
N/A + 原因
```

不能简单省略。

---

# 27. 最终必须给出的正式实验命令

报告中必须给出可以复制执行的命令或 shell entry，至少包括：

```text
A. TimeMixer baseline
B. Uniform Multi-Patch
C. Dynamic Patch Only
D. Scale Router Only
E. Full Joint Router
F1. Fixed Patch 4
F2. Fixed Patch 8
G. Full Joint Router without balance regularization
```

至少以：

```text
ETTm1 / seq_len=96 / pred_len=96
```

给出完整命令。

然后说明怎样把：

```text
pred_len
```

替换成：

```text
192 / 336 / 720
```

不要在报告中只写伪代码参数组合。

---

# 28. 推荐正式 benchmark 顺序

在最终报告里给出推荐的实际跑实验顺序：

```text
Stage 1
TimeMixer baseline + Full Joint Router
pred_len=96
```

先判断主方法是否有信号。

如果 Full Joint Router 明显不能训练或明显劣于 baseline，再分析而不是立刻把全部矩阵跑完。

然后：

```text
Stage 2
pred_len=96 全部 ablations
```

再然后：

```text
Stage 3
主 baseline + full model
pred_len=192/336/720
```

最后再决定是否值得把所有 ablations 扩展到所有 horizon。

这样可以节约大量 GPU 时间。

---

# 29. 最终结果报告格式

完成后不要只说“可以训练”。

必须输出结构化报告：

## A. Final File Changes

新增 / 修改文件。

## B. Final Architecture

完整数据流：

```text
x
→ multiscale
→ embedding
→ adaptive patch
→ scale/patch joint routing
→ PDM
→ prediction
```

## C. Mathematical Definition

明确写：

```text
pi(m,t,k)
alpha_m
P(m,t,k|X)
L_patch_balance
L_scale_balance
L_total
```

## D. Experiment Matrix

A-G 每一组含义与配置。

## E. Test Matrix

逐项 PASS / FAIL / N/A。

## F. Parameter Overhead

baseline vs adaptive。

## G. Runtime Overhead

时间和相对 slowdown；显存如有则报告。

## H. Router Example

真实 ETTm1 batch 的 scale / patch routing 示例。

## I. Formal Commands

所有正式实验命令。

## J. Known Limitations

诚实说明。

## K. Remaining Risks

包括：

```text
Git index broken
x_mark.repeat baseline risk
fixed patch > 12 limitation
```

## L. Final Recommendation

最后必须明确回答：

> 当前 TimeMixerAdaptivePatch 是否已经达到“可以开始正式 ETTm1 baseline / full model / ablation benchmark”的工程状态？

答案只能是：

```text
YES
```

或：

```text
NO
```

如果是 NO，列出所有 blocker。

如果是 YES，再给出**第一条最推荐执行的正式实验命令**。

---

# 30. 完成后停止

本任务完成后：

- 不要自行开始四个 horizon 的完整训练；
- 不要大规模 grid search；
- 不要 commit；
- 不要 push；
- 不要修 Git index；
- 不要继续扩展 architecture。

只完成最终工程验收与正式实验准备，然后停止。
