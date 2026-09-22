# Stage 01 — Scale-Aware EMA 数学与工程设计冻结

Status: **PASS**

Stage 00 prerequisite: **PASS**（baseline reproducibility 为 `PARTIAL`，但已明确允许继续）

TSLib audited base commit: `4e938a1767106324dd753b2a44832bf870a0252e`

TSLib source changes in Stage 01: **none**

Training / evaluation / preprocessing run in Stage 01: **none**

## 1. Scope and decision summary

本阶段仅冻结数学定义、工程接口、`channel_independence` v1 边界和 Stage 02 开始前的 baseline-freeze 方案。没有修改 Time-Series-Library 源码，没有实现 EMA，没有执行训练，也没有对 TSLib 执行 `git clean`、`git reset`、`git stash` 或 `git commit`。

v1 的核心决定如下：

- 正式主实验固定 `channel_independence=1`，与已复现 ETTh1 baseline 一致。
- 新增两种相互可区分的分解模式：`ema` 与 `scale_aware_ema`。
- `ema` 在所有尺度使用同一个 `alpha_base`；`scale_aware_ema` 按固定连续时间常数换算每个尺度的系数。
- EMA 只替换 `PastDecomposableMixing`（PDM）内的 decomposition；seasonal bottom-up mixing、trend top-down mixing、FMM、loss、数据划分和评价流程均保持不变。
- v1 对 `EMA + channel_independence=0` 采用显式拒绝，而不是静默保留或替换 `Model.preprocess` 的额外 moving-average split。现有 `moving_avg` / `dft_decomp` 与 `channel_independence=0` 的行为不受影响。
- Stage 02 应在独立 Git worktree 中先创建只包含 seed / DataLoader RNG 必要差异的 baseline-freeze commit；当前 dirty worktree 保持原样。

## 2. Mathematical definition

### 2.1 Recurrence and initialization

对 PDM 输入张量 `x` 的每个 batch-like 行和每个特征维度，沿时间维独立执行：

```text
trend_0    = x_0
trend_t    = alpha_k * x_t + (1 - alpha_k) * trend_{t-1},  t >= 1
seasonal_t = x_t - trend_t
```

`alpha_k` 是该尺度唯一的固定标量，在 batch、原始 channel 和 embedding feature 之间广播；模块不保存跨 batch 状态。

令基础采样间隔为 `Delta`，基础尺度系数为 `alpha_base in (0, 1)`。TimeMixer 的 `down_sampling_window = w`，第 `k` 个尺度的有效采样间隔倍率为：

```text
r_k = w^k,  k = 0, 1, ..., down_sampling_layers
```

Scale-Aware EMA 使用：

```text
alpha_k = 1 - (1 - alpha_base)^(r_k)
```

其中 `k=0` 时 `r_0=1`，所以 `alpha_0=alpha_base`。

### 2.2 Continuous-time interpretation

若固定连续时间常数为 `tau`，指数衰减在一个基础采样间隔后的保留比例是：

```text
1 - alpha_base = exp(-Delta / tau)
```

粗尺度的一个采样步跨越 `r_k * Delta`，因此它的保留比例应为：

```text
exp(-r_k * Delta / tau)
= [exp(-Delta / tau)]^(r_k)
= (1 - alpha_base)^(r_k)
```

于是得到规定的 `alpha_k`。该换算保持的是连续时间意义下相同的 EMA time constant / decay horizon；它不是人为规定“粗尺度必须更追随当前点”。粗尺度单步覆盖的物理时间更长，所以在一次粗尺度更新中需要消耗更多历史权重。

基础时间步单位下的时间常数还可写为：

```text
tau / Delta = -1 / ln(1 - alpha_base)
```

### 2.3 Monotonicity

当 `0 < alpha_base < 1` 时，令 `q = 1 - alpha_base`，则 `0 < q < 1`。当真正下采样，即 `w > 1` 时，`r_k=w^k` 随 `k` 严格增大，`q^(r_k)` 严格减小，因此 `alpha_k=1-q^(r_k)` 严格增大。若 `w=1`，所有 `r_k=1`，Scale-Aware EMA 退化为各尺度相同的 EMA；这不是异常，而是“没有扩大采样间隔”的正确结果。

### 2.4 Why not independent hand-tuned alphas

不为每个尺度分别手工设置 `alpha_0, alpha_1, ...`，原因是：

1. 一个 `alpha_base` 加确定性换算只对应一个研究假设，能够把“EMA 本身”和“scale-aware 换算”分开比较；
2. 独立 alpha 会随尺度数线性增加超参数，扩大 validation 搜索空间，并提高使用 test set 反向调参的风险；
3. 公式对任意合法 `down_sampling_layers` 和 `w` 自动扩展，不需要为每个模型配置重新定义表格；
4. 各尺度系数受同一个物理衰减时间约束，结果更可解释、可复现。

### 2.5 Small and large `alpha_base`

- 当 `alpha_base` 很小时，利用一阶近似，`alpha_k = 1-(1-alpha_base)^(r_k) ~= r_k * alpha_base`（在乘积仍较小时成立）。基础尺度趋势变化缓慢，粗尺度系数随采样间隔近似线性增大。
- 当 `alpha_base` 很大且接近 1 时，`1-alpha_base` 已很小，随 `r_k` 次幂迅速趋近 0，因而 `alpha_k` 很快接近 1。此时趋势几乎等于当前观测，seasonal 残差趋近 0；这也是后续 alpha 搜索必须基于 validation、且不能任意靠近边界的原因。
- `alpha_base=0` 会使趋势永远停留在初值；`alpha_base=1` 会使趋势逐点等于输入。两者均不属于 v1 合法开区间。

### 2.6 Numeric table for `w=2`, `down_sampling_layers=3`

此时四个尺度为 `k={0,1,2,3}`，`r_k={1,2,4,8}`；对应 baseline 的长度为 `96,48,24,12`。

| `alpha_base` | `k=0`, `r=1` | `k=1`, `r=2` | `k=2`, `r=4` | `k=3`, `r=8` |
|---:|---:|---:|---:|---:|
| 0.05 | 0.050000000 | 0.097500000 | 0.185493750 | 0.336579569 |
| 0.10 | 0.100000000 | 0.190000000 | 0.343900000 | 0.569532790 |
| 0.20 | 0.200000000 | 0.360000000 | 0.590400000 | 0.832227840 |

## 3. Frozen v1 research boundary and ablation

### 3.1 Fixed boundary

v1 中：

- alpha 不是 `nn.Parameter`，不可学习；
- alpha 不依赖 instance、channel、embedding feature 或时间位置 `t`；
- 同一尺度上的所有输入共用一个 `alpha_k`；
- EMA 状态只存在于单次 `forward` 的局部计算图中，不跨样本或 batch 持久化；
- 只替换 `models/TimeMixer.py` 中 `PastDecomposableMixing` 的 decomposition 调用；
- `MultiScaleSeasonMixing`、`MultiScaleTrendMixing`、`future_multi_mixing`（FMM）、projection、normalization、loss、optimizer、数据划分和评价指标保持不变；
- 原 `moving_avg` 与 `dft_decomp` 分支保留原构造与原调用语义；
- 正式 v1 主实验固定 `channel_independence=1`。

### 3.2 `channel_independence` v1 boundary

Stage 00 确认：当 `channel_independence=0` 时，`Model.pre_enc()` 会在 embedding 前通过 `self.preprocess = series_decomp(configs.moving_avg)` 再做一次独立的 moving-average split；它与 PDM 内的 decomposition 是两个不同位置。

因此 v1 冻结如下行为：

- `decomp_method in {'ema', 'scale_aware_ema'}` 时，必须要求 `channel_independence == 1`；否则在 `Model.__init__` 尽早抛出带解释的 `ValueError`。
- 错误信息必须指出：`channel_independence=0` 包含额外的 moving-average preprocess，不属于 EMA v1 实验边界。
- 不修改 `Model.preprocess`，也不在 EMA 模式下静默留下“preprocess 用 moving average、PDM 用 EMA”的混合方法。
- `moving_avg` 和 `dft_decomp` 原有的 `channel_independence=0/1` 行为保持不变。
- 若以后研究 `channel_independence=0`，必须另立实验设计，分别定义 preprocess 与 PDM 两个分解点的处理和相应 baseline；不能作为 v1 的顺手扩展。

基线主配置 `channel_independence=1` 会在 embedding 前把原始输入从 `[B,T,N]` 折叠为 `[B*N,T,1]`，PDM 实际收到 `[B*N,T,d_model]`。固定标量 alpha 会广播到这些独立的 batch-like 行，既不会混合原始变量，也不会跨 batch 保存状态。

### 3.3 Required ablation modes

| `decomp_method` | Per-scale coefficient | Purpose |
|---|---|---|
| `moving_avg` | 现有 window=25 行为 | 已复现 baseline |
| `ema` | 对所有 `k`，`alpha_k = alpha_base` | 测量 Moving Average → EMA 的变化 |
| `scale_aware_ema` | `alpha_k = 1-(1-alpha_base)^(w^k)` | 测量 EMA → Scale-Aware EMA 的额外变化 |
| `dft_decomp` | 现有 `top_k` 行为 | 兼容保留，不属于本轮主消融 |

`ema` 与 `scale_aware_ema` 必须使用相同的 `alpha_base` 候选、相同 seed、相同训练配置和相同数据划分。alpha 的选择只能依据 validation；test 指标不得参与选择。

## 4. Engineering interface design

### 4.1 Planned files and minimal edit surface

Stage 02 建议的最小改动面为：

1. **新文件** `layers/EMADecomp.py`：只放固定系数 EMA decomposition 及 alpha 换算 helper；该路径是 Stage 02 计划新增，不是声称当前已经存在。
2. `models/TimeMixer.py`：在现有 `DFT_series_decomp` / PDM 选择逻辑附近导入新模块；仅新增两个显式 `elif` 分支、EMA 的 scale-index 调用和 `channel_independence` guard。
3. `run.py`：在现有 `--moving_avg`、`--decomp_method`、down-sampling 参数附近增加 EMA CLI 参数并更新帮助文字；默认仍是 `moving_avg`。
4. 后续实验脚本应新建在 `scripts/long_term_forecast/ETT_script/` 下，不能覆盖现有 `TimeMixer_ETTh1.sh`；这属于后续阶段，不在 Stage 01 创建。

不把 EMA 塞进 `layers/Autoformer_EncDec.py`，以免扩大对其他模型共享 decomposition 代码的影响面。新模块可以独立单测，同时 TimeMixer 的 moving-average/DFT 路径保持原状。

### 4.2 Class and method contract

冻结建议接口：

```text
class EMADecomposition(nn.Module):
    __init__(
        self,
        alpha_base: float,
        down_sampling_window: int,
        mode: str,  # exactly 'ema' or 'scale_aware_ema'
    )

    alpha_for_scale(self, scale_index: int) -> float

    forward(
        self,
        x: torch.Tensor,       # [batch_like, time, feature]
        scale_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]  # (seasonal, trend)
```

接口约束：

- `forward` 输入/输出均严格为 `[batch_like,time,feature]`，shape 完全相同；返回顺序与现有 `series_decomp` / `DFT_series_decomp` 一致。
- `PastDecomposableMixing.forward` 使用 `enumerate(x_list)` 获得真实 list 顺序的 `scale_index`，而不是从长度反推尺度。
- `down_sampling_window` 在 PDM 构造 EMA 模块时从 `configs.down_sampling_window` 传入一次；系数由模块内部根据 mode 计算。
- `ema` 虽然各尺度 alpha 相同，仍走同一个显式 `scale_index` 接口，使两种 EMA 的张量路径一致，唯一差异是 alpha 换算。
- `down_sampling_layers=0` 时只有 `k=0`，两种 EMA 数学上相同；`w=1` 时两种 EMA 也相同。
- 对 avg/max/conv 下采样，`k` 仍表示第几次 stride=`w` 的采样，故 `r_k=w^k` 按采样间隔定义，不由可能为 floor/ceil 的输出长度决定。正式 v1 主实验仍使用已复现的 avg、`w=2` 配置。

### 4.3 PDM integration without refactoring legacy branches

`models/TimeMixer.py:129-134` 当前只构造 `series_decomp` 或 `DFT_series_decomp`，`models/TimeMixer.py:164-165` 在每个尺度调用同一个 decomposition 实例。Stage 02 只应做以下定向扩展：

- 保存明确的 `self.decomp_method`；
- 对 `ema` / `scale_aware_ema` 构造 `EMADecomposition`；
- 只有 EMA 分支调用 `decomposition(x, scale_index)`；
- legacy 分支继续调用原来的 `decomposition(x)`；
- 不重写 `series_decomp`、`DFT_series_decomp`、season/trend mixing 类或 FMM。

应增加回归检查：同一 baseline checkpoint、同一固定输入、`eval()` 模式下，改动前后 `decomp_method=moving_avg` 的 state-dict keys 和输出保持一致；`dft_decomp` 至少验证其原调用路径与 state-dict keys 未变。

### 4.4 CLI contract and defaults

冻结 CLI：

```text
--decomp_method {moving_avg,dft_decomp,ema,scale_aware_ema}
--ema_alpha 0.10
```

- `--decomp_method` 的默认值继续是 `moving_avg`。
- `--ema_alpha` 表示公式中的 `alpha_base`，默认 `0.10`；仅当方法为 `ema` 或 `scale_aware_ema` 时读取和校验，legacy 模式完全忽略它。
- `--moving_avg`、`--top_k` 和现有 down-sampling 参数名称及默认值不变。
- 可以为 `decomp_method` 增加上述四个显式 choices；现有两个合法值的语义不得改变。
- 不新增 learnable-alpha、per-channel-alpha、per-instance-alpha 或 time-varying-alpha 参数。

### 4.5 Logging and artifact identity

每次 EMA 运行在模型/实验启动时只打印一次结构化摘要，至少包含：

```text
decomp_method, ema_alpha(alpha_base), down_sampling_window,
down_sampling_layers, channel_independence,
[(scale_index, r_k, alpha_k), ...]
```

不能在每个 PDM layer 或每个 batch 重复打印。`run.py` 的 args 输出会保留原始配置；上面的摘要负责记录实际派生 alpha。

当前 checkpoint setting 字符串不直接编码 decomposition 和 alpha，所以后续 EMA 启动脚本必须让 `model_id` 和/或 `des` 明确包含方法、alpha、seed 与 horizon，例如 `saema_a0p10_seed2021`。不得复用或覆盖 baseline 的日志、checkpoint、`test_results` 或 `results` 目录。

### 4.6 Checkpoint behavior

v1 的 alpha、mode、`w` 和 scale index 都是配置/普通 Python 值：

- 不创建 `nn.Parameter`；
- 优先不注册 persistent buffer；若实现需要 tensor 标量，应在 `forward` 中由输入创建，或使用 `persistent=False` buffer；
- 新 EMA 模块不增加 state-dict key；
- 默认 moving-average 模型的旧 checkpoint 应继续 strict-load；
- EMA 模型除配置外没有新增可训练状态。恢复 EMA 训练时必须同时保存/记录完整 CLI 与实验标识，不能只靠 checkpoint 猜 alpha。

## 5. Numerical stability and edge behavior

### 5.1 Validation

构造时必须验证：

- `alpha_base` 是有限实数且严格满足 `0 < alpha_base < 1`；NaN、Inf、0、1 和越界值立即报错；
- `down_sampling_window` 是整数且 `>=1`；
- `mode` 只能是 `ema` 或 `scale_aware_ema`；
- `scale_index` 是非负整数；
- `x` 是三维浮点 tensor，时间维为 dimension 1。

为减少 `alpha_base` 很小或 `r_k` 较大时的消减误差，派生系数应使用 float64 标量形式：

```text
alpha_k = -expm1(r_k * log1p(-alpha_base))
```

它与规定公式数学等价。若极端配置使保留比例在 float64 下溢，`alpha_k` 可数值饱和为 `1.0`，含义是历史权重已低于可表示精度；不得产生 NaN 或超出 `[0,1]` 的值。正式候选范围与四尺度配置不会触发该饱和。

### 5.2 Dtype, device, AMP, and autograd

- 所有工作 tensor 从输入 `x` 派生，绝不隐式创建 CPU/default-dtype 状态；模块支持 CPU/GPU 语义，但项目内验证只能按既定规则在远程环境执行。
- float32/float64 输入分别以原精度递推。
- 允许 fp16/bfloat16 输入，但递推 accumulator 必须显式提升到 float32，最后把 `(seasonal, trend)` 转回输入 dtype；输出 device 与输入一致。
- 递推使用 out-of-place 运算和逐步 list/stack（或等价、不破坏 autograd 的实现）；不得对需要梯度的 view 做危险的原地覆盖，也不得使用 `detach()` / `no_grad()`。
- alpha 固定不代表切断输入梯度；梯度应能从所有 `trend_t` 回传至当前和历史输入。
- Stage 02 必须覆盖 AMP 关闭与开启时的有限输出/有限梯度检查；不能静默把 DFT 分支的 autocast 行为一起改变。

### 5.3 Length and non-finite values

- 空序列 `T=0` 无法定义 `trend_0`，应立即抛出清楚的 `ValueError`。
- `T=1` 时 `trend=x`、`seasonal=zeros_like(x)`，shape、dtype 和 device 均保持。
- 任意合法 `T>=1` 都精确保长，不依赖 moving-average window，因此也不受“长度小于 window”影响。
- 不使用 `nan_to_num` 静默修复输入。NaN/Inf 输入自然传播；单元测试和 smoke test 必须用 `torch.isfinite` 检查输出与梯度，并在失败时报告首个出问题的 mode/scale/dtype。
- 运行期逐 batch 全量 finite scan 可能带来开销，v1 不强制常驻；可以只在测试/调试路径启用。

## 6. Stage 01 verification and non-actions

本阶段重新读取并以其为唯一依据：

- `README.md`
- `CURRENT_STAGE.md`
- `results/00_baseline_audit.md`
- `prompts/01_method_spec.md`

并只读核对了 Stage 00 指向的真实代码位置：

- `models/TimeMixer.py:9-26,118-184,187-216,277-287,289-395`
- `run.py:62-89`
- `layers/Autoformer_EncDec.py:21-53`

数学表按给定公式重新计算。未运行 Python、CUDA、训练、评估、推理或数据预处理；未修改 TSLib 文件；未执行任何 TSLib Git 写操作；未开始 Stage 02。

## Frozen v1 specification

1. PDM 输入 `x:[batch_like,T,D]` 使用 `trend_0=x_0`、`trend_t=alpha_k*x_t+(1-alpha_k)*trend_{t-1}`、`seasonal=x-trend`，输出顺序固定为 `(seasonal, trend)`。
2. `ema` 对所有尺度使用 `alpha_k=alpha_base`；`scale_aware_ema` 使用 `r_k=w^k` 和 `alpha_k=1-(1-alpha_base)^(r_k)`。`alpha_base` 是有限常数且严格位于 `(0,1)`。
3. `scale_index` 来自 `enumerate(x_list)`；`w` 来自 `configs.down_sampling_window`；alpha 不学习、不依赖 instance/channel/time，不跨 forward 保存状态。
4. CLI 固定为 `--decomp_method {moving_avg,dft_decomp,ema,scale_aware_ema}` 和 `--ema_alpha`（默认 `0.10`）。全局默认仍是 `decomp_method=moving_avg`，因此不显式选择 EMA 时 baseline graph、state dict 和数值行为必须保持不变。
5. v1 只替换 PDM decomposition，不改变 preprocess、seasonal/trend mixing、FMM、loss、数据划分、训练超参数或评价指标。
6. 正式 v1 只支持 `channel_independence=1`。EMA 与 `channel_independence=0` 的组合必须显式报错；绝不静默修改 `Model.preprocess` 的 moving-average split。
7. EMA 模块没有 parameter 或 persistent state。fp16/bfloat16 输入用 fp32 accumulator，返回原 dtype；空序列报错，长度 1 返回零 seasonal 和原输入 trend；所有实现保持 autograd。
8. 启动日志必须记录 base alpha、每尺度实际 alpha、`w`、尺度数和 channel 模式；实验标识必须编码 method/alpha/seed/horizon，不能覆盖 baseline artifacts。
9. `moving_avg` 与 `dft_decomp` 的构造、调用和 checkpoint 兼容路径不得重构式修改。alpha 选择只基于 validation，不得使用 test set 调参。

## Stage 02 baseline-freeze procedure

以下步骤是 Stage 02 在首次修改 TSLib 源码前必须执行的方案；Stage 01 **没有执行**这些 Git 写操作。

1. **复核而不改动原 dirty worktree。** 在原工作区记录 `HEAD`、branch、remote、完整 `git status --short`、staged/unstaged diff 和 untracked manifest。若 `HEAD` 不再是 `4e938a1767106324dd753b2a44832bf870a0252e`，或状态与 Stage 00 相比出现无法解释的变化，立即停止并重新审计。
2. **建立私有恢复副本。** 在 TSLib 与 bridge 仓库之外的私有目录保存 tracked binary patch、index patch、untracked 文件清单及其文件副本/校验和。该备份可能含本地配置，禁止提交到任何仓库。验证备份可读后再继续。原 dirty worktree 本身仍保持原样，因此备份是第二重保障。
3. **禁止在原 worktree 清理或切分。** 不在原工作区执行 `clean`、`reset`、`stash`、checkout/switch、commit，也不移动/删除现有文件。先检查 `git worktree list` 和目标 branch 是否已存在；若存在则停止核对，不能覆盖或删除。
4. **从审计 SHA 建立独立 baseline worktree。** 从 `4e938a1767106324dd753b2a44832bf870a0252e` 新建 sibling worktree 与分支 `codex/saema-v1-baseline`。新 worktree 必须显示 clean，且初始 HEAD 必须等于审计 SHA。
5. **只移植真正需要的可复现性差异。** 在新 worktree 手工应用并逐 hunk 审核以下两类非-upstream 改动：
   - `run.py`：`set_random_seed(args.seed)` 的 Python/NumPy/PyTorch/CUDA seed 初始化、`--seed` 默认 2021，以及 parse 后调用；不要带入 AdaptivePatch CLI 参数。
   - `data_provider/data_factory.py`：由 `torch.initial_seed()` 派生 worker 的 Python/NumPy seed，并用以 `args.seed` 初始化的 `torch.Generator` 驱动 DataLoader。
6. **明确排除无关内容。** baseline snapshot 不包含 `exp/exp_long_term_forecasting.py` 的 router auxiliary-loss 路径、M4 修复、AdaptivePatch 文件/脚本、mode-only 改动、README 删除、本地 `.codex`/`AGENTS.md`、报告文件或其他未跟踪实验代码。排除的理由和文件清单写入 freeze commit 的 accompanying report，避免它们被误认为 SAEMA 贡献。
7. **验证最小 snapshot。** 检查 `git diff --check`、完整 diff 和 status；确认相对 upstream 只有 `run.py` 与 `data_provider/data_factory.py` 的上述必要 hunks。随后按项目远程规则完成环境检查，并用不会覆盖既有 artifacts 的最小确定性检查确认相同 seed 的 DataLoader 顺序可重现。若验证失败，停止并报告，不进入 EMA 实现。
8. **提交 baseline freeze。** 只在独立 worktree/branch 上创建一个 commit，建议消息为 `baseline-freeze: preserve deterministic seed and dataloader rng`。记录该 commit SHA 和精确的“非-upstream 改动清单”；可建立本地只读标记 `saema-v1-baseline`。除非另有明确授权和可写 remote，不推送 TSLib 分支。这里的 commit 只属于 Stage 02；Stage 01 未执行。
9. **从 freeze commit 再建实现 worktree。** 从 baseline-freeze SHA 新建第二个 sibling worktree/分支 `codex/saema-v1`。baseline worktree 固定保留用于对照；所有 Stage 02 EMA 源码改动只发生在实现 worktree，绝不回到原 dirty worktree。
10. **建立可归因的 commit 边界。** 后续首个实现 commit 只能包含 `layers/EMADecomp.py`、TimeMixer 的定向接入、CLI 和相应测试；实验脚本/集成按后续阶段另行提交。每个 EMA commit 都以 baseline-freeze SHA 为祖先。
11. **保证公平比较。** baseline rerun 使用 baseline-freeze commit；`ema` 与 `scale_aware_ema` 使用其直接后代，并保持 seed/RNG、数据、训练配置一致。报告同时记录 baseline-freeze SHA 与 EMA implementation SHA，使可观察差异只归因于明确列出的 EMA delta。
12. **失败即停。** 任一步发现备份不完整、branch/worktree 冲突、非预期 diff、环境不符或最小可复现性检查失败，都停止 Stage 02 并写明原因；不得以 clean/reset/stash 处理原 dirty tree，也不得继续实现 EMA。
