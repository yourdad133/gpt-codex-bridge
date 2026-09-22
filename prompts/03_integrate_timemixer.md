# Stage 03 — 将 EMA / Scale-Aware EMA 接入 TimeMixer，并证明 baseline 路径未改变

## 前置条件

必须完整读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`
- `results/02_implement_ema_module.md`

Stage 02 必须为 PASS。

TSLib 冻结点：

- audited upstream SHA: `4e938a1767106324dd753b2a44832bf870a0252e`
- baseline-freeze SHA: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- EMA module implementation SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`
- implementation branch/worktree: `codex/saema-v1`

如果这些 SHA、branch ancestry 或 worktree 状态与 Stage 02 报告不一致，停止并报告。

## 本阶段目标

把 Stage 02 已通过测试的 `EMADecomposition` 接入 TimeMixer 的 PDM decomposition。

最终支持：

1. `moving_avg` — 原始 TimeMixer 行为；
2. `dft_decomp` — 原始 TimeMixer 行为；
3. `ema` — 所有尺度共享 `alpha_base = ema_alpha`；
4. `scale_aware_ema` — 每个尺度按冻结公式计算 `alpha_k`。

本阶段只做：

- TimeMixer 集成；
- CLI；
- guard；
- logging；
- integration/regression tests；
- 模型级和极短数据级 smoke。

**不进行正式训练，不做 alpha 搜索。**

---

# 1. Git / worktree 边界

所有源码修改只允许发生在：

- worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- branch: `codex/saema-v1`

开始前必须：

- `git status --short` 为空；
- HEAD == `8f3a6b951360053ec1afd5f747f11beb7195d30d`；
- parent chain 包含 baseline-freeze SHA `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`。

不要修改：

- 原 dirty worktree；
- baseline-freeze worktree/branch；
- Stage 02 EMA 数学实现，除非 integration test 发现真实 bug。若发现 bug，可以修复，但必须单独记录原因并重新运行 Stage 02 的 65 tests。

---

# 2. CLI 冻结

严格使用 Stage 01 已冻结的参数名：

`--ema_alpha`

**不要使用或新增 `--ema_alpha_base`。**

CLI 契约：

- `--decomp_method` 默认仍为 `moving_avg`；
- 支持：
  - `moving_avg`
  - `dft_decomp`
  - `ema`
  - `scale_aware_ema`
- `--ema_alpha` default = `0.10`；
- legacy 方法不应读取/依赖 EMA alpha 的有效性；
- 不改变 `--moving_avg`、`--top_k`、down-sampling 参数现有默认值。

如果给 `decomp_method` 增加 argparse `choices` 可能影响仓库中其他模型/历史脚本，则不要机械增加 choices；优先保持兼容，只在 TimeMixer 内校验支持值。

---

# 3. TimeMixer 集成位置

根据 Stage 00/01 已审计的真实代码：

- 导入 `layers/EMADecomp.py`；
- 只定向修改 `PastDecomposableMixing` 的 decomposition selector 与 per-scale 调用；
- 使用 `enumerate(x_list)` 得到真实 `scale_index`；
- EMA 分支调用：
  `self.decompsition(x, scale_index)` 或等价已冻结 API；
- legacy `moving_avg` / `dft_decomp` 分支仍保持原来的：
  `self.decompsition(x)`。

禁止为了“统一代码”重写：

- `series_decomp`；
- `DFT_series_decomp`；
- `MultiScaleSeasonMixing`；
- `MultiScaleTrendMixing`；
- `future_multi_mixing`；
- normalization / embedding / prediction head；
- downsampling。

---

# 4. channel_independence guard

Frozen v1 正式边界：

`ema/scale_aware_ema + channel_independence=1`：允许。

`ema/scale_aware_ema + channel_independence=0`：必须在模型初始化早期显式 `ValueError`。

错误信息需要清楚说明：

- `channel_independence=0` 在 embedding 前还有独立 moving-average preprocess；
- 该组合不属于 SAEMA v1 实验定义；
- 不是运行时 shape bug。

必须验证：

- `moving_avg + channel_independence=0` 仍可实例化；
- `dft_decomp + channel_independence=0` 保持 legacy 行为；
- guard 只针对两个 EMA mode。

---

# 5. Logging

EMA 模式启动时打印一次结构化摘要，至少包含：

- decomp_method；
- ema_alpha；
- down_sampling_window；
- down_sampling_layers；
- channel_independence；
- 每个尺度：
  - scale_index；
  - `r_k = w^k`；
  - actual alpha_k。

Baseline 示例配置：

- `w=2`
- `down_sampling_layers=3`
- `ema_alpha=0.10`

Scale-Aware 应打印约：

- k0: 0.100000
- k1: 0.190000
- k2: 0.343900
- k3: 0.569533

不得：

- 每个 batch 打印；
- 每个 PDM block 重复打印相同摘要；
- 在 moving_avg / DFT 模式制造 EMA 日志噪声。

如果最自然的实现需要在 `Model.__init__` 统一打印一次，可这样做。

---

# 6. 参数 / state_dict / checkpoint 兼容性

必须证明：

### 6.1 Parameter count

对相同 TimeMixer 配置：

- baseline-freeze moving_avg；
- integrated moving_avg；
- fixed EMA；
- scale-aware EMA；

trainable parameter count 必须一致。

Stage 02 EMA module 自身仍必须：

- `parameters() == []`；
- 不新增 persistent state-dict key。

### 6.2 State-dict keys

对相同 TimeMixer config：

- baseline-freeze moving_avg 的 `state_dict().keys()`
- integrated moving_avg 的 `state_dict().keys()`

必须完全一致。

EMA 两模式相对 integrated moving_avg 也不应因 EMA coefficient 增加新的 key。

如果模型本身因 decomposition 分支是否含参数导致已有 legacy 差异，必须明确报告；不要掩盖。

### 6.3 Old checkpoint strict-load

选择 Stage 00 已确认存在的 ETTh1 baseline checkpoint，例如 pred_len=96 / seed=2021。

在 integrated code 上以完全匹配的 moving_avg baseline config：

`load_state_dict(..., strict=True)`

必须成功。

不要覆盖原 checkpoint。

---

# 7. 最关键：跨 worktree moving_avg 数值回归

本阶段必须证明：

> 集成 EMA 后，只要 decomp_method 仍为 moving_avg，TimeMixer 数值行为没有改变。

请使用：

- baseline-freeze worktree @ `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- integrated worktree @ 本阶段代码

设计 deterministic regression。

要求：

1. 固定完全相同 random seed；
2. 完全相同模型 config；
3. CPU eval 模式优先，避免 GPU nondeterminism；
4. 完全相同 synthetic input；
5. 确保两边模型初始权重完全相同：
   - 可使用相同 seed 初始化后比较 state dict；
   - 或从 baseline 生成 state dict，再 strict-load 到 integrated moving_avg；
6. 比较最终 output；
7. 最好同时比较至少一个 PDM decomposition 相关中间输出（如果不需要侵入式改代码即可 capture）。

验收：

- state-dict keys 相同；
- 参数 tensor 相同；
- final output 应 exact equal 或达到非常严格的 allclose；
- 如非 bitwise exact，报告 max_abs_diff、max_rel_diff 与原因。

**如果 moving_avg regression 不通过，不得进入训练 smoke；必须定位并修复。**

---

# 8. Integration tests

在 Stage 02 test 基础上新增针对 TimeMixer 的集成测试，文件位置自行按项目结构合理选择。

至少覆盖：

1. moving_avg model instantiation；
2. dft_decomp model instantiation；
3. ema model instantiation；
4. scale_aware_ema model instantiation；
5. EMA + CI=0 明确报错；
6. moving_avg + CI=0 不被新 guard 拦截；
7. 同一合法输入下四种 mode forward shape 一致；
8. fixed EMA 在不同 scale 使用同 alpha；
9. scale-aware EMA 使用预期 alpha table；
10. parameter count 不增加；
11. state-dict key regression；
12. old baseline checkpoint strict-load；
13. moving_avg 跨 worktree numerical regression；
14. Stage 02 原 65 tests 仍通过；
15. CPU forward finite；
16. CUDA forward finite（可用时）。

注意：不要“修复” Stage 00 已指出的 DFT 既有问题。这里只确认新集成没有破坏 legacy DFT 路径。

---

# 9. 数据级 smoke

完成模型级 regression 后，做**极短** smoke，验证真实 ETTh1 DataLoader → TimeMixer → loss/backward 能走通。

固定：

- ETTh1；
- long_term_forecast；
- seq_len=96；
- pred_len=96；
- channel_independence=1；
- baseline 架构参数；
- seed=2021。

至少对：

- moving_avg；
- ema with `--ema_alpha 0.10`；
- scale_aware_ema with `--ema_alpha 0.10`

各跑极少量 batch（例如 1–3 个 train batches + 1 validation batch），而不是完整 epoch。

要求检查：

- forward finite；
- loss finite；
- backward finite；
- optimizer step 可执行；
- shape 正确。

**不要产生正式 benchmark 指标，不要把 smoke MSE 当研究结果。**

如果现有 runner 不容易限制到几个 batch，可以写独立 smoke test/harness，而不是修改正式训练流程。

---

# 10. 实验脚本/命令模板

本阶段可以新增一个**不会执行正式训练**的 SAEMA 命令模板或后续实验脚本，例如：

`scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_SAEMA.sh`

但必须满足：

- 不覆盖官方 `TimeMixer_ETTh1.sh`；
- 显式包含：
  - decomp_method；
  - `--ema_alpha`；
  - seed；
  - pred_len；
- experiment identity 能区分 method / alpha / seed / horizon；
- 不与旧 baseline checkpoint/log 路径冲突。

若新增脚本，请只做语法/命令检查，本阶段不要执行完整 10 epoch。

---

# 11. Commit 边界

Stage 03 implementation commit 应主要只包含：

- `models/TimeMixer.py`
- `run.py`
- integration test(s)
- 如有：新的 SAEMA 实验脚本/command template

不要混入其他模型/数据集/旧实验修改。

建议 commit message：

`feat: integrate scale-aware EMA into TimeMixer`

若先后修复 integration bug，可有多个清晰 commit，但报告必须给最终 SHA 与 commit list。

---

# 12. 输出

写入 bridge：

`results/03_integrate_timemixer.md`

必须包含：

## Git
- Stage 03 start SHA；
- final SHA；
- commits；
- changed files；
- `git diff --stat 8f3a6b9..HEAD`；
- final `git status --short`。

## Integration
- CLI；
- decomposition selector；
- per-scale call；
- CI guard；
- logging example。

## Regression
- parameter counts；
- state-dict key comparison；
- old checkpoint strict-load；
- baseline-freeze vs integrated moving_avg numerical regression；
- exact/max diff。

## Tests
- Stage 02 65 tests；
- new integration tests；
- CPU/CUDA；
- all pass/fail details。

## Data smoke
- actual commands/harness；
- moving_avg / ema / scale-aware EMA；
- finite forward/loss/backward/optimizer；
- 不把 smoke 指标作为 benchmark。

## Final
- `Status: PASS | FAIL | BLOCKED`
- legacy moving_avg behavior unchanged: YES/NO
- old checkpoint compatible: YES/NO
- EMA adds trainable params: YES/NO
- ready for Stage 04: YES/NO

任何关键 regression 失败都必须停止，不进入 Stage 04。

完成并 push bridge 结果后停止。
