# Stage 02 — 冻结可追溯 baseline + 实现独立 EMA decomposition 与单元测试

**Status: PASS**

- Bridge execution base: `0bb17879af67662ab489ea805a949d8de4e63208`
- TSLib audited upstream SHA: `4e938a1767106324dd753b2a44832bf870a0252e`
- Baseline-freeze SHA: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- EMA implementation SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`
- Formal ETTh1 training/evaluation: **not run**
- Stage 03: **not run**

本阶段完整读取并遵守了 `README.md`、`CURRENT_STAGE.md`、`results/00_baseline_audit.md`、`results/01_method_spec.md` 和 `prompts/02_implement_ema_module.md`。执行严格分为串行的 baseline freeze 与独立 EMA 模块实现两部分。

## 1. Baseline Freeze

### 1.1 原 dirty worktree 只读核对

- 原 worktree: `/home/ouyangguanghong/projects/Time-Series-Library`
- Windows 映射: `X:\Time-Series-Library`
- branch: `main`
- HEAD: `4e938a1767106324dd753b2a44832bf870a0252e`
- remote: `origin https://github.com/thuml/Time-Series-Library.git`
- `core.filemode`: `true`

开始 Stage 02 时的 `git status --short`：

```text
AM .codex/config.toml
AM AGENTS.md
 D README.md
 M data_provider/data_factory.py
 M data_provider/data_loader.py
 M exp/exp_long_term_forecasting.py
 M run.py
 M utils/m4_summary.py
?? TimeMixerBaseline_CrossBenchmark_Report_20260914.md
?? TimeMixer_ETTh1_Paper_Reproduction_20260922.md
?? scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh
?? scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_paper.sh
?? scripts/long_term_forecast/ETT_script/analyze_router_dynamics.py
?? scripts/long_term_forecast/ETT_script/validate_TimeMixerAdaptivePatch.py
?? scripts/long_term_forecast/LocalDynamicPatch_CrossBenchmark.sh
?? scripts/long_term_forecast/TimeMixer_CrossBenchmark_Baseline.sh
```

差异摘要：

- unstaged: 8 files, 318 insertions, 389 deletions；
- staged: `.codex/config.toml`、`AGENTS.md`，共 25 insertions；
- untracked manifest: 上述 8 个报告/脚本文件；
- 初始 `git worktree list --porcelain` 只有原 worktree。

与 Stage 00 记录相比，没有新增无法解释的实质修改。Stage 00 中若干 `models/TimeMixer.py`、`layers/StandardNorm.py` 和 shell script 的 mode-only 项已不再出现；这些项当时已确认没有内容差异，本次状态是其消失而不是增加新的内容 hunk。其余实质 dirty 路径和内容与 Stage 00 的审计说明一致，因此没有触发 BLOCKED 条件。

在整个 Stage 02 前后，未在原 worktree 执行 `clean`、`reset`、`stash`、`checkout/switch`、`commit`、删除或覆盖操作。结束前把以下三项与恢复备份逐字节比较，全部一致：

- `git status --porcelain=v1 -z`；
- `git diff --binary HEAD`；
- `git diff --cached --binary`。

结论：原 dirty worktree 内容与 index 均保持未修改。

### 1.2 私有恢复备份

- 私有目录：`/home/ouyangguanghong/.codex-stage02-recovery/Time-Series-Library-20260922T135945`
- 位置在 TSLib 与 bridge 仓库之外；未提交到任何仓库；
- 保存了完整 tracked working-tree binary patch、unstaged patch、staged/index patch、NUL 分隔 status、NUL 分隔 untracked manifest、逐文件 SHA-256、8 个 untracked 文件的压缩副本、HEAD/branch/remote/worktree metadata；
- 共 12 个恢复材料文件，untracked 文件计数为 8；
- `sha256sum -c`、`gzip -t`、tracked patch reverse check、staged patch cached reverse check 全部通过。

报告不记录本地配置内容或恢复材料校验值。

### 1.3 Baseline worktree 与允许的差异

- worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1-baseline`
- branch: `codex/saema-v1-baseline`
- initial HEAD: `4e938a1767106324dd753b2a44832bf870a0252e`
- initial status: clean
- freeze commit: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- commit message: `baseline-freeze: preserve deterministic seed and dataloader rng`
- final baseline status: clean

相对 audited upstream 的完整变更边界：

```text
 data_provider/data_factory.py | 34 ++++++++++++++++++++++++++++------
 run.py                        | 19 +++++++++++++------
 2 files changed, 41 insertions(+), 12 deletions(-)
```

精确的非-upstream 改动：

1. `run.py`
   - 将入口处固定 `fix_seed` 改为 `set_random_seed(args.seed)`；
   - 设置 Python `random`、NumPy、PyTorch CPU、CUDA 单卡与全部 CUDA device seed；
   - `--seed` 默认值由 `2` 改为已复现 baseline 使用的 `2021`；
   - 在参数解析后、模型与 DataLoader 构造前调用 seed 初始化。
2. `data_provider/data_factory.py`
   - 新增 `_seed_worker`，从 `torch.initial_seed()` 派生 32-bit Python/NumPy worker seed；
   - 新增 `_loader_kwargs`，用 `args.seed` 初始化独立 `torch.Generator`；
   - anomaly、classification 与常规 DataLoader 路径均使用同一 generator/worker init/num_workers 配置。

程序化边界核对结果：

- `data_provider/data_factory.py` 与原 dirty tree 中已复现运行使用的版本逐字节一致；
- `run.py` 中 `patch_sizes`、routing、temperature、AdaptivePatch alpha/balance 等无关参数全部不存在；
- `git diff --check` 通过；
- 相对 upstream 的实质 changed files 恰好只有上述两个文件。

### 1.4 明确排除的无关内容

Baseline freeze 没有带入：

- `.codex/config.toml`、`AGENTS.md`、`README.md` 删除；
- `exp/exp_long_term_forecasting.py` router auxiliary-loss 路径；
- `data_provider/data_loader.py`、`utils/m4_summary.py` 的 M4 相关修改；
- AdaptivePatch、router analysis/validation、旧 baseline 脚本和报告；
- Stage 00 记录的所有 mode-only 变化；
- 任何旧日志、checkpoint、test_results、results 或未跟踪实验文件。

未 push TSLib 分支；本阶段只在 bridge 中记录 SHA 与证据。

### 1.5 环境与 deterministic check

Baseline worktree 中重新执行环境检查：

```text
HOST=ps
PROJECT=/home/ouyangguanghong/projects/Time-Series-Library-saema-v1-baseline
PYTHON=/home/ouyangguanghong/miniconda3/envs/tslib/bin/python
GPU 0/1=NVIDIA GeForce RTX 4090 D, 24564 MiB each
driver=570.211.01
PyTorch=2.5.1+cu121
CUDA_AVAILABLE=True
CUDA_COUNT=2
```

核心检查命令：

```bash
source /home/ouyangguanghong/miniconda3/etc/profile.d/conda.sh
conda activate tslib
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

轻量 deterministic 检查使用 `TensorDataset(torch.arange(40))`、`batch_size=5`、`num_workers=2`，分别用 seed 2022、2022、2023 新建 DataLoader；同时重复调用 `_seed_worker` 检查 Python/NumPy RNG。实际结果：

```text
DETERMINISTIC_DATALOADER_ORDER=PASS
DIFFERENT_SEED_ORDER=PASS
WORKER_PYTHON_NUMPY_RNG=PASS
FIRST_ORDER_HEAD=[13, 16, 17, 23, 36, 35, 32, 10, 6, 29]
```

没有创建或覆盖任何 baseline artifact，没有运行正式训练。

### 1.6 Baseline freeze 过程中的失败尝试

第一次 `git commit` 因仓库未配置作者姓名/邮箱而失败，没有产生 commit。为避免修改共享 Git config，随后仅对单次 commit 使用 `git -c user.name=... -c user.email=... commit`，成功创建 freeze commit。原 worktree 未受影响。

## 2. EMA Implementation

### 2.1 Worktree 与 commit 边界

- worktree: `/home/ouyangguanghong/projects/Time-Series-Library-saema-v1`
- branch: `codex/saema-v1`
- parent baseline-freeze SHA: `462875291aa7eb5963e9ba81dc5ab86c6b8f5230`
- implementation SHA: `8f3a6b951360053ec1afd5f747f11beb7195d30d`
- commit message: `feat: add scale-aware EMA decomposition module`
- final status: clean

Implementation commit 相对 baseline-freeze 的完整 diff：

```text
A       layers/EMADecomp.py
A       tests/test_ema_decomp.py

 layers/EMADecomp.py      | 123 +++++++++++++++++++
 tests/test_ema_decomp.py | 300 +++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 423 insertions(+)
```

`run.py` 的 seed 变化只来自 parent baseline-freeze，不属于 implementation delta。本提交没有修改 `models/TimeMixer.py`、`run.py`、PDM/FMM/mixing、moving-average/DFT、正式 shell script 或其他文件。

### 2.2 核心 API 与数学行为

新增接口：

```python
class EMADecomposition(nn.Module):
    def __init__(self, alpha_base, down_sampling_window, mode): ...
    def alpha_for_scale(self, scale_index): ...
    def forward(self, x, scale_index): ...  # -> (seasonal, trend)
```

- 输入/输出：严格 `[batch_like, time, feature]`，返回 `(seasonal, trend)`；
- 初始化：`trend[:, 0, :] = x[:, 0, :]`；
- 递推：`trend_t = alpha_k * x_t + (1 - alpha_k) * trend_{t-1}`；
- residual：`seasonal = x - trend`；
- `mode='ema'`：所有尺度 `alpha_k = alpha_base`；
- `mode='scale_aware_ema'`：`r_k = w ** scale_index`，并用 float64 Python scalar 稳定计算 `alpha_k = -expm1(r_k * log1p(-alpha_base))`；极端倍率导致 float64 retention 下溢时数值饱和到 `1.0`；
- `scale_index` 由 API 显式传入，不由 tensor 长度反推。

输入校验覆盖 finite 且开区间内的 alpha、整数且 `>=1` 的 window、两个合法 mode、非负整数 scale index、3D floating tensor 和非空时间维。

FP16/BF16 输入递推时显式提升到 FP32 accumulator，最后恢复输入 dtype；所有 tensor 运算由输入派生，输出 device 与输入一致。实现没有 `detach()`、没有在正常 forward 中使用 `no_grad()`、没有破坏 autograd 的 inplace 更新，也不保存跨 forward 状态。

参数/状态检查：

```text
list(module.parameters()) == []
module.state_dict() == {}
```

因此 alpha 不是 `nn.Parameter`，也没有新增 persistent state-dict key。

## 3. Tests

### 3.1 实际命令

远端 `tslib` 初始没有 pytest；按远程执行规则安装测试运行器，没有修改 requirements 或项目配置：

```bash
python -m pip install pytest
# installed: pytest 9.1.1, pluggy 1.6.0, iniconfig 2.3.0
```

完整最终测试命令：

```bash
cd /home/ouyangguanghong/projects/Time-Series-Library-saema-v1
source /home/ouyangguanghong/miniconda3/etc/profile.d/conda.sh
conda activate tslib
export CUDA_VISIBLE_DEVICES=0,1
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_ema_decomp.py
```

最终结果：

```text
.................................................................        [100%]
65 passed in 1.60s
```

### 3.2 覆盖结果

- 数学：shape 保持、`seasonal + trend ~= x`、常数序列、手算 fixed EMA、Stage 01 数值表、单调性、`w=1`、`k=0`、down-sampling-layers=0 语义；
- 边界：alpha 0/1/越界/NaN/Inf/非法类型、非法 mode/window/scale index、空序列、长度 1、非浮点、非 3D、非 tensor；
- CPU：float32/float64，两个 mode，forward/backward 全 finite；
- CUDA：float32/float16/bfloat16，两个 mode，dtype/device 保持，forward/backward 全 finite；
- AMP：CUDA autocast float16/bfloat16，输出与 input gradient 全 finite；
- 回归性质：重复 forward deterministic、fixed EMA 忽略 scale index、scale-aware 不同 scale 使用不同 alpha、不修改输入 tensor；
- 状态：无 parameter、空 state dict；
- 非有限输入自然传播，没有静默 `nan_to_num`。

### 3.3 失败测试与修复历史

第一次 pytest collection 在执行任何 test 前报 1 个 `SyntaxError`。原因是传输到远端的初版新增文件 patch 中 hunk 行数元数据写短，导致两个新文件尾部被截断。补齐 `EMADecomp.forward` 尾部和 CUDA autocast test 后，完整 suite 首次执行即 65/65 通过；实现 commit 后再次执行仍为 65/65 通过。没有隐藏或删除失败断言。

## 4. Performance Smoke

仅做 forward 粗测，不作为论文 efficiency 结论。外部 timing harness 使用 `torch.no_grad()`；EMA 模块自身的正常 forward 不包含 `no_grad()`。

- module: `EMADecomposition(0.1, 2, 'scale_aware_ema')`
- `scale_index=3`
- tensor shape: `(896, 96, 16)`，其中 `896 = 128 * 7`
- dtype: float32
- CPU: warmup 5，repeats 20
- CUDA: warmup 10，repeats 30，每次计时同步 CUDA

结果：

```text
CPU  mean_ms=5.217 median_ms=4.497 min_ms=3.540 max_ms=9.166
CUDA mean_ms=2.187 median_ms=1.251 min_ms=1.179 max_ms=5.979
```

输出 shape 正确且 finite；未发现数量级异常。该结果只用于 Stage 02 smoke，不代表端到端模型速度。

## 5. Git verification

最终 worktree 图：

```text
/home/ouyangguanghong/projects/Time-Series-Library
  main @ 4e938a1767106324dd753b2a44832bf870a0252e (original dirty, unchanged)
/home/ouyangguanghong/projects/Time-Series-Library-saema-v1-baseline
  codex/saema-v1-baseline @ 462875291aa7eb5963e9ba81dc5ab86c6b8f5230 (clean)
/home/ouyangguanghong/projects/Time-Series-Library-saema-v1
  codex/saema-v1 @ 8f3a6b951360053ec1afd5f747f11beb7195d30d (clean)
```

验证结果：

- `git diff --check` for baseline snapshot: PASS；
- `git diff --check baseline-freeze..implementation`: PASS；
- implementation `git status --short`: empty；
- implementation parent exactly equals baseline-freeze SHA；
- implementation diff files exactly equal `layers/EMADecomp.py` and `tests/test_ema_decomp.py`；
- original dirty worktree status/tracked diff/index diff 与 pre-freeze backup 完全一致。

## 6. Final

**Status: PASS**

Stage 02 满足 Frozen v1 specification 在本阶段要求的独立模块边界：fixed EMA、Scale-Aware EMA、稳定 alpha helper、输入校验、dtype/device/autograd 行为、无参数/无 persistent state 和完整单测均已实现并验证。

本阶段没有接入 TimeMixer PDM/forward，没有修改 `models/TimeMixer.py`，没有新增 EMA CLI，没有修改正式训练脚本，没有运行 ETTh1 正式训练或 alpha search，也没有执行 Stage 03。

从代码与验证角度可以交由 Stage 03 继续集成，但必须先由 bridge 的 `CURRENT_STAGE.md` 明确推进并提供 Stage 03 prompt；本次执行在 Stage 02 完成后停止。
