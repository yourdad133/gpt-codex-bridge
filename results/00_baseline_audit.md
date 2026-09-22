Status: PASS
TSLib commit: `4e938a1767106324dd753b2a44832bf870a0252e`
TSLib branch: `main`
Baseline reproducible: PARTIAL

# Stage 00 — TimeMixer baseline audit

## 1. Audit verdict

Stage 00 已完成。本阶段只做了只读审计；没有修改 Time-Series-Library 源代码，也没有重跑 12 次训练，更没有执行 Stage 01。

既有 ETTh1 baseline 的 12 份成功日志、12 个最佳 checkpoint、12 个 `test_results` 目录和最终 MSE/MAE 均已核实，数值与既有报告一致。但 baseline 标记为 `PARTIAL`，原因是：

1. TSLib 当前工作树存在大量未提交内容，精确运行态不能只由上述 commit SHA 还原；
2. 三 seed 运行依赖未提交的 seed/DataLoader 改动；
3. `results/` 当前为空，已生成的 `metrics.npy`、`pred.npy`、`true.npy` 不再存在，现存证据是日志、checkpoint 和 `test_results`；
4. `models/TimeMixer.py` 的内容与 commit 一致，只有文件模式从 `100755` 变为 `100644`，但训练入口与数据加载代码并非 commit-clean。

因此，已有结果可信且可从 checkpoint 重新测试，但在开始研究改动前仍应先冻结一个可追溯、干净的 baseline 代码状态。

## 2. TSLib workspace and environment

### 2.1 Workspace identity

- 远程绝对路径：`/home/ouyangguanghong/projects/Time-Series-Library`
- Windows 映射路径：`X:\Time-Series-Library`
- Git remote：`origin https://github.com/thuml/Time-Series-Library.git`（fetch/push）
- 当前 branch：`main`
- 当前 commit：`4e938a1767106324dd753b2a44832bf870a0252e`

### 2.2 Current `git status --short`

```text
AM .codex/config.toml
AM AGENTS.md
 D README.md
 M data_provider/data_factory.py
 M data_provider/data_loader.py
 M exp/exp_long_term_forecasting.py
 M layers/StandardNorm.py
 M models/TimeMixer.py
 M run.py
 M scripts/long_term_forecast/ECL_script/TSMixer.sh
 M scripts/long_term_forecast/ECL_script/TimeMixer.sh
 M scripts/long_term_forecast/ETT_script/TSMixer_ETTh1.sh
 M scripts/long_term_forecast/ETT_script/TSMixer_ETTh2.sh
 M scripts/long_term_forecast/ETT_script/TSMixer_ETTm1.sh
 M scripts/long_term_forecast/ETT_script/TSMixer_ETTm2.sh
 M scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
 M scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2.sh
 M scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
 M scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2.sh
 M scripts/long_term_forecast/Traffic_script/TSMixer.sh
 M scripts/long_term_forecast/Traffic_script/TimeMixer.sh
 M scripts/long_term_forecast/Weather_script/TSMixer.sh
 M scripts/long_term_forecast/Weather_script/TimeMixer.sh
 M scripts/short_term_forecast/TSMixer_M4.sh
 M scripts/short_term_forecast/TimeMixer_M4.sh
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

未清理或 reset 上述内容。差异核对显示：

- `models/TimeMixer.py`、`layers/StandardNorm.py` 及所列 TimeMixer/TSMixer shell 脚本是 mode-only 变化；
- `run.py` 增加了显式 seed 初始化，并把 seed 默认值改为 2021；还包含与另一模型相关的未提交 CLI 参数；
- `data_provider/data_factory.py` 增加了按 run seed 固定 DataLoader generator/worker RNG 的逻辑；
- `exp/exp_long_term_forecasting.py` 增加了 router auxiliary-loss 路径。普通 TimeMixer 不提供该接口，实际仍走原预测张量路径，辅助损失为 0；
- `data_provider/data_loader.py` 与 `utils/m4_summary.py` 的实质改动只涉及 M4；
- `README.md` 当前被删除。

### 2.3 Runtime environment

- 环境名称：`tslib`
- Python：`3.11.16`
- Python executable：`/home/ouyangguanghong/miniconda3/envs/tslib/bin/python`
- PyTorch：`2.5.1+cu121`
- PyTorch CUDA runtime：`12.1`
- `torch.cuda.is_available()`：`True`
- 可见 GPU 数：`2`
- GPU：2 × NVIDIA GeForce RTX 4090 D，单卡 24564 MiB
- NVIDIA driver：`570.211.01`
- 审计时 GPU 0/1 均约 1 MiB 占用，利用率 0%

实际环境检查在远程项目目录中执行，核心命令为：

```bash
source /home/ouyangguanghong/miniconda3/etc/profile.d/conda.sh
conda activate tslib
export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
python -c 'import os, sys, torch; print(os.getenv("CONDA_DEFAULT_ENV")); print(sys.version); print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

## 3. Existing ETTh1 baseline

### 3.1 Configuration confirmed from logs

| Item | Confirmed value |
|---|---|
| Task/data | `long_term_forecast`, `ETTh1`, `features=M`, `enc_in=7`, `c_out=7` |
| Input | `seq_len=96`, `label_len=0` |
| Horizons | `pred_len={96,192,336,720}` |
| Seeds | `2021, 2022, 2023` |
| Model width/depth | `d_model=16`, `d_ff=32`, `e_layers=2` |
| Decomposition | `decomp_method=moving_avg`, `moving_avg=25` |
| Multiscale | `down_sampling_layers=3`, `down_sampling_window=2`, `down_sampling_method=avg` |
| Normalization | `use_norm=1` |
| Optimization | Adam, initial LR `0.01`, `lradj=type1`, MSE loss |
| Training | `train_epochs=10`, `patience=10`, `batch_size=128`, `num_workers=10`, `itr=1` |
| AMP | disabled |
| Channel mode | default `channel_independence=1` |

`scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh` 覆盖四个 horizon，并明确设置三层、窗口 2、avg pooling、`d_model=16`、`d_ff=32`、`e_layers=2`、LR 0.01、10 epochs 和 batch size 128。该脚本顶部存在未使用的 `batch_size=16` 变量，四个调用实际都硬编码 `--batch_size 128`；它未显式写 decomposition 参数，而是依赖 `run.py` 的 `moving_avg/25/use_norm=1` 默认值。既有复现实验命令则显式传入了这些参数。

### 3.2 Actual successful command template

以下模板是既有 12 次成功运行的实际命令形式；`${pred_len}` 展开为 `96 192 336 720`，`${seed}` 展开为 `2021 2022 2023`。物理 GPU 1 被映射为进程内 `cuda:0`。

```bash
export CUDA_VISIBLE_DEVICES=1
/home/ouyangguanghong/miniconda3/envs/tslib/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id "ETTh1_96_${pred_len}_paper_seed${seed}_gpu1fixed" \
  --model TimeMixer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len "${pred_len}" \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --learning_rate 0.01 \
  --train_epochs 10 \
  --patience 10 \
  --batch_size 128 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --decomp_method moving_avg \
  --moving_avg 25 \
  --use_norm 1 \
  --loss MSE \
  --seed "${seed}" \
  --gpu 0 \
  --num_workers 10 \
  --des PaperRep_20260922 \
  --itr 1 \
  2>&1 | tee "experiment_logs/TimeMixer_ETTh1_${pred_len}_seed${seed}_gpu1fixed_20260922.log"
```

### 3.3 Metrics

| Horizon | Seed 2021 MSE / MAE | Seed 2022 MSE / MAE | Seed 2023 MSE / MAE | Three-seed mean MSE / MAE |
|---:|---:|---:|---:|---:|
| 96 | 0.376218 / 0.399333 | 0.373363 / 0.393359 | 0.382440 / 0.398855 | **0.377340 / 0.397182** |
| 192 | 0.447379 / 0.432038 | 0.432129 / 0.431674 | 0.428995 / 0.432161 | **0.436168 / 0.431958** |
| 336 | 0.497219 / 0.461007 | 0.470660 / 0.444710 | 0.474299 / 0.449385 | **0.480726 / 0.451701** |
| 720 | 0.493089 / 0.475488 | 0.519979 / 0.489870 | 0.505935 / 0.479702 | **0.506334 / 0.481687** |
| Four-horizon mean | — | — | — | **0.450142 / 0.440632** |

代表性日志中的数据量与最终 shape：

| Horizon | Train / Val / Test windows | Final prediction/target shape |
|---:|---:|---:|
| 96 | 8449 / 2785 / 2785 | `(2785, 96, 7)` |
| 192 | 8353 / 2689 / 2689 | `(2689, 192, 7)` |
| 336 | 8209 / 2545 / 2545 | `(2545, 336, 7)` |
| 720 | 7825 / 2161 / 2161 | `(2161, 720, 7)` |

### 3.4 Artifact locations

- 汇总报告：`/home/ouyangguanghong/projects/Time-Series-Library/TimeMixer_ETTh1_Paper_Reproduction_20260922.md`
- 成功日志：`/home/ouyangguanghong/projects/Time-Series-Library/experiment_logs/TimeMixer_ETTh1_{96,192,336,720}_seed{2021,2022,2023}_gpu1fixed_20260922.log`
- 成功 checkpoint：`/home/ouyangguanghong/projects/Time-Series-Library/checkpoints/long_term_forecast_ETTh1_96_{pred}_paper_seed{seed}_gpu1fixed_TimeMixer_ETTh1_ftM_sl96_ll0_pl{pred}_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_PaperRep_20260922_0/checkpoint.pth`
- `test_results/`：12 个对应设置目录仍存在。
- `results/`：审计时为空；`metrics.npy`、`pred.npy`、`true.npy` 不再存在。

逐一检查确认 12 个 `checkpoint.pth` 全部存在；文件大小按 horizon 分别为 655906、726562、832546、1115170 bytes。

### 3.5 Failed attempt retained for audit

首次启动检查使用 `CUDA_VISIBLE_DEVICES=0,1` 和 `--gpu 1`。训练完成后加载最佳 checkpoint 时失败：

```text
RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1.
```

失败日志：`experiment_logs/TimeMixer_ETTh1_96_seed2021_20260922.log`。该次结果未进入统计。随后改为 `CUDA_VISIBLE_DEVICES=1` 与 `--gpu 0`，即把物理 GPU 1 映射为进程内 `cuda:0`，12 次运行全部完成测试。

## 4. TimeMixer decomposition call chain

### 4.1 Key source locations

| File | Symbol / role | Current lines |
|---|---|---:|
| `run.py` | CLI parser: `moving_avg`, `decomp_method`, multiscale args | 72, 81–89 |
| `run.py` | Select `Exp_Long_Term_Forecast`; train/test dispatch | 209–211, 231–267 |
| `exp/exp_basic.py` | Dynamic model discovery and `Model` class loading | 10–24, 79–111 |
| `exp/exp_long_term_forecasting.py` | `Exp_Long_Term_Forecast`, model/optimizer/loss/data entry | 19–40 |
| `exp/exp_long_term_forecasting.py` | Validation, training, checkpoint load, test metrics | 173–205, 207–338, 338–438 |
| `layers/Autoformer_EncDec.py` | `moving_avg`, `series_decomp` | 21–53 |
| `models/TimeMixer.py` | `DFT_series_decomp` | 9–26 |
| `models/TimeMixer.py` | `MultiScaleSeasonMixing` / `MultiScaleTrendMixing` | 29–115 |
| `models/TimeMixer.py` | `PastDecomposableMixing` | 118–184 |
| `models/TimeMixer.py` | `Model`, multiscale inputs, forecast, future mixing | 187–396 |
| `scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh` | Four ETTh1 horizon launches | 1–125 |

Call chain for long-term forecasting:

```text
run.py
  -> Exp_Long_Term_Forecast(args)
  -> dynamic import models.TimeMixer.Model
  -> Exp_Long_Term_Forecast.train/test
  -> TimeMixer.Model.forward
  -> TimeMixer.Model.forecast
  -> __multi_scale_process_inputs
  -> normalize + optional channel-independent reshape
  -> embedding
  -> pdm_blocks[i](enc_out_list)
  -> PastDecomposableMixing.decompsition(x) for every scale
  -> seasonal bottom-up mixing + trend top-down mixing
  -> future_multi_mixing
  -> sum scale predictions + denormalize
```

### 4.2 Answers to the audit questions

1. **Supported `decomp_method` values.** The current PDM constructor implements exactly `moving_avg` and `dft_decomp`. `run.py` documents those two values but does not enforce `choices`; any other string reaches `PastDecomposableMixing.__init__` and raises `ValueError('decompsition is error')`.

2. **Origin of moving-average kernel/window.** `run.py:72` defines `--moving_avg` with default 25. It is passed as `configs.moving_avg` to `series_decomp` in `PastDecomposableMixing` (`models/TimeMixer.py:129–130`) and to `Model.preprocess` (`models/TimeMixer.py:201`). `series_decomp` constructs `moving_avg(kernel_size, stride=1)`; the implementation repeats the first/last value at both ends, applies `AvgPool1d`, then returns `season=x-moving_mean` and `trend=moving_mean` (`layers/Autoformer_EncDec.py:21–53`). The successful baseline explicitly used 25.

3. **Where decomposition is called in PDM.** Each encoder PDM block is created in `Model.__init__` (`models/TimeMixer.py:198–199`). `forecast()` calls the two PDM blocks at lines 367–369. Within `PastDecomposableMixing.forward`, lines 164–170 iterate over every scale and invoke `self.decompsition(x)` at line 165.

4. **Number of scales.** The model creates the original resolution plus one result for every down-sampling layer: `down_sampling_layers + 1`. The baseline therefore has four scales.

5. **Scale lengths.** For avg/max pooling with kernel=stride=`w`, `T_i=floor(seq_len / w^i)`. With `seq_len=96`, `w=2`, and three layers, lengths are `96, 48, 24, 12`. The time-mixing Linear layers are built from the same integer-division formula. Conv down-sampling uses kernel 3, stride `w`, circular padding 1 and can produce `ceil(T/w)` for non-divisible lengths, while the Linear dimensions still use floor division; non-divisible configurations are therefore a shape-risk. Repeated down-sampling must also keep every scale non-empty.

6. **Whether scales share decomposition.** Within one `PastDecomposableMixing` block, a single `self.decompsition` instance is reused for all scales. With `e_layers=2`, there are two independent PDM instances, one per encoder layer. The current moving-average module has no learned parameters, so sharing is operational rather than parameter sharing. A scale-aware EMA cannot obtain distinct coefficients from the present single-instance call unless scale index/alpha is passed explicitly or per-scale modules are constructed.

7. **Seasonal and trend paths.** Decomposition outputs `[B*,T_i,D]`. If `channel_independence=0`, both tensors first pass through the same `cross_layer`. They are permuted to `[B*,D,T_i]`. Seasonal components then use `MultiScaleSeasonMixing`, a high-to-low bottom-up path with temporal Linear/GELU/Linear transforms and residual addition into the next lower resolution. Trend components use `MultiScaleTrendMixing`, a low-to-high top-down path with temporal Linear/GELU/Linear transforms and residual addition into the next higher resolution. The two paths are returned to `[B*,T_i,D]`, summed, and, for `channel_independence=1`, added to `ori` through `out_cross_layer`. Future predictors project every scale to `pred_len`; scale predictions are stacked and summed.

8. **Tensor shape convention at decomposition.** `series_decomp` and `DFT_series_decomp` expect time on dimension 1: `[batch_like, time, feature]`. Raw ETTh1 input is `[B,96,7]`. After multiscale pooling it is `[B,T_i,7]`. With the baseline `channel_independence=1`, normalization is followed by reshape to `[B*7,T_i,1]`; embedding then produces `[B*7,T_i,16]`, which is the actual PDM decomposition input for `T_i in {96,48,24,12}`. With `channel_independence=0`, embedding produces `[B,T_i,16]`. The season/trend mixers temporarily permute these to `[batch_like,16,T_i]`.

9. **Minimum-intrusion EMA insertion point.** The narrowest PDM change is the decomposition selector and per-scale call in `PastDecomposableMixing` (`models/TimeMixer.py:129–165`), plus a TimeMixer-local EMA decomposition class next to `DFT_series_decomp` and explicit CLI configuration next to `run.py:72–89`. The existing moving-average and DFT branches should remain unchanged. `Model.preprocess`/`pre_enc` must be handled deliberately because, when `channel_independence=0`, it performs an additional moving-average decomposition before embedding that is independent of `decomp_method`.

## 5. Risk audit for later EMA work

### 5.1 DFT branch compatibility

- EMA must be a separate explicit branch; existing `dft_decomp` behavior and `top_k` must not inherit EMA arguments or state.
- Current DFT code uses `rfft(..., dim=1)`, but `freq[0]=0` indexes the first batch element rather than the DC frequency bin, and `torch.topk` has no `dim` argument so it selects along the last feature dimension. These are existing behaviors and were not changed in Stage 00.
- `irfft` does not pass `n=x.size(1)`; odd time lengths can reconstruct to `T-1`. Baseline lengths are all even, but other multiscale settings may not be.
- CUDA half-precision FFT support is length/dtype sensitive. A future EMA branch must not alter or accidentally autocast the DFT path differently.

### 5.2 `channel_independence`

- Baseline is `channel_independence=1`; each original variable is folded into the batch axis before embedding. EMA state must be local to each sample/variable/feature and must never persist across batches.
- With `channel_independence=0`, `Model.preprocess` performs a separate moving-average split before PDM. Replacing only the PDM decomposition would leave this earlier split as moving average. This needs an explicit method decision and tests rather than an implicit change.

### 5.3 Sequence-length boundaries

- Baseline scales are safe: `96/48/24/12`.
- Later implementations must test `down_sampling_layers=0`, length 1, non-divisible lengths, and lengths shorter than any nominal smoothing window.
- Scale coefficients should be selected from the actual enumerated scale or actual `x.size(1)`, not inferred through unchecked indexing.
- The existing centered moving average preserves length for odd kernels such as 25; even kernels do not receive symmetric total padding and can shorten the sequence. EMA should guarantee exact input/output length.

### 5.4 AMP, dtype, device, and autograd

- Any EMA initializer/state must be created from the input tensor (or cast/moved to it), not as a CPU/default-dtype tensor.
- Under AMP, long recurrent accumulation in fp16/bf16 may lose precision. If accumulation is intentionally promoted to fp32, the output dtype and gradient path must be tested and documented.
- Avoid in-place recurrence on views that can break autograd. Fixed alpha should not accidentally become a trainable parameter in this research stage.

### 5.5 Checkpoint compatibility

- Keeping the default `decomp_method=moving_avg` and constructing EMA only when explicitly selected preserves the old moving-average state dict.
- Registering new persistent EMA buffers/parameters unconditionally would make strict loading of old checkpoints fail. A fixed coefficient should be a Python value or a non-persistent buffer unless a later stage explicitly requires learned state.
- The current checkpoint setting string does not encode `decomp_method`, moving-average window, or future EMA coefficients. New experiment IDs/descriptions must prevent overwriting or confusing baseline and EMA checkpoints.

### 5.6 Default behavior and experimental hygiene

- The default CLI value must remain `moving_avg`; existing scripts must continue to produce the original TimeMixer graph and outputs unless an EMA option is explicitly supplied.
- Do not edit or reuse the existing baseline logs/checkpoint directories for EMA experiments.
- Before implementation, preserve the exact meaningful dirty-tree changes needed for reproducibility on an isolated research branch; do not fold unrelated AdaptivePatch/M4 work into the EMA comparison.
- Hyperparameter selection must remain validation-based; test metrics must not be used to choose EMA coefficients.

## 6. Files changed and verification performed in Stage 00

- TSLib files changed by this stage: none.
- Bridge file added by this stage: `results/00_baseline_audit.md`.
- Verified: remote Python/PyTorch/CUDA/GPU environment; Git remote/branch/SHA/status; 13 dated logs of which 12 contain successful final metrics and one contains the documented failure; all 12 successful checkpoint files; all per-seed metrics and four horizon means; source call chain and current line locations.
- Not run: training, evaluation, inference, data preprocessing, source-formatting commands, Stage 01.

## 7. Final conclusion

The existing ETTh1 TimeMixer result set is sufficient as an empirical comparison point, and its metrics/checkpoints are intact. It is not yet a fully frozen code baseline because the exact run depended on uncommitted runtime/data-loader changes and the array-level result artifacts have been removed. Stage 00 itself passes because the discrepancy and all relevant code paths, artifacts, commands, risks, and edit boundaries are now documented without modifying TSLib.

### Recommended edit points

- `models/TimeMixer.py:9–26` — adjacent TimeMixer-local decomposition implementation area.
- `models/TimeMixer.py:129–165` — PDM decomposition selection and per-scale invocation.
- `models/TimeMixer.py:201,277–287` — channel-dependent pre-embedding decomposition decision point.
- `run.py:72,81–89` — explicit decomposition/EMA CLI configuration while retaining the moving-average default.
- A new EMA-specific ETTh1 launch script beside `scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh`, leaving the baseline script unchanged.
