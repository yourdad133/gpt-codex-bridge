# TimeMixer 审查与自适应 Patch 设计

> 来源对话：审查 TimeMixer 并设计自适应 Patch
> 导出时间：2026-09-12
> 说明：以下内容为该对话的最终回答；报告中的源码路径对应原始 Time-Series-Library 工作区。

审查完成。本阶段未修改文件、未训练、未使用 GPU、未 commit/push。

## 1. Git 状态

- 当前分支：`main`
- 分支状态：与 `origin/main` 同步
- 远程：

```text
origin https://github.com/thuml/Time-Series-Library.git (fetch)
origin https://github.com/thuml/Time-Series-Library.git (push)
```

存在本地修改：

```text
layers/StandardNorm.py
models/TimeMixer.py
scripts/long_term_forecast/ECL_script/TSMixer.sh
scripts/long_term_forecast/ECL_script/TimeMixer.sh
scripts/long_term_forecast/ETT_script/TSMixer_ETTh1.sh
scripts/long_term_forecast/ETT_script/TSMixer_ETTh2.sh
scripts/long_term_forecast/ETT_script/TSMixer_ETTm1.sh
scripts/long_term_forecast/ETT_script/TSMixer_ETTm2.sh
scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
scripts/long_term_forecast/ETT_script/TimeMixer_ETTh2.sh
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2.sh
scripts/long_term_forecast/Traffic_script/TSMixer.sh
scripts/long_term_forecast/Traffic_script/TimeMixer.sh
scripts/long_term_forecast/Weather_script/TSMixer.sh
scripts/long_term_forecast/Weather_script/TimeMixer.sh
scripts/short_term_forecast/TSMixer_M4.sh
scripts/short_term_forecast/TimeMixer_M4.sh
```

未跟踪文件：

```text
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_paper.sh
```

`git diff --stat`：

```text
18 files changed, 0 insertions(+), 0 deletions(-)
```

进一步检查显示，18 个已跟踪文件主要是文件模式从 `100755` 变为 `100644`，没有内容增删。未跟踪的 `TimeMixer_ETTm1_paper.sh` 应视为用户文件，未触碰。

## 2. TimeMixer 实际数据流

源码：[models/TimeMixer.py](X:/Time-Series-Library/models/TimeMixer.py:187)

设：

```text
B = batch size
L = seq_len
C = enc_in
D = d_model
M = down_sampling_layers
S = M + 1
T_m = L // down_sampling_window**m
```

实际流程：

```text
x_enc                  [B, L, C]
x_mark_enc             [B, L, F]
   │
   ▼
multi-scale downsampling
   │
   ├── X0                [B, T0, C]
   ├── X1                [B, T1, C]
   ├── ...
   └── XM                [B, TM, C]
   │
   ▼
per-scale normalization
   │
   ▼
channel independence reshape / pre-encoding
   │
   ▼
DataEmbedding_wo_pos
   │
   ├── H0                [B_eff, T0, D]
   ├── H1                [B_eff, T1, D]
   ├── ...
   └── HM                [B_eff, TM, D]
   │
   ▼
PastDecomposableMixing × e_layers
   │
   ▼
future_multi_mixing
   │
   ├── Y0                [B, pred_len, C]
   ├── Y1                [B, pred_len, C]
   └── ...
   │
   ▼
stack + sum + denormalization
   │
   ▼
forecast               [B, pred_len, C]
```

`x_dec` 和 `x_mark_dec` 在 TimeMixer 的 `forecast` 中实际上没有参与预测计算。

### ETTm1 当前 baseline 的各 scale

当前脚本使用：

```text
L = 96
C = 7
D = 16
down_sampling_window = 2
down_sampling_layers = 3
```

因此：

| scale | 时间长度 | 原始输入 | CI=1 后 embedding |
|---|---:|---|---|
| 0 | 96 | `[B, 96, 7]` | `[7B, 96, 16]` |
| 1 | 48 | `[B, 48, 7]` | `[7B, 48, 16]` |
| 2 | 24 | `[B, 24, 7]` | `[7B, 24, 16]` |
| 3 | 12 | `[B, 12, 7]` | `[7B, 12, 16]` |

当前脚本未传 `--freq`，所以 `run.py` 默认使用 `freq=h`，`timeF` 时间特征维度为 `F=4`。

## 3. PDM 结构

### `PastDecomposableMixing`

输入：

```text
H_m: [B_eff, T_m, D]
```

首先按时间维做 decomposition：

```text
season: [B_eff, T_m, D]
trend:  [B_eff, T_m, D]
```

随后转置为：

```text
[B_eff, D, T_m]
```

`MultiScaleSeasonMixing` 在时间维上进行：

```text
T_m → T_{m+1}
```

`MultiScaleTrendMixing` 反向进行：

```text
T_{m+1} → T_m
```

最终每个 scale 恢复为：

```text
[B_eff, T_m, D]
```

PDM 不改变每个 scale 的 token 数量。

### `future_multi_mixing`

CI 模式下：

```text
[B*C, T_m, D]
   │
   ▼
Linear(T_m, pred_len)
   │
   ▼
[B*C, pred_len, D]
   │
   ▼
projection D → 1
   │
   ▼
reshape
   │
   ▼
[B, pred_len, C]
```

所有 scale 的预测结果：

```text
[B, pred_len, C, S]
```

最后沿 scale 求和，得到：

```text
[B, pred_len, C]
```

## 4. 固定时间长度依赖

TimeMixer 中所有关键固定长度依赖如下：

1. `MultiScaleSeasonMixing`

```python
configs.seq_len // configs.down_sampling_window ** i
```

对应：

```text
T_i → T_{i+1}
```

2. `MultiScaleTrendMixing`

```python
configs.seq_len // configs.down_sampling_window ** i
```

对应：

```text
T_{i+1} → T_i
```

3. `predict_layers`

```python
nn.Linear(T_i, pred_len)
```

4. 非 CI 模式的残差路径：

```python
out_res_layers[i]      : T_i → T_i
regression_layers[i]   : T_i → pred_len
```

5. 分类任务还有：

```python
nn.Linear(d_model * seq_len, num_class)
```

这些定义位于：[models/TimeMixer.py](X:/Time-Series-Library/models/TimeMixer.py:29)

### 问题 A：为什么不能直接使用不同数量的 patch token？

不能直接使用，原因是：

- PyTorch batch tensor 必须是规则的 dense tensor，不能让同一个 scale 内不同样本拥有不同长度。
- PDM 的 `out_low + out_low_res`、`out_high + out_high_res` 要求对应 scale 的时间维完全一致。
- 所有 `nn.Linear(T_i, ...)` 已经在初始化时固定了输入维度。
- 即使 padding 到最大长度，当前 PDM 没有 mask 机制，padding 会参与 decomposition、scale mixing 和 prediction。
- CI 模式下 batch 已经变成 `B*C`，动态长度还必须同时保持样本和变量的对应关系。

### 问题 B：恢复为 `[B, L_m, D]` 后能否送入 PDM？

可以，但必须满足：

```text
L_m == 原 TimeMixer 配置中的 T_m
```

并且：

- 每个 batch 样本的 `L_m` 必须相同；
- scale 顺序不能改变；
- `D` 必须保持不变；
- `channel_independence=1` 时，实际输入应为：

```text
[B*C, L_m, D]
```

而不是 `[B, L_m, D]`；
- 如果 `L_m` 只是一个不同的固定长度，也不能直接送入现有 PDM，除非同时重建所有相关 Linear 层。

因此后续 Dynamic Patch 应采用：

```text
输入  [B_eff, T_m, D]
输出  [B_eff, T_m, D]
```

即 shape-preserving residual enhancement。

## 5. `channel_independence`

源码位置：[models/TimeMixer.py](X:/Time-Series-Library/models/TimeMixer.py:329)

当：

```python
channel_independence = 1
```

时：

```text
[B, T, C]
 → permute
[B, C, T]
 → reshape
[B*C, T, 1]
 → embedding
[B*C, T, D]
```

所以答案是：是的，embedding 后实际 batch 类似：

```text
[B*C, T, D]
```

其中每一行对应一个独立的单变量时间序列：

```text
row = b * C + c
```

后续 Router 应把 `B*C` 当作有效 batch：

```text
B_eff = B*C
```

建议：

- `π(m,t,k)` 在 `[B_eff, T_m, K]` 上计算；
- scale router 的结果也应按 `B_eff` 对齐；
- load-balance 或 entropy 辅助损失应在 `B_eff` 上聚合；
- 如果需要恢复原始样本结构，应显式 reshape 为 `[B,C,...]`；
- 不应未经处理地把 `B*C` 当作普通样本维度。

另外发现一个已有 baseline 风险：

```python
x_mark = x_mark.repeat(N, 1, 1)
```

而输入数据的 reshape 顺序是：

```text
[B,C,T] → [B*C,T,1]
```

`repeat(N,...)` 通常会产生：

```text
[b0,b1,...,b0,b1,...]
```

但输入 reshape 更接近：

```text
[b0c0,b0c1,b1c0,b1c1,...]
```

因此时间标记在 `B>1`、`C>1` 时可能与通道序列错位。此阶段不修改 baseline，但后续需要单独确认是否修正为按样本连续重复。

## 6. 模型注册方式

当前版本不是手动 `model_dict`。

源码：[exp/exp_basic.py](X:/Time-Series-Library/exp/exp_basic.py:25)

实际过程是：

```text
扫描 models/*.py
    ↓
文件名去掉 .py 作为模型名
    ↓
动态 import models.<model_name>
    ↓
查找模块中的 class Model
```

因此未来只需新增：

```text
models/TimeMixerAdaptivePatch.py
```

并定义：

```python
class Model(...):
    ...
```

不需要修改：

```text
models/__init__.py
run.py
手动 model_dict
```

`run.py` 只负责参数解析和选择 Experiment。

## 7. Loss 计算位置

源码：[exp/exp_long_term_forecasting.py](X:/Time-Series-Library/exp/exp_long_term_forecasting.py:37)

当前 criterion：

```python
criterion = nn.MSELoss()
```

实际训练损失位置：

```text
AMP 分支：exp/exp_long_term_forecasting.py:122
非 AMP 分支：exp/exp_long_term_forecasting.py:130
```

验证损失：

```text
exp/exp_long_term_forecasting.py:69
```

最小后续修改方案：

```text
prediction_loss = criterion(outputs, batch_y)
loss = prediction_loss + λ * router_aux_loss
```

只在训练的 AMP/非 AMP 两个分支加入辅助损失。验证和测试仍使用 prediction loss，避免改变 early stopping 和测试指标含义。

不建议让模型直接返回 `(prediction, aux_loss)`，因为当前验证和测试代码默认 `outputs` 是 tensor。更稳妥的是让新模型保存：

```python
self.router_aux_loss
```

训练入口读取该属性。使用 `DataParallel` 时要注意：

```python
self.model.module.router_aux_loss
```

## 8. ETTm1 baseline 参数

正式已跟踪脚本：

[TimeMixer_ETTm1.sh](X:/Time-Series-Library/scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1.sh:1)

| 参数 | 值 |
|---|---|
| `seq_len` | 96 |
| `label_len` | 0 |
| `pred_len` | 96 / 192 / 336 / 720 |
| `e_layers` | 2 |
| `d_model` | 16 |
| `d_ff` | 32 |
| `batch_size` | 16 |
| `learning_rate` | 0.01 |
| `train_epochs` | 10，来自 `run.py` 默认值 |
| `patience` | 3，来自 `run.py` 默认值 |
| `down_sampling_layers` | 3 |
| `down_sampling_window` | 2 |
| `down_sampling_method` | `avg` |
| `channel_independence` | 1，来自默认值 |
| `moving_avg` | 25，来自默认值 |

另有未跟踪文件：

```text
scripts/long_term_forecast/ETT_script/TimeMixer_ETTm1_paper.sh
```

它不是当前正式 tracked baseline，主要差异是：

```text
batch_size = 128
patience = 15
itr = 3
train_epochs = 10
moving_avg = 25
```

公平实验应优先以已跟踪的 `TimeMixer_ETTm1.sh` 为准，除非明确指定使用 paper 版本。

## 9. 推荐文件结构

建议：

```text
models/TimeMixerAdaptivePatch.py
layers/AdaptivePatchRouter.py
```

暂不新增：

```text
utils/router_utils.py
```

除非后续发现路由损失或统计函数被多个模块复用。

### 继承还是复制？

推荐继承 TimeMixer：

```python
from models.TimeMixer import Model as BaseTimeMixerModel

class Model(BaseTimeMixerModel):
    ...
```

理由：

- 直接复用现有 PDM；
- 复用 `future_multi_mixing`；
- 复用 normalization、projection 和 baseline 参数；
- 避免复制 500 多行 baseline；
- 保持 baseline 与新模型的结构差异最小；
- 不修改 `models/TimeMixer.py`。

需要注意，baseline 的 `__multi_scale_process_inputs` 是私有方法，子类重写 `forecast` 时需要显式调用基类的私有方法，或者在新文件中复用一小段输入处理逻辑。这是继承方案的唯一主要耦合点，但仍比完整复制 baseline 更易维护。

## 10. 后续设计图

```text
x_enc
[B, L, C]
   │
   ▼
Multi-scale TimeMixer inputs
   │
   ├── X0 [B, T0, C]
   ├── X1 [B, T1, C]
   ├── X2 [B, T2, C]
   └── ...
   │
   ▼
Normalization + Channel Independence
   │
   ▼
Embedding
   │
   ├── H0 [B_eff, T0, D]
   ├── H1 [B_eff, T1, D]
   ├── H2 [B_eff, T2, D]
   └── ...
   │
   ▼
Adaptive Patch Router
   │
   ├── Local Patch scale 0
   ├── Local Patch scale 1
   ├── Local Patch scale 2
   └── ...
   │
   ▼
Local routing
π(m,t,k)
   │
   ▼
Scale routing
α(m)
   │
   ▼
Joint routing
P(m,t,k) = α(m) π(m,t,k)
   │
   ▼
Residual Patch Enhancement
H'_m = H_m + ΔH_m
   │
   │  输出必须保持 [B_eff, T_m, D]
   ▼
Original TimeMixer PDM
   │
   ▼
future_multi_mixing
   │
   ▼
Prediction
[B, pred_len, C]
```

第一版建议采用 soft routing，不引入 RL、FFT、diffusion 或 causal module。

## 11. 当前技术风险

1. PDM 不是任意长度 token 的模块，Dynamic Patch 必须恢复为固定 `T_m`。
2. `seq_len` 不能随意换成不可整除 `down_sampling_window` 的值。
3. `avg/max/conv` 的实际下采样长度可能与 `seq_len // window**m` 不一致。
4. CI 模式下 Router 的有效 batch 是 `B*C`。
5. PDM 后的 `predict_layers`、非 CI residual layers 仍依赖固定长度。
6. 路由辅助损失需要兼容 AMP 和 `DataParallel`。
7. 当前 `x_mark.repeat(N,...)` 可能造成时间特征与通道序列错位。
8. `run.py` 中的 `--loss` 参数目前没有真正参与 criterion 选择，始终使用 MSE。

## 12. 最终结论

当前项目适合进入子任务 2：实现独立的 Local Dynamic Patch 模块。

建议子任务 2 只实现：

```text
layers/AdaptivePatchRouter.py
```

并建立严格接口：

```text
List[[B_eff, T_m, D]]
    →
List[[B_eff, T_m, D]]
```

先不要修改 baseline TimeMixer、训练入口或实验脚本。