# Stage 04 — ETTh1 三种 decomposition 的训练 Smoke Test

## 目标

确认：

- moving average baseline；
- fixed EMA；
- Scale-Aware EMA；

都能从训练 → validation → checkpoint → test 完整跑通，并且不会 NaN/Inf。

这不是正式结果，不用于论文结论。

## 固定条件

使用与此前 TimeMixer ETTh1 baseline 相同的核心配置：

- long-term forecasting；
- ETTh1；
- seq_len=96；
- pred_len=96；
- seed=2021；
- 其余模型结构参数保持 baseline 一致。

为了 smoke，可以把 epoch 降到 2（或项目中合理的最短完整训练轮数），但三种 decomposition 必须完全一致。

EMA 暂用：

`ema_alpha_base = 0.10`

Scale-Aware EMA 必须在日志中记录全部 scale alpha。

## 三个实验

A. original moving average  
B. fixed EMA(alpha=0.10)  
C. Scale-Aware EMA(alpha_base=0.10)

## 检查项

每个实验记录：

- 完整命令；
- GPU；
- start/end 时间；
- epoch loss；
- validation loss；
- 是否成功保存 checkpoint；
- 是否成功 load best checkpoint；
- test MSE / MAE；
- 是否存在 NaN / Inf；
- peak GPU memory（能方便获得时记录）；
- 总 wall-clock time。

注意：smoke test 指标只用来判断代码是否工作，不比较“谁更好”。

如果出现 bug：
- 可以修复实现 bug；
- 必须在结果里写清楚 bug、原因、改动；
- 不允许为了让 EMA 指标变好而改变训练参数。

## 输出

写：

`results/04_smoke_training.md`

如果有结构化数据，同时写：

`results/data/04_smoke_training.csv`

完成后停止，不执行 Stage 05。
