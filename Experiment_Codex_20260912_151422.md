# TimeMixer AdaptivePatch 子任务 4 执行归档

## 请求

执行 [TimeMixer AdaptivePatch 子任务 4](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask4.md)：

- 将 patch/scale balance auxiliary loss 接入 TSLib 长期预测训练；
- 补齐 CLI、消融开关、AMP/非 AMP 兼容和 router 统计；
- 完成最小真实 ETTm1 smoke test；
- 不执行完整 benchmark，不修改原始 TimeMixer 架构，不修复 Git index。

## 实现结果

训练目标为：

L_total = L_pred + lambda_patch * L_patch_balance + lambda_scale * L_scale_balance

默认两个权重均为 0.001。auxiliary loss 只参与训练优化；validation、early stopping 和 test 仍只使用 prediction loss/forecasting metrics。DataParallel 训练通过模型 forward 显式返回 replica-produced auxiliary tensors，再做跨 replica mean，避免从临时 replica 之外读取失真的缓存属性。

## 改动文件

- run.py：新增 patch candidate、patch/scale routing、router hidden width、temperature、adaptive alpha 以及两个 balance weight 参数。
- exp/exp_long_term_forecasting.py：新增能力检测、训练 loss 合成、AMP/非 AMP 的 total-loss backward、epoch loss 分项日志和 router 统计/轻量 collapse warning。
- models/TimeMixerAdaptivePatch.py：新增可选 return_router_aux=True 的 forward 返回接口；默认 forward 返回值保持不变。

## 验证

- 远程环境：PyTorch 2.5.1+cu121，CUDA 可用，2 张 RTX 4090 D 可见。
- Python compile：通过。
- CLI：--patch_sizes 4 8 16 32 与 --patch_sizes 16 解析通过；learned/uniform routing 解析与构造通过。
- 真实 ETTm1：读取真实 dataloader，batch shape (2, 96, 7)，完成单 batch 非 AMP 与 AMP 的实际 Exp_Long_Term_Forecast.train()。
- 非 AMP 数值：prediction 0.94601750，patch balance 0.02026952，scale balance 0.01266158，total 0.94605041，公式最大误差 2.934e-08。
- zero weights：total_loss == prediction_loss，误差 0。
- 梯度与更新：patch router、scale router、patch encoder、beta、backbone 梯度均 finite；patch/scale router 参数均发生更新。
- AMP：单 batch 实际训练入口通过。
- learned stats：概率归一化、entropy、patch probabilities、beta 均 finite。
- uniform ablation：scale probabilities 为 [0.25, 0.25, 0.25, 0.25]，patch/scale balance 均为 0。
- single patch：p=4、p=8 全尺度通过；p=16 在最短 T=12 scale 按设计报 No valid patch size。
- baseline compatibility：TimeMixer、TimesNet train-style path 通过，普通模型 auxiliary terms 为零。
- DataParallel：两卡 auxiliary loss 可微、finite，scale router 有梯度并完成 optimizer step。
- 未执行完整 ETTm1 benchmark、四个 pred_len 全量训练或超参搜索。

## 限制与安全

- DataParallel auxiliary loss 已验证；其 base module 的缓存 router stats 不作为多卡聚合统计源，单卡 epoch-level router logging 已验证。
- 保留固定 patch 网格、最终恢复原序列长度和 channel-independence 约束。
- 原始 TimeMixer 源文件、本阶段之外的工作区修改均未触碰；Git index 损坏未修复。
- 未 reset、clean、checkout 覆盖工作区，未提交或推送 TSLib 工作区。

