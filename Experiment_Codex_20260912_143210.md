# TimeMixerAdaptivePatch 子任务 3 执行归档

日期：2026-09-12

## 请求摘要

读取当前项目 `AGENTS.md` 后，按 [TimeMixerAdaptivePatch 子任务 3](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask3.md) 接入原始 TimeMixer、Local Dynamic Patch、Scale Router 和 factorized joint scale/patch routing。本阶段不改训练 loss、CLI、baseline 或正式训练脚本，不进行完整训练。

## 最终结果

子任务 3 架构接入完成。新增 `models/TimeMixerAdaptivePatch.py`，继承并复用原始 `models.TimeMixer.Model`；`layers/AdaptivePatchRouter.py` 未修改。

实现包含：

- embedding 与 PDM 之间的 Adaptive Patch enhancement；
- 从原始 `H_m` temporal mean pooling 得到 scale representation；
- 每个 scale 独立 scoring MLP 的 learned/uniform Scale Router；
- `Hfinal_m = H_m + alpha[:,m].view(B_eff,1,1) * (Hpatch_m - H_m)`；
- `get_router_aux_losses()`，返回可微 `patch_balance` 与 `scale_balance`；
- `get_router_stats()`，返回 detached scale/patch probabilities、entropy 和 beta。

## 数学定义

`pi(m,t,k) = P(k | m,t,X)` 是 scale 内部的 Patch Router，`alpha_m = P(m | X)` 是 Scale Router，联合路由为：

`P(m,t,k | X) = alpha_m * pi(m,t,k)`。

Scale balance 使用 `KL(q || Uniform)`，其中 `q_m = mean_B_eff(alpha[:,m])`；patch/scale entropy 仅作为诊断统计，不加入本阶段 loss。

## 验证结果

所有 Python/CUDA 相关验证均在项目规定的远程 `tslib` 环境完成；PyTorch 为 `2.5.1+cu121`，CUDA 可用且两张 GPU 可见。

- import 与自动模型发现：PASS
- ETTm1 典型 embedding shape：`[(14,96,16), (14,48,16), (14,24,16), (14,12,16)]`
- Adaptive Patch shape preservation：同上，PASS
- learned alpha shape：`(14,4)`；最大概率和误差：`0.0`
- uniform scale routing 最大误差：`0.0`
- full forecast：`(2,96,7)`，PASS
- scale balance：`0.0035427846014499664`
- patch balance：`0.04157369211316109`
- scale entropy：`1.3827369213104248`
- mean scale probabilities：`[0.24991877377033234, 0.28472068905830383, 0.23401610553264618, 0.2313443273305893]`
- factorized joint gating 最大恒等式误差：`2.2887252271175385e-07`
- shared-state loading：unexpected keys `[]`；132 个 missing keys 全部属于新增 `adaptive_patch`/`scale_router` 模块
- beta=0 baseline degeneration：max abs diff `0.0`；mean abs diff `0.0`
- 代表性梯度范数：Patch Encoder `0.0005429930170066655`；Patch Router `0.11934448156118393`；beta `0.00687468471005559`；Scale Router `0.04334789514541626`；baseline backbone `2.5838546752929688`
- router stats detached、模型级 uniform patch/scale、单尺度 router 边界：PASS
- finite/no NaN/no Inf 与 CPU smoke：PASS

## 参数量

- 原始 TimeMixer：`75,497`
- TimeMixerAdaptivePatch：`98,561`
- 新增：`23,064`，约 `30.549558260593138%`
- Adaptive Patch：`18,452`
- Scale Router：`4,612`

## 文件变化

- 新增：`models/TimeMixerAdaptivePatch.py`
- 修改：无 `layers/AdaptivePatchRouter.py` 修改；无 baseline、训练入口、CLI 或脚本修改

项目工作区在本任务开始时已有其他改动，本次未覆盖或清理。普通 Git index 当前损坏，常规 `git status`/`git diff` 会报 index error；使用临时只读 index 检查时确认新增模型文件及既有工作区状态，未执行 reset/clean/checkout/commit/push 项目代码。

## Known Limitations

固定 patch candidate 集合和固定网格边界；Dynamic Patch 最终恢复原时间长度；Scale Router 当前按 `B_eff` 独立运行；channel-independence 模式不做跨变量 routing；baseline `x_mark.repeat` 风险未修复；训练 auxiliary loss/CLI/日志接入留到子任务 4；未开始正式 benchmark。

## 结论

子任务 3 的架构级测试全部通过，`TimeMixerAdaptivePatch` 已具备进入子任务 4（训练框架、CLI、auxiliary loss integration、router logging）的条件。未执行子任务 4。
