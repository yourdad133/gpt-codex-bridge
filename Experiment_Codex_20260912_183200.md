# TimeMixerAdaptivePatch 正式实验归档

日期：2026-09-12

## 请求摘要

执行 TimeMixerAdaptivePatch 的正式实验阶段：ETTm1 96→96，完成 fixed2/fixed4/fixed6/fixed8/fixed12 强 Fixed-Patch 对照，并对 baseline、Best Fixed、patch_only、joint 使用 seed 2021/2022/2023 做多 seed 稳定性验证；同时复核路由动态性并避免实验结果重名。

## 最终结果

- Stage A 按验证集最优 loss 选择 Best Fixed：**fixed4**，最优验证 loss **0.392381**。
- Stage A 的 fixed12 在 epoch 2 后出现明显数值爆炸，标记为 unstable，不作为推荐固定 patch 配置。
- Stage B 三 seed 平均测试结果：
  - baseline：MSE 0.327095 ± 0.009885，MAE 0.364252 ± 0.005849
  - fixed4：MSE 0.325828 ± 0.008756，MAE 0.363884 ± 0.005105
  - patch_only：MSE **0.320359 ± 0.002480**，MAE **0.360579 ± 0.001064**
  - joint：MSE 0.322785 ± 0.004253，MAE 0.362672 ± 0.004552
- 相对 baseline 的平均 Test MSE 差值：fixed4 -0.001267、patch_only -0.006736、joint -0.004310。
- patch_only 的 scale 路由在三个 seed 中严格保持 uniform；局部 patch 路由仍表现出随输入和时间位置变化的动态性。joint 同时表现出跨尺度重加权和局部 patch 动态性。
- 全部 12 次正式 seed 运行均完成并输出 early-stopping、Test MSE、Test MAE；未停止或修改其他用户的 GPU 进程。

## 改动与产物

项目工作区新增/修改：

- `run.py`：统一 Python/NumPy/Torch/CUDA seed 设置，`--seed` 在解析后生效。
- `data_provider/data_factory.py`：DataLoader generator 与 worker seed 绑定到实验 seed。
- `exp/exp_long_term_forecasting.py`：fixed 模式下抑制错误的 collapse warning。
- `scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh`：新增 fixed2/fixed6/fixed12，并让显式 seed 进入实验命名与输出目录。
- `scripts/long_term_forecast/ETT_script/analyze_router_dynamics.py`：从最佳 checkpoint 统计验证集 scale/patch 路由动态性。
- `TimeMixerAdaptivePatch_FormalExperiment_Report_20260912.md`：完整正式实验报告。

日志保存在项目 `experiment_logs/` 目录，包含 Stage A、Stage B 全部运行和 `router_dynamics_formal_20260912_201500.log`。

## 验证

- 远程 `tslib` 环境、Torch CUDA 与两张 GPU 检查通过。
- 修改后的 Python 文件和新增分析脚本远程 `py_compile` 通过。
- patch size 2、6、12 及多尺度时序长度的构造/前向 sanity check 通过。
- 12 次正式运行日志均存在且包含完整测试指标。
- 未执行 git reset、clean、checkout、commit 或 push；未覆盖用户既有代码变更。

## 参考

- 任务说明来源：[TimeMixer_AdaptivePatch_Subtask5.md](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
- 完整报告：`X:\\Time-Series-Library\\TimeMixerAdaptivePatch_FormalExperiment_Report_20260912.md`
