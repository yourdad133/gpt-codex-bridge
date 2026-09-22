# Stage 03 — 将 EMA / Scale-Aware EMA 接入 TimeMixer，保持 baseline 完全兼容

## 前置条件

Stage 02 必须 PASS。

## 目标

把已测试的 EMA decomposition 接入 TimeMixer PDM，但不改变 TimeMixer 其他结构。

最终需要支持至少三种 decomposition：

- 原始 moving average（原行为）；
- `ema`：所有尺度共享同一个 alpha_base；
- `scale_aware_ema`：每个尺度按公式得到 alpha_k。

如果原项目还有 `dft_decomp`，必须保留并确保不被破坏。

## 配置要求

基于实际 CLI 结构扩展，不要复制一套新的运行框架。

建议但需结合真实代码确认：

- `--decomp_method scale_aware_ema`
- `--ema_alpha_base <float>`

同时让：

`--decomp_method ema`

使用同一个 `ema_alpha_base` 作为固定 alpha。

关键要求：

- 原项目默认值保持不变；
- 不传任何新参数时，原 TimeMixer 的行为应与修改前一致；
- 新增参数不能影响非 TimeMixer 模型；
- 日志中打印 decomposition 类型；
- 对 EMA 模式打印每个 scale 实际 alpha。

## PDM 集成要求

在 multi-scale x_list decomposition 时：

- `ema` 对每个 scale 使用相同 alpha_base；
- `scale_aware_ema` 根据当前 scale index k 计算 alpha_k；
- seasonal / trend 的后续 mixing 路径不能改；
- 不改变 tensor shape；
- 不改变 downsampling；
- 不改变 normalize / embedding / prediction head。

## Baseline regression tests

必须至少验证：

1. 原 moving-average 配置可正常实例化；
2. 原 DFT 配置（若存在）可正常实例化；
3. EMA 配置可正常实例化；
4. Scale-Aware EMA 可正常实例化；
5. 对同一个随机输入，所有模式 forward shape 相同；
6. 原默认配置下没有引入新增可训练参数；
7. EMA v1 本身没有新增可训练参数；
8. 原 baseline 的参数量不发生意外变化。

如果可以，固定 random seed，对修改前后的原 moving-average 模式做一次 deterministic forward regression；若无法获得修改前输出，至少通过现有 baseline smoke command 验证。

## 实验脚本

不要覆盖官方/原有脚本。

新增独立的 SAEMA 实验脚本或命令模板，命名清楚，例如：

`TimeMixer_ETTh1_SAEMA.sh`

实际位置根据当前项目结构决定。

脚本必须显式写出：
- decomp_method；
- ema_alpha_base；
- seed；
- pred_len。

## 本阶段只做 smoke forward

只需要模型级/数据级短 smoke，不进行完整 10 epoch 正式训练。

## 输出

写入：

`results/03_integrate_timemixer.md`

包含：

- changed files；
- 新 CLI；
- 三种 decomposition 的实际调用关系；
- 每个 scale alpha 的日志示例；
- regression/smoke test；
- 参数量对比；
- diff stat；
- 已知风险。

完成并 push 后停止，不执行 Stage 04。
