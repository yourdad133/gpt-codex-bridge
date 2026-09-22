# Stage 02 — 实现 EMA decomposition 模块与单元测试（暂不接入 TimeMixer）

## 前置条件

读取：

- `results/00_baseline_audit.md`
- `results/01_method_spec.md`

严格遵守其中 Frozen v1 specification。

## 目标

在 TSLib 中独立实现：

- fixed EMA decomposition；
- Scale-Aware EMA decomposition；

并通过单元测试。

**本阶段不要把它接入 TimeMixer forward/PDM。**

## Git 安全

在第一次修改源码前：

1. 记录当前 commit；
2. 如果当前不是专门的研究分支，创建并切换到：
   `research/scale-aware-ema`
3. 不要 reset 或覆盖 baseline 分支已有内容。

## 实现要求

根据 Stage 01 确认的真实工程结构，新建最合适的 EMA decomposition 模块。

核心行为：

`trend[:,0,:] = x[:,0,:]`

对 t>=1：

`trend[:,t,:] = alpha * x[:,t,:] + (1-alpha) * trend[:,t-1,:]`

`seasonal = x - trend`

Scale-Aware EMA:

`alpha_k = 1 - (1-alpha_base)^(down_sampling_window^k)`

要求：

- 输入输出 shape 与 TimeMixer 当前 decomposition 完全兼容；
- 不使用 numpy 参与 forward；
- 不 `.detach()`；
- 不进行 in-place 操作导致 autograd 异常；
- 保持 input 的 device；
- 尽量保持 input dtype；
- alpha 做严格范围验证，非法值给出清楚错误；
- sequence length=1 可正常运行。

## 单元测试

至少覆盖：

1. 输出 shape 与输入一致；
2. `seasonal + trend == x`（数值误差范围内）；
3. 常数序列：trend 应保持常数、seasonal 近似 0；
4. 固定 EMA 的手算小样本正确；
5. Scale-Aware alpha 公式正确；
6. 当 w>1 时 alpha_k 随 k 单调增大；
7. alpha 始终位于 (0,1)；
8. CPU forward；
9. CUDA 可用时 GPU forward；
10. backward 能产生有限梯度；
11. float32；环境允许时额外检查 float16/bfloat16；
12. length=1；
13. 非法 alpha_base 报错。

## 性能记录

用一个与 ETTh1 常见输入量级接近的随机张量，记录 EMA forward 的粗略耗时。这里只用于发现极端低效实现，不做正式性能结论。

## 本阶段禁止

- 不修改 TimeMixer PDM 调用逻辑；
- 不跑正式训练；
- 不改 baseline 脚本；
- 不加 instance adaptive；
- 不把 alpha 变成 Parameter。

## 输出

在 bridge 仓库写：

`results/02_implement_ema_module.md`

必须包含：

- Status；
- TSLib branch / commit before / commit after（如有 commit）；
- changed files；
- 核心 API；
- 所有测试命令与结果；
- 失败测试及修复；
- 简短性能记录；
- `git diff --stat`；
- 是否满足 Frozen v1 specification。

如测试失败无法修复，Status=FAIL 并停止。

完成后 push 结果，停止。不要执行 Stage 03。
