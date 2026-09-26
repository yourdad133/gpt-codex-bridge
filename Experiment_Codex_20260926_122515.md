# Weather TimeXer / TimeXerGTR 实验脚本归档

- 日期：2026-09-26
- 请求：在服务器工作区编写两份分别运行 TimeXer 和 TimeXerGTR 的 Weather 数据集脚本，除模型专属参数外保持其余超参数一致；执行前读取 GTR-main 的 AGENTS.md。

## 最终结果

已在 GTR-main 仓库新增两份脚本：

- `scripts/TimeXer_Weather_M.sh`
- `scripts/TimeXerGTR_Weather_M.sh`

两份脚本分别训练并测试 96、192、336、720 步预测，均使用 Weather 数据集的 21 个数值通道、96 步输入、相同随机种子、网络规模、训练参数和 GPU 配置。TimeXerGTR 脚本只额外指定 `--gtr_cycle_len 144` 和 `--gtr_gate_init 0.1`。

## 修改文件

- `GTR-main/scripts/TimeXer_Weather_M.sh`
- `GTR-main/scripts/TimeXerGTR_Weather_M.sh`

## 验证

- 已读取 GTR-main 的 AGENTS.md。
- 已确认 Weather CSV 存在，含日期列及 21 个数值通道。
- 两份脚本均通过 Bash 语法检查；去除模型名称及 GTR 专属参数后，其余脚本文本完全一致。
- 远程 SSH 认证未通过，因此尚未完成远程环境、CUDA 或训练验证；本次没有实验指标。

## 运行方式

在服务器的 GTR-main 项目目录中分别运行：

```bash
bash scripts/TimeXer_Weather_M.sh
bash scripts/TimeXerGTR_Weather_M.sh
```
