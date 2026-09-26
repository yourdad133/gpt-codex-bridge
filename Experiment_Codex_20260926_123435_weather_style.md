# Weather TimeXer 对比脚本风格调整

- 日期：2026-09-26
- 请求：将两份 Weather 运行脚本改为与提供的 ETTm2 示例一致的逐条命令风格。

## 结果

已更新 GTR-main 中的以下文件：

- `scripts/TimeXer_Weather_M.sh`
- `scripts/TimeXerGTR_Weather_M.sh`

每份脚本现有四段独立的 `python -u run.py` 命令，对应预测长度 96、192、336、720。均使用 Weather 多变量预测及相同的公共超参数；TimeXerGTR 仅额外包含 `--gtr_cycle_len 144` 与 `--gtr_gate_init 0.1`。两份脚本的训练轮数已统一为 10，GPU 设置按示例统一为 0。

## 验证

- 修改前已读取 GTR-main 的 AGENTS.md。
- 两份脚本的 Bash 语法检查通过。
- 去除模型名称及 GTR 专属参数后，两份脚本文本完全一致；每份均包含四条运行命令。
- 本次未执行远程训练或 CUDA 测试，因此没有实验指标。

## 使用

在服务器的 GTR-main 项目目录中激活 `tslib` 环境后，分别运行：

```bash
bash scripts/TimeXer_Weather_M.sh
bash scripts/TimeXerGTR_Weather_M.sh
```
