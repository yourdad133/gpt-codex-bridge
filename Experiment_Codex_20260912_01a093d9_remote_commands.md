# 实验项目 Codex 回复

- 日期：2026-09-12
- 项目：实验
- 主题：把可执行的远程实验命令写入 X 盘项目配置

## 本次请求

用户要求在 X 盘项目的 `AGENTS.md` 和 `.codex/config.toml` 中写入固定 SSH 运行命令，使 X 盘负责修改服务器文件，SSH 负责使用服务器 GPU 运行代码，并固定使用服务器中的 `tslib` 环境。

## 已完成

- 更新 X 盘项目根目录的 `AGENTS.md`：
  - 写入固定 SSH 主机、用户、远程项目目录、环境 Python 和 GPU 参数。
  - 写入实验开始前的 GPU/CUDA/Python 检查命令。
  - 写入 Conda 激活和直接调用环境 Python 的远程运行命令。
  - 明确禁止在本机 Python/GPU 上运行或回退。
- 更新 X 盘项目的 `.codex/config.toml`：
  - 写入远程连接参数、`tslib`、GPU `0,1`、检查命令和运行命令模板。
- 修正了 SSH 远程 shell 与 Python 字符串之间的引号转义问题。
- 临时文件已清理，原有项目文件未覆盖。

## 验证

- 命令模板执行成功。
- 远程 Python 来自 `tslib` 环境。
- `torch.cuda.is_available()` 为 True，设备数为 2。
- 两张 NVIDIA GeForce RTX 4090 D 可见。
- TOML 配置解析通过。

## 安全处理

本次 X 盘本地配置包含连接参数，但未包含密码；连接参数未同步到公共 GitHub 仓库或回复归档。
