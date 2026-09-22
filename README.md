# GPT ↔ Codex Experiment Bridge

这个仓库用于 **ChatGPT 规划实验 → Codex 在 Time-Series-Library 中执行 → Codex 将证据与结果回写**。

当前研究主题：**Scale-Aware EMA for TimeMixer**。

## 工作规则

1. Codex 每次开始前先读取 `CURRENT_STAGE.md`。
2. **一次只执行一个阶段**，不得自动跳到下一阶段。
3. 具体任务只以 `prompts/<阶段>.md` 为准。
4. Time-Series-Library 的代码修改应在独立研究分支进行，不能破坏已经复现成功的 baseline。
5. 每个阶段结束后，Codex 必须把完整结果写入本仓库指定的 `results/*.md`。
6. 报告必须记录：TSLib commit、分支、代码改动、实际命令、环境、关键日志、指标、失败尝试及原因。
7. 大量原始训练日志不要直接塞进报告；报告写日志路径和关键片段即可。
8. 如果阶段失败，**停止**，把失败原因和建议写入结果文件，不得擅自进入下一阶段。
9. 除非 prompt 明确要求，不要改变 baseline 超参数、数据划分、评价指标或训练流程。
10. 禁止为了获得更好 test 指标而使用 test set 调参；超参数选择必须基于 validation。

## 目录

- `CURRENT_STAGE.md`：当前允许执行的唯一阶段。
- `prompts/`：ChatGPT 写给 Codex 的阶段任务。
- `results/`：Codex 的阶段回复、实验数据和结论。

## 当前方法边界

第一阶段研究只做：

**TimeMixer + Scale-Aware EMA decomposition**

暂不做：
- instance-adaptive EMA；
- channel-adaptive EMA；
- 动态 `alpha_t`；
- CCO 等超参数优化器；
- 其他额外网络结构。

目的是先验证：**“EMA 本身”与“scale-aware EMA 系数”到底分别贡献多少。**
