# Codex Results

Codex 每完成一个阶段，在此目录写对应结果。

推荐报告结构：

1. Status: PASS / FAIL / BLOCKED
2. TSLib workspace / branch / commit
3. 本阶段实际做了什么
4. 修改文件
5. 实际执行命令
6. 测试或训练结果
7. 关键指标
8. 失败与修复
9. 与 prompt 的偏差（若有）
10. 下一阶段前置条件
11. 最终结论

涉及数值实验时，除 Markdown 汇总外，尽量同时保存机器可读 CSV。
