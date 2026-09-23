# TimeMixer 接入 GTR

- 日期：2026-09-23
- 请求：参考 [macovaseas/GTR](https://github.com/macovaseas/GTR)，将 GTR 作为 TimeMixer 输入增强模块；只做输入级 GTR，不做多尺度 GTR。
- 结果：新增 `TimeMixerGTR`，在 TimeMixer 多尺度下采样前检索可学习的全局周期，并用可学习 sigmoid 门控进行残差融合；后续沿用 TimeMixer 的下采样、PDM、FMM 和预测路径。原 `TimeMixer` 基线保持独立。
- 周期相位：从 `x_mark_enc` 读取时间特征；新增周期长度、GTR 局部卷积宽度、门控初始值和 dropout 参数，并在中文 README 中记录配置示例。
- 改动文件：`layers/GTR.py`、`models/TimeMixerGTR.py`、`run.py`、`README_zh.md`、`third_party/GTR_LICENSE`。
- 验证：完成代码结构与配置接入的静态检查；未运行训练、评估或测试。
