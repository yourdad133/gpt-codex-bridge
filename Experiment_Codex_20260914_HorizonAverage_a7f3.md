# Weather / Electricity 四 horizon 平均结果

日期：2026-09-14

本次请求按 TimeMixer 的四个预测步长 96、192、336、720 对 Weather 和 Electricity 的 MSE、MAE 做等权算术平均。

结果：

| Benchmark | Average MSE | Average MAE |
|---|---:|---:|
| Weather | 0.245411538 | 0.275632892 |
| Electricity | 0.186691631 | 0.278634407 |

计算方式：四个 horizon 的对应指标之和除以 4。原始结果来自本轮 Local Dynamic Patch 实验日志。