# TimeMixerAdaptivePatch ETTm1 Subtask 5 Codex Reply Archive

日期：2026-09-12

## 请求摘要

读取 X 盘项目约定与 GitHub 上的 `TimeMixer_AdaptivePatch_Subtask5.md`，完成 TimeMixerAdaptivePatch 的 ETTm1 最终工程验收、A-G 消融实验入口准备和正式 benchmark 前的 sanity checks。

## 最终结果

结论：YES。当前 TimeMixerAdaptivePatch 已达到可以开始正式 ETTm1 baseline、full model 和 ablation benchmark 的工程状态。

本阶段没有启动四个预测长度的完整 benchmark，也没有进行大规模超参数搜索、Git index 修复、commit 或 push。

## 改动文件

新增：

- `TimeMixerAdaptivePatch_NOTES.md`
- `scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh`
- `scripts/long_term_forecast/ETT_script/validate_TimeMixerAdaptivePatch.py`

核心模型和 baseline 文件在本阶段未修改。

## 运行验证

远程 tslib/CUDA 环境验证通过：

- Python compile/import：PASS
- TimeMixer construction、forward、optimizer step：PASS
- Adaptive forward：pred_len 96/192/336/720 全部 PASS
- seq_len=192（尺度 192/96/48/24）：PASS
- patch edge cases、uniform routing、learned routing：PASS
- patch/scale probability normalization、invalid patch probability=0：PASS
- patch router / scale router 参数更新：PASS
- auxiliary loss、zero-balance、AMP、non-AMP：PASS
- two-GPU DataParallel：PASS
- beta=0 baseline degeneration：PASS，max abs diff=0
- A-G variant construction/forward：PASS
- shell syntax 和 CLI parse：PASS
- real ETTm1 batch forward/router example：PASS

当前参数统计：

- TimeMixer：75,513
- TimeMixerAdaptivePatch：98,577
- 增量：23,064（30.5431%）
- Adaptive Patch：18,452
- Scale Router：4,612

工程 forward benchmark（batch=2、seq_len=96、pred_len=96、float32）：

- TimeMixer：5.3122 ms
- Full Joint Router：9.2749 ms
- 相对 slowdown：1.7460x
- FLOPs：未测量

## 正式入口

推荐第一条正式命令：

```bash
EXPERIMENT=baseline SEQ_LEN=96 PRED_LEN=96 bash scripts/long_term_forecast/ETT_script/TimeMixerAdaptivePatch_ETTm1.sh
```

该脚本通过 `EXPERIMENT` 支持 baseline、uniform、patch_only、scale_only、joint、fixed4、fixed8、joint_nobalance；通过 `PRED_LEN` 支持 96/192/336/720。

## 参考

[TimeMixerAdaptivePatch Subtask 5](https://github.com/yourdad133/gpt-codex-bridge/blob/main/TimeMixer_AdaptivePatch_Subtask5.md)
