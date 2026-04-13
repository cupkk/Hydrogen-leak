# 导师汇报结果包 2026-04-11

本目录保留的是 2026-04-11 阶段的旧版结果摘要，主要对应早期 `cfd56`/过渡阶段实验。它现在只作为历史归档，不再作为正式汇报或报告中的主结果来源。

## 当前状态

- 保留：JSON / CSV 结果摘要文件，便于回溯早期实验结论。
- 删除：旧版原始绘图子目录 `plots_sensor`、`plots_scale_none`、`plots_scale_focus`、`plots_scale_balanced`。
- 原因：这些图片已经被 clean48 阶段的正式图替代，且不再被当前报告引用，继续保留只会增加混淆。

## 当前应优先使用的正式材料

- 正式报告图：[`docs/report_figures_20260413`](../report_figures_20260413)
- 过渡但仍在用的图：[`docs/report_figures_20260412`](../report_figures_20260412)
- 服务器同步下来的 clean48 正式结果：[`docs/server_results_20260413`](../server_results_20260413)

## 本目录保留文件

- [`holdout400_test_aggregate_metrics.json`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/holdout400_test_aggregate_metrics.json)
- [`sensor_condition_study.csv`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/sensor_condition_study.csv)
- [`sensor_condition_study.json`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/sensor_condition_study.json)
- [`sensor_param_baseline.json`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/sensor_param_baseline.json)
- [`train_scale_none_summary.csv`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/train_scale_none_summary.csv)
- [`train_scale_lowflow_focus_summary.csv`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/train_scale_lowflow_focus_summary.csv)
- [`train_scale_lowflow_balanced_summary.csv`](/d:/github/Hydrogen-leak/docs/advisor_results_20260411/train_scale_lowflow_balanced_summary.csv)

## 使用建议

如果是给导师汇报或继续写报告，不要再直接引用本目录中的旧结论图。当前统一以 clean48 结果为准。
