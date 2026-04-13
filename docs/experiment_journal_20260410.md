## 2026-04-10 训练进度核查

截至 2026-04-10 上午，`56-case` 正式基线主链已经完成，训练数据量多次重复实验仍在继续，传感器数量/观测时长实验已修正配置后重新启动。

当前服务器状态如下：

- `GPU1~GPU3`：分别运行三种 `sample_weight_mode`（`none / lowflow_focus_v1 / lowflow_balanced_v1`）下的训练规模重复实验。
- `GPU0`：已从正式基线主线释放出来，用于补跑传感器条件实验。

从三路重复实验日志和 checkpoint 落盘情况看，三条链路都已经完成：

- `n=6` 的 `r00 / r01 / r02`
- `n=12` 的 `r00 / r01 / r02`

并进入：

- `n=24, r00`

的 `GPSD` 阶段，当前推进到约 `55%~58%`。这表明三条训练规模实验均超过半程，但距离全部 `12` 组任务完成仍有明显剩余。

## 正式基线当前结果

正式基线结果已经落盘在：

- `exps/gp-edm_holdout400_cfd56_cfd56_holdout400_val0200_train_20260409-2213/val_eval/aggregate_metrics.json`
- `exps/gp-edm_holdout400_cfd56_cfd56_holdout400_val0200_train_20260409-2213/test_eval/aggregate_metrics.json`
- `results/cfd56_holdout400_val0200_sensor_param_baseline/sensor_param_baseline.json`

其中 `test_eval` 已完整生成，说明：

- `56-case` 正式基线模型训练
- `holdout test` 全场反演评估
- `sensor` 源参数基线

这一主链已经跑通。

当前 `test_eval` 平均指标为：

- `global_rmse = 0.005997`
- `global_mae = 0.004019`
- `global_rel_l1_active_mean = 29.463`
- `global_rel_l2 = 19.190`
- `mass_mean_rel_error = 6.553`

低泄漏率 `50/100 mL/min` 子集仍明显更差：

- `global_rel_l2 = 48.918`
- `global_rel_l1_active_mean = 70.005`
- `mass_mean_rel_error = 16.761`

源参数 `sensor` 基线在 `test` 上当前为：

- 源位置平均误差 `214.29 mm`
- 泄漏率 MAE `34.94 mL/min`
- 泄漏率平均相对误差 `10.19%`

## 当前异常与处理

本日上午发现两个关键情况：

1. `val_eval` 的 7 个 case 指标完全相同，异常明显，暂不宜直接作为正式结论使用，后续需优先检查 `val` 的 sample index 或聚合逻辑。
2. 传感器条件实验原先误用了 `data/sensors_real_12.csv`，却要求生成 `30` 个传感器子集，导致报错。现已改为使用 `data/sensors_real.csv`（33 点）并重新启动，当前正在运行。

## 当前判断

当前训练仍在正常进行，且不需要人为中断：

- 三张卡上的训练规模重复实验仍在持续推进
- `GPU0` 上的传感器条件实验已经重新接上

若中途不再出现新的配置错误，训练规模三路实验预计还需要约 `12~16` 小时。
