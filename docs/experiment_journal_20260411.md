# 2026-04-11 实验日志

## 1. 本次核查目标

本次工作不是重新启动训练，而是对当前 56-case 正式实验结果做一次完整核查，明确三件事：

1. 导师要求的“未见位置闭环验证”是否已经形成可汇报结果。
2. 量化评价指标是否已经固定，能否直接用于报告与答辩。
3. “模型能力 / 模型成本 / 泛化能力”三条实验线中，哪些已经完成，哪些仍然缺口明显。

当前本地工作目录为 `d:\github\Hydrogen-leak`，正式实验仓库为 [SDIFT模型](/d:/github/Hydrogen-leak/SDIFT模型)。远端核查目录为 `/hy-tmp/SDIFT_model56`。

## 2. 已确认存在的正式结果

远端已经存在以下关键产物，说明正式 56-case 实验链路不是停留在脚本层面，而是已经生成了可分析结果：

- `exps/gp-edm_holdout400_cfd56_cfd56_holdout400_val0200_train_20260409-2213/test_eval/aggregate_metrics.json`
- `results/advisor_study/sensor_conditions_cfd56_holdout400/sensor_condition_study.csv`
- `results/advisor_study/sensor_conditions_cfd56_holdout400/sensor_condition_study.json`
- `results/advisor_study/train_scale_none_cfd56/training_scale_repeated_summary.csv`
- `results/advisor_study/train_scale_lowflow_focus_cfd56/training_scale_repeated_summary.csv`
- `results/advisor_study/train_scale_lowflow_balanced_cfd56/training_scale_repeated_summary.csv`
- `results/cfd56_holdout400_val0200_sensor_param_baseline/sensor_param_baseline.json`

与此同时，远端当前没有持续运行中的训练进程，GPU 处于空闲状态。因此本轮分析基于“已有完成结果”，而不是“训练尚未结束的中间状态”。

## 3. 导师任务 1：未见位置闭环验证

这一项已经具备正式结果，而且数据分层方式与导师要求一致。

当前 56-case 划分为：

- train: 42 组
- val: 7 组，对应未见位置 `(200, 0, 0)`
- test: 7 组，对应未见位置 `(400, 0, 0)`

也就是说，测试集全部来自训练中未出现过的泄漏源位置，满足“用没训练过的位置 CFD 结果抽样传感器，再做全场反演”的要求。对应 split 文件位于 [split.json](/d:/github/Hydrogen-leak/SDIFT模型/data/splits/holdout_400_0_0_val_0200/split.json)。

当前正式 holdout test 的聚合结果为：

- 测试样本数：7
- 全局 RMSE：0.005997
- 全局 MAE：0.004019
- 全局相对 L2 误差：19.1905
- 活跃区相对 L1 平均误差：29.4631
- 质量守恒相对误差均值：6.5529

其中 test 集 7 个样本全部是 `(400, 0, 0)` 这一未见位置下不同泄漏率工况：

- 50
- 100
- 200
- 400
- 600
- 800
- 1000 mL/min

这意味着当前已经能够给导师展示一个完整的闭环：

CFD 真值全场 -> 抽样成传感器时序 -> 模型反演全场 -> 与原始 CFD 全场逐点比较

因此，从“有没有一个可控真值下的可验证结果”这个角度，答案已经是有。

## 4. 导师任务 2：量化评价指标

这一项也已经基本完成，而且代码里定义是清晰的，不是口头描述。指标定义位于 [evaluate_reconstruction.py](/d:/github/Hydrogen-leak/SDIFT模型/evaluate_reconstruction.py)。

当前已经实现并可直接汇报的指标包括：

- `global_rmse`：全时空整体 RMSE
- `global_mae`：全时空整体 MAE
- `global_rel_l1_mean`：全时空逐点相对误差平均值
- `global_rel_l1_active_mean`：仅在真值高于阈值的活跃区域统计相对误差
- `global_rel_l2`：整体相对 L2 误差
- `mass.mean_rel_error`：总质量随时间序列的相对误差均值
- `per_time_rmse`
- `per_time_mae`
- `per_time_rel_l1`
- `per_time_rel_l1_active`

其中，导师提到的“每个时刻逐点做差，再按相对值做平均，形成随时间变化的误差指标”，在当前实现中已经直接对应：

- `per_time_rel_l1`
- `per_time_rel_l1_active`

这两个时间曲线就是后续报告里最应该重点展示的时变误差指标。建议正式汇报时采用以下组合：

- 主指标：`global_rel_l2`
- 辅指标：`global_rmse`、`global_mae`
- 重点图：`per_time_rel_l1_active` 随时间变化曲线
- 物理一致性指标：`mass.mean_rel_error`

原因是：

- `global_rmse` 和 `global_mae` 便于和其他模型比较；
- `global_rel_l2` 能压缩整个时空场误差，适合做总体排序；
- `per_time_rel_l1_active` 最贴合导师“逐时刻相对误差”的要求；
- `mass.mean_rel_error` 能说明结果是否至少保住了总体量级。

## 5. 当前结果的主要优点与主要短板

当前结果最大的优点，不是绝对误差数值本身，而是验证链路已经成立。也就是说，现在已经不是“只能画图”，而是已经有严格的真值回比。

但也必须明确短板：

### 5.1 低泄漏率仍然是主要薄弱点

在 50/100 mL/min 低泄漏率子集上，误差显著高于整体均值。当前 holdout test 中：

- 低泄漏率子集相对 L2 误差均值：48.9183
- 低泄漏率子集活跃区相对 L1 平均误差：70.0049

对比整体：

- 整体相对 L2 误差均值：19.1905
- 整体活跃区相对 L1 平均误差：29.4631

这说明当前模型在强泄漏工况上已经有可用趋势，但在低泄漏、小浓度工况下仍然明显不稳。导师如果追问“最难工况表现如何”，这一点不能回避。

### 5.2 传感器数量 / 观测时长实验已经做完，但趋势还不够干净

远端 `sensor_condition_study.csv` 已经存在 9 组结果，即：

- 传感器数：6 / 12 / 30
- 观测时长：20 / 60 / 120 步

但目前结果不是单调改善，甚至出现“传感器更多、观测更长，指标反而更差”的现象。例如：

- 30 传感器 + 20 步：相对 L2 误差 16.90
- 30 传感器 + 120 步：相对 L2 误差 24.65

这类现象说明当前实验已经有数据，但还不能直接解释为“更多传感器无用”或“更长观测有害”。更合理的判断是：当前观测注入机制、观测权重和采样过程之间仍存在耦合，导致该曲线可以作为阶段性结果，但还不适合作为最终科学结论。

### 5.3 训练数据量实验已经做成重复分层版本，但方差较大

这一项已经不是单次抽样，而是每个规模做了 3 次重复，并分成：

- `none`
- `lowflow_focus_v1`
- `lowflow_balanced_v1`

目前从结果看，`lowflow_balanced_v1` 是三者中最稳的一条线，小样本规模下尤其明显更优。例如：

- size 6：相对 L2 误差 2.92
- size 12：相对 L2 误差 2.86
- size 24：相对 L2 误差 13.36
- size 42：相对 L2 误差 9.14

但这条曲线仍然不是严格单调下降，说明当前关系不能简单概括成“近线性提升”或“边际收益快速递减”，更准确的说法应当是：

在当前数据分布和训练配置下，性能对训练集规模较敏感，但受样本构成和低流量样本权重影响明显，呈现出高方差、非单调特征。

这句话适合写进报告，比强行说“是线性的”更稳妥。

### 5.4 尺寸泛化目前没有数据支撑

这点已经通过脚本检查确认。当前 56-case manifest 中没有 `space_size_x_m / space_size_y_m / space_size_z_m` 等尺寸字段，也没有至少两种不同空间尺寸。对应检查脚本为 [assess_size_generalization_readiness.py](/d:/github/Hydrogen-leak/SDIFT模型/assess_size_generalization_readiness.py)，本地检查输出为 [size_readiness_cfd56.json](/d:/github/Hydrogen-leak/tmp/size_readiness_cfd56.json)。

结论非常明确：

当前数据还不支持“尺寸稍大或稍小场景迁移”的正式实验。

所以导师第三项任务目前只能给出规划，不能给出实证结果。

## 6. 源位置与泄漏率估计的当前位置

这一条线已经有独立 baseline，不必再完全依赖峰值搜索。当前远端 `sensor_param_baseline.json` 显示：

- 源位置平均误差：214.29 mm
- 泄漏率 MAE：34.94 mL/min
- 泄漏率相对误差均值：10.19%

这说明“源位置 + 泄漏率”已经不是完全空白，而是有一个可以汇报的轻量回归 baseline。但这个 baseline 仍然更适合作为辅助结果，而不是主结果，原因有两点：

1. 导师当前最关心的是浓度场反演能否在真值下被验证。
2. 源位置平均误差仍然偏大，离“高精度定位”还有差距。

因此，现阶段最合理的表达是：

浓度场反演主线已经形成正式闭环验证；源参数估计已有可训练 baseline，但仍处于辅助验证与后续强化阶段。

## 7. 当前最合理的导师汇报口径

如果按“先证明有效 -> 再证明值得做 -> 再证明能推广”这条科研逻辑来组织，本阶段最稳妥的口径如下。

第一步，先证明有效：

已经完成。可以用 56-case 中未见位置 `(400,0,0)` 的 7 个测试工况作为正式验证集，展示从 CFD 真值抽样传感器，到模型反演，再回到全场真值比较的完整闭环，并报告 RMSE / MAE / 相对 L2 / 活跃区逐时刻相对误差 / 质量误差。

第二步，再证明值得做：

已经部分完成。传感器数量 / 观测时长实验和训练数据量重复实验都已经跑出结果，但目前趋势仍受观测机制与样本构成影响，结论应写成“已形成阶段性趋势与敏感性分析”，不宜夸大为最终定律。

第三步，再证明能推广：

尚未完成。当前缺少不同空间尺寸的数据集，因此无法对尺寸迁移给出严谨结论。下一批 CFD 工况如果要服务导师第三项任务，优先级应该是补不同箱体尺寸，而不是继续只在同一尺寸里加位置。

## 8. 下一步任务建议

下一步不建议盲目继续大规模训练，而应按以下顺序推进。

### 8.1 先把“已经完成的导师任务”整理成正式图表

这是最高优先级。原因是结果已经有了，但还没有被压缩成最适合汇报和写报告的图表包。建议直接产出：

- 未见位置 holdout 的总表
- 7 个测试 case 的逐 case 指标表
- `per_time_rel_l1_active` 时间曲线
- 传感器数量 / 观测时长曲线
- 训练数据量 `mean ± std` 曲线
- 低泄漏率子集单独结果表

### 8.2 然后针对低泄漏率补强

当前最明显短板就在 50/100 mL/min。下一轮真正值得花算力做的，不是全量无差别重训，而是：

- 优先补 50/100 mL/min 的缺失位置数据
- 保留 `lowflow_balanced_v1` 作为当前首选加权策略
- 所有结果同时汇报整体指标和低泄漏率子集指标

### 8.3 最后再进入尺寸泛化

只有在拿到至少两种不同空间尺寸，并把尺寸字段写进 manifest 以后，才能正式启动导师第三项任务。否则现在做“泛化能力”只能停留在口头方案。

## 9. 本次结论

截至 2026-04-11，可以下一个清晰结论：

1. 导师要求的“未见位置 CFD -> 传感器抽样 -> 模型反演 -> 回比真值”闭环已经完成，且有正式量化结果。
2. 量化指标体系已经落地，尤其是逐时刻相对误差曲线已经具备实现与输出能力。
3. 模型能力与模型成本两条实验线已经产出结果，但当前仍应作为阶段性分析，不宜过度解释。
4. 尺寸泛化尚未开始，不是方法失败，而是数据条件尚不具备。
5. 当前最该做的不是重新从零训练，而是把已有结果整理成导师可直接审核的图表与表格，同时集中补低泄漏率和尺寸变化数据。

## 10. 56-case 坐标元数据纠偏

本日进一步核对 `F:\AI反演_CFD计算` 批次的原始 case 命名后，确认此前对五字段目录名的 `x/y` 解析存在反置。典型例子是 `6,0,100,0,50` 与 `6,0,200,0,1000`，此前被写成 `(0,100,0)` 与 `(0,200,0)`，但结合用户确认和原始命名语义，正确坐标应为 `(100,0,0)` 与 `(200,0,0)`。为避免后续继续写错，已在 [build_cfd_multicase_dataset.py](/d:/github/Hydrogen-leak/SDIFT模型/build_cfd_multicase_dataset.py) 中把五字段命名规则正式修正为 `6,y,x,z,rate`。

与此同时，已同步修正本地权威元数据文件，包括：

- [cfd56_all_T120_interp48_manifest.csv](/d:/github/Hydrogen-leak/SDIFT模型/data/cfd56_all_T120_interp48_manifest.csv)
- [cfd49_all_T120_interp48_manifest.csv](/d:/github/Hydrogen-leak/SDIFT模型/data/cfd49_all_T120_interp48_manifest.csv)
- `cfd11_newbatch_*_manifest.csv`
- [split.json](/d:/github/Hydrogen-leak/SDIFT模型/data/splits/holdout_400_0_0_val_0200/split.json)
- [train_manifest.csv](/d:/github/Hydrogen-leak/SDIFT模型/data/splits/holdout_400_0_0_val_0200/train_manifest.csv)
- [val_manifest.csv](/d:/github/Hydrogen-leak/SDIFT模型/data/splits/holdout_400_0_0_val_0200/val_manifest.csv)

纠偏后，56-case 数据池的实际覆盖更整齐：共有 `8` 个泄漏位置，并且每个位置都已覆盖 `50 / 100 / 200 / 400 / 600 / 800 / 1000 mL/min` 七个泄漏率等级，不再存在“`(100,0,0)` 只有 3 组、`(0,100,0)` 只有 4 组”的假象。当前正式划分也更清晰：`train` 为 6 个完整位置、`val = (200,0,0)`、`test = (400,0,0)`。

这次纠偏对已有实验的影响需要分开看。对于主线浓度场重建，`HDF5` 张量本身未改，`train / val / test` 的 case ID 归属也未改，因此已经完成的 `FTM + GPSD + holdout_400` 重建结果在数值上仍然有效，变化仅限于位置标签解释。尤其正式 test 集仍然是 `(400,0,0)` 未见位置，因此“未见位置闭环验证”这一主结论不受影响。另一方面，凡是直接使用坐标标签做监督或分层抽样的实验，理论上都受到影响，最典型的是源位置/泄漏率回归器以及按位置分层的训练数据量重复实验。这意味着后续若要继续推进源参数估计和 train-size 曲线，应基于修正后的 manifest 重新生成结果，而不应继续沿用旧标签版本。
