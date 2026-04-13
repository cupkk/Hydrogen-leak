# Experiment Journal 20260408

## 记录范围

本日志用于补录截至 `2026-04-08` 的主要工程进展和实验活动。由于此前没有严格按日沉淀日志，本文件同时承担“历史补录 + 今日记录”双重作用。后续建议严格按天新增新文件，不再在本文件中混杂新日期内容。

## 一、项目状态基线

截至本日志写入时，项目主线已经从“方法思路验证”推进到“有正式训练集、正式 holdout 测试集、正式定量指标、正式源参数基线”的阶段。当前主要资产包括：

- 16 组早期工况张量数据
- 22 组新增工况张量数据
- 合并后的 38 组正式数据集
- 一套固定的 `holdout_400_0_0` 未见位置测试划分
- 远端完成训练的 `FTM + GPSD` 主模型
- 已落地的 `sensor` 源参数回归器基线

当前仓库的主要工作重心已经不再是“能不能跑通”，而是：

- 观测约束是否足够强
- 指标是否能回答导师提出的三个问题
- 低泄漏率工况是否被整体平均掩盖

## 二、历史补录

### 2026-04-02：当前仓库单工况闭环跑通

在当前主仓库上补齐了原始 CFD 数据接入、传感器生成、FTM/GPSD/MPDPS 三阶段脚本，并在单 case 上跑出第一版完整三维浓度场重建结果。该阶段的意义是确认当前仓库不是只停留在论文 demo，而是能够开始接氢泄漏数据。

该阶段主要结论：

- 当前仓库能够从稀疏测点恢复 `48 x 48 x 48` 三维场
- 结果可导出为 `.mat` 和 ParaView 时序可视化
- 但此时还没有多工况、未见位置测试和正式量化评估

### 2026-04-04：16 组数据转换与 smoke train

完成前 16 组 CFD 数据的统一张量化，形成：

- `cfd16_all_T120_interp48.h5`
- `cfd16_all_T120_interp48_meta.npy`
- `cfd16_all_T120_interp48_manifest.csv`
- `cfd16_all_T120_interp48_report.json`

完成内容：

- 原始 ASCII -> 规则 3D 网格
- 固定 120 步时间长度
- `48^3` 网格
- `IDW(k=8,p=2)` 插值
- FTM smoke train
- GPSD smoke train

在此阶段发现并处理了一个重要问题：

- `case_0015` 的标签误写为 `(200,-300,0,200)`，后续确认应修正为 `(200,-150,0,200)`，该元数据问题后来已统一修正

### 2026-04-07：22 组新增数据转换、38 组合并、正式 holdout 训练

新增数据来自 `E:` 和 `F:` 的 22 组工况。完成内容：

- 新 22 组张量化
- 老 16 组与新 22 组合并成 `38` 组正式数据集
- 生成 `holdout_400_0_0` 的训练/测试划分
- 远端服务器完成 `FTM` 和 `GPSD` 正式训练
- 完成未见位置 `(400,0,0)` 的 7 组 holdout 测试

正式 holdout 基线平均结果为：

- `global_rmse = 0.002540`
- `global_mae = 0.000700`
- `global_rel_l1_active_mean = 4.434`
- `global_rel_l2 = 8.417`
- `mass_mean_rel_error = 4.430`

测试集 7 个 case 为：

- `Q50-X400-Y0-Fraction`
- `Q100-X400-Y0-Fraction`
- `Q200-X400-Y0-Fraction`
- `Q400-X400-Y0-Fraction`
- `Q600-X400-Y0-Fraction`
- `Q800-X400-Y0-Fraction`
- `Q1000-X400-Y0-Fraction`

这一阶段的核心意义是：第一次形成了可向导师展示的“真值可控、位置未见、全场回归、逐点量化”的正式验证。

### 2026-04-07：源参数模型比较

在 `sensor / core / hybrid` 三种源参数估计模式上完成了正式比较。当前最佳方案是 `sensor`：

- 源位置平均误差 `160.37 mm`
- 泄漏率平均相对误差 `1.71%`
- 泄漏率 MAE `3.16 mL/min`

而 `hybrid` 和 `core` 显著更差。当前判断是：

- 小样本阶段，传感器统计特征回归比 `FTM core` 或简单拼接 `hybrid` 更稳
- 短期内应冻结 `sensor` 为正式基线
- “峰值搜索”不再作为主结果，只能作为早期启发式参考

## 三、2026-04-08 当日记录

### 1. 对导师要求重新梳理实验逻辑

导师要求的三条主线被重新拆解为：

- 模型能力：传感器数量、观测时长等输入条件是否影响精度
- 模型成本：训练数据量增加后，精度关系是线性、递减还是其他模式
- 泛化能力：是否能从一个空间迁移到尺寸稍有变化的空间

重新评估后发现：

- 当前仓库已具备前两条实验的代码基础
- 第三条“尺寸泛化”暂时还没有数据支撑，因为当前 manifest 本质仍是同一箱体尺寸

### 2. 低泄漏率问题被正式确认

从 holdout 结果中单独抽取 `50/100 mL·min⁻¹` 子集，发现整体平均明显掩盖了低流量工况的困难。低流量子集指标显著劣化，尤其是：

- `global_rel_l1_active_mean`
- `global_rel_l2`
- `mass_mean_rel_error`

基于这一结果，代码层已加入加权训练入口，准备比较：

- `none`
- `lowflow_focus_v1`
- `lowflow_balanced_v1`

这一步的结论已经足够明确，不再只是猜测。

### 3. “观测是否真的起作用”的 sanity check

为避免错误地解释“传感器数量/时长影响”，新增了观测 sanity check，额外跑：

- 正确传感器输入
- 打乱的传感器输入
- 错位或全零传感器输入

已经拿到的前几组结果表明：

- `correct` 与 `shuffled` 几乎重合
- 差异小到不足以支撑“传感器变化显著影响精度”的结论

因此当前得出关键判断：

- 当前 `MPDPS` 的观测约束偏弱
- 在先调强观测权重之前，不能直接把“传感器数量/时长曲线”当成可信结论

### 4. 已开始的修正动作

为解决上述问题，今天补齐并启用了以下脚本：

- `run_observation_sanity_check.py`
- `run_mpdps_observation_tuning.py`
- `build_repeated_train_size_subsets.py`
- `run_training_scale_repeated_study.py`
- `run_sensor_param_baseline.py`

这意味着实验策略发生了调整：

- 先做 `MPDPS` 调参，让观测真正起作用
- 再做传感器数量/观测时长影响实验
- 再做多次重复的训练规模实验

### 5. 远端运行状态

当前远端主机：

- `ssh -p 63260 root@i-1.gpushare.com`

远端当前最重要的活跃任务是：

- `MPDPS` 观测调参

远端重点结果目录包括：

- `/root/Hydrogen-leak/SDIFT_model/results/advisor_study/mpdps_tuning_holdout400`
- `/root/Hydrogen-leak/SDIFT_model/results/source_param_sensor_baseline_holdout400`

当前仍未完成的不是主模型训练，而是“导师要求实验”的第二阶段验证。

### 6. 远端磁盘打满与处理

本次继续推进时，发现远端根盘一度达到约 `99.96%`，`MPDPS` 调参在保存 `.mat` 重建结果时中断，报错为：

- `OSError: [Errno 28] No space left on device`

此次问题的本质不是模型数值发散，而是实验产物管理失控。占用空间最大的对象包括：

- 完整训练数据打包文件 `hydrogen_leak_train_bundle_20260407.tar.gz`
- 旧版单次训练规模实验生成的子集 `train_scale_lowflow_focus/subsets`
- 已过时的一轮 `advisor_scale_lowflow_gpsd_*` 训练目录

处理措施如下：

- 删除远端冗余打包文件
- 删除已被新实验方案替代的旧训练规模实验目录
- 删除旧的临时子集目录
- 保留正式基线模型、正式 holdout 结果和当前参数基线结果

清理后远端根盘恢复到约：

- `Used = 22G / 30G`
- `Avail = 9.0G`
- `Use% = 71%`

### 7. 为避免再次打满盘所做的代码修正

本次不只做了清理，还修正了实验脚本本身的磁盘占用策略：

- `run_holdout_reconstruction_eval.py`
  - 新增 `--cleanup_recon_mat`
  - 新增 `--cleanup_sensor_cache`
  - 即：评估完成后自动删除中间 `.mat`、`_summary.json`、`_source_est.npy` 和临时传感器缓存

- `run_observation_sanity_check.py`
  - 新增 `--skip_existing_modes`
  - 支持在某个 mode 已完成时跳过，避免重跑

- `run_mpdps_observation_tuning.py`
  - 新增 `--skip_existing_tags`
  - 支持在某个参数组合已完成时跳过

因此，本次恢复之后的 `MPDPS` 调参不再采用“所有中间重建结果都长期落盘”的方式，而是改为“只保留评估结果和汇总表”。

### 8. 本次恢复后的当前状态

恢复后，`MPDPS` 调参已经重新启动，当前采用：

- 自动清理中间重建结果
- 自动清理传感器缓存
- 跳过已完成参数组合

也就是说，这次恢复不是简单重跑，而是带着实验管理改进继续推进。

### 9. 远端数据盘情况

远端当前额外可用数据盘挂载点为：

- `/hy-tmp`

容量为：

- `100G available`

当前根盘状态约为：

- `30G total`
- `21G used`
- `9.3G available`
- `70% used`

结论是：

- 当前这轮 `MPDPS` 调参在清理冗余文件后已经足够继续跑完，不必因空间问题立刻停机。
- 后续新增的 14 组工况、合并后的新训练集、以及下一轮正式训练产物，优先应放到 `/hy-tmp`，而不是继续全部堆在根盘 overlay 上。

## 四、当前明确结论

### 1. 已经确定成立的结论

- 38 组数据上的 `FTM + GPSD + MPDPS` 主链路已完成训练和未见位置验证。
- 三维浓度场反演已经不是概念验证，而是具备正式定量结果。
- `sensor` 回归器是当前正式的源参数估计基线。
- 低泄漏率工况是现阶段最薄弱环节。

### 2. 已经确定不应再继续沿用的做法

- 不能再把“峰值搜索”作为主源定位结果。
- 不能只报整体平均指标。
- 不能在 `MPDPS` 观测约束还没调强时，直接解释传感器数量和时长对性能的影响。
- 不能在缺少不同箱体尺寸数据时，提前声称已经验证尺寸泛化。

## 五、下一步任务清单

当前优先级应固定为：

1. 完成 `MPDPS` 调参，使 `correct` 明显优于 `shuffled`
2. 固定最佳观测参数后，重做传感器数量 / 观测时长影响实验
3. 固定测试集，做 `6/12/24/31` 的多次重复分层子集实验，并报告 `mean ± std`
4. 比较 `none / lowflow_focus_v1 / lowflow_balanced_v1`
5. 继续补充 `50/100 mL·min⁻¹` 且覆盖缺失位置的 CFD 数据
6. 下一批 CFD 优先补不同箱体尺寸数据，为尺寸泛化实验做准备

## 六、后续日志维护约定

从本文件开始，后续建议按日新增：

- `docs/experiment_journal_20260409.md`
- `docs/experiment_journal_20260410.md`

每日日志至少应包含：

- 当天修改了哪些脚本
- 当天跑了哪些实验
- 结果文件在哪里
- 得到了什么结论
- 哪些问题还没解决

## 七、2026-04-08 当前远端核查补充

本次补查时，远端主机 `ssh -p 63260 root@i-1.gpushare.com` 上的状态如下。

### 1. 主模型训练是否完成

主模型训练已经完成，不再需要继续训练主干模型。这里的“主模型”指：

- `FTM` 训练
- `GPSD` 训练
- 基于 `holdout_400_0_0` 的正式全场反演评估
- `sensor / core / hybrid` 三种源参数模型比较

因此，当前不缺“主模型是否收敛”的答案，缺的是“导师要求的扩展实验是否可信、是否完成”。

### 2. 当前仍在运行的任务

当前正在运行的是 `MPDPS` 观测调参，而不是主模型训练。进程包括：

- `run_mpdps_observation_tuning.py`
- 其内部调用的 `run_observation_sanity_check.py`
- 以及子进程 `run_holdout_reconstruction_eval.py`

本次核查时，正在跑的配置是：

- `mpdps_weight = 1.0`
- `obs_rho = 0.01`
- `total_steps = 20`
- `posterior_samples = 1`
- 模式：`correct` 与 `shuffled`

### 3. 已完成结果的再确认

当前正式 holdout 基线结果没有变化，仍为：

- `global_rmse = 0.002540`
- `global_mae = 0.000700`
- `global_rel_l1_active_mean = 4.434`
- `global_rel_l2 = 8.417`
- `mass_mean_rel_error = 4.430`

`sensor` 源参数基线仍是当前最优正式方案：

- 源位置平均误差 `160.37 mm`
- 泄漏率平均相对误差 `1.71%`
- 泄漏率 MAE `3.16 mL/min`

### 4. 传感器数量 / 观测时长实验的当前判断

远端已经生成了 `sensor_condition_study.csv`，但当前结果基本呈现“几乎不变”的状态。例如：

- `6 / 12 / 30` 个传感器下，`global_rmse` 都约为 `0.002540`
- `20 / 60 / 120` 步观测时长下，`global_rmse` 也几乎不变
- 低流量子集指标同样变化极小

这说明当前这批“传感器数量 / 时长”结果不能直接用于解释模型能力曲线，因为它们更像是对“观测约束太弱”的再次佐证，而不是有效的条件影响实验。

### 5. 当前阶段结论更新

截至本次补查，结论应进一步明确为：

- 主模型训练已完成。
- 正式全场反演结果已可用于阶段性汇报。
- 正式源参数基线已建立，短期应固定 `sensor` 回归器。
- `MPDPS` 观测调参尚未完成，因此“传感器数量/观测时长影响”实验暂不能作为最终可信结论。
- 多次重复的训练数据量实验尚未正式启动到可汇报状态。

## 八、2026-04-08 服务器再核查

本次再次核查远端服务器后，状态更新如下。

### 1. 当前是否还在训练

主模型训练已经完成，当前没有任何 `FTM`、`GPSD`、`MPDPS` 调参或重复训练规模实验仍在运行。

远端活跃 Python 进程仅剩：

- `jupyter-lab`

因此，当前服务器并未继续进行有效训练计算。

### 2. 当前服务器是否还在使用

服务器实例仍然开着，但 GPU 基本空闲：

- GPU：`NVIDIA L40S`
- 利用率：`0%`
- 显存占用：约 `1 MiB / 46068 MiB`

这意味着：

- 服务器仍然在线
- 但当前没有训练任务在消耗 GPU
- 若继续保持开机，将主要消耗实例租用时间，而不是计算资源

### 3. 当前未完成任务为什么停了

`MPDPS` 观测调参任务不是自然完成，而是在某组配置上报错退出。当前日志显示失败发生在：

- 配置：`mpdps_weight = 2.0`
- `obs_rho = 0.003`
- `total_steps = 30`
- 模式：`correct`
- 测试样本：`case_0031`

调用链为：

- `run_mpdps_observation_tuning.py`
- `run_observation_sanity_check.py`
- `run_holdout_reconstruction_eval.py`
- `message_passing_DPS.py`

其中最外层日志已经确认本轮调参流程以 `CalledProcessError` 结束，因此当前“观测调参实验”处于中断状态，而不是仍在后台继续推进。

### 4. 当前状态的准确结论

截至这次服务器核查：

- 主模型训练：完成
- 正式 holdout 基线评估：完成
- 源参数正式基线：完成
- MPDPS 观测调参：未完成，且已报错中断
- 训练规模重复实验：未启动到正式结果阶段

## 九、2026-04-08 MPDPS 第一轮调参完成补记

本轮 `MPDPS` 第一轮观测调参现已全部跑完，汇总文件位于：

- `tmp/mpdps_observation_tuning_20260408.json`
- `tmp/mpdps_observation_tuning_20260408.csv`
- 远端同步副本：`results/advisor_study/mpdps_tuning_holdout400/mpdps_observation_tuning.json`
- 远端同步副本：`results/advisor_study/mpdps_tuning_holdout400/mpdps_observation_tuning.csv`

本轮共完成 `16` 组参数组合。按“正确观测输入下的绝对重建指标”看，当前最优配置是：

- `mpdps_weight = 4.0`
- `total_steps = 30`
- `num_posterior_samples = 1`
- `correct_global_rmse_mean = 0.002525823744081851`
- `correct_global_rel_l2_mean = 8.368168422540837`

但本轮更关键的结论不是“哪组绝对 RMSE 最低”，而是“观测是否真正起作用”。按 `correct` 对比 `shuffled` 的分离度看，最大也只有：

- `shuffled_minus_correct_rel_l2 = 0.00016989905692810225`
- `shuffled_over_correct_rel_l2 = 1.0000201851331987`

这说明即使在当前分离度最好的配置上，`correct` 与 `shuffled` 之间也几乎没有可用差异，仍不足以支撑“MPDPS 已经显著利用了观测信息”的结论。

另外，本轮还确认了一个实验框架层面的关键事实：在当前 `sensor_path` 反演流程下，`obs_rho` 实际不会改变观测输入，因此把 `obs_rho` 作为本轮有效调参维度是不成立的。其直接证据是：在相同 `mpdps_weight` 和 `total_steps` 下，`obs_rho = 0.003` 与 `obs_rho = 0.01` 的结果完全一致。

因此，本轮调参的正式结论应写为：

- 第一轮 `MPDPS` 参数网格已完整跑完。
- `total_steps = 30` 相比 `20` 确实略有改善。
- `mpdps_weight = 4.0` 给出了当前最优绝对重建指标。
- 但当前观测约束仍然偏弱，`correct / shuffled` 尚未被有效拉开。
- 下一轮调参不应继续把 `obs_rho` 当成主维度，而应转向更强的 `mpdps_weight / total_steps / num_posterior_samples / zeta`，并补充 `zeros` 与 `wrong_positions` 验证。

## 十、2026-04-08 MPDPS 第二轮强观测调参已启动

基于第一轮结果，当前已经启动第二轮更强的观测调参，远端任务特征如下：

- 输出目录改为数据盘：`/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2`
- 当前模式先只跑：`correct, shuffled`
- 当前参数网格：
  - `mpdps_weight = 4, 8, 16`
  - `total_steps = 30, 50`
  - `num_posterior_samples = 1, 2`
  - `zeta = 0.009, 0.03`
- `obs_rho` 不再作为有效主维度，仅保留单一占位值 `0.01`

本轮目的不是继续追求轻微的绝对 RMSE 改善，而是优先验证：在更强后验约束下，`correct` 是否能够稳定且明显优于 `shuffled`。只有这一点成立，后续“传感器数量 / 观测时长影响”实验才具有正式解释价值。

本轮远端查看入口：

- 日志：`/root/Hydrogen-leak/SDIFT_model/logs/mpdps_tuning_holdout400_stage2.log`
- 进程：`ps -ef | grep -E 'run_mpdps_observation_tuning.py .*mpdps_tuning_holdout400_stage2' | grep -v grep`
- 输出：`/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2`

## 十一、2026-04-08 MPDPS 第二轮强观测调参完成补记

第二轮 `stage2` 调参现已全部完成，结果文件位于：

- `/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2/mpdps_observation_tuning.json`
- `/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2/mpdps_observation_tuning.csv`

按“正确观测输入下的绝对重建指标”看，本轮最优配置是：

- `mpdps_weight = 16`
- `total_steps = 50`
- `num_posterior_samples = 2`
- `zeta = 0.03`
- `correct_global_rmse_mean = 0.0019759424491068963`
- `correct_global_rel_l2_mean = 6.344596685235556`
- `correct_global_rel_l1_active_mean_mean = 2.6161191807820634`

与第一轮最优配置相比，绝对重建误差继续下降，说明更强的后验采样和更大的 `zeta` 的确可以改善重建质量。

但若按“观测是否真正起作用”这一核心问题判断，本轮仍未达到理想状态。分离度最好的配置为：

- `w8_rho0.01_steps30_post01_z0.03`
- `shuffled_minus_correct_rel_l2 = 0.0001268754167522701`
- `shuffled_over_correct_rel_l2 = 1.0000151679715354`

这仍然只是 `1e-4` 量级差异，远不足以支撑“correct 明显优于 shuffled”的正式结论。也就是说：

- 第二轮调参显著降低了绝对误差；
- 但依然没有实质性增强观测输入对反演结果的支配作用；
- 因此后续不应继续只在 `mpdps_weight / total_steps / posterior_samples / zeta` 上做同类微调。

截至本条记录写入时，GPU 已空闲，第二轮任务已完全结束。

## 十二、2026-04-08 当前最优绝对精度基线已冻结

当前已将第二轮 `stage2` 中按绝对重建指标最优的配置正式冻结为“当前最优绝对精度配置”，对应文件：

- `SDIFT模型/exps/mpdps_best_absolute_holdout400_stage2.json`

冻结内容为：

- `mpdps_weight = 16`
- `total_steps = 50`
- `num_posterior_samples = 2`
- `zeta = 0.03`

必须明确说明：该配置只表示“当前全场反演绝对精度最优”，不表示“观测依赖最强”。

## 十三、2026-04-08 完整 sanity check 已补跑并在进行中

基于上述冻结配置，当前已启动完整 sanity check，模式扩展为：

- `correct`
- `shuffled`
- `zeros`
- `wrong_positions`

输出目录沿用：

- `/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2/w16_rho0.01_steps50_post02_z0.03`

运行方式为复用已完成的 `correct / shuffled`，只增量补跑 `zeros / wrong_positions`，避免重复消耗算力。

本条记录写入时：

- `correct / shuffled` 结果已存在；
- `zeros / wrong_positions` 尚在远端继续生成；
- 当前不能提前写结论，必须等四种模式全部落盘后再做最终判断。

## 十四、2026-04-08 MPDPS 诊断能力已接入

为定位“为什么观测项始终没把 correct 与 shuffled 拉开”，当前已经在 `message_passing_DPS.py` 中接入诊断输出，新增内容包括：

- 每一步 `llk_grad_norm`
- 每一步 `diffusion_update_norm`
- 每一步 `obs_update_norm`
- `obs_over_diffusion_ratio`
- `equal_norm_obs_scale`
- 直接残差与时间传播残差的平均范数
- `poest_matrix1 / poest_matrix2 / llk_grad` 的整体范数

同时新增了单 case 诊断入口：

- `SDIFT模型/run_mpdps_diagnostic_case.py`

其用途是：在单个测试样本上分别用 `correct / shuffled / zeros / wrong_positions` 跑同一配置，并直接导出诊断 JSON，检查观测项是否在数值尺度上被扩散更新压制。

## 十一、2026-04-08 MPDPS 第二轮阶段性结果

第二轮当前尚未全部完成，但已经产出前 `5` 组完整 `observation_sanity_check.json`。从现有结果看，绝对重建指标确实继续改善，尤其在增加 posterior sample 后改善明显。例如：

- `w4_rho0.01_steps30_post01_z0.03`
  - `correct_global_rmse_mean = 0.0025254184732347073`
  - `correct_global_rel_l2_mean = 8.366507882610305`
- `w4_rho0.01_steps50_post01_z0.009`
  - `correct_global_rmse_mean = 0.0025201603014136527`
  - `correct_global_rel_l2_mean = 8.348474141277544`
- `w4_rho0.01_steps30_post02_z0.03`
  - `correct_global_rmse_mean = 0.0019821603265076726`
  - `correct_global_rel_l2_mean = 6.368061136044628`

这说明更强的采样配置可以显著改善“绝对重建质量”。

但更关键的观察是：截至目前，`correct` 与 `shuffled` 仍几乎完全重合，甚至在若干已完成配置上 `shuffled` 还略优于 `correct`，但差异量级仍只有 `1e-5 ~ 1e-4`。因此当前不能把这类改善解释为“观测约束更强了”，更合理的判断是：

- 第二轮目前主要提升了采样/后验平均带来的绝对误差。
- 观测主导性问题尚未被解决。
- 即使 absolute RMSE 明显下降，也暂时不能据此恢复“传感器数量 / 观测时长影响实验”的正式解释资格。
## 十五、2026-04-11 17:45 远端 56-case 训练进度检查

服务器当前主工作目录为 `/hy-tmp/SDIFT_model56`，数据盘空间充足，`/hy-tmp` 约 150 GB，总占用约 72 GB；系统盘 `/` 约 70% 占用，暂未达到危险状态。当前可见 4 张 NVIDIA L40S，其中 GPU0、GPU1、GPU2 正在满载训练，GPU3 在完成未见泄漏率分支后已空闲。

当前仍在运行的是 56-case 修正标签后的训练数据量重复实验，分为三种权重模式并行：

- `none`
- `lowflow_focus_v1`
- `lowflow_balanced_v1`

截至本次检查，`none` 已完成 `6×3`、`12×3` 和 `24×1`，正在训练 `24×repeat_01`，进度约 `89%`；`lowflow_focus_v1` 已完成 `6×3` 和 `12×1`，正在训练 `12×repeat_01`，进度约 `45%`；`lowflow_balanced_v1` 已完成 `6×3` 和 `12×1`，正在训练 `12×repeat_01`，进度约 `45%`。三条链均未发现新的 Python traceback，训练仍在正常推进。

已完成的中间指标只能作为趋势观察，不能作为最终结论。当前 `none` 模式下，`n=24, repeat_00` 的全场重建平均指标为 `RMSE=8.238e-4`、`MAE=5.144e-4`、`active relative L1=2.391`，暂时优于 `n=6/12` 的均值；但由于 `n=24` 只完成 1 次重复，尚不能判断训练数据量曲线是否线性、递减或波动。`lowflow_focus_v1` 在 `n=6` 的整体均值约为 `RMSE=1.069e-3`、`MAE=7.361e-4`，低泄漏率子集表现也优于 `none` 和 `balanced` 的部分结果，但 `n=12` 当前只完成 1 次重复且结果变差，必须等完整 3 次重复后再下结论。

未见泄漏率实验 `holdout_rate_0100_val_0200` 已完成训练与修复后的测试集评估，输出位于 `exps/gp-edm_holdoutrate0100_cfd56_cfd56_holdoutrate0100_val0200_train_20260411-1202/test_eval/aggregate_metrics.json`。测试集 8 个 case 的均值为 `RMSE=0.14079`、`MAE=0.09661`、`active relative L1=157.20`。该结果被 `case_0049` 明显拉高；除该异常样本外，多数 100 ml/min 样本 RMSE 约为 `0.026`。因此，未见泄漏率实验已经形成结果，但必须后续单独排查 `case_0049` 的真值尺度、标签、传感器抽样或重建输出是否存在异常。

源参数方面，目前可确认 `sensor` 参数基线已生成，路径为 `results/cfd56_holdoutrate0100_val0200_sensor_param_baseline/sensor_param_baseline.json`。该基线在测试集上的源位置平均误差约 `176.64 mm`，泄漏率 MAE 约 `68.24 ml/min`。这只能作为临时基线；`sensor / core / hybrid` 三种源参数模型的正式对比仍需在修正 manifest 后单独补齐。

按当前不重排任务的方式估算，训练数据量重复实验的瓶颈在 `lowflow_focus_v1` 和 `lowflow_balanced_v1` 两条链。每个 subset 从 FTM、GPSD 到 7 个测试 case 评估大约需要 2 小时上下；当前剩余量约为 `none` 5 个 subset、`focus` 8 个 subset、`balanced` 8 个 subset。由于三条链只占用 3 张卡，若不人工拆分 GPU3，整体大约还需 15-17 小时；若后续将空闲 GPU3 和先完成的 GPU0 用于安全拆分剩余 subset，理论上可压缩到约 9-12 小时，但需要避免与当前原始脚本发生重复训练或写文件冲突。

当前建议是先不中断正在运行的 3 个训练进程，等当前 subset 完成后再根据输出目录状态决定是否拆分 GPU3。当前最重要的监控文件为：

- `logs/cfd56_scale_none_relabel20260411.log`
- `logs/cfd56_scale_lowflow_focus_relabel20260411.log`
- `logs/cfd56_scale_lowflow_balanced_relabel20260411.log`
- `logs/cfd56_holdoutrate0100_val0200_eval_fix.log`

## 十六、2026-04-12 09:20 远端训练进度检查

远端主目录仍为 `/hy-tmp/SDIFT_model56`。当前数据盘 `/hy-tmp` 占用约 `91 GB / 150 GB`，剩余约 `60 GB`，系统盘 `/` 占用约 `70%`，暂未出现磁盘风险。当前 4 张 L40S 中只有 GPU2 满载，GPU0、GPU1、GPU3 已空闲。

训练进程方面，主脚本 `run_56case_relabel_and_rate_tasks_4gpu.sh` 仍在等待最后一条分支结束。`none` 与 `lowflow_focus_v1` 两条训练数据量重复实验已经完整完成，均已生成 `training_scale_repeated_detail.*` 和 `training_scale_repeated_summary.*`。当前只剩 `lowflow_balanced_v1` 分支仍在运行。

截至本次检查，三条训练数据量曲线完成情况如下：

- `none`：`6/12/24/42 × 3 repeats` 已全部完成。
- `lowflow_focus_v1`：`6/12/24/42 × 3 repeats` 已全部完成。
- `lowflow_balanced_v1`：已完成 `6×3`、`12×3`、`24×3`、`42×1`；`42×repeat_01` 正在训练，`42×repeat_02` 尚未启动。

当前正在运行的具体任务是：

- `train_GPSD.py`
- 数据集：`cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_n042_r01`
- 当前日志进度约 `3328 / 4000`，约 `83%`
- 日志显示剩余 GPSD 时间约 `22-23 min`

按最近一个完整 `n=42` 子任务的耗时估算，`lowflow_balanced_v1 n042_r01` 完成 GPSD 后还需要进行 7 个测试 case 的 MPDPS 重建与指标汇总；随后 `n042_r02` 还需要完整执行 FTM、GPSD 和评估。因此若保持当前脚本顺序不拆分任务，保守估计总剩余时间约 `3.5-4.5 h`。如果在 `r01` 完成后人工把 `r02` 拆到空闲 GPU 上，理论上可以略微节省等待，但由于只剩最后一个 subset，收益有限，且需要避免与主脚本自动启动的 `r02` 发生重复写入。

已完成曲线的阶段性观察如下。`none` 完整结果显示 `n=6` 与 `n=12` 的均值接近，`n=24` 和 `n=42` 反而出现较大波动，说明训练数据量增加并未稳定带来误差下降；这需要结合重复实验的标准差正式解释。`lowflow_focus_v1` 的 `n=6` 当前表现较好，整体 `RMSE` 均值约 `1.069e-3`，`MAE` 均值约 `7.361e-4`，但 `n=12/24/42` 并未继续单调变好。`lowflow_balanced_v1` 目前还缺最后两个 `n=42` repeat，暂不做最终判断。

## 十六、2026-04-11 22:22 训练必要性与算力状态复查

远端当前 4 张 L40S 均已有负载。GPU0、GPU1、GPU2 分别继续跑 `none`、`lowflow_focus_v1`、`lowflow_balanced_v1` 的训练数据量重复实验；GPU3 已被用于额外拆分 `lowflow_focus_v1 n=42` 子任务。当前未见新 traceback，训练仍在推进。

截至本次复查，`none` 已完成 `n=6/12/24` 的 3 次重复，但结果显示方差很大：`n=24` 的三次重复分别约为 `RMSE=8.238e-4`、`5.055e-3`、`2.047e-2`，均值反而劣于 `n=6/12`。这说明当前不能简单得出“训练数据越多越好”的结论，更可能存在子集分布、低泄漏率占比、训练随机性和生成模型稳定性共同影响。因此该实验如果要正式支撑导师提出的“模型成本-精度关系”，必须继续完成重复实验，否则只能作为不完整现象观察。

`lowflow_focus_v1` 已完成 `n=6/12` 的 3 次重复，当前均值显示 `n=6` 反而优于 `n=12`：`n=6 RMSE≈1.069e-3`，`n=12 RMSE≈1.884e-3`。这提示低流量加权不一定随训练样本数单调提升，必须继续看 `n=24/42`。`lowflow_balanced_v1` 已完成 `n=6/12` 的 3 次重复，当前 `n=12` 略优于 `n=6`，但差异仍需结合 `n=24/42` 判断。

当前判断：如果只是证明“模型能在未见位置上完成 CFD 真值闭环验证”，现有结果已经足够；如果要回答导师的第二个硬问题“训练数据量增加后能力怎么变化”，继续训练仍有必要，尤其要补齐 `n=24/42` 的重复结果。否则报告中只能写“初步结果显示关系不稳定”，不能给出可靠的 mean ± std 曲线。

算力方面，单个 GPSD 训练进程当前约占用 23 GB 显存，L40S 48 GB 比较稳妥；24 GB 显卡理论上可能勉强运行，但很容易因 batch size、并行评估或显存碎片触顶。当前代码不是 DDP，多卡不会加速单个模型，只能并行跑不同 subset 或不同实验分支。因此想缩短总耗时，优先增加可并行任务数对应的 GPU 数量，而不是只换一张更强的卡。

## 十六、2026-04-11 22:15 训练进度复查

本次复查时，服务器时间为 `2026-04-11 22:14`。当前 4 张 L40S 均有任务占用，其中 GPU0、GPU1、GPU2 正在跑 GPSD，GPU3 正在跑额外拆出的 `lowflow_focus_v1, train_size=42` 分支。GPU3 利用率低于前三张卡，是因为当前处于 FTM 阶段，显存和 GPU 利用率天然低于 GPSD 阶段。

当前进度如下：

- `none`：`6×3`、`12×3`、`24×3` 已完成；正在跑 `42×repeat_00`，GPSD 进度约 `57%`。
- `lowflow_focus_v1`：`6×3`、`12×3` 已完成；主链正在跑 `24×repeat_00`，GPSD 进度约 `45%`；GPU3 已额外完成 `42×repeat_00`，正在跑 `42×repeat_01` 的 FTM。
- `lowflow_balanced_v1`：`6×3`、`12×3` 已完成；正在跑 `24×repeat_00`，GPSD 进度约 `45%`。

需要明确纠正此前“今晚 12 点前可能完成”的估计：按完整任务口径，这个估计偏乐观。慢的原因不是单卡性能不足，而是当前实验并不是一次多卡分布式训练，而是大量独立模型重复训练。每一种权重模式下都要对 `6/12/24/42` 四个训练规模做 `3` 次分层重复，每个 subset 都包含 FTM、GPSD 和 7 个 holdout case 的 MPDPS 反演评估。也就是说，这不是“一个模型跑 4 卡”，而是几十个模型/评估链条在串并混合推进。

如果保持当前脚本自然推进，瓶颈会落在 `lowflow_balanced_v1` 后续的 `24×repeat_01/02` 和 `42×repeat_00/01/02`，预计仍需约 `12-14` 小时。若后续在 GPU0/GPU3 释放后继续手动拆分剩余 subset，理论上可压缩到约 `7-9` 小时，但需要避免与主脚本重复写同一输出目录。当前不建议粗暴启动重复任务，否则可能造成 HDF5 锁冲突或同一 subset 被两条进程同时训练。

## 十六、2026-04-11 18:06 GPU3 追加调度已挂起

按“先等当前 subset 完成，再启用 GPU3 拆分”的原则，已在服务器上挂起一个后台监控任务，PID 为 `46511`。该任务不会立即启动训练，而是等待以下两个当前 subset 都完成并生成 `aggregate_metrics.json`：

- `lowflow_focus_v1_n012_r01`
- `lowflow_balanced_v1_n012_r01`

满足条件后，该任务会在 GPU3 上额外运行 `lowflow_focus_v1` 的远端剩余大样本任务：

- `train_size = 42`
- `repeat = 0, 1, 2`
- 输出目录仍为 `results/advisor_study/train_scale_lowflow_focus_cfd56_relabel20260411`

这样安排的原因是 `focus size=42` 距离主脚本当前进度较远，GPU3 有较大安全窗口先完成这些 subset；当原主脚本后续走到同一 subset 时，会因为已存在 FTM、GPSD 和 `aggregate_metrics.json` 而自动跳过，避免重复训练。监控日志为：

- `logs/gpu3_focus42_after_current_20260411.log`

当前配置并不是“最快训练配置”，而是“正式可比配置”。其中 `direct_inner` 观测注入、`mpdps_weight=16`、`obs_rho=0.01`、`zeta=0.03` 是为了保持与当前最优反演基线一致；`eval_total_steps=20`、`num_posterior_samples=1` 已经是训练数据量实验中的加速评估设置。如果只追求筛选速度，可以把 GPSD 训练步数从 `4000` 降到 `2000` 或 `1000`，并减少 FTM 迭代和评估 case 数，但这样得到的是筛选结果，不能直接作为导师汇报中的最终定量结论。

## 十七、2026-04-11 18:48 训练状态复查

本次复查时，服务器不是 4 卡满载。`nvidia-smi` 显示 GPU0 满载，GPU1、GPU2、GPU3 接近空闲。原因不是主任务退出，而是三条训练数据量实验链处在不同阶段：`none` 分支正在 GPU0 上训练 `n024_r02` 的 GPSD，进度约 `20%`；`lowflow_focus_v1` 和 `lowflow_balanced_v1` 的 `n012_r01` GPSD 已经跑到 `4000/4000`，当前进入 `run_holdout_reconstruction_eval.py` 评估阶段，GPU 占用会比训练阶段低且可能间歇性使用；GPU3 当前仍在等待追加调度条件触发。

截至本次复查，完成情况如下：

- `none`：已完成 `n006_r00/r01/r02`、`n012_r00/r01/r02`、`n024_r00/r01`，当前正在跑 `n024_r02`，后面还剩 `n042_r00/r01/r02`。
- `lowflow_focus_v1`：已完成 `n006_r00/r01/r02`、`n012_r00`，当前正在评估 `n012_r01`，后面还剩 `n012_r02`、`n024_r00/r01/r02`、`n042_r00/r01/r02`。
- `lowflow_balanced_v1`：已完成 `n006_r00/r01/r02`、`n012_r00`，当前正在评估 `n012_r01`，后面还剩 `n012_r02`、`n024_r00/r01/r02`、`n042_r00/r01/r02`。

按当前原始脚本顺序继续跑，不人为拆分更多任务的情况下，瓶颈仍是 `focus` 和 `balanced` 两条链，预计还需要约 `14-16 小时`。如果 GPU3 追加调度顺利触发，并且后续先完成的 GPU0 也被安全用于 disjoint subset，理论上可压缩到约 `8-10 小时`，但必须避免与当前父进程同时写同一个输出目录。
## 十七、2026-04-12 09:25 降为 1 卡后的中断与恢复

用户将远端实例从 4 卡降为 1 卡后，原训练主进程已退出。检查时服务器仅剩 1 张 L40S 可见，且没有 `run_56case_relabel_and_rate_tasks_4gpu.sh`、`run_training_scale_repeated_study.py`、`train_GPSD.py` 或 `message_passing_DPS.py` 进程。说明降卡操作已经中断原训练链。

中断点位于 `lowflow_balanced_v1_n042_r01` 的 GPSD 训练阶段，日志停在约 `3446 / 4000`，没有 traceback、`Killed`、`CUDA` 或 `RuntimeError` 字样，但没有生成 `ema_3999.pth`，也没有生成该 subset 的 `aggregate_metrics.json`。因此不能把该中断结果作为正式完成结果。`lowflow_balanced_v1_n042_r02` 尚未开始。

为避免脚本误把不完整的 `ema_3000.pth` 当成已完成模型使用，已将中断实验目录移动为备份：

- `exps/cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_gpsd_cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_n042_r01_20260412-0729_interrupted_20260412_0924`

随后已用当前单卡重新启动 `lowflow_balanced_v1` 分支恢复任务，日志为：

- `logs/cfd56_scale_lowflow_balanced_relabel20260412_resume_1gpu.log`

恢复进程 PID 为 `601`。恢复脚本会重新检查/生成 repeated subset 文件，然后自动跳过已有 `aggregate_metrics.json` 的完成项，只补跑缺失的 `n042_r01` 与 `n042_r02`。截至本条记录写入时，恢复进程仍在运行，当前处于 subset HDF5/manifest 重建阶段，因此 GPU 暂时空闲；进入 FTM/GPSD 后 GPU 会重新满载。

在只保留 1 张 L40S 的情况下，剩余任务预计需要约 `4-5 h`。其中 `n042_r01` 需要从 GPSD 重新训练到 4000 steps，再做 7 个 case 的 MPDPS 评估；`n042_r02` 还需要完整执行 FTM、GPSD 和评估。若期间不中断，预计今天中午到下午早些时候可以完成 `lowflow_balanced_v1` 曲线并生成完整 summary。

## 十八、2026-04-12 09:52 单卡恢复后的实时状态

再次检查确认：当前远端只可见 1 张 NVIDIA L40S，训练任务仍在运行，没有再次中断。当前运行的是恢复后的 `lowflow_balanced_v1` 分支，进程为 `run_training_scale_repeated_study.py`，其子进程正在执行最后一个 subset：

- `train_FTM.py`
- 数据集：`cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_n042_r02`
- 输入：`results/advisor_study/train_scale_lowflow_balanced_cfd56_relabel20260411/subsets/size_042/repeat_02/train_042_r02.h5`

当前完成状态如下：

- `none`：12/12 个 subset 完成，summary 已生成。
- `lowflow_focus_v1`：12/12 个 subset 完成，summary 已生成。
- `lowflow_balanced_v1`：11/12 个 subset 完成；`n042_r01` 已完成并生成 `aggregate_metrics.json`；只剩 `n042_r02`。

因此当前已经不是大规模训练阶段，只是在补最后一个 `lowflow_balanced_v1 n042_r02`。按日志和历史耗时估计，剩余时间约为 `2.8-3.5 h`：FTM 还需约 `20-25 min`，GPSD 约 `2.0-2.3 h`，7 个测试 case 的 MPDPS 评估和汇总约 `15-30 min`。降为 1 卡后不会影响已完成结果，只会让最后一个 subset 不能再并行加速。

## 十九、2026-04-12 10:08 单卡恢复后的最新状态

再次检查远端状态，当前服务器时间为 `2026-04-12 10:07:53`。远端只显示 1 张 L40S，GPU 显存约 `6.2 GB / 46 GB`，利用率约 `45%`。当前仍只剩最后一个任务：`lowflow_balanced_v1_n042_r02`。

已完成状态：

- `none`：12/12 个 subset 完成，summary 已生成。
- `lowflow_focus_v1`：12/12 个 subset 完成，summary 已生成。
- `lowflow_balanced_v1`：11/12 个 subset 完成；`n042_r01` 已在 `09:42:22` 完成。

当前运行进程为 `train_FTM.py`，数据集为 `cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_n042_r02`，说明最后一个 subset 正处于 FTM 分解阶段，还未进入 GPSD。根据前一个 `n042` repeat 的实际耗时，最后一个 subset 从 FTM、GPSD 到 7 个测试 case 评估总计约 2.5-3 小时；当前已运行约 25 分钟，因此保守估计剩余时间约 `2-2.5 h`。若不中断，预计 `12:00-12:40` 左右完成全部训练数据量实验。
