# 2026-04-11 追加实验日志

## 1. 本次任务目标

本轮工作承接前一版 56-case 正式结果，目标不是重跑主线浓度场模型，而是处理“坐标标签纠偏后真正会受影响的部分”，并补出下一条更有科研价值的泛化实验线。具体包括四件事：

1. 将修正后的本地元数据同步到远端服务器。
2. 保留现有 `holdout_400` 浓度场重建主结果，不因标签纠偏重新训练主线模型。
3. 基于修正后的 manifest，重跑会受坐标标签影响的实验：
   - `sensor / core / hybrid` 源参数模型比较
   - `training_scale repeated study`
4. 正式补一个“未见泄漏率”实验，而不是继续只强调“未见位置”。

## 2. 已完成：修正后的本地元数据同步到服务器

以下文件已经同步到远端目录 `/hy-tmp/SDIFT_model56`：

- `build_cfd_multicase_dataset.py`
- `data/cfd56_all_T120_interp48_manifest.csv`
- `data/splits/holdout_400_0_0_val_0200/split.json`
- `data/splits/holdout_400_0_0_val_0200/train_manifest.csv`
- `data/splits/holdout_400_0_0_val_0200/val_manifest.csv`

同时新增并同步了两份后续实验脚本：

- `subset_h5_by_rate_3way.py`
- `run_56case_relabel_and_rate_tasks.sh`

其中 `build_cfd_multicase_dataset.py` 已修正五字段 case 名解析规则，后续像 `6,0,100,0,50` 和 `6,0,200,0,1000` 这类目录不再被误写成 `(0,100,0)` 与 `(0,200,0)`，而是正确解析为 `(100,0,0)` 与 `(200,0,0)`。

## 3. 已完成：保持主线浓度场模型不重跑

这一点已经严格执行。

当前 `holdout_400` 主线全场重建模型没有重新训练，仍沿用已有正式结果。原因是：

- `HDF5` 张量本身没有变化；
- `train / val / test` 的 case 归属没有变化；
- 正式测试集仍然是未见位置 `(400,0,0)`；
- 因此主线浓度场重建的数值结果仍然有效，变化只发生在元数据解释层。

换言之，本轮没有浪费算力去重跑已经足够稳定的 `FTM + GPSD + MPDPS` 主验证结果。

## 4. 已完成：source / core / hybrid 三种源参数模型重新比较

基于修正后的 `holdout_400_0_0_val_0200` manifest，远端已经重新完成 `sensor / core / hybrid` 三种源参数模型的正式比较，结果目录为：

- `/hy-tmp/SDIFT_model56/results/cfd56_holdout400_val0200_source_param_compare_relabel20260411`

核心排序结果如下：

| 模型 | selected_alpha | 源位置平均误差 / mm | 泄漏率相对误差均值 | 泄漏率 MAE / (mL/min) |
| --- | ---: | ---: | ---: | ---: |
| sensor | 10.0 | 217.10 | 0.1019 | 34.94 |
| hybrid | 100.0 | 1111.61 | 9.8869 | 1563.17 |
| core | 100.0 | 922.00 | 20.1334 | 3100.49 |

结论非常明确：

- `sensor` 仍然是当前唯一可作为正式基线汇报的源参数模型。
- `core` 与 `hybrid` 在当前 56-case / holdout_400 任务上明显失效，至少目前不应作为主结果。
- 坐标纠偏后，原先“先用重建场峰值搜索定位”的做法仍然可以保留作参考，但主汇报口径应切换为“sensor regression baseline”。

## 5. 已完成：新增“未见泄漏率”正式 split

本轮新建了按泄漏率切分的数据划分工具：

- [subset_h5_by_rate_3way.py](/d:/github/Hydrogen-leak/SDIFT模型/subset_h5_by_rate_3way.py)

其作用是像按位置切分一样，直接从统一 `HDF5 + manifest` 生成 `train / val / test` 三向切分，但切分依据从“泄漏位置”改为“泄漏率”。

本次正式设定为：

- `val_rate = 200 mL/min`
- `test_rate = 100 mL/min`

对应远端已生成的新 split 目录为：

- `/hy-tmp/SDIFT_model56/data/splits/holdout_rate_0100_val_0200`

切分规模为：

- `train = 40`
- `val = 8`
- `test = 8`

该切分有两个优点：

1. `test` 集是真正的“未见泄漏率”，不再只是“未见位置”。
2. 每个 split 都保留全部 8 个泄漏位置，因此它主要检验的是“泄漏率泛化”，而不是位置泛化。

这条实验线比继续重复未见位置结果更有价值，因为它直接回答了“模型能不能从已见离散泄漏率迁移到未见泄漏率等级”。

## 6. 已启动：训练数据量重复实验重跑

由于 `build_repeated_train_size_subsets.py` 会按位置分层抽样，而位置标签在本轮被纠偏，因此旧版 `train_scale_*_cfd56` 结果不能再直接作为最终版。

为保证可复现，本轮新增了远端批处理脚本：

- [run_56case_relabel_and_rate_tasks.sh](/d:/github/Hydrogen-leak/SDIFT模型/run_56case_relabel_and_rate_tasks.sh)

该脚本会顺序执行：

1. `train_scale_none_cfd56_relabel20260411`
2. `train_scale_lowflow_focus_cfd56_relabel20260411`
3. `train_scale_lowflow_balanced_cfd56_relabel20260411`
4. `holdout_rate_0100_val_0200` 正式训练与评估

当前远端后台总任务已经启动，主日志为：

- `/hy-tmp/SDIFT_model56/logs/relabel_rate_master_20260411.log`

首段执行已经确认正常开始，`scale_none` 的 subset 构建已成功写出，不是“起进程即退出”的假启动。

## 7. 远端查看命令

如果需要随时查看远端进度，可在服务器中执行：

```bash
cd /hy-tmp/SDIFT_model56
ps -ef | grep -E 'run_training_scale_repeated_study.py|run_56case_relabel_and_rate_tasks.sh|train_FTM.py|train_GPSD.py|run_holdout_reconstruction_eval.py' | grep -v grep
tail -f logs/relabel_rate_master_20260411.log
tail -f logs/cfd56_scale_none_relabel20260411.log
nvidia-smi
```

后续阶段性完成标志如下：

- `results/advisor_study/train_scale_none_cfd56_relabel20260411/training_scale_repeated_summary.csv`
- `results/advisor_study/train_scale_lowflow_focus_cfd56_relabel20260411/training_scale_repeated_summary.csv`
- `results/advisor_study/train_scale_lowflow_balanced_cfd56_relabel20260411/training_scale_repeated_summary.csv`
- `exps/gp-edm_holdoutrate0100_cfd56_cfd56_holdoutrate0100_val0200_train_*/test_eval/aggregate_metrics.json`

## 8. 当前阶段判断

截至本次追加日志，任务状态可以概括为：

- 元数据纠偏：完成
- 服务器同步：完成
- 主线浓度场结果保留：完成
- source/core/hybrid 重跑比较：完成
- 未见泄漏率 split：完成
- 训练数据量实验重跑：已启动
- 未见泄漏率正式训练与评估：已排队，待前序训练数据量实验完成后自动继续

## 9. 下一步重点

下一轮最重要的不是再改模型结构，而是拿到两类正式新结果：

1. `relabel20260411` 版训练数据量曲线  
   目标是给导师一条不再受错误位置标签污染的 `mean ± std` 关系曲线。

2. `holdout_rate_0100_val_0200` 未见泄漏率结果  
   目标是把“泛化”从当前的未见位置，扩展到更有说服力的未见泄漏率。

如果这两条线都完成，那么导师布置的三部分任务就会形成更完整的结构：

- 已证实：未见位置闭环可验证
- 已量化：误差指标和成本曲线可汇报
- 新补强：未见泄漏率泛化开始形成正式结果

## 10. 当前恢复状态与续跑命令

本次追加之后，远端链条已经重新恢复并继续执行。当前不是“尚未启动”，而是已经在跑 `train_scale_none_cfd56_relabel20260411`，其后会自动串行进入：

1. `train_scale_lowflow_focus_cfd56_relabel20260411`
2. `train_scale_lowflow_balanced_cfd56_relabel20260411`
3. `holdout_rate_0100_val_0200` 正式训练与评估

如果服务器后续再次重启，不需要手工判断从哪一步继续。因为：

- [run_56case_relabel_and_rate_tasks.sh](/d:/github/Hydrogen-leak/SDIFT模型/run_56case_relabel_and_rate_tasks.sh) 已改成按最终产物文件判断是否跳过阶段；
- [run_56case_formal_pipeline.sh](/d:/github/Hydrogen-leak/SDIFT模型/run_56case_formal_pipeline.sh) 也已补成可跳过已完成的 `FTM / GPSD / sensor baseline / val-test eval`。

后续如需手工恢复，只要在服务器里执行：

```bash
cd /hy-tmp/SDIFT_model56
nohup bash ./run_56case_relabel_and_rate_tasks.sh . > logs/relabel_rate_master_20260411.log 2>&1 &
```

查看是否已经续上，可执行：

```bash
cd /hy-tmp/SDIFT_model56
ps -ef | grep -E 'run_training_scale_repeated_study.py|run_56case_relabel_and_rate_tasks.sh|train_FTM.py|train_GPSD.py|run_holdout_reconstruction_eval.py' | grep -v grep
tail -f logs/relabel_rate_master_20260411.log
tail -f logs/cfd56_scale_none_relabel20260411.log
```

## 11. 4 卡并行切换与冗余文件清理

在本次后续推进中，远端资源已从单卡切换为 4 张 `L40S` 可见。经核查，远端实际可见卡为：

- GPU0: L40S
- GPU1: L40S
- GPU2: L40S
- GPU3: L40S

由于此前误标签版本的 `train_scale_*_cfd56` 三组旧结果各占约 `9.7G`，远端 `/hy-tmp` 仅剩约 `9.3G` 可用空间，继续训练存在中途写爆磁盘风险。因此本轮先做了保守清理，只删除明确冗余且已被新任务替代的目录：

远端已删除：

- `results/advisor_study/train_scale_none_cfd56`
- `results/advisor_study/train_scale_lowflow_focus_cfd56`
- `results/advisor_study/train_scale_lowflow_balanced_cfd56`

清理后远端磁盘由约 `91G used / 9.3G free` 变为约 `62G used / 39G free`，足以支撑并行继续训练。

本地已删除：

- `results/tmp_holdout_rate100_val200`
- `results/study_dryrun`
- `results/tmp_param_compare_sensor`
- `results/tmp_sensor_param_model.json`
- `results/tmp_sensor_param_train.csv`
- `data/splits/holdout_400_0_0/train_size_subsets_demo`

以上删除对象均为临时、dryrun 或已被正式版本替代的中间产物，不影响当前正式 56-case 数据、正式 split、正式 holdout 结果或研究日志。

为真正用满 4 卡，本轮新增并同步了 [run_56case_relabel_and_rate_tasks_4gpu.sh](/d:/github/Hydrogen-leak/SDIFT模型/run_56case_relabel_and_rate_tasks_4gpu.sh)。该脚本将四条任务拆到 4 张卡并行：

1. GPU0: `train_scale_none_cfd56_relabel20260411`
2. GPU1: `train_scale_lowflow_focus_cfd56_relabel20260411`
3. GPU2: `train_scale_lowflow_balanced_cfd56_relabel20260411`
4. GPU3: `holdout_rate_0100_val_0200` 正式训练与评估

切换完成后，旧的单卡 orphan 训练进程已被显式终止，新的 4 卡并行链已启动。当前总控日志为：

- `/hy-tmp/SDIFT_model56/logs/relabel_rate_master_4gpu_20260411.log`

四路核心日志为：

- `logs/cfd56_scale_none_relabel20260411.log`
- `logs/cfd56_scale_lowflow_focus_relabel20260411.log`
- `logs/cfd56_scale_lowflow_balanced_relabel20260411.log`
- `logs/cfd56_holdoutrate0100_val0200_formal.log`

## 12. 2026-04-11 15:15 远端训练状态复核

本次复核时，远端工作目录为 `/hy-tmp/SDIFT_model56`，4 卡并行链仍在运行。训练数据量重复实验的三条链均未异常退出，其中 `none` 已完成 `6/12` 个聚合结果，`lowflow_focus_v1` 已完成 `3/12` 个聚合结果，`lowflow_balanced_v1` 已完成 `3/12` 个聚合结果。当前正在运行的子任务分别为：

- GPU0: `cfd56_scale_none_relabel20260411_none_n024_r00`
- GPU1: `cfd56_scale_focus_relabel20260411_lowflow_focus_v1_n012_r00`
- GPU2: `cfd56_scale_balanced_relabel20260411_lowflow_balanced_v1_n012_r00`

未见泄漏率实验的 FTM/GPSD 主训练已经完成，但首次评估阶段因路径参数传递问题中断，表现为 `core_mean_std.mat` 默认路径找不到。该问题已经定位为评估命令没有正确传入 `model_path` 与 `core_mean_std_path`，不是模型训练失败。已使用已有训练产物重新启动评估，不需要重训未见泄漏率模型。当前使用的关键产物为：

- `ckp/basis_cfd56_holdoutrate0100_val0200_train_4x8x8_2026_04_11_120228_last.pth`
- `exps/gp-edm_holdoutrate0100_cfd56_cfd56_holdoutrate0100_val0200_train_20260411-1202/checkpoints/ema_7999.pth`
- `exps/gp-edm_holdoutrate0100_cfd56_cfd56_holdoutrate0100_val0200_train_20260411-1202/core_mean_std.mat`

复核时发现未见泄漏率评估最初未正确绑定到 GPU3，导致 GPU3 空闲。已停止旧评估进程，并通过 `/tmp/start_rate_eval_gpu3.sh` 重新启动，显式设置 `CUDA_VISIBLE_DEVICES=3`。重启后 GPU3 已开始执行 MPDPS 采样，日志为：

- `logs/cfd56_holdoutrate0100_val0200_eval_fix.log`

服务器磁盘在并行训练后升至约 `90%` 使用率。为避免 checkpoint 持续写入导致磁盘打满，已执行保守清理：每个 `exps/*/checkpoints` 目录只保留最新 `model_*.pth` 和最新 `ema_*.pth`，删除旧中间 checkpoint 共 `294` 个，释放约 `21.1GB`。清理后 `/hy-tmp` 由约 `90G used / 11G free` 降至约 `69G used / 32G free`。此次清理不删除 HDF5、manifest、最终模型、评估 JSON 或聚合指标。

截至本次复核，训练仍在正常推进，但训练数据量重复实验尚未完成。按当前进度估计，未见泄漏率 val/test 评估预计约 `0.5-1.0 h` 完成；训练数据量三条链仍是总耗时瓶颈，若不中断，预计还需要约 `6-9 h` 才能全部跑完。该估计会随 `n024` 和 `n042` 子集的实际耗时调整。
