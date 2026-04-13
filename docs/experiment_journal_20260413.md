# Experiment Journal 2026-04-13

## 2026-04-13 00:56 clean48 三卡恢复与任务分配

服务器主目录为 `/hy-tmp/SDIFT_model56`，当前可见 3 张 NVIDIA L40S。`cfd48_clean_T120_interp48.h5` 已完成上传、解压、质检和 split 重建，clean 数据质量检查通过，`flagged_count=0`。因此本轮训练与评估只使用 clean48 数据，不再使用包含 `case_0048-case_0055` 异常样本的 cfd56 数据。

上一轮 2 卡启动的 clean formal 任务已经完成 FTM 与 GPSD 训练，模型权重和 `core_mean_std.mat` 均存在，但自动进入评估阶段时路径传参错误，导致 `metadata_path` 和 `sensor_csv` 被展开为空字符串，评估进程失败。该问题不是模型训练失败，而是 shell 后台函数传参问题。已改用显式脚本 `run_clean48_formal_eval_fix_20260413.sh` 重新启动两个 formal eval。

当前三卡任务分配如下。

| GPU | 任务 | 状态 | 关键日志 |
| ---: | --- | --- | --- |
| 0 | clean48 未见泄漏率 formal eval | 正在运行 MPDPS，`steps50/post2/direct_inner` | `/hy-tmp/SDIFT_model56/logs/clean48_rate_formal_eval_fix_20260413.log` |
| 1 | clean48 未见位置 formal eval | 正在运行 MPDPS，`steps50/post2/direct_inner` | `/hy-tmp/SDIFT_model56/logs/clean48_position_formal_eval_fix_20260413.log` |
| 2 | clean48 训练数据量 repeated study | 正在跑 `lowflow_focus_v1, n=6, repeat=0` 的 FTM | `/hy-tmp/SDIFT_model56/logs/clean48_scale_lowflow_focus_20260413.log` |

GPU0/GPU1 当前用于生成正式可汇报的 clean 版闭环验证结果，分别对应导师要求中的“未见泄漏率泛化”和“未见位置泛化”。GPU2 用于补 clean 数据上的训练数据量影响曲线，当前配置为 `train_sizes=6/12/24/31`、每个规模 3 次分层重复、`sample_weight_mode=lowflow_focus_v1`，重建评估配置沿用当前最优 `direct_inner` 观测注入方案。

预计耗时如下。

| 任务 | 预计剩余时间 | 说明 |
| --- | ---: | --- |
| clean48 未见泄漏率 formal eval | 约 1.0-1.5 h | 已有模型，只需 val/test MPDPS 反演与指标汇总 |
| clean48 未见位置 formal eval | 约 1.0-1.5 h | 已有模型，只需 val/test MPDPS 反演与指标汇总 |
| clean48 训练数据量 repeated study | 约 6-10 h | 需要 12 个 subset 的 FTM、GPSD 与 7 case 评估，当前只占 1 卡串行推进 |

短期汇报优先级为先拿到 GPU0/GPU1 的两个 clean formal 结果。它们完成后即可更新报告中的核心闭环验证指标。训练数据量曲线属于导师任务中的“模型成本”部分，耗时最长，可以在 GPU0/GPU1 释放后继续拆分剩余 subset 到空闲 GPU，以压缩总时间。

监控命令：

```bash
watch -n 30 nvidia-smi
tail -f /hy-tmp/SDIFT_model56/logs/clean48_rate_formal_eval_fix_20260413.log
tail -f /hy-tmp/SDIFT_model56/logs/clean48_position_formal_eval_fix_20260413.log
tail -f /hy-tmp/SDIFT_model56/logs/clean48_scale_lowflow_focus_20260413.log
ps -ef | grep -E 'run_holdout_reconstruction_eval|message_passing_DPS|train_FTM|train_GPSD|run_training_scale_repeated_study' | grep -v grep
```

## 2026-04-13 01:02 运行状态复查

复查确认两个 formal eval 已经修复并正常产出指标文件，不再出现 `metadata_path=''` 或 `sensor_csv=''` 的错误。未见泄漏率分支已完成 2 个 val case，正在跑第 3 个 val case；未见位置分支已完成 2 个 val case，正在评估第 3 个 val case。每个 case 当前耗时约 1.5 min 左右，因此两个 formal eval 预计在 30-60 min 内完成 val/test 全部结果。

训练数据量 repeated study 已完成第一个 subset 的 FTM，并进入 `train_GPSD.py`。当前 subset 为 `lowflow_focus_v1, n=6, repeat=0`，GPU2 显存约 9.3 GB，利用率约 94-99%。该研究链路仍包含 12 个 subset，完整跑完预计需要 6-10 h。

`run_training_scale_repeated_study.py` 会在已有 `aggregate_metrics.json` 时跳过完成项，但没有运行中锁机制。为了避免父进程和手动拆分任务同时写同一个 subset，暂不在 GPU0/GPU1 尚未释放时强行拆分训练数据量实验。等 formal eval 完成后，应优先在 subset 边界检查已完成项，再把未开始的 size/repeat 拆到空闲 GPU。

## 2026-04-13 01:14 夜间自动调度

新增并上传了两个服务器脚本：

- `/hy-tmp/SDIFT_model56/scripts/remote_clean48_train_scale_helper_20260413.sh`
- `/hy-tmp/SDIFT_model56/scripts/remote_clean48_nightly_scheduler_20260413.sh`

夜间调度器 PID 为 `12032`，日志为 `/hy-tmp/SDIFT_model56/logs/clean48_nightly_scheduler_20260413.log`。调度逻辑如下：

1. 等待 clean48 未见泄漏率 formal eval 的 test aggregate 生成。
2. 等待 clean48 未见位置 formal eval 的 test aggregate 生成。
3. 在 GPU0 上补跑 `lowflow_focus_v1, train_size=24, repeat=0/1/2`。
4. 在 GPU1 上补跑 `lowflow_focus_v1, train_size=31, repeat=0/1/2`。
5. GPU2 原父进程继续顺序跑 `n=6/12`，后续遇到已完成的 `n=24/31` 会读取已有 `aggregate_metrics.json` 并跳过重复训练。
6. 所有任务结束后写入最终 error scan。

这样安排的原因是 GPU0/GPU1 当前还在跑 formal eval，直接抢占会影响最核心的汇报结果；等 formal eval 完成后，大样本 subset 最耗时，优先拆到空闲 GPU 能显著缩短总耗时。当前预计：formal eval 还需约 30-60 min，完整 clean48 训练数据量 repeated study 预计明早前完成；如果 helper 与父进程都正常，预计总剩余时间约 4-7 h。

监控命令：

```bash
tail -f /hy-tmp/SDIFT_model56/logs/clean48_nightly_scheduler_20260413.log
tail -f /hy-tmp/SDIFT_model56/logs/clean48_scale_helper_n024_gpu0_20260413.log
tail -f /hy-tmp/SDIFT_model56/logs/clean48_scale_helper_n031_gpu1_20260413.log
cat /hy-tmp/SDIFT_model56/logs/clean48_nightly_status_20260413.txt
```

## 2026-04-13 01:30 四卡调度与 formal 结果

用户将实例升级为 4 张 L40S 后，服务器已实际可见 GPU0-GPU3。clean48 的两个 formal eval 已全部完成 val/test 聚合，未见泄漏率与未见位置各 14 个 case 的 `eval.json` 均已生成，日志错误关键字扫描为空。

formal eval 阶段性指标如下。

| 实验 | split | count | RMSE | MAE | active relative L1 | mass relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 未见泄漏率 | val | 7 | 1.304e-3 | 5.546e-4 | 1.862 | 0.739 |
| 未见泄漏率 | test | 7 | 8.600e-4 | 3.548e-4 | 2.050 | 0.932 |
| 未见位置 | val | 7 | 6.439e-4 | 3.734e-4 | 3.521 | 3.894 |
| 未见位置 | test | 7 | 6.521e-4 | 3.717e-4 | 1.379 | 0.899 |

formal eval 完成后，4 卡分工已经切换为训练数据量 repeated study：

| GPU | 任务 | 状态 |
| ---: | --- | --- |
| 0 | `lowflow_focus_v1, n=24, repeat=0/1/2` | helper 已启动，正在跑 `n024_r00` 的 FTM |
| 1 | `lowflow_focus_v1, n=31, repeat=0/1/2` | helper 已启动，正在跑 `n031_r00` 的 FTM |
| 2 | `lowflow_focus_v1, n=12, repeat=2` | extra helper 已启动，正在跑 `n012_r02` 的 FTM |
| 3 | 原父进程顺序任务 | 正在跑 `n006_r00` 的 GPSD |

当前 4 张卡均有任务负载，错误扫描仍为空。预计剩余总耗时约 4-7 h，主要取决于 `n=24/31` 三次重复的 GPSD 和 MPDPS 评估耗时。若实例不中断，预计早上可以看到完整 clean48 formal 结果和 clean48 训练数据量 repeated study 结果。

## 2026-04-13 08:39 clean48 结果复查与文档同步

服务器路径仍为 `/hy-tmp/SDIFT_model56`。本次复查确认两个 clean formal 闭环验证任务已经完成，四个 aggregate 文件均存在，分别对应未见泄漏率 val/test 和未见泄漏位置 val/test。服务器端 `eval.json` 数量为 rate 14 个、position 14 个，数量与 val/test 各 7 个 case 的设计一致。

clean formal 指标如下。

| 实验 | split | count | RMSE | MAE | active relative L1 | mass relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 未见泄漏率 `100 mL/min` | val | 7 | 1.304e-3 | 5.546e-4 | 1.862 | 0.739 |
| 未见泄漏率 `100 mL/min` | test | 7 | 8.600e-4 | 3.548e-4 | 2.050 | 0.932 |
| 未见泄漏位置 `(400,0,0)` | val | 7 | 6.439e-4 | 3.734e-4 | 3.521 | 3.894 |
| 未见泄漏位置 `(400,0,0)` | test | 7 | 6.521e-4 | 3.717e-4 | 1.379 | 0.899 |

训练数据量 repeated study 当前完成 `10/12` 个重复，缺少 `n=12,r=1` 和 `n=31,r=2`。服务器上两个缺失任务仍在运行：GPU2 正在跑 `n012_r01` 的 GPSD，GPU1 正在跑 `n031_r02` 的 GPSD。GPU0 和 GPU3 当前空闲，因为只剩两个不可再细拆的独立子任务。

已完成重复的中间统计如下。

| 训练工况数 | 完成重复数 | RMSE mean | RMSE std |
| ---: | ---: | ---: | ---: |
| 6 | 3/3 | 1.918e-3 | 2.197e-4 |
| 12 | 2/3 | 7.686e-4 | 8.521e-6 |
| 24 | 3/3 | 2.658e-3 | 2.597e-3 |
| 31 | 2/3 | 1.346e-3 | 6.020e-4 |

初步解释是：clean48 训练数据量与重建精度之间目前不是简单单调关系，`n=24` 存在明显高方差，说明模型成本实验必须报告 `mean ± std`，不能只报单次曲线。最终结论等待 12 个重复全部完成。

本地已同步以下文件：

- `docs/server_results_20260413/clean48_formal_summary.csv`
- `docs/server_results_20260413/clean48_train_scale_partial.csv`
- `docs/server_results_20260413/clean48_train_scale_partial_summary.csv`
- `docs/report_figures_20260413/clean48_formal_rmse_mae.png`
- `docs/report_figures_20260413/clean48_train_scale_partial_rmse.png`
- `docs/clean48_results_update_20260413.md`

`报告.md` 已追加第 9 节，写入 clean48 formal 指标和训练数据量当前进度。

## 2026-04-13 08:48 服务器重启后恢复

用户将服务器从 4 卡降为 2 卡并重启实例。重启后复查确认 `/hy-tmp/SDIFT_model56` 数据仍在，clean formal 四个 aggregate 文件仍完整存在；训练数据量 repeated study 已完成 `10/12`，缺失项仍为 `n=12,r=1` 和 `n=31,r=2`。重启导致所有训练进程中断，GPU0/GPU1 初始为空闲状态。

恢复前检查发现两个缺失任务的 GPSD 目录中只有 `ema_1000.pth` 和 `model_1000.pth`，没有最终步 checkpoint。为避免拿中断半成品进行评估，已将这两个旧 GPSD 目录移动到 `exps_interrupted_20260413/`，保留痕迹但不参与正式结果。FTM 的 core 和 basis 文件完整，因此无需重跑 FTM，只从 GPSD 阶段恢复。

恢复后的任务分配如下。

| GPU | 任务 | 状态 |
| ---: | --- | --- |
| 0 | `lowflow_focus_v1, n=12, repeat=1` | GPSD 已重新开始，日志为 `logs/clean48_resume_n012_r01_gpu0_20260413.log` |
| 1 | `lowflow_focus_v1, n=31, repeat=2` | GPSD 已重新开始，日志为 `logs/clean48_resume_n031_r02_gpu1_20260413.log` |

自动汇总脚本已重启：`scripts/remote_clean48_finalize_results_20260413.sh`。它会等待 `12/12` 个 `aggregate_metrics.json` 全部出现，然后生成最终训练数据量汇总文件到 `results/advisor_study/clean48_final_reports_20260413/`。

恢复后复查显示两张 GPU 均为 100% 利用率，目标日志未发现新的 `Traceback`、`FileNotFoundError`、`RuntimeError`、`CUDA out`、`Killed` 等错误。当前预计两个缺失任务还需要约 `2-3 h` 完成 GPSD 和后续 MPDPS 评估。

## 2026-04-13 11:28 clean48 训练数据量实验完成

服务器端 `train_scale_lowflow_focus_clean48_20260413` 已完成 `12/12` 个重复，最终汇总文件已经由 `remote_clean48_finalize_results_20260413.sh` 写入 `/hy-tmp/SDIFT_model56/results/advisor_study/clean48_final_reports_20260413/`。本地已同步最终 `CSV/JSON/Markdown`，并生成最终图表 `docs/report_figures_20260413/clean48_train_scale_final_rmse.png` 和 `docs/report_figures_20260413/clean48_train_scale_final_repeats.png`。

最终训练数据量结果如下。

| 训练工况数 | 重复次数 | RMSE mean | RMSE std | MAE mean | 低泄漏率 RMSE mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 3/3 | 1.918e-3 | 2.197e-4 | 1.015e-3 | 1.805e-3 |
| 12 | 3/3 | 7.361e-4 | 4.647e-5 | 4.628e-4 | 5.041e-4 |
| 24 | 3/3 | 2.658e-3 | 2.597e-3 | 1.212e-3 | 2.450e-3 |
| 31 | 3/3 | 1.180e-3 | 5.446e-4 | 5.840e-4 | 5.982e-4 |

结论：当前 clean48 配置下，训练数据量增加没有带来单调收益。`n=12` 的平均 RMSE 最低，`n=24` 出现明显高方差，`n=31` 较 `n=24` 恢复但仍未超过 `n=12`。这说明导师要求的“模型成本”问题目前不能写成线性提升或简单边际收益递减，而应写成“当前呈非单调波动，主要受训练稳定性、子集分布、低泄漏率样本难度和观测注入瓶颈共同影响”。

严格扫描当前正式日志 `clean48_rate_formal_eval_fix_20260413.log`、`clean48_position_formal_eval_fix_20260413.log`、`clean48_resume_n012_r01_gpu0_20260413.log`、`clean48_resume_n031_r02_gpu1_20260413.log`，未发现新的 `Traceback`、`FileNotFoundError`、`RuntimeError`、`CUDA out` 或 `Killed`。服务器旧日志中仍保留此前错误参数和 `cfd56` 阶段的 Traceback，但不属于当前 clean48 正式结果。

## 2026-04-13 n=24 高误差重复排查

对 `n=24` 的三个重复进行 case 级拆解后确认，高误差主要来自 `repeat=1`。该重复不是单个测试 case 爆掉，而是 7 个未见位置测试 case 全部系统性偏高：`repeat=1` 的 case 级 RMSE 基本集中在 `6.25e-3` 到 `6.50e-3`，而 `repeat=0/2` 通常在 `4.5e-4` 到 `1.35e-3`。

训练子集分布排查显示，`n=24` 三个重复的位置覆盖一致，均包含 5 个训练位置；泄漏率分布也接近。`repeat=1` 包含 5 个 `50 mL/min` 和 4 个 `100 mL/min` 样本，并不缺少低泄漏率。因此，不能将异常解释为“低泄漏率训练样本不足”或“某个位置没覆盖”。

质量积分诊断显示 `repeat=1` 的预测质量积分出现负均值和尺度偏移。7 个测试 case 平均后，`repeat=1` 的 `mean(pred mass)/mean(true mass)` 约为 `-1.178`，而 `repeat=2` 约为 `1.027`。这说明 `repeat=1` 的 GPSD/MPDPS 链路生成了与真实泄漏率不匹配的振荡场，低泄漏率 case 的相对误差更高只是分母较小导致的放大效应；全局 RMSE 异常在高低泄漏率上都存在。

已新增诊断文档和图表：

- `docs/n24_outlier_diagnostic_20260413.md`
- `docs/server_results_20260413/n24_case_diagnostics.csv`
- `docs/server_results_20260413/n24_mass_diagnostics.csv`
- `docs/report_figures_20260413/n24_case_rmse_outlier.png`
- `docs/report_figures_20260413/n24_mass_scale_diagnostic.png`

## 2026-04-13 10:57 训练进度复查

服务器仍在 `/hy-tmp/SDIFT_model56`，当前仅可见 2 张 L40S。`n=12,r=1` 已完成 GPSD、MPDPS 评估和指标聚合，已生成 `results/advisor_study/train_scale_lowflow_focus_clean48_20260413/lowflow_focus_v1_n012_r01/aggregate_metrics.json`。因此训练数据量 repeated study 已从 `10/12` 推进到 `11/12`。

当前唯一缺失项为 `n=31,r=2`。该任务正在 GPU1 上运行 GPSD，进度约 `3509/4000`，GPU1 显存约 `23.2 GB`，利用率约 `92%`。GPU0 当前空闲是正常现象，因为 `n=12,r=1` 已完成，且只剩一个不可再拆分的子任务。

自动汇总脚本 `remote_clean48_finalize_results_20260413.sh` 仍在运行，最近记录为 `aggregate_count=11`。当 `n=31,r=2` 生成最后一个 aggregate 后，会自动写出最终汇总到 `results/advisor_study/clean48_final_reports_20260413/`。

目标日志扫描仍为空，未发现新的 `Traceback`、`FileNotFoundError`、`RuntimeError`、`CUDA out`、`Killed` 等错误。预计剩余时间约 `25-40 min`，其中 GPSD 约 `20 min`，后续 MPDPS 评估和 aggregate 约 `5-20 min`。

## 2026-04-13 研究汇报边界与下一阶段规划整理

基于当前 `cfd48_clean` 正式结果，已确认现在可以向导师做阶段性汇报，但只能按“前两步任务已完成、第三步已给出明确路线”的边界来讲。可正式汇报的核心是 clean 数据闭环验证、未见位置与未见泄漏率定量结果、训练数据量多重复实验以及 `n=24` 异常的系统诊断；不可夸大为“严格尺寸泛化已完成”或“真实实验外推已完成”。

同时补写了后续执行版路线文档 [docs/cfd_training_plan_20260413.md](/d:/github/Hydrogen-leak/docs/cfd_training_plan_20260413.md)，内容包括同尺度基准数据池目标、通风/障碍物/尺寸扩展顺序、张量转换与原始数据删除规则、物理规律融合方向、真实实验数据融入方式以及后续训练与评估顺序。后续如果切换窗口或重新开始新会话，应优先读取 [docs/研究进展总览.md](/d:/github/Hydrogen-leak/docs/研究进展总览.md) 和该规划文档，再决定是继续训练还是先补 CFD 数据。
