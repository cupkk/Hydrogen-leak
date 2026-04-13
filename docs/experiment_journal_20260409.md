# Experiment Journal 20260409

## 2026-04-09 新增 CFD 数据本地分析与张量转换补记

今天在不依赖服务器的前提下，完成了新增原始 CFD 数据的本地结构分析、转换脚本修复、分批张量化和总数据集合并。新增数据根目录为 `E:\氢泄漏` 与 `F:\AI反演_CFD计算`。这批数据与前一版 38 组工况相比，主要新增了 `(0,0,0)` 与 `(100,0,0)` 两个位置，其中 `(0,0,0)` 已覆盖 `50, 100, 200, 400, 600, 800, 1000 mL/min` 共 7 组，`(100,0,0)` 已覆盖 `50, 100, 200, 400 mL/min` 共 4 组。

在正式转换前，先修复了 `SDIFT模型/build_cfd_multicase_dataset.py` 中两个会影响新数据接入的问题。第一，原始 `discover_case_dirs()` 会把同时包含压缩包和已解压目录的根目录误判为单个 case，导致 `F:\AI反演_CFD计算` 不能正确扫描到子工况目录；现已改为优先识别子目录 case。第二，原始 `parse_case_name()` 对新出现的五字段命名模式存在 `x/y` 解析歧义，如 `6,0,100,0,50` 实际应解释为 `6,y,x,z,rate`；现已修正为能够正确解析 `x/y/z` 与泄漏率。脚本修复后，新增数据的 case 发现与元数据提取已恢复正常。

经扫描，这批新增目录一共发现 18 个候选工况，其中可立即用于训练转换的完整工况为 11 组，另有 7 组暂时不能纳入。不能纳入的 7 组全部集中在位置 `(200,0,0)`，对应泄漏率 `50, 100, 200, 400, 600, 800, 1000 mL/min`。它们当前仍处于解压未完成或唯一可读时间帧不足的状态，尚不能满足统一截取前 120 个时间步的要求，因此本轮转换中被明确排除，没有混入训练张量。

新增 11 组完整工况采用与前面 38 组完全一致的转换参数：前 120 s、`48 × 48 × 48` 规则体素网格、`float32`、`IDW(k=8, power=2)`、统一物理坐标轴 `u∈[-0.5,0.5]`、`v∈[0,0.8]`、`w∈[-0.4,0.4]`。为了避免单次长跑超时，本次没有一次性整体转换，而是按 `3 + 3 + 3 + 2` 四个批次分别落盘，再统一合并。四个中间批次文件分别为：

- `SDIFT模型/data/cfd11_newbatch_b1.h5`
- `SDIFT模型/data/cfd11_newbatch_b2.h5`
- `SDIFT模型/data/cfd11_newbatch_b3.h5`
- `SDIFT模型/data/cfd11_newbatch_b4.h5`

最终合并后的新增数据主文件为：

- `SDIFT模型/data/cfd11_newbatch_T120_interp48.h5`
- `SDIFT模型/data/cfd11_newbatch_T120_interp48_meta.npy`
- `SDIFT模型/data/cfd11_newbatch_T120_interp48_manifest.csv`
- `SDIFT模型/data/cfd11_newbatch_T120_interp48_report.json`

核验结果表明，这 11 组新增张量的形状为 `11 × 120 × 48 × 48 × 48`，主 HDF5 文件体积约 `0.3769 GB`。全部样本 `frame_selection_mode` 均为 `integer_aligned`，`repaired_frame_count` 均为 `0`，说明这 11 组都属于无修复、整数秒对齐的干净样本，可直接用于后续训练或划分。

在此基础上，已将新增 11 组与现有 38 组合并，生成新的总训练张量：

- `SDIFT模型/data/cfd49_all_T120_interp48.h5`
- `SDIFT模型/data/cfd49_all_T120_interp48_meta.npy`
- `SDIFT模型/data/cfd49_all_T120_interp48_manifest.csv`
- `SDIFT模型/data/cfd49_all_T120_interp48_report.json`

合并后总形状为 `49 × 120 × 48 × 48 × 48`，主 HDF5 文件体积约 `1.8765 GB`。当前 49 组工况的空间覆盖已更新为：

- `(0, 0, 0)`：7 组，泄漏率 `50, 100, 200, 400, 600, 800, 1000`
- `(0, 100, 0)`：4 组，泄漏率 `50, 100, 200, 400`
- `(0, -150, 0)`：7 组
- `(100, 0, 0)`：3 组
- `(200, -150, 0)`：7 组
- `(200, -300, 0)`：7 组
- `(300, 0, 0)`：7 组
- `(400, 0, 0)`：7 组

这意味着现阶段数据集在同一箱体尺寸下的空间覆盖已经进一步扩展，但尺寸泛化问题仍未开始解决，因为新增 11 组依然属于同一空间尺度。

基于这次本地转换，下一步建议如下。第一，继续等待并跟踪 `F:\AI反演_CFD计算` 中 `(200,0,0)` 的 7 组工况完成解压，随后按完全相同的参数补转并并入 `cfd49_all`。第二，在观测注入机制修正之前，不建议立刻对 `49` 组数据发起新一轮正式大训练；更合理的做法是先用这份更完整的数据做 split 预案、低流量覆盖检查和尺寸数据清单规划。第三，后续若新增数据中出现不同箱体尺寸，应优先独立建 manifest 字段，把“尺寸泛化”从口头目标转成可执行实验。

## 范围

本文记录 `2026-04-09` 的结果确认与阶段判断，重点是：

- 当前最优绝对精度配置下的完整 sanity check 是否结束
- `correct / shuffled / zeros / wrong_positions` 四种输入是否被明显区分
- 当前是否还需要继续等待训练

## 一、完整 sanity check 已结束

本次检查的目标配置为当前冻结的“最优绝对精度配置”：

- `mpdps_weight = 16`
- `total_steps = 50`
- `num_posterior_samples = 2`
- `zeta = 0.03`

对应配置文件：

- `SDIFT模型/exps/mpdps_best_absolute_holdout400_stage2.json`

完整 sanity check 结果文件位于：

- `/hy-tmp/SDIFT_runs/advisor_study/mpdps_tuning_holdout400_stage2/w16_rho0.01_steps50_post02_z0.03/observation_sanity_check.json`

四种模式均已完成：

- `correct`
- `shuffled`
- `zeros`
- `wrong_positions`

本次核查时远端 GPU 已空闲，说明这轮任务已经完全结束。

## 二、四种输入模式结果

平均指标如下：

- `correct`
  - `global_rmse_mean = 0.0019759424491068963`
  - `global_rel_l2_mean = 6.344596685235556`
  - `global_rel_l1_active_mean_mean = 2.6161191807820634`

- `shuffled`
  - `global_rmse_mean = 0.001975994356933904`
  - `global_rel_l2_mean = 6.344528392828382`
  - `global_rel_l1_active_mean_mean = 2.616397205024518`

- `zeros`
  - `global_rmse_mean = 0.0019763443403679284`
  - `global_rel_l2_mean = 6.344697818886994`
  - `global_rel_l1_active_mean_mean = 2.616192693362143`

- `wrong_positions`
  - `global_rmse_mean = 0.001976928530745975`
  - `global_rel_l2_mean = 6.347862534646759`
  - `global_rel_l1_active_mean_mean = 2.617931247450881`

## 三、当前结论

结论已经足够明确：

1. 当前最优绝对精度配置确实能继续降低重建误差。
2. 但四种观测输入的结果仍然极其接近。
3. `zeros` 和 `wrong_positions` 并没有显著破坏反演结果。
4. 因此可以正式坐实：当前问题不在于 `MPDPS` 超参数还没扫够，而在于观测注入机制本身太弱。

这意味着后续不应优先继续重复：

- `mpdps_weight / zeta / total_steps / posterior_samples` 的同类微调

而应转向：

- 诊断观测梯度项与扩散更新项的量级关系
- 检查时间核传播与基函数投影是否把观测残差过度平滑
- 评估是否需要归一化或自适应放大观测项
- 必要时改成更直接的条件注入方式

## 四、当前是否还需要继续等待

不需要。

截至本日志写入时，这一轮完整 sanity check 已经结束，当前没有相关训练继续运行。下一阶段属于“机制定位与代码修改”，而不是继续等待同一轮训练出结果。

## 五、下一步

下一步最合理的是：

1. 用新增诊断接口在单个 case 上分别跑 `correct / shuffled / zeros / wrong_positions`
2. 读取每一步的：
   - `llk_grad_norm`
   - `diffusion_update_norm`
   - `obs_update_norm`
   - `obs_over_diffusion_ratio`
3. 判断观测项是被量级压制，还是被时间传播/基函数结构冲淡
4. 基于诊断结果修改 `message_passing_DPS.py` 的观测注入方式

## 观测注入机制改造（本地代码）

基于前序单 case 诊断，已经确认当前 `MPDPS` 的观测更新量通常只有扩散更新量的 `0.1%` 左右，最大也只有约 `2%`，因此问题不再归因于超参数未扫够，而归因于 `message_passing_DPS.py` 中观测项注入机制本身过弱。围绕这一判断，今天已在本地完成观测注入链路改造，并保持与旧实验的兼容。

本次改造的核心文件为 `SDIFT模型/message_passing_DPS.py`。旧版固定采用 `obs_update = zeta/(i+1) * llk_grad` 的单次轻微后验修正；新版在保留 `legacy` 模式的同时，新增了两种可切换模式：一是 `adaptive_ratio`，即根据当前步扩散更新范数与观测梯度范数，自动计算目标观测缩放因子，使观测修正量能够达到设定的相对比例；二是 `direct_inner`，即在单个扩散步内部执行多次观测修正，并在内层循环中重新用当前隐变量通过去噪网络估计浓度场 core，再次计算观测梯度，实现更直接的条件注入，而不再局限于一次性的小幅后验修正。

为支撑后续定位机制问题，本次改造还增加了更完整的诊断量记录。每个扩散步现在都会额外记录 `obs_injection_mode`、`obs_base_scale`、`obs_target_scale`、`obs_inner_scales`、`obs_inner_grad_norms`、`obs_inner_update_norms`、`obs_inner_total_update_norm` 等信息，并保留 `llk_grad_norm`、`diffusion_update_norm`、`obs_over_diffusion_ratio` 等关键指标。这样后续在服务器上重跑单 case 诊断时，可以直接判断自适应放大和内层直接修正是否真正把观测项拉到了与扩散项可比的量级。

外围实验脚本也已同步更新，包括 `run_holdout_reconstruction_eval.py`、`run_observation_sanity_check.py`、`run_mpdps_observation_tuning.py` 和 `run_mpdps_diagnostic_case.py`。这些脚本现在都支持透传新的观测注入参数：`obs_injection_mode`、`obs_scale_schedule`、`obs_scale_blend`、`obs_target_ratio`、`obs_min_scale`、`obs_max_scale`、`obs_inner_steps` 和 `obs_inner_decay`。这意味着后续无论是单 case 诊断、完整 sanity check，还是重新跑 holdout 评估，都不需要再手工改脚本内部逻辑。

为方便下一轮服务器实验，本地新增了候选配置文件 `SDIFT模型/exps/mpdps_direct_inner_candidate_holdout400.json`，其默认建议为：`obs_injection_mode = direct_inner`、`obs_scale_schedule = constant`、`obs_scale_blend = replace`、`obs_target_ratio = 0.25`、`obs_inner_steps = 3`、`obs_inner_decay = 0.7`。这份配置不是最终结论，而是为下一轮“观测项是否真正生效”的服务器诊断准备的起点。

本次改造完成后，已在本地执行了 `py_compile` 语法校验，并用 `run_mpdps_observation_tuning.py --dry_run` 做了参数通路检查。结果表明新参数已经能够从调参脚本一路传递到 `message_passing_DPS.py`，当前代码状态可直接用于下一轮服务器侧机制验证。

## direct_inner 服务器单 case 诊断结果

本日随后已将新版 `message_passing_DPS.py` 与外围脚本同步到服务器，并基于当前正式训练资产启动了 `direct_inner` 机制诊断。使用的模型与基函数保持不变，仍采用 `holdout_400_0_0` 的正式 `FTM/GPSD` 结果，仅替换观测注入机制。诊断配置为：`mpdps_weight = 16`、`total_steps = 50`、`num_posterior_samples = 2`、`zeta = 0.03`、`obs_injection_mode = direct_inner`、`obs_scale_schedule = constant`、`obs_scale_blend = replace`、`obs_target_ratio = 0.25`、`obs_inner_steps = 3`、`obs_inner_decay = 0.7`。实验对象仍取两个代表 case：低泄漏率 `case_0033 (50 mL/min)` 与高泄漏率 `case_0027 (1000 mL/min)`，并分别测试 `correct / shuffled / zeros / wrong_positions` 四种输入。

结果显示，观测项量级已经被成功抬升。此前旧机制下 `obs_over_diffusion_ratio` 平均仅有 `0.0013 ~ 0.0015`；在 `direct_inner` 配置下，两组 case 的该比值平均值均提升到约 `0.42`，最大值约 `0.547`。这说明观测更新量已经不再是扩散更新量的千分之一量级，而是进入了同一数量级的可比范围。从数值机制上看，这一步已经解决了“观测项根本压不进去”的核心问题。

高泄漏率 `case_0027` 的区分效果已经比较清楚。`correct` 模式下：
- `global_rmse = 0.0015699`
- `global_rel_l2 = 0.8452`
- `global_rel_l1_active_mean = 1.0940`

而相同 case 下：
- `shuffled`: `global_rel_l2 = 0.9516`
- `zeros`: `global_rel_l2 = 1.1708`
- `wrong_positions`: `global_rel_l2 = 1.1532`

这意味着在高流量工况上，`correct` 已经明显优于打乱、清零和错误位置输入，说明新的观测注入机制开始真正利用传感器信息，而不再像旧版那样四种输入几乎重合。

低泄漏率 `case_0033` 的情况则更复杂。`correct` 模式下：
- `global_rmse = 0.0010344`
- `global_rel_l2 = 11.4166`
- `global_rel_l1_active_mean = 9.0418`

对比可见：
- `shuffled`: `global_rel_l2 = 11.4355`
- `zeros`: `global_rel_l2 = 11.7200`
- `wrong_positions`: `global_rel_l2 = 20.8192`

这里已经能明显区分 `wrong_positions`，也能看出 `zeros` 较差，但 `correct` 与 `shuffled` 的差距仍然很小。这说明机制层面已经不是完全失效，但在低泄漏率工况上，观测信号仍然较弱，尚不足以稳定拉开与“打乱观测”的差距。换言之，`direct_inner` 已经把问题从“观测项完全无效”推进到了“高流量有效、低流量仍然偏弱”的阶段。

基于这一轮服务器结果，当前最合理的结论是：新版观测注入机制已经值得继续向正式实验推进，不需要回退到旧版超参数扫描；但在重新启动导师要求的整套正式曲线前，仍应先做一次完整 holdout `7` 个 case 的 sanity check，确认这种改进不是只在两个代表 case 上偶然成立。如果完整 holdout 上也能保持 `correct` 平均优于 `shuffled / zeros / wrong_positions`，则可以正式重启“传感器数量 / 观测时长影响”和“训练数据量多次重复分层实验”。若低泄漏率子集仍无法拉开，则下一步应优先增强低流量样本权重，并将 `50/100 mL/min` 工况作为单独子集持续监控。

## 补充数据纳入顺序与时间估算

本日进一步核对了补充数据目录。`E:\氢泄漏` 当前为位置 `(0,0,0)` 的完整 7 组泄漏率工况；`F:\AI反演_CFD计算` 当前已完整解析为 11 组，其中 `(100,0,0)` 有 `50/100/200/400 mL/min` 共 4 组，`(200,0,0)` 有 `50/100/200/400/600/800/1000 mL/min` 共 7 组。此前未完成转换的 `(200,0,0)` 那 7 组现在已经具备纳入总数据池的条件。若将其按既定参数全部转成张量并并入现有 `cfd49_all`，则总样本池可从 `49` 组提升到 `56` 组。

这里需要明确区分“全部纳入总数据池”和“全部用来训练”。当前建议是：新增数据应全部纳入 master dataset 和新的 manifest/split 设计，但不应全部直接塞进 train split。原因有三。第一，如果把所有位置都放入训练集，就无法再保留“未见位置”测试，导师最关心的泛化验证会失去证据基础。第二，当前 `direct_inner` 机制刚刚在两个代表 case 上验证出有效趋势，若此时同时更换训练数据版本和训练集划分，将无法判断后续指标变化究竟来自机制改进还是来自数据扩充。第三，后续若补入不同尺寸数据，这些尺寸本身就应该作为保留测试条件，而不应该在初轮中全部泄漏进训练集。

按当前最合理的顺序，后续前三步的时间估算如下。第一步是在现有正式训练资产上，用新版 `direct_inner` 机制完成完整 `holdout_400_0_0` 的 7-case sanity check；该步骤不需要重训主模型，在 1 张 L40S 上大约需要 `1 ~ 1.5` 小时。第二步是将 `(200,0,0)` 这 7 组补充数据转换并并入总数据池，同时重建新的 split；这一步主要是本地 CPU/磁盘工作，预计 `1 ~ 2` 小时。第三步是在新的 `56` 组数据划分上重训正式基线模型（`FTM + GPSD + sensor 参数基线 + holdout 评估`）；若仍采用当前量级的训练配置，在 1 张 L40S 上大约需要 `2 ~ 3` 小时。换言之，如果只完成这三步而不展开整套导师曲线，1 张 L40S 当天内可以完成。

真正耗时的是后面的正式曲线实验。其中“传感器数量 / 观测时长”实验约为 `9` 个条件，每个条件都要跑完整 holdout 反演，1 张 L40S 预计需要 `2 ~ 3` 小时；而“训练数据量 6/12/24/31 的多次重复分层实验”若按 `4 个规模 × 3 次重复 × 3 种 sample_weight_mode` 计算，共有 `36` 组独立训练任务，每组都包含 `FTM + GPSD + holdout 评估`，在 1 张 L40S 上总体大约需要 `36 ~ 54` 小时。由于当前代码并未做单任务多卡并行，额外租卡的价值主要体现在“并行跑多个独立实验”，而不是加速单个训练任务。

因此，若目标是最短墙钟时间，较合理的配置是租 `4` 张 L40S：它足以把 `36` 组独立训练任务并行摊开到大约 `9 ~ 14` 小时内完成，是当前代码结构下性价比较高的选择。若预算充足、目标是尽量压到一个夜间窗口内完成，则可以考虑 `8` 张 L40S，预计可将这部分时间继续压缩到 `5 ~ 8` 小时；但再往上增加卡数，收益会明显下降，因为瓶颈开始转向磁盘 I/O、脚本调度和评估串行部分。若只打算先完成机制确认和新的正式基线，`1` 张 L40S 已经足够，不必一开始就上多卡。

## 56-case 总数据池与 split 已完成

本地补充数据 `(200,0,0)` 的 7 组工况已经完成张量化，并与既有 `49` 组数据合并成新的 `56-case` 总数据池。合并后的主文件为 `SDIFT模型/data/cfd56_all_T120_interp48.h5`，对应 manifest 为 `SDIFT模型/data/cfd56_all_T120_interp48_manifest.csv`，数据形状为 `56 × 120 × 48 × 48 × 48`。新增部分沿用与既有数据一致的转换参数：前 `120` 个时间步、`48^3` 规则体素网格、`IDW(k=8, power=2)` 插值、统一物理坐标轴 `u∈[-0.5,0.5]、v∈[0,0.8]、w∈[-0.4,0.4]`。所有用于转换的临时分片目录和中间日志已经清理，仅保留最终 `cfd56_all_T120_interp48.*` 和正式 split 文件。

基于这套 `56-case` 数据，三分割方案已经重建完成：`test = (400,0,0)`，`val = (200,0,0)`，其余位置为 train。最终切分结果为 `42 / 7 / 7`，分别对应 train、val、test。这个划分既保留了未见位置泛化验证，也给新补充的 `(200,0,0)` 工况单独预留了验证位，避免把所有新数据都泄漏进训练集。

这一步之后，后续正式训练可以直接基于 `cfd56_all_T120_interp48.h5` 以及 `data/splits/holdout_400_0_0_val_0200/` 下的 `train/val/test` 文件展开，不需要再回头拼接旧的临时 part 文件。
## 56-case 正式 4 卡训练启动

2026-04-09 晚间，已将 `56-case` 正式任务通过“个人数据 OSS -> 服务器数据盘 `/hy-tmp`”链路部署到新工作目录 `/hy-tmp/SDIFT_model56`，并启动 4 卡并行训练脚本 `run_56case_advisor_tasks_4gpu.sh`。当前使用的数据与划分为：

- 总数据：`cfd56_all_T120_interp48.h5`
- 划分：`train = 42`，`val = 7`，`test = 7`
- 训练主线：`FTM + GPSD + sensor 参数基线 + val/test 全场反演评估`
- 机制配置：`mpdps_weight=16`，`obs_rho=0.01`，`zeta=0.03`，`obs_injection_mode=direct_inner`

启动后确认后台进程已建立。当前 `GPU0` 先进入正式基线训练的 `FTM` 阶段，`GPU1~3` 正在分别构建训练数据量重复实验所需的分层子集；这一阶段主要消耗 CPU 和磁盘 I/O，因此会出现“只有 0 号卡先吃满、1~3 号卡暂时空闲”的现象，属于正常行为，不是脚本卡死。

按当前脚本结构估算，本轮总墙钟时间大约为 `10 ~ 14` 小时。瓶颈不在 0 号卡的正式基线，而在 `GPU1~3` 上三组 `12` 次重复训练规模实验。若中途不报错，预计可以在一夜内完成。当前最关键的远端查看命令为：

```bash
ssh -p 63260 root@i-1.gpushare.com
tail -f /hy-tmp/SDIFT_model56/logs/advisor_56case_4gpu.nohup.log
```

分任务日志：

```bash
tail -f /hy-tmp/SDIFT_model56/logs/cfd56_formal_pipeline_gpu0.log
tail -f /hy-tmp/SDIFT_model56/logs/cfd56_scale_none_gpu1.log
tail -f /hy-tmp/SDIFT_model56/logs/cfd56_scale_focus_gpu2.log
tail -f /hy-tmp/SDIFT_model56/logs/cfd56_scale_balanced_gpu3.log
```

GPU 状态：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

## 2026-04-10 早间进度核查

截至 2026-04-10 上午，`56-case` 正式任务已出现分化结果。正式基线主线 `GPU0` 已经完成一轮完整产出，关键文件已落盘在：

- `exps/gp-edm_holdout400_cfd56_cfd56_holdout400_val0200_train_20260409-2213/val_eval/aggregate_metrics.json`
- `exps/gp-edm_holdout400_cfd56_cfd56_holdout400_val0200_train_20260409-2213/test_eval/aggregate_metrics.json`
- `results/cfd56_holdout400_val0200_sensor_param_baseline/sensor_param_baseline.json`

其中 `test_eval` 已成功汇总完成，说明“正式基线模型训练 + test holdout 反演评估 + sensor 参数基线”这一条主链已经跑通。与此同时，传感器数量/观测时长实验最初失败，原因不是模型问题，而是脚本仍引用 `data/sensors_real_12.csv` 却要求生成 `30` 个传感器子集，导致 `make_nested_sensor_subsets.py` 报错 “requested 30 sensors but only 12 are available”。该问题随后已修正为使用 `data/sensors_real.csv`（33 点），并已在空闲的 `GPU0` 上重新启动。

训练数据量多次重复实验方面，`GPU1~GPU3` 当前正在并行执行三种权重模式：

- `none`
- `lowflow_focus_v1`
- `lowflow_balanced_v1`

从日志与 checkpoint 落盘情况看，三路都已经完成：

- `n=6` 的 `r00/r01/r02`
- `n=12` 的 `r00/r01/r02`

并已进入：

- `n=24, r00`

的 `GPSD` 阶段，当前大约推进到 `55%~58%`。这意味着三路重复实验已超过一半，但距离全部 `12` 组任务完成仍有明显剩余。按当前速度估算，三路训练规模实验还需约 `12~16` 小时。

需要额外记录一个风险点：`val_eval/aggregate_metrics.json` 中 7 个验证 case 的指标完全相同（`global_rmse ≈ 0.57698`，`global_rel_l2 ≈ 0.57698`），这一现象高度可疑，不符合不同泄漏率工况应有的差异性。因此当前对 `val_eval` 不宜直接下结论，后续应优先核查 `val` 路径中的 sample index 或聚合映射逻辑；相对而言，`test_eval` 的 7 个 case 指标是分开的，暂时更可信。

## 2026-04-10 上午二次核查（显卡调整后）

用户反馈远端“关闭了一张显卡”后，再次检查训练状态。系统层面 `nvidia-smi` 仍然显示 4 张 `L40S` 可见，因此从操作系统视角看，这次调整并未把 GPU 数量真正降为 3。当前实况为：

- `GPU1~GPU3` 正在继续三条训练规模重复实验，并都已进入 `n=24, r00` 的 `GPSD` 阶段，单路进度约 `55%~58%`
- `GPU0` 之前用于正式基线与 test/val 评估，现已空出，并被重新分配给“传感器数量 / 观测时长”实验

这一轮复查确认了三件事。

第一，`56-case` 正式基线主链已经完整结束并落盘，不再需要重训主模型。正式基线 `test` 汇总结果为：

- `global_rmse = 0.005997`
- `global_mae = 0.004019`
- `global_rel_l1_active_mean = 29.463`
- `global_rel_l2 = 19.190`
- `mass_mean_rel_error = 6.553`

其中低泄漏率子集 `50/100 mL/min` 更差：

- `global_rel_l2 = 48.918`
- `global_rel_l1_active_mean = 70.005`
- `mass_mean_rel_error = 16.761`

这再次说明当前模型在低泄漏率工况上仍然明显偏弱。

第二，源参数 `sensor` 基线也已完成，当前 `test` 集上的结果为：

- 源位置平均误差 `214.29 mm`
- 泄漏率 `MAE = 34.94 mL/min`
- 泄漏率平均相对误差 `10.19%`

第三，传感器条件实验之前的失败原因已被确认并修正。此前脚本错误地用 `data/sensors_real_12.csv` 去生成 `30` 个传感器子集，导致直接报错；现已改为 `data/sensors_real.csv`（33 点）重启。因此，当前“正式基线”已完成，“训练规模重复实验”仍在进行，而“传感器数量 / 观测时长”实验已重新接上。

按这一时点重新估算，若不再出现新的中断，剩余墙钟时间大致为：

- 传感器条件实验：约 `2 ~ 4` 小时
- 三条训练规模重复实验：约 `10 ~ 14` 小时

因此全部导师任务更现实的完工窗口应为“今晚到明早”之间，而非上午内全部结束。
