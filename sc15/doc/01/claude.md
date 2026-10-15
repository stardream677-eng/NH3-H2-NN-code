审查完成。我读完了全部 10 个脚本（约 1.5 万行），并用 Cantera 环境对几处关键策略做了数值验证。下面是结论。

总结论
代码工程质量较高，语法、导入、路径解析、断点续训都没问题。但在策略层面有三处会直接决定模型能否用于 CFD 的硬伤，其中一处是 v69 方案新引入的，会让 QSSA 分支的训练目标自相矛盾。

严重度	问题	位置
P0	QSSA 阻尼上限 max_damping=15 让积分器无法复现数据，速率损失与状态损失直接冲突	train_sc_v68.py:2948-2957
P0	LHS 的 N2 独立采样，真实"燃料+空气"混合气在训练集中几乎为零	lhs_sampler_sc_v11.py:193-196
P0	2 ms 平衡判据把慢着火工况当平衡截断，低温高压诱导期从未进入数据	runner_sc_v21_qssa.py:62-64
P1	温度始终用真值回灌，pure-AR 指标对 CFD 部署是乐观的；autonomous 模式热力学闭合用 1000 K 常数系数	train_sc_v68.py:4394
P1	P_avg/D_avg 取端点算术平均，着火行上即使不加 cap 也无法精确复现 Y_next	runner_sc_v21_qssa.py:443-444
P1	trust_region_fraction、experiment_mode、rate_z_limit 在状态更新里全是空操作，v34/v68 预设的消融对比无意义	train_sc_v68.py:2902-3045
P1	QSSA 预设关掉了全部守恒项，且 qssa 更新路径从不调用已有的 Dykstra 投影	train_all_v68.py 预设
P2	QSSA 模式下 scaler 文件名和校验用的是 legacy 签名，同一 output_root 第二次运行会抛 "Refusing incompatible scaler transform"	train_sc_v68.py:5027-5067
P2	log10 目标下限 -20 形成尖峰，标准化后 MSE 对物理区间不敏感	transform_target_array
P2	conservative_simplex 对速率头做 softmax，语义不通，属死代码	train_sc_v68.py:2395
P0 细节与证据
1. 阻尼上限摧毁了 QSSA 积分器。 runner 生成的 P、D 使得精确指数积分 Y·e^(-D·dt) + P(1-e^(-D·dt))/D 能复现 Y_next。但训练时把 D 截到 15/dt。对刚性物种这是物理常态而非异常：

条件	活跃物种中 D·dt>15 的比例
1200 K、40 atm、φ=1 轨迹上的 OH / H / O / HO2 / NH2	83% / 96% / 91% / 78% / 80%
库中现有 4 条样本轨迹	99%
截断后 Y_next ≈ P·dt/15，而真实准稳态值是 P/D，二者相差 D·dt/15 倍，可达 10³ 到 10⁶。用真值 P、D 反算 Y_next 的对数误差：

| 积分方式 | 活跃项 |Δlog10 Y| 中位数 | 偏差 >1 dex 比例 |
|---|---|---|
| 精确指数积分 | 5.7e-13 | 0.2% |
| cap=15 | 1.63 | 61% |

这意味着网络要同时满足"logP 逼近真值"和"状态逼近真值"是不可能的，二者必然打架。v69 计划 Task 1.4 想解决的"过阻尼塌陷"是 D 被高估的问题，正确的做法是改参数化，例如直接预测 log(P/D) 即准稳态值和 log D，让渐近值成为网络输出，而不是截 D·dt。

2. 训练集里几乎没有空气配比的混合气。 采样器先独立采 X_N2，再用燃料和 O2 填余量。化学计量 NH3-空气的 X_N2 约 0.62，低于 Engine_Core 下限 0.70。我用采样器生成 2 万点统计：

分层	N2/O2 中位数	空气配比样本占比
Spec_Engine_Core	12.3	0.3%
Spec_Cold_Oxidation	1046	0.7%
Spec_Radical_Burst	15.3	5.6%
Engine_Core 且 φ≥0.7 且空气配比	无	0.0%
空气 N2/O2 是 3.76。整套数据实际是 65% 到 99% EGR 稀释工况，绝热火焰温度、自由基池、着火延迟都与真实发动机工况系统性不同。N2 应由 φ 和稀释率推导，不该独立采样。

3. 平衡判据截断慢着火。 判据是 t>2 ms 后连续 3 步 |dT/dt|<1 K/s 且 max|dY/dt|<0.01。低温高压工况在 2 ms 时速率远低于阈值，但远未着火：

| 工况 | 着火延迟 | 2 ms 时 max|dY/dt| | 结果 |
|---|---|---|---|
| 800 K、100 atm | 0.30 s | 2.4e-5 | 2.09 ms 时按"平衡"终止，T 仍 800 K |
| 850 K、60 atm | 0.086 s | 1.6e-4 | 同样会被截断 |

判据把"冻结的诱导期"误判为"平衡"。再叠加 Cold_Oxidation 占 30% 且 T 为 300 到 800 K，大约一半数据是"2 ms 内什么都没发生"。建议对比 Cantera 平衡态组分或绝热温度来判定平衡，诱导期工况应跑到估算 IDT 的若干倍。

训练与部署策略的其他问题
温度闭合缺失。 rollout 每步从真实轨迹取 T_next，模型只预测物种。0D 验证里着火时刻由真实 T 泄露给模型，pure-AR RMSE 偏乐观。autonomous 模式用 1000 K 处的常数 u 和 cv，跨 1500 K 的着火会偏差数百 K，每步 ΔT 还被夹到 250 K。定容反应器的正确闭合是 u(T_next, Y_next)=u(T, Y)，NASA7 多项式在 torch 里可微实现并不难。任何 QSSA 预设都没开这个模式，耦合回归从未被训练或验证过。
目标定义本身不自洽。 端点算术平均的 P、D 在着火行上无法复现 Y_next。1200 K/40 atm 轨迹的瞬态行上，精确指数积分的 |Δlog10 Y| p90 为 0.23 dex，最大 4.3 dex。最需要学准的行恰是目标最不自洽的行。可考虑让 D 由 Y、Y_next 和 P 的时间积分反解，保证积分器按构造精确。
多个策略开关是空操作。 predict_next_state 收到 trust_region_fraction 和 experiment_mode 后只回写到返回字典，从不改变更新。v34 和 v68 预设声称的 0.25 信赖域不存在。validate 脚本却按 experiment_mode 推导 update/feedback 约束，训练与验证语义不一致。
守恒。 QSSA 更新对每个物种独立，Σ(P_k - D_k·Y_k) 不为零，预设里 lambda_element、lambda_mass、lambda_positive 全为 0。project_physical_state 的 Dykstra 投影已写好，但 qssa 路径不调用。
反应器类型。 数据用定容反应器。若目标 CFD 的化学子步是定压假设，大 dt 跨着火时物种增量的热力学路径不同，会有系统偏差。需要和目标求解器对齐。
指标与选模。 按 oracle 总损失选模和调学习率。标准化 RMSE 被每条轨迹约 190 行的平衡段主导，着火错相在指标上不显著，这与 v69 计划诊断的"假着火"一致。建议加 IDT 误差和自由基峰值时刻误差。
目标分布。 库样本中 57% 的 logP、46% 的 logD 和 68% 的 log10 Y 落在 -20 下限。缺失物种的 D=Dest/Y 本身不适定。应对缺失物种屏蔽损失。
NUIG 机理第 123 号反应 N+NO<=>N2+O 的 A 因子为负。 真实工况下未产生负的生成/消耗率，但库中那 4 条 10⁵ atm 的测试轨迹里 46% 的 D 为负，被 clip 静默映射到 -20。runner 没有检查。
现状与环境
磁盘上唯一的 QSSA 库在 sc14/07/01 下，只有 4 条轨迹，且输入 CSV 的 P 以 Pa 写入却被当作 atm，等效 101325 atm，属退化测试数据。1.9 亿行的完整库是旧 V21 格式，位于 D:\zmwj\sc01\06，没有 P_avg 和 D_coef_avg 列。QSSA 训练需要全量重新生成。
环境 Cantera 3.0.1 可加载 3.1.0 生成的 yaml，torch 为 2.11 nightly 带 CUDA，GPU 可用。所有脚本语法与导入正常。
建议的处理顺序
去掉或改参数化 max_damping，这一项最便宜且影响最大。
重做采样器的组分逻辑和 runner 的终止判据，再全量重生成 QSSA 库。
补上可微的能量闭合，把耦合 rollout 纳入验证。
清理空操作开关，否则消融结论不可信。
你现在可以说说遇到的具体问题，上面几条很可能已经覆盖了根因。



Write C:\Users\123\.claude\projects\C--Users-123\memory\project_nh3_h2_nn_surrogate_sc15.md
26 lines
---
name: project-nh3-h2-nn-surrogate-sc15
description: NH3/H2 combustion NN surrogate project (V68/V69 QSSA) in Desktop/2STest/second chapter/009/xl1/sc15; key strategy findings and data locations
metadata:
  type: project
---

Project: neural operator replacing NH3/H2 chemical mechanism in CFD. Code in
`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc15` (earlier iterations in sibling `sc14/07/0x`).
Pipeline: lhs_sampler_sc_v11 -> runner_sc_v21_qssa (Cantera 0D const-V, QSSA P/D targets) ->
split_trajectories_v21 -> train_sc_v68 (MLP, species-AR rollout, external recorded T/P/dt) -> validate_sc_v68.
Env: `C:\Users\123\.conda\envs\cantera\python.exe` (Cantera 3.0.1, torch 2.11 nightly cu130, GPU).

Data locations (as of 2026-10-09): full 190M-row V21 database (no QSSA columns) at `D:\zmwj\sc01\06`;
only QSSA database on disk is a 4-trajectory degenerate smoke test in `sc14/07/01/database_Transient_V21_QSSA`.

Review findings (2026-10-09), see [[technique-focal-loss-v33]] for unrelated earlier work:
- `max_damping=15` cap in QSSA predict_next_state makes targets inconsistent (D*dt>15 is normal for radicals).
- LHS samples X_N2 independently; air-like NH3/H2-air mixtures ~0% of training data.
- Equilibrium stop at 2 ms truncates slow-ignition (800-900 K, high P) cases before ignition.
- trust_region_fraction / experiment_mode are no-ops in the state update.
- QSSA runs reuse legacy TRANSFORM_SIGNATURE for scaler filenames -> second run in same output_root raises.

**Why:** user asked for a strategy review before describing their actual problem; these are the likely root causes.
**How to apply:** when the user reports QSSA training/rollout issues, check these first before deeper debugging.
Bash Add memory index entry for the project review
IN
cd "/c/Users/123/.claude/projects/C--Users-123/memory" && printf '%s\n' '- [NH3/H2 NN Surrogate sc15 Review](project_nh3_h2_nn_surrogate_sc15.md) — Pipeline layout, data locations, and the P0 strategy findings (damping cap, LHS N2 sampling, 2 ms equilibrium cutoff)' >> MEMORY.md && tail -5 MEMORY.md

OUT
- [Cantera Conda Environment](reference_cantera_env.md) — Conda environment path for Cantera-based combustion simulations
- [Regression Focal Loss V33](technique_focal_loss_v33.md) — Breakthrough technique achieving +26.4% average improvement for trace species prediction
- [IVC Temperature Fix with CP Equation](fix_ivc_temperature_cp_equation.md) — Successfully fixed IVC temperature using cp energy equation, improved from +138K error to -13K
- [NH3/H2 NN Surrogate sc15 Review](project_nh3_h2_nn_surrogate_sc15.md) — Pipeline layout, data locations, and the P0 strategy findings (damping cap, LHS N2 sampling, 2 ms equilibrium cutoff)
审查报告已在上一条消息中完整给出，审查结论也已存入记忆以便后续对话直接引用。

简要回顾：代码工程层面没有阻塞问题，但策略上有三处 P0 硬伤，分别是 QSSA 积分器的阻尼上限让训练目标自相矛盾、LHS 采样几乎不覆盖空气配比混合气、以及 2 ms 平衡判据截断慢着火工况。另有温度闭合缺失、多个策略开关为空操作、QSSA 预设关闭全部守恒项等 P1 问题。

接下来请描述你实际遇到的问题，我会结合上面的发现定位根因。如果你愿意，也可以告诉我目标 CFD 求解器的化学子步假设是定容还是定压、以及实际关心的工况范围（T、P、φ、EGR 率），这两点会直接决定采样器和 runner 的修改方向。


震荡和发散的根因已经定位，而且用你的 v70 checkpoint 做了可复现的实验验证。结论是：两者是同一个根因，即 QSSA 积分器里的阻尼上限 max_damping=15 与每步随机采样的 dt 相互作用。它不是网络容量或正则化的问题，v69 方案的噪声注入、平滑损失、微观加权都治不了它。

震荡（恒定浓度上的锯齿）
机制。 平台区的真实状态几乎不变，每一步输入里唯一显著变化的量是 log_dt，它在 1e-8 到 1e-4 之间随机抽样。积分器在 D·dt>15 时把 D 截成 15/dt，于是 Y_next ≈ P·dt/15。要让平台区的 Y_next 等于 Y，网络必须输出 P ∝ 1/dt；要让速率损失小，网络又必须输出 P = P_true，与 dt 无关。它学到的是折中，所以 Y_next 随 dt 单调漂移。随机 dt 序列就被直接印成了锯齿。

证据一，oracle 误差统计（从你保存的 trajectory_states.csv 算出）。 平台区单步 log10 误差的符号翻转率 0.67 且 lag-1 自相关约 0，这是每步独立重抽样才有的白噪声特征，不是模型状态漂移。

物种	平台区误差 std（dex）	lag-1 自相关	符号翻转率
OH	0.40	0.07	0.675
O	0.53	0.11	0.665
HO2	0.43	0.09	0.667
NNH	0.49	0.04	0.678
H2O	0.02	0.02	0.671
证据二，固定状态只扫 dt（Cantera 平衡态，2542 K，79.5 atm）。 物理上 Y_next/Y 应恒为 1。模型输出：

物种	dt=1e-8	1e-7	1e-6	1e-5	1e-4	跨度（dex）
OH	+0.36	+0.40	+0.50	+0.86	+1.14	0.78
HO2	-0.18	-0.19	+0.33	+0.60	+0.83	1.02
NH3	+0.01	+0.14	+0.34	+1.07	+1.77	1.77
NO	0.00	0.00	0.00	-0.01	-0.03	0.03
NO 的 D 只有 1e5，D·dt 全程不超过 15，不受截断，所以完全无震荡。这与图里 NO、H2O、N2 平滑而自由基锯齿的现象一一对应。

证据三，随机 dt 与固定 dt 的 oracle 对比（128 步平台区）。 随机 dt 下 OH 误差 std 为 0.28 dex，固定 dt 为 1e-7、1e-6、1e-5 时均为 0.00。锯齿完全消失，只剩一个随 dt 变化的常值偏差。

发散（长程 pure-AR）
速率头已经不是速率。 同一平衡态下，log10(P_pred/P_true) 跟着 log10(15/(D·dt)) 走，斜率约 0.8：

物种	dt	D_true·dt	log10(P_pred/P_true)	log10(15/(D·dt))
H	1e-6	3.1e2	-0.78	-1.32
H	1e-4	3.1e4	-2.42	-3.32
HO2	1e-4	1.2e4	-2.08	-2.91
NO	任意	<12	-0.2 到 +0.1	0
这解释了训练日志里 rate loss 从第 2 个 epoch 起就卡在 0.38 到 0.42 再也不降：目标自相矛盾，网络无解。闭环里每一步都带着这个随 dt 变化的偏差，偏差又反馈进下一步输入，于是：

从平衡态出发、给真实 T/P 的 pure-AR 128 步，质量和从 1.000 漂到 1.078，OH 高出 0.43 dex。
从 1300 K 预混出发，模型在着火后 OH 高出一个数量级并持续振荡，NH3 在 1e-7 和 4e-6 间乱跳。
测试集 10 条样本轨迹里 4 条终点 O2 相对误差超过 7 倍，最差 161 倍。pure-AR 的质量和在 H=128 降到 0.990，N 元素误差 3.2%。
v69 与 v70 两次训练命令完全相同，差别只是代码里 raw 状态损失加了 tanh 限幅，所以 v70 的 state loss 从 0.42 降到 0.11，但 rate loss 和锯齿一样没变，印证了问题不在正则项。

修改建议
去掉 max_damping 截断，用精确指数积分 Y·e^(-D·dt) + (P/D)(1 - e^(-D·dt))。exp(-D·dt) 下溢为 0 在数值上是安全的，不需要截。
改输出参数化为 log10(P/D) 加 log10 D。P/D 是准稳态渐近值，和 dt 无关，网络只需学状态到渐近值的映射。v69 计划 Task 1.4 想解决的"过阻尼塌陷"也应通过给 log10(P/D) 加物理下界来做，而不是截 D。
让目标按构造自洽。runner 里 P、D 取端点算术平均，着火行上用真值 P、D 也复现不了 Y_next。可改为给定 Y、Y_next、dt 和 P 的时间积分反解等效 D，这样积分器对训练数据是精确的。
把 dt 从随机抽样改成与 CFD 一致的分布，或至少在验证时用固定 dt 看一遍，把 dt 敏感度作为独立指标监控。
探针脚本已保存为 probe_dt_sensitivity_v70.py，运行前需设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE，否则 torch 与 Cantera 的 OpenMP 运行时会冲突。改完积分器后可以直接用它复查 dt 跨度是否归零。