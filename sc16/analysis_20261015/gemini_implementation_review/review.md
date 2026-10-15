# Gemini 实施结果复审

审查日期：2026-10-15。项目：`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16`。

**结论：部分基础修复有效，但本次交付尚不能认定 Task 0–7 全部完成，也不具备按报告所列命令启动新模式全量训练的条件。** 80 项原测试通过可以确认；新模式注册、标签编码、分组读取、验证选模和低浓度梯度仍有实质问题。没有新的训练/独立测试精度证据支持“高精度模型已经完成”。

本次仅审阅及新增本目录的复现材料；没有修改训练源码、原始测试、模型权重或训练数据，没有启动全量训练。

## 验证范围与结果

- 审阅用户提供的 Gemini 报告，逐项核对当前源码与前一轮规划。
- 使用前次保存的 SHA256 确认 sc15 中对应文件仍与原审查版本一致，再生成当前 sc16 的差异。本轮发现原有文件改动 6 份，另有 5 个新增模块；`train_all_v68.py`、`dual_branch_network.py`、`validate_all_best_models_v68.py` 没有改动。
- 环境：`C:\Users\123\.conda\envs\cantera\python.exe`，PyTorch `2.11.0.dev20260120+cu130`，Cantera `3.0.1`。
- 原测试：`python -m pytest test -q --disable-warnings`，**80 passed, 2 warnings in 18.86s**。本轮为该进程设置了 CANTERA_DATA 指向相邻 sc15；当前 sc16 也已有 NUIG2024.yaml。
- 独立补充回归：`test_review_regressions.py`，**10 failed in 8.51s**。这些测试断言计划要求的正确行为，失败意味着缺陷被复现，不能把失败数解释为随机抽样的质量比例。文件放在本审查目录，未混入原 `test/`。
- 数值/数据探针：`probe_integration.py`，输出 `evidence.json`。只使用小型合成数据与现有 4,800 条 mini 样本；临时 Parquet 和快照均写入临时目录。
- 与现有测试的环境做法一致，独立探针与补充测试进程设置了 `KMP_DUPLICATE_LIB_OK=TRUE` 以绕过此前已存在的 OpenMP 重复运行时问题；没有永久修改环境。这是本次运行条件，不是环境故障的正式修复。

## 已确认有效的部分

1. **本地 NUIG 元素矩阵修复有效。** 对现有 `dual_branch` 模式、4,800 条真实标签做“编码→解码→约束投影”的无网络误差实验，最大绝对误差为 `1.9422058529e-7`，前次原代码为 `5.2265989887e-6`。实际使用的 float32 矩阵与 Cantera 矩阵最大差约 `2.84e-8`。这是标签表示/投影误差改善，不是训练后网络精度提高 27 倍。
2. **实际解码路径保留大 dt。** 输入 `1.625e-4 s`，`get_current_dt` 返回约 `1.6249998589e-4 s`，差别来自 float32 对数表示。原 `1e-4` 上限不再截断该路径。
3. **全冷 NO 专项损失生效。** 真值 0、预测 `1e-3` 时损失约 `0.185104`，梯度约 `43.4273`，方向能够压低假阳性。
4. **投影 floor 已对齐。** 用真正的 `project_physical_state` 运行静止休眠物种，`1e-40` 不再被强制抬到 `1e-30`。
5. **主循环参数监督与 AR 状态监督已分开。** `train_sc_v68.py:6573` 起确实使用真实输入另做参数预测；诱导权重改为使用真实相邻状态。这里确认的是接线逻辑，尚不能宣称泛化精度提高。
6. 新 `phi1_decay` 的大 `D*dt` 分母保持真实值，小参数分支避免显式 0/0，独立算子测试通过。但其外围的 float32 对数梯度仍有问题，见 R2。

## 需要先修复的发现

P1：阻碍新模式使用、可能破坏训练/验证有效性，应在长训练前修复。P2：契约、诊断和特定条件下的缺陷，应补齐后再宣称对应任务完成。

### R1 · P1：新 PD 模式没有贯通实际训练和加载流程

位置：

- `train_sc_v68.py:8029`：CLI 的 `--output_parameterization` choices 不含 `dual_branch_pd_v1`。
- `train_sc_v68.py:5633`：即使绕过 CLI，`train_model_v68` 自身也拒绝新模式。
- `train_sc_v68.py:1972`：真实目标编码 `transform_target_array` 拒绝新模式。
- `train_sc_v68.py:2242`、`:2322`、`:2592`：scaler 签名、数据列选择仍只处理旧双分支。
- `train_sc_v68.py:2982`：`CoarseOperatorTools` 将新模式当 legacy 校验签名。
- `train_sc_v68.py:2767`、`:6171`：默认 MLP 模式白名单及输出损失权重尚未匹配新输出维度。
- `validate_sc_v68.py:134`、`:413`：checkpoint/feature_info 允许的签名未加入新模式。
- `train_all_v68.py` 尚未接入新模式。

实测：报告中的 CLI 参数以 **exit code 2** 被拒绝；目标变换抛 `ValueError`；新 feature_info 构造工具时提示期望 `macro_asinh_p90_v2`，实际为 `dual_branch_pd_v1`。生产源码中 `encode_pd_targets` 的调用数为零。

影响：报告所列新模型命令不能开始训练。只补 CLI choices 仍会在后续环节失败；直接把旧 scaler 或输出层改名还会造成目标语义错误。

建议：一次打通“真实 Parquet→列读取→scaler→目标编码→模型→损失权重→状态解码→保存→重新加载→验证”全链条，复用统一模式配置。验收应包含一个真实小 batch 的前向、反向、保存和加载，而不只测独立积分器。

### R2 · P1：新解码分支重新引入极低浓度的 float32 非有限梯度

位置：`train_sc_v68.py:3448–3450`。

当前新分支先在线性域得到 `Y_micro_next`，再执行：

```python
logY_micro_next_candidate = torch.log10(torch.clamp_min(Y_micro_next, 1e-40))
```

实测：单独调用这条新分支，`log10(Y_old)=-39`、`zP=-20`、`zD=-10`、`dt=1e-6`、float32 时，前向得到 `logY_next=-39`，但输入梯度为 `[0,0,0,inf,nan]`，输出梯度为 `[0,inf,nan]`；`-39.5` 同样失败。

这是 `d log10(Y)/dY=1/(Y ln 10)` 在 float32 极小浓度处溢出，并在链式法则中形成 `0*inf`。后续 log-STE 不能消除已经在 raw 对数更新中产生的奇异梯度。`Y=1e-40` 单点可能因截断得到零梯度而掩盖问题，所以必须扫 floor 附近的多个数量级。

复现为了隔离此数值分支，使用旧合法工具 fixture 再显式指定新解码路径；这不是证明当前新模式已经能够训练，也不是把它归因于过去的 H=128 崩溃。

建议：实现数值稳定的对数域 PD 更新，或采用经过验证的局部高精度/自定义导数设计；同时处理零生成、纯衰减和指数输出范围。测试真实 float32 的完整“网络输出→解码→log 状态损失→反向”，不能只测 float64 的线性 `Y_next`。

### R3 · P1：condition_group_id 没有从 Parquet 读入，所谓零泄漏未实现

位置：`data_pipeline/split_trajectories_v21.py:145–156`，下游 `:212–215`。

`_available_condition_columns` 漏掉 `condition_group_id`。因此新增读取逻辑永远拿不到该列，随后用 `sample_id` 充当组 ID。

实测：10 个真实工况组、每组 3 条轨迹，读入后变成 30 个组。按 60%/20%/20% 划分后，训练与验证共享 5 个真实工况组，训练与测试也共享 5 个真实工况组。

现有 `test_condition_group_isolation_in_splits` 的断言再次读取已经损坏的 metadata，检查的是“伪组”是否分离，所以仍能通过。

建议：把组列加入 Parquet 投影读取，断言读入 ID 与源数据一致，以原始工况 ID 独立检查划分。不同来源、不同 dt 的相同物理初态还需使用一致的组键；仅给每条轨迹起不同名字并不能隔离同工况。生成的 24 个 benchmark 工况仍写入通用训练目录，不能据当前代码宣称基准工况已从训练隔离。

### R4 · P1：固定验证协议只是新增模块，没有驱动真实选模

位置：`train_sc_v68.py:6418`、`:6452`、`:4616`、`:7242–7244`、`:7425–7428`。

`HorizonSentinelTracker` 被创建，但未调用 `consider_candidate`；生产文件中没有 `compute_species_trajectory_metrics` 调用。真实训练仍执行：

- `val_dataset.set_epoch(epoch)`，抽样随 epoch 改变；
- 验证 `H=min(active_horizon, ...)`，受课程阶段限制；
- `val_stats=val_stats_oracle`，scheduler 和新 `_H...pth` 保存仍由旧 oracle total loss 驱动。

新增“按当前 active_horizon 保存文件”与“课程切换重置 patience”是有效的局部改进，但并不等于在固定样本上独立评估 H=1/8/32/128，更不等于按照新物种指标选模。

建议：真正建立固定哨兵输入/物理时间序列，独立于训练课程执行各 horizon；连接新指标、物理门槛和 tracker，保存并恢复其状态；正式模型选择依据固定 AR 协议，oracle 指标单独报告。

### R5 · P1：新指标会奖励自由基完全漏预测，物理门槛也不完整

位置：`species_validation_protocol.py:131–137`、`:230`。

活跃误差掩码使用 `(Y_true>threshold) & (Y_pred>0)`，把预测为零或负数的活跃真值排除；没有剩余样本时误差返回 0。

实测：真实 H=`1e-4`，预测 H=0 时 `mean_radical_log_rmse=0`；预测 H=`1e-5` 时误差为 1 dex。包含平衡 H2 的守恒示例中，前者总分约 `4.0e-5`，后者约 `0.350036`，错误模型反而更优。

另一个实测输入预测 `[-0.1,2.0]`，存在负质量分数、质量和 1.9、元素残差最大 1.5，但 `is_valid_candidate=True`。函数记录这些指标，却只用非有限值和冷态误报率构成候选门槛。报告宣称的 NO/N2O 终态误差也尚未实现，现代码用的是全时间聚合 NMAE。

建议：活跃掩码只由真实物理状态决定；预测零值按明确浓度下限计入误差，负值和物理残差设置门槛；按轨迹保留终点和峰时信息。先修评分，再接入 R4，防止新协议系统性选错模型。

### R6 · P1：D mask 新接口使旧 qssa/qssa_asymptotic 训练存在越界崩溃

位置：`train_sc_v68.py:935–941`、`:6628–6633`；另有 `:928–929` 和 `:951–952` 的无状态回退问题。

主循环传入完整的反标准化输入（T/P、宏观 Y、微观 logY、dt）。qssa 分支的 `n_micro=output_dim//2` 实际是全部反应物种数，却用仅包含微观物种的 `tools.micro_idx[i]` 遍历。

实测：一个宏观 H2、一个微观 H 的合法 qssa fixture，完整输入时抛 `IndexError: index 1 is out of bounds ... size 1`。实际机理只要宏观与微观同时存在，也有相同长度不匹配。

此外，无起终点状态时并没有按注释返回 None：目标 `[0,-16,8]` 仍得到 `[1,1,0]`，强衰减 D 被屏蔽。验证端 `compute_training_loss` 恰好仍不传状态，因此主训练与验证掩码不一致。

建议：明确“物种物理质量分数”和“混合输入特征”两个不同接口；qssa 按全部反应物种映射、双分支按微观物种映射，不靠数值是否为负来猜表示。没有状态就不屏蔽；同步训练、验证调用。

### R7 · P2：实验 manifest 没有接入 checkpoint，严格校验本身也漏检

位置：`experiment_manifest.py:138–184`，主训练保存 `train_sc_v68.py:7435–7466`。

生产文件未调用 `create_experiment_manifest`、`audit_parquet_dataset`、`verify_checkpoint_compatibility`；checkpoint 仍未写入新 manifest。新模块不能保护实际训练/验证的机理、数据和 scaler 配对。

即便单独调用严格检查，`transform_signature` 错误且缺少所有 scaler 哈希的 checkpoint 仍返回 `(True,[])`：签名未比较，scaler 只检查“双方都有同名键但值不同”，不检查缺失键。数据审计函数还只查同目录 `metadata.json` 的 `training_rows`，没有覆盖前次实际发现过期的 `database_summary.json`。

建议：绑定所有保存/恢复/验证入口，检查必需字段存在、签名/eps/物种顺序/机理/scaler 等一致；从实际使用的元数据层级审计。补充回归已覆盖错误签名与缺失 scaler 仍被接受的问题。

### R8 · P2：快照记录发生在失败后，缺少实际重放条件

位置：`training_diagnostics.py:169–180`、`:105–118`；主循环 `train_sc_v68.py:7054–7068`、`:7011–7014`、`:7040–7042`。

问题包括：

- RNG 在失败之后采集，无法还原本 batch 前向前的随机状态；调度采样/噪声等随机选择可能不同。
- 主循环没有传 sample_ids/window_start，也没有保存有效训练配置、反馈掩码、scaler/化学工具和 chunk 起点。
- 非有限 chunk loss、非有限 rollout loss 的提前 continue 不会走 tracker；异常 backward 也未被统一记录。
- replay 需要调用者额外提供 forward_fn，并没有实现真实训练 batch 的完整重放。

实测：含 dropout 的简单模型，先执行前向后按当前 API 存失败快照，再 replay，输出最大差约 `1.49146`。这是随机重放契约的反例；不是声称当前默认网络开启了 dropout。

建议：在 batch 前保存 RNG/配置/输入元信息，或保存实际已生成的随机决策；把所有失败出口统一接入快照；编写真实 chunked-AR 重放入口和一次可重现的失败回归。

### R9 · P2：float64 范数仅用于诊断，实际裁剪仍可溢出并清零梯度

位置：`train_sc_v68.py:7054`、`:7072–7076`。

新增检查算出了 `grad_norm_f64`，但后续 `clip_grad_norm_` 重新用原梯度类型计算范数，未使用该安全值。实测两个元素均为 `1e20` 的 float32 梯度：诊断认为有限，float64 范数 `1.414e20`；实际裁剪返回 Inf，梯度被变成 `[0,0]`。

建议：使用安全范数实施缩放，并在缩放后验证梯度/更新是否有效；将失败记录为训练异常。该问题是有限大梯度的具体反例，不是本次已证明的历史崩溃原因。

### R10 · P2：机理矩阵仍可静默回退，训练并未绑定实际机理契约

位置：`train_sc_v68.py:2883–2904`、`:3114–3125`。

当前本机 NUIG 文件存在，修复确实生效。但 `build_element_matrix` 仍自动尝试固定 NUIG 路径，捕获异常后回退手写 AW；调用者没有把实际数据机理路径传入。`build_feature_info` 也没有保存矩阵/元素顺序/机理哈希。

实测：显式传入不存在的机理路径仍成功返回旧 AW 矩阵；真实 mini 往返所用 feature_info 不含 `element_matrix` 与 `mechanism_sha256`。

建议：训练入口从数据契约解析明确机理，构建并存储 A 与哈希；指定路径失败时明确报错，测试 fixture 的旧矩阵只能显式选择。不能把“本机碰巧找到 NUIG”当作可迁移的严格契约。

### R11 · P2：mini 转换把初始当量比重新解释为每步剩余燃料/O2 比值

位置：`data_pipeline/build_training_dataset_v72.py:124–133`。

每个时间步用剩余 NH3/H2/O2 重新计算 `phi`、`phi_actual`，燃烧产物与其他中间物种未计入；这不是该轨迹的初始工况当量比。

实测 30 条轨迹中 15 条的写入 phi 范围超过 0.1；sample_id=34 的初值约 `2.67105`，过程最大约 `97089.84`，末值约 `43898.39`。用它做按工况采样或全行条件统计会产生错误分布。当前 splitter 取初始行，因此此问题不意味着所有现有分层结果都已被这些末态数值污染。

建议：从每条轨迹真实初态计算一次并固定保存；如果需要瞬时剩余燃料/O2 指标，应另命名，不覆盖初始工况字段。缺机理时用所有物种分子量 28.0 计算终压的回退也应改为明确失败；当前本机有机理时该回退未触发。

## 为什么 80 项测试没有挡住这些问题

- 新 PD 测试主要测 `kinetics_parameterization` 的独立函数，没有通过主训练 CLI、真实列读取、scaler、模式签名和 checkpoint 加载。
- 低浓度测试测旧双分支 log-STE，新积分器主要测 float64 线性状态；遗漏新分支 float32 的线性转对数。
- 分组隔离测试用读入后的错误 metadata 作为真值，形成自证。
- floor 测试在本地构造 `p0=scaled_dyn.clamp_min(1e-40)` 后验证这个局部变量，并没有调用实际投影层；本次另做实际投影才确认该修复有效。
- 元素矩阵 roundtrip 测试用 Yn 自己的元素量作为目标，旧矩阵也可天然满足自身约束；它不足以复现 Y_old→Y_next 的原问题。本次另做 4,800 条完整标签往返补足证据。
- 监督对齐测试手写玩具训练循环，没有调用生产训练循环；固定协议测试只测新类本身，没有断言主程序调用它。

这些测试可继续保留，但应新增跨模块的集成检查。测试数量不能代替路径覆盖，也不能代替真实训练后的独立精度验证。

## 建议的返修顺序与完成条件

1. **先修 R1/R2/R6。** 新模式完成真实小数据的编码、训练一小批、有限梯度、保存重载、验证；同时确保旧 qssa/asymptotic 不回归。
2. **修 R3/R5/R4。** 用源工况组验证无泄漏；先修评分，再接入固定多 horizon 验证与选模。不得用新增文件名冒充完整固定协议。
3. **补 R7/R10/R8/R9/R11。** 绑定机理与实验哈希；补齐故障重放和安全裁剪；修初态元数据。
4. 原 80 项测试与本次 10 项回归均通过后，再做真实 mini 数据的小型端到端训练。记录每物种单步、短程 AR、长程 AR、冷态误报、负值与守恒残差，以及失败 batch 数。
5. 最后按原计划做配对消融与独立留出工况测试。只有这些结果支持时，才能说“物种预测精度达标”，再安排工作站长训练。任务 8–10 的数据补充、架构比较、完整热化学闭环仍应单独验收。

## 复现文件与命令

本目录文件：

- `probe_integration.py`：数值与接线探针。
- `evidence.json`：包含源码 SHA256、全部具体输入/输出和静态调用核查。
- `test_review_regressions.py`：10 项期望正确行为的回归测试，当前全部失败。
- `regression_results.txt`：失败输出与源码行号。
- `changed_sources.json` 和 `*.diff`：与前次已确认基线的差异。

```powershell
Set-Location -LiteralPath 'C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16'
& 'C:\Users\123\.conda\envs\cantera\python.exe' -m pytest test -q --disable-warnings
& 'C:\Users\123\.conda\envs\cantera\python.exe' -m pytest analysis_20261015/gemini_implementation_review/test_review_regressions.py -q --tb=short --disable-warnings
& 'C:\Users\123\.conda\envs\cantera\python.exe' analysis_20261015/gemini_implementation_review/probe_integration.py
```

命令应在 OpenMP 环境已正确配置的进程运行；本次额外运行条件已在前文记录。所有审查结论对应 evidence.json 中记录的源码哈希，后续代码变化后应重跑，不能沿用本轮结果作为新版本保证。
