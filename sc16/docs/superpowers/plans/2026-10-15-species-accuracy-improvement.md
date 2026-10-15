# sc16 物种浓度预测精度改进 Implementation Plan

> **For agentic workers:** 按本计划逐项执行，先建立失败回归，再做最小修改、验证和记录。每项使用复选框跟踪；实验建议不得未经对照就写成默认配置。本次只制定计划，不执行训练源码修改。执行时使用本环境可用的调试与测试技能；无需创建额外聊天。

**Goal:** 修复已证实的数值与数据不一致，定位非有限梯度来源，建立能可靠比较单步和长期物种浓度精度的训练与验证流程。

**Architecture:** 保留现有训练主线及旧输出模式，先完成物理与数据一致性修复。新的连续输出表示使用独立模式和变换版本，以相同数据、验证集和训练预算进行对照；通过实验后再决定默认架构和损失。

**Tech Stack:** Python 3.12、Cantera 3.0.1、PyTorch 2.11 开发版、NumPy、PyArrow、pytest。执行器为 `C:\Users\123\.conda\envs\cantera\python.exe`。

---

**范围与本次核实结论**

当前目标仍是物种浓度预测精度。CONVERGE 接口、ONNX/LibTorch 导出和微秒级性能不纳入本轮交付。下面对 Gemini 的建议作了筛选，而非直接合并全部结论。

项目根目录：`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16`。

本计划依据：本目录当前源码；前一轮 `analysis_20261015/results.json`；本轮 `analysis_20261015/gemini_verification_evidence.json`；相邻 sc15 中的历史日志和 benchmark；sc14 中辅助脚本指向的小数据集。当前 sc15 与 sc16 的 8 个核心 Python 文件逐字节相同，且 sc16 自前次审查后未变，但这不能证明历史训练时使用的源码与当前一致。

|Gemini 结论或建议|核实结果|计划如何处理|
|---|---|---|
|元素矩阵、dt 裁切、sentinel 跳变、NO 冷态 return、floor、D mask 等问题|已有代码和数值证据|纳入确定修复与回归检查|
|历史模型 IDT 和 NO 表现差|历史 JSON 确实有这些记录|作为故障线索；不直接归因于当前所有模型|
|6 倍诱导期权重导致 NO 崩溃|没有只改变该权重的消融；训练架构、标签、scaler 等也变化|作为待验证假设，不立即删除权重，不宣称独立 NO 头必然解决|
|第 14 轮没有有效 batch|历史日志证实；第 9 轮起就出现非有限梯度|优先保存失败输入并定位首个非有限运算|
|NaN 已确定来自 raw Y 飙升到 1e6 和温度 Arrhenius 指数|证据不足。当前 raw 宏观状态还有 [-1,10] 截断；NASA7 层没有 Arrhenius 反应速率指数计算|不能把“大 sinh”“温度敏感”当已定位根因；对相关路径逐个实验|
|KL 每步必做 8 次矩阵求逆，肯定慢于详细机理|默认至多 8 次迭代，有提前停止；`solve` 是解线性方程，不等于显式求逆。没有同硬件、同误差容限基准|保留现算法作为对照，不因未经测量的性能结论先换算法|
|仅 N2/H2O 作为吸收池即可消除任意元素残差|一般不成立：N/H/O 三约束对应的 3×2 子矩阵秩只有 2；还需满足非负可行性|不纳入保证守恒的默认方案|
|1 微秒以内推理、无缝接入 CONVERGE UDF|无实测、无接口契约依据|删除性能保证和部署承诺，后续独立评估|
|数据只有 15,857 行导致训练崩溃|混合了不同实验：本地小数据确为 15,857 行，崩溃日志元数据是 206,607,041 行|先建立数据/源码/checkpoint 对应关系，不能用扩容小数据解释大数据实验失败|
|样本少即“极端欠拟合”|不成立；样本少可能引起过拟合或覆盖不足，欠拟合须看训练误差|用训练误差、独立工况误差和学习曲线判断|
|训练数据缺少小 dt|分来源看：mini 的 67.54% 区间 dt≤5e-6；增广部分没有该小步长|审计联合分布 T/P/Y/dt；按同状态多 dt 补样，不笼统认定都在外推|
|统一 dt 上限到 2e-4、floor 到 1e-35|任意改常数不等于解决一致性|保留真实 dt、元数据边界显式检查；floor 采用可追踪策略和敏感性验证|
|永久 detach 温度、最大 H 改成 16–32|可作稳定性消融，但会改变梯度和长期约束，不能保证足够|分开前向长度 H 与反向截断长度 K；先定位再决定|

补充科学表述：NO 在氨燃烧中不应一概当成仅在后燃高温区累积的被动污染物；其生成和消耗也可能参与反应耦合。因此独立 NO 头可以是数值设计候选，不能据“NO 只在最后产生”直接决定采样和监督。

**本轮新增的实际证据**

1. 相邻 sc15 的 `v68_training_log_20261014_205615.txt` 记录 D 盘上的训练，元数据为 206,607,041 行、19,374 条成功轨迹、13,562 个训练 ID、2,906 个验证 ID。第 9–14 轮的非有限梯度警告分别为 30、193、201、219、221、242 次，第 14 轮以无有效 batch 结束。这里是日志声明的数据库规模，不代表每轮实际读入了全部行。
2. sc14 小数据现有 817+4800+3840+6400=15,857 行，而 database_summary.json 仍只写 817 行，说明元信息已过期。对应 6,400 行增广分片中，**2,404 行（37.5625%）dt>1e-4**，所以前次发现的裁切问题在这份实际数据中已经出现，不只是可能出现。
3. `retrained_benchmark_v72_audited.json` 的 NO 平均相对误差约 99.921%、IDT 比值中位数 0.333 可由逐工况记录重算。但 summary 写 NN 点火数 23，cases 中实际 24 条都有 NN IDT；其中 1 条参考在该时间窗未点火。其声明的 `best_composite_epoch4.pth` 在指定目录当前不存在，不能宣称已重新运行并复现该 checkpoint。
4. 因此“15/24 提升到 23/24”不能直接当成严格的点火分类成功率，更不能把相关变化归因于诱导权重。应分别统计正确点火、漏点火、参考未点火而 NN 点火、数值失败，以及双方均点火时的时刻误差。

**实施总顺序**

数据与版本溯源 → 数值契约修复 → 失败 batch 梯度定位 → 连续输出表示对照 → 监督与采样对照 → 固定验证选模 → 必要的数据补充 → 架构比较。完整热化学闭环作为物种阶段通过后的独立任务。

每个任务都应单独记录改动和结果；若项目有 Git，则单独提交。当前计划不要求先把非 Git 目录迁移为仓库，也不要求重构整个 8,000 行训练文件。

**任务 0：冻结数据、源码、评估口径，阻断实验结果混用（P0）**

文件：

- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\experiment_manifest.py`，负责源码/机理/scaler/数据分片/split/验证协议的记录与核对。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的训练入口与 checkpoint 元数据。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\validate_sc_v68.py` 和 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\benchmark_cfd_closed_loop.py` 的加载与结果输出。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_experiment_manifest.py`。

- [ ] 从实际 Parquet footer 统计行数，从 sample_id/初态索引统计轨迹数；与 summary 不一致时报错并给出两者，不静默相信旧元数据。
- [ ] manifest 保存 `source_hashes, mechanism_sha256, species_order, dataset_parts, split_hash, scaler_hashes, output_parameterization, transform_signature, dt_policy, concentration_policy, evaluation_protocol_id`。大分片使用生成时哈希或既有内容哈希，避免每个 epoch 重扫巨量数据。
- [ ] 在读 checkpoint 时核对模式、特征顺序、scaler、机理；旧文件缺字段可用于明确标记的历史复现，不允许自动混入新模型对比。
- [ ] 对 benchmark 的 summary 一律从 cases 重算；遗漏/重复 case、NaN、时间未覆盖和提前失败均单独记录，不能以删掉失败样本来改善平均指标。
- [ ] 给测试制造 summary=23、cases=24 个非空 IDT 的记录，验证审计可发现差异；给机理/scaler 哈希变化的 checkpoint 验证拒绝新实验加载。
- [ ] 单项验证：`python -m pytest test/test_experiment_manifest.py -q`。验收：真实历史不一致被报告，正常配对不受影响。完整命令的 Python 路径见文末。

**任务 1：统一机理元素矩阵，保留投影方法做干净对照（P0）**

文件：

- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\chemistry_contract.py`，负责机理元素矩阵、单位和浓度策略。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的 `build_feature_info`、`CoarseOperatorTools.__init__`、元素损失和投影调用；检查 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\validate_sc_v68.py` 的 `load_physical_element_matrix`。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_mechanism_contract.py`。

- [ ] 先添加测试：在 NUIG 物种顺序下，训练和验证的 A 均等于机理原子组成/分子量定义；不再由手写 AW 生成新实验矩阵。
- [ ] 新构造函数的核心代码为：

```python
def element_mass_matrix(gas, species_order):
    import numpy as np
    elements = [e for e in gas.element_names
                if any(gas.n_atoms(s, e) > 0 for s in species_order)]
    A = np.asarray([
        [gas.n_atoms(s, e) * gas.atomic_weight(e)
         / gas.molecular_weights[gas.species_index(s)]
         for s in species_order]
        for e in elements
    ], dtype=np.float64)
    return elements, A
```

- [ ] 将 A、元素顺序、物种顺序、机理哈希存入 feature_info；训练只读这些元数据生成 device/dtype 张量。测试人工 fixture 可显式传入 A，不隐式回退到手写 AW。
- [ ] 保持现有 KL 算法不变，只替换矩阵，重跑前一轮真实标签往返试验。对该 4,800 条样本，双分支投影后最大绝对误差应低于 5e-7；这只是本样本的数值回归门槛，不是全域预测精度要求。
- [ ] 同时确认宏微拼接及 inert 扣除顺序，输出投影后的真实元素残差；失败标志必须进入验证结果。
- [ ] 验证：`python -m pytest test/test_mechanism_contract.py -q`。与修复前输出记录做单因素对照，不借此声称网络精度提升 27 倍。

**任务 2：真实 dt、数据辅助脚本和工况隔离（P0）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的 `sanitize_dt_numpy`、`sanitize_dt_tensor`、`get_current_dt`、输入变换与元数据。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\data_pipeline\build_training_dataset_v72.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\data_pipeline\generate_augmented_trajectories.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\data_pipeline\split_trajectories_v21.py`。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_dt_and_dataset_contract.py`。

- [ ] 回归样本包含 dt 为 1e-8、1e-6、1e-4、1.5e-4、1.625e-4，验证编码/解码恢复真实 dt；0、负值、NaN、Inf 明确拒绝。
- [ ] 新实验对正有限 dt 不作隐式裁切；训练范围从实际数据统计并存入元数据。指定域外输入应由显式策略拒绝或子步进，不能改 dt 而保留原标签。
- [ ] NumPy 核心检查用如下语义，Torch 入口做同等检查；现有推理 dt 解码中也必须移除固定上限，否则只修输入没有完成闭环：

```python
def require_positive_dt(dt):
    import numpy as np
    value = np.asarray(dt, dtype=np.float64)
    if not np.all(np.isfinite(value) & (value > 0)):
        raise ValueError("dt must be finite and strictly positive")
    return value
```

- [ ] 数据脚本项目根改为 `Path(__file__).resolve().parents[1]`；机理、输入、输出通过显式参数传入；写新实验目录。先写临时文件、校验后原子替换，不覆盖 sc14 的固定分片。
- [ ] 从真实初态/机理重算 phi；mini 的 P_next 按下一源状态或恒容 EOS 恢复，无法可靠恢复的末行重积分或标记不可用，不伪造 P_next=P。
- [ ] benchmark 条件从训练候选中剥离；对同初态不同随机 dt 的轨迹用共同 condition_group_id 分组。已经进入训练的 24 条 benchmark 可保留为开发回归，但新建未使用工况的独立测试集。
- [ ] 重建实际数据 manifest。对那 2,404 条超原上限的增广区间，保留真实 dt 或重新生成到新契约，不能把它们当作原协议正常样本。
- [ ] 验证：`python -m pytest test/test_dt_and_dataset_contract.py test/test_split_trajectory_strata.py -q`。验收：标签终点与真实 dt 一致、分组无泄漏、脚本不会意外写旧目录。

**任务 3：修正冷态损失、floor 契约与 D mask（P0/P1，分别提交）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\enhanced_loss_and_weights.py`。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的投影参考值、`compute_qssa_rate_mask`、训练与验证调用。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_cold_state_contract.py`。

- [ ] 首先单独修 NO 提前返回：active 集合为空时 active loss 取与预测关联的零值，但仍执行 cold loss。宏观 asinh 与微观 log 两分支阈值分别使用物理浓度转换。
- [ ] 测试真实 NO=0、预测=1e-3 的全冷 batch：专项损失>0、对预测的梯度有限且方向指向减小 NO；全冷且预测低于冷态容许阈值时不产生无意义惩罚。
- [ ] 然后统一数值 floor 策略。先保留现有标签 1e-40 作为对照基准，移除投影中不一致的 1e-30；局部投影求解必要时用 float64。数值表示 floor 与物理活跃阈值分开，均写入 manifest，不直接统一改成 1e-35。
- [ ] 测试休眠物种无生成无消耗时，不因为投影参考值而跳高 10 dex；测试极低浓度前向、后向、log-STE 和元素残差均有限。
- [ ] 再改 D mask：函数增加真实起点与终点信息。首先以“只屏蔽 Y_old/Y_next 均低于该物种 active 阈值且变化低于参考噪声”的保守规则做对照；没有这些状态信息时不屏蔽 D。低 Y_qssa 不能单独触发 mask。
- [ ] 测试 Y_old=1e-4、Y_next≈1e-16、D=1e8、dt=1e-6 的强衰减样本，其 D 监督必须保留；报告各物种 mask 比例。
- [ ] 验证：`python -m pytest test/test_cold_state_contract.py test/test_micro_feedback_stability.py -q`。每个子项独立记录单步与 AR 变化，不一次把阈值和权重都改掉。

**任务 4：先定位非有限梯度，再选择稳定化措施（P0，优先于长训练）**

文件：

- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\training_diagnostics.py`，只负责失败快照、首个异常运算和有限梯度统计。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的分块 backward、非有限分支、梯度裁剪与 optimizer.step 周边。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_failure_snapshot.py`。

- [ ] 添加 independent attempted_batch_idx，避免日志所有失败都重复写 batch=1；保存实际 sample_id、window_start、dt、输入、标签、当前模型状态、优化器状态、RNG、有效配置和 chunk 起点。
- [ ] 调试模式在 batch 之前保存可重放状态；遇到第一处 NaN/Inf 后停止该诊断运行，不能无限跳过以伪装正常训练。正常训练模式只允许预设的有限跳过预算，并报告比率。
- [ ] 用失败快照执行 `torch.autograd.detect_anomaly(check_nan=True)`，记录 forward 非有限值、backward 首个异常运算、各损失梯度范数和输出饱和比例。
- [ ] 梯度检查覆盖“各元素有限但 float32 总范数溢出”的情况，范数诊断用 float64；裁剪前确认有限。已有 NaN 不能靠把 clip_norm 从 1.0 改为 0.5 修好。
- [ ] 分开实验前向长度 H 和反向截断 K：保持 H=128，先比较 K=4、8、16；另测 H=32 的成本/精度，但不预设 32 步足以覆盖诱导期。
- [ ] 温度反馈 `detach()` 只作为明确开关的消融；同时对比 recorded T/P、完整温度梯度与 stop-gradient。若异常来自热力学求根，先修正求根/导数；允许后续比较隐式热力学导数，不直接默认断开所有温度梯度。
- [ ] 只有确认宏观速率饱和是问题后，再设计与真实标签范围一致的平滑上限；不用 `tanh(z/3)` 无标定替换原目标。报告被限幅的参考标签比例。
- [ ] 验证：`python -m pytest test/test_failure_snapshot.py test/test_rollout_chunked_backward.py -q`。验收：故障快照可定位到运算；若尚无原始大数据失败快照，只能报告“小型稳定性测试通过”，不能宣称已解决历史崩溃。

PyTorch 官方将 anomaly detection 用于追踪异常反向运算；梯度裁剪文档也单列非有限范数处理，二者不能互相替代：[Autograd](https://docs.pytorch.org/docs/stable/autograd.html)、[clip_grad_norm_](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html)。

**任务 5：连续 QSSA 参数化作为新模式，保留旧模式对照（P1）**

文件：

- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\kinetics_parameterization.py`，封装新目标编码、解码、稳定指数积分。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_all_v68.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\validate_sc_v68.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\validate_all_best_models_v68.py`；必要时调整 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\dual_branch_network.py` 的模式校验，不改变隐藏层容量。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_continuous_parameterization.py`。

- [ ] 先审计已有 `qssa` 的 logP/logD 实现，复用已验证的稳定算子；不是另写一套无测试重复实现。
- [ ] 新模式名 `dual_branch_pd_v1`：宏观头暂保持原定义，微观头始终对应 P、D，不再让同一 z1 在 logP 和 log(P/D) 间切换。新 transform signature、scaler、checkpoint 独立，严禁用旧输出头直接解释新目标。
- [ ] 目标候选为 `log10(P+eps_P)` 和 `log10(D+eps_D)`，解码对应 `max(10**z-eps,0)`；记录零附近的截断梯度。它连续但不应被宣传为处处光滑。若零附近优化不足，再与 softplus 正值表示做单独对照。
- [ ] eps_P 与 eps_D 按各自单位和可接受状态误差选择，不能因方便取同一个 1e-30。要求在最长验证物理时间内，零生成的数值偏置不超过冷态容许浓度；零消耗近似造成的衰减也需有上界。
- [ ] 使用 `Y_next=Y*exp(-D*dt)+P*dt*phi1(D*dt)`。下列稳定 phi1 核心同时避免未选分支的 0/0 和巨大 x 的 Taylor 溢出：

```python
def phi1_decay(x):
    import torch
    small = x.abs() < 1e-4
    x_small = torch.where(small, x, torch.zeros_like(x))
    x_regular = torch.where(small, torch.ones_like(x), x)
    series = 1 - x_small / 2 + x_small.square() / 6 - x_small.pow(3) / 24
    regular = -torch.expm1(-x_regular) / x_regular
    return torch.where(small, series, regular)
```

- [ ] 微量物种低浓度输出保留对数域实现或稳定 log-state 梯度，不能退回 float32 的 `log10(Y≈1e-40)` 奇异反传；复用现有 log-STE 回归用例。
- [ ] 测试 P=0、D=0、两者都为零、小/大 Ddt、耗尽、初始零值、多数量级扫描；验证 D→0 时趋于 Y+Pdt，阈值附近值连续、梯度有限。
- [ ] 真实标签往返与旧模式比较；检查 P/D 大动态范围引起的回归条件数，避免把“消除了 sentinel”误当成已证明所有精度都提高。
- [ ] 验证：`python -m pytest test/test_continuous_parameterization.py test/test_exponential_integrator.py test/test_micro_feedback_stability.py -q`。先通过积分器测试，再开始相同网络容量的小型训练对照。

**任务 6：区分真实状态参数监督与预测轨迹监督（P1）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的主训练循环和辅助 rollout loss。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\enhanced_loss_and_weights.py` 的诱导期权重来源与损失统计。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_supervision_alignment.py`。

- [ ] 将参数监督放在真实输入：`rate_pred_true=model(batch_X[:,step])` 对 `batch_Y[:,step]`；AR 分支 `model(current_X)` 主要接受真实轨迹的状态/终点损失。真实输入与反馈输入相同的时候复用一次前向，避免无意义重复。
- [ ] 移除 AR 验证中将“预测状态参数”与“真实状态参数”直接混成同一监督准确率的口径；参数误差作为 oracle 指标，AR 主要报告状态误差。
- [ ] 诱导期权重由真实相邻状态计算；先比较 boost=1、2、6，其他配置不变。记录未燃、诱导、快速反应、后期各组对总梯度的贡献。
- [ ] NO 先使用已经修复的独立损失，不立即加新头。只有在样本/损失平衡改善后仍出现任务冲突，再比较共享头与 NO/N2O 独立头。
- [ ] 损失分母按有效样本数和特征数一致归一化；记录 masks、各损失值及梯度范数，防止新增权重等效改变整体学习率。
- [ ] 用线性衰减的玩具问题 `f(y)=-k*y` 验证两分支输入与标签配对正确；用全冷、混合冷/热 batch 检查 NO 梯度；用不同 dt 的同状态样本检查诱导权重解释。
- [ ] 验证：`python -m pytest test/test_supervision_alignment.py test/test_cold_state_contract.py -q`，再执行配对训练消融。只有独立验证结果支持时，才把 boost 或新头设为默认。

**任务 7：固定验证集、物理时间与选模标准（P0/P1）**

文件：

- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\species_validation_protocol.py`。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\validate_sc_v68.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\compute_physical_metrics.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的 scheduler/早停/最佳模型保存。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_fixed_validation_protocol.py`。

- [ ] 建立固定哨兵集：固定 condition_group_id、起点、dt 序列、物理终止时间和 H=1/8/32/128。验证 H 独立于当前训练 curriculum；不能训练 H=8 时用 8 步分数覆盖原 128 步最佳值。
- [ ] 每轮评固定小集，按固定频率评完整 validation；test 不参与调参和选模。已训练过的历史 24 工况仅作开发回归，不能再标为独立 test。
- [ ] 记录逐物种绝对 MAE、按真实峰值归一化的轨迹误差、活跃区 log 误差、冷态假阳性、峰值/峰时、NO/N2O 终值；主要物种与关键自由基分别汇总，不能只看一个全物种均值。
- [ ] raw/constrained 都输出，记录投影修正分位数、负值、元素残差、投影失败。参考近零时不用不带下限的相对误差。
- [ ] 点火定义使用一致的参考温升阈值或共同计算的 dT/dt 指标；参考未点火是“本时间窗未点火”，需与数值失败分开。禁止各轨迹各自用自身 Tmax 定义不同的比较阈值。
- [ ] scheduler/早停先采用固定协议的 AR 主要物种与活跃自由基组平衡分数，同时把冷态假阳性、非有限结果和重要物种退化设为独立门槛；每个 horizon 单独保存最佳值。具体权重冻结在 protocol_id，不能在观察 test 后修改。
- [ ] 测试同 checkpoint 连续两次评估结果一致；curriculum 变化不改变验证样本/horizon；汇总必须等于逐 case 重算。
- [ ] 验证：`python -m pytest test/test_fixed_validation_protocol.py test/test_experiment_manifest.py -q`。

**任务 8：按缺口补数据，修正窗口采样效率（P1，先审计后扩容）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\data_pipeline\runner_sc_v21_qssa.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\data_pipeline\generate_augmented_trajectories.py`。
- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的 `CoarseOperatorDataset` 窗口选择。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_multidt_and_windows.py`。

- [ ] 按真实训练 manifest 审计 T/P/phi/H2比例/物理阶段/dt 的联合覆盖，分别报告每个来源；不再把 15,857 行与 2.07 亿行实验混合统计。
- [ ] 在同一个锚点状态上分别重置独立反应器，生成 dt={1e-8,1e-7,5e-7,1e-6,5e-6,1e-5,1e-4} 中落在目标范围内的区间。所有同锚点/同母轨迹数据进入同 split，避免泄漏。
- [ ] 增加完整长诱导期、燃料接近耗尽、NO/N2O 生成/消耗区的样本，而非只均匀加行；先扩固定规模的开发集并画学习曲线，只有覆盖和泛化收益支持时才扩大到 30–50 万或更多区间。
- [ ] 初始工况范围从已有目标研究范围和验证缺口确定，不直接缩到 T≥800 K、P≤100 atm，以免删掉原模型承诺覆盖的低温/高压状态。
- [ ] 160 步轨迹、H=128、stride=64 时补入最后合法起点 32；训练阶段动态 H 时允许短轨迹参与短窗口，不能一律按最大 H=128 丢弃。短窗口或 mask 必须正确归一化损失。
- [ ] 比较随机窗口、物理阶段分层窗口和按轨迹等权采样，避免长冷轨迹仅靠区间数支配训练；保留窗口元信息用于失败重放。
- [ ] 验证：`python -m pytest test/test_multidt_and_windows.py test/test_split_trajectory_strata.py -q`。验收：同锚点起点严格相同、窗口无越界和尾段遗漏、物理阶段覆盖可量化。

**任务 9：最后比较网络结构和宏观低浓度表示（P2）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\dual_branch_network.py`、`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_all_v68.py` 的显式实验配置。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_architecture_contract.py`。

- [ ] 以已修复、验证口径固定的当前 MLP/ResMLP/双分支为对照；隐藏宽度与块数显式保存，不在验证端写死 512×4。
- [ ] 每个候选固定数据、更新步数和参数化，至少 3 个随机种子；同时报告训练误差、验证误差、最坏工况和失败率，不仅比较最后 epoch。
- [ ] 对 NH3/H2 等接近耗尽状态，先对比局部高精度解码和物理尺度混合损失；若标签 float32 已失去信息，需要一起修改标签表示，单改最终加法 dtype 不够。
- [ ] 将 N2O 的实际分支归属写入 feature_info，按数据灵敏度选择；不要因为 DEFAULT_SLOW_SPECIES 注释含 N2O 就假设当前已进入慢分支。
- [ ] 架构修改仅在独立验证收益超过多种子波动、关键物种不明显退化时保留。没有收益则保留较简单模型。
- [ ] 验证：`python -m pytest test/test_architecture_contract.py test/test_dual_branch.py -q`；最终结论来自训练对照，不来自网络参数数量。

**任务 10：独立热化学闭环，物种阶段达标后再执行（P2）**

文件：

- 修改 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\thermo_nasa7_layer.py` 和 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\train_sc_v68.py` 的 `state_to_input_scaled/update_temperature_autonomous`。
- 新建 `C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16\test\test_thermochemical_closure.py`。

- [ ] 分开三个协议：参考 T/P 下物种 AR；自主 T、参考 P 的条件实验；自主 T/P 的完整闭环，报告中禁止统称“完全自主”。
- [ ] 恒容保存初始 rho 与 u_target，用预测 Y 解 T，并按 EOS 更新 P；恒压保存 P 与 h_target。压力单位只在输入边界转换。
- [ ] Newton 中间步可限幅，但最终 T 必须满足能量残差；不再对求得的物理终点直接裁 ΔT=500 K。不能收敛时报告失败或连同组成子步推进，不能静默改变能量。
- [ ] 对 1200 K→约 3039.8 K 的已知守恒测试，检查最终能量而非仅检查 T 在范围内；无组成变化时 T/P 应保持一致。
- [ ] 完整闭环梯度与 stop-gradient 分开实验，长期验证仍涵盖点火与后燃阶段；若精度退化，回到数据/表示诊断，不直接宣称模型可替代 CFD 机理。
- [ ] 验证：`python -m pytest test/test_thermochemical_closure.py -q`，并与相同机理 Cantera UV/HP 结果比较。

**训练对照的顺序与停止条件**

|编号|与前一基线相比只改变什么|决定|
|---|---|---|
|B0|现代码的可复现实验；不掩盖发现的数据或评估缺陷|建立历史参照，不作为最终有效性证据|
|B1|任务 0/1/2/7 的一致性与验证修复|数据契约和真实标签往返合格后才继续|
|B2a/b/c|依次修 NO 冷态、floor、D mask|逐项看冷态误报和关键物种影响|
|B3|连续 PD 参数化，架构不变|不连续消失且状态预测有收益才选用|
|B4|真实参数监督与 AR 状态监督分开|比较单步与长程，不能只降低训练总 loss|
|B5|固定 H 下改变 K；再独立比较温度梯度策略|依据失败定位和精度决定，不预设永久 detach|
|B6|boost=1/2/6、分层采样；必要时独立 NO 头|验证 NO 与自由基是否真的存在可缓解冲突|
|B7|有针对性地补数据|学习曲线和留出工况收益决定是否扩容|
|B8|MLP/ResMLP/双分支容量比较|使用最终冻结协议及至少 3 个种子|

出现输入/标签不一致、无效梯度、错误机理/scaler 或 benchmark 工况泄漏时，不进入大规模训练。任何“优化成功”须同时满足：可重放、验证协议相同、关键物种不被均值掩盖、失败工况没有被删去。最终精度验收线由研究所需的主要物种误差、自由基活跃阈值、残余 NH3 和 NO/N2O 浓度范围共同确定，不以未经约定的 1% 或 1 微秒作通用标准。

**本轮明确不采用的默认修改**

- 不把训练样本数机械扩到 50 万当作解决崩溃的方法；已有历史实验的元数据规模远大于此。
- 不用统一改变 dt 上限来掩盖实际区间与标签不一致。
- 不直接永久禁用温度梯度、把前向 H 降到 16、或把宏观输出压成未经标定的 tanh。
- 不把 KL 投影当“金标准标签”。真实 Cantera 终点才是参考；投影只是对候选状态施加约束，距离度量不保证物理最准确。
- 不承诺仅 N2/H2O 的显式修正能普遍保正且同时守恒，不承诺 <1 微秒推理。未来若做投影替代，要在同一算法下重新训练/微调并验证，避免训练与推理约束算子不同。
- 不把有限时间推进算子直接导出后当成瞬时化学源项；未来接口需先明确 dt、守恒量、热力学约束和调用语义。CONVERGE 官方说明其包含 SAGE 化学求解器，但这不足以证明 SAGE 与 Cantera/CVODES 是同一实现或某个 UDF 能直接替换它：[CONVERGE 官方介绍](https://convergecfd.com/about/converge-cfd-software)。

**执行命令与交付物**

以下 pytest 文件是计划新增文件，在对应任务完成前不应把这些命令当作当前已能运行的脚本。执行环境示例：

```powershell
Set-Location -LiteralPath 'C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc16'
$env:CANTERA_DATA = (Resolve-Path '..\sc15').Path
& 'C:\Users\123\.conda\envs\cantera\python.exe' -m pytest test -q --disable-warnings
```

正式实验应把所用机理以哈希管理的显式路径交给训练，不长期依赖旁边目录。原环境出现过 OpenMP 重复运行时问题；先用独立进程和小型算子验证确认环境稳定，不在正式计划里无条件设置绕过错误的环境变量。

每项交付包括：最小源码修改、对应回归测试、变化前后诊断记录、manifest、固定验证指标和仍未解决的问题。变换契约改变时独立保存 scaler/checkpoint；可以复用兼容的骨干权重做暖启动，但必须重置不兼容输出层及优化器状态，不冒充断点续训。

本次已完成的是证据核实和实施规划；以上复选框均为待执行内容。
