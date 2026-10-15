# Gemini 第二轮返修后的第三次复审

审查日期：2026-10-15。范围：物种预测模型的训练、数据、数值更新、验证和保存重载。此次没有修改训练源码、原测试、现有数据或已有权重；试验训练仅使用临时目录中的小型数据副本。

**结论：本轮确有实质改进，90 项既有测试全部通过；但“所有问题已经修复、全链路验证完成、可直接开展全量训练”的结论仍不成立。实际首次训练能够完成，随后同配置续训立即失败；固定哨兵的样本、质量门禁和恢复状态仍有缺口。** 当前也没有独立测试集上的新精度结果，不能据此认定“高精度物种模型已经完成”。

## 1. 本次实际验证

环境：`C:\Users\123\.conda\envs\cantera\python.exe`，Python 3.12.12，PyTorch 2.11.0.dev20260120+cu130，Cantera 3.0.1。小型训练使用 CUDA；独立数值探针使用 CPU/float32。

| 验证 | 结果 | 可以说明什么 |
| --- | --- | --- |
| 原 `test/` + 上轮 10 项回归测试 | **90 passed, 2 warnings in 18.55s** | 既有测试覆盖的行为已通过；上轮回归测试内容未被改写 |
| mini 数据取 4 条轨迹、每条 16 行，显式分开 3 条训练/1 条验证 | 64 行；1 epoch 成功 | 真实 Parquet、scaler、PD 标签、网络前后向、验证、保存可以运行 |
| 训练 H=1，独立评估 H=1/8 | 两份 `_H1.pth`、`_H8.pth` 成功生成 | 验证展开不再受当前训练 H 限制，哨兵保存已真正接入 |
| 使用同一输出目录 `resume='auto'` 再次训练 | **RuntimeError: Refusing incompatible scaler transform** | 保存与续训链路仍有阻塞 |
| 完整 PD 解码 float32 梯度扫描 | **168 组，0 组非有限梯度** | 上轮极低浓度梯度问题在所测范围内已修复 |
| 本轮新增的独立反例回归测试 | **5 failed in 2.40s** | 质量门禁、两种模式的无状态 D mask、惰性物种质量统计、报告命令仍有错误 |

168 组扫描覆盖 `log10(Y)∈{-40,-39.5,-39,-38,-30,-20,-6}`、`zP∈{-20,-19,-6}`、`zD∈{-10,0,4,10}`、`dt∈{1e-8,1e-4}`。这不是“任意网络输出下导数全局有界”的证明，也不是长程训练稳定性或泛化精度验收。

新增测试有意断言应当成立的契约，其失败是可复现的缺陷证据，不是随机抽样得到的代码故障比例。没有运行全量训练。

## 2. 需要返修的问题

### F1 · P1：新模式首次训练成功，但自产 scaler 阻断第二次启动

位置：[主入口签名计算](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:5801>)、[加载时比较](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:6085>)。

主入口局部 `transform_signature` 的分支仍未包含 `dual_branch_pd_v1`，因此取到 `macro_asinh_p90_v2`。与此同时，`build_feature_info` 已正确生成 `dual_branch_pd_v1`。第一次训练不触发已有 scaler 校验，所以可以完成；第二次启动读取自己生成的文件便被拒绝。

实际产物：

- 文件名：`feature_info_nuig_2024_macro_asinh_p90_v2_split_5025cb2308c5ecf8.pkl`。
- 文件内 feature_info 签名：`dual_branch_pd_v1`。
- checkpoint 顶层及 manifest 签名：`macro_asinh_p90_v2`。
- checkpoint 内 feature_info 签名：`dual_branch_pd_v1`。
- 续训在第 6086 行报错，尚未进入权重恢复。

建议：所有入口、文件名、feature_info、manifest 和 checkpoint 复用唯一的模式签名解析函数。对本轮错误命名产物提供明确迁移或重建办法，不能只改一处字符串后继续混用。验收必须包含同目录的“首次运行→保存→第二次运行续训”，并校验恢复的 epoch、模型、优化器和选模状态。

### F2 · P1：质量守恒门禁实际上没有生效

位置：[质量统计字段](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/species_validation_protocol.py:191>)、[门禁读取](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/species_validation_protocol.py:222>)。

统计写入 `max_unity_error`，门禁却读取 `max_abs_unity_error` 并默认返回 0。结果是质量和检验永远按零误差通过。

独立反例：H2、N2、O2 的真值各为 1/3，预测各减少 0.04。预测全部非负，各元素误差均为 0.04，低于当前元素门槛 0.05；但总质量和为 **0.88**，偏差 **0.12**，超过质量门槛 0.05，仍得到 **`is_valid_candidate=True`**。该反例独立隔离了质量门禁，不会被负值门禁代替检验。

建议：使用一致字段并对缺失字段报错，避免以 0 默默放行；分别覆盖负值、非有限值、总质量和各元素门禁。质量/元素允许误差的具体数值应根据精度目标另行确定。

### F3 · P2：验证计算总质量时漏掉 AR/HE，修复 F2 后会误拒正确预测

位置：[哨兵收集状态](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:4876>)、[哨兵物种列表](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:4927>)。

调用端只传入 `macro_species + micro_species`，排除了惰性物种；评分函数却要求传入物种的质量和等于 1。

通过实际 `validate_model_v68` 验证一个静止状态：`Y_AR=0.1`、`Y_H2≈0.9`、极微量 H，模型准确保持该状态。主物种误差为 0，但报告质量误差为 **0.1000000238**。现在由于 F2 的错误，该候选暂时被放行；只修键名后，正确的含惰性气体工况将被拒绝。

建议：总质量统计使用包含惰性物种的完整状态，或显式传入反应物种质量目标 `1-sum(Y_inert)`。同时保持元素矩阵、物种顺序和质量统计的契约一致。F2/F3 应一起修复。

### F4 · P1：哨兵已接入，但验证样本仍变化，默认模型仍由旧损失选择

位置：[验证数据 shuffle](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:6147>)、[每轮设置 epoch](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:6550>)、[默认模型保存](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:7550>)。

已修复部分：pure-AR 按最大评估 H 展开；物种指标和 `consider_candidate` 已调用，确实保存 `_H{h}.pth`。

仍存在：

1. 验证 dataset 使用 `shuffle_parts=True`，并每轮更新 epoch。当验证文件多于 `val_parts_per_epoch`（默认 9）时，每轮评估的工况发生变化。用 12 个文件、每轮取 9 个的探针，epoch 0/1/2 的成员集合不同。若全部文件均评估且不截断 batch，成员集合不受此项影响；但设置 `max_val_batches` 时窗口 shuffle 仍会改变评估样本。
2. scheduler、早停及默认 `best_v68_net_{mechanism}.pth` 仍依据 oracle total loss。默认模型保存分支不经过新的候选门禁。[批量验证器默认文件模式](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/validate_all_best_models_v68.py:72>) 也仍指向该文件，而不是指定的哨兵模型。
3. 当前采集每个 H 的终点，拼成二维数组后评分，没有保存每条轨迹的时间序列。产物中 `timing_metrics={}`；目前的 peak 是这些终点样本之间的最大值，不是每条轨迹的自由基峰值/峰时。终点哨兵本身有用，但不能当作完整轨迹精度报告。

建议：冻结明确的验证工况、窗口、步长及采样成员，记录其哈希；明确定义最终模型采用哪个 H 或什么聚合规则，并让默认验证入口使用这一选择结果。oracle loss 可以保留作诊断，不能把它选出的权重称为物种哨兵最优模型。

### F5 · P2：哨兵最优分数没有保存与恢复，续训会重置排行榜

位置：[无条件新建 tracker](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:6516>)、[latest checkpoint 保存](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:7829>)。

在模型恢复之后，仍重新构造 `HorizonSentinelTracker()` 并把 `best_val_loss_by_horizon` 置空。latest checkpoint 没有该 tracker 的 best_scores/best_epochs/best_records。修复 F1 后，续训第一个合格候选会相对于 `inf` 被视为最佳，可能覆盖磁盘上更好的 `_H{h}.pth`。

建议：保存/恢复 tracker 和按 H 的早停状态，或严格读取并校验已有每 H 最佳产物。验收：先保存优模型，重启后提交较差候选，原文件必须保持不变。

### F6 · P1：D mask 的无状态回退仍把真实衰减当作休眠

位置：[dual 分支回退](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:935>)、[验证损失没有传入状态](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:4325>)。

新回退条件是 `(z1 < -15) & (z2 <= 4)`。它使上轮 `D=1e8` 的测试通过，却不能判定物种是否休眠；而 PD 模式的 z1 已表示生成率，不再表示 QSSA 浓度。

反例：`Y_old=1e-4`、`D=1e4 s^-1`、`dt=1e-4 s`、生成可忽略，则 `Y_next≈3.6788e-5`，单步消耗约 **63.2%**。这显然不是休眠态。旧 dual 目标 `[0,-16,4]` 和 PD 目标 `[0,-20,4]` 均得到 mask `[1,1,0]`，把 D 监督移除；给定实际起终状态后则正确不屏蔽。

影响：主训练的带状态调用已改善，但 `compute_training_loss` 的实际验证调用仍不传状态，导致训练与验证损失含义不一致，进一步影响仍依据 oracle loss 的调度/默认选模。

建议：没有状态就不判休眠、返回不屏蔽；或把真实起终状态贯通全部调用链。用 `D*dt`、活跃起点、纯衰减和零生成等多组物理反例测试，不能再用一个 D 阈值代替状态判据。

### F7 · P2：manifest 已写入，但机理哈希和加载校验仍未真正接入

位置：[manifest 构造](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/train_sc_v68.py:6258>)、[验证加载入口](<C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc16/validate_sc_v68.py:120>)。

实际生成的 checkpoint 显示：

```text
mechanism_sha256 = "unverified"
transform_signature = "macro_asinh_p90_v2"
feature_info.transform_signature = "dual_branch_pd_v1"
split_hash = null
dataset_audit = {}
dt_policy = {"min": 1e-8, "max": 1e-4}
```

scaler 数组哈希已有实际计算，是有效进展。但 `tools.mech_sha256` 没有赋值；元素名仍硬编码；主路径没有调用 `verify_checkpoint_compatibility`，也没有调用 `audit_parquet_dataset`。独立验证器允许多个签名，只检查成员资格，未比较顶层与 feature_info 的签名一致性。保存一个 manifest 字典并不等于加载时校验。

`dt_policy` 也仍记录旧上下限，而现在实际 dt 路径不再按这些值裁剪。最新/周期 checkpoint 的 manifest 存在于嵌套 feature_info/training_config 中，缺少 verifier 所期望的统一顶层入口。

建议：从实际使用的机理契约获取哈希、物种/元素顺序；统一保存位置；在 resume、warm-start、独立验证加载时比较实际机理、scaler、模式及划分契约，并以真实篡改产物测试拒绝行为。参数化和验证协议源码也应纳入版本记录。

### F8 · P2：报告给出的工作站命令仍不能执行

对报告完整参数直接调用当前 parser，得到 exit code 2：

```text
unrecognized arguments: --curriculum_warmup_epochs 5 --weight_decay 1e-5
```

现有课程参数是 `--curriculum_horizons` 和 `--curriculum_start_epochs`，没有上述 warmup 参数；AdamW 的 weight_decay 在源码中固定为 `1e-5`，没有 CLI 参数。另外，单独设置 `--induction_boost 6.0` 并不会开启 `--use_induction_weights`。这里仅核对命令行为，不建议未经消融就重新采用 6 倍加权。

建议：从 parser 实际能力生成命令，并在临时小数据上原样运行。先完成 F1/F2/F4/F6，再给出全量训练指令。

## 3. 上轮尚未关闭的其他问题

这些问题不因为 10 个回归测试全部通过而自动关闭。本次保留原报告和证据，并重新检查关键路径：

| 上轮问题 | 本轮状态与证据 | 后续验收 |
| --- | --- | --- |
| R8 故障快照不能精确重放 | `training_diagnostics.py` 未变；RNG 仍在失败后捕获。重新运行 Dropout 重放，输出最大差 **1.49146**，并非同一前向 | 保存 batch 前 RNG，恢复相同模型状态/模式和完整计算配置；随机前向必须逐元素重现；所有跳过路径进入统一预算 |
| R9 梯度检查后的裁剪仍溢出 | float64 检查认为 `[1e20,1e20]` 有限，norm≈`1.4142e20`；随后原 `clip_grad_norm_` 返回 **Inf** 并将梯度变为 **0** | 使用稳定范数执行实际缩放，并检查裁剪后的梯度；不能只稳定诊断函数 |
| R10 机理绑定/静默回退 | 显式给出不存在的机理路径，`build_element_matrix` 仍返回手写常数构造的矩阵，不报错 | 显式机理失败应拒绝继续；确认元素矩阵与实际数据生成机理一致 |
| R11 phi 元数据仍取燃烧中剩余燃料比 | builder 文件 SHA256 与上轮一致，仍逐行从剩余 NH3/H2/O2 重算。上轮实测 30 条轨迹中 15 条变化超过 0.1，最差约 2.67→97089.84 | 将初始当量比作为工况元数据保持不变；若还需要剩余燃料比，另设字段。当前 phi 非直接网络输入，不能把它直接解释为物种误差的已证实根因 |

R8/R9/R10 本次复现见 `supplement_evidence.json`；R11 数值沿用上轮证据并核对相关文件未改，未冒充本轮重新生成数据的结果。

## 4. 对“上轮全部修复”的逐项判定

| 上轮发现 | 判定 |
| --- | --- |
| R1 新模式不可达 | **部分修复**：首次训练通了，保存签名与续训仍失败（F1） |
| R2 极低浓度梯度 | **所测范围内修复**：168 组完整解码梯度均有限；没有证明长程全局稳定 |
| R3 condition_group_id 白名单遗漏 | **已修复该缺陷**：原始组 ID 驱动的回归通过；这不替代对实际生成数据/独立 benchmark 隔离的验收 |
| R4 固定验证未接入 | **部分修复**：展开及按 H 保存有效，固定样本、默认选模和恢复状态仍不完整（F4/F5） |
| R5 零自由基及物理门禁 | **部分修复**：零预测误差、负值/元素门禁已改善；总质量门禁及惰性质量仍错（F2/F3） |
| R6 D mask 越界与回退 | **部分修复**：旧 qssa 布局越界测试通过；无状态回退仍错误（F6） |
| R7 manifest | **部分修复**：生成已接入，真实契约和加载校验未接入（F7） |
| R8/R9/R10/R11 | **未关闭**，见上一节 |

## 5. 建议的返修顺序与放行条件

1. **先修签名和实际续训**：统一模式契约，完成真实数据的两次启动；同时保存恢复哨兵状态。
2. **修评分和选择链路**：质量键、惰性质量、固定样本、默认权重选择、每 H 最优不可被更差续训候选覆盖。
3. **修 D mask 及诊断链路**：用物理状态判休眠，接通稳定裁剪和可精确重放的快照。
4. **使契约可校验**：真实机理哈希、scaler/划分校验和严格失败；修正报告命令。
5. **然后做精度验收**：使用冻结、与训练工况隔离的数据，分别报告主物种、活跃自由基、NO/N2O 的单步及多步误差；同协议比较旧/新模式。训练通过、守恒通过、测试通过与预测精度达标分别报告。

无需因本次审查先扩展 CFD 部署工作。当前更值得投入的是让训练目标、验证分数和最终选出的权重在同一套物理契约下可重复比较。

## 6. 复现材料

全部位于本目录，历史 `gemini_implementation_review` 未被覆盖：

- `probe_revision2.py` / `evidence.json`：真实小型训练、续训失败、保存签名、manifest、门禁/掩码反例及源码哈希。
- `smoke_train.log`：完整微型训练和第二次启动日志。
- `probe_supplement.py` / `supplement_evidence.json`：168 组梯度、含惰性状态验证、抽样成员、旧诊断问题。
- `test_followup_regressions.py` / `followup_regression_results.txt`：5 项独立反例及失败信息。
- `source_snapshot/`：本轮审查时训练/数据脚本副本；`gemini_report_reviewed.md`：被审阅报告副本。
- `baseline_test_result.txt`：本轮已执行的 90 项测试结果记录。

在 sc16 工作目录执行：

```powershell
$env:CANTERA_DATA = (Get-Location).Path
& 'C:\Users\123\.conda\envs\cantera\python.exe' -m pytest test analysis_20261015/gemini_implementation_review/test_review_regressions.py -q --disable-warnings
& 'C:\Users\123\.conda\envs\cantera\python.exe' analysis_20261015/gemini_revision2_review/probe_revision2.py
& 'C:\Users\123\.conda\envs\cantera\python.exe' analysis_20261015/gemini_revision2_review/probe_supplement.py
& 'C:\Users\123\.conda\envs\cantera\python.exe' -m pytest analysis_20261015/gemini_revision2_review/test_followup_regressions.py -q --disable-warnings
```

这些探针重跑会更新本目录的证据文件，但不写入原有训练产物。临时训练数据、权重自动清理；日志和汇总元数据保留。
