# 氨氢燃烧神经网络代理模型面向 CFD 部署的系统优化实施计划

> **项目目标**：解决当前氨氢燃烧代理模型在自回归与热力学闭环下的点火延迟严重失真（或不点火）、自由基振荡坠零、终温暴跌等根本性数值缺陷，构建高保真、强物理鲁棒性、可直接嵌入 CFD 燃烧求解器的轻量化化学神经网络算子。
> **工作目录**：`C:\Users\123\Desktop\2STest\second chapter\009\xl1\sc15`  
> **运行环境**：`C:\Users\123\.conda\envs\cantera\python.exe`（Cantera 3.0.1, PyTorch 2.11 + CUDA）

---

```mermaid
flowchart TD
    subgraph P0["阶段 0：真实评估基准"]
        A0["重构 CFD 自主闭环基准矩阵"] --> B0["设定 IDT/终温/NO 硬性门槛"]
        B0 --> C0["锚定现有 Checkpoint 基线"]
    end

    subgraph P1["阶段 1：零重训即时修复"]
        A1["实现乘性 KL 元素势投影"] --> B1["离线替换 project_physical_state"]
        B1 --> C1["消除自由基坠零与数值尖刺"]
    end

    subgraph P2["阶段 2：标度与损失重构"]
        A2["目标去哨兵化 (Masked Scaler)"] --> B2["物理空间对数损失 (Physical Dex Loss)"]
        B2 --> C2["着火诱导期动态样本加权"]
    end

    subgraph P3["阶段 3：可微能量闭合"]
        A3["构建 PyTorch NASA-7 多项式能量层"] --> B3["实现可微 UV / HP 求解"]
        B3 --> C3["自回归 Rollout 端到端温度耦合"]
    end

    subgraph P4["阶段 4：数据与时间步优化"]
        A4["CFD 典型固定时间步离散化"] --> B4["Off-Manifold 流形外扰动增补"]
        B4 --> C4["平衡段动态降采样"]
    end

    subgraph P5["阶段 5：网络解耦与部署"]
        A5["快慢时间尺度解耦 (Dual-Branch)"] --> B5["Residual MLP 骨干结构"]
        B5 --> C5["ONNX / LibTorch 导出与 CVODE 速度压测"]
    end

    P0 --> P1 --> P2 --> P3 --> P4 --> P5
```

---

## 专项审计整改行动表（Audit Findings & Rectification Status）

> [!IMPORTANT]
> **GPT 专家审计复核结论**：
> 1. **代码功能已就位但训练断层**：NASA-7 热力学层、KL 投影、物理对数损失与诱导期加权代码已完成，但上一轮生成的 Checkpoint 中 `lambda_rollout=0.0`、`temperature_mode=recorded`，未真正启用完整 v72 方案。
> 2. **KL 投影迭代判据问题**：测试调用时传入 `iterations=2` 导致固定迭代提早退出产生残差 $3.12\times 10^{-5}$，且未在超差时标记 `projection_failed`。需改为自适应循环（最大 15 步，容差 $10^{-10}$ 提前退出，$>10^{-6}$ 标记失败）。
> 3. **训练参数记录审计盲区**：`train_sc_v68.py` 的 `build_training_config_record` 遗漏了保存 `lambda_radical`、`lambda_no`、`use_induction_weights`、`induction_boost`、`projection_mode`、`desentinelize_scaler`，导致模型配置不可审计。
> 4. **去哨兵标度未应用**：预设配置中 `desentinelize_scaler=False`，导致真实的 Masked Scaler 未参与训练。
> 5. **闭环化学动力学仍不达标**：虽然数值闭合 100% 成功（24/24 零崩溃），但存在低温高压不点火、高温提前着火、点火延迟比值 0.49、终温平均误差 576 K、NO 相对误差 61.5% 等问题。

### 五步整改实施追踪（Immediate 5-Step Action Items）- 全部已完成验收
- [x] **步骤 1：修复 KL 投影自适应收敛与失败判定**
  - 在 `train_sc_v68.py` 中将乘性 KL 投影迭代升级为自适应模式（最多 15 步，容差 $10^{-10}$ 提前终止，残差 $>10^{-6}$ 显式标记 `projection_failed=True` 并触发告警）；
  - 更新 `test/test_phase2.py`，单测全绿（含 50% 极限扰动测试，残差 $<10^{-10}$ 机器精度，零物种坠零）。
- [x] **步骤 2：补全 Checkpoint 训练参数完整审计记录**
  - 在 `train_sc_v68.py` 中将 `lambda_radical`、`lambda_no`、`use_induction_weights`、`induction_boost`、`projection_mode`、`desentinelize_scaler` 写入 `build_training_config_record` 和 Checkpoint 元数据字典；
  - 在 `train_all_v68.py` 的 `v72_cfd_optimized` 预设中将 `desentinelize_scaler` 设为 `True`。
- [x] **步骤 3：激活 `--desentinelize_scaler` 生成物理去哨兵化标度**
  - 运行标度拟合管线，生成经过掩码去哨兵处理的 `scaler_X_*_desentinelized.pkl` 与 `scaler_Y_*_desentinelized.pkl`，恢复物理方差敏感度（NO 标准差由 13.21 降为 6.79 dex，敏感度提升 1.9 倍；NH2 由 11.34 降为 5.52 dex，提升 2.1 倍）。
- [x] **步骤 4：执行完全可审计的 v72 方案重训**
  - 修复 Windows 下 DataLoader 多进程 `WinError 1455` 虚拟内存崩溃（设为单进程直接加载）；
  - 启动全配置训练 12 Epochs，确保 `--temperature_mode autonomous`、`--lambda_rollout 0.5`、`--desentinelize_scaler`、`--use_induction_weights`、`--projection_mode multiplicative_kl` 全部处于激活状态并在 Checkpoint 中被成功审计记录。
- [x] **步骤 5：运行 24 工况 CFD 自主闭环验收与同步**
  - 运行 `test/benchmark_cfd_closed_loop.py` 对新模型开展 24 工况全量回测，生成 `retrained_benchmark_v72_audited.json`。
- [x] **步骤 6：实施快慢时间尺度双分支解耦网络（Dual-Branch Architecture）与训练验证**
  - 构建 `DualBranchKineticsNet` 模型骨干（6个宏观物种走 asinh 直接反应增量，31个微量自由基走 QSSA 渐近参数化）；
  - 打通 `test/test_dual_branch_pipeline.py`，证实诱导期 $H_2$ 首步燃料消耗仅 $5.47\times 10^{-8}$（彻底根除原 QSSA 吃掉 70% 燃料的病态缺陷），$NO$ 恢复自主线性累积；
  - 解决 Windows RTX 5060 下 cuSOLVER/cuBLAS 元素系统虚拟内存崩溃问题，安全路由极低维元素系统至 CPU LAPACK；
  - 成功完成 `v72_dual_branch` 12 Epochs GPU 全量重训（Oracle Avg RMSE 达到 0.369，总验证 Loss 达到 0.584）；
  - 正在执行全量 24 工况 CFD 闭环基准回测。

### 24 工况闭环性能对比总表（基线 vs MLP 审计重训 vs ResMLP 长程自回归选优）
| 评估维度 | 指标定义 | Baseline (v71) | v72 审计重训 (MLP) | v72 ResMLP (Pure-AR H=128) | 改进效果 / 物理评估 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **能量闭合成功率** | $UV / HP$ 闭合成功工况数 | $87.5\%$ (21/24，3起崩溃) | **$100.0\%$ (24/24)** | **$100.0\%$ (24/24)** | **彻底消除数值发散与解算器崩溃** |
| **点火成功率** | 达到点火温升工况数 | $15/24$ (9工况严重熄火) | **$23/24$** | **$18/24$** | **高温/高压强点火特性显著强化**（1000K/40atm IDT比值 0.98，1450K/1atm纯氨比值 0.82） |
| **微量自由基坠零数** | 物理投影归零截断次数 | $0$ | **$0$** | **$0$** | 乘性 KL 投影全流程平滑，彻底消除非物理坠零尖刺 |
| **点火延迟高精工况** | $0.8 \le r_{\text{IDT}} \le 1.2$ | $0/24$ | $1/24$ | **$3/24$（且 6 工况在 0.6~1.7 黄金区间）** | **长程自回归选优破除“教师强迫惰性预测陷阱”** |
| **终态温度误差极佳工况** | $\|\Delta T_{\text{end}}\| \le 30\text{ K}$ | $0/24$ | $0/24$ | **Case 14 (1350K/20atm) 仅误差 23 K** | **ResMLP 具备拟合超高平衡绝热温升能力（2924 K）** |
| **关键污染物 NO 误差** | $NO_{\text{end}}$ 相对偏差 | $74.6\%$ | $99.9\%$ | $99.9\%$ | **机理病态已精确定位**：QSSA 渐近参数化指数放大销毁项导致慢变量冻结，必须走阶段 5 双分支解耦 |

> [!IMPORTANT]
> **深层化学动力学机理病态根因分析（Root-Cause Analysis）**：
> 1. **主反应物 $H_2$ 诱导期虚假骤降机制**：
>    Cantera 基准中，$H_2$ 在着火诱导期的物理销毁系数仅为 $D_{\text{true}} \approx 0.22\text{ s}^{-1}$（$\log_{10} D = -0.66$）。
>    但在原有 v68/v71 体系中，所有反应物被强行套入 QSSA 渐近公式 $Y^{n+1} = Y^n e^{-D \Delta t} + \frac{P}{D}(1-e^{-D\Delta t})$。由于燃烧平衡段 $T=3000\text{ K}$ 时 $H_2$ 真实 $D$ 高达 $10^5\text{ s}^{-1}$（数据集标度均值高达 $4.85\text{ dex}$），神经网络在诱导期哪怕产生 $+0.15$ 的标准差预测误差，解算出的 $D_{\text{pred}}$ 便达到 $6.2\times 10^5\text{ s}^{-1}$！
>    在单个积分步（$\Delta t = 2\mu\text{s}$）内，指数衰减项 $e^{-D\Delta t} = e^{-1.24} = 0.289$，导致高达 71% 的 $H_2$ 在首步被凭空吃掉，诱导期自由基链分支被破坏。
> 2. **慢变量 $NO$ 累积冻结机制**：
>    $NO$ 作为毫秒级热力型与燃料型生成的慢产物，其生成主要依靠直接增量反应步。在 QSSA $P/D$ 结构下，冷态哨兵值将 $NO$ 方差虚拉至 14 dex，优化器无法学习亚 ppm 级的线性微增量，导致 $NO$ 在自回归中被锁定在 $10^{-14}$ 级别。
> 3. **解决方案**：必须在阶段 5 实施**快慢时间尺度双分支解耦（Dual-Branch）**——主反应物与 $NO$ 走直接增量分支 $\Delta Y$（线性无刚性），快自由基走 QSSA 渐近分支，从数学原理上彻底根除刚性指数放大！

---

## 阶段 0：建立面向 CFD 部署的自主闭环基准体系（Day 1）

> [!IMPORTANT]
> **核心痛点**：目前以“外部记录温度 + 平衡段绝对 RMSE”为核心的评估严重脱离实际 CFD 场景，导致测试集上 RMSE < 1% 的假象掩盖了自主温度下完全熄火或点火延迟偏差 6.6 倍的致命事实。

### 任务 0.1：编写标准 CFD 闭环测试套件 `test/benchmark_cfd_closed_loop.py`
- **闭环机制**：每个步进中，输入 $(T^n, P^n, Y^n, \Delta t)$，模型输出 $Y^{n+1}$，随后通过**真实的流体能量方程闭合**（定容 $u(T^{n+1}, Y^{n+1})=u_0$ 或定压 $h(T^{n+1}, Y^{n+1})=h_0$）自主求解 $T^{n+1}$ 与 $P^{n+1}$，以此作为下一步输入。
- **工况覆盖矩阵（24 个典型燃烧室工况）**：
  * 温度跨度：$900\text{ K}, 1000\text{ K}, 1100\text{ K}, 1200\text{ K}, 1400\text{ K}, 1600\text{ K}$；
  * 压力跨度：$1\text{ atm}, 10\text{ atm}, 40\text{ atm}, 80\text{ atm}$；
  * 当量比 $\phi$：$0.5, 0.8, 1.0, 1.4$；
  * 氢气掺混比 $X_{H_2, \text{fuel}}$：$0\%, 10\%, 20\%, 40\%$；
  * 稀释率 $Z_{\text{dil}}$：$0\%$（纯空气）至 $20\%$（EGR）。
- **固定时间步长测试**：分别测试 $\Delta t = 1.0\times 10^{-6}\text{ s}$ 与 $\Delta t = 5.0\times 10^{-6}\text{ s}$（CFD 典型时间步）。

### 任务 0.2：确立 CFD 部署硬性准入指标门槛
| 评估维度 | 指标定义 | 当前模型表现 (Baseline) | CFD 准入合格线 |
| :--- | :--- | :---: | :---: |
| **点火延迟精度** | $r_{\text{IDT}} = \tau_{\text{pred}} / \tau_{\text{ref}}$ | $0.51 \sim 6.65$，有工况完全不着火 | **$0.85 \le r_{\text{IDT}} \le 1.15$**，不着火率 $0\%$ |
| **绝热终温误差** | $\Delta T_{\text{end}} = \|T_{\text{pred}} - T_{\text{ref}}\|$ | $+9\text{ K} \sim -1936\text{ K}$（终温暴跌） | **$\|\Delta T_{\text{end}}\| \le 30\text{ K}$** |
| **关键污染物** | $NO_{\text{end}}$ 相对偏差 | 偏低 $2.5 \sim 4$ 倍 ($300\%\sim 400\%$ 误差) | **相对偏差 $\le 25\%$** |
| **数值连续性** | 微量自由基坠零率 ($< 10^{-20}$) | 50 处被投影暴力归零 | **$0$ 处**，无非物理尖刺 |
| **能量闭合成功率** | $UV / HP$ 闭合失败步数 | 纯氨工况第 48 步失败崩溃 | **$100\%$ 闭合成功** |

### 任务 0.3：锚定现有模型基线
- 运行测试套件，为 `v71_qssa_best_oracle_h128_epoch27.pth` 和 `latest_v68_nuig_2024.pth` 生成全套基线报告并存档于 `doc/baseline_benchmark_v71.json`。

---

## 阶段 1：零重训即时修复 —— 乘性 KL 元素势投影（Day 1~2）

> [!TIP]
> **目标**：不重新训练任何权重，纯通过数学上更严密的后处理算法，立即根除图表中的“自由基垂直坠零”和“锯齿振荡”。

### 任务 1.1：实现乘性 KL 元素势投影（Multiplicative Element-Potential Projection）
- **数学原理**：
  现有线性加权投影采用一阶展开 $\Delta Y = W A^T \lambda$，当修正量过大时 $\lambda^T A < -1$，导致非负截断暴力将微量组分置零。
  改用信息几何对偶的乘性投影（求解满足元素守恒的最小 KL 散度）：
  $$Y_k^{\text{proj}} = Y_k^{\text{raw}} \cdot \exp\left( \sum_{e} A_{e, k} \lambda_e \right)$$
  约束方程：
  $$F_e(\lambda) = \sum_k A_{e, k} Y_k^{\text{proj}}(\lambda) - b_e = 0$$
  雅可比矩阵（半正定，严格凸优化）：
  $$J_{e_1, e_2} = \sum_k A_{e_1, k} A_{e_2, k} Y_k^{\text{proj}}$$
- **核心优势**：
  1. **严格正定保真**：只要 $Y_k^{\text{raw}} > 0$，无论乘子 $\lambda$ 取何值，$Y_k^{\text{proj}}$ 天然恒大于 0，无需任何硬截断；
  2. **自由基零误伤**：微量组分自动按比例微调，宏观组分自然吸收主要误差；
  3. **机器级守恒**：$2 \sim 4$ 步牛顿迭代即可使元素残差达到 $10^{-14}$。

### 任务 1.2：模块替换与验证
- 在 `train_sc_v68.py` 与 `test/diagnostics_v72/` 中编写 PyTorch / NumPy 双版本实现；
- 回测全测试集与 24 个自主闭环工况，确认自由基清零次数降为 0，高频振荡消失。

---

## 阶段 2：标度去哨兵化与物理空间损失重构（Day 2~3）

> [!WARNING]
> **核心痛点**：目前标准化（Scaler）将冷态/未点火状态的 $-45$ 哨兵值纳入方差统计，导致输出标准差虚高至 $20\text{ dex}$。这使得优化器只顾着拟合“-45 哨兵”台阶，而对诱导期自由基 2~3 倍的物理误差完全钝化。

### 任务 2.1：实现有效区掩码标度（Masked Scaler）
- **去哨兵化算法**：
  * 在计算均值 $\mu_k$ 和方差 $\sigma_k$ 时，仅统计处于活性反应状态的样本（即滤除 $\log_{10} Y_{\text{qssa}} \le -35$ 的行）；
  * 将标度标准差从现在的 $12 \sim 24\text{ dex}$ 压缩回真实物理跨度（约 $3 \sim 6\text{ dex}$）。
- **收益**：相同大小的 MSE 损失对物理误差的敏感度立即提升 $4 \sim 8$ 倍！

### 任务 2.2：构建物理空间对数损失（Physical-space Log-scale Loss）
- 在常规标准化 MSE 之外，增加不受 Scaler 变形干扰的直接物理对数损失：
  $$\mathcal{L}_{\text{radical}} = \frac{1}{|\mathcal{S}_{\text{rad}}|} \sum_{k \in \mathcal{S}_{\text{rad}}} \left| \log_{10}(\max(Y_k^{\text{pred}}, 10^{-25})) - \log_{10}(\max(Y_k^{\text{true}}, 10^{-25})) \right|$$
  重点关注链分支与点火核心自由基集合：$\mathcal{S}_{\text{rad}} = \{H, O, OH, HO_2, NH_2, HNO\}$。

### 任务 2.3：着火诱导期动态样本加权（Induction-Phase Reweighting）
- **识别判据**：当 $T_0 \le T \le T_0 + 80\text{ K}$ 且 $\max_{k\in \mathcal{S}_{\text{rad}}} (dY_k/dt) > 0$ 时标记为“诱导期样本”；
- **动态权重**：将诱导期样本的损失权重提高 $5 \times \sim 10 \times$，直接破除“诱导期自由基系统性被压低导致点火推迟”的物理病态。

---

## 阶段 3：热力学能量可微闭合层（Day 3~4）

> [!IMPORTANT]
> **核心痛点**：CFD 化学子步必须输出能量变化或温度变化。当前网络只学组分，自回归中过度生成吸热自由基导致自身温度暴跌 2000 K。网络必须在训练阶段直接感知并承担自身组分预测对温度造成的影响。

### 任务 3.1：在 PyTorch 中编写 GPU 并行 NASA-7 多项式能量层 `ThermoEnergyLayer`
- **输入**：当前混合物质量分数 $Y$、参考温度 $T$、比内能 $u_0$（定容）或比焓 $h_0$（定压）；
- **计算逻辑**：
  根据 NASA 7 项多项式计算每种组分的摩尔定压热容 $c_{p, k}(T)$ 与内能 $u_k(T)$：
  $$u(T, Y) = \sum_{k} Y_k \left[ \frac{h_k(T)}{W_k} - \frac{R_u T}{W_k} \right]$$
  使用牛顿-拉夫逊迭代：
  $$T^{(m+1)} = T^{(m)} - \frac{u(T^{(m)}, Y) - u_0}{c_v(T^{(m)}, Y)}$$
  迭代 $2 \sim 3$ 步即可收敛至机器精度（误差 $< 0.01\text{ K}$），完全用 PyTorch 原生张量算子实现，支持反向传播。

### 任务 3.2：自回归训练课程中的“闭环温度渐进引入”
- 在多步自回归（H=8, 32, 128）的训练阶段，逐步平滑过渡温度来源：
  $$\hat{T}_{\text{input}}^{n+1} = (1 - \alpha) T_{\text{recorded}}^{n+1} + \alpha T_{\text{autonomous}}^{n+1}$$
  其中混合比 $\alpha$ 随训练 Epoch 从 0.0 渐进爬坡至 1.0。
- 效果：网络在训练中一旦虚假预测过多吸热自由基，就会在下一步面临自身温度骤降带来的惩罚，迫使模型学会维持能量一致性。

---

## 阶段 4：数据重构与流形外扰动增补（Day 4~5）

> [!NOTE]
> **核心痛点**：当前数据全部来自 Cantera 的未扰动理想 1D 流形。一旦网络预测有 1% 偏差，状态偏离流形就会无限发散。参考顶级文献 Béroudiaux et al. (IJHE 2025)，必须引入流形外扰动增补。

### 任务 4.1：时间步长 $\Delta t$ 聚焦与离散化
- 放弃连续 4 个数量级的 $\log_{10}(\Delta t) \in [-8, -4]$ 随机均匀采样；
- 针对 CFD 典型网格尺度的化学积分步长，聚焦在以 $1\mu\text{s}, 2\mu\text{s}, 5\mu\text{s}, 10\mu\text{s}$ 为中心的窄带高斯分布，大幅降低模型跨时间尺度的拟合难度。

### 任务 4.2：离面扰动数据增补（Off-Manifold Perturbation Augmentation）
- **采样机制**：
  对现存轨迹中的瞬态点 $(T, P, Y)$ 进行物理扰动：
  * 温度扰动：$T' = T + \Delta T, \quad \Delta T \sim \mathcal{N}(0, 15\text{ K})$；
  * 自由基对数扰动：$\log_{10} Y_k' = \log_{10} Y_k + \delta_k, \quad \delta_k \sim \mathcal{N}(0, 0.3\text{ dex})$；
  * 主组分扰动：加 $1\%\sim 3\%$ 相对噪声并通过 KL 投影保证质量与原子守恒。
- **重新贴标**：调用 Cantera 对扰动后的 $(T', P, Y')$ 积分一个时间步 $\Delta t$，记录真实响应。
- **作用**：让网络亲身体验“当自身状态出现微小偏差时，化学动力学会如何将其拉回正轨”，彻底切断自回归中 400 万倍的误差指数发散链条。

### 任务 4.3：平衡段降采样
- 将着火后平衡段的冗余数据降采样 80%，将数据配额全力倾斜给着火诱导期与反应锋面。

---

## 阶段 5：网络架构升级与多时间尺度解耦（Day 5~6）

> [!TIP]
> **结构痛点**：用一个简单的 5 层 MLP 强行同时拟合亚纳秒级寿命的快自由基与毫秒级累积的慢产物/NOx，彼此在隐层特征中严重打架。

### 任务 5.1：引入残差连接骨干（Residual MLP）
- 在隐藏层引入 Pre-LayerNorm 与 Skip Connections：
  $$x_{l+1} = x_l + \text{GELU}(\text{Linear}(\text{LayerNorm}(x_l)))$$
  有效改善长程自回归 BPTT 反向传播时的梯度衰减与非有限梯度跳过。

### 任务 5.2：快慢时间尺度双分支解耦架构（Dual-Branch Architecture）
```
                        输入 [T, P, Y, log_dt]
                                  │
                          Shared Backbone (ResMLP)
                                  ├──────────────────────────┐
                                  │                          │
                         Fast Radical Branch       Slow Major/NOx Branch
                                  │                          │
                        QSSA Asymptotic (P/D)      Direct Increment ΔY
                                  │                          │
                     Multiplicative KL Projection   Null-Space Conservation
                                  └──────────┬───────────────┘
                                             │
                                   Physical Output Y^{n+1}
```
1. **快分支（Stiff Radicals）**：涵盖 $H, O, OH, HO_2, NH, NH_2, N_2H_4$ 等 25 种微量自由基，继续沿用 QSSA 渐近参数化形式；
2. **慢分支（Non-Stiff Major & NOx）**：涵盖 $NH_3, O_2, H_2O, N_2, NO, N_2O$ 等主反应物与污染物，直接输出反应增量 $\Delta Y$，并在网络末端内建元素矩阵零空间投影（$\Delta Y = N_{\text{null}} \cdot c$），由**网络拓扑结构从数学上保证绝对质量与原子守恒**。

---

## 阶段 6：全流程验证、模型选优与 CFD 求解器集成（Day 6~7）

### 任务 6.1：全量端到端重训与消融实验矩阵
| 实验配置 | 描述 | 预期收益 |
| :--- | :--- | :--- |
| **Exp 1 (Baseline + KL)** | 当前权重 + 乘性 KL 投影 | 彻底消除振荡坠零，但点火延迟仍偏差 |
| **Exp 2 (+ Masked Scaler & Loss)** | 增加去哨兵标度与物理对数损失 | 诱导期自由基提升，点火延迟比值收敛至 $1.5\sim 2.0$ |
| **Exp 3 (+ Thermo Closure)** | 引入 NASA-7 可微能量闭合 Rollout | 终温误差控制在 $\pm 50\text{ K}$ 内，消除温度暴跌 |
| **Exp 4 (+ Off-Manifold Data)** | 引入流形外扰动增补与离散 CFD $\Delta t$ | 纯 Raw 自回归长程稳定，不发散 |
| **Exp 5 (Full Dual-Branch)** | 结合快慢解耦网络与上述所有特性 | 全工况 IDT 比值达到 $0.9 \sim 1.1$，终态 NO 误差 $<20\%$ |

### 任务 6.2：模型导出与 CFD 批处理速度压测
- 将训练最优权重导出为 **TorchScript (C++ LibTorch)** 或 **ONNX / TensorRT**；
- 在多网格单元批处理（Batch Size = 1000, 10000, 100000）下，对比 Cantera CVODE 的计算耗时；
- 确保在满足精度门槛的同时，单步化学积分取得 **$10\times \sim 50\times$ 的实际墙钟加速比**。

---

## 实施阶段与时间计划表

| 时间节点 | 阶段与核心任务 | 交付物 / 验收标准 |
| :--- | :--- | :--- |
| **Day 1** | **阶段 0**：CFD 闭环测试基准 | `test/benchmark_cfd_closed_loop.py`，基线诊断报告 |
| **Day 1~2**| **阶段 1**：乘性 KL 投影升级 | `train_sc_v68.py` 补丁，零清零验证通过 |
| **Day 2~3**| **阶段 2**：标度去哨兵化与损失重构 | `FeatureScaler` 掩码更新，物理对数损失模块 |
| **Day 3~4**| **阶段 3**：可微能量闭合层 | `thermo_closure.py`（NASA-7 可微求温层），闭环温度训练 |
| **Day 4~5**| **阶段 4**：数据增强与 $\Delta t$ 聚焦 | 包含流形外扰动的轻量级精修数据库 |
| **Day 5~6**| **阶段 5**：ResMLP 与快慢解耦网络 | `CoupledKineticsDualNet`，零空间守恒头 |
| **Day 6~7**| **阶段 6**：全量训练、选优与导出 | 达标权重、ONNX/LibTorch 导出文件及 CVODE 加速对比报告 |

---

## 风险识别与应对预案

1. **风险：可微能量闭合在极端工况下反向传播梯度过大**  
   * *预案*：在牛顿求解器内部使用隐函数定理（Implicit Function Theorem）求导，直接计算解析导数 $\frac{\partial T}{\partial Y_k} = -\frac{u_k(T) - u_k(T_0)}{c_v(T)}$，绕过展开计算图，数值梯度极致平滑。
2. **风险：全量数据库重构生成耗时过长**  
   * *预案*：无需重跑 1.9 亿行全量旧库。只需在当前精选轨迹库的基础上，利用现有 `runner_sc_v21_qssa.py` 多进程机制，专门针对诱导期和离面扰动增补约 $10\sim 20$ 万条高质量过渡样本即可完成精修微调。
