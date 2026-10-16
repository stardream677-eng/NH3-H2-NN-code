# sc16 — 自主 T/P/Y、H=128 全轨迹基准

这是一套**评估工具，不是预先得出的精度结论**。运行者必须先拥有独立训练产生的、配置为 `temperature_mode=autonomous` 且 `feedback_state_source=constrained` 的检查点。PR #3–#6 的单元测试通过，并不表示神经网络已经在生产数据上完成 H=128 训练。

## 与旧 CFD 基准的区别

`test/benchmark_cfd_closed_loop.py` 是历史分析脚本，它调用 Cantera `UVY/HPY` 为每一步神经预测求解温度，且另有独立的 NumPy 投影，**不能把它的结果称为生产网络完全自主闭环精度**。

新工具 `test/benchmark_autonomous_h128.py` 使用真实检查点和训练同款状态推进：预测物种 → 现有 KL/Dykstra 投影 → NASA7 能量闭合 → EOS 压力反馈。下一步只接收本步预测 T/P/Y 和预定的下一步 dt。Cantera 在另一份反应器中生成参考轨迹，并在步进结束后独立复算物理残差，绝不反馈修复神经预测。

## 运行准备

- 要求 `NUIG2024.yaml`（或与检查点相匹配的 TH/UCF 机理）、保存于训练目录的 **原始** Scaler 与 `best_v68_net_*_H128.pth`。
- 验证器会严格检查机理 SHA-256、Scaler Hash、Checkpoint Feature Contract、`pressure_feedback_contract` 和 `projection_validation_contract`。不允许自动重建 Scaler。
- 例子文件 `benchmark_autonomous_cases.example.json` 提供四个 UV/HP 场景，每个固定 128 步。它们**只是探索性案例**，并没有证明与训练的 `condition_group_id` 严格隔离。若要报告独立泛化精度，需自行检查工况分组并提供被冻结的真正外推集。

示例（PowerShell，工作目录为 `sc16`）：

```powershell
& 'C:\Users\123\.conda\envs\cantera\python.exe' test/benchmark_autonomous_h128.py `
  --checkpoint output/weights_v68/best_v68_net_nuig_2024_H128.pth `
  --scalers-root output `
  --mechanism nuig_2024 `
  --mechanism-path NUIG2024.yaml `
  --cases test/benchmark_autonomous_cases.example.json `
  --output output/benchmark_autonomous_h128 `
  --device cpu
```

按实际训练产物替换 `--checkpoint` 与 `--scalers-root`。不要使用旧版仅 `recorded` 温度、仅自主 T 而不是 T/P、或不具备严格 KL 版本契约的检查点冒充完整自主模型。

## 指标定义

- **IDT**：从原始 T0 起首次达到 `T0+400 K` 的时间，以固定时间节点之间**线性插值**估计。这是显式选择的温升阈值 IDT，**不是**最大 `dT/dt`、OH 峰值或定压传统 IDT；使用其他定义需重新标注，不允许混用。
- **未点火**：128 步观察窗内没达到阈值则以 `null` 标记右删失，不当作零时延。只有参考与模型两者均在窗内点火时才计算 IDT 误差及比值。
- **温压**：逐时刻 MAE/RMSE、峰值、峰值时刻、最大温升。
- **组分**：NH3/H2/O2/H2O、OH/H/O 以及 NO/NO2/N2O 的逐轨迹峰值、峰值时间与末态误差；活跃自由基计算 log10 RMSE；完整全部组分储存在每一条 `.npz` 中。
- **NOx**：若机理含 NO 和 NO2，额外报告 `Y_NO + Y_NO2` 的轨迹统计（质量分数口径）。
- **物理门禁**：每一步从完整预测物理状态重新计算质量、元素、绝热 UV/HP 能量及理想气体 EOS 的残差。默认上限是质量 `5e-6`、元素 `1e-6`、能量 `10 J/kg`、EOS 相对误差 `1e-5`。
- **失败**：NASA7 未收敛、投影无法满足约束、压力异常、参考轨迹故障等均保留失败案例及步号，不纳入成功案例的平均 IDT。命令行有失败案例时退出码为 `2`。

## 输出与限制

`benchmark_report.json`：版本协议、病例清单和 SHA-256、完整场景记录与失败原因、软硬件版本、源码/机理/检查点哈希及统计汇总。`<case_name>.npz`：全时序 `time_s/ref_T_K/ref_P_atm/ref_Y/nn_T_K/nn_P_atm/nn_Y`（失败案例的网络数组可以比参考短）。物种数组的列顺序由报告 `species_order` 指定。

这套脚本不会在这里自动运行训练或生产基准。只有跑完独立条件的真实检查点 H=128 自回归，并报告实际误差、失败率与 IDT 事件一致性之后，才能判断是否适合 CFD 使用。
