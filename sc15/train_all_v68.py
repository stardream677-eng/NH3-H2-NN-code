"""V68 batch training launcher.

Runs one V68 long-horizon species-autoregressive operator training per mechanism,
streams child output into a timestamped log, and writes a machine-readable summary.
The launcher does not alter the scientific training code; all V68 horizon, curriculum,
oracle/AR metrics, and artifact isolation are handled by train_sc_v68.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


VERSION = "V68"
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = SCRIPT_DIR / "train_sc_v68.py"
PYTHON_EXECUTABLE = Path(sys.executable).resolve()
DEFAULT_MECHANISMS = ["nuig_2024"]
TRANSFORM_SIGNATURE = "macro_asinh_p90_v2"
ROLLOUT_SIGNATURE = "external_recorded_t_p_dt_species_ar_experiments_v3"


def get_transform_signature(output_param: Optional[str]) -> str:
    """Resolve true transform signature from output parameterization."""
    if output_param == "qssa_asymptotic":
        return "qssa_asymptotic_v2"
    elif output_param == "qssa":
        return "qssa_log10_P_D_v2"
    elif output_param == "dual_branch":
        return "dual_branch_asinh_macro_qssa_micro_v1"
    return TRANSFORM_SIGNATURE

def _resolve_default_paths():
    configured_root = os.environ.get("V68_DATA_ROOT")
    roots = (
        (Path(configured_root).expanduser() if configured_root else None),
        Path.cwd(),
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
    )
    for root in roots:
        if root is None:
            continue
        for cand in (root, root / "database_Transient_V21_QSSA"):
            if not cand.exists() or not (cand / "trajectories_nuig_2024").exists():
                continue
            # Only accept directories that contain the expected data structure
            base = cand if not cand.name.startswith("database_") else cand.parent
            return cand, base / "phase68_runs"
    # Fallback: return SCRIPT_DIR even if data is not present (will fail validation later)
    return SCRIPT_DIR / "database_Transient_V21_QSSA", SCRIPT_DIR / "phase68_runs"

DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT = _resolve_default_paths()
# No implicit warm-start checkpoint: a removed or stale legacy weight must not
# prevent a fresh split-aware training run from starting. Use --warm_start
# explicitly when a compatible checkpoint is intentionally available.
DEFAULT_WARM_START = None

# V3 is the only trainable model. V1/V2/V4 are validation-only ablations that
# reuse the same V3 checkpoints and therefore must not be offered here.
PLAN_PRESETS = {
    "v68_qssa": {
        "name": "v68_qssa_exponential_integrator",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.0,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.0,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa",
        "rate_decode_mode": "qssa",
        "rate_z_limit": 10.0,
    },
    "v71_qssa": {
        "name": "v71_qssa_asymptotic_exact",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa_asymptotic",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa_asymptotic",
        "rate_decode_mode": "qssa_asymptotic",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "constrained",
    },

    "v72_dual_branch": {
        "name": "v72_dual_branch",
        "experiment_mode": "qssa",
        "state_update_mode": "dual_branch",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.1,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "lambda_radical": 1.0,
        "lambda_no": 1.0,
        "use_induction_weights": True,
        "induction_boost": 6.0,
        "projection_mode": "multiplicative_kl",
        "desentinelize_scaler": True,
        "network_arch": "dual_branch",
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 2.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "autonomous",
        "max_delta_temperature": 500.0,
        "output_parameterization": "dual_branch",
        "rate_decode_mode": "dual_branch",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "constrained",
        "batch_size": 1024,
        "val_batch_size": 2048,
    },

    "v72_cfd_optimized": {
        "name": "v72_cfd_optimized",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa_asymptotic",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.1,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "lambda_radical": 1.0,
        "lambda_no": 0.5,
        "use_induction_weights": True,
        "induction_boost": 6.0,
        "projection_mode": "multiplicative_kl",
        "desentinelize_scaler": True,
        "network_arch": "resmlp",
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 2.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "autonomous",
        "max_delta_temperature": 500.0,
        "output_parameterization": "qssa_asymptotic",
        "rate_decode_mode": "qssa_asymptotic",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "constrained",
    },

    "v71_raw_feedback": {
        "name": "v71_qssa_raw_feedback",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa_asymptotic",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa_asymptotic",
        "rate_decode_mode": "qssa_asymptotic",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "raw",
    },

    "v69_qssa": {
        "name": "v69_qssa_stabilized",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa",
        "rate_decode_mode": "qssa",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "raw",
    },

    # --- Phase 1 单因素消融对比方案 ---
    "v69_no_noise": {
        "name": "v69_qssa_no_noise",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.0,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa",
        "rate_decode_mode": "qssa",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "raw",
    },

    "v69_no_stability": {
        "name": "v69_qssa_no_stability",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.0,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 2.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa",
        "rate_decode_mode": "qssa",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "raw",
    },

    "v69_no_microweight": {
        "name": "v69_qssa_no_microweight",
        "experiment_mode": "qssa",
        "state_update_mode": "qssa",
        "lambda_rate": 1.0,
        "lambda_state": 1.0,
        "lambda_element": 0.0,
        "lambda_mass": 0.0,
        "lambda_positive": 0.0,
        "rate_loss_kind": "mse",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.01,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.005,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.0,
        "key_species_loss_weight": 1.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "qssa",
        "rate_decode_mode": "qssa",
        "rate_z_limit": 10.0,
        "state_loss_source": "raw",
        "feedback_state_source": "raw",
    },

    "v3": {
        "name": "v3_constrained_weighted",
        "experiment_mode": "constrained_delta",
        "lambda_rate": 0.5,
        "lambda_state": 2.0,
        "lambda_element": 0.2,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.1,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.01,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.5,
        "key_species_loss_weight": 3.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "direct_rate",
        "rate_decode_mode": "direct_rate",
        "rate_z_limit": 10.0,
    },
    "v34": {
        "name": "v3_4_soft_feedback_trust_region",
        "experiment_mode": "constrained_delta",
        "lambda_rate": 0.5,
        "lambda_state": 2.0,
        "lambda_element": 0.2,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.1,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.01,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.5,
        "key_species_loss_weight": 3.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        # V3.4: gradually enable predicted feedback and limit macro jumps.
        "feedback_blend_start": 0.35,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 8,
        "trust_region_fraction": 0.25,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "direct_rate",
        "rate_decode_mode": "direct_rate",
        "rate_z_limit": 10.0,
    },
    "v35": {
        "name": "v3_5_autonomous_temperature",
        "experiment_mode": "constrained_delta",
        "lambda_rate": 0.5,
        "lambda_state": 2.0,
        "lambda_element": 0.2,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.1,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.01,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.5,
        "key_species_loss_weight": 3.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        # V3.5 keeps the mature feedback path; only T update is autonomous.
        "feedback_blend_start": 1.0,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 0,
        "trust_region_fraction": 0.0,
        "temperature_mode": "autonomous",
        "max_delta_temperature": 250.0,
        "output_parameterization": "direct_rate",
        "rate_decode_mode": "direct_rate",
        "rate_z_limit": 10.0,
    },
    "v68": {
        "name": "v68_bounded_direct_rate",
        "experiment_mode": "constrained_delta",
        "lambda_rate": 0.5,
        "lambda_state": 2.0,
        "lambda_element": 0.2,
        "lambda_mass": 0.05,
        "lambda_positive": 0.01,
        "rate_loss_kind": "huber",
        "rate_huber_beta": 0.5,
        "lambda_rollout": 0.5,
        "lambda_stability": 0.1,
        "transition_weight_alpha": 2.0,
        "transition_weight_clip": 8.0,
        "feedback_noise_std": 0.01,
        "macro_loss_weight": 1.0,
        "micro_loss_weight": 1.5,
        "key_species_loss_weight": 3.0,
        "tail_start": 0.80,
        "tail_end": 0.95,
        "tail_ramp_epochs": 8,
        "backward_chunk_steps": 16,
        "feedback_blend_start": 0.35,
        "feedback_blend_end": 1.0,
        "feedback_blend_ramp_epochs": 8,
        "trust_region_fraction": 0.25,
        "temperature_mode": "recorded",
        "max_delta_temperature": 1000.0,
        "output_parameterization": "direct_rate",
        "rate_decode_mode": "bounded_direct_rate",
        "rate_z_limit": 10.0,
    },
}

ACTIVE_COMPARISON_PLANS = [
    "v71_qssa",
    "v71_raw_feedback",
    "v69_qssa",
    "v68_qssa",
    "v69_no_noise",
    "v69_no_stability",
    "v69_no_microweight",
    "v68",
]

PLAN_MENU = {
    "1": "v71_qssa",
    "1b": "v71_raw_feedback",
    "2": "v69_qssa",
    "3": "v68_qssa",
    "4": "v69_no_noise",
    "5": "v69_no_stability",
    "6": "v69_no_microweight",
    "7": "v68",
    "8": "v71_raw_feedback",
    "0": "v72_dual_branch",
    "dual": "v72_dual_branch",
    "dual_branch": "v72_dual_branch",
    "v72_dual": "v72_dual_branch",
    "v72_dual_branch": "v72_dual_branch",
    "0b": "v72_cfd_optimized",
    "9": "v72_cfd_optimized",
    "A": "all",
    "a": "all",
    "all": "all",
    "v72": "v72_dual_branch",
    "v72_cfd": "v72_cfd_optimized",
    "v72_cfd_optimized": "v72_cfd_optimized",
    "v71": "v71_qssa",
    "v71_qssa": "v71_qssa",
    "v71_raw": "v71_raw_feedback",
    "v71_raw_feedback": "v71_raw_feedback",
    "v69": "v69_qssa",
    "v69_qssa": "v69_qssa",
    "v68": "v68",
    "v68_qssa": "v68_qssa",
    # Legacy backward compatibility mappings:
    "v3": "v3",
    "v34": "v34",
    "v35": "v35",
}


def select_plan_interactively(input_fn=input) -> str:
    """Select one training strategy or all comparison strategies."""
    print("\n" + "=" * 68)
    print("请选择训练方案 (支持单选或一键批量运行所有对比方案):")
    print("=" * 68)
    print("  0.  v72_dual_branch: 双分支多时间尺度解耦架构 (宏观asinh直接率+微观QSSA渐近率+交叉耦合，彻底根除H2衰减与NO冻结) [Phase 5 终极方案]")
    print("  0b. v72_cfd_optimized: 单分支 QSSA 优化版 (KL元素势投影 + 诱导期加权 + 自由基/NO损失 + NASA-7闭环) [对照方案]")
    print("  1.  v71_qssa: 渐近 QSSA + 物理加权投影闭环回灌 (训练与CFD部署严格统一，消除暴露偏差) [标准推荐]")
    print("  1b. v71_raw_feedback: 消融对照：相同渐近 QSSA + raw 自由状态自回归回灌 (Claude消融方案) [对照实验]")
    print("  2.  v69_qssa: 闭环稳定增强版 QSSA (Phase 1 旧方案：对数加权 + 状态噪声 + 导数平滑 + 阻尼保护)")
    print("  3.  v68_qssa: 纯指数积分器 QSSA 基线 (无噪声扰动，无稳定性损失，用于对照)")
    print("  4.  v69_no_noise: 消融实验：仅稳定性损失 + 对数加权 (无状态噪声)")
    print("  5.  v69_no_stability: 消融实验：仅状态噪声 + 对数加权 (无稳定性损失)")
    print("  6.  v69_no_microweight: 消融实验：仅状态噪声 + 稳定性损失 (微观权重为 1.0)")
    print("  7.  v68: 经典直接速率回归 (Bounded direct-rate baseline)")
    print("  A.  all: 一键顺序训练上述所有方案 (批量消融与系统性对比)")
    while True:
        choice = input_fn("请输入方案编号 (0/0b/1/1b/2/3/4/5/6/7 或 A): ").strip()
        if choice in PLAN_MENU:
            selected = PLAN_MENU[choice]
            if selected == "all":
                print("已选择: 批量运行所有对比方案 (all)")
            else:
                print(f"已选择 {selected}: {PLAN_PRESETS[selected]['name']}")
            return selected
        print("输入无效，请输入 0、1、1b、2、3、4、5、6、7 或 A。")


def select_mode_interactively(input_fn=input) -> str:
    """Select between test mode (5 epochs) and full mode (30 epochs)."""
    print("\n" + "=" * 68)
    print("请选择训练模式:")
    print("=" * 68)
    print("  1. 测试/消融模式 (5 Epochs 快速验证，轻量采样，用于快速检查 raw 稳定性和指标趋势) [推荐测试]")
    print("  2. 正式训练模式 (30 Epochs 完整生产训练，全量数据收敛)")
    while True:
        choice = input_fn("请输入模式编号 (1/2): ").strip()
        if choice in ("1", "test", "t"):
            print("已选择: 测试/消融模式 (5 Epochs)")
            return "test"
        elif choice in ("2", "full", "f"):
            print("已选择: 正式训练模式 (30 Epochs)")
            return "full"
        print("输入无效，请输入 1 或 2。")


def apply_plan_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply a built-in plan and isolate its artifacts under one directory."""
    if args.plan is None:
        return args

    explicit_keys = getattr(args, "_explicit_keys", set())

    if args.plan == "all":
        args.preset_name = "batch_all_plans"
        if not getattr(args, "_curriculum_explicit", False):
            if getattr(args, "mode", None) == "test":
                args.curriculum_horizons = [1, 8, 32, 128]
                args.curriculum_start_epochs = [0, 1, 2, 3]
            else:
                args.curriculum_horizons = [1, 8, 32, 128]
                args.curriculum_start_epochs = [0, 2, 5, 8]
        return args

    preset = PLAN_PRESETS[args.plan]
    for key in (
        "experiment_mode",
        "lambda_rate",
        "lambda_state",
        "lambda_element",
        "lambda_mass",
        "lambda_positive",
        "lambda_rollout",
        "lambda_stability",
        "transition_weight_alpha",
        "transition_weight_clip",
        "feedback_noise_std",
        "macro_loss_weight",
        "micro_loss_weight",
        "key_species_loss_weight",
        "feedback_blend_start",
        "feedback_blend_end",
        "feedback_blend_ramp_epochs",
        "trust_region_fraction",
        "temperature_mode",
        "max_delta_temperature",
        "rate_loss_kind",
        "rate_huber_beta",
        "output_parameterization",
        "rate_decode_mode",
        "rate_z_limit",
        "state_update_mode",
        "state_loss_source",
        "feedback_state_source",
        "micro_rate_z_limit",
        "lambda_radical",
        "lambda_no",
        "use_induction_weights",
        "induction_boost",
        "projection_mode",
        "desentinelize_scaler",
        "network_arch",
        "batch_size",
        "val_batch_size",
        "train_stride",
    ):
        if key in preset and key not in explicit_keys:
            setattr(args, key, preset[key])

    args.preset_name = preset["name"]

    # New split-aware runs use staged curriculum unless explicitly overridden.
    if "max_horizon" not in explicit_keys:
        args.max_horizon = max(args.max_horizon, 128)
    if not getattr(args, "_curriculum_explicit", False):
        if getattr(args, "mode", None) == "test":
            args.curriculum_horizons = [1, 8, 32, 128]
            args.curriculum_start_epochs = [0, 1, 2, 3]
        else:
            args.curriculum_horizons = [1, 8, 32, 128]
            args.curriculum_start_epochs = [0, 2, 5, 8]
    elif len(args.curriculum_horizons) != len(args.curriculum_start_epochs) and "curriculum_start_epochs" not in explicit_keys:
        n = len(args.curriculum_horizons)
        if args.epochs <= n:
            args.curriculum_start_epochs = list(range(n))
        else:
            args.curriculum_start_epochs = [int(i * args.epochs / n) for i in range(n)]
    if "eval_horizons" not in explicit_keys:
        args.eval_horizons = [h for h in [1, 2, 4, 8, 16, 32, 64, 128] if h <= args.max_horizon]
        if not args.eval_horizons:
            args.eval_horizons = [args.max_horizon]
    # V3 starts with more predicted-state feedback because it is intended for
    # warm continuation from a mature checkpoint.
    if "tail_start" not in explicit_keys:
        args.tail_start = preset["tail_start"]
    if "tail_end" not in explicit_keys:
        args.tail_end = preset["tail_end"]
    if "tail_ramp_epochs" not in explicit_keys:
        args.tail_ramp_epochs = preset["tail_ramp_epochs"]
    if getattr(args, "mode", None) == "test":
        if "tail_ramp_epochs" not in explicit_keys:
            args.tail_ramp_epochs = min(args.tail_ramp_epochs, 2)
        if "feedback_blend_ramp_epochs" not in explicit_keys:
            args.feedback_blend_ramp_epochs = min(getattr(args, "feedback_blend_ramp_epochs", 0), 2)
    if "backward_chunk_steps" in preset and "backward_chunk_steps" not in explicit_keys:
        args.backward_chunk_steps = preset["backward_chunk_steps"]
    args.backward_chunk_steps = max(1, min(args.backward_chunk_steps, args.max_horizon))

    current_root = Path(args.output_root)
    if current_root.name != preset["name"]:
        args.output_root = current_root / preset["name"]
    else:
        args.output_root = current_root

    # A preset is a fresh run by default, but must not override an explicit
    # user supplied resume path.  ``_resume_explicit`` is populated by
    # ``parse_args_from`` before the preset is applied.
    if not getattr(args, "_resume_explicit", False):
        args.resume = "none"
    if args.log_dir is None:
        args.log_dir = args.output_root / "logs"
    else:
        log_dir_path = Path(args.log_dir)
        if log_dir_path.name != preset["name"]:
            args.log_dir = log_dir_path / preset["name"]
        else:
            args.log_dir = log_dir_path
    return args


def build_training_command(
    mechanism: str,
    data_dir: str,
    output_root: Path,
    split_dir: str | None = None,
    epochs: int = 30,
    parts_per_epoch: int = 42,
    val_parts_per_epoch: int | None = 9,
    batch_size: int = 1024,
    val_batch_size: int | None = 2048,
    lr: float = 2e-4,
    backward_chunk_steps: int = 5,
    max_horizon: int = 128,
    curriculum_horizons: list[int] | None = None,
    curriculum_start_epochs: list[int] | None = None,
    eval_horizons: list[int] | None = None,
    train_stride: int | None = None,
    val_stride: int | None = None,
    tail_start: float = 0.05,
    tail_end: float = 1.0,
    tail_ramp_epochs: int = 0,
    val_ratio: float = 0.10,
    scaler_parts: int = 10,
    scaler_sample_rate: float = 0.02,
    scaler_max_rows: int = 2_000_000,
    lambda_rate: float = 1.0,
    lambda_state: float = 1.0,
    lambda_element: float = 0.2,
    lambda_mass: float = 0.05,
    lambda_positive: float = 0.01,
    rate_loss_kind: str = "mse",
    rate_huber_beta: float = 1.0,
    output_parameterization: str = "direct_rate",
    rate_decode_mode: str = "direct_rate",
    rate_z_limit: float = 10.0,
    lambda_rollout: float = 0.0,  # 新增
    lambda_stability: float = 0.0,
    transition_weight_alpha: float = 0.0,
    transition_weight_clip: float = 8.0,
    feedback_noise_std: float = 0.0,
    macro_loss_weight: float = 1.0,
    micro_loss_weight: float = 1.0,
    key_species_loss_weight: float = 1.0,
    feedback_blend_start: float = 1.0,
    feedback_blend_end: float = 1.0,
    feedback_blend_ramp_epochs: int = 0,
    trust_region_fraction: float = 0.0,
    temperature_mode: str = "recorded",
    max_delta_temperature: float = 1000.0,
    resume: str = "auto",
    warm_start: str | None = None,  # 新增
    experiment_mode: str = "baseline",  # 新增
    state_update_mode: str = "legacy",  # Task8: legacy or task8_conservative_state
    state_loss_source: str = "raw",
    feedback_state_source: str = "raw",
    micro_rate_z_limit: float | None = None,
    checkpoint_every: int = 10,
    full_val_every: int = 5,
    max_val_batches: int | None = None,
    lambda_radical: float = 0.0,
    lambda_no: float = 0.0,
    use_induction_weights: bool = False,
    induction_boost: float = 6.0,
    projection_mode: str = "multiplicative_kl",
    desentinelize_scaler: bool = False,
    network_arch: str = "mlp",
) -> list[str]:
    """Build the exact command used for one V68 mechanism."""
    curriculum_horizons = curriculum_horizons or [1, 8, 32, 128]
    curriculum_start_epochs = curriculum_start_epochs or [0, 2, 5, 8]
    eval_horizons = eval_horizons or [1, 2, 4, 8, 16, 32, 64, 128]

    cmd = [
        str(PYTHON_EXECUTABLE),
        str(TRAINING_SCRIPT),
        "--data_dir", str(data_dir),
        "--mechanism", mechanism,
        "--epochs", str(epochs),
        "--parts_per_epoch", str(parts_per_epoch),
        "--batch_size", str(batch_size),
        "--backward_chunk_steps", str(backward_chunk_steps),
        "--lr", str(lr),
        "--max_horizon", str(max_horizon),
        "--curriculum_horizons", *map(str, curriculum_horizons),
        "--curriculum_start_epochs", *map(str, curriculum_start_epochs),
        "--eval_horizons", *map(str, eval_horizons),
        "--tail_start", str(tail_start),
        "--tail_end", str(tail_end),
        "--tail_ramp_epochs", str(tail_ramp_epochs),
        "--val_ratio", str(val_ratio),
        "--scaler_parts", str(scaler_parts),
        "--scaler_sample_rate", str(scaler_sample_rate),
        "--scaler_max_rows", str(scaler_max_rows),
        "--lambda_rate", str(lambda_rate),
        "--lambda_state", str(lambda_state),
        "--lambda_element", str(lambda_element),
        "--lambda_mass", str(lambda_mass),
        "--lambda_positive", str(lambda_positive),
        "--rate_loss_kind", rate_loss_kind,
        "--rate_huber_beta", str(rate_huber_beta),
        "--output_parameterization", output_parameterization,
        "--rate_decode_mode", rate_decode_mode,
        "--rate_z_limit", str(rate_z_limit),
        "--lambda_rollout", str(lambda_rollout),  # 新增
        "--lambda_stability", str(lambda_stability),
        "--transition_weight_alpha", str(transition_weight_alpha),
        "--transition_weight_clip", str(transition_weight_clip),
        "--feedback_noise_std", str(feedback_noise_std),
        "--macro_loss_weight", str(macro_loss_weight),
        "--micro_loss_weight", str(micro_loss_weight),
        "--key_species_loss_weight", str(key_species_loss_weight),
        "--feedback_blend_start", str(feedback_blend_start),
        "--feedback_blend_end", str(feedback_blend_end),
        "--feedback_blend_ramp_epochs", str(feedback_blend_ramp_epochs),
        "--trust_region_fraction", str(trust_region_fraction),
        "--temperature_mode", temperature_mode,
        "--max_delta_temperature", str(max_delta_temperature),
        "--experiment_mode", experiment_mode,  # 新增
        "--state_update_mode", state_update_mode,  # Task8
        "--state-loss-source", state_loss_source,
        "--feedback-state-source", feedback_state_source,
        "--resume", resume,
        "--checkpoint_every", str(checkpoint_every),
        "--full_val_every", str(full_val_every),
        "--output_root", str(output_root),
        "--lambda_radical", str(lambda_radical),
        "--lambda_no", str(lambda_no),
        "--induction_boost", str(induction_boost),
        "--projection_mode", str(projection_mode),
    ]

    if use_induction_weights:
        cmd.append("--use_induction_weights")
    if desentinelize_scaler:
        cmd.append("--desentinelize_scaler")
    if network_arch != "mlp":
        cmd.extend(["--network_arch", str(network_arch)])

    if micro_rate_z_limit is not None:
        cmd.extend(["--micro-rate-z-limit", str(micro_rate_z_limit)])

    if split_dir is not None:
        cmd.extend(["--split_dir", str(split_dir)])

    if train_stride is not None:
        cmd.extend(["--train_stride", str(train_stride)])
    if val_stride is not None:
        cmd.extend(["--val_stride", str(val_stride)])
    if val_parts_per_epoch is not None:
        cmd.extend(["--val_parts_per_epoch", str(val_parts_per_epoch)])
    if val_batch_size is not None:
        cmd.extend(["--val_batch_size", str(val_batch_size)])
    if max_val_batches is not None:
        cmd.extend(["--max_val_batches", str(max_val_batches)])
    if warm_start is not None:  # 新增
        cmd.extend(["--warm_start", str(warm_start)])

    return cmd


class TrainingLogger:
    def __init__(self, log_dir: Path | None = None):
        self.log_dir = Path(log_dir) if log_dir is not None else SCRIPT_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"v68_training_log_{timestamp}.txt"
        self.results: dict[str, dict] = {}
        self.start_time = time.time()
        self.plan = "custom"
        self.preset_name = None
        self.output_parameterization = "direct_rate"
        self.rate_decode_mode = "direct_rate"
        self.rate_z_limit = 10.0
        self.state_update_mode = "legacy"
        self.warm_start = None

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg, flush=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(log_msg + "\n")

    def save_summary(self) -> Path:
        total_time = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_param = getattr(self, "output_parameterization", "direct_rate")
        plan_sig = get_transform_signature(out_param)

        summary = {
            "version": VERSION,
            "plan": getattr(self, "plan", "custom"),
            "preset_name": getattr(self, "preset_name", None),
            "output_parameterization": out_param,
            "rate_decode_mode": getattr(self, "rate_decode_mode", "direct_rate"),
            "rate_z_limit": getattr(self, "rate_z_limit", 10.0),
            "state_update_mode": getattr(self, "state_update_mode", "legacy"),
            "warm_start": getattr(self, "warm_start", None),
            "transform_signature": plan_sig,
            "rollout_signature": ROLLOUT_SIGNATURE,
            "method": "Long-horizon species-autoregressive operator with external recorded T/P/dt",
            "data_source": "database_Transient_V21 (dynamic horizon windows, pure species AR rollout)",
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration_hours": total_time / 3600,
            "mechanisms": self.results,
            "log_file": str(self.log_file),
            "artifact_directories": [
                "scalers_v68",
                "weights_v68",
                "checkpoints_v68",
                "history_v68",
                "metrics_v68",
            ],
            "key_features": [
                "Dynamic window horizon up to 128 steps",
                "Horizon curriculum: 1 -> 8 -> 32 -> 128",
                "Continuous prediction-tail scheduled sampling",
                "High-activity/ignition transition loss weighting (V3 preset)",
                "Segmented closed-loop BPTT with feedback-state perturbation (V3 preset)",
                "Adjacent-state increment stability loss (V3 preset)",
                "Oracle one-step and pure species-AR validation",
                "Per-horizon metrics at 1/2/4/8/16/32/64/128",
                "External T/P/dt from recorded trajectory (not predicted)",
                "Atomic latest checkpoint and resume with full RNG state",
            ],
        }
        summary_file = self.log_dir / f"v68_training_summary_{timestamp}.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        self.log(f"TRAINING SUMMARY SAVED: {summary_file}")
        return summary_file


def check_gpu() -> bool:
    """Print basic GPU status without making nvidia-smi mandatory."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            print("[WARNING] nvidia-smi unavailable or no GPU found")
            return False
        for index, line in enumerate(result.stdout.strip().splitlines()):
            values = [value.strip() for value in line.split(",")]
            if len(values) == 4:
                print(
                    f"GPU {index}: "
                    f"Free {values[0]} MiB / Total {values[1]} MiB, "
                    f"Util {values[2]}%, Temp {values[3]}C"
                )
        return True
    except Exception as exc:
        print(f"[WARNING] GPU check failed: {exc}")
        return False


def _metadata_candidates(data_root: Path, mechanism: str) -> list[Path]:
    """Return likely metadata locations without assuming a database name."""
    trajectory_name = f"trajectories_{mechanism}"
    candidates = [
        data_root / trajectory_name / "metadata.json",
        data_root / "database_Transient_V21_QSSA" / trajectory_name / "metadata.json",
        data_root / "database_Transient_V21" / trajectory_name / "metadata.json",
    ]
    try:
        child_dirs = sorted(
            child for child in data_root.iterdir() if child.is_dir()
        )
    except OSError:
        child_dirs = []
    for child in child_dirs:
        candidate = child / trajectory_name / "metadata.json"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def validate_training_inputs(
    data_dir: str,
    mechanisms: list[str],
    warm_start: str | None,
) -> None:
    """Fail early with actionable paths before starting a long training run."""
    data_root = Path(data_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    for mechanism in mechanisms:
        metadata_candidates = _metadata_candidates(data_root, mechanism)
        metadata = None
        for candidate in metadata_candidates:
            if candidate.is_file():
                metadata = candidate
                break
        if metadata is None:
            raise FileNotFoundError(
                f"Metadata not found for mechanism {mechanism}. Checked: {metadata_candidates}"
            )

    if warm_start:
        checkpoint = Path(warm_start)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Warm-start checkpoint not found: {checkpoint}"
            )


def run_training_for_mechanism(
    mechanism: str,
    logger: TrainingLogger,
    **config,
) -> bool:
    logger.log(f"MECHANISM {len(logger.results) + 1}/{len(config.get('mechanisms', [mechanism]))}: {mechanism.upper()}")
    logger.log(f"STARTING TRAINING: {mechanism.upper()}")

    # 过滤掉不属于 build_training_command 的参数
    training_config = {
        k: v for k, v in config.items()
        if k not in {"mechanisms", "plan", "preset_name"}
    }
    command = build_training_command(mechanism=mechanism, **training_config)
    transform_sig = get_transform_signature(training_config.get("output_parameterization"))
    logger.log(f"Rollout mode: {ROLLOUT_SIGNATURE}")
    logger.log(f"Transform: {transform_sig}")
    logger.log(f"Command: {' '.join(command)}")

    started = time.time()
    process = None
    success = False

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            cwd=str(SCRIPT_DIR),
        )

        for line in process.stdout:
            logger.log(line.rstrip())

        exit_code = process.wait()
        duration_hours = (time.time() - started) / 3600

        if exit_code == 0:
            logger.results[mechanism] = {
                "status": "success",
                "duration_hours": duration_hours,
            }
            logger.log(
                f"SUCCESS: {mechanism.upper()} completed "
                f"(exit={exit_code}, {duration_hours:.2f}h)"
            )
            success = True
        else:
            logger.results[mechanism] = {
                "status": "failed",
                "exit_code": exit_code,
                "duration_hours": duration_hours,
            }
            logger.log(
                f"FAILED: {mechanism.upper()} "
                f"(exit={exit_code}, {duration_hours:.2f}h)"
            )
        return success
    except KeyboardInterrupt:
        logger.log("Training interrupted by user")
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    except Exception as exc:
        duration_hours = (time.time() - started) / 3600
        logger.results[mechanism] = {
            "status": "error",
            "error": str(exc),
            "duration_hours": duration_hours,
        }
        logger.log(f"ERROR: {mechanism.upper()}: {exc}")
        return False


def main(
    data_dir: str,
    mechanisms: list[str] | None = None,
    output_root: Path | None = None,
    log_dir: Path | None = None,
    **config,
) -> bool:
    mechanisms = mechanisms or DEFAULT_MECHANISMS
    output_root = Path(output_root) if output_root is not None else SCRIPT_DIR
    logger = TrainingLogger(log_dir)
    logger.plan = config.get("plan", "custom")
    logger.preset_name = config.get(
        "preset_name",
        PLAN_PRESETS.get(logger.plan, {}).get("name", "custom"),
    )
    logger.warm_start = config.get("warm_start")
    logger.output_parameterization = config.get("output_parameterization", "direct_rate")
    logger.rate_decode_mode = config.get("rate_decode_mode", "direct_rate")
    logger.rate_z_limit = config.get("rate_z_limit", 10.0)
    logger.state_update_mode = config.get("state_update_mode", "legacy")

    active_transform = get_transform_signature(config.get("output_parameterization"))
    logger.log(f"{VERSION} AUTOMATED TRAINING SESSION STARTED")
    logger.log(f"Method: {active_transform} + {ROLLOUT_SIGNATURE}")
    logger.log("Data: database_Transient_V21 long-horizon streaming")
    logger.log(f"Data directory: {data_dir}")
    logger.log(f"Training script: {TRAINING_SCRIPT}")
    logger.log(f"Mechanisms: {mechanisms}")
    logger.log(f"Epochs: {config.get('epochs', 30)}")
    logger.log(f"Parts per epoch: {config.get('parts_per_epoch', 42)}")
    logger.log(
        f"Validation parts per epoch: {config.get('val_parts_per_epoch', 'all')}"
    )
    logger.log(f"Batch size: {config.get('batch_size', 1024)}")
    logger.log(
        f"Validation batch size: {config.get('val_batch_size', config.get('batch_size', 2048))}"
    )
    logger.log(f"Max horizon: {config.get('max_horizon', 128)}")
    logger.log(f"Backward chunk steps: {config.get('backward_chunk_steps', 5)}")
    logger.log(
        f"Transition weighting: alpha={config.get('transition_weight_alpha', 0.0)}, "
        f"clip={config.get('transition_weight_clip', 8.0)}"
    )
    logger.log(f"Feedback state noise std: {config.get('feedback_noise_std', 0.0)}")
    logger.log(
        "Feedback blend: "
        f"{config.get('feedback_blend_start', 1.0)} -> "
        f"{config.get('feedback_blend_end', 1.0)} over "
        f"{config.get('feedback_blend_ramp_epochs', 0)} epochs"
    )
    logger.log(
        f"Trust-region fraction: {config.get('trust_region_fraction', 0.0)}"
    )
    logger.log(
        f"Temperature mode: {config.get('temperature_mode', 'recorded')}"
    )
    logger.log(
        "Species loss weights: "
        f"macro={config.get('macro_loss_weight', 1.0)}, "
        f"micro={config.get('micro_loss_weight', 1.0)}, "
        f"key={config.get('key_species_loss_weight', 1.0)}"
    )
    logger.log(f"Stability loss weight: {config.get('lambda_stability', 0.0)}")
    logger.log(f"Plan: {config.get('plan', 'custom')}")
    logger.log(f"Preset name: {logger.preset_name}")
    logger.log(f"Output root: {output_root}")
    logger.log(f"Warm start: {logger.warm_start or 'none'}")
    logger.log(f"Output parameterization: {logger.output_parameterization}")
    logger.log(f"Rate decode mode: {logger.rate_decode_mode}")
    logger.log(f"Rate z limit: {logger.rate_z_limit}")
    logger.log(f"State update mode: {logger.state_update_mode}")

    validate_training_inputs(
        data_dir,
        mechanisms,
        config.get("warm_start"),
    )

    gpu_available = check_gpu()
    logger.log(f"GPU available: {gpu_available}")

    config["mechanisms"] = mechanisms
    config["data_dir"] = data_dir
    config["output_root"] = output_root

    all_success = True
    for mechanism in mechanisms:
        success = run_training_for_mechanism(mechanism, logger, **config)
        all_success = all_success and success
        if not success:
            logger.log(f"Mechanism {mechanism} failed; stopping batch")
            break

    logger.save_summary()
    return all_success


def parse_args_from(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{VERSION} batch training launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Complete-trajectory split directory; auto-detected by train_sc_v68 when omitted.",
    )
    parser.add_argument(
        "--plan", "--方案",
        type=str,
        choices=sorted(list(PLAN_PRESETS.keys()) + ["all"]),
        default=None,
        help="Built-in continuation experiment: v71_qssa, v69_qssa, v68_qssa, v69_no_noise, v69_no_stability, v69_no_microweight, v68, or all.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("test", "full"),
        default=None,
        help="Training mode: 'test' (5 epochs, fast verification) or 'full' (30 epochs, production run).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Shortcut to enable test mode (5 epochs, 14 parts/epoch, 3 val parts/epoch).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Shortcut to enable full mode (30 epochs, 42 parts/epoch, 9 val parts/epoch).",
    )
    parser.add_argument("--mechanisms", type=str, nargs="+", default=DEFAULT_MECHANISMS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--parts_per_epoch", type=int, default=42)
    parser.add_argument(
        "--val_parts_per_epoch",
        type=int,
        default=9,
        help="每个验证 epoch 随机抽取的 parquet 文件数；默认 9。",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=2048,
        help="验证 batch size；默认 2048。",
    )
    parser.add_argument("--backward_chunk_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_horizon", type=int, default=128)
    parser.add_argument("--curriculum_horizons", type=str, default="1,8,32,128")
    parser.add_argument("--curriculum_start_epochs", type=str, default="0,2,5,8")
    parser.add_argument("--eval_horizons", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--train_stride", type=int, default=None)
    parser.add_argument("--val_stride", type=int, default=None)
    parser.add_argument("--tail_start", type=float, default=0.05)
    parser.add_argument("--tail_end", type=float, default=1.0)
    parser.add_argument("--tail_ramp_epochs", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--scaler_parts", type=int, default=10)
    parser.add_argument("--scaler_sample_rate", type=float, default=0.02)
    parser.add_argument("--scaler_max_rows", type=int, default=2_000_000)
    parser.add_argument("--lambda_rate", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=1.0)
    parser.add_argument("--lambda_element", type=float, default=0.2)
    parser.add_argument("--lambda_mass", type=float, default=0.05)
    parser.add_argument("--lambda_positive", type=float, default=0.01)
    parser.add_argument("--rate_loss_kind", choices=("mse", "huber"), default="mse")
    parser.add_argument("--rate_huber_beta", type=float, default=1.0)
    parser.add_argument("--output_parameterization", choices=("direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch"), default="direct_rate")
    parser.add_argument("--rate_decode_mode", choices=("direct_rate", "bounded_direct_rate", "qssa", "qssa_asymptotic", "dual_branch"), default="direct_rate")
    parser.add_argument("--rate_z_limit", type=float, default=10.0)
    parser.add_argument("--lambda_rollout", type=float, default=0.0)  # 新增
    parser.add_argument("--lambda_stability", type=float, default=0.0)
    parser.add_argument("--transition_weight_alpha", type=float, default=0.0)
    parser.add_argument("--transition_weight_clip", type=float, default=8.0)
    parser.add_argument("--feedback_noise_std", type=float, default=0.0)
    parser.add_argument("--macro_loss_weight", type=float, default=1.0)
    parser.add_argument("--micro_loss_weight", type=float, default=1.0)
    parser.add_argument("--key_species_loss_weight", type=float, default=1.0)
    parser.add_argument("--feedback_blend_start", type=float, default=1.0)
    parser.add_argument("--feedback_blend_end", type=float, default=1.0)
    parser.add_argument("--feedback_blend_ramp_epochs", type=int, default=0)
    parser.add_argument("--trust_region_fraction", type=float, default=0.0)
    parser.add_argument(
        "--temperature_mode",
        type=str,
        choices=("recorded", "autonomous"),
        default="recorded",
    )
    parser.add_argument("--max_delta_temperature", type=float, default=1000.0)
    parser.add_argument("--experiment_mode", type=str, default="baseline",
                        choices=["baseline", "diagnostic", "constrained_delta", "conserve_project", "qssa"],
                        help="Experiment mode for state update strategy")  # 新增
    parser.add_argument("--state_update_mode", type=str, default="legacy",
                        choices=["legacy", "task8_conservative_state", "qssa", "qssa_asymptotic", "dual_branch"],
                        help="Task8 state update strategy: 'legacy', 'task8_conservative_state', 'qssa', 'qssa_asymptotic', or 'dual_branch'")
    parser.add_argument("--state-loss-source", "--state_loss_source", dest="state_loss_source", type=str, default="raw",
                        choices=["constrained", "raw"],
                        help="State loss target: 'constrained' or 'raw'")
    parser.add_argument("--feedback-state-source", "--feedback_state_source", dest="feedback_state_source", type=str, default="raw",
                        choices=["constrained", "raw"],
                        help="Feedback state target: 'constrained' or 'raw'")
    parser.add_argument("--micro-rate-z-limit", "--micro_rate_z_limit", dest="micro_rate_z_limit", type=float, default=None,
                        help="Positive z limit for micro rate protection")
    parser.add_argument("--lambda_radical", type=float, default=0.0,
                        help="Weight for physical radical log loss on fast chain-branching species")
    parser.add_argument("--lambda_no", type=float, default=0.0,
                        help="Weight for pollutant NO log loss")
    parser.add_argument("--use_induction_weights", action="store_true", default=False,
                        help="Boost loss weights for pre-ignition induction period samples")
    parser.add_argument("--induction_boost", type=float, default=6.0,
                        help="Loss boost factor for induction period samples")
    parser.add_argument("--projection_mode", choices=("multiplicative_kl", "weighted_linear"), default="multiplicative_kl",
                        help="Simplex and element projection algorithm")
    parser.add_argument("--desentinelize_scaler", action="store_true", default=False,
                        help="Use MaskedOnlineMoments to exclude frozen sentinel values from scaler")
    parser.add_argument("--network_arch", choices=("mlp", "resmlp", "dual_branch"), default="mlp",
                        help="Neural network architecture backbone")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip interactive confirmation prompts")
    parser.add_argument("--warm_start", type=str, default=DEFAULT_WARM_START,
                        help="Optional checkpoint path for warm start; omitted means train from scratch.")  # 新增
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--full_val_every", type=int, default=5)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log_dir", type=Path, default=None)
    parser.add_argument("--no-confirm", action="store_true")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._explicit_keys = {
        token.lstrip("-").split("=")[0].replace("-", "_")
        for token in raw_argv
        if token.startswith("--")
    }

    if args.test:
        args.mode = "test"
    elif args.full:
        args.mode = "full"

    # argparse does not expose whether a value came from the command line or
    # from the default.  Keep that distinction so plan presets can preserve
    # an explicit ``--resume`` path.
    args._curriculum_explicit = any(
        token == "--curriculum_horizons" or token.startswith("--curriculum_horizons=")
        or token == "--curriculum_start_epochs" or token.startswith("--curriculum_start_epochs=")
        for token in raw_argv
    )
    args._resume_explicit = any(
        token == "--resume" or token.startswith("--resume=")
        for token in raw_argv
    )

    if args.mode == "test":
        if "epochs" not in args._explicit_keys:
            args.epochs = 5
        if "parts_per_epoch" not in args._explicit_keys:
            args.parts_per_epoch = 14
        if "val_parts_per_epoch" not in args._explicit_keys:
            args.val_parts_per_epoch = 3
        if not args._curriculum_explicit:
            args.curriculum_horizons = "1,8,32,128"
            args.curriculum_start_epochs = "0,1,2,3"
        if "checkpoint_every" not in args._explicit_keys:
            args.checkpoint_every = 1
        if "full_val_every" not in args._explicit_keys:
            args.full_val_every = 1

    args.curriculum_horizons = [int(x) for x in args.curriculum_horizons.split(",")]
    args.curriculum_start_epochs = [int(x) for x in args.curriculum_start_epochs.split(",")]
    args.eval_horizons = [int(x) for x in args.eval_horizons.split(",")]

    args = apply_plan_preset(args)
    if args.log_dir is None and args.plan is not None and args.plan != "all":
        args.log_dir = args.output_root / "logs"

    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from()


if __name__ == "__main__":
    args = parse_args()

    # If mode was not explicitly supplied via CLI, ask interactively unless --no-confirm
    if args.mode is None and not args.no_confirm:
        args.mode = select_mode_interactively()
        if args.mode == "test":
            if "epochs" not in args._explicit_keys:
                args.epochs = 5
            if "parts_per_epoch" not in args._explicit_keys:
                args.parts_per_epoch = 14
            if "val_parts_per_epoch" not in args._explicit_keys:
                args.val_parts_per_epoch = 3
            if not args._curriculum_explicit:
                args.curriculum_horizons = [1, 8, 32, 128]
                args.curriculum_start_epochs = [0, 1, 2, 3]
            if "checkpoint_every" not in args._explicit_keys:
                args.checkpoint_every = 1
            if "full_val_every" not in args._explicit_keys:
                args.full_val_every = 1
            if "tail_ramp_epochs" not in args._explicit_keys:
                args.tail_ramp_epochs = min(args.tail_ramp_epochs, 2)
            if "feedback_blend_ramp_epochs" not in args._explicit_keys:
                args.feedback_blend_ramp_epochs = min(getattr(args, "feedback_blend_ramp_epochs", 0), 2)
    elif args.mode is None:
        args.mode = "full"

    if args.plan is None:
        if args.no_confirm:
            args.plan = "v71_qssa"
            print("未指定方案且使用 --no-confirm，默认选择 v71_qssa。")
        else:
            args.plan = select_plan_interactively()
        args = apply_plan_preset(args)
        if args.log_dir is None and args.plan != "all":
            args.log_dir = args.output_root / "logs"

    if args.plan == "all":
        base_output_root = Path(args.output_root)
        print("\n" + "=" * 72)
        print(f"{VERSION} 批量对比训练启动 (共 {len(ACTIVE_COMPARISON_PLANS)} 个方案)")
        print("=" * 72)
        print(f"模式 (mode): {args.mode} (epochs={args.epochs}, parts={args.parts_per_epoch}, val_parts={args.val_parts_per_epoch})")
        print(f"对比方案列表: {', '.join(ACTIVE_COMPARISON_PLANS)}")
        print(f"基础输出根目录: {base_output_root}")
        print(f"机制: {', '.join(args.mechanisms)}")
        if not args.no_confirm:
            answer = input("确认启动所有方案的批量顺序训练？输入 y/yes 继续，其他输入取消: ")
            if answer.lower() not in ("y", "yes"):
                print("Cancelled by user")
                sys.exit(0)

        all_batch_success = True
        overall_results = {}
        for plan_name in ACTIVE_COMPARISON_PLANS:
            print("\n" + "#" * 72)
            print(f"=== 开始运行对比方案: {plan_name} ===")
            print("#" * 72)
            plan_args = copy.copy(args)
            plan_args.plan = plan_name
            plan_args.output_root = base_output_root
            plan_args.log_dir = None
            plan_args = apply_plan_preset(plan_args)

            success = main(
                data_dir=plan_args.data_dir,
                split_dir=plan_args.split_dir,
                mechanisms=plan_args.mechanisms,
                output_root=plan_args.output_root,
                log_dir=plan_args.log_dir,
                epochs=plan_args.epochs,
                parts_per_epoch=plan_args.parts_per_epoch,
                val_parts_per_epoch=plan_args.val_parts_per_epoch,
                batch_size=plan_args.batch_size,
                val_batch_size=plan_args.val_batch_size,
                lr=plan_args.lr,
                backward_chunk_steps=plan_args.backward_chunk_steps,
                max_horizon=plan_args.max_horizon,
                curriculum_horizons=plan_args.curriculum_horizons,
                curriculum_start_epochs=plan_args.curriculum_start_epochs,
                eval_horizons=plan_args.eval_horizons,
                train_stride=plan_args.train_stride,
                val_stride=plan_args.val_stride,
                tail_start=plan_args.tail_start,
                tail_end=plan_args.tail_end,
                tail_ramp_epochs=plan_args.tail_ramp_epochs,
                val_ratio=plan_args.val_ratio,
                scaler_parts=plan_args.scaler_parts,
                scaler_sample_rate=plan_args.scaler_sample_rate,
                scaler_max_rows=plan_args.scaler_max_rows,
                lambda_rate=plan_args.lambda_rate,
                lambda_state=plan_args.lambda_state,
                lambda_element=plan_args.lambda_element,
                lambda_mass=plan_args.lambda_mass,
                lambda_positive=plan_args.lambda_positive,
                rate_loss_kind=plan_args.rate_loss_kind,
                rate_huber_beta=plan_args.rate_huber_beta,
                output_parameterization=plan_args.output_parameterization,
                rate_decode_mode=plan_args.rate_decode_mode,
                rate_z_limit=plan_args.rate_z_limit,
                lambda_rollout=plan_args.lambda_rollout,
                lambda_stability=plan_args.lambda_stability,
                transition_weight_alpha=plan_args.transition_weight_alpha,
                transition_weight_clip=plan_args.transition_weight_clip,
                feedback_noise_std=plan_args.feedback_noise_std,
                macro_loss_weight=plan_args.macro_loss_weight,
                micro_loss_weight=plan_args.micro_loss_weight,
                key_species_loss_weight=plan_args.key_species_loss_weight,
                feedback_blend_start=plan_args.feedback_blend_start,
                feedback_blend_end=plan_args.feedback_blend_end,
                feedback_blend_ramp_epochs=plan_args.feedback_blend_ramp_epochs,
                trust_region_fraction=plan_args.trust_region_fraction,
                temperature_mode=plan_args.temperature_mode,
                max_delta_temperature=plan_args.max_delta_temperature,
                experiment_mode=plan_args.experiment_mode,
                state_update_mode=plan_args.state_update_mode,
                state_loss_source=getattr(plan_args, "state_loss_source", "raw"),
                feedback_state_source=getattr(plan_args, "feedback_state_source", "raw"),
                micro_rate_z_limit=getattr(plan_args, "micro_rate_z_limit", None),
                lambda_radical=getattr(plan_args, "lambda_radical", 0.0),
                lambda_no=getattr(plan_args, "lambda_no", 0.0),
                use_induction_weights=getattr(plan_args, "use_induction_weights", False),
                induction_boost=getattr(plan_args, "induction_boost", 6.0),
                projection_mode=getattr(plan_args, "projection_mode", "multiplicative_kl"),
                desentinelize_scaler=getattr(plan_args, "desentinelize_scaler", False),
                network_arch=getattr(plan_args, "network_arch", "mlp"),
                warm_start=plan_args.warm_start,
                resume=plan_args.resume,
                checkpoint_every=plan_args.checkpoint_every,
                full_val_every=plan_args.full_val_every,
                max_val_batches=plan_args.max_val_batches,
                plan=plan_args.plan,
                preset_name=getattr(plan_args, "preset_name", None) or PLAN_PRESETS[plan_name]["name"],
            )
            overall_results[plan_name] = success
            all_batch_success = all_batch_success and success
            if not success:
                print(f"[ERROR] 方案 {plan_name} 训练失败，终止后续批处理。")
                break

        print("\n" + "=" * 72)
        print("所有方案训练执行完成汇总:")
        for p, s in overall_results.items():
            print(f"  - {p}: {'成功' if s else '失败'}")
        print("=" * 72)
        sys.exit(0 if all_batch_success else 1)

    if not args.no_confirm:
        preset_name = getattr(args, "preset_name", None) or (
            PLAN_PRESETS[args.plan]["name"] if args.plan in PLAN_PRESETS else "custom"
        )
        print("\n" + "=" * 72)
        print(f"{VERSION} 训练启动配置")
        print("=" * 72)
        print(f"模式 (mode): {args.mode}")
        print(f"方案 (plan): {args.plan}")
        print(f"预设名称 (preset name): {preset_name}")
        print(f"输出根目录 (output_root): {args.output_root}")
        print(f"warm_start: {args.warm_start or 'none'}")
        print(f"output_parameterization: {args.output_parameterization}")
        print(f"rate_decode_mode: {args.rate_decode_mode}")
        print(f"rate_z_limit: {args.rate_z_limit}")
        print(f"state_update_mode: {args.state_update_mode}")
        print(f"机制: {', '.join(args.mechanisms)}")
        print(f"epochs: {args.epochs}; batch_size: {args.batch_size}; lr: {args.lr}")
        print(f"split_dir: {args.split_dir or '[auto-detect]'}")
        print(f"resume: {args.resume}")
        print(f"curriculum: {args.curriculum_horizons} @ {args.curriculum_start_epochs}")
        print(f"backward_chunk_steps: {args.backward_chunk_steps}")
        if not getattr(args, "yes", False):
            answer = input("确认启动训练？输入 y/yes 继续，其他输入取消: ")
            if answer.lower() not in ("y", "yes"):
                print("Cancelled by user")
                sys.exit(0)

    success = main(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        mechanisms=args.mechanisms,
        output_root=args.output_root,
        log_dir=args.log_dir,
        epochs=args.epochs,
        parts_per_epoch=args.parts_per_epoch,
        val_parts_per_epoch=args.val_parts_per_epoch,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        backward_chunk_steps=args.backward_chunk_steps,
        max_horizon=args.max_horizon,
        curriculum_horizons=args.curriculum_horizons,
        curriculum_start_epochs=args.curriculum_start_epochs,
        eval_horizons=args.eval_horizons,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        tail_start=args.tail_start,
        tail_end=args.tail_end,
        tail_ramp_epochs=args.tail_ramp_epochs,
        val_ratio=args.val_ratio,
        scaler_parts=args.scaler_parts,
        scaler_sample_rate=args.scaler_sample_rate,
        scaler_max_rows=args.scaler_max_rows,
        lambda_rate=args.lambda_rate,
        lambda_state=args.lambda_state,
        lambda_element=args.lambda_element,
        lambda_mass=args.lambda_mass,
        lambda_positive=args.lambda_positive,
        rate_loss_kind=args.rate_loss_kind,
        rate_huber_beta=args.rate_huber_beta,
        output_parameterization=args.output_parameterization,
        rate_decode_mode=args.rate_decode_mode,
        rate_z_limit=args.rate_z_limit,
        lambda_rollout=args.lambda_rollout,  # 新增
        lambda_stability=args.lambda_stability,
        transition_weight_alpha=args.transition_weight_alpha,
        transition_weight_clip=args.transition_weight_clip,
        feedback_noise_std=args.feedback_noise_std,
        macro_loss_weight=args.macro_loss_weight,
        micro_loss_weight=args.micro_loss_weight,
        key_species_loss_weight=args.key_species_loss_weight,
        feedback_blend_start=args.feedback_blend_start,
        feedback_blend_end=args.feedback_blend_end,
        feedback_blend_ramp_epochs=args.feedback_blend_ramp_epochs,
        trust_region_fraction=args.trust_region_fraction,
        temperature_mode=args.temperature_mode,
        max_delta_temperature=args.max_delta_temperature,
        experiment_mode=args.experiment_mode,  # 新增
        state_update_mode=args.state_update_mode,  # Task8
        state_loss_source=getattr(args, "state_loss_source", "raw"),
        feedback_state_source=getattr(args, "feedback_state_source", "raw"),
        micro_rate_z_limit=getattr(args, "micro_rate_z_limit", None),
        lambda_radical=getattr(args, "lambda_radical", 0.0),
        lambda_no=getattr(args, "lambda_no", 0.0),
        use_induction_weights=getattr(args, "use_induction_weights", False),
        induction_boost=getattr(args, "induction_boost", 6.0),
        projection_mode=getattr(args, "projection_mode", "multiplicative_kl"),
        desentinelize_scaler=getattr(args, "desentinelize_scaler", False),
        network_arch=getattr(args, "network_arch", "mlp"),
        warm_start=args.warm_start,  # 新增
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        full_val_every=args.full_val_every,
        max_val_batches=args.max_val_batches,
        plan=args.plan or "custom",
        preset_name=getattr(args, "preset_name", None) or (
            PLAN_PRESETS[args.plan]["name"] if args.plan in PLAN_PRESETS else "custom"
        ),
    )

    sys.exit(0 if success else 1)



