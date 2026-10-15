"""
V68 long-horizon species-autoregressive operator training.

Macro targets use a complete, invertible representation:
    rate_scale_i = P90(abs(dYdt_i)) on sampled training data
    z_i = asinh(dYdt_i / rate_scale_i)
    target_i = standardize(z_i)

Rollout uses external recorded T/P/dt sequence from the trajectory; only species
states are predicted autoregressively. Training uses continuous prediction-tail
scheduled sampling over dynamic horizons up to 128 steps. Validation reports
both oracle one-step and pure species-AR metrics at multiple horizons.
"""

from __future__ import annotations

import sys
import os
import json
import math
import random
import re
import gc
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
try:
    import torch._dynamo
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import pyarrow.parquet as pq
import joblib
import training_diagnostics as td
import kinetics_parameterization as kp
import species_validation_protocol as svp
import experiment_manifest as em
import hashlib


# ============================================================
# Windows UTF-8
# ============================================================

if sys.platform == "win32":
    # Reconfigure the existing streams in place.  Wrapping ``.buffer`` with
    # a new TextIOWrapper can close pytest/IDE capture streams at teardown.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Some embedded/captured streams do not expose reconfigure().
            pass


# ============================================================
# Device / Seed
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


def restore_cuda_rng_state(states) -> bool:
    """Restore CUDA RNG only when the checkpoint matches this host's GPUs."""
    if states is None or not torch.cuda.is_available():
        return False
    states = list(states)
    if len(states) != torch.cuda.device_count():
        return False
    torch.cuda.set_rng_state_all(states)
    return True


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # 为了可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(SEED)

print(f"Using device: {DEVICE}")


# ============================================================
# Chemistry constants
# ============================================================

AW = {
    "N": 14.0067,
    "H": 1.00784,
    "O": 15.999,
    "Ar": 39.948,
    "He": 4.002602,
}

# V21 中 dt 是 1e-8 ~ 1e-4 s
DEFAULT_DT_MIN = 1e-8
DEFAULT_DT_MAX = 1e-4
DT_NUMERIC_FLOOR = 1e-12

# 对微量物种不能再用 1e-10，
# 因为你的数据图中已经出现远低于 1e-10 的物种。
LOG_EPS = 1e-40

# 数值安全
MICRO_LOG_MIN = float(np.log10(LOG_EPS))
MICRO_LOG_MAX = 0.0

# V68 version and transform contract.
VERSION = "v68"
LEGACY_TRANSFORM_SIGNATURE = "macro_asinh_p90_v2"
QSSA_TRANSFORM_SIGNATURE = "qssa_log10_P_D_v2"
QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE = "qssa_asymptotic_v2"
DUAL_BRANCH_TRANSFORM_SIGNATURE = "dual_branch_asinh_macro_qssa_micro_v1"
DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE = "dual_branch_pd_v1"
TRANSFORM_SIGNATURE = LEGACY_TRANSFORM_SIGNATURE


def transform_signature_for(output_parameterization):
    """One mode-to-signature contract for files, scalers and checkpoints."""
    signatures = {
        "direct_rate": LEGACY_TRANSFORM_SIGNATURE,
        "conservative_simplex": LEGACY_TRANSFORM_SIGNATURE,
        "qssa": QSSA_TRANSFORM_SIGNATURE,
        "qssa_asymptotic": QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE,
        "dual_branch": DUAL_BRANCH_TRANSFORM_SIGNATURE,
        "dual_branch_pd_v1": DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE,
    }
    return signatures[output_parameterization]
ROLLOUT_SIGNATURE = "external_recorded_t_p_dt_species_ar_experiments_v3"
RATE_SCALE_FLOOR = 1e-12
RATE_DECODE_MODES = ("direct_rate", "bounded_direct_rate", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")
DEFAULT_RATE_Z_LIMIT = 10.0
# Keep direct CLI use portable.  A deployment can provide a data root through
# V68_DATA_ROOT; otherwise resolve relative to this source tree.
DEFAULT_DATA_DIR = os.environ.get("V68_DATA_ROOT", str(Path(__file__).resolve().parent))
# ``torch.quantile`` has an implementation-dependent input-size limit.  Keep
# the audit's percentile calculation bounded while retaining exact mean/max
# statistics over all transitions.
TRANSITION_AUDIT_MAX_QUANTILE_SAMPLES = 65536

STATE_LOSS_SOURCES = ("constrained", "raw")
DEFAULT_STATE_LOSS_SOURCE = "raw"
# Raw rollouts can be many orders of magnitude outside the standardized
# training state.  This scale keeps the state-loss coordinate locally linear
# in-distribution while bounding the OOD residual and its backward gradient.
RAW_STATE_LOSS_SCALE = 32.0
# Keep raw closed-loop feedback representable to the MLP.  The returned raw
# state remains untouched for diagnostics; only the next model input is
# saturated after standardization when a trajectory leaves the data domain.
RAW_FEEDBACK_INPUT_LIMIT = 32.0

FEEDBACK_STATE_SOURCES = ("constrained", "raw")
DEFAULT_FEEDBACK_STATE_SOURCE = "raw"


def constrained_micro_feedback(
    raw_log_micro: torch.Tensor,
    projected_micro: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the projected micro state forward and log-state coordinates backward.

    The physical projection is intentionally detached.  A linear-space STE
    followed by ``log10`` would expose the backward pass to ``1 / Y`` and can
    overflow for dormant radicals near ``LOG_EPS``.  The equivalent STE in log
    coordinates has identity gradient for ``logY``; the derived linear state
    then has a bounded ``ln(10) * Y`` derivative.
    """
    projected_log = torch.log10(
        torch.clamp(projected_micro.detach(), min=LOG_EPS, max=1.0)
    )
    constrained_log = projected_log + (
        raw_log_micro - raw_log_micro.detach()
    )
    constrained_micro = torch.pow(10.0, constrained_log)
    return constrained_log, constrained_micro


def _log10_add_exp(log10_a: torch.Tensor, log10_b: torch.Tensor) -> torch.Tensor:
    """Return log10(10**log10_a + 10**log10_b) without linear underflow."""
    ln10 = math.log(10.0)
    return torch.logaddexp(log10_a * ln10, log10_b * ln10) / ln10


def dual_branch_micro_log_update(
    log_y_old: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Integrate dual-branch QSSA micro-species directly in log10 space.

    Ordinary targets store ``(log10(Y_qssa), log10(D))``.  The ``z2 < -40``
    sentinel stores production-only ``log10(P)`` in ``z1``; ``z1 < -40``
    denotes zero production.  Keeping the update logarithmic avoids taking a
    derivative of ``log10(Y)`` at subnormal radical mass fractions.
    """
    ln10 = math.log(10.0)
    dt_safe = torch.clamp(dt, min=DT_NUMERIC_FLOOR)
    log_dt = torch.log10(dt_safe)

    is_prod_only = z2 < -40.0
    is_zero_creation = z1 < -40.0
    log_y_qssa = torch.clamp(z1, min=-40.0, max=30.0)
    log_d = torch.clamp(z2, min=-40.0, max=30.0)
    log_d_dt = log_d + log_dt

    # Evaluate exp(-D*dt) without underflowing D*dt for very small rates.
    log_d_dt_bounded = torch.clamp(log_d_dt, min=-45.0, max=math.log10(80.0))
    d_dt_for_decay = torch.pow(10.0, log_d_dt_bounded)
    log_decay_term = log_y_old - d_dt_for_decay / ln10

    # log10(1 - exp(-x)); for x < 1e-4 use its stable small-x expansion.
    is_small_d_dt = log_d_dt < -4.0
    x_small = torch.pow(10.0, torch.clamp(log_d_dt, min=-45.0, max=-4.0))
    log_one_minus_exp_small = log_d_dt - x_small / (2.0 * ln10)
    x_regular = torch.pow(
        10.0,
        torch.clamp(log_d_dt, min=-4.0, max=math.log10(80.0)),
    )
    log_one_minus_exp_regular = torch.log(-torch.expm1(-x_regular)) / ln10
    log_one_minus_exp = torch.where(
        is_small_d_dt,
        log_one_minus_exp_small,
        log_one_minus_exp_regular,
    )

    log_generation_term = log_y_qssa + log_one_minus_exp
    log_generation_term = torch.where(
        is_zero_creation,
        torch.full_like(log_generation_term, -1.0e30),
        log_generation_term,
    )
    ordinary_update = _log10_add_exp(log_decay_term, log_generation_term)

    log_p = torch.clamp(z1, min=-40.0, max=30.0)
    production_only_update = _log10_add_exp(log_y_old, log_p + log_dt)
    production_only_update = torch.where(
        is_zero_creation,
        log_y_old,
        production_only_update,
    )

    log_y_next = torch.where(is_prod_only, production_only_update, ordinary_update)
    return torch.clamp(log_y_next, min=MICRO_LOG_MIN, max=1.0)


def select_training_state(
    pred_state: dict[str, Any],
    source: str = DEFAULT_STATE_LOSS_SOURCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select (Y_macro, logY_micro) for state loss calculation or rollout state tracking.

    In raw mode:
    - macro uses raw candidate Y_macro_next_raw
    - micro uses raw candidate logY_micro_next_raw without clamping to MICRO_LOG_MAX
    """
    if source == "raw":
        macro = pred_state.get("Y_macro_next_raw", pred_state["Y_macro_next"])
        log_micro = pred_state.get("logY_micro_next_raw", pred_state["logY_micro_next"])
    elif source == "constrained":
        macro = pred_state.get("Y_macro_next_constrained", pred_state["Y_macro_next"])
        log_micro = pred_state.get("logY_micro_next_constrained", pred_state["logY_micro_next"])
    else:
        raise ValueError(f"Unknown state_loss_source: {source}, must be one of {STATE_LOSS_SOURCES}")
    return macro, log_micro


def select_feedback_state(
    pred_state: dict[str, Any],
    source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select (Y_macro, logY_micro) for feeding back into next autoregressive input step."""
    if source == "raw":
        macro = pred_state.get("Y_macro_next_raw", pred_state["Y_macro_next"])
        log_micro = pred_state.get("logY_micro_next_raw", pred_state["logY_micro_next"])
    elif source == "constrained":
        macro = pred_state.get("Y_macro_next_constrained", pred_state["Y_macro_next"])
        log_micro = pred_state.get("logY_micro_next_constrained", pred_state["logY_micro_next"])
    else:
        raise ValueError(f"Unknown feedback_state_source: {source}, must be one of {FEEDBACK_STATE_SOURCES}")
    return macro, log_micro


def state_loss_features(
    state_features_scaled: torch.Tensor,
    source: str,
) -> torch.Tensor:
    """Return finite state-loss features without changing raw feedback states."""
    if source not in STATE_LOSS_SOURCES:
        raise ValueError(
            f"Unknown state_loss_source: {source}, must be one of {STATE_LOSS_SOURCES}"
        )
    if source == "raw":
        scale = float(RAW_STATE_LOSS_SCALE)
        return scale * torch.tanh(state_features_scaled / scale)
    return state_features_scaled


def conservative_nonnegative_decode(logits: torch.Tensor, total: float = 1.0) -> torch.Tensor:
    """Decode logits to a nonnegative simplex."""
    if logits.ndim < 1 or not math.isfinite(float(total)) or total <= 0:
        raise ValueError("invalid logits or total")
    return torch.softmax(torch.nan_to_num(logits), dim=-1) * float(total)


def rate_outlier_statistics(rate: torch.Tensor, threshold: float = 1.0e5) -> dict:
    values = torch.as_tensor(rate).detach()
    finite = torch.isfinite(values)
    av = values.abs(); out = finite & (av > float(threshold)); fv = av[finite]
    return {"count": int(values.numel()), "finite_count": int(finite.sum()), "nonfinite_count": int((~finite).sum()), "outlier_count": int(out.sum()), "outlier_fraction": float(out.float().mean()) if values.numel() else 0.0, "abs_p99": float(torch.quantile(fv.float(),0.99)) if fv.numel() else float("nan"), "abs_p999": float(torch.quantile(fv.float(),0.999)) if fv.numel() else float("nan"), "abs_max": float(fv.max()) if fv.numel() else float("nan")}

def robust_rate_target(rate: torch.Tensor, scale: float = 1.0, clip: float = 20.0) -> torch.Tensor:
    if scale <= 0 or clip <= 0: raise ValueError("scale and clip must be positive")
    return torch.asinh(torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0) / float(scale)).clamp(-float(clip), float(clip))


# ============================================================
# V68 实验模式：状态更新策略
# ============================================================
# 实验A：诊断模式 - 输出 raw 和 clamped 统计
# 实验B：受限增量 - ΔY_consume ≥ -Y_k
# 实验C：守恒投影 - 确保 Y_k+1 ≥ 0 且元素守恒
EXPERIMENT_MODES = ["baseline", "diagnostic", "constrained_delta", "conserve_project", "qssa", "dual_branch_pd_v1"]


def artifact_subdir(kind: str) -> str:
    # V68 uses isolated artifact directories
    return f"{kind}_v68"


# ============================================================
# Task8 物理状态更新与投影纯函数 (Task8-only Pure Functions)
# ============================================================

STATE_UPDATE_MODES = ("legacy", "task8_conservative_state", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")
DEFAULT_STATE_UPDATE_MODE = "legacy"
TASK8_VERSION = "task8_v1"
STATE_CONSTRAINT_EPSILON = 1e-12


def decode_model_output(
    model_output: torch.Tensor,
    tools: Any,
    bound_macro: bool = True,
) -> dict[str, torch.Tensor]:
    """Pure function: Decode raw model output into physical macro rates and micro log-rates.

    Task8 specification:
    - Only performs decoding via tools.inverse_rate.
    - Obeys checkpoint's rate_decode_mode (direct_rate or bounded_direct_rate).
    - Does NOT apply any state projection or target clipping.
    """
    rate_raw = tools.inverse_rate(model_output, bound_macro=bound_macro)
    rate_macro = rate_raw[..., :tools.n_macro]
    rate_micro_log = rate_raw[..., tools.n_macro:]
    return {
        "rate_raw": rate_raw,
        "rate_macro": rate_macro,
        "rate_micro_log": rate_micro_log,
    }


def project_physical_state(
    Y_macro_raw: torch.Tensor,
    Y_micro_raw: torch.Tensor,
    inert_sum: float | torch.Tensor = 0.0,
    element_matrix: torch.Tensor | None = None,
    target_elements: torch.Tensor | None = None,
    iterations: int = 8,
    epsilon: float = STATE_CONSTRAINT_EPSILON,
    projection_mode: str = "multiplicative_kl",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Pure function: Project physical species mass fractions onto the nonnegative mass-conserving simplex.

    Task8 Core Guarantees:
    1. Operates exclusively in physical Y space (never on 37-dim raw logits).
    2. Guarantees nonnegativity: Y_constrained >= 0.
    3. Guarantees mass conservation: sum(Y_dyn) + inert_sum == 1.0 (to machine precision).
    4. Supports 'multiplicative_kl' (default): Element-potential information projection solving
       Y_i = r_i * exp(A^T lambda)_i via damped Newton-Raphson. Strictly protects all trace radicals
       from Euclidean zeroing and preserves positivity to machine precision.
       Supports 'weighted_linear' as fallback.
    5. Returns (Y_macro_constrained, Y_micro_constrained, diagnostics).
    6. Does not modify or overwrite raw inputs; records intervention and correction norms.
    """
    orig_shape_macro = Y_macro_raw.shape
    if Y_macro_raw.ndim == 1:
        macro_raw = Y_macro_raw.unsqueeze(0)
        micro_raw = Y_micro_raw.unsqueeze(0)
    else:
        macro_raw = Y_macro_raw
        micro_raw = Y_micro_raw

    dyn_raw = torch.cat([macro_raw, micro_raw], dim=-1)
    b_size, n_dyn = dyn_raw.shape
    n_macro = macro_raw.shape[-1]

    if isinstance(inert_sum, torch.Tensor):
        i_sum = inert_sum.to(dtype=dyn_raw.dtype, device=dyn_raw.device)
        if i_sum.ndim == 0:
            i_sum = i_sum.unsqueeze(0).expand(b_size, 1)
        elif i_sum.ndim == 1:
            i_sum = i_sum.unsqueeze(1)
    else:
        i_sum = torch.full((b_size, 1), float(inert_sum), dtype=dyn_raw.dtype, device=dyn_raw.device)

    # 1. Raw candidate diagnostics
    neg_mask = dyn_raw < 0.0
    neg_count = neg_mask.sum(dim=-1)
    neg_fraction = neg_count.to(dtype=torch.float32) / float(n_dyn)
    raw_mass_sum = dyn_raw.sum(dim=-1, keepdim=True) + i_sum
    mass_sum_deviation = (raw_mass_sum - 1.0).abs()

    # Target sum for dynamic species
    target_sum = torch.clamp(1.0 - i_sum, min=1e-8, max=1.0)

    # 2. Nonnegative truncation
    pos_dyn = torch.clamp(dyn_raw, min=0.0)
    pos_sum = pos_dyn.sum(dim=-1, keepdim=True)

    # 3. Simplex mass conservation scaling
    valid_sum_mask = pos_sum > epsilon
    scaled_dyn = torch.where(
        valid_sum_mask,
        pos_dyn * (target_sum / torch.clamp(pos_sum, min=epsilon)),
        torch.full_like(pos_dyn, 1.0 / float(n_dyn)) * target_sum,
    )

    # 4. Optional elemental conservation projection
    projected_dyn = scaled_dyn
    proj_failed = False
    elem_res_max = torch.zeros(b_size, dtype=scaled_dyn.dtype, device=scaled_dyn.device)
    if element_matrix is not None and target_elements is not None:
        try:
            A = element_matrix.to(dtype=scaled_dyn.dtype, device=scaled_dyn.device)
            target_elem = target_elements.to(dtype=scaled_dyn.dtype, device=scaled_dyn.device)
            if target_elem.ndim == 1:
                target_elem = target_elem.unsqueeze(0).expand(b_size, -1)

            # Check if element matrix A already spans total mass conservation.
            sp_elem_sum = A.sum(dim=0)
            already_spans_mass = torch.allclose(sp_elem_sum, torch.ones_like(sp_elem_sum), atol=1e-3)
            if already_spans_mass:
                A_eff = A
                target_eff = target_elem
            else:
                ones_row = torch.ones((1, A.shape[1]), dtype=A.dtype, device=A.device)
                A_eff = torch.cat([A, ones_row], dim=0)
                target_eff = torch.cat([target_elem, target_sum], dim=-1)

            if projection_mode == "multiplicative_kl":
                # Multiplicative KL (Information / Element-Potential) Projection:
                # Solves: min sum Y_i log(Y_i / r_i) s.t. A_eff @ Y = target_eff
                # Explicit form: Y_i = r_i * exp(A_eff^T @ lambda)_i
                # Solved via damped Newton-Raphson:
                # res = A_eff @ Y - target_eff
                # J = A_eff @ diag(Y) @ A_eff^T
                # d_lambda = (J + reg)^-1 @ res
                # Strictly guarantees Y_i > 0 for all species; NEVER wipes out trace radicals!
                p0 = scaled_dyn.clamp_min(1e-40)
                K = A_eff.shape[0]
                lam = torch.zeros(b_size, K, dtype=p0.dtype, device=p0.device)
                p = p0.clone()
                reg = 1e-11 * torch.eye(K, dtype=p0.dtype, device=p0.device).unsqueeze(0)
                # Eight damped Newton steps are the default budget; callers can
                # request more for unusually difficult projections.
                max_proj_iters = max(int(iterations), 8) if iterations > 0 else 8
                projection_tolerance = max(
                    1e-10,
                    8.0 * torch.finfo(p.dtype).eps,
                )
                for _ in range(max_proj_iters):
                    r = torch.matmul(p, A_eff.T) - target_eff
                    if (r.abs().max(dim=-1).values < projection_tolerance).all():
                        break
                    A_w = A_eff.unsqueeze(0) * p.unsqueeze(1)
                    J = torch.matmul(A_w, A_eff.T)
                    try:
                        d_lam = torch.linalg.solve(J + reg, r.unsqueeze(-1)).squeeze(-1)
                    except Exception:
                        d_lam = torch.matmul(torch.linalg.pinv(J + reg), r.unsqueeze(-1)).squeeze(-1)
                    d_lam = torch.clamp(d_lam, -10.0, 10.0)
                    lam = lam - 0.8 * d_lam
                    exponent = torch.matmul(lam, A_eff).clamp(-30.0, 30.0)
                    p = p0 * torch.exp(exponent)

                p_sum = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                p = p * (target_sum / p_sum)
                elem_res_max = (torch.matmul(p, A.T) - target_elem).abs().max(dim=-1).values
                if (elem_res_max > 1e-6).any():
                    proj_failed = True
                projected_dyn = p
            else:
                # Legacy concentration-weighted closed-form linear projection
                p = scaled_dyn.clone()
                K = A_eff.shape[0]
                reg = 1e-12 * torch.eye(K, dtype=p.dtype, device=p.device).unsqueeze(0)
                n_proj_iters = min(max(iterations, 1), 3) if iterations > 0 else 2
                for _ in range(n_proj_iters):
                    r = target_eff - torch.matmul(p, A_eff.T)
                    W = torch.clamp(p, min=1e-40)
                    A_w = A_eff.unsqueeze(0) * W.unsqueeze(1)
                    M_gram = torch.matmul(A_w, A_eff.T)
                    try:
                        lam = torch.linalg.solve(M_gram + reg, r.unsqueeze(-1)).squeeze(-1)
                    except Exception:
                        lam = torch.matmul(torch.linalg.pinv(M_gram + reg), r.unsqueeze(-1)).squeeze(-1)
                    dY = W * torch.matmul(lam, A_eff)
                    p = torch.clamp(p + dY, min=0.0)
                    p_sum = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    p = p * (target_sum / p_sum)

                elem_res_max = (torch.matmul(p, A.T) - target_elem).abs().max(dim=-1).values
                projected_dyn = p
        except Exception:
            proj_failed = True
            projected_dyn = scaled_dyn
            elem_res_max = (torch.matmul(scaled_dyn, A.T) - target_elem).abs().max(dim=-1).values

    # 5. Separation into macro and micro
    macro_constrained = projected_dyn[..., :n_macro]
    micro_constrained = projected_dyn[..., n_macro:]

    # 6. Correction norms & intervention diagnostics
    correction = projected_dyn - dyn_raw
    correction_l1 = correction.abs().sum(dim=-1)
    correction_l2 = torch.norm(correction, p=2, dim=-1)
    max_correction = correction.abs().max(dim=-1).values
    intervention = max_correction > 1e-6
    zeroed_species = (projected_dyn < 1e-40).sum(dim=-1)

    diagnostics = {
        "intervention": intervention,
        "correction_norm_l1": correction_l1,
        "correction_norm_l2": correction_l2,
        "max_correction": max_correction,
        "correction_per_species": correction,
        "zeroed_species_count": zeroed_species,
        "negative_count": neg_count,
        "negative_fraction": neg_fraction,
        "mass_sum_deviation": mass_sum_deviation.squeeze(-1),
        "target_sum": target_sum.squeeze(-1),
        "constrained_mass_sum": (projected_dyn.sum(dim=-1, keepdim=True) + i_sum).squeeze(-1),
        "element_residual_max": elem_res_max,
        "projection_failed": proj_failed,
        "projection_warning": (elem_res_max > 1e-4),
        "projection_mode": projection_mode,
    }

    if len(orig_shape_macro) == 1:
        macro_constrained = macro_constrained.squeeze(0)
        micro_constrained = micro_constrained.squeeze(0)

    return macro_constrained, micro_constrained, diagnostics


def encode_state_for_model(
    tools: Any,
    base_next_X_scaled: torch.Tensor,
    Y_macro: torch.Tensor,
    logY_micro: torch.Tensor,
    temperature_mode: str = "recorded",
    thermo_closure: dict | None = None,
    current_X_scaled: torch.Tensor | None = None,
    max_delta_temperature: float = 1000.0,
    feedback_input_limit: float | None = None,
) -> torch.Tensor:
    """Pure function: Encode physical macro Y and micro logY into model input space."""
    return tools.state_to_input_scaled(
        base_next_X_scaled=base_next_X_scaled,
        Y_macro_next=Y_macro,
        logY_micro_next=logY_micro,
        temperature_mode=temperature_mode,
        thermo_closure=thermo_closure,
        current_X_scaled=current_X_scaled,
        max_delta_temperature=max_delta_temperature,
        feedback_input_limit=feedback_input_limit,
    )


def apply_task8_state_update(
    tools: Any,
    current_X_scaled: torch.Tensor,
    rate_pred_scaled: torch.Tensor,
    inert_sum: float | torch.Tensor = 0.0,
    element_matrix: torch.Tensor | None = None,
    target_elements: torch.Tensor | None = None,
    iterations: int = 8,
    projection_mode: str = "multiplicative_kl",
) -> dict[str, Any]:
    """Pure function: Task8 physical state update producing strictly separated raw and constrained candidates."""
    current_X, Y_macro_old, logY_micro_old, Y_micro_old = tools.get_current_state(current_X_scaled)
    dt = tools.get_current_dt(current_X_scaled)

    decoded = decode_model_output(rate_pred_scaled, tools, bound_macro=True)
    rate_raw = decoded["rate_raw"]
    rate_macro = decoded["rate_macro"]
    rate_micro_log = decoded["rate_micro_log"]

    # --- 1. Raw Candidate (genuine unconstrained Euler candidate) ---
    delta_Y_macro_raw = rate_macro * dt
    Y_macro_candidate_raw = Y_macro_old + delta_Y_macro_raw
    logY_micro_candidate_raw = logY_micro_old + rate_micro_log * dt
    Y_micro_candidate_raw = torch.pow(
        10.0,
        torch.clamp(logY_micro_candidate_raw, MICRO_LOG_MIN, MICRO_LOG_MAX),
    )

    # --- 2. Constrained Candidate (Task8 physical state projection) ---
    Y_macro_constrained, Y_micro_constrained, diagnostics = project_physical_state(
        Y_macro_raw=Y_macro_candidate_raw,
        Y_micro_raw=Y_micro_candidate_raw,
        inert_sum=inert_sum,
        element_matrix=element_matrix,
        target_elements=target_elements,
        iterations=iterations,
        projection_mode=projection_mode,
    )
    logY_micro_constrained = torch.log10(
        torch.clamp(Y_micro_constrained, min=LOG_EPS, max=1.0)
    )

    return {
        "current_X": current_X,
        "dt": dt,
        "Y_macro_old": Y_macro_old,
        "logY_micro_old": logY_micro_old,
        "rate_raw": rate_raw,
        "rate_macro": rate_macro,
        "rate_micro_log": rate_micro_log,

        # Raw branch (unconstrained Euler step)
        "Y_macro_next_raw": Y_macro_candidate_raw,
        "Y_macro_next_unconstrained": Y_macro_candidate_raw,
        "logY_micro_next_raw": logY_micro_candidate_raw,
        "Y_micro_next_raw": Y_micro_candidate_raw,
        "Y_micro_next_unconstrained": Y_micro_candidate_raw,
        "Y_next_raw_full": torch.cat([Y_macro_candidate_raw, Y_micro_candidate_raw], dim=-1),

        # Constrained branch (Task8 physical state projection)
        "Y_macro_next_constrained": Y_macro_constrained,
        "logY_micro_next_constrained": logY_micro_constrained,
        "Y_micro_next_constrained": Y_micro_constrained,
        "Y_next_constrained_full": torch.cat([Y_macro_constrained, Y_micro_constrained], dim=-1),

        # Compatibility fields for legacy consumers
        "Y_macro_next": Y_macro_constrained,
        "logY_micro_next": logY_micro_constrained,
        "Y_micro_next": Y_micro_constrained,
        "Y_next_safe_full": torch.cat([Y_macro_constrained, Y_micro_constrained], dim=-1),

        # Task8 Diagnostics
        "state_update_mode": "task8_conservative_state",
        "task8_diagnostics": diagnostics,
        "correction_norm_l1": diagnostics["correction_norm_l1"],
        "negative_count": diagnostics["negative_count"],
        "mass_defect": diagnostics["mass_sum_deviation"],
        "intervention": diagnostics["intervention"],
        "diagnostic": {
            "negative_count": int(diagnostics["negative_count"].sum().item()),
            "negative_ratio": float(diagnostics["negative_fraction"].mean().item()),
            "mass_sum_deviation_mean": float(diagnostics["mass_sum_deviation"].mean().item()),
            "max_correction_mean": float(diagnostics["max_correction"].mean().item()),
            "intervention_count": int(diagnostics["intervention"].sum().item()),
        },
        "experiment_mode": "task8_conservative_state",
        "projection_element_error": None,
    }


def apply_legacy_state_update(
    tools: Any,
    current_X_scaled: torch.Tensor,
    rate_pred_scaled: torch.Tensor,
    safe_for_rollout: bool = True,
    experiment_mode: str = "baseline",
    trust_region_fraction: float = 0.0,
    delta_limit: float | None = None,
) -> dict[str, Any]:
    """Pure function: Wrapper for state update logic."""
    return tools.predict_next_state(
        current_X_scaled=current_X_scaled,
        rate_pred_scaled=rate_pred_scaled,
        safe_for_rollout=safe_for_rollout,
        experiment_mode=experiment_mode,
        trust_region_fraction=trust_region_fraction,
        delta_limit=delta_limit,
        state_update_mode="legacy",
    )


# ============================================================


def build_horizon_schedule(horizons: list[int], start_epochs: list[int]) -> dict:
    """Build horizon curriculum schedule."""
    if len(horizons) != len(start_epochs):
        raise ValueError("horizons and start_epochs must have same length")
    return {"horizons": horizons, "start_epochs": start_epochs}


def horizon_for_epoch(epoch: int, schedule: dict) -> int:
    """Return active horizon for given epoch."""
    horizons = schedule["horizons"]
    starts = schedule["start_epochs"]
    for i in range(len(starts) - 1, -1, -1):
        if epoch >= starts[i]:
            return horizons[i]
    return horizons[0]


def build_prediction_tail_mask(
    batch_size: int,
    horizon: int,
    tail_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Build boolean mask for continuous prediction tail.

    Returns shape (batch_size, horizon) where True means use predicted species.
    Each sample transitions from teacher forcing to prediction once and never returns.
    """
    device = tail_lengths.device
    step_indices = torch.arange(horizon, device=device).unsqueeze(0)  # (1, H)
    cutoff = horizon - tail_lengths.view(-1, 1)  # (B, 1)
    return step_indices >= cutoff  # (B, H)


def compute_transition_weights(
    rate_physical: torch.Tensor,
    delta_state_physical: torch.Tensor | None = None,
    alpha: float = 0.0,
    clip: float = 8.0,
) -> torch.Tensor:
    """Return per-transition weights for high-activity/transition samples.

    ``rate_physical`` and ``delta_state_physical`` may have shape ``(B, H, D)``
    or ``(B, D)``.  Activity is normalized by the batch median so the weighting
    is insensitive to the absolute units of a mechanism.  A zero alpha keeps
    the legacy uniform loss exactly unchanged.
    """
    if alpha <= 0.0:
        shape = rate_physical.shape[:-1]
        return torch.ones(shape, dtype=rate_physical.dtype, device=rate_physical.device)
    rate = torch.as_tensor(rate_physical)
    rate_activity = torch.nan_to_num(
        rate.abs(), nan=0.0, posinf=0.0, neginf=0.0
    ).mean(dim=-1)

    def normalize_activity(values: torch.Tensor) -> torch.Tensor:
        flat_values = values.reshape(-1)
        finite_values = flat_values[torch.isfinite(flat_values)]
        if finite_values.numel() == 0:
            return torch.zeros_like(values)
        pos_values = finite_values[finite_values > 1e-8]
        if pos_values.numel() > 0:
            scale = torch.median(pos_values).clamp_min(1e-6)
        else:
            scale = torch.tensor(1.0, device=values.device, dtype=values.dtype)
        return values / scale

    activity = normalize_activity(rate_activity)
    if delta_state_physical is not None:
        delta = torch.as_tensor(delta_state_physical)
        delta_activity = torch.nan_to_num(
            delta.abs(), nan=0.0, posinf=0.0, neginf=0.0
        ).mean(dim=-1)
        activity = activity + normalize_activity(delta_activity)
    weights = 1.0 + float(alpha) * activity
    return torch.clamp(weights, min=1.0, max=max(float(clip), 1.0))


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    feature_weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    stable_accumulation: bool = False,
) -> torch.Tensor:
    """MSE with optional sample, feature weights, and element mask.

    Raw QSSA rollouts can temporarily leave the training distribution by many
    orders of magnitude before the rate head has learned useful bounds.  When
    ``stable_accumulation`` is enabled, accumulate the squared residual in
    float64 so a finite outlier cannot overflow float32 and discard the batch.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if stable_accumulation and prediction.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        residual = prediction.to(torch.float64) - target.to(torch.float64)
    else:
        residual = prediction - target
    squared = residual.pow(2)
    if sample_weights is None and feature_weights is None and mask is None:
        return squared.mean()

    if sample_weights is None:
        sample = torch.ones(
            squared.shape[0], 1, dtype=squared.dtype, device=squared.device
        )
    else:
        sample = sample_weights.to(
            device=squared.device, dtype=squared.dtype
        ).reshape(-1, 1)
        if sample.shape[0] != squared.shape[0]:
            raise ValueError("sample_weights must have one value per sample")

    if feature_weights is None:
        feature = torch.ones(
            1, squared.shape[-1], dtype=squared.dtype, device=squared.device
        )
    else:
        feature = feature_weights.to(
            device=squared.device, dtype=squared.dtype
        ).reshape(1, -1)
        if feature.shape[-1] != squared.shape[-1]:
            raise ValueError("feature_weights must have one value per feature")

    weights = sample * feature
    if mask is not None:
        weights = weights * mask.to(device=weights.device, dtype=weights.dtype)

    denominator = weights.sum().clamp_min(
        torch.finfo(squared.dtype).eps
    )
    return (squared * weights).sum() / denominator


def weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    feature_weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Huber/SmoothL1 loss with the same weighting contract as MSE.

    The quadratic region is retained for small residuals while large source
    outliers contribute linearly, preventing a few extreme rates from
    dominating a batch.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=float(beta))
    if sample_weights is None and feature_weights is None and mask is None:
        return elementwise.mean()
    if sample_weights is None:
        sample = torch.ones((elementwise.shape[0], 1), dtype=elementwise.dtype, device=elementwise.device)
    else:
        sample = sample_weights.to(device=elementwise.device, dtype=elementwise.dtype).reshape(-1, 1)
        if sample.shape[0] != elementwise.shape[0]:
            raise ValueError("sample_weights must have one value per sample")
    if feature_weights is None:
        feature = torch.ones((1, elementwise.shape[-1]), dtype=elementwise.dtype, device=elementwise.device)
    else:
        feature = feature_weights.to(device=elementwise.device, dtype=elementwise.dtype).reshape(1, -1)
        if feature.shape[-1] != elementwise.shape[-1]:
            raise ValueError("feature_weights must have one value per feature")

    weights = sample * feature
    if mask is not None:
        weights = weights * mask.to(device=weights.device, dtype=weights.dtype)

    denominator = weights.sum().clamp_min(torch.finfo(elementwise.dtype).eps)
    return (elementwise * weights).sum() / denominator


def compute_qssa_rate_mask(
    tools: Any,
    true_rate_scaled: torch.Tensor,
    Y_old_unscaled: torch.Tensor | None = None,
    Y_next_unscaled: torch.Tensor | None = None,
    active_threshold: float = 1e-12,
    noise_threshold: float = 1e-12,
) -> torch.Tensor | None:
    """Mask destruction rate heads (odd columns) ONLY for dormant trace radicals.

    Conservative rule:
    Only masks destruction head D when BOTH Y_old and Y_next are below active_threshold
    AND |Y_next - Y_old| < noise_threshold.
    If state information (Y_old, Y_next) is not supplied, no masking is performed (returns None)
    to prevent erroneously masking strong decay transitions (e.g. Y_old=1e-4 -> Y_next=1e-16 with D=1e8).
    """
    rate_mode = getattr(tools, "rate_decode_mode", None)
    if rate_mode not in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"):
        return None

    mask = torch.ones_like(true_rate_scaled)
    scale_y = tools.scale_Y.to(device=true_rate_scaled.device)
    mean_y = tools.mean_Y.to(device=true_rate_scaled.device)
    target_denorm = true_rate_scaled * scale_y + mean_y

    has_states = (Y_old_unscaled is not None and Y_next_unscaled is not None)
    if not has_states:
        # P/D labels do not establish whether the initial concentration is dormant.
        return None

    if rate_mode in ("dual_branch", "dual_branch_pd_v1"):
        slow_dim = len(tools.macro_species) if hasattr(tools, "macro_species") else len(tools.macro_idx)
        micro_targets = target_denorm[:, slow_dim:]
        n_micro = micro_targets.shape[1] // 2
        micro_mask = torch.ones_like(micro_targets)
        for i in range(n_micro):
            if has_states:
                if Y_old_unscaled.shape[1] == n_micro:
                    y_old_i = Y_old_unscaled[:, i]
                    y_next_i = Y_next_unscaled[:, i]
                else:
                    col_idx = tools.micro_idx[i]
                    y_old_i = Y_old_unscaled[:, col_idx]
                    y_next_i = Y_next_unscaled[:, col_idx]

                if (y_old_i < 0).any():
                    y_old_phys = torch.pow(10.0, y_old_i.clamp(min=-40.0, max=0.0))
                    y_next_phys = torch.pow(10.0, y_next_i.clamp(min=-40.0, max=0.0))
                else:
                    y_old_phys = y_old_i
                    y_next_phys = y_next_i
                is_dormant = (y_old_phys < active_threshold) & (y_next_phys < active_threshold) & ((y_next_phys - y_old_phys).abs() < noise_threshold)
            else:
                z1 = micro_targets[:, 2 * i]
                z2 = micro_targets[:, 2 * i + 1]
                is_dormant = (z1 < -15.0) & (z2 <= 4.0)

            micro_mask[:, 2 * i + 1] = torch.where(is_dormant, 0.0, 1.0)

        mask[:, slow_dim:] = micro_mask
        if (mask == 1.0).all():
            return None
        return mask
    elif rate_mode in ("qssa", "qssa_asymptotic"):
        reactive_cols = list(tools.macro_idx) + list(tools.micro_idx)
        n_reactive = true_rate_scaled.shape[1] // 2
        for i in range(n_reactive):
            if has_states:
                if Y_old_unscaled.shape[1] == n_reactive:
                    y_old_i = Y_old_unscaled[:, i]
                    y_next_i = Y_next_unscaled[:, i]
                    is_micro = (i >= len(tools.macro_idx))
                else:
                    col_idx = reactive_cols[i]
                    y_old_i = Y_old_unscaled[:, col_idx]
                    y_next_i = Y_next_unscaled[:, col_idx]
                    is_micro = col_idx in tools.micro_idx

                if is_micro or (y_old_i < 0).any():
                    y_old_phys = torch.pow(10.0, y_old_i.clamp(min=-40.0, max=0.0))
                    y_next_phys = torch.pow(10.0, y_next_i.clamp(min=-40.0, max=0.0))
                else:
                    y_old_phys = y_old_i
                    y_next_phys = y_next_i
                is_dormant = (y_old_phys < active_threshold) & (y_next_phys < active_threshold) & ((y_next_phys - y_old_phys).abs() < noise_threshold)
            else:
                z1 = target_denorm[:, 2 * i]
                z2 = target_denorm[:, 2 * i + 1]
                is_dormant = (z1 < -15.0) & (z2 <= 4.0)

            mask[:, 2 * i + 1] = torch.where(is_dormant, 0.0, 1.0)

        if (mask == 1.0).all():
            return None
        return mask
    if (mask == 1.0).all():
        return None
    return mask



def build_species_loss_weights(
    feature_info: dict,
    macro_weight: float = 1.0,
    micro_weight: float = 1.0,
    key_species_weight: float = 1.0,
    key_species: tuple[str, ...] = ("NH2", "HO2", "OH", "H2O2"),
) -> tuple[torch.Tensor, dict]:
    """Build mean-one rate/state feature weights from the saved species order."""
    for name, value in (
        ("macro_weight", macro_weight),
        ("micro_weight", micro_weight),
        ("key_species_weight", key_species_weight),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    macro_species = list(feature_info["macro_species"])
    micro_species = list(feature_info["micro_species"])
    species = macro_species + micro_species
    weights = torch.full((len(species),), float(micro_weight), dtype=torch.float32)
    weights[:len(macro_species)] = float(macro_weight)
    positions = {species_name: index for index, species_name in enumerate(species)}
    present_key_species = []
    for species_name in key_species:
        index = positions.get(species_name)
        if index is not None:
            weights[index] = float(key_species_weight)
            present_key_species.append(species_name)

    weights = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    metadata = {
        "macro_weight": float(macro_weight),
        "micro_weight": float(micro_weight),
        "key_species_weight": float(key_species_weight),
        "key_species_requested": list(key_species),
        "key_species_present": present_key_species,
        "feature_order": species,
    }
    return weights, metadata


def initialize_transition_audit() -> dict:
    """Create scalar accumulators for one epoch of ignition-weight auditing."""
    return {
        "sample_count": 0,
        "high_activity_count": 0,
        "weight_sum": 0.0,
        "weight_max": 0.0,
        "rate_squared_error_high": 0.0,
        "rate_squared_error_normal": 0.0,
        "state_squared_error_high": 0.0,
        "state_squared_error_normal": 0.0,
        "rate_high_count": 0,
        "rate_normal_count": 0,
        "state_high_count": 0,
        "state_normal_count": 0,
    }


def update_transition_audit(
    audit: dict,
    sample_weights: torch.Tensor | None,
    rate_prediction: torch.Tensor,
    rate_target: torch.Tensor,
    state_prediction: torch.Tensor,
    state_target: torch.Tensor,
) -> torch.Tensor:
    """Accumulate unweighted errors by high-activity status and return weights."""
    if sample_weights is None:
        weights = torch.ones(
            rate_prediction.shape[0], dtype=rate_prediction.dtype,
            device=rate_prediction.device,
        )
    else:
        weights = sample_weights.to(
            device=rate_prediction.device, dtype=rate_prediction.dtype
        ).reshape(-1)
    if weights.numel() != rate_prediction.shape[0]:
        raise ValueError("sample_weights must have one value per sample")
    high = weights > (1.0 + 1e-6)
    normal = ~high
    rate_error = (rate_prediction - rate_target).pow(2).mean(dim=1)
    state_error = (
        state_prediction.to(torch.float64)
        - state_target.to(torch.float64)
    ).pow(2).mean(dim=1)

    audit["sample_count"] += int(weights.numel())
    audit["high_activity_count"] += int(high.sum().item())
    audit["weight_sum"] += float(weights.sum().detach().item())
    if weights.numel():
        audit["weight_max"] = max(
            float(audit.get("weight_max", 0.0)),
            float(weights.max().detach().item()),
        )
    for name, values in (("rate", rate_error), ("state", state_error)):
        for region, mask in (("high", high), ("normal", normal)):
            count = int(mask.sum().item())
            audit[f"{name}_{region}_count"] += count
            if count:
                audit[f"{name}_squared_error_{region}"] += float(
                    values[mask].sum().detach().item()
                )
    return weights.detach()


def merge_transition_weight_samples(
    existing: list[torch.Tensor],
    incoming: list[torch.Tensor],
    max_samples: int = TRANSITION_AUDIT_MAX_QUANTILE_SAMPLES,
) -> list[torch.Tensor]:
    """Merge percentile samples while keeping epoch memory use bounded."""
    chunks = [
        values.reshape(-1).to(dtype=torch.float32, device="cpu")
        for values in (*existing, *incoming)
        if values.numel()
    ]
    if not chunks:
        return []
    combined = torch.cat(chunks)
    if combined.numel() <= max_samples:
        return [combined]
    indices = torch.linspace(
        0,
        combined.numel() - 1,
        steps=max_samples,
        dtype=torch.float32,
    ).round().to(dtype=torch.long)
    return [combined[indices]]


def finalize_transition_audit(audit: dict, weight_values: list[torch.Tensor]) -> dict:
    """Convert epoch accumulators to JSON-safe ignition audit metrics."""
    sample_count = int(audit["sample_count"])
    count = max(sample_count, 1)

    # Build only a bounded, deterministic sample for percentiles.  The old
    # implementation concatenated every transition in an epoch, which can
    # exceed torch.quantile's input limit on long runs.
    total_weight_values = sum(int(values.numel()) for values in weight_values)
    if total_weight_values:
        quantile_chunks = []
        remaining_budget = TRANSITION_AUDIT_MAX_QUANTILE_SAMPLES
        for values in weight_values:
            n_values = int(values.numel())
            if n_values == 0 or remaining_budget <= 0:
                continue
            take = min(
                n_values,
                max(
                    1,
                    int(round(
                        TRANSITION_AUDIT_MAX_QUANTILE_SAMPLES
                        * n_values
                        / total_weight_values
                    )),
                ),
                remaining_budget,
            )
            if take == n_values:
                sampled = values.reshape(-1).to(dtype=torch.float32, device="cpu")
            else:
                indices = torch.linspace(
                    0,
                    n_values - 1,
                    steps=take,
                    dtype=torch.float32,
                ).round().to(dtype=torch.long)
                sampled = values.reshape(-1).to(device="cpu")[indices].to(dtype=torch.float32)
            quantile_chunks.append(sampled)
            remaining_budget -= take
        weights = torch.cat(quantile_chunks)[:TRANSITION_AUDIT_MAX_QUANTILE_SAMPLES]
    else:
        weights = torch.ones(1, dtype=torch.float32)

    recorded_weight_sum = float(audit.get("weight_sum", 0.0))
    if sample_count and recorded_weight_sum:
        weight_mean = recorded_weight_sum / sample_count
    else:
        weight_mean = float(weights.mean().item())
    recorded_weight_max = float(audit.get("weight_max", 0.0))
    weight_max = recorded_weight_max if recorded_weight_max > 0.0 else float(weights.max().item())

    def mean_error(region: str, name: str) -> float | None:
        region_count = int(audit[f"{name}_{region}_count"])
        if region_count == 0:
            return None
        return audit[f"{name}_squared_error_{region}"] / region_count

    return {
        "sample_count": int(audit["sample_count"]),
        "high_activity_fraction": audit["high_activity_count"] / count,
        "weight_mean": float(weight_mean),
        "weight_p50": float(torch.quantile(weights, 0.50).item()),
        "weight_p90": float(torch.quantile(weights, 0.90).item()),
        "weight_max": float(weight_max),
        "rate_mse_high_activity": mean_error("high", "rate"),
        "rate_mse_normal": mean_error("normal", "rate"),
        "state_mse_high_activity": mean_error("high", "state"),
        "state_mse_normal": mean_error("normal", "state"),
    }


def compute_stability_loss(
    predicted_state_sequence: torch.Tensor,
    target_state_sequence: torch.Tensor,
) -> torch.Tensor:
    """Penalize errors in state increments, which drive closed-loop drift.

    Inputs are scaled state features with shape ``(B, H, D)`` (or ``(B, D)``
    for a single state).  The loss compares adjacent predicted and target
    increments, so a correct but non-zero physical transition has zero loss.
    A sequence shorter than two states has no increment and returns a finite
    zero scalar.
    """
    if predicted_state_sequence.shape != target_state_sequence.shape:
        raise ValueError("predicted and target state sequences must have the same shape")
    if predicted_state_sequence.ndim == 2:
        predicted_state_sequence = predicted_state_sequence.unsqueeze(1)
        target_state_sequence = target_state_sequence.unsqueeze(1)
    if predicted_state_sequence.ndim != 3:
        raise ValueError("state sequences must have shape (B, H, D) or (B, D)")
    if predicted_state_sequence.size(1) < 2:
        return predicted_state_sequence.sum() * 0.0
    predicted_delta = predicted_state_sequence[:, 1:] - predicted_state_sequence[:, :-1]
    target_delta = target_state_sequence[:, 1:] - target_state_sequence[:, :-1]
    return weighted_mse_loss(
        predicted_delta,
        target_delta,
        stable_accumulation=True,
    )


def compute_rate_stability_loss(
    predicted_log_rate_sequence: torch.Tensor,
    target_log_rate_sequence: torch.Tensor,
) -> torch.Tensor:
    """Match adjacent log-rate changes for a sequence of QSSA parameters.

    The inputs are already in log10(P), log10(D) feature space with shape
    ``(B, H, D)`` (or ``(B, D)`` for a single state).  Comparing increments
    in this space targets the high-frequency rate jumps that are hidden by a
    state-only smoothness loss.
    """
    if predicted_log_rate_sequence.shape != target_log_rate_sequence.shape:
        raise ValueError("predicted and target log-rate sequences must have the same shape")
    if predicted_log_rate_sequence.ndim == 2:
        predicted_log_rate_sequence = predicted_log_rate_sequence.unsqueeze(1)
        target_log_rate_sequence = target_log_rate_sequence.unsqueeze(1)
    if predicted_log_rate_sequence.ndim != 3:
        raise ValueError("log-rate sequences must have shape (B, H, D) or (B, D)")
    if predicted_log_rate_sequence.size(1) < 2:
        return predicted_log_rate_sequence.sum() * 0.0
    predicted_delta = (
        predicted_log_rate_sequence[:, 1:] - predicted_log_rate_sequence[:, :-1]
    )
    target_delta = target_log_rate_sequence[:, 1:] - target_log_rate_sequence[:, :-1]
    return weighted_mse_loss(
        predicted_delta,
        target_delta,
        stable_accumulation=True,
    )


def qssa_log_rate_features_from_scaled(
    rate_sequence_scaled: torch.Tensor,
    tools,
) -> torch.Tensor:
    """Decode scaled QSSA outputs into log10(P), log10(D) features."""
    if rate_sequence_scaled.ndim == 2:
        flat = rate_sequence_scaled
        shape = None
    elif rate_sequence_scaled.ndim == 3:
        shape = rate_sequence_scaled.shape[:-1]
        flat = rate_sequence_scaled.reshape(-1, rate_sequence_scaled.shape[-1])
    else:
        raise ValueError("scaled rate sequence must have shape (B, D) or (B, H, D)")

    if getattr(tools, "rate_decode_mode", None) == "qssa_asymptotic":
        logP, logD = tools.inverse_qssa_asymptotic_log_parameters(flat)
        log_features = torch.cat([logP, logD], dim=-1)
    else:
        generation, destruction = tools.inverse_qssa_parameters(flat)
        log_features = torch.cat(
            [
                torch.log10(generation.clamp_min(LOG_EPS)),
                torch.log10(destruction.clamp_min(LOG_EPS)),
            ],
            dim=1,
        )
    return log_features if shape is None else log_features.reshape(*shape, -1)


def state_features_scaled_from_prediction(
    pred_state: dict,
    tools,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
) -> torch.Tensor:
    """Convert a predicted physical state to the scaled species feature space."""
    macro, log_micro = select_training_state(pred_state, state_loss_source)
    device = tools.mean_X.device
    macro = macro.to(device)
    log_micro = log_micro.to(device)
    pred_macro_scaled = (
        macro - tools.mean_X[tools.macro_idx]
    ) / tools.scale_X[tools.macro_idx]
    pred_micro_log_scaled = (
        log_micro - tools.mean_X[tools.micro_idx]
    ) / tools.scale_X[tools.micro_idx]
    return torch.cat([pred_macro_scaled, pred_micro_log_scaled], dim=1)


def _call_state_features_scaled(func, pred_state, tools, state_loss_source=DEFAULT_STATE_LOSS_SOURCE):
    """Call state_features_scaled_from_prediction supporting both 2-arg and 3-arg signatures/mocks."""
    try:
        return func(pred_state, tools, state_loss_source=state_loss_source)
    except TypeError:
        return func(pred_state, tools)



def state_features_scaled_from_input(
    state_X_scaled: torch.Tensor,
    tools,
) -> torch.Tensor:
    """Extract species features from an already scaled input state."""
    return torch.cat(
        [state_X_scaled[:, tools.macro_idx], state_X_scaled[:, tools.micro_idx]],
        dim=1,
    )


def add_feedback_state_noise(
    state_scaled: torch.Tensor,
    noise_std: float,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inject zero-mean scaled-state noise into model feedback states."""
    if noise_std <= 0.0:
        return state_scaled
    noise = torch.randn_like(state_scaled) * float(noise_std)
    if active_mask is not None:
        noise = noise * active_mask.to(device=state_scaled.device, dtype=state_scaled.dtype)
    return state_scaled + noise


def apply_feedback_state_noise(
    feedback_macro: torch.Tensor,
    feedback_log_micro: torch.Tensor,
    feedback_noise_std: float,
    feedback_state_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject training noise while preserving the selected feedback semantics.

    Raw feedback is intentionally allowed to leave the constrained physical
    ranges.  The raw-only standardized input trust region is applied later by
    ``state_to_input_scaled``.  Constrained feedback keeps the historical
    physical clamps.
    """
    if feedback_state_source not in FEEDBACK_STATE_SOURCES:
        raise ValueError(
            f"Unknown feedback_state_source: {feedback_state_source}, "
            f"must be one of {FEEDBACK_STATE_SOURCES}"
        )
    if feedback_noise_std <= 0.0:
        return feedback_macro, feedback_log_micro

    macro_noise = (
        torch.randn_like(feedback_macro)
        * float(feedback_noise_std)
        * feedback_macro.clamp_min(1e-4)
    )
    feedback_macro = feedback_macro + macro_noise
    micro_log_noise = torch.randn_like(feedback_log_micro) * (
        float(feedback_noise_std) * 2.0
    )
    feedback_log_micro = feedback_log_micro + micro_log_noise

    if feedback_state_source == "constrained":
        feedback_macro = feedback_macro.clamp(0.0, 1.0)
        feedback_log_micro = feedback_log_micro.clamp(MICRO_LOG_MIN, 0.0)
    return feedback_macro, feedback_log_micro


def checkpoint_compatible(
    checkpoint: dict,
    expected_version: str,
    expected_transform: str,
    expected_rollout: str,
    expected_training_config: dict[str, Any] | None = None,
) -> bool:
    """Check whether a checkpoint can safely resume the current experiment.

    A checkpoint may have the same tensor shapes while using different
    physical target semantics or optimization weights.  When the caller
    supplies the current training configuration, every listed key must be
    present and equal before optimizer/RNG state is restored.
    """
    # The rate heads have identical shapes across several transforms, but their
    # semantics differ (for example, qssa uses log(P/D) while asymptotic-QSSA
    # uses log(Y_qssa/D)).  A shape-compatible checkpoint with another
    # transform would silently resume with the wrong physical decoder.
    compatible = (
        checkpoint.get("version") == expected_version
        and checkpoint.get("transform_signature") == expected_transform
        and checkpoint.get("rollout_signature") == expected_rollout
    )
    if not compatible or expected_training_config is None:
        return compatible

    saved_config = checkpoint.get("training_config")
    if not isinstance(saved_config, dict):
        return False

    for key, expected_value in expected_training_config.items():
        if key not in saved_config:
            return False
        saved_value = saved_config[key]
        if isinstance(expected_value, (float, int)) and isinstance(saved_value, (float, int)):
            if not math.isclose(
                float(saved_value),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
        elif saved_value != expected_value:
            return False
    return True


def make_window(X: np.ndarray, Y: np.ndarray, start: int, horizon: int):
    """Extract (H+1, H) window for dynamic horizon."""
    x_window = X[start : start + horizon + 1]
    y_window = Y[start : start + horizon]
    return x_window, y_window


class HorizonMetricAccumulator:
    """Accumulate per-horizon validation metrics."""

    def __init__(self, eval_horizons: list[int]):
        self.eval_horizons = sorted(eval_horizons)
        self.accum = {str(h): {"sum": 0.0, "count": 0} for h in self.eval_horizons}

    def update(self, horizon: int, pred: torch.Tensor, target: torch.Tensor, valid_count: int):
        """Update metric for specific horizon endpoint."""
        h_str = str(horizon)
        if h_str in self.accum:
            # Raw rollouts can be far outside the training range; accumulate
            # the metric in float64 so a finite outlier remains reportable.
            error = (
                pred.to(torch.float64) - target.to(torch.float64)
            ).pow(2).mean().item()
            self.accum[h_str]["sum"] += error * valid_count
            self.accum[h_str]["count"] += valid_count

    def finalize(self) -> dict:
        """Return per-horizon RMSE."""
        result = {}
        for h_str, acc in self.accum.items():
            if acc["count"] > 0:
                result[h_str] = {
                    "rmse": (acc["sum"] / acc["count"]) ** 0.5,
                    "count": acc["count"],
                }
            else:
                result[h_str] = {"rmse": float("nan"), "count": 0}
        return result


def require_positive_dt(dt):
    """Validate that dt values are finite and strictly positive."""
    value = np.asarray(dt, dtype=np.float64)
    if not np.all(np.isfinite(value) & (value > 0)):
        raise ValueError("dt must be finite and strictly positive")
    return value


def require_positive_dt_tensor(dt):
    """Validate that torch dt tensor values are finite and strictly positive."""
    if not torch.all(torch.isfinite(dt) & (dt > 0)):
        raise ValueError("dt must be finite and strictly positive")
    return dt


def sanitize_dt_numpy(dt, dt_min=None, dt_max=None):
    """Validate strictly positive dt without silent implicit clipping.

    If dt_min or dt_max is explicitly provided, clips to that range.
    Otherwise preserves exact positive dt values.
    """
    dt_arr = require_positive_dt(dt)
    if dt_min is not None or dt_max is not None:
        low = dt_min if dt_min is not None else -np.inf
        high = dt_max if dt_max is not None else np.inf
        return np.clip(dt_arr, low, high)
    return dt_arr


def sanitize_dt_tensor(dt, dt_min=None, dt_max=None):
    """Validate strictly positive dt tensor without silent implicit clipping."""
    dt_t = require_positive_dt_tensor(dt)
    if dt_min is not None or dt_max is not None:
        low = dt_min if dt_min is not None else 1e-12
        high = dt_max if dt_max is not None else float("inf")
        return torch.clamp(dt_t, min=low, max=high)
    return dt_t


def compute_macro_rate_scale(raw_rates: np.ndarray) -> np.ndarray:
    raw_rates = np.asarray(raw_rates, dtype=np.float64)
    finite = np.isfinite(raw_rates).all(axis=1)
    if not finite.any():
        raise ValueError("No finite macro rates available for rate scales")
    scales = np.percentile(np.abs(raw_rates[finite]), 90.0, axis=0)
    return np.maximum(scales, RATE_SCALE_FLOOR)


def transform_macro_rates(raw_rates, rate_scale):
    raw_rates = np.asarray(raw_rates, dtype=np.float64)
    rate_scale = np.asarray(rate_scale, dtype=np.float64)
    return np.arcsinh(raw_rates / rate_scale)


def inverse_macro_rates(z, rate_scale):
    z = np.asarray(z, dtype=np.float64)
    rate_scale = np.asarray(rate_scale, dtype=np.float64)
    return rate_scale * np.sinh(z)


def partition_files_for_worker(files, worker_id, num_workers):
    if num_workers <= 1:
        return list(files)
    return list(files)[worker_id::num_workers]


def select_epoch_files(files, count, seed, epoch, shuffle=True):
    files = list(files)
    if not files:
        return []
    if count is None:
        count = len(files)
    if int(count) < 1:
        raise ValueError("file sample count must be >= 1")
    rng = np.random.default_rng(seed + 1009 * int(epoch))
    if shuffle:
        rng.shuffle(files)
    return files[:min(int(count), len(files))]


def find_cross_part_sample_ids(part_sample_ids):
    owner = {}
    duplicate = {}
    for part, sample_ids in part_sample_ids.items():
        for sample_id in sample_ids:
            if sample_id in owner and sample_id not in duplicate:
                duplicate[sample_id] = (owner[sample_id], part)
            else:
                owner.setdefault(sample_id, part)
    return duplicate


def validate_loaded_feature_info(loaded, expected):
    for key in ("all_species", "input_cols", "output_cols"):
        if list(loaded.get(key, [])) != list(expected.get(key, [])):
            raise RuntimeError(f"Loaded feature_info mismatch for {key}")


def require_detached_rollout(detach_rollout: bool):
    if not detach_rollout:
        raise ValueError(
            "Segmented rollout requires detach_rollout=True; "
            "set it explicitly only after implementing a different BPTT path"
        )


def backward_scaled_step_loss(step_loss, unroll_steps):
    """Backpropagate one detached rollout step and return its mean-loss value."""
    scaled_step_loss = step_loss / unroll_steps
    if not torch.isfinite(scaled_step_loss):
        return False, 0.0
    scaled_step_loss.backward()
    return True, scaled_step_loss.detach().item()


def backward_loss_chunk(chunk_loss, unroll_steps):
    """Backpropagate one detached chunk scaled by the full unroll length."""
    scaled_chunk_loss = chunk_loss / unroll_steps
    if not torch.isfinite(scaled_chunk_loss):
        return False, 0.0
    scaled_chunk_loss.backward()
    return True, scaled_chunk_loss.detach().item()


def validate_part_sample_isolation(parquet_files):
    part_sample_ids = {}
    for path in parquet_files:
        sample_ids = set()
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=262_144,
            columns=["sample_id"],
        ):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            sample_ids.update(np.unique(values).tolist())
        part_sample_ids[path.name] = sample_ids
    duplicates = find_cross_part_sample_ids(part_sample_ids)
    if duplicates:
        examples = list(duplicates.items())[:5]
        raise RuntimeError(
            "sample_id trajectories span parquet parts; part-level train/val "
            f"split would leak or truncate trajectories. Examples: {examples}"
        )


def resolve_trajectory_split_dir(data_dir: str, split_dir=None) -> Optional[Path]:
    """Resolve the optional complete-trajectory split directory.

    An explicit ``split_dir`` always wins.  When omitted, the standard
    ``<data_dir>/trajectory_splits_QSSA`` sibling is used if present; otherwise
    the legacy parquet-part split remains available for backward compatibility.
    """
    if split_dir is not None and str(split_dir).strip():
        path = Path(split_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Trajectory split directory not found: {path}")
        return path

    candidates = (
        Path(data_dir).expanduser() / "trajectory_splits_QSSA",
        Path(data_dir).expanduser().parent / "trajectory_splits_QSSA",
        Path(__file__).resolve().parent / "trajectory_splits_QSSA",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def load_trajectory_sample_ids(split_dir, mechanism: str, split: str) -> set[str]:
    """Load and validate one split's sample-id list."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported trajectory split: {split!r}")
    path = Path(split_dir) / f"{mechanism}_{split}_sample_ids.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory split file not found: {path}")
    frame = pd.read_csv(path, usecols=["sample_id"])
    if frame.empty:
        raise ValueError(f"Trajectory split file is empty: {path}")
    if frame["sample_id"].isna().any():
        raise ValueError(f"Trajectory split contains null sample_id values: {path}")
    values = {str(value) for value in frame["sample_id"].tolist()}
    if len(values) != len(frame):
        raise ValueError(f"Trajectory split contains duplicate sample_id values: {path}")
    return values


def trajectory_split_signature(split_dir, mechanism: str, split: str = "train") -> str:
    """Return a content-based signature for a split list."""
    path = Path(split_dir) / f"{mechanism}_{split}_sample_ids.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{split}:{digest}"


# ============================================================
# 1. Feature scaler
# ============================================================

class FeatureScaler:
    """
    简单、明确的 StandardScaler 替代品。

    注意：
    这里统计的是“网络真正使用的特征空间”，
    而不是 parquet 原始变量空间。
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        n_samples: int,
    ):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.n_samples_seen_ = int(n_samples)

        self.scale_[self.scale_ < 1e-12] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


# ============================================================
# 2. Online statistics
# ============================================================

class OnlineMoments:
    """
    批量版 Welford / parallel variance accumulation。
    """

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray):
        if batch is None or len(batch) == 0:
            return

        batch = np.asarray(batch, dtype=np.float64)

        if not np.isfinite(batch).all():
            mask = np.isfinite(batch).all(axis=1)
            batch = batch[mask]

        if len(batch) == 0:
            return

        n2 = batch.shape[0]

        batch_mean = batch.mean(axis=0)

        batch_var_sum = (
            (batch - batch_mean) ** 2
        ).sum(axis=0)

        if self.n == 0:
            self.n = n2
            self.mean = batch_mean
            self.M2 = batch_var_sum
            return

        delta = batch_mean - self.mean

        total_n = self.n + n2

        self.mean = (
            self.mean
            + delta * n2 / total_n
        )

        self.M2 = (
            self.M2
            + batch_var_sum
            + delta**2 * self.n * n2 / total_n
        )

        self.n = total_n

    def finalize(self) -> FeatureScaler:
        if self.n <= 1:
            scale = np.ones_like(self.mean)
        else:
            var = self.M2 / (self.n - 1)
            scale = np.sqrt(
                np.maximum(var, 0.0)
            )

        scale[scale < 1e-12] = 1.0

        return FeatureScaler(
            self.mean.astype(np.float32),
            scale.astype(np.float32),
            self.n,
        )


# ============================================================
# 3. Metadata / schema
# ============================================================

def resolve_mechanism_trajectory_dir(data_dir: str | Path, mechanism: str) -> Path:
    p = Path(data_dir)
    trajectory_name = f"trajectories_{mechanism}"
    candidates = [
        p / trajectory_name,
        p / "database_Transient_V21_QSSA" / trajectory_name,
        p / "database_Transient_V21" / trajectory_name,
    ]
    try:
        child_dirs = sorted(child for child in p.iterdir() if child.is_dir())
    except OSError:
        child_dirs = []
    candidates.extend(
        child / trajectory_name
        for child in child_dirs
        if child / trajectory_name not in candidates
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return p / f"trajectories_{mechanism}"


def load_metadata(
    data_dir: str,
    mechanism: str,
):
    traj_dir = resolve_mechanism_trajectory_dir(data_dir, mechanism)
    path = traj_dir / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # V21 写的是 orient=records
    if isinstance(metadata, list):
        metadata = metadata[0]

    print("\n[Metadata]")
    print(
        f"  mechanism = {mechanism}"
    )
    print(
        f"  rows      = "
        f"{metadata.get('training_rows', 'N/A')}"
    )
    print(
        f"  success   = "
        f"{metadata.get('success', 'N/A')}"
    )
    print(
        f"  failed    = "
        f"{metadata.get('failed', 'N/A')}"
    )
    print(
        f"  parts     = "
        f"{metadata.get('parquet_parts', 'N/A')}"
    )

    return metadata


def get_schema(
    data_dir: str,
    mechanism: str,
):
    parquet_dir = resolve_mechanism_trajectory_dir(data_dir, mechanism)

    files = sorted(
        parquet_dir.glob("part-*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files in {parquet_dir}"
        )

    schema = pq.read_schema(files[0])

    columns = schema.names

    Y_cols = [
        c for c in columns
        if c.startswith("Y_")
        and not c.startswith("Y_next_")
    ]

    dYdt_cols = [
        c for c in columns
        if c.startswith("dYdt_")
    ]

    species = [
        c.replace("Y_", "")
        for c in Y_cols
    ]

    return (
        columns,
        species,
        dYdt_cols,
        files,
    )


# ============================================================
# 4. Species classification
# ============================================================

def classify_species(species: List[str], include_no_in_macro: bool = False):
    inert_targets = {
        "AR",
        "HE",
        "ARGON",
        "HELIUM",
    }

    macro_targets = {
        "NH3",
        "H2O",
        "N2",
        "H2",
        "O2",
    }
    if include_no_in_macro:
        macro_targets.add("NO")

    inert = [
        sp for sp in species
        if sp.upper() in inert_targets
    ]

    macro = [
        sp for sp in species
        if sp.upper() in macro_targets
    ]

    micro = [
        sp for sp in species
        if sp.upper()
        not in inert_targets
        and sp.upper()
        not in macro_targets
    ]

    return inert, macro, micro


# ============================================================
# 5. Input / target transformation
# ============================================================

def transform_input_array(
    df: pd.DataFrame,
    inert_species: List[str],
    macro_species: List[str],
    micro_species: List[str],
):
    """
    真正的网络输入空间：

        T
        P
        Y_inert
        Y_macro
        log10(Y_micro)
        log10(dt)

    不返回原始 dt/Y。
    """

    arrays = []

    arrays.append(
        df["T"].to_numpy(dtype=np.float32)
    )

    arrays.append(
        df["P"].to_numpy(dtype=np.float32)
    )

    for sp in inert_species:
        arrays.append(
            df[f"Y_{sp}"].to_numpy(dtype=np.float32)
        )

    for sp in macro_species:
        arrays.append(
            df[f"Y_{sp}"].to_numpy(dtype=np.float32)
        )

    for sp in micro_species:
        y = df[f"Y_{sp}"].to_numpy(
            dtype=np.float64
        )

        y_log = np.log10(
            np.clip(
                y,
                LOG_EPS,
                None,
            )
        )

        arrays.append(
            y_log.astype(np.float32)
        )

    dt = sanitize_dt_numpy(
        df["dt"].to_numpy(dtype=np.float64)
    )

    dt_log = np.log10(dt)

    arrays.append(
        dt_log.astype(np.float32)
    )

    return np.column_stack(arrays)


def transform_target_array(
    df,
    macro_species,
    micro_species,
    macro_rate_scale,
    output_parameterization: str = "direct_rate",
):
    if output_parameterization not in {"direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"}:
        raise ValueError(f"invalid output_parameterization: {output_parameterization!r}")
    arrays = []
    reactive_species = list(macro_species) + list(micro_species)
    if output_parameterization == "qssa":
        for sp in reactive_species:
            P = df[f"P_avg_{sp}"].to_numpy(dtype=np.float64)
            D = df[f"D_coef_avg_{sp}"].to_numpy(dtype=np.float64)
            arrays.append(np.log10(np.clip(P, LOG_EPS, None)).astype(np.float32))
            arrays.append(np.log10(np.clip(D, LOG_EPS, None)).astype(np.float32))
    elif output_parameterization == "qssa_asymptotic":
        for sp in reactive_species:
            P = df[f"P_avg_{sp}"].to_numpy(dtype=np.float64)
            D = df[f"D_coef_avg_{sp}"].to_numpy(dtype=np.float64)
            # Dedicated production-only branch is reserved for exactly D=0.
            # Positive coefficients down to LOG_EPS remain ordinary logarithmic
            # targets instead of being silently changed into pure production.
            is_prod_only = D <= 0.0
            is_zero_creation = P <= 0.0
            logP = np.log10(np.clip(P, LOG_EPS, None))
            logD = np.log10(np.clip(D, LOG_EPS, None))
            logY_qssa = np.clip(logP - logD, -40.0, 30.0)
            z1 = np.where(
                is_zero_creation,
                -45.0,
                np.where(is_prod_only, logP, logY_qssa),
            ).astype(np.float32)
            z2 = np.where(is_prod_only, -45.0, logD).astype(np.float32)
            arrays.append(z1)
            arrays.append(z2)
    elif output_parameterization == "dual_branch":
        # Slow branch (macro species): asinh direct reaction rates
        scales = np.asarray(macro_rate_scale, dtype=np.float64)
        if scales.size != len(macro_species):
            raise ValueError("macro_rate_scale must have one value per macro species")
        for sp, scale in zip(macro_species, scales):
            dYdt = df[f"dYdt_{sp}"].to_numpy(dtype=np.float64)
            arrays.append(np.arcsinh(dYdt / max(float(scale), RATE_SCALE_FLOOR)).astype(np.float32))
        # Fast branch (micro radicals): QSSA asymptotic (logY_qssa, logD)
        for sp in micro_species:
            P = df[f"P_avg_{sp}"].to_numpy(dtype=np.float64)
            D = df[f"D_coef_avg_{sp}"].to_numpy(dtype=np.float64)
            is_prod_only = D <= 0.0
            is_zero_creation = P <= 0.0
            logP = np.log10(np.clip(P, LOG_EPS, None))
            logD = np.log10(np.clip(D, LOG_EPS, None))
            logY_qssa = np.clip(logP - logD, -40.0, 30.0)
            z1 = np.where(
                is_zero_creation,
                -45.0,
                np.where(is_prod_only, logP, logY_qssa),
            ).astype(np.float32)
            z2 = np.where(is_prod_only, -45.0, logD).astype(np.float32)
            arrays.append(z1)
            arrays.append(z2)
    elif output_parameterization == "dual_branch_pd_v1":
        # Slow branch (macro species): asinh direct reaction rates
        scales = np.asarray(macro_rate_scale, dtype=np.float64)
        if scales.size != len(macro_species):
            raise ValueError("macro_rate_scale must have one value per macro species")
        for sp, scale in zip(macro_species, scales):
            dYdt = df[f"dYdt_{sp}"].to_numpy(dtype=np.float64)
            arrays.append(np.arcsinh(dYdt / max(float(scale), RATE_SCALE_FLOOR)).astype(np.float32))
        # Fast branch (micro radicals): continuous log10(P + eps_P) and log10(D + eps_D)
        for sp in micro_species:
            P = df[f"P_avg_{sp}"].to_numpy(dtype=np.float64)
            D = df[f"D_coef_avg_{sp}"].to_numpy(dtype=np.float64)
            z_P, z_D = kp.encode_pd_targets(P, D)
            arrays.append(np.asarray(z_P, dtype=np.float32))
            arrays.append(np.asarray(z_D, dtype=np.float32))
    else:
        scales = np.asarray(macro_rate_scale, dtype=np.float64)
        if scales.size != len(macro_species):
            raise ValueError("macro_rate_scale must have one value per macro species")
        for sp, scale in zip(macro_species, scales):
            dYdt = df[f"dYdt_{sp}"].to_numpy(dtype=np.float64)
            arrays.append(np.arcsinh(dYdt / max(float(scale), RATE_SCALE_FLOOR)).astype(np.float32))
        if micro_species:
            if "dt" not in df.columns:
                raise ValueError("direct_rate target transform requires dt for micro species")
            dt = np.clip(df["dt"].to_numpy(dtype=np.float64), DT_NUMERIC_FLOOR, None)
            for sp in micro_species:
                y = np.clip(df[f"Y_{sp}"].to_numpy(dtype=np.float64), LOG_EPS, None)
                if f"Y_next_{sp}" in df.columns:
                    y_next = np.clip(df[f"Y_next_{sp}"].to_numpy(dtype=np.float64), LOG_EPS, None)
                    dlog = (np.log10(y_next) - np.log10(y)) / dt
                else:
                    dYdt = df[f"dYdt_{sp}"].to_numpy(dtype=np.float64)
                    dlog = dYdt / (y * np.log(10.0))
                arrays.append(dlog.astype(np.float32))
    return np.column_stack(arrays)


# ============================================================
# 6. Build feature information
# ============================================================

def build_feature_info(
    species,
    inert_species,
    macro_species,
    micro_species,
    output_parameterization: str = "direct_rate",
    rate_decode_mode: str = "direct_rate",
    rate_z_limit: float = 10.0,
):
    input_cols = ["T", "P"]
    for sp in inert_species:
        input_cols.append(f"Y_{sp}")
    for sp in macro_species:
        input_cols.append(f"Y_{sp}")
    for sp in micro_species:
        input_cols.append(f"logY_{sp}")
    input_cols.append("log_dt")

    if output_parameterization not in {"direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"}:
        raise ValueError(f"invalid output_parameterization: {output_parameterization!r}")
    if rate_decode_mode not in RATE_DECODE_MODES:
        raise ValueError(f"invalid rate_decode_mode: {rate_decode_mode!r}")
    if (output_parameterization in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")) != (rate_decode_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")):
        raise ValueError("qssa and dual_branch output_parameterization and rate_decode_mode must be paired")
    if (output_parameterization == "qssa_asymptotic") != (rate_decode_mode == "qssa_asymptotic"):
        raise ValueError("qssa_asymptotic output_parameterization and rate_decode_mode must match")
    if (output_parameterization == "dual_branch") != (rate_decode_mode == "dual_branch"):
        raise ValueError("dual_branch output_parameterization and rate_decode_mode must match")
    if (output_parameterization == "dual_branch_pd_v1") != (rate_decode_mode == "dual_branch_pd_v1"):
        raise ValueError("dual_branch_pd_v1 output_parameterization and rate_decode_mode must match")

    output_cols = []
    reactive_species = list(macro_species) + list(micro_species)
    if output_parameterization == "qssa":
        for sp in reactive_species:
            output_cols.extend((f"log10_P_avg_{sp}", f"log10_D_coef_avg_{sp}"))
        transform_signature = QSSA_TRANSFORM_SIGNATURE
    elif output_parameterization == "qssa_asymptotic":
        for sp in reactive_species:
            output_cols.extend((f"log10_Y_qssa_{sp}", f"log10_D_coef_avg_{sp}"))
        transform_signature = QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE
    elif output_parameterization == "dual_branch":
        for sp in macro_species:
            output_cols.append(f"asinh_dYdt_{sp}")
        for sp in micro_species:
            output_cols.extend((f"log10_Y_qssa_{sp}", f"log10_D_coef_avg_{sp}"))
        transform_signature = DUAL_BRANCH_TRANSFORM_SIGNATURE
    elif output_parameterization == "dual_branch_pd_v1":
        for sp in macro_species:
            output_cols.append(f"asinh_dYdt_{sp}")
        for sp in micro_species:
            output_cols.extend((f"log10_P_avg_{sp}", f"log10_D_coef_avg_{sp}"))
        transform_signature = DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE
    else:
        output_cols.extend(f"asinh_dYdt_{sp}" for sp in macro_species)
        output_cols.extend(f"dlog10Ydt_{sp}" for sp in micro_species)
        transform_signature = LEGACY_TRANSFORM_SIGNATURE

    return {
        "version": VERSION,
        "transform_signature": transform_signature,
        "all_species": list(species),

        "inert_species": list(
            inert_species
        ),

        "macro_species": list(
            macro_species
        ),

        "micro_species": list(
            micro_species
        ),

        "input_cols": input_cols,

        "output_cols": output_cols,

        "n_inert": len(inert_species),

        "n_macro": len(macro_species),

        "n_micro": len(micro_species),

        "epsilon": LOG_EPS,

        "dt_min": DEFAULT_DT_MIN,

        "dt_max": DEFAULT_DT_MAX,
        "macro_rate_scale": np.ones(len(macro_species), dtype=np.float32),
        "output_parameterization": output_parameterization,
        "rate_decode_mode": rate_decode_mode,
        "rate_z_limit": float(rate_z_limit),
    }


# ============================================================
# 7. Streaming scaler computation
# ============================================================

def compute_streaming_scalers(
    data_dir: str,
    mechanism: str,
    feature_info: dict,
    parquet_files: List[Path],
    max_parts: int = 10,
    sample_rate: float = 0.02,
    max_rows: int = 2_000_000,
    sample_ids=None,
    desentinelize_scaler: bool = False,
):
    """
    注意：

    scaler 统计的是：
        网络真正输入/输出的 transformed space

    而不是 parquet 的原始变量。

    """

    rng = np.random.default_rng(
        SEED
    )

    if max_parts is not None:
        n = min(
            max_parts,
            len(parquet_files),
        )

        selected = rng.choice(
            parquet_files,
            size=n,
            replace=False,
        ).tolist()
    else:
        selected = list(
            parquet_files
        )

    print(
        "\n[Scaler] Selected parts:"
    )

    for p in selected:
        print(
            f"  - {p.name}"
        )

    sample_ids = None if sample_ids is None else {str(value) for value in sample_ids}

    # Pass 1: estimate a robust per-macro-species physical rate scale.
    macro_cols = [
        f"dYdt_{sp}"
        for sp in feature_info["macro_species"]
    ]
    macro_chunks = []
    scale_rows = 0
    for p in selected:
        if scale_rows >= max_rows:
            break
        read_cols = ["sample_id", *macro_cols] if sample_ids is not None else macro_cols
        raw_macro = pd.read_parquet(p, columns=read_cols)
        if sample_ids is not None:
            raw_macro = raw_macro[raw_macro["sample_id"].astype(str).isin(sample_ids)][macro_cols]
        if sample_rate < 1.0:
            raw_macro = raw_macro.sample(frac=sample_rate, random_state=SEED)
        remaining = max_rows - scale_rows
        if len(raw_macro) > remaining:
            raw_macro = raw_macro.iloc[:remaining]
        values = raw_macro.to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        if finite.any():
            values = values[finite]
            macro_chunks.append(values)
            scale_rows += len(values)
        del raw_macro

    if not macro_chunks:
        raise RuntimeError("No finite macro rates available for scaler computation")

    macro_rate_scale = compute_macro_rate_scale(
        np.concatenate(macro_chunks, axis=0)
    ).astype(np.float32)
    feature_info["macro_rate_scale"] = macro_rate_scale
    feature_info["version"] = VERSION
    feature_info["transform_signature"] = (
        DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE
        if feature_info.get("output_parameterization") == "dual_branch_pd_v1"
        else (
            DUAL_BRANCH_TRANSFORM_SIGNATURE
            if feature_info.get("output_parameterization") == "dual_branch"
            else (
                QSSA_ASYMPTOTIC_TRANSFORM_SIGNATURE
                if feature_info.get("output_parameterization") == "qssa_asymptotic"
                else (
                    QSSA_TRANSFORM_SIGNATURE
                    if feature_info.get("output_parameterization") == "qssa"
                    else LEGACY_TRANSFORM_SIGNATURE
                )
            )
        )
    )

    print("[Scaler] Macro P90 rate scales:")
    for sp, scale in zip(feature_info["macro_species"], macro_rate_scale):
        print(f"  {sp}: {scale:.6e}")

    del macro_chunks
    gc.collect()

    x_dim = len(
        feature_info["input_cols"]
    )

    y_dim = len(
        feature_info["output_cols"]
    )

    moments_X = OnlineMoments(
        x_dim
    )

    if desentinelize_scaler:
        from enhanced_loss_and_weights import MaskedOnlineMoments
        moments_Y = MaskedOnlineMoments(
            y_dim, sentinel_threshold=-35.0
        )
        print("[Scaler] Enabled MaskedOnlineMoments to prevent frozen sentinels from inflating variance!")
    else:
        moments_Y = OnlineMoments(
            y_dim
        )

    processed = 0

    columns_needed = [
        "T",
        "P",
        "dt",
    ]

    for sp in feature_info[
        "inert_species"
    ]:
        columns_needed.append(
            f"Y_{sp}"
        )

    for sp in feature_info[
        "macro_species"
    ]:
        columns_needed.append(
            f"Y_{sp}"
        )

    for sp in feature_info[
        "micro_species"
    ]:
        columns_needed.append(
            f"Y_{sp}"
        )
        columns_needed.append(
            f"Y_next_{sp}"
        )

    if feature_info.get("output_parameterization", "direct_rate") in ("qssa", "qssa_asymptotic"):
        for sp in feature_info["macro_species"] + feature_info["micro_species"]:
            columns_needed.append(f"P_avg_{sp}")
            columns_needed.append(f"D_coef_avg_{sp}")
    elif feature_info.get("output_parameterization") in ("dual_branch", "dual_branch_pd_v1"):
        for sp in feature_info["macro_species"]:
            columns_needed.append(f"dYdt_{sp}")
        for sp in feature_info["micro_species"]:
            columns_needed.append(f"P_avg_{sp}")
            columns_needed.append(f"D_coef_avg_{sp}")
    else:
        for sp in feature_info["macro_species"]:
            columns_needed.append(f"dYdt_{sp}")
        for sp in feature_info["micro_species"]:
            columns_needed.extend((f"dYdt_{sp}", f"Y_{sp}", f"Y_next_{sp}"))
        if feature_info["micro_species"]:
            columns_needed.append("dt")

    for p in selected:

        if processed >= max_rows:
            break

        read_cols = list(dict.fromkeys(["sample_id", *columns_needed])) if sample_ids is not None else list(dict.fromkeys(columns_needed))
        df = pd.read_parquet(p, columns=read_cols)
        if sample_ids is not None:
            df = df[df["sample_id"].astype(str).isin(sample_ids)].drop(columns=["sample_id"])

        if sample_rate < 1.0:
            df = df.sample(
                frac=sample_rate,
                random_state=SEED,
            )

        remaining = (
            max_rows - processed
        )

        if len(df) > remaining:
            df = df.iloc[
                :remaining
            ]

        if len(df) == 0:
            continue

        X = transform_input_array(
            df,
            feature_info[
                "inert_species"
            ],
            feature_info[
                "macro_species"
            ],
            feature_info[
                "micro_species"
            ],
        )

        Y = transform_target_array(
            df,
            feature_info[
                "macro_species"
            ],
            feature_info[
                "micro_species"
            ],
            macro_rate_scale,
            output_parameterization=feature_info.get("output_parameterization", "direct_rate"),
        )

        moments_X.update(X)
        moments_Y.update(Y)

        processed += len(df)

        print(
            f"[Scaler] "
            f"processed={processed:,}"
        )

        del df
        del X
        del Y

        gc.collect()

    scaler_X = moments_X.finalize()
    scaler_Y = moments_Y.finalize()

    print("\n[Scaler Summary]")

    print(
        "X mean range:",
        f"{scaler_X.mean_.min():.3e}",
        "to",
        f"{scaler_X.mean_.max():.3e}",
    )

    print(
        "X std range:",
        f"{scaler_X.scale_.min():.3e}",
        "to",
        f"{scaler_X.scale_.max():.3e}",
    )

    print(
        "Y mean range:",
        f"{scaler_Y.mean_.min():.3e}",
        "to",
        f"{scaler_Y.mean_.max():.3e}",
    )

    print(
        "Y std range:",
        f"{scaler_Y.scale_.min():.3e}",
        "to",
        f"{scaler_Y.scale_.max():.3e}",
    )

    return scaler_X, scaler_Y


# ============================================================
# 8. Streaming dataset
# ============================================================

class CoarseOperatorDataset(
    IterableDataset
):

    def __init__(
        self,
        parquet_files,
        feature_info,
        scaler_X,
        scaler_Y,
        max_horizon=128,  # V68: 最大 horizon，替代 unroll_steps
        shuffle_parts=True,
        stride=None,
        seed=42,
        parts_per_epoch=None,
        sample_ids=None,
        include_metadata=False,
    ):
        super().__init__()

        self.parquet_files = [
            Path(p)
            for p in parquet_files
        ]

        self.feature_info = (
            feature_info
        )

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        # V68: 使用 max_horizon 作为窗口大小
        self.max_horizon = max_horizon
        self.unroll_steps = max_horizon  # 向后兼容

        self.shuffle_parts = (
            shuffle_parts
        )

        self.stride = (
            stride
            if stride is not None
            else max(
                1,
                max_horizon // 2,
            )
        )

        self.seed = seed
        self.epoch = 0

        self.parts_per_epoch = (
            parts_per_epoch
            if parts_per_epoch is not None
            else len(self.parquet_files)
        )
        self.sample_ids = None if sample_ids is None else {str(x) for x in sample_ids}
        self.include_metadata = include_metadata

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):

        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        # Select once conceptually for the epoch, then shard disjointly by worker.
        selected_files = select_epoch_files(
            self.parquet_files,
            self.parts_per_epoch,
            self.seed,
            self.epoch,
            self.shuffle_parts,
        )
        selected_files = partition_files_for_worker(
            selected_files,
            worker_id,
            num_workers,
        )
        rng = np.random.default_rng(
            self.seed + 1009 * self.epoch + worker_id
        )

        # ✅ 新增：Window Shuffle Buffer
        # 解决同一轨迹的连续窗口导致 batch 内动力学 regime 单一的问题
        buffer = []
        buffer_size = 2048  # 缓冲 2048 个窗口

        for part in selected_files:
            for window in self._process_part(part):
                buffer.append(window)

                # 当缓冲区满时，随机打乱并输出一半
                if len(buffer) >= buffer_size:
                    if self.shuffle_parts:
                        rng.shuffle(buffer)
                    # 输出前一半
                    half = buffer_size // 2
                    to_yield = buffer[:half]
                    buffer = buffer[half:]
                    for window in to_yield:
                        yield window

        # 输出剩余窗口
        if len(buffer) > 0:
            if self.shuffle_parts:
                rng.shuffle(buffer)
            for window in buffer:
                yield window

    def _process_part(
        self,
        part_path: Path,
    ):

        columns = [
            "sample_id",
            "time",
            "T",
            "P",
            "T_next",
            "P_next",
            "dt",
        ]

        for sp in (
            self.feature_info[
                "inert_species"
            ]
            + self.feature_info[
                "macro_species"
            ]
            + self.feature_info[
                "micro_species"
            ]
        ):
            columns.append(
                f"Y_{sp}"
            )

            columns.append(
                f"Y_next_{sp}"
            )

        for sp in self.feature_info["macro_species"] + self.feature_info["micro_species"]:
            if self.feature_info.get("output_parameterization", "direct_rate") in ("qssa", "qssa_asymptotic"):
                columns.append(f"P_avg_{sp}")
                columns.append(f"D_coef_avg_{sp}")
            elif self.feature_info.get("output_parameterization") in ("dual_branch", "dual_branch_pd_v1"):
                if sp in self.feature_info["macro_species"]:
                    columns.append(f"dYdt_{sp}")
                else:
                    columns.append(f"P_avg_{sp}")
                    columns.append(f"D_coef_avg_{sp}")
            else:
                columns.append(f"dYdt_{sp}")
                if sp in self.feature_info["micro_species"]:
                    columns.append(f"Y_next_{sp}")

        selected_columns = list(dict.fromkeys(columns))
        try:
            df = pd.read_parquet(part_path, columns=selected_columns)
        except Exception as exc:
            # Older trajectory parts may not contain the optional endpoint
            # temperature/pressure columns.  The species endpoint remains
            # usable; fall back to the source T/P for that terminal row.
            legacy_columns = [
                column
                for column in selected_columns
                if column not in {"T_next", "P_next"}
            ]
            try:
                df = pd.read_parquet(part_path, columns=legacy_columns)
            except Exception:
                raise exc

        if df.empty:
            return

        for sample_id, group in df.groupby(
            "sample_id",
            sort=False,
        ):
            if self.sample_ids is not None and str(sample_id) not in self.sample_ids:
                continue

            group = (
                group.sort_values(
                    "time"
                )
                .reset_index(
                    drop=True
                )
            )

            n = len(group)

            # A trajectory with exactly max_horizon transitions is still a
            # valid single window once its terminal state is appended below.
            if n < self.max_horizon:
                continue

            X = transform_input_array(
                group,
                self.feature_info[
                    "inert_species"
                ],
                self.feature_info[
                    "macro_species"
                ],
                self.feature_info[
                    "micro_species"
                ],
            )

            # Each parquet row stores the transition Y(t)->Y_next(t).  The
            # input transform above contains only the n source states, so add
            # the final endpoint explicitly to make the last transition
            # available to a (H+1)-state window.  The endpoint dt is not used
            # for another transition; carrying the last dt keeps the input
            # schema finite and consistent.
            terminal = group.iloc[[-1]].copy()
            if "T_next" in terminal:
                terminal.loc[:, "T"] = terminal["T_next"].to_numpy()
            if "P_next" in terminal:
                terminal.loc[:, "P"] = terminal["P_next"].to_numpy()
            for sp in (
                self.feature_info["inert_species"]
                + self.feature_info["macro_species"]
                + self.feature_info["micro_species"]
            ):
                next_col = f"Y_next_{sp}"
                if next_col in terminal:
                    terminal.loc[:, f"Y_{sp}"] = terminal[next_col].to_numpy()
            X_terminal = transform_input_array(
                terminal,
                self.feature_info["inert_species"],
                self.feature_info["macro_species"],
                self.feature_info["micro_species"],
            )
            X = np.concatenate([X, X_terminal], axis=0)

            Y = transform_target_array(
                group,
                self.feature_info[
                    "macro_species"
                ],
                self.feature_info[
                    "micro_species"
                ],
                self.feature_info[
                    "macro_rate_scale"
                ],
                output_parameterization=self.feature_info.get("output_parameterization", "direct_rate"),
            )

            X_scaled = self.scaler_X.transform(
                X
            ).astype(
                np.float32
            )

            Y_scaled = self.scaler_Y.transform(
                Y
            ).astype(
                np.float32
            )

            # V68: 窗口包含 (max_horizon + 1) 个状态点，对应 max_horizon 个 transition
            #   X: t0 ... t_max_horizon
            #   Y: r(t0->t1) ... r(t_{H-1}->t_H)
            # There are n transitions and n+1 state points after appending
            # the terminal endpoint, so the final legal start is n-H.
            max_start = n - self.max_horizon + 1

            for start in range(
                0,
                max_start,
                self.stride,
            ):
                # V68: 使用 make_window 辅助函数生成窗口
                x_window, y_window = make_window(
                    X_scaled,
                    Y_scaled,
                    start,
                    self.max_horizon,
                )

                # 防御式检查
                if (
                    len(x_window) != self.max_horizon + 1
                    or len(y_window) != self.max_horizon
                ):
                    continue

                window = (
                    torch.from_numpy(
                        x_window
                    ),
                    torch.from_numpy(
                        y_window
                    ),
                )
                if self.include_metadata:
                    yield (*window, {"sample_id": str(sample_id), "window_start": start,
                                     "part": str(part_path)})
                else:
                    yield window

        del df
        gc.collect()


# ============================================================
# 9. Network
# ============================================================

class CoupledKineticsNet(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        output_dim,
        output_parameterization: str = "direct_rate",
    ):
        super().__init__()
        if output_parameterization not in {"direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"}:
            raise ValueError(f"invalid output_parameterization: {output_parameterization!r}")
        self.output_parameterization = output_parameterization

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                1024,
            ),

            nn.GELU(),

            nn.Linear(
                1024,
                512,
            ),

            nn.GELU(),

            nn.Linear(
                512,
                512,
            ),

            nn.GELU(),

            nn.Linear(
                512,
                256,
            ),

            nn.GELU(),

            nn.Linear(
                256,
                output_dim,
            ),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 10. Element matrix
# ============================================================

def parse_formula(
    name: str,
):

    result = {}

    # Cantera accepts noble-gas species in either canonical or all-caps form.
    # Treat these names before the generic element tokenizer so ``AR`` is not
    # split into the fictitious elements ``A`` and ``R``.
    special_formula = {"AR": {"Ar": 1}, "HE": {"He": 1}}
    special = special_formula.get(str(name).upper())
    if special is not None:
        return special.copy()

    # Cantera species names may use C-/T- as cis/trans configuration
    # prefixes (for example C-HNN(O)OH); they are not carbon atoms.
    normalized_name = re.sub(
        r"^(?:C-|T-|cis-|trans-)",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # In this mechanism family a terminal V is Cantera's vibrational
    # excitation marker (OHV, O2V, NH3V), not elemental vanadium.
    normalized_name = re.sub(r"(?<=[A-Za-z0-9)])V$", "", normalized_name)

    tokens = re.findall(
        r"([A-Z][a-z]?)(\d*)",
        normalized_name,
    )

    unknown = sorted({elem for elem, _ in tokens if elem not in AW})
    if unknown:
        raise ValueError(
            f"Unsupported element(s) {unknown} in species {name!r}; "
            f"extend AW before training this mechanism"
        )

    for elem, count in tokens:

        n = (
            int(count)
            if count
            else 1
        )

        result[elem] = (
            result.get(
                elem,
                0,
            )
            + n
        )

    return result


def build_element_matrix(
    output_species,
    device: torch.device | str | None = None,
    mechanism_path: str | Path | None = None,
):
    """Build the matrix from the actual mechanism; explicit failures are errors."""
    import chemistry_contract as cc
    path = Path(mechanism_path) if mechanism_path is not None else cc.resolve_mechanism_file()
    return cc.get_mechanism_contract(
        path, output_species, device=DEVICE if device is None else device
    )["element_matrix_tensor"]


# ============================================================
# 11. State reconstruction
# ============================================================

class CoarseOperatorTools:

    def __init__(
        self,
        feature_info,
        scaler_X,
        scaler_Y,
        rate_decode_mode: str | None = None,
        rate_z_limit: float | None = None,
        micro_rate_z_limit: float | None = None,
        device: torch.device | str | None = None,
    ):

        self.device = DEVICE if device is None else torch.device(device)

        self.feature_info = (
            feature_info
        )

        self.scale_X = torch.tensor(
            scaler_X.scale_,
            dtype=torch.float32,
            device=self.device,
        )

        self.mean_X = torch.tensor(
            scaler_X.mean_,
            dtype=torch.float32,
            device=self.device,
        )

        self.scale_Y = torch.tensor(
            scaler_Y.scale_,
            dtype=torch.float32,
            device=self.device,
        )

        self.mean_Y = torch.tensor(
            scaler_Y.mean_,
            dtype=torch.float32,
            device=self.device,
        )

        if feature_info.get("version") != VERSION:
            raise ValueError(
                f"Expected feature_info version {VERSION}, "
                f"got {feature_info.get('version')!r}"
            )
        expected_signature = transform_signature_for(feature_info.get("output_parameterization", "direct_rate"))
        if feature_info.get("transform_signature") != expected_signature:
            raise ValueError(
                f"Incompatible V68 transform signature: expected {expected_signature!r}, "
                f"got {feature_info.get('transform_signature')!r}"
            )
        self.macro_rate_scale = torch.tensor(
            feature_info["macro_rate_scale"],
            dtype=torch.float32,
            device=self.device,
        )
        metadata_rate_decode_mode = feature_info.get("rate_decode_mode")
        if (
            rate_decode_mode is not None
            and metadata_rate_decode_mode is not None
            and rate_decode_mode != metadata_rate_decode_mode
        ):
            raise ValueError(
                "rate_decode_mode disagrees with feature_info metadata"
            )
        if rate_decode_mode is None:
            self.rate_decode_mode = metadata_rate_decode_mode or "direct_rate"
        else:
            self.rate_decode_mode = rate_decode_mode
        metadata_rate_z_limit = feature_info.get("rate_z_limit")
        if (
            rate_z_limit is not None
            and metadata_rate_z_limit is not None
            and not np.isclose(float(rate_z_limit), float(metadata_rate_z_limit))
        ):
            raise ValueError("rate_z_limit disagrees with feature_info metadata")
        self.rate_z_limit = float(
            rate_z_limit
            if rate_z_limit is not None
            else feature_info.get("rate_z_limit", DEFAULT_RATE_Z_LIMIT)
        )
        if self.rate_decode_mode not in RATE_DECODE_MODES:
            raise ValueError(
                f"invalid rate_decode_mode: {self.rate_decode_mode!r}"
            )
        if not np.isfinite(self.rate_z_limit) or self.rate_z_limit <= 0.0:
            raise ValueError("rate_z_limit must be a finite positive value")
        metadata_micro_rate_z_limit = feature_info.get("micro_rate_z_limit")
        if (
            micro_rate_z_limit is not None
            and metadata_micro_rate_z_limit is not None
            and not np.isclose(float(micro_rate_z_limit), float(metadata_micro_rate_z_limit))
        ):
            raise ValueError("micro_rate_z_limit disagrees with feature_info metadata")
        if micro_rate_z_limit is not None:
            self.micro_rate_z_limit = float(micro_rate_z_limit)
        elif metadata_micro_rate_z_limit is not None:
            self.micro_rate_z_limit = float(metadata_micro_rate_z_limit)
        else:
            self.micro_rate_z_limit = None
        if self.micro_rate_z_limit is not None:
            if not np.isfinite(self.micro_rate_z_limit) or self.micro_rate_z_limit <= 0.0:
                raise ValueError("micro_rate_z_limit must be a finite positive value")

        input_cols = (
            feature_info[
                "input_cols"
            ]
        )

        self.macro_species = (
            feature_info[
                "macro_species"
            ]
        )

        self.micro_species = (
            feature_info[
                "micro_species"
            ]
        )

        # X 中的 species feature indexes
        macro_idx = [
            input_cols.index(
                f"Y_{sp}"
            )
            for sp in self.macro_species
        ]

        micro_idx = [
            input_cols.index(
                f"logY_{sp}"
            )
            for sp in self.micro_species
        ]

        self.macro_idx = torch.tensor(
            macro_idx,
            dtype=torch.long,
            device=self.device,
        )

        self.micro_idx = torch.tensor(
            micro_idx,
            dtype=torch.long,
            device=self.device,
        )

        self.dt_idx = input_cols.index(
            "log_dt"
        )
        self.temperature_idx = input_cols.index("T")

        # output dimension:
        # macro rate + micro log-rate
        self.n_macro = len(
            self.macro_species
        )

        self.n_micro = len(
            self.micro_species
        )
        self.n_reactive = self.n_macro + self.n_micro

        if "element_matrix" in feature_info and feature_info["element_matrix"] is not None:
            elem_mat = feature_info["element_matrix"]
            if isinstance(elem_mat, torch.Tensor):
                self.A = elem_mat.to(device=self.device, dtype=torch.float32)
            else:
                self.A = torch.tensor(elem_mat, dtype=torch.float32, device=self.device)
        else:
            self.A = build_element_matrix(
                self.macro_species
                + self.micro_species,
                device=self.device,
            )


        # Constant pseudo-inverse used by the affine conservation projection.
        A_cpu = self.A.cpu()
        self.A_gram_pinv = torch.linalg.pinv(A_cpu @ A_cpu.T).to(self.device)

        # inert species input indexes
        self.inert_idx = torch.tensor(
            [
                input_cols.index(
                    f"Y_{sp}"
                )
                for sp in feature_info[
                    "inert_species"
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

    def inverse_qssa_parameters(self, rate_scaled):
        """Decode standardized interleaved QSSA ``(log10(P), log10(D))`` targets."""
        target_denorm = (
            rate_scaled
            * self.scale_Y
            + self.mean_Y
        )

        target_dim = target_denorm.shape[-1]
        expected_direct_dim = getattr(self, "n_reactive", None)
        if expected_direct_dim is None:
            expected_direct_dim = self.n_macro + self.n_micro
        expected_qssa_dim = 2 * expected_direct_dim
        if self.rate_decode_mode != "qssa":
            raise ValueError(
                f"inverse_qssa_parameters requires rate_decode_mode='qssa', got {self.rate_decode_mode!r}"
            )
        if target_dim != expected_qssa_dim:
            raise ValueError(
                f"qssa parameter output has dimension {target_dim}; expected {expected_qssa_dim}"
            )
        logP = torch.clamp(target_denorm[..., 0::2], -40.0, 30.0)
        logD = torch.clamp(target_denorm[..., 1::2], -40.0, 30.0)
        return torch.pow(10.0, logP), torch.pow(10.0, logD)

    def inverse_qssa_asymptotic_log_parameters(self, rate_scaled):
        """Decode standardized QSSA asymptotic targets into log10(P) and log10(D) in log-space.

        Guarantees exact cancellation without linear float32 overflow (inf),
        and handles pure production (D=0) losslessly via the z2 < -40.0 sentinel.
        """
        target_denorm = (
            rate_scaled
            * self.scale_Y
            + self.mean_Y
        )
        z1 = target_denorm[..., 0::2]
        z2 = target_denorm[..., 1::2]
        is_prod_only = z2 < -40.0
        is_zero_creation = z1 < -40.0

        logD_normal = torch.clamp(z2, -40.0, 30.0)
        logP_normal = torch.clamp(torch.clamp(z1, -40.0, 30.0) + logD_normal, -40.0, 30.0)

        logP_prod = torch.clamp(z1, -40.0, 30.0)
        logD_prod = torch.full_like(z2, -45.0)

        logP = torch.where(is_zero_creation, torch.full_like(z1, -45.0), torch.where(is_prod_only, logP_prod, logP_normal))
        logD = torch.where(is_prod_only, logD_prod, logD_normal)
        return logP, logD

    def inverse_qssa_asymptotic_parameters(self, rate_scaled):
        """Decode standardized interleaved QSSA asymptotic ``(log10(Y_qssa), log10(D))`` targets."""
        target_denorm = (
            rate_scaled
            * self.scale_Y
            + self.mean_Y
        )

        target_dim = target_denorm.shape[-1]
        expected_direct_dim = getattr(self, "n_reactive", None)
        if expected_direct_dim is None:
            expected_direct_dim = self.n_macro + self.n_micro
        expected_qssa_dim = 2 * expected_direct_dim
        if self.rate_decode_mode != "qssa_asymptotic":
            raise ValueError(
                f"inverse_qssa_asymptotic_parameters requires rate_decode_mode='qssa_asymptotic', got {self.rate_decode_mode!r}"
            )
        if target_dim != expected_qssa_dim:
            raise ValueError(
                f"qssa asymptotic output has dimension {target_dim}; expected {expected_qssa_dim}"
            )
        z1 = target_denorm[..., 0::2]
        z2 = target_denorm[..., 1::2]
        is_prod_only = z2 < -40.0
        is_zero_creation = z1 < -40.0

        logY_qssa = torch.clamp(z1, -40.0, 30.0)
        logD = torch.where(is_prod_only, torch.full_like(z2, -45.0), torch.clamp(z2, -40.0, 30.0))
        D = torch.where(is_prod_only, torch.zeros_like(logD), torch.pow(10.0, logD))
        Y_qssa = torch.where(is_zero_creation, torch.zeros_like(logY_qssa), torch.pow(10.0, logY_qssa))
        return Y_qssa, D

    def inverse_rate(
        self,
        rate_scaled,
        bound_macro: bool = True,
        bound_micro: bool | None = None,
    ):
        """Invert the standardized direct-rate target representation."""
        if self.rate_decode_mode in ("qssa", "qssa_asymptotic"):
            raise ValueError(
                "inverse_rate is undefined for QSSA targets; "
                "use inverse_qssa_parameters or inverse_qssa_asymptotic_parameters instead"
            )

        target_denorm = (
            rate_scaled
            * self.scale_Y
            + self.mean_Y
        )

        # Support both one-step tensors ``(B, D)`` and rollout tensors
        # ``(B, H, D)`` by slicing the feature axis explicitly.  The decode
        # mode is authoritative: a direct-rate vector must never be inferred
        # as QSSA merely because its width happens to match another shape.
        target_dim = target_denorm.shape[-1]
        expected_direct_dim = getattr(self, "n_reactive", None)
        if expected_direct_dim is None:
            expected_direct_dim = self.n_macro + self.n_micro

        if target_dim != expected_direct_dim:
            raise ValueError(
                f"{self.rate_decode_mode} rate output has dimension {target_dim}; expected {expected_direct_dim}"
            )

        macro_z = target_denorm[..., :self.n_macro]
        if not torch.isfinite(macro_z).all():
            raise FloatingPointError("Non-finite macro asinh prediction")

        if bound_macro and self.rate_decode_mode == "bounded_direct_rate":
            macro_z = self.rate_z_limit * torch.tanh(
                macro_z / self.rate_z_limit
            )
        else:
            macro_z = torch.clamp(macro_z, -20.0, 20.0)
        # A fully initialized operator follows the asinh target contract.  A
        # small compatibility fallback keeps lightweight test/dummy tools that
        # only provide tensors usable as direct physical-rate decoders.
        if hasattr(self, "feature_info"):
            rate_macro = self.macro_rate_scale * torch.sinh(macro_z)
        else:
            rate_macro = macro_z
        rate_micro_log = target_denorm[..., self.n_macro:]
        effective_bound_micro = bound_macro if bound_micro is None else bound_micro
        if effective_bound_micro and self.micro_rate_z_limit is not None:
            micro_z = rate_scaled[..., self.n_macro:]
            micro_z_bounded = self.micro_rate_z_limit * torch.tanh(
                micro_z / self.micro_rate_z_limit
            )
            rate_micro_log = (
                micro_z_bounded * self.scale_Y[self.n_macro:] + self.mean_Y[self.n_macro:]
            )

        return torch.cat([rate_macro, rate_micro_log], dim=-1)

    def inverse_state(
        self,
        state_scaled,
    ):
        return (
            state_scaled
            * self.scale_X
            + self.mean_X
        )

    def get_current_state(
        self,
        current_X_scaled,
    ):

        current_X = (
            self.inverse_state(
                current_X_scaled
            )
        )

        Y_macro = current_X[
            :,
            self.macro_idx,
        ]

        logY_micro = current_X[
            :,
            self.micro_idx,
        ]

        Y_micro = torch.pow(
            10.0,
            torch.clamp(
                logY_micro,
                MICRO_LOG_MIN,
                MICRO_LOG_MAX,
            ),
        )

        return (
            current_X,
            Y_macro,
            logY_micro,
            Y_micro,
        )

    def get_current_dt(
        self,
        current_X_scaled,
    ):

        current_X = (
            self.inverse_state(
                current_X_scaled
            )
        )

        dt_log = current_X[
            :,
            self.dt_idx,
        ]

        dt = sanitize_dt_tensor(
            torch.pow(10.0, dt_log)
        ).unsqueeze(1)

        return dt

    def project_conservation(
        self,
        Y_old_full,
        Y_candidate_full,
        iterations=8,
    ):
        """Project concentrations onto nonnegative elemental conservation with concentration weighting."""
        target_elements = Y_old_full @ self.A.T
        projected = Y_candidate_full.clone()
        K = self.A.shape[0]
        reg = 1e-10 * torch.eye(K, dtype=projected.dtype, device=projected.device).unsqueeze(0)
        n_proj_iters = min(max(iterations, 1), 3) if iterations > 0 else 2
        for _ in range(n_proj_iters):
            r = target_elements - torch.matmul(projected, self.A.T)
            W = torch.clamp(projected, min=1e-40)
            A_w = self.A.unsqueeze(0) * W.unsqueeze(1)
            M_gram = torch.matmul(A_w, self.A.T)
            try:
                lam = torch.linalg.solve((M_gram + reg).cpu(), r.unsqueeze(-1).cpu()).squeeze(-1).to(projected.device)
            except Exception:
                lam = torch.matmul(torch.linalg.pinv((M_gram + reg).cpu()).to(projected.device), r.unsqueeze(-1)).squeeze(-1)
            dY = W * torch.matmul(lam, self.A)
            projected = torch.clamp(projected + dY, min=0.0)

        return projected

    def predict_next_state(
        self,
        current_X_scaled,
        rate_pred_scaled,
        safe_for_rollout=True,
        experiment_mode="baseline",
        trust_region_fraction=0.0,
        delta_limit=None,
        state_update_mode="qssa",
        inert_sum=None,
        element_matrix=None,
        target_elements=None,
        iterations=8,
        projection_mode=None,
    ):
        eff_proj_mode = projection_mode if projection_mode is not None else getattr(self, "projection_mode", "multiplicative_kl")
        if state_update_mode in ("task8_conservative_state", "constrained_delta"):
            return apply_task8_state_update(
                tools=self,
                current_X_scaled=current_X_scaled,
                rate_pred_scaled=rate_pred_scaled,
                inert_sum=inert_sum if inert_sum is not None else 0.0,
                element_matrix=element_matrix,
                target_elements=target_elements,
                iterations=iterations,
                projection_mode=eff_proj_mode,
            )

        (
            current_X,
            Y_macro_old,
            logY_micro_old,
            _,
        ) = self.get_current_state(current_X_scaled)

        dt = self.get_current_dt(current_X_scaled)
        
        if state_update_mode in ("qssa", "qssa_asymptotic") and not safe_for_rollout:
            # Raw QSSA rollouts must preserve a previous unconstrained state
            # such as Y > 1; only the lower numerical floor is applied.
            Y_micro_old = torch.pow(10.0, torch.clamp(logY_micro_old, min=MICRO_LOG_MIN))
        else:
            Y_micro_old = torch.pow(10.0, torch.clamp(logY_micro_old, MICRO_LOG_MIN, 0.0))
        Y_old = torch.cat([Y_macro_old, Y_micro_old], dim=1)
        logY_micro_next_candidate = None

        if state_update_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1") or self.rate_decode_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"):
            if state_update_mode == "dual_branch_pd_v1" or self.rate_decode_mode == "dual_branch_pd_v1":
                # 1. Macro species (first n_macro channels): direct asinh rate
                macro_pred_scaled = rate_pred_scaled[..., :self.n_macro]
                macro_denorm = macro_pred_scaled * self.scale_Y[:self.n_macro] + self.mean_Y[:self.n_macro]
                macro_z = torch.clamp(macro_denorm, -20.0, 20.0)
                rate_macro_cand = self.macro_rate_scale * torch.sinh(macro_z)
                # Genuine unconstrained candidate: do not clamp at 0.0 to preserve non-zero gradients for backpropagation
                Y_macro_next = Y_macro_old + rate_macro_cand * dt

                # 2. Micro species (remaining 2 * n_micro channels): continuous log10(P + eps_P), log10(D + eps_D)
                micro_pred_scaled = rate_pred_scaled[..., self.n_macro:]
                micro_denorm = micro_pred_scaled * self.scale_Y[self.n_macro:] + self.mean_Y[self.n_macro:]
                z_P = micro_denorm[..., 0::2]
                z_D = micro_denorm[..., 1::2]
                P_pred, D_pred = kp.decode_pd_outputs(z_P, z_D)
                Y_micro_next, logY_micro_next_candidate = kp.pd_v1_micro_log_update(logY_micro_old, z_P, z_D, dt)

                Y_next_reactive = torch.cat([Y_macro_next, Y_micro_next], dim=-1)
                rate_macro = rate_macro_cand
                rate_micro_log = (
                    logY_micro_next_candidate - logY_micro_old
                ) / dt.clamp_min(1e-12)
                rate_raw = rate_pred_scaled
            elif state_update_mode == "dual_branch" or self.rate_decode_mode == "dual_branch":
                # 1. Macro species (first n_macro channels): direct asinh rate
                macro_pred_scaled = rate_pred_scaled[..., :self.n_macro]
                macro_denorm = macro_pred_scaled * self.scale_Y[:self.n_macro] + self.mean_Y[:self.n_macro]
                macro_z = torch.clamp(macro_denorm, -20.0, 20.0)
                rate_macro_cand = self.macro_rate_scale * torch.sinh(macro_z)
                # Genuine unconstrained candidate: do not clamp at 0.0 to preserve non-zero gradients for backpropagation
                Y_macro_next = Y_macro_old + rate_macro_cand * dt

                # 2. Micro species (remaining 2 * n_micro channels): QSSA asymptotic
                micro_pred_scaled = rate_pred_scaled[..., self.n_macro:]
                micro_denorm = micro_pred_scaled * self.scale_Y[self.n_macro:] + self.mean_Y[self.n_macro:]
                z1 = micro_denorm[..., 0::2]
                z2 = micro_denorm[..., 1::2]
                is_prod_only = z2 < -40.0
                is_zero_creation = z1 < -40.0
                logD = torch.where(is_prod_only, torch.full_like(z2, -45.0), torch.clamp(z2, -40.0, 30.0))
                logP = torch.where(is_prod_only, z1, z1 + z2)
                logP = torch.clamp(logP, -40.0, 30.0)

                P_pred = torch.where(is_zero_creation, torch.zeros_like(logP), torch.pow(10.0, logP))
                D_pred = torch.where(is_prod_only, torch.zeros_like(logD), torch.pow(10.0, logD))
                logY_micro_next_candidate = dual_branch_micro_log_update(
                    log_y_old=logY_micro_old,
                    z1=z1,
                    z2=z2,
                    dt=dt,
                )
                Y_micro_next = torch.pow(10.0, logY_micro_next_candidate)

                Y_next_reactive = torch.cat([Y_macro_next, Y_micro_next], dim=-1)
                rate_macro = rate_macro_cand
                rate_micro_log = (
                    logY_micro_next_candidate - logY_micro_old
                ) / dt.clamp_min(1e-12)
                rate_raw = rate_pred_scaled
            elif state_update_mode == "qssa_asymptotic" or self.rate_decode_mode == "qssa_asymptotic":
                logP, logD = self.inverse_qssa_asymptotic_log_parameters(rate_pred_scaled)
                P_pred = torch.where(logP < -40.0, torch.zeros_like(logP), torch.pow(10.0, torch.clamp(logP, -40.0, 30.0)))
                is_prod_only = logD < -40.0
                D_pred = torch.where(is_prod_only, torch.zeros_like(logD), torch.pow(10.0, torch.clamp(logD, -40.0, 30.0)))
                D_eff = torch.clamp_min(D_pred, 0.0)
                D_dt = D_eff * dt

                # Prevent autograd NaN: sanitize inputs to large-D branch before evaluating division
                is_small_D = D_dt < 1e-6
                D_safe_large = torch.where(is_small_D, torch.ones_like(D_eff), D_eff)
                D_dt_large = D_safe_large * dt
                D_dt_bounded = torch.clamp(D_dt_large, max=80.0)

                prod_term_small = P_pred * dt * (1.0 - 0.5 * D_dt)
                prod_term_large = (P_pred / D_safe_large) * (1.0 - torch.exp(-D_dt_bounded))
                prod_term = torch.where(is_small_D, prod_term_small, prod_term_large)

                exp_neg_Ddt = torch.where(is_prod_only, torch.ones_like(D_eff), torch.exp(-torch.clamp(D_dt, max=80.0)))
                Y_next_reactive = Y_old * exp_neg_Ddt + prod_term
            else:
                if self.rate_decode_mode != "qssa":
                    raise ValueError(f"state_update_mode='qssa' requires rate_decode_mode='qssa', got {self.rate_decode_mode!r}")
                P_pred, D_pred = self.inverse_qssa_parameters(rate_pred_scaled)
                # Pure exact exponential QSSA integration without artificial max_damping clamp
                D_eff = torch.clamp_min(D_pred, 0.0)
                D_dt = D_eff * dt
                D_dt_bounded = torch.clamp(D_dt, max=80.0)
                exp_neg_Ddt = torch.exp(-D_dt_bounded)
                safe_factor = torch.where(
                    D_dt < 1e-6,
                    dt - D_eff * (dt ** 2) / 2.0,
                    -torch.expm1(-D_dt_bounded) / torch.clamp_min(D_eff, LOG_EPS),
                )
                Y_next_reactive = Y_old * exp_neg_Ddt + P_pred * safe_factor

            # Defensively guard against astronomical unconstrained exploration spikes
            # in raw intermediate states (e.g. > 10.0 kg/kg mass fraction)
            Y_macro_next = torch.clamp(Y_next_reactive[:, :self.n_macro], min=-1.0, max=10.0)
            if logY_micro_next_candidate is None:
                Y_micro_next = torch.clamp(
                    Y_next_reactive[:, self.n_macro:],
                    min=LOG_EPS,
                    max=10.0,
                )
                logY_micro_next_raw = torch.log10(
                    torch.clamp(Y_micro_next, min=LOG_EPS)
                )
            else:
                logY_micro_next_raw = torch.clamp(
                    logY_micro_next_candidate,
                    min=MICRO_LOG_MIN,
                    max=1.0,
                )
                Y_micro_next = torch.pow(10.0, logY_micro_next_raw)
            rate_macro = (Y_macro_next - Y_macro_old) / dt.clamp_min(1e-12)
            rate_micro_log = (
                logY_micro_next_raw - logY_micro_old
            ) / dt.clamp_min(1e-12)
            if state_update_mode not in ("dual_branch", "dual_branch_pd_v1") and self.rate_decode_mode not in ("dual_branch", "dual_branch_pd_v1"):
                rate_raw = torch.cat([P_pred, D_pred], dim=1)
        else:
            if self.rate_decode_mode not in ("direct_rate", "bounded_direct_rate"):
                raise ValueError(
                    f"state_update_mode={state_update_mode!r} requires a direct-rate decode mode"
                )
            decoded_rate = self.inverse_rate(rate_pred_scaled)
            rate_macro = decoded_rate[:, :self.n_macro]
            rate_micro_log = decoded_rate[:, self.n_macro:]
            Y_macro_next = Y_macro_old + rate_macro * dt
            logY_micro_next = logY_micro_old + rate_micro_log * dt
            Y_micro_next = torch.pow(10.0, torch.clamp(logY_micro_next, MICRO_LOG_MIN, 0.0))
            P_pred = rate_macro
            D_pred = rate_micro_log
            rate_raw = decoded_rate
        
        # 1. Raw candidates (genuine unconstrained integration candidate)
        Y_macro_next_raw = Y_macro_next
        Y_micro_next_raw = Y_micro_next
        if logY_micro_next_candidate is None and not (
            state_update_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")
            or self.rate_decode_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")
        ):
            logY_micro_next_raw = torch.log10(
                torch.clamp(Y_micro_next_raw, min=LOG_EPS)
            )

        # Trust region limitation if enabled (> 0)
        if trust_region_fraction > 0.0:
            max_delta_macro = trust_region_fraction * torch.clamp_min(torch.abs(Y_macro_old), 1e-4)
            delta_macro = torch.clamp(Y_macro_next - Y_macro_old, -max_delta_macro, max_delta_macro)
            Y_macro_next_raw = torch.clamp_min(Y_macro_old + delta_macro, 0.0)

        # 2. Constrained candidates (Physical Simplex & Elemental Projection)
        if inert_sum is None:
            if hasattr(self, "inert_idx") and len(self.inert_idx) > 0:
                inert_idx_dev = self.inert_idx.to(current_X.device)
                inert_sum_eff = torch.clamp_min(current_X[:, inert_idx_dev], 0.0).sum(dim=-1, keepdim=True)
            else:
                inert_sum_eff = 0.0
        else:
            inert_sum_eff = inert_sum

        eff_element_matrix = element_matrix if element_matrix is not None else getattr(self, "A", None)
        if target_elements is None and eff_element_matrix is not None:
            dyn_old = torch.cat([Y_macro_old, Y_micro_old], dim=-1)
            eff_target_elements = torch.matmul(dyn_old, eff_element_matrix.to(dyn_old.device).T)
        else:
            eff_target_elements = target_elements

        with torch.no_grad():
            Y_macro_proj, Y_micro_proj, proj_diag = project_physical_state(
                Y_macro_raw=Y_macro_next_raw.detach(),
                Y_micro_raw=Y_micro_next_raw.detach(),
                inert_sum=inert_sum_eff,
                element_matrix=eff_element_matrix,
                target_elements=eff_target_elements,
                iterations=iterations,
                projection_mode=eff_proj_mode,
            )
        # Straight-Through Estimator (STE): forward value is exact physical projection,
        # backward gradient flows cleanly via identity d(Y_proj)/d(Y_raw) = I.
        Y_macro_next_constrained = Y_macro_next_raw + (Y_macro_proj - Y_macro_next_raw).detach()
        logY_micro_next_constrained, Y_micro_next_constrained = (
            constrained_micro_feedback(
                raw_log_micro=logY_micro_next_raw,
                projected_micro=Y_micro_proj,
            )
        )

        # Active state (used for feedback in rollout when safe_for_rollout=True)
        if safe_for_rollout:
            Y_macro_next_active = Y_macro_next_constrained
            Y_micro_next_active = Y_micro_next_constrained
            logY_micro_next_active = logY_micro_next_constrained
        else:
            Y_macro_next_active = torch.clamp(Y_macro_next_raw, min=0.0, max=1.0)
            if logY_micro_next_candidate is None:
                Y_micro_next_active = torch.clamp(Y_micro_next_raw, min=LOG_EPS, max=1.0)
                logY_micro_next_active = torch.log10(Y_micro_next_active)
            else:
                logY_micro_next_active = torch.clamp(
                    logY_micro_next_raw,
                    min=MICRO_LOG_MIN,
                    max=0.0,
                )
                Y_micro_next_active = torch.pow(10.0, logY_micro_next_active)

        # Rate calculations for downstream consumers (validators, diagnostics, metrics)
        dt_safe = dt.clamp_min(1e-12)
        rate_macro = (Y_macro_next_active - Y_macro_old) / dt_safe
        rate_micro_log = (logY_micro_next_active - logY_micro_old) / dt_safe

        return {
            "current_X": current_X,
            "dt": dt,
            "Y_macro_old": Y_macro_old,
            "logY_micro_old": logY_micro_old,
            
            "P_pred": P_pred,
            "D_pred": D_pred,
            "rate_raw": rate_raw,
            "rate_macro": rate_macro,
            "rate_micro_log": rate_micro_log,
            
            # Active outputs (honoring safe_for_rollout)
            "Y_macro_next": Y_macro_next_active,
            "logY_micro_next": logY_micro_next_active,
            "Y_micro_next": Y_micro_next_active,

            # Explicit constrained candidate
            "Y_macro_next_constrained": Y_macro_next_constrained,
            "logY_micro_next_constrained": logY_micro_next_constrained,
            "Y_micro_next_constrained": Y_micro_next_constrained,
            "Y_next_constrained_full": torch.cat([Y_macro_next_constrained, Y_micro_next_constrained], dim=-1),
            "Y_next_projected_full": torch.cat([Y_macro_next_constrained, Y_micro_next_constrained], dim=-1),

            # Explicit raw candidate (for unconstrained loss supervision in dual-track training)
            "Y_macro_next_raw": Y_macro_next_raw,
            "Y_macro_next_unconstrained": Y_macro_next_raw,
            "logY_micro_next_raw": logY_micro_next_raw,
            "Y_micro_next_raw": Y_micro_next_raw,
            "Y_micro_next_unconstrained": Y_micro_next_raw,
            "logY_micro_next_unconstrained": logY_micro_next_raw,
            "Y_next_raw_full": torch.cat([Y_macro_next_raw, Y_micro_next_raw], dim=-1),

            "Y_next_safe_full": torch.cat([Y_macro_next_active, Y_micro_next_active], dim=-1),
            "diagnostic": proj_diag,
            "experiment_mode": experiment_mode,
            "projection_element_error": None,
        }

    def state_to_input_scaled(
        self,
        base_next_X_scaled,
        Y_macro_next,
        logY_micro_next,
        temperature_mode="recorded",
        thermo_closure=None,
        current_X_scaled=None,
        max_delta_temperature=1000.0,
        feedback_input_limit: float | None = None,
    ):
        """
        使用：
            true T_next
            true P_next
            true dt_next

        但 species 使用预测值。
        """

        next_X = (
            base_next_X_scaled.clone()
        )

        current_unscaled = (
            self.inverse_state(
                next_X
            )
        )

        # ----------------------------------------------------
        # Macro
        # ----------------------------------------------------

        current_unscaled[
            :,
            self.macro_idx
        ] = Y_macro_next

        # ----------------------------------------------------
        # Micro
        # ----------------------------------------------------

        current_unscaled[
            :,
            self.micro_idx
        ] = logY_micro_next

        if temperature_mode == "autonomous":
            if thermo_closure is None or current_X_scaled is None:
                raise ValueError(
                    "autonomous temperature requires thermo_closure and current_X_scaled"
                )
            current_X, Y_macro_old, logY_micro_old, Y_micro_old = self.get_current_state(
                current_X_scaled
            )
            has_inert_in_closure = (
                thermo_closure is not None
                and "species_order" in thermo_closure
                and hasattr(self, "inert_idx")
                and len(self.inert_idx) > 0
                and len(thermo_closure["species_order"]) == (len(self.inert_idx) + self.n_macro + self.n_micro)
            )
            if has_inert_in_closure:
                Y_inert = current_X[:, self.inert_idx]
                Y_old_full = torch.cat([Y_inert, Y_macro_old, Y_micro_old], dim=1)
                Y_next_full = torch.cat(
                    [Y_inert, Y_macro_next, torch.pow(10.0, torch.clamp(logY_micro_next, MICRO_LOG_MIN, MICRO_LOG_MAX))],
                    dim=1,
                )
            else:
                Y_old_full = torch.cat([Y_macro_old, Y_micro_old], dim=1)
                Y_next_full = torch.cat(
                    [Y_macro_next, torch.pow(10.0, torch.clamp(logY_micro_next, MICRO_LOG_MIN, MICRO_LOG_MAX))],
                    dim=1,
                )
            delta_y = Y_next_full - Y_old_full
            heat_capacity_coeff = torch.as_tensor(
                thermo_closure["heat_capacity_coeff"],
                dtype=current_X_scaled.dtype,
                device=current_X_scaled.device,
            )
            heat_capacity = (
                Y_old_full * heat_capacity_coeff.reshape(1, -1)
            ).sum(dim=1, keepdim=True)
            current_temperature = self.inverse_state(current_X_scaled)[
                :, self.temperature_idx:self.temperature_idx + 1
            ]
            next_temperature = update_temperature_autonomous(
                current_temperature,
                delta_y,
                torch.as_tensor(
                    thermo_closure["energy_coeff"],
                    dtype=current_X_scaled.dtype,
                    device=current_X_scaled.device,
                ),
                heat_capacity,
                max_delta_temperature=max_delta_temperature,
                thermo_closure=thermo_closure,
                Y_old_full=Y_old_full,
                Y_next_full=Y_next_full,
            )
            current_unscaled[:, self.temperature_idx:self.temperature_idx + 1] = next_temperature

        next_X_scaled = (
            current_unscaled
            - self.mean_X
        ) / self.scale_X

        if feedback_input_limit is not None:
            limit = float(feedback_input_limit)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("feedback_input_limit must be finite and positive")
            species_idx = torch.cat([self.macro_idx, self.micro_idx])
            bounded_species = next_X_scaled.index_select(1, species_idx).clamp(
                -limit,
                limit,
            )
            next_X_scaled = next_X_scaled.index_copy(
                1,
                species_idx,
                bounded_species,
            )

        return next_X_scaled


# ============================================================
# 12. Physics soft loss
# ============================================================

def compute_element_conservation_loss(
    tools: CoarseOperatorTools,
    pred_state: dict,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
):
    """
    Evaluate element and mass conservation.
    In dual-track training (state_loss_source='raw'), evaluates directly on the
    model's raw unprojected predictions so the network is strictly penalized
    for off-manifold drift.
    """

    Y_old = torch.cat(
        [
            pred_state[
                "Y_macro_old"
            ],

            torch.pow(
                10.0,
                torch.clamp(
                    pred_state[
                        "logY_micro_old"
                    ],
                    MICRO_LOG_MIN,
                    MICRO_LOG_MAX,
                ),
            ),
        ],
        dim=1,
    )

    if state_loss_source == "constrained":
        Y_next = pred_state.get(
            "Y_next_constrained_full",
            pred_state["Y_next_safe_full"],
        )
    else:
        Y_next = pred_state.get(
            "Y_next_raw_full",
            pred_state["Y_next_safe_full"],
        )


    delta_Y = (
        Y_next - Y_old
    )

    element_delta = (
        delta_Y
        @ tools.A.T
    )

    # 归一化：
    # 不同元素质量储量尺度可能不同。
    b = (
        Y_old
        @ tools.A.T
    )

    scale = torch.clamp(
        torch.abs(b),
        min=1e-6,
    )

    normalized_error = (
        element_delta / scale
    )

    loss_element = torch.clamp(
        torch.mean(normalized_error**2),
        max=100.0,
    )

    # 总质量变化
    mass_delta = (
        delta_Y.sum(
            dim=1,
            keepdim=True,
        )
    )

    loss_mass = torch.clamp(
        torch.mean(mass_delta**2),
        max=100.0,
    )

    return (
        loss_element,
        loss_mass,
    )


def compute_positivity_loss(
    pred_state: dict,
):
    """
    只有 macro 需要额外正值约束。

    Micro 已经通过 log 表示天然为正。
    """

    negative_macro = F.relu(
        -pred_state[
            "Y_macro_next_raw"
        ]
    )

    return torch.mean(
        negative_macro**2
    )


# ============================================================
# 12.5. Rollout Loss（阶段五新增）
# ============================================================

def compute_rollout_loss(
    pred_states_at_horizons: dict,
    true_states_at_horizons: dict,
    tools: 'CoarseOperatorTools',
    target_horizons: list = None,
    horizon_weights: dict = None,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
):
    """
    计算多步 rollout 的状态误差。

    Args:
        pred_states_at_horizons: {h: pred_state_dict} - 各 horizon 的预测状态
        true_states_at_horizons: {h: true_X_scaled} - 各 horizon 的真实状态
        tools: CoarseOperatorTools 实例
        target_horizons: 需要计算 loss 的 horizon 列表，默认 [4, 8, 16, 32, 64, 128]
        horizon_weights: 各 horizon 的权重，默认后期horizon权重更大
        state_loss_source: 状态误差源 ('constrained' 或 'raw')

    Returns:
        rollout_loss: 加权平均的 rollout 状态误差
        rollout_breakdown: 各 horizon 的详细损失
    """
    if target_horizons is None:
        target_horizons = [4, 8, 16, 32, 64, 128]

    if horizon_weights is None:
        # 默认权重：H4/H8 次要，H16/H32 次之，H64/H128 最重要
        horizon_weights = {
            4: 0.1,
            8: 0.1,
            16: 0.15,
            32: 0.15,
            64: 0.2,
            128: 0.3,
        }

    rollout_breakdown = {}
    total_loss = 0.0
    total_weight = 0.0

    for h in target_horizons:
        if h not in pred_states_at_horizons or h not in true_states_at_horizons:
            continue

        pred_state = pred_states_at_horizons[h]
        true_X_scaled = true_states_at_horizons[h]

        pred_state_scaled = _call_state_features_scaled(
            state_features_scaled_from_prediction,
            pred_state,
            tools,
            state_loss_source=state_loss_source,
        )

        # 提取真实状态
        true_macro_scaled = true_X_scaled[:, tools.macro_idx]
        true_micro_scaled = true_X_scaled[:, tools.micro_idx]
        true_state_scaled = torch.cat([true_macro_scaled, true_micro_scaled], dim=1)

        # 计算 MSE
        pred_state_scaled = state_loss_features(pred_state_scaled, state_loss_source)
        true_state_scaled = state_loss_features(true_state_scaled, state_loss_source)
        h_loss = weighted_mse_loss(
            pred_state_scaled,
            true_state_scaled,
            stable_accumulation=(state_loss_source == "raw"),
        )

        weight = horizon_weights.get(h, 0.1)
        rollout_breakdown[f"H{h}"] = h_loss.item()

        total_loss += h_loss * weight
        total_weight += weight

    if total_weight > 0:
        rollout_loss = total_loss / total_weight
    else:
        rollout_loss = torch.tensor(0.0, device=pred_states_at_horizons[list(pred_states_at_horizons.keys())[0]]["Y_macro_next_raw"].device)

    groups = {"one_step": [1], "short": [8, 16, 32], "long": [64, 128]}
    for name, hs in groups.items():
        vals = [rollout_breakdown[f"H{h}"] for h in hs if f"H{h}" in rollout_breakdown]
        if vals: rollout_breakdown[name] = float(sum(vals) / len(vals))
    return rollout_loss, rollout_breakdown


def compute_training_rollout_loss(
    model,
    tools,
    batch_X,
    active_horizon,
    experiment_mode,
    batch_Y=None,
    horizons=(4, 8, 16, 32, 64, 128),
    detach_interval: int = 16,
    lambda_stability: float = 0.0,
    trust_region_fraction: float = 0.0,
    temperature_mode: str = "recorded",
    thermo_closure: dict | None = None,
    max_delta_temperature: float = 1000.0,
    state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
    feedback_state_source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
    feedback_noise_std: float = 0.0,
    backward_callback=None,
):
    """Run a segmented pure-species AR rollout for training loss.

    The graph remains connected within ``detach_interval`` steps and is then
    detached. When ``backward_callback`` is provided, each segment loss is
    backpropagated immediately so its activations can be released. Without a
    callback, the differentiable full rollout loss is returned as before.
    """
    available = [h for h in horizons if h <= active_horizon]
    if not available:
        return torch.zeros((), device=batch_X.device), {}

    current_X = batch_X[:, 0, :]
    predicted = {}
    targets = {}
    rollout_breakdown = {}
    previous_predicted = state_features_scaled_from_input(batch_X[:, 0, :], tools)
    previous_target = previous_predicted.detach()
    previous_pred_rate = None
    previous_target_rate = None
    stability_sum = torch.zeros((), device=batch_X.device)
    stability_report_sum = torch.zeros((), device=batch_X.device)
    stability_chunk_sum = torch.zeros((), device=batch_X.device)
    chunk_loss = None
    streamed_loss = torch.zeros((), device=batch_X.device)
    horizon_weights = {
        4: 0.1,
        8: 0.1,
        16: 0.15,
        32: 0.15,
        64: 0.2,
        128: 0.3,
    }
    rollout_weight_sum = sum(horizon_weights.get(h, 0.1) for h in available)
    is_safe = (feedback_state_source != "raw")
    for step in range(active_horizon):
        pred_rate = model(current_X)
        pred_state = tools.predict_next_state(
            current_X,
            pred_rate,
            safe_for_rollout=is_safe,
            experiment_mode=experiment_mode,
            trust_region_fraction=trust_region_fraction if is_safe else 0.0,
            state_update_mode=state_update_mode,
        )
        horizon = step + 1
        if horizon in available:
            if backward_callback is None:
                predicted[horizon] = pred_state
                targets[horizon] = batch_X[:, horizon, :]
            else:
                endpoint_loss, endpoint_breakdown = compute_rollout_loss(
                    {horizon: pred_state},
                    {horizon: batch_X[:, horizon, :]},
                    tools,
                    target_horizons=[horizon],
                    state_loss_source=state_loss_source,
                )
                weighted_endpoint_loss = (
                    endpoint_loss
                    * horizon_weights.get(horizon, 0.1)
                    / max(rollout_weight_sum, 1e-12)
                )
                chunk_loss = (
                    weighted_endpoint_loss
                    if chunk_loss is None
                    else chunk_loss + weighted_endpoint_loss
                )
                streamed_loss = streamed_loss + weighted_endpoint_loss.detach()
                rollout_breakdown[f"H{horizon}"] = endpoint_breakdown[
                    f"H{horizon}"
                ]
        current_predicted = _call_state_features_scaled(
            state_features_scaled_from_prediction,
            pred_state,
            tools,
            state_loss_source=state_loss_source,
        )
        current_target = state_features_scaled_from_input(batch_X[:, horizon, :], tools)
        target_rate = batch_Y[:, step, :] if batch_Y is not None else None
        if lambda_stability > 0.0:
            if state_update_mode in ("qssa", "qssa_asymptotic"):
                stability_term = None
                # A rate-jump penalty needs two adjacent QSSA predictions.
                # The first transition has no previous rate and contributes
                # a finite zero term instead of comparing raw states.
                if (
                    target_rate is not None
                    and previous_pred_rate is not None
                    and previous_target_rate is not None
                ):
                    predicted_log_rates = qssa_log_rate_features_from_scaled(
                        torch.stack([previous_pred_rate, pred_rate], dim=1), tools
                    )
                    target_log_rates = qssa_log_rate_features_from_scaled(
                        torch.stack([previous_target_rate, target_rate], dim=1), tools
                    )
                    stability_term = compute_rate_stability_loss(
                        predicted_log_rates, target_log_rates
                    )
            else:
                stability_term = compute_stability_loss(
                    torch.stack([previous_predicted, current_predicted], dim=1),
                    torch.stack([previous_target, current_target], dim=1),
                )
            if stability_term is not None:
                if backward_callback is None:
                    stability_sum = stability_sum + stability_term
                else:
                    stability_chunk_sum = stability_chunk_sum + stability_term
                    stability_report_sum = stability_report_sum + stability_term.detach()
        previous_predicted = current_predicted
        previous_target = current_target.detach()
        previous_pred_rate = pred_rate
        previous_target_rate = target_rate.detach() if target_rate is not None else None
        if horizon < active_horizon:
            feedback_macro, feedback_log_micro = select_feedback_state(
                pred_state, feedback_state_source
            )
            if feedback_noise_std > 0.0 and model.training:
                feedback_macro, feedback_log_micro = apply_feedback_state_noise(
                    feedback_macro,
                    feedback_log_micro,
                    feedback_noise_std,
                    feedback_state_source,
                )
            current_X = tools.state_to_input_scaled(
                batch_X[:, horizon, :],
                feedback_macro,
                feedback_log_micro,
                temperature_mode=temperature_mode,
                thermo_closure=thermo_closure,
                current_X_scaled=current_X,
                max_delta_temperature=max_delta_temperature,
                feedback_input_limit=(
                    RAW_FEEDBACK_INPUT_LIMIT
                    if feedback_state_source == "raw"
                    else None
                ),
            )
            if detach_interval > 0 and horizon % detach_interval == 0:
                current_X = current_X.detach()
                previous_predicted = previous_predicted.detach()
                if previous_pred_rate is not None:
                    previous_pred_rate = previous_pred_rate.detach()

        if backward_callback is not None and (
            horizon == active_horizon
            or (detach_interval > 0 and horizon % detach_interval == 0)
        ):
            if lambda_stability > 0.0:
                stability_chunk_loss = (
                    float(lambda_stability)
                    * stability_chunk_sum
                    / max(int(active_horizon), 1)
                )
                chunk_loss = (
                    stability_chunk_loss
                    if chunk_loss is None
                    else chunk_loss + stability_chunk_loss
                )
            if chunk_loss is not None and chunk_loss.requires_grad:
                backward_callback(chunk_loss)
            chunk_loss = None
            stability_chunk_sum = torch.zeros((), device=batch_X.device)

    if backward_callback is None:
        try:
            rollout_loss, rollout_breakdown = compute_rollout_loss(
                predicted,
                targets,
                tools,
                target_horizons=available,
                state_loss_source=state_loss_source,
            )
        except TypeError:
            rollout_loss, rollout_breakdown = compute_rollout_loss(
                predicted,
                targets,
                tools,
                target_horizons=available,
            )
        if lambda_stability > 0.0:
            stability_loss = stability_sum / max(int(active_horizon), 1)
            rollout_loss = rollout_loss + float(lambda_stability) * stability_loss
            rollout_breakdown["stability"] = stability_loss.detach().item()
    else:
        rollout_loss = streamed_loss
        if lambda_stability > 0.0:
            stability_loss = stability_report_sum / max(int(active_horizon), 1)
            rollout_loss = rollout_loss + float(lambda_stability) * stability_loss
            rollout_breakdown["stability"] = stability_loss.item()

    groups = {"one_step": [1], "short": [8, 16, 32], "long": [64, 128]}
    for name, hs in groups.items():
        vals = [rollout_breakdown[f"H{h}"] for h in hs if f"H{h}" in rollout_breakdown]
        if vals:
            rollout_breakdown[name] = float(sum(vals) / len(vals))
    return rollout_loss, rollout_breakdown


# ============================================================
# 13. Loss
# ============================================================

def compute_training_loss(
    pred_rate_scaled,
    target,
    pred_state,
    tools,
    true_next_X_scaled,
    lambda_rate=1.0,
    lambda_state=1.0,
    lambda_element=0.2,
    lambda_mass=0.05,
    lambda_positive=0.01,
    rate_loss_kind="mse",
    rate_huber_beta=1.0,
    lambda_rollout=0.0,  # 新增：rollout loss 权重
    rollout_states=None,  # 新增：多步 rollout 状态
    rollout_targets=None,  # 新增：多步 rollout 目标
    rate_feature_weights=None,
    rate_sample_weights=None,
    state_feature_weights=None,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
    lambda_radical: float = 0.0,
    lambda_no: float = 0.0,
):
    """
    最终 loss：

        L =
            lambda_rate * rate_loss
          + lambda_state * integrated_state_loss
          + lambda_element * element_loss
          + lambda_mass * mass_loss
          + lambda_positive * positivity
          + lambda_rollout * rollout_loss
          + lambda_radical * radical_loss
          + lambda_no * no_loss
    """

    # Both macro z and micro log-rate targets are already standardized
    # consistently by scaler_Y in V68.
    true_rate_scaled = target
    rate_mask = compute_qssa_rate_mask(tools, true_rate_scaled)
    if rate_loss_kind == "mse":
        rate_loss = weighted_mse_loss(
            pred_rate_scaled,
            true_rate_scaled,
            sample_weights=rate_sample_weights,
            feature_weights=rate_feature_weights,
            mask=rate_mask,
        )
    elif rate_loss_kind == "huber":
        rate_loss = weighted_smooth_l1_loss(
            pred_rate_scaled,
            true_rate_scaled,
            sample_weights=rate_sample_weights,
            feature_weights=rate_feature_weights,
            mask=rate_mask,
            beta=rate_huber_beta,
        )
    else:
        raise ValueError("rate_loss_kind must be 'mse' or 'huber'")

    # ========================================================
    # 2. State loss
    # ========================================================

    # 当前 predicted next species
    pred_macro, pred_micro_log = select_training_state(
        pred_state, state_loss_source
    )

    pred_state_features = torch.cat(
        [
            pred_macro,
            pred_micro_log,
        ],
        dim=1,
    )

    # true next state 已经位于 X scaled space
    true_macro_scaled = (
        true_next_X_scaled[
            :,
            tools.macro_idx
        ]
    )

    true_micro_log_scaled = (
        true_next_X_scaled[
            :,
            tools.micro_idx
        ]
    )

    true_state_features_scaled = torch.cat(
        [
            true_macro_scaled,
            true_micro_log_scaled,
        ],
        dim=1,
    )

    # 将 predicted state 转到与 X scaler 相同的空间
    pred_state_unscaled_for_loss = (
        pred_state_features
    )

    pred_state_scaled = torch.cat(
        [
            (
                pred_state_unscaled_for_loss[
                    :,
                    :tools.n_macro
                ]
                - tools.mean_X[
                    tools.macro_idx
                ]
            )
            / tools.scale_X[
                tools.macro_idx
            ],

            (
                pred_state_unscaled_for_loss[
                    :,
                    tools.n_macro:
                ]
                - tools.mean_X[
                    tools.micro_idx
                ]
            )
            / tools.scale_X[
                tools.micro_idx
            ],
        ],
        dim=1,
    )

    pred_state_scaled = state_loss_features(pred_state_scaled, state_loss_source)
    true_state_features_scaled = state_loss_features(
        true_state_features_scaled,
        state_loss_source,
    )

    state_loss = weighted_mse_loss(
        pred_state_scaled,
        true_state_features_scaled,
        feature_weights=state_feature_weights,
        stable_accumulation=(state_loss_source == "raw"),
    )

    # ========================================================
    # 3. Physics soft loss
    # ========================================================

    loss_element, loss_mass = (
        compute_element_conservation_loss(
            tools,
            pred_state,
            state_loss_source=state_loss_source,
        )
    )

    # ========================================================
    # 4. Positivity
    # ========================================================

    positivity_loss = (
        compute_positivity_loss(
            pred_state
        )
    )

    # ========================================================
    # 5. Rollout Loss
    # ========================================================

    rollout_loss = torch.tensor(0.0, device=rate_loss.device)
    rollout_breakdown = {}

    if lambda_rollout > 0 and rollout_states is not None and rollout_targets is not None:
        rollout_loss, rollout_breakdown = compute_rollout_loss(
            rollout_states,
            rollout_targets,
            tools,
        )

    # ========================================================
    # 6. Physical radical and NO loss (Direct uninflated dex space)
    # ========================================================
    radical_loss = torch.tensor(0.0, device=rate_loss.device)
    if lambda_radical > 0.0:
        from enhanced_loss_and_weights import compute_physical_radical_loss
        radical_loss = compute_physical_radical_loss(
            tools=tools,
            pred_state=pred_state,
            true_next_X_scaled=true_next_X_scaled,
            state_source=state_loss_source,
        )

    no_loss = torch.tensor(0.0, device=rate_loss.device)
    if lambda_no > 0.0:
        from enhanced_loss_and_weights import compute_pollutant_no_loss
        no_loss = compute_pollutant_no_loss(
            tools=tools,
            pred_state=pred_state,
            true_next_X_scaled=true_next_X_scaled,
            state_source=state_loss_source,
        )

    total = (
        lambda_rate
        * rate_loss

        + lambda_state
        * state_loss

        + lambda_element
        * loss_element

        + lambda_mass
        * loss_mass

        + lambda_positive
        * positivity_loss

        + lambda_rollout
        * rollout_loss

        + lambda_radical
        * radical_loss

        + lambda_no
        * no_loss
    )

    metrics = {
        "total": total,
        "rate": rate_loss,
        "state": state_loss,
        "element": loss_element,
        "mass": loss_mass,
        "positive": positivity_loss,
        "rollout": rollout_loss,
        "radical": radical_loss,
        "no": no_loss,
        "rollout_breakdown": rollout_breakdown,
    }

    return metrics


# ============================================================
# 14. Scheduled sampling
# ============================================================

def teacher_forcing_probability(
    epoch: int,
    epochs: int,
    start: float = 1.0,
    end: float = 0.10,
):
    """
    平滑退火。

    不建议完全降到 0：
    在训练后期仍保留少量 teacher forcing，
    避免模型训练完全锁死在早期预测误差的 OOD 状态。
    """

    progress = (
        epoch / max(
            epochs - 1,
            1,
        )
    )

    progress = np.clip(
        progress,
        0.0,
        1.0,
    )

    # cosine decay
    alpha = (
        0.5
        * (
            1
            - np.cos(
                np.pi
                * progress
            )
        )
    )

    return (
        start
        + (end - start)
        * alpha
    )


# ============================================================
# 15. Validation
# ============================================================

@torch.no_grad()
def validate_model_v68(
    model,
    val_loader,
    tools,
    device,
    eval_horizons,
    active_horizon,
    lambda_rate=1.0,
    lambda_state=1.0,
    lambda_element=0.2,
    lambda_mass=0.05,
    lambda_positive=0.01,
    rate_loss_kind="mse",
    rate_huber_beta=1.0,
    lambda_radical: float = 0.0,
    lambda_no: float = 0.0,
    max_batches=None,
    mode="oracle",  # "oracle" or "pure_ar"
    experiment_mode="baseline",  # 新增：实验模式
    trust_region_fraction: float = 0.0,
    temperature_mode: str = "recorded",
    thermo_closure: dict | None = None,
    max_delta_temperature: float = 1000.0,
    rate_feature_weights=None,
    state_feature_weights=None,
    state_update_mode: str = "legacy",
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
    feedback_state_source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
):
    """
    V68 双模式验证：
    - mode="oracle": Oracle one-step，每步用真实物种，验证单步准确性
    - mode="pure_ar": Pure species-AR，完全自回归，T/P/dt 保持真实

    返回：
    - mode="oracle": 标准 loss dict + per-horizon oracle metrics
    - mode="pure_ar": 标准 loss dict + per-horizon pure-AR metrics
    """
    model.eval()

    # 标准 loss 累积（向后兼容 V68 报告格式）
    total = 0.0
    total_rate = 0.0
    total_state = 0.0
    total_element = 0.0
    total_mass = 0.0
    total_positive = 0.0
    total_radical = 0.0
    total_no = 0.0

    # Per-horizon 指标累积器
    horizon_metrics = HorizonMetricAccumulator(eval_horizons)
    sentinel_accum = {str(h): {"true": [], "pred": [], "inert_mass": []} for h in eval_horizons}

    batches = 0

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:

            if max_batches is not None and batches >= max_batches:
                break

            batch_X = batch_X.to(device, non_blocking=True)
            batch_Y = batch_Y.to(device, non_blocking=True)

            B = batch_X.size(0)
            if mode == "pure_ar":
                H = min(max(eval_horizons), batch_X.size(1) - 1)
            else:
                H = min(active_horizon, batch_X.size(1) - 1)

            # 截取到 H
            batch_X = batch_X[:, :H + 1, :]
            batch_Y = batch_Y[:, :H, :]

            seq_total = 0.0
            seq_rate = 0.0
            seq_state = 0.0
            seq_element = 0.0
            seq_mass = 0.0
            seq_positive = 0.0
            seq_radical = 0.0
            seq_no = 0.0

            if mode == "oracle":
                # Oracle one-step: 每步用真实物种
                for step in range(H):
                    current_X = batch_X[:, step, :]

                    pred_rate = model(current_X)
                    pred_state = tools.predict_next_state(
                        current_X,
                        pred_rate,
                        safe_for_rollout=(feedback_state_source != "raw"),
                        experiment_mode=experiment_mode,
                        trust_region_fraction=trust_region_fraction,
                        state_update_mode=state_update_mode,
                    )

                    true_next_X = batch_X[:, step + 1, :]

                    loss_dict = compute_training_loss(
                        pred_rate,
                        batch_Y[:, step, :],
                        pred_state,
                        tools,
                        true_next_X,
                        lambda_rate,
                        lambda_state,
                        lambda_element,
                        lambda_mass,
                        lambda_positive,
                        state_feature_weights=state_feature_weights,
                        rate_loss_kind=rate_loss_kind,
                        rate_huber_beta=rate_huber_beta,
                        state_loss_source=state_loss_source,
                    )

                    loss_rad_val = torch.tensor(0.0, device=device)
                    if lambda_radical > 0.0:
                        from enhanced_loss_and_weights import compute_physical_radical_loss
                        loss_rad_val = compute_physical_radical_loss(
                            tools=tools,
                            pred_state=pred_state,
                            true_next_X_scaled=true_next_X,
                            state_source=state_loss_source,
                        )

                    loss_no_val = torch.tensor(0.0, device=device)
                    if lambda_no > 0.0:
                        from enhanced_loss_and_weights import compute_pollutant_no_loss
                        loss_no_val = compute_pollutant_no_loss(
                            tools=tools,
                            pred_state=pred_state,
                            true_next_X_scaled=true_next_X,
                            state_source=state_loss_source,
                        )

                    step_total = (
                        loss_dict["total"].item()
                        + (lambda_radical * loss_rad_val).item()
                        + (lambda_no * loss_no_val).item()
                    )
                    seq_total += step_total
                    seq_rate += loss_dict["rate"].item()
                    seq_state += loss_dict["state"].item()
                    seq_element += loss_dict["element"].item()
                    seq_mass += loss_dict["mass"].item()
                    seq_positive += loss_dict["positive"].item()
                    seq_radical += loss_rad_val.item()
                    seq_no += loss_no_val.item()

                    # Per-horizon oracle metrics: state MSE
                    if (step + 1) in eval_horizons:
                        val_macro, val_micro_log = select_training_state(
                            pred_state, state_loss_source
                        )
                        pred_macro_scaled = (
                            (val_macro - tools.mean_X[tools.macro_idx])
                            / tools.scale_X[tools.macro_idx]
                        )
                        pred_micro_log_scaled = (
                            (val_micro_log - tools.mean_X[tools.micro_idx])
                            / tools.scale_X[tools.micro_idx]
                        )
                        pred_state_scaled = torch.cat([pred_macro_scaled, pred_micro_log_scaled], dim=1)

                        true_macro_scaled = true_next_X[:, tools.macro_idx]
                        true_micro_scaled = true_next_X[:, tools.micro_idx]
                        true_state_scaled = torch.cat([true_macro_scaled, true_micro_scaled], dim=1)

                        horizon_metrics.update(
                            step + 1,
                            pred_state_scaled,
                            true_state_scaled,
                            B,
                        )

            elif mode == "pure_ar":
                # Pure species-AR: 完全自回归物种，T/P/dt 保持真实
                current_X = batch_X[:, 0, :]

                for step in range(H):
                    true_current_X = batch_X[:, step, :]
                    pred_rate = model(current_X)
                    if step == 0 or torch.equal(current_X, true_current_X):
                        rate_pred_true = pred_rate
                    else:
                        rate_pred_true = model(true_current_X)

                    pred_state = tools.predict_next_state(
                        current_X,
                        pred_rate,
                        safe_for_rollout=(feedback_state_source != "raw"),
                        experiment_mode=experiment_mode,
                        trust_region_fraction=(
                            trust_region_fraction
                            if feedback_state_source != "raw"
                            else 0.0
                        ),
                        state_update_mode=state_update_mode,
                    )

                    true_next_X = batch_X[:, step + 1, :]

                    loss_dict = compute_training_loss(
                        rate_pred_true,
                        batch_Y[:, step, :],
                        pred_state,
                        tools,
                        true_next_X,
                        lambda_rate,
                        lambda_state,
                        lambda_element,
                        lambda_mass,
                        lambda_positive,
                        state_feature_weights=state_feature_weights,
                        rate_loss_kind=rate_loss_kind,
                        rate_huber_beta=rate_huber_beta,
                        state_loss_source=state_loss_source,
                    )

                    loss_rad_val = torch.tensor(0.0, device=device)
                    if lambda_radical > 0.0:
                        from enhanced_loss_and_weights import compute_physical_radical_loss
                        loss_rad_val = compute_physical_radical_loss(
                            tools=tools,
                            pred_state=pred_state,
                            true_next_X_scaled=true_next_X,
                            state_source=state_loss_source,
                        )

                    loss_no_val = torch.tensor(0.0, device=device)
                    if lambda_no > 0.0:
                        from enhanced_loss_and_weights import compute_pollutant_no_loss
                        loss_no_val = compute_pollutant_no_loss(
                            tools=tools,
                            pred_state=pred_state,
                            true_next_X_scaled=true_next_X,
                            state_source=state_loss_source,
                        )

                    step_total = (
                        loss_dict["total"].item()
                        + (lambda_radical * loss_rad_val).item()
                        + (lambda_no * loss_no_val).item()
                    )
                    seq_total += step_total
                    seq_rate += loss_dict["rate"].item()
                    seq_state += loss_dict["state"].item()
                    seq_element += loss_dict["element"].item()
                    seq_mass += loss_dict["mass"].item()
                    seq_positive += loss_dict["positive"].item()
                    seq_radical += loss_rad_val.item()
                    seq_no += loss_no_val.item()

                    # Per-horizon pure-AR metrics
                    if (step + 1) in eval_horizons:
                        val_macro, val_micro_log = select_training_state(
                            pred_state, state_loss_source
                        )
                        pred_macro_scaled = (
                            (val_macro - tools.mean_X[tools.macro_idx])
                            / tools.scale_X[tools.macro_idx]
                        )
                        pred_micro_log_scaled = (
                            (val_micro_log - tools.mean_X[tools.micro_idx])
                            / tools.scale_X[tools.micro_idx]
                        )
                        pred_state_scaled = torch.cat([pred_macro_scaled, pred_micro_log_scaled], dim=1)

                        true_macro_scaled = true_next_X[:, tools.macro_idx]
                        true_micro_scaled = true_next_X[:, tools.micro_idx]
                        true_state_scaled = torch.cat([true_macro_scaled, true_micro_scaled], dim=1)

                        horizon_metrics.update(
                            step + 1,
                            pred_state_scaled,
                            true_state_scaled,
                            B,
                        )

                        h_key = str(step + 1)
                        val_micro = torch.pow(10.0, torch.clamp(val_micro_log, min=-40.0, max=0.0))
                        pred_species = torch.cat([val_macro, val_micro], dim=-1).detach().cpu().numpy()

                        true_macro_raw = true_next_X[:, tools.macro_idx] * tools.scale_X[tools.macro_idx] + tools.mean_X[tools.macro_idx]
                        true_micro_log_raw = true_next_X[:, tools.micro_idx] * tools.scale_X[tools.micro_idx] + tools.mean_X[tools.micro_idx]
                        true_micro_raw = torch.pow(10.0, torch.clamp(true_micro_log_raw, min=-40.0, max=0.0))
                        true_species = torch.cat([true_macro_raw, true_micro_raw], dim=-1).detach().cpu().numpy()

                        sentinel_accum[h_key]["pred"].append(pred_species)
                        sentinel_accum[h_key]["true"].append(true_species)
                        inert = tools.inverse_state(true_next_X)[:, tools.inert_idx].sum(dim=-1)
                        sentinel_accum[h_key]["inert_mass"].append(inert.detach().cpu().numpy())

                    # 构造下一步输入：预测物种 + 真实 T/P/dt
                    if step < H - 1:
                        next_base = batch_X[:, step + 1, :]
                        feedback_macro, feedback_log_micro = select_feedback_state(
                            pred_state, feedback_state_source
                        )
                        current_X = tools.state_to_input_scaled(
                            next_base,
                            feedback_macro,
                            feedback_log_micro,
                            temperature_mode=temperature_mode,
                            thermo_closure=thermo_closure,
                            current_X_scaled=current_X,
                            max_delta_temperature=max_delta_temperature,
                            feedback_input_limit=(
                                RAW_FEEDBACK_INPUT_LIMIT
                                if feedback_state_source == "raw"
                                else None
                            ),
                        )

            else:
                raise ValueError(f"Unknown validation mode: {mode}")

            total += seq_total / H
            total_rate += seq_rate / H
            total_state += seq_state / H
            total_element += seq_element / H
            total_mass += seq_mass / H
            total_positive += seq_positive / H
            total_radical += seq_radical / H
            total_no += seq_no / H

            batches += 1

    if batches == 0:
        raise RuntimeError("No validation batches.")

    sentinel_metrics = {}
    species_order = list(tools.macro_species) + list(tools.micro_species)
    elem_mat = getattr(tools, "A", None)
    elem_mat_np = elem_mat.detach().cpu().numpy() if elem_mat is not None else None
    for h_str, acc in sentinel_accum.items():
        if acc["pred"]:
            all_preds = np.concatenate(acc["pred"], axis=0)
            all_trues = np.concatenate(acc["true"], axis=0)
            sentinel_metrics[h_str] = svp.compute_species_trajectory_metrics(
                true_states=all_trues,
                pred_states=all_preds,
                species_names=species_order,
                element_matrix=elem_mat_np,
                inert_mass=np.concatenate(acc["inert_mass"]),
            )

    stats = {
        "total": total / batches,
        "rate": total_rate / batches,
        "state": total_state / batches,
        "element": total_element / batches,
        "mass": total_mass / batches,
        "positive": total_positive / batches,
        "radical": total_radical / batches,
        "no": total_no / batches,
        "per_horizon": horizon_metrics.finalize(),
        "sentinel_metrics": sentinel_metrics,
        "mode": mode,
    }

    return stats


def validate_model(
    model,
    val_loader,
    tools,
    device,
    lambda_rate=1.0,
    lambda_state=1.0,
    lambda_element=0.2,
    lambda_mass=0.05,
    lambda_positive=0.01,
    max_batches=None,
    state_update_mode="qssa",  # Add missing parameter with default
    feedback_state_source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
):
    model.eval()

    total = 0.0
    total_rate = 0.0
    total_state = 0.0
    total_element = 0.0
    total_mass = 0.0
    total_positive = 0.0

    batches = 0

    for batch_X, batch_Y in val_loader:

        batch_X = batch_X.to(
            device,
            non_blocking=True,
        )

        batch_Y = batch_Y.to(
            device,
            non_blocking=True,
        )

        current_X = (
            batch_X[:, 0, :]
        )

        seq_total = 0.0
        seq_rate = 0.0
        seq_state = 0.0
        seq_element = 0.0
        seq_mass = 0.0
        seq_positive = 0.0

        for step in range(
            batch_Y.shape[1]
        ):

            pred_rate = model(
                current_X
            )

            pred_state = (
                tools.predict_next_state(
                    current_X,
                    pred_rate,
                    safe_for_rollout=(feedback_state_source != "raw"),
                    state_update_mode=state_update_mode,
                )
            )

            # 每个 source target 都对应严格的 t -> t+1 endpoint。
            true_next_X = batch_X[:, step + 1, :]

            loss_dict = (
                compute_training_loss(
                    pred_rate,
                    batch_Y[:, step, :],
                    pred_state,
                    tools,
                    true_next_X,
                    lambda_rate,
                    lambda_state,
                    lambda_element,
                    lambda_mass,
                    lambda_positive,
                )
            )

            seq_total += (
                loss_dict["total"].item()
            )

            seq_rate += (
                loss_dict["rate"].item()
            )

            seq_state += (
                loss_dict["state"].item()
            )

            seq_element += (
                loss_dict["element"].item()
            )

            seq_mass += (
                loss_dict["mass"].item()
            )

            seq_positive += (
                loss_dict["positive"].item()
            )

            # 最后一步不再构造 next state
            if step < batch_Y.shape[1] - 1:

                next_base = (
                    batch_X[
                        :,
                        step + 1,
                        :
                    ]
                )

                current_X = (
                    tools.state_to_input_scaled(
                        next_base,
                        pred_state[
                            "Y_macro_next"
                        ],
                        pred_state[
                            "logY_micro_next"
                        ],
                    )
                )

        total += (
            seq_total
            / batch_Y.shape[1]
        )

        total_rate += (
            seq_rate
            / batch_Y.shape[1]
        )

        total_state += (
            seq_state
            / batch_Y.shape[1]
        )

        total_element += (
            seq_element
            / batch_Y.shape[1]
        )

        total_mass += (
            seq_mass
            / batch_Y.shape[1]
        )

        total_positive += (
            seq_positive
            / batch_Y.shape[1]
        )

        batches += 1

        if (
            max_batches is not None
            and batches >= max_batches
        ):
            break

    if batches == 0:
        return {
            "total": float("inf"),
            "rate": float("inf"),
            "state": float("inf"),
            "element": float("inf"),
            "mass": float("inf"),
            "positive": float("inf"),
        }

    return {
        "total":
            total / batches,

        "rate":
            total_rate / batches,

        "state":
            total_state / batches,

        "element":
            total_element / batches,

        "mass":
            total_mass / batches,

        "positive":
            total_positive / batches,
    }


# ============================================================
# 16. Training
# ============================================================


def scheduled_tail_fraction(
    epoch: int,
    tail_start: float,
    tail_end: float,
    tail_ramp_epochs: int,
) -> float:
    """Return the AR tail fraction for a warm continuation schedule."""
    if not 0.0 <= tail_start <= 1.0:
        raise ValueError("tail_start must be between 0 and 1")
    if not 0.0 <= tail_end <= 1.0:
        raise ValueError("tail_end must be between 0 and 1")
    if tail_end < tail_start:
        raise ValueError("tail_end must be >= tail_start")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if tail_ramp_epochs < 0:
        raise ValueError("tail_ramp_epochs must be non-negative")
    if tail_ramp_epochs == 0:
        return float(tail_end)
    progress = min(1.0, epoch / float(tail_ramp_epochs))
    return float(tail_start + (tail_end - tail_start) * progress)


def scheduled_feedback_strength(
    epoch: int,
    start: float,
    end: float,
    ramp_epochs: int,
) -> float:
    """Schedule the strength of predicted-state feedback for V3.4."""
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("feedback blend bounds must be between 0 and 1")
    if end < start:
        raise ValueError("feedback blend end must be >= start")
    if epoch < 0 or ramp_epochs < 0:
        raise ValueError("epoch and ramp_epochs must be non-negative")
    if ramp_epochs == 0:
        return float(end)
    progress = min(1.0, epoch / float(ramp_epochs))
    return float(start + (end - start) * progress)


def blend_feedback_state(
    base_state: torch.Tensor,
    predicted_state: torch.Tensor,
    use_prediction: torch.Tensor,
    feedback_strength: float,
) -> torch.Tensor:
    """Blend teacher and predicted states without a hard discontinuity."""
    alpha = use_prediction.to(dtype=base_state.dtype) * float(feedback_strength)
    while alpha.ndim < base_state.ndim:
        alpha = alpha.unsqueeze(-1)
    return base_state + alpha * (predicted_state - base_state)


def update_temperature_autonomous(
    temperature: torch.Tensor,
    delta_y: torch.Tensor,
    energy_coeff: torch.Tensor | None = None,
    heat_capacity: torch.Tensor | None = None,
    max_delta_temperature: float = 1000.0,
    thermo_closure: dict | None = None,
    Y_old_full: torch.Tensor | None = None,
    Y_next_full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update temperature from composition energy change.

    When thermo_closure provides 'nasa7_layer' and full species mass fractions,
    evaluates exact mixture internal energy (or enthalpy) and inverts for T_next
    via differentiable Newton-Raphson with NASA-7 polynomials.
    Falls back to first-order linear approximation around reference temperature if needed.
    """
    if (
        thermo_closure is not None
        and thermo_closure.get("nasa7_layer") is not None
        and Y_old_full is not None
        and Y_next_full is not None
    ):
        try:
            layer = thermo_closure["nasa7_layer"]
            reactor_type = thermo_closure.get("reactor_type", "IdealGasReactor")
            if reactor_type == "IdealGasConstPressureReactor":
                _, _, h_old, _ = layer.compute_mixture_properties(temperature, Y_old_full)
                T_solved = layer.solve_temperature_hp(h_old, Y_next_full, temperature)
            else:
                _, _, _, u_old = layer.compute_mixture_properties(temperature, Y_old_full)
                T_solved = layer.solve_temperature_uv(u_old, Y_next_full, temperature)

            dT = torch.clamp(
                T_solved - temperature,
                -float(max_delta_temperature),
                float(max_delta_temperature),
            )
            return torch.clamp(temperature + dT, min=200.0, max=5000.0)
        except Exception:
            pass  # Fall through to linear fallback

    if energy_coeff is None or heat_capacity is None:
        return temperature

    coeff = energy_coeff.to(device=delta_y.device, dtype=delta_y.dtype).reshape(1, -1)
    capacity = heat_capacity.to(device=delta_y.device, dtype=delta_y.dtype)
    if capacity.ndim == 1:
        capacity = capacity.unsqueeze(-1)
    delta_temperature = -(delta_y * coeff).sum(dim=-1, keepdim=True) / capacity.clamp_min(1e-6)
    delta_temperature = torch.clamp(
        delta_temperature,
        min=-float(max_delta_temperature),
        max=float(max_delta_temperature),
    )
    return torch.clamp(temperature + delta_temperature, min=200.0, max=5000.0)


def resolve_temperature_mechanism_path(
    metadata: dict,
    data_dir: str | Path,
    mechanism: str,
) -> Path:
    """Resolve Cantera input locally before accepting a stale metadata path."""
    data_root = Path(data_dir)
    known_names = {
        "nuig_2024": "NUIG2024.yaml",
        "tsinghua_2024c": "TH2024.yaml",
        "ucf_2024": "UCF2024.yaml",
    }
    candidates = []
    mapped_name = known_names.get(str(mechanism).lower())
    if mapped_name:
        candidates.append(data_root / mapped_name)
    candidates.extend(
        [
            data_root / f"{mechanism}.yaml",
            data_root / f"{mechanism}.yml",
            data_root / "database_Transient_V21_QSSA" / mapped_name
            if mapped_name else None,
            data_root / "database_Transient_V21_QSSA" / f"{mechanism}.yaml",
            data_root / "mechanisms" / mapped_name
            if mapped_name else None,
            data_root / "mechanisms" / f"{mechanism}.yaml",
        ]
    )
    recorded = Path(metadata["mechanism_path"]) if metadata.get("mechanism_path") else None
    if recorded is not None:
        # A moved workspace often keeps the mechanism beside the data root or
        # in a mechanisms subdirectory under it.  Search by basename first;
        # only then accept the original absolute metadata path.
        candidates.extend(
            [
                data_root / recorded.name,
                data_root.parent / recorded.name,
                data_root / "mechanisms" / recorded.name,
                data_root.parent / "mechanisms" / recorded.name,
                recorded,
            ]
        )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()

    attempted = "\n  ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Cantera mechanism file not found. Tried:\n  " + attempted
    )


def build_temperature_closure(
    metadata: dict,
    species_order: list[str],
    data_dir: str | Path | None = None,
    mechanism: str | None = None,
) -> dict:
    """Build differentiable thermodynamic closure layer from Cantera mechanism."""
    try:
        import cantera as ct
    except ImportError as exc:
        raise RuntimeError("V3.5 autonomous temperature requires Cantera") from exc

    if data_dir is not None and mechanism is not None:
        mechanism_path = resolve_temperature_mechanism_path(
            metadata, data_dir, mechanism
        )
    else:
        mechanism_path = metadata.get("mechanism_path")
        if not mechanism_path:
            raise ValueError("Dataset metadata has no mechanism_path for V3.5")
    gas = ct.Solution(str(mechanism_path))
    missing = [sp for sp in species_order if sp not in gas.species_names]
    if missing:
        raise ValueError(f"Mechanism is missing dataset species: {missing[:5]}")

    reference_temperature = 1000.0
    gas.TP = reference_temperature, ct.one_atm
    species_indices = [gas.species_index(sp) for sp in species_order]
    molecular_weights = gas.molecular_weights[species_indices]
    gas_constant = ct.gas_constant
    reactor_type = metadata.get("reactor_type", "IdealGasReactor")
    if reactor_type == "IdealGasConstPressureReactor":
        energy_molar = gas.standard_enthalpies_RT[species_indices] * gas_constant * reference_temperature
        heat_capacity = gas.standard_cp_R[species_indices] * gas_constant / molecular_weights
        energy_kind = "enthalpy/cp"
    else:
        energy_molar = gas.standard_int_energies_RT[species_indices] * gas_constant * reference_temperature
        heat_capacity = (gas.standard_cp_R[species_indices] - 1.0) * gas_constant / molecular_weights
        energy_kind = "internal_energy/cv"

    nasa7_layer = None
    try:
        from thermo_nasa7_layer import ThermoNASA7Layer
        nasa7_layer = ThermoNASA7Layer(
            mechanism_path=str(mechanism_path),
            species_order=species_order,
        )
        print(f"[THERMO] Exact ThermoNASA7Layer initialized successfully ({len(species_order)} species).")
    except Exception as exc:
        print(f"[THERMO] Warning: ThermoNASA7Layer init note: {exc}, using linear fallback.")

    return {
        "energy_coeff": (energy_molar / molecular_weights).astype(np.float32),
        "heat_capacity_coeff": heat_capacity.astype(np.float32),
        "reactor_type": reactor_type,
        "energy_kind": energy_kind,
        "reference_temperature": reference_temperature,
        "nasa7_layer": nasa7_layer,
        "species_order": species_order,
    }


def build_training_config_record(
    max_horizon: int = 128,
    curriculum_horizons: list[int] | None = None,
    curriculum_start_epochs: list[int] | None = None,
    tail_start: float = 0.05,
    tail_end: float = 1.0,
    tail_ramp_epochs: int = 0,
    backward_chunk_steps: int = 5,
    lambda_rate: float = 1.0,
    lambda_state: float = 1.0,
    lambda_element: float = 0.2,
    lambda_mass: float = 0.05,
    lambda_positive: float = 0.01,
    rate_loss_kind: str = "mse",
    rate_huber_beta: float = 1.0,
    output_parameterization: str = "direct_rate",
    rate_decode_mode: str = "direct_rate",
    rate_z_limit: float = DEFAULT_RATE_Z_LIMIT,
    micro_rate_z_limit: float | None = None,
    lambda_rollout: float = 0.0,
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
    detach_rollout: bool = True,
    experiment_mode: str = "baseline",
    state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
    feedback_state_source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
    rollout_signature: str = ROLLOUT_SIGNATURE,
    lambda_radical: float = 0.0,
    lambda_no: float = 0.0,
    use_induction_weights: bool = False,
    induction_boost: float = 6.0,
    projection_mode: str = "multiplicative_kl",
    desentinelize_scaler: bool = False,
    network_arch: str = "mlp",
    trajectory_split_train_signature: str | None = None,
    trajectory_split_validation_signature: str | None = None,
    trajectory_split_dir: str | None = None,
    species_loss_metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Construct a clean, audited training config dictionary for checkpoints and manifests."""
    curriculum_horizons = curriculum_horizons or [1, 8, 32, 128]
    curriculum_start_epochs = curriculum_start_epochs or [0, 2, 5, 8]
    record = {
        "max_horizon": int(max_horizon),
        "curriculum_horizons": list(curriculum_horizons),
        "curriculum_start_epochs": list(curriculum_start_epochs),
        "tail_start": float(tail_start),
        "tail_end": float(tail_end),
        "tail_ramp_epochs": int(tail_ramp_epochs),
        "backward_chunk_steps": int(backward_chunk_steps),
        "lambda_rate": float(lambda_rate),
        "lambda_state": float(lambda_state),
        "lambda_element": float(lambda_element),
        "lambda_mass": float(lambda_mass),
        "lambda_positive": float(lambda_positive),
        "rate_loss_kind": rate_loss_kind,
        "rate_huber_beta": float(rate_huber_beta),
        "output_parameterization": output_parameterization,
        "rate_decode_mode": rate_decode_mode,
        "rate_z_limit": float(rate_z_limit),
        "micro_rate_z_limit": (float(micro_rate_z_limit) if micro_rate_z_limit is not None else None),
        "lambda_rollout": float(lambda_rollout),
        "lambda_stability": float(lambda_stability),
        "transition_weight_alpha": float(transition_weight_alpha),
        "transition_weight_clip": float(transition_weight_clip),
        "feedback_noise_std": float(feedback_noise_std),
        "macro_loss_weight": float(macro_loss_weight),
        "micro_loss_weight": float(micro_loss_weight),
        "key_species_loss_weight": float(key_species_loss_weight),
        "feedback_blend_start": float(feedback_blend_start),
        "feedback_blend_end": float(feedback_blend_end),
        "feedback_blend_ramp_epochs": int(feedback_blend_ramp_epochs),
        "trust_region_fraction": float(trust_region_fraction),
        "delta_limit": float(trust_region_fraction),
        "temperature_mode": temperature_mode,
        "max_delta_temperature": float(max_delta_temperature),
        "detach_rollout": bool(detach_rollout),
        "experiment_mode": experiment_mode,
        "state_update_mode": state_update_mode,
        "task8_version": (TASK8_VERSION if state_update_mode == "task8_conservative_state" else "none"),
        "state_constraint_type": ("physical_simplex_nonnegative" if state_update_mode == "task8_conservative_state" else "none"),
        "state_constraint_epsilon": STATE_CONSTRAINT_EPSILON,
        "state_loss_source": state_loss_source,
        "feedback_state_source": feedback_state_source,
        "rollout_signature": rollout_signature,
        "lambda_radical": float(lambda_radical),
        "lambda_no": float(lambda_no),
        "use_induction_weights": bool(use_induction_weights),
        "induction_boost": float(induction_boost),
        "projection_mode": str(projection_mode),
        "desentinelize_scaler": bool(desentinelize_scaler),
        "network_arch": str(network_arch),
    }
    if trajectory_split_train_signature is not None:
        record["trajectory_split_train_signature"] = trajectory_split_train_signature
    if trajectory_split_validation_signature is not None:
        record["trajectory_split_validation_signature"] = trajectory_split_validation_signature
    if trajectory_split_dir is not None:
        record["trajectory_split_dir"] = trajectory_split_dir
    if species_loss_metadata is not None:
        record["species_loss_metadata"] = species_loss_metadata
    for k, v in kwargs.items():
        if k not in record:
            record[k] = v
    return record


def evaluate_robust_raw_checkpoint(
    pure_ar_h128: float,
    pure_ar_mass: float,
    pure_ar_element: float,
    pure_ar_rate: float,
    h128_limit: float = 1.0,
    mass_limit: float = 0.2,
    element_limit: float = 0.2,
    ref_h128: float = 0.02,
    ref_mass: float = 0.05,
    ref_element: float = 0.02,
    ref_rate: float = 0.5,
) -> tuple[bool, float | None]:
    """In-training coarse screening gate for robust checkpoint selection.

    Scientific Scope & Physical Semantics:
    - Inputs are derived from internal `val_stats_pure_ar` during training:
      * `pure_ar_h128`: Normalized scaled state RMSE at H=128 (using tools scaling);
      * `pure_ar_mass`, `pure_ar_element`: Soft conservation losses from compute_training_loss();
      * `pure_ar_rate`: Normalized rate loss.
    - Role: Serves as an in-training gate (训练期粗筛守门员) to reject divergent, non-finite,
      or unphysical checkpoints, preventing selection of early unconverged epochs.
    - Limitation: This in-training metric does NOT substitute for post-training offline
      multi-metric physical evaluation on the test split (which rigorously verifies
      unconstrained `pure_ar_raw` physical state RMSE, physical mass/element closure via
      compute_physical_metrics.py, active-region micro log RMSE, and N2 rate jump metrics).

    Returns:
        (is_accepted, score): (True, float_score) if passing hard gates, else (False, None).
    """
    is_finite = (
        math.isfinite(pure_ar_h128)
        and math.isfinite(pure_ar_mass)
        and math.isfinite(pure_ar_element)
        and math.isfinite(pure_ar_rate)
    )
    if not is_finite:
        return False, None

    is_bounded = (
        pure_ar_h128 < h128_limit
        and pure_ar_mass < mass_limit
        and pure_ar_element < element_limit
    )
    if not is_bounded:
        return False, None

    # The champion is selected lexicographically: hard conservation gates are
    # applied above, then H=128 closed-loop error alone determines the score.
    # Rate and conservation values remain recorded diagnostics, but cannot
    # compensate for even a small deterioration in long-horizon stability.
    score = math.log1p(max(0.0, pure_ar_h128) / ref_h128)
    return True, score


def train_model_v68(
    data_dir: str,
    mechanism: str = "nuig_2024",
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 2e-4,
    max_horizon: int = 128,  # V68: 最大 horizon
    curriculum_horizons: list = None,  # V68: horizon curriculum
    curriculum_start_epochs: list = None,  # V68: curriculum 起始 epoch
    eval_horizons: list = None,  # V68: 评估的 horizon 列表
    tail_start: float = 0.05,  # V68: 初始连续预测尾段比例
    tail_end: float = 1.0,  # V68: 最终连续预测尾段比例
    tail_ramp_epochs: int = 0,  # V68: AR 尾段升温所需的新 epoch 数
    train_stride: int = None,  # V68: 训练窗口 stride
    val_stride: int = None,  # V68: 验证窗口 stride
    parts_per_epoch: int = 42,
    val_parts_per_epoch: int | None = 9,
    val_batch_size: int | None = 2048,
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
    rate_z_limit: float = DEFAULT_RATE_Z_LIMIT,
    micro_rate_z_limit: float | None = None,
    lambda_rollout: float = 0.0,  # 新增：rollout loss 权重
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
    detach_rollout: bool = True,
    backward_chunk_steps: int = 5,
    resume: str = "auto",  # V68: resume 模式
    warm_start: str = None,  # 新增：暖机启动模式，指定预训练checkpoint路径
    experiment_mode: str = "baseline",  # 新增：实验模式 (baseline/diagnostic/constrained_delta/conserve_project)
    state_update_mode: str = DEFAULT_STATE_UPDATE_MODE,  # Task8: legacy or task8_conservative_state
    state_loss_source: str = DEFAULT_STATE_LOSS_SOURCE,
    feedback_state_source: str = DEFAULT_FEEDBACK_STATE_SOURCE,
    checkpoint_every: int = 10,  # V68: checkpoint 间隔
    full_val_every: int = 5,  # V68: 完整验证间隔（包含Pure-AR）
    max_val_batches: int = None,  # V68: 限制验证 batch 数
    num_workers: int = 0,
    output_root: str = ".",
    split_dir: str = None,
    lambda_radical: float = 0.0,
    lambda_no: float = 0.0,
    use_induction_weights: bool = False,
    induction_boost: float = 6.0,
    projection_mode: str = "multiplicative_kl",
    desentinelize_scaler: bool = False,
    network_arch: str = "mlp",
):
    """
    V68 训练入口：长 horizon 物种自回归，外部真实 T/P/dt。

    核心改进：
    - 连续预测尾段 scheduled sampling（vs V68 独立步 teacher forcing）
    - Horizon curriculum：8 → 16 → 32 → 64 → 128
    - 双模式验证：oracle one-step + pure species-AR
    - 多 horizon 指标：1/2/4/8/16/32/64/128 步
    - Resume 支持完整 RNG 状态

    新增（阶段四/五）：
    - experiment_mode: 状态更新实验模式
    - warm_start: 暖机启动，加载预训练权重但不加载优化器状态
    - lambda_rollout: 多步rollout损失权重
    - transition_weight_alpha: 高活性/点火转变步骤的损失放大系数
    - feedback_noise_std: 预测反馈状态的 scaled-space 扰动强度
    """

    # 设置默认值
    if curriculum_horizons is None:
        curriculum_horizons = [1, 8, 32, 128]
    if curriculum_start_epochs is None:
        curriculum_start_epochs = [0, 2, 5, 8]
    if eval_horizons is None:
        eval_horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    if train_stride is None:
        train_stride = max(1, max_horizon // 2)
    if val_stride is None:
        val_stride = max(1, max_horizon // 4)
    if val_batch_size is None:
        val_batch_size = batch_size

    # 构建 horizon schedule
    horizon_schedule = build_horizon_schedule(curriculum_horizons, curriculum_start_epochs)

    # 验证参数
    require_detached_rollout(detach_rollout)
    if not 1 <= backward_chunk_steps <= max_horizon:
        raise ValueError(
            "backward_chunk_steps must be between 1 and max_horizon"
        )
    if max(eval_horizons) > max_horizon:
        raise ValueError(
            f"eval_horizons max ({max(eval_horizons)}) exceeds max_horizon ({max_horizon})"
        )
    if transition_weight_alpha < 0.0:
        raise ValueError("transition_weight_alpha must be >= 0")
    if transition_weight_clip < 1.0:
        raise ValueError("transition_weight_clip must be >= 1")
    if feedback_noise_std < 0.0:
        raise ValueError("feedback_noise_std must be >= 0")
    if tail_ramp_epochs < 0:
        raise ValueError("tail_ramp_epochs must be >= 0")
    if macro_loss_weight <= 0.0 or micro_loss_weight <= 0.0 or key_species_loss_weight <= 0.0:
        raise ValueError("species loss weights must be positive")
    if lambda_stability < 0.0:
        raise ValueError("lambda_stability must be >= 0")
    if not 0.0 <= feedback_blend_start <= 1.0:
        raise ValueError("feedback_blend_start must be in [0, 1]")
    if not 0.0 <= feedback_blend_end <= 1.0:
        raise ValueError("feedback_blend_end must be in [0, 1]")
    if feedback_blend_end < feedback_blend_start:
        raise ValueError("feedback_blend_end must be >= feedback_blend_start")
    if feedback_blend_ramp_epochs < 0:
        raise ValueError("feedback_blend_ramp_epochs must be >= 0")
    if not 0.0 <= trust_region_fraction <= 1.0:
        raise ValueError("trust_region_fraction must be in [0, 1]")
    if temperature_mode not in {"recorded", "autonomous"}:
        raise ValueError("temperature_mode must be 'recorded' or 'autonomous'")
    if max_delta_temperature <= 0.0:
        raise ValueError("max_delta_temperature must be > 0")
    if rate_loss_kind not in {"mse", "huber"}:
        raise ValueError("rate_loss_kind must be 'mse' or 'huber'")
    if rate_huber_beta <= 0.0:
        raise ValueError("rate_huber_beta must be > 0")
    if output_parameterization not in {"direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"}:
        raise ValueError(f"invalid output_parameterization: {output_parameterization!r}")
    if (output_parameterization in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")) != (rate_decode_mode in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1")):
        raise ValueError("qssa and dual_branch output_parameterization and rate_decode_mode must be paired")
    if (output_parameterization == "qssa_asymptotic") != (rate_decode_mode == "qssa_asymptotic"):
        raise ValueError("qssa_asymptotic output_parameterization and rate_decode_mode must match")
    if (output_parameterization == "dual_branch") != (rate_decode_mode == "dual_branch"):
        raise ValueError("dual_branch output_parameterization and rate_decode_mode must match")
    if (output_parameterization == "dual_branch_pd_v1") != (rate_decode_mode == "dual_branch_pd_v1"):
        raise ValueError("dual_branch_pd_v1 output_parameterization and rate_decode_mode must match")
    if rate_decode_mode not in RATE_DECODE_MODES:
        raise ValueError(f"invalid rate_decode_mode: {rate_decode_mode!r}")
    if not np.isfinite(rate_z_limit) or rate_z_limit <= 0.0:
        raise ValueError("rate_z_limit must be a finite positive value")
    if state_loss_source not in STATE_LOSS_SOURCES:
        raise ValueError(f"invalid state_loss_source: {state_loss_source!r}")
    if feedback_state_source not in FEEDBACK_STATE_SOURCES:
        raise ValueError(f"invalid feedback_state_source: {feedback_state_source!r}")
    if micro_rate_z_limit is not None and (not np.isfinite(micro_rate_z_limit) or micro_rate_z_limit <= 0.0):
        raise ValueError("micro_rate_z_limit must be a finite positive value")

    # 验证实验模式
    if experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(
            f"experiment_mode must be one of {EXPERIMENT_MODES}, got {experiment_mode!r}"
        )

    # Persist the knobs that change the optimization behavior alongside every
    # new checkpoint.  This does not affect loading of legacy checkpoints.
    training_config = build_training_config_record(
        max_horizon=max_horizon,
        curriculum_horizons=curriculum_horizons,
        curriculum_start_epochs=curriculum_start_epochs,
        tail_start=tail_start,
        tail_end=tail_end,
        tail_ramp_epochs=tail_ramp_epochs,
        backward_chunk_steps=backward_chunk_steps,
        lambda_rate=lambda_rate,
        lambda_state=lambda_state,
        lambda_element=lambda_element,
        lambda_mass=lambda_mass,
        lambda_positive=lambda_positive,
        rate_loss_kind=rate_loss_kind,
        rate_huber_beta=rate_huber_beta,
        output_parameterization=output_parameterization,
        rate_decode_mode=rate_decode_mode,
        rate_z_limit=rate_z_limit,
        micro_rate_z_limit=micro_rate_z_limit,
        lambda_rollout=lambda_rollout,
        lambda_stability=lambda_stability,
        transition_weight_alpha=transition_weight_alpha,
        transition_weight_clip=transition_weight_clip,
        feedback_noise_std=feedback_noise_std,
        macro_loss_weight=macro_loss_weight,
        micro_loss_weight=micro_loss_weight,
        key_species_loss_weight=key_species_loss_weight,
        feedback_blend_start=feedback_blend_start,
        feedback_blend_end=feedback_blend_end,
        feedback_blend_ramp_epochs=feedback_blend_ramp_epochs,
        trust_region_fraction=trust_region_fraction,
        temperature_mode=temperature_mode,
        max_delta_temperature=max_delta_temperature,
        detach_rollout=detach_rollout,
        experiment_mode=experiment_mode,
        state_update_mode=state_update_mode,
        state_loss_source=state_loss_source,
        feedback_state_source=feedback_state_source,
        rollout_signature=ROLLOUT_SIGNATURE,
        lambda_radical=lambda_radical,
        lambda_no=lambda_no,
        use_induction_weights=use_induction_weights,
        induction_boost=induction_boost,
        projection_mode=projection_mode,
        desentinelize_scaler=desentinelize_scaler,
        network_arch=network_arch,
    )

    # Supplying warm_start is explicit intent; do not let the default auto
    # resume silently select a checkpoint from the new experiment directory.
    if warm_start and resume == "auto":
        resume = "none"
    if warm_start and resume != "none":
        raise ValueError(
            "warm_start and resume are mutually exclusive. "
            "Use warm_start to load only model weights, or resume to continue training."
        )

    print("\n" + "#" * 90)
    print(
        "V68 Long-Horizon Species-Autoregressive Operator Training"
    )
    print("#" * 90)
    print(f"Version: {VERSION}")
    transform_signature = transform_signature_for(output_parameterization)
    print(f"Transform: {transform_signature}")
    print(f"Rollout: {ROLLOUT_SIGNATURE}")
    print(f"Max horizon: {max_horizon}")
    print(f"Curriculum horizons: {curriculum_horizons}")
    print(f"Curriculum start epochs: {curriculum_start_epochs}")
    print(f"Eval horizons: {eval_horizons}")
    print(
        f"Tail fraction: {tail_start:.2f} → {tail_end:.2f} "
        f"(ramp={tail_ramp_epochs} epochs)"
    )
    print(f"Backward chunk steps: {backward_chunk_steps}")
    print(f"Parts per epoch: {parts_per_epoch}")
    print(
        f"Validation parts per epoch: "
        f"{val_parts_per_epoch if val_parts_per_epoch is not None else 'all'}"
    )
    print(
        f"Batch size: {batch_size}; "
        f"validation batch size: {val_batch_size if val_batch_size is not None else batch_size}"
    )
    print(
        f"Transition weighting: alpha={transition_weight_alpha}, "
        f"clip={transition_weight_clip}"
    )
    print(f"Feedback state noise std: {feedback_noise_std}")
    print(
        f"Feedback blend: {feedback_blend_start:.2f} -> "
        f"{feedback_blend_end:.2f} over {feedback_blend_ramp_epochs} epochs"
    )
    print(f"Trust-region fraction: {trust_region_fraction:.3f}")
    print(f"Rate decode: {rate_decode_mode} (z_limit={rate_z_limit:.3f})")
    print(f"Temperature mode: {temperature_mode}")
    print(f"Stability loss weight: {lambda_stability}")
    print(f"Experiment mode: {experiment_mode}")
    print(f"State update mode: {state_update_mode}")
    print(f"Resume: {resume}")
    if split_dir:
        print(f"Trajectory split dir: {split_dir}")
    if warm_start:
        print(f"Warm start: {warm_start}")
    print(f"Lambda weights: rate={lambda_rate}, state={lambda_state}, "
          f"element={lambda_element}, mass={lambda_mass}, positive={lambda_positive}, "
          f"rollout={lambda_rollout}, stability={lambda_stability}")

    metadata = load_metadata(
        data_dir,
        mechanism,
    )

    (
        columns,
        species,
        dYdt_cols,
        parquet_files,
    ) = get_schema(
        data_dir,
        mechanism,
    )

    inert_species, macro_species, micro_species = (
        classify_species(
            species,
            include_no_in_macro=(output_parameterization == "dual_branch"),
        )
    )

    feature_info = build_feature_info(
        species,
        inert_species,
        macro_species,
        micro_species,
        output_parameterization=output_parameterization,
        rate_decode_mode=rate_decode_mode,
        rate_z_limit=rate_z_limit,
    )
    expected_feature_info = feature_info.copy()

    # V3.5 uses a fixed, differentiable thermodynamic closure.  If metadata
    # or Cantera is unavailable, remain finite and use recorded temperature.
    thermo_closure = None
    effective_temperature_mode = temperature_mode
    if temperature_mode == "autonomous":
        try:
            thermo_closure = build_temperature_closure(
                metadata,
                feature_info.get("inert_species", []) + feature_info["macro_species"] + feature_info["micro_species"],
                data_dir=data_dir,
                mechanism=mechanism,
            )
            print(
                "[THERMO] Autonomous temperature enabled "
                f"({thermo_closure['energy_kind']}, "
                f"reactor={thermo_closure['reactor_type']}, "
                f"{len(thermo_closure['species_order'])} species incl inert)"
            )
        except Exception as exc:
            effective_temperature_mode = "recorded"
            print(
                "[THERMO] Autonomous temperature unavailable; "
                f"falling back to recorded T: {exc}"
            )
    training_config["effective_temperature_mode"] = effective_temperature_mode

    print(
        "\nSpecies summary:"
    )

    print(
        "  inert =",
        len(inert_species),
    )

    print(
        "  macro =",
        len(macro_species),
    )

    print(
        "  micro =",
        len(micro_species),
    )

    # --------------------------------------------------------
    # Split by complete trajectories when a manifest is available.  The
    # parquet files remain the scan source; sample_ids enforce disjoint
    # train/validation rows and keep scaler statistics train-only.
    # --------------------------------------------------------

    resolved_split_dir = resolve_trajectory_split_dir(data_dir, split_dir)
    train_sample_ids = None
    val_sample_ids = None
    if resolved_split_dir is not None:
        train_sample_ids = load_trajectory_sample_ids(
            resolved_split_dir, mechanism, "train"
        )
        val_sample_ids = load_trajectory_sample_ids(
            resolved_split_dir, mechanism, "validation"
        )
        train_parts = list(parquet_files)
        val_parts = list(parquet_files)
        print(
            f"\n[Split] Trajectory manifest: {resolved_split_dir}\n"
            f"  train sample_ids = {len(train_sample_ids):,}\n"
            f"  validation sample_ids = {len(val_sample_ids):,}"
        )
        training_config["trajectory_split_dir"] = str(resolved_split_dir)
        training_config["trajectory_split_train_signature"] = (
            trajectory_split_signature(resolved_split_dir, mechanism, "train")
        )
        training_config["trajectory_split_validation_signature"] = (
            trajectory_split_signature(resolved_split_dir, mechanism, "validation")
        )
    else:
        print("\n[Split] Verifying that sample_id does not cross parquet parts...")
        try:
            validate_part_sample_isolation(parquet_files)
        except Exception as exc:
            print(f"  [Split Notice] {exc}")

        rng = np.random.default_rng(SEED)
        shuffled = list(parquet_files)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            val_parts = list(shuffled)
            train_parts = list(shuffled)
            print("  [Split Notice] 仅发现 1 个 parquet 分卷且未指定 split_dir；复用该分卷兼作训练与验证。")
        else:
            n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio > 0 else 0
            if n_val >= len(shuffled):
                n_val = max(1, len(shuffled) - 1)
            val_parts = shuffled[:n_val]
            train_parts = shuffled[n_val:]

    print("\nDataset split:")
    print(
        f"  total parts = "
        f"{len(parquet_files)}"
    )
    print(
        f"  train parts = "
        f"{len(train_parts)}"
    )
    print(
        f"  val parts   = "
        f"{len(val_parts)}"
    )

    # Keep the historical behavior when no validation sampling limit is given:
    # use every validation part. This must be resolved after the split exists.
    if val_parts_per_epoch is None:
        val_parts_per_epoch = len(val_parts)

    training_config.update(
        {
            "parts_per_epoch": int(parts_per_epoch),
            "val_parts_per_epoch": int(val_parts_per_epoch),
            "batch_size": int(batch_size),
            "val_batch_size": int(val_batch_size),
        }
    )

    if not train_parts:
        raise RuntimeError(
            "No train parts."
        )

    # --------------------------------------------------------
    # scaler
    # --------------------------------------------------------

    scaler_split_tag = ""
    if resolved_split_dir is not None:
        split_sig = trajectory_split_signature(
            resolved_split_dir, mechanism, "train"
        ).split(":", 1)[-1]
        scaler_split_tag = f"_split_{split_sig}"
    if desentinelize_scaler:
        scaler_split_tag = f"{scaler_split_tag}_desentinelized"

    scaler_dir = (
        Path(output_root)
        / artifact_subdir("scalers")
    )

    scaler_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sx_path = (
        scaler_dir
        / f"scaler_X_{mechanism}_{transform_signature}{scaler_split_tag}.pkl"
    )

    sy_path = (
        scaler_dir
        / f"scaler_Y_{mechanism}_{transform_signature}{scaler_split_tag}.pkl"
    )

    fi_path = (
        scaler_dir
        / f"feature_info_{mechanism}_{transform_signature}{scaler_split_tag}.pkl"
    )

    if (
        sx_path.exists()
        and sy_path.exists()
        and fi_path.exists()
    ):

        print(
            "\nLoading existing V68 scalers..."
        )

        scaler_X = joblib.load(
            sx_path
        )

        scaler_Y = joblib.load(
            sy_path
        )

        feature_info = joblib.load(
            fi_path
        )

        if feature_info.get("version") != VERSION:
            raise RuntimeError(
                f"Refusing incompatible scaler version: "
                f"{feature_info.get('version')!r}"
            )
        if feature_info.get("transform_signature") != transform_signature:
            raise RuntimeError("Refusing incompatible scaler transform")
        validate_loaded_feature_info(feature_info, expected_feature_info)
        loaded_rate_decode_mode = feature_info.get("rate_decode_mode", "direct_rate")
        loaded_rate_z_limit = float(
            feature_info.get("rate_z_limit", DEFAULT_RATE_Z_LIMIT)
        )
        if loaded_rate_decode_mode != rate_decode_mode or not np.isclose(
            loaded_rate_z_limit, rate_z_limit
        ):
            raise RuntimeError(
                "Loaded scaler rate decode metadata does not match training configuration"
            )

    else:

        scaler_X, scaler_Y = (
            compute_streaming_scalers(
                data_dir,
                mechanism,
                feature_info,
                train_parts,
                max_parts=scaler_parts,
                sample_rate=scaler_sample_rate,
                max_rows=scaler_max_rows,
                sample_ids=train_sample_ids,
                desentinelize_scaler=desentinelize_scaler,
            )
        )

        joblib.dump(
            scaler_X,
            sx_path,
        )

        joblib.dump(
            scaler_Y,
            sy_path,
        )

        joblib.dump(
            feature_info,
            fi_path,
        )

    # --------------------------------------------------------
    # dataset
    # --------------------------------------------------------

    train_dataset = CoarseOperatorDataset(
        train_parts,
        feature_info,
        scaler_X,
        scaler_Y,
        max_horizon=max_horizon,  # V68
        shuffle_parts=True,
        stride=train_stride,  # V68
        seed=SEED,
        parts_per_epoch=parts_per_epoch,
        sample_ids=train_sample_ids,
    )

    val_dataset = CoarseOperatorDataset(
        val_parts,
        feature_info,
        scaler_X,
        scaler_Y,
        max_horizon=max_horizon,  # V68
        shuffle_parts=True,
        stride=val_stride,  # V68
        seed=SEED,
        parts_per_epoch=val_parts_per_epoch,
        sample_ids=val_sample_ids,
    )

    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")
    if parts_per_epoch < 1:
        raise ValueError("parts_per_epoch must be >= 1")
    if val_parts_per_epoch < 1:
        raise ValueError("val_parts_per_epoch must be >= 1")
    if val_batch_size < 1:
        raise ValueError("val_batch_size must be >= 1")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # model/tools
    # --------------------------------------------------------

    D_X = len(
        feature_info["input_cols"]
    )

    D_Y = len(
        feature_info["output_cols"]
    )

    if network_arch == "resmlp":
        from dual_branch_network import CoupledKineticsResMLP
        model = CoupledKineticsResMLP(
            D_X,
            D_Y,
            hidden_dim=512,
            num_blocks=4,
            output_parameterization=output_parameterization,
        ).to(DEVICE)
        print(f"Instantiated ResMLP Backbone model (dim={D_X}->512x4->{D_Y}) on {DEVICE}")
    elif network_arch == "dual_branch":
        from dual_branch_network import DualBranchKineticsNet
        slow_species = feature_info["macro_species"]
        fast_species = feature_info["micro_species"]
        slow_dim = len(slow_species)
        fast_dim = len(fast_species)

        model = DualBranchKineticsNet(
            input_dim=D_X,
            slow_dim=slow_dim,
            fast_dim=fast_dim,
            hidden_dim=512,
            num_blocks=4,
            output_parameterization=output_parameterization,
            use_null_space=False,
            element_matrix_slow=None,
        ).to(DEVICE)
        print(f"Instantiated DualBranchKineticsNet (dim={D_X}->512x4->slow={slow_dim}+fast={2*fast_dim}={D_Y}) on {DEVICE}")
    else:
        model = CoupledKineticsNet(
            D_X,
            D_Y,
            output_parameterization=output_parameterization,
        ).to(DEVICE)
        print(f"Instantiated baseline CoupledKineticsNet (dim={D_X}->{D_Y}) on {DEVICE}")

    tools = CoarseOperatorTools(
        feature_info,
        scaler_X,
        scaler_Y,
        rate_decode_mode=rate_decode_mode,
        rate_z_limit=rate_z_limit,
        micro_rate_z_limit=micro_rate_z_limit,
    )
    tools.projection_mode = projection_mode

    species_loss_weights, species_loss_metadata = build_species_loss_weights(
        feature_info,
        macro_weight=macro_loss_weight,
        micro_weight=micro_loss_weight,
        key_species_weight=key_species_loss_weight,
    )
    species_loss_weights = species_loss_weights.to(DEVICE)
    if output_parameterization in ("dual_branch", "dual_branch_pd_v1"):
        macro_w = species_loss_weights[:len(feature_info["macro_species"])]
        micro_w = species_loss_weights[len(feature_info["macro_species"]):].repeat_interleave(2)
        rate_feature_weights = torch.cat([macro_w, micro_w], dim=0)
    elif D_Y == 2 * len(species_loss_weights) or output_parameterization in ("qssa", "qssa_asymptotic"):
        rate_feature_weights = species_loss_weights.repeat_interleave(2)
    else:
        rate_feature_weights = species_loss_weights
    state_feature_weights = species_loss_weights
    training_config["species_loss_metadata"] = species_loss_metadata
    scaler_hashes = {
        "scaler_X": hashlib.sha256(scaler_X.scale_.tobytes() + scaler_X.mean_.tobytes()).hexdigest(),
        "scaler_Y": hashlib.sha256(scaler_Y.scale_.tobytes() + scaler_Y.mean_.tobytes()).hexdigest(),
    }
    chem_contract = {
        "mechanism_name": mechanism,
        "mechanism_sha256": getattr(tools, "mech_sha256", "unverified"),
        "species_order": feature_info["all_species"],
        "element_names": ["H", "N", "O", "AR"],
    }
    experiment_manifest_data = em.create_experiment_manifest(
        mechanism_contract=chem_contract,
        output_parameterization=output_parameterization,
        transform_signature=transform_signature,
        dt_policy={"min": float(feature_info.get("dt_min", 1e-8)), "max": float(feature_info.get("dt_max", 2e-4))},
        concentration_policy={"floor": float(LOG_EPS)},
        evaluation_protocol_id=svp.PROTOCOL_ID,
        scaler_hashes=scaler_hashes,
    )
    feature_info["manifest"] = experiment_manifest_data
    training_config["manifest"] = experiment_manifest_data
    print(
        "Species loss weights: "
        f"macro={macro_loss_weight}, micro={micro_loss_weight}, "
        f"key={key_species_loss_weight}; "
        f"present={species_loss_metadata['key_species_present']}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=7,
            factor=0.5,
        )
    )

    weights_dir = (
        Path(output_root)
        / artifact_subdir("weights")
    )

    weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        Path(output_root)
        / artifact_subdir("checkpoints")
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    # V68: 多指标最佳模型跟踪
    best_oracle_h128 = float("inf")
    best_oracle_avg = float("inf")
    best_pure_ar_h128 = float("inf")
    best_composite = float("inf")
    best_robust_raw = float("inf")

    patience_counter = 0

    history = []
    start_epoch = 0

    # ========================================================
    # V68 断点续训逻辑：支持完整 RNG 状态恢复
    # ========================================================
    resume_path = None

    if resume == "auto":
        latest_path = weights_dir / f"latest_v68_{mechanism}.pth"
        if latest_path.exists():
            resume_path = latest_path
    elif resume != "none":
        resume_path = Path(resume)

    if resume_path and resume_path.exists():
        print(f"\n[RESUME] Loading checkpoint: {resume_path.name}")

        try:
            ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)

            # V68 签名校验
            resume_config_keys = {
                key: training_config[key]
                for key in (
                    "output_parameterization",
                    "rate_decode_mode",
                    "rate_z_limit",
                    "micro_rate_z_limit",
                    "state_update_mode",
                    "state_loss_source",
                    "feedback_state_source",
                    "lambda_element",
                    "lambda_mass",
                    "lambda_positive",
                    "lambda_rollout",
                    "lambda_stability",
                    "trust_region_fraction",
                    "feedback_noise_std",
                    "detach_rollout",
                )
            }
            if not checkpoint_compatible(
                ckpt,
                VERSION,
                transform_signature,
                ROLLOUT_SIGNATURE,
                expected_training_config=resume_config_keys,
            ):
                print(f"[WARNING] Checkpoint incompatible with V68.")
                print(f"  Expected: {VERSION}/{transform_signature}/{ROLLOUT_SIGNATURE}")
                print(f"  Found: {ckpt.get('version')}/{ckpt.get('transform_signature')}/{ckpt.get('rollout_signature')}")
                print("  Training configuration differs or is missing; optimizer/RNG state will not be resumed.")
                print("[WARNING] Starting training from scratch.")
            else:
                model_state = ckpt["model_state_dict"]
                legacy_projection = (
                    network_arch == "dual_branch"
                    and "P_null" in model_state
                    and getattr(model, "P_null", None) is None
                )
                if network_arch == "dual_branch":
                    from dual_branch_network import load_dual_branch_state_dict
                    load_dual_branch_state_dict(model, model_state)
                else:
                    model.load_state_dict(model_state)

                if legacy_projection:
                    # Folding P_null changes optimizer coordinates; retain weights
                    # but initialize optimizer, scheduler, and horizon curriculum anew.
                    print("[RESUME] Legacy P_null folded into slow-head weights.")
                    print("[RESUME] Optimizer state is reset; restarting curriculum at epoch 0.")
                    start_epoch = 0
                    best_val_loss = float("inf")
                    patience_counter = 0
                    history = []
                    best_oracle_h128 = float("inf")
                    best_oracle_avg = float("inf")
                    best_pure_ar_h128 = float("inf")
                    best_composite = float("inf")
                    best_robust_raw = float("inf")
                else:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

                    start_epoch = ckpt["epoch"] + 1
                    best_val_loss = ckpt["best_val_loss"]
                    patience_counter = ckpt.get("patience_counter", 0)
                    history = ckpt.get("history", [])

                    # V68: 恢复多指标最佳值（如果存在）
                    best_oracle_h128 = ckpt.get("best_oracle_h128", float("inf"))
                    best_oracle_avg = ckpt.get("best_oracle_avg", float("inf"))
                    best_pure_ar_h128 = ckpt.get("best_pure_ar_h128", float("inf"))
                    best_composite = ckpt.get("best_composite", float("inf"))
                    best_robust_raw = ckpt.get("best_robust_raw", float("inf"))

                    # V68: 恢复完整 RNG 状态
                    if "rng_states" in ckpt:
                        random.setstate(ckpt["rng_states"]["python"])
                        np.random.set_state(ckpt["rng_states"]["numpy"])
                        torch.set_rng_state(ckpt["rng_states"]["torch"])
                        cuda_states = ckpt["rng_states"].get("cuda")
                        cuda_restored = cuda_states is None or restore_cuda_rng_state(cuda_states)
                        if cuda_states is not None and not cuda_restored:
                            print(
                                "[RESUME] CUDA RNG state skipped: checkpoint/device count mismatch."
                            )
                        print(
                            "[RESUME] RNG states restored."
                            if cuda_restored
                            else "[RESUME] CPU RNG states restored; CUDA RNG skipped."
                        )

                    print(f"[RESUME] Resuming from epoch {start_epoch}")
                    print(f"[RESUME] Best val loss: {best_val_loss:.4f}")
                    print(f"[RESUME] History length: {len(history)}")

        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print("[WARNING] Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float("inf")
            patience_counter = 0
            history = []

    # ========================================================
    # V68 暖机启动逻辑：只加载模型权重，不加载优化器/scheduler/history
    # ========================================================
    if warm_start and not resume_path:
        warm_start_path = Path(warm_start)
        if warm_start_path.exists():
            print(f"\n[WARM START] Loading pretrained weights from: {warm_start_path.name}")

            try:
                ckpt = torch.load(warm_start_path, map_location=DEVICE, weights_only=False)

                if ckpt.get("version") != VERSION:
                    raise RuntimeError(
                        f"Warm-start version mismatch: {ckpt.get('version')!r}"
                    )
                if ckpt.get("transform_signature") != transform_signature:
                    raise RuntimeError(
                        f"Warm-start transform mismatch: expected {transform_signature!r}, got {ckpt.get('transform_signature')!r}"
                    )
                if "feature_info" in ckpt:
                    validate_loaded_feature_info(
                        ckpt["feature_info"],
                        expected_feature_info,
                    )

                # 只加载模型权重
                if network_arch == "dual_branch":
                    from dual_branch_network import load_dual_branch_state_dict
                    load_dual_branch_state_dict(model, ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt["model_state_dict"])

                print("[WARM START] Model weights loaded successfully.")
                print("[WARM START] Optimizer, scheduler, and history are freshly initialized.")
                print("[WARM START] Training will start from epoch 0 with pretrained weights.")

                # 不加载 epoch/optimizer/scheduler/history，保持初始状态
                # start_epoch 保持为 0
                # optimizer, scheduler 使用新创建的实例
                # best_val_loss 保持为 inf
                # history 保持为空列表

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load warm-start checkpoint {warm_start_path}: {e}"
                ) from e
        else:
            raise FileNotFoundError(
                f"Warm-start checkpoint does not exist: {warm_start_path}"
            )

    tracker = td.TrainingDiagnosticTracker(
        snapshot_dir=weights_dir.parent / "diagnostics",
        max_skip_budget=50,
        debug_mode=False,
    )
    sentinel_tracker = svp.HorizonSentinelTracker()
    best_val_loss_by_horizon = {}
    previous_horizon = None

    # ========================================================
    # V68 训练循环：horizon curriculum + 连续预测尾段
    # ========================================================
    for epoch in range(start_epoch, epochs):

        # 当前 epoch 的 active horizon
        active_horizon = horizon_for_epoch(epoch, horizon_schedule)
        if previous_horizon is not None and active_horizon != previous_horizon:
            print(f"\n[CURRICULUM] Transitioning horizon from {previous_horizon} to {active_horizon}. Resetting patience counter.")
            patience_counter = 0
        previous_horizon = active_horizon

        # The epoch is relative to this invocation. Warm-start runs therefore
        # continue from the mature AR fraction instead of restarting low.
        tail_fraction = scheduled_tail_fraction(
            epoch,
            tail_start,
            tail_end,
            tail_ramp_epochs,
        )
        feedback_strength = scheduled_feedback_strength(
            epoch,
            feedback_blend_start,
            feedback_blend_end,
            feedback_blend_ramp_epochs,
        )

        train_dataset.set_epoch(epoch)
        # Validation samples are re-selected deterministically per epoch. A
        # fixed epoch would repeatedly validate the same leading files.
        val_dataset.set_epoch(epoch)

        model.train()

        train_stats = {
            "total": 0.0,
            "rate": 0.0,
            "state": 0.0,
            "element": 0.0,
            "mass": 0.0,
            "positive": 0.0,
            "rollout": 0.0,
            "stability": 0.0,
            "transition_weight": 0.0,
            "radical": 0.0,
            "no": 0.0,
        }
        transition_audit = initialize_transition_audit()
        audit_weight_values = []

        n_train_batches = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/{epochs} "
                f"H={active_horizon} "
                f"tail={tail_fraction:.2f}"
            ),
            ncols=120,
        )

        for batch_X, batch_Y in pbar:
            attempted_batch_idx = tracker.step_attempted()

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True,
            )

            batch_Y = batch_Y.to(
                DEVICE,
                non_blocking=True,
            )

            # V68: 截取到 active_horizon
            B = batch_X.size(0)
            actual_H = min(active_horizon, batch_X.size(1) - 1)
            batch_X = batch_X[:, :actual_H + 1, :]
            batch_Y = batch_Y[:, :actual_H, :]

            # Weight difficult physical transitions (especially ignition
            # windows) without changing the standardized target contract.
            if transition_weight_alpha > 0.0:
                true_states_physical = tools.inverse_state(batch_X)
                true_macro_states = true_states_physical[:, :, tools.macro_idx]
                macro_deltas = true_macro_states[:, 1:, :] - true_macro_states[:, :-1, :]
                # IMPORTANT: Use dt from physical states, not from standardized batch_X
                dt_steps = torch.pow(10.0, true_states_physical[:, :-1, tools.dt_idx]).unsqueeze(-1)
                true_rates_physical = macro_deltas / dt_steps.clamp_min(1e-12)
                transition_weights = compute_transition_weights(
                    true_rates_physical,
                    macro_deltas,
                    alpha=transition_weight_alpha,
                    clip=transition_weight_clip,
                )
            else:
                transition_weights = None

            # V68: 生成每个样本的连续预测尾段长度
            tail_min = max(1, int(actual_H * tail_fraction * 0.8))
            tail_max = max(tail_min, int(actual_H * tail_fraction))
            tail_lengths = torch.randint(
                tail_min,
                tail_max + 1,
                (B,),
                device=DEVICE,
            )
            use_pred_mask = build_prediction_tail_mask(B, actual_H, tail_lengths)

            current_X = batch_X[:, 0, :]

            optimizer.zero_grad(
                set_to_none=True
            )

            batch_total = 0.0
            batch_valid = True
            chunk_loss = None
            chunk_steps = 0

            local_stats = {
                "rate": 0.0,
                "state": 0.0,
                "element": 0.0,
                "mass": 0.0,
                "positive": 0.0,
                "rollout": 0.0,
                "stability": 0.0,
                "transition_weight": 0.0,
                "radical": 0.0,
                "no": 0.0,
            }
            local_transition_audit = initialize_transition_audit()
            local_audit_weight_values = []

            # Use the true initial state as the anchor for increment loss;
            # subsequent anchors follow the predicted trajectory and are
            # detached at the same chunk boundaries as the rollout graph.
            previous_pred_state_scaled = state_features_scaled_from_input(
                batch_X[:, 0, :], tools
            )
            previous_true_state_scaled = previous_pred_state_scaled.detach()
            previous_pred_rate_scaled = None
            previous_true_rate_scaled = None

            for step in range(actual_H):

                # ----------------------------------------------
                # network & disentangled parameter supervision
                # ----------------------------------------------
                true_current_X = batch_X[:, step, :]
                is_true_input = (step == 0) or torch.equal(current_X, true_current_X)

                if is_true_input:
                    pred_rate_scaled = model(current_X)
                    rate_pred_true = pred_rate_scaled
                else:
                    pred_rate_scaled = model(current_X)
                    rate_pred_true = model(true_current_X)

                # ----------------------------------------------
                # next state
                # ----------------------------------------------

                pred_state = (
                    tools.predict_next_state(
                        current_X,
                        pred_rate_scaled,
                        safe_for_rollout=(feedback_state_source != "raw"),
                        experiment_mode=experiment_mode,
                        trust_region_fraction=trust_region_fraction,
                        state_update_mode=state_update_mode,
                    )
                )

                # ✅ 修复：True endpoint of this transition
                #
                # Dataset window now contains t0...t50 (51 points)
                # So all 50 transitions have their endpoints:
                # step 0: t0 -> t1
                # step 1: t1 -> t2
                # ...
                # step 49: t49 -> t50 ✅ t50 在窗口中！
                #
                # 所有 50 步都计算 state loss

                true_next_X = (
                    batch_X[
                        :,
                        step + 1,
                        :
                    ]
                )
                step_weights = transition_weights[:, step] if transition_weights is not None else None
                true_cur_X_unscaled = true_current_X * tools.scale_X + tools.mean_X
                true_nxt_X_unscaled = true_next_X * tools.scale_X + tools.mean_X
                if use_induction_weights:
                    from enhanced_loss_and_weights import compute_induction_sample_weights
                    ind_w = compute_induction_sample_weights(
                        true_cur_X_unscaled, true_nxt_X_unscaled, tools, boost_factor=induction_boost
                    )
                    step_weights = ind_w if step_weights is None else step_weights * ind_w

                # V68 targets are already standardized
                true_rate_scaled = batch_Y[:, step, :]
                rate_mask = compute_qssa_rate_mask(
                    tools,
                    true_rate_scaled,
                    Y_old_unscaled=true_cur_X_unscaled,
                    Y_next_unscaled=true_nxt_X_unscaled,
                )
                if rate_loss_kind == "mse":
                    rate_loss = weighted_mse_loss(
                        rate_pred_true,
                        true_rate_scaled,
                        sample_weights=step_weights,
                        feature_weights=rate_feature_weights,
                        mask=rate_mask,
                    )
                else:
                    rate_loss = weighted_smooth_l1_loss(
                        rate_pred_true,
                        true_rate_scaled,
                        sample_weights=step_weights,
                        feature_weights=rate_feature_weights,
                        mask=rate_mask,
                        beta=rate_huber_beta,
                    )
                # ----------------------------------------------
                # integrated state loss
                # ✅ 修复：所有 50 步都计算 state loss
                # ----------------------------------------------

                train_macro, train_micro_log = select_training_state(
                    pred_state, state_loss_source
                )

                pred_macro_scaled = (
                    (
                        train_macro
                        - tools.mean_X[
                            tools.macro_idx
                        ]
                    )
                    / tools.scale_X[
                        tools.macro_idx
                    ]
                )

                pred_micro_log_scaled = (
                    (
                        train_micro_log
                        - tools.mean_X[
                            tools.micro_idx
                        ]
                    )
                    / tools.scale_X[
                        tools.micro_idx
                    ]
                )

                pred_state_scaled = (
                    torch.cat(
                        [
                            pred_macro_scaled,
                            pred_micro_log_scaled,
                        ],
                        dim=1,
                    )
                )

                true_macro_scaled = (
                    true_next_X[
                        :,
                        tools.macro_idx
                    ]
                )

                true_micro_scaled = (
                    true_next_X[
                        :,
                        tools.micro_idx
                    ]
                )

                true_state_scaled = (
                    torch.cat(
                        [
                            true_macro_scaled,
                            true_micro_scaled,
                        ],
                        dim=1,
                    )
                )

                pred_state_scaled = state_loss_features(
                    pred_state_scaled,
                    state_loss_source,
                )
                true_state_scaled = state_loss_features(
                    true_state_scaled,
                    state_loss_source,
                )

                if lambda_stability > 0.0:
                    if state_update_mode in ("qssa", "qssa_asymptotic"):
                        if (
                            previous_pred_rate_scaled is not None
                            and previous_true_rate_scaled is not None
                        ):
                            stability_loss = compute_rate_stability_loss(
                                qssa_log_rate_features_from_scaled(
                                    torch.stack(
                                        [previous_pred_rate_scaled, pred_rate_scaled], dim=1
                                    ),
                                    tools,
                                ),
                                qssa_log_rate_features_from_scaled(
                                    torch.stack(
                                        [previous_true_rate_scaled, true_rate_scaled], dim=1
                                    ),
                                    tools,
                                ),
                            )
                        else:
                            stability_loss = pred_state_scaled.sum() * 0.0
                    else:
                        stability_loss = compute_stability_loss(
                            torch.stack(
                                [previous_pred_state_scaled, pred_state_scaled], dim=1
                            ),
                            torch.stack(
                                [previous_true_state_scaled, true_state_scaled], dim=1
                            ),
                        )
                else:
                    stability_loss = pred_state_scaled.sum() * 0.0

                state_loss = weighted_mse_loss(
                    pred_state_scaled,
                    true_state_scaled,
                    step_weights,
                    species_loss_weights,
                    stable_accumulation=(state_loss_source == "raw"),
                )

                audit_weights = update_transition_audit(
                    local_transition_audit,
                    step_weights,
                    pred_rate_scaled,
                    true_rate_scaled,
                    pred_state_scaled,
                    true_state_scaled,
                )
                local_audit_weight_values.append(audit_weights.cpu())

                # ----------------------------------------------
                # physics
                # ----------------------------------------------

                loss_element, loss_mass = (
                    compute_element_conservation_loss(
                        tools,
                        pred_state,
                        state_loss_source=state_loss_source,
                    )
                )

                loss_positive = (
                    compute_positivity_loss(
                        pred_state
                    )
                )

                loss_radical = torch.tensor(0.0, device=rate_loss.device)
                if lambda_radical > 0.0:
                    from enhanced_loss_and_weights import compute_physical_radical_loss
                    loss_radical = compute_physical_radical_loss(
                        tools=tools,
                        pred_state=pred_state,
                        true_next_X_scaled=true_next_X,
                        state_source=state_loss_source,
                    )

                loss_no = torch.tensor(0.0, device=rate_loss.device)
                if lambda_no > 0.0:
                    from enhanced_loss_and_weights import compute_pollutant_no_loss
                    loss_no = compute_pollutant_no_loss(
                        tools=tools,
                        pred_state=pred_state,
                        true_next_X_scaled=true_next_X,
                        state_source=state_loss_source,
                    )

                # ----------------------------------------------
                # total
                # ----------------------------------------------

                step_loss = (
                    lambda_rate
                    * rate_loss

                    + lambda_state
                    * state_loss

                    + lambda_element
                    * loss_element

                    + lambda_mass
                    * loss_mass

                    + lambda_positive
                    * loss_positive
                    + lambda_stability
                    * stability_loss
                    + lambda_radical
                    * loss_radical
                    + lambda_no
                    * loss_no
                )

                # Accumulate a small detached chunk to reduce backward
                # engine launches while bounding retained activations.
                if not torch.isfinite(step_loss):
                    print(f"DEBUG METRICS: rate={rate_loss}, state={state_loss}, element={loss_element}, mass={loss_mass}, pos={loss_positive}, rad={loss_radical}, no={loss_no}")
                chunk_loss = (
                    step_loss
                    if chunk_loss is None
                    else chunk_loss + step_loss
                )
                chunk_steps += 1
                chunk_completed = False

                if (
                    chunk_steps >= backward_chunk_steps
                    or step == actual_H - 1
                ):
                    step_valid, step_value = backward_loss_chunk(
                        chunk_loss,
                        actual_H,
                    )
                    if not step_valid:
                        print(
                            f"\n[WARNING] Non-finite loss; discarding batch "
                            f"at epoch={epoch + 1}, "
                            f"batch={n_train_batches + 1}, "
                            f"step={step}."
                        )
                        batch_valid = False
                        break
                    batch_total += step_value
                    chunk_loss = None
                    chunk_steps = 0
                    chunk_completed = True

                local_stats[
                    "rate"
                ] += rate_loss.item()

                local_stats[
                    "state"
                ] += state_loss.item()

                local_stats[
                    "element"
                ] += loss_element.item()

                local_stats[
                    "mass"
                ] += loss_mass.item()

                local_stats[
                    "positive"
                ] += loss_positive.item()

                local_stats["stability"] += stability_loss.item()
                local_stats["radical"] += loss_radical.item()
                local_stats["no"] += loss_no.item()

                # Record the effective per-transition weighting so V3.1 can
                # be audited from the training log.  Legacy modes report 1.
                local_stats["transition_weight"] += (
                    step_weights.mean().item()
                    if step_weights is not None
                    else 1.0
                )

                # ----------------------------------------------
                # V68 rollout: 连续预测尾段
                # ----------------------------------------------

                if step < actual_H - 1:

                    base_next_X = (
                        batch_X[
                            :,
                            step + 1,
                            :
                        ]
                    )

                    feed_macro, feed_micro_log = select_feedback_state(
                        pred_state, feedback_state_source
                    )
                    predicted_next_X = (
                        tools.state_to_input_scaled(
                            base_next_X,
                            feed_macro,
                            feed_micro_log,
                            temperature_mode=effective_temperature_mode,
                            thermo_closure=thermo_closure,
                            current_X_scaled=current_X,
                            max_delta_temperature=max_delta_temperature,
                            feedback_input_limit=(
                                RAW_FEEDBACK_INPUT_LIMIT
                                if feedback_state_source == "raw"
                                else None
                            ),
                        )
                    )

                    species_idx = torch.cat(
                        [
                            tools.macro_idx,
                            tools.micro_idx,
                        ]
                    )
                    feedback_idx = species_idx
                    if effective_temperature_mode == "autonomous":
                        # V3.5 predicts T from composition energy change; T
                        # must enter the next feedback state together with Y.
                        feedback_idx = torch.cat(
                            [
                                species_idx,
                                torch.tensor(
                                    [tools.temperature_idx],
                                    device=species_idx.device,
                                    dtype=species_idx.dtype,
                                ),
                            ]
                        )

                    # V68: 连续预测尾段 - 每个样本根据 mask 选择
                    # mask[b, step] = True 表示样本 b 在步 step 使用预测值
                    use_pred = use_pred_mask[:, step].unsqueeze(1)  # (B, 1)

                    selected_next_X = (
                        base_next_X.clone()
                    )

                    selected_next_X[:, feedback_idx] = blend_feedback_state(
                        base_next_X[:, feedback_idx],
                        predicted_next_X[:, feedback_idx],
                        use_pred,
                        feedback_strength,
                    )

                    if feedback_noise_std > 0.0:
                        selected_next_X[:, species_idx] = add_feedback_state_noise(
                            selected_next_X[:, species_idx],
                            feedback_noise_std * tail_fraction,
                            use_pred,
                        )

                    current_X = (
                        selected_next_X
                    )

                    # Keep gradients through predicted feedback within a
                    # chunk; detach only after that chunk was backpropagated.
                    if detach_rollout and chunk_completed:
                        current_X = current_X.detach()

                if chunk_completed:
                    previous_pred_state_scaled = pred_state_scaled.detach()
                    previous_true_state_scaled = true_state_scaled.detach()
                    previous_pred_rate_scaled = pred_rate_scaled.detach()
                    previous_true_rate_scaled = true_rate_scaled.detach()
                else:
                    previous_pred_state_scaled = pred_state_scaled
                    previous_true_state_scaled = true_state_scaled.detach()
                    previous_pred_rate_scaled = pred_rate_scaled
                    previous_true_rate_scaled = true_rate_scaled.detach()

            # ----------------------------------------------------
            # finalize accumulated gradients
            # ----------------------------------------------------

            if not batch_valid:
                # Earlier steps may already have accumulated gradients.
                optimizer.zero_grad(set_to_none=True)
                continue

            rollout_loss = torch.zeros((), device=batch_X.device)
            rollout_breakdown = {}
            if lambda_rollout > 0:
                rollout_loss, rollout_breakdown = compute_training_rollout_loss(
                    model,
                    tools,
                    batch_X,
                    actual_H,
                    experiment_mode,
                    batch_Y=batch_Y,
                    detach_interval=backward_chunk_steps,
                    lambda_stability=lambda_stability,
                    trust_region_fraction=trust_region_fraction,
                    state_update_mode=state_update_mode,
                    state_loss_source=state_loss_source,
                    feedback_state_source=feedback_state_source,
                    temperature_mode=effective_temperature_mode,
                    thermo_closure=thermo_closure,
                    max_delta_temperature=max_delta_temperature,
                    feedback_noise_std=feedback_noise_std * tail_fraction,
                    backward_callback=lambda chunk_loss: (
                        lambda_rollout * chunk_loss
                    ).backward(),
                )
                if not torch.isfinite(rollout_loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # H=1 is the first curriculum stage, while the auxiliary
                # rollout targets begin at H=4.  In that stage the rollout
                # helper intentionally returns a constant zero, so there is
                # no graph to backpropagate through.  The one-step loss above
                # has already supplied the gradients for this batch.
                batch_total += float(lambda_rollout * rollout_loss.detach().item())
                local_stats["rollout"] = rollout_loss.detach().item()
                local_stats["stability"] += (
                    float(lambda_stability) * rollout_breakdown.get("stability", 0.0)
                )

            is_finite, grad_norm_f64, grad_diag = td.check_gradient_finite_and_norm(model)
            if not is_finite:
                print(
                    f"\n[WARNING] Non-finite gradients ({grad_diag.get('non_finite_params')}) "
                    f"at epoch={epoch + 1}, attempted_batch={attempted_batch_idx}."
                )
                tracker.record_failure_and_check_budget(
                    model=model,
                    batch_X=batch_X,
                    batch_Y=batch_Y,
                    epoch=epoch + 1,
                    optimizer=optimizer,
                    error_msg=f"Non-finite gradients: {grad_diag.get('non_finite_params')}",
                    diagnostics=grad_diag,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            tracker.record_success()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            for key in transition_audit:
                if key == "weight_max":
                    transition_audit[key] = max(
                        transition_audit[key],
                        local_transition_audit[key],
                    )
                else:
                    transition_audit[key] += local_transition_audit[key]
            audit_weight_values = merge_transition_weight_samples(
                audit_weight_values,
                local_audit_weight_values,
            )

            # ----------------------------------------------------
            # statistics
            # ----------------------------------------------------

            train_stats["total"] += (
                batch_total
            )

            train_stats["rate"] += (
                local_stats["rate"]
                / actual_H
            )

            train_stats["state"] += (
                local_stats["state"]
                / actual_H
            )

            train_stats["element"] += (
                local_stats["element"]
                / actual_H
            )

            train_stats["mass"] += (
                local_stats["mass"]
                / actual_H
            )

            train_stats["positive"] += (
                local_stats["positive"]
                / actual_H
            )

            train_stats["rollout"] += local_stats["rollout"]

            train_stats["stability"] += (
                local_stats["stability"] / actual_H
            )

            train_stats["transition_weight"] += (
                local_stats["transition_weight"] / actual_H
            )

            train_stats["radical"] += local_stats["radical"] / actual_H
            train_stats["no"] += local_stats["no"] / actual_H

            n_train_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{batch_total:.4f}",
                    "rate": f"{local_stats['rate']/actual_H:.4f}",
                    "state": f"{local_stats['state']/actual_H:.4f}",
                    "roll": f"{local_stats['rollout']:.4f}",
                    "elem": f"{local_stats['element']/actual_H:.2e}",
                    "mass": f"{local_stats['mass']/actual_H:.2e}",
                    "w": f"{local_stats['transition_weight']/actual_H:.2f}",
                }
            )

        if n_train_batches == 0:
            raise RuntimeError(
                "No valid training batches produced."
            )

        for key in train_stats:
            train_stats[key] /= (
                n_train_batches
            )
        transition_audit_summary = finalize_transition_audit(
            transition_audit,
            audit_weight_values,
        )
        train_stats["transition_weight"] = transition_audit_summary["weight_mean"]
        train_stats["transition_audit"] = transition_audit_summary

        # --------------------------------------------------------
        # V68 双模式验证
        # --------------------------------------------------------

        # Oracle one-step validation (主要用于 scheduler 和 early stopping)
        val_stats_oracle = validate_model_v68(
            model,
            val_loader,
            tools,
            DEVICE,
            eval_horizons,
            active_horizon,
            lambda_rate,
            lambda_state,
            lambda_element,
            lambda_mass,
            lambda_positive,
            max_batches=max_val_batches,
            mode="oracle",
            rate_loss_kind=rate_loss_kind,
            rate_huber_beta=rate_huber_beta,
            lambda_radical=lambda_radical,
            lambda_no=lambda_no,
            experiment_mode=experiment_mode,
            trust_region_fraction=trust_region_fraction,
            temperature_mode=effective_temperature_mode,
            thermo_closure=thermo_closure,
            max_delta_temperature=max_delta_temperature,
            rate_feature_weights=rate_feature_weights,
            state_feature_weights=state_feature_weights,
            state_update_mode=state_update_mode,
            state_loss_source=state_loss_source,
            feedback_state_source=feedback_state_source,
        )

        # Pure species-AR validation (用于评估长 horizon 累积误差)
        # 每 full_val_every epoch 运行一次完整的 pure-AR 验证
        if (epoch + 1) % full_val_every == 0:
            val_stats_pure_ar = validate_model_v68(
                model,
                val_loader,
                tools,
                DEVICE,
                eval_horizons,
                active_horizon,
                lambda_rate,
                lambda_state,
                lambda_element,
                lambda_mass,
                lambda_positive,
                max_batches=max_val_batches,
                mode="pure_ar",
                rate_loss_kind=rate_loss_kind,
                rate_huber_beta=rate_huber_beta,
                lambda_radical=lambda_radical,
                lambda_no=lambda_no,
                experiment_mode=experiment_mode,
                trust_region_fraction=trust_region_fraction,
                temperature_mode=effective_temperature_mode,
                thermo_closure=thermo_closure,
                max_delta_temperature=max_delta_temperature,
                rate_feature_weights=rate_feature_weights,
                state_feature_weights=state_feature_weights,
                state_update_mode=state_update_mode,
                state_loss_source=state_loss_source,
                feedback_state_source=feedback_state_source,
            )
        else:
            val_stats_pure_ar = None

        # Oracle validation intentionally monitors the unweighted supervised
        # objective; it excludes train-only transition/induction weights,
        # stability, and rollout losses.
        val_stats = val_stats_oracle
        scheduler.step(
            val_stats["total"]
        )

        # --------------------------------------------------------
        # report
        # --------------------------------------------------------

        lr_now = (
            optimizer.param_groups[0]["lr"]
        )

        print("\n" + "-" * 90)
        print(
            f"Epoch {epoch + 1:3d}"
        )

        print(
            f"  Horizon        : "
            f"{active_horizon}"
        )

        print(
            f"  Tail Fraction  : "
            f"{tail_fraction:.3f}"
        )

        print(
            f"  Learning Rate  : "
            f"{lr_now:.3e}"
        )

        print(
            f"  Train Total    : "
            f"{train_stats['total']:.6e}"
        )

        print(
            f"  Train Rate     : "
            f"{train_stats['rate']:.6e}"
        )

        print(
            f"  Train State    : "
            f"{train_stats['state']:.6e}"
        )

        print(
            f"  Train Rollout  : "
            f"{train_stats['rollout']:.6e}"
        )

        print(
            f"  Train Stability: "
            f"{train_stats['stability']:.6e}"
        )

        print(
            f"  Train Transition Weight: "
            f"{train_stats['transition_weight']:.4f}"
        )

        audit = train_stats["transition_audit"]
        print(
            "  Ignition Audit : "
            f"high={audit['high_activity_fraction']:.2%}, "
            "w(mean/p50/p90/max)="
            f"{audit['weight_mean']:.3f}/"
            f"{audit['weight_p50']:.3f}/"
            f"{audit['weight_p90']:.3f}/"
            f"{audit['weight_max']:.3f}"
        )
        print(
            "  Ignition MSE   : "
            "rate high/normal="
            f"{audit['rate_mse_high_activity'] if audit['rate_mse_high_activity'] is not None else 'n/a'}/"
            f"{audit['rate_mse_normal'] if audit['rate_mse_normal'] is not None else 'n/a'}, "
            "state high/normal="
            f"{audit['state_mse_high_activity'] if audit['state_mse_high_activity'] is not None else 'n/a'}/"
            f"{audit['state_mse_normal'] if audit['state_mse_normal'] is not None else 'n/a'}"
        )

        print(
            f"  Train Element  : "
            f"{train_stats['element']:.6e}"
        )

        print(
            f"  Train Mass     : "
            f"{train_stats['mass']:.6e}"
        )

        print(
            f"  Val Total      : "
            f"{val_stats['total']:.6e}"
        )

        print(
            f"  Val Rate       : "
            f"{val_stats['rate']:.6e}"
        )

        print(
            f"  Val State      : "
            f"{val_stats['state']:.6e}"
        )

        print(
            f"  Val Element    : "
            f"{val_stats['element']:.6e}"
        )

        print(
            f"  Val Mass       : "
            f"{val_stats['mass']:.6e}"
        )

        print(
            f"  Val Positive   : "
            f"{val_stats['positive']:.6e}"
        )

        # V68: Per-horizon oracle 指标
        print("\n  Oracle per-horizon RMSE:")
        oracle_horizons = val_stats["per_horizon"]
        for h_str in sorted(oracle_horizons.keys(), key=lambda x: int(x)):
            h_data = oracle_horizons[h_str]
            if h_data["count"] > 0:
                print(f"    H={h_str:>3s}: {h_data['rmse']:.6e} (n={h_data['count']})")

        # V68: Pure-AR 指标（如果本 epoch 运行了）
        if val_stats_pure_ar is not None:
            print("\n  Pure-AR per-horizon RMSE:")
            pure_ar_horizons = val_stats_pure_ar["per_horizon"]
            for h_str in sorted(pure_ar_horizons.keys(), key=lambda x: int(x)):
                h_data = pure_ar_horizons[h_str]
                if h_data["count"] > 0:
                    print(f"    H={h_str:>3s}: {h_data['rmse']:.6e} (n={h_data['count']})")

        print("-" * 90)

        # --------------------------------------------------------
        # history
        # --------------------------------------------------------

        history_entry = {
            "epoch": epoch + 1,
            "horizon": active_horizon,
            "tail_fraction": tail_fraction,
            "lr": lr_now,
            "train": train_stats.copy(),
            "validation_oracle": {
                "total": val_stats["total"],
                "rate": val_stats["rate"],
                "state": val_stats["state"],
                "element": val_stats["element"],
                "mass": val_stats["mass"],
                "positive": val_stats["positive"],
                "per_horizon": val_stats["per_horizon"],
            },
        }

        if val_stats_pure_ar is not None:
            history_entry["validation_pure_ar"] = {
                "total": val_stats_pure_ar["total"],
                "rate": val_stats_pure_ar["rate"],
                "state": val_stats_pure_ar["state"],
                "element": val_stats_pure_ar["element"],
                "mass": val_stats_pure_ar["mass"],
                "positive": val_stats_pure_ar["positive"],
                "per_horizon": val_stats_pure_ar["per_horizon"],
            }

        history.append(history_entry)

        # --------------------------------------------------------
        # save best - 原有总损失模型
        # --------------------------------------------------------

        # --------------------------------------------------------
        # save best - Fixed Horizon Sentinel Tracking & Model Selection
        # --------------------------------------------------------
        if val_stats_pure_ar is not None and "sentinel_metrics" in val_stats_pure_ar:
            for h_int in eval_horizons:
                h_str = str(h_int)
                if h_str in val_stats_pure_ar["sentinel_metrics"]:
                    h_metrics = val_stats_pure_ar["sentinel_metrics"][h_str]
                    if sentinel_tracker.consider_candidate(h_int, h_metrics, epoch + 1):
                        h_path = weights_dir / f"best_v68_net_{mechanism}_H{h_int}.pth"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "mechanism": mechanism,
                                "feature_info": feature_info,
                                "version": VERSION,
                                "transform_signature": transform_signature,
                                "rollout_signature": ROLLOUT_SIGNATURE,
                                "training_config": training_config,
                                "manifest": experiment_manifest_data,
                                "best_val_loss": val_stats["total"],
                                "sentinel_score": h_metrics.get("composite_score"),
                                "sentinel_metrics": h_metrics,
                                "epoch": epoch + 1,
                                "horizon": h_int,
                            },
                            h_path,
                        )
                        print(f"[SENTINEL] New best valid H={h_int} model -> {h_path.name} (score={h_metrics.get('composite_score'):.6e})")

        horizon_best_loss = best_val_loss_by_horizon.get(active_horizon, float("inf"))
        improved_horizon = val_stats["total"] < horizon_best_loss
        if improved_horizon:
            best_val_loss_by_horizon[active_horizon] = val_stats["total"]
            best_val_loss = val_stats["total"]
            patience_counter = 0

            # Update overall benchmark model if this is a mature horizon (H >= 32 or max_horizon)
            if active_horizon >= 32 or active_horizon == max_horizon:
                best_path = weights_dir / f"best_v68_net_{mechanism}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "mechanism": mechanism,
                        "feature_info": feature_info,
                        "version": VERSION,
                        "transform_signature": transform_signature,
                        "rollout_signature": ROLLOUT_SIGNATURE,
                        "training_config": training_config,
                        "manifest": experiment_manifest_data,
                        "best_val_loss": best_val_loss,
                        "epoch": epoch + 1,
                        "horizon": active_horizon,
                    },
                    best_path,
                )
                print(f"[SAVE] Best mature model -> {best_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print("\nEarly stopping.")
                break

        # --------------------------------------------------------
        # V68 多指标最佳模型保存
        # --------------------------------------------------------

        # 1. Best Oracle H=128
        if "128" in val_stats["per_horizon"] and val_stats["per_horizon"]["128"]["count"] > 0:
            oracle_h128_rmse = val_stats["per_horizon"]["128"]["rmse"]
            if oracle_h128_rmse < best_oracle_h128:
                best_oracle_h128 = oracle_h128_rmse
                best_oracle_h128_path = weights_dir / f"best_oracle_h128_epoch{epoch+1}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "mechanism": mechanism,
                        "feature_info": feature_info,
                        "version": VERSION,
                        "transform_signature": transform_signature,
                        "rollout_signature": ROLLOUT_SIGNATURE,
                        "training_config": training_config,
                        "metric_name": "oracle_h128_rmse",
                        "metric_value": best_oracle_h128,
                        "epoch": epoch + 1,
                        "all_metrics": val_stats.copy(),
                    },
                    best_oracle_h128_path,
                )
                print(f"[SAVE] Best Oracle H=128 model -> {best_oracle_h128_path.name} (RMSE={best_oracle_h128:.6e})")

        # 2. Best Oracle Average (所有horizon平均)
        oracle_rmse_list = []
        for h_str, h_data in val_stats["per_horizon"].items():
            if h_data["count"] > 0:
                oracle_rmse_list.append(h_data["rmse"])
        if oracle_rmse_list:
            oracle_avg_rmse = sum(oracle_rmse_list) / len(oracle_rmse_list)
            if oracle_avg_rmse < best_oracle_avg:
                best_oracle_avg = oracle_avg_rmse
                best_oracle_avg_path = weights_dir / f"best_oracle_avg_epoch{epoch+1}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "mechanism": mechanism,
                        "feature_info": feature_info,
                        "version": VERSION,
                        "transform_signature": transform_signature,
                        "rollout_signature": ROLLOUT_SIGNATURE,
                        "training_config": training_config,
                        "metric_name": "oracle_avg_rmse",
                        "metric_value": best_oracle_avg,
                        "epoch": epoch + 1,
                        "all_metrics": val_stats.copy(),
                    },
                    best_oracle_avg_path,
                )
                print(f"[SAVE] Best Oracle Average model -> {best_oracle_avg_path.name} (Avg RMSE={best_oracle_avg:.6e})")

        # 3. Best Pure-AR H=128 (只在有Pure-AR数据时)
        if val_stats_pure_ar is not None:
            if "128" in val_stats_pure_ar["per_horizon"] and val_stats_pure_ar["per_horizon"]["128"]["count"] > 0:
                pure_ar_h128_rmse = val_stats_pure_ar["per_horizon"]["128"]["rmse"]
                if pure_ar_h128_rmse < best_pure_ar_h128:
                    best_pure_ar_h128 = pure_ar_h128_rmse
                    best_pure_ar_h128_path = weights_dir / f"best_pure_ar_h128_epoch{epoch+1}.pth"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "mechanism": mechanism,
                            "feature_info": feature_info,
                            "version": VERSION,
                            "transform_signature": transform_signature,
                            "rollout_signature": ROLLOUT_SIGNATURE,
                            "training_config": training_config,
                            "metric_name": "pure_ar_h128_rmse",
                            "metric_value": best_pure_ar_h128,
                            "epoch": epoch + 1,
                            "all_metrics": {
                                "oracle": val_stats.copy(),
                                "pure_ar": val_stats_pure_ar.copy(),
                            },
                        },
                        best_pure_ar_h128_path,
                    )
                    print(f"[SAVE] Best Pure-AR H=128 model -> {best_pure_ar_h128_path.name} (RMSE={best_pure_ar_h128:.6e})")

            # 4. Best Composite (组合指标: 0.2*Oracle + 0.8*PureAR, 自适应使用当前最大可用 Horizon)
            target_h_key = None
            if val_stats_pure_ar is not None:
                for hk in [str(h) for h in sorted(eval_horizons, reverse=True)]:
                    if (hk in val_stats["per_horizon"] and val_stats["per_horizon"][hk]["count"] > 0 and
                        hk in val_stats_pure_ar["per_horizon"] and val_stats_pure_ar["per_horizon"][hk]["count"] > 0):
                        target_h_key = hk
                        break

            if target_h_key is not None:
                oracle_h_for_composite = val_stats["per_horizon"][target_h_key]["rmse"]
                pure_ar_h_for_composite = val_stats_pure_ar["per_horizon"][target_h_key]["rmse"]
                composite_metric = 0.2 * oracle_h_for_composite + 0.8 * pure_ar_h_for_composite

                if composite_metric < best_composite:
                    best_composite = composite_metric
                    best_composite_path = weights_dir / f"best_composite_epoch{epoch+1}.pth"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "mechanism": mechanism,
                            "feature_info": feature_info,
                            "version": VERSION,
                            "transform_signature": transform_signature,
                            "rollout_signature": ROLLOUT_SIGNATURE,
                            "training_config": training_config,
                            "metric_name": "composite",
                            "metric_value": best_composite,
                            "metric_formula": "0.2*oracle_h128 + 0.8*pure_ar_h128",
                            "oracle_h": oracle_h_for_composite,
                            "pure_ar_h": pure_ar_h_for_composite,
                            "horizon": target_h_key,
                            "epoch": epoch + 1,
                            "all_metrics": {
                                "oracle": val_stats.copy(),
                                "pure_ar": val_stats_pure_ar.copy(),
                            },
                        },
                        best_composite_path,
                    )
                    print(f"[SAVE] Best Composite model -> {best_composite_path.name} (Composite={best_composite:.6e})")

            # 5. Best Robust Raw (稳健选模: 硬门槛拦截发散 + 归一化综合评分)
            if (target_h_key is not None and
                val_stats_pure_ar["per_horizon"][target_h_key]["count"] > 0):

                pure_ar_h_metric = val_stats_pure_ar["per_horizon"][target_h_key]["rmse"]
                pure_ar_mass = float(val_stats_pure_ar.get("mass", 0.0))
                pure_ar_element = float(val_stats_pure_ar.get("element", 0.0))
                pure_ar_rate = float(val_stats_pure_ar.get("rate", 0.0))

                is_accepted, robust_raw_score = evaluate_robust_raw_checkpoint(
                    pure_ar_h_metric, pure_ar_mass, pure_ar_element, pure_ar_rate
                )

                if is_accepted and robust_raw_score is not None:

                    if robust_raw_score < best_robust_raw:
                        best_robust_raw = robust_raw_score
                        best_robust_raw_path = weights_dir / f"best_robust_raw_epoch{epoch+1}.pth"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "mechanism": mechanism,
                                "feature_info": feature_info,
                                "version": VERSION,
                                "transform_signature": transform_signature,
                                "rollout_signature": ROLLOUT_SIGNATURE,
                                "training_config": training_config,
                                "metric_name": "robust_raw_score",
                                "metric_value": best_robust_raw,
                                "metric_formula": "log1p(h128/ref_h128); mass/element are hard gates; rate is diagnostic only",
                                "pure_ar_h": pure_ar_h_metric,
                                "pure_ar_mass": pure_ar_mass,
                                "pure_ar_element": pure_ar_element,
                                "pure_ar_rate": pure_ar_rate,
                                "epoch": epoch + 1,
                                "all_metrics": {
                                    "oracle": val_stats.copy(),
                                    "pure_ar": val_stats_pure_ar.copy(),
                                },
                            },
                            best_robust_raw_path,
                        )
                        print(
                            f"[SAVE] Best Robust Raw model -> {best_robust_raw_path.name} "
                            f"(Score={best_robust_raw:.4f}, H={pure_ar_h_metric:.6e}, Mass={pure_ar_mass:.6e})"
                        )

        # --------------------------------------------------------
        # V68 checkpoint with RNG state
        # --------------------------------------------------------

        if (
            (epoch + 1) % checkpoint_every
            == 0
        ):

            checkpoint_path = (
                checkpoint_dir
                / (
                    f"checkpoint_v68_"
                    f"{mechanism}_"
                    f"epoch{epoch+1}.pth"
                )
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_val_loss":
                        best_val_loss,

                    "patience_counter":
                        patience_counter,

                    "history":
                        history,

                    "feature_info":
                        feature_info,

                    "version": VERSION,

                    "transform_signature": transform_signature,

                    "rollout_signature": ROLLOUT_SIGNATURE,
                    "training_config": training_config,

                    "rng_states": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                },
                checkpoint_path,
            )

            print(
                f"[SAVE] checkpoint -> "
                f"{checkpoint_path}"
            )

        # --------------------------------------------------------
        # V68 latest checkpoint (atomic save every epoch)
        # --------------------------------------------------------

        latest_path = weights_dir / f"latest_v68_{mechanism}.pth"
        temp_path = latest_path.with_suffix(".tmp")

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "patience_counter":
                    patience_counter,

                "history":
                    history,

                "feature_info":
                    feature_info,

                "version": VERSION,

                "transform_signature": transform_signature,

                "rollout_signature": ROLLOUT_SIGNATURE,
                "training_config": training_config,

                # V68: 保存多指标最佳值
                "best_oracle_h128": best_oracle_h128,
                "best_oracle_avg": best_oracle_avg,
                "best_pure_ar_h128": best_pure_ar_h128,
                "best_composite": best_composite,
                "best_robust_raw": best_robust_raw,

                "rng_states": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            temp_path,
        )

        temp_path.replace(latest_path)

        # --------------------------------------------------------
        # history
        # --------------------------------------------------------

        history_path = (
            output_root_path(
                output_root
            )
            / artifact_subdir("history")
        )

        history_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            history_path
            / f"history_{mechanism}.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
                ensure_ascii=False,
            )

    print("\n" + "=" * 90)
    print(
        "V68 training finished."
    )

    print(
        f"Best validation loss (total) = "
        f"{best_val_loss:.6e}"
    )

    print("\nBest metrics achieved:")
    if best_oracle_h128 < float("inf"):
        print(f"  Oracle H=128 RMSE      : {best_oracle_h128:.6e}")
    if best_oracle_avg < float("inf"):
        print(f"  Oracle Average RMSE    : {best_oracle_avg:.6e}")
    if best_pure_ar_h128 < float("inf"):
        print(f"  Pure-AR H=128 RMSE     : {best_pure_ar_h128:.6e}")
    if best_composite < float("inf"):
        print(f"  Composite (0.2O+0.8P)  : {best_composite:.6e}")

    print("=" * 90)


# ============================================================
# Utility
# ============================================================

def output_root_path(
    output_root: str,
):
    path = Path(
        output_root
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# Main CLI
# ============================================================

def get_training_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V68 long-horizon species-autoregressive "
            "chemistry operator training"
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help=(
            "Complete-trajectory split directory containing "
            "<mechanism>_train/validation_sample_ids.csv. "
            "When omitted, trajectory_splits_QSSA is auto-detected."
        ),
    )

    parser.add_argument(
        "--mechanism",
        type=str,
        default="nuig_2024",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max_horizon",
        type=int,
        default=128,
        help="Maximum horizon for training windows.",
    )

    parser.add_argument(
        "--curriculum_horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizon curriculum stages (default: 1 8 32 128).",
    )

    parser.add_argument(
        "--curriculum_start_epochs",
        type=int,
        nargs="+",
        default=None,
        help="Start epochs for each horizon stage (default: 0 2 5 8).",
    )

    parser.add_argument(
        "--eval_horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizons to evaluate (default: 1 2 4 8 16 32 64 128).",
    )

    parser.add_argument(
        "--tail_start",
        type=float,
        default=0.05,
        help="Initial continuous prediction tail fraction.",
    )

    parser.add_argument(
        "--tail_end",
        type=float,
        default=1.0,
        help="Final continuous prediction tail fraction.",
    )

    parser.add_argument(
        "--tail_ramp_epochs",
        type=int,
        default=0,
        help="New epochs used to ramp tail_start to tail_end (0 = immediate).",
    )

    parser.add_argument(
        "--train_stride",
        type=int,
        default=None,
        help="Training window stride (default: max_horizon // 2).",
    )

    parser.add_argument(
        "--val_stride",
        type=int,
        default=None,
        help="Validation window stride (default: max_horizon // 4).",
    )

    parser.add_argument(
        "--backward_chunk_steps",
        type=int,
        default=5,
        help="Connected rollout steps per backward chunk before detaching.",
    )

    parser.add_argument(
        "--parts_per_epoch",
        type=int,
        default=42,
        help="每个训练 epoch 随机抽取的 parquet 文件数。",
    )

    parser.add_argument(
        "--val_parts_per_epoch",
        type=int,
        default=9,
        help="每个验证 epoch 随机抽取的 parquet 文件数；默认 9。",
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=2048,
        help="验证 batch size；默认 2048。",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--scaler_parts",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--scaler_sample_rate",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--scaler_max_rows",
        type=int,
        default=2_000_000,
    )

    parser.add_argument(
        "--lambda_rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--rate_loss_kind",
        choices=("mse", "huber"),
        default="mse",
        help="Source/rate loss: MSE or robust SmoothL1/Huber.",
    )
    parser.add_argument("--output_parameterization", choices=("direct_rate", "conservative_simplex", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"), default="direct_rate")

    parser.add_argument(
        "--rate_huber_beta",
        type=float,
        default=1.0,
        help="SmoothL1 transition scale in standardized rate space.",
    )
    parser.add_argument(
        "--rate_decode_mode",
        choices=RATE_DECODE_MODES,
        default="direct_rate",
        help="Macro-rate decode: legacy direct_rate or smoothly bounded asinh-z.",
    )
    parser.add_argument(
        "--rate_z_limit",
        type=float,
        default=DEFAULT_RATE_Z_LIMIT,
        help="Positive asinh-z limit for bounded_direct_rate.",
    )

    parser.add_argument(
        "--lambda_state",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda_element",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lambda_mass",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lambda_positive",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lambda_rollout",
        type=float,
        default=0.0,
        help="Weight for multi-horizon pure species-AR state loss.",
    )
    parser.add_argument(
        "--lambda_stability",
        type=float,
        default=0.0,
        help="Weight for adjacent-state increment stability loss.",
    )

    parser.add_argument(
        "--transition_weight_alpha",
        type=float,
        default=0.0,
        help="Loss emphasis for high-activity/ignition transitions.",
    )
    parser.add_argument(
        "--transition_weight_clip",
        type=float,
        default=8.0,
        help="Maximum per-transition loss weight.",
    )
    parser.add_argument(
        "--feedback_noise_std",
        type=float,
        default=0.0,
        help="Scaled-space noise applied to predicted feedback states.",
    )
    parser.add_argument(
        "--macro_loss_weight",
        type=float,
        default=1.0,
        help="Relative loss weight for macro-species features before normalization.",
    )
    parser.add_argument(
        "--micro_loss_weight",
        type=float,
        default=1.0,
        help="Relative loss weight for ordinary micro-species features before normalization.",
    )
    parser.add_argument(
        "--key_species_loss_weight",
        type=float,
        default=1.0,
        help="Relative loss weight for NH2, HO2, OH, and H2O2 before normalization.",
    )

    parser.add_argument("--feedback_blend_start", type=float, default=1.0)
    parser.add_argument("--feedback_blend_end", type=float, default=1.0)
    parser.add_argument("--feedback_blend_ramp_epochs", type=int, default=0)
    parser.add_argument(
        "--trust_region_fraction",
        "--delta_limit",
        dest="trust_region_fraction",
        type=float,
        default=0.0,
        help="Relative macro increment bound used by V3.4 (0 disables it).",
    )
    parser.add_argument(
        "--temperature_mode",
        choices=("recorded", "autonomous"),
        default="recorded",
        help="Use recorded T or a bounded composition-energy temperature update.",
    )
    parser.add_argument("--max_delta_temperature", type=float, default=1000.0)

    parser.add_argument(
        "--experiment_mode",
        type=str,
        default="baseline",
        choices=EXPERIMENT_MODES,
        help="State update strategy for phase 4/5 experiments.",
    )

    parser.add_argument(
        "--state_update_mode",
        type=str,
        default=DEFAULT_STATE_UPDATE_MODE,
        choices=STATE_UPDATE_MODES,
        help="Task8 state update strategy: 'legacy' or 'task8_conservative_state'.",
    )

    parser.add_argument(
        "--warm_start",
        type=str,
        default=None,
        help="Checkpoint path; load model weights only and reset optimizer state.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume mode: 'auto', 'none', or path to checkpoint.",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help="Save numbered checkpoint every N epochs.",
    )

    parser.add_argument(
        "--full_val_every",
        type=int,
        default=5,
        help="Run full validation every N epochs.",
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Limit validation batches (None = all).",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--state-loss-source",
        "--state_loss_source",
        dest="state_loss_source",
        choices=STATE_LOSS_SOURCES,
        default=DEFAULT_STATE_LOSS_SOURCE,
        help="State loss target: 'constrained' (default) or 'raw' (for unconstrained micro error probe).",
    )

    parser.add_argument(
        "--feedback-state-source",
        "--feedback_state_source",
        dest="feedback_state_source",
        choices=FEEDBACK_STATE_SOURCES,
        default=DEFAULT_FEEDBACK_STATE_SOURCE,
        help="Feedback state target for rollout next step: 'constrained' (default) or 'raw'.",
    )

    parser.add_argument(
        "--micro-rate-z-limit",
        "--micro_rate_z_limit",
        dest="micro_rate_z_limit",
        type=float,
        default=None,
        help="Positive z limit for micro rate protection (default: None).",
    )

    parser.add_argument(
        "--lambda_radical",
        type=float,
        default=0.0,
        help="Weight for physical radical log loss on fast chain-branching species (H, O, OH, HO2, NH2, HNO).",
    )
    parser.add_argument(
        "--lambda_no",
        type=float,
        default=0.0,
        help="Weight for pollutant NO log loss.",
    )
    parser.add_argument(
        "--use_induction_weights",
        action="store_true",
        default=False,
        help="Boost loss weights for pre-ignition induction period samples.",
    )
    parser.add_argument(
        "--induction_boost",
        type=float,
        default=6.0,
        help="Loss boost factor for induction period samples (default: 6.0).",
    )
    parser.add_argument(
        "--projection_mode",
        type=str,
        default="multiplicative_kl",
        choices=("multiplicative_kl", "weighted_linear"),
        help="Simplex and element projection algorithm: 'multiplicative_kl' (default) or 'weighted_linear'.",
    )
    parser.add_argument(
        "--desentinelize_scaler",
        action="store_true",
        default=False,
        help="Use MaskedOnlineMoments to exclude frozen sentinel values from scaler standard deviation.",
    )
    parser.add_argument(
        "--network_arch",
        type=str,
        default="mlp",
        choices=("mlp", "resmlp", "dual_branch"),
        help="Neural network architecture: 'mlp' (baseline 4-layer), 'resmlp' (deep Pre-LN ResMLP), or 'dual_branch'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )

    return parser


def main():
    parser = get_training_arg_parser()
    args = parser.parse_args()

    train_model_v68(
        data_dir=args.data_dir,
        mechanism=args.mechanism,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_horizon=args.max_horizon,
        curriculum_horizons=args.curriculum_horizons,
        curriculum_start_epochs=args.curriculum_start_epochs,
        eval_horizons=args.eval_horizons,
        tail_start=args.tail_start,
        tail_end=args.tail_end,
        tail_ramp_epochs=args.tail_ramp_epochs,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        parts_per_epoch=args.parts_per_epoch,
        val_parts_per_epoch=args.val_parts_per_epoch,
        val_batch_size=args.val_batch_size,
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
        micro_rate_z_limit=args.micro_rate_z_limit,
        lambda_rollout=args.lambda_rollout,
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
        experiment_mode=args.experiment_mode,
        state_update_mode=args.state_update_mode,
        state_loss_source=args.state_loss_source,
        feedback_state_source=args.feedback_state_source,
        warm_start=args.warm_start,
        backward_chunk_steps=args.backward_chunk_steps,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        full_val_every=args.full_val_every,
        max_val_batches=args.max_val_batches,
        output_root=args.output_root,
        split_dir=args.split_dir,
        lambda_radical=args.lambda_radical,
        lambda_no=args.lambda_no,
        use_induction_weights=args.use_induction_weights,
        induction_boost=args.induction_boost,
        projection_mode=args.projection_mode,
        desentinelize_scaler=args.desentinelize_scaler,
        network_arch=args.network_arch,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()



