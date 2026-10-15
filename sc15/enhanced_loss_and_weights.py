"""Enhanced Physical Losses, Induction Reweighting, and De-sentinelized Scalers.

Addresses Phase 2 of the CFD deployment optimization plan:
  1. compute_physical_radical_loss: Direct physical log10-space error on fast chain-branching
     radicals (H, O, OH, HO2, NH2, HNO) with active-state masking (no artificial cold gradients).
  2. compute_pollutant_no_loss: Decoupled log-space loss for slow pollutant NO.
  3. compute_induction_sample_weights: Identifies pre-ignition induction samples (800-1600 K, dT < 60 K,
     active radical pool growth) and boosts sample weight by 5x-8x.
  4. MaskedOnlineMoments: Eliminates -45 sentinel floor distortion from scaler variance,
     without artificial variance capping, and implements .finalize() for seamless drop-in usage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fast chain-branching radicals governing ignition delay (NO is excluded as it is a slow pollutant)
CORE_CHAIN_BRANCHING_RADICALS = ("H", "O", "OH", "HO2", "NH2", "HNO")
POLLUTANT_SPECIES = ("NO",)


def compute_physical_radical_loss(
    tools: Any,
    pred_state: Dict[str, torch.Tensor],
    true_next_X_scaled: torch.Tensor,
    radicals: Tuple[str, ...] = CORE_CHAIN_BRANCHING_RADICALS,
    state_source: str = "raw",
    active_log_threshold: float = -20.0,
    cold_penalty_threshold: float = -15.0,
    loss_type: str = "huber",
    huber_beta: float = 0.5,
) -> torch.Tensor:
    """Computes direct logarithmic error on physical mass fractions of key chain-branching radicals.

    Addresses GPT review:
    1. Active-state masking: Only active samples (true_logy > active_log_threshold, e.g. -20)
       contribute to regression error.
    2. Cold-state safety: When true_logy <= active_log_threshold, penalizes false-positive
       over-prediction only if pred_logy > cold_penalty_threshold (-15 dex).
    3. Prevents artificial massive gradients on frozen cold states.
    """
    if state_source == "raw":
        pred_micro = pred_state.get("Y_micro_next_raw")
        pred_macro = pred_state.get("Y_macro_next_raw")
        pred_log_micro = pred_state.get("logY_micro_next_raw")
    else:
        pred_micro = pred_state.get("Y_micro_next_constrained")
        pred_macro = pred_state.get("Y_macro_next_constrained")
        pred_log_micro = pred_state.get("logY_micro_next_constrained")

    device = true_next_X_scaled.device
    dtype = true_next_X_scaled.dtype

    # Unscale true next X
    true_X_unscaled = true_next_X_scaled * tools.scale_X + tools.mean_X

    rad_losses = []

    for sp in radicals:
        if sp in tools.micro_species:
            micro_idx = tools.micro_species.index(sp)
            if pred_log_micro is not None:
                pred_logy = pred_log_micro[:, micro_idx]
            else:
                pred_y = pred_micro[:, micro_idx].clamp_min(1e-12)
                pred_logy = torch.log10(pred_y)

            col_idx = tools.micro_idx[micro_idx]
            true_logy = true_X_unscaled[:, col_idx]
        elif sp in tools.macro_species:
            macro_idx = tools.macro_species.index(sp)
            s_rad = 1e-6
            ln10 = 2.302585092994046
            pred_y = pred_macro[:, macro_idx]
            pred_logy = torch.asinh(pred_y / s_rad) / ln10
            col_idx = tools.macro_idx[macro_idx]
            true_y = true_X_unscaled[:, col_idx]
            true_logy = torch.asinh(torch.clamp_min(true_y, 0.0) / s_rad) / ln10
        else:
            continue

        # Active mask: sample is active if true concentration > 10^-20
        active_mask = (true_logy > active_log_threshold)
        cold_mask = ~active_mask

        loss_sp = torch.tensor(0.0, device=device, dtype=dtype)
        n_active = active_mask.sum()

        if n_active > 0:
            diff_active = pred_logy[active_mask] - true_logy[active_mask]
            if loss_type == "l1":
                active_loss = diff_active.abs().mean()
            elif loss_type == "huber":
                active_loss = F.smooth_l1_loss(
                    diff_active, torch.zeros_like(diff_active), beta=huber_beta
                )
            else:
                active_loss = diff_active.pow(2).mean()
            loss_sp = loss_sp + active_loss

        # Cold false-positive penalty: if cold but predicted > -15 dex
        n_cold = cold_mask.sum()
        if n_cold > 0:
            cold_over = F.relu(pred_logy[cold_mask] - cold_penalty_threshold)
            cold_loss = cold_over.pow(2).mean() * 0.1
            loss_sp = loss_sp + cold_loss

        rad_losses.append(loss_sp)

    if not rad_losses:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.stack(rad_losses).mean()


def compute_pollutant_no_loss(
    tools: Any,
    pred_state: Dict[str, torch.Tensor],
    true_next_X_scaled: torch.Tensor,
    state_source: str = "raw",
    active_threshold: float = -18.0,
    loss_type: str = "huber",
    huber_beta: float = 0.5,
) -> torch.Tensor:
    """Decoupled physical log-space loss for pollutant NO.

    NO is a slow accumulation product rather than a fast chain-branching radical.
    Supervises active NO (true_logy > -18 dex) with an independent loss weight,
    preventing long-horizon under-prediction without distorting radical gradients.
    """
    if "NO" not in tools.micro_species and "NO" not in tools.macro_species:
        return torch.tensor(0.0, device=true_next_X_scaled.device, dtype=true_next_X_scaled.dtype)

    if state_source == "raw":
        pred_micro = pred_state.get("Y_micro_next_raw")
        pred_macro = pred_state.get("Y_macro_next_raw")
        pred_log_micro = pred_state.get("logY_micro_next_raw")
    else:
        pred_micro = pred_state.get("Y_micro_next_constrained")
        pred_macro = pred_state.get("Y_macro_next_constrained")
        pred_log_micro = pred_state.get("logY_micro_next_constrained")

    true_X_unscaled = true_next_X_scaled * tools.scale_X + tools.mean_X

    if "NO" in tools.micro_species:
        idx = tools.micro_species.index("NO")
        if pred_log_micro is not None:
            pred_logy = pred_log_micro[:, idx]
        else:
            pred_y = pred_micro[:, idx].clamp_min(1e-12)
            pred_logy = torch.log10(pred_y)
        col_idx = tools.micro_idx[idx]
        true_logy = true_X_unscaled[:, col_idx]
        active_mask = (true_logy > active_threshold)
    else:
        idx = tools.macro_species.index("NO")
        pred_y = pred_macro[:, idx]
        col_idx = tools.macro_idx[idx]
        true_y = true_X_unscaled[:, col_idx]

        # Use smooth asinh-dex transform:
        # asinh(y / s) / ln(10) is linear for small y (|y| << s) with bounded slope 1 / (s * ln(10)),
        # and asymptotically approaches log10(2 y / s) for large y (y >> s).
        # With s = 1e-5 (10 ppm, representative of nascent NO formation),
        # the gradient magnitude is bounded by 1 / (1e-5 * 2.30) ~ 4.34e4 (well-behaved in Huber),
        # completely preventing gradient explosion when pred_y <= 0 while preserving full log sensitivity.
        s_no = 1e-5
        ln10 = 2.302585092994046
        pred_logy = torch.asinh(pred_y / s_no) / ln10
        true_logy = torch.asinh(torch.clamp_min(true_y, 0.0) / s_no) / ln10
        active_mask = (true_y > 1e-7)

    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=true_next_X_scaled.device, dtype=true_next_X_scaled.dtype)

    diff = pred_logy[active_mask] - true_logy[active_mask]
    if loss_type == "l1":
        loss = diff.abs().mean()
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=huber_beta)
    else:
        loss = diff.pow(2).mean()

    # Small penalty against cold false-positives (pred NO > 10 ppm when true NO <= 0.1 ppm)
    cold_mask = ~active_mask
    if cold_mask.sum() > 0:
        cold_penalty = F.relu(pred_logy[cold_mask] - 0.5).pow(2).mean() * 0.05
        loss = loss + cold_penalty

    return loss


def compute_induction_sample_weights(
    current_X_unscaled: torch.Tensor,
    next_X_unscaled: torch.Tensor,
    tools: Any,
    boost_factor: float = 6.0,
    delta_t_ignition_threshold: float = 60.0,
) -> torch.Tensor:
    """Assigns boosted sample weights (e.g. 6x) to samples in the pre-ignition induction period.

    Induction detection criteria:
      1. T in warm reaction regime: T_cur in [850 K, 1800 K]
      2. Flame front has not yet ignited: 0 <= (T_next - T_current) < delta_t_ignition_threshold
      3. Radical pool is actively growing: d(log10 Y)/dt > 0 for H, OH, or HO2.
    """
    B = current_X_unscaled.shape[0]
    weights = torch.ones(B, dtype=current_X_unscaled.dtype, device=current_X_unscaled.device)

    # Temperature is column 0 (temperature_idx)
    T_cur = current_X_unscaled[:, tools.temperature_idx]
    T_nxt = next_X_unscaled[:, tools.temperature_idx]
    dT = T_nxt - T_cur

    rad_growing = torch.zeros(B, dtype=torch.bool, device=current_X_unscaled.device)
    for sp in ("OH", "H", "HO2"):
        if sp in tools.micro_species:
            idx = tools.micro_idx[tools.micro_species.index(sp)]
            dlogY = next_X_unscaled[:, idx] - current_X_unscaled[:, idx]
            rad_growing = rad_growing | (dlogY > 0.02)

    is_warm = (T_cur >= 850.0) & (T_cur <= 1800.0)
    is_pre_ignition = (dT >= -5.0) & (dT < delta_t_ignition_threshold)
    is_induction = is_warm & is_pre_ignition & rad_growing

    weights = torch.where(
        is_induction,
        torch.full_like(weights, float(boost_factor)),
        weights,
    )
    return weights


class MaskedOnlineMoments:
    """Welford online moments accumulator that excludes cold sentinel values (<= sentinel_threshold)
    from contaminating the variance of QSSA rate parameters.
    
    Addresses GPT review:
    1. Removes arbitrary np.clip(std, 0.5, 8.0) ceiling, computing true unmasked sample variance.
    2. Implements .finalize() to return a standard FeatureScaler object.
    """
    def __init__(self, dim: int, sentinel_threshold: float = -35.0):
        self.dim = dim
        self.sentinel_threshold = float(sentinel_threshold)
        self.n = np.zeros(dim, dtype=np.int64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray):
        if batch is None or len(batch) == 0:
            return
        batch = np.asarray(batch, dtype=np.float64)

        for j in range(self.dim):
            col = batch[:, j]
            valid = np.isfinite(col) & (col > self.sentinel_threshold)
            vals = col[valid]
            if len(vals) == 0:
                continue

            n2 = len(vals)
            mean2 = np.mean(vals)
            m2_batch = np.sum((vals - mean2) ** 2)

            if self.n[j] == 0:
                self.n[j] = n2
                self.mean[j] = mean2
                self.M2[j] = m2_batch
            else:
                delta = mean2 - self.mean[j]
                total_n = self.n[j] + n2
                self.mean[j] += delta * (n2 / total_n)
                self.M2[j] += m2_batch + (delta ** 2) * (self.n[j] * n2 / total_n)
                self.n[j] = total_n

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        var = np.where(self.n > 1, self.M2 / np.maximum(self.n - 1, 1), 1.0)
        std = np.sqrt(np.maximum(var, 1e-8))
        # Protect against exact zero variance without artificial upper cap
        std = np.maximum(std, 1e-4)
        return self.mean.copy(), std.copy(), self.n.copy()

    def finalize(self):
        """Finalize into FeatureScaler compatible with train_sc_v68."""
        mean, std, n = self.get_stats()
        # Local import to avoid circular dependency
        from train_sc_v68 import FeatureScaler
        return FeatureScaler(
            mean.astype(np.float32),
            std.astype(np.float32),
            int(n.max()) if len(n) > 0 else 0,
        )
