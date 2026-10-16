"""Continuous kinetics parameterization and exponential integrator module.

Provides the 'dual_branch_pd_v1' parameterization where:
1. Macro species use standardized asinh-scaled production rates (continuous and unbounded).
2. Micro species use continuous log10(P + eps_P) and log10(D + eps_D) heads without sentinels.
3. State evolution uses the unified phi1 exponential integrator:
       Y_next = Y_old * exp(-D*dt) + P*dt*phi1(D*dt)
   which smoothly transitions from pure production (D=0 -> Y + P*dt) to stiff equilibrium
   (D*dt >> 1 -> P/D) with strictly finite gradients and zero discontinuity jumps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from numerical_contract import mass_from_log10

# Mode signatures
DUAL_BRANCH_PD_V1_SIGNATURE = "dual_branch_pd_v1"

# Numerical floor offsets for target encoding
EPS_P = 1e-20  # [s^-1] Mass fraction production rate floor (bias < 1e-22 in 1e-2 s)
EPS_D = 1e-10  # [s^-1] Destruction frequency floor (false decay < 1e-12 in 1e-2 s)


def phi1_decay(x: torch.Tensor) -> torch.Tensor:
    """Compute phi1(x) = (1 - exp(-x)) / x with Taylor series for |x| < 1e-4.

    Avoids 0/0 division on inactive branches and Taylor overflow for large x.
    Note: Only the exponent -x is bounded to prevent exp overflow, while the
    denominator remains the true x so that phi1(x) -> 1/x exactly for stiff x >> 1.
    """
    small = x.abs() < 1e-4
    x_small = torch.where(small, x, torch.zeros_like(x))
    x_regular = torch.where(small, torch.ones_like(x), x)
    series = 1.0 - x_small / 2.0 + x_small.square() / 6.0 - x_small.pow(3) / 24.0
    x_for_exp = torch.clamp(x_regular, max=80.0)
    regular = -torch.expm1(-x_for_exp) / x_regular
    return torch.where(small, series, regular)


def phi1_decay_np(x: np.ndarray) -> np.ndarray:
    """NumPy version of phi1_decay."""
    x_arr = np.asarray(x, dtype=np.float64)
    small = np.abs(x_arr) < 1e-4
    x_small = np.where(small, x_arr, 0.0)
    x_regular = np.where(small, 1.0, x_arr)
    series = 1.0 - x_small / 2.0 + x_small ** 2 / 6.0 - x_small ** 3 / 24.0
    x_for_exp = np.clip(x_regular, 0.0, 80.0)
    regular = -np.expm1(-x_for_exp) / x_regular
    return np.where(small, series, regular)


def encode_pd_targets(
    P: torch.Tensor | np.ndarray,
    D: torch.Tensor | np.ndarray,
    eps_P: float = EPS_P,
    eps_D: float = EPS_D,
) -> Tuple[Any, Any]:
    """Encode physical production rate P and destruction frequency D into continuous log10 targets."""
    if isinstance(P, torch.Tensor):
        P_clamped = torch.clamp_min(P, 0.0)
        D_clamped = torch.clamp_min(D, 0.0)
        z_P = torch.log10(P_clamped + eps_P)
        z_D = torch.log10(D_clamped + eps_D)
        return z_P, z_D
    else:
        P_arr = np.maximum(np.asarray(P, dtype=np.float64), 0.0)
        D_arr = np.maximum(np.asarray(D, dtype=np.float64), 0.0)
        z_P = np.log10(P_arr + eps_P)
        z_D = np.log10(D_arr + eps_D)
        return z_P, z_D


def decode_pd_outputs(
    z_P: torch.Tensor | np.ndarray,
    z_D: torch.Tensor | np.ndarray,
    eps_P: float = EPS_P,
    eps_D: float = EPS_D,
) -> Tuple[Any, Any]:
    """Decode continuous log10 outputs back to nonnegative physical (P, D)."""
    if isinstance(z_P, torch.Tensor):
        # Diagnostics may exceed float32 even when the integrated P/D ratio is
        # perfectly physical. Integration below never forms these huge rates.
        if not all(torch.isfinite(z).all() and (z<300).all() for z in (z_P,z_D)):
            raise FloatingPointError('P/D log rates exceed the finite diagnostic domain')
        P = torch.clamp_min(torch.pow(10.0, z_P.double()) - eps_P, 0.0)
        D = torch.clamp_min(torch.pow(10.0, z_D.double()) - eps_D, 0.0)
        return P, D
    else:
        if not all(np.isfinite(z).all() and (np.asarray(z)<300).all() for z in (z_P,z_D)):
            raise FloatingPointError('P/D log rates exceed the finite diagnostic domain')
        P = np.maximum(np.power(10.0, np.asarray(z_P, dtype=np.float64)) - eps_P, 0.0)
        D = np.maximum(np.power(10.0, np.asarray(z_D, dtype=np.float64)) - eps_D, 0.0)
        return P, D


def integrate_qssa_continuous(
    Y_old: torch.Tensor,
    P: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Unified, continuous exact exponential integrator for combustion radicals:

    Y_next = Y_old * exp(-D * dt) + P * dt * phi1(D * dt)

    Smoothly and continuously bridges:
    - Pure production (D=0) -> Y_next = Y_old + P*dt (no 0/0!)
    - Stiff steady state (D*dt >> 1) -> Y_next = P/D
    - Pure decay (P=0) -> Y_next = Y_old * exp(-D*dt)
    """
    D_eff = torch.clamp_min(D, 0.0)
    P_eff = torch.clamp_min(P, 0.0)
    dt_eff = torch.clamp_min(dt, 0.0)

    x = D_eff * dt_eff
    x_bounded = torch.clamp(x, max=80.0)
    exp_neg_x = torch.exp(-x_bounded)
    phi = phi1_decay(x)

    Y_next = Y_old * exp_neg_x + P_eff * dt_eff * phi
    return Y_next


def integrate_qssa_continuous_np(
    Y_old: np.ndarray,
    P: np.ndarray,
    D: np.ndarray,
    dt: np.ndarray | float,
) -> np.ndarray:
    """NumPy version of the continuous exponential integrator."""
    Y_arr = np.asarray(Y_old, dtype=np.float64)
    P_eff = np.maximum(np.asarray(P, dtype=np.float64), 0.0)
    D_eff = np.maximum(np.asarray(D, dtype=np.float64), 0.0)
    dt_eff = np.maximum(np.asarray(dt, dtype=np.float64), 0.0)

    x = D_eff * dt_eff
    x_bounded = np.clip(x, 0.0, 80.0)
    exp_neg_x = np.exp(-x_bounded)
    phi = phi1_decay_np(x)

    return Y_arr * exp_neg_x + P_eff * dt_eff * phi


LN10 = 2.302585092994046
INV_LN10 = 0.4342944819032518


def log10_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log10(10^a + 10^b) with finite gradients via logaddexp."""
    return INV_LN10 * torch.logaddexp(a * LN10, b * LN10)


def pd_v1_micro_log_update(
    logY_old: torch.Tensor,
    z_P: torch.Tensor,
    z_D: torch.Tensor,
    dt: torch.Tensor,
    eps_P: float = EPS_P,
    eps_D: float = EPS_D,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate micro radical state directly in log10 space to avoid 1/Y gradient overflow.

    Returns
    -------
    Y_next_linear : torch.Tensor
        Physical linear mass fraction.
    logY_next : torch.Tensor
        Candidate state in log10 space with strictly finite gradients in float32.
    """
    if not all(torch.isfinite(v).all() for v in (logY_old,z_P,z_D,dt)) or (dt<=0).any():
        raise ValueError('P/D update requires finite states, log rates and positive dt')
    dtype=logY_old.dtype
    def effective_log(z,eps):
        z=z.double(); floor=math.log10(eps); active=z>floor
        distance=torch.where(active,z-floor,torch.ones_like(z))
        correction=torch.log10(-torch.expm1(-distance*LN10))
        return torch.where(active,z+correction,torch.full_like(z,-1e4)),active
    lp,has_p=effective_log(z_P,eps_P)
    ld,has_d=effective_log(z_D,eps_D)
    ldt=torch.log10(dt.double())
    lx=ld+ldt
    stiff=has_d & (lx>math.log10(80.))
    # Bound only the phi evaluation. Pure decay retains its exact log state,
    # including across D*dt=80; no fictitious concentration floor is introduced.
    x=torch.where(has_d,torch.pow(10.,lx.clamp(max=math.log10(80.))),torch.zeros_like(lx))
    if (lx>300.).any():raise FloatingPointError('D*dt exceeds the finite integration domain')
    exact_x=torch.where(has_d,torch.pow(10.,lx.clamp(max=300.)),torch.zeros_like(lx))
    decay=logY_old.double()-exact_x*INV_LN10
    log_phi=torch.log10(phi1_decay(x))
    production=torch.where(stiff,lp-ld,lp+ldt+log_phi)
    production=torch.where(has_p,production,torch.full_like(production,-torch.inf))
    log_next=log10_add_exp(decay,production).to(dtype)
    if not torch.isfinite(log_next).all():raise FloatingPointError('P/D log state exceeds the feedback dtype domain')
    return mass_from_log10(log_next),log_next

