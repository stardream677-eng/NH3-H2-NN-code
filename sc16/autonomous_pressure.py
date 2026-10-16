"""Ideal-gas pressure feedback for adiabatic closed-reactor neural rollouts.

The V21 data contract stores pressure in atm (Cantera uses Pa internally).
A pressure *ratio* is unit invariant, and avoids an implicit density state:

    P_next / P_old = (T_next / T_old)
        * sum_i(Y_next_i / W_i) / sum_i(Y_old_i / W_i)

For a constant-volume ideal-gas reactor, repeated exact EOS steps preserve the
density implicit in the initial P/T/Y. A constant-pressure reactor keeps P_old.

Mass fractions must include the same full mechanism species (including inerts)
used by the NASA7 layer, in the same order as its molecular weight buffer.
"""
from __future__ import annotations

import torch


PRESSURE_FEEDBACK_CONTRACT = "eos_uv_constant_hp_v1"


def autonomous_pressure_next(
    P_old: torch.Tensor,
    T_old: torch.Tensor,
    T_next: torch.Tensor,
    Y_old: torch.Tensor,
    Y_next: torch.Tensor,
    molecular_weights: torch.Tensor,
    reactor_type: str,
    *,
    mass_tolerance: float = 2e-3,
    min_pressure_atm: float = 1e-6,
    max_pressure_atm: float = 1e4,
) -> torch.Tensor:
    """Compute constant-volume EOS pressure or constant-pressure feedback.

    Inputs P_old are in *atm* for sc16; output retains this unit and dtype.
    Float64 computations are retained through EOS evaluation; returned
    float32 pressure is checked against the EOS within rounding tolerance.
    The operator is differentiable with respect to T and composition.

    The tolerance on mass normalization is an explicit guard, not an
    implicit renormalization: incorrect species outputs must not silently
    change density or pressure.
    """
    if reactor_type not in ("IdealGasReactor", "IdealGasConstPressureReactor"):
        raise ValueError(f"Unsupported ideal-gas reactor type: {reactor_type!r}")
    if P_old.ndim != 2 or P_old.shape[1] != 1:
        raise ValueError("P_old must be shaped (batch, 1)")
    if T_old.shape != P_old.shape or T_next.shape != P_old.shape:
        raise ValueError("T_old and T_next must match P_old shape")
    if Y_old.ndim != 2 or Y_next.shape != Y_old.shape or Y_old.shape[0] != P_old.shape[0]:
        raise ValueError("Y_old and Y_next must have shape (batch, n_species)")
    weights = molecular_weights.reshape(-1)
    if weights.numel() != Y_old.shape[1]:
        raise ValueError("Molecular-weight count must equal full species count")
    if not 0.0 < mass_tolerance < 1.0:
        raise ValueError("mass_tolerance must lie between 0 and 1")

    P = P_old.to(torch.float64)
    T0 = T_old.to(torch.float64)
    T1 = T_next.to(torch.float64)
    Y0 = Y_old.to(torch.float64)
    Y1 = Y_next.to(torch.float64)
    W = weights.to(device=P.device, dtype=torch.float64)

    if not all(torch.isfinite(x).all() for x in (P, T0, T1, Y0, Y1, W)):
        raise ValueError("Non-finite pressure, temperature, composition or weights")
    if (W <= 0.0).any() or (T0 <= 0.0).any() or (T1 <= 0.0).any():
        raise ValueError("Molecular weights and absolute temperatures must be positive")
    if (Y0 < 0.0).any() or (Y1 < 0.0).any():
        raise ValueError("EOS cannot use negative mass fractions")
    if (
        ((Y0.sum(dim=1) - 1.0).abs() > mass_tolerance).any()
        or ((Y1.sum(dim=1) - 1.0).abs() > mass_tolerance).any()
    ):
        raise ValueError("EOS requires full, approximately unit-normalized mass fractions")
    if (P <= 0.0).any():
        raise ValueError("EOS requires positive current pressure")

    if reactor_type == "IdealGasConstPressureReactor":
        updated = P
    else:
        inverse_mw_old = (Y0 / W.unsqueeze(0)).sum(dim=1, keepdim=True)
        inverse_mw_next = (Y1 / W.unsqueeze(0)).sum(dim=1, keepdim=True)
        if (inverse_mw_old <= 0.0).any() or (inverse_mw_next <= 0.0).any():
            raise ValueError("EOS requires positive inverse mixture molecular weight")
        updated = P * (T1 / T0) * (inverse_mw_next / inverse_mw_old)

    if not torch.isfinite(updated).all() or bool(
        ((updated < min_pressure_atm) | (updated > max_pressure_atm)).any()
    ):
        raise RuntimeError(
            f"Autonomous pressure outside [{min_pressure_atm:g}, {max_pressure_atm:g}] atm; "
            "check the predicted composition, temperature and timestep"
        )
    returned = updated.to(P_old.dtype)
    if not torch.isfinite(returned).all():
        raise RuntimeError("Autonomous pressure overflow after feedback dtype conversion")
    relative = ((returned.to(torch.float64) - updated).abs() / updated).max()
    if bool(relative > max(4 * torch.finfo(P_old.dtype).eps, 1e-12)):
        raise RuntimeError("Autonomous EOS pressure residual too large after dtype conversion")
    return returned
