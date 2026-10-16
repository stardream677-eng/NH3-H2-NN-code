"""Differentiable NASA-7 Polynomial Thermochemistry Layer for PyTorch.

Provides GPU-capable, differentiable thermodynamic
calculations directly from Cantera mechanism files (e.g. NUIG2024.yaml, TH2024.yaml, UCF2024.yaml):
  - Heat capacity (cp, cv)
  - Specific enthalpy (h)
  - Specific internal energy (u)
  - Const-Volume energy closure: solves T from internal energy u(T, Y) = u_0
  - Const-Pressure energy closure: solves T from enthalpy h(T, Y) = h_0

Replaces the legacy 1000 K constant-cp approximation with an iterative NASA-7
energy closure that verifies the final energy residual and rejects failed solves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cantera as ct
import numpy as np
import torch
import torch.nn as nn


class ThermoNASA7Layer(nn.Module):
    def __init__(
        self,
        mechanism_path: str | Path = "NUIG2024.yaml",
        species_order: Optional[List[str]] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.mechanism_path = str(mechanism_path)
        self.custom_dtype = dtype
        
        # Load Cantera solution to extract NASA polynomials
        gas = ct.Solution(self.mechanism_path)
        all_names = list(gas.species_names)
        
        if species_order is None:
            self.species_names = all_names
        else:
            self.species_names = list(species_order)
            
        self.n_species = len(self.species_names)
        gas_constant = ct.gas_constant
        
        # Extract molecular weights and coeffs for ordered species
        mw_list = []
        t_mid_list = []
        coeffs_list = []
        
        for sp_name in self.species_names:
            sp = gas.species(sp_name)
            mw_list.append(gas.molecular_weights[gas.species_index(sp_name)])
            c = sp.thermo.coeffs
            # c[0]: T_mid, c[1:8]: high-T, c[8:15]: low-T
            t_mid_list.append(c[0])
            coeffs_list.append(c[1:15])
            
        # Register persistent buffers (move with model.to(device))
        self.register_buffer("W", torch.tensor(np.array(mw_list), dtype=dtype, device=device))  # (N_sp,)
        self.register_buffer("t_mids", torch.tensor(np.array(t_mid_list), dtype=dtype, device=device))  # (N_sp,)
        self.register_buffer("coeffs", torch.tensor(np.array(coeffs_list), dtype=dtype, device=device))  # (N_sp, 14)
        self.register_buffer("Ru", torch.tensor(gas_constant, dtype=dtype, device=device))
        
    def _evaluate_nasa_coeffs(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns standard cp_R and h_RT per species at temperature T.
        
        T: shape (B, 1) or scalar
        Returns:
            cp_R: (B, N_sp)
            h_RT: (B, N_sp)
        """
        # Ensure T has shape (B, 1)
        if T.ndim == 0:
            T_col = T.view(1, 1)
        elif T.ndim == 1:
            T_col = T.unsqueeze(-1)
        else:
            T_col = T
            
        # Select low vs high polynomial: (B, N_sp)
        t_mids = self.t_mids.to(device=T_col.device, dtype=T_col.dtype)
        is_high = (T_col >= t_mids.unsqueeze(0))
        
        coeffs = self.coeffs.to(device=T_col.device, dtype=T_col.dtype)
        a_high = coeffs[:, :7].unsqueeze(0)  # (1, N_sp, 7)
        a_low = coeffs[:, 7:].unsqueeze(0)   # (1, N_sp, 7)
        
        a = torch.where(is_high.unsqueeze(-1), a_high, a_low)  # (B, N_sp, 7)
        
        # T powers: (B, 1)
        T1 = T_col
        T2 = T1 * T1
        T3 = T2 * T1
        T4 = T3 * T1
        
        # cp/Ru = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
        cp_R = (
            a[..., 0]
            + a[..., 1] * T1
            + a[..., 2] * T2
            + a[..., 3] * T3
            + a[..., 4] * T4
        )
        
        # h/(Ru*T) = a0 + a1/2*T + a2/3*T^2 + a3/4*T^3 + a4/5*T^4 + a5/T
        h_RT = (
            a[..., 0]
            + 0.5 * a[..., 1] * T1
            + (1.0 / 3.0) * a[..., 2] * T2
            + 0.25 * a[..., 3] * T3
            + 0.2 * a[..., 4] * T4
            + a[..., 5] / T1.clamp_min(10.0)
        )
        
        return cp_R, h_RT

    def compute_mixture_properties(
        self, T: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mixture cp, cv, h, u per unit mass.
        
        T: (B, 1) or (B,) in Kelvin
        Y: (B, N_sp) mass fractions
        Returns:
            cp_mix: (B, 1)  [J/(kg K)]
            cv_mix: (B, 1)  [J/(kg K)]
            h_mix: (B, 1)   [J/kg]
            u_mix: (B, 1)   [J/kg]
        """
        if T.ndim == 1:
            T_col = T.unsqueeze(-1)
        else:
            T_col = T
            
        cp_R, h_RT = self._evaluate_nasa_coeffs(T_col)  # (B, N_sp)
        
        # Convert molar dimensionless to mass-specific properties:
        W = self.W.to(device=T_col.device, dtype=T_col.dtype)
        Ru = self.Ru.to(device=T_col.device, dtype=T_col.dtype)
        Y_cast = Y.to(device=T_col.device, dtype=T_col.dtype)

        Ru_W = Ru / W.unsqueeze(0)            # (1, N_sp)
        cp_mass = cp_R * Ru_W                 # (B, N_sp)
        cv_mass = (cp_R - 1.0) * Ru_W         # (B, N_sp)
        
        Ru_T_W = (Ru * T_col) / W.unsqueeze(0)  # (B, N_sp)
        h_mass = h_RT * Ru_T_W                  # (B, N_sp)
        u_mass = (h_RT - 1.0) * Ru_T_W          # (B, N_sp)
        
        # Mixture mass averages
        cp_mix = (Y_cast * cp_mass).sum(dim=-1, keepdim=True)
        cv_mix = (Y_cast * cv_mass).sum(dim=-1, keepdim=True)
        h_mix = (Y_cast * h_mass).sum(dim=-1, keepdim=True)
        u_mix = (Y_cast * u_mass).sum(dim=-1, keepdim=True)
        
        return cp_mix, cv_mix, h_mix, u_mix

    def _solve_temperature_energy(
        self,
        target: torch.Tensor,
        Y: torch.Tensor,
        T_init: torch.Tensor,
        mode: str,
        max_iters: int = 24,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """Newton energy closure with an explicit *final* residual check.

        Newton iterations use float64 and bounded temperature steps; the final
        residual is checked after converting to the caller's temperature dtype.
        The returned tensor remains differentiable.  Unreachable targets or
        failed solves raise RuntimeError instead of returning a clipped state.
        """
        if max_iters < 1 or not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("max_iters must be >= 1 and tol must be finite and positive")
        if mode not in ("uv", "hp"):
            raise ValueError(f"Unsupported energy closure mode: {mode}")

        original_dtype = T_init.dtype
        work_dtype = torch.float64
        T = T_init.reshape(-1, 1).to(dtype=work_dtype)
        Y_work = Y.to(device=T.device, dtype=work_dtype)
        target_work = torch.as_tensor(target, device=T.device, dtype=work_dtype)
        if target_work.ndim == 0:
            target_work = target_work.reshape(1, 1)
        elif target_work.ndim == 1:
            target_work = target_work[:, None]
        try:
            target_work = torch.broadcast_to(target_work, T.shape)
        except RuntimeError as exc:
            raise ValueError("Energy target must broadcast to the temperature batch") from exc
        if Y_work.ndim != 2 or Y_work.shape != (T.shape[0], self.n_species):
            raise ValueError("Y must have shape (batch, n_species)")
        if not (
            torch.isfinite(T).all()
            and torch.isfinite(Y_work).all()
            and torch.isfinite(target_work).all()
        ):
            raise ValueError("Energy closure requires finite temperatures, fractions and targets")
        if (T < 200.0).any() or (T > 5000.0).any():
            raise ValueError("Initial temperatures must lie in [200, 5000] K")

        def energy_and_capacity(T_current):
            cp, cv, h, u = self.compute_mixture_properties(T_current, Y_work)
            return (u, cv) if mode == "uv" else (h, cp)

        for _ in range(max_iters):
            energy, capacity = energy_and_capacity(T)
            residual = energy - target_work
            if not (torch.isfinite(residual).all() and torch.isfinite(capacity).all()):
                raise RuntimeError("NASA7 energy closure produced non-finite values")
            if (capacity <= 0.0).any():
                raise RuntimeError("NASA7 energy closure requires positive heat capacity")
            # tol is a temperature-equivalent tolerance; also bound the
            # absolute/relative energy residual in J/kg.
            energy_tol = 0.1 + 1e-8 * torch.maximum(energy.abs(), target_work.abs())
            energy_tol = torch.maximum(energy_tol, capacity * float(tol))
            if bool((residual.abs() <= energy_tol).all()):
                break
            step = torch.clamp(residual / capacity, min=-500.0, max=500.0)
            T = torch.clamp(T - step, min=200.0, max=5000.0)

        # Check the *returned* state: float32 rounding may be larger than
        # the residual at an intermediate float64 Newton iterate.
        final_T = T.to(original_dtype)
        energy, capacity = energy_and_capacity(final_T.to(work_dtype))
        residual = (energy - target_work).abs()
        rounding_K = 2.0 * torch.finfo(original_dtype).eps * final_T.to(work_dtype).abs()
        energy_tol = torch.maximum(
            0.1 + 1e-8 * torch.maximum(energy.abs(), target_work.abs()),
            capacity.abs() * (float(tol) + rounding_K),
        )
        if not torch.isfinite(residual).all() or not bool((residual <= energy_tol).all()):
            max_abs = float(residual.detach().max())
            max_limit = float(energy_tol.detach().max())
            raise RuntimeError(
                f"NASA7 {mode} energy closure did not converge within {max_iters} iterations "
                f"(max residual={max_abs:.6e} J/kg, allowed={max_limit:.6e} J/kg); "
                "reduce the integration interval or inspect the predicted composition"
            )
        return final_T

    def solve_temperature_uv(
        self,
        u_target: torch.Tensor,
        Y: torch.Tensor,
        T_init: torch.Tensor,
        max_iters: int = 24,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """Solve constant-volume u(T,Y)=u_target; reject non-converged states."""
        return self._solve_temperature_energy(u_target, Y, T_init, "uv", max_iters, tol)

    def solve_temperature_hp(
        self,
        h_target: torch.Tensor,
        Y: torch.Tensor,
        T_init: torch.Tensor,
        max_iters: int = 24,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """Solve constant-pressure h(T,Y)=h_target; reject non-converged states."""
        return self._solve_temperature_energy(h_target, Y, T_init, "hp", max_iters, tol)
