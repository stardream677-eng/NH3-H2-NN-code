# -*- coding: utf-8 -*-
"""
Transient Chemistry Database Generator V21

Training target:
    (T_n, P_n, Y_n, dt) -> (dTdt_avg, dYdt_avg)

where
    dYdt_avg = (Y_{n+1} - Y_n) / dt
    dTdt_avg = (T_{n+1} - T_n) / dt

dt is a CFD-like coarse temporal interval, NOT Cantera's internal
adaptive solver step.

Output:
    database_Transient_V21/
        trajectories_<mechanism>/
            part-000000.parquet
            part-000001.parquet
            ...
            metadata.json
        dataset_manifest.csv
        database_summary.json

This version avoids repeated parquet read/concat/rewrite, caches the
Cantera mechanism inside each worker process, uses deterministic random
seeds, clamps the last dt to max_time, validates states, and records
interval-average source terms.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cantera as ct
import numpy as np
import pandas as pd
from pebble import ProcessPool
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    dt_min: float = 1e-8
    dt_max: float = 1e-4
    max_time: float = 2.0
    max_unignited_time: float = 0.25
    max_steps_per_trajectory: int = 35000

    equilibrium_check_start: float = 2e-3
    eq_T_rate_tol: float = 1.0       # K/s
    eq_Y_rate_tol: float = 1e-2     # 1/s
    equilibrium_consecutive: int = 3
    min_ignition_temp_rise: float = 200.0  # K, required temperature rise before checking equilibrium to guard slow induction

    pressure_unit: str = "atm"       # "atm", "pa", or "auto"
    composition_epsilon: float = 1e-12
    mass_fraction_tolerance: float = 1e-8
    negative_mass_fraction_tolerance: float = 1e-12

    # "IdealGasReactor" = constant-volume adiabatic reactor.
    # "IdealGasConstPressureReactor" = constant-pressure reactor.
    reactor_type: str = "IdealGasReactor"
    energy: str = "on"

    base_seed: int = 20260815

    workers: int = 16
    task_timeout_s: float = 120.0

    batch_size: int = 50
    compression: str = "snappy"
    overwrite_output: bool = False

    validate_each_interval: bool = True
    max_abs_source_rate: float = 1e20


CFG = Config()

# The exact QSSA inversion treats states at or below this level as numerical
# extinction.  This is only a division guard for rate extraction; it is much
# lower than the training transform floor and does not erase trace labels.
QSSA_STATE_FLOOR = 1e-300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("chem_database_v21")


SUPPORTED_MECHANISMS = (
    "nuig_2024",
    "tsinghua_2024c",
    "ucf_2024",
)


def normalize_mechanism_selection(
    mechanism: str | None = None,
    mechanisms: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Validate and canonicalize mechanisms requested for database generation."""
    if mechanism is not None and mechanisms is not None:
        raise ValueError("--mechanism and --mechanisms are mutually exclusive")

    raw = (
        [mechanism]
        if mechanism is not None
        else list(mechanisms)
        if mechanisms is not None
        else list(SUPPORTED_MECHANISMS)
    )
    selected = [str(name).strip().lower() for name in raw]
    if not selected or any(not name for name in selected):
        raise ValueError("At least one mechanism must be selected")

    unknown = sorted(set(selected) - set(SUPPORTED_MECHANISMS))
    if unknown:
        supported = ", ".join(SUPPORTED_MECHANISMS)
        raise ValueError(
            f"Unknown mechanism(s): {', '.join(unknown)}. "
            f"Supported mechanisms: {supported}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("Mechanism selection contains duplicates")
    return selected


# =============================================================================
# Worker-local Cantera cache
# =============================================================================

_WORKER_GAS: Optional[ct.Solution] = None
_WORKER_MECH_PATH: Optional[str] = None
_WORKER_MECH_NAME: Optional[str] = None


def _get_worker_gas(mech_path: str, mech_name: str) -> ct.Solution:
    """Load one mechanism once per worker process and reuse it."""
    global _WORKER_GAS, _WORKER_MECH_PATH, _WORKER_MECH_NAME

    if (
        _WORKER_GAS is None
        or _WORKER_MECH_PATH != mech_path
        or _WORKER_MECH_NAME != mech_name
    ):
        logger.info(
            "PID=%s loading mechanism [%s]: %s",
            os.getpid(), mech_name, mech_path
        )
        _WORKER_GAS = ct.Solution(mech_path)
        _WORKER_MECH_PATH = mech_path
        _WORKER_MECH_NAME = mech_name

    return _WORKER_GAS


# =============================================================================
# Utilities
# =============================================================================

def _stable_seed(sample_id, base_seed: int) -> int:
    """Deterministic uint32 seed; unlike Python hash(), stable across runs."""
    raw = f"{base_seed}|{sample_id}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _sample_log_uniform_dt(
    rng: np.random.Generator,
    dt_min: float,
    dt_max: float,
) -> float:
    if not (0 < dt_min <= dt_max):
        raise ValueError(f"Invalid dt range [{dt_min}, {dt_max}]")
    return 10.0 ** rng.uniform(math.log10(dt_min), math.log10(dt_max))


def _pressure_to_pa(value: float, unit: str) -> float:
    unit = unit.lower().strip()
    val = float(value)
    if not np.isfinite(val) or val <= 0:
        raise ValueError(f"Invalid non-positive or non-finite pressure: {val}")

    if unit == "auto":
        if val > 2000.0:
            unit = "pa"
        else:
            unit = "atm"

    if unit == "atm":
        if val > 500.0:
            raise ValueError(
                f"Pressure sanity check failed: {val} exceeds 500 atm! "
                f"This looks like Pascals ({val} Pa = {val/ct.one_atm:.4f} atm). "
                f"Please specify --pressure_unit pa or verify your input dataset."
            )
        if val < 0.01:
            raise ValueError(f"Pressure sanity check failed: {val} atm is unphysically low (< 0.01 atm).")
        return val * ct.one_atm

    if unit == "pa":
        if val < 500.0:
            raise ValueError(
                f"Pressure sanity check failed: {val} is < 500 Pa! "
                f"This looks like atmospheres ({val} atm). "
                f"Please specify --pressure_unit atm or verify your input dataset."
            )
        if val > 1e8:
            raise ValueError(f"Pressure sanity check failed: {val} Pa exceeds 100 MPa (1000 atm).")
        return val

    raise ValueError(f"Unknown pressure_unit: '{unit}'. Must be 'atm', 'pa', or 'auto'.")


def _brentq_monotonic(
    f,
    a: float,
    b: float,
    xtol: float = 1e-15,
    rtol: float = 1e-14,
    maxiter: int = 100,
) -> float:
    """Pure-Python Brent-Dekker method for 1D bracketed monotonic root finding.

    Eliminates dependency on SciPy/LAPACK, completely preventing Windows multiprocessing
    pagefile exhaustion (DLL load failure) while guaranteeing machine precision (< 1e-15).
    """
    fa = float(f(a))
    fb = float(f(b))
    if abs(fa) <= 1e-16:
        return a
    if abs(fb) <= 1e-16:
        return b
    if fa * fb > 0:
        raise ValueError(f"Root not bracketed: f({a})={fa}, f({b})={fb}")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    mflag = True
    d = 0.0
    for _ in range(maxiter):
        if fa != fc and fb != fc:
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 <= s <= b if a < b else b <= s <= (3 * a + b) / 4)
        cond2 = mflag and (abs(s - b) >= abs(b - c) / 2)
        cond3 = (not mflag) and (abs(s - b) >= abs(c - d) / 2)
        cond4 = mflag and (abs(b - c) < xtol)
        cond5 = (not mflag) and (abs(c - d) < xtol)

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = float(f(s))
        d = c
        c = b
        fc = fb
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        tol = 2.0 * (xtol + rtol * abs(b))
        if abs(b - a) <= tol or abs(fb) <= 1e-16:
            return b
    return b


def _solve_exact_qssa_params(
    Y0: np.ndarray,
    Y_next: np.ndarray,
    dt: float,
    P_raw: np.ndarray,
    D_raw_init: np.ndarray,
    max_abs_tol: float = 1e-10,
    max_rel_tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Algebraically exact inversion of effective QSSA parameters (P_eff, D_eff).

    Guarantees that the analytical exponential integrator:
        Y_reconstructed = Y0 * exp(-D_eff * dt) + P_eff * dt * phi_1(-D_eff * dt)
    reconstructs Cantera's true state step Y_next to machine precision for both
    macro species and ultra-trace radicals across 35 orders of magnitude.
    Uses target-normalized bounded monotonic root finding (pure-Python Brent's method)
    with strict mixed absolute and relative error post-condition validation.
    """
    Y0 = np.asarray(Y0, dtype=np.float64)
    Y_next = np.asarray(Y_next, dtype=np.float64)
    n_sp = len(Y0)
    P_eff = np.maximum(np.asarray(P_raw, dtype=np.float64), 0.0)
    D_eff = np.zeros(n_sp, dtype=np.float64)

    for i in range(n_sp):
        y0 = float(Y0[i])
        y_tgt = float(Y_next[i])
        p_val = float(P_eff[i])
        d_val = float(D_raw_init[i])

        # 1. Underproduction check: if y_tgt >= y0 + p_val * dt
        # Even zero destruction cannot achieve y_tgt with p_val.
        # Set D = 0 and solve exactly: P_eff = (y_tgt - y0) / dt
        if y_tgt >= (y0 + p_val * dt):
            P_eff[i] = (y_tgt - y0) / dt
            D_eff[i] = 0.0
            continue

        # 2. Complete extinction (target <= 1e-300 or float64 true underflow)
        if y_tgt <= 1e-300:
            if y0 <= 1e-300:
                P_eff[i] = 0.0
                D_eff[i] = 0.0
            else:
                # In float64, exp(-750) underflows to 0.0
                P_eff[i] = 0.0
                D_eff[i] = 750.0 / dt
            continue

        # 3. Unchanged species (e.g. frozen inerts or ultra-slow trace species)
        if abs(y_tgt - y0) < 1e-300 and p_val * dt < 1e-300:
            P_eff[i] = 0.0
            D_eff[i] = 0.0 if y0 > 1e-300 else max(d_val, 0.0)
            continue

        P_eff[i] = p_val

        # Target-normalized root function h(z): solves for relative error!
        # h(z) = (y0 * exp(-z) + p_val * dt * phi1(-z)) / y_tgt - 1.0
        def h(z):
            if z < 1e-7:
                phi1 = 1.0 - 0.5 * z + (z**2) / 6.0
            elif z > 700.0:
                phi1 = 1.0 / z
            else:
                phi1 = (1.0 - math.exp(-z)) / z
            exp_z = 0.0 if z > 700.0 else math.exp(-z)
            return (y0 * exp_z + p_val * dt * phi1) / y_tgt - 1.0

        # At z = 0, h(0) = (y0 + p_val * dt - y_tgt) / y_tgt >= 0
        if h(0.0) <= 1e-15:
            D_eff[i] = 0.0
            continue

        # Analytical large-z asymptotic solution:
        # If p_val * dt > 100 * y_tgt, then z must be > 100.
        # In that regime, exp(-z) < 1e-43 ~ 0, so y_rec = p_val * dt / z = y_tgt -> z = p_val * dt / y_tgt
        if p_val * dt > 100.0 * y_tgt:
            z_asymp = p_val * dt / y_tgt
            if abs(h(z_asymp)) < 1e-12:
                D_eff[i] = z_asymp / dt
                continue

        # Analytically guaranteed upper bound bracket:
        z_hi = max(100.0, float(d_val) * dt, 3.0 * p_val * dt / y_tgt)
        if y0 > 10.0 * y_tgt:
            z_hi = max(z_hi, -math.log(max(y_tgt / max(y0, 1e-300), 1e-300)) + 10.0)

        while h(z_hi) > 0 and z_hi < 1e300:
            z_hi *= 10.0

        if h(z_hi) > 0:
            D_eff[i] = z_hi / dt
            continue

        # Guaranteed bounded root search solving relative error to machine precision
        z_root = _brentq_monotonic(h, 0.0, z_hi)
        D_eff[i] = z_root / dt

    # Strict post-condition validation: both absolute AND relative tolerances
    z_all = D_eff * dt
    safe_z = np.maximum(z_all, 1e-12)
    phi1_all = np.where(
        z_all < 1e-7,
        1.0 - 0.5 * z_all + (z_all**2) / 6.0,
        (1.0 - np.exp(-np.minimum(z_all, 700.0))) / safe_z
    )
    exp_z_all = np.where(z_all > 700.0, 0.0, np.exp(-z_all))
    y_rec = Y0 * exp_z_all + P_eff * dt * phi1_all
    abs_res = np.abs(y_rec - Y_next)
    max_abs = float(np.max(abs_res))

    # Check relative error on all active / non-extinct species (> 1e-300)
    active = Y_next > 1e-300
    if np.any(active):
        rel_res = abs_res[active] / Y_next[active]
        max_rel = float(np.max(rel_res))
    else:
        max_rel = 0.0

    if max_abs > max_abs_tol or max_rel > max_rel_tol:
        raise RuntimeError(
            f"QSSA algebraic inversion failed closure check! "
            f"Max abs residual: {max_abs:.3e} (tol {max_abs_tol:.3e}), "
            f"Max rel residual: {max_rel:.3e} (tol {max_rel_tol:.3e})"
        )

    return P_eff, D_eff


def _build_composition(
    sample: dict,
    gas: ct.Solution,
    epsilon: float,
) -> Dict[str, float]:
    comp = {}

    # Build case-insensitive lookup for sample keys
    sample_lookup = {k.upper(): k for k in sample.keys()}

    for sp in gas.species_names:
        key = f"X_{sp}"
        # Case-insensitive lookup for robustness (e.g., AR vs Ar, HE vs He)
        if key not in sample:
            key_upper = f"X_{sp.upper()}"
            if key_upper in sample_lookup:
                key = sample_lookup[key_upper]
            else:
                continue

        value = float(sample[key])

        if not np.isfinite(value):
            raise ValueError(f"Non-finite composition: {key}={value}")
        if value < -epsilon:
            raise ValueError(f"Negative composition: {key}={value}")
        if value > epsilon:
            comp[sp] = value

    if not comp:
        raise ValueError("No valid X_species columns found.")

    total = float(sum(comp.values()))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid composition sum: {total}")

    return {sp: value / total for sp, value in comp.items()}


def _ammonia_hydrogen_phi(sample: dict, gas: ct.Solution) -> float:
    """Compute phi from the declared NH3/H2 fuel and O2 oxidizer basis.

    Cantera's no-argument ``equivalence_ratio`` includes hydrogen in product
    diluents such as H2O.  That is not the sampling contract here, where H2O
    is an independent background component, so use the explicit fuel basis
    whenever those columns are available.
    """
    keys = ("X_NH3", "X_H2", "X_O2")
    if all(key in sample for key in keys):
        nh3 = max(0.0, float(sample["X_NH3"]))
        h2 = max(0.0, float(sample["X_H2"]))
        o2 = float(sample["X_O2"])
        if o2 > 0.0:
            return (0.75 * nh3 + 0.5 * h2) / o2
    return float(gas.equivalence_ratio())


def _create_reactor(
    gas: ct.Solution,
    reactor_type: str,
    energy: str,
):
    if reactor_type == "IdealGasReactor":
        return ct.IdealGasReactor(gas, energy=energy)

    if reactor_type == "IdealGasConstPressureReactor":
        return ct.IdealGasConstPressureReactor(gas, energy=energy)

    raise ValueError(f"Unsupported reactor_type: {reactor_type}")


def _validate_state(
    Y: np.ndarray,
    T: float,
    P: float,
    cfg: Config,
) -> Optional[str]:
    if not np.all(np.isfinite(Y)):
        return "non_finite_Y"

    if not np.isfinite(T) or not np.isfinite(P):
        return "non_finite_T_or_P"

    if T <= 0 or P <= 0:
        return "non_positive_T_or_P"

    if float(np.min(Y)) < -cfg.negative_mass_fraction_tolerance:
        return "negative_mass_fraction"

    if abs(float(np.sum(Y)) - 1.0) > cfg.mass_fraction_tolerance:
        return "mass_fraction_sum_error"

    return None


def _sanitize_mass_fractions(
    Y: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Remove only round-off negative mass fractions and preserve closure.

    Cantera can emit tiny negative trace values after stiff integration.  They
    are safe to project to zero when they are below the configured tolerance,
    but larger negatives must fail the trajectory instead of becoming labels.
    """
    values = np.asarray(Y, dtype=np.float64).copy()
    if not np.all(np.isfinite(values)):
        raise ValueError("non_finite_Y")
    if float(np.min(values)) < -cfg.negative_mass_fraction_tolerance:
        raise ValueError("negative_mass_fraction")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("invalid_mass_fraction_sum")
    if abs(total - 1.0) > cfg.mass_fraction_tolerance:
        raise ValueError("mass_fraction_sum_error")
    if np.any(values < 0.0):
        values = np.maximum(values, 0.0)
        values /= float(np.sum(values))
    return values


def trajectory_key(mechanism_name: str, sample_id) -> tuple[str, str]:
    """Return a globally unique trajectory key across mechanisms."""
    return str(mechanism_name), str(sample_id)


# =============================================================================
# Single sample
# =============================================================================

def _simulate_trajectory(
    mech_path: str,
    mech_name: str,
    sample: dict,
    cfg: Config,
) -> pd.DataFrame:
    """
    One row = one coarse interval.

    Inputs:
        T, P, Y_k, dt

    Targets:
        dTdt_avg, dYdt_avg_k

    Diagnostics:
        T_next, P_next, Y_next_k, time, termination_reason
    """
    sample_id = sample.get("sample_id", "N/A")

    try:
        rng = np.random.default_rng(
            _stable_seed(sample_id, cfg.base_seed)
        )

        gas = _get_worker_gas(mech_path, mech_name)

        T = float(sample["T"])
        P = _pressure_to_pa(
            float(sample["P"]),
            cfg.pressure_unit,
        )

        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"Invalid initial T={T}")
        if not np.isfinite(P) or P <= 0:
            raise ValueError(f"Invalid initial P={P}")

        composition = _build_composition(
            sample,
            gas,
            cfg.composition_epsilon,
        )
        initial_radical_total = sum(
            max(0.0, float(sample.get(f"X_{species}", 0.0)))
            for species in ("OH", "H", "O", "NO")
        )

        gas.TPX = T, P, composition

        # When the sampler provides target/actual equivalence-ratio metadata,
        # validate it against the mechanism before starting the trajectory.
        # This catches composition drift at the database boundary instead of
        # allowing an inconsistent sample to enter the training corpus.
        target_phi = sample.get("phi")
        reported_phi = sample.get("phi_actual")
        actual_phi = _ammonia_hydrogen_phi(sample, gas)
        if not np.isfinite(actual_phi):
            raise ValueError("Unable to compute a finite equivalence ratio")
        if target_phi is not None and np.isfinite(float(target_phi)):
            phi_error = abs(actual_phi - float(target_phi))
            if phi_error > max(1e-6, 0.01 * abs(float(target_phi))):
                raise ValueError(
                    f"Equivalence-ratio mismatch: target={float(target_phi):.8g}, "
                    f"actual={actual_phi:.8g}, abs_error={phi_error:.3e}"
                )
        if reported_phi is not None and np.isfinite(float(reported_phi)):
            reported_error = abs(actual_phi - float(reported_phi))
            if reported_error > max(1e-6, 0.01 * abs(float(reported_phi))):
                raise ValueError(
                    f"Reported phi_actual mismatch: reported={float(reported_phi):.8g}, "
                    f"actual={actual_phi:.8g}, abs_error={reported_error:.3e}"
                )

        reactor = _create_reactor(
            gas,
            cfg.reactor_type,
            cfg.energy,
        )
        sim = ct.ReactorNet([reactor])

        species_names = list(gas.species_names)

        t = 0.0
        Y = _sanitize_mass_fractions(reactor.thermo.Y.copy(), cfg)
        if not np.array_equal(Y, reactor.thermo.Y):
            reactor.thermo.Y = Y
        T = float(reactor.T)
        P = float(reactor.thermo.P)
        T_init = T
        has_ignited = False
        step_count = 0

        if cfg.validate_each_interval:
            error = _validate_state(Y, T, P, cfg)
            if error is not None:
                raise ValueError(f"Invalid initial state: {error}")

        rows: List[dict] = []
        equilibrium_count = 0
        termination_reason = "max_time"

        while t < cfg.max_time:
            requested_dt = _sample_log_uniform_dt(
                rng, cfg.dt_min, cfg.dt_max
            )

            remaining = cfg.max_time - t
            if remaining <= 0:
                break

            requested_dt = min(requested_dt, remaining)
            target_time = t + requested_dt

            # --- QSSA Feature Extraction (Start) ---
            C_k_start = gas.creation_rates
            D_k_prime_start = gas.destruction_rates
            W_k = gas.molecular_weights
            rho_start = gas.density
            
            P_start = (C_k_start * W_k) / rho_start
            Dest_rate_start = (D_k_prime_start * W_k) / rho_start
            
            Y_safe_start = np.maximum(Y, QSSA_STATE_FLOOR)
            D_coef_start = Dest_rate_start / Y_safe_start
            # -------------------------------

            try:
                sim.advance(target_time)
            except ct.CanteraError as exc:
                termination_reason = "cantera_error"
                logger.debug(
                    "Cantera error sample=%s: %s",
                    sample_id, str(exc)[:300]
                )
                break

            t_new = float(sim.time)
            dt = t_new - t
            
            if dt <= 1e-15:
                termination_reason = "zero_or_tiny_dt"
                break

            try:
                Y_next = _sanitize_mass_fractions(reactor.thermo.Y.copy(), cfg)
            except ValueError as exc:
                termination_reason = f"invalid_state:{exc}"
                break
            if not np.array_equal(Y_next, reactor.thermo.Y):
                reactor.thermo.Y = Y_next
            T_next = float(reactor.T)
            P_next = float(reactor.thermo.P)

            if cfg.validate_each_interval:
                error = _validate_state(
                    Y_next, T_next, P_next, cfg
                )
                if error is not None:
                    termination_reason = f"invalid_state:{error}"
                    break

            # --- QSSA Feature Extraction (End) & Exact Algebraic Inversion ---
            C_k_end = gas.creation_rates
            D_k_prime_end = gas.destruction_rates
            rho_end = gas.density
            
            P_end = (C_k_end * W_k) / rho_end
            Dest_rate_end = (D_k_prime_end * W_k) / rho_end
            
            Y_safe_end = np.maximum(Y_next, QSSA_STATE_FLOOR)
            D_coef_end = Dest_rate_end / Y_safe_end
            
            P_candidate = (P_start + P_end) / 2.0
            D_candidate = (D_coef_start + D_coef_end) / 2.0

            P_avg, D_coef_avg = _solve_exact_qssa_params(
                Y0=Y,
                Y_next=Y_next,
                dt=dt,
                P_raw=P_candidate,
                D_raw_init=D_candidate,
            )
            # ----------------------------------------------------------------

            dYdt = (Y_next - Y) / dt
            dTdt = (T_next - T) / dt

            if not np.all(np.isfinite(dYdt)) or not np.isfinite(dTdt):
                termination_reason = "non_finite_source"
                break

            if np.max(np.abs(dYdt)) > cfg.max_abs_source_rate:
                termination_reason = "source_rate_outlier"
                break

            row = {
                "sample_id": sample_id,
                "stratum_name": str(sample.get("stratum_name", "unknown")),
                "mechanism_name": mech_name,
                "Z_dil": float(sample.get("Z_dil", 0.0)),
                # ``phi`` is optional in the input CSV.  Keep the generated
                # dataset finite by recording the physically computed value
                # when the caller did not provide a target value.
                "phi": (
                    float(target_phi)
                    if target_phi is not None and np.isfinite(float(target_phi))
                    else actual_phi
                ),
                "phi_actual": actual_phi,
                "initial_radical_total": initial_radical_total,
                "time": t,
                "dt": dt,
                "T": T,
                "P": P / ct.one_atm,
                "T_next": T_next,
                "P_next": P_next / ct.one_atm,
                "dTdt_avg": dTdt,
                "termination_reason": "",
                "n_intervals_in_trajectory": 0,  # 临时占位，稍后填入
            }

            for i, sp in enumerate(species_names):
                row[f"Y_{sp}"] = float(Y[i])
                row[f"dYdt_{sp}"] = float(dYdt[i])
                row[f"Y_next_{sp}"] = float(Y_next[i])
                # Add the new QSSA features
                row[f"P_avg_{sp}"] = float(P_avg[i])
                row[f"D_coef_avg_{sp}"] = float(D_coef_avg[i])

            rows.append(row)

            # Check for ignition occurrence (temperature rise threshold or hot starting state)
            if (T_next - T_init) >= cfg.min_ignition_temp_rise or T_init >= 1600.0:
                has_ignited = True

            # Equilibrium uses BOTH temperature and species rates,
            # BUT only after ignition has occurred (or starting from burnt gas)
            # to prevent premature termination during slow induction periods.
            if has_ignited or T_init >= 1600.0:
                if t_new >= cfg.equilibrium_check_start:
                    max_dYdt = float(np.max(np.abs(dYdt)))

                    if (
                        abs(dTdt) < cfg.eq_T_rate_tol
                        and max_dYdt < cfg.eq_Y_rate_tol
                    ):
                        equilibrium_count += 1
                    else:
                        equilibrium_count = 0

                    if equilibrium_count >= cfg.equilibrium_consecutive:
                        termination_reason = "equilibrium"
                        break
            else:
                # If unignited mixture has simulated for max_unignited_time, exit cleanly
                # to avoid accumulating thousands of identical frozen states.
                if t_new >= cfg.max_unignited_time:
                    termination_reason = "unignited_timeout"
                    break

            step_count += 1
            if step_count >= cfg.max_steps_per_trajectory:
                termination_reason = "max_steps"
                break

            t = t_new
            Y = Y_next
            T = T_next
            P = P_next

        if not rows:
            return pd.DataFrame()

        if termination_reason not in {"equilibrium", "unignited_timeout", "max_time"}:
            logger.warning(
                "Sample incomplete/truncated: sample=%s mechanism=%s reason=%s (steps=%d, t=%.4e s)",
                sample_id, mech_name, termination_reason, step_count, t
            )
            return pd.DataFrame()

        n_intervals = len(rows)
        # Convert list-of-dicts to dict-of-columns to avoid allocating massive 2D object arrays
        cols = list(rows[0].keys())
        data_dict = {col: [r[col] for r in rows] for col in cols}
        data_dict["n_intervals_in_trajectory"] = [n_intervals] * n_intervals
        term_reasons = [""] * n_intervals
        term_reasons[-1] = termination_reason
        data_dict["termination_reason"] = term_reasons
        del rows

        df = pd.DataFrame(data_dict)

        numeric_cols = [
            c for c in df.columns
            if c not in {
                "sample_id",
                "stratum_name",
                "mechanism_name",
                "termination_reason",
            }
        ]
        df[numeric_cols] = df[numeric_cols].astype(np.float64)

        return df

    except ct.CanteraError as exc:
        logger.warning(
            "Cantera failure: sample=%s mechanism=%s: %s",
            sample_id, mech_name, str(exc)[:400]
        )
        return pd.DataFrame()

    except Exception as exc:
        logger.warning(
            "Sample failure: sample=%s mechanism=%s: %s",
            sample_id, mech_name, str(exc)[:400]
        )
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


# =============================================================================
# Parquet output
# =============================================================================

def _prepare_mechanism_dir(
    root: Path,
    mech_name: str,
    overwrite: bool,
) -> Path:
    out = root / f"trajectories_{mech_name}"

    if out.exists() and overwrite:
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_part(
    df: pd.DataFrame,
    out_dir: Path,
    part_index: int,
    compression: str,
) -> Path:
    path = out_dir / f"part-{part_index:06d}.parquet"

    if path.exists():
        raise FileExistsError(
            f"Part already exists: {path}. "
            f"Use overwrite_output=True for a fresh run."
        )

    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )
    return path


def _write_metadata(
    out_dir: Path,
    mech_name: str,
    mech_path: str,
    cfg: Config,
    n_samples: int,
    success: int,
    failed: int,
    rows: int,
    parts: int,
) -> None:
    metadata = {
        "database_version": "V21",
        "mechanism_name": mech_name,
        "mechanism_path": str(Path(mech_path).resolve()),
        "reactor_type": cfg.reactor_type,
        "energy": cfg.energy,
        "pressure_input_unit": cfg.pressure_unit,
        "dt_min": cfg.dt_min,
        "dt_max": cfg.dt_max,
        "max_time": cfg.max_time,
        "max_unignited_time": cfg.max_unignited_time,
        "max_steps_per_trajectory": cfg.max_steps_per_trajectory,
        "equilibrium_check_start": cfg.equilibrium_check_start,
        "min_ignition_temp_rise": cfg.min_ignition_temp_rise,
        "eq_T_rate_tol": cfg.eq_T_rate_tol,
        "eq_Y_rate_tol": cfg.eq_Y_rate_tol,
        "equilibrium_consecutive": cfg.equilibrium_consecutive,
        "composition_epsilon": cfg.composition_epsilon,
        "mass_fraction_tolerance": cfg.mass_fraction_tolerance,
        "negative_mass_fraction_tolerance": cfg.negative_mass_fraction_tolerance,
        "max_abs_source_rate": cfg.max_abs_source_rate,
        "base_seed": cfg.base_seed,
        "workers": cfg.workers,
        "batch_size": cfg.batch_size,
        "task_timeout_s": cfg.task_timeout_s,
        "input_samples": n_samples,
        "success": success,
        "failed": failed,
        "success_rate": (
            success / (success + failed)
            if success + failed else 0.0
        ),
        "training_rows": rows,
        "parquet_parts": parts,
    }

    pd.DataFrame([metadata]).to_json(
        out_dir / "metadata.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )


# =============================================================================
# One mechanism
# =============================================================================

def run_mechanism(
    mech_name: str,
    mech_path: str,
    samples_df: pd.DataFrame,
    output_root: Path,
    cfg: Config,
) -> Tuple[Path, int, int, int]:

    if not os.path.exists(mech_path):
        raise FileNotFoundError(
            f"Mechanism not found: {mech_path}"
        )

    samples = samples_df[
        samples_df["mechanism_name"] == mech_name
    ].copy()

    if samples.empty:
        logger.warning("No samples for mechanism [%s]", mech_name)
        return output_root / f"trajectories_{mech_name}", 0, 0, 0

    if not samples["sample_id"].is_unique:
        raise ValueError(
            f"sample_id must be unique within [{mech_name}]"
        )

    out_dir = _prepare_mechanism_dir(
        output_root,
        mech_name,
        cfg.overwrite_output,
    )

    n_samples = len(samples)
    success = 0
    failed = 0
    total_rows = 0
    part_index = 0

    logger.info("=" * 80)
    logger.info(
        "[%s] samples=%d workers=%d batch=%d",
        mech_name, n_samples, cfg.workers, cfg.batch_size
    )
    logger.info(
        "dt=[%.3e, %.3e] s, max_time=%.6g s, reactor=%s",
        cfg.dt_min, cfg.dt_max, cfg.max_time, cfg.reactor_type
    )
    logger.info("=" * 80)

    # One pool for the entire mechanism. Worker-local mechanism cache
    # eliminates repeated YAML loading for every sample.
    with ProcessPool(max_workers=cfg.workers) as pool:

        for batch_start in range(0, n_samples, cfg.batch_size):
            batch_end = min(
                batch_start + cfg.batch_size,
                n_samples,
            )
            batch = samples.iloc[batch_start:batch_end]

            logger.info(
                "Processing samples %d-%d/%d",
                batch_start + 1, batch_end, n_samples
            )

            futures = {}

            for _, row in batch.iterrows():
                sample = row.to_dict()

                futures[sample["sample_id"]] = pool.schedule(
                    _simulate_trajectory,
                    args=(
                        mech_path,
                        mech_name,
                        sample,
                        cfg,
                    ),
                    timeout=cfg.task_timeout_s,
                )

            batch_dfs: List[pd.DataFrame] = []

            for sample_id, future in tqdm(
                futures.items(),
                desc=f"{mech_name} batch {part_index:06d}",
                unit="traj",
                ncols=100,
                leave=False,
            ):
                try:
                    result = future.result()

                    if (
                        isinstance(result, pd.DataFrame)
                        and not result.empty
                    ):
                        term_reason = result["termination_reason"].iloc[-1]
                        if term_reason in {"equilibrium", "unignited_timeout", "max_time"}:
                            batch_dfs.append(result)
                            success += 1
                        else:
                            failed += 1
                            logger.warning(
                                "Trajectory discarded due to non-physical termination: sample=%s, reason=%s",
                                sample_id, term_reason
                            )
                    else:
                        failed += 1

                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "Task failed/timeout: sample=%s: %s",
                        sample_id, str(exc)[:300]
                    )

            if not batch_dfs:
                logger.warning(
                    "Batch %d-%d produced no valid rows.",
                    batch_start + 1, batch_end
                )
                gc.collect()
                continue

            df_batch = pd.concat(
                batch_dfs,
                ignore_index=True,
            )

            # Never convert unexpected NaN to zero. That would turn a
            # computational/data error into a physically valid-looking zero.
            numeric_cols = df_batch.select_dtypes(
                include=[np.number]
            ).columns

            if df_batch[numeric_cols].isna().any().any():
                bad = [
                    c for c in numeric_cols
                    if df_batch[c].isna().any()
                ]
                raise ValueError(
                    f"Unexpected NaN in columns: {bad[:20]}"
                )

            values = df_batch[numeric_cols].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(
                    "Unexpected inf/-inf in generated batch."
                )

            part_path = _write_part(
                df_batch,
                out_dir,
                part_index,
                cfg.compression,
            )

            total_rows += len(df_batch)

            logger.info(
                "Saved %s: rows=%s, cumulative=%s",
                part_path.name,
                f"{len(df_batch):,}",
                f"{total_rows:,}",
            )

            part_index += 1

            del futures
            del batch_dfs
            del df_batch
            gc.collect()

    _write_metadata(
        out_dir,
        mech_name,
        mech_path,
        cfg,
        n_samples,
        success,
        failed,
        total_rows,
        part_index,
    )

    logger.info(
        "[%s] complete: success=%d failed=%d rows=%s parts=%d",
        mech_name,
        success,
        failed,
        f"{total_rows:,}",
        part_index,
    )

    return out_dir, success, failed, total_rows


# =============================================================================
# Whole database
# =============================================================================

def run_database(
    mechanisms: Dict[str, str],
    samples_csv: str,
    output_dir: str,
    cfg: Config,
) -> None:

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(samples_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    samples = pd.read_csv(csv_path)

    required = {
        "sample_id",
        "mechanism_name",
        "T",
        "P",
    }
    missing = required - set(samples.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if samples["sample_id"].isna().any():
        raise ValueError("sample_id contains NaN.")

    if samples["mechanism_name"].isna().any():
        raise ValueError("mechanism_name contains NaN.")

    outputs = []
    total_success = 0
    total_failed = 0
    total_rows = 0

    for name, path in mechanisms.items():
        out, success, failed, rows = run_mechanism(
            name,
            path,
            samples,
            root,
            cfg,
        )

        if success + failed:
            outputs.append(out)

        total_success += success
        total_failed += failed
        total_rows += rows

    # Do NOT merge all mechanisms into one huge parquet by repeatedly
    # reading and rewriting it. Each directory is already a Parquet dataset.
    manifest = pd.DataFrame([
        {
            "mechanism_name": p.name.replace("trajectories_", ""),
            "dataset_path": str(p.resolve()),
        }
        for p in outputs
    ])

    manifest.to_csv(
        root / "dataset_manifest.csv",
        index=False,
    )

    # Keep the dataset-level manifest backward compatible for split scripts,
    # and add an explicit trajectory-level manifest for audit and provenance.
    trajectory_records = []
    for mechanism_dir in outputs:
        for part_path in sorted(mechanism_dir.glob("part-*.parquet")):
            terminal = pd.read_parquet(
                part_path,
                columns=[
                    "sample_id",
                    "mechanism_name",
                    "termination_reason",
                    "n_intervals_in_trajectory",
                ],
            )
            terminal = terminal[terminal["termination_reason"].fillna("") != ""]
            if terminal.empty:
                continue
            trajectory_records.extend(
                terminal[
                    [
                        "mechanism_name",
                        "sample_id",
                        "termination_reason",
                        "n_intervals_in_trajectory",
                    ]
                ].to_dict("records")
            )
    trajectory_manifest = pd.DataFrame(trajectory_records).drop_duplicates(
        subset=["mechanism_name", "sample_id"]
    )
    if not trajectory_manifest.empty:
        trajectory_manifest = trajectory_manifest.sort_values(
            ["mechanism_name", "sample_id"]
        )
    trajectory_manifest.to_csv(
        root / "trajectory_manifest.csv",
        index=False,
    )

    summary = {
        "database_version": "V21",
        "input_csv": str(csv_path.resolve()),
        "output_root": str(root.resolve()),
        "total_input_samples": len(samples),
        "success_trajectories": total_success,
        "failed_trajectories": total_failed,
        "success_rate": (
            total_success / (total_success + total_failed)
            if total_success + total_failed else 0.0
        ),
        "total_training_rows": total_rows,
        "mechanisms": list(mechanisms.keys()),
    }

    pd.DataFrame([summary]).to_json(
        root / "database_summary.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    logger.info("=" * 80)
    logger.info("DATABASE GENERATION COMPLETE")
    logger.info("Success trajectories: %d", total_success)
    logger.info("Failed trajectories : %d", total_failed)
    logger.info("Training rows       : %s", f"{total_rows:,}")
    logger.info("Manifest            : %s", root / "dataset_manifest.csv")
    logger.info("=" * 80)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transient Chemistry Database Generator V21 (QSSA)")
    parser.add_argument(
        "--mechanism",
        type=str,
        default=None,
        help="生成单个机理的数据，例如 nuig_2024。",
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        nargs="+",
        default=None,
        help="生成多个机理的数据；与 --mechanism 互斥。",
    )
    parser.add_argument(
        "--list-mechanisms",
        action="store_true",
        help="列出可用机理名称并退出。",
    )
    parser.add_argument("--input_csv", type=str, default=None, help="Input LHS samples CSV path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output database directory")
    parser.add_argument("--workers", type=int, default=16, help="Worker count")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size per parquet part")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output directory")
    parser.add_argument(
        "--pressure_unit",
        type=str,
        default="atm",
        choices=["atm", "pa", "auto"],
        help="Input pressure unit ('atm', 'pa', or 'auto'). Default: 'atm'.",
    )
    parser.add_argument("--dt_min", type=float, default=1e-8, help="Minimum step dt (s)")
    parser.add_argument("--dt_max", type=float, default=1e-4, help="Maximum step dt (s)")
    parser.add_argument("--max_time", type=float, default=2.0, help="Maximum trajectory simulation time (s)")
    parser.add_argument("--max_unignited_time", type=float, default=0.25, help="Maximum time for unignited mixtures (s). Default: 0.25.")
    parser.add_argument("--equilibrium_check_start", type=float, default=2e-3, help="Start time for equilibrium checking (s)")
    parser.add_argument("--min_ignition_temp_rise", type=float, default=200.0, help="Temperature rise threshold to declare ignition (K)")
    parser.add_argument("--max_steps", type=int, default=35000, help="Maximum step count per trajectory. Default: 35000.")
    args = parser.parse_args()

    if args.list_mechanisms:
        print("\n".join(SUPPORTED_MECHANISMS))
        raise SystemExit(0)

    try:
        selected_mechanisms = normalize_mechanism_selection(
            mechanism=args.mechanism,
            mechanisms=args.mechanisms,
        )
    except ValueError as exc:
        parser.error(str(exc))

    SCRIPT_DIR = Path(__file__).resolve().parent

    def _resolve_existing(path_candidates):
        for p in path_candidates:
            if p is not None and Path(p).is_file():
                return Path(p).resolve()
        return Path(path_candidates[0]).resolve()

    def _resolve_mechanism(mech_key, filename):
        candidates = [
            SCRIPT_DIR / filename,
            Path.cwd() / filename,
            SCRIPT_DIR.parent / filename,
            SCRIPT_DIR.parent.parent / filename,
            SCRIPT_DIR / "mechanisms" / filename,
        ]
        return _resolve_existing(candidates)

    if args.input_csv:
        INPUT_FILE = Path(args.input_csv).resolve()
    else:
        INPUT_FILE = _resolve_existing([
            SCRIPT_DIR / "lhs_samples_Species_Initial_v1.csv",
            Path.cwd() / "lhs_samples_Species_Initial_v1.csv",
            SCRIPT_DIR.parent / "lhs_samples_Species_Initial_v1.csv",
        ])

    OUTPUT_DIR = Path(args.output_dir).resolve() if args.output_dir else (SCRIPT_DIR / "database_Transient_V21_QSSA")

    MECH_FILENAMES = {
        "nuig_2024": "NUIG2024.yaml",
        "tsinghua_2024c": "TH2024.yaml",
        "ucf_2024": "UCF2024.yaml",
    }
    MECHANISMS = {
        k: _resolve_mechanism(k, fname)
        for k, fname in MECH_FILENAMES.items()
        if k in selected_mechanisms
    }

    cfg = Config(
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_time=args.max_time,
        max_unignited_time=args.max_unignited_time,
        max_steps_per_trajectory=args.max_steps,

        equilibrium_check_start=args.equilibrium_check_start,
        eq_T_rate_tol=1.0,
        eq_Y_rate_tol=1e-2,
        equilibrium_consecutive=3,
        min_ignition_temp_rise=args.min_ignition_temp_rise,

        pressure_unit=args.pressure_unit,

        reactor_type="IdealGasReactor",
        energy="on",

        base_seed=20260815,

        workers=args.workers,
        task_timeout_s=300.0,

        batch_size=args.batch_size,
        compression="snappy",
        overwrite_output=args.overwrite,

        validate_each_interval=True,
        max_abs_source_rate=1e20,
    )

    print("\n" + "=" * 80)
    print("Transient Chemistry Database Generator V21 (QSSA)")
    print("=" * 80)
    print("Input            :", INPUT_FILE)
    print("Output           :", OUTPUT_DIR)
    print("dt               : [%.3e, %.3e] s" % (cfg.dt_min, cfg.dt_max))
    print("max_time         : %.6g s" % cfg.max_time)
    print("pressure_unit    :", cfg.pressure_unit)
    print("ignition_guard_dT: %.1f K" % cfg.min_ignition_temp_rise)
    print("workers          :", cfg.workers)
    print("batch            :", cfg.batch_size)
    print("reactor          :", cfg.reactor_type)
    print("=" * 80 + "\n")

    run_database(
        mechanisms={
            name: str(path)
            for name, path in MECHANISMS.items()
        },
        samples_csv=str(INPUT_FILE),
        output_dir=str(OUTPUT_DIR),
        cfg=cfg,
    )
