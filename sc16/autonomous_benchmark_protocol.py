"""Independent, fixed-grid trajectory metrics for NH3/H2 autonomous benchmarking.

Metrics intentionally use *full, time-aligned reactor trajectories*, never
window endpoints or Cantera-corrected network states. A missing ignition event
is right-censored at t_end; it must not be counted as a zero ignition delay.
"""
from __future__ import annotations

from typing import Any
import numpy as np

BENCHMARK_PROTOCOL = "autonomous_h128_trajectories_v1"
DEFAULT_IDT_RISE_K = 400.0
RADICALS = ("OH", "H", "O", "HO2", "NH2", "HNO")
KEY_SPECIES = ("NH3", "H2", "O2", "H2O", "OH", "H", "O", "NO", "N2O")


def first_temperature_rise_time(
    times_s: np.ndarray, temperatures_K: np.ndarray, initial_temperature_K: float,
    rise_K: float = DEFAULT_IDT_RISE_K,
) -> float | None:
    """First T=T_initial+rise_K crossing, linearly interpolated on fixed grid."""
    t = np.asarray(times_s, dtype=np.float64)
    T = np.asarray(temperatures_K, dtype=np.float64)
    if t.ndim != 1 or T.shape != t.shape or t.size < 2:
        raise ValueError("IDT needs matching one-dimensional time/temperature arrays")
    if not (np.isfinite(t).all() and np.isfinite(T).all()
            and np.isfinite(initial_temperature_K) and np.isfinite(rise_K)
            and rise_K > 0 and (np.diff(t) > 0).all()):
        raise ValueError("IDT requires finite, increasing times, finite T, positive rise_K")
    threshold = initial_temperature_K + rise_K
    crossed = np.flatnonzero(T >= threshold)
    if not crossed.size:
        return None
    j = int(crossed[0])
    if j == 0:
        return float(t[0])
    a, b = float(T[j - 1]), float(T[j])
    if b <= a:
        return float(t[j])
    fraction = np.clip((threshold - a) / (b - a), 0.0, 1.0)
    return float(t[j - 1] + fraction * (t[j] - t[j - 1]))


def _trace_stats(times: np.ndarray, ref: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if ref.shape != times.shape or pred.shape != times.shape:
        raise ValueError("Physical traces must use the same time grid")
    if not np.isfinite(ref).all() or not np.isfinite(pred).all():
        raise ValueError("Cannot score non-finite trajectory")
    ref_peak = int(np.argmax(ref))
    pred_peak = int(np.argmax(pred))
    return {
        "mae": float(np.mean(np.abs(pred - ref))),
        "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "endpoint_abs_error": float(abs(pred[-1] - ref[-1])),
        "peak_ref": float(ref[ref_peak]),
        "peak_pred": float(pred[pred_peak]),
        "peak_amplitude_abs_error": float(abs(pred[pred_peak] - ref[ref_peak])),
        "peak_time_ref_s": float(times[ref_peak]),
        "peak_time_pred_s": float(times[pred_peak]),
        "peak_time_abs_error_s": float(abs(times[pred_peak] - times[ref_peak])),
    }


def compare_complete_trajectories(
    times_s: np.ndarray,
    reference_temperature_K: np.ndarray,
    predicted_temperature_K: np.ndarray,
    reference_pressure_atm: np.ndarray,
    predicted_pressure_atm: np.ndarray,
    reference_mass_fractions: np.ndarray,
    predicted_mass_fractions: np.ndarray,
    species_names: list[str],
    *,
    idt_rise_K: float = DEFAULT_IDT_RISE_K,
) -> dict[str, Any]:
    """Score complete, independent, time-matched reference and NN trajectories.

    This is a performance comparison, not the physical-conservation audit.
    Element/mass/energy/EOS residuals are calculated separately against the
    predicted physical trajectory and *initial* reactor invariants.
    """
    t = np.asarray(times_s, dtype=np.float64)
    rt, nt, rp, np_ = [
        np.asarray(v, dtype=np.float64) for v in (
            reference_temperature_K, predicted_temperature_K,
            reference_pressure_atm, predicted_pressure_atm,
        )
    ]
    yr = np.asarray(reference_mass_fractions, dtype=np.float64)
    yn = np.asarray(predicted_mass_fractions, dtype=np.float64)
    if t.ndim != 1 or t.size < 2 or not (np.isfinite(t).all() and (np.diff(t) > 0).all()):
        raise ValueError("Benchmark times must be finite and strictly increasing")
    if yr.shape != yn.shape or yr.shape != (t.size, len(species_names)):
        raise ValueError("Both mass-fraction trajectories must be (n_times, n_species)")
    if len(species_names) != len(set(species_names)):
        raise ValueError("Species names must be unique")
    if not (np.isfinite(yr).all() and np.isfinite(yn).all()):
        raise ValueError("Non-finite species trajectories must be classified as failures")

    ref_idt = first_temperature_rise_time(t, rt, rt[0], idt_rise_K)
    pred_idt = first_temperature_rise_time(t, nt, rt[0], idt_rise_K)
    both_ignited = ref_idt is not None and pred_idt is not None
    scores: dict[str, Any] = {
        "idt_definition": f"first_T_rise_{idt_rise_K:g}K_linear_interpolation",
        "idt_ref_s": ref_idt,
        "idt_nn_s": pred_idt,
        "idt_abs_error_s": abs(pred_idt - ref_idt) if both_ignited else None,
        "idt_ratio": pred_idt / ref_idt if both_ignited and ref_idt > 0 else None,
        "ref_ignited": ref_idt is not None,
        "nn_ignited": pred_idt is not None,
        "ignition_agreement": (ref_idt is None) == (pred_idt is None),
        "idt_censored_at_s": float(t[-1]),
        "temperature_K": _trace_stats(t, rt, nt),
        "pressure_atm": _trace_stats(t, rp, np_),
        "species": {},
    }
    for index, name in enumerate(species_names):
        if name not in KEY_SPECIES:
            continue
        a, b = yr[:, index], yn[:, index]
        stats = _trace_stats(t, a, b)
        if name in RADICALS:
            active = a > 1e-12
            stats["active_log10_rmse"] = (
                float(np.sqrt(np.mean((
                    np.log10(np.maximum(b[active], 1e-40)) - np.log10(a[active])
                ) ** 2)))
                if active.any() else None
            )
        scores["species"][name] = stats
    return scores


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Surface every failed trajectory; do not average censored IDTs or failures."""
    if not cases:
        raise ValueError("At least one case is required")
    done = [c for c in cases if c.get("status") == "completed"]
    failed = [c for c in cases if c.get("status") != "completed"]
    valid_idt = [
        c["metrics"]["idt_abs_error_s"] for c in done
        if c["metrics"]["idt_abs_error_s"] is not None
    ]
    agreement = [bool(c["metrics"]["ignition_agreement"]) for c in done]
    ref_ignited = sum(bool(c["metrics"]["ref_ignited"]) for c in done)
    pred_ignited = sum(bool(c["metrics"]["nn_ignited"]) for c in done)
    return {
        "cases": len(cases),
        "completed": len(done),
        "failed": len(failed),
        "completion_rate": len(done) / len(cases),
        "failure_names": [c.get("name") for c in failed],
        "reference_ignition_cases": ref_ignited,
        "neural_ignition_cases": pred_ignited,
        "matched_ignition_status_cases": sum(agreement),
        "ignition_agreement_rate_completed": sum(agreement) / len(done) if done else None,
        "cases_with_both_idt": len(valid_idt),
        "mean_abs_idt_error_s_both_ignited": float(np.mean(valid_idt)) if valid_idt else None,
        "max_mass_residual": max((c["physics"]["max_mass_residual"] for c in done), default=None),
        "max_element_residual": max((c["physics"]["max_element_residual"] for c in done), default=None),
        "max_energy_residual_J_per_kg": max((c["physics"]["max_energy_residual_J_per_kg"] for c in done), default=None),
        "max_eos_relative_residual": max((c["physics"]["max_eos_relative_residual"] for c in done), default=None),
    }
