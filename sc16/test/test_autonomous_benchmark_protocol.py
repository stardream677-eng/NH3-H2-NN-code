"""Regression tests for complete, time-aligned H=128 benchmark statistics."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autonomous_benchmark_protocol as ab


def _traces():
    t = np.array([0., 1., 2., 3., 4.]) * 1e-3
    ref_T = np.array([1000., 1090., 1260., 1420., 1700.])
    pred_T = np.array([1000., 1020., 1190., 1390., 1630.])
    ref_P = np.array([10., 11., 12., 13., 14.])
    pred_P = np.array([10., 10.8, 11.9, 12.7, 13.2])
    names = ["H2", "OH", "NO"]
    Yr = np.array([
        [.60, 1e-16, .40],
        [.59, 1e-14, .41],
        [.55, 1e-7, .4499999],
        [.49, 1e-4, .5099],
        [.40, 1e-3, .599],
    ])
    Yn = Yr.copy()
    Yn[2, 1] = 2e-7
    Yn[3, 1] = 2e-4
    Yn[4, 1] = 0.0005
    Yn[:, 2] = 1 - Yn[:, 0] - Yn[:, 1]
    return t, ref_T, pred_T, ref_P, pred_P, Yr, Yn, names


def test_crossing_time_is_linearly_interpolated_not_grid_rounded():
    t = np.array([0., 1., 2.])
    T = np.array([1000., 1250., 1500.])
    assert ab.first_temperature_rise_time(t, T, 1000., 400.) == pytest.approx(1.6)


def test_idt_never_reports_event_when_window_does_not_ignite():
    t = np.linspace(0, 0.128, 129)
    T = np.linspace(900, 1040, 129)
    assert ab.first_temperature_rise_time(t, T, 900, 400.) is None


def test_censored_reference_vs_predicted_ignition_is_disagreement():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    no_ignition = np.full_like(nt, 1000.)
    m = ab.compare_complete_trajectories(t, rt, no_ignition, rp, np_, yr, yn, names)
    assert m["ref_ignited"]
    assert not m["nn_ignited"]
    assert m["ignition_agreement"] is False
    assert m["idt_nn_s"] is None
    assert m["idt_abs_error_s"] is None
    assert m["idt_ratio"] is None
    assert m["idt_censored_at_s"] == pytest.approx(t[-1])


def test_both_nonignited_is_agreement_but_not_zero_idt_error():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    a = np.full_like(rt, 1000.)
    b = np.full_like(nt, 1020.)
    m = ab.compare_complete_trajectories(t, a, b, rp, np_, yr, yn, names)
    assert m["ignition_agreement"] is True
    assert m["idt_abs_error_s"] is None
    assert m["idt_ref_s"] is None


def test_full_trajectory_metrics_include_peaks_timing_and_radicals():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    m = ab.compare_complete_trajectories(t, rt, nt, rp, np_, yr, yn, names)
    assert m["temperature_K"]["mae"] > 0
    assert m["temperature_K"]["peak_time_abs_error_s"] >= 0
    assert m["species"]["OH"]["active_log10_rmse"] > 0
    assert m["species"]["OH"]["peak_amplitude_abs_error"] > 0
    assert m["species"]["NO"]["endpoint_abs_error"] > 0
    assert m["idt_ref_s"] == pytest.approx(3e-3 - (20. / 160.) * 1e-3)
    assert m["idt_nn_s"] == pytest.approx(3e-3 + (10. / 240.) * 1e-3)
    assert m["idt_abs_error_s"] > 0


def test_nox_is_no_plus_no2_not_no_alone():
    t = np.array([0.0, 1e-3, 2e-3])
    T = np.array([1000.0, 1200.0, 1500.0])
    P = np.ones(3)
    names = ["NO", "NO2", "O2"]
    true = np.array([[0.01, 0.002, .988], [.02, .003, .977], [.03, .004, .966]])
    pred = np.array([[0.01, 0.002, .988], [.022, .002, .976], [.025, .006, .969]])
    result = ab.compare_complete_trajectories(t, T, T, P, P, true, pred, names)
    nox = result["species"]["NOx_mass_fraction"]
    assert nox["endpoint_abs_error"] == pytest.approx(abs((.025 + .006) - (.03 + .004)))

    no_no2 = ab.compare_complete_trajectories(
        t, T, T, P, P, true[:, [0, 2]], pred[:, [0, 2]], ["NO", "O2"]
    )
    assert "NOx_mass_fraction" not in no_no2["species"]


def test_nonfinite_species_trajectory_is_hard_failure():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    yn[2, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite species"):
        ab.compare_complete_trajectories(t, rt, nt, rp, np_, yr, yn, names)


def test_misaligned_times_or_species_are_rejected():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    with pytest.raises(ValueError, match="same time grid"):
        ab.compare_complete_trajectories(t, rt[:-1], nt, rp, np_, yr, yn, names)
    with pytest.raises(ValueError, match="matching one-dimensional"):
        ab.first_temperature_rise_time(t, nt[:-1], nt[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        ab.compare_complete_trajectories(t[::-1], rt, nt, rp, np_, yr, yn, names)


def test_summary_never_hides_failed_case_or_averages_censored_idt():
    t, rt, nt, rp, np_, yr, yn, names = _traces()
    full = ab.compare_complete_trajectories(t, rt, nt, rp, np_, yr, yn, names)
    censored = ab.compare_complete_trajectories(
        t, np.full_like(rt, 1000.), np.full_like(nt, 1100.),
        rp, np_, yr, yn, names,
    )
    cases = [
        {"name": "ok", "status": "completed", "metrics": full,
         "physics": {"max_mass_residual": 1e-7, "max_element_residual": 1e-8,
                     "max_energy_residual_J_per_kg": 0.5, "max_eos_relative_residual": 1e-7}},
        {"name": "no-ignition", "status": "completed", "metrics": censored,
         "physics": {"max_mass_residual": 1e-7, "max_element_residual": 1e-8,
                     "max_energy_residual_J_per_kg": 0.5, "max_eos_relative_residual": 1e-7}},
        {"name": "unstable", "status": "failed", "failure_step": 28,
         "failure_message": "NASA7 did not converge"},
    ]
    summary = ab.summarize_cases(cases)
    assert summary["cases"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["completion_rate"] == pytest.approx(2 / 3)
    assert summary["failure_names"] == ["unstable"]
    assert summary["cases_with_both_idt"] == 1
    assert summary["mean_abs_idt_error_s_both_ignited"] == pytest.approx(full["idt_abs_error_s"])
    assert summary["matched_ignition_status_cases"] == 2


def test_summary_rejects_empty_results():
    with pytest.raises(ValueError, match="At least one case"):
        ab.summarize_cases([])
