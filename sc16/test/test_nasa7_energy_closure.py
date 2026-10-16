"""Numerical regression tests for residual-verified autonomous thermochemistry.

Run against sc16/NUIG2024.yaml using the Cantera workstation environment.
These tests are not a substitute for full H=128 combustion rollouts.
"""
from pathlib import Path
import sys

import cantera as ct
import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from thermo_nasa7_layer import ThermoNASA7Layer
from train_sc_v68 import update_temperature_autonomous


@pytest.fixture(scope="module")
def layer():
    species = ["NH3", "H2", "O2", "N2", "H2O"]
    return ThermoNASA7Layer(PROJECT / "NUIG2024.yaml", species_order=species)


def _mass_fractions():
    y = torch.tensor(
        [[0.06, 0.04, 0.21, 0.68, 0.01],
         [0.02, 0.03, 0.15, 0.72, 0.08],
         [0.03, 0.02, 0.10, 0.68, 0.17]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(y.sum(dim=1), torch.ones(3, dtype=torch.float64))
    return y


@pytest.mark.parametrize("kind", ["uv", "hp"])
def test_nasa7_temperature_roundtrip_uses_final_energy_residual(layer, kind):
    y = _mass_fractions()
    T = torch.tensor([[950.0], [1550.0], [2550.0]], dtype=torch.float64)
    cp, cv, h, u = layer.compute_mixture_properties(T, y)
    target = u if kind == "uv" else h
    solve = layer.solve_temperature_uv if kind == "uv" else layer.solve_temperature_hp
    recovered = solve(target, y, T + 350.0)
    torch.testing.assert_close(recovered, T, atol=2e-4, rtol=0.0)
    cp_r, cv_r, h_r, u_r = layer.compute_mixture_properties(recovered, y)
    residual = ((u_r if kind == "uv" else h_r) - target).abs()
    assert residual.max() < 1.0  # J/kg


def test_nasa7_properties_agree_with_cantera(layer):
    gas = ct.Solution(str(PROJECT / "NUIG2024.yaml"))
    y = _mass_fractions()[:1]
    T = torch.tensor([[1350.0]], dtype=torch.float64)
    cp, cv, h, u = layer.compute_mixture_properties(T, y)
    gas.TPY = float(T.item()), ct.one_atm, {
        species: float(y[0, i]) for i, species in enumerate(layer.species_names)
    }
    assert cp.item() == pytest.approx(gas.cp_mass, rel=2e-8)
    assert cv.item() == pytest.approx(gas.cv_mass, rel=2e-8)
    assert h.item() == pytest.approx(gas.enthalpy_mass, rel=2e-8, abs=0.1)
    assert u.item() == pytest.approx(gas.int_energy_mass, rel=2e-8, abs=0.1)


@pytest.mark.parametrize("kind", ["uv", "hp"])
def test_nasa7_rejects_nonconvergence_and_unreachable_targets(layer, kind):
    y = _mass_fractions()[:1]
    initial = torch.tensor([[1200.0]], dtype=torch.float64)
    cp, cv, h, u = layer.compute_mixture_properties(initial, y)
    target = u if kind == "uv" else h
    solve = layer.solve_temperature_uv if kind == "uv" else layer.solve_temperature_hp
    with pytest.raises(RuntimeError, match="did not converge"):
        solve(target, y, initial + 1000.0, max_iters=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve(torch.tensor([[1e10]], dtype=torch.float64), y, initial)


@pytest.mark.parametrize("kind", ["uv", "hp"])
def test_nasa7_float32_return_has_bounded_energy_residual(layer, kind):
    y = _mass_fractions()[:1]
    expected_T = torch.tensor([[1350.0]], dtype=torch.float64)
    cp, cv, h, u = layer.compute_mixture_properties(expected_T, y)
    target = u if kind == "uv" else h
    solve = layer.solve_temperature_uv if kind == "uv" else layer.solve_temperature_hp
    recovered = solve(target, y.to(torch.float32), (expected_T + 220.0).to(torch.float32))
    assert recovered.dtype == torch.float32
    cp_r, cv_r, h_r, u_r = layer.compute_mixture_properties(recovered.to(torch.float64), y)
    residual = ((u_r if kind == "uv" else h_r) - target).abs()
    assert residual.max() < 5.0  # J/kg, including float32 temperature rounding


def test_nasa7_temperature_gradient_is_finite(layer):
    y_ref = _mass_fractions()
    y_next = y_ref.clone().detach().requires_grad_(True)
    T = torch.tensor([[1000.0], [1500.0], [2400.0]], dtype=torch.float64)
    target = layer.compute_mixture_properties(T, y_ref)[3].detach()
    recovered = layer.solve_temperature_uv(target, y_next, T + 220.0)
    recovered.sum().backward()
    assert y_next.grad is not None
    assert torch.isfinite(y_next.grad).all()
    assert (y_next.grad.abs() > 0.0).any()


@pytest.mark.parametrize("kind", ["uv", "hp"])
def test_autonomous_update_preserves_energy_without_terminal_clipping(layer, kind):
    y_old = _mass_fractions()[:1]
    y_new = y_old.clone()
    # Small combustion-like composition shift with a solvable energy target.
    y_new[0, 0] -= 1e-4
    y_new[0, 4] += 1e-4
    T = torch.tensor([[1150.0]], dtype=torch.float64)
    updated = update_temperature_autonomous(
        T, y_new - y_old, max_delta_temperature=1000.0,
        thermo_closure={
            "nasa7_layer": layer,
            "reactor_type": "IdealGasReactor" if kind == "uv" else "IdealGasConstPressureReactor",
        },
        Y_old_full=y_old,
        Y_next_full=y_new,
    )
    before = layer.compute_mixture_properties(T, y_old)
    after = layer.compute_mixture_properties(updated, y_new)
    index = 3 if kind == "uv" else 2
    assert (after[index] - before[index]).abs().max() < 1.0


def test_autonomous_update_rejects_absent_nasa7_instead_of_linear_fallback():
    T = torch.tensor([[1000.0]])
    Y = torch.tensor([[1.0]])
    with pytest.raises(RuntimeError, match="requires an initialized layer"):
        update_temperature_autonomous(
            T, torch.zeros_like(Y),
            energy_coeff=torch.tensor([0.0]),
            heat_capacity=torch.tensor([[1000.0]]),
            thermo_closure={"nasa7_layer": None},
            Y_old_full=Y, Y_next_full=Y,
        )


def test_autonomous_closure_rejects_unknown_reactor_type(layer):
    y = _mass_fractions()[:1]
    T = torch.tensor([[1100.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="Unsupported autonomous reactor"):
        update_temperature_autonomous(
            T, torch.zeros_like(y),
            thermo_closure={"nasa7_layer": layer, "reactor_type": "SomeOtherReactor"},
            Y_old_full=y, Y_next_full=y,
        )


def test_autonomous_builder_rejects_failed_nasa7_initialization(monkeypatch):
    import thermo_nasa7_layer
    import train_sc_v68 as training

    class BrokenNASA:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("test coefficient error")

    monkeypatch.setattr(thermo_nasa7_layer, "ThermoNASA7Layer", BrokenNASA)
    with pytest.raises(RuntimeError, match="refusing silent linear-temperature fallback"):
        training.build_temperature_closure(
            {"reactor_type": "IdealGasReactor"},
            ["NH3", "H2", "O2", "N2", "H2O"],
            data_dir=PROJECT,
            mechanism="nuig_2024",
        )


def test_autonomous_update_rejects_unphysical_energy_step(layer):
    # A tiny allowed delta-T must reject, rather than silently clip, the
    # different-energy composition after a valid Newton solution.
    y_old = _mass_fractions()[:1]
    y_new = y_old.clone()
    y_new[0, 0] -= 1e-3
    y_new[0, 4] += 1e-3
    T = torch.tensor([[1000.0]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="max_delta_temperature"):
        update_temperature_autonomous(
            T, y_new - y_old, max_delta_temperature=1e-6,
            thermo_closure={"nasa7_layer": layer, "reactor_type": "IdealGasReactor"},
            Y_old_full=y_old, Y_next_full=y_new,
        )
