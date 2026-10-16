"""Autonomous T/P closed-reactor feedback regression tests.

Pressure in the V21/V72 Parquet feature contract is atm. Cantera uses Pa.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomous_pressure import PRESSURE_FEEDBACK_CONTRACT, autonomous_pressure_next


def _state(dtype=torch.float64):
    W = torch.tensor([28.0134, 2.01588, 31.9988, 18.01528], dtype=dtype)
    initial_Y = torch.tensor([
        [0.74, 0.04, 0.21, 0.01],
        [0.73, 0.02, 0.23, 0.02],
    ], dtype=dtype)
    final_Y = torch.tensor([
        [0.74, 0.03, 0.20, 0.03],
        [0.73, 0.015, 0.22, 0.035],
    ], dtype=dtype)
    T0 = torch.tensor([[950.0], [1450.0]], dtype=dtype)
    T1 = torch.tensor([[1200.0], [1700.0]], dtype=dtype)
    P0 = torch.tensor([[15.0], [30.0]], dtype=dtype)
    return W, initial_Y, final_Y, T0, T1, P0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_volume_eos_pressure_preserves_density(dtype):
    W, Y0, Y1, T0, T1, P0 = _state(dtype)
    P1 = autonomous_pressure_next(
        P0, T0, T1, Y0, Y1, W, "IdealGasReactor"
    )
    density_factor0 = P0.to(torch.float64) / (
        T0.to(torch.float64) * (Y0.to(torch.float64) / W).sum(-1, keepdim=True)
    )
    density_factor1 = P1.to(torch.float64) / (
        T1.to(torch.float64) * (Y1.to(torch.float64) / W).sum(-1, keepdim=True)
    )
    torch.testing.assert_close(density_factor1, density_factor0, rtol=2e-6, atol=0)
    assert P1.dtype == dtype


def test_constant_pressure_feedback_ignores_future_reference_pressure():
    W, Y0, Y1, T0, T1, P0 = _state()
    P1 = autonomous_pressure_next(P0, T0, T1, Y0, Y1, W, "IdealGasConstPressureReactor")
    torch.testing.assert_close(P1, P0, rtol=0, atol=0)


def test_pressure_eos_multistep_retains_original_density():
    W, Y0, Y1, T0, T1, P0 = _state()
    P1 = autonomous_pressure_next(P0, T0, T1, Y0, Y1, W, "IdealGasReactor")
    Y2 = torch.tensor([[0.74, .025, .195, .04],
                       [0.73, .01, .215, .045]], dtype=torch.float64)
    T2 = torch.tensor([[1250.0], [1600.0]], dtype=torch.float64)
    P2 = autonomous_pressure_next(P1, T1, T2, Y1, Y2, W, "IdealGasReactor")
    P2_direct = autonomous_pressure_next(P0, T0, T2, Y0, Y2, W, "IdealGasReactor")
    torch.testing.assert_close(P2, P2_direct, rtol=5e-14, atol=0)


def test_eos_keeps_autograd_for_temperature_and_species():
    W, Y0, Y1, T0, T1, P0 = _state()
    T1 = T1.detach().requires_grad_(True)
    Y1 = Y1.detach().requires_grad_(True)
    P1 = autonomous_pressure_next(P0, T0, T1, Y0, Y1, W, "IdealGasReactor")
    P1.sum().backward()
    assert torch.isfinite(T1.grad).all()
    assert torch.isfinite(Y1.grad).all()
    assert (T1.grad.abs() > 0).any()
    assert (Y1.grad.abs() > 0).any()


@pytest.mark.parametrize("bad", ["negative", "missing_species", "nonfinite", "not_normalized", "overpressure"])
def test_eos_fails_closed_on_nonphysical_inputs(bad):
    W, Y0, Y1, T0, T1, P0 = _state()
    if bad == "negative":
        Y1[0, 0] = -0.1
    elif bad == "missing_species":
        W = W[:-1]
    elif bad == "nonfinite":
        T1[0, 0] = float("nan")
    elif bad == "not_normalized":
        Y1[0, 2] *= 0.5
    elif bad == "overpressure":
        P0[0, 0] = 1e5
    with pytest.raises((ValueError, RuntimeError)):
        autonomous_pressure_next(P0, T0, T1, Y0, Y1, W, "IdealGasReactor")


def test_eos_rejects_unknown_reactor_type():
    W, Y0, Y1, T0, T1, P0 = _state()
    with pytest.raises(ValueError, match="Unsupported ideal-gas reactor"):
        autonomous_pressure_next(P0, T0, T1, Y0, Y1, W, "NotAReactor")


def test_eos_contract_signature_is_explicit():
    assert PRESSURE_FEEDBACK_CONTRACT == "eos_uv_constant_hp_v1"


@pytest.fixture(scope="module")
def cantera_tools():
    import cantera as ct
    import train_sc_v68 as train
    from thermo_nasa7_layer import ThermoNASA7Layer

    species = ["N2", "H2", "O2", "H2O", "OH"]
    inert, macro, micro = ["N2"], ["H2", "O2", "H2O"], ["OH"]
    fi = train.build_feature_info(species, inert, macro, micro)
    fi.update(mechanism="nuig_2024", mechanism_path=str(ROOT / "NUIG2024.yaml"))

    class IdentityScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)

    tools = train.CoarseOperatorTools(
        fi, IdentityScaler(len(fi["input_cols"])),
        IdentityScaler(len(fi["output_cols"])), device="cpu"
    )
    closure = {
        "nasa7_layer": ThermoNASA7Layer(
            ROOT / "NUIG2024.yaml", species_order=inert + macro + micro
        ),
        "reactor_type": "IdealGasReactor",
        "species_order": inert + macro + micro,
        "energy_coeff": np.zeros(len(species), dtype=np.float32),
        "heat_capacity_coeff": np.ones(len(species), dtype=np.float32) * 1000,
    }
    return tools, closure


@pytest.mark.parametrize("reactor", ["IdealGasReactor", "IdealGasConstPressureReactor"])
def test_autonomous_feedback_replaces_both_recorded_temperature_and_pressure(
    cantera_tools, reactor
):
    import cantera as ct
    tools, closure = cantera_tools
    closure = {**closure, "reactor_type": reactor}
    # Physical: T, P[atm], Y_N2, Y_H2, Y_O2, Y_H2O, logY_OH, log_dt
    current = torch.tensor(
        [[1150.0, 12.0, 0.75, 0.04, 0.20, 0.01, -20.0, -6.0]],
        dtype=torch.float32,
    )
    future_teacher = torch.tensor(
        [[1850.0, 450.0, 0.75, 0.0, 0.1, 0.15, -6.0, -5.5]],
        dtype=torch.float32,
    )
    macro_prediction = torch.tensor([[0.039, 0.199, 0.012]], dtype=torch.float32)
    micro_prediction = torch.tensor([[-20.0]], dtype=torch.float32)
    out = tools.state_to_input_scaled(
        future_teacher, macro_prediction, micro_prediction,
        temperature_mode="autonomous",
        thermo_closure=closure,
        current_X_scaled=current,
    )
    assert out.shape == current.shape
    assert out[0, 0].item() != pytest.approx(1850.0)
    assert out[0, 1].item() != pytest.approx(450.0)
    assert out[0, 7].item() == pytest.approx(-5.5)  # only future dt stays recorded
    if reactor == "IdealGasConstPressureReactor":
        assert out[0, 1].item() == pytest.approx(12.0, rel=1e-7)
    else:
        gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
        names = ["N2", "H2", "O2", "H2O", "OH"]
        y0 = dict(zip(names, [0.75, 0.04, 0.20, 0.01, 1e-20]))
        y1 = dict(zip(names, [0.75, 0.039, 0.199, 0.012, 1e-20]))
        gas.TPY = float(current[0, 0]), float(current[0, 1]) * ct.one_atm, y0
        rho0 = gas.density
        gas.TDY = float(out[0, 0]), rho0, y1
        assert out[0, 1].item() * ct.one_atm == pytest.approx(
            gas.P, rel=5e-6
        )


def test_autonomous_feedback_is_independent_of_recorded_future_T_and_P(cantera_tools):
    tools, closure = cantera_tools
    current = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    teacher_a = torch.tensor([[1300., 20., .75, .0, .1, .15, -6., -5.5]])
    teacher_b = teacher_a.clone()
    teacher_b[0, 0] = 2700.
    teacher_b[0, 1] = 450.
    macro = torch.tensor([[.039, .199, .012]])
    micro = torch.tensor([[-20.]])
    a = tools.state_to_input_scaled(
        teacher_a, macro, micro,
        temperature_mode="autonomous", thermo_closure=closure,
        current_X_scaled=current,
    )
    b = tools.state_to_input_scaled(
        teacher_b, macro, micro,
        temperature_mode="autonomous", thermo_closure=closure,
        current_X_scaled=current,
    )
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_scheduled_sampling_blend_recloses_selected_rows_only(cantera_tools):
    import cantera as ct
    import train_sc_v68 as train

    tools, closure = cantera_tools
    current_row = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    current = torch.cat([current_row, current_row], dim=0)
    # Same mass-normalized mixed species, but deliberately wrong blended T/P.
    mixed = torch.tensor([
        [1700., 400., .75, .0395, .1995, .011, -20., -5.5],
        [1300., 230., .75, .0395, .1995, .011, -20., -5.5],
    ])
    selected = torch.tensor([[True], [False]])
    closed = train.reclose_blended_autonomous_feedback(
        tools, current, mixed, selected, closure,
    )
    # Teacher row is untouched, even when its reference pressure is unrelated
    # to the current predicted thermodynamic state.
    torch.testing.assert_close(closed[1], mixed[1])
    assert closed[0, 0] != mixed[0, 0]
    assert closed[0, 1] != mixed[0, 1]
    torch.testing.assert_close(closed[0, [2, 3, 4, 5, 6, 7]], mixed[0, [2, 3, 4, 5, 6, 7]])

    gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    names = ["N2", "H2", "O2", "H2O", "OH"]
    y0 = dict(zip(names, [0.75, .04, .20, .01, 1e-20]))
    y1 = dict(zip(names, [0.75, .0395, .1995, .011, 1e-20]))
    gas.TPY = 1150., 12. * ct.one_atm, y0
    rho0 = gas.density
    u0 = gas.int_energy_mass
    gas.TDY = float(closed[0, 0]), rho0, y1
    assert float(closed[0, 1]) * ct.one_atm == pytest.approx(gas.P, rel=5e-6)
    assert gas.int_energy_mass == pytest.approx(u0, rel=1e-6, abs=0.5)


def test_no_prediction_mask_skips_reclosure(cantera_tools):
    import train_sc_v68 as train
    tools, closure = cantera_tools
    x = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    recorded = x.clone()
    recorded[0, 0] = 2000.
    recorded[0, 1] = 500.
    closed = train.reclose_blended_autonomous_feedback(
        tools, x, recorded, torch.tensor([[False]]), closure,
    )
    torch.testing.assert_close(closed, recorded)


def test_legacy_autonomous_checkpoint_rejected_by_validator():
    import validate_sc_v68 as validator
    with pytest.raises(ValueError, match="Legacy autonomous checkpoint"):
        validator.checkpoint_rollout_config({
            "training_config": {"temperature_mode": "autonomous"}
        })
    valid = validator.checkpoint_rollout_config({
        "training_config": {
            "temperature_mode": "autonomous",
            "pressure_feedback_contract": PRESSURE_FEEDBACK_CONTRACT,
        }
    })
    assert valid["temperature_mode"] == "autonomous"
    recorded = validator.checkpoint_rollout_config({
        "training_config": {"temperature_mode": "recorded"}
    })
    assert recorded["temperature_mode"] == "recorded"


def test_autonomous_feedback_never_reads_future_reference_inert_species(cantera_tools):
    tools, closure = cantera_tools
    current = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    future_a = torch.tensor([[1600., 15., .75, .02, .21, .02, -10., -5.5]])
    future_b = future_a.clone()
    # A future recorded inert fraction is deliberately corrupted; it must not
    # influence even the next *scaled* input in fully autonomous mode.
    future_b[0, 2] = 0.03
    macro = torch.tensor([[.039, .199, .012]])
    micro_log = torch.tensor([[-20.]])
    a = tools.state_to_input_scaled(
        future_a, macro, micro_log, temperature_mode="autonomous",
        thermo_closure=closure, current_X_scaled=current,
    )
    b = tools.state_to_input_scaled(
        future_b, macro, micro_log, temperature_mode="autonomous",
        thermo_closure=closure, current_X_scaled=current,
    )
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert a[0, tools.inert_idx[0]] == current[0, tools.inert_idx[0]]


def test_recorded_mode_retains_teacher_T_and_P(cantera_tools):
    tools, closure = cantera_tools
    current = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    teacher = torch.tensor([[1850., 450., .75, .0, .1, .15, -6., -5.5]])
    out = tools.state_to_input_scaled(
        teacher, torch.tensor([[.039, .199, .012]]),
        torch.tensor([[-20.]]), temperature_mode="recorded",
        current_X_scaled=current, thermo_closure=closure,
    )
    assert out[0, 0] == teacher[0, 0]
    assert out[0, 1] == teacher[0, 1]


def test_autonomous_rejects_post_closure_species_clipping(cantera_tools):
    tools, closure = cantera_tools
    current = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    macro = torch.tensor([[.039, .199, .012]])
    # Tiny artificial scaled-feedback limit would alter the mass fractions
    # *after* the energy/EOS solve: refuse the inconsistency instead.
    with pytest.raises(RuntimeError, match="clipping would change the species"):
        tools.state_to_input_scaled(
            current, macro, torch.tensor([[-20.]]),
            temperature_mode="autonomous", thermo_closure=closure,
            current_X_scaled=current, feedback_input_limit=0.01,
        )


def test_new_pressure_module_is_in_experiment_manifest_source_hashes():
    import experiment_manifest as em
    hashes = em.compute_source_hashes(ROOT)
    assert "autonomous_pressure.py" in hashes
    assert len(hashes["autonomous_pressure.py"]) == 64


def test_autonomous_closure_rejects_missing_inert_species(cantera_tools):
    tools, closure = cantera_tools
    corrupted = {**closure, "species_order": ["H2", "O2", "H2O", "OH"]}
    current = torch.tensor([[1150., 12., .75, .04, .20, .01, -20., -6.]])
    with pytest.raises(ValueError, match="species order"):
        tools.state_to_input_scaled(
            current, torch.tensor([[.039, .199, .012]]),
            torch.tensor([[-20.]]), temperature_mode="autonomous",
            thermo_closure=corrupted, current_X_scaled=current,
        )
