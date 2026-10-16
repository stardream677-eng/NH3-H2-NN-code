"""Small Cantera integration checks for autonomous H128 benchmark plumbing.

The one-/few-step test validates that neural states remain neural states. It
does not claim to measure the learned model's H128 accuracy or IDT quality.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ct = pytest.importorskip("cantera")
import chemistry_contract as cc
import train_sc_v68 as tr

SPEC = importlib.util.spec_from_file_location(
    "benchmark_autonomous_h128", ROOT / "test" / "benchmark_autonomous_h128.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_case_manifest_requires_named_explicit_128_step_reactor(tmp_path):
    example = ROOT / "test" / "benchmark_autonomous_cases.example.json"
    cases = bench.load_cases(example)
    assert len(cases) == 4
    assert all(case["steps"] == 128 for case in cases)
    assert len({case["name"] for case in cases}) == len(cases)
    data = json.loads(example.read_text(encoding="utf-8"))
    data["cases"][0]["steps"] = 127
    target = tmp_path / "invalid.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 128"):
        bench.load_cases(target)
    data["cases"][0]["steps"] = 128
    data["cases"][0]["name"] = "../../unsafe"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Case name"):
        bench.load_cases(target)


def test_benchmark_rejects_short_horizon_checkpoint_even_if_max_horizon_is_128(
    tmp_path, monkeypatch,
):
    from autonomous_pressure import PRESSURE_FEEDBACK_CONTRACT
    from physical_projection_contract import PROJECTION_VALIDATION_CONTRACT

    checkpoint_path = tmp_path / "dummy_H32.pth"
    mechanism_path = ROOT / "NUIG2024.yaml"
    checkpoint_path.write_bytes(b"fake_checkpoint_for_gate_test")

    fake = {
        "training_config": {
            "temperature_mode": "autonomous",
            "feedback_state_source": "constrained",
            "max_horizon": 128,
        },
        "horizon": 32,
        "sentinel_metrics": {"is_valid_candidate": True},
    }
    monkeypatch.setattr(bench.validator, "load_checkpoint", lambda *args, **kwargs: fake)
    monkeypatch.setattr(
        bench.validator, "checkpoint_rollout_config",
        lambda _: {
            "temperature_mode": "autonomous",
            "pressure_feedback_contract": PRESSURE_FEEDBACK_CONTRACT,
            "projection_validation_contract": PROJECTION_VALIDATION_CONTRACT,
            "autonomous_input_contract": "autonomous_tpy_inert_current_v1",
        },
    )
    with pytest.raises(ValueError, match="not an explicitly selected H=128"):
        bench.load_model_components(
            checkpoint_path, tmp_path, "nuig_2024",
            mechanism_path, torch.device("cpu"),
        )


def test_independent_cantera_energy_eos_audit_cannot_correct_wrong_pressure():
    gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    gas.TPX = 1220., 24. * ct.one_atm, bench.fuel_air_mole_fractions(0.9, 0.25)
    Y = gas.Y.copy()
    elements, A = cc.compute_element_mass_matrix(
        gas, gas.species_names, preferred_elements=None
    )
    audit_gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    result = bench.audit_neural_state(
        audit_gas, gas.T, gas.P / ct.one_atm, Y, "IdealGasReactor",
        gas.density, gas.P / ct.one_atm, gas.int_energy_mass, A @ Y, A,
    )
    assert result["mass"] < 1e-12
    assert result["element"] < 1e-12
    assert result["energy_J_per_kg"] < 1e-6
    assert result["eos_relative"] < 1e-12

    with pytest.raises(RuntimeError, match="Thermodynamic drift"):
        bench.audit_neural_state(
            audit_gas, gas.T, gas.P / ct.one_atm * 1.5, Y,
            "IdealGasReactor", gas.density, gas.P / ct.one_atm,
            gas.int_energy_mass, A @ Y, A,
        )


def test_constant_pressure_independent_audit_requires_fixed_pressure():
    gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    gas.TPX = 1280., 8. * ct.one_atm, bench.fuel_air_mole_fractions(1.1, 0.3)
    Y = gas.Y.copy()
    _, A = cc.compute_element_mass_matrix(
        gas, gas.species_names, preferred_elements=None
    )
    reader = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    result = bench.audit_neural_state(
        reader, gas.T, gas.P / ct.one_atm, Y,
        "IdealGasConstPressureReactor",
        gas.density, gas.P / ct.one_atm, gas.enthalpy_mass, A @ Y, A,
    )
    assert result["energy_J_per_kg"] < 1e-6
    with pytest.raises(RuntimeError, match="Thermodynamic drift"):
        bench.audit_neural_state(
            reader, gas.T, gas.P / ct.one_atm * 1.2, Y,
            "IdealGasConstPressureReactor",
            gas.density, gas.P / ct.one_atm, gas.enthalpy_mass, A @ Y, A,
        )


@pytest.mark.parametrize(
    "reactor_type",
    ["IdealGasReactor", "IdealGasConstPressureReactor"],
)
def test_three_step_zero_kinetics_rollout_has_no_cantera_temperature_feedback(
    reactor_type,
):
    gas = ct.Solution(str(ROOT / "NUIG2024.yaml"))
    species = list(gas.species_names)
    inert = ["N2"]
    macro = [sp for sp in species if sp not in inert]
    fi = tr.build_feature_info(species, inert, macro, [], "direct_rate", "direct_rate")
    fi.update(mechanism="nuig_2024", mechanism_path=str(ROOT / "NUIG2024.yaml"))

    class IdentityScaler:
        def __init__(self, n):
            self.mean_ = np.zeros(n, dtype=np.float32)
            self.scale_ = np.ones(n, dtype=np.float32)

    tools = tr.CoarseOperatorTools(
        fi, IdentityScaler(len(fi["input_cols"])),
        IdentityScaler(len(fi["output_cols"])),
        device="cpu", rate_decode_mode="direct_rate",
    )
    tools._benchmark_state_update_mode = "legacy"
    tools._benchmark_trust_region_fraction = 0.0
    tools._benchmark_max_delta_temperature = 1000.0

    class ZeroRateModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(
                (x.shape[0], len(fi["output_cols"])),
                device=x.device, dtype=x.dtype,
            )

    case = {
        "name": "zero_rate_physical_smoke",
        "T0_K": 1600., "P0_atm": 12., "phi": 1.0,
        "h2_fraction": 0.2, "dt_s": 1e-6, "steps": 3,
        "reactor_type": reactor_type,
    }
    ordered_species = inert + macro
    record, trace = bench.run_case(
        case, ZeroRateModel(), tools, ROOT / "NUIG2024.yaml", ordered_species,
        idt_rise_K=400.0,
    )
    assert record["status"] == "completed", record.get("failure_message")
    assert record["completed_steps"] == 3
    assert trace["nn_T_K"].shape == (4,)
    assert trace["ref_T_K"].shape == (4,)
    assert trace["nn_Y"].shape == (4, gas.n_species)
    # Zero-rate model must remain near its initial state, regardless of
    # Cantera's independent chemical reaction on the reference trajectory.
    assert np.max(np.abs(trace["nn_T_K"] - case["T0_K"])) < 1e-2
    assert np.max(np.abs(trace["nn_P_atm"] - case["P0_atm"])) < 1e-4
    assert record["physics"]["max_energy_residual_J_per_kg"] < 10.
    assert record["physics"]["max_eos_relative_residual"] < 1e-5
