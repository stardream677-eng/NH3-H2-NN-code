"""Strict H=128 autonomous T/P/Y benchmark; Cantera is reference/auditor only.

Unlike the historical benchmark_cfd_closed_loop.py, the neural trajectory is
advanced *only* by the checkpoint model, production constrained species
projection, differentiable NASA7 energy solver and EOS pressure feedback.
Cantera is NEVER called to repair neural T, P or Y during rollout.

Example (run from sc16 after training an autonomous + constrained checkpoint):
  python test/benchmark_autonomous_h128.py \
    --checkpoint output/weights_v68/best_v68_net_nuig_2024_H128.pth \
    --scalers-root output --mechanism nuig_2024 --mechanism-path NUIG2024.yaml \
    --cases test/benchmark_autonomous_cases.example.json \
    --output output/benchmark_h128

The supplied cases are not automatically independent of the training data.
This script explicitly reports them as *independence unverified*: use frozen,
held-out condition groups for publication-quality test metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import os
import sys
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cantera as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autonomous_benchmark_protocol as protocol
import chemistry_contract as cc
import experiment_manifest as em
import train_sc_v68 as tr
import validate_sc_v68 as validator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuel_air_mole_fractions(phi: float, hydrogen_mole_fraction: float) -> dict[str, float]:
    if not (np.isfinite(phi) and 0.1 <= phi <= 5):
        raise ValueError("phi must be finite and in [0.1, 5]")
    if not (np.isfinite(hydrogen_mole_fraction) and 0 <= hydrogen_mole_fraction <= 1):
        raise ValueError("h2_fraction must be in [0, 1]")
    nh3 = 1.0 - hydrogen_mole_fraction
    h2 = hydrogen_mole_fraction
    o2 = (0.75 * nh3 + 0.5 * h2) / phi
    mixture = {"O2": o2, "N2": 3.7619 * o2}
    if nh3:
        mixture["NH3"] = nh3
    if h2:
        mixture["H2"] = h2
    return mixture


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("protocol") != protocol.BENCHMARK_PROTOCOL:
        raise ValueError("Cases JSON must declare protocol=autonomous_h128_trajectories_v1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases JSON must contain a nonempty cases list")
    names = set()
    clean = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a dictionary")
        name = case.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("Case name must contain only letters, digits, '_' or '-'")
        if name in names:
            raise ValueError(f"Duplicate case name: {name}")
        names.add(name)
        if case.get("reactor_type") not in (
            "IdealGasReactor", "IdealGasConstPressureReactor"
        ):
            raise ValueError(f"{name}: unsupported reactor_type")
        steps = case.get("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError(f"{name}: steps must be a positive integer")
        if steps != 128:
            raise ValueError(f"{name}: H=128 protocol requires exactly 128 model steps")
        values = ("T0_K", "P0_atm", "phi", "h2_fraction", "dt_s")
        for key in values:
            if key not in case or not isinstance(case[key], (int, float)) or not np.isfinite(case[key]):
                raise ValueError(f"{name}: finite numeric {key} is required")
        if not (200 <= case["T0_K"] <= 5000):
            raise ValueError(f"{name}: initial T outside NASA7 valid range")
        if not (1e-6 <= case["P0_atm"] <= 1e4):
            raise ValueError(f"{name}: initial P outside autonomous EOS guard")
        if not (0 < case["dt_s"] <= 1e-3):
            raise ValueError(f"{name}: dt_s outside (0, 1e-3] seconds")
        fuel_air_mole_fractions(float(case["phi"]), float(case["h2_fraction"]))
        clean.append(dict(case))
    return clean


def load_model_components(
    checkpoint_path: Path, scalers_root: Path, mechanism_name: str,
    mechanism_path: Path, device: torch.device,
):
    if not checkpoint_path.is_file() or not mechanism_path.is_file():
        raise FileNotFoundError("Checkpoint and mechanism YAML must exist")
    tr.DEVICE = device
    checkpoint = validator.load_checkpoint(checkpoint_path, mechanism_name)
    config = validator.checkpoint_rollout_config(checkpoint)
    if config["temperature_mode"] != "autonomous":
        raise ValueError("H=128 closed-loop benchmark requires an autonomous checkpoint")
    recorded = checkpoint.get("training_config") or {}
    if recorded.get("feedback_state_source") != "constrained":
        raise ValueError("Autonomous benchmark requires constrained-species checkpoint")
    if config["pressure_feedback_contract"] != tr.ap.PRESSURE_FEEDBACK_CONTRACT:
        raise ValueError("Checkpoint lacks the autonomous EOS pressure contract")
    if config["projection_validation_contract"] != tr.ppc.PROJECTION_VALIDATION_CONTRACT:
        raise ValueError("Checkpoint lacks the strict elemental projection contract")
    if config["autonomous_input_contract"] != tr.ap.AUTONOMOUS_INPUT_CONTRACT:
        raise ValueError("Checkpoint was not trained with leak-free autonomous inert feedback")
    if int(recorded.get("max_horizon", 0) or 0) < 128:
        raise ValueError(
            "This H=128 benchmark requires a checkpoint from a run configured "
            "for max_horizon >= 128; select an H128-trained candidate"
        )
    if checkpoint.get("horizon") != 128:
        raise ValueError(
            "Checkpoint is not an explicitly selected H=128 Sentinel checkpoint; "
            "use best_v68_net_<mechanism>_H128.pth, not latest or a short-H model"
        )
    if not bool((checkpoint.get("sentinel_metrics") or {}).get("is_valid_candidate", False)):
        raise ValueError("H128 checkpoint has no valid frozen Sentinel selection record")
    fi = checkpoint["feature_info"]
    actual_hash = sha256(mechanism_path)
    expected_hash = (checkpoint.get("manifest") or {}).get("mechanism_sha256")
    if actual_hash != expected_hash:
        raise ValueError("Mechanism YAML SHA-256 differs from checkpoint manifest")

    sx, sy, _ = validator.load_scalers(
        output_root=scalers_root, mechanism=mechanism_name,
        feature_info=fi, data_dir=mechanism_path.parent,
        rebuild_scalers=False, checkpoint_root=checkpoint_path.parent.parent,
    )
    em.verify_runtime_artifacts(checkpoint, fi, sx, sy, (mechanism_path.parent,))
    model = validator._build_checkpoint_model(checkpoint, fi, device=device)
    model.eval()
    tools = tr.CoarseOperatorTools(
        fi, sx, sy, rate_decode_mode=recorded.get(
            "rate_decode_mode", fi.get("rate_decode_mode", "direct_rate")
        ),
        rate_z_limit=float(recorded.get("rate_z_limit", tr.DEFAULT_RATE_Z_LIMIT)),
        device=device,
    )
    tools.projection_mode = recorded.get("projection_mode", "multiplicative_kl")
    gas = ct.Solution(str(mechanism_path))
    names = list(fi["inert_species"]) + list(fi["macro_species"]) + list(fi["micro_species"])
    if len(names) != len(set(names)) or set(names) != set(gas.species_names):
        missing = sorted(set(gas.species_names) - set(names))
        extra = sorted(set(names) - set(gas.species_names))
        raise ValueError(
            "Autonomous benchmark requires full mechanism species coverage; "
            f"missing={missing}, extra={extra}"
        )
    if list(fi["all_species"]) != list(gas.species_names):
        raise ValueError("Checkpoint mechanism species order differs from Cantera")
    return checkpoint, config, model, tools, gas, names


def physical_input(
    tools, gas: ct.Solution, T: float, P_atm: float, Y: np.ndarray, dt: float,
) -> torch.Tensor:
    mapping = {"T": T, "P": P_atm, "log_dt": np.log10(dt)}
    for name in tools.feature_info["inert_species"] + tools.feature_info["macro_species"]:
        mapping[f"Y_{name}"] = float(Y[gas.species_index(name)])
    for name in tools.feature_info["micro_species"]:
        mapping[f"logY_{name}"] = float(np.log10(max(Y[gas.species_index(name)], tr.LOG_EPS)))
    row = torch.tensor(
        [[mapping[name] for name in tools.feature_info["input_cols"]]],
        device=tools.device, dtype=tools.mean_X.dtype,
    )
    return (row - tools.mean_X) / tools.scale_X


def decode_feedback_state(tools, scaled, names, gas):
    state = tools.inverse_state(scaled)[0].detach().cpu().numpy()
    out = np.zeros(gas.n_species, dtype=np.float64)
    for sp in tools.feature_info["inert_species"] + tools.feature_info["macro_species"]:
        out[gas.species_index(sp)] = state[tools.feature_info["input_cols"].index(f"Y_{sp}")]
    for sp in tools.feature_info["micro_species"]:
        z = state[tools.feature_info["input_cols"].index(f"logY_{sp}")]
        out[gas.species_index(sp)] = 10.0 ** np.clip(z, tr.MICRO_LOG_MIN, tr.MICRO_LOG_MAX)
    return (
        float(state[tools.temperature_idx]),
        float(state[tools.pressure_idx]),
        out,
    )


def generate_cantera_reference(initial_gas: ct.Solution, reactor_type: str,
                               grid_s: np.ndarray, mechanism_path: Path) -> dict[str, np.ndarray]:
    """Integrate a *separate* Cantera reactor at the exact NN sample times."""
    gas = ct.Solution(str(mechanism_path))
    gas.TPY = initial_gas.T, initial_gas.P, initial_gas.Y
    if reactor_type == "IdealGasReactor":
        reactor = ct.IdealGasReactor(gas)
    else:
        reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    T = [float(reactor.thermo.T)]
    P = [float(reactor.thermo.P / ct.one_atm)]
    Y = [reactor.thermo.Y.copy()]
    for instant in grid_s[1:]:
        net.advance(float(instant))
        T.append(float(reactor.thermo.T))
        P.append(float(reactor.thermo.P / ct.one_atm))
        Y.append(reactor.thermo.Y.copy())
    return {"T": np.asarray(T), "P": np.asarray(P), "Y": np.asarray(Y)}


def audit_neural_state(
    gas: ct.Solution, T: float, P_atm: float, Y: np.ndarray,
    reactor_type: str, initial_density: float, initial_pressure_atm: float,
    initial_energy: float, initial_elements: np.ndarray,
    A_full: np.ndarray, *, mass_tolerance=5e-6,
    element_tolerance=1e-6, energy_tolerance_Jkg=10.0,
    eos_relative_tolerance=1e-5,
) -> dict[str, float]:
    """Independent Cantera *audit*, without overwriting neural feedback values."""
    if not (np.isfinite(T) and np.isfinite(P_atm) and np.isfinite(Y).all()):
        raise RuntimeError("Non-finite autonomous T/P/Y")
    mass = float(abs(Y.sum() - 1.0))
    negative = float(max(0.0, -Y.min()))
    elements = float(np.abs(A_full @ Y - initial_elements).max())
    if mass > mass_tolerance or negative > 0 or elements > element_tolerance:
        raise RuntimeError(
            f"Physical species drift: mass={mass:.4e}, negative={negative:.4e}, "
            f"elements={elements:.4e}"
        )
    if reactor_type == "IdealGasReactor":
        gas.TDY = T, initial_density, Y
        independent_P_atm = gas.P / ct.one_atm
        energy = gas.int_energy_mass
    else:
        gas.TPY = T, initial_pressure_atm * ct.one_atm, Y
        independent_P_atm = initial_pressure_atm
        energy = gas.enthalpy_mass
    energy_residual = float(abs(energy - initial_energy))
    eos_residual = float(abs(P_atm - independent_P_atm) / independent_P_atm)
    if energy_residual > energy_tolerance_Jkg or eos_residual > eos_relative_tolerance:
        raise RuntimeError(
            f"Thermodynamic drift: energy={energy_residual:.4e} J/kg, "
            f"EOS relative={eos_residual:.4e}"
        )
    return {
        "mass": mass, "element": elements,
        "energy_J_per_kg": energy_residual, "eos_relative": eos_residual,
    }


def run_case(
    case: dict, model, tools, mechanism_path: Path, names: list[str] | None = None,
    idt_rise_K: float = protocol.DEFAULT_IDT_RISE_K,
) -> tuple[dict, dict[str, np.ndarray]]:
    name = case["name"]
    expected_species = (
        list(tools.feature_info["inert_species"])
        + list(tools.feature_info["macro_species"])
        + list(tools.feature_info["micro_species"])
    )
    if names is None or (set(names) == set(expected_species) and list(names) != expected_species):
        names = expected_species
    gas_initial = ct.Solution(str(mechanism_path))
    gas_initial.TPX = (
        float(case["T0_K"]), float(case["P0_atm"]) * ct.one_atm,
        fuel_air_mole_fractions(float(case["phi"]), float(case["h2_fraction"])),
    )
    Y0 = gas_initial.Y.copy()
    initial_density = float(gas_initial.density)
    reactor_type = case["reactor_type"]
    energy0 = (
        float(gas_initial.int_energy_mass) if reactor_type == "IdealGasReactor"
        else float(gas_initial.enthalpy_mass)
    )
    elements, A_full = cc.compute_element_mass_matrix(
        gas_initial, list(gas_initial.species_names), preferred_elements=None
    )
    initial_elements = A_full @ Y0
    dt = float(case["dt_s"])
    times = np.arange(case["steps"] + 1, dtype=np.float64) * dt
    result = {
        "name": name, "status": "failed", "failure_step": None,
        "reactor_type": reactor_type, "dt_s": dt, "steps": case["steps"],
        "t_end_s": float(times[-1]), "initial_T_K": float(case["T0_K"]),
        "initial_P_atm": float(case["P0_atm"]),
        "phi": float(case["phi"]),
        "h2_fraction": float(case["h2_fraction"]),
        "initial_density_kg_per_m3": initial_density,
        "initial_energy_J_per_kg": energy0,
        "energy_constraint": "UV" if reactor_type == "IdealGasReactor" else "HP",
        "failure_phase": None, "failure_message": None,
    }
    traces = {"time_s": times}

    try:
        reference = generate_cantera_reference(gas_initial, reactor_type, times, mechanism_path)
    except Exception as exc:
        result.update(failure_step=0, failure_phase="reference_cantera",
                      failure_message=f"{type(exc).__name__}: {exc}")
        return result, traces

    traces.update(ref_T_K=reference["T"], ref_P_atm=reference["P"], ref_Y=reference["Y"])
    auditor = ct.Solution(str(mechanism_path))
    try:
        closure = tr.build_temperature_closure(
            {"mechanism_path": str(mechanism_path), "reactor_type": reactor_type},
            species_order=names,
        )
    except Exception as exc:
        result.update(
            failure_step=0, failure_phase="autonomous_thermo_initialization",
            failure_message=f"{type(exc).__name__}: {exc}",
        )
        return result, traces
    current = physical_input(
        tools, gas_initial, float(case["T0_K"]), float(case["P0_atm"]), Y0, dt
    )
    predicted_T = [float(case["T0_K"])]
    predicted_P = [float(case["P0_atm"])]
    predicted_Y = [Y0.copy()]
    physical_checks = []
    failure_step = None
    failure_phase = None
    failure_message = None
    with torch.no_grad():
        for index in range(case["steps"]):
            try:
                prediction = model(current)
                pred = tools.predict_next_state(
                    current, prediction, safe_for_rollout=True,
                    state_update_mode=tools._benchmark_state_update_mode,
                    trust_region_fraction=tools._benchmark_trust_region_fraction,
                )
                next_state = tools.state_to_input_scaled(
                    current.clone(),  # no future teacher row; dt stays fixed
                    pred["Y_macro_next_constrained"],
                    pred["logY_micro_next_constrained"],
                    temperature_mode="autonomous",
                    thermo_closure=closure, current_X_scaled=current,
                    max_delta_temperature=tools._benchmark_max_delta_temperature,
                )
                new_T, new_P, new_Y = decode_feedback_state(
                    tools, next_state, names, gas_initial
                )
                checks = audit_neural_state(
                    auditor, new_T, new_P, new_Y, reactor_type,
                    initial_density, float(case["P0_atm"]), energy0,
                    initial_elements, A_full,
                )
                physical_checks.append(checks)
                predicted_T.append(new_T)
                predicted_P.append(new_P)
                predicted_Y.append(new_Y.copy())
                current = next_state
            except Exception as exc:
                failure_step = index + 1
                failure_phase = "neural_physical_rollout"
                failure_message = f"{type(exc).__name__}: {exc}"
                break

    traces.update(
        nn_T_K=np.asarray(predicted_T), nn_P_atm=np.asarray(predicted_P),
        nn_Y=np.asarray(predicted_Y),
    )
    if failure_step is not None:
        result.update(
            status="failed", failure_step=failure_step,
            failure_phase=failure_phase, failure_message=failure_message,
            completed_steps=len(predicted_T) - 1,
        )
        return result, traces

    physics = {
        "max_mass_residual": max(v["mass"] for v in physical_checks),
        "max_element_residual": max(v["element"] for v in physical_checks),
        "max_energy_residual_J_per_kg": max(v["energy_J_per_kg"] for v in physical_checks),
        "max_eos_relative_residual": max(v["eos_relative"] for v in physical_checks),
    }
    metrics = protocol.compare_complete_trajectories(
        times, reference["T"], np.asarray(predicted_T),
        reference["P"], np.asarray(predicted_P),
        reference["Y"], np.asarray(predicted_Y),
        list(gas_initial.species_names),
        idt_rise_K=idt_rise_K,
    )
    result.update(status="completed", completed_steps=case["steps"], physics=physics, metrics=metrics)
    return result, traces


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scalers-root", type=Path, required=True)
    parser.add_argument("--mechanism", required=True, choices=("nuig_2024", "tsinghua_2024c", "ucf_2024"))
    parser.add_argument("--mechanism-path", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="Explicit holdout candidates. Disjointness is not inferred.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--idt-rise-K", type=float, default=protocol.DEFAULT_IDT_RISE_K)
    args = parser.parse_args(argv)
    if not np.isfinite(args.idt_rise_K) or args.idt_rise_K <= 0:
        parser.error("--idt-rise-K must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA selected but torch.cuda.is_available() is False")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    mechanism_path = args.mechanism_path.expanduser().resolve()
    cases_path = args.cases.expanduser().resolve()
    cases = load_cases(cases_path)
    checkpoint, cfg, model, tools, _, names = load_model_components(
        checkpoint_path, args.scalers_root.expanduser().resolve(),
        args.mechanism, mechanism_path, torch.device(args.device),
    )
    training_cfg = checkpoint["training_config"]
    tools._benchmark_state_update_mode = cfg["state_update_mode"]
    tools._benchmark_trust_region_fraction = cfg["trust_region_fraction"]
    tools._benchmark_max_delta_temperature = cfg["max_delta_temperature"]

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for c in cases:
        print(f"[H128] {c['name']} reactor={c['reactor_type']} dt={c['dt_s']:.3g}s", flush=True)
        try:
            record, traces = run_case(c, model, tools, mechanism_path, names, args.idt_rise_K)
        except Exception as exc:
            record = {
                "name": c["name"], "status": "failed",
                "failure_step": None, "failure_phase": "benchmark_runner",
                "failure_message": f"{type(exc).__name__}: {exc}",
                "failure_traceback": traceback.format_exc(limit=8),
            }
            traces = {}
        records.append(record)
        np.savez_compressed(output / f"{c['name']}.npz", **traces)
        print(
            f"[H128] {record['status']}: {record.get('failure_message') or 'full 128 steps'}",
            flush=True,
        )
    report = {
        "protocol": protocol.BENCHMARK_PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mechanism": args.mechanism,
        "mechanism_path": str(mechanism_path),
        "mechanism_sha256": sha256(mechanism_path),
        "cases_path": str(cases_path), "cases_sha256": sha256(cases_path),
        "device": str(args.device), "torch_version": str(torch.__version__),
        "cantera_version": str(ct.__version__),
        "species_order": list(names),
        "source_hashes": {
            file: sha256(ROOT / file) for file in (
                "train_sc_v68.py", "validate_sc_v68.py",
                "thermo_nasa7_layer.py", "autonomous_pressure.py",
                "physical_projection_contract.py", "autonomous_benchmark_protocol.py",
            )
        },
        "benchmark_runner_sha256": sha256(Path(__file__)),
        "source_training_config": {
            k: training_cfg.get(k) for k in (
                "temperature_mode", "feedback_state_source", "state_update_mode",
                "projection_mode", "projection_validation_contract",
                "pressure_feedback_contract", "autonomous_input_contract",
                "max_horizon",
            )
        },
        "independence_from_training": "unverified: explicit holdout condition audit required",
        "summary": protocol.summarize_cases(records),
        "cases": records,
    }
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    # Explicit non-zero code for *any* failed case, never silent success.
    return 0 if report["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
