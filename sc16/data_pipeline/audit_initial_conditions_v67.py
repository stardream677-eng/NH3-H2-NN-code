"""Task 3 audit for initial composition, phi metadata, and elemental closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def _resolve_dataset_folder(root: Path, mechanism: str, recorded: str | Path) -> Path:
    local = root / f"trajectories_{mechanism}"
    if local.is_dir():
        return local
    path = Path(str(recorded))
    return path if path.is_absolute() else root / path


def _manifest_parts(root: Path):
    manifest = root / "dataset_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    for entry in pd.read_csv(manifest).to_dict("records"):
        folder = _resolve_dataset_folder(root, str(entry["mechanism_name"]), entry["dataset_path"])
        for part in sorted(folder.glob("part-*.parquet")):
            yield str(entry["mechanism_name"]), part, folder


def _metadata_mechanism_path(folder: Path, database_root: Path | None = None, mechanism: str | None = None) -> Path | None:
    metadata = folder / "metadata.json"
    if not metadata.exists():
        return None
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Some database writers store one metadata record in a one-item list.
    if isinstance(payload, list):
        payload = payload[0] if payload and isinstance(payload[0], dict) else {}
    if not isinstance(payload, dict):
        return None

    value = payload.get("mechanism_path")
    if not value:
        return None
    recorded = Path(str(value))

    # Metadata stores the path used when the database was generated.  After a
    # workstation move, resolve the same filename in the current workspace
    # before falling back to that stale absolute path.
    if database_root is not None:
        filename = recorded.name
        candidates = [
            database_root / filename,
            database_root.parent / filename,
            database_root / "mechanisms" / filename,
            database_root.parent / "mechanisms" / filename,
        ]
        known_names = {
            "nuig_2024": "NUIG2024.yaml",
            "tsinghua_2024c": "TH2024.yaml",
            "ucf_2024": "UCF2024.yaml",
        }
        mapped = known_names.get(str(mechanism).lower()) if mechanism else None
        if mapped:
            candidates = [
                database_root / mapped,
                database_root.parent / mapped,
                database_root / "mechanisms" / mapped,
                database_root.parent / "mechanisms" / mapped,
            ] + candidates
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    if recorded.is_file():
        return recorded.resolve()
    return recorded


def _element_mass_matrix(gas, species: list[str]):
    matrix = np.zeros((gas.n_elements, len(species)), dtype=np.float64)
    for j, name in enumerate(species):
        # Dataset mass-fraction columns are conventionally prefixed with
        # ``Y_`` (for example ``Y_H2``), while Cantera uses the bare species
        # name (``H2``).  Resolve the dataset label once at this boundary so
        # the elemental matrix cannot silently become all zeros.
        species_name = name[2:] if name.startswith("Y_") else name
        if species_name not in gas.species_names:
            continue
        k = gas.species_index(species_name)
        molecular_weight = float(gas.molecular_weights[k])
        for i in range(gas.n_elements):
            matrix[i, j] = gas.n_atoms(k, i) * float(gas.atomic_weights[i]) / molecular_weight
    return matrix, list(gas.element_names)


def _ammonia_hydrogen_phi(row: dict, gas) -> float:
    """Use the dataset's declared NH3/H2 fuel and O2 oxidizer basis."""
    if all(key in row for key in ("Y_NH3", "Y_H2", "Y_O2")):
        nh3 = max(0.0, float(row["Y_NH3"])) / float(gas.molecular_weights[gas.species_index("NH3")])
        h2 = max(0.0, float(row["Y_H2"])) / float(gas.molecular_weights[gas.species_index("H2")])
        o2 = float(row["Y_O2"]) / float(gas.molecular_weights[gas.species_index("O2")])
        if o2 > 0.0:
            return (0.75 * nh3 + 0.5 * h2) / o2
    if all(key in row for key in ("X_NH3", "X_H2", "X_O2")):
        nh3 = max(0.0, float(row["X_NH3"]))
        h2 = max(0.0, float(row["X_H2"]))
        o2 = float(row["X_O2"])
        if o2 > 0.0:
            return (0.75 * nh3 + 0.5 * h2) / o2
    return float(gas.equivalence_ratio())


def _audit_mechanism(mechanism: str, parts: list[tuple[Path, Path]], mechanism_path: Path | None = None) -> dict:
    first: dict[str, dict] = {}
    last: dict[str, dict] = {}
    first_times: dict[str, float | None] = {}
    last_times: dict[str, float | None] = {}
    y_columns: list[str] = []
    phi_values = []
    missing_columns = set()
    for part, _folder in parts:
        parquet = pq.ParquetFile(part)
        for batch in parquet.iter_batches(batch_size=65536):
            frame = batch.to_pandas()
            required = {"sample_id", "time"}
            missing_columns.update(required - set(frame.columns))
            if missing_columns:
                continue
            if not y_columns:
                y_columns = sorted(c for c in frame.columns if c.startswith("Y_") and not c.startswith("Y_next_"))
            for sample_id, group in frame.groupby("sample_id", sort=False):
                key = str(sample_id)
                times = pd.to_numeric(group["time"], errors="coerce").to_numpy(float)
                finite = np.isfinite(times)
                if finite.any():
                    positions = np.flatnonzero(finite)
                    first_pos = int(positions[np.argmin(times[finite])])
                    last_pos = int(positions[np.argmax(times[finite])])
                    first_time = float(times[first_pos])
                    last_time = float(times[last_pos])
                else:
                    first_pos = 0
                    last_pos = len(group) - 1
                    first_time = None
                    last_time = None

                if key not in first or (
                    first_time is not None
                    and (first_times[key] is None or first_time < first_times[key])
                ):
                    first[key] = group.iloc[first_pos].to_dict()
                    first_times[key] = first_time
                if key not in last or (
                    last_time is not None
                    and (last_times[key] is None or last_time > last_times[key])
                ):
                    last[key] = group.iloc[last_pos].to_dict()
                    last_times[key] = last_time
    if missing_columns:
        raise ValueError(f"{mechanism}: missing required columns {sorted(missing_columns)}")
    if not first:
        return {"mechanism": mechanism, "trajectories": 0, "empty": True}

    phi_values = [float(row["phi"]) for row in first.values() if "phi" in row and pd.notna(row["phi"])]
    radical_columns = [f"Y_{name}" for name in ("OH", "H", "O", "NO") if f"Y_{name}" in y_columns]
    initial_radical_total = (
        np.nansum(np.asarray([[float(row.get(column, 0.0)) for column in radical_columns] for row in first.values()]), axis=1)
        if radical_columns else np.zeros(len(first), dtype=float)
    )

    initial = np.asarray([[float(row.get(c, np.nan)) for c in y_columns] for row in first.values()])
    endpoint_columns = [f"Y_next_{column[2:]}" for column in y_columns]
    has_transition_endpoints = bool(endpoint_columns) and all(
        all(column in row for column in endpoint_columns)
        for row in last.values()
    )
    final_columns = endpoint_columns if has_transition_endpoints else y_columns
    final = np.asarray([[float(row.get(c, np.nan)) for c in final_columns] for row in last.values()])
    initial_sums = np.nansum(initial, axis=1)
    final_sums = np.nansum(final, axis=1)
    result = {
        "mechanism": mechanism,
        "trajectories": len(first),
        "species_columns": y_columns,
        "final_state_source": "Y_next" if has_transition_endpoints else "Y",
        "nonfinite_initial_values": int((~np.isfinite(initial)).sum()),
        "negative_initial_values": int((initial < -1e-12).sum()),
        "initial_mass_sum_max_abs_error": float(np.nanmax(np.abs(initial_sums - 1.0))),
        "final_mass_sum_max_abs_error": float(np.nanmax(np.abs(final_sums - 1.0))),
        "trajectory_element_closure_available": False,
        "phi_column_present": bool(phi_values),
        "phi_min": float(np.min(phi_values)) if phi_values else None,
        "phi_max": float(np.max(phi_values)) if phi_values else None,
        "phi_actual_available": False,
        "initial_radical_columns": radical_columns,
        "initial_radical_total_mean": float(np.mean(initial_radical_total)) if len(initial_radical_total) else 0.0,
        "initial_radical_total_max": float(np.max(initial_radical_total)) if len(initial_radical_total) else 0.0,
    }

    if mechanism_path and mechanism_path.exists():
        try:
            import cantera as ct

            gas = ct.Solution(str(mechanism_path))
            matrix, element_names = _element_mass_matrix(gas, y_columns)
            initial_elements = initial @ matrix.T
            final_elements = final @ matrix.T
            result["trajectory_element_closure_available"] = True
            result["element_names"] = element_names
            result["element_closure_max_abs_error"] = {
                name: float(np.nanmax(np.abs(initial_elements[:, i] - final_elements[:, i])))
                for i, name in enumerate(element_names)
            }
            actual_phi = []
            for row in first.values():
                composition = {name[2:]: float(row[name]) for name in y_columns if float(row[name]) > 0.0}
                try:
                    gas.TPY = float(row.get("T", 300.0)), float(row.get("P", 1.0)) * ct.one_atm, composition
                    actual_phi.append(_ammonia_hydrogen_phi(row, gas))
                except Exception:
                    actual_phi.append(np.nan)
            actual_phi = np.asarray(actual_phi, dtype=float)
            finite_phi = actual_phi[np.isfinite(actual_phi)]
            result["phi_actual_available"] = bool(finite_phi.size)
            result["actual_phi_min"] = float(np.min(finite_phi)) if finite_phi.size else None
            result["actual_phi_max"] = float(np.max(finite_phi)) if finite_phi.size else None
            result["actual_phi_nonfinite"] = int((~np.isfinite(actual_phi)).sum())
            if phi_values and len(phi_values) == len(actual_phi):
                target_phi = np.asarray(phi_values, dtype=float)
                finite_pair = np.isfinite(target_phi) & np.isfinite(actual_phi)
                result["phi_abs_error_max"] = (
                    float(np.max(np.abs(target_phi[finite_pair] - actual_phi[finite_pair])))
                    if finite_pair.any() else None
                )
        except Exception as exc:
            result["mechanism_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["mechanism_path"] = str(mechanism_path) if mechanism_path else None
        result["mechanism_unavailable"] = True
    return result


def audit_database(root: Path, mechanism_paths: dict[str, str] | None = None) -> dict:
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for mechanism, part, folder in _manifest_parts(Path(root)):
        grouped.setdefault(mechanism, []).append((part, folder))
    output = {}
    for mechanism, parts in grouped.items():
        configured = Path(mechanism_paths[mechanism]).expanduser() if mechanism_paths and mechanism in mechanism_paths else None
        configured = configured or _metadata_mechanism_path(parts[0][1], Path(root), mechanism)
        output[mechanism] = _audit_mechanism(mechanism, parts, configured)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit initial composition and phi for V67 Task 3")
    parser.add_argument("--database", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mechanism-path", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    paths = {}
    for item in args.mechanism_path:
        if "=" not in item:
            parser.error("--mechanism-path must use NAME=PATH")
        name, value = item.split("=", 1)
        paths[name] = value
    result = {"database": str(Path(args.database).resolve()), "task": "Task3", "audit": audit_database(Path(args.database), paths)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "initial_conditions_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    pd.DataFrame(result["audit"]).T.to_csv(out / "initial_conditions_audit.csv")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
