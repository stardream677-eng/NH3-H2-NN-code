"""Shared initial-condition and moved-database helpers for Task2/Task4."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "_task3_phi", Path(__file__).with_name("audit_initial_conditions_v67.py")
)
_task3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task3)


def dataset_parts(root: Path):
    """Resolve each manifest dataset once; fail on missing/ambiguous inputs."""
    root = Path(root)
    manifest = root / "dataset_manifest.csv"
    seen = set()
    for row in pd.read_csv(manifest).to_dict("records"):
        def field(names):
            return next((row[name] for name in names
                         if name in row and pd.notna(row[name]) and str(row[name]).strip()), None)
        mechanism = field(("mechanism_name", "mechanism"))
        recorded = field(("dataset_path", "dataset_folder", "path"))
        if mechanism is None or recorded is None:
            raise ValueError(f"Missing mechanism/dataset path in {manifest}: {row}")
        mechanism = str(mechanism)
        local = root / f"trajectories_{mechanism}"
        folder = Path(str(recorded))
        folder = folder if folder.is_absolute() else root / folder
        folder = local if local.is_dir() else folder
        if mechanism in seen:
            raise ValueError(f"Duplicate mechanism in manifest: {mechanism}; cannot merge sample IDs safely")
        seen.add(mechanism)
        parts = sorted(folder.glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet parts for {mechanism}: {folder}")
        yield mechanism, parts


def mechanism_path(folder: Path, root: Path, mechanism: str) -> Path | None:
    path = _task3._metadata_mechanism_path(folder, root, mechanism)
    if path is not None and path.is_file():
        return path
    known = {"nuig_2024": "NUIG2024.yaml", "tsinghua_2024c": "TH2024.yaml", "ucf_2024": "UCF2024.yaml"}
    name = known.get(mechanism)
    if name:
        for base in (root, root.parent, root / "mechanisms", root.parent / "mechanisms"):
            if (base / name).is_file():
                return (base / name).resolve()
    return path


def load_phi_gas(folder: Path, root: Path, mechanism: str):
    path = mechanism_path(folder, root, mechanism)
    try:
        if path is None:
            raise FileNotFoundError("mechanism_unavailable")
        import cantera as ct
        return ct.Solution(str(path)), path, None
    except Exception as exc:
        return None, path, f"{type(exc).__name__}: {exc}"


def actual_phi(row: dict, gas, mechanism_error=None):
    """Return value, failure reason and source without masking bad fuel inputs.

    Reuse Task3's NH3/H2/O2 mole-basis definition. This ratio is independent
    of T/P; no Cantera state mutation or fallback to its previous state occurs.
    """
    try:
        recorded = float(row.get("phi_actual", np.nan))
        if np.isfinite(recorded) and recorded >= 0:
            return recorded, None, "phi_actual_column"
        columns = next((tuple(f"{prefix}_{sp}" for sp in ("NH3", "H2", "O2"))
                        for prefix in ("Y", "X")
                        if all(f"{prefix}_{sp}" in row for sp in ("NH3", "H2", "O2"))), None)
        if columns is None:
            raise ValueError("missing_NH3_H2_O2_columns")
        basis = np.asarray([row[c] for c in columns], dtype=float)
        if not np.isfinite(basis).all():
            raise ValueError("nonfinite_NH3_H2_O2")
        if (basis < 0).any():
            raise ValueError("negative_NH3_H2_O2")
        if basis[2] <= 0:
            raise ValueError("nonpositive_O2")
        if columns[0].startswith("Y_") and gas is None:
            raise ValueError(mechanism_error or "mechanism_unavailable")
        value = float(_task3._ammonia_hydrogen_phi(row, gas))
        if not np.isfinite(value):
            raise ValueError("nonfinite_phi")
        return value, None, "task3_composition"
    except Exception as exc:  # Cantera species lookup errors are reported too.
        return None, f"{type(exc).__name__}: {exc}", None
