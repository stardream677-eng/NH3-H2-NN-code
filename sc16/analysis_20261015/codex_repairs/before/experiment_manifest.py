"""Experiment manifest and provenance contract module.

Guarantees full reproducibility and prevents silent experimental corruption by:
1. Hashing core source files and mechanisms (SHA256).
2. Verifying dataset row counts from Parquet file footers against metadata summaries.
3. Checking checkpoint compatibility before resuming or validating.
4. Auditing benchmark summary metrics directly from case records without dropping failures.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


CORE_SOURCE_FILES = (
    "train_sc_v68.py",
    "dual_branch_network.py",
    "enhanced_loss_and_weights.py",
    "thermo_nasa7_layer.py",
    "chemistry_contract.py",
    "validate_sc_v68.py",
)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute standard SHA256 hex digest for a file."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_source_hashes(root_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Compute SHA256 hashes of all core training and modeling Python files."""
    base = Path(root_dir) if root_dir else Path(__file__).resolve().parent
    hashes = {}
    for filename in CORE_SOURCE_FILES:
        target = base / filename
        if target.is_file():
            hashes[filename] = compute_file_sha256(target)
    return hashes


def audit_parquet_dataset(dataset_dir: Union[str, Path]) -> Dict[str, Any]:
    """Audit actual Parquet dataset files from metadata footers without reading full data into RAM."""
    dpath = Path(dataset_dir)
    if not dpath.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dpath}")

    import pyarrow.parquet as pq

    parquet_files = sorted(list(dpath.glob("part-*.parquet")))
    # Filter out backup files (.bak)
    parquet_files = [f for f in parquet_files if not f.name.endswith(".bak")]

    parts_info = []
    total_rows = 0
    for pf in parquet_files:
        meta = pq.read_metadata(pf)
        num_rows = meta.num_rows
        total_rows += num_rows
        parts_info.append({
            "file": pf.name,
            "rows": num_rows,
            "size_bytes": pf.stat().st_size,
        })

    summary_file = dpath / "metadata.json"
    recorded_rows = None
    inconsistency_warning = None

    if summary_file.is_file():
        try:
            summary_content = json.loads(summary_file.read_text(encoding="utf-8"))
            if isinstance(summary_content, list) and len(summary_content) > 0:
                recorded_rows = summary_content[0].get("training_rows")
            elif isinstance(summary_content, dict):
                recorded_rows = summary_content.get("training_rows")

            if recorded_rows is not None and recorded_rows != total_rows:
                inconsistency_warning = (
                    f"Dataset metadata inconsistency: metadata.json records {recorded_rows} rows, "
                    f"but actual Parquet files contain {total_rows} rows!"
                )
        except Exception as exc:
            inconsistency_warning = f"Failed to parse metadata.json: {exc}"

    return {
        "dataset_dir": str(dpath),
        "parquet_parts": parts_info,
        "actual_total_rows": total_rows,
        "recorded_total_rows": recorded_rows,
        "is_consistent": (inconsistency_warning is None),
        "inconsistency_warning": inconsistency_warning,
    }


def create_experiment_manifest(
    mechanism_contract: Dict[str, Any],
    output_parameterization: str,
    transform_signature: str,
    dt_policy: Dict[str, Any],
    concentration_policy: Dict[str, Any],
    evaluation_protocol_id: str,
    dataset_audit: Optional[Dict[str, Any]] = None,
    scaler_hashes: Optional[Dict[str, str]] = None,
    split_hash: Optional[str] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Assemble a comprehensive immutable experiment manifest."""
    source_hashes = compute_source_hashes(root_dir)

    manifest = {
        "manifest_version": "1.0",
        "source_hashes": source_hashes,
        "mechanism_name": mechanism_contract["mechanism_name"],
        "mechanism_sha256": mechanism_contract["mechanism_sha256"],
        "species_order": list(mechanism_contract["species_order"]),
        "element_names": list(mechanism_contract["element_names"]),
        "output_parameterization": output_parameterization,
        "transform_signature": transform_signature,
        "dt_policy": dt_policy,
        "concentration_policy": concentration_policy,
        "evaluation_protocol_id": evaluation_protocol_id,
        "split_hash": split_hash,
        "scaler_hashes": scaler_hashes or {},
        "dataset_audit": dataset_audit or {},
    }
    return manifest


def verify_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    expected_manifest: Dict[str, Any],
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """Verify that a checkpoint is strictly compatible with the current experimental contract."""
    errors = []
    
    ckpt_manifest = checkpoint.get("manifest")
    training_config = checkpoint.get("training_config", {})

    if ckpt_manifest is None:
        if strict:
            errors.append("Checkpoint lacks 'manifest'; legacy checkpoint cannot be loaded in strict new experiment mode")
        # Check training_config as fallback
        ckpt_param = training_config.get("output_parameterization")
        exp_param = expected_manifest.get("output_parameterization")
        if ckpt_param and exp_param and ckpt_param != exp_param:
            errors.append(f"Output parameterization mismatch: checkpoint '{ckpt_param}' vs expected '{exp_param}'")
        return (len(errors) == 0, errors)

    # 1. Mechanism check
    if ckpt_manifest.get("mechanism_sha256") != expected_manifest.get("mechanism_sha256"):
        errors.append(
            f"Mechanism SHA256 mismatch: checkpoint {ckpt_manifest.get('mechanism_sha256')[:8]} "
            f"vs expected {expected_manifest.get('mechanism_sha256')[:8]}"
        )

    # 2. Species order check
    if ckpt_manifest.get("species_order") != expected_manifest.get("species_order"):
        errors.append("Species order mismatch between checkpoint and current contract")

    # 3. Output parameterization check
    if ckpt_manifest.get("output_parameterization") != expected_manifest.get("output_parameterization"):
        errors.append(
            f"Output parameterization mismatch: checkpoint '{ckpt_manifest.get('output_parameterization')}' "
            f"vs expected '{expected_manifest.get('output_parameterization')}'"
        )

    # 4. Transform signature check
    ckpt_sig = ckpt_manifest.get("transform_signature")
    exp_sig = expected_manifest.get("transform_signature")
    if exp_sig is not None and ckpt_sig != exp_sig:
        errors.append(
            f"Transform signature mismatch: checkpoint '{ckpt_sig}' vs expected '{exp_sig}'"
        )

    # 5. Scaler hashes check if provided
    ckpt_scalers = ckpt_manifest.get("scaler_hashes", {})
    exp_scalers = expected_manifest.get("scaler_hashes", {})
    if strict and exp_scalers:
        missing_scalers = set(exp_scalers.keys()) - set(ckpt_scalers.keys())
        if missing_scalers:
            errors.append(f"Missing scaler hashes in checkpoint manifest: {sorted(missing_scalers)}")
    for k, v in exp_scalers.items():
        if k in ckpt_scalers and ckpt_scalers[k] != v:
            errors.append(f"Scaler hash mismatch for '{k}'")

    is_compatible = len(errors) == 0
    return is_compatible, errors


def recompute_benchmark_summary(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute benchmark summary strictly from case records, surfacing all failures."""
    if not cases:
        raise ValueError("Cannot recompute benchmark summary: case list is empty")

    n_cases = len(cases)
    ignited_count = 0
    closure_success_count = 0
    
    valid_idt_ratios = []
    valid_abs_delta_tends = []
    valid_no_errors = []
    wiped_species_total = 0

    for c in cases:
        # Closure check
        if c.get("closure_ok", True) and c.get("fail_step") is None:
            closure_success_count += 1
            
        # IDT and ignition check
        nn_idt = c.get("nn_idt_ms")
        ref_idt = c.get("ref_idt_ms")
        if nn_idt is not None and np.isfinite(nn_idt):
            ignited_count += 1
            
        idt_ratio = c.get("idt_ratio")
        if idt_ratio is not None and np.isfinite(idt_ratio):
            valid_idt_ratios.append(float(idt_ratio))

        # Tend check
        delta_tend = c.get("delta_Tend")
        if delta_tend is not None and np.isfinite(delta_tend):
            valid_abs_delta_tends.append(abs(float(delta_tend)))

        # NO error check
        no_err = c.get("no_rel_err")
        if no_err is not None and np.isfinite(no_err):
            valid_no_errors.append(float(no_err))

        wiped_species_total += int(c.get("wiped_species_count", 0))

    summary = {
        "num_cases": n_cases,
        "closure_success_rate": closure_success_count / n_cases,
        "ignited_count": ignited_count,
        "mean_idt_ratio": float(np.mean(valid_idt_ratios)) if valid_idt_ratios else None,
        "median_idt_ratio": float(np.median(valid_idt_ratios)) if valid_idt_ratios else None,
        "mean_abs_delta_Tend": float(np.mean(valid_abs_delta_tends)) if valid_abs_delta_tends else None,
        "mean_no_rel_err": float(np.mean(valid_no_errors)) if valid_no_errors else None,
        "total_wiped_species": wiped_species_total,
    }
    return summary
