"""Standalone V68 model validation and plotting.

The script reuses the exact model, feature transforms, scalers and rollout
implementation from ``train_sc_v68.py``.  It evaluates both teacher-forced
(oracle) and species-autoregressive (pure_ar) rollouts and writes machine-
readable metrics together with diagnostic plots.

Example (run from the ``train`` directory)::

    python validate_sc_v68.py --mechanism nuig_2024 --split val

The default ``data_dir`` is discovered beside this script when the local data
layout is present. On another machine, pass ``--data_dir`` explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import joblib
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise RuntimeError("joblib is required; install the training dependencies") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plotting can be disabled while metrics still run
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_sc_v68 as tr  # noqa: E402

# Prioritize local directory first, then optional tools subdirectory
for cand_dir in (SCRIPT_DIR, SCRIPT_DIR / "tools"):
    if str(cand_dir) not in sys.path:
        sys.path.insert(0, str(cand_dir))
try:
    import compute_physical_metrics as physical_metrics  # noqa: E402
    PHYSICAL_METRICS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - standalone validator still runs
    physical_metrics = None
    PHYSICAL_METRICS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Scalers created by train_sc_v68.py were pickled as __main__.FeatureScaler.
# During validation this script is __main__, so register the training class
# under that name before any joblib.load() call.
import __main__  # noqa: E402

if hasattr(tr, "FeatureScaler"):
    __main__.FeatureScaler = tr.FeatureScaler


DEFAULT_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128)


def discover_default_data_root() -> Path:
    """Find a useful local default without baking in one machine's drive."""
    configured = os.environ.get("V68_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()

    candidates = (
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "data",
        SCRIPT_DIR.parent.parent / "data",
    )
    for candidate in candidates:
        if (candidate / "database_Transient_V21_QSSA").exists():
            return candidate
    # The explicit --data_dir option remains the reliable choice when data
    # lives on another machine or outside the code directory.
    return SCRIPT_DIR


DEFAULT_DATA_ROOT = discover_default_data_root()
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("V68_OUTPUT_ROOT", str(DEFAULT_DATA_ROOT))
).expanduser()
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "V68_CHECKPOINT",
        str(
            DEFAULT_DATA_ROOT
            / "checkpoints_v68"
            / "checkpoint_v68_nuig_2024_epoch100.pth"
        ),
    )
).expanduser()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path, mechanism: str | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=tr.DEVICE, weights_only=False)
    except TypeError:  # older torch versions
        checkpoint = torch.load(path, map_location=tr.DEVICE)
    required = {"model_state_dict", "feature_info"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if checkpoint.get("version") != tr.VERSION:
        raise ValueError(
            f"Checkpoint version {checkpoint.get('version')!r} is not {tr.VERSION!r}"
        )
    valid_transforms = {
        "macro_asinh_p90_v2",
        "qssa_log10_P_D_v2",
        "qssa_asymptotic_v2",
        getattr(tr, "DUAL_BRANCH_TRANSFORM_SIGNATURE", "dual_branch_asinh_macro_qssa_micro_v1"),
        getattr(tr, "DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE", "dual_branch_pd_v1"),
        tr.TRANSFORM_SIGNATURE,
    }
    if checkpoint.get("transform_signature") not in valid_transforms:
        raise ValueError(
            f"Checkpoint transform signature {checkpoint.get('transform_signature')!r} "
            f"does not match expected {sorted(valid_transforms)}"
        )
    if checkpoint.get("rollout_signature") not in (None, tr.ROLLOUT_SIGNATURE):
        raise ValueError("Checkpoint rollout signature does not match V68")
    if mechanism:
        checkpoint_mechanism = checkpoint.get("mechanism")
        feature_info = checkpoint.get("feature_info", {})
        if not isinstance(feature_info, dict):
            raise ValueError("Checkpoint feature_info must be a dictionary")
        feature_mechanism = feature_info.get("mechanism")
        for source, value in (
            ("checkpoint", checkpoint_mechanism),
            ("feature_info", feature_mechanism),
        ):
            if value and value != mechanism:
                raise ValueError(
                    f"Checkpoint mechanism mismatch: {source} contains "
                    f"{value!r}, requested {mechanism!r}"
                )
    fi = checkpoint["feature_info"]
    expected_signature = tr.transform_signature_for(fi.get("output_parameterization", "direct_rate"))
    if checkpoint.get("transform_signature") != expected_signature or fi.get("transform_signature") != expected_signature:
        raise ValueError("Transform signature differs between checkpoint, mode and feature_info")
    import experiment_manifest as em
    em.verify_runtime_artifacts(checkpoint, fi)
    return checkpoint


def checkpoint_rollout_config(checkpoint: dict) -> dict:
    """Return rollout settings saved with a V3.x checkpoint."""
    config = checkpoint.get("training_config") or {}
    return {
        "experiment_mode": config.get("experiment_mode", "baseline"),
        "state_update_mode": config.get("state_update_mode", "legacy"),
        "trust_region_fraction": float(config.get("trust_region_fraction", 0.0) or 0.0),
        "temperature_mode": config.get(
            "effective_temperature_mode",
            config.get("temperature_mode", "recorded"),
        ),
        "max_delta_temperature": float(config.get("max_delta_temperature", 1000.0) or 1000.0),
        "feedback_blend_start": float(config.get("feedback_blend_start", 1.0) or 1.0),
        "feedback_blend_end": float(config.get("feedback_blend_end", 1.0) or 1.0),
        "feedback_blend_ramp_epochs": int(config.get("feedback_blend_ramp_epochs", 0) or 0),
    }


def build_temperature_closure_for_validation(
    metadata: dict,
    feature_info: dict,
    data_dir: Path,
    mechanism: str,
) -> dict:
    """Build the same closure as training, preferring the validation data root."""
    return tr.build_temperature_closure(
        metadata,
        feature_info.get("inert_species", []) + feature_info["macro_species"] + feature_info["micro_species"],
        data_dir=data_dir,
        mechanism=mechanism,
    )


def choose_checkpoint(
    output_root: Path,
    mechanism: str,
    requested: str | None,
    data_dir: Path | None = None,
) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()

    roots = [output_root]
    if data_dir is not None:
        roots.append(data_dir)
    roots.append(DEFAULT_DATA_ROOT)
    seen = set()
    candidates = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        weights = root / tr.artifact_subdir("weights")
        checkpoints = root / tr.artifact_subdir("checkpoints")
        candidates.extend(
            (
                weights / f"best_v68_net_{mechanism}.pth",
                weights / f"latest_v68_{mechanism}.pth",
                checkpoints / f"checkpoint_v68_{mechanism}_epoch100.pth",
            )
        )
    if mechanism == "nuig_2024":
        candidates.append(DEFAULT_CHECKPOINT)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No checkpoint found in the configured artifact roots. "
        "Pass --checkpoint explicitly. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _safe_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "checkpoint"


def checkpoint_label(
    checkpoint: dict,
    path: Path,
    include_stem: bool = False,
) -> str:
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        base = f"epoch{int(epoch)}"
    else:
        match = re.search(r"(?:epoch|ep)[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
        base = f"epoch{match.group(1)}" if match else _safe_label(path.stem)
    return f"{base}_{_safe_label(path.stem)}" if include_stem else base


def unique_checkpoint_labels(checkpoints: List[dict], paths: List[Path]) -> List[str]:
    """Return stable, collision-free directory labels for checkpoint runs."""
    include_stem = len(checkpoints) > 1
    labels = []
    used = {}
    for checkpoint, path in zip(checkpoints, paths):
        base = checkpoint_label(checkpoint, path, include_stem=include_stem)
        count = used.get(base, 0) + 1
        used[base] = count
        labels.append(base if count == 1 else f"{base}_{count}")
    return labels


def _validate_scaler(scaler, name: str, expected_dim: int) -> None:
    mean = np.asarray(getattr(scaler, "mean_", None))
    scale = np.asarray(getattr(scaler, "scale_", None))
    if mean.shape != (expected_dim,) or scale.shape != (expected_dim,):
        raise ValueError(
            f"{name} has shape mean={mean.shape}, scale={scale.shape}; "
            f"expected ({expected_dim},)"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"{name} contains non-finite mean/scale values")
    if (scale <= 0).any():
        raise ValueError(f"{name} contains non-positive scale values")


def load_scalers(
    output_root: Path,
    mechanism: str,
    feature_info: dict,
    data_dir: Path | None = None,
    rebuild_scalers: bool = False,
    split_dir: Path | None = None,
    checkpoint_root: Path | list[Path] | None = None,
):
    """Load the training scalers, optionally rebuilding them by explicit request.

    Validation must use the training transform.  The fallback mirrors the
    training script's streaming-scaler defaults and is intended for machines
    where only the checkpoint and parquet database were copied.
    """
    scaler_suffix = ""
    if split_dir is not None:
        split_path = Path(split_dir) / f"{mechanism}_train_sample_ids.csv"
        if split_path.is_file():
            signature = tr.trajectory_split_signature(split_dir, mechanism, "train").split(":", 1)[-1]
            scaler_suffix = f"_split_{signature}"
    candidates = [output_root / tr.artifact_subdir("scalers")]
    if checkpoint_root is not None:
        if isinstance(checkpoint_root, (list, tuple)):
            for cr in checkpoint_root:
                candidates.append(Path(cr) / tr.artifact_subdir("scalers"))
        else:
            candidates.append(Path(checkpoint_root) / tr.artifact_subdir("scalers"))
    # Also search current working directory, script directory, and parent of weights
    candidates.extend([
        Path.cwd() / tr.artifact_subdir("scalers"),
        SCRIPT_DIR / tr.artifact_subdir("scalers"),
    ])
    if data_dir is not None:
        candidates.extend([data_dir / tr.artifact_subdir("scalers"), data_dir.parent / tr.artifact_subdir("scalers")])
    target_signatures = []
    ckpt_sig = (feature_info or {}).get("transform_signature")
    if ckpt_sig:
        target_signatures.append(ckpt_sig)
    for sig in ("macro_asinh_p90_v2", getattr(tr, "TRANSFORM_SIGNATURE", None), "qssa_log10_P_D_v2", "qssa_asymptotic_v2"):
        if sig and sig not in target_signatures:
            target_signatures.append(sig)

    scaler_dir = None
    matched_sig = None
    matched_suffix = None
    for candidate in dict.fromkeys(candidates):
        for sig in target_signatures:
            for s_suff in ([scaler_suffix, ""] if scaler_suffix else [""]):
                if all((candidate / f).exists() for f in (
                    f"scaler_X_{mechanism}_{sig}{s_suff}.pkl",
                    f"scaler_Y_{mechanism}_{sig}{s_suff}.pkl",
                    f"feature_info_{mechanism}_{sig}{s_suff}.pkl",
                )):
                    scaler_dir = candidate
                    matched_sig = sig
                    matched_suffix = s_suff
                    break
            if scaler_dir is not None:
                break
        if scaler_dir is not None:
            break

    if scaler_dir is None:
        # Check if any candidate has any matching split-suffixed scaler files
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_dir():
                continue
            for sig in target_signatures:
                matches = list(candidate.glob(f"scaler_X_{mechanism}_{sig}_split_*.pkl"))
                for m in matches:
                    prefix = f"scaler_X_{mechanism}_{sig}"
                    suff = m.name[len(prefix):-4]
                    if (candidate / f"scaler_Y_{mechanism}_{sig}{suff}.pkl").exists() and \
                       (candidate / f"feature_info_{mechanism}_{sig}{suff}.pkl").exists():
                        scaler_dir = candidate
                        matched_sig = sig
                        matched_suffix = suff
                        break
                if scaler_dir is not None:
                    break
            if scaler_dir is not None:
                break

    if scaler_dir is None:
        sig_to_use = target_signatures[0] if target_signatures else tr.TRANSFORM_SIGNATURE
        expected = candidates[0] / f"scaler_X_{mechanism}_{sig_to_use}.pkl"
        if not rebuild_scalers:
            raise FileNotFoundError(
                f"Training scalers not found (expected {expected}). "
                "Pass --rebuild_scalers only when rebuilding from the validation data "
                "is intentional."
            )
        if data_dir is None:
            raise FileNotFoundError(
                "Cannot rebuild scalers without --data_dir."
            )
        print("[validate] scalers not found; rebuilding with training streaming defaults...", flush=True)
        _, _, _, parquet_files = tr.get_schema(str(data_dir), mechanism)
        train_sample_ids = None
        if split_dir is not None:
            train_sample_ids = tr.load_trajectory_sample_ids(split_dir, mechanism, "train")
        scaler_x, scaler_y = tr.compute_streaming_scalers(
            str(data_dir), mechanism, feature_info, parquet_files,
            max_parts=10, sample_rate=0.02, max_rows=2_000_000,
            sample_ids=train_sample_ids,
        )
        _validate_scaler(scaler_x, "scaler_X", len(feature_info["input_cols"]))
        _validate_scaler(scaler_y, "scaler_Y", len(feature_info["output_cols"]))
        import experiment_manifest as em
        em.verify_runtime_artifacts({"manifest": feature_info.get("manifest")}, feature_info,
                                    scaler_x, scaler_y, (data_dir,))
        scaler_dir = output_root / tr.artifact_subdir("scalers")
        scaler_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_x, scaler_dir / f"scaler_X_{mechanism}_{sig_to_use}{scaler_suffix}.pkl")
        joblib.dump(scaler_y, scaler_dir / f"scaler_Y_{mechanism}_{sig_to_use}{scaler_suffix}.pkl")
        saved_info = dict(feature_info)
        saved_info["_scaler_source"] = f"rebuilt:{scaler_dir}"
        joblib.dump(saved_info, scaler_dir / f"feature_info_{mechanism}_{sig_to_use}{scaler_suffix}.pkl")
        return scaler_x, scaler_y, saved_info
    sx_path = scaler_dir / f"scaler_X_{mechanism}_{matched_sig}{matched_suffix}.pkl"
    sy_path = scaler_dir / f"scaler_Y_{mechanism}_{matched_sig}{matched_suffix}.pkl"
    fi_path = scaler_dir / f"feature_info_{mechanism}_{matched_sig}{matched_suffix}.pkl"
    scaler_x = joblib.load(sx_path)
    scaler_y = joblib.load(sy_path)
    saved_info = joblib.load(fi_path)
    _validate_scaler(scaler_x, "scaler_X", len(feature_info["input_cols"]))
    _validate_scaler(scaler_y, "scaler_Y", len(feature_info["output_cols"]))
    if not isinstance(saved_info, dict):
        raise ValueError(f"Saved feature_info is not a dictionary: {fi_path}")
    valid_transforms = {
        "macro_asinh_p90_v2",
        "qssa_log10_P_D_v2",
        "qssa_asymptotic_v2",
        getattr(tr, "DUAL_BRANCH_TRANSFORM_SIGNATURE", "dual_branch_asinh_macro_qssa_micro_v1"),
        getattr(tr, "DUAL_BRANCH_PD_V1_TRANSFORM_SIGNATURE", "dual_branch_pd_v1"),
        tr.TRANSFORM_SIGNATURE,
    }
    if saved_info.get("transform_signature") not in valid_transforms:
        raise ValueError(f"Saved feature_info has an incompatible transform signature: {saved_info.get('transform_signature')!r}")
    # The scaler-side feature contract is authoritative. This catches an
    # accidental mechanism/checkpoint mix-up before any predictions are made.
    for key in ("input_cols", "output_cols", "all_species", "macro_species", "micro_species", "inert_species"):
        if list(saved_info.get(key, [])) != list(feature_info.get(key, [])):
            raise ValueError(f"Checkpoint and scaler feature_info differ in {key}")
    if not np.array_equal(saved_info.get("macro_rate_scale"), feature_info.get("macro_rate_scale")):
        raise ValueError("Checkpoint and scaler feature_info differ in macro_rate_scale")
    if saved_info.get("rate_decode_mode", "direct_rate") != feature_info.get(
        "rate_decode_mode", "direct_rate"
    ):
        raise ValueError("Checkpoint and scaler feature_info differ in rate_decode_mode")
    saved_rate_z_limit = float(
        saved_info.get("rate_z_limit", tr.DEFAULT_RATE_Z_LIMIT)
    )
    checkpoint_rate_z_limit = float(
        feature_info.get("rate_z_limit", tr.DEFAULT_RATE_Z_LIMIT)
    )
    if not np.isclose(saved_rate_z_limit, checkpoint_rate_z_limit):
        raise ValueError("Checkpoint and scaler feature_info differ in rate_z_limit")
    saved_info["_scaler_source"] = str(scaler_dir)
    import experiment_manifest as em
    em.verify_runtime_artifacts({"manifest": feature_info.get("manifest")}, feature_info,
                                scaler_x, scaler_y, (data_dir,) if data_dir is not None else ())
    return scaler_x, scaler_y, saved_info


def split_parts(parts: List[Path], val_ratio: float, seed: int, split: str):
    if split == "all":
        return list(parts)
    shuffled = list(parts)
    np.random.default_rng(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    if split == "val":
        return shuffled[:n_val]
    if split == "train":
        if len(shuffled) == 1:
            raise ValueError("Cannot create a train split from a single parquet part")
        return shuffled[n_val:]
    raise ValueError(f"Unknown split: {split}")


class MetricAccumulator:
    """Online per-column MAE/RMSE/bias/R2 and relative error."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        n = len(names)
        self.abs_error = np.zeros(n, dtype=np.float64)
        self.sq_error = np.zeros(n, dtype=np.float64)
        self.sum_error = np.zeros(n, dtype=np.float64)
        self.sum_true = np.zeros(n, dtype=np.float64)
        self.sum_true_sq = np.zeros(n, dtype=np.float64)
        self.abs_true = np.zeros(n, dtype=np.float64)
        self.count = np.zeros(n, dtype=np.int64)
        self.nonfinite = 0

    def update(self, pred, true) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred[:, None]
            true = true[:, None]
        finite = np.isfinite(pred) & np.isfinite(true)
        self.nonfinite += int((~finite).sum())
        err = np.where(finite, pred - true, 0.0)
        true_safe = np.where(finite, true, 0.0)
        self.abs_error += np.abs(err).sum(axis=0)
        self.sq_error += np.square(err).sum(axis=0)
        self.sum_error += err.sum(axis=0)
        self.sum_true += true_safe.sum(axis=0)
        self.sum_true_sq += np.square(true_safe).sum(axis=0)
        self.abs_true += np.abs(true_safe).sum(axis=0)
        self.count += finite.sum(axis=0)

    def as_dict(self) -> dict:
        result = {}
        for i, name in enumerate(self.names):
            n = int(self.count[i])
            if n == 0:
                result[name] = {"n": 0, "mae": None, "rmse": None, "bias": None,
                                "r2": None, "relative_mae": None}
                continue
            sse = self.sq_error[i]
            sst = self.sum_true_sq[i] - self.sum_true[i] ** 2 / n
            result[name] = {
                "n": n,
                "mae": float(self.abs_error[i] / n),
                "rmse": float(math.sqrt(max(0.0, sse / n))),
                "bias": float(self.sum_error[i] / n),
                "r2": float(1.0 - sse / sst) if sst > 1e-30 else None,
                "relative_mae": float(self.abs_error[i] / max(self.abs_true[i], 1e-30)),
            }
        return {"per_species": result, "nonfinite_values": self.nonfinite}


class DiagnosticAccumulator:
    """Aggregate raw-negative and clamp statistics by macro species."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        n = len(self.names)
        self.total_predictions = 0
        self.predictions_with_negative = 0
        self.negative_count = np.zeros(n, dtype=np.int64)
        self.negative_sum = np.zeros(n, dtype=np.float64)
        self.clamp_sum = np.zeros(n, dtype=np.float64)
        self.clamp_sq_sum = np.zeros(n, dtype=np.float64)
        self.clamp_max = np.zeros(n, dtype=np.float64)
        self.raw_min = np.full(n, np.inf, dtype=np.float64)
        self.raw_error = MetricAccumulator(self.names)
        self.clamped_error = MetricAccumulator(self.names)

    def update(self, raw, true=None) -> None:
        raw = np.asarray(raw, dtype=np.float64).reshape(-1, len(self.names))
        finite = np.isfinite(raw)
        negative = finite & (raw < 0.0)
        clamped = np.maximum(raw, 0.0)
        magnitude = np.where(negative, clamped - raw, 0.0)

        self.total_predictions += raw.shape[0]
        self.predictions_with_negative += int(negative.any(axis=1).sum())
        self.negative_count += negative.sum(axis=0)
        self.negative_sum += np.where(negative, raw, 0.0).sum(axis=0)
        self.clamp_sum += magnitude.sum(axis=0)
        self.clamp_sq_sum += np.square(magnitude).sum(axis=0)
        self.clamp_max = np.maximum(self.clamp_max, magnitude.max(axis=0))
        finite_min = np.where(finite, raw, np.inf).min(axis=0)
        self.raw_min = np.minimum(self.raw_min, finite_min)

        if true is not None:
            true = np.asarray(true, dtype=np.float64).reshape(-1, len(self.names))
            self.raw_error.update(raw, true)
            self.clamped_error.update(clamped, true)

    def as_dict(self) -> dict:
        total_values = self.total_predictions * len(self.names)
        negative_values = int(self.negative_count.sum())
        total_clamp = float(self.clamp_sum.sum())
        per_species = {}
        for i, name in enumerate(self.names):
            count = int(self.negative_count[i])
            per_species[name] = {
                "total_predictions": self.total_predictions,
                "negative_values": count,
                "negative_ratio": (
                    float(count / self.total_predictions)
                    if self.total_predictions else 0.0
                ),
                "mean_negative_value": (
                    float(self.negative_sum[i] / count) if count else None
                ),
                "minimum_raw_value": (
                    float(self.raw_min[i]) if np.isfinite(self.raw_min[i]) else None
                ),
                "total_clamp_magnitude": float(self.clamp_sum[i]),
                "mean_clamp_magnitude": (
                    float(self.clamp_sum[i] / count) if count else 0.0
                ),
                "max_clamp_magnitude": float(self.clamp_max[i]),
            }

        return {
            "total_predictions": self.total_predictions,
            "negative_predictions": self.predictions_with_negative,
            "negative_ratio": (
                float(self.predictions_with_negative / self.total_predictions)
                if self.total_predictions else 0.0
            ),
            "total_values": total_values,
            "negative_values": negative_values,
            "negative_value_ratio": (
                float(negative_values / total_values) if total_values else 0.0
            ),
            "total_clamp_magnitude": total_clamp,
            "mean_clamp_magnitude": (
                float(total_clamp / negative_values) if negative_values else 0.0
            ),
            "max_clamp_magnitude": float(self.clamp_max.max(initial=0.0)),
            "clamp_magnitude_rmse": (
                float(math.sqrt(self.clamp_sq_sum.sum() / total_values))
                if total_values else 0.0
            ),
            "raw_vs_true": self.raw_error.as_dict(),
            "clamped_vs_true": self.clamped_error.as_dict(),
            "per_species": per_species,
        }


class ConstraintInterventionAccumulator:
    """Count actual update, feedback, and projection interventions."""

    def __init__(self, names: List[str], tolerance: float = 1e-12):
        self.names = list(names)
        self.tolerance = float(tolerance)
        n = len(self.names)
        self.total_predictions = 0
        self.update_count = np.zeros(n, dtype=np.int64)
        self.update_sum = np.zeros(n, dtype=np.float64)
        self.update_sq_sum = np.zeros(n, dtype=np.float64)
        self.update_max = np.zeros(n, dtype=np.float64)
        self.feedback_count = np.zeros(n, dtype=np.int64)
        self.feedback_sum = np.zeros(n, dtype=np.float64)
        self.feedback_sq_sum = np.zeros(n, dtype=np.float64)
        self.feedback_max = np.zeros(n, dtype=np.float64)
        self.projection_count = np.zeros(n, dtype=np.int64)
        self.projection_sum = np.zeros(n, dtype=np.float64)
        self.projection_sq_sum = np.zeros(n, dtype=np.float64)
        self.projection_max = np.zeros(n, dtype=np.float64)
        self.update_rows = 0
        self.feedback_rows = 0
        self.projection_rows = 0

    def _record(self, raw, applied, count, total, sq_total, maximum):
        raw = np.asarray(raw, dtype=np.float64).reshape(-1, len(self.names))
        applied = np.asarray(applied, dtype=np.float64).reshape(-1, len(self.names))
        finite = np.isfinite(raw) & np.isfinite(applied)
        correction = np.where(finite, np.abs(applied - raw), 0.0)
        intervened = finite & (correction > self.tolerance)
        count += intervened.sum(axis=0)
        total += correction.sum(axis=0)
        sq_total += np.square(correction).sum(axis=0)
        maximum[:] = np.maximum(maximum, correction.max(axis=0, initial=0.0))
        return int(intervened.any(axis=1).sum())

    def update(self, raw_update, applied_update, raw_feedback, applied_feedback,
               projected=None) -> None:
        raw_update = np.asarray(raw_update, dtype=np.float64)
        applied_update = np.asarray(applied_update, dtype=np.float64)
        raw_feedback = np.asarray(raw_feedback, dtype=np.float64)
        applied_feedback = np.asarray(applied_feedback, dtype=np.float64)
        self.total_predictions += raw_update.shape[0]
        self.update_rows += self._record(
            raw_update, applied_update, self.update_count, self.update_sum,
            self.update_sq_sum, self.update_max,
        )
        self.feedback_rows += self._record(
            raw_feedback, applied_feedback, self.feedback_count, self.feedback_sum,
            self.feedback_sq_sum, self.feedback_max,
        )
        if projected is not None:
            self.projection_rows += self._record(
                raw_update, projected, self.projection_count, self.projection_sum,
                self.projection_sq_sum, self.projection_max,
            )

    def _summary(self, count, total, sq_total, maximum, intervened_rows):
        values = int(count.sum())
        total_values = self.total_predictions * len(self.names)
        return {
            "intervened_values": values,
            "intervened_value_ratio": float(values / total_values) if total_values else 0.0,
            "intervened_rows": int(intervened_rows),
            "total_correction": float(total.sum()),
            "mean_correction": float(total.sum() / values) if values else 0.0,
            "max_correction": float(maximum.max(initial=0.0)),
            "correction_rmse": (
                float(math.sqrt(sq_total.sum() / total_values)) if total_values else 0.0
            ),
            "per_species": {
                name: {
                    "intervened_values": int(count[i]),
                    "intervened_ratio": (
                        float(count[i] / self.total_predictions)
                        if self.total_predictions else 0.0
                    ),
                    "total_correction": float(total[i]),
                    "mean_correction": float(total[i] / count[i]) if count[i] else 0.0,
                    "max_correction": float(maximum[i]),
                }
                for i, name in enumerate(self.names)
            },
        }

    def as_dict(self) -> dict:
        result = {
            "total_predictions": self.total_predictions,
            "update": self._summary(
                self.update_count, self.update_sum, self.update_sq_sum, self.update_max,
                self.update_rows,
            ),
            "feedback": self._summary(
                self.feedback_count, self.feedback_sum, self.feedback_sq_sum, self.feedback_max,
                self.feedback_rows,
            ),
            "projection": self._summary(
                self.projection_count, self.projection_sum,
                self.projection_sq_sum, self.projection_max, self.projection_rows,
            ),
        }
        result["update"]["intervened_rows"] = self.update_rows
        result["feedback"]["intervened_rows"] = self.feedback_rows
        result["projection"]["intervened_rows"] = self.projection_rows
        for key in ("update", "feedback", "projection"):
            result[key]["intervened_row_ratio"] = float(
                result[key]["intervened_rows"] / self.total_predictions
            ) if self.total_predictions else 0.0
        return result


class RateDiagnosticAccumulator:
    """Summarize physical macro-rate errors and temporal rate jumps."""

    def __init__(self, names: List[str], rate_scales, steady_rate_threshold: float = 0.05):
        self.names = list(names)
        self.rate_scales = np.maximum(np.asarray(rate_scales, dtype=np.float64), 1e-30)
        self.steady_rate_threshold = float(steady_rate_threshold)
        n = len(self.names)
        self.count = np.zeros(n, dtype=np.int64)
        self.pred_sum = np.zeros(n, dtype=np.float64)
        self.pred_sq_sum = np.zeros(n, dtype=np.float64)
        self.true_sum = np.zeros(n, dtype=np.float64)
        self.true_sq_sum = np.zeros(n, dtype=np.float64)
        self.abs_error = np.zeros(n, dtype=np.float64)
        self.sq_error = np.zeros(n, dtype=np.float64)
        self.sum_error = np.zeros(n, dtype=np.float64)
        self.max_abs_pred = np.zeros(n, dtype=np.float64)
        self.max_abs_error = np.zeros(n, dtype=np.float64)
        self.steady_count = np.zeros(n, dtype=np.int64)
        self.steady_abs_error = np.zeros(n, dtype=np.float64)
        self.steady_sq_error = np.zeros(n, dtype=np.float64)
        self.steady_sum_error = np.zeros(n, dtype=np.float64)
        self.jump_count = np.zeros(n, dtype=np.int64)
        self.max_pred_jump = np.zeros(n, dtype=np.float64)
        self.pred_jump_sq = np.zeros(n, dtype=np.float64)
        self.true_jump_sq = np.zeros(n, dtype=np.float64)
        self.residual_jump_sq = np.zeros(n, dtype=np.float64)
        self.state_jump_count = np.zeros(n, dtype=np.int64)
        self.max_state_jump = np.zeros(n, dtype=np.float64)
        self.state_jump_sq = np.zeros(n, dtype=np.float64)
        self.true_state_jump_sq = np.zeros(n, dtype=np.float64)
        self.residual_state_jump_sq = np.zeros(n, dtype=np.float64)

    def update(
        self,
        pred,
        true,
        previous_pred=None,
        previous_true=None,
        state_pred=None,
        state_true=None,
        previous_state_pred=None,
        previous_state_true=None,
    ) -> None:
        pred = np.asarray(pred, dtype=np.float64).reshape(-1, len(self.names))
        true = np.asarray(true, dtype=np.float64).reshape(-1, len(self.names))
        finite = np.isfinite(pred) & np.isfinite(true)
        err = np.where(finite, pred - true, 0.0)
        count = finite.sum(axis=0)
        self.count += count
        self.pred_sum += np.where(finite, pred, 0.0).sum(axis=0)
        self.pred_sq_sum += np.square(np.where(finite, pred, 0.0)).sum(axis=0)
        self.true_sum += np.where(finite, true, 0.0).sum(axis=0)
        self.true_sq_sum += np.square(np.where(finite, true, 0.0)).sum(axis=0)
        self.abs_error += np.abs(err).sum(axis=0)
        self.sq_error += np.square(err).sum(axis=0)
        self.sum_error += err.sum(axis=0)
        self.max_abs_pred = np.maximum(self.max_abs_pred, np.where(finite, np.abs(pred), 0.0).max(axis=0))
        self.max_abs_error = np.maximum(self.max_abs_error, np.where(finite, np.abs(err), 0.0).max(axis=0))

        steady = finite & (np.abs(true) / self.rate_scales <= self.steady_rate_threshold)
        self.steady_count += steady.sum(axis=0)
        self.steady_abs_error += np.where(steady, np.abs(err), 0.0).sum(axis=0)
        self.steady_sq_error += np.where(steady, np.square(err), 0.0).sum(axis=0)
        self.steady_sum_error += np.where(steady, err, 0.0).sum(axis=0)

        if previous_pred is None or previous_true is None:
            return
        previous_pred = np.asarray(previous_pred, dtype=np.float64).reshape(-1, len(self.names))
        previous_true = np.asarray(previous_true, dtype=np.float64).reshape(-1, len(self.names))
        pair_finite = finite & np.isfinite(previous_pred) & np.isfinite(previous_true)
        pred_jump = np.where(pair_finite, pred - previous_pred, 0.0)
        true_jump = np.where(pair_finite, true - previous_true, 0.0)
        residual_jump = pred_jump - true_jump
        self.jump_count += pair_finite.sum(axis=0)
        self.max_pred_jump = np.maximum(self.max_pred_jump, np.abs(pred_jump).max(axis=0))
        self.pred_jump_sq += np.square(pred_jump).sum(axis=0)
        self.true_jump_sq += np.square(true_jump).sum(axis=0)
        self.residual_jump_sq += np.square(residual_jump).sum(axis=0)

        if (
            state_pred is None or state_true is None
            or previous_state_pred is None or previous_state_true is None
        ):
            return
        state_pred = np.asarray(state_pred, dtype=np.float64).reshape(-1, len(self.names))
        state_true = np.asarray(state_true, dtype=np.float64).reshape(-1, len(self.names))
        previous_state_pred = np.asarray(previous_state_pred, dtype=np.float64).reshape(-1, len(self.names))
        previous_state_true = np.asarray(previous_state_true, dtype=np.float64).reshape(-1, len(self.names))
        state_finite = (
            np.isfinite(state_pred) & np.isfinite(state_true)
            & np.isfinite(previous_state_pred) & np.isfinite(previous_state_true)
        )
        pred_state_jump = np.where(state_finite, state_pred - previous_state_pred, 0.0)
        true_state_jump = np.where(state_finite, state_true - previous_state_true, 0.0)
        residual_state_jump = pred_state_jump - true_state_jump
        self.state_jump_count += state_finite.sum(axis=0)
        self.max_state_jump = np.maximum(self.max_state_jump, np.abs(pred_state_jump).max(axis=0))
        self.state_jump_sq += np.square(pred_state_jump).sum(axis=0)
        self.true_state_jump_sq += np.square(true_state_jump).sum(axis=0)
        self.residual_state_jump_sq += np.square(residual_state_jump).sum(axis=0)

    def as_dict(self) -> dict:
        per_species = {}
        for i, name in enumerate(self.names):
            n = int(self.count[i])
            steady_n = int(self.steady_count[i])
            jumps = int(self.jump_count[i])
            pred_mean = self.pred_sum[i] / n if n else None
            pred_var = max(0.0, self.pred_sq_sum[i] / n - (pred_mean or 0.0) ** 2) if n else None
            true_mean = self.true_sum[i] / n if n else None
            true_var = max(0.0, self.true_sq_sum[i] / n - (true_mean or 0.0) ** 2) if n else None
            per_species[name] = {
                "n": n,
                "pred_mean": float(pred_mean) if pred_mean is not None else None,
                "pred_std": float(math.sqrt(pred_var)) if pred_var is not None else None,
                "true_mean": float(true_mean) if true_mean is not None else None,
                "true_std": float(math.sqrt(true_var)) if true_var is not None else None,
                "mae": float(self.abs_error[i] / n) if n else None,
                "rmse": float(math.sqrt(self.sq_error[i] / n)) if n else None,
                "bias": float(self.sum_error[i] / n) if n else None,
                "max_abs_pred": float(self.max_abs_pred[i]) if n else None,
                "max_abs_error": float(self.max_abs_error[i]) if n else None,
                "steady_n": steady_n,
                "steady_mae": float(self.steady_abs_error[i] / steady_n) if steady_n else None,
                "steady_rmse": float(math.sqrt(self.steady_sq_error[i] / steady_n)) if steady_n else None,
                "steady_bias": float(self.steady_sum_error[i] / steady_n) if steady_n else None,
                "jump_n": jumps,
                "max_pred_jump": float(self.max_pred_jump[i]) if jumps else None,
                "pred_jump_rmse": float(math.sqrt(self.pred_jump_sq[i] / jumps)) if jumps else None,
                "true_jump_rmse": float(math.sqrt(self.true_jump_sq[i] / jumps)) if jumps else None,
                "residual_jump_rmse": float(math.sqrt(self.residual_jump_sq[i] / jumps)) if jumps else None,
                "state_jump_n": int(self.state_jump_count[i]),
                "max_state_jump": float(self.max_state_jump[i]) if self.state_jump_count[i] else None,
                "state_jump_rmse": float(math.sqrt(self.state_jump_sq[i] / self.state_jump_count[i])) if self.state_jump_count[i] else None,
                "true_state_jump_rmse": float(math.sqrt(self.true_state_jump_sq[i] / self.state_jump_count[i])) if self.state_jump_count[i] else None,
                "residual_state_jump_rmse": float(math.sqrt(self.residual_state_jump_sq[i] / self.state_jump_count[i])) if self.state_jump_count[i] else None,
            }
        return {
            "steady_rate_threshold": self.steady_rate_threshold,
            "rate_scales": self.rate_scales.tolist(),
            "per_species": per_species,
        }


class TrajectoryAccumulator:
    """Trajectory-ordered residual statistics for one rollout horizon."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        n = len(names)
        self.error = MetricAccumulator(names)
        self.sign_flip_count = np.zeros(n, dtype=np.int64)
        self.pair_count = np.zeros(n, dtype=np.int64)
        self.prev_sum = np.zeros(n, dtype=np.float64)
        self.curr_sum = np.zeros(n, dtype=np.float64)
        self.prev_sq_sum = np.zeros(n, dtype=np.float64)
        self.curr_sq_sum = np.zeros(n, dtype=np.float64)
        self.product_sum = np.zeros(n, dtype=np.float64)

    def update(self, pred, true, previous_error=None) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if pred.ndim == 1:
            pred = pred[:, None]
            true = true[:, None]
        error = pred - true
        self.error.update(pred, true)
        if previous_error is None:
            return
        previous_error = np.asarray(previous_error, dtype=np.float64)
        finite = np.isfinite(previous_error) & np.isfinite(error)
        self.sign_flip_count += (
            finite & (previous_error != 0.0) & (error != 0.0)
            & ((previous_error * error) < 0.0)
        ).sum(axis=0)
        self.pair_count += finite.sum(axis=0)
        prev = np.where(finite, previous_error, 0.0)
        curr = np.where(finite, error, 0.0)
        self.prev_sum += prev.sum(axis=0)
        self.curr_sum += curr.sum(axis=0)
        self.prev_sq_sum += np.square(prev).sum(axis=0)
        self.curr_sq_sum += np.square(curr).sum(axis=0)
        self.product_sum += (prev * curr).sum(axis=0)

    def as_dict(self, baseline_rmse: dict | None = None) -> dict:
        metrics = self.error.as_dict()
        per_species = metrics["per_species"]
        for i, name in enumerate(self.names):
            n = int(self.pair_count[i])
            denom = n * self.prev_sq_sum[i] - self.prev_sum[i] ** 2
            denom_curr = n * self.curr_sq_sum[i] - self.curr_sum[i] ** 2
            covariance = n * self.product_sum[i] - self.prev_sum[i] * self.curr_sum[i]
            corr = (
                float(covariance / math.sqrt(max(denom * denom_curr, 1e-30)))
                if n > 1 and denom > 1e-30 and denom_curr > 1e-30 else None
            )
            entry = per_species[name]
            entry["sign_flip_rate"] = (
                float(self.sign_flip_count[i] / n) if n else None
            )
            entry["sign_flip_count"] = int(self.sign_flip_count[i])
            entry["lag1_corr"] = corr
            if baseline_rmse:
                baseline = baseline_rmse.get(name)
                entry["error_growth_ratio"] = (
                    float(entry["rmse"] / baseline)
                    if baseline is not None and baseline > 0 and entry["rmse"] is not None
                    else None
                )
        return metrics


def state_from_scaled(tools, x_scaled: torch.Tensor):
    x = tools.inverse_state(x_scaled)
    macro = x[:, tools.macro_idx]
    micro_log = x[:, tools.micro_idx]
    micro = torch.pow(10.0, torch.clamp(micro_log, tr.MICRO_LOG_MIN, tr.MICRO_LOG_MAX))
    return macro, micro, micro_log


def validation_true_rate(tools, target_scaled, current_x_scaled, next_x_scaled):
    """Return the physical rate represented by one validation transition.

    Direct-rate targets already contain rates and are decoded through the
    model's inverse transform. QSSA targets contain ``P,D`` coefficients, so
    the measured rate comes from the two endpoint states of this transition.
    """
    if getattr(tools, "rate_decode_mode", "direct_rate") not in ("qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"):
        return tools.inverse_rate(target_scaled, bound_macro=False)

    current = tools.inverse_state(current_x_scaled)
    endpoint = tools.inverse_state(next_x_scaled)
    dt = tools.get_current_dt(current_x_scaled).reshape(-1, 1).clamp_min(1e-12)
    macro_rate = (endpoint[:, tools.macro_idx] - current[:, tools.macro_idx]) / dt
    micro_log_rate = (endpoint[:, tools.micro_idx] - current[:, tools.micro_idx]) / dt
    return torch.cat([macro_rate, micro_log_rate], dim=1)


def species_names(feature_info: dict) -> List[str]:
    return list(feature_info["macro_species"]) + list(feature_info["micro_species"])


def select_trajectory_indices(
    temperatures,
    pressures,
    activities,
    count: int,
    selection: str = "combustion",
    min_temperature: float = 800.0,
    max_temperature: float = 3000.0,
):
    """Choose representative windows, preferring normal combustion conditions.

    Activity is the mean absolute physical macro rate at the first transition.
    The helper is independent of torch/data loading so it can be regression-tested.
    """
    temperatures = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    pressures = np.asarray(pressures, dtype=np.float64).reshape(-1)
    activities = np.asarray(activities, dtype=np.float64).reshape(-1)
    if not (len(temperatures) == len(pressures) == len(activities)):
        raise ValueError("temperature, pressure and activity lengths must match")
    if count <= 0 or len(temperatures) == 0:
        return [], []
    finite = np.isfinite(temperatures) & np.isfinite(pressures) & np.isfinite(activities)
    combustion = finite & (temperatures >= min_temperature) & (temperatures <= max_temperature)
    if selection == "first":
        candidates = np.flatnonzero(finite)
    elif selection == "combustion":
        candidates = np.flatnonzero(combustion)
        if len(candidates) == 0:
            candidates = np.flatnonzero(finite)
    elif selection == "diverse":
        candidates = np.flatnonzero(finite)
    else:
        raise ValueError(f"Unknown trajectory selection: {selection}")

    metadata = [
        {
            "temperature": float(temperatures[i]),
            "pressure": float(pressures[i]),
            "activity": float(activities[i]),
        }
        for i in range(len(temperatures))
    ]
    if selection == "first":
        chosen = candidates[:count].tolist()
    else:
        # Sort by activity, then interleave temperature bins to retain multiple
        # operating conditions instead of taking one nearly identical cluster.
        pool = list(candidates)
        if selection == "combustion":
            pool.sort(key=lambda i: (-activities[i], abs(temperatures[i] - 1500.0), i))
        else:
            pool.sort(key=lambda i: (temperatures[i], pressures[i], -activities[i], i))
        chosen = []
        used_bins = set()
        for i in pool:
            temp_bin = int(np.floor(temperatures[i] / 200.0))
            pressure_bin = int(np.floor(pressures[i] / 10000.0))
            key = (temp_bin, pressure_bin)
            if key not in used_bins or len(chosen) >= len(pool) - count:
                chosen.append(int(i))
                used_bins.add(key)
            if len(chosen) >= count:
                break
        if len(chosen) < count:
            chosen.extend(int(i) for i in pool if int(i) not in chosen[:])
            chosen = chosen[:count]
    return chosen, metadata


def update_plot_samples(store: dict, mode: str, horizon: int, true, pred, limit: int):
    key = (mode, horizon)
    if key not in store:
        store[key] = {"true": [], "pred": []}
    room = limit - sum(len(x) for x in store[key]["true"])
    if room <= 0:
        return
    take = min(room, len(true))
    store[key]["true"].append(true[:take].copy())
    store[key]["pred"].append(pred[:take].copy())


def select_trajectory_species(
    names: List[str],
    trajectory_samples: List[dict],
    requested: List[str] | None,
    limit: int | None = None,
) -> List[str]:
    """Select species for readable concentration-vs-time trajectory plots."""
    if requested:
        unknown = [name for name in requested if name not in names]
        if unknown:
            raise ValueError(f"Unknown trajectory species: {unknown}")
        return list(dict.fromkeys(requested))
    if limit is None:
        return list(names)
    if not trajectory_samples:
        return list(names[:limit])
    true_values = np.concatenate(
        [sample["true"] for sample in trajectory_samples], axis=0
    )
    score = np.nanmax(true_values, axis=0) - np.nanmin(true_values, axis=0)
    order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
    return [names[i] for i in order[: min(limit, len(names))]]


def _log_safe_values(values):
    """Keep non-positive values out of a log axis without changing the data."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def save_concentration_trajectory_plots(
    trajectory_samples: List[dict],
    names: List[str],
    output_dir: Path,
    requested_species: List[str] | None = None,
) -> None:
    """Plot actual, oracle and autoregressive concentrations on same trajectories."""
    if plt is None or not trajectory_samples:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = select_trajectory_species(
        names, trajectory_samples, requested_species, limit=None
    )
    indices = [names.index(name) for name in selected]
    rollout_modes = [mode for mode in ("oracle", "pure_ar", "pure_ar_raw") if mode in trajectory_samples[0]]
    colors = {"true": "black", "oracle": "tab:blue", "pure_ar": "tab:orange", "pure_ar_raw": "tab:red"}
    labels = {
        "true": "actual",
        "oracle": "single-step (oracle)",
        "pure_ar": "autoregressive",
        "pure_ar_raw": "autoregressive (raw)",
    }
    for trajectory_index, sample in enumerate(trajectory_samples):
        ncols = 4
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(18, max(2.6 * nrows, 8.0)),
            squeeze=False,
        )
        steps = np.arange(sample["true"].shape[0])
        for row, (name, column) in enumerate(zip(selected, indices)):
            ax = axes[row // ncols, row % ncols]
            series = {
                mode: np.asarray(sample[mode][:, column], dtype=np.float64)
                for mode in ("true", *rollout_modes)
            }
            positive_values = np.concatenate(
                [values[np.isfinite(values) & (values > 0.0)] for values in series.values()]
            )
            floor = (
                max(float(np.min(positive_values)) * 0.5, 1e-30)
                if positive_values.size
                else 1e-30
            )
            for mode in ("true", *rollout_modes):
                ax.plot(
                    steps,
                    _log_safe_values(series[mode]),
                    color=colors[mode],
                    lw=1.35 if mode == "true" else 1.05,
                    label=labels[mode],
                )
                if mode.endswith("raw"):
                    raw_nonpositive = np.isfinite(series[mode]) & (series[mode] <= 0.0)
                    if raw_nonpositive.any():
                        ax.scatter(
                            steps[raw_nonpositive],
                            np.full(int(raw_nonpositive.sum()), floor),
                            color=colors[mode],
                            marker="x",
                            s=24,
                            label=f"{labels[mode]} <= 0",
                            zorder=4,
                        )
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor)
            ax.set_ylabel(name)
            ax.grid(True, which="both", alpha=0.25)
            if row == 0:
                ax.legend(loc="best", ncol=3)
        for unused in range(len(selected), nrows * ncols):
            axes[unused // ncols, unused % ncols].axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel("time step")
        fig.suptitle(
            f"Concentration trajectory comparison (window {trajectory_index + 1})"
        )
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"concentration_trajectory_{trajectory_index + 1:03d}.png",
            dpi=180,
        )
        plt.close(fig)


def write_trajectory_states_csv(
    trajectory_samples: List[dict], names: List[str], path: Path
) -> None:
    """Write selected continuous trajectories with raw/clamped/true states."""
    rows = []
    for trajectory_id, sample in enumerate(trajectory_samples, start=1):
        n_steps = sample["true"].shape[0]
        for step in range(n_steps):
            for species_index, species in enumerate(names):
                row = {
                    "trajectory_id": trajectory_id,
                    "step": step,
                    "species": species,
                    "true": float(sample["true"][step, species_index]),
                }
                for key in (
                    "oracle_raw", "oracle_clamped", "oracle",
                    "pure_ar_raw", "pure_ar_raw_clamped", "pure_ar",
                ):
                    if key in sample:
                        row[key] = float(sample[key][step, species_index])
                rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_phase2_summary_csv(result: dict, path: Path) -> None:
    """Write the compact epoch/mode/species comparison requested for phase 2."""
    rows = []
    trajectory_diagnostic = result.get("trajectory_diagnostic", {}).get("by_horizon", {})
    for mode, by_horizon in trajectory_diagnostic.items():
        summary = by_horizon.get(str(max(result.get("horizons", [1]))), {})
        for species, species_summary in summary.get("per_species", {}).items():
            row = {
                "epoch": result.get("epoch"),
                "mode": mode,
                "species": species,
                "sign_flip_rate": species_summary.get("sign_flip_rate"),
                "lag1_corr": species_summary.get("lag1_corr"),
            }
            for horizon in result.get("horizons", []):
                metrics = by_horizon.get(str(horizon), {})
                species_metrics = metrics.get("per_species", {}).get(species, {})
                row[f"H{horizon}_RMSE"] = species_metrics.get("rmse")
                row[f"H{horizon}_MAE"] = species_metrics.get("mae")
                row[f"H{horizon}_growth"] = (
                    float(species_metrics["rmse"] / by_horizon[str(result["horizons"][0])]["per_species"][species]["rmse"])
                    if species_metrics.get("rmse") is not None
                    and by_horizon[str(result["horizons"][0])]["per_species"][species].get("rmse")
                    else None
                )
            if result.get("diagnostic"):
                overall = result["diagnostic"]["overall"].get(mode, {})
                row["raw_negative_rate"] = overall.get("negative_value_ratio")
            rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_macro_for_update(pred_state: dict, update_constraint: str):
    """Return the macro state represented by the selected update policy."""
    if update_constraint == "raw":
        return pred_state.get(
            "Y_macro_next_unconstrained", pred_state["Y_macro_next_raw"]
        )
    if update_constraint == "constrained":
        return pred_state["Y_macro_next"]
    raise ValueError(f"Unsupported update_constraint: {update_constraint!r}")


def select_macro_for_feedback(raw_macro, applied_macro, feedback_constraint: str):
    """Return the macro state that would be fed into the next AR step."""
    if feedback_constraint == "raw":
        return raw_macro
    if feedback_constraint == "constrained":
        return applied_macro
    raise ValueError(f"Unsupported feedback_constraint: {feedback_constraint!r}")


def feedback_input_limit_for_mode(mode: str) -> float | None:
    """Return the feedback standardization limit used by each rollout mode."""
    return tr.RAW_FEEDBACK_INPUT_LIMIT if mode == "pure_ar_raw" else None


def macro_rate_from_prediction(tools, current_x_scaled, pred_state: dict, pred_macro):
    """Return the rate represented by the state selected for reporting.

    QSSA returns both raw and clamped state candidates.  Reusing a generic
    ``rate_macro`` field here can silently report the clamped branch even when
    the validator selected ``pure_ar_raw``.  Deriving the rate from the
    selected endpoint keeps the diagnostic semantics aligned with the rollout
    mode.
    """
    current = pred_state.get("Y_macro_old")
    if current is None:
        current = tools.inverse_state(current_x_scaled)[:, tools.macro_idx]
    dt = pred_state.get("dt")
    if dt is None:
        dt = tools.get_current_dt(current_x_scaled)
    dt = dt.reshape(-1, 1).clamp_min(1e-12)
    return (pred_macro - current) / dt


def micro_log_rate_from_prediction(tools, current_x_scaled, pred_state: dict, pred_micro_log):
    """Return the log-space rate represented by the selected micro state."""
    current = pred_state.get("logY_micro_old")
    if current is None:
        current = tools.inverse_state(current_x_scaled)[:, tools.micro_idx]
    dt = pred_state.get("dt")
    if dt is None:
        dt = tools.get_current_dt(current_x_scaled)
    dt = dt.reshape(-1, 1).clamp_min(1e-12)
    return (pred_micro_log - current) / dt


def load_physical_element_matrix(data_dir: Path, mechanism: str, names: list[str]):
    """Resolve the actual Cantera mechanism, including inert species."""
    import cantera as ct
    import chemistry_contract as cc

    metadata = tr.load_metadata(str(data_dir), mechanism)
    path = tr.resolve_temperature_mechanism_path(metadata, data_dir, mechanism)
    gas = ct.Solution(str(path))
    missing = sorted(set(names) - set(gas.species_names))
    if missing:
        raise ValueError(f"Physical metric mechanism is missing species: {missing}")
    elements, matrix = cc.compute_element_mass_matrix(gas, names, preferred_elements=None)
    return matrix, elements, {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "elements": elements, "species": names,
    }



def _build_checkpoint_model(checkpoint, feature_info, device=None):
    model_config = checkpoint.get("training_config") or {}
    state_dict = checkpoint["model_state_dict"]
    model_arch = model_config.get("network_arch", "mlp")
    state_keys = set(state_dict)
    is_dual_branch = (
        model_arch == "dual_branch"
        or "P_null" in state_keys
        or {"slow_head.2.weight", "fast_head.2.weight"}.issubset(state_keys)
    )
    output_dim = len(feature_info["output_cols"])
    input_dim = len(feature_info["input_cols"])
    output_parameterization = model_config.get("output_parameterization", "direct_rate")
    device = tr.DEVICE if device is None else device

    if is_dual_branch:
        from dual_branch_network import DualBranchKineticsNet, load_dual_branch_state_dict

        model = DualBranchKineticsNet(
            input_dim=input_dim,
            slow_dim=len(feature_info["macro_species"]),
            fast_dim=len(feature_info["micro_species"]),
            hidden_dim=model_config.get("hidden_dim", 512),
            num_blocks=model_config.get("num_blocks", 4),
            output_parameterization=output_parameterization,
            use_null_space=False,
        ).to(device)
        load_dual_branch_state_dict(model, state_dict)
    elif model_arch == "resmlp":
        from dual_branch_network import CoupledKineticsResMLP

        model = CoupledKineticsResMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=model_config.get("hidden_dim", 512),
            num_blocks=model_config.get("num_blocks", 4),
            output_parameterization=output_parameterization,
        ).to(device)
        model.load_state_dict(state_dict)
    else:
        model = tr.CoupledKineticsNet(
            input_dim,
            output_dim,
            output_parameterization=output_parameterization,
        ).to(device)
        model.load_state_dict(state_dict)
    return model


def evaluate(args, checkpoint, feature_info, scaler_x, scaler_y, parts, output_dir: Path, data_dir: Path):
    tr.seed_everything(args.seed)
    model_config = checkpoint.get("training_config") or {}
    rate_decode_mode = model_config.get(
        "rate_decode_mode", feature_info.get("rate_decode_mode", "direct_rate")
    )
    rate_z_limit = float(
        model_config.get(
            "rate_z_limit", feature_info.get("rate_z_limit", tr.DEFAULT_RATE_Z_LIMIT)
        )
    )
    tools = tr.CoarseOperatorTools(
        feature_info,
        scaler_x,
        scaler_y,
        rate_decode_mode=rate_decode_mode,
        rate_z_limit=rate_z_limit,
    )
    model = _build_checkpoint_model(checkpoint, feature_info)
    model.eval()

    saved_rollout = checkpoint_rollout_config(checkpoint)
    selected_experiment_mode = args.experiment_mode
    if selected_experiment_mode == "baseline" and saved_rollout["experiment_mode"] != "baseline":
        selected_experiment_mode = saved_rollout["experiment_mode"]
    selected_state_update_mode = getattr(args, "state_update_mode", "legacy")
    if selected_state_update_mode == "legacy" and saved_rollout.get("state_update_mode", "legacy") != "legacy":
        selected_state_update_mode = saved_rollout.get("state_update_mode", "legacy")
    trust_region_fraction = saved_rollout["trust_region_fraction"]
    temperature_mode = saved_rollout["temperature_mode"]
    max_delta_temperature = saved_rollout["max_delta_temperature"]
    thermo_closure = None
    if temperature_mode == "autonomous":
        metadata = tr.load_metadata(str(data_dir), args.mechanism)
        try:
            thermo_closure = build_temperature_closure_for_validation(
                metadata, feature_info, data_dir, args.mechanism
            )
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint requests autonomous temperature, but the validation "
                f"thermodynamic closure could not be loaded: {exc}"
            ) from exc
        print(
            "[validate] autonomous temperature enabled; "
            f"thermo_closure={thermo_closure.get('energy_kind', 'unknown')}"
            f"; trust_region={trust_region_fraction:.3f}",
            flush=True,
        )
    else:
        print(
            f"[validate] temperature_mode=recorded; trust_region={trust_region_fraction:.3f}",
            flush=True,
        )

    max_horizon = args.max_horizon or max(args.horizons)
    horizons = sorted({h for h in args.horizons if h > 0 and h <= max_horizon})
    if not horizons:
        raise ValueError("No valid horizons")
    split_ids = None
    if args.split_dir and args.split != "all":
        split_name = "validation" if args.split == "val" else args.split
        split_path = Path(args.split_dir) / f"{args.mechanism}_{split_name}_sample_ids.csv"
        split_ids = pd.read_csv(split_path)["sample_id"].astype(str).tolist()
        print(f"[validate] active split manifest: {split_path.name} (filtered to {len(split_ids):,} trajectories)", flush=True)
    dataset = tr.CoarseOperatorDataset(
        parts, feature_info, scaler_x, scaler_y,
        max_horizon=max_horizon,
        shuffle_parts=False,
        stride=args.stride if args.stride is not None else max(1, max_horizon // 4),
        seed=args.seed,
        parts_per_epoch=len(parts),
        sample_ids=split_ids,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    names = species_names(feature_info)
    macro_names = list(feature_info["macro_species"])
    micro_names = list(feature_info["micro_species"])
    rollout_modes = ["oracle", "pure_ar"]
    # Keep update and feedback constraints independent for ablation studies.
    update_constraint = args.update_constraint
    if update_constraint is None:
        update_constraint = (
            "constrained"
            if selected_experiment_mode in ("constrained_delta", "conserve_project", "qssa")
            else "raw"
        )
    feedback_constraint = args.feedback_constraint
    if feedback_constraint is None:
        feedback_constraint = (
            "constrained" if selected_experiment_mode != "baseline" else "raw"
        )
    if args.diagnostic:
        rollout_modes.append("pure_ar_raw")
    selected_experiment_mode = (
        selected_experiment_mode
        if selected_experiment_mode in ("conserve_project", "qssa")
        else "constrained_delta" if update_constraint == "constrained" else "baseline"
    )
    rollout_experiment_modes = {
        "oracle": selected_experiment_mode,
        "pure_ar": selected_experiment_mode,
        # Diagnostic path remains genuinely raw at both boundaries.
        "pure_ar_raw": "baseline",
    }
    if physical_metrics is None or not hasattr(physical_metrics, "PhysicalMetricAccumulator"):
        raise RuntimeError("Full Task5 requires the matching tools/compute_physical_metrics.py")
    physical_names = list(feature_info["inert_species"]) + names
    element_matrix, element_names, element_metadata = load_physical_element_matrix(
        data_dir, args.mechanism, physical_names
    )
    inert_set = list(feature_info.get("inert_species", []))
    physical_stats = {
        mode: {str(h): {scope: physical_metrics.PhysicalMetricAccumulator(
            physical_names, element_matrix, element_names,
            negative_tolerance=1e-12, inert_species=inert_set
        ) for scope in ("raw", "constrained")} for h in horizons}
        for mode in rollout_modes
    }
    stats = {
        mode: {str(h): {"state": MetricAccumulator(names),
                        "log_state_micro": MetricAccumulator(micro_names),
                        "macro_rate": MetricAccumulator(macro_names),
                        "micro_log_rate": MetricAccumulator(micro_names)}
               for h in horizons}
        for mode in rollout_modes
    }
    diagnostic = {
        mode: {str(h): DiagnosticAccumulator(macro_names) for h in horizons}
        for mode in ("oracle", "pure_ar", "pure_ar_raw")
    } if args.diagnostic else {}
    constraint_intervention = {
        mode: {str(h): ConstraintInterventionAccumulator(macro_names) for h in horizons}
        for mode in rollout_modes
    }
    trajectory_stats = {
        mode: {str(h): TrajectoryAccumulator(names) for h in range(1, max_horizon + 1)}
        for mode in rollout_modes
    }
    rate_diagnostic = {
        mode: {
            str(h): RateDiagnosticAccumulator(
                macro_names,
                feature_info["macro_rate_scale"],
                args.steady_rate_threshold,
            )
            for h in horizons
        }
        for mode in rollout_modes
    }
    plot_samples = {}
    trajectory_samples = []
    trajectory_conditions = []
    selected_condition_keys = set()
    batches = 0
    windows = 0
    negative_macro = {mode: {str(h): 0 for h in horizons} for mode in stats}

    try:
        loader_length = len(loader)
    except TypeError:
        # IterableDataset may not expose a length; tqdm can still count batches.
        loader_length = None
    total_batches = (
        min(loader_length, args.max_batches)
        if loader_length is not None and args.max_batches is not None
        else args.max_batches if args.max_batches is not None
        else loader_length
    )
    pbar = tqdm(loader, total=total_batches, desc="[validate]", dynamic_ncols=True)

    with torch.no_grad():
        for batch_x, batch_y in pbar:
            if args.max_batches is not None and batches >= args.max_batches:
                break
            batch_x = batch_x.to(tr.DEVICE, non_blocking=True)
            batch_y = batch_y.to(tr.DEVICE, non_blocking=True)
            bsz, steps = batch_y.shape[:2]
            steps = min(steps, max_horizon)
            windows += bsz
            current = {mode: batch_x[:, 0, :] for mode in rollout_modes}
            previous_errors = {mode: None for mode in rollout_modes}
            previous_rates = {mode: None for mode in rollout_modes}
            previous_true_rates = {mode: None for mode in rollout_modes}
            previous_states = {mode: None for mode in rollout_modes}
            previous_true_states = {mode: None for mode in rollout_modes}
            current_physical = tools.inverse_state(batch_x[:, 0, :])
            temperature_idx = feature_info["input_cols"].index("T")
            pressure_idx = feature_info["input_cols"].index("P")
            first_true_rate = validation_true_rate(
                tools,
                batch_y[:, 0, :],
                batch_x[:, 0, :],
                batch_x[:, 1, :],
            )[:, :len(macro_names)]
            initial_activity = torch.mean(torch.abs(first_true_rate), dim=1)
            candidate_indices, candidate_metadata = select_trajectory_indices(
                current_physical[:, temperature_idx].detach().cpu().numpy(),
                current_physical[:, pressure_idx].detach().cpu().numpy(),
                initial_activity.detach().cpu().numpy(),
                count=max(0, args.plot_trajectories - len(trajectory_samples)),
                selection=args.trajectory_selection,
                min_temperature=args.combustion_min_temperature,
                max_temperature=args.combustion_max_temperature,
            )
            selected_indices = []
            selected_metadata = []
            for index in candidate_indices:
                meta = candidate_metadata[index]
                condition_key = (
                    int(np.floor(meta["temperature"] / 200.0)),
                    int(np.floor(meta["pressure"] / 10000.0)),
                )
                if args.trajectory_selection != "first" and condition_key in selected_condition_keys:
                    continue
                selected_condition_keys.add(condition_key)
                selected_indices.append(index)
                selected_metadata.append(meta)
            trajectory_count = len(selected_indices)
            trajectory_records = []
            if trajectory_count:
                initial_macro, initial_micro, _ = state_from_scaled(
                    tools, batch_x[selected_indices, 0, :]
                )
                initial_species = torch.cat(
                    [initial_macro, initial_micro], dim=1
                ).cpu().numpy()
                for sample_index in range(trajectory_count):
                    initial = initial_species[sample_index]
                    trajectory_records.append(
                        {
                            "true": [initial],
                            **{mode: [initial] for mode in rollout_modes},
                            "oracle_raw": [initial],
                            "oracle_clamped": [initial],
                            "pure_ar_raw_clamped": [initial],
                            "_metadata": selected_metadata[sample_index],
                        }
                    )
            for step in range(steps):
                endpoint = step + 1
                true_x = batch_x[:, endpoint, :]
                true_macro, true_micro, true_micro_log = state_from_scaled(tools, true_x)
                true_rate = validation_true_rate(
                    tools,
                    batch_y[:, step, :],
                    batch_x[:, step, :],
                    true_x,
                )
                predicted_species = {}
                true_species = torch.cat([true_macro, true_micro], dim=1)
                for mode in rollout_modes:
                    pred_rate_scaled = model(current[mode])
                    pred_state = tools.predict_next_state(
                        current[mode], pred_rate_scaled,
                        safe_for_rollout=(mode != "pure_ar_raw"),
                        experiment_mode=rollout_experiment_modes[mode],
                        trust_region_fraction=(
                            trust_region_fraction if mode != "pure_ar_raw" else 0.0
                        ),
                        state_update_mode=(
                            selected_state_update_mode
                            if mode != "pure_ar_raw" or selected_state_update_mode in ("qssa", "qssa_asymptotic")
                            else "legacy"
                        ),
                    )
                    # Update and feedback are separate ablation axes. The
                    # reported state always follows the selected update rule.
                    pred_macro = select_macro_for_update(
                        pred_state,
                        "raw" if mode == "pure_ar_raw" else update_constraint,
                    )
                    if mode == "pure_ar_raw":
                        pred_micro = pred_state.get(
                            "Y_micro_next_unconstrained", pred_state["Y_micro_next"]
                        )
                        pred_micro_log = pred_state.get(
                            "logY_micro_next_unconstrained", pred_state["logY_micro_next"]
                        )
                    else:
                        pred_micro = pred_state["Y_micro_next"]
                        pred_micro_log = pred_state["logY_micro_next"]
                    pred_species = torch.cat([pred_macro, pred_micro], dim=1)
                    predicted_species[mode] = pred_species
                    raw_macro = pred_state.get(
                        "Y_macro_next_unconstrained", pred_state["Y_macro_next_raw"]
                    )
                    projected_macro = None
                    if pred_state.get("Y_next_projected_full") is not None:
                        projected_macro = pred_state["Y_next_projected_full"][:, :len(macro_names)]
                    raw_species = torch.cat(
                        [raw_macro, pred_micro], dim=1
                    )
                    clamped_species = torch.cat(
                        [torch.clamp(raw_macro, min=0.0), pred_micro],
                        dim=1,
                    )
                    if args.diagnostic and mode == "oracle":
                        predicted_species["oracle_raw"] = raw_species
                        predicted_species["oracle_clamped"] = clamped_species
                    if args.diagnostic and mode == "pure_ar":
                        predicted_species["pure_ar_raw_clamped"] = clamped_species
                    if endpoint in horizons:
                        key = str(endpoint)
                        # Inert species are carried from the current input; include
                        # them so absolute mass normalization is meaningful.
                        true_inert = tools.inverse_state(true_x)[:, tools.inert_idx].clamp_min(0.0)
                        pred_inert = tools.inverse_state(current[mode])[:, tools.inert_idx].clamp_min(0.0)
                        full_true = torch.cat([true_inert, true_species], dim=1).cpu().numpy()
                        physical_stats[mode][key]["raw"].update(
                            full_true, torch.cat([pred_inert, raw_species], dim=1).cpu().numpy()
                        )
                        physical_stats[mode][key]["constrained"].update(
                            full_true, torch.cat([pred_inert, pred_species], dim=1).cpu().numpy()
                        )
                        # Update intervention compares the unconstrained Euler
                        # candidate with the state actually reported. Feedback
                        # intervention compares raw feedback with the state sent
                        # to the next model step.
                        applied_feedback = raw_macro
                        if mode != "pure_ar_raw" and feedback_constraint == "constrained":
                            applied_feedback = select_macro_for_feedback(
                                raw_macro,
                                pred_state["Y_macro_next"],
                                "constrained",
                            )
                        constraint_intervention[mode][key].update(
                            raw_macro.detach().cpu().numpy(),
                            pred_macro.detach().cpu().numpy(),
                            raw_macro.detach().cpu().numpy(),
                            applied_feedback.detach().cpu().numpy(),
                            projected_macro.detach().cpu().numpy()
                            if projected_macro is not None else None,
                        )
                        stats[mode][key]["state"].update(
                            pred_species.cpu().numpy(), true_species.cpu().numpy()
                        )
                        stats[mode][key]["log_state_micro"].update(
                            pred_micro_log.cpu().numpy(), true_micro_log.cpu().numpy()
                        )
                        macro_rate_val = macro_rate_from_prediction(
                            tools, current[mode], pred_state, pred_macro
                        )
                        stats[mode][key]["macro_rate"].update(
                            macro_rate_val.cpu().numpy(),
                            true_rate[:, :len(macro_names)].cpu().numpy(),
                        )
                        micro_log_rate_val = micro_log_rate_from_prediction(
                            tools, current[mode], pred_state, pred_micro_log
                        )
                        stats[mode][key]["micro_log_rate"].update(
                            micro_log_rate_val.cpu().numpy(),
                            true_rate[:, len(macro_names):].cpu().numpy(),
                        )
                        rate_diagnostic[mode][key].update(
                            macro_rate_val.cpu().numpy(),
                            true_rate[:, :len(macro_names)].cpu().numpy(),
                            previous_rates[mode],
                            previous_true_rates[mode],
                            pred_macro.cpu().numpy(),
                            true_macro.cpu().numpy(),
                            previous_states[mode],
                            previous_true_states[mode],
                        )
                        negative_macro[mode][key] += int(
                            (raw_macro < 0).sum().item()
                        )
                        if args.diagnostic:
                            diagnostic[mode][key].update(
                                raw_macro.cpu().numpy(),
                                true_macro.cpu().numpy(),
                            )
                        update_plot_samples(
                            plot_samples, mode, endpoint,
                            true_species.cpu().numpy(), pred_species.cpu().numpy(),
                            args.plot_samples,
                        )
                    if args.diagnostic:
                        trajectory_stats[mode][str(endpoint)].update(
                            pred_species.cpu().numpy(), true_species.cpu().numpy(), previous_errors[mode]
                        )
                        previous_errors[mode] = (
                            pred_species - true_species
                        ).detach().cpu().numpy()
                    previous_rates[mode] = macro_rate_from_prediction(
                        tools, current[mode], pred_state, pred_macro
                    ).detach().cpu().numpy()
                    previous_true_rates[mode] = true_rate[:, :len(macro_names)].detach().cpu().numpy()
                    previous_states[mode] = pred_macro.detach().cpu().numpy()
                    previous_true_states[mode] = true_macro.detach().cpu().numpy()
                    if mode == "oracle":
                        current[mode] = batch_x[:, endpoint, :]
                    elif endpoint < steps:
                        if mode == "pure_ar_raw":
                            macro_for_feedback = raw_macro
                        else:
                            macro_for_feedback = select_macro_for_feedback(
                                raw_macro,
                                pred_macro,
                                feedback_constraint,
                            )
                        feedback_micro_log = (
                            pred_state.get("logY_micro_next_unconstrained", pred_state["logY_micro_next"])
                            if mode == "pure_ar_raw"
                            else pred_state["logY_micro_next"]
                        )
                        current[mode] = tools.state_to_input_scaled(
                            batch_x[:, endpoint, :],
                            macro_for_feedback, feedback_micro_log,
                            temperature_mode=temperature_mode,
                            thermo_closure=thermo_closure,
                            current_X_scaled=current[mode],
                            max_delta_temperature=max_delta_temperature,
                            feedback_input_limit=feedback_input_limit_for_mode(mode),
                        )
                if trajectory_records:
                    true_values = true_species[selected_indices].cpu().numpy()
                    for sample_index, record in enumerate(trajectory_records):
                        record["true"].append(true_values[sample_index])
                        for mode in rollout_modes:
                            record[mode].append(
                                predicted_species[mode][selected_indices[sample_index]].detach().cpu().numpy()
                            )
                        if args.diagnostic:
                            record["oracle_raw"].append(
                                predicted_species["oracle_raw"][selected_indices[sample_index]].detach().cpu().numpy()
                            )
                            record["oracle_clamped"].append(
                                predicted_species["oracle_clamped"][selected_indices[sample_index]].detach().cpu().numpy()
                            )
                            record["pure_ar_raw_clamped"].append(
                                predicted_species["pure_ar_raw_clamped"][selected_indices[sample_index]].detach().cpu().numpy()
                            )
            for record in trajectory_records:
                trajectory_conditions.append(dict(record["_metadata"]))
                trajectory_samples.append(
                    {
                        key: np.stack(values, axis=0)
                        for key, values in record.items()
                        if key != "_metadata"
                    }
                )
            batches += 1
            pbar.set_postfix({"windows": windows})

    result = {
        "checkpoint": str(checkpoint.get("_path", "")),
        "epoch": checkpoint.get("epoch"),
        "mechanism": checkpoint.get("mechanism", args.mechanism),
        "device": str(tr.DEVICE),
        "split": args.split,
        "split_dir": str(Path(args.split_dir).resolve()) if args.split_dir else None,
        "split_signatures": {
            split: tr.trajectory_split_signature(args.split_dir, args.mechanism, split)
            for split in ("train", "validation", "test")
        } if args.split_dir else {},
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "rebuild_scalers": args.rebuild_scalers,
        "stride": args.stride if args.stride is not None else max(1, max_horizon // 4),
        "parts": [str(p) for p in parts],
        "batches": batches,
        "windows": windows,
        "horizons": horizons,
        "experiment_mode": selected_experiment_mode,
        "state_update_mode": selected_state_update_mode,
        "checkpoint_training_config": checkpoint.get("training_config", {}),
        "temperature_mode": temperature_mode,
        "trust_region_fraction": trust_region_fraction,
        "max_delta_temperature": max_delta_temperature,
        "update_constraint": update_constraint,
        "feedback_constraint": feedback_constraint,
        "rollout_experiment_modes": rollout_experiment_modes,
        "state_semantics": {
            "raw_macro": "unconstrained Euler candidate (Y_old + rate * dt)",
            "constrained_macro": "state after the selected nonnegative/update constraint",
            "micro_species": "always reconstructed with the configured log-space floor/ceiling",
            "oracle_feedback": (
                "hypothetical intervention statistic only; the next oracle step uses truth"
            ),
        },
        "transform_signature": (
            checkpoint.get("transform_signature")
            or (feature_info or {}).get("transform_signature")
            or tr.TRANSFORM_SIGNATURE
        ),
        "rollout_signature": getattr(tr, "ROLLOUT_SIGNATURE", None),
        "scaler_source": feature_info.get("_scaler_source"),
        "trajectory_selection": {
            "strategy": args.trajectory_selection,
            "count": len(trajectory_samples),
            "min_temperature": args.combustion_min_temperature,
            "max_temperature": args.combustion_max_temperature,
            "conditions": trajectory_conditions,
        },
        "modes": {},
    }
    for mode in stats:
        result["modes"][mode] = {}
        for h in horizons:
            key = str(h)
            result["modes"][mode][key] = {
                metric_name: metric.as_dict()
                for metric_name, metric in stats[mode][key].items()
            }
            result["modes"][mode][key]["negative_macro_predictions"] = negative_macro[mode][key]
    if args.diagnostic:
        result["diagnostic"] = {
            "overall": {
                mode: _merge_diagnostic_accumulators(diagnostic[mode].values()).as_dict()
                for mode in diagnostic
            },
            "by_horizon": {
                mode: {h: accumulator.as_dict() for h, accumulator in horizons_by_mode.items()}
                for mode, horizons_by_mode in diagnostic.items()
            },
        }
        trajectory_by_horizon = {}
        for mode, accumulators in trajectory_stats.items():
            baseline = accumulators["1"].error.as_dict()["per_species"]
            baseline_rmse = {name: item.get("rmse") for name, item in baseline.items()}
            trajectory_by_horizon[mode] = {
                horizon: accumulator.as_dict(baseline_rmse)
                for horizon, accumulator in accumulators.items()
                if int(horizon) in horizons
            }
        result["trajectory_diagnostic"] = {"by_horizon": trajectory_by_horizon}
    result["constraint_intervention"] = {
        "update_constraint": update_constraint,
        "feedback_constraint": feedback_constraint,
        "overall": {
            mode: _merge_constraint_accumulators(constraint_intervention[mode].values()).as_dict()
            for mode in constraint_intervention
        },
        "by_horizon": {
            mode: {h: accumulator.as_dict() for h, accumulator in horizons_by_mode.items()}
            for mode, horizons_by_mode in constraint_intervention.items()
        },
    }
    result["rate_diagnostic"] = {
        "steady_rate_threshold": args.steady_rate_threshold,
        "by_horizon": {
            mode: {
                horizon: accumulator.as_dict()
                for horizon, accumulator in horizons_by_mode.items()
            }
            for mode, horizons_by_mode in rate_diagnostic.items()
        },
    }
    result["physical_metrics_streaming"] = {
        mode: {h: {scope: accumulator.as_dict() for scope, accumulator in scopes.items()}
               for h, scopes in by_horizon.items()}
        for mode, by_horizon in physical_stats.items()
    }
    result["physical_element_metadata"] = element_metadata
    result["physical_metrics_protocol"] = (
        physical_metrics.summarize_validation_result(result)
        if physical_metrics is not None else
        {
            "schema_version": "physical_metrics_v68",
            "available": False,
            "reason": "physical_metrics_import_failed",
            "error": PHYSICAL_METRICS_IMPORT_ERROR,
            "modes": {},
        }
    )
    del result["physical_metrics_streaming"]
    return result, plot_samples, trajectory_samples, names


def _merge_diagnostic_accumulators(accumulators: Iterable[DiagnosticAccumulator]) -> DiagnosticAccumulator:
    """Merge horizon accumulators without retaining batch tensors."""
    accumulators = list(accumulators)
    if not accumulators:
        raise ValueError("Cannot merge an empty diagnostic accumulator sequence")
    merged = DiagnosticAccumulator(accumulators[0].names)
    for acc in accumulators:
        merged.total_predictions += acc.total_predictions
        merged.predictions_with_negative += acc.predictions_with_negative
        merged.negative_count += acc.negative_count
        merged.negative_sum += acc.negative_sum
        merged.clamp_sum += acc.clamp_sum
        merged.clamp_sq_sum += acc.clamp_sq_sum
        merged.clamp_max = np.maximum(merged.clamp_max, acc.clamp_max)
        merged.raw_min = np.minimum(merged.raw_min, acc.raw_min)
        for target, source in ((merged.raw_error, acc.raw_error), (merged.clamped_error, acc.clamped_error)):
            target.abs_error += source.abs_error
            target.sq_error += source.sq_error
            target.sum_error += source.sum_error
            target.sum_true += source.sum_true
            target.sum_true_sq += source.sum_true_sq
            target.abs_true += source.abs_true
            target.count += source.count
            target.nonfinite += source.nonfinite
    return merged


def _merge_constraint_accumulators(
    accumulators: Iterable[ConstraintInterventionAccumulator],
) -> ConstraintInterventionAccumulator:
    """Merge intervention counters across horizons."""
    accumulators = list(accumulators)
    if not accumulators:
        raise ValueError("Cannot merge an empty constraint accumulator sequence")
    merged = ConstraintInterventionAccumulator(accumulators[0].names)
    for acc in accumulators:
        merged.total_predictions += acc.total_predictions
        merged.update_rows += acc.update_rows
        merged.feedback_rows += acc.feedback_rows
        merged.projection_rows += acc.projection_rows
        for target, source in (
            (merged.update_count, acc.update_count),
            (merged.feedback_count, acc.feedback_count),
            (merged.projection_count, acc.projection_count),
        ):
            target += source
        for target, source in (
            (merged.update_sum, acc.update_sum),
            (merged.feedback_sum, acc.feedback_sum),
            (merged.projection_sum, acc.projection_sum),
            (merged.update_sq_sum, acc.update_sq_sum),
            (merged.feedback_sq_sum, acc.feedback_sq_sum),
            (merged.projection_sq_sum, acc.projection_sq_sum),
        ):
            target += source
        for target, source in (
            (merged.update_max, acc.update_max),
            (merged.feedback_max, acc.feedback_max),
            (merged.projection_max, acc.projection_max),
        ):
            target[:] = np.maximum(target, source)
    return merged


def write_csv(result: dict, path: Path) -> None:
    rows = []
    for mode, horizons in result["modes"].items():
        for horizon, values in horizons.items():
            for metric_name, metric_data in values.items():
                if metric_name.startswith("negative_"):
                    continue
                for species, values_by_metric in metric_data["per_species"].items():
                    rows.append({
                        "mode": mode, "horizon": horizon, "metric": metric_name,
                        "species": species, **values_by_metric,
                    })
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_constraint_intervention_csv(result: dict, path: Path) -> None:
    payload = result.get("constraint_intervention")
    if not payload:
        return
    rows = []
    for scope, modes in (("overall", payload.get("overall", {})),
                         ("by_horizon", payload.get("by_horizon", {}))):
        for mode, values in modes.items():
            horizon_values = {"overall": values} if scope == "overall" else values
            for horizon, summary in horizon_values.items():
                for kind in ("update", "feedback", "projection"):
                    item = summary.get(kind, {})
                    rows.append({
                        "scope": scope, "mode": mode, "horizon": horizon,
                        "intervention": kind,
                        "intervened_rows": item.get("intervened_rows", 0),
                        "intervened_row_ratio": item.get("intervened_row_ratio", 0.0),
                        "intervened_values": item.get("intervened_values", 0),
                        "intervened_value_ratio": item.get("intervened_value_ratio", 0.0),
                        "total_correction": item.get("total_correction", 0.0),
                        "mean_correction": item.get("mean_correction", 0.0),
                        "max_correction": item.get("max_correction", 0.0),
                        "correction_rmse": item.get("correction_rmse", 0.0),
                    })
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_plots(result: dict, samples: dict, names: List[str], output_dir: Path) -> None:
    if plt is None:
        print("[plot] matplotlib is unavailable; metrics were still written")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Error growth with rollout horizon, averaged over output species.
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, marker in (("oracle", "o"), ("pure_ar", "s")):
        hs, values = [], []
        for h in result["horizons"]:
            state = result["modes"][mode][str(h)]["state"]["per_species"]
            rmse = [v["rmse"] for v in state.values() if v["rmse"] is not None]
            if rmse:
                hs.append(h); values.append(float(np.mean(rmse)))
        if hs:
            ax.plot(hs, values, marker=marker, label=mode)
    ax.set(xlabel="Horizon (steps)", ylabel="Mean physical-state RMSE", title="Error growth by horizon")
    ax.set_xscale("symlog", linthresh=1); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "rmse_by_horizon.png", dpi=180); plt.close(fig)

    final_h = str(max(result["horizons"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mode, color in (("oracle", "tab:blue"), ("pure_ar", "tab:orange")):
        state = result["modes"][mode][final_h]["state"]["per_species"]
        ordered = [state[n]["rmse"] if state[n]["rmse"] is not None else np.nan for n in names]
        axes[0].plot(names, ordered, marker="o", label=mode, color=color)
    axes[0].set_title(f"Per-species RMSE at horizon {final_h}"); axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=75); axes[0].grid(True, alpha=0.25); axes[0].legend()
    for mode, color in (("oracle", "tab:blue"), ("pure_ar", "tab:orange")):
        item = samples.get((mode, int(final_h)))
        if not item: continue
        true = np.concatenate(item["true"], axis=0)
        pred = np.concatenate(item["pred"], axis=0)
        axes[1].scatter(true[:, 0], pred[:, 0], s=8, alpha=0.35, label=mode, color=color)
    if axes[1].has_data():
        lo, hi = axes[1].get_xlim(); axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].set_xlabel(f"True {names[0]}"); axes[1].set_ylabel(f"Predicted {names[0]}")
    axes[1].set_title(f"Parity sample at horizon {final_h}"); axes[1].grid(True, alpha=0.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(plot_dir / "species_rmse_and_parity.png", dpi=180); plt.close(fig)

    # A trajectory-style diagnostic from the capped endpoint sample.
    for mode in ("oracle", "pure_ar"):
        item = samples.get((mode, int(final_h)))
        if not item: continue
        true = np.concatenate(item["true"], axis=0)
        pred = np.concatenate(item["pred"], axis=0)
        nplot = min(6, true.shape[1])
        fig, axes = plt.subplots(nplot, 1, figsize=(9, max(2.2, 1.8 * nplot)), squeeze=False)
        for i in range(nplot):
            ax = axes[i, 0]
            ax.plot(true[:, i], label="true", lw=1.2)
            ax.plot(pred[:, i], label="pred", lw=1.0, alpha=0.85)
            ax.set_ylabel(names[i]); ax.grid(True, alpha=0.25)
            if i == 0: ax.legend()
        axes[-1, 0].set_xlabel("sample")
        fig.suptitle(f"{mode} endpoint state samples (horizon {final_h})")
        fig.tight_layout(); fig.savefig(plot_dir / f"endpoint_samples_{mode}.png", dpi=180); plt.close(fig)


def write_diagnostic_csv(result: dict, path: Path) -> None:
    diagnostic = result.get("diagnostic")
    if not diagnostic:
        return
    rows = []
    for scope, modes in (("overall", diagnostic["overall"]), ("by_horizon", diagnostic["by_horizon"])):
        for mode, payload in modes.items():
            items = [("overall", payload)] if scope == "overall" else payload.items()
            for horizon, values in items:
                for species, species_values in values.get("per_species", {}).items():
                    rows.append({
                        "scope": scope,
                        "horizon": horizon,
                        "mode": mode,
                        "species": species,
                        **species_values,
                        "negative_ratio_all_samples": values.get("negative_ratio"),
                        "negative_value_ratio": values.get("negative_value_ratio"),
                        "mean_clamp_magnitude_all": values.get("mean_clamp_magnitude"),
                    })
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_rate_diagnostic_csv(result: dict, path: Path) -> None:
    """Write physical rate, near-steady and temporal-jump diagnostics."""
    rows = []
    for mode, horizons in result.get("rate_diagnostic", {}).get("by_horizon", {}).items():
        for horizon, payload in horizons.items():
            for species, values in payload.get("per_species", {}).items():
                rows.append({"mode": mode, "horizon": horizon, "species": species, **values})
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_diagnostic_plots(result: dict, names: List[str], output_dir: Path) -> None:
    if plt is None or not result.get("diagnostic"):
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizons = result["horizons"]
    colors = {"oracle": "tab:blue", "pure_ar": "tab:orange", "pure_ar_raw": "tab:red"}
    labels = {"oracle": "oracle", "pure_ar": "pure AR (clamped)", "pure_ar_raw": "pure AR (raw)"}
    for metric, filename, ylabel, title in (
        ("negative_value_ratio", "negative_ratio_by_horizon.png", "negative raw-value ratio", "Raw negative values by horizon"),
        ("mean_clamp_magnitude", "clamp_magnitude_by_horizon.png", "mean clamp magnitude", "Clamp magnitude by horizon"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, values_by_h in result["diagnostic"]["by_horizon"].items():
            ys = [values_by_h[str(h)].get(metric, 0.0) for h in horizons]
            ax.plot(horizons, ys, marker="o", label=labels.get(mode, mode), color=colors.get(mode))
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlabel="horizon (steps)", ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, color in (("pure_ar", "tab:orange"), ("pure_ar_raw", "tab:red")):
        if mode not in result["diagnostic"]["by_horizon"]:
            continue
        raw_rmse, clamped_rmse = [], []
        for h in horizons:
            payload = result["diagnostic"]["by_horizon"][mode][str(h)]
            for target, output in ((raw_rmse, "raw_vs_true"), (clamped_rmse, "clamped_vs_true")):
                values = [item["rmse"] for item in payload[output]["per_species"].values() if item["rmse"] is not None]
                target.append(float(np.mean(values)) if values else np.nan)
        ax.plot(horizons, raw_rmse, marker="o", color=color, linestyle="--", label=f"{labels.get(mode, mode)} raw")
        ax.plot(horizons, clamped_rmse, marker="s", color=color, linestyle="-", label=f"{labels.get(mode, mode)} clamped")
    ax.set_xscale("symlog", linthresh=1)
    ax.set(xlabel="horizon (steps)", ylabel="mean macro RMSE", title="Raw versus clamped macro-state error")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "raw_vs_clamped_rmse_by_horizon.png", dpi=180); plt.close(fig)

    h = max(horizons)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 5))
    for mode, color in colors.items():
        if mode not in result["diagnostic"]["by_horizon"]:
            continue
        payload = result["diagnostic"]["by_horizon"][mode][str(h)]
        ys = [payload["per_species"][name]["negative_ratio"] for name in names if name in payload["per_species"]]
        ax.plot(names[:len(ys)], ys, marker="o", label=labels[mode], color=color)
    ax.set_title(f"Per-species raw negative ratio at horizon {h}")
    ax.set_ylabel("negative ratio"); ax.tick_params(axis="x", rotation=75); ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "negative_ratio_by_species.png", dpi=180); plt.close(fig)


def save_rate_diagnostic_plots(result: dict, names: List[str], output_dir: Path) -> None:
    """Plot rate error/noise diagnostics on a comparable linear scale."""
    if plt is None or not result.get("rate_diagnostic"):
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizons = result["horizons"]
    colors = {"oracle": "tab:blue", "pure_ar": "tab:orange", "pure_ar_raw": "tab:red"}
    labels = {"oracle": "oracle", "pure_ar": "pure AR (clamped)", "pure_ar_raw": "pure AR (raw)"}
    for field, filename, ylabel, title in (
        ("rmse", "rate_rmse_by_horizon.png", "macro-rate RMSE", "Physical macro-rate error"),
        ("pred_std", "rate_pred_std_by_horizon.png", "predicted-rate standard deviation", "Predicted macro-rate noise"),
        ("steady_rmse", "steady_rate_rmse_by_horizon.png", "near-steady macro-rate RMSE", "Near-steady rate error"),
        ("max_pred_jump", "rate_max_jump_by_horizon.png", "maximum adjacent predicted-rate jump", "Largest macro-rate jump"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, by_horizon in result["rate_diagnostic"]["by_horizon"].items():
            values = []
            for h in horizons:
                species_values = [
                    item.get(field) for item in by_horizon[str(h)]["per_species"].values()
                    if item.get(field) is not None
                ]
                values.append(float(np.mean(species_values)) if species_values else np.nan)
            ax.plot(horizons, values, marker="o", label=labels.get(mode, mode), color=colors.get(mode))
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlabel="horizon (steps)", ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)


def save_phase2_plots(result: dict, output_dir: Path) -> None:
    if plt is None or not result.get("trajectory_diagnostic"):
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_horizon = result["trajectory_diagnostic"]["by_horizon"]
    horizons = result["horizons"]
    colors = {"oracle": "tab:blue", "pure_ar": "tab:orange", "pure_ar_raw": "tab:red"}
    labels = {"oracle": "oracle", "pure_ar": "pure AR (clamped)", "pure_ar_raw": "pure AR (raw)"}
    for field, filename, ylabel, title in (
        ("sign_flip_rate", "trajectory_sign_flip_by_horizon.png", "mean sign-flip rate", "Adjacent residual sign flips"),
        ("lag1_corr", "trajectory_lag1_corr_by_horizon.png", "mean lag-1 correlation", "Adjacent residual correlation"),
        ("error_growth_ratio", "trajectory_error_growth_by_horizon.png", "mean RMSE(h) / RMSE(1)", "Trajectory error amplification"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, values_by_h in by_horizon.items():
            ys = []
            for h in horizons:
                species_values = [
                    item.get(field) for item in values_by_h[str(h)]["per_species"].values()
                    if item.get(field) is not None
                ]
                ys.append(float(np.mean(species_values)) if species_values else np.nan)
            ax.plot(horizons, ys, marker="o", label=labels.get(mode, mode), color=colors.get(mode))
        ax.set_xscale("symlog", linthresh=1)
        ax.set(xlabel="horizon (steps)", ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate and visualize a V68 kinetics model")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_ROOT), help="Directory containing database_Transient_V21_QSSA")
    parser.add_argument("--mechanism", default="nuig_2024")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Training artifact root")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Exact directory for this validation result; bypasses the default output layout.",
    )
    parser.add_argument("--checkpoint", default=None, help="One explicit .pth checkpoint")
    parser.add_argument(
        "--checkpoints", nargs="+", default=None,
        help="Two or more checkpoints to validate with identical data settings.",
    )
    parser.add_argument("--split", choices=("val", "train", "test", "all"), default="val")
    parser.add_argument("--split_dir", default=None, help="Trajectory split directory produced by split_trajectories_v21.py")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=tr.SEED)
    parser.add_argument("--max_horizon", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--plot_samples", type=int, default=2000)
    parser.add_argument(
        "--plot_trajectories",
        type=int,
        default=12,
        help="Number of concentration trajectories to plot.",
    )
    parser.add_argument(
        "--trajectory_selection",
        choices=("first", "diverse", "combustion"),
        default="combustion",
        help="Select plotted windows by first occurrence, diverse conditions, or combustion-like conditions.",
    )
    parser.add_argument(
        "--combustion_min_temperature",
        type=float,
        default=800.0,
        help="Minimum initial temperature (K) for combustion trajectory selection.",
    )
    parser.add_argument(
        "--combustion_max_temperature",
        type=float,
        default=3000.0,
        help="Maximum initial temperature (K) for combustion trajectory selection.",
    )
    parser.add_argument(
        "--steady_rate_threshold",
        type=float,
        default=0.05,
        help="Near-steady threshold for |true macro rate| / P90 rate scale.",
    )
    parser.add_argument(
        "--trajectory_species",
        nargs="+",
        default=None,
        help="Species to plot; default is all species.",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--diagnostic", dest="diagnostic", action="store_true", default=True,
                        help="Run raw/clamped closed-loop diagnostics (default: enabled).")
    parser.add_argument("--no-diagnostic", dest="diagnostic", action="store_false",
                        help="Disable raw/clamped diagnostics.")
    parser.add_argument(
        "--experiment_mode",
        choices=("baseline", "diagnostic", "constrained_delta", "conserve_project", "qssa"),
        default="baseline",
        help="State update strategy for oracle and clamped pure-AR rollouts.",
    )
    parser.add_argument(
        "--state_update_mode",
        choices=("legacy", "task8_conservative_state", "qssa", "qssa_asymptotic"),
        default="legacy",
        help="State update strategy: 'legacy', 'task8_conservative_state', 'qssa', or 'qssa_asymptotic'.",
    )
    parser.add_argument(
        "--update_constraint",
        choices=("constrained", "raw"),
        default=None,
        help="Macro state update used by rollout; defaults from --experiment_mode.",
    )
    parser.add_argument(
        "--feedback_constraint",
        choices=("constrained", "raw"),
        default=None,
        help="Whether the next-step feedback is clamped to non-negative concentrations.",
    )
    parser.add_argument(
        "--rebuild_scalers",
        action="store_true",
        help="Recompute missing scalers from data; disabled by default for trustworthy comparisons.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio must be between 0 and 1")
    seed_everything(args.seed)
    output_root = Path(args.output_root).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()

    # Auto-discover split_dir from data_dir if not explicitly passed
    if not args.split_dir:
        discovered_split = tr.resolve_trajectory_split_dir(str(data_dir))
        if discovered_split is not None:
            args.split_dir = str(discovered_split)

    requested_checkpoints = args.checkpoints or ([args.checkpoint] if args.checkpoint else None)
    if requested_checkpoints is None:
        requested_checkpoints = [
            str(choose_checkpoint(output_root, args.mechanism, None, data_dir=data_dir))
        ]
    checkpoint_paths = [Path(value).expanduser().resolve() for value in requested_checkpoints]
    print(f"[validate] checkpoints: {len(checkpoint_paths)}", flush=True)
    checkpoints = []
    for checkpoint_path in checkpoint_paths:
        print(f"[validate] loading checkpoint: {checkpoint_path}", flush=True)
        checkpoint = load_checkpoint(checkpoint_path, args.mechanism)
        checkpoint["_path"] = str(checkpoint_path)
        checkpoints.append(checkpoint)

    # Fallback to checkpoint recorded split_dir if still not discovered
    if not args.split_dir and checkpoints:
        cfg = checkpoints[0].get("training_config") or {}
        cfg_split = cfg.get("trajectory_split_dir")
        if cfg_split and Path(cfg_split).is_dir():
            args.split_dir = str(Path(cfg_split).resolve())

    if args.split_dir:
        split_file = Path(args.split_dir) / f"{args.mechanism}_{'validation' if args.split == 'val' else args.split}_sample_ids.csv"
        if args.split != "all" and not split_file.exists():
            raise FileNotFoundError(f"Trajectory split file not found: {split_file}")
    labels = unique_checkpoint_labels(checkpoints, checkpoint_paths)
    feature_info = checkpoints[0]["feature_info"]
    for checkpoint in checkpoints[1:]:
        for key in ("input_cols", "output_cols", "all_species", "macro_species", "micro_species"):
            if list(checkpoint["feature_info"].get(key, [])) != list(feature_info.get(key, [])):
                raise ValueError(
                    f"Checkpoint {checkpoint['_path']} has a different feature contract ({key})"
                )
    # Ensure a split-aware checkpoint is evaluated with the exact train split used to fit its scalers.
    requested_train_signature = None
    if args.split_dir:
        requested_train_signature = tr.trajectory_split_signature(
            Path(args.split_dir).expanduser().resolve(), args.mechanism, "train"
        )
    if requested_train_signature:
        for checkpoint in checkpoints:
            training_config = checkpoint.get("training_config") or {}
            recorded_signature = training_config.get("trajectory_split_train_signature")
            if args.split == "test" and not recorded_signature:
                raise ValueError("Test validation requires a checkpoint with a recorded train split signature")
            if recorded_signature and recorded_signature != requested_train_signature:
                raise ValueError(
                    "Checkpoint trajectory split signature does not match --split_dir: "
                    + repr(recorded_signature) + " != " + repr(requested_train_signature)
                )

    print("[validate] loading scalers and feature metadata...", flush=True)
    scaler_x, scaler_y, saved_info = load_scalers(
        output_root,
        args.mechanism,
        feature_info,
        data_dir=data_dir,
        rebuild_scalers=args.rebuild_scalers,
        split_dir=Path(args.split_dir).expanduser().resolve() if args.split_dir else None,
        checkpoint_root=[cp.parent.parent for cp in checkpoint_paths],
    )
    print(f"[validate] scanning parquet schema in: {data_dir}", flush=True)
    _, _, _, parquet_files = tr.get_schema(str(data_dir), args.mechanism)
    print(
        f"[validate] checking trajectory isolation across "
        f"{len(parquet_files)} parquet parts (this may take a while)...",
        flush=True,
    )
    tr.validate_part_sample_isolation(parquet_files)
    print("[validate] trajectory isolation check complete", flush=True)
    parts = parquet_files if args.split_dir else split_parts(parquet_files, args.val_ratio, args.seed, args.split)
    if not parts:
        raise RuntimeError("Selected split contains no parquet parts")
    output_base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else output_root / "diagnostic_v68" / args.mechanism
        if args.diagnostic
        else output_root / tr.artifact_subdir("validation") / args.mechanism
    )
    output_base.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for checkpoint, label in zip(checkpoints, labels):
        output_dir = output_base / label if len(checkpoints) > 1 else output_base
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[validate] checkpoint={checkpoint['_path']}", flush=True)
        parts_desc = f"source_parts={len(parts)}" + (f" (filtered by {Path(args.split_dir).name})" if args.split_dir else "")
        print(f"[validate] split={args.split}, {parts_desc}, device={tr.DEVICE}", flush=True)
        print("[validate] starting model evaluation...", flush=True)
        result, samples, trajectory_samples, names = evaluate(
            args, checkpoint, saved_info, scaler_x, scaler_y, parts, output_dir,
            data_dir,
        )
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=True)
        write_csv(result, output_dir / "metrics.csv")
        protocol = result.get("physical_metrics_protocol")
        if protocol:
            with (output_dir / "physical_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=True)
            protocol_rows = physical_metrics.flatten_protocol(protocol) if physical_metrics is not None else []
            if protocol_rows:
                pd.DataFrame(protocol_rows).to_csv(output_dir / "physical_metrics.csv", index=False)
                for scope in ("raw", "constrained"):
                    scoped_rows = [row for row in protocol_rows if row.get("scope") == scope]
                    if not scoped_rows:
                        continue
                    pd.DataFrame(scoped_rows).to_csv(output_dir / f"{scope}_metrics.csv", index=False)
                    with (output_dir / f"{scope}_metrics.json").open("w", encoding="utf-8") as handle:
                        json.dump(
                            {"schema_version": "physical_metrics_v68", "scope": scope, "rows": scoped_rows},
                            handle,
                            indent=2,
                            ensure_ascii=False,
                            allow_nan=True,
                        )
        if args.diagnostic:
            with (output_dir / "diagnostic.json").open("w", encoding="utf-8") as handle:
                json.dump(result["diagnostic"], handle, indent=2, ensure_ascii=False, allow_nan=True)
            write_diagnostic_csv(result, output_dir / "diagnostic.csv")
            with (output_dir / "rate_diagnostic.json").open("w", encoding="utf-8") as handle:
                json.dump(result["rate_diagnostic"], handle, indent=2, ensure_ascii=False, allow_nan=True)
            write_rate_diagnostic_csv(result, output_dir / "rate_diagnostic.csv")
            with (output_dir / "constraint_intervention.json").open("w", encoding="utf-8") as handle:
                json.dump(result["constraint_intervention"], handle, indent=2, ensure_ascii=False, allow_nan=True)
            write_constraint_intervention_csv(result, output_dir / "constraint_intervention.csv")
            write_phase2_summary_csv(result, output_dir / "phase2_summary.csv")
            write_trajectory_states_csv(trajectory_samples, names, output_dir / "trajectory_states.csv")
        if not args.no_plots:
            save_plots(result, samples, names, output_dir)
            if args.diagnostic:
                save_diagnostic_plots(result, names, output_dir)
                save_rate_diagnostic_plots(result, names, output_dir)
                save_phase2_plots(result, output_dir)
            save_concentration_trajectory_plots(
                trajectory_samples, names, output_dir,
                requested_species=args.trajectory_species,
            )
        print(f"[validate] metrics: {output_dir / 'metrics.json'}")
        if not args.no_plots:
            print(f"[validate] plots:   {output_dir / 'plots'}")
        comparison_rows.append({"epoch": result.get("epoch"), "checkpoint": result.get("checkpoint"), "output_dir": str(output_dir)})
    if len(comparison_rows) > 1:
        with (output_base / "checkpoint_comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(comparison_rows, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()



