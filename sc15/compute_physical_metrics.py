"""Shared raw/constrained physical metrics for V67 validation.

The module is deliberately array-based so it can be used by the streaming
validator and by the post-hoc ``trajectory_states.csv`` audit tool.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# Match the model's LOG_EPS floor.  Values around 1e-20 are valid resolved
# trace states in the revised pipeline and must not be reported as floor hits.
TRACE_FLOOR_THRESHOLD = 1.0001e-40


class PhysicalMetricAccumulator:
    """Bounded-memory mass-fraction metrics over every validation window.

    Invalid values are counted, never replaced with zero in conservation
    statistics. Element error is prediction minus the corresponding reference
    state; mass error is reported both against that reference and against one.
    """

    def __init__(
        self,
        species_names,
        element_matrix=None,
        element_names=None,
        negative_tolerance: float = 1e-12,
        inert_species=None,
    ):
        self.names = list(species_names)
        n = len(self.names)
        self.matrix = None if element_matrix is None else np.asarray(element_matrix, dtype=np.float64)
        self.elements = list(element_names or [])
        self.negative_tolerance = float(negative_tolerance)
        self.inert_species = set(inert_species or ["AR", "Ar", "HE", "He"])
        self.is_inert = np.array([name in self.inert_species for name in self.names], dtype=bool)
        if self.matrix is not None:
            if self.matrix.ndim != 2 or self.matrix.shape != (len(self.elements), n):
                raise ValueError("element matrix dimensions do not match names")
            if not np.isfinite(self.matrix).all():
                raise ValueError("element matrix contains nonfinite values")
        self.rows = self.nonfinite = self.true_nonfinite = self.negative_rows = 0
        self.negative = np.zeros(n, dtype=np.int64)
        self.count = np.zeros(n, dtype=np.int64)
        self.abs_sum = np.zeros(n)
        self.sq_sum = np.zeros(n)
        self.error_sum = np.zeros(n)
        self.log_count = np.zeros(n, dtype=np.int64)
        self.log_sq_sum = np.zeros(n)
        self.floor_hit = np.zeros(n, dtype=np.int64)
        self.active_log_count = np.zeros(n, dtype=np.int64)
        self.active_log_sq_sum = np.zeros(n)
        self.valid_rows = 0
        self.pred_mass_sum = self.mass_sq = self.mass_max = 0.0
        self.unity_sq = self.unity_max = 0.0
        self.element_sq = np.zeros(len(self.elements))
        self.element_max = np.zeros(len(self.elements))

    def update(self, true, prediction):
        true = np.asarray(true, dtype=np.float64)
        pred = np.asarray(prediction, dtype=np.float64)
        if true.shape != pred.shape or true.ndim != 2 or true.shape[1] != len(self.names):
            raise ValueError("physical metrics require matching (rows, species) arrays")
        pred_finite = np.isfinite(pred)
        true_finite = np.isfinite(true)
        finite = pred_finite & true_finite
        negative = pred_finite & (pred < -self.negative_tolerance)
        self.rows += len(pred)
        self.nonfinite += int((~pred_finite).sum())
        self.true_nonfinite += int((~true_finite).sum())
        self.negative_rows += int(negative.any(axis=1).sum())
        self.negative += negative.sum(axis=0)
        error = np.where(finite, pred, 0.0) - np.where(finite, true, 0.0)
        self.count += finite.sum(axis=0)
        self.abs_sum += np.abs(error).sum(axis=0)
        self.sq_sum += np.square(error).sum(axis=0)
        self.error_sum += error.sum(axis=0)
        positive = finite & (pred > 0.0) & (true > 0.0)
        log_error = np.log(np.where(positive, pred, 1.0)) - np.log(np.where(positive, true, 1.0))
        self.log_count += positive.sum(axis=0)
        self.log_sq_sum += np.square(log_error).sum(axis=0)
        floor_hit = finite & (pred > 0.0) & (pred <= TRACE_FLOOR_THRESHOLD)
        self.floor_hit += floor_hit.sum(axis=0)
        active = positive & (true > 1e-15)
        active_log_error = np.log(np.where(active, pred, 1.0)) - np.log(np.where(active, true, 1.0))
        self.active_log_count += active.sum(axis=0)
        self.active_log_sq_sum += np.square(active_log_error).sum(axis=0)
        valid_rows = finite.all(axis=1)
        if valid_rows.any():
            p = pred[valid_rows]
            t = true[valid_rows]
            mass = p.sum(axis=1)
            mass_error = mass - t.sum(axis=1)
            unity_error = mass - 1.0
            self.valid_rows += len(p)
            self.pred_mass_sum += float(mass.sum())
            self.mass_sq += float(np.square(mass_error).sum())
            self.mass_max = max(self.mass_max, float(np.abs(mass_error).max()))
            self.unity_sq += float(np.square(unity_error).sum())
            self.unity_max = max(self.unity_max, float(np.abs(unity_error).max()))
            if self.matrix is not None:
                elemental_error = (p - t) @ self.matrix.T
                self.element_sq += np.square(elemental_error).sum(axis=0)
                self.element_max = np.maximum(self.element_max, np.abs(elemental_error).max(axis=0))

    def as_dict(self):
        def rmse(squared, n):
            return float(np.sqrt(squared / n)) if n else None

        n = self.valid_rows
        reactive_neg = int(self.negative[~self.is_inert].sum()) if hasattr(self, "is_inert") else int(self.negative.sum())
        inert_neg = int(self.negative[self.is_inert].sum()) if hasattr(self, "is_inert") else 0
        result = {
            "rows": self.rows,
            "negative_tolerance": self.negative_tolerance,
            "nonfinite_values": self.nonfinite,
            "true_nonfinite_values": self.true_nonfinite,
            "negative_value_count": int(self.negative.sum()),
            "reactive_negative_value_count": reactive_neg,
            "inert_negative_value_count": inert_neg,
            "negative_value_ratio": float(self.negative.sum() / (self.rows * len(self.names))) if self.rows and self.names else 0.0,
            "negative_row_count": self.negative_rows,
            "negative_row_ratio": self.negative_rows / self.rows if self.rows else 0.0,
            "mass_sum": {
                "n": n,
                "predicted_mean": self.pred_mass_sum / n if n else None,
                "max_abs_error": self.mass_max if n else None,
                "rmse": rmse(self.mass_sq, n),
                "max_abs_error_from_one": self.unity_max if n else None,
                "rmse_from_one": rmse(self.unity_sq, n),
            },
            "element_closure_available": self.matrix is not None and n > 0,
            "per_species": {},
        }
        for i, name in enumerate(self.names):
            count = int(self.count[i])
            result["per_species"][name] = {
                "n": count,
                "mae": float(self.abs_sum[i] / count) if count else None,
                "rmse": rmse(self.sq_sum[i], count),
                "bias": float(self.error_sum[i] / count) if count else None,
                "log_n": int(self.log_count[i]),
                "log_rmse": rmse(self.log_sq_sum[i], int(self.log_count[i])),
                "floor_hit_count": int(self.floor_hit[i]),
                "floor_hit_ratio": float(self.floor_hit[i] / count) if count else 0.0,
                "active_log_n": int(self.active_log_count[i]),
                "active_log_rmse": rmse(self.active_log_sq_sum[i], int(self.active_log_count[i])),
            }
        if self.matrix is not None:
            result["element_closure"] = {
                name: {"n": n, "max_abs_error": float(self.element_max[i]) if n else None,
                       "rmse": rmse(self.element_sq[i], n)}
                for i, name in enumerate(self.elements)
            }
        return result


def _metric(values: np.ndarray, truth: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(truth)
    result = {}
    for i in range(values.shape[1]):
        mask = finite[:, i]
        if not mask.any():
            result[i] = {
                "n": 0, "mae": None, "rmse": None, "bias": None, "log_rmse": None,
                "floor_hit_count": 0, "floor_hit_ratio": 0.0, "active_log_n": 0, "active_log_rmse": None,
            }
            continue
        error = values[mask, i] - truth[mask, i]
        positive = mask & (values[:, i] > 0.0) & (truth[:, i] > 0.0)
        log_rmse = None
        if positive.any():
            log_error = np.log(values[positive, i]) - np.log(truth[positive, i])
            log_rmse = float(np.sqrt(np.mean(log_error ** 2)))
        active = positive & (truth[:, i] > 1e-15)
        active_log_rmse = None
        if active.any():
            active_log_error = np.log(values[active, i]) - np.log(truth[active, i])
            active_log_rmse = float(np.sqrt(np.mean(active_log_error ** 2)))
        floor_hits = mask & (values[:, i] <= TRACE_FLOOR_THRESHOLD)
        floor_hit_ratio = float(floor_hits.mean()) if mask.any() else 0.0
        result[i] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "log_rmse": log_rmse,
            "floor_hit_count": int(floor_hits.sum()),
            "floor_hit_ratio": floor_hit_ratio,
            "active_log_n": int(active.sum()),
            "active_log_rmse": active_log_rmse,
        }
    return result


def compute_physical_metrics(
    true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    species_names: Sequence[str],
    element_matrix: np.ndarray | None = None,
    element_names: Sequence[str] | None = None,
    negative_tolerance: float = 1e-12,
) -> dict:
    """Compute comparable metrics for raw and constrained predictions.

    ``true`` and prediction arrays are shaped ``(rows, species)`` and must use
    mass-fraction columns in the same order.  Element metrics are optional and
    require a matrix shaped ``(elements, species)`` containing elemental mass
    fractions per species.
    """
    true = np.asarray(true, dtype=np.float64)
    if true.ndim == 1:
        true = true[:, None]
    names = list(species_names)
    if true.ndim != 2 or true.shape[1] != len(names):
        raise ValueError("true must have one column per species")
    
    inert_mask = np.array([name.upper() in {"N2", "AR", "HE"} for name in names])
    reactive_mask = ~inert_mask

    output = {}
    for label, prediction in predictions.items():
        values = np.asarray(prediction, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape != true.shape:
            raise ValueError(f"prediction {label!r} shape {values.shape} != true {true.shape}")
        finite = np.isfinite(values)
        negative = finite & (values < -float(negative_tolerance))
        row_negative = negative.any(axis=1)

        negative_reactive = negative[:, reactive_mask] if np.any(reactive_mask) else np.zeros((values.shape[0], 0), dtype=bool)
        negative_inert = negative[:, inert_mask] if np.any(inert_mask) else np.zeros((values.shape[0], 0), dtype=bool)

        mass_sum = np.nansum(values, axis=1)
        true_sum = np.nansum(true, axis=1)
        item = {
            "rows": int(values.shape[0]),
            "nonfinite_values": int((~finite).sum()),
            "negative_tolerance": float(negative_tolerance),
            "negative_value_count": int(negative.sum()),
            "negative_value_ratio": float(negative.mean()) if negative.size else 0.0,
            "negative_row_count": int(row_negative.sum()),
            "negative_row_ratio": float(row_negative.mean()) if len(row_negative) else 0.0,
            "negative_reactive_count": int(negative_reactive.sum()),
            "negative_reactive_ratio": float(negative_reactive.mean()) if negative_reactive.size else 0.0,
            "negative_inert_count": int(negative_inert.sum()),
            "negative_inert_ratio": float(negative_inert.mean()) if negative_inert.size else 0.0,
            "mass_sum": {
                "predicted_mean": float(np.nanmean(mass_sum)) if len(mass_sum) else None,
                "max_abs_error": float(np.nanmax(np.abs(mass_sum - true_sum))) if len(mass_sum) else None,
                "rmse": float(np.sqrt(np.nanmean((mass_sum - true_sum) ** 2))) if len(mass_sum) else None,
            },
            "per_species": {
                names[index]: metrics
                for index, metrics in _metric(values, true).items()
            },
        }
        if element_matrix is not None:
            matrix = np.asarray(element_matrix, dtype=np.float64)
            if matrix.shape[1] != values.shape[1]:
                raise ValueError("element_matrix must have one column per species")
            element_labels = list(element_names or [f"element_{i}" for i in range(matrix.shape[0])])
            if len(element_labels) != matrix.shape[0]:
                raise ValueError("element_names length does not match element_matrix")
            pred_element = values @ matrix.T
            true_element = true @ matrix.T
            item["element_closure"] = {
                label: {
                    "max_abs_error": float(np.nanmax(np.abs(pred_element[:, i] - true_element[:, i]))),
                    "rmse": float(np.sqrt(np.nanmean((pred_element[:, i] - true_element[:, i]) ** 2))),
                }
                for i, label in enumerate(element_labels)
            }
        output[label] = item
    return output


def summarize_validation_result(result: dict) -> dict:
    """Normalize metrics already accumulated by ``validate_sc_v67.py``.

    The validator keeps streaming accumulators on GPU batches, so this helper
    exposes the same raw/constrained naming without retaining all predictions.
    """
    if "physical_metrics_streaming" in result:
        modes = result["physical_metrics_streaming"]
        scopes = [scope for horizons in modes.values() for payload in horizons.values() for scope in payload.values()]
        available = bool(scopes) and all(scope.get("rows", 0) > 0 for scope in scopes)
        return {
            "schema_version": "physical_metrics_v67",
            "available": available,
            "full_physics_available": available and all(scope.get("element_closure_available") for scope in scopes),
            "coverage_scope": "streaming_validation_windows",
            "species_scope": "all_species_including_inerts",
            "raw_semantics": "unconstrained macro Euler candidate; micro species retain model log-space bounds",
            "constrained_semantics": "reported state under selected update rule; pure_ar_raw retains the raw update",
            "mass_error_reference": "corresponding true endpoint; *_from_one additionally tests normalization",
            "element_error_reference": "corresponding true endpoint",
            "element_metadata": result.get("physical_element_metadata"),
            "modes": modes,
        }
    summary = {
        "schema_version": "physical_metrics_v67",
        "available": False,
        "full_physics_available": False,
        "modes": {},
    }
    diagnostics = result.get("diagnostic", {}).get("by_horizon", {})
    for mode, horizons in result.get("modes", {}).items():
        summary["modes"][mode] = {}
        for horizon, values in horizons.items():
            state = values.get("state", {}).get("per_species", {})
            item = {"constrained": {"per_species": state}}
            diagnostic = diagnostics.get(mode, {}).get(str(horizon), {})
            if diagnostic:
                raw = diagnostic.get("raw_vs_true", {})
                item["raw"] = {
                    "negative_value_ratio": diagnostic.get("negative_value_ratio"),
                    "negative_row_ratio": diagnostic.get("negative_ratio"),
                    "per_species": raw.get("per_species", {}),
                }
            summary["modes"][mode][str(horizon)] = item
    valid_horizons = [
        horizon
        for horizons in summary["modes"].values()
        for horizon, scopes in horizons.items()
        if scopes.get("constrained", {}).get("per_species")
    ]
    if valid_horizons:
        summary["available"] = True
        summary["coverage_scope"] = "streaming_validation_windows"
    else:
        summary["reason"] = "no_valid_metrics"
    return summary


def flatten_protocol(protocol: dict) -> list[dict]:
    """Flatten the protocol summary into one row per mode/horizon/species."""
    rows = []
    for mode, horizons in protocol.get("modes", {}).items():
        for horizon, scopes in horizons.items():
            for scope, payload in scopes.items():
                for species, metrics in payload.get("per_species", {}).items():
                    rows.append({
                        "mode": mode,
                        "horizon": horizon,
                        "scope": scope,
                        "species": species,
                        **metrics,
                        "negative_value_count": payload.get("negative_value_count"),
                        "reactive_negative_value_count": payload.get("reactive_negative_value_count"),
                        "inert_negative_value_count": payload.get("inert_negative_value_count"),
                        "negative_value_ratio": payload.get("negative_value_ratio"),
                        "negative_row_count": payload.get("negative_row_count"),
                        "negative_row_ratio": payload.get("negative_row_ratio"),
                        "rows": payload.get("rows"),
                        "nonfinite_values": payload.get("nonfinite_values"),
                        "true_nonfinite_values": payload.get("true_nonfinite_values"),
                        **{f"mass_sum_{key}": value for key, value in payload.get("mass_sum", {}).items()},
                        "element_closure_available": payload.get("element_closure_available"),
                        **{f"element_{element}_{key}": value
                           for element, item in payload.get("element_closure", {}).items()
                           for key, value in item.items()},
                    })
    return rows


def _read_trajectory_states(path: Path, predictions: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    import pandas as pd

    frame = pd.read_csv(path)
    ref_cols = [c for c in frame.columns if c.startswith("reference_")]
    if ref_cols and "species" not in frame.columns:
        # Wide-format compatibility (reference_{species}, raw_{species}, constrained_{species})
        names = [c[len("reference_"):] for c in ref_cols]
        true_arr = frame[ref_cols].to_numpy(dtype=float)
        result = {}
        for label in predictions:
            cand_cols = [f"{label}_{sp}" for sp in names]
            if all(c in frame.columns for c in cand_cols):
                result[label] = frame[cand_cols].to_numpy(dtype=float)
            elif label == "oracle":
                result[label] = true_arr
            elif label in ("oracle_raw", "pure_ar_raw", "raw"):
                raw_cols = [f"raw_{sp}" for sp in names]
                if all(c in frame.columns for c in raw_cols):
                    result[label] = frame[raw_cols].to_numpy(dtype=float)
                else:
                    raise ValueError(f"prediction columns not found for: {label}")
            elif label in ("constrained", "pure_ar", "pure_ar_raw_clamped"):
                constr_cols = [f"constrained_{sp}" for sp in names]
                if all(c in frame.columns for c in constr_cols):
                    result[label] = frame[constr_cols].to_numpy(dtype=float)
                else:
                    raise ValueError(f"prediction columns not found for: {label}")
            else:
                raise ValueError(f"prediction column not found: {label}")
        return true_arr, result, names

    required = {"species", "true"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory state file is missing columns: {sorted(missing)}")
    names = list(dict.fromkeys(frame["species"].astype(str)))
    true = frame.pivot(index=[c for c in ("trajectory_id", "step") if c in frame], columns="species", values="true")
    true = true.reindex(columns=names)
    arrays = {"true": true.to_numpy(dtype=float)}
    result = {}
    for label in predictions:
        if label not in frame:
            raise ValueError(f"prediction column not found: {label}")
        pivot = frame.pivot(index=[c for c in ("trajectory_id", "step") if c in frame], columns="species", values=label)
        result[label] = pivot.reindex(columns=names).to_numpy(dtype=float)
    return arrays["true"], result, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute V67 raw/constrained physical metrics")
    parser.add_argument("--trajectory-states", type=Path)
    parser.add_argument("--predictions", nargs="+", default=["oracle_raw", "oracle", "pure_ar_raw_clamped", "pure_ar"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negative-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if not args.trajectory_states:
        parser.error("--trajectory-states is required")
    true, predictions, names = _read_trajectory_states(args.trajectory_states, args.predictions)
    result = {
        "source": str(args.trajectory_states.resolve()),
        "species": names,
        "metrics": compute_physical_metrics(true, predictions, names, negative_tolerance=args.negative_tolerance),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
