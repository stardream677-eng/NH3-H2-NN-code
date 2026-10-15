"""Species Validation Protocol and Sentinel Suite for Ammonia-Hydrogen Combustion.

Standardizes model evaluation, validation metrics, and model selection:
1. Frozen Protocol ID: v68_species_sentinel_protocol_v1
2. Fixed horizons: H in (1, 8, 32, 128) evaluated independently
3. Comprehensive metrics:
   - Per-species MAE and peak-normalized MAE (NMAE)
   - Active-zone log-RMSE for fast radicals (OH, H, O, HO2, NH2, HNO)
   - Cold-state false-positive auditing (Y_true < 1e-12, Y_pred > 1e-6)
   - Peak value and peak timing errors
   - Final pollutant concentrations (NO, N2O)
   - Physical element conservation and mass fraction sum residuals
   - Strict gating: zero non-finite values, finite gradient norm, bounded cold false-positives
4. Horizon-specific checkpointing: prevents short curriculum stages (H=1, 8)
   from prematurely overwriting or locking out long-horizon (H=32, 128) best models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROTOCOL_ID = "v68_species_sentinel_protocol_v1"
STANDARD_HORIZONS = (1, 8, 32, 128)

MAJOR_SPECIES = ("NH3", "H2", "O2", "N2", "H2O")
KEY_RADICALS = ("OH", "H", "O", "HO2", "NH2", "HNO")
POLLUTANT_SPECIES = ("NO", "N2O")

# Cold state threshold below which species is considered dormant
COLD_TRUE_THRESHOLD = 1e-12
# Predicted mass fraction above which a cold dormant species is flagged as false positive
COLD_FALSE_POSITIVE_THRESHOLD = 1e-6
# Active radical threshold for logarithmic tracking
RADICAL_ACTIVE_THRESHOLD = 1e-12


def compute_species_trajectory_metrics(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    species_names: Sequence[str],
    time_points: Optional[np.ndarray] = None,
    element_matrix: Optional[np.ndarray] = None,
    negative_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compute detailed per-species and trajectory-level validation metrics.

    Parameters
    ----------
    true_states : np.ndarray
        Shape (T, n_species) or (B, T, n_species) reference states.
    pred_states : np.ndarray
        Shape matching true_states predicted mass fractions.
    species_names : Sequence[str]
        List of species names.
    time_points : Optional[np.ndarray]
        Physical time corresponding to the T steps. If None, step indices are used.
    element_matrix : Optional[np.ndarray]
        (n_elements, n_species) elemental mass matrix for conservation tracking.
    negative_tolerance : float
        Tolerance for negative mass fractions.

    Returns
    -------
    Dict[str, Any] containing per-species, group-level, and physical closure metrics.
    """
    true_arr = np.asarray(true_states, dtype=np.float64)
    pred_arr = np.asarray(pred_states, dtype=np.float64)

    # Flatten batch dimension if present: (B, T, S) -> (-1, S) for aggregate error
    if true_arr.ndim == 3:
        B, T, S = true_arr.shape
        flat_true = true_arr.reshape(-1, S)
        flat_pred = pred_arr.reshape(-1, S)
    elif true_arr.ndim == 2:
        T, S = true_arr.shape
        B = 1
        flat_true = true_arr
        flat_pred = pred_arr
    else:
        raise ValueError(f"Expected 2D or 3D states, got shape {true_arr.shape}")

    n_species = len(species_names)
    assert S == n_species, f"Mismatch: {S} channels vs {n_species} names"

    # 1. Nonfinite and negative counts
    nonfinite_count = int((~np.isfinite(flat_pred)).sum())
    negative_count = int((flat_pred < -negative_tolerance).sum())

    # 2. Per-species metrics
    per_species: Dict[str, Dict[str, Any]] = {}
    major_nmae_list = []
    radical_log_rmse_list = []
    pollutant_nmae_list = []
    cold_fp_total = 0
    cold_eval_total = 0

    for i, sp in enumerate(species_names):
        y_t = flat_true[:, i]
        y_p = flat_pred[:, i]

        valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
        if not valid_mask.any():
            per_species[sp] = {
                "n": 0, "mae": None, "rmse": None, "nmae": None,
                "peak_true": None, "peak_pred": None, "peak_error": None,
                "active_log_rmse": None, "cold_fp_count": 0, "cold_fp_ratio": 0.0,
            }
            continue

        yt_v = y_t[valid_mask]
        yp_v = y_p[valid_mask]

        abs_err = np.abs(yp_v - yt_v)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(abs_err ** 2)))

        peak_true = float(np.max(yt_v))
        peak_pred = float(np.max(yp_v))
        peak_error = abs(peak_pred - peak_true)

        # Peak normalized MAE
        norm_denom = max(peak_true, 1e-6)
        nmae = mae / norm_denom

        # Active radical log-RMSE
        active_mask = (yt_v > RADICAL_ACTIVE_THRESHOLD)
        if active_mask.any():
            yp_active = np.maximum(yp_v[active_mask], 1e-12)
            log_err = np.log10(yp_active) - np.log10(yt_v[active_mask])
            active_log_rmse = float(np.sqrt(np.mean(log_err ** 2)))
        else:
            active_log_rmse = 0.0

        # Cold false positive auditing
        cold_mask = (yt_v < COLD_TRUE_THRESHOLD)
        n_cold = int(cold_mask.sum())
        if n_cold > 0:
            cold_fp = int((yp_v[cold_mask] > COLD_FALSE_POSITIVE_THRESHOLD).sum())
            cold_fp_ratio = cold_fp / n_cold
            cold_fp_total += cold_fp
            cold_eval_total += n_cold
        else:
            cold_fp = 0
            cold_fp_ratio = 0.0

        per_species[sp] = {
            "n": int(valid_mask.sum()),
            "mae": mae,
            "rmse": rmse,
            "nmae": nmae,
            "peak_true": peak_true,
            "peak_pred": peak_pred,
            "peak_error": peak_error,
            "active_log_rmse": active_log_rmse,
            "cold_fp_count": cold_fp,
            "cold_fp_ratio": cold_fp_ratio,
        }

        if sp in MAJOR_SPECIES:
            major_nmae_list.append(nmae)
        elif sp in KEY_RADICALS:
            radical_log_rmse_list.append(active_log_rmse)
        elif sp in POLLUTANT_SPECIES:
            pollutant_nmae_list.append(nmae)

    # 3. Trajectory timing error if 3D (B, T, S) or 2D single trajectory
    timing_metrics: Dict[str, Any] = {}
    if true_arr.ndim == 3 and time_points is not None and len(time_points) == T:
        for sp in ("OH", "T"):
            if sp in species_names:
                idx = species_names.index(sp)
                t_peaks_true = []
                t_peaks_pred = []
                for b in range(B):
                    idx_true = int(np.argmax(true_arr[b, :, idx]))
                    idx_pred = int(np.argmax(pred_arr[b, :, idx]))
                    t_peaks_true.append(time_points[idx_true])
                    t_peaks_pred.append(time_points[idx_pred])
                timing_metrics[f"{sp}_peak_time_mae"] = float(np.mean(np.abs(np.array(t_peaks_pred) - np.array(t_peaks_true))))

    # 4. Physical conservation closure
    mass_sum_pred = flat_pred.sum(axis=1)
    unity_error = np.abs(mass_sum_pred - 1.0)
    mass_closure = {
        "mean_mass_sum": float(np.mean(mass_sum_pred)),
        "max_unity_error": float(np.max(unity_error)),
        "rmse_unity_error": float(np.sqrt(np.mean(unity_error ** 2))),
    }

    element_closure = {}
    if element_matrix is not None:
        E_mat = np.asarray(element_matrix, dtype=np.float64)
        elem_err = (flat_pred - flat_true) @ E_mat.T
        max_elem_err = float(np.max(np.abs(elem_err)))
        rmse_elem_err = float(np.sqrt(np.mean(elem_err ** 2)))
        element_closure = {
            "max_element_error": max_elem_err,
            "rmse_element_error": rmse_elem_err,
        }

    # 5. Composite balanced score
    mean_major_nmae = float(np.mean(major_nmae_list)) if major_nmae_list else 0.0
    mean_rad_log_rmse = float(np.mean(radical_log_rmse_list)) if radical_log_rmse_list else 0.0
    mean_pollutant_nmae = float(np.mean(pollutant_nmae_list)) if pollutant_nmae_list else 0.0
    cold_fp_rate = (cold_fp_total / max(cold_eval_total, 1))

    # Balanced penalty function
    composite_score = (
        0.40 * mean_major_nmae
        + 0.35 * mean_rad_log_rmse
        + 0.25 * mean_pollutant_nmae
        + 10.0 * cold_fp_rate
    )
    if nonfinite_count > 0:
        composite_score += 1e6 * nonfinite_count

    max_unity_err = float(mass_closure.get("max_abs_unity_error", 0.0)) if mass_closure else 0.0
    max_elem_err = float(element_closure.get("max_element_error", 0.0)) if element_closure else 0.0
    is_valid_candidate = (
        (nonfinite_count == 0)
        and (negative_count == 0)
        and (cold_fp_rate < 0.10)
        and (max_unity_err <= 0.05)
        and (max_elem_err <= 0.05)
    )

    return {
        "protocol_id": PROTOCOL_ID,
        "composite_score": composite_score,
        "mean_major_nmae": mean_major_nmae,
        "mean_radical_log_rmse": mean_rad_log_rmse,
        "mean_pollutant_nmae": mean_pollutant_nmae,
        "cold_fp_rate": cold_fp_rate,
        "nonfinite_count": nonfinite_count,
        "negative_count": negative_count,
        "is_valid_candidate": is_valid_candidate,
        "mass_closure": mass_closure,
        "element_closure": element_closure,
        "timing_metrics": timing_metrics,
        "per_species": per_species,
    }


class HorizonSentinelTracker:
    """Tracks best model checkpoints independently per horizon (H=1, 8, 32, 128)."""

    def __init__(self, horizons: Sequence[int] = STANDARD_HORIZONS):
        self.horizons = tuple(horizons)
        self.best_scores: Dict[int, float] = {h: float("inf") for h in self.horizons}
        self.best_epochs: Dict[int, int] = {h: -1 for h in self.horizons}
        self.best_records: Dict[int, Dict[str, Any]] = {}

    def consider_candidate(
        self,
        horizon: int,
        metrics: Dict[str, Any],
        epoch: int,
    ) -> bool:
        """Return True if the candidate achieves a new best valid score for this horizon."""
        if horizon not in self.best_scores:
            self.best_scores[horizon] = float("inf")
            self.best_epochs[horizon] = -1

        score = float(metrics.get("composite_score", float("inf")))
        is_valid = bool(metrics.get("is_valid_candidate", False))

        if is_valid and score < self.best_scores[horizon]:
            self.best_scores[horizon] = score
            self.best_epochs[horizon] = epoch
            self.best_records[horizon] = metrics
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "horizons": list(self.horizons),
            "best_scores": {str(h): s for h, s in self.best_scores.items()},
            "best_epochs": {str(h): e for h, e in self.best_epochs.items()},
            "best_records": {str(h): r for h, r in self.best_records.items()},
        }
