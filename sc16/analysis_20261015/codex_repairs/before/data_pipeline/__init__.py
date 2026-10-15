# -*- coding: utf-8 -*-
"""Data Pipeline Package for Combustion Surrogate Modeling.

This package contains tools for:
  - Latin Hypercube Sampling of combustion initial states (lhs_sampler_sc_v11)
  - 0D Cantera reactor trajectories and QSSA asymptotic rate inversion (runner_sc_v21_qssa)
  - Augmented trajectory generation (generate_augmented_trajectories)
  - Dataset conversion and building (build_training_dataset_v72)
  - Trajectory condition stratification (trajectory_conditions_v67)
  - Database consistency and conservation auditing (audit_generated_database)
  - Initial condition distribution auditing (audit_initial_conditions_v67)
  - Zero-leakage stratified train/val/test splitting (split_trajectories_v21)
"""

from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

__all__ = [
    "lhs_sampler_sc_v11",
    "runner_sc_v21_qssa",
    "generate_augmented_trajectories",
    "build_training_dataset_v72",
    "trajectory_conditions_v67",
    "audit_generated_database",
    "audit_initial_conditions_v67",
    "split_trajectories_v21",
]
