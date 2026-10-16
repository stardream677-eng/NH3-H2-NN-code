"""Regression tests for condition-group and trajectory isolation in sc16."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import split_trajectories_v21 as splitter
import train_sc_v68 as training


def _sample_metadata(n_groups=9):
    rows = []
    for group_idx in range(n_groups):
        for member_idx in range(2):
            rows.append({
                "sample_id": f"traj-{group_idx}-{member_idx}",
                "condition_group_id": f"condition-{group_idx}",
                "stratum_name": "test",
                "T": 900.0 + 70 * group_idx,
                "P": 1.0 + group_idx,
                "phi": 0.7 + 0.1 * group_idx,
                "H2": 0.1 + 0.01 * group_idx,
                "phi_actual_error": None,
            })
    frame = pd.DataFrame(rows).set_index("sample_id")
    return frame


@pytest.mark.parametrize("stratify", [False, True])
def test_condition_group_cannot_cross_random_or_stratified_splits(stratify):
    metadata = _sample_metadata()
    parts = splitter._split_ids(
        metadata.index.to_numpy(dtype=object),
        np.random.default_rng(2026),
        train=0.6,
        val=0.2,
        metadata=metadata,
        stratify=stratify,
    )
    seen = {}
    for split, ids in parts.items():
        for group in metadata.loc[list(ids), "condition_group_id"]:
            assert group not in seen or seen[group] == split
            seen[group] = split
    assert len(seen) == 9
    assert sum(map(len, parts.values())) == len(metadata)


def test_extrapolation_reserves_whole_condition_group(tmp_path, monkeypatch):
    metadata = _sample_metadata()
    counts = {sid: 5 for sid in metadata.index}
    part = tmp_path / "trajectories_nuig_2024" / "part-000001.parquet"
    part.parent.mkdir()
    monkeypatch.setattr(
        splitter._conditions, "dataset_parts",
        lambda root: iter([("nuig_2024", [part])]),
    )
    monkeypatch.setattr(
        splitter, "_collect_trajectory_metadata",
        lambda folder, gas=None, mechanism_error=None: (counts, metadata),
    )
    result = splitter.split_database(
        tmp_path, tmp_path / "splits", seed=12, stratify=False,
        extrapolation_fraction=1 / len(metadata),
    )
    seen = {}
    for entry in result["summary"]:
        split = entry["split"]
        ids = pd.read_csv(entry["path"])["sample_id"].tolist()
        for group in metadata.loc[ids, "condition_group_id"]:
            assert group not in seen or seen[group] == split
            seen[group] = split
    assert len(seen) == 9
    extra = next(entry for entry in result["summary"] if entry["split"] == "extrapolation")
    assert extra["trajectories"] == 2


def test_overlapping_train_validation_manifests_rejected():
    with pytest.raises(ValueError, match="overlap"):
        training.require_disjoint_split_ids({"101", "102"}, {"102", "103"})
    training.require_disjoint_split_ids({"101"}, {"102"})
