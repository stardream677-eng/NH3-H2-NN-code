import numpy as np
import pandas as pd

import split_trajectories_v21 as splitter


def test_stratified_split_preserves_sampler_layers(tmp_path):
    strata = (
        "Spec_Engine_Core",
        "Spec_Cold_Oxidation",
        "Spec_Radical_Burst",
        "Spec_Extreme_Boundaries",
    )
    rows = []
    for layer_index, layer in enumerate(strata):
        for index in range(40):
            value = layer_index * 100.0 + index
            rows.append(
                {
                    "sample_id": f"{layer}-{index}",
                    "stratum_name": layer,
                    "time": 0.0,
                    "T": value,
                    "P": value + 1.0,
                    "phi_actual": value + 2.0,
                    "Y_H2": value + 3.0,
                }
            )

    pd.DataFrame(rows).to_parquet(tmp_path / "part-00000.parquet", index=False)
    counts, metadata = splitter._collect_trajectory_metadata(tmp_path)

    assert metadata["stratum_name"].value_counts().to_dict() == {name: 40 for name in strata}
    split_ids = splitter._split_ids(
        np.asarray(sorted(counts), dtype=object),
        np.random.default_rng(17),
        train=0.7,
        val=0.15,
        metadata=metadata,
    )

    for ids in split_ids.values():
        assert set(metadata.loc[ids, "stratum_name"]) == set(strata)
