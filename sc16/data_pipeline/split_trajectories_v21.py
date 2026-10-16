from __future__ import annotations

"""Create leakage-safe trajectory train/validation/test split manifests.

Trajectories sharing condition_group_id are always kept together. Optional
stratification balances physical conditions; optional extrapolation reserves
whole condition groups. Generated split manifests are new experiment artifacts:
never silently reuse earlier checkpoint/scaler signatures.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Self-contained initial-condition and dataset resolution helpers
class _ConditionsHelper:
    @staticmethod
    def dataset_parts(root: Path):
        root = Path(root)
        manifest = root / "dataset_manifest.csv"
        seen = set()
        if manifest.is_file():
            for row in pd.read_csv(manifest).to_dict("records"):
                def field(names):
                    return next((row[name] for name in names
                                 if name in row and pd.notna(row[name]) and str(row[name]).strip()), None)
                mechanism = field(("mechanism_name", "mechanism"))
                recorded = field(("dataset_path", "dataset_folder", "path"))
                if mechanism is None or recorded is None:
                    continue
                mechanism = str(mechanism)
                local = root / f"trajectories_{mechanism}"
                folder = Path(str(recorded))
                folder = folder if folder.is_absolute() else root / folder
                folder = local if local.is_dir() else folder
                if mechanism in seen:
                    continue
                seen.add(mechanism)
                parts = sorted(folder.glob("part-*.parquet"))
                if parts:
                    yield mechanism, parts
        else:
            for traj_dir in sorted(root.glob("trajectories_*")):
                if not traj_dir.is_dir():
                    continue
                mechanism = traj_dir.name.replace("trajectories_", "")
                if mechanism in seen:
                    continue
                seen.add(mechanism)
                parts = sorted(traj_dir.glob("part-*.parquet"))
                if parts:
                    yield mechanism, parts

    @staticmethod
    def load_phi_gas(folder: Path, root: Path, mechanism: str):
        known = {"nuig_2024": "NUIG2024.yaml", "tsinghua_2024c": "TH2024.yaml", "ucf_2024": "UCF2024.yaml"}
        name = known.get(mechanism)
        cand_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path(root), Path(folder)]
        cand_dirs.extend([d.parent for d in list(cand_dirs)])
        yaml_path = None
        if name:
            for d in cand_dirs:
                if (d / name).is_file():
                    yaml_path = (d / name).resolve()
                    break
        if yaml_path is not None:
            try:
                import cantera as ct
                return ct.Solution(str(yaml_path)), yaml_path, None
            except Exception as exc:
                return None, yaml_path, f"{type(exc).__name__}: {exc}"
        return None, None, "mechanism_yaml_not_found"

    @staticmethod
    def actual_phi(row: dict, gas=None, mechanism_error=None):
        try:
            recorded = float(row.get("phi_actual", np.nan))
            if np.isfinite(recorded) and recorded >= 0:
                return recorded, None, "phi_actual_column"
            mass_keys = ("Y_NH3", "Y_H2", "Y_O2")
            mole_keys = ("X_NH3", "X_H2", "X_O2")
            if all(pd.notna(row.get(key, np.nan)) for key in mass_keys):
                # Convert mass fractions to mole-proportional amounts before
                # applying the stoichiometric coefficients.  Use the loaded
                # mechanism weights when available, with standard molecular
                # weights as a deterministic fallback for custom CSVs.
                standard_weights = {
                    "NH3": 17.03052,
                    "H2": 2.01588,
                    "O2": 31.9988,
                }
                weights = standard_weights
                if gas is not None:
                    try:
                        weights = {
                            name: float(gas.molecular_weights[gas.species_index(name)])
                            for name in ("NH3", "H2", "O2")
                        }
                    except (KeyError, IndexError, TypeError, ValueError):
                        weights = standard_weights
                nh3 = max(0.0, float(row["Y_NH3"])) / weights["NH3"]
                h2 = max(0.0, float(row["Y_H2"])) / weights["H2"]
                o2 = float(row["Y_O2"]) / weights["O2"]
                val = (0.75 * nh3 + 0.5 * h2) / o2 if o2 > 0 else np.nan
                if np.isfinite(val):
                    return float(val), None, "simple_stoich"
            if all(pd.notna(row.get(key, np.nan)) for key in mole_keys):
                nh3 = max(0.0, float(row["X_NH3"]))
                h2 = max(0.0, float(row["X_H2"]))
                o2 = float(row["X_O2"])
                val = (0.75 * nh3 + 0.5 * h2) / o2 if o2 > 0 else np.nan
                if np.isfinite(val):
                    return float(val), None, "simple_stoich"
            return float(row.get("phi", np.nan)), None, "phi_fallback"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", None

_conditions = _ConditionsHelper()


DEFAULT_SPLITS = ("train", "validation", "test")
CONDITION_FIELDS = {
    "T": ("T",),
    "P": ("P",),
    # Prefer the recorded actual value.  Older files only contain ``phi``.
    "phi": ("phi_actual", "phi"),
    # ``Y_H2`` is the first-row H2 mass fraction used by the trajectory
    # model.  ``X_H2_fuel`` is retained as a fallback for sampler exports.
    "H2": ("Y_H2", "X_H2_fuel", "X_H2"),
}

def _resolve_dataset_folder(root: Path, mechanism: str, recorded: str | Path) -> Path:
    local = root / f'trajectories_{mechanism}'
    if local.is_dir():
        return local
    path = Path(str(recorded))
    return path if path.is_absolute() else root / path


def _available_condition_columns(part: Path) -> list[str]:
    names = set(pq.ParquetFile(part).schema.names)
    columns = ["sample_id"]
    if "time" in names:
        columns.append("time")
    if "condition_group_id" in names:
        columns.append("condition_group_id")
    if "stratum_name" in names:
        columns.append("stratum_name")
    for candidates in CONDITION_FIELDS.values():
        columns.extend(column for column in candidates if column in names)
    columns.extend(column for column in ("Y_NH3", "Y_O2", "X_NH3", "X_O2") if column in names)
    return list(dict.fromkeys(columns))


def _collect_trajectory_metadata(folder: Path, gas=None, mechanism_error=None) -> tuple[dict[str, int], pd.DataFrame]:
    """Return row counts and one deterministic initial condition per trajectory."""

    counts: dict[str, int] = {}
    first_rows: dict[str, dict[str, float]] = {}
    first_times: dict[str, float | None] = {}
    for part in sorted(folder.glob("part-*.parquet")):
        columns = _available_condition_columns(part)
        for batch in pq.ParquetFile(part).iter_batches(columns=columns, batch_size=65536):
            frame = batch.to_pandas()
            if "sample_id" not in frame:
                raise ValueError(f"Missing sample_id in {part}")
            if frame["sample_id"].isna().any():
                raise ValueError(f"Null sample_id in {part}")
            grouped = frame.groupby("sample_id", sort=False, dropna=False)
            sizes = grouped.size()
            for sample_id, n_rows in sizes.items():
                key = str(sample_id)
                counts[key] = counts.get(key, 0) + int(n_rows)
            # Select the row at the minimum finite time.  This remains
            # deterministic when a trajectory spans parquet parts or a part
            # happens to contain rows out of order.
            for sample_id, group in frame.groupby("sample_id", sort=False, dropna=False):
                key = str(sample_id)
                if "time" in group:
                    times = pd.to_numeric(group["time"], errors="coerce").to_numpy(float)
                    finite = np.isfinite(times)
                    row = group.iloc[int(np.flatnonzero(finite)[np.argmin(times[finite])])] if finite.any() else group.iloc[0]
                    candidate_time = float(np.min(times[finite])) if finite.any() else None
                else:
                    row = group.iloc[0]
                    candidate_time = None
                if key in first_rows:
                    previous_time = first_times.get(key)
                    if candidate_time is None or (previous_time is not None and candidate_time >= previous_time):
                        continue
                values: dict[str, object] = {}
                for name, candidates in CONDITION_FIELDS.items():
                    value = np.nan
                    for column in candidates:
                        if column in row:
                            try:
                                value = float(row[column])
                            except (TypeError, ValueError):
                                value = np.nan
                            if np.isfinite(value):
                                break
                    values[name] = value
                stratum_name = row.get("stratum_name", "unknown")
                values["stratum_name"] = (
                    str(stratum_name).strip()
                    if pd.notna(stratum_name) and str(stratum_name).strip()
                    else "unknown"
                )
                condition_group_id = row.get("condition_group_id")
                values["condition_group_id"] = (
                    str(condition_group_id).strip()
                    if pd.notna(condition_group_id) and str(condition_group_id).strip()
                    else str(key)
                )
                actual, reason, _ = _conditions.actual_phi(row.to_dict(), gas, mechanism_error)
                if actual is not None:
                    values["phi"] = actual
                values["phi_actual_error"] = reason
                first_rows[key] = values
                first_times[key] = candidate_time
    metadata = pd.DataFrame.from_dict(first_rows, orient="index").reindex(
        columns=[*CONDITION_FIELDS, "stratum_name", "phi_actual_error", "condition_group_id"]
    )
    metadata.index.name = "sample_id"
    metadata = metadata.reindex(sorted(counts), fill_value=np.nan)
    return counts, metadata


def _condition_strata(metadata: pd.DataFrame, bins: int) -> pd.Series:
    """Build deterministic physical-layer and quantile-bin labels."""

    bins = max(1, int(bins))
    labels = pd.Series("", index=metadata.index, dtype=object)
    pieces = []
    if "stratum_name" in metadata:
        layer = metadata["stratum_name"].fillna("unknown").astype(str).str.strip()
        layer = layer.mask(layer.eq(""), "unknown")
        labels = "layer=" + layer + "|"
    for name in CONDITION_FIELDS:
        values = pd.to_numeric(metadata.get(name), errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            pieces.append(pd.Series("missing", index=metadata.index, dtype=object))
            continue
        edges = np.unique(np.quantile(finite.to_numpy(dtype=float), np.linspace(0, 1, bins + 1)))
        if len(edges) <= 1:
            codes = pd.Series(0, index=metadata.index, dtype=object)
            codes.loc[~np.isfinite(values)] = "missing"
        else:
            codes = pd.Series("missing", index=metadata.index, dtype=object)
            mask = np.isfinite(values)
            codes.loc[mask] = np.searchsorted(edges[1:-1], values.loc[mask], side="right").astype(str)
        pieces.append(codes.rename(name))
    for index, (name, piece) in enumerate(zip(CONDITION_FIELDS, pieces)):
        labels = labels + ("" if index == 0 else "|") + name + "=" + piece.astype(str)
    return labels.astype(str)


def _select_extrapolation(metadata: pd.DataFrame, fraction: float, axes: tuple[str, ...]) -> set[str]:
    """Reserve high-distance condition tails as an independent set."""

    if fraction <= 0:
        return set()
    n = len(metadata)
    count = min(n, max(1, int(round(n * fraction))))
    scores = pd.Series(0.0, index=metadata.index)
    usable = False
    for axis in axes:
        if axis not in metadata:
            continue
        values = pd.to_numeric(metadata[axis], errors="coerce")
        mask = np.isfinite(values)
        if not mask.any():
            continue
        usable = True
        ranks = values[mask].rank(method="average", pct=True)
        scores.loc[mask] = np.maximum(scores.loc[mask], (ranks - 0.5).abs() * 2.0)
    if not usable:
        raise ValueError("Cannot select extrapolation trajectories: no finite condition columns")
    return set(scores.sort_values(ascending=False, kind="mergesort").head(count).index.astype(str))


def _split_ids(ids: np.ndarray, rng: np.random.Generator, train: float, val: float,
               metadata: pd.DataFrame | None = None, strata_bins: int = 4) -> dict[str, np.ndarray]:
    ids = np.asarray(ids, dtype=object).copy()
    if metadata is None:
        rng.shuffle(ids)
        n_train = int(round(len(ids) * train))
        n_val = int(round(len(ids) * val))
        return {"train": ids[:n_train], "validation": ids[n_train:n_train+n_val], "test": ids[n_train+n_val:]}

    if "condition_group_id" in metadata:
        meta_ids = metadata.loc[ids]
        cg = meta_ids["condition_group_id"].fillna(pd.Series(ids, index=ids)).astype(str).str.strip()
        cg = cg.mask(cg.eq(""), ids.astype(str))
        if cg.nunique() < len(ids):
            group_to_ids: dict[str, list] = {}
            group_first_row: dict[str, pd.Series] = {}
            for sid, grp in zip(ids, cg):
                group_to_ids.setdefault(grp, []).append(sid)
                if grp not in group_first_row:
                    group_first_row[grp] = meta_ids.loc[sid]

            group_keys = np.asarray(list(group_to_ids.keys()), dtype=object)
            group_meta = pd.DataFrame.from_dict(group_first_row, orient="index")

            group_splits = _split_ids(
                group_keys, rng, train, val, metadata=group_meta if stratify else None,
                strata_bins=strata_bins, stratify=stratify
            )
            result = {}
            for split_name, g_keys in group_splits.items():
                split_sids = []
                for gk in g_keys:
                    split_sids.extend(group_to_ids[gk])
                result[split_name] = np.asarray(split_sids, dtype=object)
            return result

    if not stratify:
        # Randomize whole condition groups, not their member trajectories.
        return _split_ids(ids, rng, train, val, metadata=None)

    strata = _condition_strata(metadata.loc[ids], strata_bins)
    groups = {label: group.to_numpy(dtype=object) for label, group in strata.groupby(strata).groups.items()}
    for group in groups.values():
        rng.shuffle(group)
    labels = sorted(groups)
    total = len(ids)
    target_train = int(round(total * train))
    target_val = min(total - target_train, int(round(total * val)))

    def allocate(capacities, target):
        # Distribute at most one remainder seat per stratum. Random ties use
        # the supplied seed instead of correlating split labels with T/P bins.
        size = sum(capacities.values())
        quotas = {label: (capacities[label] * target / size if size else 0.0) for label in labels}
        counts = {label: int(np.floor(quotas[label])) for label in labels}
        tie_break = {label: float(rng.random()) for label in labels}
        order = sorted(labels, key=lambda label: (quotas[label] - counts[label], tie_break[label]), reverse=True)
        for label in order[:target - sum(counts.values())]:
            counts[label] += 1
        return counts

    alloc_train = allocate({label: len(groups[label]) for label in labels}, target_train)
    alloc_val = allocate({label: len(groups[label]) - alloc_train[label] for label in labels}, target_val)
    result = {"train": [], "validation": [], "test": []}
    for label in labels:
        values = groups[label]
        n_train = alloc_train[label]
        n_val = alloc_val[label]
        result["train"].extend(values[:n_train])
        result["validation"].extend(values[n_train:n_train+n_val])
        result["test"].extend(values[n_train+n_val:])
    return {name: np.asarray(values, dtype=object) for name, values in result.items()}

def split_database(database, out, seed=20260912, train=0.7, val=0.15,
                   stratify=False, strata_bins=4, extrapolation_fraction=0.0,
                   extrapolation_axes=("T", "P", "phi", "H2"), force=False):
    if not (np.isfinite(train) and np.isfinite(val) and train >= 0 and val >= 0 and train + val < 1):
        raise ValueError("ratios must be nonnegative and sum to less than 1")
    if strata_bins < 1 or not 0 <= extrapolation_fraction < 1:
        raise ValueError("invalid strata_bins/extrapolation_fraction")
    if any(axis not in CONDITION_FIELDS for axis in extrapolation_axes):
        raise ValueError("unknown extrapolation axis")
    database, out = Path(database), Path(out); out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("*_sample_ids.csv")) if out.is_dir() else []
    if (existing or (out / "split_manifest.json").exists()) and not force:
        raise FileExistsError(f"Refusing to overwrite existing split output: {out}. Use --force to overwrite.")
    rng = np.random.default_rng(seed); rows=[]; summary=[]
    condition_diagnostics = {}
    for mech, parts in _conditions.dataset_parts(database):
        folder = parts[0].parent
        gas, _, mechanism_error = _conditions.load_phi_gas(folder, database, mech) if stratify or extrapolation_fraction else (None, None, None)
        counts, metadata = _collect_trajectory_metadata(folder, gas, mechanism_error)
        condition_diagnostics[mech] = {
            "mechanism_error": mechanism_error,
            "finite_conditions": {axis: int(np.isfinite(metadata[axis]).sum()) for axis in CONDITION_FIELDS},
            "phi_actual_failure_reasons": metadata["phi_actual_error"].dropna().value_counts().to_dict(),
        }
        all_ids=np.array(sorted(counts), dtype=object)
        extrapolation = _select_extrapolation(metadata, float(extrapolation_fraction), tuple(extrapolation_axes))
        # A single selected trajectory must reserve its entire physical condition group.
        if extrapolation and "condition_group_id" in metadata:
            group_labels = metadata["condition_group_id"].fillna(
                pd.Series(metadata.index.astype(str), index=metadata.index)
            ).astype(str).str.strip()
            group_labels = group_labels.mask(group_labels.eq(""), metadata.index.astype(str))
            selected_groups = set(group_labels.loc[list(extrapolation)])
            extrapolation = set(group_labels.index[group_labels.isin(selected_groups)].astype(str))
        remaining = np.asarray([sample_id for sample_id in all_ids if str(sample_id) not in extrapolation], dtype=object)
        split_ids = _split_ids(remaining, rng, train, val, metadata=metadata,
                               strata_bins=strata_bins, stratify=stratify)
        # Fail closed rather than writing a manifest with shared trajectories or groups.
        groups = metadata["condition_group_id"].fillna(
            pd.Series(metadata.index.astype(str), index=metadata.index)
        ).astype(str).str.strip()
        groups = groups.mask(groups.eq(""), metadata.index.astype(str))
        split_groups = {name: set(groups.loc[list(vals)]) for name, vals in split_ids.items()}
        if extrapolation:
            split_groups["extrapolation"] = set(groups.loc[list(extrapolation)])
        for i, name in enumerate(split_groups):
            for other in list(split_groups)[i + 1:]:
                if split_groups[name] & split_groups[other]:
                    raise RuntimeError(f"Condition groups overlap: {name}/{other}")
        if extrapolation:
            split_ids["extrapolation"] = np.asarray(sorted(extrapolation), dtype=object)
        for split, vals in split_ids.items():
            path=out/f'{mech}_{split}_sample_ids.csv'; pd.DataFrame({'sample_id':vals}).to_csv(path,index=False)
            summary.append({'mechanism_name':mech,'split':split,'trajectories':len(vals),'rows':sum(counts[str(x)] for x in vals),'path':str(path.resolve())})
    result={'database':str(database.resolve()),'seed':seed,
            'ratios':{'train':train,'validation':val,'test':1-train-val},
            'stratify':bool(stratify), 'strata_bins':int(strata_bins),
            'extrapolation_fraction':float(extrapolation_fraction),
            'extrapolation_axes':list(extrapolation_axes), 'summary':summary,
            'condition_diagnostics':condition_diagnostics}
    (out/'split_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=True),encoding='utf-8'); pd.DataFrame(summary).to_csv(out/'split_summary.csv',index=False)
    return result

def main():
    p=argparse.ArgumentParser(description='Split complete trajectories without window leakage')
    p.add_argument('--database', default='database_Transient_V21_QSSA'); p.add_argument('--out', default='trajectory_splits_QSSA')
    p.add_argument('--seed',type=int,default=20260912); p.add_argument('--train-ratio',type=float,default=.7); p.add_argument('--val-ratio',type=float,default=.15)
    p.add_argument('--stratify', action=argparse.BooleanOptionalAction, default=True, help='Stratify trajectory IDs by initial T/P/phi/H2 quantile bins (supports --stratify / --no-stratify)')
    p.add_argument('--strata-bins',type=int,default=4,help='Number of quantile bins per condition axis within each sampler layer')
    p.add_argument('--extrapolation-fraction',type=float,default=0.0,help='Reserve this fraction as an independent condition-tail set')
    p.add_argument('--extrapolation-axes',nargs='+',default=['T','P','phi','H2'],choices=list(CONDITION_FIELDS))
    p.add_argument('--force', action='store_true', help='Overwrite existing split output')
    args=p.parse_args()
    if args.train_ratio<0 or args.val_ratio<0 or args.train_ratio+args.val_ratio>=1: p.error('ratios must be nonnegative and sum to less than 1')
    if args.strata_bins < 1: p.error('--strata-bins must be >= 1')
    if not 0 <= args.extrapolation_fraction < 1: p.error('--extrapolation-fraction must be in [0, 1)')
    print(json.dumps(split_database(args.database,args.out,args.seed,args.train_ratio,args.val_ratio,
                                    args.stratify,args.strata_bins,args.extrapolation_fraction,
                                    tuple(args.extrapolation_axes),args.force),indent=2))
if __name__=='__main__': main()
