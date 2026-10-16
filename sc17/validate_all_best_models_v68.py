"""Batch validation launcher for the Phase 45 comparison checkpoints.

The launcher keeps training artifacts and validation outputs separate. Each
invocation writes to a unique run directory and promotes it to
``diagnostic_v68`` only when every selected checkpoint validates successfully.
Use ``--base_dir``, ``--data_dir`` and ``--output_root`` when the data or model
artifacts live outside the directory containing this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import pickle
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent


def discover_default_base_dir() -> Path:
    configured = os.environ.get("V68_BASE_DIR") or os.environ.get("V67_BASE_DIR")
    if configured:
        return Path(configured).expanduser()
    for cand in (
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        Path.cwd(),
        SCRIPT_DIR.parent / "data",
    ):
        if (
            (cand / "phase69_runs").exists()
            or (cand / "phase68_runs").exists()
            or (cand / "phase45_runs").exists()
            or (cand / "v69_qssa_stabilized").exists()
            or (cand / "database_nuig").exists()
            or (cand / "database_Transient_V21_QSSA").exists()
            or (cand / "database_Transient_V21").exists()
        ):
            return cand
    return SCRIPT_DIR


BASE_DIR = discover_default_base_dir()
DATA_DIR = BASE_DIR
OUTPUT_ROOT = BASE_DIR
MECHANISM = "nuig_2024"

# V3 is the only trained model; V1/V2/V4 are validation-only ablations.
MODEL_VERSIONS = [
    "v1_ablation", "v2_ablation", "v3_full", "v3_4", "v3_5", "v68", "v68_qssa",
    "v69_qssa", "v69_no_noise", "v69_no_stability", "v69_no_microweight", "v71_qssa", "v4_ablation", "v72_dual_branch", "v72_cfd_optimized",
]
ABLATION_CONFIGS = {
    "v1_ablation": {"update_constraint": "constrained", "feedback_constraint": "raw"},
    "v2_ablation": {"update_constraint": "raw", "feedback_constraint": "constrained"},
    "v3_full": {"update_constraint": "constrained", "feedback_constraint": "constrained"},
    "v4_ablation": {"update_constraint": "raw", "feedback_constraint": "raw"},
}


BEST_MODEL_TYPES = {
    "best_total": {
        "pattern": "best_v68_net_{mechanism}.pth",
        "name": "最佳验证损失",
        "description": "总验证损失最低的模型",
    },
    "best_oracle_h128": {
        "pattern": "best_oracle_h128_epoch*.pth",
        "name": "最佳 Oracle H=128",
        "description": "单步预测 H=128 RMSE 最低",
    },
    "best_oracle_avg": {
        "pattern": "best_oracle_avg_epoch*.pth",
        "name": "最佳 Oracle 平均",
        "description": "所有 horizon 平均 RMSE 最低",
    },
    "best_pure_ar": {
        "pattern": "best_pure_ar_h128_epoch*.pth",
        "name": "最佳 Pure-AR H=128",
        "description": "自回归 H=128 RMSE 最低",
    },
    "best_composite": {
        "pattern": "best_composite_epoch*.pth",
        "name": "最佳综合指标",
        "description": "多指标加权最佳",
    },
    "best_robust_raw": {
        "pattern": "best_robust_raw_epoch*.pth",
        "name": "最佳稳健 Raw",
        "description": "Raw长程无爆炸且综合守恒/状态最佳",
    },
    "latest": {
        "pattern": "latest_v68_{mechanism}.pth",
        "name": "最新模型",
        "description": "最新 checkpoint",
    },
}

# When several "best" checkpoints point to the same training epoch, their
# weights are normally identical. Keep one deterministic representative for
# bulk validation; the full six-file discovery list remains available for
# audit and targeted validation.
EPOCH_DEDUP_PRIORITY = {
    "best_robust_raw": 0,
    "best_total": 1,
    "best_composite": 2,
    "best_pure_ar": 3,
    "best_oracle_h128": 4,
    "best_oracle_avg": 5,
    "latest": 6,
}

def _resolve_default_split_dir() -> Path | None:
    for cand in (
        BASE_DIR / "trajectory_splits_QSSA",
        Path.cwd() / "trajectory_splits_QSSA",
        BASE_DIR / "trajectory_splits_v1",
        Path.cwd() / "trajectory_splits_v1",
    ):
        if cand.is_dir():
            return cand
    return None

VALIDATION_PARAMS = {
    "horizons": [1, 2, 4, 8, 16, 32, 64, 128],
    "batch_size": 2048,
    "split": "val",
    "split_dir": _resolve_default_split_dir(),
    "diagnostic": True,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the best checkpoints for all V68 plans"
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=BASE_DIR,
        help="Directory containing phase68_runs and the validation script.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Directory containing database_Transient_V21_QSSA; defaults to --base_dir.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Root for diagnostic_v68_runs and diagnostic_v68; defaults to --base_dir.",
    )
    parser.add_argument("--mechanism", default=MECHANISM)
    parser.add_argument("--split", choices=("val", "train", "test", "all"), default=None,
                        help="Legacy single-split selector; omitted means interactive selection.")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "val", "test"),
        default=None,
        help="Run each requested split in an independent output directory; --split remains compatible.",
    )
    parser.add_argument("--split_dir", type=Path, default=None, help="Trajectory split directory")
    parser.add_argument("--batch_size", type=int, default=VALIDATION_PARAMS["batch_size"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Validation horizons to evaluate (e.g. 1 2 4 8 16 32 64 128).",
    )
    parser.add_argument("--plot_trajectories", type=int, default=12)
    parser.add_argument(
        "--trajectory_selection",
        choices=("first", "diverse", "combustion"),
        default="combustion",
    )
    parser.add_argument("--combustion_min_temperature", type=float, default=800.0)
    parser.add_argument("--combustion_max_temperature", type=float, default=3000.0)
    parser.add_argument("--steady_rate_threshold", type=float, default=0.05)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument(
        "--rebuild_scalers",
        action="store_true",
        help="Allow validation to rebuild missing scalers (not recommended for comparisons).",
    )
    parser.add_argument(
        "--state_update_mode",
        choices=("legacy", "task8_conservative_state", "qssa", "qssa_asymptotic", "dual_branch", "dual_branch_pd_v1"),
        default=None,
        help="Task8/QSSA state update strategy for validation (passed to validate_sc_v68.py).",
    )
    parser.add_argument(
        "--plan",
        choices=(
            "v3", "v34", "v35", "v68", "v68_qssa", "v69_qssa",
            "v69_no_noise", "v69_no_stability", "v69_no_microweight", "v71_qssa", "v72_dual_branch", "v72_cfd_optimized", "all",
        ),
        default=None,
        help="Preset plan to validate (e.g. v71_qssa, v69_qssa or v69_no_noise), bypassing interactive menu.",
    )
    parser.add_argument(
        "--version",
        dest="target_version",
        default=None,
        help="Target model version to validate (e.g. v69_qssa or v69_no_noise).",
    )
    parser.add_argument(
        "--choice",
        type=str,
        default=None,
        help="Menu option choice (e.g. '1', '4', '6', '7', '8') to run non-interactively.",
    )
    parser.add_argument(
        "--no_pause",
        action="store_true",
        help="Do not wait for Enter after the interactive run.",
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Discover every epoch checkpoint instead of one representative per checkpoint type.",
    )
    parser.add_argument(
        "--checkpoint_types",
        nargs="+",
        choices=tuple(BEST_MODEL_TYPES),
        default=None,
        help="Restrict discovery to selected checkpoint types (use with --all_checkpoints for epoch sweeps).",
    )
    return parser.parse_args(argv)


def configure_paths(args) -> None:
    global BASE_DIR, DATA_DIR, OUTPUT_ROOT, MECHANISM
    BASE_DIR = Path(args.base_dir).expanduser().resolve()
    DATA_DIR = (
        Path(args.data_dir).expanduser().resolve()
        if args.data_dir is not None
        else BASE_DIR
    )
    OUTPUT_ROOT = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else BASE_DIR
    )
    if args.split_dir is None:
        args.split_dir = _resolve_default_split_dir()
    else:
        args.split_dir = Path(args.split_dir).expanduser().resolve()
    MECHANISM = args.mechanism


def print_header(text, char="="):
    print(f"\n{char * 70}")
    print(text)
    print(char * 70)


def experiment_mode_for_version(version: str) -> str:
    return "constrained_delta" if ABLATION_CONFIGS[version]["update_constraint"] == "constrained" else "baseline"


def find_best_model_file(weights_dir: Path, pattern: str):
    files = sorted(weights_dir.glob(pattern), key=lambda path: path.name)
    if not files and "v68" in pattern:
        alt_pattern = pattern.replace("v68", "v67")
        files = sorted(weights_dir.glob(alt_pattern), key=lambda path: path.name)
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    def extract_epoch(filepath: Path) -> int:
        match = re.search(r"epoch[_-]?(\d+)", filepath.name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else -1

    # Prefer the highest epoch, then a deterministic filename. Modification
    # time is only a final tie-breaker for legacy files.
    return max(
        files,
        key=lambda path: (extract_epoch(path), path.stat().st_mtime, path.name),
    )


def extract_epoch_from_filename(filepath: Path) -> int | None:
    match = re.search(r"epoch[_-]?(\d+)", filepath.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_checkpoint_epoch(filepath: Path) -> int | None:
    """Read epoch metadata only when it is not encoded in the filename."""
    epoch = extract_epoch_from_filename(filepath)
    if epoch is not None:
        return epoch
    try:
        import torch

        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        except TypeError:  # Older PyTorch without weights_only.
            checkpoint = torch.load(filepath, map_location="cpu")
        if isinstance(checkpoint, dict):
            value = checkpoint.get("epoch")
            if value is None:
                value = checkpoint.get("global_epoch", checkpoint.get("current_epoch"))
            if value is not None:
                return int(value)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        ImportError,
        EOFError,
        pickle.UnpicklingError,
    ):
        pass
    return None


def deduplicate_models_by_epoch(models: list[dict], announce: bool = True) -> list[dict]:
    """Keep one representative per known epoch, preserving unknown epochs."""
    selected: list[dict] = []
    by_epoch: dict[tuple[str, int], dict] = {}
    for model in models:
        epoch = model.get("epoch")
        if epoch is None:
            selected.append(model)
            continue
        key = (str(model.get("source_version", model.get("version", ""))), int(epoch))
        current = by_epoch.get(key)
        if current is None:
            by_epoch[key] = model
            selected.append(model)
            continue
        current_priority = EPOCH_DEDUP_PRIORITY.get(current.get("type"), 99)
        new_priority = EPOCH_DEDUP_PRIORITY.get(model.get("type"), 99)
        if new_priority < current_priority:
            selected[selected.index(current)] = model
            by_epoch[key] = model
            kept, skipped = model, current
        else:
            kept, skipped = current, model
        if announce:
            print(
                f"  - 跳过重复 epoch {epoch}: {skipped.get('type_name', skipped.get('type'))}; "
                f"保留 {kept.get('type_name', kept.get('type'))}"
            )
    return selected


def create_run_directory(output_root: Path) -> Path:
    runs_root = output_root / "diagnostic_v68_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = runs_root / stamp
    suffix = 2
    while candidate.exists():
        candidate = runs_root / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def promote_successful_run(run_dir: Path, output_root: Path) -> Path:
    """Promote a complete run, backing up the previous final result."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Validation run directory not found: {run_dir}")
    final_dir = output_root / "diagnostic_v68"
    if final_dir.exists():
        backup_stamp = f"diagnostic_v68_backup_{int(time.time())}"
        backup_dir = output_root / backup_stamp
        suffix = 2
        while backup_dir.exists():
            backup_dir = output_root / f"{backup_stamp}_{suffix}"
            suffix += 1
        shutil.move(str(final_dir), str(backup_dir))
        print(f"Backed up previous results to: {backup_dir}")
    shutil.move(str(run_dir), str(final_dir))
    return final_dir


def write_run_manifest(run_dir: Path, results: list[dict], all_success: bool, requested_split: str | None = None) -> None:
    payload = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "all_success": bool(all_success),
        "requested_split": requested_split,
        "requested_splits": list(dict.fromkeys(item.get("split", requested_split) for item in results)),
        "results": [
            {
                "version": item["model"]["version"],
                "type": item["model"]["type"],
                "checkpoint": str(item["model"]["file"]),
                "epoch": item["model"].get("epoch"),
                "experiment_mode": item["model"]["experiment_mode"],
                "source_version": item["model"].get("source_version"),
                "update_constraint": item["model"].get("update_constraint"),
                "feedback_constraint": item["model"].get("feedback_constraint"),
                "success": bool(item["success"]),
                "failure_reason": item.get("failure_reason"),
                "elapsed_seconds": item["time"],
                "split": item.get("split", requested_split),
            }
            for item in results
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def inspect_physical_metrics_artifact(
    path: Path,
    expected_horizons: list[int] | tuple[int, ...],
) -> tuple[bool, str | None]:
    """Validate the Task 5 protocol, rather than only checking its filename.

    A validator can finish with a syntactically valid ``physical_metrics.json``
    whose ``available`` flag is false (for example after an import mismatch),
    so file existence alone is not evidence of a successful validation.
    """
    if not path.is_file():
        return False, "physical_metrics_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"physical_metrics_invalid_json: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return False, "physical_metrics_not_object"
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            return False, "physical_metrics_nonfinite_json_value"
    if payload.get("available") is not True:
        reason = payload.get("reason") or "available_false"
        error = payload.get("error")
        return False, f"physical_metrics_unavailable: {reason}" + (f": {error}" if error else "")
    modes = payload.get("modes")
    if not isinstance(modes, dict) or not modes:
        return False, "physical_metrics_modes_empty"
    available_horizons = {
        str(horizon)
        for horizon_map in modes.values()
        if isinstance(horizon_map, dict)
        for horizon in horizon_map
    }
    missing = [str(horizon) for horizon in expected_horizons if str(horizon) not in available_horizons]
    if missing:
        return False, f"physical_metrics_horizons_missing: {','.join(missing)}"
    if payload.get("full_physics_available") is not True:
        return False, "physical_metrics_incomplete: mass_or_element_metrics_missing"
    for mode in ("oracle", "pure_ar", "pure_ar_raw"):
        for horizon in expected_horizons:
            scopes = modes.get(mode, {}).get(str(horizon))
            if not isinstance(scopes, dict):
                return False, f"physical_metrics_horizon_missing: {mode}/{horizon}"
            for scope in ("raw", "constrained"):
                item = scopes.get(scope, {})
                if not item.get("rows", 0) or not item.get("per_species") or not item.get("mass_sum", {}).get("n", 0):
                    return False, f"physical_metrics_scope_incomplete: {mode}/{horizon}/{scope}"
                if item.get("element_closure_available") is not True or not item.get("element_closure"):
                    return False, f"physical_metrics_elements_missing: {mode}/{horizon}/{scope}"
                if item.get("nonfinite_values", 0) or item.get("true_nonfinite_values", 0):
                    return False, f"physical_metrics_nonfinite: {mode}/{horizon}/{scope}"
    return True, None


def run_validation(
    model_name,
    checkpoint_path: Path,
    model_type_name,
    model_type_key,
    *,
    artifact_root: Path | None = None,
    output_dir: Path | None = None,
    data_dir: Path | None = None,
    base_dir: Path | None = None,
    mechanism: str | None = None,
    experiment_mode: str | None = None,
    update_constraint: str | None = None,
    feedback_constraint: str | None = None,
    batch_size: int | None = None,
    horizons: list[int] | None = None,
    max_batches: int | None = None,
    diagnostic: bool | None = None,
    no_plots: bool | None = None,
    rebuild_scalers: bool = False,
    split_dir: Path | None = None,
    plot_trajectories: int = 12,
    trajectory_selection: str = "combustion",
    combustion_min_temperature: float = 800.0,
    combustion_max_temperature: float = 3000.0,
    steady_rate_threshold: float = 0.05,
    split: str | None = None,
    state_update_mode: str | None = None,
):
    """Run one validation process and return ``(success, elapsed_seconds, reason)``."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    base_dir = Path(base_dir or BASE_DIR).expanduser().resolve()
    data_dir = Path(data_dir or DATA_DIR).expanduser().resolve()
    mechanism = mechanism or MECHANISM
    artifact_root = Path(artifact_root or checkpoint_path.parent.parent).resolve()
    output_dir = Path(
        output_dir or OUTPUT_ROOT / "diagnostic_v68" / mechanism / model_type_key
    ).expanduser().resolve()
    batch_size = batch_size or VALIDATION_PARAMS["batch_size"]
    horizons = horizons or VALIDATION_PARAMS["horizons"]
    diagnostic = VALIDATION_PARAMS["diagnostic"] if diagnostic is None else diagnostic
    no_plots = False if no_plots is None else no_plots
    experiment_mode = experiment_mode or "baseline"
    # Preserve ``None`` so validate_sc_v68 can apply its mode-specific
    # defaults (baseline/raw, QSSA/constrained) in one place.

    print(f"\n{'=' * 70}")
    print(f"验证: {model_name} - {model_type_name}")
    print(f"检查点: {checkpoint_path}")
    print(f"实验模式: {experiment_mode}")
    print(f"输出目录: {output_dir}")
    if not checkpoint_path.is_file():
        print(f"验证失败: checkpoint 不存在: {checkpoint_path}")
        return False, 0.0, "checkpoint_missing"

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_script = SCRIPT_DIR / "validate_sc_v68.py"
    cmd = [
        sys.executable,
        str(validation_script),
        "--data_dir",
        str(data_dir),
        "--mechanism",
        mechanism,
        "--output_root",
        str(artifact_root),
        "--output_dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split or VALIDATION_PARAMS["split"],
        "--horizons",
        *[str(horizon) for horizon in horizons],
        "--batch_size",
        str(batch_size),
        "--experiment_mode",
        experiment_mode,
    ]
    if update_constraint is not None:
        cmd.extend(["--update_constraint", update_constraint])
    if feedback_constraint is not None:
        cmd.extend(["--feedback_constraint", feedback_constraint])
    cmd.extend([
        "--plot_trajectories",
        str(plot_trajectories),
        "--trajectory_selection",
        trajectory_selection,
        "--combustion_min_temperature",
        str(combustion_min_temperature),
        "--combustion_max_temperature",
        str(combustion_max_temperature),
        "--steady_rate_threshold",
        str(steady_rate_threshold),
    ])
    if split_dir is not None and Path(split_dir).is_dir():
        cmd.extend(["--split_dir", str(split_dir)])
    if state_update_mode is not None:
        cmd.extend(["--state_update_mode", str(state_update_mode)])
    if diagnostic:
        cmd.append("--diagnostic")
    else:
        cmd.append("--no-diagnostic")
    if no_plots:
        cmd.append("--no_plots")
    if max_batches is not None:
        cmd.extend(["--max_batches", str(max_batches)])
    if rebuild_scalers:
        cmd.append("--rebuild_scalers")

    start_time = time.time()
    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        elapsed = time.time() - start_time
        code = getattr(exc, "returncode", "n/a")
        print(f"验证失败 (耗时 {elapsed / 60:.1f} 分钟, code={code})")
        return False, elapsed, f"validation_process_failed: code={code}"
    if diagnostic:
        artifact_ok, artifact_reason = inspect_physical_metrics_artifact(
            output_dir / "physical_metrics.json", horizons
        )
        if not artifact_ok:
            elapsed = time.time() - start_time
            print(f"验证失败：Task 5 指标协议无效 ({artifact_reason})")
            return False, elapsed, artifact_reason
    elapsed = time.time() - start_time
    print(f"验证成功 (耗时 {elapsed / 60:.1f} 分钟)")
    return True, elapsed, None


def resolve_requested_splits(args, input_fn=input) -> list[str]:
    """Resolve CLI or interactive split selection to validator names."""
    if args.splits:
        values = list(args.splits)
    elif args.split:
        values = [args.split]
    else:
        print("\n请选择验证数据集：")
        print("1. 只验证 train")
        print("2. 只验证 validation")
        print("3. 只验证 test")
        print("4. 三个都验证（默认）")
        choice = input_fn("请选择 (1/2/3/4，直接回车默认 4): ").strip()
        values = {"1": ["train"], "2": ["validation"], "3": ["test"],
                  "4": ["train", "validation", "test"],
                  "": ["train", "validation", "test"]}.get(choice)
        if values is None:
            raise ValueError("无效的数据集选择，请输入 1、2、3 或 4")
    expanded = []
    for value in values:
        if value == "all":
            expanded.extend(["train", "val", "test"])
        elif value in ("validation", "val"):
            expanded.append("val")
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def normalize_requested_splits(args, input_fn=input) -> list[str]:
    return resolve_requested_splits(args, input_fn=input_fn)


def split_specific_run_dir(run_dir: Path, split: str) -> Path:
    return run_dir / f"split_{split}"


def scan_available_models(
    base_dir: Path | None = None,
    mechanism: str | None = None,
    all_checkpoints: bool = False,
    checkpoint_types: list[str] | None = None,
):
    """Scan the canonical V68/V69 checkpoints once.

    V1/V2/V4 are inference-only ablations and are expanded after scanning.
    """
    base_dir = Path(base_dir or BASE_DIR).expanduser().resolve()
    mechanism = mechanism or MECHANISM
    print_header("扫描可用的最佳模型")
    all_models = []

    version_specs = {
        "v3_full": ("phase45_runs", "v3_constrained_weighted", True),
        "v3_4": ("phase45_runs", "v3_4_soft_feedback_trust_region", False),
        "v3_5": ("phase45_runs", "v3_5_autonomous_temperature", False),
        "v68": ("phase45_runs", "v68_bounded_direct_rate", False),
        "v68_qssa": ("phase68_runs", "v68_qssa_exponential_integrator", False),
        "v69_qssa": ("phase69_runs", "v69_qssa_stabilized", False),
        "v69_no_noise": ("phase68_runs", "v69_qssa_no_noise", False),
        "v69_no_stability": ("phase68_runs", "v69_qssa_no_stability", False),
        "v69_no_microweight": ("phase68_runs", "v69_qssa_no_microweight", False),
        "v71_qssa": ("phase68_runs", "v71_qssa_asymptotic_exact", False),
        "v72_dual_branch": ("phase68_runs", "v72_dual_branch", False),
        "v72_cfd_optimized": ("phase68_runs", "v72_cfd_optimized", False),
    }
    for version, (phase_dir, dirname, expand_ablation) in version_specs.items():
        if version.startswith("v69"):
            # The launcher currently defaults to phase68_runs; phase69_runs is
            # retained as a legacy fallback when both roots are present.
            phase_dirs = ["phase68_runs", "phase69_runs"]
        else:
            phase_dirs = [phase_dir]
        candidate_dirs = []
        for root in (base_dir, base_dir.parent):
            for root_name in phase_dirs:
                candidate_dirs.append(root / root_name / dirname)
            candidate_dirs.extend(
                [root / dirname, root / "output_v69_smoke" / dirname]
            )
        for output_name in ("output2", "output1"):
            for root_name in phase_dirs:
                candidate_dirs.append(base_dir / output_name / root_name / dirname)
        candidate_dirs.append(base_dir / "output1" / phase_dir / phase_dir / dirname)
        version_dir = None
        for cand in candidate_dirs:
            if cand.is_dir():
                version_dir = cand
                break
        if version_dir is None:
            continue

        weights_dir = None
        for w_name in ("weights_v68", "weights_v67", "weights_v66", "weights"):
            cand_w = version_dir / w_name
            if cand_w.is_dir():
                weights_dir = cand_w
                break

        if weights_dir is None:
            print(f"\n{version}: 未找到 weights 目录 (已检查: {version_dir})")
            continue
        print(f"\n{version} canonical checkpoints:")
        model_specs = BEST_MODEL_TYPES.items()
        if checkpoint_types is not None:
            selected_types = set(checkpoint_types)
            model_specs = (
                (model_type, config)
                for model_type, config in BEST_MODEL_TYPES.items()
                if model_type in selected_types
            )
        for model_type, config in model_specs:
            pattern = config["pattern"].format(mechanism=mechanism)
            all_files = list(weights_dir.glob(pattern))
            if not all_files and "v68" in pattern:
                all_files = list(weights_dir.glob(pattern.replace("v68", "v67")))
            model_file = find_best_model_file(weights_dir, pattern)
            if model_file is None:
                print(f"  - {config['name']}: 未找到")
                continue
            selected_files = all_files if all_checkpoints else [model_file]
            for selected_file in selected_files:
                size_mb = selected_file.stat().st_size / (1024 * 1024)
                suffix = (
                    f" [全部 {len(all_files)} 个文件]"
                    if all_checkpoints and len(all_files) > 1
                    else (f" [从 {len(all_files)} 个文件中选择]" if len(all_files) > 1 else "")
                )
                epoch = read_checkpoint_epoch(selected_file)
                epoch_text = f", epoch={epoch}" if epoch is not None else ", epoch=unknown"
                print(f"  + {config['name']}: {selected_file.name} ({size_mb:.2f} MB{epoch_text}){suffix}")
                all_models.append(
                    {
                        "version": version,
                        "source_version": dirname,
                        "type": model_type,
                        "type_name": config["name"],
                        "description": config["description"],
                        "file": selected_file,
                        "epoch": epoch,
                        "artifact_root": version_dir,
                        "experiment_mode": "qssa" if version in {
                            "v68_qssa", "v69_qssa", "v69_no_noise",
                            "v69_no_stability", "v69_no_microweight", "v71_qssa", "v72_dual_branch", "v72_cfd_optimized",
                        } else None,
                        "state_update_mode": (
                            "dual_branch" if version == "v72_dual_branch" else
                            "qssa_asymptotic"
                            if version in ("v71_qssa", "v72_cfd_optimized")
                            else "qssa"
                            if version in {
                                "v68_qssa", "v69_qssa", "v69_no_noise",
                                "v69_no_stability", "v69_no_microweight",
                            }
                            else None
                        ),
                        "update_constraint": None,
                        "feedback_constraint": None,
                        "expand_ablation": expand_ablation,
                    }
                )

    # Check local weights directories in base_dir and base_dir.parent if not already covered
    for local_root in (base_dir, base_dir.parent):
        for w_name in ("weights_v68", "weights_v67"):
            local_weights = local_root / w_name
            if local_weights.is_dir() and not any(m.get("file").parent == local_weights for m in all_models):
                found_local = False
                model_specs = BEST_MODEL_TYPES.items()
                if checkpoint_types is not None:
                    selected_types = set(checkpoint_types)
                    model_specs = (
                        (model_type, config)
                        for model_type, config in BEST_MODEL_TYPES.items()
                        if model_type in selected_types
                    )
                for model_type, config in model_specs:
                    pattern = config["pattern"].format(mechanism=mechanism)
                    model_file = find_best_model_file(local_weights, pattern)
                    if model_file is None:
                        continue
                    if not found_local:
                        print(f"\nlocal ({w_name}) checkpoints:")
                        found_local = True
                    all_files = list(local_weights.glob(pattern))
                    if not all_files and "v68" in pattern:
                        all_files = list(local_weights.glob(pattern.replace("v68", "v67")))
                    selected_files = all_files if all_checkpoints else [model_file]
                    for selected_file in selected_files:
                        size_mb = selected_file.stat().st_size / (1024 * 1024)
                        suffix = (
                            f" [全部 {len(all_files)} 个文件]"
                            if all_checkpoints and len(all_files) > 1
                            else (f" [从 {len(all_files)} 个文件中选择]" if len(all_files) > 1 else "")
                        )
                        epoch = read_checkpoint_epoch(selected_file)
                        epoch_text = f", epoch={epoch}" if epoch is not None else ", epoch=unknown"
                        print(f"  + {config['name']}: {selected_file.name} ({size_mb:.2f} MB{epoch_text}){suffix}")
                        all_models.append(
                            {
                                "version": "v68_qssa",
                                "source_version": "local",
                                "type": model_type,
                                "type_name": config["name"],
                                "description": config["description"],
                                "file": selected_file,
                                "epoch": epoch,
                                "artifact_root": local_root,
                                "experiment_mode": "qssa",
                                "state_update_mode": "qssa",
                                "update_constraint": None,
                                "feedback_constraint": None,
                                "expand_ablation": False,
                            }
                        )
    return all_models


def expand_ablation_models(canonical_models: list[dict]) -> list[dict]:
    """Expand canonical checkpoints into inference-only ablation configs."""
    expanded = []
    for model in canonical_models:
        if not model.get("expand_ablation", model.get("version") == "v3_full"):
            expanded.append(dict(model))
            continue
        for version, config in ABLATION_CONFIGS.items():
            item = dict(model)
            item.update(
                {
                    "version": version,
                    "update_constraint": config["update_constraint"],
                    "feedback_constraint": config["feedback_constraint"],
                    "experiment_mode": experiment_mode_for_version(version),
                }
            )
            expanded.append(item)
    return expanded


def run_selected_models(
    models: list[dict],
    run_dir: Path,
    *,
    data_dir: Path,
    base_dir: Path,
    mechanism: str,
    batch_size: int,
    max_batches: int | None,
    no_plots: bool,
    rebuild_scalers: bool,
    plot_trajectories: int,
    trajectory_selection: str,
    combustion_min_temperature: float,
    combustion_max_temperature: float,
    steady_rate_threshold: float,
    quick: bool = False,
    split_dir: Path | None = None,
    split: str = "val",
    state_update_mode: str | None = None,
):
    results = []
    for index, model in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}]")
        key = f"{model['version']}_{model['type']}"
        model_output_dir = run_dir / mechanism / key
        success, elapsed, failure_reason = run_validation(
            f"{model['version']} - {model['type_name']}",
            model["file"],
            model["description"],
            key,
            artifact_root=model["artifact_root"],
            output_dir=model_output_dir,
            data_dir=data_dir,
            base_dir=base_dir,
            mechanism=mechanism,
            experiment_mode=model["experiment_mode"],
            update_constraint=model["update_constraint"],
            feedback_constraint=model["feedback_constraint"],
            batch_size=64 if quick else batch_size,
            horizons=[128] if quick else VALIDATION_PARAMS["horizons"],
            max_batches=10 if quick else max_batches,
            diagnostic=VALIDATION_PARAMS["diagnostic"],
            no_plots=True if quick else no_plots,
            rebuild_scalers=rebuild_scalers,
            plot_trajectories=3 if quick else plot_trajectories,
            trajectory_selection=trajectory_selection,
            combustion_min_temperature=combustion_min_temperature,
            combustion_max_temperature=combustion_max_temperature,
            steady_rate_threshold=steady_rate_threshold,
            split_dir=split_dir,
            split=split,
            state_update_mode=model.get("state_update_mode") or state_update_mode,
        )
        results.append({
            "model": model,
            "success": success,
            "failure_reason": failure_reason,
            "time": elapsed,
        })
    return results


def print_results_summary(results: list[dict], title: str) -> bool:
    print_header(title)
    if not results:
        print("没有选中的验证任务")
        return False
    success_count = sum(1 for item in results if item["success"])
    all_success = success_count == len(results)
    print(f"成功: {success_count}/{len(results)}")
    for item in results:
        model = item["model"]
        status = "OK" if item["success"] else "FAILED"
        reason = f" ({item.get('failure_reason')})" if item.get("failure_reason") else ""
        print(f"[{status}] {model['version']} - {model['type_name']}{reason}")
    return all_success


def finish(no_pause: bool) -> None:
    if not no_pause:
        input("\n按 Enter 键退出...")


def main(argv=None):
    args = parse_args(argv)
    if args.horizons:
        VALIDATION_PARAMS["horizons"] = list(args.horizons)
    requested_splits = resolve_requested_splits(args)
    VALIDATION_PARAMS["split"] = requested_splits[0] if len(requested_splits) == 1 else "all"
    configure_paths(args)
    print_header("Phase 45 最佳模型验证工具")
    print(f"代码/模型根目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"结果根目录: {OUTPUT_ROOT}")

    canonical_models = scan_available_models(
        BASE_DIR,
        MECHANISM,
        all_checkpoints=args.all_checkpoints,
        checkpoint_types=args.checkpoint_types,
    )
    if not canonical_models:
        print("未找到任何可用的模型文件")
        finish(args.no_pause)
        return 1

    print(f"\n共发现 {len(canonical_models)} 个最佳模型文件")
    if args.all_checkpoints:
        unique_epoch_models = canonical_models
        print("已启用 --all_checkpoints：保留每个 checkpoint 文件，不按 epoch 去重")
    else:
        unique_epoch_models = deduplicate_models_by_epoch(canonical_models)
        print(
            f"按已知 epoch 去重后，批量验证 {len(unique_epoch_models)} 个代表模型 "
            f"（未知 epoch 的文件保留，不强行合并）"
        )
    print_header("验证选项")
    print("1. 验证所有找到的最佳模型")
    print("2. 仅验证 V3 的所有最佳模型")
    print("3. 选择特定版本和模型类型")
    print("4. 快速测试最佳验证损失模型（H=128，10 batches）")
    print("5. 验证 V3.4 和 V3.5 的所有唯一 epoch 代表")
    print("6. 验证 V68 / V69 全系（含消融）的所有唯一 epoch 代表")
    print("7. 仅验证 V68 QSSA 的所有唯一 epoch 代表")
    print("8. 仅验证 V69 QSSA 的所有唯一 epoch 代表")
    print("9. 仅验证 V69 无噪声消融的所有唯一 epoch 代表")
    print("10. 仅验证 V69 无稳定性损失消融的所有唯一 epoch 代表")
    print("11. 仅验证 V69 无微观加权消融的所有唯一 epoch 代表")
    print("12. 仅验证 V71 QSSA Asymptotic 的所有唯一 epoch 代表")
    print("0. 退出")
    choice = args.choice
    if choice is None and (args.plan or args.target_version):
        target = args.plan or args.target_version
        if target in ("v72_dual_branch", "v72_cfd_optimized"):
            choice = "13"
        elif target in ("v71_qssa", "v71"):
            choice = "12"
        elif target in ("v69_qssa", "v69"):
            choice = "8"
        elif target == "v69_no_noise":
            choice = "9"
        elif target == "v69_no_stability":
            choice = "10"
        elif target == "v69_no_microweight":
            choice = "11"
        elif target in ("v68_qssa",):
            choice = "7"
        elif target in ("v68",):
            choice = "6"
        elif target in ("v34", "v35"):
            choice = "5"
        elif target in ("v3", "v3_full"):
            choice = "2"
        elif target == "all":
            choice = "1"
    if choice is None:
        choice = input("\n请选择 (1/2/3/4/5/6/7/8/9/10/11/12/0): ").strip()

    if choice == "0":
        finish(args.no_pause)
        return 0
    if choice == "1":
        selected_models = expand_ablation_models(unique_epoch_models)
        title = (
            f"验证所有 {len(selected_models)} 个消融配置 "
            f"（{len(unique_epoch_models)} 个唯一 epoch 代表，发现 {len(canonical_models)} 个权重）"
        )
        quick = False
    elif choice == "2":
        selected_models = [
            dict(model, version="v3_full", update_constraint="constrained",
                 feedback_constraint="constrained", experiment_mode="constrained_delta")
            for model in unique_epoch_models
            if model.get("version") == "v3_full"
        ]
        title = f"验证 V3 的所有唯一 epoch 代表 ({len(selected_models)} 个；发现 {len(canonical_models)} 个)"
        quick = False
    elif choice == "3":
        expanded_models = expand_ablation_models(unique_epoch_models)
        all_known = list(MODEL_VERSIONS)
        versions = [v for v in all_known if any(m["version"] == v for m in expanded_models)]
        for m in expanded_models:
            if m["version"] not in versions:
                versions.append(m["version"])
        if not versions:
            versions = all_known
        print("\n可用版本:")
        for index, version in enumerate(versions, 1):
            count = sum(model["version"] == version for model in expanded_models)
            print(f"  {index}. {version} ({count} 个模型)")
        try:
            version_index = int(input(f"选择版本 (1-{len(versions)}): ").strip()) - 1
            selected_version = versions[version_index]
            version_models = [
                model for model in expanded_models
                if model["version"] == selected_version
            ]
            print(f"\n{selected_version} 的可用模型:")
            for index, model in enumerate(version_models, 1):
                print(f"  {index}. {model['type_name']} - {model['description']}")
            model_index = int(input(f"选择模型 (1-{len(version_models)}): ").strip()) - 1
            selected_models = [version_models[model_index]]
            title = f"验证 {selected_models[0]['version']} 的一个模型"
            quick = False
        except (ValueError, IndexError):
            print("无效的选择")
            finish(args.no_pause)
            return 2
    elif choice == "4":
        selected_models = [
            dict(
                model,
                version="v3_full",
                update_constraint="constrained",
                feedback_constraint="constrained",
                experiment_mode="constrained_delta",
            )
            for model in canonical_models
            if model["type"] == "best_total" and model.get("version") == "v3_full"
        ]
        if not selected_models:
            selected_models = [
                dict(model)
                for model in canonical_models
                if model["type"] == "best_total" and model.get("version") in {
                    "v68", "v68_qssa", "v69_qssa", "v69_no_noise",
                    "v69_no_stability", "v69_no_microweight",
                }
            ]
        if not selected_models:
            selected_models = [
                dict(model)
                for model in canonical_models
                if model["type"] == "best_total"
            ]
        if not selected_models:
            print("未找到任何可用的最佳验证损失 (best_v68_net) 模型文件")
            finish(args.no_pause)
            return 1
        ver_name = selected_models[0].get("version", "V68")
        title = f"快速测试 {ver_name} - 最佳验证损失模型"
        quick = True
    elif choice == "5":
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") in {"v3_4", "v3_5"}
        ]
        title = f"验证 V3.4/V3.5 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    elif choice == "6":
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") in {
                "v68", "v68_qssa", "v69_qssa", "v69_no_noise",
                "v69_no_stability", "v69_no_microweight", "v71_qssa",
            }
        ]
        title = f"验证 V68 / V69 / V71 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    elif choice == "7":
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") == "v68_qssa"
        ]
        title = f"验证 V68 QSSA 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    elif choice == "8":
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") == "v69_qssa"
        ]
        title = f"验证 V69 QSSA 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    elif choice in {"9", "10", "11"}:
        selected_version = {
            "9": "v69_no_noise",
            "10": "v69_no_stability",
            "11": "v69_no_microweight",
        }[choice]
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") == selected_version
        ]
        title = f"验证 {selected_version} 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    elif choice == "13":
        selected_models = [model for model in unique_epoch_models
                           if model.get("version") == (args.plan or args.target_version or "v72_dual_branch")]
        title = f"验证 V72 ({len(selected_models)} 个)"
        quick = False
    elif choice == "12":
        selected_models = [
            model for model in unique_epoch_models
            if model.get("version") == "v71_qssa"
        ]
        title = f"验证 V71 QSSA Asymptotic 的所有唯一 epoch 代表 ({len(selected_models)} 个)"
        quick = False
    else:
        print("无效的选择")
        finish(args.no_pause)
        return 2

    if not selected_models:
        print("没有选中的验证任务")
        finish(args.no_pause)
        return 1

    run_dir = create_run_directory(OUTPUT_ROOT)
    print(f"\n本次运行目录: {run_dir}")
    results = []
    split_success = {}
    for requested_split in requested_splits:
        split_run_dir = split_specific_run_dir(run_dir, requested_split)
        split_run_dir.mkdir(parents=True, exist_ok=True)
        split_results = run_selected_models(
            selected_models, split_run_dir, data_dir=DATA_DIR, base_dir=BASE_DIR,
            mechanism=MECHANISM, batch_size=args.batch_size, max_batches=args.max_batches,
            no_plots=args.no_plots, rebuild_scalers=args.rebuild_scalers,
            plot_trajectories=args.plot_trajectories, trajectory_selection=args.trajectory_selection,
            combustion_min_temperature=args.combustion_min_temperature,
            combustion_max_temperature=args.combustion_max_temperature,
            steady_rate_threshold=args.steady_rate_threshold, quick=quick,
            split_dir=args.split_dir, split=requested_split,
            state_update_mode=args.state_update_mode,
        )
        for item in split_results:
            item["split"] = requested_split
        split_success[requested_split] = print_results_summary(split_results, f"{title} | split={requested_split}")
        write_run_manifest(split_run_dir, split_results, split_success[requested_split], requested_split=requested_split)
        results.extend(split_results)
    all_success = bool(results) and all(split_success.values())
    write_run_manifest(run_dir, results, all_success, requested_split="all" if len(requested_splits) > 1 else requested_splits[0])

    if all_success:
        final_dir = promote_successful_run(run_dir, OUTPUT_ROOT)
        manifest_path = final_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_dir"] = str(final_dir)
        manifest["promoted_at"] = datetime.now().isoformat(timespec="seconds")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # The run directory is moved into its promoted location after the
        # per-split manifests are written.  Repair those paths so every
        # manifest remains self-consistent after promotion.
        for requested_split in requested_splits:
            child_manifest_path = final_dir / f"split_{requested_split}" / "run_manifest.json"
            if not child_manifest_path.is_file():
                continue
            child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
            child_manifest["run_dir"] = str(final_dir / f"split_{requested_split}")
            child_manifest_path.write_text(
                json.dumps(child_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(f"\n验证结果已整理到: {final_dir}")
        return_code = 0
    else:
        print(
            "\n本次验证未全部成功，未覆盖旧的 diagnostic_v68。"
            f" 不完整结果保留在: {run_dir}"
        )
        return_code = 1
    finish(args.no_pause)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())


