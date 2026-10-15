"""Training diagnostics, failure snapshot recorder, and gradient anomaly isolation.

This module guarantees that any non-finite gradient (NaN/Inf) or forward/backward anomaly
is immediately captured with full replay context (exact inputs, targets, model weights,
RNG state, optimizer state, and monotonic attempted_batch_idx) without silent skipping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import torch
import torch.nn as nn


def capture_rng_state() -> Dict[str, Any]:
    """Capture full RNG state across Python, NumPy, and PyTorch (CPU and CUDA)."""
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore full RNG state from captured dictionary."""
    if "python_rng" in state:
        random.setstate(state["python_rng"])
    if "numpy_rng" in state:
        np.random.set_state(state["numpy_rng"])
    if "torch_cpu_rng" in state:
        torch.set_rng_state(state["torch_cpu_rng"])
    if "torch_cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])


def check_gradient_finite_and_norm(model: nn.Module) -> Tuple[bool, float, Dict[str, Any]]:
    """Verify that all parameter gradients are finite and compute total L2 norm in float64.

    Using float64 avoids false-positive norm overflow when many parameters have large
    but finite float32 gradients.
    """
    total_norm_sq = 0.0
    non_finite_params = []
    param_diagnostics = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            if not torch.all(torch.isfinite(grad)):
                non_finite_params.append(name)
                param_diagnostics[name] = "non-finite (contains NaN or Inf)"
            else:
                p_norm = grad.to(torch.float64).norm(2).item()
                total_norm_sq += p_norm ** 2
                if p_norm > 1e4:
                    param_diagnostics[name] = f"large_norm={p_norm:.2e}"

    total_norm = float(np.sqrt(total_norm_sq))
    is_finite = (len(non_finite_params) == 0) and np.isfinite(total_norm)

    diagnostics = {
        "is_finite": is_finite,
        "total_norm_float64": total_norm,
        "non_finite_params": non_finite_params,
        "large_grad_params": param_diagnostics,
    }
    return is_finite, total_norm, diagnostics


def save_failure_snapshot(
    snapshot: Dict[str, Any],
    snapshot_dir: str | Path = "diagnostics/failure_snapshots",
) -> Path:
    """Save failure snapshot atomically to disk."""
    sdir = Path(snapshot_dir)
    sdir.mkdir(parents=True, exist_ok=True)

    batch_idx = snapshot.get("attempted_batch_idx", 0)
    epoch = snapshot.get("epoch", 0)
    filename = f"failure_snapshot_epoch{epoch}_batch{batch_idx}.pt"
    out_path = sdir / filename
    tmp_path = out_path.with_suffix(".tmp")

    torch.save(snapshot, tmp_path)
    tmp_path.replace(out_path)
    return out_path


@torch.no_grad()
def clip_gradients_stable(model: nn.Module, max_norm: float = 1.0, total_norm: float | None = None) -> float:
    """Scale with the float64 norm; never recompute a float32 sum of squares."""
    if total_norm is None:
        finite, total_norm, _ = check_gradient_finite_and_norm(model)
        if not finite:
            raise ValueError("Cannot clip non-finite gradients")
    if not np.isfinite(total_norm) or not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("Gradient norm and positive clipping limit must be finite")
    factor = min(1.0, max_norm / max(total_norm, 1e-300))
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)
    return total_norm


def replay_snapshot_with_anomaly_detection(
    snapshot_path: str | Path,
    model: nn.Module,
    forward_fn: Any,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Replay a saved failure snapshot under torch.autograd.detect_anomaly to pinpoint failure operation."""
    ckpt = torch.load(str(snapshot_path), map_location=device, weights_only=False)
    restore_rng_state(ckpt["rng_state"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    for name, module in model.named_modules():
        module.training = ckpt.get("module_training", {}).get(name, module.training)
    model.zero_grad(set_to_none=True)

    batch_X = ckpt["batch_X"].to(device)
    batch_Y = ckpt["batch_Y"].to(device) if ckpt.get("batch_Y") is not None else None

    anomaly_detected = False
    anomaly_msg = None

    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            loss = forward_fn(model, batch_X, batch_Y)
            loss.backward()
    except Exception as exc:
        anomaly_detected = True
        anomaly_msg = f"{type(exc).__name__}: {exc}"

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_msg": anomaly_msg,
        "attempted_batch_idx": ckpt.get("attempted_batch_idx"),
        "epoch": ckpt.get("epoch"),
    }


class TrainingDiagnosticTracker:
    """Monitors training batches, records snapshots on anomaly, and enforces skip budgets."""

    def __init__(
        self,
        snapshot_dir: str | Path = "diagnostics/failure_snapshots",
        max_skip_budget: int = 20,
        debug_mode: bool = False,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.max_skip_budget = max_skip_budget
        self.debug_mode = debug_mode
        self.attempted_batch_idx = 0
        self.skipped_batches = 0
        self.successful_batches = 0
        self.batch_rng = None
        self.batch_buffers = {}
        self.module_training = {}

    def step_attempted(self, model: Optional[nn.Module] = None) -> int:
        """Increment monotonic attempted batch index."""
        self.attempted_batch_idx += 1
        self.batch_rng = capture_rng_state()
        self.batch_buffers = {} if model is None else {
            name: buf.detach().cpu().clone() for name, buf in model.named_buffers()
        }
        self.module_training = {} if model is None else {
            name: module.training for name, module in model.named_modules()
        }
        return self.attempted_batch_idx

    def record_success(self) -> None:
        self.successful_batches += 1

    def record_failure_and_check_budget(
        self,
        model: nn.Module,
        batch_X: torch.Tensor,
        batch_Y: Optional[torch.Tensor] = None,
        sample_ids: Optional[Sequence[Any]] = None,
        window_start: Optional[int] = None,
        epoch: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        error_msg: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
        replay_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Record failure snapshot and check whether skip budget is exceeded."""
        self.skipped_batches += 1

        snapshot = {
            "attempted_batch_idx": self.attempted_batch_idx,
            "epoch": epoch,
            "sample_ids": list(sample_ids) if sample_ids is not None else None,
            "window_start": window_start,
            "batch_X": batch_X.detach().cpu(),
            "batch_Y": batch_Y.detach().cpu() if batch_Y is not None else None,
            "model_state_dict": {k: self.batch_buffers.get(k, v.detach().cpu()).clone()
                                 for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "rng_state": self.batch_rng if self.batch_rng is not None else capture_rng_state(),
            "rng_captured_before_batch": self.batch_rng is not None,
            "module_training": self.module_training,
            "replay_context": replay_context or {},
            "error_msg": error_msg,
            "diagnostics": diagnostics or {},
        }

        saved_path = save_failure_snapshot(snapshot, self.snapshot_dir)
        print(
            f"[DIAGNOSTICS] Non-finite batch recorded to {saved_path.name} "
            f"(Attempted #{self.attempted_batch_idx}, Total Skipped: {self.skipped_batches})"
        )

        if self.debug_mode:
            raise RuntimeError(
                f"Debug mode halt: Non-finite gradient encountered at attempted batch {self.attempted_batch_idx}. "
                f"Snapshot saved to {saved_path}"
            )

        if self.skipped_batches > self.max_skip_budget:
            raise RuntimeError(
                f"Training failed: Exceeded maximum skip budget ({self.skipped_batches} > {self.max_skip_budget}) "
                f"at attempted batch {self.attempted_batch_idx}. Last snapshot: {saved_path}"
            )

        return saved_path
