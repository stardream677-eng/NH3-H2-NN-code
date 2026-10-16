"""Multi-Timescale Dual-Branch Kinetics Neural Network with Residual MLP Backbone.

Phase 5 Implementation for Ammonia-Hydrogen Combustion Neural Surrogate:
  1. ResMLPBlock / ResMLPBackbone: Pre-LayerNorm Residual MLP with Skip Connections,
     mitigating gradient vanishing across long-horizon (H=128) BPTT.
  2. Dual-Branch Decomposition:
     - Slow Branch (Major reactants & pollutants: NH3, H2, O2, H2O, N2, NO, N2O):
       Outputs direct reaction increments ΔY (or ΔY/dt). An optional legacy
       null-space projection can be folded from older checkpoints; current training
       leaves it disabled and applies physical constraints during state projection.
       This allows pollutant NO to accumulate from trace to major-product levels.
     - Fast Branch (Stiff chain-branching radicals: H, O, OH, HO2, NH2, HNO, etc.):
       Outputs QSSA asymptotic log-parameters (log10(Y_qssa), log10(D)), integrated via
       analytical exponential formula and stabilized by multiplicative KL projection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SLOW_SPECIES = ("H2", "H2O", "N2", "NH3", "O2", "NO", "N2O")


class ResMLPBlock(nn.Module):
    """Pre-LayerNorm Residual Multi-Layer Perceptron Block."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return res + x


class ResMLPBackbone(nn.Module):
    """Deep Residual MLP representation encoder for thermochemical states [T, P, Y, log10(dt)]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, expansion=expansion, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


class DualBranchKineticsNet(nn.Module):
    """Decoupled Multi-Timescale Kinetics Network.
    
    Splits the chemical evolution into:
      - Slow Non-Stiff Branch: Predicts physical reaction increments ΔY_slow.
      - Fast Stiff Radical Branch: Predicts QSSA asymptotic parameters (logY_qssa, logD).
    """

    def __init__(
        self,
        input_dim: int,
        slow_dim: int,
        fast_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        output_parameterization: str = "qssa_asymptotic",
        use_null_space: bool = False,
        element_matrix_slow: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        self.output_parameterization = output_parameterization
        self.use_null_space = use_null_space

        # Shared Backbone
        self.backbone = ResMLPBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )

        # Slow Head: Direct Increment for major species & NOx
        # Output: ΔY_slow (scaled)
        self.slow_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, slow_dim),
        )

        # Fast Head: QSSA asymptotic parameters for stiff radicals
        # Output: [log10(Y_qssa), log10(D)] interleaved -> 2 * fast_dim
        self.fast_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2 * fast_dim),
        )

        # Optional analytical elemental null-space projector: P_null = I - A^T (A A^T)^-1 A
        if use_null_space and element_matrix_slow is not None:
            A = element_matrix_slow.float()
            # A has shape (n_elements, slow_dim)
            A_pinv = torch.linalg.pinv(A)
            P_null = torch.eye(slow_dim, device=A.device, dtype=A.dtype) - torch.matmul(A_pinv, A)
            self.register_buffer("P_null", P_null)
        else:
            self.P_null = None

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, input_dim] or [batch, horizon, input_dim].
            return_dict: If True, returns dict with separate slow and fast outputs.
                         If False, returns concatenated tensor for backward compatibility.
        """
        features = self.backbone(x)

        # 1. Slow Branch Prediction
        delta_slow = self.slow_head(features)
        if self.P_null is not None:
            # Enforce exact mathematical elemental conservation on ΔY_slow
            delta_slow = torch.matmul(delta_slow, self.P_null.T)

        # 2. Fast Branch Prediction
        fast_qssa = self.fast_head(features)

        if return_dict:
            return {
                "slow_delta": delta_slow,
                "fast_qssa": fast_qssa,
            }

        # For seamless integration with flat loss functions:
        # Concatenate into a combined vector [slow_delta, fast_qssa]
        return torch.cat([delta_slow, fast_qssa], dim=-1)


def load_dual_branch_state_dict(model: DualBranchKineticsNet, state_dict, strict: bool = True):
    """Load a checkpoint from either projected or unprojected dual-branch models.

    Older checkpoints applied ``P_null`` after the slow head. When loading into
    a model without that runtime projection, fold the linear projection into
    the final slow-head layer so predictions remain unchanged.
    """
    legacy_projector = state_dict.get("P_null")
    model_projector = getattr(model, "P_null", None)
    if legacy_projector is not None and model_projector is None:
        adapted = dict(state_dict)
        adapted.pop("P_null")
        weight_key = "slow_head.2.weight"
        bias_key = "slow_head.2.bias"
        if weight_key not in adapted or legacy_projector.ndim != 2:
            raise ValueError("Legacy dual-branch checkpoint has an invalid P_null projector")

        weight = adapted[weight_key]
        projector = legacy_projector.to(device=weight.device, dtype=weight.dtype)
        if projector.shape != (weight.shape[0], weight.shape[0]):
            raise ValueError(
                "Legacy P_null shape does not match the slow-head output dimension: "
                f"{tuple(projector.shape)} vs {tuple(weight.shape)}"
            )
        adapted[weight_key] = projector @ weight
        if bias_key in adapted:
            bias = adapted[bias_key]
            adapted[bias_key] = projector.to(device=bias.device, dtype=bias.dtype) @ bias
        state_dict = adapted

    return model.load_state_dict(state_dict, strict=strict)


class CoupledKineticsResMLP(nn.Module):
    """Direct drop-in replacement for CoupledKineticsNet with ResMLP backbone."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        output_parameterization: str = "qssa_asymptotic",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_parameterization = output_parameterization

        self.backbone = ResMLPBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
