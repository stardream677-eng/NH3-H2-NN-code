"""Fail-closed validation and nonnegative affine recovery for sc16 species projection.

The KL information projection is the primary map. If it misses the conservation
constraints, an independent Dykstra projection onto the intersection of the
nonnegative orthant and the exact affine conservation manifold may rescue the
state. The recovery is *not* a chemistry integration substep: if both solvers
fail, the caller must stop the rollout, record a snapshot, and reconsider dt
or its predicted composition. Neither solver may return an unchecked state.
"""
from __future__ import annotations

import torch


MASS_ATOL = 5e-6
ELEMENT_ATOL = 1e-6
NEGATIVE_ATOL = 0.0


class ProjectionConstraintError(RuntimeError):
    """Unusable projected state (invalid input, infeasible target, or no convergence)."""

    def __init__(self, reason: str, rows: list[int], details: str):
        self.reason = reason
        self.rows = rows
        super().__init__(
            f"Physical projection {reason} for batch rows {rows[:16]}"
            f"{'...' if len(rows) > 16 else ''}: {details}"
        )


def _rows(mask: torch.Tensor) -> list[int]:
    return mask.nonzero(as_tuple=False).reshape(-1).detach().cpu().tolist()


def prepare_projection_constraints(
    candidate: torch.Tensor,
    inert_sum: torch.Tensor,
    element_matrix: torch.Tensor | None,
    target_elements: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Check the physical target before running a numerical optimizer.

    Inert fraction is not clipped: a value greater than one must never
    silently yield a positive fictitious reactive mass. If the elemental
    targets and mass budget disagree, there is no feasible solution.
    """
    if candidate.ndim != 2 or not candidate.is_floating_point():
        raise ValueError("Candidate must have shape (batch, n_species), floating point")
    batch, count = candidate.shape
    if batch == 0 or count == 0:
        raise ValueError("Projection requires nonempty batch and species")
    if inert_sum.shape != (batch, 1):
        raise ValueError("inert_sum must have shape (batch, 1)")
    if not torch.isfinite(candidate).all() or not torch.isfinite(inert_sum).all():
        raise ProjectionConstraintError(
            "invalid_input", list(range(batch)), "non-finite candidate or inert mass fraction"
        )
    invalid_inert = (inert_sum[:, 0] < 0.0) | (inert_sum[:, 0] > 1.0)
    if invalid_inert.any():
        raise ProjectionConstraintError(
            "invalid_inert_mass", _rows(invalid_inert),
            "inert mass fraction must be in [0, 1], not silently clamped"
        )
    target_sum = 1.0 - inert_sum
    if (element_matrix is None) != (target_elements is None):
        raise ValueError("element_matrix and target_elements must be supplied together")
    if element_matrix is None:
        return None, None, target_sum

    A = torch.as_tensor(element_matrix, device=candidate.device, dtype=candidate.dtype)
    target = torch.as_tensor(target_elements, device=candidate.device, dtype=candidate.dtype)
    if A.ndim != 2 or A.shape[1] != count or A.shape[0] == 0:
        raise ValueError("Element matrix must be (n_elements, n_reactive_species)")
    if target.ndim == 1:
        target = target.unsqueeze(0).expand(batch, -1)
    if target.shape != (batch, A.shape[0]):
        raise ValueError("target_elements must have shape (batch, n_elements)")
    if not torch.isfinite(A).all() or not torch.isfinite(target).all():
        raise ProjectionConstraintError(
            "invalid_input", list(range(batch)), "element matrix or targets contain NaN/Inf"
        )
    if (A < -1e-10).any() or (A > 1.0 + 1e-6).any():
        raise ValueError("Element mass-fraction matrix must lie in [0, 1]")

    # Bounds from the convex hull of the columns of A. This is only a
    # necessary feasibility check; Dykstra's residual check handles the rest.
    A64, t64, mass64 = A.double(), target.double(), target_sum.double()
    lo = A64.min(dim=1).values.unsqueeze(0) * mass64
    hi = A64.max(dim=1).values.unsqueeze(0) * mass64
    invalid_bounds = ((t64 < lo - ELEMENT_ATOL) | (t64 > hi + ELEMENT_ATOL)).any(dim=1)
    if invalid_bounds.any():
        raise ProjectionConstraintError(
            "infeasible_target", _rows(invalid_bounds),
            "element fractions outside nonnegative simplex convex-hull bounds"
        )

    # For ordinary Cantera element mass matrices every species column sums
    # to 1, so elemental target masses must equal the reactive mass budget.
    column_sums = A64.sum(dim=0)
    if bool(torch.all((column_sums - 1.0).abs() <= 1e-5)):
        invalid_total = (t64.sum(dim=1, keepdim=True) - mass64).abs().squeeze(-1) > MASS_ATOL
        if invalid_total.any():
            raise ProjectionConstraintError(
                "infeasible_target", _rows(invalid_total),
                "sum(target_elements) disagrees with 1 - inert_sum"
            )
    return A, target, target_sum


def projection_residuals(
    state: torch.Tensor,
    inert_sum: torch.Tensor,
    element_matrix: torch.Tensor | None,
    target_elements: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Physical residuals per batch row, recomputed at the returned dtype."""
    y = state.double()
    mass = (y.sum(dim=1) + inert_sum.double().reshape(-1) - 1.0).abs()
    negativity = (-y.min(dim=1).values).clamp_min(0.0)
    finite = torch.isfinite(y).all(dim=1)
    if element_matrix is None:
        element = torch.zeros_like(mass)
    else:
        A = element_matrix.double()
        tgt = target_elements.double()
        element = (y @ A.T - tgt).abs().amax(dim=1)
    good = (
        finite
        & (mass <= MASS_ATOL)
        & (negativity <= NEGATIVE_ATOL)
        & (element <= ELEMENT_ATOL)
    )
    return {"valid": good, "mass": mass, "element": element, "negativity": negativity}


@torch.no_grad()
def recover_nonnegative_affine(
    candidate: torch.Tensor,
    element_matrix: torch.Tensor,
    target_elements: torch.Tensor,
    target_sum: torch.Tensor,
    *,
    iterations: int = 256,
) -> tuple[torch.Tensor, int]:
    """Dykstra projection onto x >= 0 and [A; 1]x = [b; mass].

    Solve in float64 to reduce loss of trace-species significance. The returned
    tensor still requires a separate physical residual check in caller dtype.
    Tiny negative roundoff is truncated before each residual test; the caller
    then rechecks *all* constraints in its actual feedback dtype. There is no
    unverified normalization after that final check.
    """
    y = candidate.double()
    batch, _ = y.shape
    A = element_matrix.to(dtype=torch.float64, device=y.device)
    ones = torch.ones((1, A.shape[1]), dtype=A.dtype, device=A.device)
    effective_A = torch.cat([A, ones], dim=0)
    effective_target = torch.cat(
        [target_elements.double(), target_sum.double()], dim=1
    )
    gram_pinv = torch.linalg.pinv(effective_A @ effective_A.T, rtol=1e-12)
    correction_orthant = torch.zeros_like(y)
    correction_affine = torch.zeros_like(y)
    x = y
    for index in range(iterations):
        before = x + correction_orthant
        nonnegative = torch.clamp_min(before, 0.0)
        correction_orthant = before - nonnegative
        before_affine = nonnegative + correction_affine
        affine = before_affine - (
            (before_affine @ effective_A.T - effective_target)
            @ gram_pinv @ effective_A
        )
        correction_affine = before_affine - affine
        x = affine
        if (index + 1) % 8 == 0:
            # The affine step may leave roundoff-sized negative entries.
            # Clamp to enforce exact nonnegativity and recheck the resulting
            # elemental/mass errors instead of accepting a tiny negative Y.
            nonnegative_candidate = x.clamp_min(0.0)
            result = projection_residuals(
                nonnegative_candidate, torch.ones_like(target_sum) - target_sum,
                A, target_elements.double(),
            )
            if bool(result["valid"].all()):
                return nonnegative_candidate.to(candidate.dtype), index + 1

    return x.clamp_min(0.0).to(candidate.dtype), iterations


def require_projection_valid(
    projected: torch.Tensor,
    inert_sum: torch.Tensor,
    element_matrix: torch.Tensor | None,
    target_elements: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Reject any unverified projection; include row indices and residuals."""
    audit = projection_residuals(
        projected, inert_sum, element_matrix, target_elements
    )
    if not bool(audit["valid"].all()):
        failed_rows = _rows(~audit["valid"])
        failed_mask = ~audit["valid"]
        max_mass = float(audit["mass"][failed_mask].max())
        max_element = float(audit["element"][failed_mask].max())
        max_negative = float(audit["negativity"][failed_mask].max())
        raise ProjectionConstraintError(
            "not_converged", failed_rows,
            f"mass_error={max_mass:.3e} (allowed {MASS_ATOL:.1e}), "
            f"element_error={max_element:.3e} (allowed {ELEMENT_ATOL:.1e}), "
            f"negative_mass={max_negative:.3e}. "
            "Reduce dt, inspect the raw predicted Y, or re-evaluate kinetics; "
            "a smaller physical integration step requires new rate evaluations"
        )
    return audit
