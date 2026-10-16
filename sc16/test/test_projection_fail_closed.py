"""Regression tests: KL/linear projection may never return invalid Y as constrained.

Use the user's Cantera pytest environment. Unit fixtures are synthetic physical
mass-fraction matrices to isolate numerical feasibility from kinetics training.
"""
import pytest
import torch

import physical_projection_contract as ppc
import train_sc_v68 as tr


def _simplex_case(dtype=torch.float64):
    # Two species with pure, independent element mass fractions.
    A = torch.eye(2, dtype=dtype)
    target = torch.tensor([[0.35, 0.65]], dtype=dtype)
    y_macro = torch.tensor([[0.6, 0.4]], dtype=dtype)
    y_micro = torch.empty((1, 0), dtype=dtype)
    return A, target, y_macro, y_micro


@pytest.mark.parametrize("mode", ["multiplicative_kl", "weighted_linear"])
def test_projection_returns_checked_mass_elements_and_nonnegativity(mode):
    A, target, macro, micro = _simplex_case()
    proj_macro, proj_micro, diag = tr.project_physical_state(
        macro, micro, inert_sum=0.0, element_matrix=A,
        target_elements=target, projection_mode=mode,
    )
    assert proj_micro.shape == (1, 0)
    torch.testing.assert_close(proj_macro, target, rtol=0, atol=ppc.ELEMENT_ATOL)
    assert (proj_macro >= 0).all()
    assert not diag["projection_failed"]
    assert diag["projection_mass_residual"].max() <= ppc.MASS_ATOL
    assert diag["element_residual_max"].max() <= ppc.ELEMENT_ATOL


def test_extinct_species_can_be_rescued_by_affine_fallback():
    # With p0 clamped at 1e-40 and exponent capped at exp(30), KL cannot
    # populate species 0 up to 0.9 in eight steps. The independent affine
    # recovery can, and must still pass a fresh elemental audit.
    A = torch.eye(2, dtype=torch.float64)
    macro, micro, diag = tr.project_physical_state(
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        torch.empty((1, 0), dtype=torch.float64),
        inert_sum=0.0,
        element_matrix=A,
        target_elements=torch.tensor([[0.9, 0.1]], dtype=torch.float64),
    )
    assert bool(diag["projection_fallback_used"].item())
    assert diag["projection_fallback_iterations"] > 0
    torch.testing.assert_close(macro, torch.tensor([[0.9, 0.1]], dtype=torch.float64), rtol=0, atol=1e-8)
    assert not diag["projection_failed"]
    assert diag["element_residual_max"].max() <= ppc.ELEMENT_ATOL


def test_only_failed_rows_are_recovered():
    A = torch.eye(2, dtype=torch.float64)
    y = torch.tensor([[0.35, 0.65], [0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([[0.35, 0.65], [0.9, 0.1]], dtype=torch.float64)
    macro, _, diag = tr.project_physical_state(
        y, torch.empty((2, 0), dtype=torch.float64),
        inert_sum=torch.zeros(2, 1, dtype=torch.float64),
        element_matrix=A, target_elements=target,
    )
    assert not diag["projection_fallback_used"][0]
    assert diag["projection_fallback_used"][1]
    torch.testing.assert_close(macro, target, rtol=0, atol=ppc.ELEMENT_ATOL)


def test_inconsistent_element_mass_target_fails_before_projection():
    A, _, macro, micro = _simplex_case()
    with pytest.raises(ppc.ProjectionConstraintError, match="infeasible_target"):
        tr.project_physical_state(
            macro, micro, 0.0, A,
            torch.tensor([[0.6, 0.6]], dtype=torch.float64),
        )


def test_feasible_mass_but_outside_element_convex_hull_rejected():
    # Each species has at least 20% of element B: pure A cannot be produced.
    A = torch.tensor([[0.8, 0.3], [0.2, 0.7]], dtype=torch.float64)
    with pytest.raises(ppc.ProjectionConstraintError, match="infeasible_target"):
        tr.project_physical_state(
            torch.tensor([[0.5, 0.5]], dtype=torch.float64),
            torch.empty((1, 0), dtype=torch.float64),
            0.0, A, torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        )


@pytest.mark.parametrize("bad_inert", [-1e-3, 1.001, float("nan")])
def test_invalid_inert_fraction_never_clamped_into_fake_mass(bad_inert):
    A, target, macro, micro = _simplex_case()
    with pytest.raises(ppc.ProjectionConstraintError):
        tr.project_physical_state(
            macro, micro, bad_inert, A, target,
        )


def test_nonfinite_prediction_cannot_be_projected_silently():
    A, target, macro, micro = _simplex_case()
    macro = macro.clone()
    macro[0, 0] = float("inf")
    with pytest.raises(ppc.ProjectionConstraintError, match="invalid_input"):
        tr.project_physical_state(macro, micro, 0.0, A, target)


def test_no_projection_algorithm_aliases_unknown_mode_to_weighted_linear():
    A, target, macro, micro = _simplex_case()
    with pytest.raises(ValueError, match="Unsupported physical projection mode"):
        tr.project_physical_state(
            macro, micro, 0.0, A, target, projection_mode="none",
        )


def test_one_sided_element_contract_is_rejected():
    A, _, macro, micro = _simplex_case()
    with pytest.raises(ValueError, match="supplied together"):
        tr.project_physical_state(macro, micro, 0.0, element_matrix=A)


def test_pure_inert_state_has_zero_reactive_mass():
    A, _, macro, micro = _simplex_case()
    projected, _, diag = tr.project_physical_state(
        macro, micro, 1.0, A, torch.zeros(1, 2, dtype=torch.float64)
    )
    assert torch.equal(projected, torch.zeros_like(projected))
    assert not diag["projection_failed"]


def test_final_audit_cannot_accept_mass_only_fallback():
    A, target, _, _ = _simplex_case()
    bad_projected = torch.tensor([[0.65, 0.35]], dtype=torch.float64)
    with pytest.raises(ppc.ProjectionConstraintError, match="not_converged") as e:
        ppc.require_projection_valid(bad_projected, torch.zeros(1, 1), A, target)
    assert e.value.rows == [0]
    assert "element_error" in str(e.value)


def test_fallback_failure_propagates_with_batch_row(monkeypatch):
    A = torch.eye(2, dtype=torch.float64)
    def fake_unusable_fallback(y, *args, **kwargs):
        return torch.tensor([[0.5, 0.5]], dtype=y.dtype), 8
    monkeypatch.setattr(ppc, "recover_nonnegative_affine", fake_unusable_fallback)
    with pytest.raises(ppc.ProjectionConstraintError, match="not_converged") as e:
        tr.project_physical_state(
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            torch.empty((1, 0), dtype=torch.float64),
            0.0, A, torch.tensor([[0.9, 0.1]], dtype=torch.float64)
        )
    assert e.value.rows == [0]


def test_float32_conservation_budget_has_physically_sensible_tolerance():
    A, target, macro, micro = _simplex_case(torch.float32)
    macro_p, _, diagnostics = tr.project_physical_state(
        macro, micro, 0.0, A, target
    )
    assert torch.isfinite(macro_p).all()
    assert (torch.matmul(macro_p, A.T) - target).abs().max() < 1e-6
    assert diagnostics["projection_mass_residual"].max() <= ppc.MASS_ATOL


def test_existing_trace_radicals_are_positive_on_success():
    # KL can retain tiny positive species when the requested state is nearby.
    macro = torch.tensor([[0.8, 0.2 - 1e-8]], dtype=torch.float64)
    micro = torch.tensor([[1e-8]], dtype=torch.float64)
    A = torch.tensor([[1., 0., 0.5], [0., 1., 0.5]], dtype=torch.float64)
    target = torch.cat([macro, micro], -1) @ A.T
    out_macro, out_micro, diag = tr.project_physical_state(
        macro * 1.001, micro, inert_sum=0.0,
        element_matrix=A, target_elements=target,
    )
    assert out_micro[0, 0] > 0.0
    assert not diag["projection_failed"]
    assert diag["element_residual_max"].max() <= ppc.ELEMENT_ATOL
