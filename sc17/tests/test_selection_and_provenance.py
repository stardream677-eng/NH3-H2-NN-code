import copy
import numpy as np
import pytest

import experiment_manifest as em
import species_validation_protocol as svp


def test_window_peaks_and_endpoints_affect_selection():
    true = np.array([[[0.0], [0.2], [0.1]]])
    pred = np.array([[[0.0], [0.1], [0.2]]])
    acc = svp.TrajectoryDiagnosticAccumulator(['OH'])
    acc.update(true, pred, np.array([[0.0, 1e-5, 2e-5]]))
    metrics = dict(composite_score=1.0, is_valid_candidate=True)
    scored = svp.apply_trajectory_selection(metrics, acc.finalize())
    assert scored['composite_score'] > 1.0
    assert scored['trajectory_selection_penalty'] > 0


def test_runtime_source_changes_are_rejected():
    manifest = dict(mechanism_sha256='a', species_order=['NH3'],
                    output_parameterization='dual_branch', transform_signature='x',
                    source_hashes={'train_sc_v68.py': 'original'})
    expected = copy.deepcopy(manifest)
    expected['source_hashes']['train_sc_v68.py'] = 'changed'
    compatible, errors = em.verify_checkpoint_compatibility({'manifest': manifest}, expected)
    assert not compatible
    assert any('source' in error.lower() for error in errors)


def test_complete_validation_failure_cannot_select_best():
    import complete_validation as cv
    result = cv.gate_primary(dict(composite_score=0.0, is_valid_candidate=True),
                             dict(selection_score=None, is_valid_candidate=False))
    assert not result['is_valid_candidate']
