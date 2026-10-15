"""Third-review counterexamples. Failures expose remaining defects; no sources are patched."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_sc_v68 as tr
import species_validation_protocol as svp

def tools_for(mode='dual_branch_pd_v1', inert=False):
    inerts = ['AR'] if inert else []
    fi = tr.build_feature_info(inerts+['H2','H'], inerts, ['H2'], ['H'], mode, mode)
    n = len(fi['input_cols'])
    sx = tr.FeatureScaler(np.zeros(n, np.float32), np.ones(n, np.float32), 1)
    sy = tr.FeatureScaler(np.zeros(3, np.float32), np.ones(3, np.float32), 1)
    return tr.CoarseOperatorTools(fi, sx, sy, device='cpu')

def test_mass_gate_rejects_large_total_defect_even_when_individual_elements_pass():
    true = np.full((1,3), 1/3)
    metrics = svp.compute_species_trajectory_metrics(true, true-.04, ['H2','N2','O2'], element_matrix=np.eye(3))
    assert metrics['mass_closure']['max_unity_error'] > .05
    assert metrics['element_closure']['max_element_error'] < .05
    assert metrics['is_valid_candidate'] is False

@pytest.mark.parametrize('mode,z1', [('dual_branch',-16.),('dual_branch_pd_v1',-20.)])
def test_missing_states_cannot_certify_dormancy_at_moderate_d(mode,z1):
    # Both labels are compatible with Y_old=1e-4, dt=1e-4, D=1e4:
    # Y_next~=3.6788e-5. Parameter labels alone cannot imply dormancy.
    mask = tr.compute_qssa_rate_mask(tools_for(mode), torch.tensor([[0.,z1,4.]]))
    assert mask is None or mask[0,-1].item() == 1.0

def inert_validation_metrics():
    op = tools_for(inert=True)
    x = torch.tensor([[[1200.,40.,.1,.9,-20.,-6.], [1200.,40.,.1,.9,-20.,-6.]]])
    y = torch.tensor([[[0.,-20.,-10.]]])
    class Constant(torch.nn.Module):
        def forward(self, inputs):
            return inputs.new_tensor([0.,-20.,-10.]).expand(len(inputs),-1)
    stats = tr.validate_model_v68(Constant(), [(x,y)], op, 'cpu', [1], 1,
                                 mode='pure_ar', state_update_mode='dual_branch_pd_v1')
    return stats['sentinel_metrics']['1']

def test_actual_validation_counts_inert_mass_in_a_perfect_prediction():
    metrics = inert_validation_metrics()
    assert metrics['mean_major_nmae'] < 1e-6
    assert metrics['mass_closure']['max_unity_error'] < 1e-6

def test_published_workstation_flags_are_accepted():
    tr.get_training_arg_parser().parse_args([
        '--mechanism','nuig_2024','--output_parameterization','dual_branch_pd_v1',
        '--rate_decode_mode','dual_branch_pd_v1','--state_update_mode','dual_branch_pd_v1',
        '--curriculum_warmup_epochs','5','--weight_decay','1e-5'])
