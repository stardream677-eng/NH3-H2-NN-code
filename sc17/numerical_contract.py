"""Numerical semantics shared by training and independent deployment checks."""
import math
import torch

RUNTIME_CONTRACT = 'sc17_repaired_kinetics_v2'
INPUT_ROUNDOFF_ATOL = 1e-12

def asinh_rate_inverse(z, scale):
    """Exact inverse on the representable domain; never silently clip labels."""
    value = scale.double() * torch.sinh(z.double())
    if not torch.isfinite(value).all() or (value.abs()>torch.finfo(z.dtype).max/16).any():
        raise FloatingPointError('Macro asinh rate exceeds the representable numerical domain')
    return value.to(z.dtype)

def mass_from_log10(log_y):
    """Exact for Y<=1, linear continuation for unphysical Y>1, with a restoring gradient.

    The log candidate remains available separately. This continuation is a loss
    surrogate outside the mass-fraction domain, never a deployed physical state.
    """
    nonpositive = log_y.clamp(max=0.)
    return torch.where(log_y>0., 1.+math.log(10.)*log_y, torch.pow(10., nonpositive))

def clean_input_mass_roundoff(scaled, physical, mean, scale, indices):
    selected=physical.index_select(-1,indices)
    # Bound cancellation by the actual feature arithmetic, not a global Y floor.
    arithmetic=(scaled.index_select(-1,indices)*scale[indices]).abs()+mean[indices].abs()
    tolerance=(4*torch.finfo(scaled.dtype).eps*arithmetic+INPUT_ROUNDOFF_ATOL).clamp(max=1e-7)
    corrected=torch.where((selected<0)&(selected>=-tolerance),torch.zeros_like(selected),selected)
    return physical.index_copy(-1,indices,corrected)
