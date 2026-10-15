"""Chemistry contract module for NUIG/Cantera mechanisms.

Guarantees exact mathematical and physical consistency between:
1. Cantera detailed mechanism definition (atomic weights, molecular weights, species atom composition)
2. Training loss element conservation matrices
3. Physical projection layer element matrices
4. Evaluation and validation metrics element matrices
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cantera as ct
import numpy as np
import torch


def compute_element_mass_matrix(
    gas: ct.Solution,
    species_order: Sequence[str],
    preferred_elements: Optional[Sequence[str]] = ("N", "H", "O", "Ar", "He"),
) -> Tuple[List[str], np.ndarray]:
    """Compute exact elemental mass fraction matrix A from Cantera Solution.

    A has shape (n_elements, n_species).
    A[e, i] = n_{e, i} * W_e / W_i, where:
      n_{e, i}: number of atoms of element e in species i
      W_e: atomic weight of element e
      W_i: molecular weight of species i
    
    Guarantees sum_{e} A[e, i] == 1.0 to floating-point precision for all species.
    """
    all_elem_names = list(gas.element_names)
    
    # Filter elements present in species_order
    present_elements = set()
    for sp in species_order:
        for elem in all_elem_names:
            if gas.n_atoms(sp, elem) > 0:
                present_elements.add(elem)
                
    if preferred_elements is not None:
        ordered_elements = [e for e in preferred_elements if e in present_elements]
        # Append any remaining elements not in preferred list
        for e in all_elem_names:
            if e in present_elements and e not in ordered_elements:
                ordered_elements.append(e)
    else:
        ordered_elements = [e for e in all_elem_names if e in present_elements]

    if not ordered_elements:
        raise ValueError("Cannot build an element matrix: no elements found for given species")

    n_elem = len(ordered_elements)
    n_spec = len(species_order)
    A = np.zeros((n_elem, n_spec), dtype=np.float64)

    for j, sp_name in enumerate(species_order):
        sp_idx = gas.species_index(sp_name)
        if sp_idx < 0:
            raise ValueError(f"Species '{sp_name}' not found in Cantera mechanism")
        mw = gas.molecular_weights[sp_idx]
        for i, elem_name in enumerate(ordered_elements):
            n_atoms = gas.n_atoms(sp_name, elem_name)
            aw = gas.atomic_weight(elem_name)
            A[i, j] = (n_atoms * aw) / mw

    return ordered_elements, A


def get_mechanism_contract(
    mechanism_path: Union[str, Path],
    species_order: Sequence[str],
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Union[List[str], np.ndarray, torch.Tensor, str]]:
    """Build a complete chemistry contract dictionary for training & validation."""
    mech_path = Path(mechanism_path).resolve()
    if not mech_path.is_file():
        raise FileNotFoundError(f"Cantera mechanism file not found: {mech_path}")
        
    mechanism_bytes = mech_path.read_bytes()
    mechanism_sha256 = hashlib.sha256(mechanism_bytes).hexdigest()
    
    gas = ct.Solution(str(mech_path))
    element_names, A_np = compute_element_mass_matrix(gas, species_order)
    
    A_tensor = torch.tensor(A_np, dtype=dtype, device=device)
    
    return {
        "mechanism_path": str(mech_path),
        "mechanism_name": mech_path.stem.lower(),
        "mechanism_sha256": mechanism_sha256,
        "element_names": element_names,
        "species_order": list(species_order),
        "element_matrix_np": A_np,
        "element_matrix_tensor": A_tensor,
    }


def parse_legacy_element_matrix(
    output_species: Sequence[str],
    aw_dict: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Reproduce the legacy hand-coded AW matrix calculation for comparison tests."""
    from train_sc_v68 import AW as DEFAULT_AW, parse_formula
    aw = DEFAULT_AW if aw_dict is None else aw_dict
    
    preferred_elements = ["N", "H", "O", "Ar", "He"]
    formulas = [parse_formula(sp) for sp in output_species]
    elements = [e for e in preferred_elements if any(e in formula for formula in formulas)]
    
    rows = []
    for j, sp in enumerate(output_species):
        formula = formulas[j]
        mw = sum(formula.get(e, 0) * aw[e] for e in elements)
        col = [formula.get(e, 0) * aw[e] / mw for e in elements]
        rows.append(col)
        
    return elements, np.array(rows, dtype=np.float64).T
