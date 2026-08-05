#!/usr/bin/env python3
"""
UMA optimization module for T-REX embedded structures.

Uses Meta's FAIR Chemistry UMA (Universal Model for Atoms) as an ASE calculator
for fast geometry optimization. Drop-in alternative to xtb_optimize.py.

The OMol task consumes total charge and spin multiplicity from ``atoms.info``.

Supports:
1. Constrained optimization (frozen metal + donor atoms, Hookean haptic springs)
2. Unconstrained optimization (full relaxation, no frozen atoms)

Requirements:
    pip install fairchem-core

Usage:
    from uma_optimize import optimize_with_uma, HapticInfo

    result = optimize_with_uma(
        mol=embedded_mol,
        metal_idx=0,
        coord_atom_indices=[1, 2, 3],
        haptic_infos=[HapticInfo(ring_atom_indices=[4,5,6,7,8], hapticity=5, m_centroid_dist=1.8)],
        charge=0,
        spin=1,
    )
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Reuse data classes and utilities from xtb_optimize
from .xtb_optimize import (
    HapticInfo,
    estimate_m_c_distance,
    mol_to_arrays,
    update_mol_coords,
    write_xyz_from_mol,
    read_xyz,
)

# =============================================================================
# UMA Calculator (cached)
# =============================================================================

_UMA_CALCULATORS = {}


def _get_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

def get_uma_calculator(
    model_name: str = "uma-s-1p1",
    device: Optional[str] = None,
    task_name: str = "omol",
):
    if device is None:
        device = _get_device()

    cache_key = (model_name, device, task_name)

    if cache_key not in _UMA_CALCULATORS:
        logger.info(f"Loading UMA model '{model_name}' on {device} (task={task_name})...")
        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate import FAIRChemCalculator
        predict_unit = pretrained_mlip.get_predict_unit(model_name, device=device)
        calc = FAIRChemCalculator(predict_unit=predict_unit, task_name=task_name)
        _UMA_CALCULATORS[cache_key] = calc
        logger.info(f"UMA model loaded successfully")

    return _UMA_CALCULATORS[cache_key]
# =============================================================================
# Result Data Class
# =============================================================================

@dataclass
class UMAResult:
    """Result of UMA optimization."""
    success: bool
    optimized_mol: Optional[object]  # RDKit Mol
    optimized_coords: Optional[np.ndarray]
    energy: Optional[float]  # Final energy in eV (UMA native units)
    energy_hartree: Optional[float] = None  # Converted to Hartree for compatibility
    error_message: str = ""
    backend: str = "uma"
    model_name: str = ""
    constrained: bool = True
    # Electronic-state inputs passed to the OMol task
    charge: int = 0
    spin: int = 1


# =============================================================================
# Core Optimization
# =============================================================================

def optimize_with_uma(
    mol,
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List[HapticInfo],
    charge: int = 0,
    spin: int = 1,
    model_name: str = "uma-s-1p1",
    device: Optional[str] = None,
    haptic_force_constant: float = 0.5,
    fmax: float = 0.05,
    max_steps: int = 200,
    constrained: bool = True,
    work_dir: Optional[str] = None,
) -> UMAResult:
    """
    Optimize embedded structure with UMA.

    Args:
        mol: RDKit molecule with 3D coordinates
        metal_idx: Index of metal atom (0-based)
        coord_atom_indices: Indices of non-haptic coordinating atoms
        haptic_infos: List of HapticInfo for haptic sites
        charge: Molecular charge
        spin: Spin multiplicity
        model_name: UMA model name ("uma-s-1p1", "uma-m-1p1")
        device: "cuda", "cpu", or None for auto-detect
        haptic_force_constant: Spring constant for haptic M-C (eV/Å²)
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum optimization steps
        constrained: If True, freeze metal + donors. If False, full relaxation.
        work_dir: Working directory for log files

    Returns:
        UMAResult with optimized structure
    """
    from ase import Atoms
    from ase.optimize import LBFGS
    from ase.constraints import FixAtoms, Hookean

    # Log haptic info
    for haptic in haptic_infos:
        target = estimate_m_c_distance(haptic.hapticity, haptic.m_centroid_dist)
        logger.info(f"    Haptic η{haptic.hapticity}: d_centroid={haptic.m_centroid_dist:.2f}Å → target M-C={target:.2f}Å")

    try:
        # Get cached calculator
        calc = get_uma_calculator(model_name=model_name, device=device)

        # Convert RDKit mol to ASE Atoms
        symbols, positions = mol_to_arrays(mol)
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info['charge'] = charge
        atoms.info['spin'] = spin
        atoms.calc = calc

        # Build constraints
        if constrained:
            constraints = []

            # Fixed atoms: metal + non-haptic donors
            fixed_indices = [metal_idx] + list(coord_atom_indices)
            fixed_indices = list(set(fixed_indices))
            if fixed_indices:
                constraints.append(FixAtoms(indices=fixed_indices))

            # Hookean springs for haptic M-C bonds
            for haptic in haptic_infos:
                target_dist = estimate_m_c_distance(
                    haptic.hapticity,
                    haptic.m_centroid_dist,
                )
                for ring_idx in haptic.ring_atom_indices:
                    constraints.append(
                        Hookean(a1=metal_idx, a2=ring_idx, rt=target_dist, k=haptic_force_constant)
                    )

            if constraints:
                atoms.set_constraint(constraints)
        else:
            logger.info("  Running unconstrained UMA optimization (no frozen atoms)")

        # Optimizer
        logfile = None
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
            logfile = os.path.join(work_dir, "uma_opt.log")

        optimizer = LBFGS(atoms, logfile=logfile)
        optimizer.run(fmax=fmax, steps=max_steps)

        # Get results
        energy_ev = atoms.get_potential_energy()
        energy_hartree = energy_ev / 27.2114  # eV → Hartree
        opt_coords = atoms.get_positions()
        opt_mol = update_mol_coords(mol, opt_coords)

        return UMAResult(
            success=True,
            optimized_mol=opt_mol,
            optimized_coords=opt_coords,
            energy=energy_ev,
            energy_hartree=energy_hartree,
            model_name=model_name,
            constrained=constrained,
            charge=charge,
            spin=spin,
        )

    except Exception as e:
        import traceback
        return UMAResult(
            success=False,
            optimized_mol=None,
            optimized_coords=None,
            energy=None,
            error_message=f"{e}\n{traceback.format_exc()}",
            model_name=model_name,
            constrained=constrained,
            charge=charge,
            spin=spin,
        )


# =============================================================================
# Convenience: from embedding result
# =============================================================================

def optimize_embedding_result_uma(
    mol,
    metal_idx: int,
    metal_binding_infos: List,
    site_r: List[float],
    charge: int = 0,
    spin: int = 1,
    model_name: str = "uma-s-1p1",
    device: Optional[str] = None,
    haptic_force_constant: float = 0.5,
    constrained: bool = True,
    work_dir: Optional[str] = None,
) -> UMAResult:
    """
    Convenience function that takes embedding output directly.
    Same interface as optimize_embedding_result() from xtb_optimize.

    Args:
        mol: Embedded RDKit molecule
        metal_idx: Metal atom index
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        site_r: Original site distances from T-REX
        charge: Molecular charge supplied to the UMA OMol model
        spin: Spin multiplicity supplied to the UMA OMol model
        model_name: UMA model name
        device: Device for UMA
        haptic_force_constant: Spring constant for haptic bonds
        constrained: Whether to freeze metal + donors
        work_dir: Working directory

    Returns:
        UMAResult with optimized structure
    """
    coord_atom_indices = []
    haptic_infos = []

    for atom_idx, site_num, is_multi, is_haptic in metal_binding_infos:
        if is_haptic:
            site_idx = site_num - 1
            centroid_dist = site_r[site_idx] if site_idx < len(site_r) else 1.8

            ghost_atom = mol.GetAtomWithIdx(atom_idx)
            ring_atoms = [n.GetIdx() for n in ghost_atom.GetNeighbors()
                         if n.GetIdx() != metal_idx]

            if ring_atoms:
                haptic_infos.append(HapticInfo(
                    ring_atom_indices=ring_atoms,
                    hapticity=len(ring_atoms),
                    m_centroid_dist=centroid_dist,
                ))
        else:
            coord_atom_indices.append(atom_idx)

    return optimize_with_uma(
        mol=mol,
        metal_idx=metal_idx,
        coord_atom_indices=coord_atom_indices,
        haptic_infos=haptic_infos,
        charge=charge,
        spin=spin,
        model_name=model_name,
        device=device,
        haptic_force_constant=haptic_force_constant,
        constrained=constrained,
        work_dir=work_dir,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    parser = argparse.ArgumentParser(description="Test UMA optimization")
    parser.add_argument("xyz", help="Input XYZ file")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--metal-idx", type=int, default=0)
    parser.add_argument("--fix-atoms", type=str, default="")
    parser.add_argument("--model", type=str, default="uma-s-1p1",
                       choices=["uma-s-1p1", "uma-m-1p1"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--unconstrained", action="store_true",
                       help="Run without frozen atoms")
    parser.add_argument("--output", type=str, default="optimized_uma.xyz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Read input
    symbols, coords = read_xyz(args.xyz)

    # Create RDKit mol
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))

    for i, (sym, coord) in enumerate(zip(symbols, coords)):
        atom = Chem.Atom(sym)
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, Point3D(*coord))

    mol.AddConformer(conf, assignId=True)
    mol = mol.GetMol()

    # Parse fixed atoms
    fix_atoms = []
    if args.fix_atoms:
        fix_atoms = [int(x) for x in args.fix_atoms.split(",")]

    # Run
    result = optimize_with_uma(
        mol=mol,
        metal_idx=args.metal_idx,
        coord_atom_indices=fix_atoms,
        haptic_infos=[],
        charge=args.charge,
        spin=args.spin,
        model_name=args.model,
        device=args.device,
        constrained=not args.unconstrained,
    )

    if result.success:
        write_xyz_from_mol(result.optimized_mol, args.output,
                          f"UMA optimized ({result.model_name}), E={result.energy_hartree:.6f} Eh")
        print(f"Success! Model: {result.model_name}, E={result.energy_hartree:.6f} Eh ({result.energy:.4f} eV)")
        print(f"Constrained: {result.constrained}")
        print(f"Written to {args.output}")
    else:
        print(f"Failed: {result.error_message}")
