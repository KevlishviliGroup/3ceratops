#!/usr/bin/env python3
"""
xTB optimization module for T-REX embedded structures.

Uses the xtb-python ASE interface. The former subprocess backend was removed
because it was not supported by the package workflow.

This module provides post-embedding optimization using xTB with:
1. Fixed constraints for non-haptic coordinating atoms (exact positions from Gram matrix)
2. Soft distance constraints for haptic M-C bonds (estimated from geometry)

Requirements:
    conda install -c conda-forge xtb-python ase

Usage:
    from xtb_optimize import optimize_with_xtb, HapticInfo
    
    result = optimize_with_xtb(
        mol=embedded_mol,
        metal_idx=0,
        coord_atom_indices=[1, 2, 3],  # Non-haptic donors
        haptic_infos=[HapticInfo(ring_atom_indices=[4,5,6,7,8], hapticity=5, m_centroid_dist=1.8)],
        charge=0,
        spin=1,
    )
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Check whether the supported backend is available
HAVE_XTB_PYTHON = False
try:
    from xtb.ase.calculator import XTB
    from ase.optimize import LBFGS
    from ase.constraints import FixAtoms, Hookean
    from ase import Atoms
    HAVE_XTB_PYTHON = True
    logger.debug("Using xtb-python backend")
except ImportError:
    logger.debug("xtb-python not available")


# =============================================================================
# Haptic Geometry Estimation
# =============================================================================

def estimate_ring_radius(n_atoms: int, bond_length: float = 1.40) -> float:
    """
    Estimate ring radius for η^n coordination.
    
    For a regular n-gon with side length s:
        radius = s / (2 × sin(π/n))
    
    Special cases:
        η2 (alkene): half of C=C bond (~1.34Å)
        η3+ (allyl, Cp, arene): regular polygon formula
    
    Args:
        n_atoms: Number of atoms in haptic coordination (hapticity)
        bond_length: Average C-C bond length (default 1.40Å for aromatic)
        
    Returns:
        Ring radius in Angstroms (centroid to atom distance)
    """
    if n_atoms < 2:
        return 0.0
    elif n_atoms == 2:
        # η2 alkene: C=C bond length ~1.34Å, radius = half
        return 1.34 / 2  # 0.67Å
    else:
        # Regular polygon
        return bond_length / (2 * math.sin(math.pi / n_atoms))


def estimate_m_c_distance(n_atoms: int, m_centroid_dist: float, 
                          bond_length: float = 1.40) -> float:
    """
    Estimate M-C distance for haptic ligand using triangle geometry.
    
    The metal, centroid, and ring carbon form a right triangle:
        M-C = sqrt(M-centroid² + centroid-C²)
    
    Args:
        n_atoms: Number of atoms in haptic coordination
        m_centroid_dist: Metal to centroid distance from T-REX
        bond_length: Average C-C bond length
        
    Returns:
        Estimated M-C distance in Angstroms
    """
    ring_radius = estimate_ring_radius(n_atoms, bond_length)
    return math.sqrt(m_centroid_dist**2 + ring_radius**2)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HapticInfo:
    """Information about a haptic binding site."""
    ring_atom_indices: List[int]  # Indices of ring atoms in mol (0-based)
    hapticity: int                 # Number of coordinating atoms
    m_centroid_dist: float         # M-centroid distance from T-REX (Angstroms)


@dataclass
class XTBResult:
    """Result of xTB optimization."""
    success: bool
    optimized_mol: Optional[object]  # RDKit Mol
    optimized_coords: Optional[np.ndarray]
    energy: Optional[float]  # Final energy in Hartree
    error_message: str = ""
    backend: str = ""  # "xtb-python", optionally with fallback provenance


# =============================================================================
# Coordinate Utilities
# =============================================================================

def mol_to_arrays(mol) -> Tuple[List[str], np.ndarray]:
    """Extract symbols and coordinates from RDKit mol."""
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    
    symbols = []
    positions = []
    
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        symbols.append(atom.GetSymbol())
        positions.append([pos.x, pos.y, pos.z])
    
    return symbols, np.array(positions)


def update_mol_coords(mol, coords: np.ndarray):
    """Update RDKit mol coordinates from numpy array."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    
    mol = Chem.RWMol(mol)
    conf = mol.GetConformer()
    
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(coords[i, 0]),
                                         float(coords[i, 1]),
                                         float(coords[i, 2])))
    
    return mol.GetMol()


def write_xyz_from_mol(mol, filepath: str, comment: str = "") -> None:
    """Write RDKit mol to XYZ file."""
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    
    with open(filepath, 'w') as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            f.write(f"{symbol:2s} {pos.x:15.8f} {pos.y:15.8f} {pos.z:15.8f}\n")


def read_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    """Read XYZ file and return symbols and coordinates."""
    symbols = []
    coords = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return symbols, np.array(coords)


# =============================================================================
# Pythonic Backend (xtb-python + ASE)
# =============================================================================

def _set_xtb_electronic_state(atoms, metal_idx: int, charge: int, spin: int) -> None:
    """Store molecular charge and unpaired-electron count where xtb-python reads them.

    The ASE xTB calculator obtains the molecular charge from the sum of
    ``initial_charges`` and UHF from the sum of ``initial_magnetic_moments``.
    Assigning the totals to the metal is only a transport convention; xTB uses
    the sums and does not interpret this as a localized atomic charge or spin.
    """
    initial_charges = np.zeros(len(atoms), dtype=float)
    initial_charges[metal_idx] = float(charge)
    atoms.set_initial_charges(initial_charges)

    initial_magmoms = np.zeros(len(atoms), dtype=float)
    initial_magmoms[metal_idx] = float(spin - 1)
    atoms.set_initial_magnetic_moments(initial_magmoms)


def optimize_with_xtb_python(
    mol,
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List[HapticInfo],
    charge: int,
    spin: int,
    method: str = "GFN2-xTB",
    haptic_force_constant: float = 0.5,
    fmax: float = 0.05,
    max_steps: int = 200,
    work_dir: Optional[str] = None,
) -> XTBResult:
    """Optimize using xtb-python + ASE."""
    
    if not HAVE_XTB_PYTHON:
        return XTBResult(
            success=False,
            optimized_mol=None,
            optimized_coords=None,
            energy=None,
            error_message="xtb-python not installed",
            backend="xtb-python",
        )
    
    try:
        # Convert to ASE Atoms
        symbols, positions = mol_to_arrays(mol)
        atoms = Atoms(symbols=symbols, positions=positions)

        # xtb-python reads charge/UHF from ASE Atoms, not calculator kwargs.
        _set_xtb_electronic_state(atoms, metal_idx, charge, spin)
        
        # Set up calculator
        calc = XTB(method=method)
        atoms.calc = calc
        
        # Build constraints
        constraints = []
        
        # Fixed atoms
        fixed_indices = [metal_idx] + list(coord_atom_indices)
        fixed_indices = list(set(fixed_indices))
        if fixed_indices:
            constraints.append(FixAtoms(indices=fixed_indices))
        
        # Hookean springs for haptic M-C
        for haptic in haptic_infos:
            target_dist = estimate_m_c_distance(
                haptic.hapticity,
                haptic.m_centroid_dist
            )
            for ring_idx in haptic.ring_atom_indices:
                constraints.append(
                    Hookean(a1=metal_idx, a2=ring_idx, rt=target_dist, k=haptic_force_constant)
                )
        
        if constraints:
            atoms.set_constraint(constraints)
        
        # Optimizer
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
            logfile = os.path.join(work_dir, "opt.log")
        else:
            logfile = None
        
        optimizer = LBFGS(atoms, logfile=logfile)
        optimizer.run(fmax=fmax, steps=max_steps)
        
        # Get results
        energy = atoms.get_potential_energy() / 27.2114  # eV to Hartree
        opt_coords = atoms.get_positions()
        opt_mol = update_mol_coords(mol, opt_coords)
        
        return XTBResult(
            success=True,
            optimized_mol=opt_mol,
            optimized_coords=opt_coords,
            energy=energy,
            backend="xtb-python",
        )
        
    except Exception as e:
        import traceback
        return XTBResult(
            success=False,
            optimized_mol=None,
            optimized_coords=None,
            energy=None,
            error_message=f"{e}\n{traceback.format_exc()}",
            backend="xtb-python",
        )


# =============================================================================
# Main Interface
# =============================================================================

def optimize_with_xtb(
    mol,
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List[HapticInfo],
    charge: int,
    spin: int,
    method: Literal["GFN2-xTB", "GFN1-xTB", "GFN-FF"] = "GFN2-xTB",
    haptic_force_constant: float = 0.5,
    fmax: float = 0.05,
    max_steps: int = 200,
    work_dir: Optional[str] = None,
    fallback_to_gfnff: bool = True,
) -> XTBResult:
    """
    Optimize embedded structure with xTB.
    
    Uses xtb-python through ASE.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        metal_idx: Index of metal atom (0-based)
        coord_atom_indices: Indices of non-haptic coordinating atoms
        haptic_infos: List of HapticInfo for haptic sites
        charge: Molecular charge
        spin: Spin multiplicity (1, 2, 3, ...)
        method: xTB method
        haptic_force_constant: Spring constant for haptic M-C (eV/Å²)
        fmax: Force convergence
        max_steps: Maximum optimization steps
        work_dir: Working directory for files
        fallback_to_gfnff: If True and primary method fails, retry with GFN-FF
        
    Returns:
        XTBResult with optimized structure
    """
    # Log haptic info
    for haptic in haptic_infos:
        target = estimate_m_c_distance(haptic.hapticity, haptic.m_centroid_dist)
        logger.info(f"    Haptic η{haptic.hapticity}: d_centroid={haptic.m_centroid_dist:.2f}Å → target M-C={target:.2f}Å")
    
    # Try primary method
    result = optimize_with_xtb_python(
        mol=mol,
        metal_idx=metal_idx,
        coord_atom_indices=coord_atom_indices,
        haptic_infos=haptic_infos,
        charge=charge,
        spin=spin,
        method=method,
        haptic_force_constant=haptic_force_constant,
        fmax=fmax,
        max_steps=max_steps,
        work_dir=work_dir,
    )
    
    # If failed and fallback enabled, try GFN-FF
    if not result.success and fallback_to_gfnff and method != "GFN-FF":
        logger.warning(f"  {method} failed, falling back to GFN-FF...")
        
        # Create separate work dir for fallback
        fallback_work_dir = None
        if work_dir:
            fallback_work_dir = work_dir + "_gfnff"
        
        result = optimize_with_xtb_python(
            mol=mol,
            metal_idx=metal_idx,
            coord_atom_indices=coord_atom_indices,
            haptic_infos=haptic_infos,
            charge=charge,
            spin=spin,
            method="GFN-FF",
            haptic_force_constant=haptic_force_constant,
            fmax=fmax,
            max_steps=max_steps,
            work_dir=fallback_work_dir,
        )
        
        if result.success:
            logger.info("  GFN-FF fallback succeeded")
            # Update backend info to indicate fallback was used
            result = XTBResult(
                success=result.success,
                optimized_mol=result.optimized_mol,
                optimized_coords=result.optimized_coords,
                energy=result.energy,
                error_message=result.error_message,
                backend=result.backend + " (GFN-FF fallback)",
            )
    
    return result


# =============================================================================
# Convenience Function
# =============================================================================

def optimize_embedding_result(
    mol,
    metal_idx: int,
    metal_binding_infos: List[Tuple[int, int, bool, bool]],
    site_r: List[float],
    charge: int,
    spin: int,
    method: str = "GFN2-xTB",
    haptic_force_constant: float = 0.5,
    work_dir: Optional[str] = None,
) -> XTBResult:
    """
    Convenience function that takes embedding output directly.
    
    Args:
        mol: Embedded RDKit molecule
        metal_idx: Metal atom index
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        site_r: Original site distances from T-REX
        charge: Molecular charge
        spin: Spin multiplicity
        method: xTB method
        haptic_force_constant: Spring constant for haptic bonds
        work_dir: Working directory
        
    Returns:
        XTBResult with optimized structure
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
    
    return optimize_with_xtb(
        mol=mol,
        metal_idx=metal_idx,
        coord_atom_indices=coord_atom_indices,
        haptic_infos=haptic_infos,
        charge=charge,
        spin=spin,
        method=method,
        haptic_force_constant=haptic_force_constant,
        work_dir=work_dir,
    )