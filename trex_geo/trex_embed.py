# trex_geo/trex_embed.py
"""
T-REX Structure Embedding Module.

This module faithfully replicates MetalloGen's embedding approach:
1. Parse ligand SMILES with AddHs to get complete molecules WITH HYDROGENS
2. Build combined molecule with all ligand atoms
3. Add ghost atoms for haptic sites
4. Replace metal with main-group dummy
5. Use LP solver to compute valid bond orders
6. Build RDKit mol and embed with constraints

V3 Enhancements:
- Progressive constraint relaxation (3 levels) when strict embedding fails
- Post-embedding clash detection to reject invalid geometries

Relaxation Levels (all use same coordinate constraints):
  0: Strict constrained (original MetalloGen: 100 iters, no basic knowledge)
  1: Relaxed params (500 iters, useBasicKnowledge=True, no chirality enforcement)
  2: ETKDG v3 (experimental torsion preferences from Cambridge Structural Database)

Reference: MetalloGen (BSD 3-Clause License)
https://github.com/seonghann/MetalloGen
"""
from __future__ import annotations

import logging
import copy
import sys
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict
import itertools

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Geometry import Point3D

# Import the LP solver
try:
    from . import compute_chg_and_bo_pulp as compute_pulp
except ImportError:
    # Running directly, not as package
    import compute_chg_and_bo_pulp as compute_pulp

if TYPE_CHECKING:
    from trex.model import TrexFull, Site

logger = logging.getLogger(__name__)


# =============================================================================
# Constants from MetalloGen
# =============================================================================

PERIODIC_TABLE = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
]

# From MetalloGen get_dummy_center_for_valid()
DUMMY_CENTER_BY_CN = {
    '2': 8,   # O
    '3': 15,  # P
    '4': 14,  # Si
    '5': 15,  # P
    '6': 16,  # S
}

# From MetalloGen get_dummy_center() - actual dummy used in coordMap
DUMMY_CENTER_ELEMENT = {
    '2': 8,   # O
    '3': 7,   # N  
    '4': 6,   # C
    '5': 7,   # N
    '6': 16,  # S
}

# From MetalloGen get_dummy_atom_list() - for haptic centroids
DUMMY_ATOM_BY_HAPTICITY = {
    1: [8, 47],      # O, Ag
    2: [13, 15],     # Al, P
    3: [14],         # Si
    4: [15],         # P
    5: [26, 16],     # Fe, S
    6: [26],         # Fe
}

# From MetalloGen get_dummy_center() - geometry-specific dummy for EMBEDDING
# These are used AFTER LP solver, for RDKit embedding
DUMMY_CENTER_BY_GEOMETRY = {
    '2_linear': 26,           # Fe
    '2_bent': 16,             # S  
    '2_bent_135': 16,         # S
    '2_bent_90': 16,          # S
    '3_t_shaped': 15,         # P
    '3_trigonal_planar': 13,  # Al
    '3_trigonal_pyramidal': 15,  # P
    '4_seesaw': 15,           # P
    '4_square_planar': 78,    # Pt
    '4_tetrahedral': 14,      # Si
    '5_pentagonal_planar': 46,  # Pd
    '5_square_pyramidal': 26,   # Fe
    '5_trigonal_bipyramidal': 15,  # P
    '6_hexagonal_planar': 26,   # Fe
    '6_octahedral': 26,         # Fe
    '6_pentagonal_pyramidal': 26,  # Fe
    '7_pentagonal_bipyramidal': 15,  # P  
}

def get_dummy_center_for_geometry(geometry_name: str) -> int:
    """Get dummy element atomic number for embedding based on full geometry name."""
    return DUMMY_CENTER_BY_GEOMETRY.get(geometry_name, 26)  # Default to Fe


# =============================================================================
# Van der Waals Radii for Clash Detection
# =============================================================================

VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    26: 1.40,  # Fe (smaller for coordination)
    27: 1.35,  # Co
    28: 1.35,  # Ni
    29: 1.40,  # Cu
    30: 1.39,  # Zn
    35: 1.85,  # Br
    44: 1.46,  # Ru
    45: 1.42,  # Rh
    46: 1.63,  # Pd
    47: 1.72,  # Ag
    53: 1.98,  # I
    74: 1.37,  # W
    77: 1.41,  # Ir
    78: 1.72,  # Pt
    79: 1.66,  # Au
    80: 1.55,  # Hg
}

def get_vdw_radius(atomic_num: int) -> float:
    """Get van der Waals radius for an element."""
    return VDW_RADII.get(atomic_num, 1.70)  # Default to carbon


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Atom:
    """Simple atom class like MetalloGen's chem.Atom."""
    atomic_number: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    formal_charge: int = 0
    
    def copy(self):
        return Atom(
            atomic_number=self.atomic_number,
            x=self.x, y=self.y, z=self.z,
            formal_charge=self.formal_charge
        )
    
    @property
    def symbol(self) -> str:
        if 1 <= self.atomic_number <= len(PERIODIC_TABLE):
            return PERIODIC_TABLE[self.atomic_number - 1]
        return "X"


@dataclass
class Ligand:
    """Ligand with atoms, bonds, and binding info."""
    atom_list: list[Atom]
    bo_matrix: np.ndarray  # Bond order matrix
    adj_matrix: np.ndarray  # Adjacency matrix
    chg_list: np.ndarray   # Formal charges
    charge: int            # Total charge
    binding_infos: list[tuple[list[int], int]]  # (local_atom_indices, site_number)


@dataclass
class ClashResult:
    """Result of post-embedding clash detection."""
    has_severe_clash: bool  # Any pair < severe_threshold × sum_vdw
    has_moderate_clash: bool  # Any pair < moderate_threshold × sum_vdw
    n_severe_clashes: int
    n_moderate_clashes: int
    min_distance: float
    min_distance_pair: Tuple[int, int]
    min_distance_ratio: float  # distance / sum_vdw_radii
    clash_details: List[dict] = field(default_factory=list)


@dataclass
class EmbedResult:
    """Result of embedding."""
    mol: Optional[Chem.Mol]
    success: bool
    metal_idx: int
    coord_atom_indices: list[int]
    charge: int
    spin: int
    diagnostics: dict
    embedding_level: int = 0  # Which relaxation level succeeded (0-4)
    embedding_level_name: str = "strict_constrained"  # Human-readable level name
    clash_result: Optional[ClashResult] = None  # Post-embedding clash detection


class Molecule:
    """
    Simple Molecule class compatible with MetalloGen's LP solver.
    
    Provides the interface expected by compute_chg_and_bo_pulp.py
    """
    def __init__(self, atom_list: list[Atom], adj_matrix: np.ndarray):
        self.atom_list = atom_list
        self.adj_matrix = adj_matrix.astype(float)
        self.bo_matrix = None
        self.atom_feature = {'chg': np.zeros(len(atom_list))}
        self.chg = 0
        self.multiplicity = 1
    
    def get_z_list(self) -> np.ndarray:
        return np.array([a.atomic_number for a in self.atom_list])
    
    def get_period_group_list(self) -> tuple[np.ndarray, np.ndarray]:
        """Return period and group for each atom."""
        z_list = self.get_z_list()
        n = len(z_list)
        period_list = np.zeros(n, dtype=int)
        group_list = np.zeros(n, dtype=int)
        
        for i, z in enumerate(z_list):
            period_list[i], group_list[i] = get_period_and_group(z)
        
        return period_list, group_list
    
    def get_matrix(self, matrix_type: str) -> np.ndarray:
        if matrix_type == "adj":
            return self.adj_matrix
        elif matrix_type == "bo":
            return self.bo_matrix
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
    
    def get_adj_matrix(self) -> np.ndarray:
        return self.adj_matrix
    
    def get_neighbor_list(self) -> list[list[int]]:
        """Return list of neighbor indices for each atom."""
        n = len(self.atom_list)
        neighbor_list = []
        for i in range(n):
            neighbors = []
            for j in range(n):
                if self.adj_matrix[i, j] > 0 and i != j:
                    neighbors.append(j)
            neighbor_list.append(neighbors)
        return neighbor_list
    
    def get_bond_list(self, include_self: bool = False) -> list[tuple[int, int]]:
        """Return list of (i, j) bond pairs with i < j."""
        n = len(self.atom_list)
        bond_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if self.adj_matrix[i, j] > 0:
                    bond_list.append((i, j))
        return bond_list
    
    def get_chg(self) -> int:
        return self.chg
    
    def get_max_valency_list(self) -> np.ndarray:
        """Return maximum valency for each atom."""
        z_list = self.get_z_list()
        period_list, group_list = self.get_period_group_list()
        n = len(z_list)
        max_val = np.zeros(n, dtype=int)
        for i in range(n):
            if period_list[i] == 1:
                max_val[i] = 1
            elif period_list[i] == 2:
                max_val[i] = 4
            else:
                max_val[i] = 6  # Allow expanded octet
        return max_val
    
    def set_bo_matrix(self, bo_matrix: np.ndarray):
        self.bo_matrix = bo_matrix


def get_period_and_group(z: int) -> tuple[int, int]:
    """Get period and group from atomic number."""
    if z <= 2:
        period = 1
        group = z
    elif z <= 10:
        period = 2
        group = z - 2
    elif z <= 18:
        period = 3
        group = z - 10
    elif z <= 36:
        period = 4
        if z <= 20:
            group = z - 18
        elif z <= 30:
            group = z - 18  # d-block
        else:
            group = z - 28
    elif z <= 54:
        period = 5
        if z <= 38:
            group = z - 36
        elif z <= 48:
            group = z - 36
        else:
            group = z - 46
    else:
        period = 6
        group = min(z - 54, 8)
    return period, group


# =============================================================================
# Ligand Parsing (with Hydrogens!)
# =============================================================================

def parse_ligand_smiles(smiles: str, add_hydrogens: bool = True) -> Optional[Ligand]:
    """
    Parse ligand SMILES to get complete molecule WITH HYDROGENS.
    
    This is the key step that was missing - MetalloGen adds H!
    """
    try:
        rd_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if rd_mol is None:
            logger.error(f"Failed to parse SMILES: {smiles}")
            return None
        
        # Sanitize (but don't adjust Hs yet)
        Chem.SanitizeMol(
            rd_mol, 
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
        
        # ADD HYDROGENS - critical step!
        if add_hydrogens:
            rd_mol = Chem.AddHs(rd_mol)
        
        # Kekulize to get explicit bond orders
        rd_mol_copy = Chem.Mol(rd_mol)
        try:
            Chem.Kekulize(rd_mol, clearAromaticFlags=True)
        except:
            rd_mol = rd_mol_copy  # Fall back if kekulization fails
        
        n = rd_mol.GetNumAtoms()
        
        # Extract atoms
        atom_list = []
        chg_list = []
        for i in range(n):
            rd_atom = rd_mol.GetAtomWithIdx(i)
            atom = Atom(
                atomic_number=rd_atom.GetAtomicNum(),
                formal_charge=rd_atom.GetFormalCharge()
            )
            atom_list.append(atom)
            chg_list.append(rd_atom.GetFormalCharge())
        
        # Extract bond orders
        bond_types = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 1,  # After kekulization, shouldn't happen
        }
        
        bo_matrix = np.zeros((n, n), dtype=int)
        for bond in rd_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bo = bond_types.get(bond.GetBondType(), 1)
            bo_matrix[i, j] = bo_matrix[j, i] = bo
        
        adj_matrix = (bo_matrix > 0).astype(int)
        
        total_charge = sum(chg_list)
        
        logger.debug(f"Parsed ligand: {n} atoms (including H), charge={total_charge}")
        
        return Ligand(
            atom_list=atom_list,
            bo_matrix=bo_matrix,
            adj_matrix=adj_matrix,
            chg_list=np.array(chg_list),
            charge=total_charge,
            binding_infos=[]  # Will be set later
        )
        
    except Exception as e:
        logger.error(f"Error parsing ligand SMILES '{smiles}': {e}")
        return None


def find_coordinating_atoms(
    ligand: Ligand,
    coord_positions: list[int],  # 1-based positions from T-REX
) -> list[int]:
    """
    Find atom indices in ligand (with H) that correspond to coordination positions.
    
    coord_positions are 1-based indices into the SMILES (heavy atoms only).
    We need to map to the actual atom indices after H addition.
    
    Special case: For single-atom ligands (like hydride H-), position 1 maps to index 0.
    """
    # Special case: single-atom ligand (e.g., hydride, halide)
    if len(ligand.atom_list) == 1:
        # Position 1 maps to the only atom (index 0)
        if coord_positions == [1] or coord_positions == []:
            return [0]
        else:
            logger.warning(f"Unexpected coord positions {coord_positions} for single-atom ligand")
            return [0]
    
    # Build mapping: SMILES position (1-based, heavy only) -> actual index (0-based, with H)
    heavy_to_full = []
    for i, atom in enumerate(ligand.atom_list):
        if atom.atomic_number != 1:  # Not hydrogen
            heavy_to_full.append(i)
    
    # Special case: ligand with only hydrogens (shouldn't happen but handle it)
    if len(heavy_to_full) == 0:
        logger.warning(f"Ligand has no heavy atoms, using index 0")
        return [0] if coord_positions else []
    
    coord_indices = []
    for pos in coord_positions:
        if 1 <= pos <= len(heavy_to_full):
            coord_indices.append(heavy_to_full[pos - 1])
        else:
            logger.warning(f"Coordination position {pos} out of range (max {len(heavy_to_full)})")
    
    return coord_indices


# =============================================================================
# Build Combined Molecule (MetalloGen's get_alternative_molecule)
# =============================================================================

def build_combined_molecule(
    ligands: list[Ligand],
    metal_z: int,
    oxidation_state: int,
    geometry_cn: int,
) -> tuple[list[Atom], np.ndarray, np.ndarray, list[int], list[tuple[int, int, bool, bool]], int]:
    """
    Build combined molecule following MetalloGen's get_alternative_molecule.
    
    Returns:
        atom_list: All atoms (metal at index 0, then ligands, then ghost atoms)
        adj_matrix: Combined adjacency matrix
        bo_matrix: Combined bond order matrix
        dummy_indices: Indices of ghost/dummy atoms
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        metal_idx: Index of metal (always 0)
    """
    metal_idx = 0
    
    # Count total atoms
    total_wo_dummy = 1  # Metal
    for lig in ligands:
        total_wo_dummy += len(lig.atom_list)
    
    # Count dummy atoms needed for haptic sites
    n_dummies = 0
    for lig in ligands:
        for binding_indices, site_num in lig.binding_infos:
            if len(binding_indices) > 1:  # Haptic
                n_dummies += 1
    
    total_atoms = total_wo_dummy + n_dummies
    
    # Initialize
    atom_list = [None] * total_atoms
    adj_matrix = np.zeros((total_atoms, total_atoms), dtype=int)
    bo_matrix = np.zeros((total_atoms, total_atoms), dtype=int)
    
    # Add metal at index 0 (will be replaced with dummy later)
    atom_list[0] = Atom(atomic_number=metal_z, formal_charge=oxidation_state)
    
    # Track atom index offsets
    current_idx = 1
    atom_indices_per_ligand = []
    
    # Add ligand atoms
    for lig in ligands:
        n_lig = len(lig.atom_list)
        indices = list(range(current_idx, current_idx + n_lig))
        atom_indices_per_ligand.append(indices)
        
        for i, atom in enumerate(lig.atom_list):
            atom_list[indices[i]] = atom.copy()
        
        # Copy ligand internal bonds
        for i in range(n_lig):
            for j in range(n_lig):
                if lig.adj_matrix[i, j] > 0:
                    gi, gj = indices[i], indices[j]
                    adj_matrix[gi, gj] = lig.adj_matrix[i, j]
                    bo_matrix[gi, gj] = lig.bo_matrix[i, j]
        
        current_idx += n_lig
    
    # Add dummy atoms and connect
    dummy_idx = total_wo_dummy
    dummy_indices = []
    metal_binding_infos = []
    
    for lig_idx, lig in enumerate(ligands):
        indices = atom_indices_per_ligand[lig_idx]
        is_multidentate = len(lig.binding_infos) > 1
        
        for binding_indices, site_num in lig.binding_infos:
            is_haptic = len(binding_indices) > 1
            
            if is_haptic:
                # Add ghost atom
                hapticity = len(binding_indices)
                dummy_z = DUMMY_ATOM_BY_HAPTICITY.get(hapticity, [26])[0]
                atom_list[dummy_idx] = Atom(atomic_number=dummy_z)
                
                # Connect ghost to ring atoms
                for local_idx in binding_indices:
                    global_idx = indices[local_idx]
                    adj_matrix[global_idx, dummy_idx] = 1
                    adj_matrix[dummy_idx, global_idx] = 1
                    bo_matrix[global_idx, dummy_idx] = 1
                    bo_matrix[dummy_idx, global_idx] = 1
                
                dummy_indices.append(dummy_idx)
                metal_binding_infos.append((dummy_idx, site_num, is_multidentate, is_haptic))
                dummy_idx += 1
            else:
                # Direct connection
                global_idx = indices[binding_indices[0]]
                metal_binding_infos.append((global_idx, site_num, is_multidentate, is_haptic))
    
    # Connect metal to binding atoms
    for binding_idx, site_num, _, _ in metal_binding_infos:
        adj_matrix[metal_idx, binding_idx] = 1
        adj_matrix[binding_idx, metal_idx] = 1
        bo_matrix[metal_idx, binding_idx] = 1  # M-L bonds are SINGLE
        bo_matrix[binding_idx, metal_idx] = 1
    
    # Replace metal with dummy element based on CN
    cn_str = str(min(geometry_cn, 6))
    dummy_metal_z = DUMMY_CENTER_BY_CN.get(cn_str, 16)
    atom_list[0] = Atom(atomic_number=dummy_metal_z, formal_charge=0)
    
    logger.info(f"Built combined molecule: {total_atoms} atoms, {n_dummies} ghosts, "
                f"metal replaced with Z={dummy_metal_z}")
    
    # Compute ligand atom ranges (start_idx, end_idx) for each ligand
    ligand_atom_ranges = []
    for indices in atom_indices_per_ligand:
        if indices:
            ligand_atom_ranges.append((indices[0], indices[-1] + 1))
        else:
            ligand_atom_ranges.append((0, 0))
    
    return atom_list, adj_matrix, bo_matrix, dummy_indices, metal_binding_infos, metal_idx, ligand_atom_ranges


# =============================================================================
# Get Valid Molecule using LP Solver (MetalloGen's get_valid_molecule)
# =============================================================================

def identify_lp_donor_shifts(
    z_list: np.ndarray,
    adj_matrix: np.ndarray,
    metal_idx: int,
) -> dict[int, int]:
    """
    Identify LP donor atoms that would cause LP solver infeasibility.
    
    These are atoms where group + adjacency > 8, which violates the octet
    constraint and causes charge imbalance issues (especially at CN=4).
    
    LP donors like N in NH3, O in H2O, P in PH3, S in RSH with high adjacency
    need charge > 0 to satisfy octet, but the metal dummy can't absorb the
    negative charge at CN=4 (Si dummy with 4 neighbors has octet constraint).
    
    Shifting these atoms to lower-group elements eliminates the need for
    any formal charges, making the LP problem feasible.
    
    Returns:
        dict mapping atom_idx -> new_z (shifted element)
    """
    n = len(z_list)
    adjacency_list = np.sum(adj_matrix, axis=0).astype(int)
    
    lp_donor_shifts = {}
    
    for i in range(n):
        if i == metal_idx:
            continue
        
        z = z_list[i]
        period, group = get_period_and_group(z)
        adj = adjacency_list[i]
        
        # Check if this atom would violate octet constraint
        # Period 2: constraint always applies (N, O, etc.)
        # Period 3 with adj <= 4: constraint applies (P, S, etc.)
        
        if period == 2 and group + adj > 8:
            # Shift to element where group + adjacency = 8
            # new_group = 8 - adjacency, so new_z = 10 - adjacency (period 2)
            # Examples: N(g=5) + adj=4 = 9 > 8 -> C(g=4), ve = 4+4 = 8
            #           O(g=6) + adj=3 = 9 > 8 -> N(g=5), ve = 5+3 = 8
            new_z = int(10 - adj)  # Convert to Python int for RDKit compatibility
            if new_z >= 3:  # At least Li
                lp_donor_shifts[i] = new_z
                sym_old = PERIODIC_TABLE[z - 1] if z <= len(PERIODIC_TABLE) else f"Z{z}"
                sym_new = PERIODIC_TABLE[new_z - 1] if new_z <= len(PERIODIC_TABLE) else f"Z{new_z}"
                logger.info(f"  LP donor fallback: atom {i} {sym_old}(g={group}, adj={adj}) -> {sym_new}")
        
        elif period == 3 and adj <= 4 and group + adj > 8:
            # new_z = 18 - adjacency (period 3)
            # Examples: P(g=5) + adj=4 = 9 > 8 -> Si(g=4), ve = 4+4 = 8
            #           S(g=6) + adj=3 = 9 > 8 -> P(g=5), ve = 5+3 = 8
            new_z = int(18 - adj)  # Convert to Python int for RDKit compatibility
            if 11 <= new_z <= 18:  # Na through Ar
                lp_donor_shifts[i] = new_z
                sym_old = PERIODIC_TABLE[z - 1] if z <= len(PERIODIC_TABLE) else f"Z{z}"
                sym_new = PERIODIC_TABLE[new_z - 1] if new_z <= len(PERIODIC_TABLE) else f"Z{new_z}"
                logger.info(f"  LP donor fallback: atom {i} {sym_old}(g={group}, adj={adj}) -> {sym_new}")
    
    return lp_donor_shifts


def _run_lp_solver(
    new_atom_list: list[Atom],
    adj_matrix: np.ndarray,
    total_charge: int,
    metal_idx: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Helper function to run LP solver and return results.
    
    Returns:
        (chg_list, bo_matrix) or (None, None) on failure
    """
    molecule = Molecule(new_atom_list, adj_matrix)
    molecule.chg = total_charge
    
    try:
        chg_list, bo_matrix = compute_pulp.compute_chg_and_bo(
            molecule, 
            total_charge, 
            resolve=True, 
            cleanUp=True,
            HalogenConstraint=True,
            MetalCenters=[metal_idx],
        )
        return chg_list, bo_matrix
    except Exception as e:
        logger.debug(f"LP solver raised exception: {e}")
        return None, None


def get_valid_molecule(
    atom_list: list[Atom],
    adj_matrix: np.ndarray,
    total_charge: int,
    metal_idx: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[Atom], np.ndarray]:
    """
    Convert molecule to valid RDKit-compatible form using LP solver.
    
    FALLBACK STRATEGY:
    1. First try LP solver with standard element replacements (original behavior)
    2. If LP solver fails, identify LP donor atoms and shift them
    3. Retry LP solver with shifted atoms
    4. Track all modifications in original_z for restoration after embedding
    
    This conservative approach preserves working cases while fixing problematic
    LP donors (N in NH3, O in H2O/alcohols, P in PH3, S in thiols).
    
    Args:
        atom_list: List of Atom objects
        adj_matrix: Adjacency matrix
        total_charge: Total molecular charge
        metal_idx: Index of metal atom
        
    Returns:
        chg_list: Formal charges (or None if failed)
        bo_matrix: Bond order matrix (or None if failed)
        new_atom_list: Modified atom list with replaced elements
        original_z: ORIGINAL atomic numbers for restoration after embedding
    """
    n = len(atom_list)
    z_list = np.array([a.atomic_number for a in atom_list])
    original_z = z_list.copy()  # SAVE ORIGINAL for restoration!
    
    # Get period/group for each atom
    period_list = np.zeros(n, dtype=int)
    group_list = np.zeros(n, dtype=int)
    for i, z in enumerate(z_list):
        period_list[i], group_list[i] = get_period_and_group(z)
    
    # Find adjacency count
    adjacency_list = np.sum(adj_matrix, axis=0).astype(int)
    
    # Get max valency for each atom - FIXED for halogens!
    max_valency = np.zeros(n, dtype=int)
    for i in range(n):
        if period_list[i] == 1:
            max_valency[i] = 1  # H
        elif period_list[i] == 2:
            max_valency[i] = 4  # C, N, O, etc.
        elif group_list[i] == 7:
            # HALOGENS (F, Cl, Br, I) - max valency 1!
            max_valency[i] = 1
        else:
            max_valency[i] = 6  # Other period 3+ atoms
    
    # Find problem indices (overvalent or period > 3)
    problem_indices = set(np.flatnonzero(adjacency_list > max_valency))
    over_period_indices = set(np.flatnonzero(period_list > 3))
    problem_indices = list(problem_indices | over_period_indices)
    
    # DEBUG: Check all halogens
    for i in range(n):
        if group_list[i] == 7:
            logger.debug(f"  Halogen at {i}: Z={z_list[i]}, adj={adjacency_list[i]}, max_val={max_valency[i]}, flagged={i in problem_indices}")
    
    logger.debug(f"Problem indices: {problem_indices}")
    for idx in problem_indices:
        logger.debug(f"  Atom {idx}: Z={z_list[idx]}, period={period_list[idx]}, "
                    f"group={group_list[idx]}, adj={adjacency_list[idx]}, max_val={max_valency[idx]}")
    
    # Create modified z_list (replace problem atoms with main-group elements)
    new_z_list = z_list.copy()
    for idx in problem_indices:
        period = period_list[idx]
        adj = adjacency_list[idx]
        
        if period == 1:
            if adj > 1:
                new_z_list[idx] = 10 - adj  # e.g., adj=2 -> O
        else:
            if adj < 5:
                if period == 2:
                    new_z_list[idx] = 10 - adj  # e.g., adj=4 -> C
                else:
                    new_z_list[idx] = 18 - adj  # e.g., adj=2 -> S, adj=4 -> Si
            elif adj < 7:
                new_z_list[idx] = 10 + adj  # e.g., adj=6 -> S
            else:
                new_z_list[idx] = 26  # Fe for very high coordination
        
        logger.debug(f"  Replaced atom {idx}: Z={z_list[idx]} -> Z={new_z_list[idx]}")
    
    # Create new atom list with modified z
    new_atom_list = []
    for i, atom in enumerate(atom_list):
        new_atom = atom.copy()
        new_atom.atomic_number = int(new_z_list[i])
        new_atom_list.append(new_atom)
    
    # =========================================================================
    # FIRST ATTEMPT: Try LP solver with standard replacements (original behavior)
    # =========================================================================
    logger.info("Calling LP solver for bond order assignment...")
    
    chg_list, bo_matrix = _run_lp_solver(new_atom_list, adj_matrix, total_charge, metal_idx)
    
    if bo_matrix is not None and chg_list is not None:
        logger.info(f"LP solver succeeded on first attempt: charges sum = {np.sum(chg_list)}")
    
    # =========================================================================
    # FALLBACK: If first attempt failed, try with LP donor shifts
    # =========================================================================
    if bo_matrix is None or chg_list is None:
        logger.info("LP solver failed, attempting fallback with LP donor shifts...")
        
        # Identify LP donor atoms that need shifting
        lp_donor_shifts = identify_lp_donor_shifts(new_z_list, adj_matrix, metal_idx)
        
        if not lp_donor_shifts:
            logger.error("LP solver failed and no LP donor shifts identified - cannot recover")
            return None, None, new_atom_list, original_z
        
        logger.info(f"Identified {len(lp_donor_shifts)} LP donor atoms to shift")
        
        # Apply LP donor shifts
        for idx, new_z in lp_donor_shifts.items():
            new_z_list[idx] = new_z
            new_atom_list[idx].atomic_number = int(new_z)  # Ensure Python int for RDKit
        
        # Retry LP solver with shifted atoms
        chg_list, bo_matrix = _run_lp_solver(new_atom_list, adj_matrix, total_charge, metal_idx)
        
        if bo_matrix is None or chg_list is None:
            logger.error("LP solver fallback also failed")
            return None, None, new_atom_list, original_z
        
        logger.info(f"LP solver FALLBACK succeeded: charges sum = {np.sum(chg_list)}")
    
    # =========================================================================
    # POST-PROCESSING (same as original)
    # =========================================================================
    
    # Following MetalloGen: further adjust atoms if needed
    bo_sum = np.sum(bo_matrix, axis=0)
    new_period_list = np.zeros(n, dtype=int)
    new_group_list = np.zeros(n, dtype=int)
    for i in range(n):
        new_period_list[i], new_group_list[i] = get_period_and_group(new_z_list[i])
    
    # Modify atoms with unpaired electrons (radical removal)
    # WARNING: This can convert S(16)→Cl(17) which we fix below
    for i in range(n):
        group = new_group_list[i]
        bo = int(bo_sum[i])
        if group % 2 != bo % 2:
            old_z = int(new_atom_list[i].atomic_number)
            new_group_list[i] += 1
            new_atom_list[i].atomic_number = old_z + 1  # Ensure Python int
            new_z = new_atom_list[i].atomic_number
            if old_z == 16 and new_z == 17:  # S→Cl
                logger.debug(f"  Radical removal: atom {i} S(16)→Cl(17), bo_sum={bo}")
    
    # Check and fix overvalent atoms
    for i in range(n):
        group = new_group_list[i]
        bo = int(bo_sum[i])
        
        if new_period_list[i] == 1:
            new_z = 10 - bo
        elif new_period_list[i] == 2:
            new_z = 10 - bo
        else:
            if bo > 4 and bo < 7:
                new_z = bo + 10
            elif bo >= 7:
                new_z = 26
            else:
                new_z = 18 - bo
        
        # Check validity and fix if needed
        octet = 4 if new_period_list[i] <= 2 else group
        if not check_atom_validity(group, bo, 0, octet):
            new_atom_list[i].atomic_number = int(new_z)
    
    # CRITICAL FIX: Halogens (group 7) can ONLY have valence 1 in RDKit
    # The generic octet rule allows valence 2, but RDKit rejects it
    # Replace any halogen with bo_sum > 1 with S (Z=16)
    halogen_groups = {7}  # F, Cl, Br, I are all group 7
    for i in range(n):
        z = new_atom_list[i].atomic_number
        period, group = get_period_and_group(z)
        bo = int(bo_sum[i])
        if group == 7 and bo > 1:
            logger.debug(f"  Halogen fix: atom {i} (Z={z}) has bo_sum={bo}, replacing with S (Z=16)")
            new_atom_list[i].atomic_number = 16  # S can have valence 2
    
    return chg_list, bo_matrix, new_atom_list, original_z


def check_atom_validity(group: int, bo: int, chg: int, octet: int) -> bool:
    """
    Check if atom valence is valid.
    
    Matches MetalloGen's process.py:check_atom_validity
    """
    # Lone pair inequality: group - bo - chg >= 0
    if group - bo - chg < 0:
        return False
    # Octet rule inequality: group + bo - chg <= 2 * octet
    if group + bo - chg > 2 * octet:
        return False
    return True


# =============================================================================
# Build RDKit Mol (MetalloGen's get_rd_mol)
# =============================================================================

def build_rdkit_mol(
    atom_list: list[Atom],
    bo_matrix: np.ndarray,
    metal_idx: int,
    chg_list: Optional[np.ndarray] = None,
) -> Optional[Chem.Mol]:
    """
    Build RDKit Mol from atom list and bond order matrix.
    
    Following MetalloGen's get_rd_mol().
    
    Args:
        atom_list: List of Atom objects
        bo_matrix: Bond order matrix from LP solver
        metal_idx: Index of metal atom
        chg_list: Formal charges (MetalloGen sets these to 0 after LP solver)
        
    Note: MetalloGen sets atom_feature['chg'] = np.zeros after LP solver,
    so all formal charges are 0 in the RDKit mol.
    """
    n = len(atom_list)
    
    # MetalloGen sets all charges to 0 after LP solver (chem.py line 1913)
    # The LP solver finds valid BOND ORDERS; charges are set to 0 for embedding
    if chg_list is None:
        chg_list = np.zeros(n, dtype=int)
    else:
        # Override with zeros to match MetalloGen behavior
        chg_list = np.zeros(n, dtype=int)
    
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    
    rd_mol = Chem.RWMol()
    
    # Add atoms with formal charge 0
    for i, atom in enumerate(atom_list):
        rd_atom = Chem.Atom(int(atom.atomic_number))  # Ensure Python int for RDKit
        rd_atom.SetFormalCharge(int(chg_list[i]))  # Always 0
        rd_atom.SetNoImplicit(True)
        rd_mol.AddAtom(rd_atom)
    
    # Add bonds
    for i in range(n):
        for j in range(i + 1, n):
            if bo_matrix[i, j] > 0:
                bo = int(bo_matrix[i, j])
                bond_type = bond_types.get(bo, Chem.BondType.SINGLE)
                rd_mol.AddBond(i, j, bond_type)
    
    # Try to sanitize (MetalloGen uses full sanitization at chem.py line 1695)
    try:
        Chem.SanitizeMol(rd_mol)
        logger.info("Full sanitization succeeded")
        return rd_mol.GetMol()
    except Exception as e:
        logger.warning(f"Full sanitization failed: {e}")
    
    # Try relaxed sanitization as fallback
    try:
        Chem.SanitizeMol(
            rd_mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL ^
                Chem.SanitizeFlags.SANITIZE_PROPERTIES ^
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            )
        )
        logger.info("Relaxed sanitization succeeded")
        return rd_mol.GetMol()
    except Exception as e:
        logger.error(f"Relaxed sanitization also failed: {e}")
        return None


# =============================================================================
# Clash Detection
# =============================================================================

def check_atom_clashes(
    mol: Chem.Mol,
    severe_threshold: float = 0.5,
    moderate_threshold: float = 0.7,
    bond_buffer: int = 3,
) -> ClashResult:
    """
    Check for atom clashes in an embedded molecule.
    
    A clash is detected when two non-bonded atoms are closer than a fraction
    of the sum of their van der Waals radii.
    
    Args:
        mol: RDKit molecule with 3D conformer
        severe_threshold: Ratio below which clash is severe (default 0.5)
        moderate_threshold: Ratio below which clash is moderate (default 0.7)
        bond_buffer: Ignore atoms within this many bonds (default 3 = 1-4 ok)
        
    Returns:
        ClashResult with statistics and details
    """
    if mol.GetNumConformers() == 0:
        return ClashResult(
            has_severe_clash=False, has_moderate_clash=False,
            n_severe_clashes=0, n_moderate_clashes=0,
            min_distance=float('inf'), min_distance_pair=(-1, -1),
            min_distance_ratio=float('inf'), clash_details=[],
        )
    
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    
    # Build bond distance matrix using Floyd-Warshall
    INF = 1000
    dist_matrix = np.full((n_atoms, n_atoms), INF, dtype=int)
    np.fill_diagonal(dist_matrix, 0)
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist_matrix[i, j] = dist_matrix[j, i] = 1
    
    for k in range(n_atoms):
        for i in range(n_atoms):
            for j in range(n_atoms):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
    
    # Check all non-bonded pairs
    clash_details = []
    min_distance = float('inf')
    min_distance_pair = (-1, -1)
    min_distance_ratio = float('inf')
    n_severe = n_moderate = 0
    
    for i in range(n_atoms):
        pos_i = conf.GetAtomPosition(i)
        atom_i = mol.GetAtomWithIdx(i)
        vdw_i = get_vdw_radius(atom_i.GetAtomicNum())
        
        for j in range(i + 1, n_atoms):
            if dist_matrix[i, j] <= bond_buffer:
                continue
            
            pos_j = conf.GetAtomPosition(j)
            atom_j = mol.GetAtomWithIdx(j)
            vdw_j = get_vdw_radius(atom_j.GetAtomicNum())
            
            dist = np.sqrt((pos_i.x - pos_j.x)**2 + 
                          (pos_i.y - pos_j.y)**2 + 
                          (pos_i.z - pos_j.z)**2)
            
            sum_vdw = vdw_i + vdw_j
            ratio = dist / sum_vdw
            
            if dist < min_distance:
                min_distance = dist
                min_distance_pair = (i, j)
                min_distance_ratio = ratio
            
            if ratio < severe_threshold:
                n_severe += 1
                clash_details.append({
                    'atoms': (i, j),
                    'symbols': (atom_i.GetSymbol(), atom_j.GetSymbol()),
                    'distance': dist, 'sum_vdw': sum_vdw,
                    'ratio': ratio, 'severity': 'severe',
                })
            elif ratio < moderate_threshold:
                n_moderate += 1
                clash_details.append({
                    'atoms': (i, j),
                    'symbols': (atom_i.GetSymbol(), atom_j.GetSymbol()),
                    'distance': dist, 'sum_vdw': sum_vdw,
                    'ratio': ratio, 'severity': 'moderate',
                })
    
    return ClashResult(
        has_severe_clash=(n_severe > 0),
        has_moderate_clash=(n_moderate > 0),
        n_severe_clashes=n_severe, n_moderate_clashes=n_moderate,
        min_distance=min_distance, min_distance_pair=min_distance_pair,
        min_distance_ratio=min_distance_ratio, clash_details=clash_details,
    )


# =============================================================================
# Constrained Embedding with Progressive Relaxation (V3)
# =============================================================================

def _embed_level_0_strict(
    rd_mol: Chem.Mol,
    cmap: Dict[int, Point3D],
    seed: int,
    max_attempts: int,
) -> Optional[Chem.Mol]:
    """
    Level 0: Original strict constrained embedding (MetalloGen behavior).
    
    This is the most faithful to the original MetalloGen approach:
    - useRandomCoords=True: Start from random initial coordinates
    - maxIterations=100: Limited refinement iterations
    - useBasicKnowledge=False: Don't use RDKit's built-in geometric preferences
    - ignoreSmoothingFailures=True: Continue even if bounds matrix smoothing fails
    
    The coordMap fixes the metal at origin and all coordinating atoms at their
    target positions. Distance geometry must satisfy ALL these constraints
    simultaneously while also satisfying bond length/angle bounds from the
    molecular graph.
    
    This fails when the constraints are geometrically incompatible with the
    ligand structures (e.g., a rigid ligand can't reach its target position).
    """
    params = rdDistGeom.EmbedParameters()
    params.useRandomCoords = True
    params.maxIterations = 100
    params.useBasicKnowledge = False
    params.ignoreSmoothingFailures = True
    params.SetCoordMap(cmap)
    
    for attempt in range(max_attempts):
        params.randomSeed = seed + attempt * 12345
        mol_copy = Chem.RWMol(Chem.Mol(rd_mol))
        try:
            if AllChem.EmbedMolecule(mol_copy, params) == 0 and mol_copy.GetNumConformers() > 0:
                return mol_copy.GetMol()
        except Exception as e:
            logger.debug(f"Level 0 attempt {attempt + 1} failed: {e}")
    return None


def _embed_level_1_relaxed(
    rd_mol: Chem.Mol,
    cmap: Dict[int, Point3D],
    seed: int,
    max_attempts: int,
) -> Optional[Chem.Mol]:
    """
    Level 1: Relaxed parameters - more iterations and basic knowledge.
    
    Key differences from Level 0:
    - maxIterations=500: 5x more iterations for distance geometry optimization,
      giving the algorithm more chances to find a valid solution
    - useBasicKnowledge=True: Uses RDKit's built-in knowledge about:
      * Preferred bond lengths (from atomic radii)
      * Preferred bond angles (sp3=109.5°, sp2=120°, sp=180°)
      * Basic ring constraints (planarity for aromatic rings)
      This makes the bounds matrix more realistic and can help find solutions
      that Level 0 misses
    - enforceChirality=False: Allows the algorithm to ignore tetrahedral
      chirality constraints, which can sometimes conflict with the coordMap
    
    The same coordMap constraints are used, but the algorithm has more
    flexibility in how it satisfies them.
    """
    params = rdDistGeom.EmbedParameters()
    params.useRandomCoords = True
    params.maxIterations = 500
    params.useBasicKnowledge = True
    params.ignoreSmoothingFailures = True
    params.enforceChirality = False
    params.SetCoordMap(cmap)
    
    for attempt in range(max_attempts):
        params.randomSeed = seed + attempt * 54321
        mol_copy = Chem.RWMol(Chem.Mol(rd_mol))
        try:
            if AllChem.EmbedMolecule(mol_copy, params) == 0 and mol_copy.GetNumConformers() > 0:
                return mol_copy.GetMol()
        except Exception as e:
            logger.debug(f"Level 1 attempt {attempt + 1} failed: {e}")
    return None


def _embed_level_2_etkdg(
    rd_mol: Chem.Mol,
    cmap: Dict[int, Point3D],
    seed: int,
    max_attempts: int,
) -> Optional[Chem.Mol]:
    """
    Level 2: ETKDG v3 with experimental torsion angle knowledge.
    
    ETKDG = "Experimental Torsion-angle preference with Distance Geometry"
    
    This uses torsion angle preferences derived from the Cambridge Structural
    Database (CSD), which contains ~1 million experimentally-determined crystal
    structures. The key insight is that certain torsion angles are statistically
    preferred in real molecules.
    
    ETKDGv3 (RDKit 2020.09+) improvements over earlier versions:
    - Better handling of macrocycles (uses distance bounds from small ring conformers)
    - Improved small ring conformations (uses templates from CSD)
    - Better treatment of conjugated systems
    - Improved sampling of flexible regions
    
    For metal complexes, ETKDG can help because:
    - Ligand conformations are more realistic
    - Ring systems (Cp, arenes, chelate rings) adopt correct geometries
    - The torsion preferences help resolve ambiguities in the bounds matrix
    
    Falls back to basic ETKDG if v3 is not available in the RDKit version.
    """
    try:
        params = rdDistGeom.ETKDGv3()
    except AttributeError:
        # Older RDKit version
        params = rdDistGeom.ETKDG()
    
    params.useRandomCoords = True
    params.maxIterations = 500
    params.ignoreSmoothingFailures = True
    params.SetCoordMap(cmap)
    
    for attempt in range(max_attempts):
        params.randomSeed = seed + attempt * 13579
        mol_copy = Chem.RWMol(Chem.Mol(rd_mol))
        try:
            if AllChem.EmbedMolecule(mol_copy, params) == 0 and mol_copy.GetNumConformers() > 0:
                return mol_copy.GetMol()
        except Exception as e:
            logger.debug(f"Level 2 attempt {attempt + 1} failed: {e}")
    return None


def constrained_embed(
    rd_mol: Chem.Mol,
    metal_idx: int,
    metal_binding_infos: list[tuple[int, int, bool, bool]],
    site_coords: np.ndarray,
    metal_radius: float = 1.5,
    scale: float = 1.0,
    max_attempts: int = 30,
    seed: int = None,
    max_relaxation_level: int = 2,
    reject_severe_clashes: bool = True,
) -> Tuple[Optional[Chem.Mol], int, str, Optional[ClashResult]]:
    """
    Constrained embedding with progressive relaxation fallback.
    
    All levels enforce the same coordinate constraints (metal at origin,
    coordinating atoms at target positions). The difference is in the
    distance geometry algorithm parameters.
    
    Relaxation Levels:
    
    Level 0 - Strict (Original MetalloGen):
        Original MetalloGen behavior. Uses minimal iterations (100) and
        no built-in geometric knowledge. Fast but fails when constraints
        are difficult to satisfy.
    
    Level 1 - Relaxed Parameters:
        More iterations (500) and enables RDKit's basic geometric knowledge
        (preferred bond lengths, angles, ring planarity). Also disables
        chirality enforcement which can conflict with coordMap constraints.
        Often succeeds where Level 0 fails due to better bounds handling.
    
    Level 2 - ETKDG v3:
        Uses Experimental Torsion-angle Distance Geometry with knowledge
        from the Cambridge Structural Database. Produces more realistic
        ligand conformations, especially for rings and conjugated systems.
        Best for complexes with rigid or semi-rigid ligands.
    
    Args:
        rd_mol: RDKit molecule with dummy metal
        metal_idx: Index of metal (dummy) atom
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        site_coords: Target coordinates for each site (from coord_reconstruction)
        metal_radius: Unused, kept for backward compatibility
        scale: Unused, kept for backward compatibility  
        max_attempts: Total attempts distributed across levels
        seed: Base random seed (default 42)
        max_relaxation_level: Maximum level to try (0-2, default 2)
        reject_severe_clashes: If True, reject structures with severe atom
            clashes (< 0.5 × sum of VDW radii) and try next level
        
    Returns:
        Tuple of (mol, level, level_name, clash_result)
        - mol: Embedded molecule or None if all levels failed
        - level: Which level succeeded (0-2) or -1 if failed
        - level_name: Human-readable level name
        - clash_result: ClashResult from post-embedding clash check
        
    Note:
        site_num in metal_binding_infos is 1-based and indexes into
        site_coords as site_coords[site_num - 1]
    """
    if seed is None:
        seed = 42
    
    # Validate site_coords
    site_coords = np.asarray(site_coords)
    if site_coords.ndim == 1 and len(site_coords) == 3:
        site_coords = site_coords.reshape(1, 3)
    
    if site_coords.ndim != 2 or site_coords.shape[1] != 3:
        logger.error(f"Invalid site_coords shape: {site_coords.shape}")
        return None, -1, "invalid_input", None
    
    # Build full coordMap: metal at origin, coordinating atoms at targets
    cmap = {metal_idx: Point3D(0.0, 0.0, 0.0)}
    for atom_idx, site_num, is_multi, is_haptic in metal_binding_infos:
        coord_idx = site_num - 1
        if coord_idx < len(site_coords):
            x, y, z = site_coords[coord_idx]
            cmap[atom_idx] = Point3D(float(x), float(y), float(z))
    
    # Distribute attempts across levels (3 levels now)
    attempts_per_level = max(max_attempts // 3, 5)
    
    levels = [
        (0, "strict_constrained", 
         lambda: _embed_level_0_strict(rd_mol, cmap, seed, attempts_per_level)),
        (1, "relaxed_params",
         lambda: _embed_level_1_relaxed(rd_mol, cmap, seed, attempts_per_level)),
        (2, "etkdg_v3",
         lambda: _embed_level_2_etkdg(rd_mol, cmap, seed, attempts_per_level)),
    ]
    
    for level, level_name, embed_func in levels:
        if level > max_relaxation_level:
            break
        
        logger.debug(f"Trying embedding level {level}: {level_name}")
        mol = embed_func()
        
        if mol is not None:
            clash_result = check_atom_clashes(mol)
            
            if reject_severe_clashes and clash_result.has_severe_clash:
                logger.warning(f"Level {level} produced {clash_result.n_severe_clashes} severe clashes, trying next level")
                continue
            
            logger.info(f"Embedding succeeded at level {level} ({level_name})")
            return mol, level, level_name, clash_result
    
    logger.error("All embedding levels failed")
    return None, -1, "all_failed", None


# =============================================================================
# Post-Embedding: Fix Small Ligand Geometry
# =============================================================================

def fix_small_ligand_geometry(
    mol: Chem.Mol,
    metal_idx: int,
    metal_binding_infos: list[tuple[int, int, bool, bool]],
    ligand_atom_ranges: list[tuple[int, int]],
    original_z: np.ndarray,
    max_ligand_size: int = 4,
) -> Chem.Mol:
    """
    Fix geometry of small ligands that may have degenerate/overlapping atoms.
    
    RDKit's distance geometry often fails for small symmetric ligands like H2O
    when only the coordinating atom is constrained. This function explicitly
    reconstructs proper geometry for such ligands.
    
    Handles:
    - H2O: Places H's at 104.5° angle, 0.96Å from O, pointing away from metal
    - NH3: Places H's in tetrahedral arrangement, pointing away from metal
    - Other small ligands: Spreads atoms around M-L axis
    
    Args:
        mol: Embedded RDKit molecule
        metal_idx: Index of metal atom
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        ligand_atom_ranges: [(start_idx, end_idx), ...] for each ligand
        original_z: Original atomic numbers BEFORE LP solver modifications
        max_ligand_size: Only fix ligands with <= this many atoms
        
    Returns:
        Molecule with fixed small ligand geometries
    """
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    
    if mol.GetNumConformers() == 0:
        return mol
    
    conf = mol.GetConformer()
    mol = Chem.RWMol(mol)
    
    # Get metal position
    metal_pos = np.array(conf.GetAtomPosition(metal_idx))
    
    for atom_idx, site_num, is_multi, is_haptic in metal_binding_infos:
        # Skip multidentate and haptic
        if is_multi or is_haptic:
            continue
        
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 0:
            continue
        
        # Find which ligand
        ligand_idx = None
        for lig_i, (start, end) in enumerate(ligand_atom_ranges):
            if start <= atom_idx < end:
                ligand_idx = lig_i
                ligand_size = end - start
                ligand_start = start
                ligand_end = end
                break
        
        if ligand_idx is None or ligand_size > max_ligand_size:
            continue
        
        # Get coordinating atom info - use ORIGINAL atomic number!
        coord_pos = np.array(conf.GetAtomPosition(atom_idx))
        original_atom_z = int(original_z[atom_idx]) if atom_idx < len(original_z) else atom.GetAtomicNum()
        coord_symbol = pt.GetElementSymbol(original_atom_z)
        
        # Get neighbors (excluding metal) - also check their ORIGINAL types
        neighbors = []
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx != metal_idx:
                orig_n_z = int(original_z[n_idx]) if n_idx < len(original_z) else neighbor.GetAtomicNum()
                orig_n_sym = pt.GetElementSymbol(orig_n_z)
                neighbors.append((n_idx, orig_n_sym))
        
        if not neighbors:
            continue
        
        # Check if atoms are degenerate (too close together)
        if len(neighbors) >= 2:
            pos1 = np.array(conf.GetAtomPosition(neighbors[0][0]))
            pos2 = np.array(conf.GetAtomPosition(neighbors[1][0]))
            dist = np.linalg.norm(pos1 - pos2)
            
            if dist > 0.5:  # Atoms are reasonably separated
                continue
        
        logger.info(f"  Fixing degenerate geometry: atom {atom_idx} ({coord_symbol}), {len(neighbors)} neighbors")
        
        # M->L vector (will place neighbors opposite to this)
        ml_vec = coord_pos - metal_pos
        ml_dist = np.linalg.norm(ml_vec)
        if ml_dist < 0.01:
            continue
        ml_unit = ml_vec / ml_dist
        
        # Build orthonormal basis with ml_unit as z-axis
        # Find a vector not parallel to ml_unit
        if abs(ml_unit[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        
        # Gram-Schmidt to get orthonormal basis
        u1 = ml_unit
        u2 = perp - np.dot(perp, u1) * u1
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(u1, u2)
        
        # Determine geometry based on coordinating atom and neighbors
        if coord_symbol == 'O' and len(neighbors) == 2:
            # Water: H-O-H angle ~104.5°, O-H ~0.96Å
            # We want H's to point AWAY from metal
            hoh_angle = 104.5 * np.pi / 180  # H-O-H angle
            bond_len = 0.96  # O-H distance
            
            # The bisector of H-O-H should point away from metal (-ml_unit direction)
            # Each H is at half the H-O-H angle from the bisector
            half_angle = hoh_angle / 2  # ~52.25°
            
            # H positions: in a plane containing M-O axis, symmetric around bisector
            # bisector points along -ml_unit (away from metal)
            # H's are at ±half_angle from bisector, in the plane defined by ml_unit and u2
            
            h_count = 0
            for n_idx, n_sym in neighbors:
                if n_sym != 'H':
                    continue
                
                # Alternate sides: first H at +half_angle, second at -half_angle
                sign = 1 if h_count == 0 else -1
                h_count += 1
                
                # Direction from O to H:
                # - Component along -ml_unit (away from metal): cos(half_angle)
                # - Component perpendicular (in u2 direction): sin(half_angle)
                h_dir = -np.cos(half_angle) * ml_unit + sign * np.sin(half_angle) * u2
                h_dir = h_dir / np.linalg.norm(h_dir)
                
                new_pos = coord_pos + bond_len * h_dir
                conf.SetAtomPosition(n_idx, Point3D(float(new_pos[0]), 
                                                     float(new_pos[1]), 
                                                     float(new_pos[2])))
                logger.debug(f"    Placed H at idx {n_idx}: {new_pos}")
                
        elif coord_symbol == 'N' and len(neighbors) == 3:
            # Ammonia: N-H ~1.01Å, H-N-H ~107°
            # The H's should point away from metal in a tetrahedral-like arrangement
            bond_len = 1.01
            
            # For NH3 coordinated to metal, the N lone pair points at metal
            # H's are in a pyramidal arrangement pointing away
            # H-N-H angle ~107°, so angle from axis (lone pair direction) is ~180-107/2 ≈ 70°
            # Actually: for tetrahedral, angle from axis = 180 - 109.5 = 70.5°
            cone_angle = 70.5 * np.pi / 180  # Angle of N-H bonds from the -M-N direction
            
            h_indices = [(n_idx, n_sym) for n_idx, n_sym in neighbors if n_sym == 'H']
            
            for i, (n_idx, _) in enumerate(h_indices):
                # Distribute H's around the axis, 120° apart
                azimuth = i * 2 * np.pi / 3
                
                # Direction from N to H:
                # - Component along -ml_unit (away from metal): cos(cone_angle)
                # - Component perpendicular: sin(cone_angle) in rotating direction
                h_dir = (-np.cos(cone_angle) * ml_unit + 
                         np.sin(cone_angle) * (np.cos(azimuth) * u2 + np.sin(azimuth) * u3))
                h_dir = h_dir / np.linalg.norm(h_dir)
                
                new_pos = coord_pos + bond_len * h_dir
                conf.SetAtomPosition(n_idx, Point3D(float(new_pos[0]), 
                                                     float(new_pos[1]), 
                                                     float(new_pos[2])))
                logger.debug(f"    Placed H at idx {n_idx}: {new_pos}")
                
        else:
            # Generic small ligand: spread neighbors evenly around M-L axis, pointing away
            n_neighbors = len(neighbors)
            logger.debug(f"    Using generic geometry for {n_neighbors} neighbors")
            
            for i, (n_idx, n_sym) in enumerate(neighbors):
                # Get current distance (or use default)
                old_pos = np.array(conf.GetAtomPosition(n_idx))
                bond_len = np.linalg.norm(old_pos - coord_pos)
                if bond_len < 0.5:
                    bond_len = 1.0  # Default
                
                # Angle around axis
                azimuth = i * 2 * np.pi / n_neighbors
                
                # Point away from metal at ~70° from -M-L axis (similar to tetrahedral)
                cone_angle = 70.0 * np.pi / 180
                
                n_dir = (-np.cos(cone_angle) * ml_unit + 
                         np.sin(cone_angle) * (np.cos(azimuth) * u2 + np.sin(azimuth) * u3))
                n_dir = n_dir / np.linalg.norm(n_dir)
                
                new_pos = coord_pos + bond_len * n_dir
                conf.SetAtomPosition(n_idx, Point3D(float(new_pos[0]), 
                                                     float(new_pos[1]), 
                                                     float(new_pos[2])))
                logger.debug(f"    Placed {n_sym} at idx {n_idx}: {new_pos}")
    
    return mol.GetMol()


# =============================================================================
# Post-Embedding: Rotate Monodentate Ligands
# =============================================================================

def rotate_monodentate_ligands(
    mol: Chem.Mol,
    metal_idx: int,
    metal_binding_infos: list[tuple[int, int, bool, bool]],
    ligand_atom_ranges: list[tuple[int, int]],
    max_ligand_size: int = 5,
) -> Chem.Mol:
    """
    Rotate small monodentate ligands so neighboring atoms point away from metal.
    
    This is a post-embedding cleanup step from MetalloGen that fixes ligand orientations.
    For example, ensures CO has C toward metal (not O), and NH3 has H pointing away.
    
    Args:
        mol: Embedded RDKit molecule
        metal_idx: Index of metal atom
        metal_binding_infos: [(atom_idx, site_num, is_multidentate, is_haptic), ...]
        ligand_atom_ranges: [(start_idx, end_idx), ...] for each ligand
        max_ligand_size: Only rotate ligands with <= this many atoms
        
    Returns:
        Molecule with rotated ligands
    """
    if mol.GetNumConformers() == 0:
        return mol
    
    conf = mol.GetConformer()
    mol = Chem.RWMol(mol)
    
    # Get metal position
    metal_pos = np.array(conf.GetAtomPosition(metal_idx))
    
    # Track which sites belong to which ligand and if monodentate
    for atom_idx, site_num, is_multi, is_haptic in metal_binding_infos:
        # Skip multidentate and haptic sites
        if is_multi or is_haptic:
            continue
        
        # Skip ghost atoms (they'll be removed anyway)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 0:
            continue
        
        # Find which ligand this atom belongs to
        ligand_idx = None
        ligand_size = 0
        for lig_i, (start, end) in enumerate(ligand_atom_ranges):
            if start <= atom_idx < end:
                ligand_idx = lig_i
                ligand_size = end - start
                break
        
        if ligand_idx is None:
            continue
        
        # Skip large ligands
        if ligand_size > max_ligand_size:
            continue
        
        # Get coordinating atom position
        coord_pos = np.array(conf.GetAtomPosition(atom_idx))
        
        # Get neighbors of coordinating atom (excluding metal)
        neighbors = []
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx != metal_idx:
                neighbors.append(n_idx)
        
        if not neighbors:
            continue
        
        # Compute centroid of neighbors
        neighbor_positions = [np.array(conf.GetAtomPosition(n)) for n in neighbors]
        neighbor_centroid = np.mean(neighbor_positions, axis=0)
        
        # Vector from metal to coordinating atom (M->L axis)
        ml_vec = coord_pos - metal_pos
        ml_dist = np.linalg.norm(ml_vec)
        if ml_dist < 0.01:
            continue
        ml_unit = ml_vec / ml_dist
        
        # Vector from coordinating atom to neighbor centroid
        ln_vec = neighbor_centroid - coord_pos
        ln_dist = np.linalg.norm(ln_vec)
        if ln_dist < 0.01:
            continue
        
        # Project neighbor centroid onto M-L axis to check direction
        # Positive = same direction as M->L (neighbors point away from metal) - GOOD
        # Negative = opposite direction (neighbors point toward metal) - NEED TO ROTATE
        projection = np.dot(ln_vec, ml_unit)
        
        if projection >= 0:
            # Already pointing away from metal
            logger.debug(f"  Ligand at atom {atom_idx}: neighbors already pointing away")
            continue
        
        # Need to rotate 180° around M-L axis
        logger.debug(f"  Ligand at atom {atom_idx}: rotating 180° (proj={projection:.2f})")
        
        # Get all atoms in this ligand (excluding the coordinating atom which stays fixed)
        start, end = ligand_atom_ranges[ligand_idx]
        ligand_atoms = [i for i in range(start, end) if i != atom_idx]
        
        # Rotate each atom 180° around the M-L axis passing through coord_pos
        for lig_atom_idx in ligand_atoms:
            old_pos = np.array(conf.GetAtomPosition(lig_atom_idx))
            
            # Translate so coord_pos is origin
            rel_pos = old_pos - coord_pos
            
            # Rodrigues rotation formula for 180° (pi radians)
            # For 180°: v_rot = 2*(v·k)*k - v where k is unit axis
            # Simplified: v_rot = 2*(projection onto axis) - v
            proj_on_axis = np.dot(rel_pos, ml_unit) * ml_unit
            new_rel_pos = 2 * proj_on_axis - rel_pos
            
            # Translate back
            new_pos = new_rel_pos + coord_pos
            
            conf.SetAtomPosition(lig_atom_idx, Point3D(float(new_pos[0]), 
                                                        float(new_pos[1]), 
                                                        float(new_pos[2])))
    
    return mol.GetMol()


# =============================================================================
# Post-processing: Restore Metal, Remove Ghosts
# =============================================================================

def restore_and_cleanup(
    mol: Chem.Mol,
    metal_idx: int,
    metal_z: int,
    oxidation_state: int,
    dummy_indices: list[int],
    original_z: np.ndarray,
) -> tuple[Chem.Mol, int, dict[int, int]]:
    """
    Restore ALL original atomic numbers and remove ghost atoms.
    
    CRITICAL: The LP solver replaces many atoms (metal, halogens, etc.) with
    different elements. We must restore ALL original atoms, not just the metal!
    
    Returns:
        cleaned_mol: Final molecule with correct elements
        new_metal_idx: Metal index after cleanup
        index_map: old_idx -> new_idx mapping
    """
    rw_mol = Chem.RWMol(Chem.Mol(mol))
    
    # RESTORE ALL ORIGINAL ATOMIC NUMBERS (not just metal!)
    for i in range(rw_mol.GetNumAtoms()):
        if i not in dummy_indices and i < len(original_z):
            atom = rw_mol.GetAtomWithIdx(i)
            atom.SetAtomicNum(int(original_z[i]))
            # Clear formal charge for final output
            atom.SetFormalCharge(0)
    
    # Set metal oxidation state specifically
    metal_atom = rw_mol.GetAtomWithIdx(metal_idx)
    metal_atom.SetAtomicNum(metal_z)
    metal_atom.SetFormalCharge(oxidation_state)
    
    # Remove ghost atoms (in reverse order to maintain indices)
    removed = set(dummy_indices)
    for idx in sorted(dummy_indices, reverse=True):
        rw_mol.RemoveAtom(idx)
    
    # Build index map
    old_n = mol.GetNumAtoms()
    index_map = {}
    new_idx = 0
    for old_idx in range(old_n):
        if old_idx not in removed:
            index_map[old_idx] = new_idx
            new_idx += 1
    
    new_metal_idx = index_map[metal_idx]
    
    logger.info(f"Cleanup: restored {len(original_z) - len(dummy_indices)} atoms, "
                f"removed {len(dummy_indices)} ghosts, metal now at idx {new_metal_idx}")
    
    return rw_mol.GetMol(), new_metal_idx, index_map


# =============================================================================
# Main Pipeline
# =============================================================================

def embed_trex_metallogen(
    t: 'TrexFull',
    site_coords: np.ndarray,
    max_attempts: int = 30,
    seed: int = None,
    geometry: str = None,
    max_relaxation_level: int = 2,
    reject_severe_clashes: bool = True,
) -> EmbedResult:
    """
    Main embedding pipeline following MetalloGen with progressive relaxation.
    
    Args:
        t: TrexFull object
        site_coords: Site coordinates from coord_reconstruction (n_sites, 3)
        max_attempts: Number of embedding attempts (distributed across levels)
        seed: Random seed for embedding (if None, uses default)
        geometry: Geometry name like "6_octahedral" (if None, inferred from t.G)
        max_relaxation_level: Maximum relaxation level to try (0-2, default 2)
            - 0: Strict constrained (original MetalloGen behavior)
            - 1: Relaxed params (more iterations, useBasicKnowledge=True)
            - 2: ETKDG v3 with experimental torsion angle preferences
        reject_severe_clashes: Reject structures with severe atom clashes
        
    Returns:
        EmbedResult with mol, diagnostics, embedding_level, and clash_result
    """
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    
    diagnostics = {}
    
    # Get site ordering (pairs interleaved, then singles)
    sites_ordered = []
    for a, b in t.pairs:
        sites_ordered.append(a)
        sites_ordered.append(b)
    for s in t.singles:
        sites_ordered.append(s)
    
    n_sites = len(sites_ordered)
    diagnostics['n_sites'] = n_sites
    
    if len(site_coords) != n_sites:
        raise ValueError(f"Expected {n_sites} site coords, got {len(site_coords)}")
    
    # Parse ligands
    logger.info("Step 1: Parsing ligands with hydrogens...")
    
    ligands = []
    for lig_idx, payload in enumerate(t.ligands):
        if payload.kind != "SMILES":
            raise ValueError(f"Only SMILES ligands supported, got {payload.kind}")
        
        lig = parse_ligand_smiles(payload.text, add_hydrogens=True)
        if lig is None:
            raise ValueError(f"Failed to parse ligand {lig_idx}: {payload.text}")
        
        # Find binding infos for this ligand
        # NOTE: site.lig is 1-based in TrexFull, lig_idx is 0-based
        lig_num = lig_idx + 1  # Convert to 1-based for comparison
        site_num = 0
        for site in sites_ordered:
            site_num += 1
            if site.lig == lig_num:
                coord_indices = find_coordinating_atoms(lig, list(site.atoms))
                lig.binding_infos.append((coord_indices, site_num))
        
        ligands.append(lig)
        logger.info(f"  Ligand {lig_idx}: {len(lig.atom_list)} atoms, "
                   f"{len(lig.binding_infos)} binding sites")
    
    diagnostics['n_ligands'] = len(ligands)
    diagnostics['total_ligand_atoms'] = sum(len(lig.atom_list) for lig in ligands)
    
    # Build combined molecule
    logger.info("Step 2: Building combined molecule...")
    
    metal_z = pt.GetAtomicNumber(t.metal)
    
    atom_list, adj_matrix, bo_matrix, dummy_indices, metal_binding_infos, metal_idx, ligand_atom_ranges = \
        build_combined_molecule(ligands, metal_z, t.ox, n_sites)
    
    diagnostics['total_atoms'] = len(atom_list)
    diagnostics['n_dummies'] = len(dummy_indices)
    
    # Calculate total charge
    total_charge = t.ox
    for lig in ligands:
        total_charge += lig.charge
    
    # Step 3: Get valid molecule using LP solver
    logger.info("Step 3: Computing valid bond orders with LP solver...")
    
    chg_list, valid_bo_matrix, valid_atom_list, original_z = get_valid_molecule(
        atom_list, adj_matrix, total_charge, metal_idx
    )
    
    if valid_bo_matrix is None:
        logger.error("LP solver failed to find valid bond orders")
        return EmbedResult(
            mol=None, success=False, metal_idx=0,
            coord_atom_indices=[], charge=total_charge, spin=t.spin or 1,
            diagnostics=diagnostics
        )
    
    diagnostics['lp_solver_success'] = True
    
    # Step 3b: Replace metal with geometry-specific dummy for embedding
    # MetalloGen does this at embed.py line 238: valid_ace_mol.atom_list[metal_index] = dummy_center
    # The dummy_center comes from get_dummy_center(geometry_name) which is geometry-specific
    
    # Infer geometry from t.G if not provided
    if geometry is None:
        geo_map = {
    # Primary geometry codes (from T-REX G field)
    'O': '6_octahedral',
    'SP': '4_square_planar',
    'T': '4_tetrahedral', 
    'TBP': '5_trigonal_bipyramidal',
    'SQP': '5_square_pyramidal',
    'L': '2_linear',
    'TP': '3_trigonal_planar',
    'TPY': '3_trigonal_pyramidal',
    # Additional codes (from get_geo_type())
    'sqpl': '4_square_planar',
    'sqpy': '5_square_pyramidal',
    'thd': '4_tetrahedral',
    'trbp': '5_trigonal_bipyramidal',
    'pntbp': '7_pentagonal_bipyramidal',
    'ln': '2_linear',
    'trpl': '3_trigonal_planar',
    'tsh': '3_t_shaped',
    'ssw': '4_seesaw',
    'bnt': '2_bent',  # <- ADDED: bent geometry
    'unk': '6_octahedral',
    # Bent variations
    'B': '2_bent',
    'BNT': '2_bent',
}
        geometry = geo_map.get(t.G, '6_octahedral')
    
    geometry_dummy_z = get_dummy_center_for_geometry(geometry)
    valid_atom_list[metal_idx].atomic_number = geometry_dummy_z
    logger.info(f"  Replaced metal with geometry-specific dummy Z={geometry_dummy_z} for {geometry}")
    
    # Build RDKit mol with valid bond orders
    # IMPORTANT: MetalloGen sets all formal charges to 0 for embedding (chem.py:1913)
    logger.info("Step 4: Building RDKit molecule...")
    
    rd_mol = build_rdkit_mol(valid_atom_list, valid_bo_matrix, metal_idx, chg_list=None)
    if rd_mol is None:
        return EmbedResult(
            mol=None, success=False, metal_idx=0,
            coord_atom_indices=[], charge=total_charge, spin=t.spin or 1,
            diagnostics=diagnostics
        )
    
    diagnostics['rdkit_atoms'] = rd_mol.GetNumAtoms()
    
    # Constrained embedding with progressive relaxation
    logger.info("Step 5: Constrained embedding (with progressive relaxation)...")
    
    embedded_mol, embed_level, embed_level_name, clash_result = constrained_embed(
        rd_mol, metal_idx, metal_binding_infos, site_coords,
        max_attempts=max_attempts,
        seed=seed,
        max_relaxation_level=max_relaxation_level,
        reject_severe_clashes=reject_severe_clashes,
    )
    
    # Store embedding level info in diagnostics
    diagnostics['embedding_level'] = embed_level
    diagnostics['embedding_level_name'] = embed_level_name
    if clash_result:
        diagnostics['n_severe_clashes'] = clash_result.n_severe_clashes
        diagnostics['n_moderate_clashes'] = clash_result.n_moderate_clashes
        diagnostics['min_nonbonded_distance'] = clash_result.min_distance
    
    if embedded_mol is None:
        return EmbedResult(
            mol=None, success=False, metal_idx=0,
            coord_atom_indices=[], charge=total_charge, spin=t.spin or 1,
            diagnostics=diagnostics,
            embedding_level=embed_level,
            embedding_level_name=embed_level_name,
        )
    
    # Step 5b: Fix geometry of small ligands (water, NH3, etc.) that may have degenerate atoms
    logger.info("Step 5b: Fixing small ligand geometries...")
    embedded_mol = fix_small_ligand_geometry(
        embedded_mol, metal_idx, metal_binding_infos, ligand_atom_ranges,
        original_z=original_z,
        max_ligand_size=4
    )
    
    # Step 5c: Rotate small monodentate ligands so neighbors point away from metal
    logger.info("Step 5c: Rotating small monodentate ligands...")
    embedded_mol = rotate_monodentate_ligands(
        embedded_mol, metal_idx, metal_binding_infos, ligand_atom_ranges,
        max_ligand_size=5
    )
    
    # Cleanup - RESTORE ALL ORIGINAL ATOMS
    logger.info("Step 6: Restoring ALL original atoms and removing ghosts...")
    
    final_mol, new_metal_idx, index_map = restore_and_cleanup(
        embedded_mol, metal_idx, metal_z, t.ox, dummy_indices, original_z
    )
    
    # Get coordinating atom indices (for xTB constraints)
    # Map the metal_binding_infos to new indices after cleanup
    coord_atoms = set()
    mapped_binding_infos = []
    haptic_ring_infos = []  # Track ring atoms for haptic sites
    
    for binding_idx, site_num, is_multi, is_haptic in metal_binding_infos:
        if is_haptic:
            # Ghost atom is removed - we need to track the ring atoms instead
            # Find ring atoms that were connected to this ghost in the ORIGINAL mol
            if binding_idx < embedded_mol.GetNumAtoms():
                ghost_atom = embedded_mol.GetAtomWithIdx(binding_idx)
                ring_atoms_old = []
                for neighbor in ghost_atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    if n_idx != metal_idx and n_idx not in dummy_indices:
                        ring_atoms_old.append(n_idx)
                
                # Map to new indices
                ring_atoms_new = [index_map[i] for i in ring_atoms_old if i in index_map]
                if ring_atoms_new:
                    haptic_ring_infos.append({
                        'site_num': site_num,
                        'ring_atom_indices': ring_atoms_new,
                        'hapticity': len(ring_atoms_new),
                    })
        else:
            # Non-haptic - direct coordination
            if binding_idx not in dummy_indices and binding_idx in index_map:
                new_idx = index_map[binding_idx]
                coord_atoms.add(new_idx)
                mapped_binding_infos.append((new_idx, site_num, is_multi, is_haptic))
    
    coord_atom_indices = sorted(coord_atoms)
    
    # Store in diagnostics for xTB
    diagnostics['metal_binding_infos'] = mapped_binding_infos
    diagnostics['coord_atom_indices'] = coord_atom_indices
    diagnostics['haptic_ring_infos'] = haptic_ring_infos
    
    logger.info(f"Embedding complete: {final_mol.GetNumAtoms()} atoms")
    logger.info(f"  Coordinating atoms for xTB: {coord_atom_indices}")
    logger.info(f"  Haptic sites: {len(haptic_ring_infos)}")
    logger.info(f"  Embedding level: {embed_level} ({embed_level_name})")
    
    return EmbedResult(
        mol=final_mol,
        success=True,
        metal_idx=new_metal_idx,
        coord_atom_indices=coord_atom_indices,
        charge=total_charge,
        spin=t.spin or 1,
        diagnostics=diagnostics,
        embedding_level=embed_level,
        embedding_level_name=embed_level_name,
        clash_result=clash_result,
    )


# =============================================================================
# Utility Functions  
# =============================================================================

def write_xyz(mol: Chem.Mol, filename: str, comment: str = "") -> None:
    """Write molecule to XYZ file."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")
    
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{comment}\n")
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")


__all__ = [
    'EmbedResult',
    'ClashResult',
    'parse_ligand_smiles',
    'build_combined_molecule',
    'build_rdkit_mol',
    'constrained_embed',
    'check_atom_clashes',
    'restore_and_cleanup',
    'embed_trex_metallogen',
    'write_xyz',
    'get_vdw_radius',
    'VDW_RADII',
]
