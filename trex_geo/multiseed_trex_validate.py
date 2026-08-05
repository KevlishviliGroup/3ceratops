#!/usr/bin/env python3
"""
Multi-Seed Conformer Generation with T-REX Back-Validation.

Workflow:
1. Generate N embeddings with different random seeds  
2. GFN-FF optimization on all (fast screening)
3. T-REX back-validation using trex.xyz_to_trex
4. Rank by energy, filter by T-REX validity
5. Optional GFN2-xTB refinement on top K

Usage:
    from multiseed_trex_validate import generate_and_validate
    
    result = generate_and_validate(
        t=trex_adapter,
        site_coords=site_coords,
        original_trex=trex_string,
        n_seeds=5,
        top_k=3,
    )
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from trex.model import TrexFull

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SeedResult:
    """Result from a single seed attempt."""
    seed: int
    mol: Optional[Chem.Mol] = None
    success: bool = False
    energy: float = float('inf')
    energy_kcal: float = float('inf')  # Relative to lowest
    method: str = ""
    trex_valid: Optional[bool] = None  # None=not checked
    trex_reconstructed: Optional[str] = None
    error: str = ""


@dataclass 
class MultiSeedResult:
    """Result from multi-seed pipeline."""
    original_trex: str
    n_seeds: int
    n_embed_ok: int
    n_opt_ok: int
    n_trex_valid: int
    best: Optional[SeedResult] = None
    all_results: List[SeedResult] = field(default_factory=list)
    selected: List[SeedResult] = field(default_factory=list)


# =============================================================================
# T-REX Back-Validation
# =============================================================================

def mol_to_xyz_string(mol: Chem.Mol) -> str:
    """Convert RDKit mol to XYZ string."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), ""]
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s} {pos.x:15.8f} {pos.y:15.8f} {pos.z:15.8f}")
    return "\n".join(lines)


def write_xyz_file(mol: Chem.Mol, path: str, comment: str = "") -> None:
    """Write RDKit mol to XYZ file."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    with open(path, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{comment}\n")
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():2s} {pos.x:15.8f} {pos.y:15.8f} {pos.z:15.8f}\n")


def canonicalize_trex_string(trex_str: str) -> Optional[str]:
    """
    Canonicalize a T-REX string.
    
    Uses trex.parse_full and trex.canon_full.
    """
    try:
        from trex.parse_full import parse_trex_full_string
        from trex.canon_full import canonicalize_full, pretty_trex_full
        
        t = parse_trex_full_string(trex_str)
        t_canon = canonicalize_full(t)
        return pretty_trex_full(t_canon)
    except Exception as e:
        logger.debug(f"Canonicalization failed: {e}")
        return None


def validate_trex_roundtrip(
    mol: Chem.Mol,
    original_trex: str,
    charge: int,
    geometry: str = None,
    work_dir: str = None,
    timeout_seconds: int = 30,
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Validate structure by converting back to T-REX and comparing.
    
    Uses trex.xyz_to_trex.xyz_to_trex_canonical which takes a FILE PATH.
    
    Args:
        mol: Optimized RDKit molecule
        original_trex: Original input T-REX string
        charge: Total molecular charge
        geometry: Optional geometry hint (e.g., "octahedral")
        work_dir: Directory for temporary XYZ file
        timeout_seconds: Max time for xyz_to_trex (default 30s)
        
    Returns:
        (is_valid, reconstructed_trex) - is_valid is None if validation unavailable/failed
    """
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("xyz_to_trex timed out")
    
    try:
        from trex.xyz_to_trex import xyz_to_trex_canonical
    except ImportError:
        logger.debug("trex.xyz_to_trex not available")
        return None, None
    
    # Write mol to temporary XYZ file
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="trex_validate_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)
    
    xyz_path = work_path / "structure.xyz"
    
    try:
        write_xyz_file(mol, str(xyz_path))
        
        # Set timeout for xyz_to_trex
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            # Convert XYZ back to canonical T-REX
            reconstructed = xyz_to_trex_canonical(
                str(xyz_path), 
                overall_charge=charge,
                geometry=geometry,
            )
        finally:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
        
        if reconstructed is None:
            return False, None
        
        # Canonicalize original for comparison
        original_canon = canonicalize_trex_string(original_trex)
        
        if original_canon is None:
            # Fall back to direct comparison
            is_valid = (original_trex.strip() == reconstructed.strip())
        else:
            is_valid = (original_canon.strip() == reconstructed.strip())
        
        return is_valid, reconstructed
    
    except TimeoutError:
        logger.warning(f"T-REX validation timed out after {timeout_seconds}s")
        return None, None
    except Exception as e:
        logger.warning(f"T-REX validation failed: {e}")
        return None, None


# =============================================================================
# Multi-Seed Generation
# =============================================================================

def generate_with_seeds(
    t: 'TrexFull',
    site_coords: np.ndarray,
    seeds: List[int],
    max_attempts_per_seed: int = 5,
    geometry: str = None,
) -> List[SeedResult]:
    """
    Generate conformers with multiple random seeds.
    
    Args:
        t: TrexFull adapter object
        site_coords: Site coordinates from coord_reconstruction
        seeds: List of random seeds to try
        max_attempts_per_seed: Internal attempts per seed
        geometry: Geometry name like "6_octahedral"
        
    Returns:
        List of SeedResult for each seed
    """
    from .trex_embed import embed_trex_metallogen
    
    results = []
    
    for seed in seeds:
        try:
            result = embed_trex_metallogen(
                t=t,
                site_coords=site_coords,
                max_attempts=max_attempts_per_seed,
                seed=seed,
                geometry=geometry,
            )
            
            results.append(SeedResult(
                seed=seed,
                mol=result.mol,
                success=result.success,
            ))
            
        except Exception as e:
            results.append(SeedResult(
                seed=seed,
                success=False,
                error=str(e)[:200],
            ))
    
    return results


def optimize_conformers(
    results: List[SeedResult],
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List = None,
    charge: int = 0,
    spin: int = 1,
    method: str = "GFN-FF",
    work_dir: str = None,
) -> List[SeedResult]:
    """
    Optimize conformers using xTB.
    
    Args:
        results: List of SeedResult with embedded molecules
        metal_idx: Metal atom index
        coord_atom_indices: Coordinating atoms to fix
        haptic_infos: Haptic site information
        charge: Total charge
        spin: Spin multiplicity
        method: "GFN-FF" or "GFN2-xTB"
        work_dir: Working directory
        
    Returns:
        Updated list with energies
    """
    from .xtb_optimize import optimize_with_xtb
    
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="multiseed_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)
    
    if haptic_infos is None:
        haptic_infos = []
    
    optimized = []
    
    for r in results:
        if not r.success or r.mol is None:
            optimized.append(r)
            continue
        
        seed_dir = work_path / f"seed_{r.seed}"
        seed_dir.mkdir(exist_ok=True)
        
        try:
            opt_result = optimize_with_xtb(
                mol=r.mol,
                metal_idx=metal_idx,
                coord_atom_indices=coord_atom_indices,
                haptic_infos=haptic_infos,
                charge=charge,
                spin=spin,
                method=method,
                work_dir=str(seed_dir),
                fallback_to_gfnff=(method != "GFN-FF"),
            )
            
            if opt_result.success:
                optimized.append(SeedResult(
                    seed=r.seed,
                    mol=opt_result.optimized_mol,
                    success=True,
                    energy=opt_result.energy,
                    method=opt_result.backend,
                ))
            else:
                optimized.append(SeedResult(
                    seed=r.seed,
                    mol=r.mol,
                    success=False,
                    error=f"Optimization failed: {opt_result.error_message[:100]}",
                ))
                
        except Exception as e:
            optimized.append(SeedResult(
                seed=r.seed,
                mol=r.mol,
                success=False,
                error=str(e)[:100],
            ))
    
    return optimized


def validate_and_rank(
    results: List[SeedResult],
    original_trex: str,
    charge: int,
    geometry: str = None,
    top_k: int = 3,
    require_trex_valid: bool = False,
    work_dir: str = None,
) -> List[SeedResult]:
    """
    Validate T-REX round-trip and rank by energy.
    
    Args:
        results: Optimized conformers
        original_trex: Original T-REX string
        charge: Total charge
        geometry: Geometry hint for xyz_to_trex (e.g., "octahedral")
        top_k: Number of conformers to keep
        require_trex_valid: If True, only keep T-REX valid conformers
        work_dir: Working directory for temp files
        
    Returns:
        Top K conformers sorted by energy
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="trex_validate_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)
    
    # Validate each conformer
    for i, r in enumerate(results):
        if not r.success or r.mol is None:
            continue
        
        seed_work_dir = work_path / f"validate_seed_{r.seed}"
        seed_work_dir.mkdir(exist_ok=True)
        
        is_valid, reconstructed = validate_trex_roundtrip(
            mol=r.mol,
            original_trex=original_trex,
            charge=charge,
            geometry=geometry,
            work_dir=str(seed_work_dir),
        )
        r.trex_valid = is_valid
        r.trex_reconstructed = reconstructed
    
    # Filter successful ones
    valid = [r for r in results if r.success and r.energy < float('inf')]
    
    # Optionally filter by T-REX validity
    if require_trex_valid:
        valid = [r for r in valid if r.trex_valid is True]
    
    # Sort by energy
    valid.sort(key=lambda r: r.energy)
    
    # Calculate relative energies
    if valid:
        e_min = valid[0].energy
        for r in valid:
            r.energy_kcal = (r.energy - e_min) * 627.509
    
    return valid[:top_k]


# =============================================================================
# Main Pipeline
# =============================================================================

def generate_and_validate(
    t: 'TrexFull',
    site_coords: np.ndarray,
    original_trex: str,
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List = None,
    charge: int = 0,
    spin: int = 1,
    geometry: str = None,
    geometry_for_validation: str = None,
    n_seeds: int = 5,
    seed_start: int = 42,
    seed_step: int = 12345,
    top_k: int = 3,
    require_trex_valid: bool = False,
    refine_gfn2: bool = True,
    skip_validation: bool = False,
    work_dir: str = None,
) -> MultiSeedResult:
    """
    Main pipeline: generate, optimize, validate, rank.
    
    Args:
        t: TrexFull adapter
        site_coords: Site coordinates
        original_trex: Original T-REX string for validation
        metal_idx: Metal atom index
        coord_atom_indices: Coordinating atoms
        haptic_infos: Haptic site info
        charge: Total charge
        spin: Spin multiplicity
        geometry: Geometry for embedding (e.g., "6_octahedral")
        geometry_for_validation: Geometry for xyz_to_trex (e.g., "octahedral")
        n_seeds: Number of seeds to try
        seed_start: Starting seed
        seed_step: Step between seeds
        top_k: Conformers to keep after GFN-FF
        require_trex_valid: Only keep T-REX valid conformers
        refine_gfn2: Refine top K with GFN2-xTB
        skip_validation: Skip T-REX validation entirely
        work_dir: Working directory
        
    Returns:
        MultiSeedResult with best conformer and all results
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="multiseed_pipeline_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)
    
    if haptic_infos is None:
        haptic_infos = []
    
    # Generate seeds
    seeds = [seed_start + i * seed_step for i in range(n_seeds)]
    logger.info(f"Multi-seed pipeline: {n_seeds} seeds")
    
    # Step 1: Generate embeddings
    logger.info("Step 1: Generating embeddings...")
    results = generate_with_seeds(t, site_coords, seeds, geometry=geometry)
    n_embed_ok = sum(1 for r in results if r.success)
    logger.info(f"  Embedding: {n_embed_ok}/{n_seeds}")
    
    if n_embed_ok == 0:
        return MultiSeedResult(
            original_trex=original_trex,
            n_seeds=n_seeds,
            n_embed_ok=0,
            n_opt_ok=0,
            n_trex_valid=0,
            all_results=results,
        )
    
    # Step 2: GFN-FF optimization
    logger.info("Step 2: GFN-FF optimization...")
    gfnff_dir = work_path / "gfnff"
    results = optimize_conformers(
        results=results,
        metal_idx=metal_idx,
        coord_atom_indices=coord_atom_indices,
        haptic_infos=haptic_infos,
        charge=charge,
        spin=spin,
        method="GFN-FF",
        work_dir=str(gfnff_dir),
    )
    n_opt_ok = sum(1 for r in results if r.success and r.energy < float('inf'))
    logger.info(f"  Optimization: {n_opt_ok}/{n_embed_ok}")
    
    if n_opt_ok == 0:
        return MultiSeedResult(
            original_trex=original_trex,
            n_seeds=n_seeds,
            n_embed_ok=n_embed_ok,
            n_opt_ok=0,
            n_trex_valid=0,
            all_results=results,
        )
    
    # Step 3: Validate and rank
    if skip_validation:
        logger.info("Step 3: Skipping T-REX validation, ranking by energy only...")
        # Just filter successful and sort by energy
        selected = [r for r in results if r.success and r.energy < float('inf')]
        selected.sort(key=lambda r: r.energy)
        if selected:
            e_min = selected[0].energy
            for r in selected:
                r.energy_kcal = (r.energy - e_min) * 627.509
        selected = selected[:top_k]
        n_trex_valid = 0
    else:
        logger.info("Step 3: T-REX validation and ranking...")
        validate_dir = work_path / "validate"
        selected = validate_and_rank(
            results=results,
            original_trex=original_trex,
            charge=charge,
            geometry=geometry_for_validation,  # Use short format for xyz_to_trex
            top_k=top_k,
            require_trex_valid=require_trex_valid,
            work_dir=str(validate_dir),
        )
        n_trex_valid = sum(1 for r in results if r.trex_valid is True)
        logger.info(f"  T-REX valid: {n_trex_valid}/{n_opt_ok}")
    logger.info(f"  Selected: {len(selected)}")
    
    if not selected:
        return MultiSeedResult(
            original_trex=original_trex,
            n_seeds=n_seeds,
            n_embed_ok=n_embed_ok,
            n_opt_ok=n_opt_ok,
            n_trex_valid=n_trex_valid,
            all_results=results,
        )
    
    # Step 4: GFN2-xTB refinement (optional)
    if refine_gfn2 and selected:
        logger.info("Step 4: GFN2-xTB refinement...")
        gfn2_dir = work_path / "gfn2"
        selected = optimize_conformers(
            results=selected,
            metal_idx=metal_idx,
            coord_atom_indices=coord_atom_indices,
            haptic_infos=haptic_infos,
            charge=charge,
            spin=spin,
            method="GFN2-xTB",
            work_dir=str(gfn2_dir),
        )
        # Re-sort by new energies
        selected = [r for r in selected if r.success]
        selected.sort(key=lambda r: r.energy)
        if selected:
            e_min = selected[0].energy
            for r in selected:
                r.energy_kcal = (r.energy - e_min) * 627.509
    
    best = selected[0] if selected else None
    if best:
        logger.info(f"  Best: seed={best.seed}, E={best.energy:.6f} Eh")
    
    return MultiSeedResult(
        original_trex=original_trex,
        n_seeds=n_seeds,
        n_embed_ok=n_embed_ok,
        n_opt_ok=n_opt_ok,
        n_trex_valid=n_trex_valid,
        best=best,
        all_results=results,
        selected=selected,
    )


# =============================================================================
# Utility
# =============================================================================

def write_seed_xyz(result: SeedResult, path: str, comment: str = "") -> None:
    """Write a SeedResult to XYZ file."""
    if result.mol is None:
        raise ValueError("No molecule")
    if not comment:
        comment = f"seed={result.seed} E={result.energy:.6f} Eh"
    write_xyz_file(result.mol, path, comment)


__all__ = [
    'SeedResult',
    'MultiSeedResult', 
    'generate_and_validate',
    'validate_trex_roundtrip',
    'canonicalize_trex_string',
    'write_seed_xyz',
]
