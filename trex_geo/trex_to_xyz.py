#!/usr/bin/env python3
"""
T-REX to XYZ Structure Generation Pipeline.

Converts T-REX canonical strings to 3D molecular structures using:
1. Explicit internal coordinates (if provided)
2. ML-predicted internal coordinates (if model available)
3. Idealized geometry (fallback)

Usage:
    from trex_to_xyz import trex_to_xyz
    
    # With explicit internals
    mol = trex_to_xyz(trex_str, internals={'site_r': [...], 'trans_cos': [...], ...})
    
    # With LF-GNN ensemble prediction
    mol = trex_to_xyz(trex_str)
    
    # Force idealized geometry
    mol = trex_to_xyz(trex_str, use_idealized=True)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


# =============================================================================
# Import from trex package
# =============================================================================

try:
    from trex.parse_full import parse_trex_full
    from trex.model import TrexFull, MapSite, LigandPayload
    HAVE_TREX = True
except ImportError:
    HAVE_TREX = False
    logger.warning("trex package not available")


# =============================================================================
# Geometry mapping
# =============================================================================

# Maps geometry codes from TrexFull.get_geo_type() to full embedding names
GEO_CODE_TO_FULL = {
    # From get_geo_type() returns
    'O': '6_octahedral',
    'sqpl': '4_square_planar',
    'sqpy': '5_square_pyramidal',
    'thd': '4_tetrahedral',
    'trbp': '5_trigonal_bipyramidal',
    'pntbp': '7_pentagonal_bipyramidal',
    'ln': '2_linear',
    'trpl': '3_trigonal_planar',
    'tsh': '3_t_shaped',
    'ssw': '4_seesaw',
    'bnt': '2_bent',
    'unk': '6_octahedral',  # fallback
    # Also support uppercase codes (from G field directly)
    'SP': '4_square_planar',
    'SQP': '5_square_pyramidal',
    'T': '4_tetrahedral',
    'TBP': '5_trigonal_bipyramidal',
    'L': '2_linear',
    'TP': '3_trigonal_planar',
    'TPY': '3_trigonal_pyramidal',
}


def get_geometry_full_name(t: TrexFull) -> str:
    """Get full geometry name like '6_octahedral' from TrexFull."""
    # Use get_geo_type() which handles G field and inference
    code = t.get_geo_type()
    return GEO_CODE_TO_FULL.get(code, '6_octahedral')


# =============================================================================
# Structure Info and Validation
# =============================================================================

def get_structure_info(t: TrexFull) -> dict:
    """
    Get structural info for validating internals.
    
    Returns:
        dict with n_sites, n_trans_pairs, n_cis_pairs, cis_pair_indices
    """
    n_pairs = len(t.pairs)
    n_singles = len(t.singles)
    n_sites = 2 * n_pairs + n_singles
    n_trans_pairs = n_pairs
    
    # Build cis pair indices (all i < j not in same trans pair)
    trans_pair_set = set()
    for pi in range(n_pairs):
        idx_a = 2 * pi
        idx_b = 2 * pi + 1
        trans_pair_set.add((idx_a, idx_b))
        trans_pair_set.add((idx_b, idx_a))
    
    cis_pair_indices = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if (i, j) not in trans_pair_set:
                cis_pair_indices.append((i, j))
    
    n_cis_pairs = len(cis_pair_indices)
    
    return {
        'n_sites': n_sites,
        'n_trans_pairs': n_trans_pairs,
        'n_cis_pairs': n_cis_pairs,
        'cis_pair_indices': cis_pair_indices,
    }


def validate_internals(internals: dict, structure_info: dict) -> bool:
    """
    Validate that internals have correct dimensions for the structure.
    
    Args:
        internals: dict with site_r, trans_cos, cis_cos, cis_pair_indices
        structure_info: dict from get_structure_info
        
    Returns:
        True if valid, raises ValueError if not
    """
    n_sites = structure_info['n_sites']
    n_trans_pairs = structure_info['n_trans_pairs']
    n_cis_pairs = structure_info['n_cis_pairs']
    
    # Check site_r
    site_r = internals.get('site_r', [])
    if len(site_r) != n_sites:
        raise ValueError(f"site_r length {len(site_r)} != n_sites {n_sites}")
    
    # Check trans_cos
    trans_cos = internals.get('trans_cos', [])
    if len(trans_cos) != n_trans_pairs:
        raise ValueError(f"trans_cos length {len(trans_cos)} != n_trans_pairs {n_trans_pairs}")
    
    # Check cis_cos
    cis_cos = internals.get('cis_cos', [])
    if len(cis_cos) != n_cis_pairs:
        raise ValueError(f"cis_cos length {len(cis_cos)} != n_cis_pairs {n_cis_pairs}")
    
    # Check cis_pair_indices if provided
    cis_pair_indices = internals.get('cis_pair_indices', None)
    if cis_pair_indices is not None:
        if len(cis_pair_indices) != n_cis_pairs:
            raise ValueError(f"cis_pair_indices length {len(cis_pair_indices)} != n_cis_pairs {n_cis_pairs}")
    
    return True


# =============================================================================
# Internals Computation
# =============================================================================

def compute_idealized_internals(t: TrexFull) -> dict:
    """
    Compute idealized internal coordinates using trex.geometry.
    
    Args:
        t: TrexFull object
        
    Returns:
        dict with site_r, trans_cos, cis_cos, cis_pair_indices
    """
    from trex.geometry import (
        get_ideal_distance, 
        get_ideal_cis_cos,
        IDEAL_TRANS_COS,
    )
    
    structure_info = get_structure_info(t)
    n_sites = structure_info['n_sites']
    n_trans_pairs = structure_info['n_trans_pairs']
    cis_pair_indices = structure_info['cis_pair_indices']
    
    # Build ordered sites list
    sites_ordered = []
    for a, b in t.pairs:
        sites_ordered.append(a)
        sites_ordered.append(b)
    for s in t.singles:
        sites_ordered.append(s)
    
    # Compute ideal distances
    site_r = []
    for site in sites_ordered:
        lig_payload = t.ligands[site.lig - 1]
        lig_smi = lig_payload.text if lig_payload.kind == 'SMILES' else ''
        r_ideal = get_ideal_distance(t.metal, site, lig_smi)
        site_r.append(r_ideal)
    
    # Trans angles: always -1 (180°)
    trans_cos = [IDEAL_TRANS_COS] * n_trans_pairs
    
    # Cis angles: geometry-dependent
    geometry = t.get_geo_type()
    n_pair_sites = 2 * n_trans_pairs
    n_singles = len(t.singles)
    
    cis_cos = []
    for i, j in cis_pair_indices:
        site_i_in_trans = i < n_pair_sites
        site_j_in_trans = j < n_pair_sites
        
        site_i_singles_idx = None if site_i_in_trans else (i - n_pair_sites)
        site_j_singles_idx = None if site_j_in_trans else (j - n_pair_sites)
        
        ideal_cos = get_ideal_cis_cos(
            geometry=geometry,
            site_i_in_trans=site_i_in_trans,
            site_j_in_trans=site_j_in_trans,
            site_i_singles_idx=site_i_singles_idx,
            site_j_singles_idx=site_j_singles_idx,
            n_singles=n_singles,
        )
        cis_cos.append(ideal_cos)
    
    return {
        'site_r': site_r,
        'trans_cos': trans_cos,
        'cis_cos': cis_cos,
        'cis_pair_indices': cis_pair_indices,
    }



# =============================================================================
# Haptic Site Handling
# =============================================================================

def is_site_haptic(site: MapSite) -> bool:
    """Check if a site is haptic (multiple binding atoms)."""
    return len(site.atoms) > 1


def get_sites_ordered(t: TrexFull) -> List[MapSite]:
    """Get sites in order: pairs first (alternating), then singles."""
    sites = []
    for a, b in t.pairs:
        sites.append(a)
        sites.append(b)
    for s in t.singles:
        sites.append(s)
    return sites


def scale_haptic_distances(
    site_r: List[float],
    sites: List[MapSite],
    scale: float,
) -> List[float]:
    """Scale distances for haptic sites."""
    scaled = []
    for r, site in zip(site_r, sites):
        if is_site_haptic(site):
            scaled.append(r * scale)
        else:
            scaled.append(r)
    return scaled


def _try_predict_internals(trex_str: str, model_path: Optional[str]) -> Optional[dict]:
    """Run the optional ML predictor and allow idealized fallback if unavailable."""
    try:
        from .predict import predict_internals_with_model
    except ImportError as exc:
        logger.info("ML internals unavailable; using idealized internals: %s", exc)
        return None

    try:
        return predict_internals_with_model(trex_str, model_path)
    except Exception as exc:
        logger.warning("ML internals prediction failed; using idealized internals: %s", exc)
        return None


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class TrexToXyzResult:
    """Result from trex_to_xyz."""
    mol: Optional[Chem.Mol]
    success: bool
    internals_source: str  # "provided", "predicted", "idealized"
    error: Optional[str] = None
    energy: Optional[float] = None
    proj_error: Optional[float] = None
    diagnostics: dict = None


# =============================================================================
# Main Pipeline
# =============================================================================

def trex_to_xyz(
    trex_str: str,
    internals: Optional[dict] = None,
    use_idealized: bool = False,
    model_path: str = None,
    n_seeds: int = 5,
    top_k: int = 3,
    refine_gfn2: bool = True,
    haptic_scale: float = 0.35,
    work_dir: Optional[str] = None,
) -> TrexToXyzResult:
    """
    Convert T-REX string to 3D molecular structure.
    
    Internal coordinates are obtained in this priority:
    1. If `internals` is provided and valid, use it
    2. If model exists at `model_path`, use ML prediction
    3. Otherwise, use idealized geometry from trex.geometry
    
    Args:
        trex_str: Canonical T-REX string
        internals: Optional dict with site_r, trans_cos, cis_cos, cis_pair_indices
        use_idealized: Force use of idealized geometry (skip ML)
        model_path: Path to an ensemble directory containing seed_*/best.pt checkpoints
        n_seeds: Number of embedding seeds to try
        top_k: Keep top K conformers after GFN-FF
        refine_gfn2: Refine with GFN2-xTB
        haptic_scale: Distance scaling factor for haptic sites
        work_dir: Working directory for temporary files
        
    Returns:
        TrexToXyzResult with mol and metadata
    """
    from .coord_reconstruction import reconstruct_coordination_sphere
    from .trex_embed import embed_trex_metallogen
    from .multiseed_trex_validate import generate_and_validate
    from .xtb_optimize import HapticInfo
    
    if not HAVE_TREX:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source="none",
            error="trex package not available",
        )
    
    # Parse T-REX
    try:
        t = parse_trex_full(trex_str)
    except Exception as e:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source="none",
            error=f"Parse failed: {e}",
        )
    
    # Get geometry name
    geometry = get_geometry_full_name(t)
    
    # Get structure info for validation
    structure_info = get_structure_info(t)
    
    # Determine internals source
    internals_source = "none"
    final_internals = None
    proj_error = None
    
    # Option 1: Use provided internals
    if internals is not None:
        try:
            validate_internals(internals, structure_info)
            final_internals = internals
            internals_source = "provided"
            logger.info("Using provided internal coordinates")
        except ValueError as e:
            return TrexToXyzResult(
                mol=None, success=False,
                internals_source="none",
                error=f"Invalid internals: {e}",
            )
    
    # Option 2: ML prediction (if not forcing idealized)
    if final_internals is None and not use_idealized:
        predicted = _try_predict_internals(trex_str, model_path)
        if predicted is not None:
            final_internals = predicted
            internals_source = "predicted"
            proj_error = predicted.get('proj_error', None)
            logger.info("Using ML-predicted internal coordinates")
    
    # Option 3: Idealized geometry (fallback)
    if final_internals is None:
        try:
            final_internals = compute_idealized_internals(t)
            internals_source = "idealized"
            logger.info("Using idealized internal coordinates")
        except Exception as e:
            return TrexToXyzResult(
                mol=None, success=False,
                internals_source="none",
                error=f"Failed to compute idealized internals: {e}",
            )
    
    # Scale haptic distances
    sites_ordered = get_sites_ordered(t)
    haptic_indices = [i for i, s in enumerate(sites_ordered) if is_site_haptic(s)]
    
    if haptic_indices:
        scaled_site_r = scale_haptic_distances(
            final_internals['site_r'], sites_ordered, haptic_scale
        )
    else:
        scaled_site_r = final_internals['site_r']
    
    # Reconstruct coordination sphere
    try:
        cis_pair_indices = final_internals.get('cis_pair_indices', structure_info['cis_pair_indices'])
        cis_pair_indices = [tuple(p) for p in cis_pair_indices]
        
        site_coords, recon_diagnostics = reconstruct_coordination_sphere(
            t=t,
            site_r=scaled_site_r,
            trans_cos=final_internals['trans_cos'],
            cis_cos=final_internals['cis_cos'],
            cis_pair_indices=cis_pair_indices,
            expected_chirality=t.X,
        )
        site_coords = np.array(site_coords)
        
        if site_coords.ndim == 1:
            site_coords = site_coords.reshape(1, -1)
        if site_coords.shape[1] != 3:
            raise ValueError(f"Invalid site_coords shape: {site_coords.shape}")
            
    except Exception as e:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source=internals_source,
            error=f"Coord reconstruction failed: {e}",
        )
    
    # Initial embedding to get metal_idx, coord_atom_indices
    try:
        test_result = embed_trex_metallogen(
            t, site_coords, max_attempts=5, seed=42, geometry=geometry
        )
        
        if not test_result.success:
            return TrexToXyzResult(
                mol=None, success=False,
                internals_source=internals_source,
                error="Initial embedding failed",
            )
        
        metal_idx = test_result.metal_idx
        coord_atom_indices = test_result.coord_atom_indices
        charge = test_result.charge
        spin = test_result.spin
        
    except Exception as e:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source=internals_source,
            error=f"Embedding failed: {e}",
        )
    
    # Build haptic infos
    haptic_infos = []
    for hd in test_result.diagnostics.get('haptic_ring_infos', []):
        site_idx = hd['site_num'] - 1
        centroid_dist = final_internals['site_r'][site_idx] if site_idx < len(final_internals['site_r']) else 1.8
        haptic_infos.append(HapticInfo(
            ring_atom_indices=hd['ring_atom_indices'],
            hapticity=hd['hapticity'],
            m_centroid_dist=centroid_dist,
        ))
    
    # Convert geometry for xyz_to_trex validation format
    geometry_for_xyz2trex = None
    if geometry:
        parts = geometry.split('_', 1)
        if len(parts) == 2:
            geometry_for_xyz2trex = parts[1].replace('_', ' ')
    
    try:
        ms_result = generate_and_validate(
            t=t,
            site_coords=site_coords,
            original_trex=trex_str,
            metal_idx=metal_idx,
            coord_atom_indices=coord_atom_indices,
            haptic_infos=haptic_infos,
            charge=charge,
            spin=spin,
            geometry=geometry,
            geometry_for_validation=geometry_for_xyz2trex,
            n_seeds=n_seeds,
            top_k=top_k,
            refine_gfn2=refine_gfn2,
            skip_validation=True,  # Skip T-REX validation for now
            work_dir=work_dir,
        )
        
        if ms_result.best:
            return TrexToXyzResult(
                mol=ms_result.best.mol,
                success=True,
                internals_source=internals_source,
                proj_error = proj_error,
                energy=ms_result.best.energy,
                diagnostics={
                    'n_seeds': ms_result.n_seeds,
                    'n_embed_ok': ms_result.n_embed_ok,
                    'n_opt_ok': ms_result.n_opt_ok,
                    'best_seed': ms_result.best.seed,
                },
            )
        else:
            return TrexToXyzResult(
                mol=None, success=False,
                internals_source=internals_source,
                error="No valid conformers generated",
            )
            
    except Exception as e:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source=internals_source,
            error=f"Multi-seed pipeline failed: {e}",
        )


def trex_to_xyz_file(
    trex_str: str,
    output_path: str,
    **kwargs,
) -> bool:
    """
    Convenience function to write result directly to XYZ file.
    
    Args:
        trex_str: Canonical T-REX string
        output_path: Path to output XYZ file
        **kwargs: Arguments passed to trex_to_xyz
        
    Returns:
        True if successful
    """
    result = trex_to_xyz(trex_str, **kwargs)
    
    if not result.success or result.mol is None:
        logger.error(f"Failed: {result.error}")
        return False
    
    # Write XYZ
    mol = result.mol
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    
    with open(output_path, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"E={result.energy:.6f} source={result.internals_source}\n")
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():2s} {pos.x:15.8f} {pos.y:15.8f} {pos.z:15.8f}\n")
    
    logger.info(f"Written: {output_path}")
    return True


__all__ = [
    'trex_to_xyz',
    'trex_to_xyz_file',
    'TrexToXyzResult',
    'get_structure_info',
    'validate_internals',
    'compute_idealized_internals',
]
