#!/usr/bin/env python3
"""
T-REX to XYZ Structure Generation Pipeline — UMA variant.

Uses Meta FAIR Chemistry UMA models instead of xTB for geometry optimization.
Three-tier optimization:
    1. UMA-small constrained (fast screening, many seeds)
    2. UMA-medium constrained (refinement, top K from tier 1)
    3. UMA unconstrained (optional full relaxation, no frozen atoms)

Usage:
    from trex_to_xyz_uma import trex_to_xyz_uma

    # Default: small→medium tiered optimization
    result = trex_to_xyz_uma(trex_str, n_seeds=20, top_k=5)

    # With unconstrained refinement as final step
    result = trex_to_xyz_uma(trex_str, n_seeds=20, top_k=5, refine_unconstrained=True)
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Reuse shared components from the xTB pipeline
from .trex_to_xyz import (
    HAVE_TREX,
    get_geometry_full_name,
    get_structure_info,
    validate_internals,
    compute_idealized_internals,
    is_site_haptic,
    get_sites_ordered,
    scale_haptic_distances,
    TrexToXyzResult,
    _try_predict_internals,
)

from .multiseed_trex_validate import (
    SeedResult,
    MultiSeedResult,
    generate_with_seeds,
    validate_and_rank,
)

from .uma_optimize import (
    optimize_with_uma,
    get_uma_calculator,
    HapticInfo,
)


# =============================================================================
# UMA Conformer Optimization (replaces optimize_conformers)
# =============================================================================

def optimize_conformers_uma(
    results: List[SeedResult],
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List[HapticInfo] = None,
    charge: int = 0,
    spin: int = 1,
    model_name: str = "uma-s-1p1",
    device: Optional[str] = None,
    constrained: bool = True,
    haptic_force_constant: float = 0.5,
    fmax: float = 0.05,
    max_steps: int = 200,
    work_dir: Optional[str] = None,
) -> List[SeedResult]:
    """
    Optimize conformers using UMA.

    Drop-in replacement for multiseed_trex_validate.optimize_conformers,
    using UMA instead of xTB.

    Args:
        results: List of SeedResult with embedded molecules
        metal_idx: Metal atom index
        coord_atom_indices: Coordinating atoms to fix
        haptic_infos: Haptic site information
        charge: Total charge supplied to the UMA OMol model
        spin: Spin multiplicity supplied to the UMA OMol model
        model_name: UMA model name ("uma-s-1p1", "uma-m-1p1")
        device: Device for UMA ("cuda", "cpu", None=auto)
        constrained: Whether to freeze metal + donors
        haptic_force_constant: Spring constant for haptic bonds
        fmax: Force convergence (eV/Å)
        max_steps: Max optimization steps
        work_dir: Working directory

    Returns:
        Updated list of SeedResult with energies
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="uma_multiseed_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)

    if haptic_infos is None:
        haptic_infos = []

    # Pre-load the calculator so it's cached for all seeds
    get_uma_calculator(model_name=model_name, device=device)

    optimized = []

    for r in results:
        if not r.success or r.mol is None:
            optimized.append(r)
            continue

        seed_dir = work_path / f"seed_{r.seed}"
        seed_dir.mkdir(exist_ok=True)

        try:
            opt_result = optimize_with_uma(
                mol=r.mol,
                metal_idx=metal_idx,
                coord_atom_indices=coord_atom_indices,
                haptic_infos=haptic_infos,
                charge=charge,
                spin=spin,
                model_name=model_name,
                device=device,
                haptic_force_constant=haptic_force_constant,
                fmax=fmax,
                max_steps=max_steps,
                constrained=constrained,
                work_dir=str(seed_dir),
            )

            if opt_result.success:
                optimized.append(SeedResult(
                    seed=r.seed,
                    mol=opt_result.optimized_mol,
                    success=True,
                    energy=opt_result.energy_hartree,  # Hartree for compatibility
                    method=f"uma:{model_name}" + ("" if constrained else ":unconstrained"),
                ))
            else:
                optimized.append(SeedResult(
                    seed=r.seed,
                    mol=r.mol,
                    success=False,
                    error=f"UMA optimization failed: {opt_result.error_message[:100]}",
                ))

        except Exception as e:
            optimized.append(SeedResult(
                seed=r.seed,
                mol=r.mol,
                success=False,
                error=str(e)[:100],
            ))

    return optimized


# =============================================================================
# Main Pipeline
# =============================================================================

def generate_and_validate_uma(
    t,
    site_coords: np.ndarray,
    original_trex: str,
    metal_idx: int,
    coord_atom_indices: List[int],
    haptic_infos: List[HapticInfo] = None,
    charge: int = 0,
    spin: int = 1,
    geometry: str = None,
    geometry_for_validation: str = None,
    n_seeds: int = 20,
    seed_start: int = 42,
    seed_step: int = 12345,
    top_k: int = 5,
    require_trex_valid: bool = False,
    skip_validation: bool = False,
    # UMA-specific
    model_small: str = "uma-s-1p1",
    model_medium: str = "uma-m-1p1",
    refine_medium: bool = True,
    refine_unconstrained: bool = False,
    device: Optional[str] = None,
    fmax_small: float = 0.05,
    fmax_medium: float = 0.03,
    fmax_unconstrained: float = 0.02,
    max_steps_small: int = 200,
    max_steps_medium: int = 300,
    max_steps_unconstrained: int = 500,
    work_dir: str = None,
) -> MultiSeedResult:
    """
    Three-tier UMA pipeline: generate, optimize (small→medium→unconstrained), validate, rank.

    Tier 1: UMA-small constrained — fast screening on all seeds
    Tier 2: UMA-medium constrained — refinement on top_k
    Tier 3: UMA unconstrained (optional) — full relaxation on best candidate

    Args:
        t: TrexFull adapter
        site_coords: Site coordinates
        original_trex: Original T-REX string for validation
        metal_idx: Metal atom index
        coord_atom_indices: Coordinating atoms
        haptic_infos: Haptic site info
        charge: Total charge supplied to the UMA OMol model
        spin: Spin multiplicity supplied to the UMA OMol model
        geometry: Geometry for embedding (e.g., "6_octahedral")
        geometry_for_validation: Geometry for xyz_to_trex (e.g., "octahedral")
        n_seeds: Number of seeds (default 20 — UMA is fast)
        seed_start: Starting seed
        seed_step: Step between seeds
        top_k: Conformers to keep after tier 1
        require_trex_valid: Only keep T-REX valid conformers
        skip_validation: Skip T-REX validation entirely
        model_small: UMA model for tier 1 screening
        model_medium: UMA model for tier 2 refinement
        refine_medium: Whether to run tier 2 (default True)
        refine_unconstrained: Whether to run tier 3 unconstrained (default False)
        device: Device for UMA ("cuda", "cpu", None=auto)
        fmax_small: Force convergence for tier 1
        fmax_medium: Force convergence for tier 2
        fmax_unconstrained: Force convergence for tier 3
        max_steps_small: Max steps for tier 1
        max_steps_medium: Max steps for tier 2
        max_steps_unconstrained: Max steps for tier 3
        work_dir: Working directory

    Returns:
        MultiSeedResult with best conformer and all results
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="uma_pipeline_")
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True, parents=True)

    if haptic_infos is None:
        haptic_infos = []

    # Generate seeds
    seeds = [seed_start + i * seed_step for i in range(n_seeds)]
    logger.info(f"UMA multi-seed pipeline: {n_seeds} seeds, tiers: small"
                f"{'+medium' if refine_medium else ''}"
                f"{'+unconstrained' if refine_unconstrained else ''}")

    # =========================================================================
    # Step 1: Generate embeddings
    # =========================================================================
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

    # =========================================================================
    # Step 2: Tier 1 — UMA-small constrained (fast screening)
    # =========================================================================
    logger.info(f"Step 2: Tier 1 — UMA-small ({model_small}) constrained optimization...")
    tier1_dir = work_path / "tier1_small"
    results = optimize_conformers_uma(
        results=results,
        metal_idx=metal_idx,
        coord_atom_indices=coord_atom_indices,
        haptic_infos=haptic_infos,
        charge=charge,
        spin=spin,
        model_name=model_small,
        device=device,
        constrained=True,
        fmax=fmax_small,
        max_steps=max_steps_small,
        work_dir=str(tier1_dir),
    )
    n_opt_ok = sum(1 for r in results if r.success and r.energy < float('inf'))
    logger.info(f"  Tier 1 optimization: {n_opt_ok}/{n_embed_ok}")

    if n_opt_ok == 0:
        return MultiSeedResult(
            original_trex=original_trex,
            n_seeds=n_seeds,
            n_embed_ok=n_embed_ok,
            n_opt_ok=0,
            n_trex_valid=0,
            all_results=results,
        )

    # =========================================================================
    # Step 3: Validate and rank (or just rank by energy)
    # =========================================================================
    if skip_validation:
        logger.info("Step 3: Skipping T-REX validation, ranking by energy only...")
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
            geometry=geometry_for_validation,
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

    # =========================================================================
    # Step 4: Tier 2 — UMA-medium constrained (refinement)
    # =========================================================================
    if refine_medium and selected:
        logger.info(f"Step 4: Tier 2 — UMA-medium ({model_medium}) constrained refinement...")
        tier2_dir = work_path / "tier2_medium"
        selected = optimize_conformers_uma(
            results=selected,
            metal_idx=metal_idx,
            coord_atom_indices=coord_atom_indices,
            haptic_infos=haptic_infos,
            charge=charge,
            spin=spin,
            model_name=model_medium,
            device=device,
            constrained=True,
            fmax=fmax_medium,
            max_steps=max_steps_medium,
            work_dir=str(tier2_dir),
        )
        # Re-sort by new energies
        selected = [r for r in selected if r.success]
        selected.sort(key=lambda r: r.energy)
        if selected:
            e_min = selected[0].energy
            for r in selected:
                r.energy_kcal = (r.energy - e_min) * 627.509

    # =========================================================================
    # Step 5: Tier 3 — UMA unconstrained (optional, full relaxation)
    # =========================================================================
    if refine_unconstrained and selected:
        logger.info(f"Step 5: Tier 3 — UMA ({model_medium}) unconstrained full relaxation...")
        tier3_dir = work_path / "tier3_unconstrained"

        # Run unconstrained on all selected candidates
        unconstrained_results = optimize_conformers_uma(
            results=selected,
            metal_idx=metal_idx,
            coord_atom_indices=coord_atom_indices,
            haptic_infos=haptic_infos,
            charge=charge,
            spin=spin,
            model_name=model_medium,
            device=device,
            constrained=False,
            fmax=fmax_unconstrained,
            max_steps=max_steps_unconstrained,
            work_dir=str(tier3_dir),
        )

        # Keep unconstrained results alongside constrained for comparison
        unconstrained_ok = [r for r in unconstrained_results if r.success]
        if unconstrained_ok:
            unconstrained_ok.sort(key=lambda r: r.energy)
            logger.info(f"  Tier 3: {len(unconstrained_ok)}/{len(selected)} succeeded")

            # Use unconstrained results as the final selected set
            selected = unconstrained_ok
            if selected:
                e_min = selected[0].energy
                for r in selected:
                    r.energy_kcal = (r.energy - e_min) * 627.509
        else:
            logger.warning("  Tier 3: All unconstrained optimizations failed, keeping constrained results")

    best = selected[0] if selected else None
    if best:
        logger.info(f"  Best: seed={best.seed}, E={best.energy:.6f} Eh, method={best.method}")

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
# Top-Level Entry Point
# =============================================================================

def trex_to_xyz_uma(
    trex_str: str,
    internals: Optional[dict] = None,
    use_idealized: bool = False,
    model_path: str = None,
    n_seeds: int = 20,
    top_k: int = 5,
    haptic_scale: float = 0.35,
    # UMA-specific
    model_small: str = "uma-s-1p1",
    model_medium: str = "uma-m-1p1",
    refine_medium: bool = True,
    refine_unconstrained: bool = False,
    device: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> TrexToXyzResult:
    """
    Convert T-REX string to 3D molecular structure using UMA optimization.

    Internal coordinates are obtained in this priority:
    1. If `internals` is provided and valid, use it
    2. If model exists at `model_path`, use ML prediction
    3. Otherwise, use idealized geometry from trex.geometry

    Args:
        trex_str: Canonical T-REX string
        internals: Optional dict with site_r, trans_cos, cis_cos, cis_pair_indices
        use_idealized: Force use of idealized geometry (skip ML)
        model_path: Path to an ensemble directory containing seed_*/best.pt checkpoints
        n_seeds: Number of embedding seeds (default 20, more than xTB since UMA is faster)
        top_k: Keep top K conformers after tier 1
        haptic_scale: Distance scaling factor for haptic sites
        model_small: UMA model for tier 1
        model_medium: UMA model for tier 2
        refine_medium: Whether to run tier 2 medium refinement
        refine_unconstrained: Whether to run tier 3 unconstrained relaxation
        device: Device for UMA
        work_dir: Working directory for temporary files

    Returns:
        TrexToXyzResult with mol and metadata
    """
    from .coord_reconstruction import reconstruct_coordination_sphere
    from .trex_embed import embed_trex_metallogen

    if not HAVE_TREX:
        return TrexToXyzResult(
            mol=None, success=False,
            internals_source="none",
            error="trex package not available",
        )

    # Parse T-REX
    try:
        from trex.parse_full import parse_trex_full
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

    # Option 2: ML prediction
    if final_internals is None and not use_idealized:
        predicted = _try_predict_internals(trex_str, model_path)
        if predicted is not None:
            final_internals = predicted
            internals_source = "predicted"
            proj_error = predicted.get('proj_error', None)
            logger.info("Using ML-predicted internal coordinates")

    # Option 3: Idealized geometry
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

    # Geometry for validation
    geometry_for_xyz2trex = None
    if geometry:
        parts = geometry.split('_', 1)
        if len(parts) == 2:
            geometry_for_xyz2trex = parts[1].replace('_', ' ')

    # Run UMA multi-seed pipeline
    try:
        ms_result = generate_and_validate_uma(
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
            skip_validation=True,  # Skip T-REX validation for now
            model_small=model_small,
            model_medium=model_medium,
            refine_medium=refine_medium,
            refine_unconstrained=refine_unconstrained,
            device=device,
            work_dir=work_dir,
        )

        if ms_result.best:
            return TrexToXyzResult(
                mol=ms_result.best.mol,
                success=True,
                internals_source=internals_source,
                proj_error=proj_error,
                energy=ms_result.best.energy,
                diagnostics={
                    'n_seeds': ms_result.n_seeds,
                    'n_embed_ok': ms_result.n_embed_ok,
                    'n_opt_ok': ms_result.n_opt_ok,
                    'best_seed': ms_result.best.seed,
                    'best_method': ms_result.best.method,
                    'optimizer': 'uma',
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
            error=f"UMA multi-seed pipeline failed: {e}",
        )


def trex_to_xyz_uma_file(
    trex_str: str,
    output_path: str,
    **kwargs,
) -> bool:
    """
    Convenience function to write result directly to XYZ file.

    Args:
        trex_str: Canonical T-REX string
        output_path: Path to output XYZ file
        **kwargs: Arguments passed to trex_to_xyz_uma

    Returns:
        True if successful
    """
    result = trex_to_xyz_uma(trex_str, **kwargs)

    if not result.success or result.mol is None:
        logger.error(f"Failed: {result.error}")
        return False

    mol = result.mol
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    method = result.diagnostics.get('best_method', 'uma') if result.diagnostics else 'uma'

    with open(output_path, 'w') as f:
        f.write(f"{n}\n")
        f.write(f"E={result.energy:.6f} source={result.internals_source} method={method}\n")
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():2s} {pos.x:15.8f} {pos.y:15.8f} {pos.z:15.8f}\n")

    logger.info(f"Written: {output_path}")
    return True


__all__ = [
    'trex_to_xyz_uma',
    'trex_to_xyz_uma_file',
    'generate_and_validate_uma',
    'optimize_conformers_uma',
]
