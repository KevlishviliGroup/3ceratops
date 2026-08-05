# trex_geo/coord_reconstruction.py
"""
Coordinate reconstruction from internal coordinates via Gram matrix projection.

Components 1-4 of the T-REX-geo pipeline:
1. Gram matrix builder
2. Rank-3 projector (ensures Euclidean embeddability)
3. Coordinate extractor
4. Chirality enforcer

This module takes distances (r) and angles (cos_θ) and produces 3D coordinates
that are guaranteed to be valid in Euclidean space.

FIXED: Now properly handles linear and bent geometries (n_sites <= 2)
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trex.model import TrexFull, MapSite


# =============================================================================
# Component 1: Gram Matrix Builder
# =============================================================================

def build_cos_theta_matrix(
    n_sites: int,
    n_trans_pairs: int,
    trans_cos: list[float],
    cis_cos: list[float],
    cis_pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build the full cos(θ) matrix from trans and cis angle data.
    
    The matrix is symmetric with 1.0 on diagonal (cos(0) = 1).
    
    Args:
        n_sites: Total number of coordination sites
        n_trans_pairs: Number of trans pairs
        trans_cos: cos(θ) values for trans pairs (one per pair)
        cis_cos: cos(θ) values for cis pairs
        cis_pair_indices: (i, j) site index pairs for cis angles
        
    Returns:
        Symmetric (n_sites × n_sites) matrix of cos(θ_ij) values
    """
    cos_theta = np.eye(n_sites)  # diagonal = 1 (same site, angle = 0)
    
    # Fill in trans pairs
    # Trans pairs are at indices: (0,1), (2,3), (4,5), ...
    for pi in range(n_trans_pairs):
        i = 2 * pi
        j = 2 * pi + 1
        cos_theta[i, j] = trans_cos[pi]
        cos_theta[j, i] = trans_cos[pi]
    
    # Fill in cis pairs
    for (i, j), cos_val in zip(cis_pair_indices, cis_cos):
        cos_theta[i, j] = cos_val
        cos_theta[j, i] = cos_val
    
    return cos_theta


def build_gram_matrix(r: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """
    Build the Gram matrix from distances and angles.
    
    The Gram matrix encodes inner products of position vectors:
        G_ij = <v_i, v_j> = |v_i| |v_j| cos(θ_ij) = r_i * r_j * cos(θ_ij)
    
    A valid Euclidean embedding exists iff G is positive semidefinite
    with rank ≤ 3.
    
    Args:
        r: Array of M-site distances (n_sites,)
        cos_theta: Matrix of cos(θ_ij) values (n_sites × n_sites)
        
    Returns:
        Gram matrix G (n_sites × n_sites)
    """
    r = np.asarray(r)
    cos_theta = np.asarray(cos_theta)
    
    # G_ij = r_i * r_j * cos(θ_ij)
    G = np.outer(r, r) * cos_theta
    
    return G


# =============================================================================
# Component 2: Rank-3 Projector
# =============================================================================

def project_rank3(G: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
    """
    Project Gram matrix to rank 3 via SVD truncation.
    
    This ensures the resulting matrix corresponds to points embeddable
    in 3D Euclidean space. Any "impossible" geometry from noisy predictions
    is projected to the nearest valid geometry.
    
    Args:
        G: Gram matrix (n_sites × n_sites)
        regularization: Small value to handle numerical issues
        
    Returns:
        Rank-3 projected Gram matrix
    """
    # Symmetrize (should already be symmetric, but ensure numerical stability)
    G = 0.5 * (G + G.T)
    
    n_sites = G.shape[0]
    
    # For very small matrices (n_sites <= 3), no projection needed
    if n_sites <= 3:
        # Just ensure positive semi-definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        eigenvalues = np.maximum(eigenvalues, regularization)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    
    # Truncate to rank 3
    S_truncated = S.copy()
    S_truncated[3:] = 0.0
    
    # Handle negative eigenvalues (shouldn't happen for valid Gram matrix,
    # but predictions might produce them)
    S_truncated = np.maximum(S_truncated, regularization)
    
    # Reconstruct
    G_proj = U @ np.diag(S_truncated) @ Vt
    
    return G_proj


def compute_projection_error(G_original: np.ndarray, G_projected: np.ndarray) -> dict:
    """
    Compute metrics for how much the rank-3 projection changed the Gram matrix.
    
    Useful for diagnostics: large errors indicate the predicted internal
    coordinates were geometrically inconsistent.
    
    Args:
        G_original: Original Gram matrix
        G_projected: Rank-3 projected Gram matrix
        
    Returns:
        Dictionary with error metrics
    """
    diff = G_original - G_projected
    
    # Frobenius norm of difference
    frobenius_error = np.linalg.norm(diff, 'fro')
    
    # Max absolute element-wise difference
    max_error = np.max(np.abs(diff))
    
    # Relative error
    G_norm = np.linalg.norm(G_original, 'fro')
    relative_error = frobenius_error / G_norm if G_norm > 1e-10 else 0.0
    
    # Eigenvalue spectrum of original (for diagnostics)
    eigenvalues = np.linalg.eigvalsh(G_original)
    n_negative = np.sum(eigenvalues < -1e-10)
    rank_deficit = np.sum(eigenvalues[:-3] > 1e-10) if len(eigenvalues) > 3 else 0
    
    return {
        'frobenius_error': float(frobenius_error),
        'max_error': float(max_error),
        'relative_error': float(relative_error),
        'n_negative_eigenvalues': int(n_negative),
        'rank_above_3': int(rank_deficit),
    }


# =============================================================================
# Component 3: Coordinate Extractor  
# =============================================================================

def extract_coordinates_linear(r: np.ndarray, cos_theta: float) -> np.ndarray:
    """
    Extract 3D coordinates for linear/bent 2-coordinate geometries.
    
    For 2 sites, we can construct coordinates directly:
    - Site 0 at (0, 0, r[0])
    - Site 1 positioned such that the angle between them (through origin) 
      has cosine = cos_theta
    
    Args:
        r: Array of 2 distances [r0, r1]
        cos_theta: cos(θ) between the two M-L vectors
        
    Returns:
        Coordinate array (2, 3), metal at origin
    """
    r0, r1 = r[0], r[1]
    
    # Place first site along z-axis
    site0 = np.array([0.0, 0.0, r0])
    
    # Place second site in xz-plane
    # cos(θ) = z1/r1 => z1 = r1 * cos(θ)
    # sin(θ) = x1/r1 => x1 = r1 * sin(θ)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
    site1 = np.array([r1 * sin_theta, 0.0, r1 * cos_theta])
    
    return np.array([site0, site1])


def extract_coordinates(G_proj: np.ndarray, reference_frame: np.ndarray | None = None) -> np.ndarray:
    """
    Extract 3D coordinates from projected Gram matrix.
    
    Since G = X @ X.T (where X is n_sites × 3), we can recover X from eigendecomposition:
        G = V @ Λ @ V.T
        X = V[:, :3] @ sqrt(Λ[:3, :3])
    
    The metal is implicitly at the origin; each row of X is a site vector.
    
    Note on degeneracy: For highly symmetric geometries (e.g., perfect octahedral),
    eigenvalues may be nearly degenerate, making the coordinate frame arbitrary.
    We handle this by:
    1. Sorting eigenvalues consistently
    2. Applying a deterministic sign convention to eigenvectors
    3. Optionally aligning to a reference frame if provided
    
    FIXED: Now handles n_sites < 3 by padding coordinates to 3D.
    
    Args:
        G_proj: Rank-3 projected Gram matrix
        reference_frame: Optional (3, 3) matrix to align axes to (for consistency)
        
    Returns:
        Coordinate array (n_sites × 3), metal at origin
    """
    n_sites = G_proj.shape[0]
    
    # Symmetrize
    G_proj = 0.5 * (G_proj + G_proj.T)
    
    # Use eigendecomposition (more stable for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(G_proj)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Handle small systems (n_sites < 3)
    # The number of dimensions is min(n_sites, 3)
    n_dims = min(n_sites, 3)
    
    # Take top n_dims eigenvalues/vectors, handle numerical issues
    top_vals = np.maximum(eigenvalues[:n_dims], 0.0)
    top_vecs = eigenvectors[:, :n_dims]
    
    # Apply deterministic sign convention to eigenvectors:
    # Make the largest component of each eigenvector positive
    for i in range(n_dims):
        max_idx = np.argmax(np.abs(top_vecs[:, i]))
        if top_vecs[max_idx, i] < 0:
            top_vecs[:, i] *= -1
    
    # Coordinates in n_dims space: X = V @ sqrt(Λ)
    coords_reduced = top_vecs * np.sqrt(top_vals)
    
    # Pad to 3D if necessary
    if n_dims < 3:
        coords = np.zeros((n_sites, 3))
        coords[:, :n_dims] = coords_reduced
    else:
        coords = coords_reduced
    
    # Optional: align to reference frame for consistency across reconstructions
    if reference_frame is not None and coords.shape[0] >= 3:
        # Procrustes alignment to reference
        coords = _align_to_reference(coords, reference_frame)
    
    return coords


def _align_to_reference(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align coordinate frame to a reference using Procrustes.
    
    This ensures consistent orientation across different reconstructions,
    which is important when eigenvalues are degenerate.
    """
    # Simple alignment: find rotation that best maps coords to reference
    # Using SVD of cross-covariance
    H = coords.T @ reference
    U, S, Vt = np.linalg.svd(H)
    
    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        Vt[-1, :] *= -1
    
    R = Vt.T @ U.T
    return coords @ R


def coords_to_internal(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert 3D coordinates back to internal coordinates (r, cos_θ).
    
    Useful for validating round-trip consistency.
    
    Args:
        coords: Site coordinates (n_sites × 3), metal at origin
        
    Returns:
        r: distances (n_sites,)
        cos_theta: angle matrix (n_sites × n_sites)
    """
    n_sites = coords.shape[0]
    
    # Distances
    r = np.linalg.norm(coords, axis=1)
    
    # Angles
    cos_theta = np.eye(n_sites)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if r[i] > 1e-10 and r[j] > 1e-10:
                cos_val = np.dot(coords[i], coords[j]) / (r[i] * r[j])
                # Clamp to valid range
                cos_val = np.clip(cos_val, -1.0, 1.0)
            else:
                cos_val = 0.0
            cos_theta[i, j] = cos_val
            cos_theta[j, i] = cos_val
    
    return r, cos_theta


# =============================================================================
# Component 4: Chirality Enforcer
# =============================================================================

def attach_coords_to_trex(t: 'TrexFull', coords: np.ndarray) -> 'TrexFull':
    """
    Attach reconstructed coordinates to a TrexFull's MapSites.
    
    Creates a copy with coord fields populated in canonical site order.
    
    Args:
        t: TrexFull object (will not be modified)
        coords: Site coordinates (n_sites × 3) in canonical order
        
    Returns:
        New TrexFull with coordinates attached to MapSites
    """
    from trex.model import TrexFull, MapSite, LigandPayload
    
    # Build new pairs with coords
    new_pairs = []
    coord_idx = 0
    for a, b in t.pairs:
        new_a = MapSite(lig=a.lig, atoms=list(a.atoms), coord=tuple(coords[coord_idx]))
        coord_idx += 1
        new_b = MapSite(lig=b.lig, atoms=list(b.atoms), coord=tuple(coords[coord_idx]))
        coord_idx += 1
        new_pairs.append((new_a, new_b))
    
    # Build new singles with coords
    new_singles = []
    for s in t.singles:
        new_s = MapSite(lig=s.lig, atoms=list(s.atoms), coord=tuple(coords[coord_idx]))
        coord_idx += 1
        new_singles.append(new_s)
    
    return TrexFull(
        metal=t.metal,
        ox=t.ox,
        spin=t.spin,
        ligands=list(t.ligands),
        pairs=new_pairs,
        singles=new_singles,
        G=t.G,
        X=t.X,
    )


def enforce_chirality(
    coords: np.ndarray,
    t: 'TrexFull',
    expected_chirality: str | None,
) -> np.ndarray:
    """
    Ensure reconstructed coordinates have correct chirality.
    
    If the computed chirality doesn't match expected, reflect the coordinates
    across a plane (flip one axis) to produce the correct enantiomer.
    
    Args:
        coords: Site coordinates (n_sites × 3)
        t: TrexFull object (used for structure, not its coords)
        expected_chirality: Expected chirality flag ("@", "@@", "Δ", "Λ", or None)
        
    Returns:
        Coordinates with correct chirality (possibly reflected)
    """
    from trex.chirality import compute_chirality
    
    if expected_chirality is None:
        return coords
    
    # For 2-coordinate systems, chirality doesn't apply
    if coords.shape[0] <= 2:
        return coords
    
    # Attach coords and compute actual chirality
    t_with_coords = attach_coords_to_trex(t, coords)
    actual_chirality = compute_chirality(t_with_coords)
    
    if actual_chirality == expected_chirality:
        return coords
    
    # Need to flip - reflect across xy plane (flip z)
    coords_flipped = coords.copy()
    coords_flipped[:, 2] *= -1
    
    # Verify the flip worked
    t_flipped = attach_coords_to_trex(t, coords_flipped)
    flipped_chirality = compute_chirality(t_flipped)
    
    if flipped_chirality == expected_chirality:
        return coords_flipped
    
    # If z-flip didn't work, try y-flip
    coords_flipped = coords.copy()
    coords_flipped[:, 1] *= -1
    
    t_flipped = attach_coords_to_trex(t, coords_flipped)
    flipped_chirality = compute_chirality(t_flipped)
    
    if flipped_chirality == expected_chirality:
        return coords_flipped
    
    # Last resort: x-flip
    coords_flipped = coords.copy()
    coords_flipped[:, 0] *= -1
    
    # Return even if it doesn't match - caller should check
    return coords_flipped


# =============================================================================
# Full Pipeline: Components 1-4 Combined
# =============================================================================

def reconstruct_coordination_sphere(
    t: 'TrexFull',
    site_r: list[float],
    trans_cos: list[float],
    cis_cos: list[float],
    cis_pair_indices: list[tuple[int, int]],
    expected_chirality: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full pipeline: internal coordinates → 3D site positions.
    
    This is the complete "Phase I" of T-REX-geo structure generation.
    
    FIXED: Now properly handles linear and bent geometries (n_sites <= 2)
    by using direct coordinate construction instead of Gram matrix decomposition.
    
    Args:
        t: TrexFull object (for structure and expected chirality)
        site_r: M-site distances in canonical order
        trans_cos: cos(θ) for trans pairs
        cis_cos: cos(θ) for cis pairs
        cis_pair_indices: Site index pairs for cis angles
        expected_chirality: Override chirality (uses t.X if None)
        
    Returns:
        coords: Site coordinates (n_sites × 3), metal at origin
        diagnostics: Dictionary with projection error metrics
    """
    n_sites = len(site_r)
    n_trans_pairs = len(trans_cos)
    
    # Determine chirality target
    if expected_chirality is None:
        expected_chirality = t.X
    
    r = np.array(site_r)
    
    # Special handling for 2-coordinate systems (linear/bent)
    if n_sites == 2:
        # Direct coordinate construction for linear/bent geometries
        cos_theta_value = trans_cos[0] if n_trans_pairs > 0 else 0.0
        coords = extract_coordinates_linear(r, cos_theta_value)
        
        diagnostics = {
            'frobenius_error': 0.0,
            'max_error': 0.0,
            'relative_error': 0.0,
            'n_negative_eigenvalues': 0,
            'rank_above_3': 0,
            'geometry_type': 'linear' if cos_theta_value < -0.5 else 'bent',
        }
        
        return coords, diagnostics
    
    # Special handling for 1-coordinate systems (rare, but handle gracefully)
    if n_sites == 1:
        coords = np.array([[0.0, 0.0, r[0]]])
        diagnostics = {
            'frobenius_error': 0.0,
            'max_error': 0.0,
            'relative_error': 0.0,
            'n_negative_eigenvalues': 0,
            'rank_above_3': 0,
            'geometry_type': 'monodentate',
        }
        return coords, diagnostics
    
    # Standard pipeline for n_sites >= 3
    # Component 1: Build matrices
    cos_theta = build_cos_theta_matrix(
        n_sites=n_sites,
        n_trans_pairs=n_trans_pairs,
        trans_cos=trans_cos,
        cis_cos=cis_cos,
        cis_pair_indices=cis_pair_indices,
    )
    
    G = build_gram_matrix(r, cos_theta)
    
    # Component 2: Project to rank 3
    G_proj = project_rank3(G)
    
    # Compute diagnostics
    diagnostics = compute_projection_error(G, G_proj)
    
    # Component 3: Extract coordinates
    coords = extract_coordinates(G_proj)
    
    # Component 4: Enforce chirality
    if expected_chirality:
        coords = enforce_chirality(coords, t, expected_chirality)
    
    return coords, diagnostics


def validate_reconstruction(
    original_r: list[float],
    original_trans_cos: list[float],
    original_cis_cos: list[float],
    cis_pair_indices: list[tuple[int, int]],
    reconstructed_coords: np.ndarray,
) -> dict:
    """
    Validate reconstruction by comparing round-trip internal coordinates.
    
    Args:
        original_*: Original internal coordinates
        reconstructed_coords: Coordinates from reconstruction pipeline
        
    Returns:
        Dictionary with validation metrics
    """
    n_trans = len(original_trans_cos)
    
    # Convert reconstructed coords back to internal
    r_recon, cos_theta_recon = coords_to_internal(reconstructed_coords)
    
    # Distance errors
    r_original = np.array(original_r)
    r_errors = np.abs(r_recon - r_original)
    
    # Trans angle errors
    trans_errors = []
    for pi in range(n_trans):
        i, j = 2 * pi, 2 * pi + 1
        trans_errors.append(abs(cos_theta_recon[i, j] - original_trans_cos[pi]))
    
    # Cis angle errors
    cis_errors = []
    for (i, j), original_cos in zip(cis_pair_indices, original_cis_cos):
        cis_errors.append(abs(cos_theta_recon[i, j] - original_cos))
    
    return {
        'r_mae': float(np.mean(r_errors)),
        'r_max_error': float(np.max(r_errors)),
        'trans_cos_mae': float(np.mean(trans_errors)) if trans_errors else 0.0,
        'trans_cos_max_error': float(np.max(trans_errors)) if trans_errors else 0.0,
        'cis_cos_mae': float(np.mean(cis_errors)) if cis_errors else 0.0,
        'cis_cos_max_error': float(np.max(cis_errors)) if cis_errors else 0.0,
    }


__all__ = [
    'build_cos_theta_matrix',
    'build_gram_matrix',
    'project_rank3',
    'compute_projection_error',
    'extract_coordinates',
    'extract_coordinates_linear',
    'coords_to_internal',
    'attach_coords_to_trex',
    'enforce_chirality',
    'reconstruct_coordination_sphere',
    'validate_reconstruction',
]
