#!/usr/bin/env python3
"""
LF-GNN ensemble prediction for the trex_geo pipeline.

Loads trained ensemble of LF-GNN models and predicts coordination geometry
(M-L bond lengths and L-M-L angles) for a given T-REX complex.

Models are cached on first use. Inference runs on CPU by default.

Usage in trex_geo pipeline:
    from hgnn_predict import EnsemblePredictor
    
    predictor = EnsemblePredictor("path/to/ensemble_dir")
    result = predictor.predict(trex_string)
    # result = {'site_r': [...], 'trans_cos': [...], 'cis_cos': [...],
    #           'cis_pair_indices': [...], 'proj_error': float}
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(_PACKAGE_DIR), "LF_GNN", "ensemble_gram")

# Lazy imports — only load heavy modules when first needed
_MODULES_LOADED = False
_trex_to_hyperdata = None
_AtomHyperNet = None
_BuildConfig = None
_parse_trex_full_string = None
_build_gram_matrix = None
_project_rank3 = None


def _ensure_imports():
    """Lazy-load GNN and trex modules on first use."""
    global _MODULES_LOADED, _trex_to_hyperdata, _AtomHyperNet, _BuildConfig
    global _parse_trex_full_string, _build_gram_matrix, _project_rank3
    
    if _MODULES_LOADED:
        return
    
    from .model import (
        trex_to_hyperdata, AtomHyperNet, BuildConfig,
    )
    from trex.parse_full import parse_trex_full_string
    from .coord_reconstruction import build_gram_matrix, project_rank3
    
    _trex_to_hyperdata = trex_to_hyperdata
    _AtomHyperNet = AtomHyperNet
    _BuildConfig = BuildConfig
    _parse_trex_full_string = parse_trex_full_string
    _build_gram_matrix = build_gram_matrix
    _project_rank3 = project_rank3
    _MODULES_LOADED = True


def _load_single_model(ckpt_path: str, d_in: int, device: torch.device) -> torch.nn.Module:
    """Load a single model from checkpoint."""
    _ensure_imports()
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    n_aux = {k: len(v) for k, v in ckpt['aux_maps'].items()}
    
    model = _AtomHyperNet(
        d_in,
        d=cfg['hidden'],
        mpnn_pre=cfg['mpnn_pre'],
        interleaved=cfg['interleaved'],
        hyper_only=cfg['hyper_only'],
        n_os=n_aux['os'],
        n_cn=n_aux['cn'],
        n_geom=n_aux['geom'],
        dropout=cfg.get('dropout', 0.1),
        edrop=cfg.get('edrop', 0.0),
        hedrop_p=cfg.get('hedrop', 0.0),
        prenorm=cfg.get('prenorm', False),
    )
    
    # Load EMA weights if available
    if ckpt.get('ema') is not None:
        model.load_state_dict(ckpt['ema'], strict=False)
    else:
        model.load_state_dict(ckpt['model'])
    
    model.to(device)
    model.eval()
    return model


class EnsemblePredictor:
    """Cached ensemble predictor for the trex_geo pipeline.
    
    Loads all model checkpoints once, then predicts on individual complexes.
    Thread-safe for read-only inference.
    
    Args:
        model_dir: Directory containing seed_*/best.pt checkpoints,
                   or a list of checkpoint paths.
        device: 'cpu' or 'cuda'. Default 'cpu'.
        project: If True, apply rank-3 Gram projection after ensembling.
                 Enforces geometric consistency at a small accuracy cost on r.
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = 'cpu',
        project: bool = True,
    ):
        _ensure_imports()
        
        self.device = torch.device(device)
        self.project = project
        self.cfg = _BuildConfig(
            include_metal_dative=False,
            include_site_hyperedges=True,
        )
        self._model_dir = model_dir
        
        # Find checkpoints
        if isinstance(model_dir, list):
            ckpt_paths = model_dir
        else:
            ckpt_paths = sorted(glob.glob(os.path.join(model_dir, 'seed_*', 'best.pt')))
            if not ckpt_paths:
                # Try flat directory
                ckpt_paths = sorted(glob.glob(os.path.join(model_dir, '*.pt')))
        
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        
        # Load aux_maps from first checkpoint (should be same across seeds)
        first_ckpt = torch.load(ckpt_paths[0], map_location=self.device, weights_only=False)
        self.aux_maps = first_ckpt['aux_maps']
        
        # Determine d_in from a dummy forward pass
        # Use a simple heuristic: load one checkpoint config
        self._d_in = None  # Will be set on first predict call
        
        # Load all models
        logger.info(f"Loading {len(ckpt_paths)} ensemble models on {device}...")
        self.models = []
        self._ckpt_paths = ckpt_paths
        self._loaded = False
    
    def _lazy_load_models(self, d_in: int):
        """Load models once we know d_in from the first data point."""
        if self._loaded:
            return
        
        for path in self._ckpt_paths:
            model = _load_single_model(path, d_in, self.device)
            self.models.append(model)
            logger.debug(f"  Loaded {path}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self.models)} models")
    
    @torch.no_grad()
    def predict(self, trex_str: str) -> Optional[Dict]:
        """Predict coordination geometry for a single T-REX complex.
        
        Args:
            trex_str: T-REX canonical string
            
        Returns:
            dict with:
                site_r: np.array of absolute M-L distances (Å)
                trans_cos: np.array of absolute cos(θ) for trans pairs
                cis_cos: np.array of absolute cos(θ) for cis pairs
                cis_pair_indices: list of (i, j) tuples for cis pairs
                proj_error: float, Gram projection error (confidence metric)
            or None if prediction fails
        """
        _ensure_imports()
        
        try:
            # Build hypergraph data
            data = _trex_to_hyperdata(
                trex_str, y=None, cfg=self.cfg, aux_maps=self.aux_maps,
            )
        except Exception as e:
            logger.warning(f"Failed to build hypergraph: {e}")
            return None
        
        # Lazy load models now that we know d_in
        d_in = data.x.size(1)
        self._lazy_load_models(d_in)
        
        # Add dummy fields expected by the model
        if not hasattr(data, 'he_target'):
            data.he_target = torch.zeros(data.he_type.size(0), dtype=torch.float32)
        if not hasattr(data, 'n_sites'):
            data.n_sites = torch.tensor([int((data.he_type == 2).sum())], dtype=torch.long)
        
        # Create a batch of 1
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(self.device)
        
        # Run inference on all models
        all_preds = []
        for model in self.models:
            he_pred, he_type, _ = model(data)
            all_preds.append(he_pred.float().cpu().numpy())
        
        he_type = he_type.cpu().numpy()
        he_ideal = data.he_ideal.cpu().numpy()
        
        # Average deltas across models
        mean_delta = np.mean(all_preds, axis=0)
        
        # Absolute values
        abs_vals = he_ideal + mean_delta
        
        # Separate by type
        site_mask = he_type == 2
        trans_mask = he_type == 1
        cis_mask = he_type == 0
        
        site_r = abs_vals[site_mask]
        trans_cos = abs_vals[trans_mask]
        cis_cos = abs_vals[cis_mask]
        
        ns = int(site_mask.sum())
        
        # Reconstruct cis_pair_indices from the canonical ordering
        # Pairs are enumerated as upper triangle: (0,1), (0,2), ..., (1,2), ...
        # with types assigned in trex_to_hyperdata
        pair_types_ordered = he_type[he_type <= 1]  # trans=1, cis=0 in pair order
        cis_pair_indices = []
        pair_idx = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                if pair_idx < len(pair_types_ordered) and pair_types_ordered[pair_idx] == 0:
                    cis_pair_indices.append((i, j))
                pair_idx += 1
        
        # Optional: Gram projection for geometric consistency
        proj_error = 0.0
        if self.project and ns >= 3:
            proj_error = self._project_and_update(
                site_r, trans_cos, cis_cos, cis_pair_indices,
                ns, he_type, abs_vals,
            )
        
        return {
            'site_r': site_r,
            'trans_cos': trans_cos,
            'cis_cos': cis_cos,
            'cis_pair_indices': cis_pair_indices,
            'proj_error': proj_error,
        }
    
    def _project_and_update(
        self, site_r, trans_cos, cis_cos, cis_pair_indices,
        ns, he_type, abs_vals,
    ) -> float:
        """Apply rank-3 Gram projection and update arrays in-place.
        
        Returns projection error (Frobenius norm).
        """
        # Build full cosine matrix
        cos_flat = abs_vals[he_type <= 1]  # all pair cosines in canonical order
        
        cos_mat = np.eye(ns)
        idx = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                cos_mat[i, j] = cos_mat[j, i] = cos_flat[idx]
                idx += 1
        
        # Build and project Gram matrix
        G = _build_gram_matrix(site_r, cos_mat)
        G_proj = _project_rank3(G)
        proj_error = float(np.linalg.norm(G - G_proj, 'fro'))
        
        # Extract projected internals
        r_proj = np.sqrt(np.maximum(np.diag(G_proj), 1e-12))
        cos_mat_proj = np.eye(ns)
        for i in range(ns):
            for j in range(i + 1, ns):
                if r_proj[i] > 1e-8 and r_proj[j] > 1e-8:
                    cos_mat_proj[i, j] = cos_mat_proj[j, i] = np.clip(
                        G_proj[i, j] / (r_proj[i] * r_proj[j]), -1.0, 1.0
                    )
        
        # Update site_r in-place
        site_r[:] = r_proj
        
        # Update trans_cos and cis_cos from projected cos_mat
        pair_idx = 0
        trans_idx = 0
        cis_idx = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                pair_type = he_type[he_type <= 1][pair_idx] if pair_idx < (he_type <= 1).sum() else -1
                if pair_type == 1:
                    trans_cos[trans_idx] = cos_mat_proj[i, j]
                    trans_idx += 1
                elif pair_type == 0:
                    cis_cos[cis_idx] = cos_mat_proj[i, j]
                    cis_idx += 1
                pair_idx += 1
        
        return proj_error
    
    def predict_batch(self, trex_strings: List[str]) -> List[Optional[Dict]]:
        """Predict for a batch of complexes.
        
        More efficient than calling predict() in a loop because it batches
        the forward pass through each model.
        """
        # For now, simple loop — could be optimized with DataLoader batching
        return [self.predict(s) for s in trex_strings]


# ============================================================
# prediction
# ============================================================

# Global singleton predictor (lazy initialized)
_predictor: Optional[EnsemblePredictor] = None


def get_predictor(model_dir: str = None, device: str = "cpu") -> EnsemblePredictor:
    global _predictor
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    if _predictor is None or _predictor._model_dir != model_dir:
        _predictor = EnsemblePredictor(model_dir, device=device)
    return _predictor


def predict_internals_with_model(
    trex_input,
    model_path: str = None,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    
    Args:
        trex_input: T-REX canonical string OR TrexFull object.
                    If TrexFull, the string is reconstructed via str(t).
        model_path: Path to ensemble directory (containing seed_*/best.pt)
        device: 'cpu' or 'cuda'
        
    Returns:
        dict with site_r, trans_cos, cis_cos, cis_pair_indices, proj_error
        (site_r, trans_cos, cis_cos are lists of floats for pipeline compat)
        or None if prediction fails
    """
    # Accept both TrexFull and string
    from trex.canon_full import pretty_trex_full
    if isinstance(trex_input, str):
        trex_str = trex_input
    else:
        # TrexFull object — reconstruct string
        # Try common trex serialization methods
        trex_str = pretty_trex_full(trex_input)
    
    try:
        predictor = get_predictor(model_path, device)
        result = predictor.predict(trex_str)
        
        if result is not None:
            # Convert numpy arrays to lists for pipeline compatibility
            result['site_r'] = result['site_r'].tolist()
            result['trans_cos'] = result['trans_cos'].tolist()
            result['cis_cos'] = result['cis_cos'].tolist()
            
            logger.info(
                f"LF-GNN prediction: {len(result['site_r'])} sites, "
                f"proj_err={result['proj_error']:.4f}"
            )
        
        return result
        
    except Exception as e:
        logger.warning(f"LF-GNN prediction failed: {e}")
        return None
