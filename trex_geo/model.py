#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-REX → Ligand-Field GNN: Per-Hyperedge Geometry Prediction

Predicts geometric deviations (Δr, Δcos θ) directly from per-hyperedge
embeddings H, with optional Gram matrix Frobenius coupling loss.

Architecture:
- Phase 1: MPNN bond-graph warm-up (ligand feature development)
- Phase 2: Interleaved MPNN + HyperCardinalDirectional (ligand-field reasoning)
- Phase 3: Hyper-only refinement
- Output: Per-hyperedge scalar heads (site Δr, trans Δcos, cis Δcos)

Loss:
- Primary: Weighted smooth L1 on per-hyperedge deltas
- Coupling (optional): Gram matrix Frobenius norm with warmup + linear ramp
- Auxiliary: OS/CN/geometry classification regularizers

Key features:
- Per-hyperedge prediction from H (no graph-level bottleneck)
- Ideal values computed from geometry.py (portable to new compounds)
- Mixed precision (AMP) with float32 safety for GRU/scatter/Gram ops
"""
from __future__ import annotations

import os, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, GraphNorm
#from torch_scatter import scatter_add

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType, HybridizationType as HT
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from trex.parse_full import parse_trex_full_string
from trex.model import TrexFull
from trex.trex2mol import trex_to_mol, SiteKey

from trex.geometry import (
    get_ideal_cis_cos,
    get_ideal_distance,
    IDEAL_TRANS_COS,
)

# ============================================================
# Performance Settings
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune for input sizes


# ============================================================
# Atom Features & Constants
# ============================================================

PAULING_EN = {
    "H": 2.20, "He": 4.16,
    "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81,
    "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
    "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 2.10, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "I": 2.66,
    "Cs": 0.79, "Ba": 0.89, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
    "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
    "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Lu": 1.27,
    "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28,
    "Am": 1.3, "Cm": 1.3, "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3,
    "Yb": 1.1, "Eu": 1.2, "Tb": 1.1, "Te": 2.10
}
ATOM_LIST = list(PAULING_EN.keys())
PT = Chem.GetPeriodicTable()

def _safe_pt(method: str, sym: str, default: float = 0.0) -> float:
    f = getattr(PT, method, None)
    if f is None: return default
    try: return float(f(sym))
    except Exception: return default

_METAL_SYMBOLS = {
    "Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Al","Ga","In","Sn","Tl","Pb","Bi","Po"
}

def is_metal_element(symbol: str) -> float:
    return 1.0 if symbol in _METAL_SYMBOLS else 0.0

_HYB_MAP = {HT.SP: 0, HT.SP2: 1, HT.SP3: 2, HT.SP3D: 3, HT.SP3D2: 4}
_HYB_DIM = 6

def hybrid_onehot_np(rd_atom) -> np.ndarray:
    idx = _HYB_MAP.get(rd_atom.GetHybridization(), None)
    one = np.zeros(_HYB_DIM, dtype=np.float32)
    one[idx if idx is not None else (_HYB_DIM - 1)] = 1.0
    return one

_DEG_DIM = 7

def degree_onehot_np(rd_atom) -> np.ndarray:
    deg = int(rd_atom.GetDegree())
    idx = deg if deg < (_DEG_DIM - 1) else (_DEG_DIM - 1)
    one = np.zeros(_DEG_DIM, dtype=np.float32)
    one[idx] = 1.0
    return one

_CHIRAL_MAP = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
_CHIRAL_DIM = 3

def chiral_onehot_np(rd_atom) -> np.ndarray:
    chiral_tag = rd_atom.GetChiralTag()
    idx = _CHIRAL_MAP.get(chiral_tag, 0)
    one = np.zeros(_CHIRAL_DIM, dtype=np.float32)
    one[idx] = 1.0
    return one

def element_numeric_props(symbol: str) -> dict:
    try: Z = float(PT.GetAtomicNumber(symbol))
    except Exception: Z = 0.0
    cov = _safe_pt('GetRcovalent', symbol, 0.0)
    en_val = PAULING_EN.get(symbol, None)
    en = float(en_val) if en_val is not None else 0.0
    ve = _safe_pt('GetNOuterElecs', symbol, 0.0)
    row = 0.0; col = 0.0
    for m in ('GetRow','GetPeriod'):
        if row == 0.0: row = _safe_pt(m, symbol, 0.0)
    for m in ('GetColumn','GetGroup'):
        if col == 0.0: col = _safe_pt(m, symbol, 0.0)
    return {
        'Z': Z,
        'cov': cov, 'has_cov': (1.0 if cov > 0 else 0.0),
        'en': en,   'has_en': (1.0 if en_val is not None else 0.0),
        've': ve,   'has_ve': (1.0 if ve > 0 else 0.0),
        'row': row, 'has_row': (1.0 if row > 0 else 0.0),
        'col': col, 'has_col': (1.0 if col > 0 else 0.0),
        'is_metal_element': is_metal_element(symbol),
    }

def _mu_sd(vals):
    arr = np.array(vals, dtype=np.float32)
    return float(arr.mean()), float(arr.std() + 1e-8)

def _compute_stats():
    Zs, covs, ens, ves, rows, cols = [], [], [], [], [], []
    for s in ATOM_LIST:
        p = element_numeric_props(s)
        Zs.append(p['Z'])
        if p['has_cov']: covs.append(p['cov'])
        if p['has_en']:  ens.append(p['en'])
        if p['has_ve']:  ves.append(p['ve'])
        if p['has_row']: rows.append(p['row'])
        if p['has_col']: cols.append(p['col'])
    if not covs: covs=[0.0]
    if not ens:  ens=[0.0]
    if not ves:  ves=[0.0]
    if not rows: rows=[0.0]
    if not cols: cols=[0.0]
    return {'Z': _mu_sd(Zs), 'cov': _mu_sd(covs), 'en': _mu_sd(ens),
            've': _mu_sd(ves), 'row': _mu_sd(rows), 'col': _mu_sd(cols)}

ATOM_SCALAR_STATS = _compute_stats()

def _z(x: float, mu_sd) -> float:
    return (float(x) - mu_sd[0]) / mu_sd[1]

SYMBOL_TO_IDX = {s:i for i,s in enumerate(ATOM_LIST)}

@dataclass
class BuildConfig:
    include_metal_dative: bool = False
    include_site_hyperedges: bool = True
    include_all_site_hyperedges: bool = True  # all M-L 1-body terms, not just haptic


def atom_feature(rd_atom: Chem.Atom, is_metal: bool = False, metal_chirality: float = 0.0) -> np.ndarray:
    symbol = rd_atom.GetSymbol()
    idx = SYMBOL_TO_IDX.get(symbol, len(ATOM_LIST))
    onehot = np.zeros(len(ATOM_LIST) + 1, dtype=np.float32)
    onehot[idx] = 1.0

    p = element_numeric_props(symbol)
    numeric = np.array([
        _z(p['Z'], ATOM_SCALAR_STATS['Z']),
        _z(p['cov'], ATOM_SCALAR_STATS['cov']) if p['has_cov'] else 0.0,
        _z(p['en'], ATOM_SCALAR_STATS['en']) if p['has_en'] else 0.0,
        _z(p['ve'], ATOM_SCALAR_STATS['ve']) if p['has_ve'] else 0.0,
        _z(p['row'], ATOM_SCALAR_STATS['row']) if p['has_row'] else 0.0,
        _z(p['col'], ATOM_SCALAR_STATS['col']) if p['has_col'] else 0.0,
        p['has_cov'], p['has_en'], p['has_ve'], p['has_row'], p['has_col'],
        float(rd_atom.GetFormalCharge()), 1.0 if is_metal else 0.0, p['is_metal_element'],
    ], dtype=np.float32)

    hyb = hybrid_onehot_np(rd_atom)
    arom = 1.0 if rd_atom.GetIsAromatic() else 0.0
    inrng = 1.0 if rd_atom.IsInRing() else 0.0
    deg1h = degree_onehot_np(rd_atom)
    
    chiral = chiral_onehot_np(rd_atom)
    metal_chiral = np.array([metal_chirality if is_metal else 0.0], dtype=np.float32)

    return np.concatenate([
        onehot, numeric, hyb, 
        np.array([arom, inrng], dtype=np.float32), 
        deg1h, chiral, metal_chiral
    ], axis=0)


# ============================================================
# Angle Class Mapping (Discrete, not continuous)
# ============================================================

ANGLE_COS_VALUES = [
    -1.000,  # 0: 180° (trans)
    -0.809,  # 1: 144° (pentagonal bipyramidal non-adjacent equatorial)
    -0.500,  # 2: 120° (trigonal)
    -0.333,  # 3: 109.5° (tetrahedral)
     0.000,  # 4: 90° (octahedral/square planar cis)
     0.309,  # 5: 72° (pentagonal bipyramidal adjacent equatorial)
]
ANGLE_CLASS_NONE = 6
NUM_ANGLE_CLASSES = 7


def cos_to_angle_class(cos_val: float) -> int:
    min_dist = float('inf')
    best_idx = 4
    for i, ref_cos in enumerate(ANGLE_COS_VALUES):
        dist = abs(cos_val - ref_cos)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx


def _compute_ideal_cos_for_pair(T: TrexFull, site_i, site_j, is_trans: bool) -> float:
    if is_trans:
        return IDEAL_TRANS_COS
    
    geometry = T.G or T.get_geo_type()
    n_singles = len(T.singles)
    
    singles_set = {(s.lig, tuple(sorted(s.atoms))): idx for idx, s in enumerate(T.singles)}
    
    site_i_key = (site_i.lig, tuple(sorted(site_i.atoms)))
    site_j_key = (site_j.lig, tuple(sorted(site_j.atoms)))
    
    site_i_in_trans = site_i_key not in singles_set
    site_j_in_trans = site_j_key not in singles_set
    
    site_i_singles_idx = singles_set.get(site_i_key)
    site_j_singles_idx = singles_set.get(site_j_key)
    
    return get_ideal_cis_cos(
        geometry=geometry,
        site_i_in_trans=site_i_in_trans,
        site_j_in_trans=site_j_in_trans,
        site_i_singles_idx=site_i_singles_idx,
        site_j_singles_idx=site_j_singles_idx,
        n_singles=n_singles,
    )


# ============================================================
# Data Classes
# ============================================================

class HyperData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index': return int(self.num_nodes)
        if key == 'he_index':
            n_nodes = int(getattr(self, 'num_nodes', 0))
            n_hes   = int(getattr(self, 'num_hyperedges', 0))
            if isinstance(value, torch.Tensor): return value.new_tensor([[n_nodes], [n_hes]])
            else: return torch.tensor([[n_nodes], [n_hes]], dtype=torch.long)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('edge_index', 'he_index'): return 1
        return 0


def bond_feature(is_dative: bool = False, bond: Optional['Chem.Bond'] = None) -> np.ndarray:
    vec = np.zeros(8, dtype=np.float32)
    if is_dative:
        vec[4] = 1.0
        return vec
    if bond is None: return vec
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE: vec[0] = 1.0
    elif btype == Chem.BondType.DOUBLE: vec[1] = 1.0
    elif btype == Chem.BondType.TRIPLE: vec[2] = 1.0
    if bond.GetIsAromatic(): vec[3] = 1.0
    vec[5] = 1.0 if bond.GetIsConjugated() else 0.0
    vec[6] = 1.0 if bond.IsInRing() else 0.0
    vec[7] = 1.0 if bond.GetIsAromatic() else 0.0
    return vec


def trex_to_hyperdata(trex_str: str, y: Optional[float] = None,
                      cfg: BuildConfig = BuildConfig(),
                      aux_maps: Optional[Dict[str, Dict]] = None) -> HyperData:
    T: TrexFull = parse_trex_full_string(trex_str)
    t2m = trex_to_mol(trex_str, add_haptic_centroids=False, set_metal_formal_charge_from_ox=True, annotate_props=False, sanitize=True)
    mol = t2m.mol
    metal_idx = int(t2m.metal_idx)
    
    metal_chirality = 0.0
    if T.X == '@':
        metal_chirality = 1.0
    elif T.X == '@@':
        metal_chirality = -1.0

    X = [atom_feature(a, is_metal=(a.GetIdx() == metal_idx), metal_chirality=metal_chirality) 
         for a in mol.GetAtoms()]
    x = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)

    edge_pairs, edge_feats = [], []
    dative_ids = set(t2m.dative_bond_ids)

    for b in mol.GetBonds():
        if b.GetBondType() == Chem.BondType.ZERO: continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        
        if i == metal_idx or j == metal_idx:
            if cfg.include_metal_dative:
                is_dat = (b.GetBondType() == Chem.BondType.DATIVE) or (b.GetIdx() in dative_ids)
                edge_pairs.append((i, j)); edge_feats.append(bond_feature(is_dat, b))
                edge_pairs.append((j, i)); edge_feats.append(bond_feature(is_dat, b))
            continue

        edge_pairs.append((i, j)); edge_feats.append(bond_feature(False, b))
        edge_pairs.append((j, i)); edge_feats.append(bond_feature(False, b))

    edge_index = (torch.tensor(np.array(edge_pairs, dtype=np.int64).T, dtype=torch.long)
                  if edge_pairs else torch.zeros((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(np.stack(edge_feats, axis=0), dtype=torch.float32)
                 if edge_feats else torch.empty((0, 8), dtype=torch.float32))

    he_node, he_id, he_role, he_type = [], [], [], []
    he_angle_class: List[int] = []
    
    def _atoms_for(site):
        key = SiteKey(lig=site.lig, atoms=tuple(sorted(site.atoms)))
        return t2m.site_to_atoms_global.get(key, [])

    trans_pairs = set()
    for a, b in T.pairs:
        ka = (a.lig, tuple(sorted(a.atoms)))
        kb = (b.lig, tuple(sorted(b.atoms)))
        trans_pairs.add(tuple(sorted([ka, kb])))

    collected = []
    for a, b in T.pairs:
        A = _atoms_for(a); B = _atoms_for(b)
        if A: collected.append((a, A))
        if B: collected.append((b, B))
    for s in T.singles:
        A = _atoms_for(s)
        if A: collected.append((s, A))

    sites = [s for (s, _) in collected]
    s_atoms = [atoms for (_, atoms) in collected]
    nS = len(sites)

    for i in range(nS):
        for j in range(i + 1, nS):
            Si, Sj = sites[i], sites[j]
            key_pair = tuple(sorted([(Si.lig, tuple(sorted(Si.atoms))), (Sj.lig, tuple(sorted(Sj.atoms)))]))
            is_trans = (key_pair in trans_pairs)
            hid = len(he_type)
            he_type.append(1 if is_trans else 0)
            
            ideal_cos = _compute_ideal_cos_for_pair(T, Si, Sj, is_trans)
            he_angle_class.append(cos_to_angle_class(ideal_cos))
            
            for gid in s_atoms[i]: he_node.append(gid); he_id.append(hid); he_role.append(0)
            he_node.append(metal_idx); he_id.append(hid); he_role.append(1)
            for gid in s_atoms[j]: he_node.append(gid); he_id.append(hid); he_role.append(2)

    if cfg.include_site_hyperedges:
        for site, atoms in collected:
            if cfg.include_all_site_hyperedges or len(atoms) > 1:
                hid = len(he_type)
                he_type.append(2)
                he_angle_class.append(ANGLE_CLASS_NONE)
                he_node.append(metal_idx); he_id.append(hid); he_role.append(1)
                for gid in atoms: he_node.append(gid); he_id.append(hid); he_role.append(0)

    he_ideal_vals = []
    for i in range(nS):
        for j in range(i + 1, nS):
            Si, Sj = sites[i], sites[j]
            key_pair = tuple(sorted([(Si.lig, tuple(sorted(Si.atoms))),
                                      (Sj.lig, tuple(sorted(Sj.atoms)))]))
            is_trans = (key_pair in trans_pairs)
            ideal_cos = _compute_ideal_cos_for_pair(T, Si, Sj, is_trans)
            he_ideal_vals.append(ideal_cos)
    if cfg.include_site_hyperedges:
        for site, atoms in collected:
            if cfg.include_all_site_hyperedges or len(atoms) > 1:
                lig_payload = T.ligands[site.lig - 1]
                lig_smi = lig_payload.text if lig_payload.kind == "SMILES" else ""
                r_ideal = get_ideal_distance(T.metal, site, lig_smi)
                he_ideal_vals.append(r_ideal)
    
    he_ideal_t = (torch.tensor(he_ideal_vals, dtype=torch.float32)
                  if he_ideal_vals else torch.zeros((0,), dtype=torch.float32))


    he_index = (torch.tensor([he_node, he_id], dtype=torch.long) if he_id else torch.zeros((2,0), dtype=torch.long))
    he_role_t = (torch.tensor(he_role, dtype=torch.long) if he_role else torch.zeros((0,), dtype=torch.long))
    he_type_t = (torch.tensor(he_type, dtype=torch.long) if he_type else torch.zeros((0,), dtype=torch.long))
    he_angle_class_t = (torch.tensor(he_angle_class, dtype=torch.long) 
                        if he_angle_class else torch.zeros((0,), dtype=torch.long))
    
    data = HyperData(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        he_index=he_index, 
        he_type=he_type_t, 
        he_angle_class=he_angle_class_t,
        num_hyperedges=int(he_type_t.numel()), 
        y=(torch.tensor([float(y)], dtype=torch.float32) if y is not None else None)
    )
    data.he_role = he_role_t
    data.he_ideal = he_ideal_t


    os_val = int(getattr(T, 'ox', 0))
    cn_val = 2 * len(T.pairs) + len(T.singles)
    try: geom_val = str(T.get_geo_type())
    except: geom_val = str(getattr(T, 'G', 'UNK'))

    if aux_maps:
        data.os = torch.tensor([aux_maps['os'].get(os_val, -1)], dtype=torch.long)
        data.cn = torch.tensor([aux_maps['cn'].get(cn_val, -1)], dtype=torch.long)
        data.geom = torch.tensor([aux_maps['geom'].get(geom_val, -1)], dtype=torch.long)
    data.num_nodes = int(x.size(0))
    return data


# ============================================================
# Model Components
# ============================================================

class HyperCardinalDirectional(nn.Module):
    def __init__(self, d: int, num_types: int = 3, num_angle_classes: int = NUM_ANGLE_CLASSES):
        super().__init__()
        self.d = d
        self.type_emb = nn.Embedding(num_types, d)
        self.angle_emb = nn.Embedding(num_angle_classes, d)
        
        self.enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.scM = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.scSite = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))  # shared A/B scorer
        self.h_site = nn.GRUCell(4 * d, d)  # single shared GRU
        self.to_site = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # shared A/B message
        self.gate_site = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        
        self.to_M = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.gate_m = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.n_upd = nn.GRUCell(d, d)
        self.ln_h = nn.LayerNorm(d)  # Stabilize H accumulation across layers

    def _attn_pool_role(self, x, node_ids, he_ids, mask, H, d, dev, scorer):
        if not mask.any(): return torch.zeros(H, d, device=dev)
        mem_idx = node_ids[mask]; hed_idx = he_ids[mask]
        feats = self.enc(x[mem_idx])
        scores = scorer(feats).squeeze(-1)
        
        # Cast to float32 for scatter operations (AMP compatibility)
        scores_f32 = scores.float()
        feats_f32 = feats.float()
        
        max_per_he = torch.full((H,), -1e4, device=dev, dtype=torch.float32)  # float16-safe sentinel
        max_per_he.index_reduce_(0, hed_idx, scores_f32, reduce='amax')
        scores_stab = scores_f32 - max_per_he[hed_idx]
        weights = torch.exp(scores_stab)
        sum_w = torch.zeros(H, device=dev, dtype=torch.float32)
        sum_w.index_add_(0, hed_idx, weights)
        sum_w = sum_w.clamp_min(1e-12)
        num = torch.zeros(H, d, device=dev, dtype=torch.float32)
        num.index_add_(0, hed_idx, feats_f32 * weights.unsqueeze(-1))
        return (num / sum_w.unsqueeze(-1)).to(x.dtype)  # Cast back to original dtype

    def forward(self, x, he_index, he_type, he_role, he_angle_class=None, H_prev=None):
        if he_index.numel() == 0 or he_type.numel() == 0:
            return (x, torch.zeros((0, x.size(1)), device=x.device)) if H_prev is None else (x, H_prev)
        
        node_ids, he_ids = he_index[0], he_index[1]
        H, d = he_type.size(0), x.size(1); dev = x.device
        rA, rM, rB = (he_role==0), (he_role==1), (he_role==2)

        a_bar = self._attn_pool_role(x, node_ids, he_ids, rA, H, d, dev, self.scSite)
        m_bar = self._attn_pool_role(x, node_ids, he_ids, rM, H, d, dev, self.scM)
        b_bar = self._attn_pool_role(x, node_ids, he_ids, rB, H, d, dev, self.scSite) 

        t = self.type_emb(torch.clamp(he_type, max=2))
        
        if he_angle_class is not None:
            angle_enc = self.angle_emb(torch.clamp(he_angle_class, max=NUM_ANGLE_CLASSES-1))
        else:
            angle_enc = torch.zeros(H, d, device=dev)
        
        if H_prev is None or H_prev.numel() == 0: 
            H_prev = t + angle_enc
        
        # GRUs in float32 always — sigmoid/tanh overflow in float16 is the #1 NaN source
        with torch.cuda.amp.autocast(enabled=False):
            h_a = self.h_site(torch.cat([a_bar, m_bar, t, angle_enc], dim=1).float(), H_prev.float())
            h_b = self.h_site(torch.cat([b_bar, m_bar, t, angle_enc], dim=1).float(), H_prev.float())
        
        H_new = H_prev + 0.5 * (h_a + h_b)
        H_new = self.ln_h(H_new)  # Prevent unbounded H growth across layers

        msg_site_a = self.to_site(h_a) * self.gate_site(h_a)
        msg_site_b = self.to_site(h_b) * self.gate_site(h_b)
        msg_M = self.to_M(H_new) * self.gate_m(H_new)

        # Use float32 for scatter operations (AMP compatibility)
        orig_dtype = x.dtype
        n_msg = torch.zeros(x.size(0), x.size(1), device=dev, dtype=torch.float32)
        n_cnt = torch.zeros(x.size(0), 1, device=dev, dtype=torch.float32)
        
        if rB.any(): 
            n_msg.index_add_(0, node_ids[rB], msg_site_b[he_ids[rB]].float())
            n_cnt.index_add_(0, node_ids[rB], torch.ones(node_ids[rB].size(0), 1, device=dev, dtype=torch.float32))
        if rA.any(): 
            n_msg.index_add_(0, node_ids[rA], msg_site_a[he_ids[rA]].float())
            n_cnt.index_add_(0, node_ids[rA], torch.ones(node_ids[rA].size(0), 1, device=dev, dtype=torch.float32))
        if rM.any(): 
            n_msg.index_add_(0, node_ids[rM], msg_M[he_ids[rM]].float())
            n_cnt.index_add_(0, node_ids[rM], torch.ones(node_ids[rM].size(0), 1, device=dev, dtype=torch.float32))
        n_cnt.clamp_min_(1.0)
        
        # Cast back to float32 for GRU — never run GRU in float16
        with torch.cuda.amp.autocast(enabled=False):
            n_msg_normalized = n_msg / n_cnt
            x_new = self.n_upd(n_msg_normalized, x.float())
        return x_new.to(orig_dtype), H_new


def edrop(edge_index, edge_attr, p: float, training: bool):
    if (not training) or p <= 0.0 or edge_index.size(1) == 0: return edge_index, edge_attr
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > p
    return edge_index[:, keep], (edge_attr[keep] if edge_attr is not None else None)


def hedrop(he_index, he_type, he_role, he_angle_class=None, p: float = 0.0, training: bool = False):
    if (not training) or p <= 0.0 or he_index.size(1) == 0: 
        return he_index, he_type, he_role, he_angle_class
    
    H = he_type.size(0)
    keep = torch.rand(H, device=he_type.device) > p
    if keep.all(): 
        return he_index, he_type, he_role, he_angle_class
    
    remap = -torch.ones(H, dtype=torch.long, device=he_type.device)
    remap[keep] = torch.arange(int(keep.sum()), device=he_type.device)
    keep_mem = keep[he_index[1]]
    
    he_index_new = torch.stack([he_index[0, keep_mem], remap[he_index[1, keep_mem]]], 0)
    he_type_new = he_type[keep]
    he_role_new = he_role[keep_mem]
    he_angle_class_new = he_angle_class[keep] if he_angle_class is not None else None
    
    return he_index_new, he_type_new, he_role_new, he_angle_class_new


class EdgeEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, d), nn.ReLU(), nn.Linear(d, d))
    def forward(self, e): return self.net(e)


class LigandFieldBlock(nn.Module):
    def __init__(self, d, bond_layer, dropout=0.1, prenorm=False):
        super().__init__()
        self.bond = bond_layer
        self.hyper = HyperCardinalDirectional(d, num_types=3)
        self.norm_in = GraphNorm(d) if prenorm else None
        self.norm_mid = GraphNorm(d) if prenorm else None
        self.norm_out = GraphNorm(d)
        self.drop = nn.Dropout(p=dropout)
        self.prenorm = prenorm

    def forward(self, x, edge_index, edge_attr, he_index, he_type, he_role, he_angle_class, batch, H_prev=None):
        h_lig = self.norm_in(x, batch) if self.prenorm else x
        h_lig = self.drop(self.bond(h_lig, edge_index, edge_attr))
        x_lig = x + h_lig
        
        h_hyp = self.norm_mid(x_lig, batch) if self.prenorm else x_lig
        h_hyp, H_new = self.hyper(h_hyp, he_index, he_type, he_role, he_angle_class, H_prev)
        h_hyp = self.drop(h_hyp)
        
        return self.norm_out(x_lig + h_hyp, batch), H_new


class MLP(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
    def forward(self, x): return self.net(x)


class AtomHyperNet(nn.Module):
    def __init__(self, d_in: int, d: int = 128, mpnn_pre=3, interleaved=3, hyper_only=0,
                 n_os: int = 0, n_cn: int = 0, n_geom: int = 0,
                 dropout: float = 0.1, edrop: float = 0.0, hedrop_p: float = 0.0,
                 prenorm: bool = False):
        super().__init__()
        self.embed = nn.Linear(d_in, d)
        self.edge_encoder = EdgeEncoder(d)
        bond = lambda: GINEConv(MLP(d, p=dropout))
        # Phase 1: MPNN-only warm-up (ligand feature development)
        self.mpnn_pre = nn.ModuleList([
            nn.ModuleDict({
                'conv': bond(),
                'norm': GraphNorm(d),
                'drop': nn.Dropout(dropout),
            }) for _ in range(mpnn_pre)
        ])

        # Phase 2: Interleaved (optional, for feedback)
        self.blocks = nn.ModuleList([
            LigandFieldBlock(d, bond(), dropout=dropout, prenorm=prenorm)
            for _ in range(interleaved)
        ])
        
        # Phase 3: Hyper-only refinement (optional)
        self.hyper_post = nn.ModuleList([
            HyperCardinalDirectional(d, num_types=3)
            for _ in range(hyper_only)
        ])

        
        self.head_site_dr  = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.head_trans_dc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.head_cis_dc   = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        
        self.aux = nn.ModuleDict({
            'os': nn.Linear(d, n_os) if n_os > 0 else None,
            'cn': nn.Linear(d, n_cn) if n_cn > 0 else None,
            'geom': nn.Linear(d, n_geom) if n_geom > 0 else None
        })
        self.edrop_p, self.hedrop_p = edrop, hedrop_p

    def forward(self, data: HyperData):
        x = self.embed(data.x)
        ea = self.edge_encoder(data.edge_attr) if getattr(data, 'edge_attr', None) is not None else None
        ei, he_i, he_t, he_r = data.edge_index, data.he_index, data.he_type, data.he_role
        he_angle = getattr(data, 'he_angle_class', None)

        if self.training:
            ei, ea = edrop(ei, ea, self.edrop_p, True)
            he_i, he_t, he_r, he_angle = hedrop(he_i, he_t, he_r, he_angle, p=self.hedrop_p, training=True)

        # Phase 1: MPNN-only (ligand features mature before any hyperedge reasoning)
        for layer in self.mpnn_pre:
            h = layer['drop'](layer['conv'](x, ei, ea))
            x = layer['norm'](x + h, data.batch)
        #Phase 2: ligand field loop
        H = None
        for blk in self.blocks:
            x, H = blk(x, ei, ea, he_i, he_t, he_r, he_angle, data.batch, H_prev=H)

        # Phase 3: Hyperedge-only refinement (no more bond updates)
        for hyper in self.hyper_post:
            x, H = hyper(x, he_i, he_t, he_r, he_angle, H)

        g_atoms = global_mean_pool(x, data.batch)
#        B = g_atoms.size(0)  # Actual batch size
#        d = g_atoms.size(1)
        
        he_pred = torch.zeros(he_t.size(0), device=x.device, dtype=x.dtype)
        if H is not None and H.numel() > 0:
            m_site  = (he_t == 2)
            m_trans = (he_t == 1)
            m_cis   = (he_t == 0)
            if m_site.any():  he_pred[m_site]  = self.head_site_dr(H[m_site]).squeeze(-1).float()
            if m_trans.any(): he_pred[m_trans] = self.head_trans_dc(H[m_trans]).squeeze(-1).float()
            if m_cis.any():   he_pred[m_cis]   = self.head_cis_dc(H[m_cis]).squeeze(-1).float()

        # Aux heads still use graph-level
        aux_out = {}
        for k, layer in self.aux.items():
            if layer is not None: aux_out[k] = layer(g_atoms)
        
        return he_pred, he_t, aux_out


# ============================================================
# Dataset & Evaluation
# ============================================================

class TrexJSONLDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, cfg: BuildConfig = BuildConfig()):
        with open(jsonl_path, 'r') as f: self.rows = [json.loads(l) for l in f]
        self.cfg = cfg
        os_vals, cn_vals, geom_vals = set(), set(), set()
        for r in self.rows:
            T = parse_trex_full_string(r['text'])
            os_vals.add(int(getattr(T, 'ox', 0)))
            cn_vals.add(2*len(T.pairs) + len(T.singles))
            try: geom_vals.add(str(T.get_geo_type()))
            except: geom_vals.add(str(getattr(T, 'G', 'UNK')))
        self.aux_maps = {
            'os': {v:i for i,v in enumerate(sorted(os_vals))},
            'cn': {v:i for i,v in enumerate(sorted(cn_vals))},
            'geom': {v:i for i,v in enumerate(sorted(geom_vals))},
        }
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        data = trex_to_hyperdata(r['text'], y=r.get('y', None),
                                 cfg=self.cfg, aux_maps=self.aux_maps)
        
        geo = r.get('geo', None)
        if geo is not None:
            he_target = torch.zeros(data.he_type.size(0), dtype=torch.float32)
            
            site_idx  = (data.he_type == 2).nonzero(as_tuple=True)[0]
            trans_idx = (data.he_type == 1).nonzero(as_tuple=True)[0]
            cis_idx   = (data.he_type == 0).nonzero(as_tuple=True)[0]
            
            dr = torch.tensor(geo['site_delta_r'], dtype=torch.float32)
            dt = torch.tensor(geo['trans_delta_cos'], dtype=torch.float32)
            dc = torch.tensor(geo['cis_delta_cos'], dtype=torch.float32)
            assert len(site_idx) == len(dr), \
                f"Site alignment mismatch: {len(site_idx)} hyperedges vs {len(dr)} targets"
            assert len(trans_idx) == len(dt), \
                f"Trans alignment mismatch: {len(trans_idx)} hyperedges vs {len(dt)} targets"
            assert len(cis_idx) == len(dc), \
                f"Cis alignment mismatch: {len(cis_idx)} hyperedges vs {len(dc)} targets"
            
            he_target[site_idx[:len(dr)]]  = dr
            he_target[trans_idx[:len(dt)]]  = dt
            he_target[cis_idx[:len(dc)]]    = dc
            
            data.he_target = he_target

        else:
            data.he_target = torch.zeros(data.he_type.size(0), dtype=torch.float32)
        
        data.n_sites = torch.tensor([int((data.he_type == 2).sum())], dtype=torch.long)
        return data


_PAIR_IDX_CACHE = {}

def _get_pair_indices(ns: int, device):
    """Return (row_idx, col_idx) tensors for upper triangle of ns×ns matrix."""
    if ns not in _PAIR_IDX_CACHE:
        rows, cols = [], []
        for i in range(ns):
            for j in range(i + 1, ns):
                rows.append(i)
                cols.append(j)
        _PAIR_IDX_CACHE[ns] = (torch.tensor(rows, dtype=torch.long),
                                torch.tensor(cols, dtype=torch.long))
    r, c = _PAIR_IDX_CACHE[ns]
    return r.to(device), c.to(device)


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def gram_frobenius_loss(he_pred, he_target, he_ideal, he_type, he_index, batch, n_sites,
                        max_samples: int = 0):
    """
    Vectorized Gram matrix Frobenius loss, grouped by n_sites.
    
    Complexes with the same coordination number are batched together
    for fully vectorized Gram construction — no Python loops over pairs.
    
    Args:
        max_samples: If > 0, randomly subsample this many complexes per group.
                     0 means use all.
    """
    dev = he_pred.device
    B = n_sites.size(0)
    
    # Build hyperedge-to-batch mapping
    he_batch = torch.zeros(he_type.size(0), dtype=torch.long, device=dev)
    if he_index.numel() > 0:
        he_batch.scatter_(0, he_index[1], batch[he_index[0]])
    
    # Absolute values
    abs_pred = he_ideal + he_pred
    abs_true = he_ideal + he_target
    
    # Group batch indices by n_sites
    unique_ns = n_sites.unique()
    
    total_loss = torch.tensor(0.0, device=dev)
    n_valid = 0
    
    for ns_tensor in unique_ns:
        ns = int(ns_tensor)
        if ns < 3:
            continue
        
        n_pairs = ns * (ns - 1) // 2
        
        # Find all batch elements with this n_sites
        group_mask = n_sites == ns
        group_idx = group_mask.nonzero(as_tuple=True)[0]
        
        if group_idx.size(0) == 0:
            continue
        
        # Optional subsampling
        if max_samples > 0 and group_idx.size(0) > max_samples:
            perm = torch.randperm(group_idx.size(0), device=dev)[:max_samples]
            group_idx = group_idx[perm]
        
        K = group_idx.size(0)  # number of complexes in this group
        
        # Collect r and cos for each complex in the group
        r_pred_list = []
        r_true_list = []
        cos_pred_list = []
        cos_true_list = []
        valid_mask = []
        
        for b in group_idx.tolist():
            he_mask = he_batch == b
            he_types_b = he_type[he_mask]
            pred_b = abs_pred[he_mask]
            true_b = abs_true[he_mask]
            
            site_mask = he_types_b == 2
            pair_mask = he_types_b <= 1
            
            r_p = pred_b[site_mask]
            r_t = true_b[site_mask]
            c_p = pred_b[pair_mask]
            c_t = true_b[pair_mask]
            
            if r_p.size(0) == ns and c_p.size(0) == n_pairs:
                r_pred_list.append(r_p)
                r_true_list.append(r_t)
                cos_pred_list.append(c_p)
                cos_true_list.append(c_t)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        if not r_pred_list:
            continue
        
        K_valid = len(r_pred_list)
        
        # Stack into batched tensors: [K_valid, ns] and [K_valid, n_pairs]
        R_pred = torch.stack(r_pred_list)   # [K, ns]
        R_true = torch.stack(r_true_list)
        C_pred = torch.stack(cos_pred_list) # [K, n_pairs]
        C_true = torch.stack(cos_true_list)
        
        # Build Gram matrices vectorized
        # Diagonal: G_ii = r_i^2
        G_pred = torch.zeros(K_valid, ns, ns, device=dev)
        G_true = torch.zeros(K_valid, ns, ns, device=dev)
        
        # Diagonal
        diag_idx = torch.arange(ns, device=dev)
        G_pred[:, diag_idx, diag_idx] = R_pred ** 2
        G_true[:, diag_idx, diag_idx] = R_true ** 2
        
        # Off-diagonal: G_ij = r_i * r_j * cos_ij
        row_idx, col_idx = _get_pair_indices(ns, dev)  # [n_pairs] each
        
        # r_i * r_j for each pair: [K, n_pairs]
        rr_pred = R_pred[:, row_idx] * R_pred[:, col_idx]
        rr_true = R_true[:, row_idx] * R_true[:, col_idx]
        
        # G_ij = r_i * r_j * cos_ij
        off_diag_pred = rr_pred * C_pred  # [K, n_pairs]
        off_diag_true = rr_true * C_true
        
        # Fill upper and lower triangle
        G_pred[:, row_idx, col_idx] = off_diag_pred
        G_pred[:, col_idx, row_idx] = off_diag_pred
        G_true[:, row_idx, col_idx] = off_diag_true
        G_true[:, col_idx, row_idx] = off_diag_true
        
        # Frobenius norm per complex: [K]
        diff = G_pred - G_true
        frob = torch.sqrt((diff ** 2).sum(dim=(1, 2)).clamp_min(1e-12))
        
        total_loss = total_loss + frob.sum()
        n_valid += K_valid
    
    return total_loss / max(n_valid, 1)

def compute_projection_error(model, loader, device, use_amp=False, max_samples=500):
    """
    Compute mean Gram projection error (unsupervised consistency metric).
    
    Returns:
        float: mean ||G_pred - project_rank3(G_pred)||_F across complexes
    """
    from .coord_reconstruction import build_gram_matrix, project_rank3
    
    model.eval()
    proj_errors = []
    sample_count = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            with autocast('cuda', enabled=use_amp):
                he_pred, he_type_out, _ = model(data)
            
            he_pred = he_pred.float().cpu()
            he_type_out = he_type_out.cpu()
            he_ideal = data.he_ideal.cpu()
            n_sites_batch = data.n_sites.cpu()
            
            he_batch = torch.zeros(he_type_out.size(0), dtype=torch.long)
            he_index_cpu = data.he_index.cpu()
            batch_cpu = data.batch.cpu()
            if he_index_cpu.numel() > 0:
                he_batch.scatter_(0, he_index_cpu[1], batch_cpu[he_index_cpu[0]])
            
            abs_pred = (he_ideal + he_pred).numpy()
            
            B = n_sites_batch.size(0)
            for b in range(B):
                ns = int(n_sites_batch[b])
                if ns < 3:
                    continue
                
                he_mask = (he_batch == b).numpy()
                types_b = he_type_out[he_mask].numpy()
                pred_b = abs_pred[he_mask]
                
                r_pred = pred_b[types_b == 2]
                cos_pred = pred_b[types_b <= 1]
                
                n_pairs = ns * (ns - 1) // 2
                if len(r_pred) != ns or len(cos_pred) != n_pairs:
                    continue
                
                cos_mat = np.eye(ns)
                idx = 0
                for i in range(ns):
                    for j in range(i + 1, ns):
                        cos_mat[i, j] = cos_mat[j, i] = cos_pred[idx]
                        idx += 1
                
                G = build_gram_matrix(r_pred, cos_mat)
                G_proj = project_rank3(G)
                proj_errors.append(np.linalg.norm(G - G_proj, 'fro'))
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
            
            if sample_count >= max_samples:
                break
    
    return float(np.mean(proj_errors)) if proj_errors else float('nan')



