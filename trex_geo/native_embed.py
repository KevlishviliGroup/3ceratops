# trex_geo/native_embed.py
"""
Embedding without bond-order reassignment.

The MetalloGen route (trex_embed._embed_trex_metallogen_route) bonds every donor
to the metal, which breaks the donors' valences, so it replaces the metal with a
main-group dummy, re-derives every bond order and charge with an LP, encodes the
charges as element swaps, embeds, and swaps the elements back.

Here the metal is never bonded during embedding, so the ligands keep exactly the
Lewis structures written in their SMILES and RDKit sanitizes them as ordinary
molecules. The metal is an isolated atom pinned at the origin, donors are pinned
at their site positions, and the metal's distance bounds to the rest of the
complex are written directly:

    metal - donor             fixed (coordMap)
    metal - donor neighbour   from an M-D-X angle window chosen by donor type
                              (sp ~180, amine ~109, sp2 ~120, chelate rings by
                              size and donor element, from tmQM)
    metal - haptic atom       sqrt(d_centroid^2 + rho^2), with a pinned centroid ghost
    metal - anything else     outside the bonding cutoff: covalent radii + 0.45 A,
                              the rule trex.xyz_to_trex reads bonds with

Bonds that run through the metal never exist, so RDKit never assigns them
generic 1-4 or 1-5 bounds; ligands on different sites only see a (scaled)
van der Waals lower bound. ETKDG's last force-field stage only knows each
ligand's own geometry and can fold a chelate ring onto the metal, so each
attempt gets a short UFF clean-up with the metal, donors and centroid ghosts
fixed and every metal distance held in its bounds. An attempt is accepted only
if nothing that should not bind sits inside bonding distance of the metal. The
M-D bonds are added to the output molecule after embedding, in the same atom
order as the MetalloGen route.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom
from rdkit.DistanceGeometry import DoTriangleSmoothing
from rdkit.Geometry import Point3D

from .trex_embed import EmbedResult, _straighten_linear_ligands, check_atom_clashes

logger = logging.getLogger(__name__)

_SANITIZE_LIGAND = (Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

# Metal-centred lower bounds for atoms that are neither donors, donor neighbours
# nor haptic atoms (A).
_MIN_M_HEAVY = 2.5
_MIN_M_H = 2.0

# An atom counts as bonded to the metal when it is closer than the sum of covalent
# radii plus 0.45 A (the Open Babel rule that trex.xyz_to_trex uses to read a
# structure back). Atoms that should not bind are kept this far plus a margin;
# donors and haptic atoms must stay inside it.
_BOND_TOL = 0.45
_CONTACT_MARGIN = 0.10

# M-D-X windows (degrees) for a neighbour X of donor D inside a chelate ring, by
# ring size and donor element: 2nd-98th percentiles of the tmQM complexes in
# regression/, widened by a few degrees.
_CHELATE_WINDOWS = {
    (4, "S"): (68.0, 94.0), (4, None): (78.0, 108.0),
    (5, "N"): (98.0, 124.0), (5, "O"): (102.0, 132.0), (5, "S"): (86.0, 112.0),
    (5, "P"): (95.0, 116.0), (5, "C"): (104.0, 126.0), (5, None): (95.0, 130.0),
    (6, "N"): (105.0, 135.0), (6, "O"): (115.0, 146.0), (6, "C"): (113.0, 131.0), (6, None): (105.0, 146.0),
    (7, None): (105.0, 150.0),
}
_ELEMENT_CLASS = {"Se": "S", "Te": "S", "As": "P", "Sb": "P"}

# Haptic ring atom a, neighbour X outside the ring: smallest M-a-X angle (tmQM:
# heavy substituents median 126 deg, H 106-134 deg).
_HAPTIC_SUBSTITUENT_MIN = {"heavy": 105.0, "H": 105.0}


def _contact_cutoff(z_metal: int, z: int) -> float:
    pt = Chem.GetPeriodicTable()
    return pt.GetRcovalent(z_metal) + pt.GetRcovalent(z) + _BOND_TOL


def _chelate_window(ring: int, element: str) -> Tuple[float, float]:
    el = _ELEMENT_CLASS.get(element, element)
    return _CHELATE_WINDOWS.get((ring, el)) or _CHELATE_WINDOWS.get((ring, None)) or (100.0, 150.0)


# =============================================================================
# Ligands and sites
# =============================================================================

def _ligand_mol(smiles: str) -> Chem.Mol:
    """Ligand with explicit H, atom order identical to trex_embed.parse_ligand_smiles."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot parse ligand SMILES {smiles!r}")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=_SANITIZE_LIGAND)
    except Exception:
        # Unusual valences: keep what can be perceived and carry on.
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                                           | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                                           | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                                           | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                                           | Chem.SanitizeFlags.SANITIZE_SYMMRINGS),
                         catchErrors=True)
    return Chem.AddHs(mol)


def _site_atoms(mol: Chem.Mol, positions: List[int]) -> List[int]:
    """T-REX 1-based heavy-atom positions -> atom indices (same rule as find_coordinating_atoms)."""
    if mol.GetNumAtoms() == 1:
        return [0]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy:
        return [0]
    out = []
    for p in positions:
        if not 1 <= p <= len(heavy):
            raise ValueError(f"Donor position {p} out of range for a ligand with {len(heavy)} heavy atoms")
        out.append(heavy[p - 1])
    return out


def _is_linear_donor(mol: Chem.Mol, d: int) -> bool:
    """sp donors (CO, CN-, nitriles, isocyanides, acetylides) and N-bound NCS-/NCO-."""
    atom = mol.GetAtomWithIdx(d)
    if atom.GetHybridization() == Chem.HybridizationType.SP:
        return True
    heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if atom.GetSymbol() == "N" and len(heavy) == 1 and atom.GetDegree() == 1:
        x = heavy[0]
        bond = mol.GetBondBetweenAtoms(d, x.GetIdx())
        return (x.GetSymbol() == "C" and x.GetHybridization() == Chem.HybridizationType.SP
                and bond.GetBondTypeAsDouble() >= 2)
    return False


def _angle_window(mol: Chem.Mol, d: int) -> Tuple[float, float]:
    """M-D-X window (degrees) for a non-chelating neighbour X of donor d."""
    if _is_linear_donor(mol, d):
        return 165.0, 180.0
    atom = mol.GetAtomWithIdx(d)
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    if sym == "O":
        return 100.0, 160.0  # water, alkoxide, phenoxide, carbonyl, carboxylate
    if sym in ("S", "Se", "Te"):
        return 95.0, 125.0
    if sym in ("P", "As", "Sb"):
        return 105.0, 125.0
    if atom.GetIsAromatic() or hyb == Chem.HybridizationType.SP2:
        return 110.0, 135.0
    if hyb == Chem.HybridizationType.SP3:
        return 100.0, 120.0
    return 95.0, 150.0


def _haptic_radii(lig: Chem.Mol, atoms: List[int], seed: int) -> List[float]:
    """Distance of each haptic atom from the centroid of the haptic set, from a free-ligand embedding."""
    m = Chem.Mol(lig)
    try:
        if rdDistGeom.EmbedMolecule(m, randomSeed=seed) == 0:
            X = m.GetConformer().GetPositions()[atoms]
            return list(np.linalg.norm(X - X.mean(axis=0), axis=1))
    except Exception:
        pass
    n = len(atoms)
    rho = 0.67 if n == 2 else 1.40 / (2 * np.sin(np.pi / n))
    return [rho] * n


@dataclass
class _System:
    mol: Chem.Mol                      # metal (0) + ligands (+H) + ghosts, no metal bonds
    cmap: Dict[int, Point3D]
    donors: List[Tuple[int, int, bool]]  # (atom index, site number, is_multidentate)
    haptic: List[dict]                 # {'site_num', 'ghost', 'atoms', 'rho', 'd_c'}
    ligand_of: np.ndarray              # ligand index per atom (-1 metal, -2 ghost)
    ligand_offsets: List[int]
    linear: List[int]
    charge: int
    windows: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)


def _build_system(t, site_coords: np.ndarray, haptic_scale: float, seed: int) -> _System:
    pt = Chem.GetPeriodicTable()
    sites = [s for pair in t.pairs for s in pair] + list(t.singles)
    if len(site_coords) != len(sites):
        raise ValueError(f"Expected {len(sites)} site coords, got {len(site_coords)}")

    ligs = []
    for payload in t.ligands:
        if payload.kind != "SMILES":
            raise ValueError(f"Only SMILES ligands supported, got {payload.kind}")
        ligs.append(_ligand_mol(payload.text))
    charge = int(t.ox) + sum(a.GetFormalCharge() for lig in ligs for a in lig.GetAtoms())

    metal = Chem.RWMol()
    m_atom = Chem.Atom(pt.GetAtomicNumber(t.metal))
    m_atom.SetNoImplicit(True)
    metal.AddAtom(m_atom)
    combined = metal.GetMol()
    offsets, ligand_of = [], [-1]
    for k, lig in enumerate(ligs):
        offsets.append(combined.GetNumAtoms())
        combined = Chem.CombineMols(combined, lig)
        ligand_of += [k] * lig.GetNumAtoms()
    rw = Chem.RWMol(combined)

    cmap = {0: Point3D(0.0, 0.0, 0.0)}
    donors, haptic = [], []
    sites_per_lig = {}
    for s in sites:
        sites_per_lig[s.lig] = sites_per_lig.get(s.lig, 0) + 1
    for k, (site, xyz) in enumerate(zip(sites, np.asarray(site_coords, dtype=float).reshape(-1, 3))):
        lig_idx = site.lig - 1
        local = _site_atoms(ligs[lig_idx], list(site.atoms))
        glob = [offsets[lig_idx] + i for i in local]
        if len(glob) == 1:
            cmap[glob[0]] = Point3D(*xyz)
            donors.append((glob[0], k + 1, sites_per_lig[site.lig] > 1))
            continue
        # Haptic site: pin a ghost at the ring centroid. trex_to_xyz scales the
        # M-centroid distance (predicted or idealized) by haptic_scale for the
        # MetalloGen ghost; divide it back out.
        norm = float(np.linalg.norm(xyz))
        d_c = norm / haptic_scale
        centroid = xyz / norm * d_c if norm > 1e-6 else xyz
        ghost = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(ghost).SetNoImplicit(True)
        ligand_of.append(-2)
        cmap[ghost] = Point3D(*centroid)
        haptic.append({"site_num": k + 1, "ghost": ghost, "atoms": glob, "d_c": d_c,
                       "rho": _haptic_radii(ligs[lig_idx], local, seed)})
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    linear = [d for d, _, _ in donors if _is_linear_donor(mol, d)]
    return _System(mol=mol, cmap=cmap, donors=donors, haptic=haptic, ligand_of=np.array(ligand_of),
                   ligand_offsets=offsets, linear=linear, charge=charge)


# =============================================================================
# Bounds
# =============================================================================

def _set(b: np.ndarray, i: int, j: int, lo: float, hi: float) -> None:
    i, j = min(i, j), max(i, j)
    b[i][j] = hi   # upper triangle: upper bound
    b[j][i] = lo   # lower triangle: lower bound


def _chelate_ring_sizes(mol: Chem.Mol, donor: int, donors_same_lig: List[int], topo: np.ndarray) -> Dict[int, int]:
    """For neighbours X of `donor` that lie on a shortest path to another donor of the
    same ligand, the size of the chelate ring M-D-X-...-D' (smallest one)."""
    out = {}
    for x in (n.GetIdx() for n in mol.GetAtomWithIdx(donor).GetNeighbors()):
        for d2 in donors_same_lig:
            if d2 == donor:
                continue
            path = topo[donor][d2]
            if path <= 5 and topo[x][d2] == path - 1:
                size = int(path) + 2
                out[x] = min(out.get(x, 99), size)
    return out


def _bounds(s: _System, inter_scale: float = 0.8, widen: float = 0.0) -> Tuple[np.ndarray, bool]:
    """Bounds matrix and whether triangle smoothing succeeded. When it fails (the
    pinned donors ask for a bite the ligand cannot make, e.g. a four-membered
    chelate at 90 deg), the unsmoothed matrix is returned and distance geometry
    distributes the strain."""
    mol, n = s.mol, s.mol.GetNumAtoms()
    RDLogger.DisableLog("rdApp.*")
    try:
        b = rdDistGeom.GetMoleculeBoundsMatrix(mol, set15bounds=True, scaleVDW=False, doTriangleSmoothing=False)
    finally:
        RDLogger.EnableLog("rdApp.*")
    b = np.array(b)
    lig = s.ligand_of
    pinned = set(s.cmap)
    ghosts = {h["ghost"] for h in s.haptic}
    haptic_atoms = {a for h in s.haptic for a in h["atoms"]}
    donor_set = {d for d, _, _ in s.donors}
    topo = Chem.GetDistanceMatrix(mol)  # metal and ghosts are isolated atoms

    # 1. Different ligands: only a softened van der Waals contact.
    for i in range(1, n):
        for j in range(i + 1, n):
            if lig[i] >= 0 and lig[j] >= 0 and lig[i] != lig[j] and not (i in pinned and j in pinned):
                b[j][i] *= inter_scale

    # 2. Metal row: no upper bound; anything that should not bind stays outside the
    #    bonding cutoff.
    z_m = mol.GetAtomWithIdx(0).GetAtomicNum()
    safe = np.zeros(n)
    for j in range(1, n):
        if j in ghosts:
            continue
        z = mol.GetAtomWithIdx(j).GetAtomicNum()
        safe[j] = _contact_cutoff(z_m, z) + _CONTACT_MARGIN
        _set(b, 0, j, max(_MIN_M_H if z == 1 else _MIN_M_HEAVY, safe[j]), 1000.0)

    # 3. Donor neighbours: M...X from the M-D-X window. A neighbour shared by two
    #    donors (a four-membered chelate) gets the overlap of both windows, or both
    #    together when they do not overlap.
    donors_by_lig = {}
    for d, _, _ in s.donors:
        donors_by_lig.setdefault(lig[d], []).append(d)
    ranges = {}
    for d, _, _ in s.donors:
        r = s.cmap[d].Length()
        chelate = _chelate_ring_sizes(mol, d, donors_by_lig[lig[d]], topo)
        default = _angle_window(mol, d)
        for x in (nb.GetIdx() for nb in mol.GetAtomWithIdx(d).GetNeighbors()):
            if x in donor_set or x in haptic_atoms:
                continue
            if x in chelate:
                lo_a, hi_a = _chelate_window(chelate[x], mol.GetAtomWithIdx(d).GetSymbol())
            else:
                lo_a, hi_a = default
            lo_a, hi_a = max(lo_a - widen, 60.0), min(hi_a + widen, 180.0)
            bl = 0.5 * (b[min(d, x)][max(d, x)] + b[max(d, x)][min(d, x)])
            lo = float(np.sqrt(r * r + bl * bl - 2 * r * bl * np.cos(np.radians(lo_a))))
            hi = float(np.sqrt(r * r + bl * bl - 2 * r * bl * np.cos(np.radians(hi_a))))
            ranges.setdefault(x, []).append((lo, hi))
            s.windows[(d, x)] = (lo_a, hi_a)
    for x, rs in ranges.items():
        lo, hi = max(r[0] for r in rs), min(r[1] for r in rs)
        if lo > hi:
            lo, hi = min(r[0] for r in rs), max(r[1] for r in rs)
        # never inside the bonding cutoff; when the window cannot reach past it
        # (a short M-D bond), the window's far end gives way
        lo = max(lo, safe[x])
        hi = max(hi, lo + 0.1)
        _set(b, 0, x, lo - 0.02, hi + 0.02)

    # 4. Haptic sites: ring atoms on a circle around the pinned centroid ghost.
    for h in s.haptic:
        g, d_c = h["ghost"], h["d_c"]
        for j in range(1, n):
            if j != g and j not in pinned:
                _set(b, g, j, min(b[max(g, j)][min(g, j)], 1.0), 1000.0)
        ring = set(h["atoms"])
        for a, rho in zip(h["atoms"], h["rho"]):
            tol = 0.08 + widen / 100.0
            _set(b, g, a, max(rho - tol, 0.0), rho + tol)
            dm = float(np.hypot(d_c, rho))
            _set(b, 0, a, dm - tol, dm + tol)
            # substituents on the ring point away from the metal
            for x in (nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()):
                if x in ring or x in donor_set or x in haptic_atoms:
                    continue
                kind = "H" if mol.GetAtomWithIdx(x).GetAtomicNum() == 1 else "heavy"
                bl = 0.5 * (b[min(a, x)][max(a, x)] + b[max(a, x)][min(a, x)])
                ang = np.radians(_HAPTIC_SUBSTITUENT_MIN[kind] - widen)
                lo = max(float(np.sqrt(dm * dm + bl * bl - 2 * dm * bl * np.cos(ang))), safe[x])
                if lo > b[x][0]:
                    b[x][0] = lo

    # 5. Everything pinned: exact.
    keys = sorted(pinned)
    for a, i in enumerate(keys):
        for j in keys[a + 1:]:
            dist = (s.cmap[i] - s.cmap[j]).Length()
            _set(b, i, j, max(dist - 0.01, 0.0), dist + 0.01)

    smoothed = b.copy()
    if DoTriangleSmoothing(smoothed):
        return smoothed, True
    return b, False


# =============================================================================
# Embedding
# =============================================================================

_STRATEGIES = (
    # name, ETKDG torsions, basic knowledge, enforce chirality
    ("native_etkdg", True, True, True),
    ("native_kdg", False, True, False),
    ("native_dg", False, False, False),
)


def _embed_once(s: _System, b: np.ndarray, strategy, seed: int, timeout: int) -> Optional[Chem.Mol]:
    name, torsions, knowledge, chirality = strategy
    p = rdDistGeom.ETKDGv3() if torsions else rdDistGeom.EmbedParameters()
    p.useExpTorsionAnglePrefs = torsions
    p.useBasicKnowledge = knowledge
    p.enforceChirality = chirality
    p.useRandomCoords = True
    p.embedFragmentsSeparately = False  # metal, ligands and ghosts are separate fragments
    p.ignoreSmoothingFailures = True
    p.randomSeed = int(seed)
    p.timeout = int(timeout)
    p.SetBoundsMat(b)
    p.SetCoordMap(s.cmap)
    m = Chem.Mol(s.mol)
    RDLogger.DisableLog("rdApp.*")
    try:
        cid = rdDistGeom.EmbedMolecule(m, p)
    except Exception as e:  # pragma: no cover - RDKit invariant violations
        logger.debug(f"{name}: {e}")
        cid = -1
    finally:
        RDLogger.EnableLog("rdApp.*")
    return m if cid == 0 else None


def _finish(s: _System, emb: Chem.Mol, ox: int) -> Tuple[Chem.Mol, Chem.Mol]:
    """(output mol without ghosts and with M-D bonds, clash-check mol with M-haptic bonds too)."""
    X = emb.GetConformer().GetPositions()
    rw = Chem.RWMol(emb)
    for d, _, _ in s.donors:
        rw.AddBond(0, d, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(0).SetFormalCharge(int(ox))
    for g in sorted((h["ghost"] for h in s.haptic), reverse=True):
        rw.RemoveAtom(g)
    out = rw.GetMol()
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*X[i]))
    check = Chem.RWMol(out)
    for h in s.haptic:
        for a in h["atoms"]:
            if check.GetBondBetweenAtoms(0, a) is None:
                check.AddBond(0, a, Chem.BondType.SINGLE)
    return out, check.GetMol()


def _cleanup(s: _System, emb: Chem.Mol, b: np.ndarray, max_iters: int = 2000) -> Chem.Mol:
    """Relax the ligands with UFF while the metal, donors and centroid ghosts stay
    fixed and every metal (and ghost) distance is held inside its bounds by a
    flat-bottomed restraint.

    ETKDG and KDG finish with a force field that only knows each ligand's own
    1-2/1-3 geometry, so they can undo the metal-side windows (a chelate ring
    folded onto the metal). This puts the windows back with real ring and
    conjugation geometry. The metal and ghosts have no UFF type and get no terms
    of their own. Kept only if it does not make the contacts or windows worse."""
    from rdkit.Chem import rdForceFieldHelpers
    m = Chem.Mol(emb)
    RDLogger.DisableLog("rdApp.*")
    try:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(m, ignoreInterfragInteractions=False)
        if ff is None:
            return emb
        ghosts = {h["ghost"] for h in s.haptic}
        fixed = set(s.cmap) | ghosts | {0}
        for i in fixed:
            ff.AddFixedPoint(int(i))
        n = m.GetNumAtoms()
        for j in range(1, n):
            if j in fixed:
                continue
            lo, hi = float(b[j][0]), float(b[0][j])
            ff.UFFAddDistanceConstraint(0, j, False, lo, min(hi, 100.0), 500.0)
            for g in ghosts:
                glo, ghi = float(b[max(g, j)][min(g, j)]), float(b[min(g, j)][max(g, j)])
                if ghi < 50.0:
                    ff.UFFAddDistanceConstraint(int(g), j, False, glo, ghi, 500.0)
        ff.Initialize()
        ff.Minimize(maxIts=max_iters)
        pos = np.array(ff.Positions()).reshape(-1, 3)
    except Exception as e:  # pragma: no cover - typing problems on unusual ligands
        logger.debug(f"UFF cleanup skipped: {e}")
        return emb
    finally:
        RDLogger.EnableLog("rdApp.*")
    if not np.all(np.isfinite(pos)):
        return emb
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*pos[i]))
    X0, X1 = emb.GetConformer().GetPositions(), pos
    before = sum(_contacts(s, X0)) * 100 + _score(s, X0)[0]
    after = sum(_contacts(s, X1)) * 100 + _score(s, X1)[0]
    return m if after <= before else emb


def _contacts(s: _System, X: np.ndarray) -> Tuple[int, int]:
    """(atoms that should not bind but sit inside the bonding cutoff, haptic atoms
    outside it), judged the way trex.xyz_to_trex reads a structure. Donors sit
    where the internals put them, so they are not counted; nor are haptic atoms
    whose target distance from the internals is already beyond the cutoff."""
    z_m = s.mol.GetAtomWithIdx(0).GetAtomicNum()
    donors = {d for d, _, _ in s.donors}
    target = {a: float(np.hypot(h["d_c"], rho)) for h in s.haptic for a, rho in zip(h["atoms"], h["rho"])}
    ghosts = {h["ghost"] for h in s.haptic}
    d = np.linalg.norm(X - X[0], axis=1)
    extra = missing = 0
    for j, atom in enumerate(s.mol.GetAtoms()):
        if j == 0 or j in ghosts or j in donors:
            continue
        cut = _contact_cutoff(z_m, atom.GetAtomicNum())
        if j in target:
            missing += target[j] <= cut < d[j]
        else:
            extra += d[j] < cut
    return int(extra), int(missing)


def _score(s: _System, X: np.ndarray) -> Tuple[float, float, float]:
    """(worst M-D-X window violation in degrees, worst haptic M-C error in A,
    shortest metal distance to a heavy atom that is neither donor nor haptic)."""
    worst = 0.0
    for (d, x), (lo, hi) in s.windows.items():
        u, v = X[0] - X[d], X[x] - X[d]
        a = float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))
        worst = max(worst, lo - a, a - hi)
    hap = 0.0
    for h in s.haptic:
        for a, rho in zip(h["atoms"], h["rho"]):
            hap = max(hap, abs(float(np.linalg.norm(X[a] - X[0])) - float(np.hypot(h["d_c"], rho))))
    skip = {d for d, _, _ in s.donors} | {a for h in s.haptic for a in h["atoms"]} | {h["ghost"] for h in s.haptic}
    heavy = [i for i, a in enumerate(s.mol.GetAtoms()) if i and i not in skip and a.GetAtomicNum() > 1]
    closest = float(np.linalg.norm(X[heavy] - X[0], axis=1).min()) if heavy else 9.9
    return max(worst, 0.0), hap, closest


def embed_trex_native(
    t: "TrexFull",
    site_coords: np.ndarray,
    max_attempts: int = 30,
    seed: int = None,
    geometry: str = None,
    reject_severe_clashes: bool = True,
    haptic_scale: float = 0.35,
    timeout: int = 10,
) -> EmbedResult:
    """
    Embed a T-REX complex keeping every ligand's Lewis structure as written.

    Same arguments and EmbedResult as trex_embed.embed_trex_metallogen; `geometry`
    is accepted for compatibility and not needed. Haptic site coordinates are
    expected scaled by `haptic_scale`, as trex_to_xyz passes them (use 1.0 for
    true M-centroid distances). `timeout` is RDKit's per-attempt limit in seconds.
    """
    seed = 42 if seed is None else seed
    diagnostics: dict = {"method": "native"}
    spin = t.spin or 1
    n_sites = 2 * len(t.pairs) + len(t.singles)
    diagnostics["n_sites"] = n_sites

    def fail(name, charge=0, **extra):
        diagnostics.update(extra)
        diagnostics["embedding_level_name"] = name
        return EmbedResult(mol=None, success=False, metal_idx=0, coord_atom_indices=[], charge=charge,
                           spin=spin, diagnostics=diagnostics, embedding_level=-1, embedding_level_name=name)

    site_coords = np.asarray(site_coords, dtype=float).reshape(-1, 3)
    norms = np.linalg.norm(site_coords, axis=1)
    collapsed = [i + 1 for i, r in enumerate(norms) if r < 0.3]
    if collapsed:
        logger.error(f"Coordination sphere has site(s) {collapsed} at the metal; not embedding")
        return fail("collapsed_site", collapsed_sites=collapsed)

    s = _build_system(t, site_coords, haptic_scale, seed)
    diagnostics["n_ligands"] = len(t.ligands)
    diagnostics["total_ligand_atoms"] = s.mol.GetNumAtoms() - 1 - len(s.haptic)

    for inter_scale, widen in ((0.8, 0.0), (0.7, 5.0), (0.6, 10.0)):
        s.windows.clear()
        b, smoothed = _bounds(s, inter_scale=inter_scale, widen=widen)
        if smoothed:
            break
    diagnostics["bounds"] = {"inter_ligand_scale": inter_scale, "angle_widening": widen, "smoothed": smoothed}

    # Strategies alternate (ETKDG torsions help flexible ligands; plain DG satisfies the
    # metal-side windows more reliably when idealized angles strain a rigid chelate).
    # The first structure that meets every window is returned; otherwise the best one.
    best, last_clash = None, None
    for attempt in range(max(1, max_attempts)):
        level = attempt % len(_STRATEGIES)
        strategy = _STRATEGIES[level]
        emb = _embed_once(s, b, strategy, seed + 7919 * attempt, timeout)
        if emb is None:
            continue
        emb = _cleanup(s, emb, b)
        X_emb = emb.GetConformer().GetPositions()
        viol, hap, closest = _score(s, X_emb)
        extra, missing = _contacts(s, X_emb)
        out, check = _finish(s, emb, t.ox)
        if s.linear:
            out, _ = _straighten_linear_ligands(out, 0, s.linear)
            check_conf = check.GetConformer()
            for i, p in enumerate(out.GetConformer().GetPositions()):
                check_conf.SetAtomPosition(i, Point3D(*p))
        clash = check_atom_clashes(check)
        last_clash = clash
        if reject_severe_clashes and clash.has_severe_clash:
            logger.debug(f"{strategy[0]} attempt {attempt}: {clash.n_severe_clashes} severe clashes")
            continue
        # A heavy atom closer than 1.9 A to the metal is kept only if nothing better turns
        # up: idealized angles can force it (a four-membered chelate held at 90 deg).
        # Wrong metal contacts change what the structure is (T-REX read back, and the
        # GFN-FF topology), so they weigh more than any angle miss.
        key = viol + 50.0 * hap + (100.0 if closest < 1.9 else 0.0) + 200.0 * (extra + missing)
        if best is None or key < best[0]:
            best = (key, level, strategy[0], out, clash, viol, hap, closest, extra, missing)
        if viol <= 5.0 and hap <= 0.2 and closest >= 1.9 and extra == 0 and missing == 0:
            break
    if best is not None:
        _, level, name, out, clash, viol, hap, closest, extra, missing = best
        donors = sorted(d for d, _, _ in s.donors)
        if viol > 15.0 or hap > 0.3 or closest < 1.9 or extra or missing:
            diagnostics["geometry_warning"] = (
                f"strained coordination sphere: worst M-D-X window miss {viol:.0f} deg, "
                f"haptic M-C error {hap:.2f} A, closest non-donor heavy atom {closest:.2f} A from the metal, "
                f"{extra} unintended metal contact(s), {missing} haptic atom(s) beyond bonding distance")
            logger.warning(diagnostics["geometry_warning"])
        diagnostics.update({
            "unintended_metal_contacts": extra,
            "missing_metal_contacts": missing,
            "embedding_level": level,
            "embedding_level_name": name,
            "window_violation_deg": round(viol, 1),
            "haptic_distance_error": round(hap, 3),
            "closest_nondonor_heavy": round(closest, 2),
            "n_severe_clashes": clash.n_severe_clashes,
            "n_moderate_clashes": clash.n_moderate_clashes,
            "min_nonbonded_distance": clash.min_distance,
            "metal_binding_infos": [(d, k, multi, False) for d, k, multi in s.donors],
            "coord_atom_indices": donors,
            "haptic_ring_infos": [{"site_num": h["site_num"], "ring_atom_indices": list(h["atoms"]),
                                   "hapticity": len(h["atoms"])} for h in s.haptic],
        })
        return EmbedResult(mol=out, success=True, metal_idx=0, coord_atom_indices=donors, charge=s.charge,
                           spin=spin, diagnostics=diagnostics, embedding_level=level,
                           embedding_level_name=name, clash_result=clash)
    if last_clash is not None:
        diagnostics["n_severe_clashes"] = last_clash.n_severe_clashes
    return fail("all_failed", s.charge)


__all__ = ["embed_trex_native"]
