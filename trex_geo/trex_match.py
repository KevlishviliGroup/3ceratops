# trex_geo/trex_match.py
"""
Tiered comparison of a built structure with the T-REX string it was built from.

The structure is read back with trex.xyz_to_trex (chirality-aware, with the input's
geometry as the hint, as the training strings were made) and compared in tiers. Each
tier includes the ones before it:

    0  built          a structure exists, but its connectivity differs (or read-back failed)
    1  connectivity   same atoms, same bonds (bond orders and charges ignored), same
                      metal-donor bonds and hapticities
    2  trans map      ... and the same trans pairs
    3  chirality      ... and the same chirality, where the input specifies one
    4  identical      the read-back is the input T-REX string (canonical), up to a
                      chirality label the input leaves unspecified

Tiers 1-2 compare graphs, not strings. The reader assigns bond orders and charges from
the geometry, and when it picks a different Lewis structure (a resonance form, where a
charge sits, a non-innocent ligand read as reduced) the ligand SMILES, the atom
numbering and the metal oxidation state all change with it. The graph keeps only
elements, hydrogen counts, bonds, metal-donor sites and trans pairs, and is compared by
isomorphism, so renumbering and symmetry-equivalent donors (either O- of a nitrate)
do not matter.

Chirality is a property of the canonical string, so tier 3 needs both strings in the
same Lewis structure. When the plain read-back differs, the structure is read again with
the input's ligand Lewis structures imposed on every fragment whose skeleton matches
(TrexMatch.guided), and that string is compared.

A structure counts as valid (TrexMatch.valid) from tier 3 up. The pipelines rank
conformers with this, and "auto" embedding falls back to the MetalloGen route when the
native result is not valid.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
from networkx.algorithms import isomorphism as nx_iso
from rdkit import Chem, rdBase

logger = logging.getLogger(__name__)

BUILT, CONNECTIVITY, TRANS_MAP, CHIRALITY, IDENTICAL = range(5)
TIER_NAMES = {-1: "not built", BUILT: "built", CONNECTIVITY: "connectivity",
              TRANS_MAP: "trans map", CHIRALITY: "chirality", IDENTICAL: "identical"}
VALID_LEVEL = CHIRALITY

# T-REX geometry code -> geometry name for trex.xyz_to_trex (sets how many trans pairs it picks)
_READER_GEOMETRY = {
    "O": "octahedral", "sqpl": "square planar", "sqpy": "square pyramidal", "thd": "tetrahedral",
    "trbp": "trigonal bipyramidal", "pntbp": "pentagonal bipyramidal", "ln": "linear",
    "trpl": "trigonal planar", "tsh": "t shaped", "ssw": "seesaw", "bnt": "bent",
    "SP": "square planar", "SQP": "square pyramidal", "T": "tetrahedral", "TBP": "trigonal bipyramidal",
    "L": "linear", "B": "bent", "BNT": "bent",
}


@dataclass
class TrexMatch:
    level: int                      # BUILT .. IDENTICAL
    readback: Optional[str] = None  # canonical read-back with chirality
    guided: Optional[str] = None    # read-back with the input's Lewis structures imposed
    detail: str = ""
    chirality_checked: bool = True  # False when tier 3 was granted without a comparable string

    @property
    def tier(self) -> str:
        return TIER_NAMES[self.level]

    @property
    def valid(self) -> bool:
        return self.level >= VALID_LEVEL

    def as_dict(self) -> dict:
        return {"level": self.level, "tier": self.tier, "valid": self.valid, "readback": self.readback,
                "guided": self.guided, "detail": self.detail, "chirality_checked": self.chirality_checked}


# =============================================================================
# Small helpers
# =============================================================================

def _parse(trex):
    from trex.parse_full import parse_trex_full
    return parse_trex_full(trex) if isinstance(trex, str) else trex


def _canonical(trex: str, drop_chirality: bool = False) -> str:
    from trex.canon_full import canonicalize_full, pretty_trex_full
    from trex.parse_full import parse_trex_full_string
    t = canonicalize_full(parse_trex_full_string(trex))
    if drop_chirality:
        t.X = None
    return pretty_trex_full(t)


def _ligand_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # unusual valences: keep what can be perceived
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"cannot parse ligand SMILES {smiles!r}")
        mol.UpdatePropertyCache(strict=False)
    return mol


def _heavy_atoms(mol: Chem.Mol) -> List[int]:
    """T-REX site positions count heavy atoms in SMILES order (a lone H ligand is position 1)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    return heavy or [a.GetIdx() for a in mol.GetAtoms()]


def total_charge(trex) -> int:
    """Metal oxidation state plus the formal charges written in the ligand SMILES."""
    t = _parse(trex)
    q = int(t.ox)
    for lig in t.ligands:
        q += sum(a.GetFormalCharge() for a in _ligand_mol(lig.text).GetAtoms())
    return q


def geometry_hint(trex) -> Optional[str]:
    """Geometry name for trex.xyz_to_trex from the T-REX geometry code. G:TP means trigonal
    prismatic at CN 6 and trigonal planar at CN 3."""
    t = _parse(trex)
    code = t.get_geo_type()
    cn = 2 * len(t.pairs) + len(t.singles)
    if code == "TP":
        return "trigonal prismatic" if cn == 6 else "trigonal planar"
    return _READER_GEOMETRY.get(code)


# =============================================================================
# Charge-free graph of a T-REX string
# =============================================================================

def trex_graph(trex, trans: bool = True) -> nx.Graph:
    """Metal, ligand heavy atoms (element, H count), bonds without orders, one node per
    coordination site joined to the metal and to its donor atom(s), and (trans=True) an
    edge between the two sites of every trans pair. Charges, bond orders and the
    oxidation state are left out."""
    t = _parse(trex)
    G = nx.Graph()
    G.add_node("M", lab=f"M:{t.metal}")
    pos: Dict[Tuple[int, int], tuple] = {}
    for k, lig in enumerate(t.ligands, start=1):
        if lig.kind != "SMILES":
            raise ValueError(f"only SMILES ligands are supported, got {lig.kind}")
        mol = _ligand_mol(lig.text)
        heavy = _heavy_atoms(mol)
        keep = set(heavy)
        for p, i in enumerate(heavy, start=1):
            a = mol.GetAtomWithIdx(i)
            G.add_node((k, i), lab=f"{a.GetSymbol()}:{a.GetTotalNumHs(includeNeighbors=True)}")
            pos[(k, p)] = (k, i)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in keep and j in keep:
                G.add_edge((k, i), (k, j), kind="b")
    n = 0

    def add_site(site) -> str:
        nonlocal n
        n += 1
        node = ("S", n)
        G.add_node(node, lab="S")
        G.add_edge("M", node, kind="c")
        for p in site.atoms:
            if (site.lig, p) not in pos:
                raise ValueError(f"site {site.lig}:{p} is not a heavy atom of ligand {site.lig}")
            G.add_edge(node, pos[(site.lig, p)], kind="m")
        return node

    for a, b in t.pairs:
        na, nb = add_site(a), add_site(b)
        if trans:
            G.add_edge(na, nb, kind="t")
    for s in t.singles:
        add_site(s)
    return G


def same_graph(g1: nx.Graph, g2: nx.Graph) -> bool:
    if g1.number_of_nodes() != g2.number_of_nodes() or g1.number_of_edges() != g2.number_of_edges():
        return False
    if Counter(d["lab"] for _, d in g1.nodes(data=True)) != Counter(d["lab"] for _, d in g2.nodes(data=True)):
        return False
    if Counter(d["kind"] for *_, d in g1.edges(data=True)) != Counter(d["kind"] for *_, d in g2.edges(data=True)):
        return False
    if (nx.weisfeiler_lehman_graph_hash(g1, node_attr="lab", edge_attr="kind")
            != nx.weisfeiler_lehman_graph_hash(g2, node_attr="lab", edge_attr="kind")):
        return False
    return nx.is_isomorphic(g1, g2, node_match=lambda a, b: a["lab"] == b["lab"],
                            edge_match=lambda a, b: a["kind"] == b["kind"])


# =============================================================================
# Read-back, plain and with the input's Lewis structures imposed
# =============================================================================

def _heavy_graph(mol: Chem.Mol, donors: Sequence[int]) -> Tuple[nx.Graph, List[int]]:
    """Heavy atoms labelled (element, explicit-H neighbours, donor?) for template matching."""
    heavy = _heavy_atoms(mol)
    keep, dset = set(heavy), set(donors)
    g = nx.Graph()
    for i in heavy:
        a = mol.GetAtomWithIdx(i)
        nh = sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 1) if a.GetAtomicNum() != 1 else 0
        g.add_node(i, lab=(a.GetSymbol(), nh, i in dset))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in keep and j in keep:
            g.add_edge(i, j)
    return g, heavy


def _templates(t) -> List[Optional[dict]]:
    """Each input ligand with explicit H, kekulized, and its donor atoms, for imposing its
    Lewis structure on a read-back fragment."""
    sites = [s for pair in t.pairs for s in pair] + list(t.singles)
    out = []
    for k, lig in enumerate(t.ligands, start=1):
        try:
            mol = Chem.MolFromSmiles(lig.text)
            if mol is None:
                out.append(None)
                continue
            mol = Chem.AddHs(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            heavy = _heavy_atoms(mol)
            d = {heavy[p - 1] for s in sites if s.lig == k for p in s.atoms if 1 <= p <= len(heavy)}
            g, _ = _heavy_graph(mol, d)
            out.append({"mol": mol, "graph": g, "n_atoms": mol.GetNumAtoms(),
                        "charge": sum(a.GetFormalCharge() for a in mol.GetAtoms())})
        except Exception:
            out.append(None)
    return out


def _impose(frag: Chem.Mol, tmpl: Chem.Mol, frag_to_tmpl: Dict[int, int]) -> Optional[Chem.Mol]:
    """The fragment (read-back atom order, explicit H) with the template's bond orders and charges."""
    rw = Chem.RWMol(frag)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumRadicalElectrons(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    tmpl_to_frag = {v: k for k, v in frag_to_tmpl.items()}
    for b in tmpl.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in tmpl_to_frag and j in tmpl_to_frag:
            fb = rw.GetBondBetweenAtoms(tmpl_to_frag[i], tmpl_to_frag[j])
            if fb is None:
                return None
            fb.SetBondType(b.GetBondType())
    for ti, fi in tmpl_to_frag.items():
        ta, fa = tmpl.GetAtomWithIdx(ti), rw.GetAtomWithIdx(fi)
        fa.SetFormalCharge(ta.GetFormalCharge())
        fa.SetNumRadicalElectrons(ta.GetNumRadicalElectrons())
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


@contextlib.contextmanager
def _imposed_lewis_structures(t):
    """Patch trex.xyz_to_trex so that every fragment whose skeleton (heavy atoms, H counts,
    bonds, donor atoms) matches an unused input ligand gets that ligand's bond orders and
    charges instead of the reader's own assignment."""
    import trex.xyz_to_trex as reader

    templates = _templates(t)
    used: set = set()
    original = reader.get_lig_mol

    def guided(mol, charge, coordinating_atoms):
        try:
            g, heavy = _heavy_graph(mol, coordinating_atoms)
            for k, tp in enumerate(templates):
                if tp is None or k in used or tp["n_atoms"] != mol.GetNumAtoms():
                    continue
                gm = nx_iso.GraphMatcher(g, tp["graph"], node_match=lambda a, b: a["lab"] == b["lab"])
                if not gm.is_isomorphic():
                    continue
                imposed = _impose(mol, tp["mol"], dict(gm.mapping))
                if imposed is not None:
                    used.add(k)
                    return imposed, tp["charge"]
        except Exception as e:  # pragma: no cover - fall back to the reader's own assignment
            logger.debug(f"imposing input Lewis structure failed: {e}")
        return original(mol, charge, coordinating_atoms)

    reader.get_lig_mol = guided
    try:
        yield
    finally:
        reader.get_lig_mol = original


def read_back(xyz_path: str, charge: int, geometry: Optional[str] = None, impose_from=None) -> str:
    """Canonical T-REX with chirality from an XYZ file. impose_from: a T-REX string (or
    TrexFull) whose ligand Lewis structures are imposed on matching fragments."""
    from trex.xyz_to_trex import xyz_to_trex_with_chirality
    _quiet = rdBase.BlockLogs()  # noqa: F841 - RDKit logging resumes when this is released
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if impose_from is None:
            return xyz_to_trex_with_chirality(str(xyz_path), int(charge), geometry=geometry)
        with _imposed_lewis_structures(_parse(impose_from)):
            return xyz_to_trex_with_chirality(str(xyz_path), int(charge), geometry=geometry)


# =============================================================================
# Comparison
# =============================================================================

def _chirality_differs(reference_canon: str, other: str) -> Optional[bool]:
    """True/False when the two strings are comparable for chirality, None when not."""
    from trex.isomer_compare import IsomerType, compare_isomers
    other_canon = _canonical(other)
    if other_canon == reference_canon:
        return False
    if _canonical(other, drop_chirality=True) == _canonical(reference_canon, drop_chirality=True):
        return True  # same string apart from the chirality flag (the reference has one)
    res = compare_isomers(_parse(reference_canon), _parse(other_canon))
    if res.isomer_type == IsomerType.ENANTIOMER or "chirality" in (res.details or "").lower():
        return True
    return None


def compare_trex(reference: str, readback: Optional[str], guided: Optional[str] = None) -> TrexMatch:
    """Tier of a read-back string against the reference T-REX (both T-REX strings)."""
    _quiet = rdBase.BlockLogs()  # noqa: F841
    if readback is None:
        return TrexMatch(BUILT, None, guided, "read-back failed")
    ref_canon = _canonical(reference)
    ref_t = _parse(ref_canon)
    rb_canon = _canonical(readback)
    if rb_canon == ref_canon or (ref_t.X is None and _canonical(readback, drop_chirality=True) == ref_canon):
        return TrexMatch(IDENTICAL, rb_canon, guided)
    try:
        if not same_graph(trex_graph(ref_canon, trans=False), trex_graph(rb_canon, trans=False)):
            return TrexMatch(BUILT, rb_canon, guided, "connectivity differs")
        if not same_graph(trex_graph(ref_canon), trex_graph(rb_canon)):
            return TrexMatch(CONNECTIVITY, rb_canon, guided, "trans map differs")
    except Exception as e:
        return TrexMatch(BUILT, rb_canon, guided, f"graph comparison failed: {e}")
    if ref_t.X is None:
        return TrexMatch(CHIRALITY, rb_canon, guided, "bond orders or charges read differently")
    for other in (guided, rb_canon):
        if other is None:
            continue
        differs = _chirality_differs(ref_canon, other)
        if differs is True:
            return TrexMatch(TRANS_MAP, rb_canon, guided, "wrong chirality")
        if differs is False:
            return TrexMatch(CHIRALITY, rb_canon, guided, "bond orders or charges read differently")
    return TrexMatch(CHIRALITY, rb_canon, guided, "chirality not comparable", chirality_checked=False)


def match_structure(xyz_path: str, trex: str, charge: Optional[int] = None,
                    geometry: Optional[str] = None) -> TrexMatch:
    """Read an XYZ file back and grade it against the T-REX string it was built from.
    charge defaults to the T-REX total charge, geometry to the hint from its geometry code."""
    _quiet = rdBase.BlockLogs()  # noqa: F841
    t = _parse(trex)
    if charge is None:
        charge = total_charge(t)
    if geometry is None:
        geometry = geometry_hint(t)
    try:
        plain = read_back(xyz_path, charge, geometry)
    except Exception as e:
        return TrexMatch(BUILT, None, None, f"read-back failed: {type(e).__name__}: {e}"[:300])
    m = compare_trex(trex, plain)
    if m.level == CHIRALITY and not m.chirality_checked:
        # The input specifies a chirality, but the read-back's Lewis structures differ from
        # the input's, so the two flags are not comparable: read again with the input's
        # Lewis structures imposed and judge on that string.
        try:
            guided = read_back(xyz_path, charge, geometry, impose_from=t)
            m = compare_trex(trex, plain, guided)
        except Exception as e:
            m.detail += f"; guided read-back failed: {e}"[:200]
    return m


def match_mol(mol: Chem.Mol, trex: str, charge: Optional[int] = None, geometry: Optional[str] = None,
              work_dir: Optional[str] = None) -> TrexMatch:
    """match_structure for an RDKit molecule with a 3D conformer."""
    work = work_dir or tempfile.mkdtemp(prefix="trex_match_")
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "structure.xyz")
    conf = mol.GetConformer()
    with open(path, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            f.write(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}\n")
    return match_structure(path, trex, charge=charge, geometry=geometry)


__all__ = ["BUILT", "CONNECTIVITY", "TRANS_MAP", "CHIRALITY", "IDENTICAL", "TIER_NAMES", "VALID_LEVEL",
           "TrexMatch", "trex_graph", "same_graph", "read_back", "compare_trex", "match_structure",
           "match_mol", "total_charge", "geometry_hint"]
