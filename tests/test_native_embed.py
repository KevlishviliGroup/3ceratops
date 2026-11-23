"""Embedding regression tests for the native (LP-free) route."""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("trex")

from rdkit import Chem  # noqa: E402
from trex.parse_full import parse_trex_full  # noqa: E402

from trex_geo import compute_idealized_internals, embed_trex_metallogen, reconstruct_coordination_sphere  # noqa: E402
from trex_geo.trex_to_xyz import get_sites_ordered, is_site_haptic, scale_haptic_distances  # noqa: E402

CN, CO, CL = "SMILES:[C-]#N", "SMILES:[C-]#[O+]", "SMILES:[Cl-]"
CASES = {
    # LP target infeasible on 03b2f79 (CN 4 halide anion)
    "PtCl4": f"Pt{{+2}}|L=[{CL},{CL},{CL},{CL}]|MAP:{{(1:1,2:1),(3:1,4:1)}}|G:SP",
    # through-metal 1-4 bounds (needed level 3 on ffcdc11)
    "cisPtBr2NH3": "Pt{+2}|L=[SMILES:[Br-],SMILES:[Br-],SMILES:N,SMILES:N]|MAP:{(1:1,3:1),(2:1,4:1)}|G:SP",
    # linear ligands
    "FeCN6": "Fe{+2}|L=[" + ",".join([CN] * 6) + "]|MAP:{(1:1,2:1),(3:1,4:1),(5:1,6:1)}|G:O",
    # short geometry code
    "NiCO4": "Ni{+0}|L=[" + ",".join([CO] * 4) + "]|MAP:{;1:1,2:1,3:1,4:1}|G:T",
    # chelates
    "Cuacac2": "Cu{+2}|L=[SMILES:CC(=O)/C=C(/C)[O-],SMILES:CC(=O)/C=C(/C)[O-]]|MAP:{(1:3,2:7),(1:7,2:3)}|G:SP",
    # haptic
    "ferrocene": "Fe{+2}|L=[SMILES:[cH-]1cccc1,SMILES:[cH-]1cccc1]|MAP:{(1:[1,2,3,4,5],2:[1,2,3,4,5])}",
}


def _build(trex_str, method="native"):
    t = parse_trex_full(trex_str)
    internals = compute_idealized_internals(t)
    sites = get_sites_ordered(t)
    site_r = internals["site_r"]
    if any(is_site_haptic(s) for s in sites):
        site_r = scale_haptic_distances(site_r, sites, 0.35)
    coords, _ = reconstruct_coordination_sphere(
        t=t, site_r=site_r, trans_cos=internals["trans_cos"], cis_cos=internals["cis_cos"],
        cis_pair_indices=[tuple(p) for p in internals["cis_pair_indices"]], expected_chirality=t.X)
    return t, embed_trex_metallogen(t, np.asarray(coords), max_attempts=5, seed=42, method=method)


def _bond_counts(mol):
    mol = Chem.RWMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return Counter(b.GetBondTypeAsDouble() for b in mol.GetBonds()
                   if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1)


@pytest.mark.parametrize("name", sorted(CASES))
def test_native_builds_and_keeps_ligands(name):
    t, r = _build(CASES[name])
    assert r.success, r.diagnostics
    assert r.diagnostics["method"] == "native"
    X = r.mol.GetConformer().GetPositions()
    metal = X[r.metal_idx]
    # donors sit at their sites; nothing else is on top of the metal
    assert all(1.8 < np.linalg.norm(X[d] - metal) < 2.8 for d in r.coord_atom_indices)
    others = [i for i in range(len(X)) if i != r.metal_idx and i not in r.coord_atom_indices]
    if others:
        assert np.linalg.norm(X[others] - metal, axis=1).min() > 1.7
    # every ligand comes out with the bonding written in its SMILES
    ligands = Chem.RWMol(r.mol)
    ligands.RemoveAtom(r.metal_idx)
    got = Counter(frozenset(_bond_counts(f).items())
                  for f in Chem.GetMolFrags(ligands.GetMol(), asMols=True, sanitizeFrags=False))
    want = Counter(frozenset(_bond_counts(Chem.MolFromSmiles(l.text)).items()) for l in t.ligands)
    assert got == want


def test_linear_ligands_are_straight():
    _, r = _build(CASES["FeCN6"])
    X = r.mol.GetConformer().GetPositions()
    for d in r.coord_atom_indices:
        (n,) = [a.GetIdx() for a in r.mol.GetAtomWithIdx(d).GetNeighbors() if a.GetIdx() != r.metal_idx]
        u, v = X[r.metal_idx] - X[d], X[n] - X[d]
        assert np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))) > 170


def test_haptic_rings_sit_at_the_centroid_distance():
    _, r = _build(CASES["ferrocene"])
    X = r.mol.GetConformer().GetPositions()
    rings = r.diagnostics["haptic_ring_infos"]
    assert len(rings) == 2
    for ring in rings:
        R = X[ring["ring_atom_indices"]]
        assert abs(np.linalg.norm(R.mean(axis=0) - X[r.metal_idx]) - 1.65) < 0.15
        assert np.all(np.abs(np.linalg.norm(R - X[r.metal_idx], axis=1) - 2.08) < 0.15)


# tmQM complexes (canonical T-REX from xyz_to_trex) whose chelate rings or ring
# substituents used to fold onto the metal
ROUND_TRIP = dict(CASES, **{
    "IJALEN": "Zn{+2} | L=[ SMILES:CCc1occc(=S)c1[O-], SMILES:CCc1occc(=S)c1[O-] ] | MAP:{ 1:8, 1:10, 2:8, 2:10 }",
    "DITLEB": "Pt{+4} | L=[ SMILES:O=C([O-])C(=O)[O-], SMILES:NC1CCCCC1N, SMILES:[OH-], SMILES:[OH-] ] "
              "| MAP:{ (1:3, 2:1), (1:6, 2:8), (3:1, 4:1) } | G:O",
    "POZMUM": "W{+2} | L=[ SMILES:S=C1SC(=S)C(=S)S1, SMILES:c1cc[cH-]c1, SMILES:c1cc[cH-]c1 ] "
              "| MAP:{ 1:5, 1:7, 2:[1,2,3,4,5], 3:[1,2,3,4,5] }",
})
# (The ansa-zirconocene TORLAN now builds without contacts too, but the reader puts the
# indenide charge on another ring carbon, so its string differs by resonance form only.)


@pytest.mark.parametrize("name", sorted(ROUND_TRIP))
def test_structure_reads_back_as_the_same_trex(name, tmp_path):
    """Read the built xyz back with trex.xyz_to_trex (bonds from coordinates only):
    the same complex, with no atom that should not bind inside bonding distance."""
    from trex.xyz_to_trex import _canonicalize_trex_string, xyz_to_trex_canonical

    t, r = _build(ROUND_TRIP[name])
    assert r.success, r.diagnostics
    assert r.diagnostics["unintended_metal_contacts"] == 0
    m = r.mol
    path = tmp_path / "built.xyz"
    path.write_text(f"{m.GetNumAtoms()}\n{name}\n" + "".join(
        f"{a.GetSymbol()} {x:.5f} {y:.5f} {z:.5f}\n" for a, (x, y, z) in zip(m.GetAtoms(), m.GetConformer().GetPositions())))
    charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
    strip = lambda s: s.replace("/", "").replace("\\", "")  # the reader writes no double-bond stereo
    assert strip(xyz_to_trex_canonical(path, charge)) == strip(_canonicalize_trex_string(ROUND_TRIP[name]))


def test_collapsed_site_is_reported():
    t = parse_trex_full(CASES["PtCl4"])
    coords = np.array([[2.3, 0, 0], [-2.3, 0, 0], [0, 2.3, 0], [0, 0, 0.0]])
    r = embed_trex_metallogen(t, coords, method="native")
    assert not r.success
    assert r.embedding_level_name == "collapsed_site"
