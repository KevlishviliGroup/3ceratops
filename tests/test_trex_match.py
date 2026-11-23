"""Tiered T-REX comparison (trex_geo.trex_match) and the auto embedding fallback (embed_choice)."""
import os

from rdkit import Chem

from trex_geo.trex_match import (
    BUILT, CHIRALITY, CONNECTIVITY, IDENTICAL, TRANS_MAP,
    compare_trex, geometry_hint, match_mol, total_charge,
)

# Read-backs of real builds whose Lewis structure the reader assigned differently
RESONANCE = {
    "Fischer carbene: C=O+ vs C=N+": (
        "Cr{0} | L=[ SMILES:Clc1ccc(P(c2ccc(Cl)cc2)c2ccc(Cl)cc2)cc1, SMILES:[C-]#CC(=[O+]C)N(C)C, SMILES:[C-]#[O+], SMILES:[C-]#[O+], SMILES:[C-]#[O+], SMILES:[C-]#[O+] ] | MAP:{ (1:6, 3:1), (2:1, 4:1), (5:1, 6:1) } | G:O",
        "Cr{0} | L=[ SMILES:Clc1ccc(P(c2ccc(Cl)cc2)c2ccc(Cl)cc2)cc1, SMILES:[C-]#CC(OC)=[N+](C)C, SMILES:[C-]#[O+], SMILES:[C-]#[O+], SMILES:[C-]#[O+], SMILES:[C-]#[O+] ] | MAP:{ (1:6, 3:1), (2:1, 4:1), (5:1, 6:1) } | G:O"),
    "azopyridine read as reduced, Ru(+2) vs Ru(+6)": (
        "Ru{+2} | L=[ SMILES:c1ccc(N=Nc2ccccn2)cc1, SMILES:c1ccc(N=Nc2ccccn2)cc1, SMILES:[C-]#N, SMILES:[C-]#N ] | MAP:{ (1:5, 2:5), (1:12, 2:12), (3:1, 4:1) } | G:O",
        "Ru{+6} | L=[ SMILES:c1ccc([N-]N=c2cccc[n-]2)cc1, SMILES:c1ccc([N-]N=c2cccc[n-]2)cc1, SMILES:[C-]#N, SMILES:[C-]#N ] | MAP:{ (1:5, 2:5), (1:12, 2:12), (3:1, 4:1) } | G:O"),
    "Cp on a borate: charge on a different carbon, renumbered MAP": (
        "V{+3} | L=[ SMILES:Fc1c(F)c(F)c([B-](c2cc[cH-]c2)(c2c(F)c(F)c(F)c(F)c2F)c2c(F)c(F)c(F)c(F)c2F)c(F)c1F, SMILES:c1cc[cH-]c1 ] | MAP:{ 1:[9,10,11,12,13], 2:[1,2,3,4,5] }",
        "V{+3} | L=[ SMILES:Fc1c(F)c(F)c([B-](c2c(F)c(F)c(F)c(F)c2F)(c2c(F)c(F)c(F)c(F)c2F)[c-]2cccc2)c(F)c1F, SMILES:c1cc[cH-]c1 ] | MAP:{ 1:[31,32,33,34,35], 2:[1,2,3,4,5] }"),
    "nitrate bound through the other, equivalent O-": (
        "Zn{+2} | L=[ SMILES:c1cc2nc(c1)COc1ccccc1CNCCNCCNCc1ccccc1OC2, SMILES:O=[N+]([O-])[O-], SMILES:O=[N+]([O-])[O-] ] | MAP:{ (1:16, 2:3), (1:19, 3:3), (1:22, 2:4) } | G:O",
        "Zn{+2} | L=[ SMILES:c1cc2nc(c1)COc1ccccc1CNCCNCCNCc1ccccc1OC2, SMILES:O=[N+]([O-])[O-], SMILES:O=[N+]([O-])[O-] ] | MAP:{ (1:16, 2:3), (1:19, 3:4), (1:22, 2:4) } | G:O"),
}

TRANS_PT = "Pt{+2} | L=[ SMILES:N, SMILES:N, SMILES:[Cl-], SMILES:[Cl-] ] | MAP:{ (1:1, 2:1), (3:1, 4:1) }"
CIS_PT = "Pt{+2} | L=[ SMILES:N, SMILES:N, SMILES:[Cl-], SMILES:[Cl-] ] | MAP:{ (1:1, 3:1), (2:1, 4:1) }"
CHIRAL_CO = ("Co{+3} | L=[ SMILES:NCCNCCCN(CCN)C(N)(C(=O)[O-])C(=O)[O-] ] | "
             "MAP:{ (1:1, 1:8), (1:4, 1:16), (1:11, 1:19) } | G:O | X:@@")


def test_identical():
    assert compare_trex(TRANS_PT, TRANS_PT).level == IDENTICAL


def test_lewis_structure_differences_still_valid():
    for name, (ref, got) in RESONANCE.items():
        m = compare_trex(ref, got)
        assert m.level == CHIRALITY, (name, m.tier, m.detail)
        assert m.valid


def test_trans_map_differs():
    m = compare_trex(TRANS_PT, CIS_PT)
    assert m.level == CONNECTIVITY and not m.valid, m.detail


def test_connectivity_differs():
    s_bound = "Ag{+1} | L=[ SMILES:[S-]C#N, SMILES:[S-]C#N ] | MAP:{ (1:1, 2:1) }"
    n_bound = "Ag{+1} | L=[ SMILES:[S-]C#N, SMILES:[S-]C#N ] | MAP:{ (1:1, 2:3) }"
    assert compare_trex(s_bound, n_bound).level == BUILT
    assert compare_trex(s_bound, None).level == BUILT


def test_geometry_hint_and_charge():
    tp = ("Hf{+4} | L=[ SMILES:[Cl-], SMILES:[Cl-], SMILES:[Cl-], SMILES:[Cl-], SMILES:[Cl-], SMILES:[Cl-] ] | "
          "MAP:{ (1:1, 2:1), (3:1, 4:1), (5:1, 6:1) } | G:TP")
    assert geometry_hint(tp) == "trigonal prismatic"
    assert geometry_hint(CHIRAL_CO) == "octahedral"
    assert total_charge(tp) == -2
    from trex.parse_full import parse_trex_full
    from trex_geo.trex_to_xyz import get_geometry_full_name
    assert get_geometry_full_name(parse_trex_full(tp)) == "6_trigonal_prismatic"


def _mirror(mol):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (-p.x, p.y, p.z))
    return m


def test_chirality_from_3d(tmp_path=None):
    from trex.parse_full import parse_trex_full
    from trex_geo import compute_idealized_internals, reconstruct_coordination_sphere
    from trex_geo.native_embed import embed_trex_native

    t = parse_trex_full(CHIRAL_CO)
    coords, _ = reconstruct_coordination_sphere(t=t, **compute_idealized_internals(t))
    r = embed_trex_native(t, coords, max_attempts=10, seed=0, haptic_scale=1.0)
    assert r.success
    work = str(tmp_path) if tmp_path is not None else None
    built = match_mol(r.mol, CHIRAL_CO, work_dir=work and os.path.join(work, "a"))
    assert built.level == IDENTICAL, (built.tier, built.detail, built.readback)
    mirrored = match_mol(_mirror(r.mol), CHIRAL_CO, work_dir=work and os.path.join(work, "b"))
    assert mirrored.level == TRANS_MAP and mirrored.detail == "wrong chirality", (mirrored.tier, mirrored.detail)


class _Fake:
    def __init__(self, success, level=None, energy=0.0):
        self.success, self.energy = success, energy
        self.diagnostics = None if not success else {
            "trex_level": level, "trex_valid": None if level is None else level >= CHIRALITY}
        self.error = None if success else "failed"


def _runner(outcomes, calls):
    def run():
        method = os.environ["TREX_EMBED_METHOD"]
        calls.append(method)
        return outcomes[method]()
    return run


def test_auto_fallback():
    from trex_geo.embed_choice import run_with_embed_method

    calls = []  # native valid: no second build
    r = run_with_embed_method(_runner({"native": lambda: _Fake(True, IDENTICAL)}, calls), "auto", True)
    assert calls == ["native"] and r.diagnostics["auto_fallback"] is False

    calls = []  # native fails to build: MetalloGen result kept
    r = run_with_embed_method(_runner({"native": lambda: _Fake(False),
                                       "metallogen": lambda: _Fake(True, TRANS_MAP)}, calls), "auto", True)
    assert calls == ["native", "metallogen"] and r.success and r.diagnostics["trex_level"] == TRANS_MAP

    calls = []  # both built, neither valid: the closer tier wins
    r = run_with_embed_method(_runner({"native": lambda: _Fake(True, BUILT, -1.0),
                                       "metallogen": lambda: _Fake(True, CONNECTIVITY, 0.0)}, calls), "auto", True)
    assert r.diagnostics["trex_level"] == CONNECTIVITY and r.diagnostics["auto_fallback"] is True

    calls = []  # same tier: the lower energy wins
    r = run_with_embed_method(_runner({"native": lambda: _Fake(True, BUILT, 0.0),
                                       "metallogen": lambda: _Fake(True, BUILT, -0.5)}, calls), "auto", True)
    assert r.energy == -0.5

    calls = []  # explicit method, and auto without validation: one build
    run_with_embed_method(_runner({"metallogen": lambda: _Fake(True, BUILT)}, calls), "metallogen", True)
    run_with_embed_method(_runner({"auto": lambda: _Fake(True, None)}, calls), "auto", False)
    assert calls == ["metallogen", "auto"]
    assert os.environ.get("TREX_EMBED_METHOD") is None
