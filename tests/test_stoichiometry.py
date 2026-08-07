import json

import pytest
from rdkit import rdBase

from trex_geo.assemble import AssemblySpace, LigandIndex, LigandSpec, enumerate_stoichiometries
from trex_geo.assemble.canonicalization import CANONICALIZATION


def _index(tmp_path):
    packaged = tmp_path / "index.json"
    packaged.write_text(json.dumps({
        "metadata": {
            "schema_version": 1,
            "rdkit_version": rdBase.rdkitVersion,
            "canonicalization": CANONICALIZATION,
        },
        "ligands": {
            "NCCN": {
                "canonical_smiles": "NCCN",
                "binding_modes": [
                    {
                        "sites": [[1], [4]],
                        "internal_trans_pairs": [],
                        "geometry_class": "cis_bidentate",
                        "observations": 10,
                    },
                    {
                        "sites": [[1]],
                        "internal_trans_pairs": [],
                        "geometry_class": "monodentate",
                        "observations": 2,
                    },
                ],
            },
            "[Cl-]": {
                "canonical_smiles": "[Cl-]",
                "binding_modes": [{
                    "sites": [[1]],
                    "internal_trans_pairs": [],
                    "geometry_class": "monodentate",
                    "observations": 100,
                }],
            },
        },
    }))
    return LigandIndex(packaged, tmp_path / "overrides.json")


def test_dominant_modes_enumerate_exact_coordination_number(tmp_path):
    space = AssemblySpace(
        metal="Pd",
        oxidation_state=2,
        ligands=(LigandSpec("NCCN", 2), LigandSpec("[Cl-]", 4)),
        geometries=("square planar",),
        spin=0,
    )
    candidates = enumerate_stoichiometries(space, ligand_index=_index(tmp_path))
    assert [candidate.stoichiometry for candidate in candidates] == [
        (0, 4), (1, 2), (2, 0)
    ]
    assert all(candidate.coordination_number == 4 for candidate in candidates)


def test_all_observed_treats_modes_as_unordered_multiset(tmp_path):
    space = AssemblySpace(
        metal="Pd",
        oxidation_state=2,
        ligands=(
            LigandSpec("NCCN", 2, binding_modes="all_observed"),
            LigandSpec("[Cl-]", 4),
        ),
        geometries=("square_planar",),
    )
    candidates = enumerate_stoichiometries(space, ligand_index=_index(tmp_path))
    assert len(candidates) == 6
    mixed = [
        candidate for candidate in candidates
        if [item.mode.mode_index for item in candidate.ligand_instances if item.spec_index == 0]
        == [0, 1]
    ]
    assert len(mixed) == 1
    assert mixed[0].stoichiometry == (2, 1)


def test_trigonal_prismatic_is_rejected(tmp_path):
    space = AssemblySpace(
        metal="Pd",
        oxidation_state=2,
        ligands=(LigandSpec("[Cl-]", 6),),
        geometries=("trigonal_prismatic",),
    )
    with pytest.raises(ValueError, match="deliberately unsupported"):
        enumerate_stoichiometries(space, ligand_index=_index(tmp_path))


def test_candidate_guard_stops_combinatorial_expansion(tmp_path):
    space = AssemblySpace(
        metal="Pd",
        oxidation_state=2,
        ligands=(LigandSpec("NCCN", 2), LigandSpec("[Cl-]", 4)),
        geometries=("square_planar",),
    )
    with pytest.raises(RuntimeError, match="max_candidates=2"):
        enumerate_stoichiometries(space, ligand_index=_index(tmp_path), max_candidates=2)
