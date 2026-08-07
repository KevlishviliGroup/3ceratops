import json

import pytest
from rdkit import rdBase
from trex.parse_full import parse_trex_full

from trex_geo.assemble import (
    AssemblyConstraints,
    AssemblySpace,
    LigandIndex,
    LigandSpec,
    SiteRef,
    assemble_trex,
)
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
                "binding_modes": [{
                    "sites": [[1], [4]],
                    "internal_trans_pairs": [],
                    "geometry_class": "cis_bidentate",
                    "observations": 10,
                }],
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


def _space():
    return AssemblySpace(
        metal="Pd",
        oxidation_state=2,
        spin=0,
        ligands=(
            LigandSpec("NCCN", 1, min_count=1, identifier="en"),
            LigandSpec("[Cl-]", 2, min_count=2, identifier="chloride"),
        ),
        geometries=("square_planar",),
    )


def test_assemble_returns_parseable_unique_canonical_trex(tmp_path):
    structures = assemble_trex(_space(), ligand_index=_index(tmp_path))
    assert structures
    assert len({item.trex for item in structures}) == len(structures)
    assert all(parse_trex_full(item.trex).cn() == 4 for item in structures)


def test_required_trans_and_cis_use_native_trex_sites(tmp_path):
    constraints = AssemblyConstraints(
        required_trans=((
            SiteRef("en", 0, (1,)),
            SiteRef("chloride", 0, (1,)),
        ),),
        required_cis=((
            SiteRef("en", 0, (1,)),
            SiteRef("en", 0, (4,)),
        ),),
    )
    structures = assemble_trex(
        _space(), constraints=constraints, ligand_index=_index(tmp_path)
    )
    assert structures


def test_contradictory_constraints_fail(tmp_path):
    pair = (
        SiteRef("en", 0, (1,)),
        SiteRef("chloride", 0, (1,)),
    )
    with pytest.raises(ValueError, match="both cis and trans"):
        assemble_trex(
            _space(),
            constraints=AssemblyConstraints(required_trans=(pair,), required_cis=(pair,)),
            ligand_index=_index(tmp_path),
        )


def test_constraint_cannot_rewrite_internal_binding_mode(tmp_path):
    with pytest.raises(ValueError, match="selected binding mode"):
        assemble_trex(
            _space(),
            constraints=AssemblyConstraints(required_trans=((
                SiteRef("en", 0, (1,)),
                SiteRef("en", 0, (4,)),
            ),)),
            ligand_index=_index(tmp_path),
        )
