import json

import pytest
from rdkit import rdBase

from trex_geo.assemble import BindingModeInput, LigandIndex, LigandNotFoundError


def test_lookup_canonicalizes_equivalent_smiles(tmp_path):
    packaged = tmp_path / "index.json"
    packaged.write_text(json.dumps({
        "metadata": {
            "schema_version": 1,
            "rdkit_version": rdBase.rdkitVersion,
            "canonicalization": {
                "engine": "RDKit", "isomeric_smiles": True, "kekule_smiles": False,
                "remove_hydrogens": True, "index_base": 1,
                "index_meaning": "heavy-atom position in canonical SMILES writing order",
            },
        },
        "ligands": {"CCO": {"canonical_smiles": "CCO", "binding_modes": []}},
    }))
    index = LigandIndex(packaged, tmp_path / "overrides.json")
    assert index.lookup("OCC")["canonical_smiles"] == "CCO"


def test_resolve_registers_missing_ligand_without_persisting(tmp_path):
    packaged = tmp_path / "index.json"
    packaged.write_text(json.dumps({
        "metadata": {
            "schema_version": 1,
            "rdkit_version": rdBase.rdkitVersion,
            "canonicalization": {
                "engine": "RDKit", "isomeric_smiles": True, "kekule_smiles": False,
                "remove_hydrogens": True, "index_base": 1,
                "index_meaning": "heavy-atom position in canonical SMILES writing order",
            },
        }, "ligands": {},
    }))
    index = LigandIndex(packaged, tmp_path / "overrides.json")
    record = index.resolve(
        "NCCN", if_missing=BindingModeInput(sites=((1,), (4,))), persist=False
    )
    assert record["binding_modes"][0]["geometry_class"] == "cis_bidentate"
    assert not (tmp_path / "overrides.json").exists()
    with pytest.raises(LigandNotFoundError):
        LigandIndex(packaged, tmp_path / "other.json").lookup("NCCN")
