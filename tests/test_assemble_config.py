import json

from trex_geo.assemble.config import config_from_mapping, load_config


def _payload():
    return {
        "metal": "Pd",
        "oxidation_state": 2,
        "spin": 0,
        "geometries": ["square_planar"],
        "ligands": [
            {"name": "ethylenediamine", "identifier": "en", "min_count": 1, "max_count": 1},
            {"smiles": "[Cl-]", "identifier": "chloride", "min_count": 2, "max_count": 2},
        ],
        "constraints": {
            "required_trans": [[
                {"ligand": "en", "copy": 0, "atoms": [1]},
                {"ligand": "chloride", "copy": 0, "atoms": [1]},
            ]]
        },
        "limits": {"max_stoichiometries": 20},
        "generation": {"output": "pd_space", "optimizer": "xtb", "options": {"n_seeds": 10}},
    }


def test_config_mapping_preserves_trex_site_conventions():
    config = config_from_mapping(_payload())
    assert config.space.ligands[0].smiles == "ethylenediamine"
    assert config.constraints.required_trans[0][0].atoms == (1,)
    assert config.constraints.required_trans[0][1].ligand == "chloride"
    assert config.generator_kwargs == {"n_seeds": 10}


def test_load_json_config(tmp_path):
    path = tmp_path / "assembly.json"
    path.write_text(json.dumps(_payload()))
    assert load_config(path).space.metal == "Pd"
