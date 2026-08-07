import json
from pathlib import Path
from types import SimpleNamespace

from trex_geo.assemble import (
    AssembledComplex,
    candidate_directory_name,
    candidate_id,
    generate_assembled,
)


class _Position:
    x = 1.0
    y = 2.0
    z = 3.0


class _Conformer:
    def GetAtomPosition(self, index):
        return _Position()


class _Atom:
    def GetSymbol(self):
        return "Pd"


class _Mol:
    def GetNumAtoms(self):
        return 1

    def GetConformer(self):
        return _Conformer()

    def GetAtomWithIdx(self, index):
        return _Atom()


def _complex(trex="Pd{+2}.LIGANDS{[Cl-]}.MAP{1:1}.X:@"):
    return AssembledComplex(trex, "linear", (1,), (0,))


def test_batch_generation_writes_xyz_and_manifest(tmp_path):
    calls = []

    def generator(trex, **kwargs):
        calls.append((trex, kwargs))
        return SimpleNamespace(
            success=True, mol=_Mol(), energy=-1.25,
            internals_source="idealized", diagnostics={"best_seed": 2}, error=None,
        )

    result = generate_assembled(
        [_complex(), _complex()], tmp_path, _generator=generator,
        generator_kwargs={"model_path": Path("models")},
    )
    assert len(calls) == 1
    assert result.succeeded == 1
    candidate_dir = tmp_path / candidate_directory_name(_complex().trex)
    xyz = candidate_dir / "structure.xyz"
    assert xyz.is_file()
    assert (candidate_dir / "topology.trex").read_text().strip() == _complex().trex
    assert json.loads((candidate_dir / "generation.json").read_text())["status"] == "success"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["summary"] == {"failed": 0, "success": 1, "total": 1}
    assert manifest["generator_kwargs"]["model_path"] == "models"
    assert manifest["candidates"][0]["xyz_file"] == str(xyz.relative_to(tmp_path))


def test_candidate_directory_is_readable_safe_and_stable():
    trex = "Pd{+2}.LIGANDS{P(\\c1ccccc1)(/c1ccccc1)c1ccccc1}.MAP{1:1}.X:@"
    name = candidate_directory_name(trex)
    assert name.startswith("Pd_+2_LIGANDS_P_")
    assert "/" not in name and "\\" not in name
    assert name.endswith(candidate_id(trex)[:8])
    assert candidate_directory_name(trex) == name


def test_batch_isolates_failure_and_resumes_success(tmp_path):
    structures = [_complex("first"), _complex("second")]

    def first_run(trex, **kwargs):
        if trex == "second":
            raise RuntimeError("boom")
        return SimpleNamespace(
            success=True, mol=_Mol(), energy=None,
            internals_source="idealized", diagnostics={}, error=None,
        )

    initial = generate_assembled(structures, tmp_path, _generator=first_run)
    assert initial.failed == 1

    retried = []

    def second_run(trex, **kwargs):
        retried.append(trex)
        return SimpleNamespace(
            success=True, mol=_Mol(), energy=None,
            internals_source="idealized", diagnostics={}, error=None,
        )

    resumed = generate_assembled(structures, tmp_path, _generator=second_run)
    assert retried == ["second"]
    assert [record.status for record in resumed.records] == ["skipped", "success"]
    assert resumed.failed == 0
