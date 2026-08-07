"""Parser and environment-reporting tests for the 3-CERATOPS CLI."""

from pathlib import Path


def test_generate_parser_accepts_file_input():
    from trex_geo.cli import build_parser

    args = build_parser().parse_args(
        [
            "generate",
            "--trex-file",
            "complex.trex",
            "--output",
            "complex.xyz",
            "--optimizer",
            "uma",
            "--internals",
            "idealized",
            "--n-seeds",
            "4",
            "--top-k",
            "2",
        ]
    )
    assert args.optimizer == "uma"
    assert args.internals == "idealized"
    assert args.n_seeds == 4
    assert args.top_k == 2


def test_doctor_reports_missing_optional_stacks(monkeypatch, tmp_path, capsys):
    from trex_geo import cli

    monkeypatch.setattr(
        cli,
        "_module_available",
        lambda name: name in {"numpy", "rdkit", "pulp", "scipy", "trex"},
    )
    args = cli.build_parser().parse_args(
        ["doctor", "--model-dir", str(tmp_path / "missing-models")]
    )
    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert "Core:" in output
    assert "xTB:" in output
    assert "ML internals:" in output
    assert "UMA:" in output
    assert "Model ensemble:" in output
    assert "MISSING" in output


def test_checkpoint_discovery(tmp_path):
    from trex_geo.cli import _find_checkpoints

    checkpoint = tmp_path / "seed_1" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test")
    assert _find_checkpoints(tmp_path) == [checkpoint]


def test_assemble_parser_accepts_operational_overrides():
    from trex_geo.cli import build_parser

    args = build_parser().parse_args([
        "assemble", "space.yaml", "--enumerate-only", "--output", "results",
        "--max-stoichiometries", "50", "--max-assignments-per-stoichiometry", "200",
    ])
    assert args.config == "space.yaml"
    assert args.enumerate_only is True
    assert args.max_stoichiometries == 50


def test_assemble_cli_enumeration_only(monkeypatch, tmp_path, capsys):
    from trex_geo import cli
    from trex_geo.assemble import AssembledComplex
    from trex_geo.assemble.config import AssembleConfig
    from trex_geo.assemble.models import AssemblySpace, LigandSpec

    config = AssembleConfig(AssemblySpace(
        metal="Pd", oxidation_state=2,
        ligands=(LigandSpec("[Cl-]", min_count=2, max_count=2),),
        geometries=("linear",),
    ), output=str(tmp_path))
    monkeypatch.setattr("trex_geo.assemble.config.load_config", lambda path: config)
    monkeypatch.setattr(
        "trex_geo.assemble.assemble_trex",
        lambda *args, **kwargs: [AssembledComplex("canonical", "linear", (2,), (0, 0))],
    )
    args = cli.build_parser().parse_args(["assemble", "space.yaml", "--enumerate-only"])
    assert args.func(args) == 0
    assert (tmp_path / "assembled.trex").read_text() == "canonical\n"
    assert "Enumerated 1" in capsys.readouterr().out
