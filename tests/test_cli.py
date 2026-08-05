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
