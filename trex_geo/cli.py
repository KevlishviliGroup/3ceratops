"""Command-line interface for 3-CERATOPS."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
from pathlib import Path
from typing import Optional, Sequence


LOGGER = logging.getLogger("3ceratops")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "LF_GNN" / "ensemble_gram"


def _find_checkpoints(model_dir: Path) -> list[Path]:
    checkpoints = sorted(model_dir.glob("seed_*/best.pt"))
    if not checkpoints:
        checkpoints = sorted(model_dir.glob("*.pt"))
    return checkpoints


def _read_trex(args: argparse.Namespace) -> str:
    if args.trex is not None:
        trex_str = args.trex.strip()
    else:
        path = Path(args.trex_file)
        try:
            trex_str = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"could not read T-REX file '{path}': {exc}") from exc

    if not trex_str:
        raise ValueError("the T-REX input is empty")
    return trex_str


def _configure_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _run_generate(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    try:
        trex_str = _read_trex(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    if args.top_k is not None and args.n_seeds is not None and args.top_k > args.n_seeds:
        LOGGER.error("--top-k cannot exceed --n-seeds")
        return 2

    if args.optimizer == "xtb" and args.device is not None:
        LOGGER.error("--device applies only to the UMA optimizer")
        return 2

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.error("could not create output directory: %s", exc)
        return 2

    use_idealized = args.internals == "idealized"
    model_path = args.model_dir

    if args.optimizer == "xtb":
        from .trex_to_xyz import trex_to_xyz_file

        n_seeds = args.n_seeds if args.n_seeds is not None else 5
        top_k = args.top_k if args.top_k is not None else min(3, n_seeds)
        success = trex_to_xyz_file(
            trex_str,
            str(output_path),
            use_idealized=use_idealized,
            model_path=model_path,
            n_seeds=n_seeds,
            top_k=top_k,
            refine_gfn2=args.refine_gfn2,
            haptic_scale=args.haptic_scale,
            work_dir=args.work_dir,
        )
    else:
        from .trex_to_xyz_uma import trex_to_xyz_uma_file

        n_seeds = args.n_seeds if args.n_seeds is not None else 20
        top_k = args.top_k if args.top_k is not None else min(5, n_seeds)
        success = trex_to_xyz_uma_file(
            trex_str,
            str(output_path),
            use_idealized=use_idealized,
            model_path=model_path,
            n_seeds=n_seeds,
            top_k=top_k,
            haptic_scale=args.haptic_scale,
            refine_medium=args.refine_medium,
            refine_unconstrained=args.refine_unconstrained,
            device=args.device,
            work_dir=args.work_dir,
        )

    if not success:
        LOGGER.error("structure generation failed")
        return 1

    print(f"Wrote {output_path}")
    return 0


def _status_line(label: str, available: bool) -> str:
    return f"  {'OK' if available else 'MISSING':7s} {label}"


def _run_doctor(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)

    groups = {
        "Core": [
            ("NumPy", "numpy"),
            ("RDKit", "rdkit"),
            ("PuLP", "pulp"),
            ("SciPy", "scipy"),
            ("T-REX", "trex"),
        ],
        "xTB": [("ASE", "ase"), ("xtb-python", "xtb")],
        "ML internals": [
            ("PyTorch", "torch"),
            ("PyTorch Geometric", "torch_geometric"),
            ("torch-scatter", "torch_scatter"),
        ],
        "UMA": [("ASE", "ase"), ("fairchem-core", "fairchem")],
    }

    missing_core = False
    for group, dependencies in groups.items():
        print(f"{group}:")
        for label, module_name in dependencies:
            available = _module_available(module_name)
            print(_status_line(label, available))
            if group == "Core" and not available:
                missing_core = True

    model_dir = Path(args.model_dir) if args.model_dir else _default_model_dir()
    checkpoints = _find_checkpoints(model_dir) if model_dir.is_dir() else []
    print("Model ensemble:")
    print(_status_line(f"{model_dir} ({len(checkpoints)} checkpoints)", bool(checkpoints)))

    return 1 if missing_core else 0


def _run_assemble(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    try:
        from .assemble import assemble_trex, generate_assembled
        from .assemble.config import load_config

        config = load_config(args.config)
        structures = assemble_trex(
            config.space,
            constraints=config.constraints,
            ligand_index=config.ligand_index(),
            max_stoichiometries=(
                args.max_stoichiometries
                if args.max_stoichiometries is not None
                else config.max_stoichiometries
            ),
            max_assignments_per_stoichiometry=(
                args.max_assignments_per_stoichiometry
                if args.max_assignments_per_stoichiometry is not None
                else config.max_assignments_per_stoichiometry
            ),
        )
        output = Path(args.output or config.output)
        output.mkdir(parents=True, exist_ok=True)
        topology_file = output / "assembled.trex"
        topology_file.write_text(
            "\n".join(item.trex for item in structures) + ("\n" if structures else ""),
            encoding="utf-8",
        )
        print(f"Enumerated {len(structures)} unique canonical T-REX structures")
        print(f"Wrote {topology_file}")
        if args.enumerate_only:
            return 0
        optimizer = args.optimizer or config.optimizer
        batch = generate_assembled(
            structures,
            output,
            optimizer=optimizer,
            overwrite=args.overwrite,
            generator_kwargs=config.generator_kwargs,
        )
        print(f"Generated {batch.succeeded}; failed {batch.failed}")
        print(f"Wrote {batch.manifest_path}")
        return 1 if batch.failed else 0
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="3ceratops",
        description="Generate 3D transition-metal structures from T-REX topology.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one XYZ structure")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--trex-file", help="file containing one T-REX string")
    source.add_argument("--trex", help="quoted inline T-REX string")
    generate.add_argument("--output", required=True, help="output XYZ path")
    generate.add_argument("--optimizer", choices=("xtb", "uma"), default="xtb")
    generate.add_argument(
        "--internals",
        choices=("predicted", "idealized"),
        default="predicted",
        help="internal-coordinate source; prediction falls back to idealized values",
    )
    generate.add_argument("--model-dir", help="ensemble directory containing seed_*/best.pt")
    generate.add_argument("--n-seeds", type=_positive_int)
    generate.add_argument("--top-k", type=_positive_int)
    generate.add_argument("--haptic-scale", type=_positive_float, default=0.35)
    generate.add_argument("--work-dir")
    generate.add_argument("--device", choices=("cpu", "cuda"))
    generate.add_argument(
        "--refine-gfn2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable final GFN2-xTB refinement",
    )
    generate.add_argument(
        "--refine-medium",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable constrained UMA-medium refinement",
    )
    generate.add_argument(
        "--refine-unconstrained",
        action="store_true",
        help="run final unconstrained UMA-medium relaxation",
    )
    generate.add_argument("-v", "--verbose", action="count", default=0)
    generate.set_defaults(func=_run_generate)

    assemble = subparsers.add_parser(
        "assemble", help="enumerate a complex space and optionally generate all structures"
    )
    assemble.add_argument("config", help="YAML or JSON assembly configuration")
    assemble.add_argument(
        "--enumerate-only",
        action="store_true",
        help="write canonical T-REX structures without 3D generation",
    )
    assemble.add_argument("--output", help="override generation.output")
    assemble.add_argument("--optimizer", choices=("xtb", "uma"), help="override generation.optimizer")
    assemble.add_argument("--overwrite", action="store_true", help="regenerate completed candidates")
    assemble.add_argument("--max-stoichiometries", type=_positive_int)
    assemble.add_argument("--max-assignments-per-stoichiometry", type=_positive_int)
    assemble.add_argument("-v", "--verbose", action="count", default=0)
    assemble.set_defaults(func=_run_assemble)

    doctor = subparsers.add_parser("doctor", help="check dependencies and model checkpoints")
    doctor.add_argument("--model-dir", help="ensemble directory containing seed_*/best.pt")
    doctor.add_argument("-v", "--verbose", action="count", default=0)
    doctor.set_defaults(func=_run_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
