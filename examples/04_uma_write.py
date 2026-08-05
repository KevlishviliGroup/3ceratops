#!/usr/bin/env python3
"""Generate and write an XYZ structure with the UMA pipeline."""

from pathlib import Path

from trex_geo import trex_to_xyz_uma_file


TREX = (
    "Co{+3}|"
    "L=[SMILES:N,SMILES:N,SMILES:N,SMILES:N,SMILES:N,SMILES:N]|"
    "MAP:{(1:1,2:1),(3:1,4:1),(5:1,6:1)}|"
    "G:O"
)


def main() -> None:
    output = Path("example_uma.xyz")
    success = trex_to_xyz_uma_file(
        TREX,
        str(output),
        use_idealized=True,
        n_seeds=2,
        top_k=1,
        refine_medium=True,
        refine_unconstrained=False,
    )

    if not success:
        raise RuntimeError("UMA structure generation failed")

    print(f"written: {output.resolve()}")


if __name__ == "__main__":
    main()
