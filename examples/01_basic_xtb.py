#!/usr/bin/env python3
"""Basic idealized 3-CERATOPS generation with the xTB pipeline."""

from trex_geo import trex_to_xyz


TREX = (
    "Fe{+2}|"
    "L=[SMILES:O,SMILES:O,SMILES:O,SMILES:O,SMILES:O,SMILES:O]|"
    "MAP:{(1:1,2:1),(3:1,4:1),(5:1,6:1)}|"
    "G:O"
)


def main() -> None:
    result = trex_to_xyz(
        TREX,
        use_idealized=True,
        n_seeds=1,
        refine_gfn2=False,
    )

    if not result.success:
        raise RuntimeError(result.error)

    print(f"atoms: {result.mol.GetNumAtoms()}")
    print(f"energy: {result.energy:.6f} Eh")
    print(f"internals: {result.internals_source}")
    print(f"diagnostics: {result.diagnostics}")


if __name__ == "__main__":
    main()
