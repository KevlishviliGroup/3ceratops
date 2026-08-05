#!/usr/bin/env python3
"""Generate a square-planar complex from explicit internal coordinates."""

from trex_geo import trex_to_xyz


TREX = (
    "Pt{+2}|"
    "L=[SMILES:Cl,SMILES:Cl,SMILES:N,SMILES:N]|"
    "MAP:{(1:1,2:1),(3:1,4:1)}|"
    "G:SP"
)

INTERNALS = {
    "site_r": [2.30, 2.30, 2.05, 2.05],
    "trans_cos": [-0.99, -0.99],
    "cis_cos": [0.01, 0.01, -0.01, -0.01],
    "cis_pair_indices": [(0, 2), (0, 3), (1, 2), (1, 3)],
}


def main() -> None:
    result = trex_to_xyz(
        TREX,
        internals=INTERNALS,
        n_seeds=1,
        refine_gfn2=False,
    )

    if not result.success:
        raise RuntimeError(result.error)

    print(f"atoms: {result.mol.GetNumAtoms()}")
    print(f"energy: {result.energy:.6f} Eh")
    print(f"internals: {result.internals_source}")


if __name__ == "__main__":
    main()
