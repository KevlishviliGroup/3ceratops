#!/usr/bin/env python3
"""Use the LF-GNN ensemble and a multi-seed xTB conformer search."""

from pathlib import Path

from trex_geo import trex_to_xyz


TREX = (
    "Zn{+2}|"
    "L=[SMILES:Cl,SMILES:Cl,SMILES:N,SMILES:N]|"
    "MAP:{;1:1,2:1,3:1,4:1}|"
    "G:T"
)


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "LF_GNN" / "ensemble_gram"

    result = trex_to_xyz(
        TREX,
        model_path=str(model_dir),
        n_seeds=5,
        top_k=3,
        refine_gfn2=False,
    )

    if not result.success:
        raise RuntimeError(result.error)

    print(f"internals: {result.internals_source}")
    print(f"energy: {result.energy:.6f} Eh")
    print(f"diagnostics: {result.diagnostics}")

    if result.internals_source != "predicted":
        print("warning: prediction was unavailable; idealized fallback was used")


if __name__ == "__main__":
    main()
