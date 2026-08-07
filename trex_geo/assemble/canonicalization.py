"""One strict SMILES canonicalization path shared by index build and lookup."""

from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem, rdBase
from trex.smiles_map_utils import remap_indices_from_smiles

CANONICALIZATION = {
    "engine": "RDKit",
    "isomeric_smiles": True,
    "kekule_smiles": False,
    "remove_hydrogens": True,
    "index_base": 1,
    "index_meaning": "heavy-atom position in canonical SMILES writing order",
}


def rdkit_version() -> str:
    return rdBase.rdkitVersion


def canonicalize_smiles(smiles: str) -> str:
    """Return the exact canonical ligand key used by the catalog."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse ligand SMILES: {smiles!r}")
    mol = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
        allHsExplicit=False,
    )
    check = Chem.MolFromSmiles(canonical)
    if check is None:
        raise ValueError(f"RDKit generated an unreadable canonical SMILES: {canonical!r}")
    round_trip = Chem.MolToSmiles(
        Chem.RemoveHs(check, sanitize=True),
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=False,
        allHsExplicit=False,
    )
    if round_trip != canonical:
        raise ValueError(f"Unstable RDKit canonicalization for {smiles!r}")
    return canonical


def canonicalize_and_remap_sites(
    smiles: str, sites: Iterable[Iterable[int]]
) -> tuple[str, tuple[tuple[int, ...], ...]]:
    """Canonicalize and remap 1-based sites supplied in input-SMILES order."""
    normalized = tuple(tuple(int(atom) for atom in site) for site in sites)
    if any(atom < 1 for site in normalized for atom in site):
        raise ValueError("Ligand atom indices are 1-based and must be positive")
    flat = [atom - 1 for site in normalized for atom in site]
    canonical, mapped = remap_indices_from_smiles(
        smiles,
        flat,
        mode="smiles",
        isomericSmiles=True,
        kekuleSmiles=False,
    )
    if canonical != canonicalize_smiles(smiles):
        raise ValueError("T-REX and runtime RDKit canonicalization disagree")
    iterator = iter(atom + 1 for atom in mapped)
    remapped = tuple(tuple(next(iterator) for _ in site) for site in normalized)
    return canonical, remapped
