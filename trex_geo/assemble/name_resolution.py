"""Resolve exact common ligand names before strict SMILES canonicalization."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from rdkit import Chem


def _ligand2smiles_resolver() -> Callable[[str], str] | None:
    """Find the exact-name API without making Ligand2SMILES a hard dependency."""
    for module_name in ("ligand2smiles", "Ligand2SMILES"):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        for function_name in ("name_to_smiles", "ligand_to_smiles"):
            function = getattr(module, function_name, None)
            if callable(function):
                return function
    return None


def resolve_ligand_input(value: str) -> str:
    """Return SMILES, using only an exact Ligand2SMILES match when needed."""
    if Chem.MolFromSmiles(value) is not None:
        return value
    resolver = _ligand2smiles_resolver()
    if resolver is None:
        raise ValueError(
            f"{value!r} is not valid SMILES and Ligand2SMILES is not installed"
        )
    resolved = resolver(value)
    if not resolved or Chem.MolFromSmiles(str(resolved)) is None:
        raise ValueError(f"No exact Ligand2SMILES match for {value!r}")
    return str(resolved)
