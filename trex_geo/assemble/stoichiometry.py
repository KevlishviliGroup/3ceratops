"""Enumerate ligand counts and binding-mode multisets for target geometries."""

from __future__ import annotations

from itertools import combinations_with_replacement

from .geometry_registry import get_geometry
from .ligand_index import LigandIndex
from .models import (
    AssemblySpace,
    LigandInstance,
    LigandSpec,
    ResolvedBindingMode,
    StoichiometricCandidate,
)
from .name_resolution import resolve_ligand_input


def _resolve_modes(spec: LigandSpec, index: LigandIndex) -> tuple[str, tuple[ResolvedBindingMode, ...]]:
    smiles = resolve_ligand_input(spec.smiles)
    record = index.resolve(
        smiles,
        if_missing=spec.if_missing,
        persist=spec.persist_if_missing,
    )
    raw_modes = record.get("binding_modes", [])
    if not raw_modes:
        raise ValueError(f"Ligand {record['canonical_smiles']!r} has no binding modes")

    if spec.binding_modes == "dominant":
        selected = (0,)
    elif spec.binding_modes == "all_observed":
        selected = tuple(range(len(raw_modes)))
    else:
        selected = tuple(dict.fromkeys(spec.binding_modes))

    modes = []
    for mode_index in selected:
        if mode_index < 0 or mode_index >= len(raw_modes):
            raise IndexError(
                f"Binding-mode index {mode_index} is out of range for "
                f"{record['canonical_smiles']!r} ({len(raw_modes)} modes)"
            )
        raw = raw_modes[mode_index]
        modes.append(ResolvedBindingMode(
            mode_index=mode_index,
            sites=tuple(tuple(int(atom) for atom in site) for site in raw["sites"]),
            internal_trans_pairs=tuple(
                tuple(int(position) for position in pair)
                for pair in raw.get("internal_trans_pairs", [])
            ),
            geometry_class=raw.get("geometry_class", "explicit"),
            observations=int(raw.get("observations", 0)),
        ))
    return record["canonical_smiles"], tuple(modes)


def enumerate_stoichiometries(
    space: AssemblySpace,
    *,
    ligand_index: LigandIndex | None = None,
    max_candidates: int | None = None,
) -> list[StoichiometricCandidate]:
    """Enumerate exact-CN ligand/mode combinations; no MAP isomers are created yet."""
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")
    index = ligand_index or LigandIndex()
    resolved = tuple(_resolve_modes(spec, index) for spec in space.ligands)
    canonical_keys = [canonical for canonical, _ in resolved]
    if len(canonical_keys) != len(set(canonical_keys)):
        raise ValueError("Each canonical ligand may appear only once; use min_count/max_count")

    results: list[StoichiometricCandidate] = []
    for geometry_name in space.geometries:
        geometry = get_geometry(geometry_name)

        def visit(spec_index: int, instances: tuple[LigandInstance, ...], sites_used: int) -> None:
            if sites_used > geometry.coordination_number:
                return
            if spec_index == len(space.ligands):
                if sites_used != geometry.coordination_number or not instances:
                    return
                fixed_pairs = sum(len(item.mode.internal_trans_pairs) for item in instances)
                if fixed_pairs > geometry.trans_pairs:
                    return
                results.append(StoichiometricCandidate(
                    geometry=geometry.name,
                    coordination_number=geometry.coordination_number,
                    ligand_instances=instances,
                    stoichiometry=tuple(
                        sum(item.spec_index == i for item in instances)
                        for i in range(len(space.ligands))
                    ),
                ))
                if max_candidates is not None and len(results) > max_candidates:
                    raise RuntimeError(
                        f"Assembly space exceeds max_candidates={max_candidates}; "
                        "reduce ligand counts, modes, or geometries"
                    )
                return

            spec = space.ligands[spec_index]
            canonical, modes = resolved[spec_index]
            for count in range(spec.min_count, spec.max_count + 1):
                for mode_indices in combinations_with_replacement(range(len(modes)), count):
                    added = tuple(
                        LigandInstance(spec_index, canonical, modes[mode_index])
                        for mode_index in mode_indices
                    )
                    added_sites = sum(item.mode.denticity for item in added)
                    visit(spec_index + 1, instances + added, sites_used + added_sites)

        visit(0, (), 0)
    return results
