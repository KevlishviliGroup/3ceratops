"""YAML/JSON configuration loading for the ``3ceratops assemble`` CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .ligand_index import BindingModeInput, LigandIndex
from .models import AssemblyConstraints, AssemblySpace, LigandSpec, SiteRef


@dataclass(frozen=True)
class AssembleConfig:
    space: AssemblySpace
    constraints: AssemblyConstraints = field(default_factory=AssemblyConstraints)
    output: str = "assembled"
    optimizer: str = "xtb"
    generator_kwargs: dict[str, Any] = field(default_factory=dict)
    max_stoichiometries: int | None = None
    max_assignments_per_stoichiometry: int | None = None
    override_path: str | None = None

    def ligand_index(self) -> LigandIndex:
        return LigandIndex(override_path=Path(self.override_path)) if self.override_path else LigandIndex()


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _site_ref(value: Any, location: str) -> SiteRef:
    item = _mapping(value, location)
    try:
        return SiteRef(
            ligand=str(item["ligand"]),
            copy=int(item.get("copy", 0)),
            atoms=tuple(int(atom) for atom in item["atoms"]),
        )
    except KeyError as exc:
        raise ValueError(f"{location} requires ligand and atoms") from exc


def _relations(values: Any, location: str) -> tuple[tuple[SiteRef, SiteRef], ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{location} must be a list")
    result = []
    for index, pair in enumerate(values):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{location}[{index}] must contain exactly two site mappings")
        result.append((
            _site_ref(pair[0], f"{location}[{index}][0]"),
            _site_ref(pair[1], f"{location}[{index}][1]"),
        ))
    return tuple(result)


def _binding_mode(value: Any, location: str) -> BindingModeInput | None:
    if value is None:
        return None
    item = _mapping(value, location)
    return BindingModeInput(
        sites=tuple(tuple(int(atom) for atom in site) for site in item["sites"]),
        trans_pairs=tuple(tuple(int(position) for position in pair) for pair in item.get("trans_pairs", [])),
        label=item.get("label"),
    )


def _ligand(value: Any, index: int) -> LigandSpec:
    item = _mapping(value, f"ligands[{index}]")
    ligand_input = item.get("smiles", item.get("name"))
    if ligand_input is None:
        raise ValueError(f"ligands[{index}] requires smiles or name")
    modes = item.get("binding_modes", "dominant")
    if isinstance(modes, list):
        modes = tuple(int(mode) for mode in modes)
    return LigandSpec(
        smiles=str(ligand_input),
        min_count=int(item.get("min_count", 0)),
        max_count=int(item["max_count"]),
        binding_modes=modes,
        if_missing=_binding_mode(item.get("if_missing"), f"ligands[{index}].if_missing"),
        persist_if_missing=bool(item.get("persist_if_missing", True)),
        identifier=item.get("identifier"),
    )


def config_from_mapping(payload: dict[str, Any]) -> AssembleConfig:
    root = _mapping(payload, "config")
    ligands = root.get("ligands")
    if not isinstance(ligands, list) or not ligands:
        raise ValueError("ligands must be a nonempty list")
    geometries = root.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("geometries must be a nonempty list")
    constraints = _mapping(root.get("constraints", {}), "constraints")
    generation = _mapping(root.get("generation", {}), "generation")
    limits = _mapping(root.get("limits", {}), "limits")
    optimizer = str(generation.get("optimizer", "xtb"))
    if optimizer not in {"xtb", "uma"}:
        raise ValueError("generation.optimizer must be 'xtb' or 'uma'")
    max_stoichiometries = limits.get("max_stoichiometries")
    max_assignments = limits.get("max_assignments_per_stoichiometry")
    if max_stoichiometries is not None:
        max_stoichiometries = int(max_stoichiometries)
        if max_stoichiometries < 1:
            raise ValueError("limits.max_stoichiometries must be at least 1")
    if max_assignments is not None:
        max_assignments = int(max_assignments)
        if max_assignments < 1:
            raise ValueError("limits.max_assignments_per_stoichiometry must be at least 1")
    return AssembleConfig(
        space=AssemblySpace(
            metal=str(root["metal"]),
            oxidation_state=int(root["oxidation_state"]),
            spin=None if root.get("spin") is None else int(root["spin"]),
            ligands=tuple(_ligand(item, i) for i, item in enumerate(ligands)),
            geometries=tuple(str(value) for value in geometries),
        ),
        constraints=AssemblyConstraints(
            required_trans=_relations(constraints.get("required_trans"), "constraints.required_trans"),
            required_cis=_relations(constraints.get("required_cis"), "constraints.required_cis"),
        ),
        output=str(generation.get("output", "assembled")),
        optimizer=optimizer,
        generator_kwargs=dict(_mapping(generation.get("options", {}), "generation.options")),
        max_stoichiometries=max_stoichiometries,
        max_assignments_per_stoichiometry=max_assignments,
        override_path=root.get("ligand_overrides"),
    )


def load_config(path: str | Path) -> AssembleConfig:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read config '{config_path}': {exc}") from exc
    try:
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            import yaml
            payload = yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"could not parse config '{config_path}': {exc}") from exc
    return config_from_mapping(_mapping(payload, "config"))
