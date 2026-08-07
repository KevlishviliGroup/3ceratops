"""Public input and output models for complex-space assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .ligand_index import BindingModeInput

ModeSelection = Literal["dominant", "all_observed"] | tuple[int, ...]


@dataclass(frozen=True)
class LigandSpec:
    smiles: str
    max_count: int
    min_count: int = 0
    binding_modes: ModeSelection = "dominant"
    if_missing: BindingModeInput | None = None
    persist_if_missing: bool = True
    identifier: str | None = None

    def __post_init__(self) -> None:
        if self.min_count < 0:
            raise ValueError("min_count cannot be negative")
        if self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count")
        if isinstance(self.binding_modes, str) and self.binding_modes not in {
            "dominant", "all_observed"
        }:
            raise ValueError("binding_modes must be 'dominant', 'all_observed', or mode indices")
        if isinstance(self.binding_modes, tuple) and not self.binding_modes:
            raise ValueError("binding mode index selection cannot be empty")
        if self.identifier is not None and not self.identifier.strip():
            raise ValueError("identifier cannot be empty")


@dataclass(frozen=True)
class SiteRef:
    """A T-REX site on a user-named ligand copy.

    ``copy`` is zero-based; ``atoms`` are 1-based canonical-SMILES atom indices.
    """

    ligand: str
    copy: int
    atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.ligand.strip():
            raise ValueError("SiteRef.ligand cannot be empty")
        if self.copy < 0:
            raise ValueError("SiteRef.copy cannot be negative")
        if not self.atoms or any(atom < 1 for atom in self.atoms):
            raise ValueError("SiteRef atoms are 1-based and cannot be empty")
        if len(self.atoms) != len(set(self.atoms)):
            raise ValueError("SiteRef atoms cannot contain duplicates")


SitePair = tuple[SiteRef, SiteRef]


@dataclass(frozen=True)
class AssemblyConstraints:
    """Hard MAP constraints, expressed using T-REX-compatible sites."""

    required_trans: tuple[SitePair, ...] = field(default_factory=tuple)
    required_cis: tuple[SitePair, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for relation, pairs in (
            ("required_trans", self.required_trans),
            ("required_cis", self.required_cis),
        ):
            for pair in pairs:
                if len(pair) != 2 or pair[0] == pair[1]:
                    raise ValueError(f"{relation} entries must contain two distinct sites")


@dataclass(frozen=True)
class AssemblySpace:
    metal: str
    oxidation_state: int
    ligands: tuple[LigandSpec, ...]
    geometries: tuple[str, ...]
    spin: int | None = None

    def __post_init__(self) -> None:
        if not self.metal.strip():
            raise ValueError("metal cannot be empty")
        if not self.ligands:
            raise ValueError("at least one ligand specification is required")
        if not self.geometries:
            raise ValueError("at least one geometry is required")


@dataclass(frozen=True)
class ResolvedBindingMode:
    mode_index: int
    sites: tuple[tuple[int, ...], ...]
    internal_trans_pairs: tuple[tuple[int, int], ...]
    geometry_class: str
    observations: int

    @property
    def denticity(self) -> int:
        return len(self.sites)


@dataclass(frozen=True)
class LigandInstance:
    spec_index: int
    canonical_smiles: str
    mode: ResolvedBindingMode


@dataclass(frozen=True)
class AssembledComplex:
    """One unique, fully canonicalized T-REX topology."""

    trex: str
    geometry: str
    stoichiometry: tuple[int, ...]
    ligand_modes: tuple[int, ...]


@dataclass(frozen=True)
class StoichiometricCandidate:
    geometry: str
    coordination_number: int
    ligand_instances: tuple[LigandInstance, ...]
    stoichiometry: tuple[int, ...]

    @property
    def fixed_internal_trans_pairs(self) -> int:
        return sum(len(instance.mode.internal_trans_pairs) for instance in self.ligand_instances)
