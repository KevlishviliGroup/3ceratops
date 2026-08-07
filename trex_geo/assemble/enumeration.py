"""Construct native T-REX seeds, enumerate MAPs, canonicalize, and deduplicate."""

from __future__ import annotations

from collections import defaultdict

from trex.canon_full import canonicalize_full, pretty_trex_full
from trex.enumerate_trex import enumerate_trex_maps_by_counts
from trex.model import LigandPayload, MapSite, TrexFull

from .geometry_registry import get_geometry
from .ligand_index import LigandIndex
from .models import (
    AssembledComplex,
    AssemblyConstraints,
    AssemblySpace,
    SiteRef,
    StoichiometricCandidate,
)
from .stoichiometry import enumerate_stoichiometries


class ConstraintNotApplicableError(ValueError):
    """A constraint references a ligand copy/site absent from a candidate."""


def _site_key(site: MapSite) -> tuple[int, tuple[int, ...]]:
    return site.lig, tuple(sorted(site.atoms))


def _pair_key(left: MapSite, right: MapSite) -> frozenset[tuple[int, tuple[int, ...]]]:
    return frozenset((_site_key(left), _site_key(right)))


def _spec_identifiers(space: AssemblySpace) -> tuple[str, ...]:
    identifiers = tuple(spec.identifier or spec.smiles for spec in space.ligands)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("LigandSpec identifiers must be unique")
    return identifiers


def _instantiate_sites(
    space: AssemblySpace, candidate: StoichiometricCandidate
) -> tuple[list[LigandPayload], dict[tuple[str, int], dict[tuple[int, ...], MapSite]], list[MapSite]]:
    identifiers = _spec_identifiers(space)
    copies: defaultdict[int, int] = defaultdict(int)
    lookup: dict[tuple[str, int], dict[tuple[int, ...], MapSite]] = {}
    payloads: list[LigandPayload] = []
    sites: list[MapSite] = []

    for ligand_position, instance in enumerate(candidate.ligand_instances, start=1):
        copy = copies[instance.spec_index]
        copies[instance.spec_index] += 1
        payloads.append(LigandPayload(kind="SMILES", text=instance.canonical_smiles))
        by_atoms: dict[tuple[int, ...], MapSite] = {}
        for atoms in instance.mode.sites:
            key = tuple(sorted(atoms))
            site = MapSite(lig=ligand_position, atoms=list(atoms))
            if key in by_atoms:
                raise ValueError("A binding mode contains the same T-REX site more than once")
            by_atoms[key] = site
            sites.append(site)
        lookup[(identifiers[instance.spec_index], copy)] = by_atoms
    return payloads, lookup, sites


def _resolve_ref(
    ref: SiteRef,
    lookup: dict[tuple[str, int], dict[tuple[int, ...], MapSite]],
) -> MapSite:
    instance = lookup.get((ref.ligand, ref.copy))
    if instance is None:
        raise ConstraintNotApplicableError(
            f"Constraint references absent ligand copy {ref.ligand}[{ref.copy}]"
        )
    site = instance.get(tuple(sorted(ref.atoms)))
    if site is None:
        available = ", ".join(str(key) for key in sorted(instance))
        raise ConstraintNotApplicableError(
            f"{ref.ligand}[{ref.copy}] has no T-REX site {ref.atoms}; available: {available}"
        )
    return site


def _seed_for_candidate(
    space: AssemblySpace,
    candidate: StoichiometricCandidate,
    constraints: AssemblyConstraints,
) -> tuple[TrexFull, set[frozenset[tuple[int, tuple[int, ...]]]]]:
    payloads, lookup, all_sites = _instantiate_sites(space, candidate)
    fixed_pairs: list[tuple[MapSite, MapSite]] = []

    offset = 0
    for instance in candidate.ligand_instances:
        instance_sites = all_sites[offset : offset + instance.mode.denticity]
        offset += instance.mode.denticity
        for left, right in instance.mode.internal_trans_pairs:
            if not (0 <= left < len(instance_sites) and 0 <= right < len(instance_sites)):
                raise ValueError("Binding mode contains an invalid internal trans-pair index")
            fixed_pairs.append((instance_sites[left], instance_sites[right]))

    required_trans = [
        (_resolve_ref(left, lookup), _resolve_ref(right, lookup))
        for left, right in constraints.required_trans
    ]
    required_cis = {
        _pair_key(_resolve_ref(left, lookup), _resolve_ref(right, lookup))
        for left, right in constraints.required_cis
    }

    indexed_internal_pairs = {_pair_key(left, right) for left, right in fixed_pairs}
    for left, right in required_trans:
        if left.lig == right.lig and _pair_key(left, right) not in indexed_internal_pairs:
            raise ValueError(
                "An intraligand required-trans constraint must already be encoded by the "
                "selected binding mode; select or register the corresponding mode"
            )

    all_fixed_pairs = fixed_pairs + required_trans
    fixed_pairs = []
    fixed_keys: set[frozenset[tuple[int, tuple[int, ...]]]] = set()
    used_endpoints: set[tuple[int, tuple[int, ...]]] = set()
    for left, right in all_fixed_pairs:
        pair = _pair_key(left, right)
        if len(pair) != 2:
            raise ValueError("A trans constraint cannot pair a site with itself")
        if pair in required_cis:
            raise ValueError("The same site pair is required to be both cis and trans")
        endpoints = {_site_key(left), _site_key(right)}
        if pair not in fixed_keys and endpoints & used_endpoints:
            raise ValueError("A coordination site cannot have two trans partners")
        if pair not in fixed_keys:
            fixed_pairs.append((left, right))
            fixed_keys.add(pair)
            used_endpoints.update(endpoints)

    free_sites = [site for site in all_sites if _site_key(site) not in used_endpoints]
    return TrexFull(
        metal=space.metal,
        ox=space.oxidation_state,
        spin=space.spin,
        ligands=payloads,
        pairs=fixed_pairs,
        singles=free_sites,
    ), required_cis


def enumerate_candidate_maps(
    space: AssemblySpace,
    candidate: StoichiometricCandidate,
    *,
    constraints: AssemblyConstraints | None = None,
    max_assignments: int | None = None,
) -> list[AssembledComplex]:
    """Return every unique canonical T-REX MAP isomer for one stoichiometry."""
    if max_assignments is not None and max_assignments < 1:
        raise ValueError("max_assignments must be at least 1")
    constraints = constraints or AssemblyConstraints()
    geometry = get_geometry(candidate.geometry)
    seed, forbidden_pairs = _seed_for_candidate(space, candidate, constraints)
    enumeration_limit = None if max_assignments is None else max_assignments + 1
    assignments = enumerate_trex_maps_by_counts(
        seed,
        preserve_intra_pairs=True,
        forbid_new_intra_pairs=True,
        preserve_cross_pairs=True,
        use_input_counts=False,
        pairs_required=geometry.trans_pairs,
        singles_required=geometry.singles,
        collapse_identical_ligands=False,
        collapse_internal_symmetry=False,
        limit=enumeration_limit,
    )
    if max_assignments is not None and len(assignments) > max_assignments:
        raise RuntimeError(
            f"MAP enumeration exceeds max_assignments={max_assignments} for "
            f"{candidate.geometry} {candidate.stoichiometry}"
        )

    unique: dict[str, AssembledComplex] = {}
    for assignment in assignments:
        if any(_pair_key(left, right) in forbidden_pairs for left, right in assignment.pairs):
            continue
        full = TrexFull(
            metal=space.metal,
            ox=space.oxidation_state,
            spin=space.spin,
            ligands=seed.ligands,
            pairs=list(assignment.pairs),
            singles=list(assignment.singles),
        )
        canonical = canonicalize_full(full)
        rendered = pretty_trex_full(canonical)
        unique.setdefault(rendered, AssembledComplex(
            trex=rendered,
            geometry=candidate.geometry,
            stoichiometry=candidate.stoichiometry,
            ligand_modes=tuple(item.mode.mode_index for item in candidate.ligand_instances),
        ))
    return list(unique.values())


def assemble_trex(
    space: AssemblySpace,
    *,
    constraints: AssemblyConstraints | None = None,
    ligand_index: LigandIndex | None = None,
    max_stoichiometries: int | None = None,
    max_assignments_per_stoichiometry: int | None = None,
) -> list[AssembledComplex]:
    """Enumerate stoichiometries and return all unique canonical T-REX structures."""
    unique: dict[str, AssembledComplex] = {}
    for candidate in enumerate_stoichiometries(
        space, ligand_index=ligand_index, max_candidates=max_stoichiometries
    ):
        try:
            structures = enumerate_candidate_maps(
                space,
                candidate,
                constraints=constraints,
                max_assignments=max_assignments_per_stoichiometry,
            )
        except ConstraintNotApplicableError:
            continue
        for structure in structures:
            unique.setdefault(structure.trex, structure)
    return list(unique.values())
