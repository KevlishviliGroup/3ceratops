"""Authoritative coordination-geometry definitions used during assembly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryTemplate:
    name: str
    coordination_number: int
    trans_pairs: int
    singles: int


_TEMPLATES = (
    GeometryTemplate("linear", 2, 1, 0),
    GeometryTemplate("bent", 2, 0, 2),
    GeometryTemplate("trigonal_planar", 3, 0, 3),
    GeometryTemplate("t_shaped", 3, 1, 1),
    GeometryTemplate("tetrahedral", 4, 0, 4),
    GeometryTemplate("square_planar", 4, 2, 0),
    GeometryTemplate("seesaw", 4, 1, 2),
    GeometryTemplate("square_pyramidal", 5, 2, 1),
    GeometryTemplate("trigonal_bipyramidal", 5, 1, 3),
    GeometryTemplate("octahedral", 6, 3, 0),
    GeometryTemplate("pentagonal_bipyramidal", 7, 1, 5),
)

GEOMETRIES = {template.name: template for template in _TEMPLATES}

_ALIASES = {
    "t-shaped": "t_shaped",
    "trigonal planar": "trigonal_planar",
    "square planar": "square_planar",
    "square pyramidal": "square_pyramidal",
    "trigonal bipyramidal": "trigonal_bipyramidal",
    "pentagonal bipyramidal": "pentagonal_bipyramidal",
}


def get_geometry(name: str) -> GeometryTemplate:
    normalized = "_".join(name.strip().lower().replace("-", " ").split())
    normalized = _ALIASES.get(name.strip().lower(), normalized)
    if normalized in {"trigonal_prismatic", "tp"}:
        raise ValueError("Trigonal-prismatic geometry is deliberately unsupported")
    try:
        return GEOMETRIES[normalized]
    except KeyError as exc:
        choices = ", ".join(GEOMETRIES)
        raise ValueError(f"Unknown geometry {name!r}; choose one of: {choices}") from exc
