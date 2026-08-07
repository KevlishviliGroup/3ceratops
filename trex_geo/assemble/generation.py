"""Batch 3D generation for canonical structures produced by :mod:`assemble`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .models import AssembledComplex

Optimizer = Literal["xtb", "uma"]
Generator = Callable[..., Any]
MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class GenerationRecord:
    """Manifest entry for one canonical T-REX topology."""

    candidate_id: str
    trex: str
    geometry: str
    stoichiometry: tuple[int, ...]
    ligand_modes: tuple[int, ...]
    optimizer: Optimizer
    configuration_id: str
    status: Literal["success", "failed", "skipped"]
    xyz_file: str | None = None
    energy: float | None = None
    internals_source: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class BatchGenerationResult:
    output_dir: Path
    manifest_path: Path
    records: tuple[GenerationRecord, ...]

    @property
    def succeeded(self) -> int:
        return sum(record.status in {"success", "skipped"} for record in self.records)

    @property
    def failed(self) -> int:
        return sum(record.status == "failed" for record in self.records)


def candidate_id(trex: str) -> str:
    """Return a stable, filename-safe identity for a canonical T-REX string."""
    return hashlib.sha256(trex.encode("utf-8")).hexdigest()[:16]


def candidate_directory_name(trex: str, *, max_prefix_length: int = 80) -> str:
    """Return a readable, stable, filesystem-safe directory name.

    The prefix is derived from the canonical T-REX string; the hash is retained
    because sanitization and truncation are not injective.
    """
    safe = []
    previous_separator = False
    for character in trex:
        if character.isalnum() or character in {"+", "-"}:
            safe.append(character)
            previous_separator = False
        elif not previous_separator:
            safe.append("_")
            previous_separator = True
    prefix = "".join(safe).strip("_")[:max_prefix_length].rstrip("_") or "complex"
    return f"{prefix}__{candidate_id(trex)[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported assembly manifest schema in '{path}'")
    return payload


def _generator_for(optimizer: Optimizer) -> Generator:
    if optimizer == "xtb":
        from ..trex_to_xyz import trex_to_xyz

        return trex_to_xyz
    if optimizer == "uma":
        from ..trex_to_xyz_uma import trex_to_xyz_uma

        return trex_to_xyz_uma
    raise ValueError("optimizer must be 'xtb' or 'uma'")


def _write_xyz(path: Path, result: Any, trex: str) -> None:
    mol = result.mol
    conf = mol.GetConformer()
    energy = "" if result.energy is None else f" E={result.energy:.10g}"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{mol.GetNumAtoms()}\n")
        handle.write(f"T-REX={trex}{energy} source={result.internals_source}\n")
        for index in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(index)
            position = conf.GetAtomPosition(index)
            handle.write(
                f"{atom.GetSymbol():2s} {position.x:15.8f} "
                f"{position.y:15.8f} {position.z:15.8f}\n"
            )


def _record_payload(record: GenerationRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["stoichiometry"] = list(record.stoichiometry)
    payload["ligand_modes"] = list(record.ligand_modes)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _configuration_id(optimizer: Optimizer, kwargs: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"optimizer": optimizer, "generator_kwargs": _jsonable(kwargs)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_assembled(
    structures: Iterable[AssembledComplex],
    output_dir: str | Path,
    *,
    optimizer: Optimizer = "xtb",
    overwrite: bool = False,
    manifest_name: str = "manifest.json",
    generator_kwargs: dict[str, Any] | None = None,
    _generator: Generator | None = None,
) -> BatchGenerationResult:
    """Generate one optimized XYZ per unique canonical assembled structure.

    Each candidate is isolated: an exception or unsuccessful generator result is
    recorded and does not terminate the batch. The manifest is replaced atomically
    after every candidate, making interrupted runs safely resumable.
    """
    if optimizer not in {"xtb", "uma"}:
        raise ValueError("optimizer must be 'xtb' or 'uma'")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / manifest_name
    previous = _load_manifest(manifest_path)
    previous_records = {
        item["candidate_id"]: item for item in previous.get("candidates", [])
    }
    generator = _generator or _generator_for(optimizer)
    kwargs = dict(generator_kwargs or {})
    configuration_id = _configuration_id(optimizer, kwargs)

    # The API also protects callers that concatenate assembly results themselves.
    unique = {structure.trex: structure for structure in structures}
    records: list[GenerationRecord] = []
    started_at = _utc_now()

    def save() -> None:
        _atomic_json(manifest_path, {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "optimizer": optimizer,
            "configuration_id": configuration_id,
            "generator_kwargs": _jsonable(kwargs),
            "candidates": [_record_payload(record) for record in records],
            "summary": {
                "total": len(records),
                "success": sum(r.status in {"success", "skipped"} for r in records),
                "failed": sum(r.status == "failed" for r in records),
            },
        })

    for structure in unique.values():
        identity = candidate_id(structure.trex)
        candidate_dir = root / candidate_directory_name(structure.trex)
        xyz_path = candidate_dir / "structure.xyz"
        topology_path = candidate_dir / "topology.trex"
        generation_path = candidate_dir / "generation.json"
        xyz_name = str(xyz_path.relative_to(root))
        old = previous_records.get(identity)
        if (
            not overwrite
            and old is not None
            and old.get("status") in {"success", "skipped"}
            and old.get("optimizer") == optimizer
            and old.get("configuration_id") == configuration_id
            and xyz_path.is_file()
        ):
            record = GenerationRecord(
                candidate_id=identity,
                trex=structure.trex,
                geometry=structure.geometry,
                stoichiometry=structure.stoichiometry,
                ligand_modes=structure.ligand_modes,
                optimizer=optimizer,
                configuration_id=configuration_id,
                status="skipped",
                xyz_file=xyz_name,
                energy=old.get("energy"),
                internals_source=old.get("internals_source"),
                diagnostics=old.get("diagnostics") or {},
            )
        else:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            topology_path.write_text(structure.trex + "\n", encoding="utf-8")
            work_dir = candidate_dir / "work"
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("work_dir", str(work_dir))
            try:
                result = generator(structure.trex, **call_kwargs)
                if result.success and result.mol is not None:
                    _write_xyz(xyz_path, result, structure.trex)
                    record = GenerationRecord(
                        candidate_id=identity, trex=structure.trex,
                        geometry=structure.geometry,
                        stoichiometry=structure.stoichiometry,
                        ligand_modes=structure.ligand_modes,
                        optimizer=optimizer, configuration_id=configuration_id,
                        status="success", xyz_file=xyz_name,
                        energy=result.energy,
                        internals_source=result.internals_source,
                        diagnostics=result.diagnostics or {},
                    )
                else:
                    record = GenerationRecord(
                        candidate_id=identity, trex=structure.trex,
                        geometry=structure.geometry,
                        stoichiometry=structure.stoichiometry,
                        ligand_modes=structure.ligand_modes,
                        optimizer=optimizer, configuration_id=configuration_id,
                        status="failed",
                        internals_source=getattr(result, "internals_source", None),
                        diagnostics=getattr(result, "diagnostics", None) or {},
                        error=getattr(result, "error", None) or "generation failed",
                    )
            except Exception as exc:  # failure isolation is intentional at this boundary
                record = GenerationRecord(
                    candidate_id=identity, trex=structure.trex,
                    geometry=structure.geometry,
                    stoichiometry=structure.stoichiometry,
                    ligand_modes=structure.ligand_modes,
                    optimizer=optimizer, configuration_id=configuration_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
        records.append(record)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        topology_path.write_text(structure.trex + "\n", encoding="utf-8")
        _atomic_json(generation_path, _record_payload(record))
        save()

    if not records:
        save()
    return BatchGenerationResult(root, manifest_path, tuple(records))
