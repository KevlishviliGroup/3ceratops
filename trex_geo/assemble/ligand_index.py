"""Read the packaged ligand index and maintain optional user overrides."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .canonicalization import CANONICALIZATION, canonicalize_and_remap_sites, canonicalize_smiles
from .canonicalization import rdkit_version


class LigandNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class BindingModeInput:
    """A user-defined binding mode; atom indices refer to the supplied SMILES."""

    sites: tuple[tuple[int, ...], ...]
    trans_pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    label: str | None = None


def _default_override_path() -> Path:
    configured = os.environ.get("THREECERATOPS_LIGAND_OVERRIDES")
    return Path(configured).expanduser() if configured else Path.home() / ".3ceratops" / "ligands.json"


def _mode_label(site_count: int, trans_pairs: tuple[tuple[int, int], ...]) -> str:
    n_trans = len(trans_pairs)
    labels = {
        (1, 0): "monodentate",
        (2, 0): "cis_bidentate",
        (2, 1): "trans_bidentate",
        (3, 0): "fac_tridentate",
        (3, 1): "mer_tridentate",
        (4, 0): "nontrans_tetradentate",
        (4, 1): "seesaw_tetradentate",
        (4, 2): "equatorial_tetradentate",
    }
    return labels.get((site_count, n_trans), f"explicit_{site_count}dentate_{n_trans}trans")


class LigandIndex:
    def __init__(self, packaged_path: Path | None = None, override_path: Path | None = None):
        if packaged_path is None:
            packaged_path = Path(__file__).parents[1] / "data" / "tmqmg_ligand_index.json"
        self.packaged_path = Path(packaged_path)
        self.override_path = Path(override_path) if override_path else _default_override_path()
        self._packaged = self._read(self.packaged_path, required=True)
        self._check_metadata(self._packaged, self.packaged_path)
        self._overrides = self._read(self.override_path, required=False)
        if self._overrides:
            self._check_metadata(self._overrides, self.override_path)

    @staticmethod
    def _read(path: Path, *, required: bool) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Ligand index not found: {path}")
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _check_metadata(data: dict[str, Any], path: Path) -> None:
        settings = data.get("metadata", {}).get("canonicalization")
        if settings != CANONICALIZATION:
            raise RuntimeError(f"Canonicalization settings mismatch in {path}")
        built_with = data.get("metadata", {}).get("rdkit_version")
        if built_with and built_with != rdkit_version():
            raise RuntimeError(
                f"RDKit version mismatch for {path}: index={built_with}, runtime={rdkit_version()}"
            )

    def lookup(self, smiles: str) -> dict[str, Any]:
        canonical = canonicalize_smiles(smiles)
        for source in (self._overrides, self._packaged):
            record = source.get("ligands", {}).get(canonical)
            if record is not None:
                return record
        raise LigandNotFoundError(canonical)

    def resolve(
        self,
        smiles: str,
        *,
        if_missing: BindingModeInput | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Lookup a ligand, optionally registering a user binding mode when absent."""
        try:
            return self.lookup(smiles)
        except LigandNotFoundError:
            if if_missing is None:
                raise
        return self.register(smiles, if_missing, persist=persist)

    def register(
        self, smiles: str, mode: BindingModeInput, *, persist: bool = True
    ) -> dict[str, Any]:
        canonical, sites = canonicalize_and_remap_sites(smiles, mode.sites)
        n_sites = len(sites)
        trans_pairs = tuple(tuple(int(x) for x in pair) for pair in mode.trans_pairs)
        for left, right in trans_pairs:
            if left == right or left < 0 or right < 0 or left >= n_sites or right >= n_sites:
                raise ValueError("trans_pairs must contain distinct 0-based site positions")
        record = {
            "canonical_smiles": canonical,
            "binding_modes": [{
                "sites": [list(site) for site in sites],
                "denticity": n_sites,
                "hapticities": [len(site) for site in sites],
                "internal_trans_pairs": [list(pair) for pair in trans_pairs],
                "geometry_class": mode.label or _mode_label(n_sites, trans_pairs),
                "observations": 0,
                "source": "user",
            }],
            "candidate_sites": [list(site) for site in sites],
            "potentially_hemilabile": False,
            "hemilabile_transitions": [],
            "source": "user",
        }
        self._overrides.setdefault("metadata", {
            "schema_version": 1,
            "rdkit_version": rdkit_version(),
            "canonicalization": CANONICALIZATION,
        })
        self._overrides.setdefault("ligands", {})[canonical] = record
        if persist:
            self._write_overrides()
        return record

    def _write_overrides(self) -> None:
        self.override_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.override_path.parent, delete=False
        ) as handle:
            json.dump(self._overrides, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.override_path)
