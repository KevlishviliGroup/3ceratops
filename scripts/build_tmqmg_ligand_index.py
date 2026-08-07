#!/usr/bin/env python3
"""Build the versioned 3-CERATOPS ligand catalog from canonical tmQMg T-REX."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from rdkit import RDLogger
from trex.parse_full import parse_trex_full

from trex_geo.assemble.canonicalization import CANONICALIZATION, canonicalize_and_remap_sites
from trex_geo.assemble.canonicalization import rdkit_version


def mode_label(site_count: int, trans_count: int) -> str:
    labels = {
        (1, 0): "monodentate", (2, 0): "cis_bidentate", (2, 1): "trans_bidentate",
        (3, 0): "fac_tridentate", (3, 1): "mer_tridentate",
        (4, 0): "nontrans_tetradentate", (4, 1): "seesaw_tetradentate",
        (4, 2): "equatorial_tetradentate",
    }
    return labels.get((site_count, trans_count), f"explicit_{site_count}dentate_{trans_count}trans")


def normalize_geometry(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def build(csv_path: Path) -> dict[str, Any]:
    modes: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    stats = Counter()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            stats["rows"] += 1
            trex_text = (row.get("trex") or "").strip()
            geometry = normalize_geometry(row.get("geometry") or "")
            if not trex_text:
                stats["missing_trex"] += 1
                continue
            # TP is deliberately unsupported: its MAP pairs are not literal trans pairs.
            if geometry == "trigonal_prismatic" or "| G:TP" in trex_text:
                stats["excluded_trigonal_prismatic"] += 1
                continue
            try:
                trex = parse_trex_full(trex_text)
            except Exception:
                stats["parse_errors"] += 1
                continue
            ligand_sites: dict[int, list[tuple[int, ...]]] = defaultdict(list)
            internal_pairs: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = defaultdict(list)
            for left, right in trex.pairs:
                ligand_sites[left.lig].append(tuple(left.atoms))
                ligand_sites[right.lig].append(tuple(right.atoms))
                if left.lig == right.lig:
                    internal_pairs[left.lig].append((tuple(left.atoms), tuple(right.atoms)))
            for site in trex.singles:
                ligand_sites[site.lig].append(tuple(site.atoms))
            for ligand_number, ligand in enumerate(trex.ligands, start=1):
                if ligand.kind != "SMILES" or not ligand_sites.get(ligand_number):
                    stats["unsupported_ligands"] += 1
                    continue
                raw_sites = ligand_sites[ligand_number]
                try:
                    canonical, canonical_sites = canonicalize_and_remap_sites(ligand.text, raw_sites)
                except Exception:
                    stats["canonicalization_errors"] += 1
                    continue
                old_to_position = {site: i for i, site in enumerate(raw_sites)}
                trans_positions = tuple(sorted(
                    tuple(sorted((old_to_position[left], old_to_position[right])))
                    for left, right in internal_pairs.get(ligand_number, [])
                ))
                key = (canonical_sites, trans_positions)
                entry = modes[canonical].setdefault(key, {
                    "sites": [list(site) for site in canonical_sites],
                    "denticity": len(canonical_sites),
                    "hapticities": [len(site) for site in canonical_sites],
                    "internal_trans_pairs": [list(pair) for pair in trans_positions],
                    "geometry_class": mode_label(len(canonical_sites), len(trans_positions)),
                    "observations": 0,
                    "metals": Counter(),
                    "complex_geometries": Counter(),
                    "source": "tmQMg",
                })
                entry["observations"] += 1
                entry["metals"][trex.metal] += 1
                entry["complex_geometries"][geometry or "unknown"] += 1
                stats["ligand_occurrences"] += 1
    records = {}
    for smiles, keyed_modes in sorted(modes.items()):
        binding_modes = sorted(keyed_modes.values(), key=lambda x: (-x["observations"], x["sites"]))
        for mode in binding_modes:
            mode["metals"] = dict(sorted(mode["metals"].items()))
            mode["complex_geometries"] = dict(sorted(mode["complex_geometries"].items()))
        site_sets = [set(map(tuple, mode["sites"])) for mode in binding_modes]
        transitions = []
        for high_index, high_sites in enumerate(site_sets):
            for low_index, low_sites in enumerate(site_sets):
                if low_sites < high_sites:
                    transitions.append({
                        "higher_mode": high_index,
                        "lower_mode": low_index,
                        "potentially_labile_sites": [
                            list(site) for site in sorted(high_sites - low_sites)
                        ],
                    })
        candidate_sites = sorted(set().union(*site_sets)) if site_sets else []
        records[smiles] = {
            "canonical_smiles": smiles,
            "binding_modes": binding_modes,
            "candidate_sites": [list(site) for site in candidate_sites],
            "potentially_hemilabile": bool(transitions),
            "hemilabile_transitions": transitions,
            "source": "tmQMg",
        }
    stats["unique_ligands"] = len(records)
    return {
        "metadata": {
            "schema_version": 1,
            "built_on": date.today().isoformat(),
            "source_file": csv_path.name,
            "rdkit_version": rdkit_version(),
            "canonicalization": CANONICALIZATION,
            "trigonal_prismatic_policy": "excluded",
            "stats": dict(sorted(stats.items())),
        },
        "ligands": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.warning")
    result = build(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    print(json.dumps(result["metadata"]["stats"], indent=2))


if __name__ == "__main__":
    main()
