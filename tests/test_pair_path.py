"""Pair-hyperedge path feature and building hypergraphs the way a checkpoint was trained."""

from collections import Counter

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("trex")

from trex_geo.model import (  # noqa: E402
    NUM_PAIR_PATH_CLASSES,
    PAIR_PATH_DIFFERENT_LIGANDS,
    PAIR_PATH_SITE,
    BuildConfig,
    pair_path_class,
    trex_to_hyperdata,
)
from trex_geo.predict import build_config_from_checkpoint  # noqa: E402

ZN_OS = "Zn{+2} | L=[ SMILES:CCc1occc(=S)c1[O-], SMILES:CCc1occc(=S)c1[O-] ] | MAP:{ 1:8, 1:10, 2:8, 2:10 }"
PD_DTC = "Pd{+2} | L=[ SMILES:CCN(CC)C(=S)[S-], SMILES:CCN(CC)C(=S)[S-] ] | MAP:{ (1:7, 2:7), (1:8, 2:8) }"
CO_EN3 = "Co{+3} | L=[ SMILES:NCCN, SMILES:NCCN, SMILES:NCCN ] | MAP:{ (1:1, 2:1), (1:4, 3:1), (2:4, 3:4) } | G:O"


def _pair_classes(trex_str):
    d = trex_to_hyperdata(trex_str, cfg=BuildConfig(include_metal_dative=True, include_pair_path=True))
    assert len(d.he_path_class) == len(d.he_type)
    assert all(int(c) == PAIR_PATH_SITE for t, c in zip(d.he_type, d.he_path_class) if int(t) == 2)
    return Counter(int(c) for t, c in zip(d.he_type, d.he_path_class) if int(t) < 2)


def test_ring_size_classes():
    # ring size = bonds between the two donors + 2 (the donors' bonds to the metal)
    assert [pair_path_class(n) for n in (2, 3, 4, 5, 6, 12)] == [1, 2, 3, 4, 5, 5]
    assert NUM_PAIR_PATH_CLASSES == 7


def test_bis_chelate_bites_differ_from_inter_ligand_pairs():
    # ZnL2 with two O,S chelates: two 5-membered-ring bites, four pairs between ligands
    assert _pair_classes(ZN_OS) == {2: 2, PAIR_PATH_DIFFERENT_LIGANDS: 4}


def test_four_membered_chelate():
    assert _pair_classes(PD_DTC) == {1: 2, PAIR_PATH_DIFFERENT_LIGANDS: 4}


def test_tris_chelate():
    assert _pair_classes(CO_EN3) == {2: 3, PAIR_PATH_DIFFERENT_LIGANDS: 12}


def test_default_build_is_unchanged():
    d = trex_to_hyperdata(ZN_OS, cfg=BuildConfig())
    assert getattr(d, "he_path_class", None) is None


def test_config_follows_the_checkpoint():
    # the shipped ensemble was trained without --no_dative, i.e. with metal-donor edges
    cfg = build_config_from_checkpoint({"no_dative": False, "no_site": False})
    assert cfg.include_metal_dative and cfg.include_site_hyperedges and not cfg.include_pair_path
    assert not build_config_from_checkpoint({"no_dative": True}).include_metal_dative
    assert build_config_from_checkpoint({"pair_path": True}).include_pair_path
