"""Short G codes must mean the same as the canonical (G-less) string."""

import pytest

pytest.importorskip("trex")

from trex.parse_full import parse_trex_full  # noqa: E402

from trex_geo import compute_idealized_internals  # noqa: E402
from trex_geo.geometry_codes import get_trex_geometry_code  # noqa: E402

NI_CO4 = "Ni{+0}|L=[SMILES:[C-]#[O+],SMILES:[C-]#[O+],SMILES:[C-]#[O+],SMILES:[C-]#[O+]]|MAP:{;1:1,2:1,3:1,4:1}"
FE_CO5 = ("Fe{+0}|L=[SMILES:[C-]#[O+],SMILES:[C-]#[O+],SMILES:[C-]#[O+],SMILES:[C-]#[O+],SMILES:[C-]#[O+]]"
          "|MAP:{(1:1,2:1);3:1,4:1,5:1}")
PRISM = ("Mo{+0}|L=[" + ",".join(["SMILES:[C-]#[O+]"] * 6) + "]|MAP:{;1:1,2:1,3:1,4:1,5:1,6:1}|G:TP")


@pytest.mark.parametrize("base, code", [(NI_CO4, "G:T"), (FE_CO5, "G:TBP")])
def test_short_code_matches_canonical(base, code):
    plain, coded = parse_trex_full(base), parse_trex_full(base + "|" + code)
    assert get_trex_geometry_code(coded) == get_trex_geometry_code(plain)
    assert compute_idealized_internals(coded)["cis_cos"] == pytest.approx(compute_idealized_internals(plain)["cis_cos"])


def test_tp_is_trigonal_prismatic_at_cn6():
    assert get_trex_geometry_code(parse_trex_full(PRISM)) == "TP"


def test_model_features_ignore_short_codes():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from trex_geo.model import _compute_ideal_cos_for_pair

    plain, coded = parse_trex_full(NI_CO4), parse_trex_full(NI_CO4 + "|G:T")
    a = [_compute_ideal_cos_for_pair(plain, plain.singles[0], s, False) for s in plain.singles[1:]]
    b = [_compute_ideal_cos_for_pair(coded, coded.singles[0], s, False) for s in coded.singles[1:]]
    assert a == pytest.approx(b)
    assert a[0] == pytest.approx(-1 / 3)
