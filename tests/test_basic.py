"""Basic tests for the trex_geo import package."""

import pytest


class TestStructureInfo:
    def test_get_structure_info_octahedral(self):
        from trex_geo import get_structure_info

        class MockTrex:
            pairs = [(None, None), (None, None), (None, None)]
            singles = []

        info = get_structure_info(MockTrex())
        assert info["n_sites"] == 6
        assert info["n_trans_pairs"] == 3
        assert info["n_cis_pairs"] == 12

    def test_get_structure_info_square_planar(self):
        from trex_geo import get_structure_info

        class MockTrex:
            pairs = [(None, None), (None, None)]
            singles = []

        info = get_structure_info(MockTrex())
        assert info["n_sites"] == 4
        assert info["n_trans_pairs"] == 2
        assert info["n_cis_pairs"] == 4


class TestInternalsValidation:
    def test_validate_internals_correct(self):
        from trex_geo import validate_internals

        structure_info = {
            "n_sites": 6,
            "n_trans_pairs": 3,
            "n_cis_pairs": 12,
            "cis_pair_indices": [
                (i, j)
                for i in range(6)
                for j in range(i + 1, 6)
                if not (i % 2 == 0 and j == i + 1 and i < 6)
            ][:12],
        }
        internals = {
            "site_r": [2.0] * 6,
            "trans_cos": [-1.0] * 3,
            "cis_cos": [0.0] * 12,
        }
        assert validate_internals(internals, structure_info) is True

    def test_validate_internals_wrong_site_r(self):
        from trex_geo import validate_internals

        structure_info = {
            "n_sites": 6,
            "n_trans_pairs": 3,
            "n_cis_pairs": 12,
        }
        internals = {
            "site_r": [2.0] * 4,
            "trans_cos": [-1.0] * 3,
            "cis_cos": [0.0] * 12,
        }
        with pytest.raises(ValueError, match="site_r length"):
            validate_internals(internals, structure_info)


class TestHapticDetection:
    def test_is_site_haptic_single_atom(self):
        from trex_geo.trex_to_xyz import is_site_haptic

        class MockSite:
            atoms = [1]

        assert is_site_haptic(MockSite()) is False

    def test_is_site_haptic_multi_atom(self):
        from trex_geo.trex_to_xyz import is_site_haptic

        class MockSite:
            atoms = [1, 2, 3, 4, 5]

        assert is_site_haptic(MockSite()) is True
