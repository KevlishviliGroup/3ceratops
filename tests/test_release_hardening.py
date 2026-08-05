"""Tests for release-hardening changes that do not run optimizers."""

import builtins


def test_uma_pipeline_is_public():
    import trex_geo

    assert callable(trex_geo.trex_to_xyz_uma)
    assert callable(trex_geo.trex_to_xyz_uma_file)
    assert callable(trex_geo.optimize_with_uma)


def test_seesaw_geometry_name_is_consistent():
    from trex_geo.trex_to_xyz import get_geometry_full_name

    class MockTrex:
        @staticmethod
        def get_geo_type():
            return "ssw"

    assert get_geometry_full_name(MockTrex()) == "4_seesaw"


def test_missing_prediction_stack_returns_none(monkeypatch):
    from trex_geo.trex_to_xyz import _try_predict_internals

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "predict" and level == 1:
            raise ImportError("prediction stack deliberately unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert _try_predict_internals("dummy", None) is None
