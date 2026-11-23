# trex_geo/embed_choice.py
"""
Which 3D embedding route a build uses, and the "auto" fallback.

    native      ligands keep the Lewis structures written in their SMILES (native_embed)
    metallogen  the MetalloGen route (LP bond orders and charges, element swaps)
    auto        native; if that build fails, or with T-REX validation on its result does
                not read back with the input's connectivity, trans map and chirality
                (trex_match tier 3), build again with MetalloGen and keep the better one:
                a valid one first, then the closer tier, then the lower energy (same
                optimizer, same atoms, so the energies compare)

The method comes from the embed_method argument, else $TREX_EMBED_METHOD, else auto.
Without validation, auto only falls back per embedding (trex_embed.embed_trex_metallogen).
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

METHODS = ("auto", "native", "metallogen")
_ENV = "TREX_EMBED_METHOD"


def resolve_embed_method(method: Optional[str] = None) -> str:
    m = (method or os.environ.get(_ENV) or "auto").lower()
    if m not in METHODS:
        raise ValueError(f"Unknown embedding method {m!r}; use one of {', '.join(METHODS)}")
    return m


@contextlib.contextmanager
def embed_method_env(method: str):
    """Set $TREX_EMBED_METHOD for the duration (the embedder reads it per call)."""
    old = os.environ.get(_ENV)
    os.environ[_ENV] = method
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = old


def _rank_key(r):
    from .trex_match import VALID_LEVEL
    if not getattr(r, "success", False):
        return (-3, 0.0)
    level = (r.diagnostics or {}).get("trex_level")
    level = -1 if level is None else min(int(level), VALID_LEVEL)
    energy = r.energy if r.energy is not None else math.inf
    return (level, -energy)


def _tag(r, requested: str, fallback: bool, levels=None):
    d = dict(r.diagnostics or {})
    d["embed_method_requested"] = requested
    d.setdefault("embed_method", None if requested == "auto" else requested)
    d["auto_fallback"] = fallback
    if levels is not None:
        d["auto_levels"] = levels
    r.diagnostics = d
    return r


def run_with_embed_method(run: Callable[[], object], embed_method: Optional[str], validate: bool):
    """Call run() (one complete build, returning a TrexToXyzResult) under the chosen method."""
    method = resolve_embed_method(embed_method)
    if method != "auto" or not validate:
        with embed_method_env(method):
            return _tag(run(), method, False)

    with embed_method_env("native"):
        first = run()
    if first.success and (first.diagnostics or {}).get("trex_valid") is True:
        return _tag(first, "auto", False)

    logger.info("auto embedding: native build %s; building again with MetalloGen",
                "failed" if not first.success else "does not read back as the input T-REX")
    with embed_method_env("metallogen"):
        second = run()
    best = first if _rank_key(first) >= _rank_key(second) else second
    levels = {"native": (first.diagnostics or {}).get("trex_level") if first.success else None,
              "metallogen": (second.diagnostics or {}).get("trex_level") if second.success else None}
    return _tag(best, "auto", True, levels)


__all__ = ["METHODS", "resolve_embed_method", "embed_method_env", "run_with_embed_method"]
