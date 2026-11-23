# trex_geo/geometry_codes.py
"""
Geometry codes in the form trex.geometry and the LF-GNN features expect.

trex.geometry's ideal-angle table, and the geometry vocabulary the LF-GNN was
trained on ('O', 'TP', 'bnt', 'ln', 'pntbp', 'sqpl', 'sqpy', 'ssw', 'thd',
'trbp', 'trpl'), use T-REX's own codes. A hand-written G field may carry the
short codes that 3ceratops also accepts ('T', 'TBP', ...). Unknown keys fall
back to 90 degrees, which for G:T puts four donors mutually perpendicular and,
as a model feature, shifts predicted M-L distances (Ni(CO)4: 1.81 A without G,
1.69 A with G:T).

Each short code is mapped only at its own coordination number: canonical T-REX
uses G:TP for trigonal prismatic (CN 6), which must not be read as trigonal
planar.
"""

GEO_CODE_TO_TREX = {
    # code: (T-REX code, coordination number)
    "SP": ("sqpl", 4),
    "SQP": ("sqpy", 5),
    "T": ("thd", 4),
    "TBP": ("trbp", 5),
    "L": ("ln", 2),
    "TP": ("trpl", 3),
    "TPY": ("thd", 3),  # trigonal pyramid = tetrahedron with one vacant site
    "B": ("bnt", 2),
    "BNT": ("bnt", 2),
}


def get_trex_geometry_code(t) -> str:
    """Geometry code of a TrexFull in T-REX's own vocabulary ('thd', 'trbp', ...)."""
    code = t.get_geo_type()
    mapped = GEO_CODE_TO_TREX.get(code)
    if mapped is not None and 2 * len(t.pairs) + len(t.singles) == mapped[1]:
        return mapped[0]
    return code


__all__ = ["GEO_CODE_TO_TREX", "get_trex_geometry_code"]
