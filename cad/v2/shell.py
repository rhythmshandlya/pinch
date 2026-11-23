"""The v2 pill shell (crab/shapes.py: a superellipsoid), as printable solids: the bottom tray and the top cover.

The outer surface is lofted through superellipse sections (the same formula as crab/shapes.py, so the printed shell
is the simulated one), the inside is the same shape 2 mm smaller (WALL), and the two halves part at SPLIT_Z.
Units mm, body frame (x forward, z up).
"""

import sys
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Box, Location, Plane, Polyline, Solid, Wire, loft, make_face  # noqa: E402, F401

from crab.params import PARAMS  # noqa: E402

WALL = 2.0
N_SECT = 14              # loft sections from the bottom to the top
N_PTS = 72               # points per section
U_MAX = np.radians(84)   # sections stop short of the poles; the end sections are the flat caps


def _sp(w, e):
    return np.sign(w) * np.abs(w) ** e


def dims(inset=0.0):
    a, b, c = (v * 1000 - inset for v in PARAMS.body.half_size)
    cx, cy, cz = (v * 1000 for v in PARAMS.body.shell_center)
    ev, eh = PARAMS.body.shell_exp
    return a, b, c, (cx, cy, cz), ev, eh


def _section(a, b, c, ctr, ev, eh, u):
    cu, zu = _sp(np.cos(u), ev), c * _sp(np.sin(u), ev)
    vs = np.linspace(-np.pi, np.pi, N_PTS, endpoint=False)
    pts = [(ctr[0] + a * cu * _sp(np.cos(v), eh), ctr[1] + b * cu * _sp(np.sin(v), eh), ctr[2] + zu) for v in vs]
    return make_face(Polyline(*pts, close=True))


def _solid(inset):
    a, b, c, ctr, ev, eh = dims(inset)
    us = np.sin(np.linspace(-np.pi / 2, np.pi / 2, N_SECT)) * U_MAX      # denser near the poles
    return loft([_section(a, b, c, ctr, ev, eh, u) for u in us])


@cache
def shell_solid():
    """The hollow shell, one piece (2 mm wall)."""
    return _solid(0.0) - _solid(WALL)


if __name__ == "__main__":
    import time
    t = time.time()
    s = shell_solid()
    bb = s.bounding_box()
    print(f"shell: {s.volume / 1000:.1f} cm3 of plastic, bbox {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm, "
          f"{time.time() - t:.1f} s")
