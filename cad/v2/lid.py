"""The v2 face lid: a printed frame around the 6.25 in HDMI panel, hinged at the top-front of the shell.

Lid frame (local): Y along the hinge axis (the robot's y), Z up the lid from the hinge line, X out of the screen
(thickness 0..LID_T). In the body it stands LID_TILT back from vertical: Location(hinge, (0, -tilt, 0)).
Hinge: a friction hinge set by hand (owner, 2026-11-21): two knuckles on the lid outside one knuckle on the shell,
an M3 bolt through all three, wave washers between them and a nylock nut (docs/phase2/parts_decision.md).
Numbers from configs/layout_v2.toml (`oled`, `visor`); the panel's outline is an estimate until its drawing is
confirmed (the lit area 144.18 x 66.60 mm is the maker's).
"""

import os
import sys
from functools import cache
from math import degrees
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Box, Cylinder, Location, Pos, Rot, fillet  # noqa: E402

from crab.params import PARAMS  # noqa: E402

LID_W, LID_H, LID_T = 165.0, 82.0, 10.0          # across (y), up the lid (z), thickness (x)
MOD_W, MOD_H, MOD_T = 158.0, 76.0, 8.0           # the display module (estimate)
LIT_W, LIT_H = 144.18, 66.60                     # its lit area (maker)
BEZEL_R = 6.0
KNUCKLE_R, M3_CLEAR = 6.0, 3.4
KNUCKLE_LID = ((-62.0, -40.0), (40.0, 62.0))     # lid knuckles along y (outer)
KNUCKLE_SHELL = (-38.0, 38.0)                    # shell knuckle (inner), wave washers fill the 2 mm gaps
CABLE_SLOT = (30.0, 6.0)                         # HDMI flat cable through the hinge (width, height), at y = 0


def hinge_location():
    hx, hy, hz = (v * 1000 for v in PARAMS.body.lid_hinge)
    return Location((hx, hy, hz), (0, -degrees(PARAMS.body.lid_tilt), 0))


@cache
def lid_frame():
    """The printed lid, in the lid frame (hinge axis on Y, Z up the lid, X out of the screen)."""
    body = Pos(LID_T / 2, 0, KNUCKLE_R + LID_H / 2) * Box(LID_T, LID_W, LID_H)
    body = fillet(body.edges().filter_by(lambda e: abs(e.tangent_at(0).X) > 0.9), 4.0)
    # module pocket from the back (x = 0 side), the lit window through the front, a 1.5 mm front ledge
    pocket = Pos(MOD_T / 2 - 0.01, 0, KNUCKLE_R + LID_H / 2) * Box(MOD_T + 0.02, MOD_W + 0.6, MOD_H + 0.6)
    window = Pos(LID_T - 1.0, 0, KNUCKLE_R + LID_H / 2) * Box(4.0, LIT_W + 1.0, LIT_H + 1.0)
    body = body - pocket - window
    for y0, y1 in KNUCKLE_LID:                   # hinge knuckles along the bottom edge
        k = Pos(LID_T / 2, (y0 + y1) / 2, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(KNUCKLE_R, y1 - y0)
        body = body + k
    bolt = Pos(LID_T / 2, 0, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(M3_CLEAR / 2, LID_W + 2)
    slot = Pos(LID_T / 2, 0, KNUCKLE_R + 4.0) * Box(4.0, CABLE_SLOT[0], CABLE_SLOT[1])
    return body - bolt - slot


@cache
def shell_knuckle():
    """The shell's centre knuckle (printed onto the top cover), in the lid frame at tilt 0 (it does not tilt)."""
    k = Pos(LID_T / 2, 0, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(KNUCKLE_R, KNUCKLE_SHELL[1] - KNUCKLE_SHELL[0])
    foot = Pos(LID_T / 2, 0, KNUCKLE_R / 2) * Box(2 * KNUCKLE_R, KNUCKLE_SHELL[1] - KNUCKLE_SHELL[0], KNUCKLE_R + 6.0,
                                                 align=(Align.CENTER, Align.CENTER, Align.MAX))
    bolt = Pos(LID_T / 2, 0, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(M3_CLEAR / 2, LID_W + 2)
    slot = Pos(LID_T / 2, 0, KNUCKLE_R) * Box(4.0, CABLE_SLOT[0], 2 * KNUCKLE_R + 2)
    return k + foot - bolt - slot


if __name__ == "__main__":
    f = lid_frame()
    bb = f.bounding_box()
    print(f"lid: {f.volume / 1000:.1f} cm3, {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm; hinge at "
          f"{hinge_location().position}")
    k = shell_knuckle()
    print(f"shell knuckle: {k.volume / 1000:.1f} cm3")
