"""The v2 face lid: a printed frame around the 6.25 in HDMI panel, hinged at the top-front of the shell.

Lid frame (local): Y along the hinge axis (the robot's y), Z up the lid from the hinge line, X out of the screen
(thickness 0..LID_T). In the body it stands LID_TILT back from vertical: Location(hinge, (0, -tilt, 0)).
Hinge: a friction hinge set by hand (owner, 2026-11-21): two knuckles on the lid outside one knuckle on the shell,
an M3 bolt through all three, wave washers between them and a nylock nut (docs/phase2/parts_decision.md).
Numbers from configs/layout_v2.toml (`oled`, `visor`); the panel's outline is an estimate until its drawing is
confirmed (the lit area 144.18 x 66.60 mm is the maker's).
"""

import sys
from functools import cache
from math import degrees
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Box, Cylinder, Location, Pos, Rot, fillet  # noqa: E402

from crab.params import PARAMS  # noqa: E402

LID_W, LID_H, LID_T = 165.0, 86.0, 10.0          # across (y), up the lid (z), thickness (x)
MOD_W, MOD_H, MOD_T = 158.0, 76.0, 8.0           # the display module (estimate)
LIT_W, LIT_H = 144.18, 66.60                     # its lit area (maker)
BEZEL_R = 6.0
KNUCKLE_R, M3_CLEAR = 6.0, 3.4
PANEL_Z = KNUCKLE_R + LID_H / 2 + 2.0             # the module's centre up the lid: its pocket clears the knuckles
KNUCKLE_LID = ((-62.0, -40.0), (40.0, 62.0))     # lid knuckles along y (outer)
KNUCKLE_SHELL = (-38.0, 38.0)                    # shell knuckle (inner), wave washers fill the 2 mm gaps
CABLE_SLOT = (30.0, 6.0)                         # HDMI flat ribbon (16 mm, adapter boards at its ends) through the hinge
BOLT_INTO = 8.0                                  # each hinge bolt's thread length into the shell knuckle's end
M3_INSERT = 4.2                                  # hole for an M3 x 5.7 heat-set insert
M2_PILOT, M2_CLEAR = 1.7, 2.2
BACK_T = 1.6                                     # the lid's back cover


def hinge_location():
    hx, hy, hz = (v * 1000 for v in PARAMS.body.lid_hinge)
    return Location((hx, hy, hz), (0, -degrees(PARAMS.body.lid_tilt), 0))


@cache
def lid_frame():
    """The printed lid, in the lid frame (hinge axis on Y, Z up the lid, X out of the screen)."""
    body = Pos(LID_T / 2, 0, KNUCKLE_R + LID_H / 2) * Box(LID_T, LID_W, LID_H)
    body = fillet(body.edges().filter_by(lambda e: abs(e.tangent_at(0).X) > 0.9), 4.0)
    # module pocket from the back (x = 0 side), the lit window through the front, a 1.5 mm front ledge
    pocket = Pos(MOD_T / 2 - 0.01, 0, PANEL_Z) * Box(MOD_T + 0.02, MOD_W + 0.6, MOD_H + 0.6)
    window = Pos(LID_T - 1.0, 0, PANEL_Z) * Box(4.0, LIT_W + 1.0, LIT_H + 1.0)
    body = body - pocket - window
    for y0, y1 in KNUCKLE_LID:                   # hinge knuckles along the bottom edge
        k = Pos(LID_T / 2, (y0 + y1) / 2, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(KNUCKLE_R, y1 - y0)
        body = body + k
    slot = Pos(LID_T / 2, 0, KNUCKLE_R + 4.0) * Box(4.0, CABLE_SLOT[0], CABLE_SLOT[1])
    screws = None
    for _x, sy, sz in back_screws():                     # pilots for the back cover's screws, 8 mm into the frame
        h = Pos(4.0, sy, sz) * Rot(0, 90, 0) * Cylinder(M2_PILOT / 2, 8.0)
        screws = h if screws is None else screws + h
    return body - hinge_bolts() - slot - screws - _top_groove()


def hinge_bolts():
    """Two short M3 bolts, one through each lid knuckle into the end of the shell's knuckle (a heat-set insert there),
    instead of one through the whole hinge: the hinge's middle stays clear for the face's flat HDMI ribbon (audit B6)."""
    out = None
    for sgn in (-1, 1):
        y0 = sgn * (KNUCKLE_SHELL[1] - BOLT_INTO)
        y1 = sgn * (KNUCKLE_LID[1][1] + 1.0)
        c = Pos(LID_T / 2, (y0 + y1) / 2, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(M3_CLEAR / 2, abs(y1 - y0))
        out = c if out is None else out + c
    return out


def back_screws():
    """The back cover's two M2 screws, in the frame's 7 mm bottom border (the top border is only 3 mm: the cover's top
    edge hooks into a groove there instead). Lid frame (x, y, z)."""
    zb = PANEL_Z - MOD_H / 2 - 2.5
    return [(0.0, y, zb) for y in (-MOD_W / 2 + 12.0, MOD_W / 2 - 12.0)]


def _top_groove():
    """The groove under the frame's top border that takes the back cover's top hook (lid frame)."""
    z0 = PANEL_Z + MOD_H / 2 + 0.3
    return Pos(0.7, 0, z0 + 1.0) * Box(1.6, MOD_W - 20.0, 2.0)


@cache
def lid_back():
    """The lid's back cover (audit B6: the panel had nothing holding it): a 1.6 mm plate over the module pocket with
    rounded corners, a hook along its top edge into the frame's groove and two M2 screws at the bottom, a notch at the bottom for the HDMI ribbon and the panel's
    USB-C lead. Lid frame; printed flat."""
    from cad.v2.finish import RBox
    z_lo, z_hi = 2 * KNUCKLE_R + 1.0, KNUCKLE_R + LID_H - 1.0            # above the knuckles to under the top edge
    plate = Pos(-BACK_T / 2, 0, (z_lo + z_hi) / 2) * RBox(BACK_T, MOD_W + 6.0, z_hi - z_lo, r=0.6)
    notch = Pos(-BACK_T / 2, 0, PANEL_Z - MOD_H / 2 + 4.0) * Box(BACK_T + 1.0, CABLE_SLOT[0], 8.0)
    holes = None
    for _x, sy, sz in back_screws():
        h = Pos(-BACK_T / 2, sy, sz) * Rot(0, 90, 0) * Cylinder(M2_CLEAR / 2, BACK_T + 1.0)
        holes = h if holes is None else holes + h
    z0 = PANEL_Z + MOD_H / 2 + 0.3                       # the hook along the top edge, into the frame's groove
    hook = Pos(0.6, 0, z0 + 1.0) * Box(1.2, MOD_W - 21.0, 1.6) + Pos(-0.1, 0, z0 + 1.0) * Box(1.6, MOD_W - 21.0, 1.6)
    return plate + hook - notch - holes


@cache
def shell_knuckle():
    """The shell's centre knuckle (printed onto the top cover), in the lid frame at tilt 0 (it does not tilt)."""
    k = Pos(LID_T / 2, 0, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(KNUCKLE_R, KNUCKLE_SHELL[1] - KNUCKLE_SHELL[0])
    foot = Pos(LID_T / 2, 0, KNUCKLE_R / 2) * Box(2 * KNUCKLE_R, KNUCKLE_SHELL[1] - KNUCKLE_SHELL[0], KNUCKLE_R + 6.0,
                                                 align=(Align.CENTER, Align.CENTER, Align.MAX))
    inserts = None
    for sgn in (-1, 1):                         # an M3 heat-set insert in each end of the shell's knuckle
        y0 = sgn * KNUCKLE_SHELL[1]
        c = Pos(LID_T / 2, y0 - sgn * BOLT_INTO / 2, KNUCKLE_R) * Rot(90, 0, 0) * Cylinder(M3_INSERT / 2, BOLT_INTO)
        inserts = c if inserts is None else inserts + c
    slot = Pos(LID_T / 2, 0, KNUCKLE_R) * Box(4.0, CABLE_SLOT[0], 2 * KNUCKLE_R + 2)
    return k + foot - inserts - slot


if __name__ == "__main__":
    f = lid_frame()
    bb = f.bounding_box()
    print(f"lid: {f.volume / 1000:.1f} cm3, {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm; hinge at "
          f"{hinge_location().position}")
    k = shell_knuckle()
    print(f"shell knuckle: {k.volume / 1000:.1f} cm3")
