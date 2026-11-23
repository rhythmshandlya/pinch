"""The STS3032 arm servo's geometry for the CAD (mm) + small solid helpers.

Servo frame: output shaft = +Z through the origin (horn on the +Z face), case length along X with the output end at +X,
width along Y, Z = 0 at the case mid-plane (case faces at Z = +-face). The STS3032 screws down through its mounting
ears into a ledge ("ears"); the profile also describes servos screwed into their case faces ("faces").

Source:
  STS3032 - Feetech's official drawing (feetechrc.com product page, ST-3032-C001): case 23 x 12,
            27.5 tall (31.9 to the spline top), shaft 5.71 mm from the output end, ears 32 mm tip to tip,
            12 wide, 1.5 thick (20.5 .. 22 above the case bottom), 4 ear holes dia 2.5: two per ear,
            28 mm apart along the case (8.21 / 19.79 from the shaft). 25T spline, aluminium disc horn.
            The robot's six arm servos (yaw, lift, claw); the legs' STS3215 is in cad/v2/servos.py.
  Anything not on that drawing (horn thickness, the ear holes' spacing across the ear, cable exit)
  is an estimate: check one servo with a calliper before printing.
"""

from dataclasses import dataclass
from math import cos, radians, sin

from build123d import Align, Box, Cylinder, Pos


@dataclass(frozen=True)
class ServoProfile:
    name: str
    L: float                  # case length (X)
    W: float                  # case width (Y)
    T: float                  # case thickness along the shaft (Z)
    x_out: float              # shaft -> output end
    horn_r: float             # horn disc radius
    horn_face: float          # Z of the horn's outer face
    horn_pcd_r: float         # radius of the horn holes we screw links to
    horn_holes: tuple         # angles (deg) of those holes
    horn_screw_clear: float   # clearance hole in the link for horn screws
    dual_horn: bool           # a second (idler) horn on the -Z face that turns with the output
    mount: str                # "faces" | "ears"
    case_holes: tuple = ()    # "faces": (x, y) of the M2 holes on both faces
    ear_len: float = 0.0      # "ears": how far ears stick out past each case end
    ear_z0: float = 0.0       # "ears": underside Z of the ears
    ear_t: float = 0.0
    ear_w: float = 0.0        # "ears": ear width (Y)
    ear_holes_x: tuple = ()   # "ears": X of the ear screw holes
    ear_holes_y: tuple = (0.0,)   # "ears": Y of the holes in each ear (one or two per ear)
    boss_r: float = 2.5       # idler boss on the -Z face (if no idler horn)
    boss_h: float = 1.0
    cable: str = "rear"       # where the lead leaves the case: "rear" (end, low) | "side"
    mass: float = 0.0

    @property
    def x_rear(self):
        return -(self.L - self.x_out)

    @property
    def half_w(self):
        return self.W / 2

    @property
    def face(self):
        return self.T / 2

    @property
    def sweep_r(self):         # keep plates outside this radius around the shaft (turning horn)
        return self.horn_r + 0.4


STS3032 = ServoProfile(
    "STS3032", L=23.0, W=12.0, T=27.5, x_out=5.71,
    horn_r=10.0, horn_face=19.2, horn_pcd_r=7.0, horn_holes=(), horn_screw_clear=2.2,
    dual_horn=False, mount="ears", ear_len=4.5, ear_z0=6.75, ear_t=1.5, ear_w=12.0,
    ear_holes_x=(8.21, -19.79), ear_holes_y=(3.25, -3.25), boss_r=2.5, boss_h=1.0, cable="side",
    mass=0.020)

PROFILES = {p.name: p for p in (STS3032,)}

M2_CLEAR, M2_PILOT = 2.2, 1.7     # M2 clearance / self-tapping pilot in printed plastic
M3_CLEAR, M3_INSERT = 3.4, 4.0    # M3 clearance / brass heat-set insert hole
M2_INSERT = 3.2                   # M2 heat-set insert hole
FIT = 0.3                         # clearance around servo cases (FDM +-0.2 mm, plus unknowns)


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
                                                                  align=(Align.CENTER,) * 3)


def disc(r, z0, z1, x=0.0, y=0.0):
    return Pos(x, y, (z0 + z1) / 2) * Cylinder(r, abs(z1 - z0))


def case(p: ServoProfile):
    """Servo envelope (case + ears or idler boss) - for interference checks and rendering."""
    body = box(p.x_rear, p.x_out, -p.half_w, p.half_w, -p.face, p.face)
    if p.mount == "ears":
        body += box(p.x_rear - p.ear_len, p.x_out + p.ear_len, -p.ear_w / 2, p.ear_w / 2, p.ear_z0, p.ear_z0 + p.ear_t)
    if not p.dual_horn and p.boss_h > 0:
        body += disc(p.boss_r, -p.face - p.boss_h, -p.face)
    return body


def horn(p: ServoProfile, idler=False):
    """Output horn on +Z (or the idler horn on -Z). Turns with the child link."""
    z0, z1 = p.face + 0.3, p.horn_face
    return disc(p.horn_r, -z1, -z0) if idler else disc(p.horn_r, z0, z1)


def horn_hole_xy(p: ServoProfile):
    return [(p.horn_pcd_r * cos(radians(a)), p.horn_pcd_r * sin(radians(a))) for a in p.horn_holes]
