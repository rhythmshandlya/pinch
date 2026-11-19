"""Servo geometry profiles for the CAD (mm) + small solid helpers.

Servo frame (all profiles): output shaft = +Z through the origin (horn on the +Z face),
case length along X with the output end at +X, width along Y, Z = 0 at the case
mid-plane (case faces at Z = +-face).

Two mounting styles:
  "faces": screws into the case faces (ST3215, HD-1910): cradle plates on both faces
  "ears":  screws down through the mounting ears into a ledge (STS3032 / SCS0009 micro servos)

Sources:
  HD-1910 - Feetech dimension diagram on the Evelta listing: case 34 x 20 x 23, shaft 7.5 mm
            off-centre, 4 x M2 holes on 30 x 16 per face, dual-axis horns dia 16 (29 mm over both).
  SCS0009 - Waveshare "SC09 Servo" dimension drawing: case 23.3 x 12.1, 25.25 tall (28.45 to the
            spline top), shaft 5.7 mm off-centre, ears 32.5 mm tip to tip, 11 wide, 1.6 thick, top of
            ear 18.4 above the case bottom, ear holes dia 2 on 28.5 mm. Dual-shaft. The round disc horn
            from the box (about dia 20) is screwed to each printed link.
  STS3032 - Feetech's official drawing (feetechrc.com product page, ST-3032-C001): case 23 x 12,
            27.5 tall (31.9 to the spline top), shaft 5.71 mm from the output end, ears 32 mm tip to tip,
            12 wide, 1.5 thick (20.5 .. 22 above the case bottom), 4 ear holes dia 2.5: two per ear,
            28 mm apart along the case (8.21 / 19.79 from the shaft). 25T spline, aluminium disc horn.
            The robot uses it for every joint (the only small bus servo in stock, 2026-11-18).
  Anything not on those drawings (horn thickness, the ear holes' spacing across the ear, cable exit)
  is an estimate: check one servo with a calliper before printing (hardware/ASSEMBLY.md, step 0).
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


HD1910 = ServoProfile(
    "HD1910", L=34.0, W=20.0, T=23.0, x_out=9.5,
    horn_r=8.0, horn_face=14.5, horn_pcd_r=6.0, horn_holes=(45, 135, 225, 315), horn_screw_clear=2.2,
    dual_horn=True, mount="faces", case_holes=((7.5, 8.0), (7.5, -8.0), (-22.5, 8.0), (-22.5, -8.0)),
    cable="side", mass=0.021)

SCS0009 = ServoProfile(
    "SCS0009", L=23.3, W=12.1, T=25.25, x_out=5.95,
    horn_r=10.0, horn_face=16.6, horn_pcd_r=6.5, horn_holes=(), horn_screw_clear=1.9,
    dual_horn=False, mount="ears", ear_len=4.6, ear_z0=4.175, ear_t=1.6, ear_w=11.0,
    ear_holes_x=(8.55, -19.95), boss_r=2.5, boss_h=1.0, cable="rear", mass=0.0132)

STS3032 = ServoProfile(
    "STS3032", L=23.0, W=12.0, T=27.5, x_out=5.71,
    horn_r=10.0, horn_face=19.2, horn_pcd_r=7.0, horn_holes=(), horn_screw_clear=2.2,
    dual_horn=False, mount="ears", ear_len=4.5, ear_z0=6.75, ear_t=1.5, ear_w=12.0,
    ear_holes_x=(8.21, -19.79), ear_holes_y=(3.25, -3.25), boss_r=2.5, boss_h=1.0, cable="side",
    mass=0.020)

# GOTECK GS-4308MG 4.3 g metal-gear PWM micro servo (Evelta, 14 in stock 2026-11-18): the arms' motors.
# Listing: 20.22 x 8.5 x 24 mm overall (ears and spline included), 0.58-0.65 kg*cm, 180 deg.
# Case, shaft and ear positions below are ESTIMATES from that and the product photo (no drawing
# published) - measure one before printing the arm parts (hardware/ASSEMBLY.md, stage 0).
GS4308 = ServoProfile(
    "GS4308", L=15.8, W=8.5, T=17.6, x_out=4.5,
    horn_r=5.5, horn_face=16.0, horn_pcd_r=0.0, horn_holes=(), horn_screw_clear=1.6,
    dual_horn=False, mount="ears", ear_len=2.2, ear_z0=5.6, ear_t=1.5, ear_w=8.5,
    ear_holes_x=(5.6, -12.4), ear_holes_y=(0.0,), boss_r=0.0, boss_h=0.0, cable="side",
    mass=0.0058)

PROFILES = {p.name: p for p in (HD1910, SCS0009, STS3032, GS4308)}

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
