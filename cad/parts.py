"""Printable parts of the hand-sized crab, built with build123d (open-source code-CAD). Units: mm.

Every part is built in the frame of the simulator body it belongs to (sim/build_model.py),
so the same geometry drives the print files, the simulator's meshes and masses, and the
interference checks.

Servos (cad/servo.py): 20 Feetech STS3032 micro serial-bus servos, one kind for every joint.
  * each servo hangs from its mounting ears on a printed "ear ledge" (4 screws)
  * hips: a hub on the yaw servo's horn, a spine, and a ledge holding the thigh servo hanging down
  * links are screwed to the servo's round aluminium disc horn
Arms: 3 tiny GS-4308MG PWM servos each (swing, lift, claw). The claw motor sits inside the palm and
its shaft is the moving finger's hinge, at the front of the palm, like a real crab's claw.
Print rules (robu.in FDM service): walls >= 1.6 mm, 0.2-0.3 mm clearance around servo cases,
largest part well under 35 cm. Left and right limbs use the same parts (the right side is
the mirror image), except the moving claw jaw (dactyl), which comes in L and R.

Body: a chassis tub (floor, servo deck, walls, battery bay with a belly door) plus a dome-shaped
top shell. All wiring runs inside: servo leads drop through the deck, the battery sits under it,
the computer (Radxa ZERO 3W, Pi Zero size) rides on a tray under the dome.
"""

import sys
from math import cos, pi, radians, sin, tan
from pathlib import Path

from build123d import Cylinder, Location, Plane, Polygon, Pos, Rot, Sphere, extrude, mirror
from build123d import scale as b3d_scale

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad.servo import FIT, GS4308 as GS, M2_CLEAR, M2_PILOT, STS3032 as SC, box, disc  # noqa: E402
from crab.params import PARAMS  # noqa: E402

MM = 1000.0
LC = PARAMS.leg.coxa_len * MM          # 22  yaw -> femur axis
LF = PARAMS.leg.femur_len * MM         # 30  femur -> knee axis
LT = PARAMS.leg.tibia_len * MM         # 52  knee -> foot centre
FOOT_R = PARAMS.leg.foot_radius * MM   # 5
LS = PARAMS.arm.shoulder_offset * MM   # 16  shoulder yaw -> pitch axis
LU = PARAMS.arm.upper_len * MM         # 30  pitch axis -> claw hinge
FINGER = PARAMS.arm.finger_len * MM    # 24  claw hinge -> finger tips
ARM_Z = PARAMS.arm_mount[0][2] * MM    # 4  shoulder frame height in the body

PLATE = 2.5          # horn plates and link plates
LEDGE = 3.0          # ear ledges (the ear screws bite 3 mm of plastic)
HUB_R = 10.5         # hub over the servo's disc horn (r 10)
PILOT = 1.5          # pilot hole for the servo's own self-tapping ear screws
WIN_X, WIN_Y = 0.13, 0.3  # clearance around a servo case in its ledge window (ends, sides). The ends
                          # are tight: the ear holes sit only 2.5 mm from the case (see ASSEMBLY.md)
LEDGE_SIDE = SC.half_w + WIN_Y + 2.0   # 8.3 half-width of an ear ledge
GS_RIM = 2.6         # tiny servos: solid rim under each ear (the servo's screws go straight in)
GS_SIDE = GS.half_w + WIN_Y + 1.8      # 6.35 half-width of a tiny-servo pocket
GS_HUB_R = GS.horn_r + 0.5             # 6.0 hub over a tiny servo's horn
GS_PLATE = 2.0

# servo heights / offsets along their shafts (chosen so that consecutive layers never meet)
H_YAW = 6.0                          # top of the yaw servos' horns (body z); all six yaw servos alike
Z_YAW = H_YAW - SC.horn_face         # -13.2 yaw servo origin (case mid-plane)
Z_DECK_TOP = Z_YAW + SC.ear_z0      # -6.45 the servo deck: every yaw servo's ears rest on it
H_SH = Z_DECK_TOP - GS.ear_z0 + GS.horn_face - ARM_Z   # -0.05 arm-swing horn top (shoulder frame)
Y_F = -4.0                           # thigh servo shifted inward
Y_K = Y_F + SC.horn_face - 0.45 - SC.face   # 1.0 knee servo as close to the thigh plate as it fits
Y_PA = -3.0                          # arm-lift servo shifted inward
Z_CA = -3.0                          # claw servo mid-plane (arm frame)


def at(origin, x_dir, z_dir):
    """Servo frame -> link frame (servo +Z = output shaft, +X = toward the output end)."""
    return Location(Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))


# where each servo sits, in the frame of the link that carries its case
L_YAW = at((0, 0, Z_YAW), (1, 0, 0), (0, 0, 1))          # coxa/body frame
L_FEMUR = at((LC, Y_F, 0), (0, 0, 1), (0, 1, 0))          # coxa frame: thigh servo hanging, horn +y
L_KNEE = at((LF, Y_K, 0), (0, 0, -1), (0, -1, 0))         # femur frame: case points up, horn -y
L_SHYAW = at((0, 0, H_SH - GS.horn_face), (1, 0, 0), (0, 0, 1))   # shoulder frame (tiny servo)
L_PITCH = at((LS, Y_PA, 0), (0, 0, 1), (0, 1, 0))         # shoulder frame: hanging, horn +y
L_CLAW = at((LU, 0, Z_CA), (1, 0, 0), (0, 0, 1))          # arm frame: in the palm, case back, horn up

# the same servos' horns, in the frame of the link that turns with them
H_LOC = {
    "yaw": L_YAW,
    "femur": at((0, Y_F, 0), (0, 0, 1), (0, 1, 0)),
    "knee": at((0, Y_K, 0), (0, 0, -1), (0, -1, 0)),
    "shyaw": L_SHYAW,
    "pitch": at((0, Y_PA, 0), (0, 0, 1), (0, 1, 0)),
    "claw": at((0, 0, Z_CA), (1, 0, 0), (0, 0, 1)),
}


# ---------------------------------------------------------------- helpers
def cyl_y(r, y0, y1, x, z):
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(r, abs(y1 - y0))


def cyl_x(r, x0, x1, y, z):
    return Pos((x0 + x1) / 2, y, z) * Rot(0, 90, 0) * Cylinder(r, abs(x1 - x0))


def sorted2(a, b):
    return (a, b) if a <= b else (b, a)


def ear_ledge(t=LEDGE):
    """Plate under a micro servo's ears with a window for its case (servo frame).
    The servo drops in from the horn side; its two ear screws bite into this plate."""
    x0 = SC.ear_holes_x[1] - PILOT / 2 - 1.75        # -22.5
    x1 = SC.ear_holes_x[0] + PILOT / 2 + 1.85        # 11.2
    z1 = SC.ear_z0
    led = box(x0, x1, -LEDGE_SIDE, LEDGE_SIDE, z1 - t, z1)
    led -= box(SC.x_rear - WIN_X, SC.x_out + WIN_X, -SC.half_w - WIN_Y, SC.half_w + WIN_Y, z1 - t - 1, z1 + 1)
    for hx in SC.ear_holes_x:
        for hy in SC.ear_holes_y:
            led -= disc(PILOT / 2, z1 - t - 1, z1 + 1, hx, hy)
    return led


def ledge_extent():
    """(x0, x1) of an ear ledge in the servo frame."""
    return SC.ear_holes_x[1] - PILOT / 2 - 1.75, SC.ear_holes_x[0] + PILOT / 2 + 1.85


def gs_extent():
    """(x0, x1) of a tiny servo's pocket (case + clearance + the rims under its ears), servo frame."""
    return GS.x_rear - WIN_X - GS_RIM, GS.x_out + WIN_X + GS_RIM


def gs_pocket(z_bottom):
    """Snug pocket for a GS-4308MG (servo frame): the case drops in from the top, its ears rest on the
    rim at ear height and its own self-tapping screws go straight into the rim through the ear holes.
    z_bottom: where the pocket's walls stop (open bottom) - pass None for a closed cup with a floor."""
    from shapely.geometry import box as sbox
    x0, x1 = gs_extent()
    outline = sbox(x0, -GS_SIDE, x1, GS_SIDE).buffer(-1.8, 8).buffer(1.8, 8)   # rounded corners
    floor = -GS.face - 0.4 - 2.0
    zb = floor if z_bottom is None else z_bottom
    p = prism(outline, zb, GS.ear_z0)
    cav_z0 = -GS.face - 0.4 if z_bottom is None else zb - 1
    p -= box(GS.x_rear - WIN_X, GS.x_out + WIN_X, -GS.half_w - WIN_Y, GS.half_w + WIN_Y, cav_z0, GS.ear_z0 + 1)
    if z_bottom is None:                                         # lead exits low at the back
        p -= box(x0 - 1, GS.x_rear + 0.5, -2.5, 2.5, cav_z0, cav_z0 + 4)
    return p


# ---------------------------------------------------------------- hips and shoulders
SPINE = (12.9, LC - SC.half_w - WIN_Y)   # x span of the hip bracket's spine (15.7): clears the yaw ears


def hip_bracket(z_hub, y_off):
    """Screwed to a yaw servo's disc horn (hub at z_hub), holds the next servo hanging down by its
    ears, shaft along +y. Frame: yaw axis at origin; that servo's axis at (LC, y_off, 0)."""
    z0, z1 = z_hub, z_hub + PLATE
    lx0, lx1 = ledge_extent()
    ly0 = y_off + SC.ear_z0 - LEDGE                        # case-side face of the ledge
    p = disc(HUB_R, z0, z1) + box(0, SPINE[1], -LEDGE_SIDE, 8.0, z0, z1)
    p += box(SPINE[0], SPINE[1], -LEDGE_SIDE, ly0 + LEDGE, lx0, z1)    # spine down beside the servo
    p += at((LC, y_off, 0), (0, 0, 1), (0, 1, 0)) * ear_ledge()
    p -= disc(2.6, z0 - 1, z1 + 1)                          # horn screw
    return p


def coxa_link():
    """Hip: on the leg's yaw servo, holds the thigh servo. Frame: yaw axis at origin, x radial out."""
    return hip_bracket(H_YAW, Y_F)


YF0, YF1 = Y_F + SC.horn_face, Y_F + SC.horn_face + PLATE   # 15.2 .. 17.7 thigh plate on the horn
KX0, KX1 = LF - LEDGE_SIDE, LF + LEDGE_SIDE      # knee pocket


def femur_link():
    """Thigh: one plate on the thigh servo's horn (+y), ending in a pocket that holds the knee servo
    standing upright (horn on -y). Frame: femur axis at origin, x along the thigh."""
    lx0, lx1 = ledge_extent()
    kz0, kz1 = -lx1, -lx0                                     # ledge span in z (servo x_s -> -z)
    ly1 = Y_K - SC.ear_z0 + LEDGE                             # thigh-side face of the knee ledge
    p = cyl_y(10.5, YF0, YF1, 0, 0) + box(0, KX0 + 1, YF0, YF1, -8, 8)
    p += box(KX0, KX1, YF0, YF1, kz0, kz1)                    # pocket back (closes the case bottom)
    for x0, x1 in ((KX0, KX0 + 2.0), (KX1 - 2.0, KX1)):       # pocket side walls
        p += box(x0, x1, ly1 - 0.5, YF0 + 0.5, kz0, kz1)
    p += L_KNEE * ear_ledge()
    p -= cyl_y(2.6, YF0 - 1, YF1 + 1, 0, 0)                   # horn screw
    p -= cyl_y(SC.boss_r + 0.7, YF0 - 1, YF1 + 1, LF, 0)        # knee servo's idler boss
    return p


YT1 = Y_K - SC.horn_face          # -18.2 knee horn face
YT0 = YT1 - PLATE                 # -20.7


def tibia():
    """Shin: screwed to the knee servo's disc horn (-y), jogs to the leg's centre plane, ends in a
    stub for the TPU foot. Frame: knee axis at origin, x along the shin."""
    p = cyl_y(HUB_R, YT0, YT1, 0, 0) + box(0, 18, YT0, YT1, -6, 6)
    p += box(14, 18, YT0, 3.5, -5, 5)                         # jog (clears the knee servo when folded)
    p += box(14, LT - 12, -3.5, 3.5, -4.5, 4.5)               # shin
    p += box(LT - 13, LT - 2.5, -2.8, 2.8, -2.8, 2.8)         # foot stub
    p -= cyl_y(2.6, YT0 - 1, YT1 + 1, 0, 0)                   # horn screw
    p -= Pos(LT - 7.5, 0, 0) * Cylinder(M2_CLEAR / 2, 20)     # foot screw (M2 x 10)
    return p


def foot_tip():
    """TPU foot: soft ball over the shin stub. Frame: tibia frame."""
    tip = Pos(LT, 0, 0) * Sphere(FOOT_R) + box(LT - 11.8, LT - 2, -4.7, 4.7, -4.7, 4.7)
    tip -= box(LT - 12.8, LT - 2.3, -3.0, 3.0, -3.0, 3.0)
    tip -= Pos(LT - 7.5, 0, 0) * Cylinder(M2_CLEAR / 2, 20)
    return tip


# ---------------------------------------------------------------- arms
SPINE_A = (8.7, LS - GS.half_w - WIN_Y)     # shoulder spine x span (11.45): clears the swing servo's ears
YU0, YU1 = Y_PA + GS.horn_face, Y_PA + GS.horn_face + PLATE   # 13.0 .. 15.5 arm plate on the lift horn
PALM_X0 = LU + gs_extent()[0]               # 15.97 back of the palm
PALM_X1 = LU + gs_extent()[1]               # 37.23 front of the palm
D_DROP = (11.5, 15.0)                       # moving finger's drop (its frame): clear of the palm's front corners
D_DROP_Y = (4.6, 8.4)                       # ... and of the gap between the jaws
# Jaws (decision 24). Blades that hang ~11 mm below the palm, so a pen lying under the palm clears it, with
# lower edges that lie flat on the desk when the arm points GRIP_PITCH down. They pinch a round thing at its
# widest point (4.5 mm up, for a pen) along their whole length, instead of above it (which squeezed pens
# out of the old jaws as they rose). TPU pads on the inner faces grip and resist twisting.
GRIP_PITCH = 20.0                           # deg below horizontal (arm, in the world) for flat jaws
JAW_X = (40.3, 55.5)                        # gripping strip, arm frame x
GRIP_W = 9.0                                # the jaws are parallel when this far apart (a pen) ...
CLAW_PARALLEL = 19.0                        # ... at this claw angle (deg); at 0 their tips just meet
JAW_LOW = -24.5                             # arm-frame z of the blades' lower edge at JAW_X[0]
STRIP_H = 10.0                              # grip pads' height (arm-frame z)
PAD_T = 1.2                                 # TPU grip pad on each jaw's inner face
BLADE_T = 2.4                               # PETG blade behind the pad


def jaw_low(x):
    """Arm-frame z of the jaw blades' lower edge at arm-frame x (rises toward the tip, so it is level with
    the desk when the arm points GRIP_PITCH down)."""
    return JAW_LOW + (x - JAW_X[0]) * tan(radians(GRIP_PITCH))


def jaw_side(x_back, y0, y1, x_shift=0.0):
    """A jaw blade: side profile in the arm's x-z plane, extruded across y0..y1. From x_back (the palm
    front, or the moving finger's drop) to the tip; the lower edge follows jaw_low(), the top edge falls
    gently to a rounded-off tip. x_shift moves it into another frame (the moving finger's: -LU)."""
    xf = PALM_X1 - 1.0
    x1 = JAW_X[1]

    def top(x):
        return -1.0 - (x - xf) * 0.22

    pts = [(x_back, top(x_back)), (x1 - 3.0, top(x1 - 3.0)), (x1, top(x1 - 3.0) - 3.0),
           (x1, jaw_low(x1)), (JAW_X[0], jaw_low(JAW_X[0]))]
    if x_back < JAW_X[0] - 0.1:                              # under the palm: 45 deg back up to it
        pts.append((x_back, jaw_low(JAW_X[0]) + (JAW_X[0] - x_back)))
    return xz_slab([(x + x_shift, z) for x, z in pts], y0, y1)


def xz_slab(pts, y0, y1):
    """A plate: outline pts in the x-z plane, spanning y0..y1 (whichever way round the outline goes)."""
    s = extrude(Plane.XZ * Polygon(*pts, align=None), y1 - y0)
    return Pos(0, y0 - s.bounding_box().min.Y, 0) * s


def jaw_pad_outline(x_shift=0.0):
    """The grip pad's outline (arm frame x, z): the bottom STRIP_H of the blade, 0.4 mm in from its edges."""
    xa, xb = JAW_X[0] + 0.4, JAW_X[1] - 0.4
    return [(x + x_shift, z) for x, z in ((xa, jaw_low(xa) + 0.4), (xb, jaw_low(xb) + 0.4),
                                         (xb, jaw_low(xb) + 0.4 + STRIP_H), (xa, jaw_low(xa) + 0.4 + STRIP_H))]


def jaw_pad():
    """TPU grip pad (4 per robot, superglued on): where it sits on the left arm's fixed finger (arm
    frame, inner face 0.5 mm off the claw's centre plane); the assembly places the other three."""
    return xz_slab(jaw_pad_outline(), -0.5 - PAD_T, -0.5)


def shoulder():
    """Shoulder: screwed to the arm-swing servo's horn, holds the lift servo hanging down in a pocket.
    Frame: swing axis at origin; lift axis at (LS, 0, 0) along y."""
    z0, z1 = H_SH, H_SH + PLATE
    px0, px1 = gs_extent()
    pk_z0 = GS.ear_z0 - 3.0                                  # pocket ring 3 mm deep
    ly0 = Y_PA + pk_z0                                       # its case-side face
    p = disc(GS_HUB_R, z0, z1) + box(0, SPINE_A[1], -GS_SIDE, 5.0, z0, z1)
    p += box(SPINE_A[0], SPINE_A[1], -GS_SIDE, ly0 + 3.0, px0, z1)          # spine down beside the servo
    p += L_PITCH * gs_pocket(pk_z0)
    p -= disc(1.3, z0 - 1, z1 + 1)                           # horn screw
    return p


def upper_arm():
    """Arm + palm + fixed finger: plate on the lift servo's horn (+y), the claw servo sitting in a
    rounded palm cup (shaft up = the moving finger's hinge, at the front), fixed finger in front.
    Frame: lift axis at origin, x along the arm."""
    cup = L_CLAW * gs_pocket(None)
    bb = cup.bounding_box()
    p = cyl_y(GS_HUB_R, YU0, YU1, 0, 0) + box(0, PALM_X0 + 7, YU0, YU1, -6, 6)
    p += box(PALM_X0, PALM_X0 + 7, GS_SIDE - 0.6, YU0 + 0.5, bb.min.Z, bb.max.Z)   # web: plate -> palm
    p += cup
    xf = PALM_X1 - 1.0                                       # fixed finger grows out of the palm front
    p += jaw_side(xf, -0.5 - PAD_T - BLADE_T, -0.5 - PAD_T)   # blade (its TPU pad glues on the inside)
    p -= cyl_y(1.3, YU0 - 1, YU1 + 1, 0, 0)                  # horn screw
    return p


ZD0 = Z_CA + GS.horn_face             # 13.0 moving finger's plate on the claw horn


def to_dactyl(solid):
    """Arm frame, claw at CLAW_PARALLEL -> the moving finger's frame (hinge at origin)."""
    return Rot(0, 0, -CLAW_PARALLEL) * Pos(-LU, 0, 0) * solid


def mirror_jaw(solid):
    """Fixed jaw -> where the moving jaw is when the claw is CLAW_PARALLEL open (arm frame): the mirror image
    across the plane midway between the pads when they are GRIP_W apart."""
    return Pos(0, GRIP_W - 1.0, 0) * mirror(solid, about=Plane.XZ)


def dactyl(side=1):
    """Moving finger: plate on the claw servo's horn (the hinge), a drop in front of the palm, and a jaw blade.
    A single pivot closes like scissors, so on a long thing (a pen) plain jaws touch only at the narrow end
    of the V and it twists out. This blade is set at an angle so the two jaws are parallel when GRIP_W apart
    (at CLAW_PARALLEL): a pen is pinched along the whole jaw; the tips meet at 0.
    Frame: hinge at origin. side=+1 left arm, -1 right (mirror image)."""
    p = disc(GS_HUB_R, ZD0, ZD0 + GS_PLATE) + box(0, D_DROP[1], -4.0, D_DROP_Y[1], ZD0, ZD0 + GS_PLATE)
    p += box(D_DROP[0], D_DROP[1], D_DROP_Y[0], D_DROP_Y[1], -6.0, ZD0 + GS_PLATE)   # drop, into the blade
    p += to_dactyl(mirror_jaw(jaw_side(JAW_X[0], -0.5 - PAD_T - BLADE_T, -0.5 - PAD_T)))  # (pad glues inside)
    p -= disc(1.3, ZD0 - 1, ZD0 + GS_PLATE + 1)
    return p if side > 0 else mirror(p, about=Plane.XZ)


# ---------------------------------------------------------------- body
def yaw_mounts():
    """(name, x, y, z, yaw_deg) of the 6 yaw servo frames in the body frame (mm); right side mirrored."""
    out = []
    for lg in ("FL", "FR", "RL", "RR"):
        (x, y, z), yaw = PARAMS.leg_mounts[lg[0]]
        s = 1 if lg[1] == "L" else -1
        out.append((lg, x * MM, s * y * MM, z * MM, s * yaw * 180 / pi))
    (x, y, z), yaw = PARAMS.arm_mount
    for a, s in (("L", 1), ("R", -1)):
        out.append((a, x * MM, s * y * MM, z * MM, s * yaw * 180 / pi))
    return out


def yaw_servo_loc(name, x, y, z, yaw):
    """Location of a yaw servo's frame in the body frame."""
    return Pos(x, y, z) * Rot(0, 0, yaw) * (L_SHYAW if name in ("L", "R") else L_YAW)


def yaw_profile(name):
    """Legs: STS3032 bus servos; arm swing: GS-4308MG."""
    return GS if name in ("L", "R") else SC


# body heights (all six yaw servos share them)
Z_FLOOR = (-29.8, -28.0)                                   # floor plate (bay deep enough for an 18 mm pack)
Z_DECK = (Z_YAW + SC.ear_z0 - LEDGE, Z_YAW + SC.ear_z0)     # servo deck (-9.45 .. -6.45)
Z_SLOT = 1.8                                               # walls stop here where hip plates swing
Z_RIM = 9.5                                                # chassis rim = shell skirt
WALL = 1.8
BATTERY = (30.5, 57.5, 18.0)      # 2S LiPo up to 57.5 x 30.5 x 18 mm, lying across the body (BOM: 850 mAh, 57 x 30)
BAY = (-15.6, 15.6, -29.0, 29.0)  # battery bay under the deck: 0.7 mm clear of the leg servo cases
CAMERA_Z = 1.0                    # lens height: in the chassis front wall, between the arm servos
CAM_HEAD = 12.0                   # pocket for the Pi Zero mini camera head (11.5 mm wide strip)
PI_ZERO = (65.0, 30.0)            # board size; holes 58 x 23 (M2.5)
PI_AT = (5.5, 0.0)                # Pi centre on the tray (camera connector toward the front)
MAGNETS = [(-30.0, 24.0), (-30.0, -24.0), (1.0, 43.0), (1.0, -43.0)]    # shell <-> chassis, 6 x 3 mm
MAG_R = 4.8                       # magnet boss radius (pocket 6.3)
TRAY_POSTS = [(-30.0, 0.0), (20.0, 0.0), (0.0, 20.0), (0.0, -20.0)]       # Pi tray rests on these
TRAY_Z = (Z_RIM, Z_RIM + 2.0)
SPEAKER = (-15.0, 0.0, 14.2)      # x, y, radius of the 28 mm speaker seat inside the dome (above the Pi)
SWITCH = (-26.0, 0.0, 13.4, 19.4) # power switch cutout in the floor (behind the battery): x, y, size x, size y
MIC = (24.0, -20.0)               # microphone board on the tray, under a 1.6 mm port in the dome
TOF = (-8.0, -17.0)               # x, z of the distance-sensor windows in the side walls (y ~ +-51)
FACE_TILT = 25.0                  # deg the face screen leans back (it looks up at you)
FACE_CENTRE = (36.0, 0.0, 36.0)   # centre of the OLED module, body frame (mm): lower edge ~16 mm up
FACE_MODULE = (61.5, 39.5)        # 2.42" SSD1309 module (width, height)
FACE_ACTIVE = (55.0, 27.5)        # its lit area
FACE_HOUSING = (67.5, 45.0)       # visor outline
FACE_HOLES = [(sy * 28.25, sz * 17.25) for sy in (-1, 1) for sz in (-1, 1)]   # module's M2 holes (ESTIMATE)
FACE_GLASS_T = 1.5                # glass in front of the module's board (ESTIMATE): standoffs stop at it
FACE_LOC = Pos(*FACE_CENTRE) * Rot(0, -FACE_TILT, 0)     # local x = out of the screen, y = across, z = up


def _xf(pts, x, y, yaw_deg):
    c, s = cos(radians(yaw_deg)), sin(radians(yaw_deg))
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in pts]


def _ellipse2d(a, b, n=240, cx=0.0):
    from shapely.geometry import Polygon as SPoly
    return SPoly([(cx + a * cos(2 * pi * k / n), b * sin(2 * pi * k / n)) for k in range(n)])


def opened(geom, r=1.0):
    """Morphological opening: removes any feature narrower than 2r, keeps the largest piece."""
    g = geom.buffer(-r, 8).buffer(r, 8)
    g = max(getattr(g, "geoms", [g]), key=lambda p: p.area)
    return g.simplify(0.06, preserve_topology=True)        # few, long edges: fast, valid booleans


def opened_all(geom, r=1.0, min_area=4.0):
    """Opening that keeps every piece bigger than min_area (mm^2)."""
    g = geom.buffer(-r, 8).buffer(r, 8)
    return [p.simplify(0.06, preserve_topology=True) for p in getattr(g, "geoms", [g]) if p.area > min_area]


def to_face(poly):
    """Shapely polygon (with holes) -> build123d face, counter-clockwise so it extrudes upward."""
    from shapely.geometry.polygon import orient
    poly = orient(poly, sign=1.0)
    f = Polygon(*list(poly.exterior.coords)[:-1], align=None)
    for ring in poly.interiors:
        f -= Polygon(*list(ring.coords)[:-1], align=None)
    return f


def prism(poly, z0, z1):
    return Pos(0, 0, z0) * extrude(to_face(poly), z1 - z0)


def _yaw_range_deg(name):
    """Angles (deg, in the mount's own frame) a hip / shoulder bracket swings through.
    The sign follows each joint's axis (see sim/build_model.py)."""
    from math import degrees
    if name in ("L", "R"):
        lo, hi = PARAMS.arm.shoulder_yaw_range
        s = 1 if name == "L" else -1           # arm yaw axis is +z on the left, -z on the right
    else:
        lo, hi = PARAMS.leg.coxa_range_for(name)
        s = -1 if name[1] == "L" else 1        # coxa axis is -z on the left, +z on the right
    return tuple(sorted((s * degrees(lo), s * degrees(hi))))


def _footprints(name, level):
    """Top view (mount frame, left-side orientation) of what a hip / shoulder carries.
    level="low":  everything hanging below the hub (cradle, thigh servo, pitch servo, links)
    level="high": the hub plate and its arm as well."""
    from shapely.geometry import Point
    from shapely.geometry import box as sbox
    from shapely.ops import unary_union
    # a bracket + a servo hanging from it + a link on that servo's horn
    if name in ("L", "R"):                                   # arm: tiny servos
        y0, y1 = Y_PA - GS.face - 0.3, Y_PA + GS.face + 0.3
        low = [sbox(SPINE_A[0], -GS_SIDE, 60, y1), sbox(SPINE_A[1], y0, 60, y1),
               sbox(LS - GS_HUB_R - 0.5, y1, 60, YU1 + 0.4)]
        high = low + [Point(0, 0).buffer(GS_HUB_R, 32), sbox(0, -GS_SIDE, SPINE_A[1], 5.0)]
    else:                                                    # leg
        y0, y1 = Y_F - SC.face - 0.3, Y_F + SC.face + 0.3
        low = [sbox(SPINE[0], -LEDGE_SIDE, 80, y1), sbox(SPINE[1], y0, 80, y1),
               sbox(LC - HUB_R - 0.5, y1, 80, YF1 + 0.4)]
        high = low + [Point(0, 0).buffer(HUB_R, 32), sbox(0, -LEDGE_SIDE, SPINE[1], 8.0)]
    return unary_union(low if level == "low" else high)


def keepout2d(level, margin=0.8):
    """Top view of everything the 6 hip / shoulder brackets sweep through (body frame)."""
    from shapely.affinity import rotate, scale, translate
    from shapely.ops import unary_union
    parts = []
    for name, x, y, _, yaw in yaw_mounts():
        foot = _footprints(name, level)
        if y < 0:
            foot = scale(foot, 1, -1, origin=(0, 0))          # right side: mirror image
        a, b = _yaw_range_deg(name)
        steps = max(2, int((b - a) / 2) + 1)
        for k in range(steps):
            ang = yaw + a + (b - a) * k / (steps - 1)
            parts.append(translate(rotate(foot, ang, origin=(0, 0)), x, y))
    return unary_union(parts).buffer(margin, 8).simplify(0.06, preserve_topology=True)


def servo_footprints2d(grow=0.0, ears=True):
    """Top view of the 6 yaw servo cases (+ ears) in the body frame."""
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    out = []
    for name, x, y, _, yaw in yaw_mounts():
        p = yaw_profile(name)
        x0, x1, h = p.x_rear - grow, p.x_out + grow, p.half_w + grow
        out.append(SPoly(_xf([(x0, -h), (x1, -h), (x1, h), (x0, h)], x, y, yaw)))
        if ears:
            e0, e1, eh = p.x_rear - p.ear_len - grow, p.x_out + p.ear_len + grow, p.ear_w / 2 + grow
            out.append(SPoly(_xf([(e0, -eh), (e1, -eh), (e1, eh), (e0, eh)], x, y, yaw)))
    return unary_union(out)


def ledges2d():
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    out = []
    for name, x, y, _, yaw in yaw_mounts():
        (x0, x1), h = (gs_extent(), GS_SIDE) if name in ("L", "R") else (ledge_extent(), LEDGE_SIDE)
        out.append(SPoly(_xf([(x0, -h), (x1, -h), (x1, h), (x0, h)], x, y, yaw)))
    return unary_union(out)


def body_outline():
    """Top view of the chassis: oval + the six servo ledges, minus the bracket sweeps."""
    base = _ellipse2d(40.0, 55.0).union(ledges2d()).convex_hull
    return opened(base.difference(keepout2d("low")), 1.0)


def shell_planform():
    """Top view of the shell: like the chassis, but it also covers the hip / shoulder hubs
    (they turn below the shell's rim)."""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    hubs = unary_union([Point(x, y).buffer(12.0, 32) for _, x, y, _, _ in yaw_mounts()])
    base = _ellipse2d(40.0, 55.0).union(ledges2d()).union(hubs).convex_hull
    return opened(base.difference(keepout2d("low", margin=1.2)), 1.2)


def chassis():
    """Body tub: floor with the battery door opening, deck holding the 6 yaw servos by their ears,
    walls up to the rim; posts for the Pi, magnets for the shell."""
    from shapely.geometry import box as sbox
    outline = body_outline()
    inner = outline.buffer(-WALL, 8)
    mid = opened(outline.difference(keepout2d("high")), 1.0)           # where walls may go higher
    servo_cut = servo_footprints2d(FIT)                                 # cases + ears, with clearance
    body = prism(outline, *Z_FLOOR)
    for g in opened_all(outline.difference(inner).difference(servo_cut), 0.8):
        body += prism(g, Z_FLOOR[0], Z_SLOT)
    for g in opened_all(mid.difference(mid.buffer(-WALL, 8)).difference(servo_cut), 0.8):
        body += prism(g, Z_SLOT, Z_RIM)
    # deck: one plate at ear height, windows for the servo cases, pilot holes for the ear screws
    deck = outline.buffer(-0.5, 8).simplify(0.06)
    body += prism(deck, *Z_DECK)
    for name, x, y, z, yaw in yaw_mounts():
        loc = yaw_servo_loc(name, x, y, z, yaw)
        p = yaw_profile(name)
        body -= loc * box(p.x_rear - WIN_X, p.x_out + WIN_X, -p.half_w - WIN_Y, p.half_w + WIN_Y,
                          p.ear_z0 - LEDGE - 1, p.ear_z0 + 1)
        if p is SC:                                          # bus servos: pilot holes for their ear screws
            for hx in SC.ear_holes_x:                        # (tiny servos: screws go straight in)
                for hy in SC.ear_holes_y:
                    body -= loc * disc(PILOT / 2, SC.ear_z0 - LEDGE - 1, SC.ear_z0 + 1, hx, hy)
        if p.boss_h > 0:
            body -= loc * disc(p.boss_r + 0.6, -p.face - 4, -p.face + 0.01)    # idler boss in the floor
    # cable hole through the deck: servo leads and battery wires meet the electronics above it
    bx0, bx1, by0, by1 = BAY
    body -= box(-12.0, 12.0, -14.0, 14.0, Z_DECK[0] - 1, Z_DECK[1] + 1)
    # camera: lens hole through the front wall, a pocket behind it holds the camera head upright
    xf = outline.bounds[2]
    h, w = CAM_HEAD / 2 + 0.2, CAM_HEAD / 2 + 1.9
    body += box(xf - 9.5, xf - 0.4, -w, w, Z_DECK[1], CAMERA_Z + w) - \
        box(xf - 10.5, xf - WALL, -h, h, CAMERA_Z - h, CAMERA_Z + h)
    body -= cyl_x(3.2, xf - WALL - 1, xf + 5, 0, CAMERA_Z)
    body -= box(xf - 10.5, xf - 8.0, -3.0, 3.0, CAMERA_Z - h, CAMERA_Z + w + 1)   # slot for its flex
    # battery door opening in the floor (door screws to two bosses)
    body -= box(bx0, bx1, by0, by1, Z_FLOOR[0] - 1, Z_FLOOR[1] + 1)
    for y in (by0 - 3.5, by1 + 3.5):
        body += disc(3.0, Z_FLOOR[1], Z_FLOOR[1] + 4, 0, y) - disc(0.85, Z_FLOOR[0] - 1, Z_FLOOR[1] + 4.5, 0, y)
    # posts for the Pi tray (M2 self-tapping from above)
    for px, py in TRAY_POSTS:
        body += disc(3.5, Z_DECK[1], Z_RIM, px, py) - disc(0.85, Z_RIM - 8, Z_RIM + 1, px, py)
    # magnets that hold the shell (6 x 3 mm discs, glued)
    for mx, my in MAGNETS:
        body += disc(MAG_R, Z_DECK[1], Z_RIM, mx, my) - disc(3.15, Z_RIM - 3.2, Z_RIM + 1, mx, my)
    # power switch in the floor, behind the battery (flip it from underneath)
    sx, sy, sw, sh = SWITCH
    body -= box(sx - sw / 2, sx + sw / 2, sy - sh / 2, sy + sh / 2, Z_FLOOR[0] - 1, Z_FLOOR[1] + 1)
    # windows for the two VL53L0X distance sensors, one in each side wall (crabs walk sideways)
    tx, tz = TOF
    for s in (1, -1):
        body -= cyl_y(3.5, s * 40, s * 62, tx, tz)
    sol = body.solids()
    return max(sol, key=lambda s: s.volume)


def pi_holes():
    """Pi Zero 2 W mounting holes (body frame): board centred at PI_AT, long side along x."""
    cx, cy = PI_AT
    return [(cx + sx * 29.0, cy + sy * 11.5) for sx in (-1, 1) for sy in (-1, 1)]


def pi_tray():
    """The 'brain' tray: Raspberry Pi Zero 2 W on 4 stand-offs, speaker seat, amp and mic spots.
    Rests on 4 chassis posts under the dome and lifts out in one piece."""
    from shapely.geometry import Point
    from shapely.geometry import box as sbox
    from shapely.ops import unary_union
    cx, cy = PI_AT
    shape = unary_union([sbox(cx - 34.5, -17.0, cx + 34.5, 17.0), sbox(-22.0, -30.0, 10.0, 30.0)]
                        + [Point(p).buffer(4.5, 32) for p in TRAY_POSTS])
    shape = shape.convex_hull.intersection(shell_planform().buffer(-2.6, 8))
    for mx, my in MAGNETS:
        shape = shape.difference(Point(mx, my).buffer(MAG_R + 0.8, 32))
    shape = opened(shape, 1.0)
    z0, z1 = TRAY_Z
    t = prism(shape, z0, z1)
    for px, py in pi_holes():                                   # stand-offs, M2.5 self-tapping
        t += disc(2.8, z1, z1 + 3.0, px, py) - disc(1.05, z0 - 1, z1 + 3.5, px, py)
    for px, py in TRAY_POSTS:
        t -= disc(M2_CLEAR / 2, z0 - 1, z1 + 1, px, py)
    for x0, x1 in ((cx - 18, cx - 4), (cx + 4, cx + 18)):                        # cable windows
        t -= box(x0, x1, -5.0, 5.0, z0 - 1, z1 + 1)
    for k in range(3):                                          # zip-tie slots for the amp / mic boards
        t -= box(-16 + 6 * k, -14 + 6 * k, -27.0, -23.0, z0 - 1, z1 + 1)
    return max(t.solids(), key=lambda s: s.volume)



def desk_tray():
    """Pinch's desk tray: where pens and small things live (16 x 7 cm inside), a 1 mm floor with a 2 cm
    ramped front lip Pinch can push things up, low walls on three sides, and an upright tab at the back
    for the 40 mm ArUco marker sticker (docs/marker_7.png) that tells Pinch where it is on the desk."""
    ix, iy = 160.0, 70.0
    t = 2.0
    p = box(-ix / 2 - t, ix / 2 + t, -iy / 2, iy / 2 + t, 0, 1.0)                  # floor
    p += box(-ix / 2 - t, ix / 2 + t, iy / 2, iy / 2 + t, 0, 12.0)                 # back wall
    p += box(-ix / 2 - t, -ix / 2, -iy / 2, iy / 2 + t, 0, 12.0)                    # side walls
    p += box(ix / 2, ix / 2 + t, -iy / 2, iy / 2 + t, 0, 12.0)
    ramp = Plane.YZ * Polygon((0, 0), (20.0, 0), (20.0, 1.0), align=None)          # rises 1 mm over 20 mm
    p += Pos(-ix / 2 - t, -iy / 2 - 20.0, 0) * extrude(ramp, ix + 2 * t)
    p += box(-24.0, 24.0, iy / 2, iy / 2 + 3.0, 0, 52.0)                          # marker tab
    p -= box(-20.5, 20.5, iy / 2 - 0.5, iy / 2 + 0.6, 6.0, 46.5)                  # sticker recess (0.5 mm)
    return p


def belly_door():
    """Battery door: screws under the floor (2 x M2)."""
    bx0, bx1, by0, by1 = BAY
    d = box(bx0 - 3, bx1 + 3, by0 - 7, by1 + 7, Z_FLOOR[0] - 1.8, Z_FLOOR[0])
    for y in (by0 - 3.5, by1 + 3.5):
        d -= disc(M2_CLEAR / 2, Z_FLOOR[0] - 3, Z_FLOOR[0] + 1, 0, y)
    d -= box(bx1 - 0.5, bx1 + 4, -8, 8, Z_FLOOR[0] - 3, Z_FLOOR[0] + 1)   # finger notch
    return d


def top_shell():
    """Dome carapace: covers the chassis and the hip hubs, eye stalks on top, camera window in front."""
    from shapely.geometry import Point
    pf = shell_planform().simplify(0.15, preserve_topology=True)
    t = 2.0
    inner_pf = opened(pf.buffer(-t, 16), 0.6).simplify(0.15, preserve_topology=True)
    bounds = pf.bounds
    cx = (bounds[0] + bounds[2]) / 2
    a, b, c = (bounds[2] - bounds[0]) / 2 + 14.0, (bounds[3] - bounds[1]) / 2 + 8.0, 19.0
    dome_o = Pos(cx, 0, Z_RIM) * b3d_scale(Sphere(1), by=(a, b, c))
    dome_i = Pos(cx, 0, Z_RIM) * b3d_scale(Sphere(1), by=(a - t, b - t, c - t))
    outer = dome_o & prism(pf, Z_RIM, Z_RIM + 40)
    hollow = dome_i & prism(inner_pf, Z_RIM - 5, Z_RIM + 40)
    shell = (outer - hollow).fix()

    def top(x, y):
        return Z_RIM + c * max(0.0, 1 - ((x - cx) / a) ** 2 - (y / b) ** 2) ** 0.5

    # the face: a forehead visor holding the 2.42" OLED (61.5 x 39.5 mm module, 55 x 27.5 mm active area),
    # tilted back FACE_TILT so it looks up at you, its lower edge above the shoulder brackets. The module
    # goes in from inside the dome (the pocket is open at the back), glass against the bezel; four M2 screws
    # from the front go through the bezel, short spacer posts and the module's corner holes, nuts behind.
    visor = FACE_LOC * (box(-3.0, 3.4, -FACE_HOUSING[0] / 2, FACE_HOUSING[0] / 2, -FACE_HOUSING[1] / 2,
                            FACE_HOUSING[1] / 2))                                # bezel 1.6 mm in front of the glass
    shell = shell.fuse(visor)
    shell -= FACE_LOC * box(-2.2, 1.8, -FACE_MODULE[0] / 2 - 0.3, FACE_MODULE[0] / 2 + 0.3,
                            -FACE_MODULE[1] / 2 - 0.3, FACE_MODULE[1] / 2 + 0.3)          # module pocket
    shell -= FACE_LOC * box(1.0, 4.4, -FACE_ACTIVE[0] / 2 - 0.5, FACE_ACTIVE[0] / 2 + 0.5,
                            -FACE_ACTIVE[1] / 2 - 0.5, FACE_ACTIVE[1] / 2 + 0.5)          # window
    shell -= FACE_LOC * box(-30, -2.0, -FACE_MODULE[0] / 2 - 0.3, FACE_MODULE[0] / 2 + 0.3,
                            -FACE_MODULE[1] / 2 - 0.3, FACE_MODULE[1] / 2 + 0.3)          # open to the dome
    # four spacer posts between the bezel and the module's board (the glass sits between them), and
    # through-holes: M2x8 screws from the front, through the bezel, the posts and the board, nuts behind
    pcb = 1.8 - FACE_GLASS_T                                 # the board's front face (visor frame x)
    for hy, hz in FACE_HOLES:
        post = FACE_LOC * (Pos((pcb + 2.3) / 2, hy, hz) * Rot(0, 90, 0) * Cylinder(2.2, 2.3 - pcb))
        shell = shell.fuse(post)
        shell -= FACE_LOC * Pos(1.0, hy, hz) * Rot(0, 90, 0) * Cylinder(M2_CLEAR / 2, 8.0)
    dome_mid = Pos(cx, 0, Z_RIM) * b3d_scale(Sphere(1), by=(a - 0.5, b - 0.5, c - 0.5))
    above = box(-200, 200, -200, 200, Z_RIM, 100)
    # magnet bosses matching the chassis
    for mx, my in MAGNETS:
        post = (disc(MAG_R, Z_RIM, Z_RIM + 30, mx, my) & dome_mid) - disc(3.15, Z_RIM - 1, Z_RIM + 3.2, mx, my)
        shell = shell.fuse(post)
    # speaker: a ring hanging from the dome holds the 28 mm speaker face-up against a grille
    sx, sy, sr = SPEAKER
    z_ring = Z_RIM + 10.5                                       # clear of the Pi on its tray below
    ring = (disc(sr + 1.9, z_ring, Z_RIM + 40, sx, sy) - disc(sr + 0.2, z_ring - 1, Z_RIM + 41, sx, sy)) & dome_mid
    shell = shell.fuse(ring)
    for k in range(-2, 3):
        for j in range(-2, 3):
            if k * k + j * j <= 5:
                shell -= Pos(sx + 3.5 * k, sy + 3.5 * j, Z_RIM + 20) * Cylinder(0.9, 30)
    shell -= Pos(MIC[0], MIC[1], Z_RIM + 20) * Cylinder(0.8, 30)
    return max(shell.solids(), key=lambda s: s.volume)
