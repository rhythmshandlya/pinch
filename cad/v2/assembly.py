"""The complete v2 assembly: every printed part, every servo, every electronic part, on the simulator's skeleton.

    python cad/v2/assembly.py --render media/review/phase-02/cad_assembly.png
    python cad/v2/assembly.py --mass                    # mass and centre of mass per simulator body

groups() maps each simulator body (sim/models/crab_v2.xml) to its parts in that body's frame (mm): (label, solid,
mass kg, material). Masses: printed parts = volume x density x fill; servos = their real masses; electronics =
configs/layout_v2.toml (measured / datasheet / estimate, as labelled there). Right-side links are mirror images of the
left (local y -> -y), as in today's CAD (cad/assembly.py).
"""

import argparse
import os
import sys
from functools import cache
from math import degrees
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Box, Location, Plane, Pos, mirror  # noqa: E402

from cad.v2 import arms, legs, lid, servos, shell  # noqa: E402
from crab.params import ARM_IDS, LEG_IDS, PARAMS  # noqa: E402

PETG, TPU = 1.27, 1.21          # g/cm3
FILL = {"shell": 1.0, "lid": 0.9, "coxa": 0.8, "femur": 0.8, "tibia": 0.8, "foot": 0.6, "shoulder": 0.9, "upper": 0.85,
        "dactyl": 0.9}          # solid 2 mm walls for the shells; links printed with 4 walls and 40-50 % infill
MODEL = ROOT / "sim" / "models" / "crab_v2.xml"


def printed(solid, kind, density=PETG):
    return solid.volume / 1000.0 * density * FILL[kind] / 1000.0          # kg


def _mirror_y(solid):
    return mirror(solid, about=Plane.XZ)


def hip_servo_location(leg):
    """The hip servo's servo frame in the BODY frame: shaft vertical through the hip point, horn up (face at z 15.0),
    case along body x running toward the middle (front: x 1.2..46.7, rear: -46.7..-1.2), agreed with cad/v2/body.py."""
    (x, y, _), _ = PARAMS.leg_mounts[leg[0]]
    side = 1 if leg[1] == "L" else -1
    x, y = x * 1000, side * y * 1000                     # (crab.params is in metres; the CAD in mm)
    front = leg[0] == "F"                                   # cases run along x toward the middle, in two side bands
    return Location((x, y, legs.HIP_HORN_Z - servos.STS3215["z_max"]), (0, 0, 0 if front else 180))


def shoulder_servo_location(arm):
    """The shoulder yaw servo in the BODY frame: upside down, horn face at body z = -5.4, case along +x."""
    (x, y, z), _ = PARAMS.arm_mount
    side = 1 if arm == "L" else -1
    return Location((x * 1000, side * y * 1000, z * 1000 + arms.YAW_HORN_Z + 19.2), (180, 0, 0))


@cache
def electronics():
    """(label, solid, mass kg) for every layout part that is not a servo, in the body frame."""
    from crab.electronics import Layout
    out = []
    for p in Layout().parts.values():
        if p.in_model or p.kind in ("servo", "printed", "display") or p.mass_only:
            continue
        size = p.d.get("size_mm")
        if not size or min(size) <= 0:
            continue
        R = p.R
        loc = Location(Plane(origin=tuple(p.pos), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))
        out.append((p.name, loc * Box(*size), p.mass_g / 1000.0))
    return out


@cache
def groups(proxy=False):
    """simulator body -> [(label, solid in the body frame (mm), mass kg, material)]. proxy=True puts simple solids of the
    STS3215's real sizes in place of its STEP (for the interference checks: booleans on the STEP fail)."""
    sts = servos.sts3215_proxy if proxy else servos.sts3215
    g = {"body": []}
    sh = shell.shell_solid()
    g["body"].append(("shell", sh, printed(sh, "shell"), "shell"))
    lf = lid.hinge_location() * lid.lid_frame()
    g["body"].append(("lid", lf, printed(lf, "lid"), "dark"))
    panel = lid.hinge_location() * Pos(lid.MOD_T / 2, 0, lid.PANEL_Z) * Box(lid.MOD_T, lid.MOD_W, lid.MOD_H)
    g["body"].append(("panel", panel, 0.0, "screen"))                 # its mass is the layout's `oled`
    for leg in LEG_IDS:
        g["body"].append((f"{leg}_hip_servo", hip_servo_location(leg) * sts(), servos.STS3215["mass"], "servo"))
    for arm in ARM_IDS:
        g["body"].append((f"{arm}_yaw_servo", shoulder_servo_location(arm) * servos.sts3032(), 0.020, "servo"))
    for name, solid, mass in electronics():
        g["body"].append((name, solid, mass, "pcb"))
    for leg in LEG_IDS:
        right = leg[1] == "R"
        f = _mirror_y if right else (lambda s: s)
        cx = legs.coxa_link()
        g[f"{leg}_coxa_link"] = [("coxa", f(cx), printed(cx, "coxa"), "printed"),
                                 ("thigh_servo", f(legs.THIGH_SERVO * sts()), servos.STS3215["mass"], "servo")]
        fm = legs.femur_link()
        g[f"{leg}_femur_link"] = [("femur", f(fm), printed(fm, "femur"), "printed"),
                                  ("knee_servo", f(legs.KNEE_SERVO * sts()), servos.STS3215["mass"], "servo")]
        tb, ft = legs.tibia(), legs.foot()
        g[f"{leg}_tibia_link"] = [("tibia", f(tb), printed(tb, "tibia"), "printed"),
                                  ("foot", f(ft), printed(ft, "foot", TPU), "tip")]
    for arm in ARM_IDS:
        f = _mirror_y if arm == "R" else (lambda s: s)
        s1, u, d = arms.shoulder_link(), arms.upper_arm(), arms.dactyl()
        lift = Location((arms.LS, 0, 0), (90, 0, 0)) * servos.sts3032()
        claw = Pos(arms.LU, 0, arms.CLAW_Z) * servos.sts3032()
        g[f"{arm}_shoulder"] = [("shoulder", f(s1), printed(s1, "shoulder"), "printed"),
                                ("lift_servo", f(lift), 0.020, "servo")]
        g[f"{arm}_upper"] = [("upper", f(u), printed(u, "upper"), "printed"), ("claw_servo", f(claw), 0.020, "servo")]
        g[f"{arm}_dactyl"] = [("dactyl", f(d), printed(d, "dactyl"), "printed")]
    return g


def body_poses(qpos=None):
    """Simulator body -> Location in the robot frame (mm), at the stand keyframe (or a given qpos)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    if qpos is not None:
        d.qpos[:] = qpos
    d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    out = {}
    for name in groups():
        bid = m.body(name).id
        R = d.xmat[bid].reshape(3, 3)
        out[name] = Location(Plane(origin=tuple(d.xpos[bid] * 1000.0), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))
    return out


def render(path, size=(1500, 520)):
    """Tessellate the posed assembly and render three views with MuJoCo."""
    import tempfile

    import mujoco
    from build123d import export_stl
    from PIL import Image
    colors = {"shell": "0.93 0.42 0.15 1", "printed": "0.93 0.45 0.18 1", "servo": "0.15 0.15 0.17 1",
              "pcb": "0.1 0.45 0.2 1", "dark": "0.12 0.12 0.13 1", "screen": "0.02 0.02 0.03 1", "tip": "0.2 0.2 0.22 1"}
    tmp = Path(tempfile.mkdtemp())
    poses = body_poses()
    assets, geoms = [], []
    k = 0
    for body, items in groups().items():
        for label, solid, _, mat in items:
            if label == "shell":
                continue                                   # shown cut away so the inside is visible
            fn = tmp / f"p{k}.stl"
            export_stl(poses[body] * solid, str(fn), tolerance=0.3, angular_tolerance=0.4)
            assets.append(f'<mesh name="p{k}" file="{fn.as_posix()}" scale="0.001 0.001 0.001"/>')
            geoms.append(f'<geom type="mesh" mesh="p{k}" rgba="{colors[mat]}"/>')
            k += 1
    fn = tmp / "shell.stl"
    export_stl(shell.shell_solid() - Pos(0, -150, 0) * Box(400, 300, 400), str(fn), tolerance=0.3, angular_tolerance=0.4)
    assets.append(f'<mesh name="shell" file="{fn.as_posix()}" scale="0.001 0.001 0.001"/>')
    geoms.append(f'<geom type="mesh" mesh="shell" rgba="{colors["shell"]}"/>')
    xml = (f'<mujoco><asset>{"".join(assets)}</asset><visual><global offwidth="{size[0]}" offheight="{size[1]}"/></visual>'
           f'<worldbody><light pos="0.2 -0.3 0.6" dir="-0.3 0.4 -1"/><light pos="-0.3 0.3 0.5" dir="0.4 -0.3 -1"/>'
           f'{"".join(geoms)}</worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w = size[0] // 3
    r = mujoco.Renderer(m, size[1], w)
    sheet = Image.new("RGB", size)
    for i, (az, el) in enumerate(((150, -20), (90, -5), (90, -89))):
        c = mujoco.MjvCamera()
        c.type, c.distance, c.azimuth, c.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 0.62, az, el
        c.lookat[:] = [0.0, 0.0, -0.01]
        r.update_scene(d, c)
        sheet.paste(Image.fromarray(r.render()), (w * i, 0))
    sheet.save(path)
    return path


def mass_report():
    tot = 0.0
    for body, items in groups().items():
        m = sum(x[2] for x in items)
        tot += m
        print(f"{body:16s} {m * 1000:7.1f} g  ({', '.join(f'{x[0]} {x[2] * 1000:.0f}' for x in items[:6])}{' ...' if len(items) > 6 else ''})")
    print(f"robot total {tot * 1000:.0f} g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default=None)
    ap.add_argument("--mass", action="store_true")
    a = ap.parse_args()
    if a.mass:
        mass_report()
    if a.render:
        print("render ->", render(a.render))
    print("lid tilt", round(degrees(PARAMS.body.lid_tilt), 1), "deg")
