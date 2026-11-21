"""Checks of the printable v2 body (cad/v2/body.py): masses, fits, windows, joint openings and finger gaps.

    python cad/v2/body_check.py            # full: builds the tray and cover (about 2 min), prints the table
    python cad/v2/body_check.py --quick    # the feature-level checks only (no tray/cover assembly, about 30 s)

Exit code 1 if any line FAILs. WARN lines are findings that do not stop the build (estimates, the owner's calls).
What is checked:
  interface   the hip servo STEP against the agreed interface; the servos against the layout's parts
  mounts      every layout part has its mount (boards: at least two posts; wall parts: a wall to sit in)
  keep-out    no printed feature inside a joint's swept volume (+2 mm) or the hip brackets' free zones
  windows     each sensor window's view cone is clear of printed solids and parts for 20 mm
  openings    the shell is clear of each joint's swept volume by 2 mm over its range (full mode)
  fingers     the gaps between moving parts and the opening edges: under 5 mm or at least 12 mm
  parts       every part sits inside the tray / cover without touching a printed solid except its own mount (full)
  mass        tray and cover in PETG, solid walls; one piece each (full)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Pos, Solid  # noqa: E402

from cad.v2 import body as B  # noqa: E402

TOL_MM3 = 1.0              # volumes under this are numerical noise (mm^3)
SAFE_GAP, OPEN_GAP = 5.0, 12.0


class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, name, status, detail):
        self.rows.append((group, name, status, detail))
        return status

    def ok(self, group, name, cond, detail, warn=False):
        return self.add(group, name, "PASS" if cond else ("WARN" if warn else "FAIL"), detail)

    def info(self, group, name, detail):
        return self.add(group, name, "INFO", detail)

    def failures(self):
        return [r for r in self.rows if r[2] == "FAIL"]

    def print(self):
        w = max(len(r[1]) for r in self.rows)
        group = None
        for g, n, s, d in self.rows:
            if g != group:
                print(f"\n== {g}")
                group = g
            print(f"  {s:4s}  {n:<{w}}  {d}")
        f = self.failures()
        print(f"\n{len(self.rows)} checks, {sum(r[2] == 'PASS' for r in self.rows)} pass, "
              f"{sum(r[2] == 'WARN' for r in self.rows)} warn, {len(f)} fail")


def vol(a, b):
    """Volume of a & b (0 when their boxes do not meet)."""
    if a is None or b is None:
        return 0.0
    ba, bb = a.bounding_box(), b.bounding_box()
    if (ba.max.X < bb.min.X or bb.max.X < ba.min.X or ba.max.Y < bb.min.Y or bb.max.Y < ba.min.Y
            or ba.max.Z < bb.min.Z or bb.max.Z < ba.min.Z):
        return 0.0
    return B.volume(a & b)


def part_solid(name):
    lay = B.layout()
    if name.startswith("hip_"):
        return B.hip_servo(name[4:].upper())
    if name.startswith("shoulder_"):
        return B.shoulder_servo(name[-1].upper())
    p = lay[name]
    return B.obox(p.pos, p.size, p.R)


def checked_parts():
    lay = B.layout()
    return [n for n, p in lay.parts.items() if not p.mass_only and n not in B.NO_MOUNT and n not in B.PADS]


# ---------------------------------------------------------------------------------------------------- quick checks
def check_interface(r):
    lo_f, _ = B.hip_case_aabb("FL")
    _, hi_r = B.hip_case_aabb("RL")
    gap = lo_f[0] - hi_r[0]
    meas = 2 * (B.STEP_CASE[0] - B.STEP_SHAFT_MEASURED) + 2 * abs(B.joints()["FL"]["pos"][0])
    r.ok("interface", "hip shaft in STEP", abs(B.STEP_SHAFT_MEASURED - B.STEP_SHAFT_X) < 0.2,
         f"interface x {B.STEP_SHAFT_X} mm, the STEP's horn holes centre at x {B.STEP_SHAFT_MEASURED} mm: placed as the "
         f"interface says the real shaft sits {B.STEP_SHAFT_MEASURED - B.STEP_SHAFT_X:.2f} mm off the hip axis", warn=True)
    r.ok("interface", "front-rear hip gap", gap >= 2 * B.FIT + 1.2,
         f"{gap:.2f} mm at x = 0 (shared rib {gap - 2 * B.FIT:.2f} mm); with the measured shaft it would be "
         f"{meas:.2f} mm")
    lay = B.layout()
    servo_solids = {f"hip_{leg.lower()}": B.hip_servo(leg) for leg in B.LEGS}
    servo_solids.update({f"shoulder_{a.lower()}": B.shoulder_servo(a) for a in B.ARMS})
    for sname, s in servo_solids.items():
        hits = []
        for n, p in lay.parts.items():
            if p.mass_only or p.d.get("in_model") or n in ("oled", "visor") or n in B.PADS:
                continue
            v = vol(s, B.obox(p.pos, p.size, p.R))
            if v > TOL_MM3:
                hits.append(f"{n} {v:.0f} mm3")
        r.ok("interface", f"{sname} vs parts", not hits, "clear of every layout part" if not hits else "; ".join(hits))


def stacked_on(name):
    """The parts right under a board (their footprints overlap its own, their tops below its bottom)."""
    lay = B.layout()
    p = lay[name]
    c = p.corners()
    out = []
    for m, q in lay.parts.items():
        if m == name or q.mass_only or m in B.PADS or m in ("oled", "visor"):
            continue
        d = q.corners()
        if (d[:, 0].max() > c[:, 0].min() and d[:, 0].min() < c[:, 0].max() and d[:, 1].max() > c[:, 1].min()
                and d[:, 1].min() < c[:, 1].max() and d[:, 2].max() <= c[:, 2].min() + 0.01):
            out.append(m)
    return out


def check_mounts(r):
    lay = B.layout()
    for name in B.STANDOFFS:
        posts = [f for f in B.features() if f.owner == name and f.op == "add"]
        kinds = {}
        for f in posts:
            k = f.name[len(name) + 1:].rstrip("0123456789")
            kinds[k] = kinds.get(k, 0) + 1
        what = ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in kinds.items()) or "nothing printed"
        if len(posts) >= 2 and set(kinds) == {"post"}:
            r.ok("mounts", name, True, f"{what} ({B.STANDOFFS[name][0]})")
        elif len(posts) >= 2:
            r.ok("mounts", name, False, f"{what}: corners over other parts, glue / zip-tie to the ribs", warn=True)
        else:
            below = stacked_on(name)
            r.ok("mounts", name, bool(below), f"{what}; stacked on {', '.join(below)} with brass standoffs or foam "
                 "tape (no room for printed supports under it)" if below else f"{what}: no support", warn=True)
    owners = {f.owner for f in B.features() if f.op == "add"}
    cut_owners = {f.owner for f in B.features() if f.op == "cut"}
    for name in checked_parts():
        if name in B.STANDOFFS:
            continue
        p = lay[name]
        if name in owners or name in cut_owners or p.on in owners:
            kinds = sorted({f.name.split("_", 1)[-1].rstrip("0123456789+-") for f in B.features() if f.owner == name})
            r.ok("mounts", name, True, ", ".join(kinds) or f"on {p.on}")
        else:
            r.ok("mounts", name, False, "no mount")
    n_grille = B.grille()[1]
    r.ok("mounts", "speaker grille", n_grille >= 12, f"{n_grille} holes of 2 mm in the cover over the speaker")


def check_keepouts(r):
    """Layout parts against the joints' swept volumes (and the hip brackets' plate zones); printed features are shaped
    around those zones in body.py, and the ones that lost material are listed."""
    lay = B.layout()
    zones = {}
    for key in B.LEGS + B.ARMS:
        zones[key] = (B.sweep(key, 0.0), B.sweep(key, B.CLEAR))
    for leg in B.LEGS:
        zones[f"{leg} bracket plates"] = (B.fuse(B.hip_keepouts(leg)), None)
    bad = 0
    for n, p in lay.parts.items():
        if p.mass_only or p.d.get("in_model") or n in ("oled", "visor") or n in B.PADS:
            continue
        s = part_solid(n)
        hits, near = [], []
        for zn, (z0, z2) in zones.items():
            v = vol(s, z0)
            if v > TOL_MM3:
                hits.append(f"{zn} {v:.0f} mm3")
            elif z2 is not None and vol(s, z2) > TOL_MM3:
                near.append(zn)
        if hits:
            bad += 1
            r.ok("keep-out", n, False, "inside the swept volume of " + "; ".join(hits))
        elif near:
            bad += 1
            r.ok("keep-out", n, False, "closer than 2 mm to the moving " + ", ".join(near), warn=True)
    if not bad:
        r.ok("keep-out", "layout parts", True, "every part clear of the joints' swept volumes by 2 mm")
    trimmed = [f for f in B.features() if f.trimmed > 0]
    for f in trimmed:
        r.ok("keep-out", f.name, False, f"shaped around a joint: {f.trimmed:.0f} mm3 taken off", warn=True)


def window_cones():
    lay = B.layout()
    out = {}
    for n, p in lay.parts.items():
        if not p.window or n == "oled":
            continue
        wp, wd = p.window_pose()
        r0 = B.APERTURE_R.get(n, B.TOF_APERTURE_R)
        out[n] = B.cone(wp, wd, r0, p.window.get("cone_deg", 10.0), B.WINDOW_CHECK_MM, back=0.0)
    return out


def check_windows_parts(r, cones):
    lay = B.layout()
    for n, c in cones.items():
        hits, links = [], []
        for m, p in lay.parts.items():
            if m in (n, lay[n].on) or p.mass_only or m in ("oled", "visor") or m in B.PADS or p.on == n:
                continue
            v = vol(c, part_solid(m))
            if v > TOL_MM3:
                hits.append(f"{m} {v:.0f} mm3")
        for key in B.LEGS + B.ARMS:
            j = B.joints()[key]
            for q in (j["range"][0], 0.0, j["range"][1]):
                v = vol(c, B.envelope_at(key, q))
                if v > TOL_MM3:
                    links.append(f"{key} at {math.degrees(q):.0f} deg")
                    break
        r.ok("windows", f"{n} vs parts", not hits, "clear for 20 mm" if not hits else "; ".join(hits))
        if links:
            r.ok("windows", f"{n} vs links", False, "a moving link enters the view cone: " + ", ".join(links), warn=True)


def finger_gaps(key):
    """The gaps between a joint's moving envelope and its opening's edges (mm), as built (the cut is the swept volume
    grown by CLEAR):
      side   the gap along the link's faces parallel to the motion: CLEAR everywhere
      face   the most at the middle of a face turning about the axis (its corners pass at CLEAR; the arc the
             corners cut bulges beyond the flat face by hypot(x1, w) - x1)
      slot   the slot left ahead of the link at a range end: the arc its innermost face travels, plus CLEAR"""
    j = B.joints()[key]
    elems = [e for e in (B.LEG_ENVELOPE if key in B.LEGS else B.ARM_ENVELOPE) if e[1] != "cyl"]
    g = B.CLEAR
    face = max(math.hypot(e[2] + g, e[3] + g) - (e[2] + g) + g for e in elems)
    span = j["world"][1] - j["world"][0]
    if key not in B.LEGS:
        return g, face, None                         # the arm's bracket is round: no slot opens ahead of it
    slot = max(e[1] for e in elems) * span + g       # the web (innermost face 28 mm out) is what crosses the wall
    return g, face, slot


def check_fingers(r):
    for key in B.LEGS + B.ARMS:
        side, face, slot = finger_gaps(key)
        ok = side < SAFE_GAP and face < OPEN_GAP and (slot is None or slot >= OPEN_GAP)
        name = ("coxa " if key in B.LEGS else "shoulder ") + key
        tail = ("; the bracket is round, nothing opens ahead of it" if slot is None else
                f"; slot ahead of the link up to {slot:.0f} mm, closing to {side:.0f} mm at the range end (a finger "
                "in the slot is pushed to the edge: only the servo's torque limit guards it)")
        r.ok("fingers", name, ok and face < SAFE_GAP,
             f"side gap {side:.1f} mm; turning faces {side:.1f}..{face:.1f} mm" + tail, warn=True)


def arm_pitch_clearance():
    """Least distance (mm) of the upper arm (pitch horn r 14, capsule r 11, palm r 19, crab_v2.xml) outside the
    shell over the shoulder yaw and pitch ranges: sampled points on the primitives against the shell surface."""
    sh = B.shell_math()
    best = 1e9
    for key in B.ARMS:
        j = B.joints()[key]
        for q in np.linspace(*j["range"], 5):
            t = j["yaw"] + j["sign"] * q
            Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            for pitch in np.radians(np.linspace(-40, 72, 9)):
                u = np.array([math.cos(pitch), 0, math.sin(pitch)])
                pts = []
                axis0 = np.array([24.0, 0, 0])
                for yy in np.linspace(-15.5, 15.5, 5):                  # pitch horn, a cylinder about local y
                    for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                        pts.append(axis0 + [14 * math.cos(a), yy, 14 * math.sin(a)])
                palm = axis0 + 39 * u + np.array([-math.sin(pitch), 0, math.cos(pitch)]) * -6
                for v in np.random.default_rng(0).normal(size=(60, 3)):
                    pts.append(palm + 19 * v / np.linalg.norm(v))
                pw = (Rz @ np.array(pts).T).T + j["pos"]
                d = [-sh.depth(p) for p in pw]
                best = min(best, min(d))
    return best


def check_arm_pitch(r):
    d = arm_pitch_clearance()
    r.ok("openings", "arms over yaw x pitch", d > B.CLEAR,
         f"the upper arm stays {d:.1f} mm outside the shell surface over yaw and pitch (sampled)")


# ---------------------------------------------------------------------------------------------------- full checks
def check_mass(r, tray, cover):
    for name, s in (("tray", tray), ("cover", cover)):
        pieces = sorted((abs(x.volume) for x in s.solids()), reverse=True)
        r.info("mass", name, f"{B.mass_g(s):.1f} g PETG ({B.volume(s) / 1000:.1f} cm3, solid walls)")
        r.ok("mass", f"{name} in one piece", len(pieces) == 1,
             "one solid" if len(pieces) == 1 else f"{len(pieces)} solids, loose: " + ", ".join(f"{v:.0f} mm3" for v in pieces[1:]))
    from crab.params import PARAMS
    total = B.mass_g(tray) + B.mass_g(cover)
    r.info("mass", "tray + cover vs model", f"{total:.0f} g printed vs {PARAMS.body.shell_mass * 1000:.0f} g in the "
           "simulator's shell (area x 2 mm x 1.27 x 1.2)")


def check_openings(r, tray, cover):
    for key in B.LEGS + B.ARMS:
        hit = vol(B.sweep(key, 0.0), tray) + vol(B.sweep(key, 0.0), cover)
        near = vol(B.sweep(key, B.CLEAR - 0.1), tray) + vol(B.sweep(key, B.CLEAR - 0.1), cover)
        j = B.joints()[key]
        rng = f"{math.degrees(j['range'][0]):.0f}..{math.degrees(j['range'][1]):.0f} deg"
        r.ok("openings", f"{j['joint']} ({rng})", hit < TOL_MM3 and near < TOL_MM3,
             f"swept link clear of the shell by >= {B.CLEAR - 0.1:.1f} mm" if near < TOL_MM3 else
             f"shell inside the sweep: {hit:.0f} mm3 (+1.9 mm: {near:.0f} mm3)")
    for leg in B.LEGS:
        v = sum(vol(z, tray) + vol(z, cover) for z in B.hip_keepouts(leg))
        r.ok("openings", f"hip {leg} bracket free zones", v < TOL_MM3, f"{v:.0f} mm3 of print inside")


def check_thigh(r, tray, cover):
    """How far the thigh (capsule r 19 from the femur axis, crab_v2.xml) with the knee servo standing up at its end
    (limbs CAD: case up to 46 mm above the knee axis) can lift at the neutral hip before it meets the print."""
    for leg in ("FL", "RL"):
        j = B.joints()[leg]
        t = j["yaw"]
        cx, cy = j["pos"][:2]
        free_up = None
        for deg in range(-40, 90, 10):
            th = math.radians(deg)
            knee = np.array([cx + 35.2 * math.cos(t), cy + 35.2 * math.sin(t), 0.0])
            u = np.array([math.cos(th) * math.cos(t), math.cos(th) * math.sin(t), math.sin(th)])
            up = np.array([-math.sin(th) * math.cos(t), -math.sin(th) * math.sin(t), math.cos(th)])
            side = np.cross(up, u)
            R = np.column_stack([u, side, up])
            kc = B.obox(knee + 56 * u + 23.0 * up, (24.7, 35.0, 46.0), R)
            cap = (B.axis_cyl(knee, u, 19.0, 56.0) + Pos(*knee) * Solid.make_sphere(19.0)
                   + Pos(*(knee + 56 * u)) * Solid.make_sphere(19.0) + kc)
            v = vol(cap, tray) + vol(cap, cover)
            if v > TOL_MM3:
                break
            free_up = deg
        r.ok("openings", f"{leg} thigh lift (femur, neutral hip)", free_up is not None and free_up >= 80,
             f"clear up to {free_up} deg of the femur range -40..86 deg" if free_up is not None else
             "meets the print at -40 deg", warn=True)


def check_windows_print(r, cones, tray, cover):
    for n, c in cones.items():
        v = vol(c, tray) + vol(c, cover)
        r.ok("windows", f"{n} through the shell", v < TOL_MM3, "clear for 20 mm" if v < TOL_MM3 else f"{v:.0f} mm3 of print in the cone")


def check_parts(r, tray, cover):
    lay = B.layout()
    owned = {}
    for f in B.features():
        if f.op == "add" and f.owner:
            owned.setdefault(f.owner, []).append(f.solid)
    for name in checked_parts():
        s = part_solid(name)
        own = B.fuse(owned.get(name, []) + (owned.get(lay[name].on, []) if lay[name].on else []))
        v = vol(s, tray) + vol(s, cover)
        v_own = vol(s, own) if own is not None else 0.0
        bad = v - v_own
        outside = B.volume(s - B.outer()) if not (lay[name].d.get("outside_ok") or lay[name].d.get("wall_mounted")) else 0.0
        msg = []
        if bad > TOL_MM3:
            msg.append(f"{bad:.0f} mm3 inside printed solids")
        if outside > TOL_MM3:
            msg.append(f"{outside:.0f} mm3 outside the shell")
        if lay[name].d.get("wall_mounted"):
            p = lay[name]
            near = vol(B.obox(p.pos, p.size + 6.0, p.R), tray) + vol(B.obox(p.pos, p.size + 6.0, p.R), cover)
            if near < 20.0:
                msg.append("no wall around it (inside a joint opening)")
        if bad > TOL_MM3:
            who = [f.name for f in B.features() if f.op == "add" and f.owner != name and vol(s, f.solid) > TOL_MM3]
            if who:
                msg.append("in " + ", ".join(who[:4]))
        r.ok("parts", name, not msg, "fits, touches only its own mount" if not msg else "; ".join(msg))


def main(quick=False):
    t0 = time.time()
    r = Report()
    check_interface(r)
    check_mounts(r)
    check_keepouts(r)
    cones = window_cones()
    check_windows_parts(r, cones)
    check_fingers(r)
    check_arm_pitch(r)
    if not quick:
        tray, cover = B.tray(), B.cover()
        check_mass(r, tray, cover)
        check_openings(r, tray, cover)
        check_thigh(r, tray, cover)
        check_windows_print(r, cones, tray, cover)
        check_parts(r, tray, cover)
    r.print()
    print(f"({time.time() - t0:.0f} s)")
    return r


if __name__ == "__main__":
    rep = main(quick="--quick" in sys.argv)
    sys.exit(1 if rep.failures() else 0)
