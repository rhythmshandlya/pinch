"""How much of the camera's view do Pinch's arms hide, and does the idle arm swing (pinch/body/life.py look) clear it?

    python sim/benches/arm_view.py            # standing, arms straight, swung out, and over one swing
    python sim/benches/arm_view.py --json out.json

Rays from eye_cam over the real lens (160 deg fisheye, decision 115; the model's pinhole is 102 deg) against the
robot's visible geometry, standing on the desk. Reported per bearing from Pinch's heading: the share of the band
20-50 deg above level that the arms hide (where the face of someone sitting at the desk 0.4-1 m away appears), for
the arms straight, swung out to look.arm_yaw_rad, and over one swing (a direction counts as hidden only if every
pose of the swing hides it). Also: the desk's sight lines 15-60 cm ahead (tests/test_v2.py) at both ends of the swing,
and the gap between each arm and its front leg over the gait's hip range (the model has no self-collision, so the
twin would let them pass through each other).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import save_json  # noqa: E402

FACE_BAND = (20.0, 50.0)                    # deg above level
BEARINGS = (0, 10, 20, 30, 40, 50, 60, 70)


class View:
    def __init__(self, step_deg=1.0):
        import mujoco

        from crab.params import PARAMS
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
        self.d = mujoco.MjData(self.m)
        self.cam = self.m.camera("eye_cam").id
        self.arm = np.array([self.m.body(i).name.startswith(("L_", "R_")) for i in range(self.m.nbody)])
        self.groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)       # what the camera sees: the visible meshes
        self.stand_z = PARAMS.stand_body_z
        u = np.radians(np.arange(-80.0, 80.0 + 1e-9, step_deg))
        uu, vv = np.meshgrid(u, u)
        r = np.hypot(uu, vv).ravel()
        keep = r <= np.radians(80.0)                                      # the lens's 160 deg disk (equidistant)
        r, ph = r[keep], np.arctan2(vv.ravel()[keep], uu.ravel()[keep])
        self.rays = np.stack([np.sin(r) * np.cos(ph), np.sin(r) * np.sin(ph), -np.cos(r)], 1)   # camera frame

    def pose(self, q18):
        from crab.params import JOINT_NAMES
        m, d = self.m, self.d
        self.mj.mj_resetDataKeyframe(m, d, m.key("stand").id)
        d.qpos[2] = self.stand_z
        for n, v in zip(JOINT_NAMES, q18):
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        self.mj.mj_forward(m, d)

    def hidden(self):
        """(elevation deg, |bearing| deg, hidden by an arm) per ray."""
        m, d = self.m, self.d
        W = self.rays @ d.cam_xmat[self.cam].reshape(3, 3).T
        n = len(W)
        gid, dist = np.full(n, -1, dtype=np.int32), np.full(n, -1.0)
        self.mj.mj_multiRay(m, d, d.cam_xpos[self.cam], W.ravel(), self.groups, 1, -1, gid, dist, None, n, 5.0)
        body = np.where(gid >= 0, m.geom_bodyid[np.maximum(gid, 0)], 0)
        el = np.degrees(np.arcsin(np.clip(W[:, 2], -1, 1)))
        az = np.abs(np.degrees(np.arctan2(W[:, 1], W[:, 0])))
        return el, az, self.arm[body]

    def desk_lines(self):
        """Sight lines from the camera to the desk 15-60 cm ahead across +-40 deg blocked by the robot (of 63)."""
        m, d = self.m, self.d
        cam, b = d.cam_xpos[self.cam].copy(), d.xpos[m.body("body").id]
        blocked = 0
        for dist in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
            for ang in np.radians(np.linspace(-40, 40, 9)):
                v = np.array([b[0] + dist * np.cos(ang), b[1] + dist * np.sin(ang), 0.005]) - cam
                L = float(np.linalg.norm(v))
                g = np.array([-1], dtype=np.int32)
                h = self.mj.mj_ray(m, d, cam, v / L, self.groups, 1, -1, g)
                blocked += int(g[0] >= 0 and h < L - 0.01 and m.geom_bodyid[g[0]] != 0)
        return blocked

    def arm_leg_gap(self):
        """Smallest distance (m) between an arm and a front leg (visible meshes); negative: they overlap."""
        m, d = self.m, self.d
        mesh = self.mj.mjtGeom.mjGEOM_MESH
        arms = [g for g in range(m.ngeom) if self.arm[m.geom_bodyid[g]] and m.geom_type[g] == mesh and m.geom_group[g] < 3]
        legs = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("FL", "FR"))
                and m.geom_type[g] == mesh and m.geom_group[g] < 3]
        fr = np.zeros(6)
        return min(self.mj.mj_geomDistance(m, d, a, b, 0.05, fr) for a in arms for b in legs)


def band_by_bearing(el, az, hid):
    lo, hi = FACE_BAND
    f = (el >= lo) & (el < hi)
    return {b: float(hid[f & (az >= b) & (az < b + 10)].mean()) for b in BEARINGS if (f & (az >= b) & (az < b + 10)).any()}


def gait_hip_range():
    """The front hips' range over the gaits (forward, back, sideways, turning)."""
    from crab.gait import GAITS, Gait
    from crab.params import JOINT_NAMES
    i = [JOINT_NAMES.index("FL_coxa"), JOINT_NAMES.index("FR_coxa")]
    qs = []
    for g in GAITS.values():
        for cmd in ((0.05, 0, 0), (-0.05, 0, 0), (0, 0.04, 0), (0, -0.04, 0), (0, 0, 0.6), (0, 0, -0.6), (0.04, 0, 0.4)):
            gait = Gait(g)
            for _ in range(200):
                qs += list(gait.step(0.02, cmd)[i])
    return float(min(qs)), float(max(qs))


def measure(step_deg=1.0, swing_samples=7):
    from crab.emotes import pose
    from pinch.body.life import LK
    v = View(step_deg)
    out = {}
    yaw_out = LK["arm_yaw_rad"]
    hid_all = None
    for k, y in enumerate(np.linspace(0.0, yaw_out, swing_samples)):
        v.pose(pose({"*_shoulder_yaw": float(y)}))
        el, az, hid = v.hidden()
        hid_all = hid if hid_all is None else hid_all & hid
        if k == 0:
            out["straight"] = dict(by_bearing=band_by_bearing(el, az, hid), desk_blocked=v.desk_lines())
        if k == swing_samples - 1:
            out["out"] = dict(by_bearing=band_by_bearing(el, az, hid), desk_blocked=v.desk_lines(),
                              yaw_deg=float(np.degrees(y)))
    out["swing"] = dict(by_bearing=band_by_bearing(el, az, hid_all))
    lo, hi = gait_hip_range()
    gaps = []
    for c in np.linspace(lo, hi, 7):
        v.pose(pose({"*_shoulder_yaw": yaw_out, "F?_coxa": float(c)}))
        gaps.append(v.arm_leg_gap())
    out["arm_leg_gap_mm"] = 1000 * float(min(gaps))
    out["hip_range_deg"] = (float(np.degrees(lo)), float(np.degrees(hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    r = measure()
    print(f"share of the face band ({FACE_BAND[0]:.0f}-{FACE_BAND[1]:.0f} deg up) the arms hide, by bearing from Pinch's heading:")
    print(f"{'bearing':>10s} " + " ".join(f"{b:>2d}-{b + 10:<3d}" for b in BEARINGS))
    for k in ("straight", "out", "swing"):
        lab = k if k != "out" else f"out {r['out']['yaw_deg']:.0f} deg"
        print(f"{lab:>10s} " + " ".join(f"{r[k]['by_bearing'][b]:6.0%}" for b in BEARINGS))
    print(f"desk sight lines 15-60 cm ahead blocked: straight {r['straight']['desk_blocked']}/63, out "
          f"{r['out']['desk_blocked']}/63")
    print(f"arm to front leg, arms out, front hips over the gait's {r['hip_range_deg'][0]:.0f}..{r['hip_range_deg'][1]:.0f} deg: "
          f"{r['arm_leg_gap_mm']:.1f} mm")
    save_json(a.json, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
