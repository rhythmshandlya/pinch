"""Pinch in the digital twin: desk world + robot runtime + body + mind + face, one 50 Hz loop.

    sim = PinchSim()
    sim.say("Pinch, put the pen back")
    sim.run(60)                         # simulated seconds
    sim.in_tray("pen")

Used by the virtual pet app (pinch/app.py) and the scenario tests (sim/pinch_scenarios.py).
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import JOINT_NAMES  # noqa: E402
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402
from pinch.body import Body  # noqa: E402
from pinch.face import Face  # noqa: E402
from pinch.hearing import Hearing  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402
from pinch.touch import BodySense, CliffReflex, StuckSense, twin_sensor  # noqa: E402
from pinch.world import Desk  # noqa: E402
from robot.backends import SimBackend  # noqa: E402
from robot.crab_robot import DT, Crab  # noqa: E402
from robot.servos import Calibration  # noqa: E402
from sim.desk import OBJECTS, START, build_desk_model, object_pose, set_face  # noqa: E402

MIND_EVERY = 5                        # mind at 10 Hz
FACE_EVERY = 3                        # face texture at ~17 Hz


SIM_EPOCH = 1795059000.0          # the twin's calendar starts 2026-11-19 09:00 IST (03:30 UTC): runs replay exactly


class PinchSim:
    def __init__(self, memory=None, llm=False, seed=0, start=None, render=False, policies=True, extras=(),
                 epoch=SIM_EPOCH, decision_log=None):
        """epoch: the wall-clock time the run starts at (what the mind's clock reads at t = 0).
        decision_log: a file to write every activity switch and its reason to (JSON lines)."""
        self.t = 0.0
        self.desk = Desk()
        self.m = build_desk_model(self.desk, start=start or START, extras=extras)
        self.hw = SimBackend(model=self.m, render=render)
        self.d = self.hw.d
        self.crab = Crab(self.hw, calibration=Calibration())
        self.crab.start()                                  # ease into the stand pose (2 s)
        self.mind = Mind(memory=memory if memory is not None else Memory(path=None), llm=llm, seed=seed,
                         clock=lambda: epoch + self.t)
        self.mind.decision_log_path = decision_log
        self.face = Face(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.senses = BodySense()
        self.stuck_sense = StuckSense()
        self.cliff = CliffReflex()
        self.stuck_events = []            # (t, "stuck"/"unstuck", why)
        self.last_recover, self.edge_recovers = -99.0, 0
        self.hearing = Hearing()
        self._tip_d = mujoco.MjData(self.m)
        emote_pol = walk_pol = None
        if policies:
            from crab.policy_np import load_policies
            emote_pol, walk_pol = load_policies()
        grasp_pol = None
        if policies and (POLICY_DIR / "grasp.npz").exists():
            grasp_pol = NumpyPolicy.load(POLICY_DIR / "grasp.npz")
        self.body = Body(self.crab, self.pose, self.obj, self.desk, tip_fn=self.tip, emote_policy=emote_pol,
                         walk_policy=walk_pol, grasp_policy=grasp_pol)
        self.body.grip_fn = self.gripping
        self.body.grasp_view_fn = self.grasp_view
        self.view_rng = np.random.default_rng(seed + 7)
        self.k = 0
        self.t = 0.0
        self.sounds_played = []           # (t, name)
        self.motions = []                 # (t, step)
        self.falls = 0
        self.edge_stops = 0
        for name in OBJECTS:              # Pinch knows its desk things where they start (homes are learned)
            self.mind.world.see(name, self.obj(name)[0][:2], graspable=OBJECTS[name][4])
        self.mind.memory.teach_home("pen", self.desk.tray_xy)
        self.mind.memory.teach_home("eraser", (self.desk.tray_xy[0] - 0.04, self.desk.tray_xy[1]))
        self.user_active = False          # the app sets this while you interact
        self.autonomous = True            # False: the mind doesn't move the body (skill tests)

    # ---------------------------------------------------------------- state readers
    def pose(self):
        q = self.d.qpos
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), float(yaw)

    def obj(self, name):
        return object_pose(self.m, self.d, name)

    def in_tray(self, name):
        (x, y, _), z = self.obj(name)
        return self.desk.in_tray((x, y)) and z < 0.03

    GAP = 0.006          # aim a thing this far off the fixed jaw, into the open gap (mid-jaw)

    def tip(self, side, q):
        """Centre of the open claw's gap, in Pinch's body frame, for joint targets q (forward kinematics).
        The fixed finger is on one side of the grip line and the moving finger swings in from the other,
        so a thing to grab has to sit a few mm toward the moving finger."""
        dd = self._tip_d
        dd.qpos[:] = self.m.qpos0
        dd.qpos[3:7] = (1, 0, 0, 0)
        for i, n in enumerate(JOINT_NAMES):
            dd.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = q[i]
        mujoco.mj_kinematics(self.m, dd)
        s = "L" if side == "left" else "R"
        site = self.m.site(f"{s}_grip_site").id
        p = dd.site_xpos[site] - dd.qpos[:3]
        R = dd.site_xmat[site].reshape(3, 3)
        # the open claw's gap: 4 mm back from the grip line's end, 8 mm off the fixed finger (measured from the
        # finger geometry: at full open the moving finger is 14-20 mm out there, the fixed one 0.5 mm in)
        off = R @ np.array([0.0, (1 if side == "left" else -1) * self.GAP, 0.0])
        return float(p[0] + off[0]), float(p[1] + off[1])

    def feet_down(self, any_surface=False):
        """How many feet touch the desk (the real robot reads this from its leg servos' load). any_surface:
        also count a foot standing on the tray or on a desk thing (it carries weight there too)."""
        if not hasattr(self, "_feet"):
            self._feet = {self.m.geom(f"{lg}_foot").id for lg in ("FL", "FR", "RL", "RR")}
            self._desk = self.m.geom("desk_top").id
            self._robot_root = self.m.body_rootid[self.m.body("body").id]
        n = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for f, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if f not in self._feet:
                    continue
                if o == self._desk or any_surface and self.m.body_rootid[self.m.geom_bodyid[o]] != self._robot_root:
                    n.add(f)
        return len(n)

    # what the learned grab sees (on the robot: the camera); kinds and half-extents as in training/grasp_env.py
    GRASP_SHAPES = {"pen": (0, (0.0695, 0.0045, 0.0045), 0.055), "eraser": (1, (0.010, 0.006, 0.0045), 0.006)}

    def grasp_view(self, name, side):
        """(rel, ang, kind, size) for the learned grab: the thing's grab point (the point of its long axis
        nearest the open claw's gap) relative to the gap, and its direction, in Pinch's body frame."""
        if name not in self.GRASP_SHAPES or side != "right":
            return None
        kind, size, half = self.GRASP_SHAPES[name]
        m, d = self.m, self.d
        site = m.site("R_grip_site").id
        gap = d.site_xpos[site] + d.site_xmat[site].reshape(3, 3) @ np.array([0.0, -0.006, 0.0])
        b = m.body(name).id
        c, ax = d.xpos[b], d.xmat[b].reshape(3, 3)[:, 0]
        grab = c + np.clip(np.dot(gap - c, ax), -half, half) * ax
        R = d.xmat[m.body("body").id].reshape(3, 3)
        rel = R.T @ (grab - gap) + self.view_rng.normal(0, 0.002, 3)
        ax_b = R.T @ ax
        return rel, float(np.arctan2(ax_b[1], ax_b[0])), kind, np.array(size)

    def gripping(self, name, side):
        """Both claw fingers touch the thing (on the robot: the camera sees it between the fingers)."""
        s = "L" if side == "left" else "R"
        g = self.m.geom(f"{name}_geom").id
        pollex = self.m.geom(f"{s}_pollex").id
        dactyl = self.m.body(f"{s}_dactyl").id
        touch_p = touch_d = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if g not in (c.geom1, c.geom2):
                continue
            o = c.geom2 if c.geom1 == g else c.geom1
            touch_p |= o == pollex
            touch_d |= self.m.geom_bodyid[o] == dactyl
        return touch_p and touch_d

    # ---------------------------------------------------------------- the user, simulated or from the app
    def say(self, text):
        self.user_active = True
        return self.mind.hear_text(text)

    def touch(self, gesture):
        self.mind.touch(gesture)

    def hear_audio(self, chunk):
        for s in self.hearing.feed(chunk):
            self.mind.hear_summary(s)
        for kind, _ in self.hearing.take_events():
            self.mind.hear_event(kind)

    def place(self, name, xy, yaw=0.0):
        """Put a desk thing somewhere (you moving the pen while you work)."""
        b = self.m.body(name)
        adr = self.m.jnt_qposadr[b.jntadr[0]]
        z = OBJECTS[name][1][0] + 0.001
        self.d.qpos[adr:adr + 3] = (xy[0], xy[1], z)
        self.d.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))   # lying flat, turned by yaw
        self.d.qvel[self.m.jnt_dofadr[b.jntadr[0]]:self.m.jnt_dofadr[b.jntadr[0]] + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    # ---------------------------------------------------------------- the loop
    def perceive(self):
        """What Pinch's camera sees: things within its field of view (with a little noise)."""
        x, y, h = self.pose()
        w = self.mind.world
        w.pose = (x, y, h)
        for name in OBJECTS:
            (ox, oy, oyaw), oz = self.obj(name)
            on_desk = oz > -0.02
            bearing = np.arctan2(oy - y, ox - x) - h
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            dist = np.hypot(ox - x, oy - y)
            held = self.body.held and self.body.held[0] == name
            if (abs(bearing) < 0.9 and 0.05 < dist < 0.6) or held or not on_desk:
                noise = self.rng.normal(0, 0.004, 2)
                w.see(name, (ox + noise[0], oy + noise[1]), oyaw, graspable=OBJECTS[name][4], on_desk=on_desk)
                if not held and self.mind.world.objects[name].still_since < self.t - 30:
                    pass
        if self.user_active:
            w.owner.present = True
            w.owner.last_seen = self.mind.t

    def step(self):
        if self.k % MIND_EVERY == 0 and not self.autonomous:
            self.perceive()
        elif self.k % MIND_EVERY == 0:
            self.perceive()
            motion, cancel = self.mind.tick(DT * MIND_EVERY, self.body.busy, self.body.ok)
            if cancel:
                self.body.cancel()
            if motion is not None:
                self.body.start(motion)
                self.motions.append((round(self.t, 2), motion))
            for s in self.mind.outputs()[2]:
                self.sounds_played.append((round(self.t, 2), s))
            # reflex: a cliff beam sees no desk while walking toward its side -> back off (latched)
            back = self.cliff.update(self.t, self.cliff_mm(), 0.0, self.body.cmd_vel)
            self.body.cliff_blocked = self.cliff.blocked
            if back is not None:
                self.body.reflex(back)
                self.edge_stops += 1
            gyro, grav = self.hw.read_imu()
            for ev in self.senses.update(DT * MIND_EVERY, gyro, grav, feet_loaded=self.feet_down()):
                self.mind.touch(ev)
            walking = self.body.busy and self.body.status in ("goto", "walk", "push", "recover", "turn_to")
            x, y, h = self.pose()
            margin = self.desk.feet_margin((x, y), h)
            # reflex: a foot over the desk edge -> drop everything and shuffle back in (before anything else)
            if margin < -0.003 and not self.senses.lifted and self.body.status != "recover" \
                    and self.t - self.last_recover > 3.0:
                self.body.reflex(("recover",))
                self.last_recover = self.t
                self.edge_recovers += 1
            for ev in self.stuck_sense.update(DT * MIND_EVERY, grav, self.feet_down(any_surface=True), (x, y),
                                              float(np.hypot(*self.body.cmd_vel[:2])), lifted=self.senses.lifted,
                                              standing=not self.body.busy, walking=walking, feet_margin=margin):
                self.mind.touch(ev)
                self.stuck_events.append((round(self.t, 1), ev, self.stuck_sense.why))
            if grav[2] > -0.5:
                self.falls += 1
        if self.k % FACE_EVERY == 0:
            expr, gaze, _ = self.mind.face_expr, self.mind.gaze, None
            self.face.update(DT * FACE_EVERY, expr, gaze)
            set_face(self.m, self.face.rgb(scale=2))
        self.body.tick()                                   # 20 ms of physics through the robot runtime
        self.k += 1
        self.t += DT

    def cliff_mm(self):
        """The cliff ring's readings (mm; the rangefinder's -1 = nothing within reach = 9999)."""
        out = {}
        for name, *_ in self.cliff.beams:
            v = float(self.d.sensor(twin_sensor(name)).data[0])
            out[name] = 9999 if v < 0 else v * 1000
        return out

    def run(self, seconds, every=None):
        for _ in range(int(seconds / DT)):
            self.step()
            if every:
                every(self)
