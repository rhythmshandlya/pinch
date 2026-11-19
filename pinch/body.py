"""Pinch's body: turns the mind's steps into joint targets every 20 ms (robot.crab_robot.Crab does the
safety checks and talks to the servos, simulated or real).

Motion skills:
  emote(name)            keyframed motion (optionally balanced by the trained emote policy)
  walk(vx, vy, wz, s)    the gait (scripted, or the trained walking policy)
  goto(xy, tol)          walk to a spot on the desk (crab-style: sideways is fine), never past the edge margin
  turn_to(xy)            face something
  pose(name, s)          hold a pose (stand, low, nap)
  push(obj, goal)        bulldoze a thing to a goal with the claws held low
  pick(obj)              line a claw up with the thing, open, reach down, close, lift
  drop(goal)             carry what it holds to the goal, put it down, let go

Positions come from pose_fn() -> (x, y, heading) and objects_fn(name) -> ((x, y, yaw), z) in the desk
frame: ground truth in the simulator; on the robot, odometry + the camera's view of the desk.
"""

import copy

import numpy as np

from crab import policy_io as pio
from crab.emotes import EMOTES, STAND, crouch, pose
from crab.gait import GAITS, Gait
from crab.params import JOINT_NAMES
from robot.crab_robot import DT
from robot.servos import SERVO_IDS
from pinch.world import Desk

J = {n: i for i, n in enumerate(JOINT_NAMES)}
SUB = 10                                               # gait integrates at 500 Hz inside each 20 ms tick
POSES = {"stand": STAND, "low": pose(crouch(0.25)), "nap": EMOTES["sleep"].keyframes[1][1]}
V_MAX, W_MAX = 0.06, 0.8
REACH_LOW = {"crouch": 0.38, "pitch": -0.13}           # jaws flat on the desk, pressing lightly (measured: -0.11)
CARRY_PITCH = 0.45                                      # carrying: the held part ~3 cm up, the short end clear of the legs
PUSH_LOWER = 0.011                                      # walk 11 mm lower while pushing: claw flat on the desk
PADDLE_PITCH = -0.28                                    # pushing: jaw blades ~1 mm above the desk (the old claw: -0.62)
PADDLE_CLAW = 0.5                                       # ... half open: the two tips cradle a round thing
PEN_HALF = 0.069                                        # half length of a pen (65 mm + rounded ends)
PEN_GRAB = 0.055                                        # hold a pen this far from its middle: its short end stays clear of the front leg
RAKE_ENABLED = False          # pulling things in from beyond them: flung a pen off the desk in tests, so off
# things the jaws must run along (their long side between the jaws): name -> grab point's offset from the middle
LONG_THINGS = {"pen": PEN_GRAB, "eraser": 0.0}
GRASP_ARM0 = (0.17, 0.08, 0.96)          # the arm when the learned grab takes over (training/grasp_env.py ARM0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def extent(name, yaw, heading, default=0.015):
    """Half-size of a thing along the push direction (a pen pushed end-on is 7 cm long, sideways 4.5 mm)."""
    if name == "pen":
        a = abs(np.cos(yaw - heading))
        return PEN_HALF * a + 0.0045 * np.sqrt(max(0.0, 1 - a * a))
    return {"cap": 0.014, "ball": 0.015, "eraser": 0.011}.get(name, default)


def arm_pose(base, side, pitch=None, yaw=None, claw=None):
    q = np.array(base, float).copy()
    s = "L" if side == "left" else "R"
    if pitch is not None:
        q[J[f"{s}_shoulder_pitch"]] = pitch
    if yaw is not None:
        q[J[f"{s}_shoulder_yaw"]] = yaw
    if claw is not None:
        q[J[f"{s}_claw"]] = claw
    return q


GAIT_LOOK = 5             # control ticks (0.1 s): how far ahead the walking policy sees the scripted gait

class Body:
    def __init__(self, crab, pose_fn, objects_fn=None, desk=None, tip_fn=None, emote_policy=None, walk_policy=None,
                 grasp_policy=None):
        self.crab, self.pose_fn, self.objects_fn = crab, pose_fn, objects_fn
        self.desk = desk or Desk()
        self.tip_fn = tip_fn                    # (side, arm joint q) -> claw tip (x, y) in the body frame
        self.emote_policy, self.walk_policy = emote_policy, walk_policy
        self._glast, self._gt, self._gseen = np.zeros(12), 0.0, -1.0
        self.grasp_policy = grasp_policy        # the learned grab (right claw; training/grasp_env.py), if exported
        self.grasp_view_fn = None               # (thing, side) -> (rel, ang, kind, size): what the camera sees
        self.task, self.busy, self.ok = None, False, True
        self.hold = STAND.copy()
        self.held = None                        # (object, side) while carrying
        self.grip_fn = None                     # (object, side) -> bool: is it between the fingers?
        self.gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * 0.52, 0.0]))
        self.t = 0.0
        self.breath = True
        self.status = ""
        self.edge_guarded = 0                   # walking commands the edge guard trimmed
        self.cmd_vel = (0.0, 0.0, 0.0)          # the walking command right now (body frame)
        self.cliff_blocked = []                 # unit vectors (body frame) the cliff ring can't vouch for
        self.pose_sigma_fn = None               # () -> how far off pose_fn() may be (m); widens the edge margin
        self.cliff_trimmed = 0                  # walking commands the cliff ring trimmed

    def trace(self, what, **kw):
        self.log.append((round(self.t, 2), what, kw))
        self.log = self.log[-200:]

    log = []

    # -- interface for the mind --------------------------------------------------------
    def start(self, step):
        kind, args = step[0], step[1:]
        make = getattr(self, f"_{kind}", None)
        if make is None:
            self.busy, self.ok = False, False
            return
        self.task = make(*args)
        self.busy, self.ok, self.status = True, True, kind
        self._reflexed = False

    def reflex(self, step):
        """A safety reflex (backing off an edge) takes over from whatever was running. When it ends, the step
        it cut short counts as failed - the mind must not take "backed off the edge" for "picked up the pen"."""
        interrupted = self.busy
        self.cancel()
        self.start(step)
        self._reflexed = interrupted

    _reflexed = False

    def cancel(self):
        if self.task is not None:
            self.task.close()
        self.task, self.busy, self.ok = None, False, False

    def tick(self):
        """One 20 ms control step."""
        self.t += DT
        self.cmd_vel = (0.0, 0.0, 0.0)
        if self.task is not None:
            try:
                q = next(self.task)
            except StopIteration as e:
                self.ok = e.value is not False and not self._reflexed
                self.task, self.busy, self._reflexed = None, False, False
                q = self.hold
        else:
            q = self.hold
        if self.task is None and self.breath:              # alive even when still: slow breathing
            q = q.copy()
            for lg in ("FL", "FR", "RL", "RR"):
                q[J[f"{lg}_femur"]] += 0.015 * np.sin(2 * np.pi * self.t / 3.5)
        if self.held:
            obj, side = self.held
            q = arm_pose(q, side, pitch=max(q[J[f"{'L' if side == 'left' else 'R'}_shoulder_pitch"]], CARRY_PITCH)
                         if self.task is None else None, claw=0.0)
        self.crab.tick(q)
        return q

    # -- basic motions -------------------------------------------------------------------
    def _blend_to(self, q1, seconds):
        q0 = self.crab.q_cmd.copy()
        n = max(1, int(seconds / DT))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            yield q0 + (q1 - q0) * s

    def _emote(self, name):
        """Keyframes; with the trained emote policy (policies/emote.npz) the legs get small live
        corrections that keep the emote balanced under shoves, slippery spots or a heavier battery."""
        emo = EMOTES.get(name)
        if emo is None:
            return False
        t, last = 0.0, np.zeros(12)
        while t < emo.duration:
            q = emo.target(t)
            if self.emote_policy is not None and not self.held:
                ahead = emo.target(min(t + 0.1, emo.duration))
                gyro, grav = self.crab.hw.read_imu()
                qj, st = self.crab.read_joints()
                qd = np.array([self.crab.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
                obs = pio.emote_observation(np, gyro, grav, qj[pio.LEG_ACT_IDX], qd[pio.LEG_ACT_IDX], last, q, ahead,
                                            None, t / emo.duration)
                last = np.clip(self.emote_policy(obs), -1, 1)
                q = pio.emote_targets(np, q, last)
            if self.held:
                q = arm_pose(q, self.held[1], claw=0.0)
            yield q
            t += DT
        self.hold = STAND.copy()
        return True

    def _pose(self, name, seconds=2.0):
        q = POSES.get(name, STAND)
        yield from self._blend_to(q, 0.6)
        for _ in range(int(seconds / DT)):
            yield q
        self.hold = q.copy()
        return True

    FEET_EDGE = 0.02          # m: no walking or turning that brings a foot closer than this to the desk edge
                              # (the gait swings each foot ~2 cm, and the desk position is good to ~1 cm)

    def _edge_guard(self, cmd):
        """Block the parts of a walking command that would carry a foot over the desk edge in the next 0.4 s
        (moving along the edge, or away from it, stays allowed). Uses Pinch's own desk position (the marker),
        so it works backwards and sideways too, where the cliff sensors can't see."""
        if self.cliff_blocked:                             # no walking toward a side no beam vouches for
            from pinch.touch import limit_velocity
            v = limit_velocity(cmd[:2], self.cliff_blocked)
            if abs(v[0] - cmd[0]) > 1e-9 or abs(v[1] - cmd[1]) > 1e-9:
                cmd = (float(v[0]), float(v[1]), cmd[2])
                self.cliff_trimmed += 1
        if self.pose_fn is None or not any(abs(c) > 1e-6 for c in cmd):
            return cmd
        x, y, h = self.pose_fn()
        now = self.desk.feet_margin((x, y), h)
        # between camera fixes the position is integrated from the walking command and can be off: keep
        # that much further from the edge (pinch_main.Localizer.sigma)
        need = self.FEET_EDGE + (self.pose_sigma_fn() if self.pose_sigma_fn is not None else 0.0)

        def after(vx, vy, wz, T=0.4):
            c, s = np.cos(h), np.sin(h)
            return self.desk.feet_margin((x + (c * vx - s * vy) * T, y + (s * vx + c * vy) * T), h + wz * T)

        vx, vy, wz = cmd
        if after(vx, vy, wz) >= min(need, now):
            return cmd
        self.edge_guarded += 1
        for alt in ((vx, vy, 0.0), (0.0, 0.0, wz), (vx, 0.0, 0.0), (0.0, vy, 0.0)):
            if any(abs(c) > 1e-6 for c in alt) and after(*alt) >= min(need, now):
                return alt
        return (0.0, 0.0, 0.0)

    def _gait_q(self, cmd, arms=None, lower=0.0):
        cmd = self._edge_guard(cmd)
        if np.hypot(cmd[0], cmd[1]) > 0.005:                    # where it was trying to go (for a climb)
            self.walk_dir = np.array(cmd[:2]) / np.hypot(cmd[0], cmd[1])
        self.cmd_vel = tuple(float(c) for c in cmd)          # what the localizer integrates on the robot
        self.gait.arms = arms
        self.gait.lower += float(np.clip(lower - self.gait.lower, -0.0004, 0.0004))   # sink / rise smoothly
        for k in range(SUB):                                   # leg IK only once per tick, after the last substep
            q = self.gait.step(DT / SUB, cmd, solve=k == SUB - 1)
        return self._residual(q, cmd, arms is None and self.gait.lower < 1e-4)

    def _residual(self, ref, cmd, plain):
        """The trained walking corrections (policies/gait.npz) on top of the scripted trot. Its look-ahead (the
        reference 0.1 s on, as in training) comes from a copy of the gait run on with the same command - no lag
        for the servos (a 0.1 s delay line cost the tidy planners most of their precision). Only for plain
        walking: pushes (paddle arms, crouched) keep the pure scripted gait, which the policy never saw."""
        if self.walk_policy is None or not plain:
            self._gseen = -1.0
            return ref
        if self.t - self._gseen > 2.5 * DT:                        # (re)starting a walk
            self._glast, self._gt = np.zeros(12), 0.0
        self._gseen = self.t
        g = copy.deepcopy(self.gait)                 # 5 coarse 20 ms steps: 0.8 ms, within 0.02 rad of fine ones
        for k in range(GAIT_LOOK):
            ahead = g.step(DT, cmd, solve=k == GAIT_LOOK - 1)
        try:
            gyro, grav = self.crab.hw.read_imu()
            qj, st = self.crab.read_joints()
        except Exception:                            # a sensor hiccup: this tick walks on the plain gait
            return ref
        qd = np.array([self.crab.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
        obs = pio.emote_observation(np, gyro, grav, qj[pio.LEG_ACT_IDX], qd[pio.LEG_ACT_IDX], self._glast, ref, ahead,
                                    None, (self._gt % 6.0) / 6.0)
        self._glast = np.clip(self.walk_policy(obs), -1, 1)
        self._gt += DT
        return pio.emote_targets(np, ref, self._glast)

    def _walk(self, vx, vy, wz, seconds):
        for _ in range(int(seconds / DT)):
            yield self._gait_q((vx, vy, wz), self._arms())
        yield from self._settle()
        return True

    def _settle(self, seconds=0.6):
        for _ in range(int(seconds / DT)):
            yield self._gait_q((0.0, 0.0, 0.0), self._arms())
        while self.gait.lower > 1e-4:
            yield self._gait_q((0.0, 0.0, 0.0), self._arms())
        self.gait = Gait(GAITS["trot"], com_offset=self.gait.com_offset)
        self.hold = STAND.copy() if not self.held else arm_pose(STAND, self.held[1], pitch=CARRY_PITCH, claw=0.0)

    def _arms(self):
        if self.held:
            return arm_pose(STAND, self.held[1], pitch=CARRY_PITCH, claw=0.0)
        return None

    # -- moving around the desk ------------------------------------------------------------
    def _goto(self, xy, tol=0.03, heading=None, timeout=40.0, slow=1.0, htol=0.15):
        xy = np.array(self.desk.clamp(xy))
        h_end = heading if heading is not None else self.pose_fn()[2]
        fitted = np.array(self.desk.fit_feet(xy, h_end, self.FEET_EDGE + 0.005))
        if np.linalg.norm(fitted - xy) > 1e-4:                # (the feet wouldn't fit there: stop short)
            self.trace("goto_fitted", asked=tuple(np.round(xy, 3)), to=tuple(np.round(fitted, 3)))
            xy = fitted
        settle_n = 0
        for k in range(int(timeout / DT)):
            if self.held and self.grip_fn is not None and k % 10 == 0 and not self.grip_fn(*self.held):
                self.trace("lost_hold", obj=self.held[0])         # it slipped out on the way: say so, and stop
                self.held = None
                self.hold = STAND.copy()
                return False
            x, y, h = self.pose_fn()
            dx, dy = xy[0] - x, xy[1] - y
            dist = np.hypot(dx, dy)
            herr = 0.0 if heading is None else _wrap(heading - h)
            if dist < tol and abs(herr) < htol:
                settle_n += 1
                if settle_n > 3:
                    break
            else:
                settle_n = 0
            ex, ey = np.cos(h) * dx + np.sin(h) * dy, -np.sin(h) * dx + np.cos(h) * dy
            v = np.array([ex, ey]) * 1.6
            sp = np.linalg.norm(v)
            vmax = V_MAX * slow
            if sp > vmax:
                v *= vmax / sp
            elif 0 < sp < 0.012 and dist > tol:                    # small steps still need some speed
                v *= 0.012 / sp
            if dist < tol:
                v[:] = 0
            wmax = W_MAX * (0.4 if self.held else 1.0)               # a long pen in the claw: turn gently
            wz = float(np.clip(2.0 * herr, -wmax, wmax))
            if abs(herr) < htol * 0.7 and dist < tol:
                wz = 0.0
            yield self._gait_q((float(v[0]), float(v[1]), wz), self._arms())
        yield from self._settle()
        x, y, h = self.pose_fn()
        ok = np.hypot(xy[0] - x, xy[1] - y) < tol * 1.6 and (heading is None or abs(_wrap(heading - h)) < 0.3)
        return bool(ok)

    def _turn_to(self, xy, tol=0.15):
        x, y, h = self.pose_fn()
        heading = np.arctan2(xy[1] - y, xy[0] - x)
        return (yield from self._goto((x, y), tol=0.5, heading=heading, timeout=12.0))

    # -- manipulation ------------------------------------------------------------------------
    def _obj(self, name):
        return self.objects_fn(name) if self.objects_fn else None

    def _push(self, name, goal, timeout=150.0, radius=0.015):
        """One claw, closed and held flat on the desk, is the paddle. The tray only opens toward the front,
        so a push into the tray goes via a spot in front of its entrance."""
        goal = np.array(goal, float)
        if np.allclose(goal, self.desk.tray_xy):
            tx, ty = self.desk.tray_xy
            entrance = np.array((tx, ty - self.desk.tray_half[1] - 0.05))
            (ox, oy, _), _ = self._obj(name)
            if oy < ty - self.desk.tray_half[1] - 0.01 or abs(ox - tx) > self.desk.tray_half[0]:
                ok = yield from self._push_to(name, entrance, timeout * 0.6, radius, tol=0.035)
                if not ok:
                    return False
            (ox, oy, _), _ = self._obj(name)
            hx = self.desk.tray_half[0]
            straight_in = (float(np.clip(ox, tx - hx + 0.025, tx + hx - 0.025)), ty + 0.01)
            return (yield from self._push_to(name, straight_in, timeout * 0.4, radius, tray=True))
        return (yield from self._push_to(name, goal, timeout, radius))

    def _push_to(self, name, goal, timeout=90.0, radius=0.015, tol=0.03, tray=False):
        t_end = self.t + timeout
        goal = np.array(goal, float)
        to_tray = tray or np.allclose(goal, self.desk.tray_xy)
        while self.t < t_end:
            o = self._obj(name)
            if o is None:
                return False
            (ox, oy, _), _ = o
            to_goal = goal - (ox, oy)
            d = np.linalg.norm(to_goal)
            if d < tol or (to_tray and self.desk.in_tray((ox, oy))):
                yield from self._blend_to(STAND, 0.5)
                yield from self._settle()
                return True
            u = to_goal / d
            heading = float(np.arctan2(u[1], u[0]))
            radius = extent(name, o[0][2], heading, radius)   # how far the thing reaches back toward the paddle
            x, y, h = self.pose_fn()
            side = self._side_for(x, y, h, ox, oy)
            paddle = arm_pose(arm_pose(pose(crouch(0.25)), side, pitch=PADDLE_PITCH, yaw=0.0, claw=PADDLE_CLAW),
                              "right" if side == "left" else "left", pitch=0.5)
            tip = np.array(self.tip_fn(side, paddle) if self.tip_fn else (0.085, 0.03 if side == "left" else -0.03))
            c, s = np.cos(heading), np.sin(heading)
            tip_w = np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
            # stand 3 cm behind it to line up; near an edge, stand closer (down to touching) before giving up
            for gap in (0.03, 0.02, 0.01, 0.0):
                start = np.array((ox, oy)) - u * (radius + gap) - tip_w
                fits = (self.desk.inside(start, self.desk.edge_margin) and           # (as _goto will clamp
                        self.desk.feet_clear(start, heading) >= self.FEET_EDGE + 0.005)  # and fit it)
                if fits:
                    break
            # can't get behind it from the desk (its feet, not just its centre, must fit there)
            if not fits:
                if RAKE_ENABLED:
                    ok = yield from self._rake(name, goal, radius)
                    if ok is not None:
                        continue
                self.trace("unreachable", obj=(round(ox, 3), round(oy, 3)), start=tuple(np.round(start, 3)),
                           feet=round(float(self.desk.feet_clear(start, heading)), 3))
                return False
            self.trace("lineup", obj=(round(ox, 3), round(oy, 3)), start=tuple(np.round(start, 3)), side=side)
            yield from self._detour((x, y), start, (ox, oy))
            ok = yield from self._goto(start, tol=0.012, heading=heading, timeout=25.0, slow=0.7)
            (ox2, oy2, _), _ = self._obj(name)
            self.trace("lined_up" if ok else "lineup_failed", obj=(round(ox2, 3), round(oy2, 3)),
                       moved=round(float(np.hypot(ox2 - ox, oy2 - oy)), 3))
            if not ok:
                continue
            yield from self._blend_to(paddle, 0.6)
            for _ in range(int(15.0 / DT)):                       # push along the line, steering on it
                (ox, oy, _), _ = self._obj(name)
                x, y, h = self.pose_fn()
                c, s = np.cos(h), np.sin(h)
                tip_now = np.array((x + c * tip[0] - s * tip[1], y + s * tip[0] + c * tip[1]))
                rel = np.array((ox, oy)) - tip_now
                ahead = c * rel[0] + s * rel[1]
                lateral = -s * rel[0] + c * rel[1]
                if (to_tray and self.desk.in_tray((ox, oy))) or np.linalg.norm(goal - (ox, oy)) < tol:
                    break
                if abs(lateral) > 0.026 or ahead < -(radius + 0.055) or ahead > 0.07 or not self.desk.inside((x, y), 0.05):
                    self.trace("slip", lateral=round(float(lateral), 3), ahead=round(float(ahead), 3),
                               obj=(round(ox, 3), round(oy, 3)))
                    break                                          # slipped off the paddle: line up again
                herr = _wrap(np.arctan2(goal[1] - oy, goal[0] - ox) - h)
                # ease off near the goal: a shoved cap slides on past it (into the tray's side wall)
                v = float(np.clip(0.042 * np.linalg.norm(goal - (ox, oy)) / 0.08, 0.015, 0.042))
                yield self._gait_q((v, float(np.clip(1.5 * lateral, -0.02, 0.02)),
                                    float(np.clip(1.5 * herr, -0.4, 0.4))), paddle, lower=PUSH_LOWER)
                self.pushes = getattr(self, "pushes", 0) + 1
            yield from self._settle(0.3)
        return False

    def _detour(self, frm, to, obj, clear=0.10, half=0.0, yaw=0.0):
        """If the straight walk from frm to `to` would bump obj, go round it first. A long thing (a pen:
        half-length `half` along `yaw`) is a line, not a point: the way round passes beyond its nearer end."""
        a, b, o = np.array(frm), np.array(to), np.array(obj)
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-6:
            return
        u = np.array((np.cos(yaw), np.sin(yaw)))
        pts = [o + u * t for t in np.linspace(-half, half, 9)] if half > 0 else [o]

        def gap(p, q0, q1):                                      # distance from p to the segment q0-q1
            d = q1 - q0
            k = np.clip(np.dot(p - q0, d) / max(np.dot(d, d), 1e-12), 0, 1)
            return np.linalg.norm(q0 + k * d - p)

        if min(gap(p, a, b) for p in pts) > clear:
            return
        if half > 0:                                             # round the nearer end of the line
            cands = [o + u * (half + clear + 0.03), o - u * (half + clear + 0.03)]
        else:
            n = np.array((-ab[1], ab[0])) / L
            cands = [o + n * (clear + 0.05), o - n * (clear + 0.05)]
        cands = [self.desk.clamp(w) for w in cands]
        w = min(cands, key=lambda p: np.linalg.norm(p - a) + np.linalg.norm(b - p) - 0.5 * self.desk.edge_distance(p))
        self.trace("detour", via=tuple(np.round(w, 3)))
        yield from self._goto(w, tol=0.04, timeout=20.0)

    def _rake(self, name, goal, radius=0.015, timeout=40.0):
        """Reach past the thing with the flat claw and walk backwards, pulling it toward the goal."""
        (ox, oy, _), _ = self._obj(name)
        u = np.array((ox, oy)) - np.array(goal)
        d = np.linalg.norm(u)
        if d < 0.03:
            return True
        u /= d
        heading = float(np.arctan2(u[1], u[0]))                  # face the thing, from the goal's side
        x, y, h = self.pose_fn()
        side = self._side_for(x, y, h, ox, oy)
        paddle = arm_pose(arm_pose(pose(crouch(0.25)), side, pitch=PADDLE_PITCH, yaw=0.0, claw=PADDLE_CLAW),
                          "right" if side == "left" else "left", pitch=0.5)
        tip = np.array(self.tip_fn(side, paddle) if self.tip_fn else (0.085, 0.03 if side == "left" else -0.03))
        c, s = np.cos(heading), np.sin(heading)
        tip_w = np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
        start = np.array((ox, oy)) + u * (radius + 0.02) - tip_w
        if not self.desk.inside(start, self.desk.edge_margin - 0.004):
            self.trace("rake_unreachable", start=tuple(np.round(start, 3)))
            return None
        raised = arm_pose(paddle, side, pitch=0.35)
        self.trace("rake", obj=(round(ox, 3), round(oy, 3)), start=tuple(np.round(start, 3)))
        yield from self._blend_to(raised, 0.4)
        yield from self._detour((x, y), start, (ox, oy))
        ok = yield from self._goto(start, tol=0.012, heading=heading, timeout=25.0, slow=0.7)
        if not ok:
            return False
        yield from self._blend_to(paddle, 0.6)                    # claw comes down beyond the thing
        t_end = self.t + timeout
        while self.t < t_end:
            (ox, oy, _), _ = self._obj(name)
            if np.linalg.norm(np.array(goal) - (ox, oy)) < 0.035:
                break
            x, y, h = self.pose_fn()
            c, s = np.cos(h), np.sin(h)
            rel = np.array((ox, oy)) - (x + c * tip[0] - s * tip[1], y + s * tip[0] + c * tip[1])
            lateral = -s * rel[0] + c * rel[1]
            if abs(lateral) > 0.03 or c * rel[0] + s * rel[1] > 0.015:
                self.trace("rake_slip", lateral=round(float(lateral), 3))
                break
            yield self._gait_q((-0.035, float(np.clip(1.5 * lateral, -0.02, 0.02)), 0.0), paddle, lower=PUSH_LOWER)
        yield from self._settle(0.3)
        return True

    def _side_for(self, x, y, h, ox, oy):
        ex, ey = np.cos(h) * (ox - x) + np.sin(h) * (oy - y), -np.sin(h) * (ox - x) + np.cos(h) * (oy - y)
        return "left" if ey > 0 else "right"

    def _pick(self, name, timeout=60.0):
        """Line a claw up with the thing (a pen lies along the finger), reach down, close, lift."""
        t_end = self.t + timeout
        for attempt in range(3):
            if self.t > t_end:
                break
            o = self._obj(name)
            if o is None:
                return False
            (ox, oy, oyaw), oz = o
            x, y, h = self.pose_fn()
            # where to stand: every approach (a pen: along its axis, either way; anything else: from any of 24
            # directions) with either claw, the nearest one whose feet fit on the desk clear of the tray. Facing
            # it from wherever Pinch happened to be put the stand off the desk for half the spots (decision 56).
            sides = ("right",) if self.grasp_policy is not None and self.grasp_view_fn is not None else \
                (self._side_for(x, y, h, ox, oy), "left" if self._side_for(x, y, h, ox, oy) == "right" else "right")
            grab0 = np.array((ox, oy))
            best = None
            for si, sd in enumerate(sides):
                sg = 1 if sd == "left" else -1
                for yw in ((0.12, 0.0, 0.24, 0.33) if name in LONG_THINGS else (0.12,)):   # the arm swings 0-20 deg
                    rq = arm_pose(pose(crouch(REACH_LOW["crouch"], "F?"), crouch(-0.20, "R?")), sd,
                                  pitch=REACH_LOW["pitch"], yaw=yw, claw=0.9)          # nose down: jaws come in low
                    tp = np.array(self.tip_fn(sd, rq) if self.tip_fn else (0.088, sg * 0.035))
                    ad = np.arctan2(tp[1], tp[0])
                    if name in LONG_THINGS:
                        # a pen: the jaws run along it, gripping it a little off its middle so the longer part
                        # points forward, away from Pinch's own legs (the palm clears the shorter part)
                        opts = []
                        for a in (oyaw, oyaw + np.pi):
                            ax = np.array((np.cos(a), np.sin(a)))
                            opts.append((_wrap(a - ad), grab0 - ax * LONG_THINGS[name]))
                    else:
                        face = np.arctan2(oy - y, ox - x)
                        opts = [(_wrap(face + k * np.pi / 12 - ad), grab0) for k in range(-12, 12)]
                    for hd, g in opts:
                        c, s_ = np.cos(hd), np.sin(hd)
                        st = g - np.array((c * tp[0] - s_ * tp[1], s_ * tp[0] + c * tp[1]))
                        room = min(self.desk.edge_distance(st) - 0.066,
                                   self.desk.feet_clear(st, hd) - self.FEET_EDGE - 0.005)
                        # cheapest: feet fit (some room to spare), little walking and turning, the nearer claw,
                        # the usual arm swing
                        cost = (100.0 if room < 0 else 0.0) + 3 * max(0.0, 0.03 - room) / 0.03 +                             np.hypot(st[0] - x, st[1] - y) / 0.10 + 0.3 * abs(_wrap(hd - h)) + 0.3 * si + abs(yw - 0.12)
                        if best is None or cost < best[0]:
                            best = (cost, sd, rq, tp, hd, st, g)
            _, side, reach, tip, heading, stand_xy, grab = best
            sgn = 1 if side == "left" else -1
            self.trace("pick_plan", side=side, tip=tuple(np.round(tip, 3)), stand=tuple(np.round(stand_xy, 3)),
                       heading=round(float(heading), 2), obj=(round(ox, 3), round(oy, 3), round(oyaw, 2)))
            fwd = np.array((np.cos(heading), np.sin(heading)))
            if self.desk.edge_distance(stand_xy) < 0.066 or \
                    self.desk.feet_clear(stand_xy, heading) < self.FEET_EDGE + 0.005:
                self.trace("pick_unreachable")                     # (too near an edge: push it, or ask for help)
                return False
            # line up 5 cm short of the spot, facing the final way, then walk the last bit straight in: the
            # legs never sweep across the thing (turning on the spot next to a pen kicks it)
            pre = stand_xy - fwd * 0.05
            half = PEN_HALF if name == "pen" else 0.0 if name not in LONG_THINGS else 0.011
            if self.desk.edge_distance(pre) >= 0.066:                # (no room for it near an edge: walk straight in)
                yield from self._detour((x, y), pre, (ox, oy), clear=0.12, half=half, yaw=oyaw)
                yield from self._goto(pre, tol=0.015, heading=heading, timeout=30.0, slow=0.8, htol=0.08)
            else:
                yield from self._detour((x, y), stand_xy, (ox, oy), clear=0.12, half=half, yaw=oyaw)
            ok = yield from self._goto(stand_xy, tol=0.010, heading=heading, timeout=30.0, slow=0.6, htol=0.06)
            xx, yy, hh = self.pose_fn()
            self.trace("pick_at", ok=ok, err=round(float(np.hypot(xx - stand_xy[0], yy - stand_xy[1])), 4),
                       herr=round(float(_wrap(heading - hh)), 3))
            if not ok:
                continue
            yaw = yield from self._align(name, side, grab, reach)
            if abs(self.last_miss) > 0.004:
                continue                                         # not lined up: walk round and try again
            reach = arm_pose(reach, side, yaw=yaw)
            if self.grasp_policy is not None and self.grasp_view_fn is not None and side == "right":
                yaw = yield from self._learned_grasp(name)       # the last few centimetres, learned
            else:
                yield from self._blend_to(arm_pose(STAND, side, pitch=0.2, yaw=yaw, claw=0.9), 0.5)
                yield from self._blend_to(reach, 0.9)
                for _ in range(15):
                    yield reach
                closed = arm_pose(reach, side, claw=0.0)
                yield from self._blend_to(closed, 0.4)
                for _ in range(10):
                    yield closed
            lift = arm_pose(pose(crouch(0.30)), side, pitch=CARRY_PITCH, yaw=yaw, claw=0.0)
            yield from self._blend_to(lift, 1.5)                   # slow: a pen held near its end pivots
            o = self._obj(name)
            held = self.grip_fn(name, side) if self.grip_fn else (o is not None and o[1] > oz + 0.004)
            self.trace("pick_lift", z0=round(oz, 4), z=round(o[1], 4) if o else None, held=held)
            if held:                                                # it came up with the claw
                self.held = (name, side)
                self.hold = lift
                return True
            yield from self._blend_to(STAND, 0.5)
        return False

    def _learned_grasp(self, name, seconds=4.0):
        """The learned grab (training/grasp_env.py): from the grasp crouch with the claw raised and open, the
        policy shifts and leans the body with the legs and swings, lowers, closes and lifts the right arm,
        watching the thing through the camera. Returns the arm's swing at the end (for the carry pose)."""
        arm = np.array(GRASP_ARM0, float)
        q = np.array(pio.GRASP_NOMINAL, float)
        q[pio.GRASP_ARM_IDX] = arm
        yield from self._blend_to(q, 0.8)
        last, t = np.zeros(pio.GRASP_N_ACT), 0.0
        while t < seconds:
            view = self.grasp_view_fn(name, "right")
            if view is None:
                break
            rel, ang, kind, size = view
            gyro, grav = self.crab.hw.read_imu()
            qj, st = self.crab.read_joints()
            qd = np.array([self.crab.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
            obs = pio.grasp_observation(np, gyro, grav, qj[pio.LEG_ACT_IDX], qd[pio.LEG_ACT_IDX], last, arm,
                                        np.asarray(rel), ang, kind, np.asarray(size), t / 4.0)
            last = np.clip(self.grasp_policy(obs), -1, 1)
            q = pio.grasp_targets(np, pio.GRASP_NOMINAL, last, arm_prev=arm)
            arm = q[pio.GRASP_ARM_IDX]
            yield q
            t += DT
        self.trace("learned_grasp", t=round(t, 2), arm=tuple(np.round(arm, 2)))
        return float(arm[0])

    def _lateral(self, name, side, grab, q):
        """Sideways miss (m) between the open claw's gap and the grab point, across the pen's axis."""
        x, y, h = self.pose_fn()
        tip = np.array(self.tip_fn(side, q))
        c, s = np.cos(h), np.sin(h)
        gap = np.array((x + c * tip[0] - s * tip[1], y + s * tip[0] + c * tip[1]))
        (ox, oy, oyaw), _ = self._obj(name)
        if name in LONG_THINGS:
            n = np.array((-np.sin(oyaw), np.cos(oyaw)))
            ax = np.array((np.cos(oyaw), np.sin(oyaw)))
            g = np.array(grab) if grab is not None else np.array((ox, oy))
            g = np.array((ox, oy)) + ax * np.dot(g - (ox, oy), ax)          # the grab point, kept on the pen
            return float(np.dot(gap - g, n)), n
        d = gap - (ox, oy)
        n = np.array((-s, c))
        return float(np.dot(d, n)), n

    def _align(self, name, side, grab, reach, tol=0.003):
        """Fine-tune the claw over the thing: swing the arm (its yaw moves the claw ~5 cm per radian),
        and take tiny side steps if the swing joint runs out of range."""
        yaw = 0.12
        for _ in range(6):
            e, n = self._lateral(name, side, grab, arm_pose(reach, side, yaw=yaw))
            if abs(e) < tol:
                break
            e2, _ = self._lateral(name, side, grab, arm_pose(reach, side, yaw=yaw + 0.05))
            slope = (e2 - e) / 0.05
            if abs(slope) > 1e-4:
                yaw_new = float(np.clip(yaw - e / slope, 0.04, 0.20))   # small swings keep the finger parallel
                e_new, _ = self._lateral(name, side, grab, arm_pose(reach, side, yaw=yaw_new))
                yaw = yaw_new
                if abs(e_new) < tol:
                    break
                e = e_new
            if abs(e) > 0.02:                                     # too far for small steps: give up here
                break
            x, y, h = self.pose_fn()                              # side step against the miss
            v_body = np.array((np.cos(h) * n[0] + np.sin(h) * n[1], -np.sin(h) * n[0] + np.cos(h) * n[1]))
            step = -np.sign(e) * v_body * 0.010
            for _ in range(int(min(0.6, abs(e) / 0.010) / DT)):
                yield self._gait_q((float(step[0]), float(step[1]), 0.0))
            yield from self._settle(0.3)
            e_after, _ = self._lateral(name, side, grab, arm_pose(reach, side, yaw=yaw))
            if abs(e_after) > abs(e):                             # made it worse: stop stepping
                break
        e, _ = self._lateral(name, side, grab, arm_pose(reach, side, yaw=yaw))
        self.trace("align", yaw=round(yaw, 3), miss_mm=round(e * 1000, 1))
        self.last_miss = e
        return yaw

    walk_dir = None

    def _recover(self, seconds=2.5):
        """Work free. A foot over the edge: shuffle the body back toward the middle of the desk with short steps
        (the stuck leg is dragged back up), then settle. All feet on the desk, so caught on something (a
        cable, a pen, a book's edge): climb on over it the way it was going, lifting its feet high."""
        x, y, h = self.pose_fn()
        if self.walk_dir is not None and self.desk.feet_margin((x, y), h) > 0.0:
            return (yield from self._climb(self.walk_dir))
        d = -np.array((x, y))
        d /= np.linalg.norm(d) + 1e-9
        c, s = np.cos(h), np.sin(h)
        bx, by = c * d[0] + s * d[1], -s * d[0] + c * d[1]           # toward the middle, in the body frame
        self.trace("recover", at=(round(x, 3), round(y, 3)), feet_margin=round(self.desk.feet_margin((x, y), h), 3))
        for _ in range(int(seconds / DT)):
            yield self._gait_q((0.025 * bx, 0.025 * by, 0.0), self._arms())
        yield from self._settle(0.4)
        return True

    def _climb(self, direction, seconds=6.0, speed=0.02):
        """The high-stepping crawl (crab/gait.py "climb": 20 mm foot lift) for a few seconds, then the trot again."""
        x, y, h = self.pose_fn()
        self.trace("climb", at=(round(x, 3), round(y, 3)), dir=tuple(np.round(direction, 2)))
        yield from self._blend_to(STAND, 0.3)
        trot, self.gait = self.gait, Gait(GAITS["climb"], com_offset=np.array([0.015 * 0.52, 0.0]))
        try:
            for _ in range(int(seconds / DT)):
                cmd = self._edge_guard((speed * direction[0], speed * direction[1], 0.0))
                self.cmd_vel = tuple(float(c) for c in cmd)
                for k in range(SUB):
                    q = self.gait.step(DT / SUB, cmd, solve=k == SUB - 1)
                yield q if self._arms() is None else arm_pose(q, self.held[1], pitch=CARRY_PITCH, claw=0.0)
            for _ in range(int(1.0 / DT)):                       # let the crawl come to rest on four feet
                for k in range(SUB):
                    q = self.gait.step(DT / SUB, (0.0, 0.0, 0.0), solve=k == SUB - 1)
                yield q
        finally:
            self.gait = trot
            self.gait.cmd[:] = 0.0
        yield from self._settle(0.4)
        return True

    def _let_go(self, q, side):
        """After opening the claw: let the thing settle, lift the arm away with the claw still wide open, and only
        then close it (closing on the way up grabbed the pen again: it rode back up in the claw)."""
        open_q = arm_pose(q, side, claw=0.9)
        for _ in range(int(0.3 / DT)):
            yield open_q
        yield from self._blend_to(arm_pose(STAND, side, pitch=0.3, claw=0.9), 0.6)
        yield from self._blend_to(STAND, 0.5)

    def _release(self):
        """Put the held thing down right here: lower the claw to just above the desk and open it."""
        if not self.held:
            return False
        name, side = self.held
        down = arm_pose(pose(crouch(REACH_LOW["crouch"], "F?"), crouch(-0.20, "R?")), side,
                        pitch=REACH_LOW["pitch"] + 0.12, claw=0.0)
        yield from self._blend_to(down, 0.7)
        yield from self._blend_to(arm_pose(down, side, claw=0.9), 0.3)
        self.held = None
        yield from self._let_go(down, side)
        self.hold = STAND.copy()
        return True

    def _drop(self, goal):
        """Carry the held thing to the goal and put it down there."""
        if not self.held:
            return False
        name, side = self.held
        goal = np.array(goal, float)
        x, y, h = self.pose_fn()
        reach = arm_pose(pose(crouch(REACH_LOW["crouch"])), side, pitch=REACH_LOW["pitch"] + 0.15, yaw=0.12, claw=0.0)
        tip = np.array(self.tip_fn(side, reach) if self.tip_fn else (0.085, (1 if side == "left" else -1) * 0.035))
        if name == "pen" and np.allclose(goal, self.desk.tray_xy):
            # lay it lengthwise in the tray: claw near the tray's left end, the pen's longer part pointing right
            goal = np.array((goal[0] - self.desk.tray_half[0] + 0.045, goal[1]))
            heading = float(-np.arctan2(tip[1], tip[0]))
        else:
            # face it from where Pinch is - or, if that puts a foot over an edge or in the tray, the nearest
            # heading that doesn't (round the tray's open front)
            direct = float(np.arctan2(goal[1] - y, goal[0] - x) - np.arctan2(tip[1], tip[0]))
            best = None
            for dh in sorted(np.linspace(-np.pi, np.pi, 25), key=abs):
                hd = direct + dh
                c, s = np.cos(hd), np.sin(hd)
                st = goal - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
                if self.desk.feet_clear(st, hd) >= self.FEET_EDGE + 0.005:
                    best = hd
                    break
            heading = _wrap(best if best is not None else direct)
        c, s = np.cos(heading), np.sin(heading)
        stand_xy = goal - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
        ok = yield from self._goto(self.desk.clamp(stand_xy, 0.06), tol=0.012, heading=heading, timeout=50.0, slow=0.5)
        if not self.held:                                    # it slipped out on the way (traced in _goto)
            return False
        yield from self._blend_to(reach, 0.8)
        yield from self._blend_to(arm_pose(reach, side, claw=0.9), 0.4)
        self.held = None
        yield from self._let_go(reach, side)
        self.hold = STAND.copy()
        o = self._obj(name)
        if o is None:
            return ok
        (ox, oy, _), _ = o
        self.trace("dropped", obj=(round(ox, 3), round(oy, 3)), in_tray=self.desk.in_tray((ox, oy)))
        return bool(np.hypot(ox - goal[0], oy - goal[1]) < 0.06 or self.desk.in_tray((ox, oy)))
