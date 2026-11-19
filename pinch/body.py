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

import numpy as np

from crab.emotes import EMOTES, STAND, crouch, pose
from crab.gait import GAITS, Gait
from crab.params import JOINT_NAMES
from robot.crab_robot import DT
from pinch.world import Desk

J = {n: i for i, n in enumerate(JOINT_NAMES)}
SUB = 10                                               # gait integrates at 500 Hz inside each 20 ms tick
POSES = {"stand": STAND, "low": pose(crouch(0.25)), "nap": EMOTES["sleep"].keyframes[1][1]}
V_MAX, W_MAX = 0.06, 0.8
REACH_LOW = {"crouch": 0.38, "pitch": -0.69}           # claw fingers touch the desk (measured in the sim)
CARRY_PITCH = -0.38                                     # carrying: near end ~2 cm up, a pen's far end drags
PUSH_LOWER = 0.011
PEN_HALF = 0.069
RAKE_ENABLED = False          # pulling things in from beyond them: flung a pen off the desk in tests, so off                                        # half length of a pen (65 mm + rounded ends)                                      # walk 11 mm lower while pushing: claw flat on the desk


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


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


class Body:
    def __init__(self, crab, pose_fn, objects_fn=None, desk=None, tip_fn=None, emote_policy=None, walk_policy=None):
        self.crab, self.pose_fn, self.objects_fn = crab, pose_fn, objects_fn
        self.desk = desk or Desk()
        self.tip_fn = tip_fn                    # (side, arm joint q) -> claw tip (x, y) in the body frame
        self.emote_policy, self.walk_policy = emote_policy, walk_policy
        self.task, self.busy, self.ok = None, False, True
        self.hold = STAND.copy()
        self.held = None                        # (object, side) while carrying
        self.grip_fn = None                     # (object, side) -> bool: is it between the fingers?
        self.gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * 0.52, 0.0]))
        self.t = 0.0
        self.breath = True
        self.status = ""

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

    def cancel(self):
        if self.task is not None:
            self.task.close()
        self.task, self.busy, self.ok = None, False, False

    def tick(self):
        """One 20 ms control step."""
        self.t += DT
        if self.task is not None:
            try:
                q = next(self.task)
            except StopIteration as e:
                self.ok = e.value is not False
                self.task, self.busy = None, False
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
        emo = EMOTES.get(name)
        if emo is None:
            return False
        t = 0.0
        while t < emo.duration:
            q = emo.target(t)
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

    def _gait_q(self, cmd, arms=None, lower=0.0):
        self.gait.arms = arms
        self.gait.lower += float(np.clip(lower - self.gait.lower, -0.0004, 0.0004))   # sink / rise smoothly
        q = None
        for _ in range(SUB):
            q = self.gait.step(DT / SUB, cmd)
        return q

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
        settle_n = 0
        for _ in range(int(timeout / DT)):
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
            wz = float(np.clip(2.0 * herr, -W_MAX, W_MAX))
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
            x, y, h = self.pose_fn()
            side = self._side_for(x, y, h, ox, oy)
            paddle = arm_pose(arm_pose(pose(crouch(0.25)), side, pitch=-0.62, yaw=0.0, claw=0.0),
                              "right" if side == "left" else "left", pitch=0.5)
            tip = np.array(self.tip_fn(side, paddle) if self.tip_fn else (0.085, 0.03 if side == "left" else -0.03))
            c, s = np.cos(heading), np.sin(heading)
            tip_w = np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
            start = np.array((ox, oy)) - u * (radius + 0.03) - tip_w
            if not self.desk.inside(start, self.desk.edge_margin - 0.004):   # can't get behind it from the desk
                if RAKE_ENABLED:
                    ok = yield from self._rake(name, goal, radius)
                    if ok is not None:
                        continue
                self.trace("unreachable", obj=(round(ox, 3), round(oy, 3)))
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
                if abs(lateral) > 0.026 or ahead < -(radius + 0.035) or ahead > 0.07 or not self.desk.inside((x, y), 0.05):
                    self.trace("slip", lateral=round(float(lateral), 3), ahead=round(float(ahead), 3),
                               obj=(round(ox, 3), round(oy, 3)))
                    break                                          # slipped off the paddle: line up again
                herr = _wrap(np.arctan2(goal[1] - oy, goal[0] - ox) - h)
                yield self._gait_q((0.042, float(np.clip(1.5 * lateral, -0.02, 0.02)),
                                    float(np.clip(1.5 * herr, -0.4, 0.4))), paddle, lower=PUSH_LOWER)
                self.pushes = getattr(self, "pushes", 0) + 1
            yield from self._settle(0.3)
        return False

    def _detour(self, frm, to, obj, clear=0.10):
        """If the straight walk from frm to `to` would bump obj, go round it first."""
        a, b, o = np.array(frm), np.array(to), np.array(obj)
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-6:
            return
        k = np.clip(np.dot(o - a, ab) / L ** 2, 0, 1)
        if np.linalg.norm(a + k * ab - o) > clear or k in (0.0, 1.0) and np.linalg.norm(o - b) > clear:
            return
        n = np.array((-ab[1], ab[0])) / L
        w1, w2 = o + n * (clear + 0.05), o - n * (clear + 0.05)
        w = max((w1, w2), key=lambda p: self.desk.edge_distance(p))
        self.trace("detour", via=tuple(np.round(w, 3)))
        yield from self._goto(self.desk.clamp(w), tol=0.04, timeout=20.0)

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
        paddle = arm_pose(arm_pose(pose(crouch(0.25)), side, pitch=-0.62, yaw=0.0, claw=0.0),
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
            side = self._side_for(x, y, h, ox, oy)
            sgn = 1 if side == "left" else -1
            reach = arm_pose(pose(crouch(REACH_LOW["crouch"], "F?"), crouch(-0.20, "R?")), side,
                             pitch=REACH_LOW["pitch"], yaw=0.12, claw=0.9)            # nose down: jaws come in low
            tip = np.array(self.tip_fn(side, reach) if self.tip_fn else (0.088, sgn * 0.035))
            arm_dir = np.arctan2(tip[1], tip[0])
            # a pen is held near one end like a person holds it: the finger runs along the pen and the
            # rest of the pen points forward, away from Pinch's own legs. Other things: face them.
            grab = np.array((ox, oy))
            if name == "pen":
                best = None
                for a in (oyaw, oyaw + np.pi):
                    ax = np.array((np.cos(a), np.sin(a)))
                    g = grab - ax * (PEN_HALF - 0.018)               # grab point; the pen extends along +ax
                    hd = _wrap(a - arm_dir)
                    c, s = np.cos(hd), np.sin(hd)
                    st = g - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
                    cost = abs(_wrap(hd - h)) + 5 * (not self.desk.inside(st, self.desk.edge_margin - 0.004))
                    if best is None or cost < best[0]:
                        best = (cost, hd, st, g)
                _, heading, stand_xy, grab = best
            else:
                heading = _wrap(np.arctan2(oy - y, ox - x) - arm_dir)
                c, s = np.cos(heading), np.sin(heading)
                stand_xy = grab - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
            self.trace("pick_plan", side=side, tip=tuple(np.round(tip, 3)), stand=tuple(np.round(stand_xy, 3)),
                       heading=round(float(heading), 2), obj=(round(ox, 3), round(oy, 3), round(oyaw, 2)))
            if not self.desk.inside(stand_xy, self.desk.edge_margin - 0.004):
                self.trace("pick_unreachable")
                return False
            yield from self._detour((x, y), stand_xy, (ox, oy), clear=0.08)
            ok = yield from self._goto(stand_xy, tol=0.010, heading=heading, timeout=30.0, slow=0.6, htol=0.06)
            xx, yy, hh = self.pose_fn()
            self.trace("pick_at", ok=ok, err=round(float(np.hypot(xx - stand_xy[0], yy - stand_xy[1])), 4),
                       herr=round(float(_wrap(heading - hh)), 3))
            if not ok:
                continue
            yaw = yield from self._align(name, side, grab, reach)
            if abs(self.last_miss) > 0.003:
                continue                                         # not lined up: walk round and try again
            reach = arm_pose(reach, side, yaw=yaw)
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

    def _lateral(self, name, side, grab, q):
        """Sideways miss (m) between the open claw's gap and the grab point, across the pen's axis."""
        x, y, h = self.pose_fn()
        tip = np.array(self.tip_fn(side, q))
        c, s = np.cos(h), np.sin(h)
        gap = np.array((x + c * tip[0] - s * tip[1], y + s * tip[0] + c * tip[1]))
        (ox, oy, oyaw), _ = self._obj(name)
        if name == "pen":
            n = np.array((-np.sin(oyaw), np.cos(oyaw)))
            ax = np.array((np.cos(oyaw), np.sin(oyaw)))
            g = np.array(grab) if grab is not None else np.array((ox, oy))
            g = np.array((ox, oy)) + ax * np.dot(g - (ox, oy), ax)          # the grab point, kept on the pen
            return float(np.dot(gap - g, n)), n
        d = gap - (ox, oy)
        n = np.array((-s, c))
        return float(np.dot(d, n)), n

    def _align(self, name, side, grab, reach, tol=0.0025):
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

    def _drop(self, goal):
        """Carry the held thing to the goal and put it down there."""
        if not self.held:
            return False
        name, side = self.held
        goal = np.array(goal, float)
        x, y, h = self.pose_fn()
        reach = arm_pose(pose(crouch(REACH_LOW["crouch"])), side, pitch=REACH_LOW["pitch"] + 0.25, yaw=0.12, claw=0.0)
        tip = np.array(self.tip_fn(side, reach) if self.tip_fn else (0.085, (1 if side == "left" else -1) * 0.035))
        if name == "pen" and np.allclose(goal, self.desk.tray_xy):
            # lay it lengthwise in the tray: claw at the tray's left end, the pen pointing right (+x)
            goal = np.array((goal[0] - self.desk.tray_half[0] + 0.03, goal[1]))
            heading = float(-np.arctan2(tip[1], tip[0]))
        else:
            heading = float(np.arctan2(goal[1] - y, goal[0] - x) - np.arctan2(tip[1], tip[0]))
        c, s = np.cos(heading), np.sin(heading)
        stand_xy = goal - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
        ok = yield from self._goto(self.desk.clamp(stand_xy, 0.06), tol=0.02, heading=heading, timeout=40.0, slow=0.7)
        yield from self._blend_to(reach, 0.8)
        yield from self._blend_to(arm_pose(reach, side, claw=0.9), 0.4)
        self.held = None
        yield from self._blend_to(arm_pose(STAND, side, pitch=0.3, claw=0.3), 0.6)
        yield from self._blend_to(STAND, 0.5)
        self.hold = STAND.copy()
        o = self._obj(name)
        if o is None:
            return ok
        (ox, oy, _), _ = o
        self.trace("dropped", obj=(round(ox, 3), round(oy, 3)), in_tray=self.desk.in_tray((ox, oy)))
        return bool(np.hypot(ox - goal[0], oy - goal[1]) < 0.06 or self.desk.in_tray((ox, oy)))
