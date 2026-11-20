"""The claws at work: pushing things (paddle), picking up, carrying, putting down."""

from pinch.body.common import (  # noqa: F401
    CARRY_PITCH,
    DT,
    GRASP_ARM0,
    JOINT_NAMES,
    LONG_THINGS,
    PADDLE_CLAW,
    PADDLE_PITCH,
    PEN_HALF,
    PUSH_LOWER,
    REACH_LOW,
    SERVO_IDS,
    STAND,
    _wrap,
    arm_pose,
    copy,
    crouch,
    extent,
    np,
    paddle_pose,
    pio,
    pose,
)


class Manipulation:
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
            near_side = self._side_for(x, y, h, ox, oy)
            # the near arm's paddle; the other arm only if the near one can't line up from the desk (decision 89:
            # choosing the far arm to spare bystanders cost more pushes than it saved)
            fits, start, side, paddle, tip, _ = self._push_lineup(name, ox, oy, u, heading, radius, near_side)
            if not fits:
                far = self._push_lineup(name, ox, oy, u, heading, radius, "right" if near_side == "left" else "left")
                if far[0]:
                    fits, start, side, paddle, tip, _ = far
            # can't get behind it from the desk (its feet, not just its centre, must fit there)
            if not fits:
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

    def _push_lineup(self, name, ox, oy, u, heading, radius, side):
        """Where to stand to push with this arm's paddle: 3 cm behind the thing, or closer near an edge.
        Returns (fits, start, side, paddle pose, paddle tip in the body frame, bystander cost)."""
        paddle = paddle_pose(side)
        tip = np.array(self.tip_fn(side, paddle) if self.tip_fn else (0.085, 0.03 if side == "left" else -0.03))
        c, s = np.cos(heading), np.sin(heading)
        tip_w = np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
        for gap in (0.03, 0.02, 0.01, 0.0):
            start = np.array((ox, oy)) - u * (radius + gap) - tip_w
            fits = (self.desk.inside(start, self.desk.edge_margin) and           # (as _goto will clamp
                    self.desk.feet_clear(start, heading) >= self.FEET_EDGE + 0.005)  # and fit it)
            if fits:
                break
        return fits, start, side, paddle, tip, self._bystander_cost(start, heading, exclude=(name,), arm_tip=tip)

    def _side_for(self, x, y, h, ox, oy):
        _ex, ey = np.cos(h) * (ox - x) + np.sin(h) * (oy - y), -np.sin(h) * (ox - x) + np.cos(h) * (oy - y)
        return "left" if ey > 0 else "right"

    def _pick(self, name, timeout=60.0):
        """Line a claw up with the thing (a pen lies along the finger), reach down, close, lift."""
        t_end = self.t + timeout
        for _attempt in range(3):
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
                        # cheapest: feet fit (some room to spare) and clear of the other things, little walking
                        # and turning, the nearer claw, the usual arm swing
                        cost = ((100.0 if room < 0 else 0.0) + 3 * max(0.0, 0.03 - room) / 0.03
                                + self._bystander_cost(st, hd, exclude=(name,), arm_tip=tp)
                                + np.hypot(st[0] - x, st[1] - y) / 0.10 + 0.3 * abs(_wrap(hd - h)) + 0.3 * si
                                + abs(yw - 0.12))
                        if best is None or cost < best[0]:
                            best = (cost, sd, rq, tp, hd, st, g)
            _, side, reach, tip, heading, stand_xy, grab = best
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

    BYSTANDER_CLEAR = 0.035   # m from any foot, or the working arm, to another thing's centre (86, 89)
    SHOULDER = 0.03           # m: the arms' roots, forward of the body centre and out to each side

    def _bystander_cost(self, xy, heading, exclude=(), arm_tip=None):
        """0 when every foot - and the working arm, shoulder to claw tip (arm_tip, body frame), if given - stays
        BYSTANDER_CLEAR from the other things on the desk, rising to 4 as one comes onto a thing: legs and
        arms bump what they work next to (the toy ball rolls off the desk)."""
        if self.bystanders_fn is None:
            return 0.0
        things = [np.asarray(p, float) for n, p in self.bystanders_fn() if n not in exclude]
        if not things:
            return 0.0
        gap = min(float(np.hypot(*(np.asarray(f) - p))) for f in self.desk.feet(xy, heading) for p in things)
        if arm_tip is not None:
            c, s = np.cos(heading), np.sin(heading)
            rot = np.array(((c, -s), (s, c)))
            root = np.array(xy) + rot @ np.array((self.SHOULDER, np.sign(arm_tip[1]) * self.SHOULDER))
            tip = np.array(xy) + rot @ np.asarray(arm_tip[:2], float)
            d = tip - root
            for p in things:
                k = np.clip(np.dot(p - root, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0)
                gap = min(gap, float(np.hypot(*(root + k * d - p))))
        return 4.0 * max(0.0, self.BYSTANDER_CLEAR - gap) / self.BYSTANDER_CLEAR

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
            # (and, among those, the nearest one that keeps its feet off the other things on the desk)
            fits = []
            for dh in sorted(np.linspace(-np.pi, np.pi, 25), key=abs):
                hd = direct + dh
                c, s = np.cos(hd), np.sin(hd)
                st = goal - np.array((c * tip[0] - s * tip[1], s * tip[0] + c * tip[1]))
                if self.desk.feet_clear(st, hd) >= self.FEET_EDGE + 0.005:
                    fits.append((self._bystander_cost(st, hd, exclude=(name,), arm_tip=tip) > 0, abs(dh), hd))
            best = min(fits)[2] if fits else None
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
