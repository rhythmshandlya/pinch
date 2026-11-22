"""Getting about the desk: the walking gait (+ the trained corrections), the foot-aware edge guard, going to a spot, detours, and getting free (recover, climb over, roll back over)."""

from pinch.body.common import (  # noqa: F401
    CARRY_PITCH,
    DT,
    GAIT_LOOK,
    GAITS,
    JOINT_NAMES,
    SERVO_IDS,
    STAND,
    SUB,
    V_MAX,
    W_MAX,
    Gait,
    _wrap,
    arm_pose,
    copy,
    np,
    pio,
)
from pinch.settings import load as _load_settings  # noqa: E402

MOTION = _load_settings("expression")["motion"]


class Locomotion:
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
        return self._residual(q, cmd, self._walking_policy(arms))

    def _walking_policy(self, arms):
        """Which learned corrections run this tick. Plain walking: the walking policy. Crouched pushing: the
        push policy if there is one, else the walking policy if it was trained on pushes too (decisions 65,
        84). Anything else (carrying, arms busy): none, the pure scripted gait the policies never saw."""
        if self.gait.lower > 1e-4:
            return next((p for p in (self.push_policy, self.walk_policy)
                         if p is not None and "push" in getattr(p, "meta", {}).get("covers", ["walk"])), None)
        return self.walk_policy if arms is None else None

    def _residual(self, ref, cmd, pol):
        """Learned walking corrections (pol) on top of the scripted trot. Its look-ahead (the reference 0.1 s on,
        as in training) comes from a copy of the gait run on with the same command - no lag for the servos (a
        0.1 s delay line cost the tidy planners most of their precision)."""
        if pol is None:
            self._gseen, self._gpol = -1.0, None
            return ref
        if self.t - self._gseen > 2.5 * DT or pol is not self._gpol:   # (re)starting a walk, or switching policy
            self._glast, self._gt = np.zeros(12), 0.0
        self._gseen, self._gpol = self.t, pol
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
        self._glast = np.clip(pol(obs), -1, 1)
        self._gt += DT
        scale = getattr(pol, "meta", {}).get("action_scale", pio.EMOTE_ACTION_SCALE)
        return pio.emote_targets(np, ref, self._glast * (scale / pio.EMOTE_ACTION_SCALE))

    def _walk(self, vx, vy, wz, seconds):
        for _ in range(int(seconds / DT)):
            yield self._gait_q((vx, vy, wz), self._arms())
        yield from self._settle()
        return True

    # -- moving around the desk ------------------------------------------------------------
    def _goto(self, xy, tol=0.03, heading=None, timeout=40.0, slow=1.0, htol=0.15):
        xy = np.array(self.desk.clamp(xy))
        h_end = heading if heading is not None else self.pose_fn()[2]
        fitted = np.array(self.desk.fit_feet(xy, h_end, self.FEET_EDGE + 0.005))
        if np.linalg.norm(fitted - xy) > 1e-4:                # (the feet wouldn't fit there: stop short)
            self.trace("goto_fitted", asked=tuple(np.round(xy, 3)), to=tuple(np.round(fitted, 3)))
            xy = fitted
        settle_n = 0
        tempo = self.walk_tempo()
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
            vmax = V_MAX * slow * tempo
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

    def walk_tempo(self):
        """How fast it walks follows how it feels (Laban effort; Cozmo/aibo mood-styled motion,
        docs/research/v2_1/expression): brisk when lively or scared, slow when sad, sleepy or tired.
        configs/mind/expression.toml [motion]; 1.0 = the gait's top speed."""
        if self.mood_fn is None:
            return 1.0
        m, c = self.mood_fn(), MOTION
        t = (c["base"] + c["arousal"] * (m.get("arousal", 0.4) - 0.4) + c["valence"] * m.get("valence", 0.0)
             - c["fatigue"] * m.get("fatigue", 0.0) - c["sleepy"] * m.get("sleepy", 0.0) + c["fear"] * m.get("fear", 0.0))
        return float(min(c["max"], max(c["min"], t)))

    def _turn_to(self, xy, tol=0.15):
        x, y, h = self.pose_fn()
        heading = np.arctan2(xy[1] - y, xy[0] - x)
        return (yield from self._goto((x, y), tol=0.5, heading=heading, timeout=12.0))

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

    walk_dir = None

    def _right_self(self, timeout=8.0):
        """Roll back over off its back: the learned get-up policy if exported (policies/righting.npz, decision
        80), else the searched swing (crab/righting.py). Done once the body has been upright for 0.2 s (then
        stand). False if it is still on its back after `timeout`."""
        from crab import righting
        self.held = None                                     # (whatever it held is gone)
        t, up, last = 0.0, 0, np.zeros(len(JOINT_NAMES))
        pol = self.righting_policy
        while t < timeout:
            if pol is not None:
                gyro, grav = self.crab.hw.read_imu()
                qj, st = self.crab.read_joints()             # (the arm servos report nothing: last command)
                qd = np.array([self.crab.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
                last = np.clip(pol(pio.righting_observation(np, grav, gyro, qj, qd, last)), -1, 1)
                yield pio.righting_targets(np, last, self.crab.lo, self.crab.hi)
            else:
                yield righting.target(t)
            t += DT
            try:
                _, grav = self.crab.hw.read_imu()
            except Exception:                                # noqa: BLE001 - a sensor hiccup: keep swinging
                continue
            up = up + 1 if grav[2] < -0.85 else 0
            if up >= int(0.2 / DT):
                self.trace("righted", after=round(t, 2))
                yield from self._blend_to(STAND, 0.6)
                yield from self._settle(0.4)
                self.gait.cmd[:] = 0.0
                return True
        self.trace("righting_failed")
        yield from self._blend_to(STAND, 0.6)
        return False

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
