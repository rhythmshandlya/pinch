"""The body's frame: starting, cancelling and ticking steps (generators that yield joint targets), reflexes, tracing, and the basic motions every skill uses (blend, hold a pose, settle)."""

from pinch.body.common import (  # noqa: F401
    CARRY_PITCH,
    DT,
    Desk,
    GAITS,
    Gait,
    J,
    POSES,
    STAND,
    arm_pose,
    copy,
    np,
)


class BodyCore:
    def __init__(self, crab, pose_fn, objects_fn=None, desk=None, tip_fn=None, emote_policy=None, walk_policy=None,
                 grasp_policy=None, life_seed=0):
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
        from pinch.body.life import IdleLife
        self.life = IdleLife(np.random.default_rng(life_seed))   # the idle life layer (seeded: replay)
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
        if self.task is None and self.breath:              # alive even when still (pinch/body/life.py)
            q = q + self.life.offsets(self.t, DT)
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

    def _pose(self, name, seconds=2.0):
        q = POSES.get(name, STAND)
        yield from self._blend_to(q, 0.6)
        for _ in range(int(seconds / DT)):
            yield q
        self.hold = q.copy()
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

    # -- manipulation ------------------------------------------------------------------------
    def _obj(self, name):
        return self.objects_fn(name) if self.objects_fn else None
