"""The body's frame: starting, cancelling and ticking steps (generators that yield joint targets), reflexes, tracing, and the basic motions every skill uses (blend, hold a pose, settle).

Body parts have owners (pinch/body/arbiter.py, configs/mind/channels.toml): the running step claims the parts it
moves at its layer (a reflex above everything); ambient life (pinch/body/life.py) moves every part no one owns and
the parts the step lets it (the claws flex while it walks), never the legs under a walk or a claw carrying something."""

from pinch.body.arbiter import ambient_parts, arbitrate, claim_for
from pinch.body.common import (  # noqa: F401
    CARRY_PITCH,
    DT,
    GAITS,
    POSES,
    STAND,
    Desk,
    Gait,
    J,
    arm_pose,
    copy,
    np,
)
from pinch.body.life import PO as _POSTURE


def _part_joints():
    out = {"legs": [J[f"{lg}_{k}"] for lg in ("FL", "FR", "RL", "RR") for k in ("coxa", "femur", "tibia")]}
    for s in "LR":
        out[f"claw_{s}"] = [J[f"{s}_shoulder_yaw"], J[f"{s}_shoulder_pitch"], J[f"{s}_claw"]]
    return out


PART_JOINTS = _part_joints()                     # body part -> joint indices (gaze, face, sound have none)


class BodyCore:
    def __init__(self, crab, pose_fn, objects_fn=None, desk=None, tip_fn=None, emote_policy=None, walk_policy=None,
                 grasp_policy=None, righting_policy=None, push_policy=None, life_seed=0):
        self.crab, self.pose_fn, self.objects_fn = crab, pose_fn, objects_fn
        self.desk = desk or Desk()
        self.tip_fn = tip_fn                    # (side, arm joint q) -> claw tip (x, y) in the body frame
        self.emote_policy, self.walk_policy = emote_policy, walk_policy
        self._glast, self._gt, self._gseen = np.zeros(12), 0.0, -1.0
        self.grasp_policy = grasp_policy        # the learned grab (right claw; training/grasp_env.py), if exported
        self.righting_policy = righting_policy  # the learned get-up (training/righting_env.py), if exported
        self.push_policy = push_policy          # walking corrections while crouched pushing, if a separate one (84)
        self._gpol = None                       # the walking policy that ran last tick
        self.bystanders_fn = None               # () -> [(name, xy)]: the things believed on the desk (decision 86)
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
        self.claim = None                       # the running step's claim on the body parts (arbiter.Claim)
        self.ambient_amp = 0.0
        self._src, self._seam, self._seam_k, self._seam_n = None, None, 0, 1     # handover blend (tick)

    def trace(self, what, **kw):
        self.log.append((round(self.t, 2), what, kw))
        self.log = self.log[-200:]

    log = []

    # -- interface for the mind --------------------------------------------------------
    def start(self, step, owner=None, layer="activity"):
        """Run a step. owner and layer say who asked and how strongly (arbiter layers: reflex, command, reaction,
        activity); the step claims the body parts it moves."""
        kind, args = step[0], step[1:]
        make = getattr(self, f"_{kind}", None)
        if make is None:
            self.busy, self.ok = False, False
            return
        self.step_goal = None
        self.task = make(*args)
        self.busy, self.ok, self.status = True, True, kind
        self.claim = claim_for(kind, owner or kind, layer, self.t)
        self._reflexed = False

    def reflex(self, step):
        """A safety reflex (backing off an edge) takes over from whatever was running. When it ends, the step
        it cut short counts as failed - the mind must not take "backed off the edge" for "picked up the pen"."""
        interrupted = self.busy
        self.cancel()
        self.start(step, owner="reflex", layer="reflex")
        self._reflexed = interrupted

    _reflexed = False

    def cancel(self):
        if self.task is not None:
            self.task.close()
        self.task, self.busy, self.ok, self.claim = None, False, False, None

    def _use_claw(self, side):
        """A step that turns out to need one claw only (a pick, a drop) gives the other one back: ambient life moves it."""
        if self.claim is None:
            return
        from dataclasses import replace
        free = "claw_R" if side == "left" else "claw_L"
        self.claim = replace(self.claim, parts=self.claim.parts - {free}, ambient_ok=self.claim.ambient_ok | {free})

    def owners(self):
        """{body part: arbiter.Claim or None} right now."""
        return arbitrate([self.claim] if self.claim is not None else [])[0]

    def _ambient_mask(self):
        """1 on the joints ambient life may move now, 0 elsewhere."""
        parts = ambient_parts(self.owners())
        if self.held:
            parts.discard("claw_L" if self.held[1] == "left" else "claw_R")
        mask = np.zeros(len(STAND))
        for p in parts:
            mask[PART_JOINTS.get(p, [])] = 1.0
        return mask

    def tick(self):
        """One 20 ms control step."""
        self.t += DT
        self.cmd_vel = (0.0, 0.0, 0.0)
        if self.task is not None:
            try:
                q = next(self.task)
            except StopIteration as e:
                self.ok = e.value is not False and not self._reflexed
                self.task, self.busy, self._reflexed, self.claim = None, False, False, None
                q = self.hold
        else:
            q = self.hold
        self.ambient_amp = 0.0                             # the largest ambient offset this tick (rad): aliveness
        if self.breath:                                    # alive always, on the parts no one is using (life.py)
            mask = self._ambient_mask()
            if mask.any():
                off = self.life.offsets(self.t, DT) * mask
                self.ambient_amp = float(np.abs(off).max())
                q = q + off
            # posture from the mood on the free parts; when a step takes a part it fades out over fade_s, not a jump
            target = self.life.posture(DT) * mask
            post = getattr(self, "_posture", None)
            post = np.zeros(len(q)) if post is None else post
            if self.claim is not None and self.claim.layer == "reflex":
                post = post * mask                           # a reflex (righting, the edge) gets its joints clean now
            self._posture = post + (target - post) * (1 - np.exp(-DT / _POSTURE["fade_s"]))
            q = q + self._posture
        if self.held:
            obj, side = self.held
            q = arm_pose(q, side, pitch=max(q[J[f"{'L' if side == 'left' else 'R'}_shoulder_pitch"]], CARRY_PITCH)
                         if self.task is None else None, claw=0.0)
        q = self._handover(q)
        self.crab.tick(q)
        return q

    HANDOVER_S, REFLEX_HANDOVER_S = 0.24, 0.08
    SNAP_RAD = 0.12                     # a handover gap above this is faded out (the servos follow ~0.16 rad a tick)

    def _handover(self, q):
        """When what drives the joints changes (a step starts, ends or is cut short), fade out the gap between the
        last target and the new one instead of jumping (a cancelled pick used to snap to the standing pose in one
        tick: 0.46 rad, more than a servo follows). A safety reflex fades faster."""
        src = id(self.task) if self.task is not None else None
        if src != self._src:
            self._src = src
            last = getattr(self.crab, "q_cmd", None)
            if last is not None and len(last) == len(q):
                gap = np.asarray(last, float) - q
                if np.abs(gap).max() > self.SNAP_RAD:      # only a real snap: a step handing over smoothly is left alone
                    reflex = self.claim is not None and self.claim.layer == "reflex"
                    self._seam = gap
                    self._seam_k = 0
                    self._seam_n = max(1, int((self.REFLEX_HANDOVER_S if reflex else self.HANDOVER_S) / DT))
        if self._seam is not None:
            self._seam_k += 1
            w = 1.0 - self._seam_k / self._seam_n
            if w <= 0:
                self._seam = None
            else:
                q = q + self._seam * (w * w * (3 - 2 * w))
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
