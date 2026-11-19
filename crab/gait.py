"""Scripted walking: a velocity-commanded gait generator + leg IK.

    gait = Gait(GAITS["trot"])
    q = gait.step(dt, cmd=(vx, vy, wz))   # -> 20 joint targets (legs + arms)

cmd is in the body frame: vx forward (m/s), vy left (m/s), wz yaw rate (rad/s).
Crabs are best at vy (sideways).

How it works
- A phase clock runs at 1/period. Each leg has a phase offset; while its phase
  is below `duty` the foot is on the ground (stance), otherwise it swings.
- Stance: the foot slides backwards in the body frame at the body's velocity at
  that foot, which pushes the body along.
- Swing: the foot lifts (step_height) and moves to where it should land so the
  next stance is centred on its neutral position (Raibert-style placement).
- Weight shift (crawl): before a leg lifts, the body leans over the triangle of
  the other three feet so the robot stays statically stable.

This is the baseline controller and the source of reference motions for RL.
"""

from dataclasses import dataclass, field

import numpy as np

from .emotes import STAND
from .kinematics import leg_fk, leg_ik
from .params import JOINT_NAMES, LEG_IDS

LEG_IDX = [[JOINT_NAMES.index(f"{lg}_{j}") for j in ("coxa", "femur", "tibia")] for lg in LEG_IDS]


@dataclass(frozen=True)
class GaitParams:
    name: str
    period: float                     # s per full cycle
    duty: float                       # fraction of cycle each foot is on the ground
    offsets: dict                     # leg -> phase offset in [0, 1)
    step_height: float = 0.025        # m
    weight_shift: float = 0.0         # 0 = off, 1 = put CoM over the support centre
    shift_mode: str = "static"        # "static": CoM over the middle of all feet (trot)
                                      # "support": lean over the feet that carry load (crawl)
    shift_lead: float = 0.10          # phase fraction over which a leg sheds / picks up load
    max_step: float = 0.045           # m, cap on half step length
    stance_inset: float = 0.015       # m, feet pulled toward the body vs STAND (keeps reach margin)
    cmd_accel: tuple = (0.25, 0.25, 1.5)   # max change of (vx, vy, wz) per second
    blend_time: float = 0.5           # s to blend from STAND into the walking stance


from .params import PARAMS as _P  # noqa: E402

_S = _P.scale                    # lengths scale with the robot
_T = _S ** 0.5                   # periods scale with sqrt(size) (Froude): small crabs step faster


def _scaled(g: "GaitParams") -> "GaitParams":
    from dataclasses import replace
    return replace(g, period=g.period * _T, step_height=g.step_height * _S, max_step=g.max_step * _S,
                   stance_inset=g.stance_inset * _S, cmd_accel=(g.cmd_accel[0] * _S, g.cmd_accel[1] * _S,
                                                                g.cmd_accel[2]))


_GAITS_100MM = {
    # One leg at a time in the lateral sequence RL -> FL -> RR -> FR, always 3 feet
    # down. duty 0.85 leaves a 4-feet window (0.10 of the cycle) before each
    # lift-off; the body shifts its weight during exactly that window.
    # Slow "sneak" gait. Re-tuned for the compact all-STS3032 crab (0.76 kg, 30 mm thigh, shins near
    # vertical) by a sweep over period, body lean, stance width and stride: tracks >= 74 % of the
    # command in every direction with the warmest servo at 64 % of rated.
    "crawl": GaitParams("crawl", period=3.6, duty=0.85,
                        offsets={"RL": 0.0, "FL": 0.75, "RR": 0.5, "FR": 0.25},
                        step_height=0.018, weight_shift=0.45, shift_mode="support",
                        shift_lead=0.10, max_step=0.030, stance_inset=0.0075),
    # Diagonal pairs: faster, dynamically balanced. weight_shift keeps the CoM
    # (which sits ahead of the body centre) over the middle of the feet.
    # period/step height chosen with the measured motor model so that even the
    # pessimistic datasheet servo has torque margin except at full sideways speed.
    "trot": GaitParams("trot", period=0.9, duty=0.55,
                       offsets={"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5},
                       step_height=0.014, weight_shift=1.0, shift_mode="static"),
}
# The values above were tuned on the original 100 mm-shin robot; scale them to this one.
GAITS = {k: _scaled(v) for k, v in _GAITS_100MM.items()}
# "climb": the crawl with feet lifted 20 mm (absolute, not scaled) - for getting over things it's caught on.
# In the twin, walking forward, it crosses a fixed 5 mm cable, a pen, and 8 and 12 mm steps; the trot
# crosses none of them forward (sideways the trot manages all but the 12 mm step). Decision 52.
GAITS["climb"] = __import__("dataclasses").replace(GAITS["crawl"], name="climb", step_height=0.020)


def _cycloid(s):
    """0->1 with zero velocity at both ends."""
    return s - np.sin(2 * np.pi * s) / (2 * np.pi)


@dataclass
class Gait:
    params: GaitParams
    com_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # CoM xy in body frame
    arms: np.ndarray = None           # 18-vector whose arm entries are used; default STAND
    lower: float = 0.0                # m: walk crouched this much lower (claws reach the desk)

    def __post_init__(self):
        from .kinematics import _mirror, _mount
        self.neutral = {}
        for lg, idx in zip(LEG_IDS, LEG_IDX):
            p = leg_fk(lg, STAND[idx])
            mount = _mirror(_mount(lg)[0], lg)
            radial = (p - mount)[:2]
            p[:2] -= self.params.stance_inset * radial / np.linalg.norm(radial)
            self.neutral[lg] = p
        self.reset()

    def reset(self):
        self.t = 0.0
        self.phase = 0.0
        self.cmd = np.zeros(3)
        self.offset = {lg: np.zeros(2) for lg in LEG_IDS}      # foot xy offset from neutral
        self.lift_start = {lg: np.zeros(2) for lg in LEG_IDS}
        self.swinging = {lg: False for lg in LEG_IDS}
        self.shift = np.zeros(2)
        self.unreachable = 0

    # -- helpers -----------------------------------------------------------
    def _leg_phase(self, lg):
        return (self.phase + self.params.offsets[lg]) % 1.0

    def _foot_vel(self, lg):
        vx, vy, wz = self.cmd
        px, py = self.neutral[lg][:2] - self.shift    # foot relative to body origin
        return np.array([vx - wz * py, vy + wz * px])

    def _support_weight(self, lg):
        """1 = carrying load, 0 = in the air. A leg picks its load up gradually
        right after touch-down and sheds it gradually just before lift-off, so
        the support centre (and the body) moves smoothly."""
        P, ph = self.params, self._leg_phase(lg)
        if ph >= P.duty:
            return 0.0
        lead = P.shift_lead
        smooth = lambda s: s * s * (3 - 2 * s)  # noqa: E731
        if ph < lead:                                # just landed
            return smooth(ph / lead)
        if ph > P.duty - lead:                       # about to lift
            return 1.0 - smooth((ph - (P.duty - lead)) / lead)
        return 1.0

    def feasible_scale(self, cmd) -> float:
        """Factor (<= 1) that shrinks cmd so every stride fits within max_step."""
        P = self.params
        vx, vy, wz = cmd
        need = max(np.hypot(vx - wz * p[1], vy + wz * p[0]) for p in self.neutral.values())
        need *= P.duty * P.period / 2
        return 1.0 if need <= P.max_step else P.max_step / need

    # -- main --------------------------------------------------------------
    def step(self, dt, cmd=(0.0, 0.0, 0.0), solve=True) -> np.ndarray:
        """Advance the gait by dt. solve=False skips the leg IK and returns None: a caller that integrates
        in several small substeps per control tick only needs the joint angles after the last one."""
        P = self.params
        # cap to what the legs can reach, then rate-limit so starts/stops are smooth
        target = np.asarray(cmd, float)
        target = target * self.feasible_scale(target)
        lim = np.asarray(P.cmd_accel) * dt
        self.cmd += np.clip(target - self.cmd, -lim, lim)

        idle = np.linalg.norm(self.cmd) < 1e-3 and np.linalg.norm(target) < 1e-3 and \
            all(np.linalg.norm(o) < 2e-3 for o in self.offset.values()) and \
            not any(self.swinging.values())
        if not idle:
            self.phase = (self.phase + dt / P.period) % 1.0

        # Speed governor: a stance foot must not run out of travel before it
        # lifts. If one would (e.g. at start-up, when feet begin at neutral, or
        # after a direction change), slow the whole body down -- all stance feet
        # together, so nothing slips.
        govern = 1.0
        budget_max = 1.25 * P.max_step
        for lg in LEG_IDS:
            ph = self._leg_phase(lg)
            v = self._foot_vel(lg)
            speed = np.linalg.norm(v)
            if idle or ph >= P.duty or speed < 1e-6:
                continue
            travelled = self.offset[lg] @ (-v / speed)            # already moved along travel dir
            budget = max(budget_max - travelled, 0.0)
            needed = speed * (P.duty - ph) * P.period
            if needed > budget:
                govern = min(govern, budget / needed)
        self.govern = govern

        feet = {}
        for lg in LEG_IDS:
            ph = self._leg_phase(lg)
            v = self._foot_vel(lg)
            if idle:
                z = 0.0
            elif ph < P.duty:                                    # stance
                self.swinging[lg] = False
                self.offset[lg] = self.offset[lg] - govern * v * dt
                z = 0.0
            else:                                                # swing
                s = (ph - P.duty) / (1.0 - P.duty)
                if not self.swinging[lg]:
                    self.swinging[lg] = True
                    self.lift_start[lg] = self.offset[lg].copy()
                land = v * P.duty * P.period / 2
                n = np.linalg.norm(land)
                if n > P.max_step:
                    land *= P.max_step / n
                self.offset[lg] = self.lift_start[lg] + (land - self.lift_start[lg]) * _cycloid(s)
                z = P.step_height * 0.5 * (1 - np.cos(2 * np.pi * s))
            n = np.linalg.norm(self.offset[lg])
            if n > 1.4 * P.max_step:                              # backstop; governor should prevent this
                self.offset[lg] *= 1.4 * P.max_step / n
            feet[lg] = self.neutral[lg] + np.array([*self.offset[lg], z + self.lower])

        # weight shift: move the body so the CoM sits over the loaded feet
        if P.weight_shift > 0:
            if P.shift_mode == "support":
                w = np.array([self._support_weight(lg) for lg in LEG_IDS])
                pts = np.array([feet[lg][:2] for lg in LEG_IDS])
            else:
                w = np.ones(len(LEG_IDS))
                pts = np.array([self.neutral[lg][:2] for lg in LEG_IDS])
            centroid = (w[:, None] * pts).sum(0) / max(w.sum(), 1e-6)
            desired = P.weight_shift * (centroid - self.com_offset)
            n = np.linalg.norm(desired)
            if n > 0.06:
                desired *= 0.06 / n
            # low-pass so the body glides rather than jerks
            a = min(1.0, dt / 0.06)
            self.shift += a * (desired - self.shift)
        else:
            self.shift[:] = 0

        self.t += dt
        if not solve:
            return None
        q = (STAND if self.arms is None else self.arms).copy()
        for lg, idx in zip(LEG_IDS, LEG_IDX):
            p = feet[lg].copy()
            p[:2] -= self.shift
            q_leg, ok = leg_ik(lg, p)
            self.unreachable += not ok
            q[idx] = q_leg
        # ease in from the STAND pose so switching from emotes to walking is smooth
        if self.t < P.blend_time:
            s = self.t / P.blend_time
            s = s * s * (3 - 2 * s)
            q = STAND + (q - STAND) * s
        return q
