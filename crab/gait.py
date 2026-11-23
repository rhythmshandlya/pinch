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

import math
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
    side_cap: float = 1.0             # sideways stride cap relative to max_step (v2 trot: sideways walking heats the knees)


from .params import PARAMS as _P  # noqa: E402

_S = _P.scale                    # lengths scale with the robot
_T = _S ** 0.5 * _P.gait_period_factor     # periods scale with sqrt(size) (Froude), x1.5 for v2 (the STS3215 rotor)


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
# style B: sideways walking at the forward cap runs the thigh and knee past the owner's torque limits (R-03: 80 % of
# stall peak, 70 % of rated RMS); a lower sideways stride cap keeps both (docs/dev/V2_MODEL.md has the numbers).
# v2.1 stance (thighs level, hips 12.5 deg forward): 0.6 instead of 0.5. Sideways is the crab's cheap direction
# (Chen et al., Bioinspir. Biomim. 17 046001, 2022: 40 % lower cost on a 3-joint-leg robot; here 17 % less energy
# per metre than forward at 8 cm/s), so it gets the most the limits allow: 6.1 cm/s instead of 5.4 with the
# hottest joint at 67 % (first-order) / 69 % (BAM) of rated RMS walking sideways nonstop; 0.65 reaches 72 % (BAM)
GAITS["trot"] = __import__("dataclasses").replace(GAITS["trot"], side_cap=float(__import__("os").environ.get("CRAB_SIDE_CAP", "0.6")))
# "climb": the crawl with feet lifted 20 mm (absolute, not scaled) - for getting over things it's caught on.
# In the twin, walking forward, it crosses a fixed 5 mm cable, a pen, and 8 and 12 mm steps; the trot
# crosses none of them forward (sideways the trot manages all but the 12 mm step). Decision 52.
GAITS["climb"] = __import__("dataclasses").replace(GAITS["crawl"], name="climb", step_height=0.020)


def _cycloid(s):
    """0->1 with zero velocity at both ends."""
    return s - math.sin(2 * math.pi * s) / (2 * math.pi)


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
        self._neutral_xy = [(float(self.neutral[lg][0]), float(self.neutral[lg][1])) for lg in LEG_IDS]
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
        n, sh = self.neutral[lg], self.shift
        px, py = n[0] - sh[0], n[1] - sh[1]           # foot relative to body origin
        return vx - wz * py, vy + wz * px

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
        need = max(math.hypot(vx - wz * p[1], (vy + wz * p[0]) / P.side_cap) for p in self._neutral_xy)
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

        c0, c1, c2 = self.cmd
        t0, t1, t2 = target
        idle = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2) < 1e-3 and math.sqrt(t0 * t0 + t1 * t1 + t2 * t2) < 1e-3 and \
            all(math.hypot(o[0], o[1]) < 2e-3 for o in self.offset.values()) and \
            not any(self.swinging.values())
        if not idle:
            self.phase = (self.phase + dt / P.period) % 1.0

        # Speed governor: a stance foot must not run out of travel before it
        # lifts. If one would (e.g. at start-up, when feet begin at neutral, or
        # after a direction change), slow the whole body down -- all stance feet
        # together, so nothing slips.
        govern = 1.0
        budget_max = 1.25 * P.max_step
        vel = {lg: self._foot_vel(lg) for lg in LEG_IDS}
        if not idle:
            for lg in LEG_IDS:
                ph = self._leg_phase(lg)
                vx, vy = vel[lg]
                speed = math.hypot(vx, vy)
                if ph >= P.duty or speed < 1e-6:
                    continue
                o = self.offset[lg]
                travelled = -(o[0] * vx + o[1] * vy) / speed           # already moved along travel dir
                budget = max(budget_max - travelled, 0.0)
                needed = speed * (P.duty - ph) * P.period
                if needed > budget:
                    govern = min(govern, budget / needed)
        self.govern = govern

        feet = {}
        lim_off = 1.4 * P.max_step
        for lg in LEG_IDS:
            ph = self._leg_phase(lg)
            vx, vy = vel[lg]
            o = self.offset[lg]
            if idle:
                z = 0.0
            elif ph < P.duty:                                    # stance
                self.swinging[lg] = False
                o[0] -= govern * vx * dt
                o[1] -= govern * vy * dt
                z = 0.0
            else:                                                # swing
                sw = (ph - P.duty) / (1.0 - P.duty)
                ls = self.lift_start[lg]
                if not self.swinging[lg]:
                    self.swinging[lg] = True
                    ls[0], ls[1] = o[0], o[1]
                lx, ly = vx * P.duty * P.period / 2, vy * P.duty * P.period / 2
                n = math.hypot(lx, ly)
                if n > P.max_step:
                    lx, ly = lx * (P.max_step / n), ly * (P.max_step / n)
                c = _cycloid(sw)
                o[0] = ls[0] + (lx - ls[0]) * c
                o[1] = ls[1] + (ly - ls[1]) * c
                z = P.step_height * 0.5 * (1 - math.cos(2 * math.pi * sw))
            n = math.hypot(o[0], o[1])
            if n > lim_off:                                       # backstop; governor should prevent this
                o[0] *= lim_off / n
                o[1] *= lim_off / n
            nl = self.neutral[lg]
            feet[lg] = (nl[0] + o[0], nl[1] + o[1], nl[2] + (z + self.lower))

        # weight shift: move the body so the CoM sits over the loaded feet
        if P.weight_shift > 0:
            if P.shift_mode == "support":
                w = [self._support_weight(lg) for lg in LEG_IDS]
                pts = [feet[lg] for lg in LEG_IDS]
            else:
                w = [1.0] * len(LEG_IDS)
                pts = self._neutral_xy
            ws = max(sum(w), 1e-6)
            cx = sum(wi * pt[0] for wi, pt in zip(w, pts)) / ws
            cy = sum(wi * pt[1] for wi, pt in zip(w, pts)) / ws
            dx = P.weight_shift * (cx - self.com_offset[0])
            dy = P.weight_shift * (cy - self.com_offset[1])
            n = math.hypot(dx, dy)
            if n > 0.06:
                dx, dy = dx * (0.06 / n), dy * (0.06 / n)
            # low-pass so the body glides rather than jerks
            a = min(1.0, dt / 0.06)
            self.shift[0] += a * (dx - self.shift[0])
            self.shift[1] += a * (dy - self.shift[1])
        else:
            self.shift[:] = 0

        self.t += dt
        if not solve:
            return None
        q = (STAND if self.arms is None else self.arms).copy()
        for lg, idx in zip(LEG_IDS, LEG_IDX):
            f = feet[lg]
            p = np.array((f[0] - self.shift[0], f[1] - self.shift[1], f[2]))
            q_leg, ok = leg_ik(lg, p)
            self.unreachable += not ok
            q[idx] = q_leg
        # ease in from the STAND pose so switching from emotes to walking is smooth
        if self.t < P.blend_time:
            s = self.t / P.blend_time
            s = s * s * (3 - 2 * s)
            q = STAND + (q - STAND) * s
        return q
