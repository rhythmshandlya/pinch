"""Keyframed emotes. Shared between the simulator and (later) the real robot.

A pose is built from STAND plus overrides. Override keys are joint names or
fnmatch patterns over JOINT_NAMES, e.g.:
    "L_*"        every left-arm joint
    "*_claw"     both claws
    "?L_femur"   femurs of the left legs (FL, RL)
    "F?_*"       every joint of the front legs
"""

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np

from .params import DESIGN, JOINT_NAMES

STAND_DICT = {
    "*_coxa": 0.0,
    "*_femur": 0.35,
    "*_tibia": -1.92,          # shins near vertical: feet under the knees, less load on the thighs
    "*_shoulder_yaw": 0.10,
    "*_shoulder_pitch": 0.35,
    "*_claw": 0.10,
}
if DESIGN == "v2":
    # style B: thighs up 35 deg from horizontal, shins 85 deg below horizontal (78 deg was the first spec: it misses the owner's RMS torque limit, docs/dev/V2_MODEL.md),
    # arms straight ahead (docs/dev/V2_MODEL.md)
    import math as _m
    import os as _os
    _thigh, _shin = float(_os.environ.get("CRAB_THIGH_DEG", "0")), float(_os.environ.get("CRAB_SHIN_DEG", "85"))   # v2.1: thighs level (owner, 2026-11-23: more ground clearance; was 5 deg up, belly 35 mm)
    # the hips swung forward at stand so the feet sit under the centre of mass (18.7 mm ahead of the body's middle:
    # the 300 g face screen): even load on the four thighs instead of 62 % on the front pair
    _coxa = float(_os.environ.get("CRAB_COXA_FWD_DEG", "12.5"))   # 22/22 moves in the limits (0: 16/22 with level thighs)
    STAND_DICT.update({"*_coxa": _m.radians(_coxa), "*_femur": _m.radians(_thigh), "*_tibia": -_m.radians(_shin + _thigh),
                       "*_shoulder_yaw": 0.0, "*_shoulder_pitch": 0.35})


TALL_STANCE = DESIGN == "v2" and STAND_DICT["*_femur"] < 0.349     # thighs under 20 deg up


def pose(*overrides) -> np.ndarray:
    """Full joint-target vector: STAND, then each layer applied in order. A layer is an override dict,
    or a whole pose vector (which replaces everything so far: "start from this pose")."""
    q = np.zeros(len(JOINT_NAMES))
    for layer in (STAND_DICT, *overrides):
        if isinstance(layer, np.ndarray):
            q = layer.astype(float).copy()
            continue
        for pattern, value in layer.items():
            hits = [i for i, n in enumerate(JOINT_NAMES) if fnmatch(n, pattern)]
            if not hits:
                raise KeyError(f"pattern {pattern!r} matches no joint")
            q[hits] = value
    return q


def crouch(amount: float, legs: str = "*") -> dict:
    """Lower the body by lifting knees while keeping tibias near vertical. In a tall stance (v2 with the thighs under
    20 deg, design v2.1) the body never rises above it: a stretch there loads the thighs past the heat limit (scared,
    yawn and sneeze measured 73-83 % of rated RMS), so the arms carry the stretch."""
    if TALL_STANCE and amount < 0:
        amount = 0.0
    return {f"{legs}_femur": STAND_DICT["*_femur"] + amount, f"{legs}_tibia": STAND_DICT["*_tibia"] - amount}


STAND = pose()


@dataclass
class Emote:
    name: str
    keyframes: list          # [(time_s, pose_vector), ...], first at t=0
    loop: bool = False

    @property
    def duration(self) -> float:
        return self.keyframes[-1][0]

    def target(self, t: float) -> np.ndarray:
        if self.loop:
            t = t % self.duration
        t = min(max(t, 0.0), self.duration)
        for (t0, q0), (t1, q1) in zip(self.keyframes, self.keyframes[1:]):
            if t <= t1:
                s = (t - t0) / max(t1 - t0, 1e-9)
                s = s * s * (3 - 2 * s)          # smoothstep easing
                return q0 + (q1 - q0) * s
        return self.keyframes[-1][1]


def _seq(steps):
    """[(dt, pose), ...] -> [(t, pose), ...] with cumulative time."""
    out, t = [], 0.0
    for dt, q in steps:
        t += dt
        out.append((t, q))
    return out


def _build() -> dict:
    E = {}

    E["idle"] = Emote("idle", _seq([
        (0.0, STAND),
        (1.6, pose(crouch(0.10), {"*_claw": 0.30})),
        (1.6, STAND),
    ]), loop=True)

    up = {"L_shoulder_pitch": 1.22, "L_shoulder_yaw": 0.30, "L_claw": 0.6}     # pitch inside its CAD limit (72 deg)
    E["wave"] = Emote("wave", _seq(                                   # claw up, swings side to side
        [(0.0, STAND), (0.6, pose(up))]
        + [(0.42, pose(up, {"L_shoulder_yaw": y, "L_claw": c})) for y, c in ((0.05, 0.2), (0.32, 0.7)) * 3]
        + [(0.7, STAND)]
    ))

    raised = {"*_shoulder_pitch": 0.9, "*_shoulder_yaw": 0.20}
    lean = crouch(0.25, "F?")
    E["angry"] = Emote("angry", _seq(
        [(0.0, STAND), (0.5, pose(lean, raised, {"*_claw": 0.7}))]
        # snaps sized for the servo's measured torque-speed curve (rotor inertia dominates the claw)
        + [(0.36, pose(lean, raised, {"*_claw": c})) for c in (0.15, 0.7) * 4]
        + [(0.6, STAND)]
    ))

    arms_up = {"*_shoulder_pitch": 1.2, "*_shoulder_yaw": 0.30}   # claws up, apart
    left_low = pose(arms_up, crouch(0.30, "?L"), crouch(-0.10, "?R"), {"*_claw": 0.8})
    right_low = pose(arms_up, crouch(0.30, "?R"), crouch(-0.10, "?L"), {"*_claw": 0.1})
    E["happy"] = Emote("happy", _seq(
        [(0.0, STAND), (0.5, pose(arms_up))]
        + [(0.4, q) for q in (left_low, right_low) * 3]
        + [(0.6, STAND)]
    ))

    # claws raised in front of the eyes, swung in as far as they go
    hide = pose(crouch(0.38), {"L_shoulder_pitch": 1.15, "R_shoulder_pitch": 0.70, "*_shoulder_yaw": 0.0,
                               "*_claw": 0.0})
    E["shy"] = Emote("shy", _seq([
        (0.0, STAND), (0.8, hide), (1.5, hide),
        (0.4, pose(crouch(0.38), {"L_shoulder_pitch": 1.15, "R_shoulder_pitch": 0.70, "*_shoulder_yaw": 0.0,
                                  "R_shoulder_yaw": 0.15, "R_claw": 0.4})),   # peek
        (0.8, hide), (0.8, STAND),
    ]))

    tuck = {"*_shoulder_pitch": -0.2, "*_shoulder_yaw": 0.10, "*_claw": 0.0}
    nap = pose(crouch(0.38), tuck)            # tibia stays inside its CAD limit (-133 deg)
    E["sleep"] = Emote("sleep", _seq([
        (0.0, STAND), (1.5, nap), (1.5, pose(crouch(0.32), tuck)),   # slow "breath"
        (1.5, nap), (1.2, STAND),
    ]))

    # ---------------------------------------------------------------- Pinch's wider repertoire
    # Moods and tricks. The face screen's eyes carry most of the emotion; these add body language.
    # Arm ranges: swing 0..20 deg (outward), lift -40..72 deg, claw 0..55 deg (CAD).

    # shh: left claw up in front of the face, a little hunched (meetings, sleeping owner)
    shh = pose(crouch(0.12), {"L_shoulder_pitch": 1.15, "L_shoulder_yaw": 0.0, "L_claw": 0.0,
                              "R_shoulder_pitch": 0.15, "R_claw": 0.0})
    E["shh"] = Emote("shh", _seq([(0.0, STAND), (0.7, shh), (1.8, shh), (0.5, pose(shh, {"L_claw": 0.25})),
                                  (0.5, shh), (0.8, STAND)]))

    # quiet fidget: tiny claw twiddles, no body motion (silent "doing stupid stuff" nearby)
    E["fidget"] = Emote("fidget", _seq(
        [(0.0, STAND)]
        + [(0.5, pose({"L_claw": a, "R_claw": b, "L_shoulder_pitch": 0.35 + 0.15 * a,
                       "R_shoulder_pitch": 0.35 + 0.15 * b}))
           for a, b in ((0.4, 0.0), (0.0, 0.4), (0.25, 0.25), (0.0, 0.0)) * 2]
        + [(0.5, STAND)]))

    def beat(s):            # dance: body twist (all coxas), bounce, alternating claw pumps
        return pose(crouch(0.12 if s > 0 else -0.05),
                    {"?L_coxa": 0.18 * s, "?R_coxa": -0.18 * s,
                     "L_shoulder_pitch": 1.1 if s > 0 else 0.4, "R_shoulder_pitch": 0.4 if s > 0 else 1.1,
                     "*_shoulder_yaw": 0.25, "*_claw": 0.6 if s > 0 else 0.1})
    E["dance"] = Emote("dance", _seq([(0.0, STAND)] + [(0.45, beat(s)) for s in (1, -1) * 4] + [(0.6, STAND)]))

    # beg: sit back on the rear legs, claws up together, little "please" pumps
    beg = pose(crouch(0.30, "R?"), crouch(-0.12, "F?"), {"*_shoulder_pitch": 0.95, "*_shoulder_yaw": 0.0})
    E["beg"] = Emote("beg", _seq(
        [(0.0, STAND), (0.7, beg)]
        + [(0.35, pose(beg, {"*_claw": c, "*_shoulder_pitch": p})) for c, p in ((0.4, 1.05), (0.1, 0.95)) * 3]
        + [(0.7, STAND)]))

    # play dead: gasp, flop flat, one claw twitches, then pops back up
    flop = pose(crouch(0.38), {"*_shoulder_pitch": -0.35, "*_shoulder_yaw": 0.30, "*_claw": 0.7})
    E["play_dead"] = Emote("play_dead", _seq([
        (0.0, STAND), (0.25, pose(crouch(0.10), {"*_shoulder_pitch": 1.0, "*_claw": 0.8})),
        (0.6, flop), (2.5, flop), (0.3, pose(flop, {"R_claw": 0.0})), (0.3, flop), (0.9, STAND)]))

    # stretch: front feet forward and low, claws reach out and open wide
    reach = pose(crouch(0.25, "F?"), crouch(-0.08, "R?"),
                 {"F?_coxa": 0.25, "*_shoulder_pitch": 0.6, "*_shoulder_yaw": 0.30, "*_claw": 0.9})
    E["stretch"] = Emote("stretch", _seq([(0.0, STAND), (1.0, reach), (1.2, reach),
                                          (0.4, pose(reach, {"*_claw": 0.3})), (0.9, STAND)]))

    # excited: quick bounces with claws up
    # v2 (2.1 kg): slower, shallower bounces keep the hottest leg joint within the owner's heat limit (R-03)
    ex_dt, ex_lo, ex_hi = (0.30, 0.12, -0.05) if DESIGN == "v2" else (0.22, 0.18, -0.08)
    E["excited"] = Emote("excited", _seq(
        [(0.0, STAND)]
        + [(ex_dt, q) for q in (pose(crouch(ex_lo), {"*_shoulder_pitch": 0.9, "*_claw": 0.7}),
                                pose(crouch(ex_hi), {"*_shoulder_pitch": 1.15, "*_claw": 0.2})) * 4]
        + [(0.5, STAND)]))

    # sad: sinks low, claws hang, a slow sigh
    droop = pose(crouch(0.30), {"*_shoulder_pitch": -0.30, "*_shoulder_yaw": 0.05, "*_claw": 0.05})
    E["sad"] = Emote("sad", _seq([(0.0, STAND), (1.4, droop), (2.0, droop),
                                  (0.6, pose(droop, {"L_shoulder_pitch": -0.15})), (0.6, droop), (1.2, STAND)]))

    # curious: body tilts (right legs lower), one claw up, little claw "hmm"
    tilt = pose(crouch(0.18, "?R"), crouch(-0.08, "?L"), {"L_shoulder_pitch": 0.75, "L_claw": 0.35})
    E["curious"] = Emote("curious", _seq([(0.0, STAND), (0.6, tilt), (1.0, tilt), (0.4, pose(tilt, {"L_claw": 0.0})),
                                          (0.3, tilt), (0.7, STAND)]))

    # scared: drops low, claws up as a shield, trembles
    cower = pose(crouch(0.36), {"*_shoulder_pitch": 0.55, "*_shoulder_yaw": 0.0, "*_claw": 0.0})
    E["scared"] = Emote("scared", _seq(
        [(0.0, STAND), (0.45 if TALL_STANCE else 0.25, cower)]          # tall stance: a slower drop (thigh heat)
        + [(0.12, pose(cower, {"?L_coxa": 0.04 * s, "?R_coxa": -0.04 * s})) for s in (1, -1) * 5]
        + [(1.0, cower), (1.0, STAND)]))

    # point: right claw forward, closed (used when it shows you something it found)
    aim = pose({"R_shoulder_pitch": 0.45, "R_shoulder_yaw": 0.0, "R_claw": 0.0})
    E["point"] = Emote("point", _seq([(0.0, STAND), (0.6, aim), (1.5, aim), (0.6, STAND)]))

    # handshake: leans in, offers the right claw, shakes
    offer = pose(crouch(0.10, "F?"), {"R_shoulder_pitch": 0.15, "R_shoulder_yaw": 0.10, "R_claw": 0.6})
    E["handshake"] = Emote("handshake", _seq(
        [(0.0, STAND), (0.7, offer)] + [(0.35, pose(offer, {"R_shoulder_pitch": p})) for p in (0.35, 0.05) * 2]
        + [(1.0, offer), (0.7, STAND)]))

    # yawn: big slow reach up with claws wide, then settles a bit lower
    yawn = pose(crouch(-0.08), {"*_shoulder_pitch": 1.2, "*_shoulder_yaw": 0.33, "*_claw": 0.9})
    E["yawn"] = Emote("yawn", _seq([(0.0, STAND), (1.2, yawn), (0.8, yawn),
                                    (1.2, pose(crouch(0.15), {"*_shoulder_pitch": 0.1, "*_claw": 0.0})),
                                    (0.8, STAND)]))

    # sneeze: ah.. ah.. CHOO (front dips)
    pre = pose(crouch(-0.06, "F?"), crouch(0.06, "R?"), {"*_shoulder_pitch": 0.6})
    choo = pose(crouch(0.22, "F?"), {"*_shoulder_pitch": 0.1, "*_claw": 0.5})
    E["sneeze"] = Emote("sneeze", _seq([(0.0, STAND), (0.5, pre), (0.4, pose(pre, {"*_shoulder_pitch": 0.75})),
                                        (0.35 if TALL_STANCE else 0.22, choo), (0.5, choo), (0.7, STAND)]))

    # look around: turns the body left and right on planted feet (all coxas together)
    left, right = pose({"?L_coxa": 0.22, "?R_coxa": -0.22}), pose({"?L_coxa": -0.22, "?R_coxa": 0.22})
    E["look_around"] = Emote("look_around", _seq([(0.0, STAND), (0.7, left), (0.8, left), (1.0, right),
                                                  (0.8, right), (0.7, STAND)]))

    # applause: claws up and clacking open / shut
    # v2: the arm servos run at a 40 % torque limit (child safety), so the claps are slower and a little smaller
    cl_dt, cl_open = (0.26, 0.6) if DESIGN == "v2" else (0.18, 0.8)
    E["applause"] = Emote("applause", _seq(
        [(0.0, STAND), (0.5, pose({"*_shoulder_pitch": 0.8, "*_shoulder_yaw": 0.0, "*_claw": cl_open}))]
        + [(cl_dt, pose({"*_shoulder_pitch": 0.8, "*_shoulder_yaw": 0.0, "*_claw": c})) for c in (0.05, cl_open) * 5]
        + [(0.5, STAND)]))

    return E


EMOTES = _build()


# ---------------------------------------------------------------------------------------------- variation
# The keyframes are the base clip; every play layers a style on top (animation layering):
#   tempo       time scale (>1 faster)                       affect as a global pass over a base motion
#   amplitude   scale of every joint's deviation from STAND  (time and size): Snibbe et al. 1999; Laban
#   asym        one arm a little smaller than the other      Effort "time"/"space" (Knight & Simmons 2015)
#   lag         one arm trails by up to 0.12 s               overlap / follow-through
#   mirror      left <-> right                               lets a reaction use the touched side
#   gaze        a body turn (all coxas) toward a side, eased in and out
# Arousal makes it faster and bigger, sleepiness slower and smaller, valence a little bigger. Drawn from a
# seeded generator (exact replay). Safe by construction: legs get half the amplitude change and no lag or
# asymmetry (they carry the balance), every target is clipped to the CAD joint limits and the shin limit, the
# peak joint speed stays within max(base peak, min(90 % of the servo limit, 1.25 x base peak)) (tempo and size
# cut together when over), and the arms cannot touch within the CAD shoulder limits (checked in the tests).

_SIDE = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL", "L": "R", "R": "L"}
MIRROR = np.array([JOINT_NAMES.index(_SIDE[n.split("_", 1)[0]] + "_" + n.split("_", 1)[1]) for n in JOINT_NAMES])
_IS_LEG = np.array([n.split("_", 1)[0] in ("FL", "FR", "RL", "RR") for n in JOINT_NAMES])
_ARM_L = np.array([n.startswith("L_") for n in JOINT_NAMES])
_ARM_R = np.array([n.startswith("R_") for n in JOINT_NAMES])
_TURN = np.array([1.0 if n in ("FL_coxa", "RL_coxa") else -1.0 if n in ("FR_coxa", "RR_coxa") else 0.0
                  for n in JOINT_NAMES])              # + = body turns left (as look_around's "left" pose)
_FEMUR = np.array([JOINT_NAMES.index(f"{lg}_femur") for lg in ("FL", "FR", "RL", "RR")])
_TIBIA = np.array([JOINT_NAMES.index(f"{lg}_tibia") for lg in ("FL", "FR", "RL", "RR")])
_YAW_PITCH_L = [JOINT_NAMES.index("L_shoulder_yaw"), JOINT_NAMES.index("L_shoulder_pitch")]
_YAW_PITCH_R = [JOINT_NAMES.index("R_shoulder_yaw"), JOINT_NAMES.index("R_shoulder_pitch")]
LIMIT_MARGIN = 0.02                    # rad inside the CAD limits
SERVO_SHARE = 0.9                      # a variant may use at most this share of a servo's commanded-speed limit
SPEED_GAIN = 1.25                      # ... and at most this much more than the base clip's own peak
SAMPLING = 1.02                        # the peaks are sampled (10 ms): 2 % allowance for the true peak
GAZE_MAX = 0.15                        # rad of body turn (look_around itself turns 0.22)


def joint_limits(margin=0.0):
    """(lo, hi) per joint from params.py (CAD-checked); robot/crab_robot.py clips to the same ranges."""
    from .params import PARAMS
    L, A = PARAMS.leg, PARAMS.arm
    r = []
    for j in JOINT_NAMES:
        side, kind = j.split("_", 1)
        r.append({"coxa": L.coxa_range_for(side), "femur": L.femur_range_for(side), "tibia": L.tibia_range,
                  "shoulder_yaw": A.shoulder_yaw_range, "shoulder_pitch": A.shoulder_pitch_range,
                  "claw": A.claw_range}[kind])
    r = np.array(r)
    return r[:, 0] + margin, r[:, 1] - margin


def servo_step(dt=0.02):
    """Per-joint max target change per control tick (the servo speed limit robot/crab_robot.py enforces)."""
    from .params import JOINT_SPECS
    return np.array([s.max_cmd_speed for s in JOINT_SPECS]) * dt


@dataclass(frozen=True)
class Style:
    tempo: float = 1.0
    amplitude: float = 1.0
    asym: float = 0.0          # -1..1: + = the right arm smaller by that share, - = the left
    lag: float = 0.0           # s the trailing arm lags
    lag_left: bool = True      # which arm trails
    mirror: bool = False
    gaze: float = 0.0          # rad, + = turn left

    def short(self):
        return dict(tempo=round(self.tempo, 3), amp=round(self.amplitude, 3), asym=round(self.asym, 3),
                    lag=round(self.lag, 3), mirror=self.mirror, gaze=round(self.gaze, 3))


PLAIN = Style()


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class VariedEmote:
    """A base emote played with a Style. Same interface as Emote (name, duration, loop, target)."""

    def __init__(self, base: Emote, style: Style = PLAIN):
        from .params import PARAMS
        self.base, self.style, self.name, self.loop = base, style, base.name, base.loop
        self.lo, self.hi = joint_limits(LIMIT_MARGIN)
        self.shin_min = PARAMS.leg.shin_min
        s = style
        amp = np.where(_IS_LEG, 1.0 + 0.5 * (s.amplitude - 1.0), s.amplitude)
        self.amp = amp * np.where(_ARM_L, 1.0 - max(0.0, -s.asym), np.where(_ARM_R, 1.0 - max(0.0, s.asym), 1.0))
        self.lag = 0.0 if base.loop else max(0.0, s.lag)
        self.lag_mask = _ARM_L if s.lag_left else _ARM_R
        self.duration = base.duration / s.tempo + self.lag

    def _base(self, u):
        q = self.base.target(u)
        return q[MIRROR] if self.style.mirror else q

    def target(self, t: float) -> np.ndarray:
        s = self.style
        if self.loop:
            t = t % self.duration
        t = min(max(t, 0.0), self.duration)
        q = self._base(t * s.tempo)
        if self.lag > 0:
            q = np.where(self.lag_mask, self._base(max(0.0, t - self.lag) * s.tempo), q)
        q = STAND + self.amp * (q - STAND)
        if s.gaze:
            env = 1.0 if self.loop else _smooth(t / 0.4) * _smooth((self.duration - t) / 0.5)
            q = q + s.gaze * env * _TURN
        q = np.clip(q, self.lo, self.hi)
        q[_TIBIA] = np.maximum(q[_TIBIA], self.shin_min - q[_FEMUR])
        return q

    def samples(self, dt=0.02):
        ts = np.arange(0.0, self.duration + dt, dt)
        return ts, np.array([self.target(t) for t in ts])


def peak_speed(emo, dt=0.02):
    """Largest per-tick joint change as a share of that servo's per-tick limit."""
    emo = emo if isinstance(emo, VariedEmote) else VariedEmote(emo)
    _, Q = emo.samples(dt)
    return float((np.abs(np.diff(Q, axis=0)) / servo_step(dt)).max()) if len(Q) > 1 else 0.0


def arms_ok(emo, dt=0.04):
    """The two arms never touch through the whole emote (crab/kinematics.py: boxes around the CAD parts)."""
    from .kinematics import arms_clear
    _, Q = emo.samples(dt)
    return all(arms_clear(q[_YAW_PITCH_L], q[_YAW_PITCH_R]) for q in Q)


def dominant_side(emo):
    """+1 if the emote mostly moves the left arm, -1 the right, 0 if both alike."""
    Q = np.array([q for _, q in emo.keyframes]) - STAND
    left, right = np.abs(Q[:, _ARM_L]).sum(), np.abs(Q[:, _ARM_R]).sum()
    return 1 if left > 1.5 * right else -1 if right > 1.5 * left else 0


class EmoteStyler:
    """Chooses a Style for every play: affect sets the mean, a seeded generator the rest, and the same emote
    twice running is pushed apart. State: its generator and the last style of each emote."""

    TEMPO_AROUSAL, AMP_AROUSAL = 0.30, 0.25        # per unit of arousal away from 0.4 (the resting liveliness)
    SLEEPY = 0.15                                  # fully sleepy: 15 % slower and smaller
    VALENCE_TEMPO, VALENCE_AMP = 0.04, 0.06
    JITTER = 0.06                                  # +- uniform on tempo and amplitude
    TEMPO_RANGE, AMP_RANGE = (0.78, 1.2), (0.6, 1.15)
    MIN_APART = 0.06                               # the same emote back to back: tempo differs by at least this
    ASYM, LAG, GAZE_JITTER = 0.15, 0.12, 0.04

    def __init__(self, rng):
        self.rng = rng
        self.last = {}
        self.last_name = None
        self._base_peak = {}

    def mean(self, arousal=0.4, valence=0.0, sleepy=0.0, gain=1.0):
        """Affect -> (tempo, amplitude) before jitter. gain (0..1]: how strongly to react (habituation)."""
        a, v, s = float(np.clip(arousal, 0, 1)), float(np.clip(valence, -1, 1)), float(np.clip(sleepy, 0, 1))
        tempo = (1 + self.TEMPO_AROUSAL * (a - 0.4) + self.VALENCE_TEMPO * v) * (1 - self.SLEEPY * s)
        amp = (1 + self.AMP_AROUSAL * (a - 0.4) + self.VALENCE_AMP * v) * (1 - self.SLEEPY * s)
        return tempo, amp * (0.6 + 0.4 * float(np.clip(gain, 0, 1)))

    def style(self, name, arousal=0.4, valence=0.0, sleepy=0.0, side=None, gaze=None, gain=1.0):
        base, r = EMOTES[name], self.rng
        t0, a0 = self.mean(arousal, valence, sleepy, gain)
        tempo = t0 * (1 + r.uniform(-self.JITTER, self.JITTER))
        amp = a0 * (1 + r.uniform(-self.JITTER, self.JITTER))
        asym = float(r.uniform(-self.ASYM, self.ASYM))
        lag = float(r.uniform(0.0, self.LAG))
        lag_left = bool(r.random() < 0.5)
        flip = bool(r.random() < 0.3)
        jitter_gaze = float(r.uniform(-self.GAZE_JITTER, self.GAZE_JITTER))
        dom = dominant_side(base)
        mirror = dom != 0 and (dom != int(np.sign(side)) if side else flip)
        gaze = jitter_gaze if gaze is None else gaze
        prev = self.last.get(name) if self.last_name == name else None
        if prev is not None and abs(tempo - prev.tempo) < self.MIN_APART:     # anti-repetition
            tempo = prev.tempo + (self.MIN_APART if prev.tempo <= t0 else -self.MIN_APART)
            lag_left = not prev.lag_left
            asym = -prev.asym if abs(asym - prev.asym) < 0.05 else asym
        st = Style(float(np.clip(tempo, *self.TEMPO_RANGE)), float(np.clip(amp, *self.AMP_RANGE)), asym, lag,
                   lag_left, bool(mirror), float(np.clip(gaze, -GAZE_MAX, GAZE_MAX)))
        st = self.make_safe(base, st)
        if prev is not None and abs(st.tempo - prev.tempo) < self.MIN_APART and abs(st.amplitude - prev.amplitude) < self.MIN_APART:
            # the speed cap pulled it back next to the last play: go the other way (slower and smaller is always safe)
            from dataclasses import replace
            tempo = max(self.TEMPO_RANGE[0], prev.tempo - self.MIN_APART)
            amp = st.amplitude if prev.tempo - tempo >= self.MIN_APART - 1e-9 else prev.amplitude * (1 - self.MIN_APART * 1.5)
            st = replace(st, tempo=min(st.tempo, tempo), amplitude=min(st.amplitude, amp))
        self.last[name], self.last_name = st, name
        return st

    def joint_peaks(self, name):
        """Per-joint peak change per tick of the plain clip, as a share of each servo's limit (computed once)."""
        if name not in self._base_peak:
            _, Q = VariedEmote(EMOTES[name]).samples(0.01)
            self._base_peak[name] = np.abs(np.diff(Q, axis=0)).max(axis=0) * 2 / servo_step()
        return self._base_peak[name]

    def base_peak(self, name):
        return float(self.joint_peaks(name).max())

    def allowed_speed(self, name):
        bp = self.base_peak(name)
        return min(1.0, max(SAMPLING * bp, min(SERVO_SHARE, SPEED_GAIN * bp)))

    def speed_bound(self, name, st):
        """An upper bound on the variant's peak speed (share of the servo limit), without sampling it: each joint
        moves tempo x its amplitude factor as fast as in the plain clip (mirrored if so); clipping to the limits
        and the shin clamp never add speed; the gaze adds at most its eased ramp's slope on the coxas."""
        p = self.joint_peaks(name)
        p = p[MIRROR] if st.mirror else p
        v = VariedEmote(EMOTES[name], st)
        gaze = abs(st.gaze) * 1.5 / 0.4 * 0.02 * np.abs(_TURN) / servo_step()
        return float((SAMPLING * st.tempo * np.abs(v.amp) * p + gaze).max())

    def make_safe(self, base, st):
        """Peak speed within the allowance: cut tempo and size together until it fits (the gaze too if needed).
        The arms cannot touch within the CAD shoulder limits (yaw >= 0; 20k random poses checked, none touch), and
        every target is clipped to those limits; tests/test_variation.py samples every variant anyway."""
        from dataclasses import replace
        allowed = self.allowed_speed(base.name)
        for k in range(6):
            b = self.speed_bound(base.name, st)
            if b <= allowed:
                return st
            k_ = (allowed / b * 0.999) ** 0.5             # share the cut between tempo and size (both carry the mood)
            st = replace(st, tempo=st.tempo * k_, amplitude=st.amplitude * k_, gaze=st.gaze * (0.5 if k >= 2 else 1.0))
        return replace(st, tempo=min(st.tempo, 1.0), amplitude=min(st.amplitude, 1.0), asym=0.0, gaze=0.0)

    def vary(self, name, **affect):
        return VariedEmote(EMOTES[name], self.style(name, **affect))
