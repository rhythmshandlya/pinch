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

from .params import JOINT_NAMES

STAND_DICT = {
    "*_coxa": 0.0,
    "*_femur": 0.35,
    "*_tibia": -1.92,          # shins near vertical: feet under the knees, less load on the thighs
    "*_shoulder_yaw": 0.10,
    "*_shoulder_pitch": 0.35,
    "*_claw": 0.10,
}


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
    """Lower the body by lifting knees while keeping tibias near vertical."""
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
    E["excited"] = Emote("excited", _seq(
        [(0.0, STAND)]
        + [(0.22, q) for q in (pose(crouch(0.18), {"*_shoulder_pitch": 0.9, "*_claw": 0.7}),
                               pose(crouch(-0.08), {"*_shoulder_pitch": 1.15, "*_claw": 0.2})) * 4]
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
        [(0.0, STAND), (0.25, cower)]
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
                                        (0.22, choo), (0.5, choo), (0.7, STAND)]))

    # look around: turns the body left and right on planted feet (all coxas together)
    left, right = pose({"?L_coxa": 0.22, "?R_coxa": -0.22}), pose({"?L_coxa": -0.22, "?R_coxa": 0.22})
    E["look_around"] = Emote("look_around", _seq([(0.0, STAND), (0.7, left), (0.8, left), (1.0, right),
                                                  (0.8, right), (0.7, STAND)]))

    # applause: claws up and clacking open / shut
    E["applause"] = Emote("applause", _seq(
        [(0.0, STAND), (0.5, pose({"*_shoulder_pitch": 0.8, "*_shoulder_yaw": 0.0, "*_claw": 0.8}))]
        + [(0.18, pose({"*_shoulder_pitch": 0.8, "*_shoulder_yaw": 0.0, "*_claw": c})) for c in (0.05, 0.8) * 5]
        + [(0.5, STAND)]))

    return E


EMOTES = _build()
