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


def pose(*overrides: dict) -> np.ndarray:
    """Full joint-target vector: STAND, then each override dict applied in order."""
    q = np.zeros(len(JOINT_NAMES))
    for layer in (STAND_DICT, *overrides):
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

    return E


EMOTES = _build()
