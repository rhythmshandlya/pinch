"""The idle life layer: small procedural motion whenever no motion step runs, so a still Pinch never looks
switched off (decision 74). Following the research plan (APPROACH_PLAN.md, idle aliveness):

  * breathing: a slow rise and fall of the body (as before)
  * weight shift: a slower, wandering sway between the legs
  * micro-actions now and then - a claw flex, a shoulder twitch, one leg resettling, a held still moment -
    at random (Poisson) times, never the same one twice running, each a smooth bump up and back
  * liveliness (from the mind: arousal) scales how often and how big; a long quiet spell settles it further

Small by construction (legs <= 0.03 rad, arms <= 0.15 rad) and seeded, so runs replay exactly. Body adds the
offsets on top of the held pose only while idle; the moment a step starts they are gone.
"""

import math

import numpy as np

from crab.params import JOINT_NAMES
from pinch.settings import load

A = load("ambient")                                   # configs/mind/ambient.toml
BR, SW, FI = A["breath"], A["sway"], A["fidget"]

J = {n: i for i, n in enumerate(JOINT_NAMES)}
LEGS = ("FL", "FR", "RL", "RR")
ACTIONS = ("claw_flex_L", "claw_flex_R", "shoulder_L", "shoulder_R", "leg_settle", "still")


class IdleLife:
    def __init__(self, rng, period_breath=BR["period_s"]):
        self.rng = rng
        self.period_breath = period_breath
        self.liveliness = 0.4                       # 0 drowsy .. 1 excited (set by the mind)
        self.quiet_s = 0.0                          # how long nothing has happened around it
        self.next_t = 2.0
        self.current = None                         # (action, t0, duration, amplitude, leg)
        self.recent = []
        self.events = []                            # (t, action): what it did, for tests and the observer
        self.sway_phase = rng.uniform(0, 2 * math.pi)

    def _pick(self):
        weights = np.array([0.0 if a in self.recent[-2:] else 1.0 for a in ACTIONS])
        weights[ACTIONS.index("still")] *= 0.5
        return ACTIONS[self.rng.choice(len(ACTIONS), p=weights / weights.sum())]

    def offsets(self, t, dt):
        """Joint offsets (18) for this tick."""
        lv = max(0.05, min(1.0, self.liveliness)) * (0.5 if self.quiet_s > 300 else 1.0)
        q = np.zeros(len(JOINT_NAMES))
        b = BR["legs_rad"] * math.sin(2 * math.pi * t / self.period_breath)          # breathing
        for lg in LEGS:
            q[J[f"{lg}_femur"]] += b
        cb = BR["claws_rad"] * math.sin(2 * math.pi * t / self.period_breath + BR["claws_lag_rad"])   # the claws too
        for side in "LR":
            q[J[f"{side}_claw"]] += cb
            q[J[f"{side}_shoulder_pitch"]] += BR["shoulders_share"] * cb
        s = SW["legs_rad"] * lv * math.sin(2 * math.pi * t / SW["period_s"] + self.sway_phase)   # weight shift
        for lg in ("FL", "RL"):
            q[J[f"{lg}_femur"]] += s
        for lg in ("FR", "RR"):
            q[J[f"{lg}_femur"]] -= s
        if self.current is None and t >= self.next_t:
            a = self._pick()
            dur = float(self.rng.uniform(*FI["duration_s"]))
            amp = float(self.rng.uniform(0.5, 1.0))
            leg = LEGS[self.rng.integers(4)]
            self.current = (a, t, dur, amp, leg)
            self.recent = (self.recent + [a])[-3:]
            self.events.append((round(t, 2), a))
        if self.current is not None:
            a, t0, dur, amp, leg = self.current
            u = (t - t0) / dur
            if u >= 1.0:
                self.current = None
                gap = float(self.rng.exponential(FI["mean_gap_s"] / (0.4 + lv)))          # livelier: more often
                self.next_t = t + max(FI["min_gap_s"], gap)
            else:
                bump = math.sin(math.pi * u) ** 2 * amp
                if a.startswith("claw_flex"):
                    q[J[f"{a[-1]}_claw"]] += FI["claw_rad"] * lv * bump
                elif a.startswith("shoulder"):
                    q[J[f"{a[-1]}_shoulder_pitch"]] += FI["shoulder_rad"] * lv * bump
                elif a == "leg_settle":
                    q[J[f"{leg}_femur"]] += FI["leg_femur_rad"] * bump
                    q[J[f"{leg}_tibia"]] -= FI["leg_tibia_rad"] * bump
        return q
