"""Needs and emotions: why Pinch wants things, and how it feels about what happens.

Needs are urges from 0 (satisfied) to 1 (urgent). They grow on their own and are satisfied by
activities. Pinch is a bit needy by design: its wish for company grows steadily, and faster when you
are right there but ignoring it.

Emotion is a point in valence (-1 unhappy .. +1 happy) x arousal (0 calm .. 1 excited), nudged by
events and pulled back to a resting mood. A named feeling (happy, lonely, scared, ...) is read off
that point plus the needs and what just happened; the face and body show the named feeling.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Personality:
    neediness: float = 1.0      # how fast it gets lonely (1 = "a bit needy")
    playfulness: float = 0.8
    timidity: float = 0.5       # how easily startled
    tidiness: float = 0.7       # how much things out of place bother it
    curiosity: float = 0.8


# per-second growth of each need (before personality), and what it feels like when urgent
NEED_RATES = dict(social=1 / 600, play=1 / 900, curiosity=1 / 700, sleep=1 / 7200, tidy=0.0,
                  energy=0.0, safety=0.0)


@dataclass
class Needs:
    social: float = 0.3      # wants company / attention
    play: float = 0.3
    curiosity: float = 0.3
    sleep: float = 0.1       # sleepiness
    energy: float = 0.1      # 1 - battery fraction (set from the battery monitor)
    tidy: float = 0.0        # things out of place that it has noticed
    safety: float = 0.0      # fear: spikes on scares, decays

    def clip(self):
        for k in self.__dataclass_fields__:
            setattr(self, k, float(np.clip(getattr(self, k), 0.0, 1.0)))

    def as_dict(self):
        return {k: round(getattr(self, k), 3) for k in self.__dataclass_fields__}


# event -> (valence push, arousal push, need changes)
EVENT_EFFECTS = {
    "petted":        (+0.35, +0.05, dict(social=-0.35, safety=-0.3)),
    "stroked":       (+0.45, -0.10, dict(social=-0.45, safety=-0.4)),
    "tapped":        (+0.05, +0.15, dict(social=-0.10)),
    "booped":        (+0.15, +0.25, dict(social=-0.10, play=-0.05)),
    "tickled":       (+0.40, +0.45, dict(social=-0.25, play=-0.30)),
    "held_leg":      (-0.05, +0.20, dict(social=-0.10)),
    "lifted":        (-0.25, +0.60, dict(safety=+0.55)),
    "shaken":        (-0.50, +0.70, dict(safety=+0.70)),
    "bumped":        (-0.15, +0.40, dict(safety=+0.30)),
    "loud_noise":    (-0.20, +0.60, dict(safety=+0.50)),
    "praised":       (+0.55, +0.35, dict(social=-0.30)),
    "scolded":       (-0.55, +0.25, dict(social=+0.10)),
    "greeted":       (+0.40, +0.40, dict(social=-0.30)),
    "name_called":   (+0.15, +0.30, dict(social=-0.10, curiosity=-0.05)),
    "spoken_to":     (+0.10, +0.10, dict(social=-0.10)),
    "played_with":   (+0.40, +0.40, dict(play=-0.50, social=-0.25)),
    "trick_done":    (+0.20, +0.20, dict(play=-0.20)),
    "tidied":        (+0.20, +0.05, dict(tidy=-0.6)),
    "novelty":       (+0.10, +0.25, dict(curiosity=-0.30)),
    "owner_left":    (-0.20, -0.05, dict(social=+0.10)),
    "owner_returned": (+0.50, +0.50, dict(social=-0.20)),
    "owner_sad":     (-0.20, -0.05, dict()),
    "owner_happy":   (+0.25, +0.15, dict()),
    "owner_stressed": (-0.10, +0.10, dict()),
    "fed":           (+0.30, +0.00, dict(energy=-0.5)),     # put on the charging tray
    "fell":          (-0.40, +0.70, dict(safety=+0.60)),
    "stuck":         (-0.35, +0.45, dict(safety=+0.35)),     # a foot over the edge, caught, going nowhere
    "unstuck":       (+0.40, -0.20, dict(safety=-0.40)),     # relief
}


@dataclass
class Affect:
    personality: Personality = field(default_factory=Personality)
    needs: Needs = field(default_factory=Needs)
    valence: float = 0.3
    arousal: float = 0.3
    resting_valence: float = 0.25
    last_event: str = ""
    last_event_t: float = -1e9
    t: float = 0.0
    quiet: bool = False          # set while the owner is in a meeting / asleep
    being_touched: bool = False
    ignored_s: float = 0.0       # owner present, but not paying attention to Pinch

    def feel(self, event: str, strength: float = 1.0):
        if event not in EVENT_EFFECTS:
            return
        dv, da, dn = EVENT_EFFECTS[event]
        p = self.personality
        if event in ("lifted", "shaken", "bumped", "loud_noise", "fell"):
            da *= 0.5 + p.timidity
            dn = {k: v * (0.5 + p.timidity) for k, v in dn.items()}
        self.valence += dv * strength
        self.arousal += da * strength
        for k, v in dn.items():
            setattr(self.needs, k, getattr(self.needs, k) + v * strength)
        if event in ("petted", "stroked", "praised", "greeted", "played_with", "tickled", "spoken_to",
                     "name_called"):
            self.ignored_s = 0.0
        self.last_event, self.last_event_t = event, self.t
        self._clip()

    def tick(self, dt, owner_present=False, owner_attending=False, battery=None, awake_hours=None):
        """Needs grow, emotions drift back to the resting mood."""
        self.t += dt
        n, p = self.needs, self.personality
        if owner_present and not owner_attending:
            self.ignored_s += dt
        elif not owner_present:
            self.ignored_s = 0.0
        # a bit needy: company matters, and being ignored while you're right there stings
        lonely_rate = NEED_RATES["social"] * p.neediness * (1.0 + (1.5 if self.ignored_s > 120 else 0.0))
        n.social += dt * lonely_rate
        n.play += dt * NEED_RATES["play"] * p.playfulness
        n.curiosity += dt * NEED_RATES["curiosity"] * p.curiosity
        n.sleep += dt * NEED_RATES["sleep"] * (2.0 if awake_hours and awake_hours > 10 else 1.0)
        n.safety -= dt * 0.05
        if battery is not None:
            n.energy = 1.0 - float(battery)
        # emotions relax toward a mood shaped by the needs
        rest_v = self.resting_valence - 0.35 * max(0, n.social - 0.6) - 0.3 * n.safety - 0.2 * max(0, n.energy - 0.7)
        rest_a = 0.25 + 0.2 * n.play * p.playfulness - 0.2 * n.sleep
        self.valence += (rest_v - self.valence) * min(1.0, dt / 30.0)
        self.arousal += (rest_a - self.arousal) * min(1.0, dt / 12.0)
        self._clip()

    def _clip(self):
        self.valence = float(np.clip(self.valence, -1, 1))
        self.arousal = float(np.clip(self.arousal, 0, 1))
        self.needs.clip()

    def feeling(self) -> str:
        """The named feeling right now (drives the face)."""
        n, v, a = self.needs, self.valence, self.arousal
        recent = self.t - self.last_event_t < 4.0
        ev = self.last_event if recent else ""
        if n.safety > 0.55:
            return "scared"
        if ev in ("shaken", "fell") and recent:
            return "dizzy"
        if self.quiet:
            return "quiet"
        if ev in ("stroked", "petted") and recent:
            return "love" if v > 0.5 else "content"
        if ev == "tickled" and recent:
            return "squeeze"
        if ev == "praised" and recent:
            return "proud"
        if ev == "scolded" and recent:
            return "sad" if v < 0 else "shy"
        if ev in ("lifted", "loud_noise", "bumped") and recent:
            return "surprised"
        if n.sleep > 0.8 or n.energy > 0.85:
            return "sleepy"
        if v > 0.55 and a > 0.6:
            return "excited"
        if v > 0.45:
            return "happy"
        if v < -0.45 and a > 0.55:
            return "angry"
        if v < -0.35:
            return "sad"
        if n.social > 0.7:
            return "lonely"
        if v < -0.15 and a > 0.45:
            return "grumpy"
        if n.curiosity > 0.65 or ev == "novelty":
            return "curious"
        if n.play > 0.75 and a < 0.35:
            return "bored"
        if v > 0.15:
            return "content" if a < 0.3 else "happy"
        return "neutral"
