"""Needs and emotions: why Pinch wants things, and how it feels about what happens.

Needs are urges from 0 (satisfied) to 1 (urgent). They grow on their own and are satisfied by
activities. Pinch is a bit needy by design: its wish for company grows steadily, and faster when you
are right there but ignoring it.

Emotion is a point in valence (-1 unhappy .. +1 happy) x arousal (0 calm .. 1 excited), nudged by
events and pulled back to a resting mood. A named feeling (happy, lonely, scared, ...) is read off
that point plus the needs and what just happened; the face and body show the named feeling.

Phase 4 (docs/research/v2_1/expression: Kismet's affect space, ALMA): each event also raises named emotions that
decay at their own pace (a fright in seconds, sadness over a minute; configs/mind/expression.toml), a slow mood
follows them over about ten minutes and shifts the resting valence, and the named feeling is the strongest emotion
(held at least 1.5 s, no flicker), else read off the mood point as before.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from pinch.settings import load as _load_settings

EXPR = _load_settings("expression")
EMO = EXPR["emotion"]


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
    fatigue: float = 0.0     # tired from moving (not sleepiness): builds while the body works, recovers at rest
    heat: float = 0.0        # its leg motors are warm: 0 up to 40 C, 1 at 60 C (set from the servos' temperatures)

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
    "high_up":       (-0.20, +0.25, dict(safety=+0.35)),     # still held in the air (repeats, grows with time)
    "put_down_relief": (+0.30, -0.30, dict(safety=-0.50)),
    "shaken":        (-0.50, +0.70, dict(safety=+0.70)),
    "bumped":        (-0.15, +0.40, dict(safety=+0.30)),
    "loud_noise":    (-0.20, +0.60, dict(safety=+0.50)),
    "sudden_motion": (-0.15, +0.50, dict(safety=+0.80)),     # something moved at it (scaled by timidity by the caller)
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
    "touched_unwanted": (-0.15, +0.30, dict(safety=+0.25)),  # touched by someone it fears
    "feared_person": (-0.12, +0.36, dict(safety=+0.30)),     # someone it is afraid of came into view
    "failed":        (-0.10, +0.10, dict()),                  # a goal-directed move didn't work
    "succeeded":     (+0.08, +0.00, dict()),                  # it did
}
FEAR_EVENTS = ("lifted", "shaken", "bumped", "loud_noise", "fell", "sudden_motion")


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
    emotions: dict = field(default_factory=dict)   # named emotion -> intensity 0..1, each decaying (expression.toml)
    mood: float = None           # slow valence (ALMA mood); None until the first tick
    _shown: str = ""             # the feeling on show and since when (hysteresis)
    _shown_t: float = -1e9

    # fear of harmless repeats wears off (a door that bangs every hour, a clumsy elbow): Stanley's habituation model
    # as in pinch/touch.py (Thompson & Spencer 1966; Rankin et al. 2009) - each repeat h <- h + alpha (1 - h), recovery
    # h <- h exp(-dt / tau) between them, the reaction never below FLOOR; falls, lifts and shakes never habituate
    HABITUATING = {"loud_noise": 0.35, "bumped": 0.3, "sudden_motion": 0.25}
    FEAR_TAU_S, FEAR_FLOOR = 600.0, 0.3

    def _habituated(self, event):
        alpha = self.HABITUATING.get(event)
        if alpha is None:
            return 1.0
        h = self.__dict__.setdefault("_fear_h", {})
        t0 = self.__dict__.get("_fear_t", {}).get(event)
        v = h.get(event, 0.0) * (math.exp(-(self.t - t0) / self.FEAR_TAU_S) if t0 is not None else 1.0)
        response = self.FEAR_FLOOR + (1 - self.FEAR_FLOOR) * (1 - v)
        h[event] = v + alpha * (1 - v)
        self.__dict__.setdefault("_fear_t", {})[event] = self.t
        return response

    def feel(self, event: str, strength: float = 1.0):
        if event not in EVENT_EFFECTS:
            return
        strength *= self._habituated(event)
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
        fear = (0.5 + p.timidity) if event in FEAR_EVENTS else 1.0
        for emo, k in EXPR["event"].get(event, {}).items():
            gain = fear if emo in ("scared", "surprised") else 1.0
            self.emotions[emo] = min(1.0, self.emotions.get(emo, 0.0) + k * strength * gain)
        self._clip()

    def tick(self, dt, owner_present=False, owner_attending=False, battery=None, awake_hours=None, sleep_drive=1.0):
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
        n.sleep += dt * NEED_RATES["sleep"] * (2.0 if awake_hours and awake_hours > 10 else 1.0) * sleep_drive
        n.safety -= dt * 0.05
        if battery is not None:
            n.energy = 1.0 - float(battery)
        # named emotions fade, each at its own pace; the mood follows the valence slowly (ALMA)
        hl = EMO["half_life"]
        self.emotions = {e: i * 0.5 ** (dt / hl.get(e, 20.0)) for e, i in self.emotions.items()
                         if i * 0.5 ** (dt / hl.get(e, 20.0)) > 0.02}
        if self.mood is None:
            self.mood = self.valence
        self.mood += (self.valence - self.mood) * min(1.0, dt / EXPR["mood"]["time_constant_s"])
        base_v = self.resting_valence + EXPR["mood"]["weight"] * (self.mood - self.resting_valence)
        # emotions relax toward a mood shaped by the needs
        rest_v = base_v - 0.35 * max(0, n.social - 0.6) - 0.3 * n.safety - 0.2 * max(0, n.energy - 0.7)
        rest_a = 0.25 + 0.2 * n.play * p.playfulness - 0.2 * n.sleep
        self.valence += (rest_v - self.valence) * min(1.0, dt / 30.0)
        self.arousal += (rest_a - self.arousal) * min(1.0, dt / 12.0)
        self._clip()

    def _clip(self):
        self.valence = float(np.clip(self.valence, -1, 1))
        self.arousal = float(np.clip(self.arousal, 0, 1))
        self.needs.clip()

    def feeling(self) -> str:
        """The named feeling right now (drives the face): fear first, quiet mode, then the strongest named emotion
        (held at least min_show_s unless another is clearly stronger), else the mood point."""
        n = self.needs
        dz = self.emotions.get("dizzy", 0.0)
        fresh = self.last_event in ("shaken", "fell") and self.t - self.last_event_t < 2.5
        if dz >= 0.4 and (fresh or dz >= max(self.emotions.values())):   # just shaken or tumbled: dizzy, then fear
            return self._show("dizzy")
        if n.safety > EMO["fear_threshold"]:
            return self._show("scared")
        if self.quiet:
            return self._show("quiet")
        cur = self._shown
        cur_i = self.emotions.get(cur, 0.0)
        best, best_i = max(self.emotions.items(), key=lambda kv: kv[1], default=("", 0.0))
        held = self.t - self._shown_t < EMO["min_show_s"] and cur and (cur_i >= 0.1 or cur not in self.emotions)
        if held and best_i < cur_i + EMO["switch_margin"]:
            return cur
        if best_i >= EMO["show_threshold"]:
            return self._show(best)
        f = self._mood_feeling()
        if held and f != cur and cur in self.emotions:
            return cur
        return self._show(f)

    def _show(self, f):
        if f != self._shown:
            self._shown, self._shown_t = f, self.t
        return f

    def _mood_feeling(self):
        """The feeling read off the mood point and the needs (when no named emotion is strong)."""
        n, v, a = self.needs, self.valence, self.arousal
        ev = self.last_event if self.t - self.last_event_t < 4.0 else ""
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
