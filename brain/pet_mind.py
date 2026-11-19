"""The crab's pet mind: what makes it curious and proactive instead of waiting to be asked.

Three layers (see docs/PET_MIND.md):
  reflexes  every control tick   breathing / fidgets, flinch, edge stop, look toward things
  heart     ~1 Hz                drives (curiosity, boredom, energy, social, fear) -> behaviour
  mind      every 10-60 s        Gemma 4 (or a stand-in) suggests a behaviour + sound; it only
                                 *biases* the heart, it never blocks the fast layers

Senses arrive as events: Event(kind, strength, data). The heart turns drives into a
behaviour choice with simple utility scores, so the crab does things on its own: a
bored crab goes exploring, a lonely one seeks attention, a startled one hides.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

BEHAVIOURS = ["idle", "explore", "approach", "play", "seek_attention", "rest", "hide", "startle"]


@dataclass
class Event:
    kind: str            # touch | voice | loud_sound | motion | face | novelty | lifted | edge | battery_low
    strength: float = 1.0
    data: dict = field(default_factory=dict)


@dataclass
class Drives:
    curiosity: float = 0.3
    boredom: float = 0.2
    energy: float = 0.8
    social: float = 0.4      # wish for company (rises when alone)
    fear: float = 0.0
    mood: float = 0.2        # -1 sad .. +1 happy

    def clip(self):
        for k in ("curiosity", "boredom", "energy", "social", "fear"):
            setattr(self, k, float(np.clip(getattr(self, k), 0, 1)))
        self.mood = float(np.clip(self.mood, -1, 1))


class Novelty:
    """How different is what I see from what I've seen lately? (tiny image fingerprints)"""

    def __init__(self, memory=40):
        self.mem, self.size = [], memory

    @staticmethod
    def fingerprint(rgb: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        small = rgb[:: max(1, h // 12), :: max(1, w // 16)].astype(float) / 255.0
        hist = np.concatenate([np.histogram(small[..., c], bins=8, range=(0, 1))[0] for c in range(3)])
        return np.concatenate([small.mean(axis=2).ravel(), hist / hist.sum()])

    def __call__(self, rgb) -> float:
        f = self.fingerprint(rgb)
        nov = 0.0 if not self.mem else float(min(np.linalg.norm(f - m) for m in self.mem) / (np.sqrt(len(f)) * 0.08))
        self.mem = (self.mem + [f])[-self.size:]
        return float(np.clip(nov, 0, 1))


class Heart:
    """Drives evolve every second; behaviours are chosen by utility."""

    def __init__(self, rng=None):
        self.d = Drives()
        self.behaviour, self.since, self.t = "idle", 0.0, 0.0
        self.bias, self.bias_until = None, -1.0        # suggestion from the mind
        self.rng = rng or np.random.default_rng(0)
        self.log = []

    def feel(self, ev: Event):
        d, s = self.d, ev.strength
        if ev.kind == "touch":
            d.social -= 0.5 * s
            d.mood += 0.4 * s
            d.fear -= 0.3 * s
            d.boredom -= 0.3 * s
            self.urge("play", 8.0)                          # petted -> a happy wiggle
        elif ev.kind in ("voice", "face"):
            d.social -= 0.2 * s
            d.curiosity += 0.3 * s
            d.boredom -= 0.3 * s
            self.urge("approach", 10.0)                     # someone's there -> go and look
        elif ev.kind == "novelty":
            # habituation: something new only counts once every few seconds, and what it sees
            # while walking about is mostly its own doing
            if self.t - self.last_novelty < 6.0:
                return
            self.last_novelty = self.t
            k = 0.3 if self.behaviour in ("explore", "approach") else 1.0
            d.curiosity += 0.35 * s * k
            d.boredom -= 0.3 * s * k * (s > 0.75)                # only something really new beats boredom
        elif ev.kind in ("loud_sound", "lifted"):
            d.fear += 0.9 * s
            d.mood -= 0.2 * s
        elif ev.kind == "motion":
            d.curiosity += 0.2 * s
            d.fear += 0.1 * s
        elif ev.kind == "battery_low":
            d.energy = min(d.energy, 0.15)
        d.clip()
        self.log.append((self.t, "event", ev.kind))

    def suggest(self, behaviour, seconds=20.0):
        """A nudge from the slow mind (Gemma): biases the choice for a while, never forces it."""
        if behaviour in BEHAVIOURS:
            self.bias, self.bias_until = behaviour, self.t + seconds

    def urge(self, behaviour, seconds):
        """A short, strong pull from something that just happened (petted, spoken to)."""
        self.urge_b, self.urge_until = behaviour, self.t + seconds

    urge_b, urge_until = None, -1.0
    last_novelty = -1e9

    def utilities(self):
        d = self.d
        u = {
            "idle": 0.30,
            "explore": 0.8 * max(0.0, d.curiosity - 0.25) + 1.6 * max(0.0, d.boredom - 0.2) * (0.5 + 0.5 * d.energy),
            "approach": 0.7 * d.curiosity * (1 - d.fear) * (self.since_face < 20),
            "play": 0.6 * d.energy * max(d.mood, 0) * (1 - d.social),
            "seek_attention": 0.8 * max(0.0, d.social - 0.35) * d.energy + 0.1 * (d.social > 0.7),
            "rest": 1.0 * (1 - d.energy) ** 2,
            "hide": 0.9 * d.fear * (d.fear > 0.35),
            "startle": 1.5 * (d.fear > 0.75),
        }
        if self.bias and self.t < self.bias_until:
            u[self.bias] = u.get(self.bias, 0) + 0.35
        if self.urge_b and self.t < self.urge_until:
            u[self.urge_b] = u.get(self.urge_b, 0) + 0.6
        u[self.behaviour] += 0.1                          # hysteresis: don't flip-flop
        return u

    since_face = 1e9

    def tick(self, dt, events=()):
        self.t += dt
        self.since += dt
        self.since_face += dt
        for ev in events:
            if ev.kind == "face":
                self.since_face = 0.0
            self.feel(ev)
        d = self.d
        active = self.behaviour in ("explore", "approach", "play", "seek_attention", "startle")
        d.boredom += dt * (0.012 if self.behaviour in ("idle", "rest") else -0.030)
        d.curiosity -= dt * (0.030 if self.behaviour in ("explore", "approach") else 0.010)
        d.social += dt * 0.005
        d.fear -= dt * 0.06
        d.energy += dt * (0.012 if self.behaviour == "rest" else (-0.008 if active else -0.0015))
        d.mood += dt * (0.01 * (0.5 - d.social) - 0.02 * d.fear)
        d.clip()
        u = self.utilities()
        best = max(u, key=u.get)
        min_time = 0.0 if best == "startle" else 3.0      # finish a behaviour before switching
        if best != self.behaviour and (self.since > min_time or best in ("startle", "hide")):
            self.behaviour, self.since = best, 0.0
            self.log.append((self.t, "behaviour", best))
        return self.behaviour


class Mind:
    """Runs the slow model (Gemma 4 via Ollama, or a stand-in) in a background thread."""

    MAP = {"wave": "seek_attention", "happy": "play", "angry": "hide", "shy": "hide", "sleep": "rest", "idle": "idle"}

    def __init__(self, brain, period=30.0):
        self.brain, self.period = brain, period
        self.busy, self.last, self.result = False, -1e9, None
        self.lock = threading.Lock()

    def maybe_think(self, t, image_png, context: str):
        if self.busy or t - self.last < self.period:
            return
        self.busy, self.last = True, t

        def work():
            try:
                act = self.brain.decide(image_png, context)
                with self.lock:
                    self.result = act
            finally:
                self.busy = False
        threading.Thread(target=work, daemon=True).start()

    def take(self):
        with self.lock:
            r, self.result = self.result, None
        return r


class Reflexes:
    """Fast, small motions layered on top of whatever pose is commanded."""

    def __init__(self, joint_names):
        self.names = list(joint_names)
        self.idx = {n: i for i, n in enumerate(self.names)}

    def overlay(self, q, t, drives: Drives, behaviour: str):
        q = np.array(q, float)
        breath = 0.02 + 0.03 * (1 - drives.energy)                      # sleepy = deeper breathing
        for n in self.names:
            if n.endswith("_femur"):
                q[self.idx[n]] += breath * np.sin(2 * np.pi * t / 3.5)
        if behaviour in ("idle", "explore", "seek_attention"):          # curious claw fidgets
            for n in ("L_claw", "R_claw"):
                q[self.idx[n]] += 0.15 * max(0.0, np.sin(2 * np.pi * t / 2.3 + (n == "R_claw") * 1.7)) * drives.curiosity
        return q
