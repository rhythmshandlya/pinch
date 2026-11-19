"""What Pinch is paying attention to (decision 72).

Stimuli - a voice and the way it came from, a touch, a person appearing, a loud sound, a thing appearing -
each get a salience: a base strength by kind x how strong this one is x novelty. Then:
  * salience fades over a few seconds (a stimulus is an event, not a state)
  * habituation: the same kind from the same place again and again counts for less, recovering over a minute
  * inhibition of return: what it has just attended to is damped for a while, so attention moves on
  * hysteresis: the focus only switches to something clearly more salient (no flicker)
The mind turns the focus into gaze (whenever no activity is aiming the eyes) and, when it is strong and off
to one side, into the Orient activity (turn toward it, look, then go back to what it was doing).
Pure logic, no physics: unit-tested in tests/test_attention.py.
"""

import math
from dataclasses import dataclass, field

BASE = {"touch": 0.9, "loud": 0.85, "voice": 0.65, "person": 0.7, "new_thing": 0.5, "motion": 0.4}
FADE_S = 3.0                # salience half-life
HABITUATE_HALF_LIFE = 60.0  # how fast familiarity with a repeated stimulus wears off
IOR_S, IOR_FACTOR = 20.0, 0.35
SWITCH_MARGIN = 1.25


@dataclass
class Stimulus:
    kind: str
    t: float
    strength: float = 1.0
    direction: float = None     # radians in Pinch's body frame (+ = left), if known
    xy: tuple = None            # where on the desk, if known
    key: str = ""               # identity for habituation ("voice:left", "touch:head", "person:sam")
    salience0: float = 0.0


@dataclass
class Attention:
    stimuli: list = field(default_factory=list)
    focus: Stimulus = None
    familiarity: dict = field(default_factory=dict)     # key -> (level, t)
    inhibited: dict = field(default_factory=dict)       # key -> until t

    def _familiar(self, key, t):
        lvl, t0 = self.familiarity.get(key, (0.0, t))
        return lvl * 0.5 ** ((t - t0) / HABITUATE_HALF_LIFE)

    def notice(self, kind, t, strength=1.0, direction=None, xy=None, key=None):
        """A stimulus arrived. Returns its starting salience."""
        if key is None:
            side = "" if direction is None else ("left" if direction > 0.4 else "right" if direction < -0.4 else "front")
            key = f"{kind}:{side}"
        fam = self._familiar(key, t)
        self.familiarity[key] = (fam + 1.0, t)
        sal = BASE.get(kind, 0.4) * min(1.0, max(0.0, strength)) / (1.0 + 0.6 * fam)
        s = Stimulus(kind, t, strength, direction, xy, key, sal)
        self.stimuli = [x for x in self.stimuli if x.key != key] + [s]
        return sal

    def salience(self, s, t):
        v = s.salience0 * 0.5 ** ((t - s.t) / FADE_S)
        if self.inhibited.get(s.key, -1.0) > t:
            v *= IOR_FACTOR
        return v

    def update(self, t):
        """The focus now (or None) and its salience."""
        self.stimuli = [s for s in self.stimuli if self.salience(s, t) > 0.02]
        if not self.stimuli:
            self.focus = None
            return None, 0.0
        best = max(self.stimuli, key=lambda s: self.salience(s, t))
        cur = self.focus if self.focus in self.stimuli else None
        if cur is None or self.salience(best, t) > SWITCH_MARGIN * self.salience(cur, t):
            self.focus = best
        return self.focus, self.salience(self.focus, t)

    def attended(self, s, t):
        """It looked at s: move on (inhibition of return)."""
        self.inhibited[s.key] = t + IOR_S
        if self.focus is s:
            self.focus = None


def direction_of(xy, pose):
    """Body-frame direction (rad, + left) of a desk point from Pinch's pose (x, y, heading)."""
    x, y, h = pose
    a = math.atan2(xy[1] - y, xy[0] - x) - h
    return (a + math.pi) % (2 * math.pi) - math.pi
