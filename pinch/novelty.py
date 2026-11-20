"""Unknown things and motion: what Pinch notices, how new and how risky each thing seems, and how that changes
with every look (the Investigate and Watch activities act on it: pinch/activities/curiosity.py).

Sensor-agnostic. Vision reports Observations - on the robot from the camera, in the twin from the simulated one
(pinch/twin/sensors.py):

    Observation(kind="unknown", xy=(0.12, 0.05), velocity=(0.0, 0.0), size=0.04, confidence=0.8, signature="box:40")

kind is "unknown" for a thing vision can't name (a known label works too, for its motion); xy on the desk (m);
velocity (m/s, None: estimated here from successive fixes); size across (m); confidence 0..1; signature: what it
looks like (on the robot an appearance fingerprint, in the twin its shape and size), so a thing taken away and
brought back is recognised and stays familiar.

The model (sources in the curiosity decision):
  * novelty N (1 = never seen) falls with every close look (x NOVELTY_PER_BOUT) and, slowly, with just seeing it;
    it is remembered per signature and recovers over days (habituation with spontaneous recovery);
  * wariness ("threat", Akiti et al. 2022: a novel thing starts with a threat prediction that safe encounters
    unlearn) starts at THREAT0 (from timidity), falls by a Rescorla-Wagner step after each safe look, rises when
    the thing startles it;
  * interest is a Wundt curve (Berlyne; Saunders & Gero's two sigmoids): rises with novelty, collapses when the
    thing is too threatening - then Pinch only watches it;
  * distance kept = D_NEAR + (D_FAR - D_NEAR) x threat: far at first, closer as it learns the thing is harmless;
  * a look is an approach-retreat excursion (Gordon, Fonio & Ahissar 2014): approach, inspect, back off, come
    back later (REVISIT_S x (1 + 2 threat)) until the thing is familiar (N < FAMILIAR_N);
  * motion: anything moving faster than MOVE_V draws the eyes; fast motion close by startles in proportion to
    how close, how fast and whether it comes toward Pinch.
Pure logic, no physics: unit-tested in tests/test_curiosity.py.
"""

import math
from dataclasses import dataclass, field

MATCH_R = 0.07              # m: a sighting this close to a tracked thing (same look) is that thing
GONE_IN_VIEW_S = 3.0        # in the camera's view but unseen this long: it was taken away
FORGET_S = 120.0            # out of view: object permanence for 2 minutes
FOV_HALF, FOV_RANGE = 0.9, 0.6   # the camera's useful field (rad either side, m): as the twin's perception
MOVE_V = 0.03               # m/s: moving (the camera's position noise is ~5 mm per fix at 10 Hz)
VEL_SMOOTH = 0.5
SUDDEN_V, SUDDEN_V_FULL = 0.10, 0.35     # m/s: startle starts / is full
STARTLE_R, STARTLE_R_FULL = 0.25, 0.12   # m: startle starts / is full (Pinch's centre to the thing)
NOVELTY_PER_BOUT = 0.55     # novelty kept after a safe close look: familiar after ~3-4 looks
NOVELTY_AFAR = 0.8          # ... after a look from a distance only
NOVELTY_VIEW_HALF_S = 240.0  # just having it in view habituates slowly
FAMILIAR_N = 0.2
THREAT_SAFE_ALPHA = 0.4     # Rescorla-Wagner steps toward 0 (safe look) / toward 1 (scare)
THREAT_BAD_ALPHA = 0.5
D_NEAR, D_FAR = 0.09, 0.25  # m: standoff (centre to the thing's edge) when trusted / when wary
REVISIT_S = 15.0
FAMILIAR_HALF_DAYS = 3.0    # remembered familiarity halves every 3 days


@dataclass
class Observation:
    kind: str = "unknown"
    xy: tuple = (0.0, 0.0)
    velocity: tuple = None
    size: float = 0.0
    confidence: float = 1.0
    signature: str = ""


@dataclass
class Thing:
    id: int
    kind: str
    xy: tuple
    size: float
    signature: str
    born_t: float
    seen_t: float
    confidence: float = 1.0
    vel: tuple = (0.0, 0.0)
    novelty: float = 1.0
    threat: float = 0.6
    bouts: int = 0
    views: list = field(default_factory=list)      # the angles (from the thing, rad) it has looked at it from
    closest: list = field(default_factory=list)    # closest approach (m, centre to centre) in each look
    next_visit_t: float = -1e9
    startled_t: float = -1e9
    startle: float = 0.0                           # how startling it is right now (decays over 3 s)

    @property
    def speed(self):
        return math.hypot(*self.vel)

    @property
    def familiar(self):
        return self.novelty < FAMILIAR_N

    @property
    def key(self):
        return f"thing:{self.id}"


def _sig(x, k=0.05):
    return 1.0 / (1.0 + math.exp(-x / k))


def interest(novelty, threat):
    """Wundt curve: positive feedback as novelty rises, negative feedback when the thing is too threatening."""
    return _sig(novelty - FAMILIAR_N) * (1.0 - _sig(threat - 0.85))


def in_view(xy, pose):
    x, y, h = pose
    d = math.hypot(xy[0] - x, xy[1] - y)
    b = (math.atan2(xy[1] - y, xy[0] - x) - h + math.pi) % (2 * math.pi) - math.pi
    return abs(b) < 0.8 * FOV_HALF and 0.08 < d < 0.9 * FOV_RANGE


class NoveltyTracker:
    def __init__(self, familiar=None, timidity=0.5):
        self.things = {}
        self.familiar = familiar if familiar is not None else {}     # signature -> {n, threat, t} (memory)
        self.threat0 = min(0.9, max(0.1, 0.3 + 0.6 * timidity))
        self._next = 1
        self.now = 0.0

    # -- sightings ---------------------------------------------------------------------------
    def _match(self, obs):
        best, bd = None, 1e9
        for th in self.things.values():
            if th.kind != obs.kind or (obs.signature and th.signature and obs.signature != th.signature):
                continue
            d = math.hypot(obs.xy[0] - th.xy[0], obs.xy[1] - th.xy[1])
            if d < MATCH_R + 0.3 * th.speed and d < bd:
                best, bd = th, d
        return best

    def _new(self, obs, t):
        th = Thing(self._next, obs.kind, tuple(obs.xy), float(obs.size), obs.signature, t, t, obs.confidence,
                   tuple(obs.velocity) if obs.velocity is not None else (0.0, 0.0), threat=self.threat0)
        self._next += 1
        if obs.kind != "unknown":                                 # a thing it knows by name: nothing new
            th.novelty, th.threat = 0.0, 0.0
        rec = self.familiar.get(obs.signature) if obs.signature else None
        if rec:
            k = 0.5 ** (max(0.0, self.now - rec["t"]) / 86400.0 / FAMILIAR_HALF_DAYS)
            th.novelty = 1.0 - (1.0 - rec["n"]) * k
            th.threat = self.threat0 - (self.threat0 - rec["threat"]) * k
        self.things[th.id] = th
        return th

    def observe(self, obs, t, pose, now=None):
        """One sighting. Returns events: ("new", thing), ("moving", thing, speed), ("sudden", thing, strength)."""
        if now is not None:
            self.now = now
        self.prune(t, pose)
        events = []
        th = self._match(obs)
        if th is None:
            th = self._new(obs, t)
            events.append(("new", th))
        else:
            dt = t - th.seen_t
            if obs.velocity is not None:
                v = tuple(obs.velocity)
            elif dt > 1e-3:
                v = ((obs.xy[0] - th.xy[0]) / dt, (obs.xy[1] - th.xy[1]) / dt)
            else:
                v = th.vel
            th.vel = tuple(VEL_SMOOTH * a + (1 - VEL_SMOOTH) * b for a, b in zip(v, th.vel))
            if th.speed < MOVE_V:                                 # still: average the fixes
                th.xy = (0.7 * th.xy[0] + 0.3 * obs.xy[0], 0.7 * th.xy[1] + 0.3 * obs.xy[1])
            else:
                th.xy = tuple(obs.xy)
            th.size = 0.8 * th.size + 0.2 * obs.size
            th.confidence = 0.7 * th.confidence + 0.3 * obs.confidence
            th.novelty *= 0.5 ** (min(dt, 1.0) / NOVELTY_VIEW_HALF_S)
            th.seen_t = t
        sp = th.speed
        if sp > MOVE_V:
            events.append(("moving", th, sp))
        x, y = pose[0], pose[1]
        dist = math.hypot(th.xy[0] - x, th.xy[1] - y)
        th.startle = max(0.0, th.startle - (t - th.startled_t) / 3.0) if th.startle else 0.0
        if sp > SUDDEN_V and dist < STARTLE_R:
            prox = min(1.0, (STARTLE_R - dist) / (STARTLE_R - STARTLE_R_FULL))
            inten = min(1.0, (sp - SUDDEN_V) / (SUDDEN_V_FULL - SUDDEN_V))
            toward = -((th.xy[0] - x) * th.vel[0] + (th.xy[1] - y) * th.vel[1]) / max(1e-6, dist * sp)
            s = prox * inten * (0.5 + 0.5 * max(0.0, toward))        # closer, faster, coming at it: worse
            if s > th.startle + 0.05:                                # (one scare per approach, as big as its peak:
                ds, th.startle = s - th.startle, s                   #  only the growth is reported)
                th.threat += THREAT_BAD_ALPHA * ds * (1.0 - th.threat)
                th.next_visit_t = max(th.next_visit_t, t + REVISIT_S)
                events.append(("sudden", th, ds))
        th.startled_t = t
        return events

    def prune(self, t, pose):
        for i, th in list(self.things.items()):
            age = t - th.seen_t
            if age > FORGET_S or (age > GONE_IN_VIEW_S and in_view(th.xy, pose)):
                self._remember(th)
                del self.things[i]

    # -- what the activities ask -------------------------------------------------------------
    def get(self, i):
        return self.things.get(i)

    def moving(self, t):
        """The most eye-catching thing moving right now, or None."""
        c = [th for th in self.things.values() if th.speed > MOVE_V and t - th.seen_t < 0.6]
        return max(c, key=lambda th: th.speed * th.confidence, default=None)

    def most_interesting(self, t, pose):
        """(thing, interest) of the best unfamiliar, still thing to go and look at now, or (None, 0)."""
        best, bs = None, 0.0
        for th in self.things.values():
            if th.familiar or t < th.next_visit_t or th.speed > MOVE_V or th.kind != "unknown":
                continue
            v = interest(th.novelty, th.threat) * th.confidence
            v /= 1.0 + math.hypot(th.xy[0] - pose[0], th.xy[1] - pose[1])
            if v > bs:
                best, bs = th, v
        return best, (interest(best.novelty, best.threat) if best else 0.0)

    def standoff(self, th):
        """How far (centre to centre, m) to stay from it now: far while wary."""
        return D_NEAR + (D_FAR - D_NEAR) * th.threat + th.size / 2

    def unknown(self, t, within=30.0):
        return [th for th in self.things.values() if th.kind == "unknown" and t - th.seen_t < within]

    # -- learning from a look ----------------------------------------------------------------
    def end_bout(self, th, closest, views, safe, t, close=True):
        """A look (an approach-retreat excursion) is over. Returns how much novelty it took away."""
        old = th.novelty
        th.bouts += 1
        th.closest.append(round(float(closest), 3))
        th.views += [round(float(v), 3) for v in views]
        if safe:
            th.novelty *= NOVELTY_PER_BOUT if close else NOVELTY_AFAR
            th.threat -= THREAT_SAFE_ALPHA * th.threat
        else:
            th.novelty *= 0.85
            th.threat += THREAT_BAD_ALPHA * 0.4 * (1.0 - th.threat)
        th.next_visit_t = t + REVISIT_S * (1.0 + 2.0 * th.threat)
        self._remember(th)
        return old - th.novelty

    def _remember(self, th):
        if th.kind == "unknown" and th.signature:
            self.familiar[th.signature] = dict(n=round(th.novelty, 3), threat=round(th.threat, 3), t=self.now)
