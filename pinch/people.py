"""People Pinch knows, and how it feels about each of them (decision 46).

Each person has three numbers, learned from how they treat Pinch:
- familiarity F (0..1): grows while they are around (a visitor becomes familiar in ~3 days of company),
  fades with a 90-day half-life when they are away
- warmth W (-1..1): petting, play and praise raise it; scolding and rough handling lower it; it drifts
  back toward 0.2F with a 30-day half-life
- fear R (0..1): rough handling (shaken, dropped, held up by someone it barely knows) raises it; it
  fades with a 3-day half-life, twice as fast while someone it trusts is near
Who did something matters: each update is weighted by how sure Pinch is that this person did it (the only
or the nearest person seen in the last 10 s). The model is small and readable on purpose (attitude
models in social robots: Kirby et al. 2010, TAME; dogs greet harder after longer absences: Rehn 2011).

Recognition: on the robot, faces from the camera (YuNet + SFace, ~40 MB of model files: not downloaded
yet, the owner decides) give each sighting a person id and a confidence. In the twin and the app the
identities are simulated. Only numbers are kept - never photos - and "forget me" deletes a person.
"""

import math

HALF_F, HALF_W, HALF_R = 90 * 24 * 3600.0, 30 * 24 * 3600.0, 3 * 24 * 3600.0
EFFECTS = {   # event -> (dW, dR) as functions of (W, R)
    "petted":  (lambda W, R: 0.05 * (1 - W), lambda W, R: -0.05),
    "stroked": (lambda W, R: 0.05 * (1 - W), lambda W, R: -0.05),
    "tickled": (lambda W, R: 0.03 * (1 - W), lambda W, R: -0.02),
    "praised": (lambda W, R: 0.03 * (1 - W), lambda W, R: -0.02),
    "played_with": (lambda W, R: 0.03 * (1 - W), lambda W, R: -0.02),
    "fed":     (lambda W, R: 0.05 * (1 - W), lambda W, R: 0.0),
    "scolded": (lambda W, R: -0.06 * (1 + W), lambda W, R: 0.05 * (1 - R)),
    "shaken":  (lambda W, R: -0.10 * (1 + W), lambda W, R: 0.30 * (1 - R)),
    "fell":    (lambda W, R: -0.10 * (1 + W), lambda W, R: 0.30 * (1 - R)),
}


class People:
    def __init__(self, data=None):
        self.people = {k: dict(v) for k, v in (data or {}).items()}
        self.seen = {}                    # id -> (time last seen, confidence) this session

    def to_dict(self):
        return self.people

    def get(self, pid):
        return self.people.get(pid)

    def ensure(self, pid, name="", owner=False, now=0.0, introduced=False):
        p = self.people.get(pid)
        if p is None:
            p = dict(name=name, owner=owner, F=0.3 if introduced or owner else 0.0,
                     W=0.2 if introduced or owner else 0.0, R=0.0, last_seen=now, t_update=now)
            if owner:
                p["F"], p["W"] = 0.6, 0.5
            self.people[pid] = p
        if name:
            p["name"] = name
        return p

    def recent(self, now, within=10.0):
        """Ids of the people seen in the last `within` seconds."""
        return [pid for pid, p in self.people.items() if now - p.get("last_seen", -1e18) <= within]

    def see(self, pid, confidence, now, name="", owner=False):
        """A sighting: returns how many hours they were away (for the greeting)."""
        p = self.ensure(pid, name=name, owner=owner, now=now)
        away_h = (now - p.get("last_seen", now)) / 3600.0
        self._fade(p, now)
        p["F"] = min(1.0, p["F"] + (1 - p["F"]) * confidence * min(60.0, now - p["last_seen"]) / (3 * 24 * 3600 / 8))
        p["last_seen"] = now
        self.seen[pid] = (now, confidence)
        return away_h

    def present(self, now, within=10.0):
        return {pid: c for pid, (t, c) in self.seen.items() if now - t < within}

    def _fade(self, p, now):
        dt = max(0.0, now - p.get("t_update", now))
        p["t_update"] = now
        trusted_near = any(self.people[o]["W"] > 0.5 for o in self.present(now) if o in self.people)
        p["F"] *= 0.5 ** (dt / HALF_F)
        p["W"] = 0.2 * p["F"] + (p["W"] - 0.2 * p["F"]) * 0.5 ** (dt / HALF_W)
        p["R"] *= 0.5 ** (dt * (2.0 if trusted_near else 1.0) / HALF_R)

    def blame(self, event, now):
        """Credit or blame the person most likely behind an event (the only or nearest one seen in 10 s)."""
        here = self.present(now)
        if not here or event not in EFFECTS:
            return None
        # the most recently seen person (who just came up to Pinch), then the surest sighting
        pid = max(here, key=lambda k: (self.seen[k][0], here[k]))
        conf = here[pid] * (1.0 if len(here) == 1 else 0.6)
        p = self.people[pid]
        self._fade(p, now)
        dw, dr = EFFECTS[event]
        p["W"] = max(-1.0, min(1.0, p["W"] + conf * dw(p["W"], p["R"])))
        p["R"] = max(0.0, min(1.0, p["R"] + conf * dr(p["W"], p["R"])))
        return pid

    def lifted_by_stranger(self, now):
        here = self.present(now)
        if not here:
            return
        pid = max(here, key=here.get)
        p = self.people[pid]
        if p["F"] < 0.3:
            p["R"] = min(1.0, p["R"] + 0.10 * (1 - p["R"]))

    def forget(self, pid):
        self.people.pop(pid, None)
        self.seen.pop(pid, None)

    def favourite(self):
        good = [(p["W"] * p["F"], pid) for pid, p in self.people.items() if p["W"] > 0.5]
        return max(good)[1] if good else None

    def feared_here(self, now):
        here = self.present(now)
        return [pid for pid in here if self.people.get(pid, {}).get("R", 0) > 0.45]

    def strangers_here(self, now):
        here = self.present(now)
        return [pid for pid in here if self.people.get(pid, {}).get("F", 0) < 0.3 and not self.people[pid].get("owner")]

    def describe(self, pid):
        p = self.people[pid]
        return f"{p['name'] or pid}: knows {p['F']:.2f}, likes {p['W']:+.2f}, fears {p['R']:.2f}"
