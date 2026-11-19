"""Self-preservation: what Pinch learns to fear, and how hungry its battery makes it (decision 44).

Danger map. Spots on the desk where something bad happened (a foot went over the edge, it got stuck,
it fell) are remembered with a fear strength v, learned by the Rescorla-Wagner rule v += a (lambda - v):
one near-fall makes a spot worrying, two make it a no-go. Fear fades (v halves every 8 awake hours) and
is unlearned faster by passing a spot safely (a "safety" strength s grows on safe passes; the effective
danger is v - s, and s itself fades every hour, so a fear that was only overlaid can come back - extinction
is new learning on top, not erasure: Bouton 2004). A cap stops it becoming phobic about the whole desk.

Battery hunger. De = (1 - charge)^2: barely noticed at 60 %, pressing at 25 %, urgent at 10 %
(homeostatic drives grow faster than the deficit: Keramati & Gutkin 2014). Play and exploring are damped
by (1 - De); HeadHome and LowPower in activities.py act on it.
"""

import math

SIGMA = 0.05                 # m: how far a remembered spot's danger spreads
ALPHA = 0.6                  # learning rate for bad outcomes
LAMBDA = {"near_fall": 0.8, "stuck": 0.5, "fell": 1.0, "cliff": 0.4}
V_HALF_H = 8.0               # fear halves every 8 awake hours
S_HALF_H = 1.0               # the safety learned on top fades faster: fear partly comes back
SAFE_GAIN = 0.08             # each safe pass through a spot unlearns a little
MAX_SPOTS = 40
NO_GO, CAREFUL = 0.5, 0.15   # danger levels: never go there / walk carefully


class DangerMap:
    def __init__(self, data=None):
        self.spots = [dict(s) for s in (data or [])]             # {x, y, v, s, what}

    def danger(self, xy):
        d = 0.0
        for s in self.spots:
            r2 = (xy[0] - s["x"]) ** 2 + (xy[1] - s["y"]) ** 2
            d += max(0.0, s["v"] - s["s"]) * math.exp(-r2 / (2 * SIGMA ** 2))
        return min(1.0, d)

    def learn(self, xy, what):
        """Something bad happened here."""
        lam = LAMBDA.get(what, 0.5)
        near = min(self.spots, key=lambda s: (xy[0] - s["x"]) ** 2 + (xy[1] - s["y"]) ** 2, default=None)
        if near is not None and math.hypot(xy[0] - near["x"], xy[1] - near["y"]) < SIGMA:
            spot = near
        else:
            spot = dict(x=float(xy[0]), y=float(xy[1]), v=0.0, s=0.0, what=what)
            self.spots.append(spot)
        spot["v"] += ALPHA * (lam - (spot["v"] - spot["s"]))
        spot["v"] = min(1.0, spot["v"])
        self.spots.sort(key=lambda s: -(s["v"] - s["s"]))
        del self.spots[MAX_SPOTS:]
        return spot

    def safe_pass(self, xy):
        """Walked through a feared spot and nothing happened."""
        for s in self.spots:
            if math.hypot(xy[0] - s["x"], xy[1] - s["y"]) < SIGMA:
                s["s"] += SAFE_GAIN * max(0.0, s["v"] - s["s"])

    def decay(self, awake_hours):
        kv, ks = 0.5 ** (awake_hours / V_HALF_H), 0.5 ** (awake_hours / S_HALF_H)
        for s in self.spots:
            s["v"] *= kv
            s["s"] *= ks
        self.spots = [s for s in self.spots if s["v"] > 0.02]

    def cap(self, desk, grid=0.03):
        """No phobia of the whole desk: if over 25 % of it is no-go-ish, scale every fear down."""
        cells = [(x * grid, y * grid) for x in range(int(-desk.half_x / grid), int(desk.half_x / grid) + 1)
                 for y in range(int(-desk.half_y / grid), int(desk.half_y / grid) + 1)]
        bad = sum(self.danger(c) >= 0.35 for c in cells) / max(1, len(cells))
        if bad > 0.25:
            for s in self.spots:
                s["v"] *= 0.8

    def to_list(self):
        return [dict(s) for s in self.spots]


def hunger(battery):
    """Battery hunger De in [0, 1]: (1 - charge)^2."""
    return max(0.0, min(1.0, 1.0 - float(battery))) ** 2
