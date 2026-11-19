"""Self-directed play (decision 45): Pinch invents games from its own skills and the toys on its desk, and
finds out by itself which are fun.

A game is verb x toy x target, built only from skills Pinch already has: push it to a spot, carry it there
(things the claw holds), or pick it up and show it off. Each game keeps a small record: how often it was
played, its wins, how well Pinch can predict where the toy ends up (prediction error), boredom, and your
praise. Choosing follows intrinsic-motivation research:
- learning progress (Oudeyer & Kaplan's curiosity, the AIBO "Playground" experiments): a game whose
  outcomes Pinch is getting better at predicting is interesting; one it has mastered stops being
- novelty: a 1/sqrt(1+n) bonus, for the game and for the toy (a new thing on the desk draws it in)
- competence in the sweet spot: 4p(1-p) peaks when it wins about half the time
- your praise (memory likes "game:<id>"), and boredom that builds with each play and fades in ~20 minutes
15 % of the time it tries a random allowed game. New *motor* skills are not invented here: they need
training (training/).

Hard safety rules, in code: the toy at least 10 cm from every edge and not in a feared spot, the target at
least 8 cm from every edge, and a game whose toy ends up near an edge is banned for a week.
"""

import math

VERBS = ("push_to", "carry_to", "show")
TARGETS = ("tray", "middle", "owner")
CARRYABLE = {"pen", "eraser"}             # what the claw can hold (decision 24)
EDGE_TOY, EDGE_GOAL, EDGE_BAN = 0.10, 0.08, 0.06
BAN_S = 7 * 24 * 3600.0


def game_id(verb, toy, target):
    return f"{verb}:{toy}:{target}"


class PlayBook:
    def __init__(self, data=None):
        data = data or {}
        self.games = {k: dict(v) for k, v in data.get("games", {}).items()}
        self.toys = dict(data.get("toys", {}))

    def to_dict(self):
        return dict(games=self.games, toys=self.toys)

    # ------------------------------------------------------------------ what can be played right now
    def target_xy(self, target, world):
        d = world.desk
        if target == "tray":
            return d.tray_xy
        if target == "owner":
            return (0.0, d.owner_xy[1] + 0.29)
        return (0.0, 0.0)

    def options(self, world, danger, now):
        out = []
        for toy, ob in world.objects.items():
            if not ob.on_desk or world.desk.edge_distance(ob.xy) < EDGE_TOY or danger(ob.xy) >= 0.15:
                continue
            for verb in VERBS:
                if verb in ("carry_to", "show") and toy not in CARRYABLE:
                    continue
                for target in (TARGETS if verb != "show" else ("here",)):
                    if target != "here":
                        goal = self.target_xy(target, world)
                        if world.desk.edge_distance(goal) < EDGE_GOAL or danger(goal) >= 0.15:
                            continue
                    gid = game_id(verb, toy, target)
                    if self.games.get(gid, {}).get("banned_until", 0) > now:
                        continue
                    out.append((verb, toy, target))
        return out

    # ------------------------------------------------------------------ choosing
    def score(self, gid, toy, like, now):
        g = self.games.get(gid, {})
        n = g.get("n", 0)
        wins = g.get("wins", [])
        err = g.get("err", [])
        lp = 0.0
        if len(err) >= 4:                                     # learning progress: is the error falling?
            lp = max(0.0, (err[-4] + err[-3]) / 2 - (err[-2] + err[-1]) / 2)
        if len(wins) >= 4:                                    # ... or the win rate changing
            lp += abs((wins[-1] + wins[-2]) / 2 - (wins[-3] + wins[-4]) / 2)
        p = (sum(wins) + 1) / (len(wins) + 2)
        nov = 0.5 / math.sqrt(1 + n) + 0.5 / math.sqrt(1 + self.toys.get(toy, {}).get("n", 0))
        boredom = g.get("boredom", 0.0) * math.exp(-(now - g.get("last_t", now)) / 1200.0)
        return lp + 0.6 * nov + 0.4 * 4 * p * (1 - p) + 0.8 * like - 0.25 * boredom

    def choose(self, world, danger, likes, rng, now):
        opts = self.options(world, danger, now)
        if not opts:
            return None, 0.0
        if rng.random() < 0.15:
            pick = opts[int(rng.integers(len(opts)))]
            return pick, self.score(game_id(*pick), pick[1], likes.get("game:" + game_id(*pick), 0.0), now)
        scored = [(self.score(game_id(*o), o[1], likes.get("game:" + game_id(*o), 0.0), now) + 0.02 * rng.random(), o)
                  for o in opts]
        s, best = max(scored)
        return best, s

    # ------------------------------------------------------------------ learning from what happened
    def predict(self, gid):
        """How far Pinch expects the toy to move (its running mean)."""
        g = self.games.get(gid, {})
        return g.get("mu", 0.05)

    def record(self, verb, toy, target, moved, won, edge_after, now):
        gid = game_id(verb, toy, target)
        g = self.games.setdefault(gid, dict(n=0, wins=[], err=[], mu=0.05, boredom=0.0, last_t=now))
        err = abs(moved - g["mu"]) / max(0.01, g["mu"])
        g["mu"] += 0.3 * (moved - g["mu"])
        g["n"] += 1
        g["wins"] = (g["wins"] + [1 if won else 0])[-10:]
        g["err"] = (g["err"] + [min(3.0, err)])[-10:]
        fun = min(1.0, max(0.0, err / 3 + (0.5 if won else 0.0)))
        g["boredom"] = g.get("boredom", 0.0) * math.exp(-(now - g.get("last_t", now)) / 1200.0) + max(0.0, 1 - 1.5 * fun)
        g["last_t"] = now
        if edge_after < EDGE_BAN:                             # that game nearly sent a toy off the desk
            g["banned_until"] = now + BAN_S
        t = self.toys.setdefault(toy, dict(n=0))
        t["n"] += 1
        return gid, err, fun
