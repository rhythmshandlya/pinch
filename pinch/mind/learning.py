"""It grows with its person: everyday feedback slowly shapes what it chooses to do (phase 4, part 3).

Praise, petting and tickles count as a reward and scolding as a punishment for what it is doing now. Something it
finished a moment ago also gets some credit, fading with a 3 s trace over 6 s (people praise after they see the
result; human feedback comes 0.2 - 4 s late: Knox & Stone's TAMER, Deep TAMER, COACH;
docs/research/v2_1/learning/notes.md). The update is a gradient bandit with a baseline (Sutton & Barto ch. 2): the
person's usual feedback is subtracted, so an owner who praises everything teaches nothing in particular.

What it learns is a small bias per person, coarse context (part of day x person present) and activity, added to
the activity's own utility. Bounds keep the personality recognisable: the bias is capped (configs/mind/learning.toml
max_bias), an activity never falls below floor of its own want, a want of zero stays zero (it learns *when* and
*how much*, not new wants), safety, needs and reflexes are never learned (Downie: crediting reflexes backfires),
and it fades with a 14-day half-life without feedback. With no feedback the bias is zero.
"""

import math
import zlib

from pinch.settings import load

CFG = load("learning")
REWARD = CFG["reward"]                       # event -> reward
LEARNABLE = set(CFG["learnable"])


def own_block(activity, clock):
    """Is this a stretch where it does this activity purely as itself? own_share of the time, in own_block_s blocks
    (fixed per block and activity: replays exactly), so nothing it was scolded for ever disappears."""
    block = int(clock // CFG["own_block_s"])
    z = (block * 0x9E3779B1 + zlib.crc32(activity.encode())) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    z = ((z ^ (z >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return ((z ^ (z >> 16)) / 2.0 ** 32) < CFG["own_share"]


class FeedbackLearner:
    def __init__(self, data=None):
        data = data or {}
        self.w = {p: {c: dict(a) for c, a in ctx.items()} for p, ctx in data.get("w", {}).items()}
        self.base = dict(data.get("base", {}))          # each person's usual feedback (the baseline)
        self.t_decay = data.get("t_decay")              # clock time the weights were last faded
        self.recent = []                                # (t, activity) of activities that just ended
        self.events = []                                # (t) of recent feedback (burst cap)
        self.repeats = {}                               # activity -> [clock t of feedback] (repeat discount)
        self.log = []                                   # (t, event, person, {activity: delta}) for the diary, tests

    def to_dict(self):
        return dict(w=self.w, base=self.base, t_decay=self.t_decay)

    # -- what it is doing ------------------------------------------------------------------------------------------
    def ended(self, t, activity):
        """An activity just ended: it can still be credited for a few seconds."""
        if activity in LEARNABLE:
            self.recent = [(t0, a) for t0, a in self.recent if t - t0 <= CFG["window_s"]] + [(t, activity)]

    def credit(self, t, current):
        """{activity: credit} for feedback given now."""
        c = {}
        for t0, a in self.recent:
            dt = t - t0
            if 0 <= dt <= CFG["window_s"]:
                c[a] = max(c.get(a, 0.0), math.exp(-dt / CFG["trace_s"]))
        if current in LEARNABLE:
            c[current] = 1.0
        return c

    # -- feedback --------------------------------------------------------------------------------------------------
    def feedback(self, event, t, clock, current, ctx, person="owner"):
        """A reward event from a person. Returns {activity: change} (empty if nothing was credited)."""
        r = REWARD.get(event)
        if r is None:
            return {}
        self.events = [x for x in self.events if t - x < CFG["burst_s"]] + [t]
        if len(self.events) > CFG["burst_max"]:          # a flurry of pats is one message, not five
            return {}
        self._fade(clock)
        b = self.base.get(person, 0.0)
        self.base[person] = b + CFG["baseline_rate"] * (r - b)
        adv = r - b
        out = {}
        for a, k in self.credit(t, current).items():
            reps = [x for x in self.repeats.get(a, []) if clock - x < CFG["repeat_s"]]
            self.repeats[a] = reps + [clock]
            step = CFG["rate"] * adv * k * CFG["repeat_discount"] ** len(reps)
            for c_key, scale in ((ctx, 1.0), ("*", CFG["global_share"])):   # this context, and all (back-off)
                row = self.w.setdefault(person, {}).setdefault(c_key, {})
                row[a] = max(-CFG["max_bias"], min(CFG["max_bias"], row.get(a, 0.0) + scale * step))
            out[a] = round(step, 4)
        self.log.append((round(t, 1), event, person, out))
        self.log = self.log[-200:]
        return out

    def _fade(self, clock):
        """Without feedback, what it learned fades back toward its own personality (14-day half-life)."""
        if self.t_decay is not None and clock > self.t_decay:
            k = 0.5 ** ((clock - self.t_decay) / (CFG["half_life_days"] * 86400.0))
            for ctxs in self.w.values():
                for row in ctxs.values():
                    for a in row:
                        row[a] *= k
        self.t_decay = clock

    # -- choosing --------------------------------------------------------------------------------------------------
    def bias(self, activity, ctx, person="owner"):
        if activity not in LEARNABLE:
            return 0.0
        p = self.w.get(person, {})
        b = p.get(ctx, {}).get(activity, 0.0) + p.get("*", {}).get(activity, 0.0)
        return max(-CFG["max_bias"], min(CFG["max_bias"], b))

    def shape(self, activity, u, ctx, person="owner"):
        """The utility with what it has learned: never a want from nothing, never below floor of its own."""
        if u <= 0 or activity not in LEARNABLE:
            return u
        return max(CFG["floor"] * u, u + self.bias(activity, ctx, person))
