"""Learning bench (phase 4, part 3): does its person's feedback shape what it does over days, within bounds?

    python sim/benches/learning.py [--days 10]

The mind alone (no physics: a stand-in body takes a few seconds per motion), two hours with its person each
simulated day, the desk things seen where they lie. Ten days with a teacher who praises exploring ("good crab" a
few seconds into it, 60 % of the time) and scolds playing; then three quiet days (no feedback) for the trained
crab and for an untrained one, same seeds - so only what it learned differs (the praise's own mood effects are
gone). Five seeds.
Pass: trained, it explores >= 25 % more and plays >= 25 % less (research target: a visible change within days,
docs/research/v2_1/learning/notes.md), and still plays >= 30 % as much (bounded: its personality stays).
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402

DT, AWAKE_S = 0.5, 7200.0
T0 = 1.8e9                        # 13:30 local at UTC+5.5 (an afternoon with its person)
MOTION_S = {"goto": 4.0, "push": 6.0, "pick": 5.0, "drop": 4.0, "emote": 2.5, "walk": 2.0, "turn_to": 1.5,
            "pose": 2.0}         # how long each kind of motion takes (a stand-in body; the twin's are similar)


def run(owner, days, seed=0, learned=None, day0=0):
    """Minutes in each activity per simulated day with this owner."""
    clock = {"t": T0}
    m = Mind(memory=Memory(path=None), llm=False, seed=seed, clock=lambda: clock["t"])
    if learned is not None:                                            # what an earlier stretch taught it
        from pinch.mind.learning import FeedbackLearner
        m.learner = FeedbackLearner(learned)
    m.memory.data["homes"] = {}
    m.routine.settings["utc_offset_h"] = 5.5
    for name, xy in (("pen", (0.10, 0.12)), ("eraser", (-0.08, 0.10)), ("cap", (0.12, -0.10))):
        m.world.see(name, xy)
    rng = np.random.default_rng(seed + 100)
    per_day = []
    for day in range(day0, day0 + days):
        clock["t"] = T0 + day * 86400.0 + 137.0 * seed                 # (seeds don't share clock times)
        m.affect.needs.sleep, m.affect.needs.fatigue = 0.0, 0.0          # (it slept overnight)
        start_t, starts, given, busy_until = m.t, Counter(), None, -1.0
        while m.t - start_t < AWAKE_S:
            o = m.world.owner
            o.present, o.last_attention, o.last_seen = True, m.t, m.t
            step, cancel = m.tick(DT, body_busy=m.t < busy_until, body_ok=True)
            if cancel:
                busy_until = -1.0
            if step is not None:
                busy_until = m.t + MOTION_S.get(step[0], 2.0)
            clock["t"] += DT
            a = m.activity.name if m.activity else "idle"
            key = (a, m.started)
            starts[a] += DT / 60.0                                 # minutes in each activity
            if owner == "teacher" and m.t - m.started > 3.0 and given != key:
                given = key
                if a == "explore" and rng.random() < 0.6:
                    m.hear_text("good crab")
                elif a == "play":
                    m.hear_text("bad crab")
            elif owner == "random" and rng.random() < DT / 90.0:   # about the teacher's rate, at random moments
                m.hear_text("good crab" if rng.random() < 0.5 else "bad crab")
        per_day.append(starts)
    return per_day, m


SEEDS = (0, 1, 2, 3, 4)
KINDS = ("explore", "play")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    trained, untrained = {k: 0.0 for k in KINDS}, {k: 0.0 for k in KINDS}
    shift = {}
    for seed in SEEDS:
        _, m = run("teacher", a.days, seed)                    # ten days with the teacher
        lw = m.learner.to_dict()
        for _what, learned, acc in (("trained", lw, trained), ("untrained", None, untrained)):
            days, _ = run("silent", 3, seed + 50, learned=learned, day0=a.days)   # then three quiet days, same seed
            for d in days:
                for k in KINDS:
                    acc[k] += d.get(k, 0.0) / (3 * len(SEEDS))
        shift[seed] = (round(m.learner.bias("explore", "afternoon:here"), 3), round(m.learner.bias("play", "afternoon:here"), 3))
    print("learned (explore, play) per seed:", shift)
    for k in KINDS:
        print(f"{k:8s} untrained {untrained[k]:5.2f} min/day   trained {trained[k]:5.2f} min/day")

    def rel(k):
        return (trained[k] - untrained[k]) / max(1e-6, untrained[k])
    checks = [
        ("trained: explores more (>= +25 %)", rel("explore") >= 0.25),
        ("trained: plays less (<= -25 %)", rel("play") <= -0.25),
        ("trained: still plays (>= 30 % of untrained)", rel("play") >= -0.7),
    ]
    ok = True
    for name, good in checks:
        ok &= bool(good)
        print(f"{'PASS' if good else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PINCH_QUIET", "1")
    sys.exit(main())
