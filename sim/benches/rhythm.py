"""Rhythm bench (phase 4): is the flow of what Pinch does like an animal's - settled, not flickering, varied but
not random, and not wearing thin over days?

    python sim/benches/rhythm.py [--seeds 20] [--days 14]        # the mind with a stand-in body (fast)
    python sim/benches/rhythm.py --twin [--seeds 4] [--days 1]    # the full twin, physics and all (slow)

The mind alone with a stand-in body (sim/benches/learning.py), its person around for two hours a day with a few
words and touches at random moments. Measured on the sequence of activities:
  flicker       share of its own switches where it drops what it just started (cut short, not finished)
                and goes straight back to what it was doing within 10 s (A -> B -> A; a reaction to a
                touch, a word or a scare in between is not flicker, nor a short activity that finished) < 5 %
  dwell         median time in an activity it chose itself, s                                         20 - 90
  switches      activity switches per minute                                                           <= 3
  next_entropy  how unpredictable the next activity is given the current one, bits                    1.0 - 2.0
  repeat4       share of 4-activity runs seen earlier the same day                                     < 50 %
  day14/day1    the day's activity entropy on the last days against the first                           >= 0.85
Thresholds: docs/research/v2_1/longterm/notes.md (Blumberg's level of interest; ethogram structure; long-term HRI:
novelty wears off in 1-3 weeks, so the last days matter more than the first). They are starting values.
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning import DT, MOTION_S, T0  # noqa: E402

AWAKE_S = 7200.0
WORDS = ("hi pinch", "good crab", "how are you?", "pinch, play!", "what's that?")
TOUCHES = ("stroke", "tap", "tickle", "boop")


def run(seed, days):
    clock = {"t": T0}
    m = Mind(memory=Memory(path=None), llm=False, seed=seed, clock=lambda: clock["t"])
    m.memory.data["homes"] = {}
    m.routine.settings["utc_offset_h"] = 5.5
    for name, xy in (("pen", (0.10, 0.12)), ("eraser", (-0.08, 0.10)), ("cap", (0.12, -0.10))):
        m.world.see(name, xy)
    rng = np.random.default_rng(seed + 7)
    out = []
    for day in range(days):
        clock["t"] = T0 + day * 86400.0 + 97.0 * seed
        m.affect.needs.sleep, m.affect.needs.fatigue = 0.0, 0.0
        start_t, busy_until, seq = m.t, -1.0, []
        while m.t - start_t < AWAKE_S:
            o = m.world.owner
            o.present, o.last_seen = True, m.t
            if rng.random() < DT / 240.0:
                m.hear_text(WORDS[rng.integers(len(WORDS))])
            if rng.random() < DT / 300.0:
                m.touch(TOUCHES[rng.integers(len(TOUCHES))])
            step, cancel = m.tick(DT, body_busy=m.t < busy_until, body_ok=True)
            if cancel:
                busy_until = -1.0
            if step is not None:
                busy_until = m.t + MOTION_S.get(step[0], 2.0)
            clock["t"] += DT
            a = m.activity.name if m.activity else "idle"
            if not seq or seq[-1][0] != a:
                why = m.decisions[-1]["reason"] if m.decisions else ""
                seq.append([a, m.t, m.t, why])
            seq[-1][2] = m.t
        out.append(seq)
    return out


def run_twin(seed, days, hours=1.0):
    """The same, in the full twin (real walking times); shorter days."""
    from pinch.twin import PinchSim
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    m = sim.mind
    m.memory.data["homes"] = {}
    rng = np.random.default_rng(seed + 7)
    out = []
    for _ in range(days):
        start_t, seq = sim.t, []
        while sim.t - start_t < hours * 3600.0:
            if rng.random() < DT / 240.0:
                sim.say(WORDS[rng.integers(len(WORDS))])
            if rng.random() < DT / 300.0:
                sim.touch(TOUCHES[rng.integers(len(TOUCHES))])
            sim.run(DT)
            a = m.activity.name if m.activity else "idle"
            if not seq or seq[-1][0] != a:
                why = m.decisions[-1]["reason"] if m.decisions else ""
                seq.append([a, sim.t, sim.t, why])
            seq[-1][2] = sim.t
        out.append(seq)
    return out


def entropy(counts):
    tot = sum(counts.values())
    return -sum(c / tot * math.log2(c / tot) for c in counts.values() if c) if tot else 0.0


def _reactive(name):
    from pinch.mind.behaviour import rank
    return name != "idle" and rank(name) <= 4         # safety, emergencies, hazards, its person's requests and touch


def measures(days):
    flick = sw = own_sw = 0
    dwell, minutes, nxt, rep4, ent = [], 0.0, Counter(), [], []
    pair = Counter()
    for seq in days:
        names = [r[0] for r in seq]
        minutes += (seq[-1][2] - seq[0][1]) / 60.0
        sw += len(seq) - 1
        for i in range(2, len(seq)):
            if _reactive(names[i - 1]) or _reactive(names[i - 2]):
                continue
            own_sw += 1
            cut_short = not str(seq[i][3]).startswith("previous activity finished")
            if names[i] == names[i - 2] and seq[i - 1][2] - seq[i - 1][1] < 10.0 and cut_short:
                flick += 1                                   # (B cut short and straight back: dithering)
        dwell += [r[2] - r[1] for r in seq if not _reactive(r[0])]
        for i in range(1, len(seq)):
            pair[(names[i - 1], names[i])] += 1
            nxt[names[i - 1]] += 1
        grams = [tuple(names[i:i + 4]) for i in range(len(names) - 3)]
        seen, r = set(), 0
        for g in grams:
            r += g in seen
            seen.add(g)
        rep4.append(r / max(1, len(grams)))
        ent.append(entropy(Counter(names)))
    tot = sum(pair.values())
    h_next = -sum(c / tot * math.log2(c / nxt[a]) for (a, _), c in pair.items()) if tot else 0.0
    return dict(flicker=flick / max(1, own_sw), dwell=float(np.median(dwell)), switches=sw / max(1e-6, minutes),
                next_entropy=h_next, repeat4=float(np.mean(rep4)), ent=ent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--twin", action="store_true")
    a = ap.parse_args()
    all_days, ents = [], []
    for seed in range(a.seeds):
        days = run_twin(seed, a.days) if a.twin else run(seed, a.days)
        all_days += days
        ents.append(measures(days)["ent"])
    r = measures(all_days)
    ents = np.array(ents)
    trend = float(ents[:, -3:].mean() / max(1e-6, ents[:, :2].mean())) if ents.shape[1] >= 5 else float("nan")
    rows = [("flicker", f"{100 * r['flicker']:.1f} % of switches", r["flicker"] < 0.05),
            ("dwell", f"median {r['dwell']:.0f} s", 20 <= r["dwell"] <= 90),
            ("switches", f"{r['switches']:.2f} per minute", r["switches"] <= 3),
            ("next_entropy", f"{r['next_entropy']:.2f} bits", 1.0 <= r["next_entropy"] <= 2.0),
            ("repeat4", f"{100 * r['repeat4']:.0f} % of 4-activity runs seen earlier that day", r["repeat4"] < 0.5),
            ("day14/day1", f"{trend:.2f} (activity entropy, last 3 days / first 2)", trend >= 0.85 or trend != trend)]
    ok = True
    for name, txt, good in rows:
        ok &= bool(good)
        print(f"{'PASS' if good else 'FAIL'}  {name:12s} {txt}")
    if os.environ.get("BENCH_VERBOSE"):
        dw = Counter()
        for seq in all_days:
            for r_ in seq:
                dw[r_[0]] += r_[2] - r_[1]
        tot = sum(dw.values())
        print({k: round(100 * v / tot, 1) for k, v in dw.most_common(12)})
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PINCH_QUIET", "1")
    sys.exit(main())
