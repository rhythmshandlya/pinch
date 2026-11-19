"""Long-duration soak: hours of simulated pet life with a randomised day around it, then a verdict.

    python sim/soak.py --hours 1 --seed 3 --out soak.json
    python sim/soak.py --hours 2 --seeds 0-5 --jobs 6          # several days side by side

The day (seeded, so a run replays exactly): you come and go; while here you talk to it, stroke or tickle it,
hold meetings, now and then nudge a desk thing out of place; a visitor drops by. Recorded: falls, anything
leaving the desk (Pinch or things), the behaviour-quality observer's findings, time in each top-level state,
how varied its choices were (entropy), and safety stops. Pass: no falls, Pinch never leaves the desk, no
observer errors, no state locked, and at least 5 different activities.
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PHRASES = ["hi pinch", "dance!", "good job pinch", "how are you?", "I'm sad today", "pinch, wave", "come here",
           "let's play", "I'm tired", "you're so cute", "pinch, put the cap away", "shake", "play dead",
           "I'm in a meeting", "bye pinch", "good morning", "no, bad pinch"]
TOUCHES = ["stroke", "stroke", "tickle", "boop", "tap", "hold"]


def schedule(rng, seconds):
    """[(t, kind, arg)] for the day."""
    ev, t = [], 30.0
    here = True
    while t < seconds:
        if here:
            r = rng.random()
            if r < 0.45:
                ev.append((t, "say", PHRASES[rng.integers(len(PHRASES))]))
            elif r < 0.75:
                ev.append((t, "touch", TOUCHES[rng.integers(len(TOUCHES))]))
            elif r < 0.85:
                ev.append((t, "nudge", ("cap", "eraser", "pen")[rng.integers(3)]))
            elif r < 0.9:
                ev.append((t, "visitor", None))
            else:
                ev.append((t, "leave", None))
                here = False
            t += rng.exponential(90.0)
        else:
            t += rng.exponential(600.0)
            ev.append((t, "arrive", None))
            here = True
    return [e for e in ev if e[0] < seconds]


def soak(args):
    seed, hours = args
    from pinch.memory import Memory
    from pinch.observer import observe
    from pinch.sim_runner import PinchSim
    rng = np.random.default_rng(10_000 + seed)
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    seconds = hours * 3600.0
    events = schedule(rng, seconds)
    off_desk, things_off, state_time = 0.0, set(), Counter()
    k = 0
    while sim.t < seconds:
        while k < len(events) and events[k][0] <= sim.t:
            _, kind, arg = events[k]
            k += 1
            if kind == "say":
                sim.say(arg)
            elif kind == "touch":
                sim.mind.touch(arg)
            elif kind == "nudge":
                xy = (rng.uniform(-0.22, 0.22), rng.uniform(-0.12, 0.12))
                if not sim.desk.in_tray(xy) and np.hypot(*xy) > 0.15:
                    sim.place(arg, xy, rng.uniform(-math.pi, math.pi))
            elif kind == "visitor" and hasattr(sim.mind, "see_person"):
                sim.mind.see_person(f"visitor-{seed}", 0.8)
            elif kind == "leave":
                sim.user_active = False
            elif kind == "arrive":
                sim.user_active = True
        sim.run(1.0)
        state_time[sim.mind.state] += 1.0
        x, y, _ = sim.pose()
        if not sim.desk.inside((x, y), 0.0):
            off_desk += 1.0
        for name in ("pen", "eraser", "cap", "ball"):
            if sim.obj(name)[1] < -0.02:
                things_off.add(name)
    findings = observe(sim.mind, seconds)
    acts = Counter(d["to"] for d in sim.mind.decisions)
    n = sum(acts.values())
    entropy = -sum(c / n * math.log2(c / n) for c in acts.values()) if n else 0.0
    errors = [f"{f.kind}: {f.evidence}" for f in findings if f.severity == "error"]
    locked = [f.evidence for f in findings if f.kind == "state_lock"]
    ok = sim.falls == 0 and off_desk == 0 and not errors and not locked and len(acts) >= 5
    return dict(seed=seed, hours=hours, ok=ok, falls=sim.falls, pinch_off_desk_s=off_desk, things_off=sorted(things_off),
                decisions=n, activities=dict(acts.most_common()), entropy_bits=round(entropy, 2),
                state_share={k: round(v / seconds, 3) for k, v in state_time.most_common()},
                findings=[dict(kind=f.kind, severity=f.severity, t0=round(f.t0), evidence=f.evidence) for f in findings],
                events=len(events), sounds=len(sim.sounds_played))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seeds", default="0", help="e.g. 3 or 0-5")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(soak, [(s, a.hours) for s in seeds]))
    for r in res:
        print(f"seed {r['seed']}: {'PASS' if r['ok'] else 'FAIL'}  {r['hours']} h  falls {r['falls']}  off desk "
              f"{r['pinch_off_desk_s']:.0f} s  things off {r['things_off']}  {r['decisions']} decisions over "
              f"{len(r['activities'])} activities (entropy {r['entropy_bits']} bits)  states {r['state_share']}")
        for f in r["findings"]:
            print(f"    {f['severity']:5s} {f['kind']}: {f['evidence']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
