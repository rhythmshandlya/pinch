"""Heat: an hour of walking on demand with the servos' heat model on (crab/servo_model.py ServoHeat), judged on
whether the heat-aware mind keeps its legs cool.

    python sim/benches/heat.py                                   # design v2, 60 min, every condition, seeds 0-1
    python sim/benches/heat.py --minutes 20 --conditions stress --seeds 0 --out heat.json

Its person keeps it walking the whole time: "come here", then "go away" as soon as it is free, then "spin", over and
over, every 2 s after the last request is done. Nothing else happens. The mind runs as on the robot: the hottest leg
servo's temperature feeds its heat need (pinch/mind/selfcare.py), the cool-down drive sits it down when warm, warm
legs make walking activities (even its person's requests) less attractive, and the body slows walking from 55 C and
stops it at 60 C (pinch/body/locomotion.py).

Conditions (the thermal resistance of a Feetech servo is not published; configs/electronics.toml [servo_heat]):
  nominal   the twin's numbers (STS3215 10 K/W, tau 240 s, fitted to SO-ARM100 users' hold tests), a 25 C room
  hot_room  the same servos in a 40 C room (a summer room without air conditioning)
  stress    the pessimistic reading of the datasheet: the maker's rated torque, held for good, settles at the 70 C
            protection limit (about 4x the nominal thermal resistance), a 25 C room

Recorded per run: the hottest leg servo every 10 s, its peak and final temperature, the cool-down rests (count, mean
length), the time share of each activity, walking commands the body's heat limit trimmed, falls and safety stops.
Pass, each run: no fall, no safety stop, the hottest leg servo never above 55 C; in the stress condition at least one
cool-down rest (the drive has to act). Design v2 (the robot being built) unless CRAB_DESIGN says otherwise.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402

CONDITIONS = ("nominal", "hot_room", "stress")
REQUESTS = ("come here", "go away", "pinch, spin")
PEAK_LIMIT_C = 55.0


def rated_rule_r_th(heat, specs):
    """The pessimistic thermal resistance: the maker's rated torque, held for good, settles at the protection limit."""
    out = []
    for i, s in enumerate(specs):
        cu = (s.rated_torque / heat.kt[i]) ** 2 * heat.R_w[i] * (1 + heat.a_cu[i] * (heat.T_protect[i] - 25.0))
        out.append((heat.T_protect[i] - 25.0) / (cu + heat.P_idle[i]))
    return np.array(out)


def heat_run(args):
    seed, condition, minutes = args
    from crab.params import servo_for
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from robot.crab_robot import SafetyStop
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    h = sim.hw.heat
    if condition == "hot_room":
        h.T_amb[:] = 40.0
    elif condition == "stress":
        h.R_th[:] = rated_rule_r_th(h, [servo_for(j) for j in sim.hw.heat_joints])
    h.T[:] = h.T_amb                                  # the servos start at the room's temperature
    h.peak[:] = h.T
    legs = [i for i, j in enumerate(sim.hw.heat_joints) if j.split("_", 1)[1] in ("coxa", "femur", "tibia")]
    trace, acts, k, free_since, stop = [], Counter(), 0, 0.0, None
    try:
        while sim.t < minutes * 60.0:
            m = sim.mind
            busy = m.request is not None or (m.activity is not None and m.activity.name == "obey" and m.gen is not None)
            if busy:
                free_since = sim.t
            elif sim.t - free_since >= 2.0:
                sim.say(REQUESTS[k % len(REQUESTS)])
                k += 1
                free_since = sim.t
            sim.run(1.0)
            acts[m.activity.name if m.activity else "none"] += 1
            if int(round(sim.t)) % 10 == 0:
                trace.append([round(sim.t), round(float(h.T[legs].max()), 2)])
    except SafetyStop as e:
        stop = str(e)
    rests = [d for d in sim.mind.decisions if d["to"] == "cool_down"]
    lengths, start = [], None
    for d in sim.mind.decisions:                      # how long each cool-down lasted
        if d["to"] == "cool_down":
            start = d["t"]
        elif start is not None:
            lengths.append(d["t"] - start)
            start = None
    peak = float(h.peak[legs].max())
    total = max(sum(acts.values()), 1)
    ok = sim.falls == 0 and stop is None and peak <= PEAK_LIMIT_C and (condition != "stress" or len(rests) >= 1)
    hot = legs[int(np.argmax(h.peak[legs]))]
    return dict(seed=seed, condition=condition, minutes=minutes, ok=ok, peak_c=round(peak, 2),
                hottest=sim.hw.heat_joints[hot], final_c=round(float(h.T[legs].max()), 2),
                room_c=float(h.T_amb[0]), r_th=round(float(h.R_th[legs[0]]), 2), tau_s=float(h.tau[legs[0]]),
                rests=len(rests), rest_mean_s=round(float(np.mean(lengths)), 1) if lengths else 0.0,
                requests=k, heat_trimmed=sim.body.heat_trimmed, falls=sim.falls, safety_stop=stop,
                share={a: round(n / total, 3) for a, n in acts.most_common()}, trace=trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seeds", default="0-1", help="e.g. 0-3 or 5")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    conds = [c for c in a.conditions.split(",") if c]
    bad = set(conds) - set(CONDITIONS)
    if bad:
        sys.exit(f"unknown conditions {sorted(bad)}: choose from {CONDITIONS}")
    res = run_parallel(heat_run, [(s, c, a.minutes) for c in conds for s in seeds], workers=a.jobs)
    for r in res:
        top = ", ".join(f"{n} {100 * v:.0f} %" for n, v in list(r["share"].items())[:4])
        print(f"{r['condition']:8s} seed {r['seed']}: room {r['room_c']:.0f} C, {r['r_th']:.1f} K/W | hottest "
              f"{r['hottest']} peak {r['peak_c']:.1f} C, final {r['final_c']:.1f} C | cool-down rests {r['rests']} "
              f"(mean {r['rest_mean_s']:.0f} s), heat-trimmed walking {r['heat_trimmed']}, falls {r['falls']}, "
              f"stop {r['safety_stop']} | {top} | {'PASS' if r['ok'] else 'FAIL'}")
    save_json(a.out, res)
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
