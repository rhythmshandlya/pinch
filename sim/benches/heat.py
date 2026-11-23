"""Heat: an hour of walking on demand with the servos' heat model on (crab/servo_model.py ServoHeat), judged on
whether the heat-aware mind keeps its legs cool.

    python sim/benches/heat.py                                   # 60 min, every condition, seeds 0-1
    python sim/benches/heat.py --minutes 20 --conditions stress --seeds 0 --out heat.json
    python sim/benches/heat.py --without-drive            # the same hours with a mind that ignores the heat (before)

Its person keeps it walking the whole time, moving from one end of the desk to the other: "come here" from the left
end, then from the right end, over and over, 2 s after the last request is done (each call is a walk of about 25 cm
and a wave). Nothing else happens. The mind runs as on the robot: the hottest leg
servo's temperature feeds its heat need (pinch/mind/selfcare.py), the cool-down drive sits it down when warm, warm
legs make walking activities (even its person's requests) less attractive, and the body slows walking from 55 C and
stops it at 60 C (pinch/body/locomotion.py).

Conditions (the thermal resistance of a Feetech servo is not published; configs/electronics.toml [servo_heat]):
  nominal   the twin's numbers (STS3215 10 K/W, tau 240 s, from SO-ARM100 users' hold tests; v1's smaller STS3032 legs
            28.8 K/W, scaled by case area), a 25 C room
  hot_room  the same servos in a 40 C room (a summer room without air conditioning)
  stress    the pessimistic reading of the datasheet: the maker's rated torque, held for good, settles at the 70 C
            protection limit (about 2-4x the nominal thermal resistance), a 25 C room
  worst     stress in the 40 C room: the case where walking on demand would go past 50 C

Recorded per run: the hottest leg servo every 10 s, its peak and final temperature, the cool-down rests (count, mean
length), the time share of each activity, walking commands the body's heat limit trimmed, falls and safety stops.
Pass, each run: no fall, no safety stop, the hottest leg servo never above 55 C; in the worst condition at least one
cool-down rest (the drive has to act). --without-drive runs the same hours with the heat need held at 0 (the body's
55-60 C walking limit and the firmware's 65 C stop stay): the comparison, judged on nothing.
Design v1 unless CRAB_DESIGN says otherwise: the twin's desk (pinch/world.py Desk: its stance numbers and 0.70 x
0.45 m top) is still sized for v1, and on it the bigger v2 parks at the back edge, where the edge guard refuses every
step, so a v2 hour would mostly stand. The v2 legs' temperatures come from sim/mixed_hour.py meanwhile.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402

CONDITIONS = ("nominal", "hot_room", "stress", "worst")
# (person's x, what it says). No "spin": spinning near an end drifts the body past the edge guard and off the desk
# (seen at x = 0.2: it reached 0.33 and fell), which would turn the hour into a fall test.
REQUESTS = ((-0.12, "come here"), (0.12, "come here"))
PEAK_LIMIT_C = 55.0


def rated_rule_r_th(heat, specs):
    """The pessimistic thermal resistance: the maker's rated torque, held for good, settles at the protection limit."""
    out = []
    for i, s in enumerate(specs):
        cu = (s.rated_torque / heat.kt[i]) ** 2 * heat.R_w[i] * (1 + heat.a_cu[i] * (heat.T_protect[i] - 25.0))
        out.append((heat.T_protect[i] - 25.0) / (cu + heat.P_idle[i]))
    return np.array(out)


def heat_run(args):
    seed, condition, minutes, drive = args
    from crab.params import servo_for
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from robot.crab_robot import SafetyStop
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    if not drive:
        sim.mind.HEAT_WARM_C, sim.mind.HEAT_FULL_C = 1e9, 2e9        # the heat need stays 0: no cool-down, no walking cost
    h = sim.hw.heat
    if condition in ("hot_room", "worst"):
        h.T_amb[:] = 40.0
    if condition in ("stress", "worst"):
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
                x, phrase = REQUESTS[k % len(REQUESTS)]
                if x is not None:                         # its person has moved along the desk
                    for desk in (sim.desk, sim.mind.world.desk):
                        desk.owner_xy = (x, desk.owner_xy[1])
                sim.say(phrase)
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
    ok = sim.falls == 0 and stop is None and peak <= PEAK_LIMIT_C and (condition != "worst" or len(rests) >= 1)
    ok = ok if drive else None
    hot = legs[int(np.argmax(h.peak[legs]))]
    return dict(seed=seed, condition=condition, minutes=minutes, drive=drive, ok=ok, peak_c=round(peak, 2),
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
    ap.add_argument("--without-drive", action="store_true", help="the mind ignores the heat (comparison, no verdict)")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    conds = [c for c in a.conditions.split(",") if c]
    bad = set(conds) - set(CONDITIONS)
    if bad:
        sys.exit(f"unknown conditions {sorted(bad)}: choose from {CONDITIONS}")
    res = run_parallel(heat_run, [(s, c, a.minutes, not a.without_drive) for c in conds for s in seeds], workers=a.jobs)
    for r in res:
        top = ", ".join(f"{n} {100 * v:.0f} %" for n, v in list(r["share"].items())[:4])
        print(f"{r['condition']:8s} seed {r['seed']}: room {r['room_c']:.0f} C, {r['r_th']:.1f} K/W | hottest "
              f"{r['hottest']} peak {r['peak_c']:.1f} C, final {r['final_c']:.1f} C | cool-down rests {r['rests']} "
              f"(mean {r['rest_mean_s']:.0f} s), heat-trimmed walking {r['heat_trimmed']}, falls {r['falls']}, "
              f"stop {r['safety_stop']} | {top} | {'-' if r['ok'] is None else 'PASS' if r['ok'] else 'FAIL'}")
    save_json(a.out, res)
    return 0 if all(r["ok"] is not False for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
