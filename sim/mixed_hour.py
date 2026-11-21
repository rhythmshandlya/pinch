"""A mixed simulated hour on design v2: is the hottest leg joint's average torque within the heat limit (R-03)?

    python sim/mixed_hour.py                    # 60 simulated minutes, first-order servos
    python sim/mixed_hour.py --servo bam --minutes 20

The hour is a seeded random mix of what a calm desk pet does: about half standing still (idle life), walking forward
at the gait's speed cap, walking sideways at its cap, turning, and the keyframed moves (crab/emotes.py), in stretches
of 5-40 s. Reported: each leg joint's RMS motor torque over the whole run as % of rated (0.637 N*m; limit 70 % for
the hottest joint), the peak as % of the maker's stall (1.91 N*m; limit 80 %), falls, and the time share of each
activity. The limit is the owner's requirement R-03 (docs/PLAN.md phase 2 exit check).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.gait import GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES, PARAMS  # noqa: E402
from sim.v2_check import DS_STALL, LEGS, RATED, Run, settle  # noqa: E402

MIX = [("stand", 0.50), ("forward", 0.15), ("sideways", 0.10), ("turn", 0.10), ("move", 0.15)]
CMDS = {"forward": (0.15, 0.0, 0.0), "sideways": (0.0, 0.15, 0.0), "turn": (0.0, 0.0, 0.4), "stand": (0.0, 0.0, 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", default="first_order", choices=["first_order", "bam"])
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if PARAMS.design != "v2":
        sys.exit("design v2 only")
    rng = np.random.default_rng(a.seed)
    run = Run(a.servo)
    com = settle(run, 1.0)
    run.reset_stats()
    gait = Gait(GAITS["trot"], com_offset=com)
    names, probs = [m[0] for m in MIX], np.array([m[1] for m in MIX])
    total, share, falls = a.minutes * 60.0, {n: 0.0 for n in names}, 0
    t = 0.0
    last = np.array(STAND, float)

    def go(q):                                                     # every command passes here (remembers the last one)
        nonlocal last
        last = np.asarray(q, float)
        run.step(last)

    def blend_to(q_start_fn, secs=0.5):
        """Blend from the last command to the next activity's first pose, as the body does between activities."""
        q0, q1 = last.copy(), np.asarray(q_start_fn, float)
        n = int(secs / run.dt)
        for k in range(n):
            u = (k + 1) / n
            go(q0 + (q1 - q0) * (0.5 - 0.5 * np.cos(np.pi * u)))

    while t < total:
        what = names[rng.choice(len(names), p=probs / probs.sum())]
        if what == "move":
            emo = EMOTES[list(EMOTES)[rng.integers(len(EMOTES))]]
            secs = emo.duration + 0.6
            blend_to(emo.target(0.0))
            for k in range(int(secs / run.dt)):
                tt = k * run.dt
                go(emo.target(min(tt, emo.duration)) if tt < emo.duration else STAND)
        else:
            secs = float(rng.uniform(5, 40))
            cmd = CMDS[what]
            if what == "stand":
                blend_to(STAND)
                for _ in range(int(secs / run.dt)):
                    go(STAND)
            else:
                gait.reset()
                blend_to(gait.step(0.0, (0.0, 0.0, 0.0)))
                for _ in range(int(secs / run.dt)):
                    go(gait.step(run.dt, cmd))
                blend_to(STAND, 0.6)                               # settle back to the stand pose
                secs += 0.6
        share[what] += secs
        t += secs
        R = run.d.xmat[1].reshape(3, 3)
        if R[2, 2] < np.cos(np.radians(45)):
            falls += 1
            sq, n, peak = run.sq, run.n, run.peak                 # keep the statistics, put it back on its feet
            run = Run(a.servo)                                     # pick it up and carry on (counted as a fall)
            com = settle(run, 1.0)
            run.sq, run.n, run.peak = sq, n, peak
            last = np.array(STAND, float)
            gait = Gait(GAITS["trot"], com_offset=com)
    rms = np.sqrt(run.sq / max(run.n, 1))
    hot = int(np.argmax(rms))
    print(f"mixed {t / 60:.1f} simulated minutes, servo model {a.servo}, seed {a.seed}; falls {falls}")
    print("time share: " + ", ".join(f"{n} {100 * v / t:.0f} %" for n, v in share.items()))
    for i, j in enumerate(LEGS):
        print(f"  {JOINT_NAMES[j]:9s} RMS {100 * rms[i] / RATED:5.1f} % of rated   peak {100 * run.peak[i] / DS_STALL:5.1f} % of stall")
    ok = 100 * rms[hot] / RATED <= 70 and 100 * run.peak.max() / DS_STALL <= 80 and falls == 0
    print(f"hottest joint {JOINT_NAMES[LEGS[hot]]}: RMS {100 * rms[hot] / RATED:.1f} % of rated (limit 70), peak "
          f"{100 * run.peak.max() / DS_STALL:.1f} % of stall (limit 80): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
