"""Every keyframed move (crab/emotes.py EMOTES) on design v2 against the servo limits (R-03), with a servo model.

    python sim/emote_check.py                     # first-order servos
    python sim/emote_check.py --servo bam         # the full measured STS3215 model on the legs
    python sim/emote_check.py --json out.json

Per move: whether it falls (tilt over 45 deg), the legs' peak motor torque as % of the maker's stall (1.91 N*m) and
the hottest leg joint's RMS torque as % of rated (0.637 N*m), and the arms' peak torque against the arm servo's
usable torque: STS3032 stall 0.44 N*m at 6 V, scaled to the 5 V rail and limited to 40 % by the firmware
(configs/electronics.toml arm torque limit), so about 0.15 N*m; the v2 model applies that limit, so the arms cannot exceed
it and the check reports how far they lag their keyframes instead. Limits: legs peak <= 80 %, RMS <= 70 %; arm lag <= 20 deg. A move that misses is listed with its worst joint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from sim.v2_check import DS_STALL, LEGS, RATED, Run, settle  # noqa: E402

ARMS = [i for i, j in enumerate(JOINT_NAMES) if i not in LEGS]
ARM_STALL_6V = 4.5 * 0.0980665                    # STS3032: 4.5 kg*cm at 6 V (Evelta / Feetech page)
ARM_USABLE = ARM_STALL_6V * 5.0 / 6.0 * 0.40      # 5 V rail B, 40 % firmware torque limit
LEG_PEAK, LEG_RMS = 80.0, 70.0
ARM_LAG = 20.0                                    # deg: an arm that lags its keyframes more than this changes the move


def play(name, servo):
    run = Run(servo)
    settle(run, 1.0)
    run.reset_stats()
    emo = EMOTES[name]
    arm_peak = np.zeros(len(ARMS))
    arm_err = np.zeros(len(ARMS))
    qadr_arm = run.qadr[ARMS]
    t, T = 0.0, emo.duration
    fell = False
    while t < T + 0.8:                                       # the move, then 0.8 s back at the stand pose
        q = emo.target(min(t, T)) if t < T else STAND
        run.step(q)
        f = np.abs(run.d.actuator_force[run.act[ARMS]])
        arm_peak = np.maximum(arm_peak, f)
        if t > 0.1:                                          # how far the arms lag the move (the torque limit slows them)
            arm_err = np.maximum(arm_err, np.abs(run.d.qpos[qadr_arm] - np.asarray(q)[ARMS]))
        R = run.d.xmat[1].reshape(3, 3)
        fell |= R[2, 2] < np.cos(np.radians(45))
        t += run.dt
    s = run.summary()
    worst_arm = JOINT_NAMES[ARMS[int(np.argmax(arm_peak))]]
    return dict(move=name, fell=bool(fell), peak=round(s["peak_ds"], 1), rms=round(s["rms_rated"], 1),
                peak_joint=s["worst"], hot_joint=s["hot"], arm_peak=round(100 * arm_peak.max() / ARM_USABLE, 1),
                arm_joint=worst_arm, tilt=round(s["tilt"], 1), arm_lag=round(float(np.degrees(arm_err.max())), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", default="first_order", choices=["first_order", "bam"])
    ap.add_argument("--only", default=None, help="comma-separated move names")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(EMOTES)
    rows = [play(n, a.servo) for n in names]
    print(f"servo model: {a.servo}; legs: peak % of {DS_STALL} N*m stall (limit {LEG_PEAK:.0f}), RMS % of {RATED} N*m "
          f"rated (limit {LEG_RMS:.0f}); arms: peak % of {ARM_USABLE:.3f} N*m usable (limit 100)")
    print(f"{'move':12s} {'fall':>4s} {'tilt':>5s} {'leg peak':>9s} {'leg RMS':>8s} {'arm peak':>9s} {'arm lag':>8s}  worst")
    bad = []
    for r in rows:
        miss = r["fell"] or r["peak"] > LEG_PEAK or r["rms"] > LEG_RMS or r["arm_peak"] > 100.5 or r["arm_lag"] > ARM_LAG
        if miss:
            bad.append(r["move"])
        print(f"{r['move']:12s} {'YES' if r['fell'] else 'no':>4s} {r['tilt']:5.1f} {r['peak']:8.1f}% {r['rms']:7.1f}% "
              f"{r['arm_peak']:8.1f}% {r['arm_lag']:6.1f}deg  {r['peak_joint']}/{r['hot_joint']}/{r['arm_joint']}{'   <- MISS' if miss else ''}")
    print(f"{len(rows) - len(bad)}/{len(rows)} moves within the limits" + (f"; miss: {', '.join(bad)}" if bad else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
