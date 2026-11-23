"""The v2 robot's power in the twin with the circuit simulation (crab/wiring.py Circuit, configs/wiring_v2.toml) against the
earlier lumped battery + regulator model: standing, walking forward and sideways at the gait's cap, a get-up attempt
(crab/righting.py swing), and the battery minutes of a mixed day.

    python sim/circuit_check.py [--bam] [--seconds 8]

Prints, per case: peak and mean pack current (as the circuit sees it, averaged over each 10 ms control tick), the lowest
voltage at the pack terminals, at a leg servo, at an arm servo and at the computer, the regulators' loads, whether the fuse
or the BMS tripped, whether the computer browned out. Then the battery life.."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from crab import righting  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.gait import GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from crab.wiring import Circuit  # noqa: E402
from robot.backends import SimBackend  # noqa: E402
from robot.servos import SERVO_IDS  # noqa: E402

DT = 0.01
MIXED_DAY = (("standing", 0.50, "camera"), ("walk forward", 0.20, "camera"), ("walk sideways", 0.10, "camera"),
             ("standing", 0.20, "idle"))      # share of the day, which case's servo load, what the computer does


def _send(hw, q):
    hw.write_positions({SERVO_IDS[j]: hw.sign[j] * q[i] for i, j in enumerate(JOINT_NAMES)})


def _gait(hw, cmd):
    g = Gait(GAITS["trot"], com_offset=(hw.d.subtree_com[1] - hw.d.xpos[1])[:2])
    return lambda t: g.step(DT, cmd)


def _case(name):
    if name == "standing":
        return lambda hw: (lambda t: STAND), 5.0
    if name == "walk forward":
        return lambda hw: _gait(hw, (0.15, 0.0, 0.0)), 8.0
    if name == "walk sideways":
        return lambda hw: _gait(hw, (0.0, 0.15, 0.0)), 8.0
    if name == "get-up swing":
        return lambda hw: (lambda t: righting.target(t)), 4.0
    raise KeyError(name)


def run(name, power_model="circuit", servo_model="first_order", seconds=None, seed=0):
    make, secs = _case(name)
    secs = seconds or secs
    hw = SimBackend(realism="datasheet", servo_model=servo_model, power_model=power_model, seed=seed)
    for _ in range(300):                                           # stand 3 s first
        _send(hw, STAND)
        hw.wait(DT)
    if name == "get-up swing":                                     # on its back
        hw.d.qpos[3:7] = (0.0, 1.0, 0.0, 0.0)
        hw.d.qpos[2] = 0.075
        hw.d.qvel[:] = 0
        import mujoco
        mujoco.mj_forward(hw.m, hw.d)
        if hw.drive is not None:
            hw.drive.reset()
    fn = make(hw)
    if hw.circuit is not None:
        hw.circuit.stats.update(i_peak=0.0, i_sum=0.0, n=0, v_servo_min=99.0, v_arm_min=99.0, v_pack_min=99.0,
                                v_computer_min=99.0, load_a_max=0.0, load_b_max=0.0, i_3v3_max=0.0)
        hw.circuit.acc.update(a=hw.circuit.acc["a"] * 0, b=hw.circuit.acc["b"] * 0, arm=hw.circuit.acc["arm"] * 0, t=0.0)
    cur, vb, vbus = [], [], []
    z_min, up_min = 9.0, 1.0
    for k in range(int(secs / DT)):
        _send(hw, fn(k * DT))
        hw.wait(DT)
        if hw.circuit is None:
            p = hw.elec.power
            cur.append(p.i_batt)
            vb.append(p.v_pack)
            vbus.append(p.v_bus)
        z_min = min(z_min, float(hw.d.qpos[2]))
        up_min = min(up_min, float(hw.d.xmat[1].reshape(3, 3)[2, 2]))
    out = dict(case=name, seconds=secs, z_min=z_min, up_min=up_min, up_end=float(hw.d.xmat[1].reshape(3, 3)[2, 2]))
    if hw.circuit is not None:
        out.update(hw.circuit.summary(secs))
        out["mean_load"] = hw.circuit.mean_load()
        out["events"] = list(hw.circuit.events)
    else:
        out.update(i_peak=max(cur), i_mean=float(np.mean(cur)), v_pack_min=min(vb), v_servo_min=min(vbus))
    return out


def battery_life(loads, enforce_module_min=False, dt=2.0):
    """Minutes of the mixed day until the pack, loaded, falls under 7 V (the regulator modules' listed minimum), under the
    firmware's 6.4 V stop, or is empty (its usable 80 %). `loads` = {case: mean (a, b, arm)} from the twin runs."""
    res = {}
    cir = Circuit(soc=1.0, enforce_module_min=enforce_module_min)
    t, marks = 0.0, {}
    n = len(cir.legs)
    # a servo draws I = a V - b, linear in the mix: the mixed day's coefficients are the weighted sums
    a = sum(w * loads[c][0] for c, w, _ in MIXED_DAY)
    b = sum(w * loads[c][1] for c, w, _ in MIXED_DAY)
    arm = sum(w * loads[c][2] for c, w, _ in MIXED_DAY)
    share_idle = sum(w for _, w, act in MIXED_DAY if act == "idle")
    assert len(a) == n
    while t < 4 * 3600 and cir.soc > 0 and not cir.dead:
        for act, share in (("camera", 1 - share_idle), ("idle", share_idle)):
            cir.activity = act
            cir.step(dt * share, a, b, arm, True)
            if cir.v_pack_term < 7.0 and "7 V" not in marks:
                marks["7 V"] = t
            if cir.v_pack_term < 6.4 and "6.4 V" not in marks:
                marks["6.4 V"] = t
            if not cir.computer_up and "brown-out" not in marks:
                marks["brown-out"] = t
        t += dt
        if "brown-out" in marks and enforce_module_min:
            break
    marks["usable"] = t
    res = {k: v / 60.0 for k, v in marks.items()}
    res["mean_pack_a"] = cir.stats["i_sum"] / max(1, cir.stats["n"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", action="store_true", help="also run walking forward with the full BAM servo model")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    names = ["standing", "walk forward", "walk sideways", "get-up swing"]
    rows, lump, loads = [], [], {}
    for n in names:
        r = run(n, "circuit", seconds=a.seconds)
        rows.append(r)
        loads[n] = r["mean_load"]
        lump.append(run(n, "lumped", seconds=a.seconds))
    print("v2 on the circuit model (first-order servo), pack 2S 2500 mAh, 100 % charge, camera on\n")
    print(f"{'case':14s} {'pack A peak':>11s} {'mean':>6s} {'pack V min':>10s} {'leg servo V':>11s} {'arm V':>6s} "
          f"{'computer V':>10s} {'reg A load':>10s} {'reg B':>6s} {'fuse':>6s}  notes")
    for r in rows:
        note = "no fuse/BMS/brown-out" if not (r["fuse_blown"] or r["bms_off"] or r["resets"]) else \
            f"fuse={r['fuse_blown']} bms={r['bms_off']} resets={r['resets']}"
        print(f"{r['case']:14s} {r['i_peak']:11.2f} {r['i_mean']:6.2f} {r['v_pack_min']:10.2f} {r['v_servo_min']:11.2f} "
              f"{r['v_arm_min']:6.2f} {r['v_computer_min']:10.2f} {r['load_a_max']:9.2f}A {r['load_b_max']:5.2f}A "
              f"{r['fuse_damage']:6.3f}  {note}; tilt min up={r['up_min']:.2f}, end up={r['up_end']:.2f}")
    print("\nthe earlier lumped model (one 5 V rail model, 7.4 V servo bus, no wires, no fuse):")
    print(f"{'case':14s} {'pack A peak':>11s} {'mean':>6s} {'pack V min':>10s} {'servo bus V':>11s}")
    for r in lump:
        print(f"{r['case']:14s} {r['i_peak']:11.2f} {r['i_mean']:6.2f} {r['v_pack_min']:10.2f} {r['v_servo_min']:11.2f}")
    if a.bam:
        r = run("walk forward", "circuit", servo_model="bam", seconds=a.seconds)
        print(f"\nfull BAM servo model, walk forward: peak {r['i_peak']:.2f} A, mean {r['i_mean']:.2f} A, "
              f"servo V min {r['v_servo_min']:.2f}")
    print("\nbattery life of the mixed day (50 % standing with the camera on, 20 % walking forward, 10 % sideways, "
          "20 % standing with the computer idle):")
    for enforce in (False, True):
        life = battery_life(loads, enforce)
        txt = ", ".join(f"{k}: {v:.0f} min" for k, v in life.items() if k != "mean_pack_a")
        print(f"  modules {'stop below 7 V (listed minimum)' if enforce else 'run down to the chip minimum   '}: {txt};"
              f" mean pack current {life['mean_pack_a']:.2f} A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
