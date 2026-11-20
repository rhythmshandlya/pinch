"""Electrical power budget from simulation.

Servo current per joint comes from that joint's servo model: I = |tau_motor| / kt, with
tau_motor = actuator force - back-EMF slope * speed (electromagnetic torque), plus each
servo's 6 mA idle draw. Back-driven phases are counted as draw (conservative).
Prints average / peak current per scenario and battery runtime estimates.

    python sim/power.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import PARAMS  # noqa: E402
from sim.servo_check import KE, KT  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
IDLE_A = 0.006
COMPUTE_W = 4.0            # Radxa ZERO 3W busy (~3 W, Wi-Fi on) + camera + sensors (W)
BUCK_EFF = 0.90
V_BATT = 7.4               # 2S LiPo nominal
USABLE = 0.80              # don't drain a LiPo below ~20 %
PACKS = (850,)             # mAh: the Bonka 2S pack in the BOM


def run(m, d, ctrl_fn, seconds):
    n = int(seconds / m.opt.timestep)
    cur = np.zeros(n)
    for i in range(n):
        d.ctrl[:] = ctrl_fn(i * m.opt.timestep)
        mujoco.mj_step(m, d)
        tau_m = d.actuator_force - KE * d.actuator_velocity
        cur[i] = np.sum(np.abs(tau_m) / KT) + m.nu * IDLE_A
    return cur


def main():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    def fresh():
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        run(m, d, lambda t: STAND, 1.0)

    v, _w = _S / _T, 1 / _T
    scen = {}
    fresh()
    scen["standing"] = run(m, d, lambda t: STAND, 5.0)
    fresh()
    scen["idle emote"] = run(m, d, EMOTES["idle"].target, 6.4)
    for name in ("wave", "happy", "angry"):
        fresh()
        scen[f"{name} emote"] = run(m, d, EMOTES[name].target, EMOTES[name].duration)
    for gname, cmd, label in (("trot", (0.08 * v, 0, 0), "trot forward"), ("trot", (0, 0.10 * v, 0), "trot sideways"),
                              ("crawl", (0.02 * v, 0, 0), "crawl")):
        fresh()
        g = Gait(GAITS[gname], com_offset=np.array([0.015 * _S, 0.0]))
        scen[label] = run(m, d, lambda t, g=g, cmd=cmd: g.step(m.opt.timestep, cmd), 8.0)[1000:]

    vb = PARAMS.bus_voltage
    print(f"Servo bus {vb} V; computer + sensors ~{COMPUTE_W:.0f} W; regulators {BUCK_EFF:.0%}; 2S LiPo\n")
    print(f"{'scenario':15s} {'servo avg A':>11s} {'peak A':>7s} {'battery avg A':>13s} "
          + " ".join(f"{p:>5d}mAh" for p in PACKS))
    rows = {}
    for k, c in scen.items():
        batt_a = (c.mean() * vb + COMPUTE_W) / BUCK_EFF / V_BATT
        rt = lambda mah: USABLE * mah / 1000 / batt_a * 60  # noqa: E731, B023 - used at once, in this iteration
        rows[k] = (c.mean(), c.max(), batt_a)
        print(f"{k:15s} {c.mean():11.2f} {c.max():7.2f} {batt_a:13.2f} " + " ".join(f"{rt(p):6.0f}min" for p in PACKS))
    return rows


if __name__ == "__main__":
    main()
