"""Design checks of the resized robot (design v2: sim/models/crab_v2.xml, STS3215 legs at 7.4 V), the v2 counterpart
of sim/check.py + sim/walk.py + sim/servo_check.py, with the servo model as a choice.

    python sim/v2_check.py                       # stand + walks, first-order servo, then the full BAM servo, compared
    python sim/v2_check.py --servo bam
    python sim/v2_check.py --servo first_order --seconds 8

Per case it reports: speed reached, tilt, joint tracking error, motor torque as a fraction of the maker's stall
(1.91 N*m) and of the rated torque (0.637 N*m, RMS), joint speed against the measured 5.1 rad/s firmware limit, and
electrical power of the 12 leg servos (copper loss R*I^2 + positive mechanical power + idle current, with
I = torque / kt; see docs/dev/V2_MODEL.md). The command delay is the measured 5 ms in every case.
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.electronics import Data  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.gait import _S, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES, JOINT_SPECS, PARAMS  # noqa: E402
from crab.servo_model import make_drive  # noqa: E402
from sim.latency import DelayLine  # noqa: E402

MODEL = Path(os.environ["CRAB_MODEL_PATH"]) if os.environ.get("CRAB_MODEL_PATH") else ROOT / "sim" / "models" / PARAMS.model_file   # a candidate (sim/situations.py)
LEGS = [i for i, j in enumerate(JOINT_NAMES) if j[0] in "FR" and j[1] in "LR"]          # the 12 leg joints
SPEC = JOINT_SPECS[LEGS[0]]
DS_STALL, RATED, VMAX = SPEC.datasheet_stall, SPEC.rated_torque, SPEC.max_cmd_speed
KT, R = SPEC.kt_current, float(Data()("leg_servo.resistance_ohm"))
IDLE_A = float(Data()("leg_servo.idle_current_a"))
DELAY = SPEC.command_delay
DT_GAIT = 0.002


def load(servo):
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    drive = make_drive(servo, m, d, JOINT_NAMES)
    drive.reset()
    return m, d, drive


class Run:
    """Steps the twin with a drive, the measured command delay, and the statistics of the check."""

    def __init__(self, servo):
        self.m, self.d, self.drive = load(servo)
        self.dt = self.m.opt.timestep
        self.delay = DelayLine(DELAY, self.dt, STAND)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINT_NAMES])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in JOINT_NAMES])
        self.act = np.array([self.m.actuator(j).id for j in JOINT_NAMES])
        self.bam = servo == "bam"
        self.reset_stats()

    def reset_stats(self):
        self.n = 0
        self.sq = np.zeros(12)            # torque^2
        self.peak = np.zeros(12)
        self.peak_w = 0.0
        self.err2 = 0.0
        self.energy = 0.0                 # J, 12 leg servos
        self.min_h, self.max_tilt = 9.0, 0.0
        self.v, self.wz, self.cnt = np.zeros(2), 0.0, 0
        self.traj = []                    # leg joint angles every 10 steps, to compare two servo models

    def motor_torque(self):
        f = self.d.actuator_force[self.act[LEGS]]
        if self.bam:                      # the BAM controller's torque already includes the back-EMF term
            return f
        return f - SPEC.back_emf * self.d.qvel[self.vadr[LEGS]]

    def step(self, q, measure=True, velocity=False):
        d = self.d
        self.drive.command(self.delay(q))
        self.drive.step()
        mujoco.mj_step(self.m, d)
        if not measure:
            return
        tau, w = self.motor_torque(), d.qvel[self.vadr[LEGS]]
        self.n += 1
        self.sq += tau ** 2
        self.peak = np.maximum(self.peak, np.abs(tau))
        self.peak_w = max(self.peak_w, float(np.abs(w).max()))
        self.err2 += float(np.mean((d.qpos[self.qadr[LEGS]] - np.asarray(q)[LEGS]) ** 2))
        amps = (tau / KT) ** 2 * R + np.maximum(0.0, tau * w)           # copper loss + positive mechanical, W
        self.energy += (float(amps.sum()) + 12 * IDLE_A * SPEC.vin) * self.dt
        if self.n % 10 == 0:
            self.traj.append(d.qpos[self.qadr[LEGS]].copy())
        R3 = d.xmat[1].reshape(3, 3)
        self.min_h = min(self.min_h, float(d.xpos[1, 2]))
        self.max_tilt = max(self.max_tilt, float(np.degrees(np.arccos(np.clip(R3[2, 2], -1, 1)))))
        if velocity:
            self.v += (R3.T @ d.qvel[0:3])[:2]
            self.wz += d.qvel[5]
            self.cnt += 1

    def summary(self, cmd=None):
        secs = max(self.n, 1) * self.dt
        rms = np.sqrt(self.sq / max(self.n, 1))
        out = dict(peak_ds=100 * self.peak.max() / DS_STALL, rms_rated=100 * rms.max() / RATED,
                   rms_rated_mean=100 * rms.mean() / RATED, peak_w=self.peak_w, track=float(np.sqrt(self.err2 / max(self.n, 1))),
                   power=self.energy / secs, tilt=self.max_tilt, min_h=self.min_h,
                   worst=JOINT_NAMES[LEGS[int(np.argmax(self.peak))]], hot=JOINT_NAMES[LEGS[int(np.argmax(rms))]],
                   traj=np.array(self.traj))
        if self.cnt:
            out["v"] = self.v / self.cnt
            out["wz"] = self.wz / self.cnt
        return out


def settle(run, seconds=1.0):
    for _ in range(int(seconds / run.dt)):
        run.step(STAND, measure=False)
    com = run.d.subtree_com[1] - run.d.xpos[1]
    return com[:2].copy()


def stand(servo, seconds=5.0):
    run = Run(servo)
    settle(run, 1.0)
    run.reset_stats()
    for _ in range(int(seconds / run.dt)):
        run.step(STAND)
    s = run.summary()
    s["z"] = float(run.d.xpos[1, 2])
    return s


def walk(servo, cmd, seconds=8.0, gait_name="trot", warm=2.0):
    run = Run(servo)
    com = settle(run, 1.0)
    gait = Gait(GAITS[gait_name], com_offset=com)
    gait.reset()
    period = gait.params.period
    cycles = max(2, int((seconds - warm) / period))
    n_warm, n_meas = int(warm / run.dt), int(round(cycles * period / run.dt))
    for _ in range(n_warm):
        run.step(gait.step(run.dt, cmd), measure=False)
    run.reset_stats()
    for _ in range(n_meas):
        run.step(gait.step(run.dt, cmd), velocity=True)
    s = run.summary()
    eff = np.asarray(cmd) * gait.feasible_scale(cmd)
    s["cmd"] = eff
    s["fell"] = s["tilt"] > 40 or s["min_h"] < 0.4 * float(run.m.key("stand").qpos[2])
    s["unreachable"] = gait.unreachable
    return s


CASES = [("forward 5 cm/s", (0.05, 0, 0)), ("forward 10 cm/s", (0.10, 0, 0)), ("sideways 5 cm/s", (0, 0.05, 0)),
         ("sideways 10 cm/s", (0, 0.10, 0)), ("forward cap", (0.15, 0, 0)), ("sideways cap", (0, 0.15, 0)),
         ("turn 0.4 rad/s", (0, 0, 0.4))]


def line(label, s):
    v = s.get("v")
    speed = "" if v is None else f"got ({v[0]*100:5.1f}, {v[1]*100:5.1f}) cm/s wz {s['wz']:5.2f} | "
    return (f"{label:17s} {speed}tilt {s['tilt']:4.1f} | peak {s['peak_ds']:4.0f}% of stall @{s['worst']:11s} | "
            f"RMS {s['rms_rated']:4.0f}% of rated @{s['hot']:11s} | speed {s['peak_w']:4.2f} rad/s | "
            f"track {np.degrees(s['track']):4.2f} deg | {s['power']:5.1f} W" + ("  FELL" if s.get("fell") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", choices=("first_order", "bam", "both"), default="both")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--cases", default="", help="comma list of case labels to run (default: all)")
    a = ap.parse_args()
    kinds = ("first_order", "bam") if a.servo == "both" else (a.servo,)
    mass = sum(mujoco.MjModel.from_xml_path(str(MODEL)).body_mass)
    print(f"design v2: {MODEL.name}, mass {mass:.3f} kg, scale {_S:.3f}, trot period {GAITS['trot'].period:.3f} s "
          f"(x{PARAMS.gait_period_factor} slower), servo {SPEC.name}: datasheet stall {DS_STALL} N*m, rated {RATED} N*m, "
          f"firmware speed limit {VMAX:.2f} rad/s; gait speed cap {GAITS['trot'].max_step / (GAITS['trot'].duty * GAITS['trot'].period / 2) * 100:.1f} cm/s")
    res = {}
    for kind in kinds:
        print(f"\n=== servo model: {kind} ===")
        s = stand(kind)
        res[kind] = {"stand": s}
        print(f"stand 5 s: z {s['z']*100:.1f} cm, tilt {s['tilt']:.2f} deg, peak {s['peak_ds']:.1f}% of stall, "
              f"RMS {s['rms_rated']:.1f}% of rated @{s['hot']}, power {s['power']:.2f} W")
        for label, cmd in CASES:
            if a.cases and label not in a.cases.split(","):
                continue
            res[kind][label] = walk(kind, cmd, a.seconds)
            print(line(label, res[kind][label]))
    if len(kinds) == 2:
        print("\n=== full BAM model minus first-order model, same gait, same command ===")
        print(f"{'case':17s} {'speed':>8s} {'peak torque':>12s} {'RMS torque':>11s} {'mean RMS':>9s} {'tracking':>9s} "
              f"{'power':>8s} {'joint angle RMS diff':>21s}")
        for label in ["stand"] + [c for c, _ in CASES if c in res["bam"]]:
            f, b = res["first_order"][label], res["bam"][label]
            n = min(len(f["traj"]), len(b["traj"]))
            dq = float(np.degrees(np.sqrt(np.mean((f["traj"][:n] - b["traj"][:n]) ** 2)))) if n else 0.0
            sp = "" if "v" not in f else f"{100 * (np.linalg.norm(b['v']) / max(np.linalg.norm(f['v']), 1e-9) - 1):+7.1f}%"
            print(f"{label:17s} {sp:>8s} {b['peak_ds'] - f['peak_ds']:+10.1f}pt {b['rms_rated'] - f['rms_rated']:+9.1f}pt "
                  f"{b['rms_rated_mean'] - f['rms_rated_mean']:+7.1f}pt {np.degrees(b['track'] - f['track']):+7.2f}deg "
                  f"{100 * (b['power'] / f['power'] - 1):+7.1f}% {dq:15.2f} deg")


if __name__ == "__main__":
    main()
