"""Runs in its own process (tests/test_v2.py starts it) and prints one JSON object of measurements of the robot."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from crab.emotes import STAND  # noqa: E402
from crab.gait import GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES, JOINT_SERVO, PARAMS, SERVOS  # noqa: E402
from robot.backends import SimBackend  # noqa: E402
from robot.servos import SERVO_IDS  # noqa: E402

out = {"model_file": PARAMS.model_file, "scale": PARAMS.scale,
       "trot_period": GAITS["trot"].period, "leg_servo": JOINT_SERVO["femur"], "arm_servo": JOINT_SERVO["claw"],
       "servos": sorted(SERVOS)}

# the committed model is what the generator makes now
sys.argv = ["build_model.py"]
sys.path.insert(0, str(ROOT / "sim"))
import build_model  # noqa: E402

xml = build_model.build()
out["model_is_current"] = (ROOT / "sim" / "models" / PARAMS.model_file).read_text(encoding="utf-8") == xml
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
out["mass"] = float(sum(m.body_mass))
out["njnt"], out["nu"] = int(m.njnt), int(m.nu)
leg_servo_mass = sum(float(m.body_mass[m.body(f"{lg}_{k}_link").id]) for lg in ("FL", "FR", "RL", "RR")
                     for k in ("coxa", "femur", "tibia"))
out["leg_mass"] = leg_servo_mass
out["min_inertia"] = float(min(m.body_inertia[1:].min(), 1.0))
out["stand_z"] = float(m.key("stand").qpos[2])


def run_backend(realism, servo, cmds, seconds=4.0):
    hw = SimBackend(realism=realism, servo_model=servo)

    def send(q):
        hw.write_positions({SERVO_IDS[j]: hw.sign[j] * q[i] for i, j in enumerate(JOINT_NAMES)})
    for _ in range(500):                                         # stand 5 s
        send(STAND)
        hw.wait(0.01)
    res = {"stand_z": float(hw.d.qpos[2]), "stand_up": float(hw.d.xmat[1].reshape(3, 3)[2, 2]),
           "stand_xy": float(np.hypot(*hw.d.qpos[:2]))}
    gait = Gait(GAITS["trot"], com_offset=(hw.d.subtree_com[1] - hw.d.xpos[1])[:2])
    for label, cmd in cmds:
        p0 = hw.d.qpos[:2].copy()
        zmin, upmin = 9.0, 1.0
        for _ in range(int(seconds / 0.01)):
            send(gait.step(0.01, cmd))
            hw.wait(0.01)
            zmin, upmin = min(zmin, float(hw.d.qpos[2])), min(upmin, float(hw.d.xmat[1].reshape(3, 3)[2, 2]))
        R = hw.d.xmat[1].reshape(3, 3)
        res[label] = {"v_body": (R.T @ np.append(hw.d.qpos[:2] - p0, 0.0))[:2].tolist(),
                      "speed": float(np.linalg.norm(hw.d.qpos[:2] - p0) / seconds), "min_z": zmin, "min_up": upmin}
    res["bus_v"], res["pack_v"] = float(hw.voltage), float(hw.battery)
    res["drive"] = getattr(hw.drive, "kind", None)
    return res


FWD, SIDE = ("forward", (0.05, 0.0, 0.0)), ("sideways", (0.0, 0.05, 0.0))
out["ideal"] = run_backend("ideal", "first_order", [FWD, SIDE])
out["datasheet"] = run_backend("datasheet", "first_order", [FWD, SIDE])
try:
    import bam  # noqa: F401
    out["bam"] = run_backend("ideal", "bam", [FWD, SIDE])
    out["bam_datasheet"] = run_backend("datasheet", "bam", [FWD], seconds=3.0)
except ImportError:
    out["bam"] = None
print("PROBE" + json.dumps(out))
