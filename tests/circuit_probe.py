"""Runs inside a process started with CRAB_DESIGN=v2 (tests/test_circuit.py starts it) and prints one JSON object of
measurements of the v2 circuit simulation in the twin. The design is read once at import, so it needs its own process."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from crab.electronics import Layout  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.params import CLIFF_BEAMS, JOINT_NAMES, PARAMS  # noqa: E402
from robot.backends import SimBackend  # noqa: E402
from robot.servos import SERVO_IDS  # noqa: E402

lay = Layout()
out = {}
out["beams"] = [[n, x, y, z, list(d)] for n, x, y, z, d in CLIFF_BEAMS]
out["chassis"] = dict(half=list(PARAMS.body.half_size), pos=list(PARAMS.body.shell_center), exp=list(PARAMS.body.shell_exp))
m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
import re  # noqa: E402

xml = (ROOT / "sim" / "models" / PARAMS.model_file).read_text(encoding="utf-8")
out["part_mass_in_model"] = float(sum(float(v) for v in re.findall(r'name="part_[a-z0-9_]+"[^>]*? mass="([0-9.e+-]+)"', xml)))
out["part_mass_in_layout"] = lay.total_mass_g(lambda p: not p.in_model) / 1000.0
out["has_face"] = bool(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "face_screen") >= 0
                       and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "face_tex") >= 0)
used = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(i)) for i in m.geom_matid if i >= 0}
out["has_eyes"] = bool(used & {"eye", "pupil"})
imu = lay["imu"]
sid = m.site("imu").id


def send(hw, q):
    hw.write_positions({SERVO_IDS[j]: hw.sign[j] * q[i] for i, j in enumerate(JOINT_NAMES)})


def stand(hw, seconds):
    for _ in range(int(seconds / 0.01)):
        send(hw, STAND)
        hw.wait(0.01)


def trace(seed):
    hw = SimBackend(realism="datasheet", seed=seed)
    rows = []
    for _ in range(100):
        send(hw, STAND)
        hw.wait(0.01)
        rows.append([float(hw.circuit.i_pack), float(hw.circuit.v_servo.mean()), float(hw.d.qpos[2]), float(hw.battery_current())])
    return rows


hw = SimBackend(realism="datasheet", seed=3)
out["site_pos_ok"] = bool(np.allclose(m.site_pos[sid], imu.pos / 1000.0, atol=1e-6))
q = np.zeros(4)
mujoco.mju_mat2Quat(q, imu.R.ravel())
out["site_quat_dot"] = float(abs(q @ m.site_quat[sid]))
stand(hw, 2.0)
out["stand_up"] = float(hw.d.xmat[1].reshape(3, 3)[2, 2])
gyro, grav = hw.read_imu()
out["grav"] = [float(v) for v in grav]
c = hw.circuit
out["stand"] = c.summary(2.0)
out["ina"] = [float(hw.battery_voltage()), float(hw.battery_current())]
out["i_pack"] = float(c.i_pack)
out["v_servo_report"] = [float(s.voltage) for s in hw.read_state().values() if s.feedback]

out["replay"] = trace(5) == trace(5)
out["replay_other_seed_differs"] = trace(5) != trace(6)

# a servo's torque limit follows the voltage at its terminals
hw2 = SimBackend(realism="datasheet", seed=0)
a = hw2.drive.leg_act
hw2.drive.set_supply(np.full(12, 6.0))
f6 = float(hw2.m.actuator_forcerange[a[0], 1])
hw2.drive.set_supply(np.full(12, 8.4))
f84 = float(hw2.m.actuator_forcerange[a[0], 1])
out["force_limit_ratio"] = f84 / f6

# brown-out: a pack so low that the regulator modules (7 V listed minimum) stop; the computer resets and ignores commands
hw3 = SimBackend(realism="datasheet", seed=0, soc=0.03, enforce_module_min=True)
stand(hw3, 0.5)
before = dict(hw3.cmd)
hw3.write_positions({1: 0.3})
out["brownout_resets"] = hw3.circuit.resets
out["brownout_down"] = not hw3.circuit.computer_up
out["brownout_ignored_command"] = dict(hw3.cmd) == before
out["brownout_events"] = [e[1] for e in hw3.circuit.events]

# without the 7 V rule the same pack keeps the computer up
hw4 = SimBackend(realism="datasheet", seed=0, soc=0.03, enforce_module_min=False)
stand(hw4, 0.5)
out["no_rule_resets"] = hw4.circuit.resets

# the lumped model is still there
hw5 = SimBackend(realism="datasheet", seed=0, power_model="lumped")
stand(hw5, 0.3)
out["lumped_has_no_circuit"] = hw5.circuit is None and hw5.elec.power.v_bus > 6.0
print("PROBE" + json.dumps(out))
