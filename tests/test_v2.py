"""Design v2, the resized STS3215 robot in the twin (docs/dev/V2_MODEL.md): selected explicitly with CRAB_DESIGN=v2,
today's design untouched. v2 needs its own process (the design is read once at import), so one probe process
(tests/v2_probe.py) measures everything and these tests read its answers."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def probe():
    if "r" not in _CACHE:
        env = dict(os.environ, CRAB_DESIGN="v2")
        p = subprocess.run([sys.executable, str(ROOT / "tests" / "v2_probe.py")], env=env, capture_output=True,
                           text=True, cwd=str(ROOT), timeout=600)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("PROBE")]
        assert line, f"the v2 probe failed:\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}"
        _CACHE["r"] = json.loads(line[0][5:])
    return _CACHE["r"]


def test_todays_design_is_the_default_and_unchanged():
    from crab.params import DESIGN, JOINT_SERVO, PARAMS, SERVOS
    assert DESIGN == "v1" and PARAMS.design == "v1" and PARAMS.model_file == "crab.xml"
    assert "STS3215M6" not in SERVOS and JOINT_SERVO["femur"] == "STS3032"
    assert PARAMS.leg.tibia_len == 0.052 and PARAMS.gait_period_factor == 1.0 and PARAMS.arm_scale == 1.0
    from crab.gait import GAITS
    assert abs(GAITS["trot"].period - 0.9 * 0.52 ** 0.5) < 1e-12


def test_v2_is_selected_explicitly_and_has_the_sizing_study_geometry():
    r = probe()
    assert r["design"] == "v2" and r["model_file"] == "crab_v2.xml"
    assert abs(r["scale"] - 0.8) < 1e-9                                       # style B: tibia 80 mm
    assert r["leg_servo"] == "STS3215M6" and r["arm_servo"] == "STS3032" and "STS3215M6" in r["servos"]
    assert abs(r["trot_period"] - 0.9 * 0.8 ** 0.5 * 1.5) < 1e-9              # the trot slowed 1.5x
    try:
        subprocess.run([sys.executable, "-c", "import crab.params"], env=dict(os.environ, CRAB_DESIGN="v3"),
                       cwd=str(ROOT), check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("an unknown design must be refused")


def test_v2_model_is_current_and_has_the_right_mass():
    r = probe()
    assert r["model_is_current"], "sim/models/crab_v2.xml is stale: python sim/build_model.py --design v2"
    assert r["nu"] == 18 and r["njnt"] == 1 + 2 * 18                          # root + 18 servo joints + 18 gear play
    assert 2.05 < r["mass"] < 2.25, r["mass"]                                 # style B: 2.135 kg with the 300 g screen module (1.985 kg with 150 g)
    assert 0.70 < r["leg_mass"] < 0.85                                        # 4 legs: 12 servos + printed covers + ball feet
    assert r["min_inertia"] > 0.0
    assert 0.052 < r["stand_z"] < 0.068


def _stands_and_walks(res):
    assert res["stand_up"] > 0.999 and res["stand_xy"] < 0.01 and 0.050 < res["stand_z"] < 0.068
    fwd, side = res["forward"], res["sideways"]
    assert fwd["v_body"][0] > 0.04 and abs(fwd["v_body"][1]) < 0.15 * fwd["v_body"][0] + 0.002, fwd
    assert side["v_body"][1] > 0.04 and abs(side["v_body"][0]) < 0.15 * side["v_body"][1] + 0.002, side
    for w in (fwd, side):
        assert w["min_up"] > 0.97 and w["min_z"] > 0.04, w                     # never close to falling over


def test_v2_stands_5_s_and_walks_forward_and_sideways_ideal():
    r = probe()["ideal"]
    assert r["drive"] == "first_order"
    _stands_and_walks(r)


def test_v2_walks_with_datasheet_realism():
    r = probe()["datasheet"]
    _stands_and_walks(r)
    assert 7.5 < r["bus_v"] < 8.4 and 7.5 < r["pack_v"] < 8.5               # the servos run straight from the 2S pack (circuit model)


def test_v2_full_bam_servo_model_stands_and_walks():
    r = probe()
    if r["bam"] is None:
        return                                                                # optional package not installed
    assert r["bam"]["drive"] == "bam"
    _stands_and_walks(r["bam"])
    assert r["bam_datasheet"]["forward"]["v_body"][0] > 0.03
    # the same walk reaches the same speed within 15 % as with the first-order servo
    a, b = r["ideal"]["forward"]["speed"], r["bam"]["forward"]["speed"]
    assert abs(b - a) < 0.15 * a, (a, b)


def test_the_electronics_file_has_the_measured_sts3215_block():
    from crab.electronics import Data
    d = Data()
    assert d.problems() == []
    m6 = "https://github.com/Rhoban/bam/blob/main/bam/params/feetech_sts3215_7_4V/m6.json"
    block = dict(d.parameters())
    for key in ("kt_nm_per_a", "resistance_ohm", "armature_kgm2", "max_velocity_rad_s", "friction_base",
                "friction_viscous", "friction_stribeck", "load_friction_motor", "load_friction_external",
                "dtheta_stribeck", "alpha", "command_delay_s", "stall_torque_nm", "no_load_speed_rad_s"):
        p = block["leg_servo." + key]
        assert p["source"] == "measured" and p["ref"] == m6, key
    vin, kt, R = d("leg_servo.supply_v"), d("leg_servo.kt_nm_per_a"), d("leg_servo.resistance_ohm")
    assert abs(d("leg_servo.stall_torque_nm") - vin * kt / R) < 1e-9          # stall from vin * kt / R
    assert abs(d("leg_servo.stall_torque_nm") - 3.4275) < 1e-3
    assert d("leg_servo.max_velocity_rad_s") == 5.095754044445978
    assert d.for_design("v1") is d
    v2 = d.for_design("v2")
    assert v2("regulator.output_v") == 7.4 and v2("battery.capacity_ah") == 2.5 and d("regulator.output_v") == 5.0
    sizing = json.loads((ROOT / "sim" / "sizing" / "bam_sts3215_m6.json").read_text())      # the BAM file's copy
    from crab.servo_model import measured_parameters
    mp = measured_parameters(d)
    assert all(mp[k] == sizing[k] for k in sizing if k not in ("model", "actuator", "q_offset")), "toml != BAM json"
