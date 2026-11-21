"""The v2 robot's circuit simulation in the twin (SimBackend(realism="datasheet") under CRAB_DESIGN=v2): the model is built from the
layout, the sensors sit at their mounts, the servos run from the voltage at their own terminals, a brown-out resets the
computer, and everything replays exactly. Design v2 needs its own process (tests/circuit_probe.py)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def probe():
    if "r" not in _CACHE:
        env = dict(os.environ, CRAB_DESIGN="v2")
        p = subprocess.run([sys.executable, str(ROOT / "tests" / "circuit_probe.py")], env=env, capture_output=True,
                           text=True, cwd=str(ROOT), timeout=900)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("PROBE")]
        assert line, f"the circuit probe failed:\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}"
        _CACHE["r"] = json.loads(line[0][5:])
    return _CACHE["r"]


def test_the_model_is_built_from_the_layout():
    from crab.electronics import Layout
    r = probe()
    assert abs(r["part_mass_in_model"] - r["part_mass_in_layout"]) < 1e-4, (r["part_mass_in_model"], r["part_mass_in_layout"])
    ch = Layout().chassis
    assert np.allclose(np.array(r["chassis"]["half"]) * 1000, ch["box_half_mm"], atol=1e-6)
    assert np.allclose(np.array(r["chassis"]["pos"]) * 1000, ch["box_center_mm"], atol=1e-6)
    assert np.allclose(np.array(r["chassis"]["dome_half"]) * 1000, ch["dome_half_mm"], atol=1e-6)
    assert r["site_pos_ok"] and r["site_quat_dot"] > 0.999999                  # the IMU site is where the layout puts the chip


def test_the_distance_beams_start_at_their_placed_windows():
    from crab.electronics import Layout
    L = Layout()
    r = probe()
    names = {"front_left": "tof_fl", "front_right": "tof_fr", "rear": "tof_rear", "side_left": "tof_left",
             "side_right": "tof_right"}
    assert {b[0] for b in r["beams"]} == set(names)
    for n, x, y, z, d in r["beams"]:
        pos, direction = L[names[n]].window_pose()
        assert np.allclose([x, y, z], pos, atol=0.01) and np.allclose(d, direction, atol=0.002), n
        assert direction[2] < -0.4                                               # looking out and down


def test_the_face_is_todays_visor_and_there_are_no_googly_eyes():
    r = probe()
    assert r["has_face"] and not r["has_eyes"]


def test_standing_on_the_circuit_model_numbers():
    r = probe()
    assert r["stand_up"] > 0.999
    s = r["stand"]
    assert 0.8 < s["i_mean"] < 1.6 and s["v_servo_min"] > 7.8 and not s["fuse_blown"] and s["resets"] == 0
    assert abs(r["ina"][1] - r["i_pack"]) < 0.06                                 # the INA219 reads the shunt current
    assert abs(r["ina"][0] - 8.3) < 0.2
    assert all(7.8 < v < 8.5 for v in r["v_servo_report"]) and len(r["v_servo_report"]) == 12
    assert r["grav"][2] < -0.99                                                  # the IMU, in body axes, feels gravity down


def test_the_circuit_replays_exactly_and_the_seed_matters():
    r = probe()
    assert r["replay"] and r["replay_other_seed_differs"]


def test_a_servos_torque_limit_follows_its_voltage():
    r = probe()
    assert abs(r["force_limit_ratio"] - 8.4 / 6.0) < 1e-9


def test_a_brown_out_resets_the_computer_and_it_stops_talking_to_the_servos():
    r = probe()
    assert r["brownout_resets"] >= 1 and r["brownout_down"] and r["brownout_ignored_command"]
    assert any("brown-out" in e for e in r["brownout_events"])
    assert r["no_rule_resets"] == 0                                              # without the 7 V rule the chip carries on


def test_the_lumped_model_is_still_available_for_comparison():
    assert probe()["lumped_has_no_circuit"]


def test_the_twin_restarts_the_mind_from_memory_after_a_brown_out():
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    sim = PinchSim(memory=Memory(path=None), seed=1)
    old_mind, mem = sim.mind, sim.mind.memory
    sim.hw.circuit = SimpleNamespace(restarts=0, computer_up=True)      # a stand-in for the circuit of design v2
    try:
        assert sim._computer_down() is False and sim.mind is old_mind        # nothing happened
        sim.hw.circuit.computer_up = False
        assert sim._computer_down() is True and sim.mind is old_mind         # booting: the loop is down, no new mind yet
        sim.hw.circuit.restarts, sim.hw.circuit.computer_up = 1, True
        assert sim._computer_down() is False
        assert sim.mind is not old_mind and sim.mind.memory is mem           # back up: a fresh mind, the same memory
        assert sim._computer_down() is False and sim._restarts_seen == 1     # once per restart
    finally:
        sim.hw.circuit = None
