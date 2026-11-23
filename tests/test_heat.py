"""Servo heat: the one-node thermal model (crab/servo_model.py), the twin's servos reporting it, the mind's heat need
and cool-down drive, and the body's heat limit on walking."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crab.params import SERVOS  # noqa: E402
from crab.servo_model import ServoHeat, fit_heat, heat_parameters  # noqa: E402


def test_heat_model_settles_where_the_steady_state_says_and_cools_back():
    h = ServoHeat([SERVOS["ST3215"]])
    p = heat_parameters(SERVOS["ST3215"])
    tau = 1.0                                                  # N*m, held
    for _ in range(int(20 * p["tau"])):                        # 20 time constants
        h.step(1.0, [tau ** 2])
    want = h.steady([tau])
    assert abs(h.T[0] - want[0]) < 0.05, (h.T, want)
    cu = tau ** 2 / p["kt"] ** 2 * p["R_w"] * (1 + p["a_cu"] * (want[0] - 25.0))
    assert abs(want[0] - (p["T_amb"] + p["R_th"] * (cu + p["P_idle"]))) < 1e-6       # the fixed point, tempco included
    assert h.steady([tau], torque_on=False)[0] < want[0]
    hot = h.T[0]
    for _ in range(int(p["tau"])):                             # one time constant with the torque off
        h.step(1.0, [0.0], torque_on=False)
    assert abs((h.T[0] - p["T_amb"]) / (hot - p["T_amb"]) - np.exp(-1)) < 0.01
    assert h.peak[0] == hot


def test_heat_model_matches_the_hold_test_it_was_estimated_from():
    """SO-ARM100 issue 134: an STS3215 holding about 15 kg*cm read 48 C after 10 min and stayed there (25 C assumed)."""
    h = ServoHeat([SERVOS["ST3215"]])
    T10 = h.predict([1.06], 600.0)[0]                          # the motor's share of the hold (BAM's load friction)
    assert 44.0 < T10 < 52.0, T10
    assert h.steady([1.06])[0] - T10 < 3.0                     # nearly settled by 10 min


def test_smaller_servo_runs_hotter_for_the_same_loss():
    big, small = heat_parameters(SERVOS["ST3215"]), heat_parameters(SERVOS["STS3032"])
    assert small["R_th"] > 2 * big["R_th"]
    assert small["tau"] == big["tau"]


def test_fit_heat_recovers_the_constants_of_a_logged_run():
    p = heat_parameters(SERVOS["ST3215"])
    true = dict(p, R_th=7.0, tau=180.0)
    h = ServoHeat(params=[true])
    t, temp, tq = [0.0], [float(h.T[0])], [0.0]
    for k in range(1, 1801):                                   # 15 min at 1.2 N*m, then 15 min cooling with torque on
        load = 1.2 ** 2 if k <= 900 else 0.0
        h.step(1.0, [load])
        t.append(float(k))
        temp.append(float(h.T[0]))
        tq.append(load)
    R, tau, err = fit_heat(t, temp, tq, p)
    assert abs(R - 7.0) < 0.3 and abs(tau - 180.0) < 12.0 and err < 0.3, (R, tau, err)


def test_twin_servos_report_the_heat_model():
    from robot.backends import SimBackend
    from robot.servos import SERVO_IDS
    hw = SimBackend()
    hw.heat.T[:] = 42.4
    st = hw.read_state()
    j = hw.heat_joints[0]
    assert st[SERVO_IDS[j]].temperature == 42.0
    for _ in range(50):                                        # standing holds the body up: the legs warm
        hw.wait(0.02)
    assert hw.heat.T.max() < 42.4                               # (cooling towards the standing temperature)
    hw2 = SimBackend()
    for _ in range(500):
        hw2.wait(0.02)
    assert np.all(hw2.heat.T >= hw2.heat.T_amb) and hw2.heat.T.max() > hw2.heat.T_amb[0] + 0.05


def _noon_mind():
    from pinch.memory import Memory
    from pinch.mind import Mind
    noon = 20776 * 86400 + 12 * 3600
    mem = Memory(path=None)
    mem.data["settings"]["utc_offset_h"] = 0
    m = Mind(memory=mem, llm=False, clock=lambda: noon + m.t)
    return m


def test_heat_need_follows_the_hottest_servo_and_feeds_tiredness():
    m = _noon_mind()
    m.tick(0.1)
    assert m.affect.needs.heat == 0.0                          # nothing reports a temperature: no heat
    for T, want in ((30.0, 0.0), (40.0, 0.0), (50.0, 0.5), (60.0, 1.0), (70.0, 1.0)):
        m.world.servo_temp = T
        m.tick(0.1)
        assert abs(m.affect.needs.heat - want) < 1e-9, (T, m.affect.needs.heat)
    assert m.affect.needs.fatigue >= 1.0 - 1e-9                 # warm motors feel like tired legs


def test_warm_legs_make_it_sit_down_and_cool():
    from pinch.activities import LIBRARY
    m = _noon_mind()
    cool = next(a for a in LIBRARY if a.name == "cool_down")
    m.world.servo_temp = 35.0
    for _ in range(20):
        m.tick(0.5)
    assert cool.utility(m) == 0.0 and m.activity.name != "cool_down"
    m.world.servo_temp = 53.0
    for _ in range(20):
        m.tick(0.5)
    assert m.activity.name == "cool_down", m.activity.name
    assert m.scores["cool_down"] > max(v for k, v in m.scores.items() if k != "cool_down")
    m.request = ("come", None)                                  # its person calls it: at 53 C it stays put
    m._interrupt()
    m.tick(0.5)
    assert m.activity.name == "cool_down", m.activity.name
    m.world.servo_temp = 42.0                                   # cooled: the drive lets go
    for _ in range(80):
        m.tick(0.5)
    assert m.activity.name != "cool_down"


def test_warm_legs_cost_walking_activities_but_not_safety():
    m = _noon_mind()
    m.affect.needs.heat = 0.5
    by = {a.name: a for a in __import__("pinch.activities", fromlist=["LIBRARY"]).LIBRARY}
    assert m._effort(by["explore"]) == 0.5 and m._effort(by["obey"]) == 0.5
    assert m._effort(by["idle"]) == 1.0                         # standing about doesn't walk
    assert m._effort(by["guard_edge"]) == 1.0 and m._effort(by["right_self"]) == 1.0
    m.affect.needs.heat = 0.0
    assert m._effort(by["explore"]) == 1.0


def test_body_slows_walking_from_55_and_stops_it_at_60():
    from pinch.body.locomotion import Locomotion
    b = Locomotion.__new__(Locomotion)
    b.heat_trimmed = 0
    cmd = (0.05, 0.01, 0.4)
    for T, k in ((None, 1.0), (40.0, 1.0), (55.0, 1.0), (57.5, 0.5), (60.0, 0.0), (64.0, 0.0)):
        b.servo_temp = T
        assert np.allclose(b._heat_limit(cmd), np.array(cmd) * k), (T, b._heat_limit(cmd))
    assert b.heat_trimmed == 3
