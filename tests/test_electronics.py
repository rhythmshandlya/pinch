"""The electronics model (configs/electronics.toml, crab/electronics.py, SimBackend realism="datasheet"): the data
file is well formed, and every model does what the file says. The default realism="ideal" is pinned by test_golden."""

import math
import tempfile
from pathlib import Path

import numpy as np

from crab.electronics import SOURCES, Data, Distance, Imu, Power
from crab.params import JOINT_NAMES
from pinch.memory import Memory
from pinch.twin import PinchSim
from robot.backends import SimBackend
from robot.crab_robot import Crab, SafetyStop
from robot.servos import SERVO_IDS


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_every_value_has_a_source_and_a_reference():
    data = Data()
    assert data.problems() == [], "\n".join(data.problems())
    params = data.parameters()
    assert len(params) > 50
    assert {p["source"] for _, p in params} == set(SOURCES), "the file should use all three kinds of source"
    for part in ("leg_servo", "arm_servo", "imu", "distance", "power_monitor", "touch", "camera", "microphones",
                 "battery", "regulator"):
        assert part in data.tree, f"the electronics file lacks {part}"


def test_the_checker_catches_bad_entries():
    bad = ('[x]\na = {value = 1, unit = "V", source = "datasheet"}\n'
           'b = {value = 1, unit = "V", source = "guess", ref = "https://example.com"}\n'
           'c = {value = 1, unit = "V", source = "estimate", ref = "no/such/file.md", why = "because I said so, ok"}\n'
           'd = {value = 1, unit = "V", source = "estimate", ref = "hardware/BOM.md"}\n')
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "bad.toml"
        f.write_text(bad)
        msgs = Data(f).problems()
    assert any(m.startswith("x.a: no ref") for m in msgs)
    assert any(m.startswith("x.b: source") for m in msgs)
    assert any(m.startswith("x.c: ref") for m in msgs)
    assert any(m.startswith("x.d: an estimate needs") for m in msgs)


def test_ideal_is_the_default_and_changes_nothing():
    sim = PinchSim(memory=Memory(path=None), seed=1)
    assert sim.realism == "ideal" and sim.hw.elec is None
    gyro, _ = sim.hw.read_imu()
    assert np.array_equal(gyro, sim.d.qvel[3:6])
    try:
        SimBackend(realism="perfect")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown realism must be refused")


def test_a_step_command_reaches_the_joint_after_the_configured_delay():
    hw = SimBackend(realism="datasheet", seed=0)
    data, dt = hw.elec.data, hw.m.opt.timestep
    leg, arm = "FL_femur", "L_shoulder_pitch"
    for j, key in ((leg, "leg_servo"), (arm, "arm_servo")):
        i = JOINT_NAMES.index(j)
        hw.write_positions({SERVO_IDS[j]: hw.cmd.get(j, 0.0) + 0.3})
        new = float(hw._target[i])                                 # the joint-frame target behind that command
        seen = None
        for k in range(1, 40):
            hw.wait(dt)
            if abs(hw.d.ctrl[hw.act[j]] - new) < 1e-9:
                seen = k
                break
        assert seen is not None
        assert seen == hw.elec.delay.n[i] + 1, (j, seen, hw.elec.delay.n[i])
        configured = data(f"{key}.command_delay_s")
        late = (seen - 1) * dt
        assert abs(late - hw.elec.delays[i]) <= dt / 2 + 1e-9
        assert hw.elec.delays[i] >= configured, f"{j}: file says {configured * 1000:.1f} ms, the bus only adds to it"
    assert hw.elec.delays[JOINT_NAMES.index(arm)] > hw.elec.delays[JOINT_NAMES.index(leg)]


def test_leg_encoder_readings_are_multiples_of_the_step():
    hw = SimBackend(realism="datasheet", seed=0)
    hw.wait(0.1)
    step = 2 * math.pi / hw.elec.data("leg_servo.encoder_steps_per_rev")
    legs = [s for s in hw.read_state().values() if s.feedback]
    assert len(legs) == 12
    for s in legs:
        assert abs(s.position / step - round(s.position / step)) < 1e-6
    ideal = SimBackend()
    assert not all(abs(s.position / step - round(s.position / step)) < 1e-6
                   for s in ideal.read_state().values() if s.feedback)


def test_imu_noise_bias_rate_and_lag():
    data = Data()
    rate = data("imu.fusion_rate_hz")
    imu = Imu(data, _rng(5))
    g0, _ = imu.read(0.0, np.zeros(3), np.array([0, 0, -1.0]))
    assert np.array_equal(g0, imu.read(0.002, np.ones(3), np.array([1.0, 0, 0]))[0]), \
        "one sample per 10 ms: reads inside it match"
    imu = Imu(data, _rng(5))
    gyro, grav = [], []
    for k in range(4000):
        g, v = imu.read(k / rate, np.zeros(3), np.array([0, 0, -1.0]))
        gyro.append(g)
        grav.append(v)
    gyro, grav = np.array(gyro), np.array(grav)
    sigma = math.radians(data("imu.gyro_noise_rms_dps"))
    assert np.allclose(gyro.std(0), sigma, rtol=0.15)
    bias_cap = math.radians(data("imu.gyro_zero_rate_offset_dps")) * data("imu.offset_left_after_calibration")
    assert np.all(np.abs(gyro.mean(0)) <= bias_cap + 4 * sigma / math.sqrt(4000))
    assert np.degrees(np.arccos(np.clip(-grav[:, 2], -1, 1))).mean() < 2.5      # a tilt bias of a degree or so at most
    # lag: after a sudden tip the gravity vector needs about the fusion lag to follow
    imu = Imu(data, _rng(5))
    tip = np.array([0, np.sin(0.5), -np.cos(0.5)])
    imu.read(0.0, np.zeros(3), np.array([0, 0, -1.0]))
    first = None
    for k in range(1, 60):
        _, v = imu.read(k / rate, np.zeros(3), tip)
        if first is None and np.arccos(np.clip(v @ tip, -1, 1)) < 0.5 * 0.37:
            first = k / rate
    lag = data("imu.fusion_lag_s") + data("imu.i2c_read_delay_s")
    assert first is not None and 0.5 * lag < first < 4 * lag, first


def test_distance_beam_noise_quantisation_rate_and_dropouts():
    data = Data()
    dist = Distance(data, _rng(2))
    rate = data("distance.rate_hz")
    near = [dist.read("front", 0.05, k / rate) for k in range(600)]
    assert all(v == -1 or abs(v / 0.001 - round(v / 0.001)) < 1e-6 for v in near), "1 mm steps"
    ok = np.array([v for v in near if v > 0])
    assert abs(ok.mean() - 0.05) < 0.003 and 0.0005 < ok.std() < 0.004
    assert (len(near) - len(ok)) / len(near) < 0.03, "a good surface almost never drops out"
    assert dist.read("front", 0.9, 599 / rate) == near[-1], "the same 33 ms sample is returned until the next one"
    far = [dist.read("rear", 0.9, k / rate) for k in range(2000)]
    p_far = np.mean([v < 0 for v in far])
    assert abs(p_far - dist.dropout_probability(0.9)) < 0.05 and p_far > 0.1, p_far
    assert dist.read("side", 3.0, 0.0) == -1, "beyond the sensor's reach: no answer"
    assert dist.read("side", -1.0, 1.0) == -1, "nothing hit: no answer"


def test_a_long_walk_lowers_the_battery_and_the_ina219_reads_it():
    hw = SimBackend(realism="datasheet", seed=3)
    crab = Crab(hw)
    crab.start()
    soc0 = hw.elec.power.soc
    crab.walk((0.05, 0.0, 0.0), 8.0)
    p = hw.elec.power
    assert p.soc < soc0 - 0.003
    assert hw.battery_current() > 0.5, "walking draws amps from the pack"
    assert p.v_pack < p.open_circuit() - 0.05, "the pack sags under load by I x internal resistance"
    volts, amps = hw.battery_voltage(), hw.battery_current()
    assert 6.0 < volts < 8.5
    assert abs(amps / 0.001 - round(amps / 0.001)) < 1e-6, "current in 10 uV / 10 mOhm = 1 mA steps"
    assert 0 <= hw.battery_fraction() <= 1
    ideal = SimBackend()
    assert ideal.battery_current() is None and ideal.battery_voltage() == ideal.battery


def test_bus_voltage_sags_with_servo_current_and_the_regulator_limit():
    data = Data()
    p = Power(data, _rng(0))
    p.step(0.0, 0.2)
    light = p.v_bus
    p.step(0.0, 3.0)
    heavy = p.v_bus
    assert light > heavy > 4.5, (light, heavy)
    p.step(0.0, 8.0)                                           # past the regulator's 5 A
    assert p.v_bus < heavy - 2.0
    low = Power(data, _rng(0), soc=0.0)
    low.step(0.0, 1.0)
    assert low.v_bus < 5.0 - 0.3 or low.v_pack < 6.8, "an empty pack cannot hold the 5 V bus"
    s = Power(data, _rng(0), soc=1.0)
    for _ in range(100):
        s.step(10.0, 1.0)
    assert 0 < s.soc < 1.0


def test_an_empty_pack_stops_the_robot():
    hw = SimBackend(realism="datasheet", seed=0, soc=0.0)
    crab = Crab(hw)
    try:
        crab.start()
        for _ in range(250):
            crab.tick(crab.q_cmd)
    except SafetyStop as e:
        assert e.kind == "battery", e.kind
    else:
        raise AssertionError("a flat pack under load should trigger the battery safety stop")


def _run(realism, seed=3, seconds=20.0):
    sim = PinchSim(memory=Memory(path=None), seed=seed, realism=realism)
    sim.say("hi pinch, I'm Sam")
    sim.run(seconds)
    return sim


def test_datasheet_realism_replays_exactly():
    a, b = _run("datasheet"), _run("datasheet")
    assert np.array_equal(a.d.qpos, b.d.qpos)
    assert a.hw.elec.power.soc == b.hw.elec.power.soc
    assert [str(m) for _, m in a.motions] == [str(m) for _, m in b.motions]
    assert a.cliff_mm() == b.cliff_mm()
    c = _run("datasheet", seed=4)
    assert not np.array_equal(a.d.qpos, c.d.qpos)


def test_datasheet_differs_from_ideal_and_drains_the_battery_in_the_twin():
    d, i = _run("datasheet"), _run("ideal")
    assert not np.array_equal(d.d.qpos, i.d.qpos)
    assert d.hw.elec.power.soc < 1.0 and d.mind.world.battery < 1.0
    assert i.mind.world.battery == 1.0 and i.hw.elec is None
    beams = d.cliff_mm()
    assert all(v == 9999 or v is None or abs(v - round(v)) < 1e-6 for v in beams.values()), "beams read whole mm"


def test_a_low_battery_reaches_the_mind():
    sim = PinchSim(memory=Memory(path=None), seed=2, realism="datasheet", soc=0.15)
    sim.run(5.0)
    assert sim.mind.world.battery < 0.3, sim.mind.world.battery
