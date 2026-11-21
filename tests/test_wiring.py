"""The v2 netlist (configs/wiring_v2.toml) and its checks (crab/wiring.py): the real wiring passes, and a netlist made bad on
purpose fails with the right kind of message. The circuit solver itself is tested here without the twin; the twin's side
(design v2 process) is in test_circuit.py."""

import copy

import numpy as np

from crab.electronics import Data, Layout, Part
from crab.wiring import Circuit, Net, check
from robot.servos import SERVO_IDS


def _fails(net=None, data=None, layout=None):
    return check(net if net is not None else Net(), layout=layout, data=data, servo_ids=SERVO_IDS)


def _edit(fn):
    t = copy.deepcopy(Net().tree)
    fn(t)
    return Net(tree=t)


def test_the_real_wiring_passes_every_check():
    rep = _fails()
    assert rep.ok, "\n".join(f"[{k}] {t}" for k, t in rep.failures)
    assert any("i2c5" in n for n in rep.notes)


def test_every_pin_is_used_once_and_has_the_function_radxa_lists():
    net = Net()
    rep = _fails(net)
    assert len(rep.pins_used) == len(set(rep.pins_used))
    assert net.pins[3] == ["GPIO1_A0", "UART3_RX_M0"]                   # Radxa's table: no I2C function on pins 3 and 5
    assert "pwm" not in net.tree                                          # the arm servos are bus servos: no PWM outputs
    assert not {7, 16, 18, 19, 21, 24} & set(rep.pins_used)               # the six former PWM pins are free
    bad = _edit(lambda t: t["i2c"][2]["devices"][1].update(reset_pin=11))   # two reset lines on one pin
    assert "pin" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][1].update(sda=6))                     # a bus on a ground pin
    assert "pin" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][2]["devices"][1].update(reset_pin=26))      # pin 26 is not connected
    assert "pin" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][2]["devices"][0].update(reset_pin=13))   # a reset line on the XSHUT pin of another sensor
    assert "pin" in _fails(bad).kinds()


def test_the_vl53_sensors_share_0x29_only_through_their_reset_pins():
    net = Net()
    devs = {d["part"]: d for b in net.i2c for d in b["devices"]}
    assert all(devs[p]["boot_addr"] == 0x29 for p in ("tof_fl", "tof_fr", "tof_rear", "tof_left", "tof_right", "tof8"))
    assert len({devs[p]["addr"] for p in devs if "reset_pin" in devs[p]}) == 6
    bad = _edit(lambda t: t["i2c"][2]["devices"][1].pop("reset_pin"))      # one without an XSHUT: two chips at 0x29 at once
    assert "i2c" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][2]["devices"][1].update(addr=0x30))     # two get the same new address
    assert "i2c" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][0]["devices"][1].update(addr=0x40))     # MPR121 cannot sit at the INA219's address
    assert "i2c" in _fails(bad).kinds()
    bad = _edit(lambda t: t["i2c"][0]["devices"].append(dict(part="ina", kind="ina219", addr=0x40)))   # two at 0x40
    assert "i2c" in _fails(bad).kinds()


def test_the_bno055_keeps_its_bus_to_itself():
    net = Net()
    bus = next(b for b in net.i2c if any(d["kind"] == "bno055" for d in b["devices"]))
    assert len(bus["devices"]) == 1 and bus["sda"] == 27 and bus["scl"] == 28
    bad = _edit(lambda t: t["i2c"][1]["devices"].append(dict(part="mpr121", kind="mpr121", addr=0x5A)))
    rep = _fails(bad)
    assert any(k == "i2c" and "stretches" in t for k, t in rep.failures), rep.failures
    bad = _edit(lambda t: t["i2c"][0]["devices"].append(dict(part="imu", kind="bno055", addr=0x28)))
    assert any("stretches" in t for _, t in _fails(bad).failures)


def test_every_load_reaches_a_supply_and_every_part_is_wired():
    bad = _edit(lambda t: t["edge"].__delitem__(next(i for i, e in enumerate(t["edge"]) if e["id"] == "trunk")))
    assert "supply" in _fails(bad).kinds()                              # the hub, and with it twelve servos, hangs in the air
    bad = _edit(lambda t: t["edge"].__delitem__(next(i for i, e in enumerate(t["edge"]) if e["id"] == "v33_bno055")))
    assert "supply" in _fails(bad).kinds()
    L = Layout()
    L.parts["ghost_sensor"] = Part("ghost_sensor", dict(L.parts["amp"].d))
    assert any(k == "coverage" and "ghost_sensor" in x for k, x in _fails(layout=L).failures)   # in the layout, no wire
    bad = _edit(lambda t: t["load"].append(dict(id="ghost", part="warp_drive", node="N_imu", kind="fixed",
                                                current="load.mic_a")))
    assert any(k == "coverage" and "warp_drive" in x for k, x in _fails(bad).failures)


def test_ratings_wire_connector_switch_shunt_regulator_and_charge_path():
    rep = _fails()
    rows = {r[0]: r for r in rep.edge_rows}
    assert rows["trunk"][3] <= 15.0 + 1e-9 and rows["trunk"][4] <= 25.0 + 1e-9      # capped by the fuse and the BMS
    assert rows["cable_FL_coxa"][4] > 3.0                                           # a stall is above the contact's rating
    assert rows["cable_FL_coxa"][4] <= 2 * rows["cable_FL_coxa"][5]                 # but within what a 2 s stall may carry
    bad = _edit(lambda t: next(e for e in t["edge"] if e["id"] == "trunk").update(awg=28))
    assert any(k == "rating" and "trunk" in x for k, x in _fails(bad).failures)
    bad = _edit(lambda t: next(e for e in t["edge"] if e["id"] == "cable_FL_femur").update(connectors=["dupont", "dupont"]))
    assert any(k == "rating" and "cable_FL_femur" in x for k, x in _fails(bad).failures)
    bad = _edit(lambda t: t["edge"].insert(0, dict(id="x", **{"from": "HUB7", "to": "S_FL_coxa"})))
    assert "supply" in _fails(bad).kinds()                                          # two feeds for one node
    d = Data()
    d.tree["shunt"]["ohm"]["value"] = 0.005                                         # the BOM's 0.005 ohm 2512 shunt
    assert any(k == "rating" and "shunt" in x for k, x in _fails(data=d).failures)
    d = Data()
    d.tree["connector"]["pogo"]["rating_a"]["value"] = 0.5
    assert any("charge path" in x for _, x in _fails(data=d).failures)
    assert rows["rega_panel"][3] > 0.4 and rows["rega_panel"][3] <= rows["rega_panel"][5]    # the HDMI panel's 5 V lead
    d = Data()
    d.tree["arm_servo"]["stall_current_a"]["value"] = 3.0                             # a hungrier arm servo overloads rail B
    assert any(k == "arm limit" for k, _ in _fails(data=d).failures)


def test_the_hdmi_panel_replaces_the_oled_and_loads_regulator_a():
    net = Net()
    assert all(d["kind"] != "ssd1309" for b in net.i2c for d in b["devices"]) and "ssd1309" not in net.devices
    assert not any(e["id"] == "v33_oled" for e in net.edges)
    panel = next(x for x in net.loads if x["id"] == "panel")
    assert panel["node"] == "PANEL5" and any(l["kind"] == "hdmi" for l in net.links)
    rep = _fails(net)
    reg_a = next(n for n in rep.notes if n.startswith("reg_a"))
    assert float(reg_a.split()[3]) >= 1.3                                             # computer + audio + sensors + panel
    bad = _edit(lambda t: t["edge"].__delitem__(next(i for i, e in enumerate(t["edge"]) if e["id"] == "rega_panel")))
    assert "supply" in _fails(bad).kinds()                                            # the panel then hangs in the air


def test_six_arm_servos_on_the_bus_and_rail_b_held_by_the_torque_limit():
    net = Net()
    arms = [ld for ld in net.loads if ld["kind"] == "arm_servo"]
    assert [ld["bus_id"] for ld in arms] == list(range(13, 19)) and [SERVO_IDS[ld["joint"]] for ld in arms] == list(range(13, 19))
    assert len([e for e in net.edges if e["id"].startswith("cable_")]) == 18          # one cable per servo
    rep = _fails(net)
    assert rep.ok and any("servo bus: 18 servos" in n for n in rep.notes) and any(n.startswith("rail B") for n in rep.notes)
    d = Data()
    assert float(d("arm_servo.stall_current_a")) * 6 > float(d("regulator_5v.current_limit_a"))   # unlimited: over the 5 A
    d.tree["arm_servo"]["torque_limit_permille"]["value"] = 1000                      # no limit: refused
    assert any(k == "arm limit" for k, _ in _fails(data=d).failures)
    d = Data()
    d.tree["arm_servo"]["torque_limit_permille"]["value"] = 0                         # a limit of zero is nonsense
    assert any(k == "arm limit" for k, _ in _fails(data=d).failures)
    bad = _edit(lambda t: next(x for x in t["load"] if x["id"] == "servo_L_claw").update(bus_id=3))   # clashes with a leg
    assert "servo bus" in _fails(bad).kinds()
    bad = _edit(lambda t: next(x for x in t["load"] if x["id"] == "servo_L_claw").pop("bus_id"))      # still a PWM servo
    assert "servo bus" in _fails(bad).kinds()
    bad = _edit(lambda t: t["rules"].update(servo_bus_baud=57600) or t["servo_bus"].update(baud=57600))   # 18 servos too slow
    assert any(k == "servo bus" and "of the bus" in x for k, x in _fails(bad).failures)


def test_servo_ids_and_the_bus_order_match_the_firmware():
    net = Net()
    order = net.tree["servo_bus"]["order"]
    assert [SERVO_IDS[j] for j in order] == list(range(1, 13))
    bad = _edit(lambda t: t["servo_bus"]["order"].reverse())
    assert "servo bus" in _fails(bad).kinds()
    bad = _edit(lambda t: next(x for x in t["load"] if x["id"] == "servo_FL_femur").update(bus_id=4))
    assert "servo bus" in _fails(bad).kinds()


# -- the circuit solver ---------------------------------------------------------------------------------------------

def _circuit(**kw):
    return Circuit(data=Data(), **kw)


def _servo_load(c, amps_each):
    """Servo coefficients that draw `amps_each` A per leg servo at about 7 V (I = a V)."""
    n = len(c.legs)
    return np.full(n, amps_each / 7.0), np.zeros(n), np.zeros(len(c.arms))


def test_the_circuit_follows_ohms_law_down_the_wires():
    c = _circuit()
    a, b, arm = _servo_load(c, 1.0)
    c.step(0.01, a, b, arm)
    i = c.i_pack
    assert 10.5 < i < 13.5                                   # 12 servos x about 0.9 A plus the computer, regulators, sensors
    # the pack sag is its resistance times the current
    voc = c.open_circuit()
    assert abs((voc - c.v_pack_term) - c.resistance() * i) < 0.02
    # a servo at the end of its 900 mm cable sees less than the hub
    assert c.v_servo.max() < c.V["HUB7"] and c.V["HUB7"] < c.v_pack_term
    drop = c.V["HUB7"] - c.v_servo[0]
    assert 0.12 < drop < 0.25, drop                          # about 0.9 A through 0.19 ohm of cable and two connectors
    # the regulators hold 5 V and carry what the netlist loads them with
    assert abs(c.rail["A"] - 5.0) < 0.1 and abs(c.rail["3V3"] - 3.3) < 1e-9
    assert 0.8 < c.i_rail["A"] < 1.1 and 0.15 < c.i_rail["3V3"] < 0.25      # the HDMI panel adds about 0.48 A to rail A
    # the INA219 sees the whole pack current, where the shunt is wired
    v, amps = c.measure()
    assert abs(amps - i) < 0.05 and abs(v - c.v_ina) < 0.03


def test_input_power_matches_output_power_through_the_regulators():
    c = _circuit()
    c.step(0.01, np.zeros(12), np.zeros(12), np.zeros(6))
    p_out = c.rail["A"] * c.i_rail["A"] + c.rail["B"] * c.i_rail["B"]
    p_in = sum(c.v_reg_in[k] * c.i_reg_in[k] for k in "AB")
    assert 0.55 < p_out / p_in < 0.95                        # light-load efficiency, not the 90 % peak
    c.activity = "npu"
    c.step(0.01, np.zeros(12), np.zeros(12), np.zeros(6))
    assert c.i_rail["A"] > 0.7                               # the computer asks for more on the NPU than idle


def test_the_fuse_blows_on_sustained_overcurrent_and_not_on_a_short_peak():
    c = _circuit()
    a, b, arm = _servo_load(c, 2.15)                         # about 22 A: 1.5 x the rating
    t = 0.0
    while not c.fuse_blown and t < 120.0 and not c.bms_off:
        c.step(0.05, a, b, arm)
        t += 0.05
    assert c.fuse_blown and not c.bms_off and 3.0 < t < 40.0, (t, c.events)
    c.step(0.05, a, b, arm)
    assert c.dead and c.v_servo.max() == 0.0 and not c.computer_up
    assert any("fuse blown" in e for _, e in c.events)
    c = _circuit()                                           # 17 A for half a second (a short stall): the fuse lives
    a, b, arm = _servo_load(c, 1.7)
    for _ in range(10):
        c.step(0.05, a, b, arm)
    assert not c.fuse_blown and 0 < c.fuse_damage < 1.0 and 10 < c.i_pack < 20
    for _ in range(400):                                     # and cools down
        c.step(0.05, np.zeros(12), np.zeros(12), np.zeros(6))
    assert c.fuse_damage < 1e-3


def test_the_bms_cuts_a_short_circuit_before_the_fuse():
    c = _circuit()
    a, b, arm = _servo_load(c, 3.3)                          # about 40 A
    for _ in range(10):
        c.step(0.05, a, b, arm)
    assert c.bms_off and c.dead and any("BMS" in e for _, e in c.events)


def test_brown_out_resets_the_computer_when_the_regulator_input_falls_below_7_v():
    c = _circuit(soc=0.03, enforce_module_min=True)          # the module's listed 7 V minimum
    c.step(0.01, np.zeros(12), np.zeros(12), np.zeros(6))
    for _ in range(5):
        c.step(0.01, np.zeros(12), np.zeros(12), np.zeros(6))
    assert c.resets == 1 and not c.computer_up and c.v_computer < 4.5
    assert any("brown-out" in e for _, e in c.events)
    c2 = _circuit(soc=0.03, enforce_module_min=False)        # the chip itself (3.8 V minimum) keeps on regulating
    for _ in range(6):
        c2.step(0.01, np.zeros(12), np.zeros(12), np.zeros(6))
    assert c2.resets == 0 and c2.computer_up and c2.stats["below_module_min_s"] > 0


def test_the_computer_boots_again_after_the_power_comes_back():
    c = _circuit(soc=1.0)
    a, b, arm = _servo_load(c, 0.2)
    # a long stall drags the rail down for a moment
    c.computer_up, c.resets, c._boot_left = False, 1, 3.0
    for _ in range(40):
        c.step(0.1, a, b, arm)
    assert c.computer_up and c.restarts == 1 and any("up again" in e for _, e in c.events)


def test_servo_voltage_sags_with_the_pack_and_with_load():
    full, flat = _circuit(soc=1.0), _circuit(soc=0.15)
    a, b, arm = _servo_load(full, 1.0)
    full.step(0.01, a, b, arm)
    flat.step(0.01, a, b, arm)
    assert full.v_servo.mean() > flat.v_servo.mean() + 0.8
    assert flat.v_servo.mean() < 6.0 and full.v_servo.mean() > 6.8


def test_the_circuit_is_deterministic():
    runs = []
    for _ in range(2):
        c = _circuit(rng=np.random.default_rng(4))
        a, b, arm = _servo_load(c, 0.7)
        for k in range(30):
            c.step(0.01, a * (1 + 0.1 * np.sin(k)), b, arm)
        runs.append((c.i_pack, c.soc, c.measure(), tuple(c.v_servo)))
    assert runs[0] == runs[1]


def test_hardware_electronics_md_agrees_with_the_netlist():
    import re
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "hardware" / "electronics.md").read_text(encoding="utf-8")
    rep = _fails()
    for pin in rep.pins_used:
        assert re.search(rf"\b{pin}\b", text), f"pin {pin} ({rep.pins_used[pin]}) is not in hardware/electronics.md"
    net = Net()
    for bus in net.i2c:
        for d in bus["devices"]:
            if d["kind"] not in ("vl53l0x", "vl53l5cx"):          # those get 0x30-0x35 from the software, listed as a range
                assert f"0x{d['addr']:02X}" in text, d
    assert "wiring_v2.toml" in text and "STS3032" in text and "HDMI" in text and "0x30-0x34" in text
    assert "PWM15_IR_M1" not in text and "SSD1309" not in text         # no PWM arms, no OLED any more
