"""The v2 robot's wiring as data (configs/wiring_v2.toml), the checks that keep it honest, and the circuit it makes.

    python -m crab.wiring            # run every check, print the table of wires, exit code 1 if anything fails

Part 1, the netlist: `Net(path)` loads the file, `check(net)` returns a `Report` of failures. It fails on
  pin          a header pin used twice, a signal on a power / ground / unconnected pin, or on a pin that does not have that
               function in Radxa's table
  i2c          an address clash on a bus (the VL53L0X / VL53L5CX that all wake at 0x29 are brought up one at a time through
               their XSHUT / LPn pins and given their own addresses: that sequence is modelled), a device that cannot have
               the address given, a clock-stretching chip (BNO055) that shares its bus, a bus more than i2c_max_utilisation busy
  supply       a load that no supply reaches, a converter with no input, a node that hangs in the air
  rating       a wire, connector or switch below the current it can see (sustained and a short stall, see below), the shunt
               over its power rating, a regulator or the Radxa's 3.3 V pins loaded beyond their limit
  coverage     a part of the layout with no wire (or a wire to a part that is not in the layout)
  servo bus    IDs (legs 1-12, arms 13-18) or header order that do not match robot/servos.py, a stub longer than the bus allows,
               a bus that cannot read all 18 servos at the control rate, and rail B: the six STS3032 arm servos would draw
               more than the 5 A regulator if they all stalled at full torque, so a torque limit (electronics.toml
               arm_servo.torque_limit_permille, written into each servo by the firmware's setup) must hold their stalled
               current inside the regulator; the check refuses a missing or too-high limit
Worst-case currents: a leg servo can stall (its locked-rotor current at the full pack) for the 2 s its overload protection
allows, and hold the maker's printed stall torque for longer; everything upstream of the fuse is capped by the fuse's rating
(sustained) and the BMS trip current (a stall). A wire or contact must carry the sustained figure at its rating and the stall
figure at `ratings.transient_factor` times its rating.

Part 2, the circuit: `Circuit` solves the power network of the netlist every control tick: pack (cells, internal resistance,
state of charge) -> fuse (it blows on sustained overcurrent, following its time-current curve) -> shunt -> switch -> the
7.4 V servo rail at every servo after the wire and connector drops, and the two 5 V regulators (efficiency, current limit,
dropout, the module's 7 V minimum) feeding the computer, the sensors, the audio and the arm servos. Each leg servo's
current comes from its measured motor model (voltage, back-EMF, winding resistance). The computer's rail dropping below its
minimum resets it (brown-out). The INA219 reads the current in the shunt where the netlist wires it.
Everything is deterministic: no random numbers beyond the INA219's seeded gain error and noise.
Depends on crab/electronics.py only."""

import math
import sys
import tomllib
from pathlib import Path

import numpy as np

from .electronics import ROOT, Data, Layout, quantise

NET_PATH = ROOT / "configs" / "wiring_v2.toml"
POWER_PINS = {1, 2, 4, 6, 9, 14, 17, 20, 25, 30, 34, 39}
NC_PINS = {26}


# -- the netlist --------------------------------------------------------------------------------------------------

class Net:
    def __init__(self, path=None, tree=None):
        self.path = Path(path) if path else NET_PATH
        if tree is None:
            with open(self.path, "rb") as f:
                tree = tomllib.load(f)
        self.tree = tree
        self.rules = tree.get("rules", {})
        self.edges = list(tree.get("edge", []))
        self.loads = list(tree.get("load", []))
        self.converters = list(tree.get("converter", []))
        self.i2c = list(tree.get("i2c", []))
        self.links = list(tree.get("link", []))
        self.charge = list(tree.get("charge_edge", []))
        self.pins = {int(k): v for k, v in tree["radxa"]["pin"].items()}
        self.devices = tree.get("device", {})

    def copy_with(self, **changes):
        """A copy of the netlist with some top-level sections replaced (tests use this to break it on purpose)."""
        import copy
        t = copy.deepcopy(self.tree)
        t.update(changes)
        return Net(self.path, t)


class Report:
    def __init__(self):
        self.failures = []
        self.notes = []

    def fail(self, kind, text):
        self.failures.append((kind, text))

    @property
    def ok(self):
        return not self.failures

    def kinds(self):
        return {k for k, _ in self.failures}


# -- geometry and electrical numbers of an edge ----------------------------------------------------------------------

class Electrical:
    """Resistance, length and ratings of the edges, from the netlist, the layout and configs/electronics.toml."""

    def __init__(self, net, data=None, layout=None):
        self.net = net
        self.d = data if isinstance(data, Data) else Data(data)
        self.layout = layout if layout is not None else Layout()
        self.slack = float(self.d("ratings.slack_factor"))
        self.loop = float(self.d("ratings.service_loop_mm"))
        self.transient = float(self.d("ratings.transient_factor"))

    def length_mm(self, e):
        v = e.get("length_mm", "auto")
        if v != "auto":
            return float(v)
        parts = e.get("parts") or [e.get("from_part"), e.get("to_part")]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return 100.0
        a, b = self.layout[parts[0]].pos, self.layout[parts[1]].pos
        return float(np.abs(a - b).sum()) * self.slack + self.loop

    def wire_ohm(self, e):
        if "awg" not in e:
            return 0.0
        par = float(e.get("parallel", 1))
        return 2.0 * float(self.d(f"awg.a{e['awg']}.ohm_per_m")) * self.length_mm(e) / 1000.0 / par

    def connector_ohm(self, e):
        par = float(e.get("parallel", 1))
        return sum(2.0 * float(self.d(f"connector.{c}.contact_mohm")) / 1000.0 for c in e.get("connectors", [])) / par

    def element_ohm(self, e):
        el = e.get("element")
        if el == "fuse":
            return float(self.d("fuse.resistance_mohm")) / 1000.0
        if el == "switch":
            return float(self.d("switch.contact_mohm")) / 1000.0
        if el == "shunt":
            return float(self.d("shunt.ohm"))
        return 0.0

    def resistance(self, e):
        return self.wire_ohm(e) + self.connector_ohm(e) + self.element_ohm(e)

    def rating(self, e):
        """The lowest continuous current rating on the edge, and what limits it."""
        par = float(e.get("parallel", 1))
        best = (1e9, "")
        if "awg" in e:
            best = (float(self.d(f"awg.a{e['awg']}.amps")) * par, f"{e['awg']} AWG wire")
        for c in e.get("connectors", []):
            r = float(self.d(f"connector.{c}.rating_a")) * par
            if r < best[0]:
                best = (r, f"{c} connector")
        if e.get("element") == "switch":
            best = (float(self.d("switch.rating_a")), "the switch")
        return best


# -- worst-case currents -----------------------------------------------------------------------------------------------

def arm_torque_limit(d):
    """The torque limit written into every arm servo, as a fraction of its maximum (0-1)."""
    return float(d("arm_servo.torque_limit_permille")) / 1000.0


def arm_stalled_a(d):
    """Current of one arm servo stalled against its torque limit: the motor current follows the torque, so the stall
    current at full torque times the limit (idle current excluded)."""
    return float(d("arm_servo.stall_current_a")) * arm_torque_limit(d)


class Currents:
    """The sustained and the stall (transient) current every load can draw, at the load's own supply, and what that is at
    the pack for converters. Pure bookkeeping over the netlist tree."""

    def __init__(self, net, el):
        self.net, self.el, self.d = net, el, el.d
        self.v_max = float(self.d("battery_v2.cells")) * 4.2
        self.v_min = float(net.rules.get("pack_min_v", 6.4))
        self.children = {}
        for e in net.edges:
            self.children.setdefault(e["from"], []).append(e)
        self.conv_in = {c["in_node"]: c for c in net.converters}
        self.loads_at = {}
        for ld in net.loads:
            self.loads_at.setdefault(ld["node"], []).append(ld)
        self.cap_sustained = float(self.d("fuse.rating_a"))
        self.cap_transient = float(self.d("battery_v2.bms_trip_a"))

    def load(self, ld):
        d, k = self.d, ld["kind"]
        if k == "leg_servo":
            return float(d("leg_servo.sustained_current_a")), self.v_max / float(d("leg_servo.resistance_ohm"))
        if k == "arm_servo":
            a = arm_stalled_a(d)                            # a stalled STS3032 at the firmware's torque limit
            return a, a
        if k == "computer":
            a = float(d("computer.power_npu_w")) / float(d("computer.min_input_v"))
            return a, a
        if k == "amp":
            a = float(d("load.amp_peak_w")) / 5.0 + float(d("load.amp_idle_a"))
            return a, a
        a = float(d(ld["current"]))
        return a, a

    def node(self, n, _seen=None):
        """(sustained, transient) current into node n from above, summed over everything below it (uncapped)."""
        s = t = 0.0
        for ld in self.loads_at.get(n, []):
            a, b = self.load(ld)
            s, t = s + a, t + b
        for e in self.children.get(n, []):
            a, b = self.node(e["to"])
            s, t = s + a, t + b
        c = self.conv_in.get(n)
        if c is not None:
            a, b = self.node(c["out_node"])
            s2, t2 = self.convert(c, a, b)
            s, t = s + s2, t + t2
        return s, t

    def convert(self, c, a_out, b_out):
        m = c["model"]
        if m == "regulator_5v":
            v, eta = float(self.d("regulator_5v.output_v")), float(self.d("regulator_5v.peak_efficiency")) * 0.95
            iq = float(self.d("regulator_5v.quiescent_a"))
            vin = self.v_min
            return v * a_out / (eta * vin) + iq, v * b_out / (eta * vin) + iq
        eta = float(self.d("computer.rail_3v3_efficiency"))                 # rail_3v3: 3.3 V from the Radxa's 5 V
        return 3.3 * a_out / (eta * 4.5), 3.3 * b_out / (eta * 4.5)

    def edge(self, e):
        """(sustained, transient) through an edge, with the fuse and the BMS capping what the pack can push."""
        s, t = self.node(e["to"])
        return min(s, self.cap_sustained), min(t, self.cap_transient)


# -- the checks ---------------------------------------------------------------------------------------------------------

def _hex(v):
    return f"0x{int(v):02X}"


def check_pins(net, rep):
    uses = []                                                    # (pin, function wanted or None, who)
    for bus in net.i2c:
        hw = bus["controller"].startswith("I2C")
        for role, pin in (("SDA", bus["sda"]), ("SCL", bus["scl"])):
            fn = f"{bus['controller']}".replace("I2C4_M0", "I2C4").replace("I2C5_M0", "I2C5")
            want = f"{fn[:4]}_{role}_M0" if hw else "GPIO"
            uses.append((pin, want, f"{bus['name']} {role}"))
        for dev in bus["devices"]:
            if "reset_pin" in dev:
                uses.append((dev["reset_pin"], "GPIO", f"{dev['part']} reset"))
    i2s = net.tree.get("i2s")
    if i2s:
        for key, fn in (("bclk", "I2S3_SCLK_M0"), ("lrck", "I2S3_LRCK_M0"), ("data_out", "I2S3_SDO_M0"),
                        ("data_in", "I2S3_SDI_M0")):
            uses.append((i2s[key], fn, f"i2s {key}"))
    seen = {}
    for pin, want, who in uses:
        if pin not in net.pins:
            rep.fail("pin", f"{who}: pin {pin} is not on the 40-pin header")
            continue
        if pin in POWER_PINS or pin in NC_PINS:
            rep.fail("pin", f"{who}: pin {pin} is a power / ground / unconnected pin ({net.pins[pin]})")
        if want and not (want == "GPIO" and any(f.startswith("GPIO") for f in net.pins[pin])) and want not in net.pins[pin]:
            rep.fail("pin", f"{who}: pin {pin} has no {want} (Radxa lists {net.pins[pin]})")
        if pin in seen:
            rep.fail("pin", f"pin {pin} is used twice: {seen[pin]} and {who}")
        seen[pin] = who
    rep.pins_used = seen


def check_i2c(net, rep):
    for bus in net.i2c:
        devs = bus["devices"]
        always, resettable = [], []
        for dev in devs:
            kind = net.devices.get(dev["kind"])
            if kind is None:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} is a {dev['kind']}, which has no entry in [device]")
                continue
            boot = dev.get("boot_addr", dev["addr"])
            if boot not in kind["addresses"]:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} wakes at {_hex(boot)}, a {dev['kind']} wakes at "
                                f"{[_hex(a) for a in kind['addresses']]}")
            if dev["addr"] != boot and not kind.get("readdressable"):
                rep.fail("i2c", f"{bus['name']}: {dev['part']} ({dev['kind']}) cannot be given another address")
            if not 0x08 <= dev["addr"] <= 0x77:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} address {_hex(dev['addr'])} is reserved")
            (resettable if "reset_pin" in dev else always).append((dev, boot))
        # the devices without a reset pin sit on their addresses all the time
        sitting = {}
        for dev, boot in always:
            for a in {dev["addr"], boot}:
                if a in sitting:
                    rep.fail("i2c", f"{bus['name']}: {dev['part']} and {sitting[a]} both answer at {_hex(a)}")
                sitting[a] = dev["part"]
        # the ones with a reset pin are brought up one at a time: its boot address must be free then, its final one always
        final = {}
        for dev, boot in resettable:
            if boot in sitting:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} wakes at {_hex(boot)} where {sitting[boot]} already sits")
            if dev["addr"] in sitting:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} is given {_hex(dev['addr'])}, taken by {sitting[dev['addr']]}")
            if dev["addr"] in final:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} and {final[dev['addr']]} are both given {_hex(dev['addr'])}")
            final[dev["addr"]] = dev["part"]
        # devices that wake at the same address without a reset pin cannot be told apart
        boots = {}
        for dev in devs:
            b = dev.get("boot_addr", dev["addr"])
            boots.setdefault(b, []).append(dev)
        for b, group in boots.items():
            if len(group) > 1 and any("reset_pin" not in x for x in group):
                rep.fail("i2c", f"{bus['name']}: {[x['part'] for x in group]} all wake at {_hex(b)} and not each has a reset pin")
        # a clock-stretching chip gets a bus of its own
        for dev in devs:
            if net.devices.get(dev["kind"], {}).get("stretches_clock") and len(devs) > 1:
                rep.fail("i2c", f"{bus['name']}: {dev['part']} ({dev['kind']}) stretches the clock and shares the bus with "
                                f"{[x['part'] for x in devs if x is not dev]}: it would block them")
        used = sum(net.devices.get(dev["kind"], {}).get("bytes_per_s", 0) + 0 for dev in devs)
        util = used * 9.0 / (float(bus["speed_khz"]) * 1000.0)
        rep.notes.append(f"{bus['name']}: {len(devs)} devices, {util:.0%} busy")
        if util > float(net.rules.get("i2c_max_utilisation", 0.6)):
            rep.fail("i2c", f"{bus['name']} is {util:.0%} busy (limit {float(net.rules.get('i2c_max_utilisation', 0.6)):.0%})")


def check_supply(net, rep):
    nodes = {"PACK"}
    parents = {}
    for e in net.edges:
        nodes |= {e["from"], e["to"]}
        if e["to"] in parents:
            rep.fail("supply", f"node {e['to']} is fed by two edges ({parents[e['to']]} and {e['id']}): the netlist is a tree")
        parents[e["to"]] = e["id"]
    out_nodes = {c["out_node"]: c for c in net.converters}
    in_nodes = {c["in_node"]: c for c in net.converters}
    reach, todo = set(), ["PACK"]
    kids = {}
    for e in net.edges:
        kids.setdefault(e["from"], []).append(e["to"])
    while todo:
        n = todo.pop()
        if n in reach:
            continue
        reach.add(n)
        todo += kids.get(n, [])
        if n in in_nodes:
            todo.append(in_nodes[n]["out_node"])
    for ld in net.loads:
        if ld["node"] not in reach:
            rep.fail("supply", f"load {ld['id']} sits on node {ld['node']}, which no supply reaches")
    for c in net.converters:
        if c["in_node"] not in reach:
            rep.fail("supply", f"converter {c['id']}: its input node {c['in_node']} is not powered")
    for n in nodes:
        if n not in reach and n not in out_nodes:
            rep.fail("supply", f"node {n} is not connected to the pack")
    rep.reach = reach


def check_ratings(net, rep, el):
    cur = Currents(net, el)
    rep.edge_rows = []
    for e in net.edges:
        try:
            s, t = cur.edge(e)
        except RecursionError:
            rep.fail("supply", "the power netlist has a loop")
            return
        rate, what = el.rating(e)
        rep.edge_rows.append((e["id"], el.length_mm(e) if "awg" in e else 0.0, el.resistance(e), s, t, rate, what))
        if s > rate + 1e-9:
            rep.fail("rating", f"{e['id']}: carries {s:.2f} A sustained, its {what} is rated {rate:.2f} A")
        elif t > rate * el.transient + 1e-9:
            rep.fail("rating", f"{e['id']}: a stall puts {t:.2f} A through it, its {what} is rated {rate:.2f} A "
                               f"(x{el.transient:g} for a stall)")
        if e.get("element") == "shunt":
            p_w = float(el.d("shunt.power_w"))
            if s * s * float(el.d("shunt.ohm")) > p_w + 1e-9:
                rep.fail("rating", f"{e['id']}: the {float(el.d('shunt.ohm')) * 1000:.0f} mOhm shunt would burn "
                                   f"{s * s * float(el.d('shunt.ohm')):.2f} W at {s:.1f} A, it is rated {p_w:g} W")
    for e in net.charge:
        i = float(el.d(e["current"]))
        rate, what = el.rating(e)
        if i > rate + 1e-9:
            rep.fail("rating", f"charge path {e['id']}: {i:.1f} A through a {what} rated {rate:.1f} A")
    for c in net.converters:
        a, b = cur.node(c["out_node"])
        if c["model"] == "regulator_5v":
            lim = float(el.d("regulator_5v.current_limit_a"))
            rep.notes.append(f"{c['id']}: worst-case load {a:.2f} A of {lim:g} A")
            if a > lim:
                rep.fail("rating", f"{c['id']}: its loads can draw {a:.2f} A, the regulator gives {lim:g} A")
        else:
            lim = float(el.d("computer.rail_3v3_limit_a"))
            rep.notes.append(f"{c['id']}: worst-case load {a:.3f} A of {lim:g} A")
            if a > lim:
                rep.fail("rating", f"the Radxa's 3.3 V pins would carry {a:.2f} A, their budget is {lim:g} A")
    rep.currents = cur


def check_coverage(net, rep, layout):
    named = set()
    for sec in (net.edges, net.loads, net.converters, net.links, net.charge):
        for x in sec:
            for k in ("part", "from_part", "to_part"):
                if x.get(k):
                    named.add(x[k])
            named |= set(x.get("parts", []))
    for bus in net.i2c:
        named |= {d["part"] for d in bus["devices"]}
    i2s = net.tree.get("i2s", {})
    named |= {x["part"] for x in i2s.get("out", []) + i2s.get("in", [])}
    exempt = set(net.tree.get("coverage", {}).get("exempt", []))
    for name in layout.parts:
        if name not in named and name not in exempt:
            rep.fail("coverage", f"{name} is in the layout but has no wire in the netlist")
    for name in sorted(named):
        if name not in layout.parts:
            rep.fail("coverage", f"{name} is in the netlist but not in the layout")


def check_bus(net, rep, el, servo_ids=None):
    """servo_ids: the firmware's joint -> ID table (robot/servos.py SERVO_IDS). crab/ may not import robot/, so the
    tests pass it in."""
    sb = net.tree.get("servo_bus", {})
    SERVO_IDS = servo_ids
    order = sb.get("order", [])
    legs = [ld for ld in net.loads if ld["kind"] == "leg_servo"]
    if sorted(order) != sorted(ld["joint"] for ld in legs):
        rep.fail("servo bus", "the bus order does not list exactly the leg servos of the loads")
    ids = [ld["bus_id"] for ld in legs]
    if len(set(ids)) != len(ids):
        rep.fail("servo bus", f"servo IDs are not unique: {sorted(ids)}")
    for ld in legs:
        if order and ld["joint"] in order and order.index(ld["joint"]) + 1 != ld["bus_id"]:
            rep.fail("servo bus", f"{ld['joint']} is header {order.index(ld['joint']) + 1} but has ID {ld['bus_id']}")
        if SERVO_IDS is not None and SERVO_IDS.get(ld["joint"]) != ld["bus_id"]:
            rep.fail("servo bus", f"{ld['joint']}: ID {ld['bus_id']} here, {SERVO_IDS.get(ld['joint'])} in robot/servos.py")
    if sb and int(sb.get("baud", 0)) != int(net.rules.get("servo_bus_baud", 1000000)):
        rep.fail("servo bus", f"the bus runs at {sb.get('baud')} baud, the rules say {net.rules.get('servo_bus_baud')}")
    by_edge = {e["to"]: e for e in net.edges}
    for ld in legs:
        e = by_edge.get(ld["node"])
        if e is None or el.length_mm(e) > float(net.rules.get("servo_bus_max_stub_mm", 1500)):
            rep.fail("servo bus", f"{ld['joint']}: its cable is missing or longer than the stub limit")
    # the six arm servos: bus servos too, IDs 13-18 in the order of arm_order, on the 5 V hub headers
    arms = [ld for ld in net.loads if ld["kind"] == "arm_servo"]
    arm_order = sb.get("arm_order", [])
    if sorted(arm_order) != sorted(ld["joint"] for ld in arms):
        rep.fail("servo bus", "arm_order does not list exactly the arm servos of the loads")
    arm_ids = [ld.get("bus_id") for ld in arms]
    if None in arm_ids:
        rep.fail("servo bus", "an arm servo has no bus_id: they are bus servos (IDs 13-18), not PWM outputs")
    elif len(set(arm_ids)) != len(arm_ids) or set(arm_ids) & set(ids):
        rep.fail("servo bus", f"arm servo IDs {sorted(arm_ids)} are not unique or clash with the legs {sorted(ids)}")
    for ld in arms:
        if ld.get("bus_id") is None:
            continue
        if arm_order and ld["joint"] in arm_order and arm_order.index(ld["joint"]) + 13 != ld["bus_id"]:
            rep.fail("servo bus", f"{ld['joint']} is arm header {arm_order.index(ld['joint']) + 1} but has ID {ld['bus_id']}")
        if servo_ids is not None and servo_ids.get(ld["joint"]) != ld["bus_id"]:
            rep.fail("servo bus", f"{ld['joint']}: ID {ld['bus_id']} here, {servo_ids.get(ld['joint'])} in robot/servos.py")
        e = by_edge.get(ld["node"])
        if e is None or el.length_mm(e) > float(net.rules.get("servo_bus_max_stub_mm", 1500)):
            rep.fail("servo bus", f"{ld['joint']}: its cable is missing or longer than the stub limit")
    # one bus, 1 Mbaud: every servo read once per control cycle must fit with room to spare
    n = len(legs) + len(arms)
    cycle_s = n * float(net.rules.get("servo_bus_exchange_bytes", 30)) * 10.0 / float(net.rules.get("servo_bus_baud", 1000000))
    util = cycle_s * float(net.rules.get("servo_bus_cycle_hz", 50))
    rep.notes.append(f"servo bus: {n} servos, {cycle_s * 1000:.1f} ms per read cycle, {util:.0%} busy at "
                     f"{net.rules.get('servo_bus_cycle_hz', 50)} Hz")
    if util > float(net.rules.get("servo_bus_max_utilisation", 0.6)):
        rep.fail("servo bus", f"{n} servos take {util:.0%} of the bus at {net.rules.get('servo_bus_cycle_hz', 50)} Hz "
                              f"(limit {float(net.rules.get('servo_bus_max_utilisation', 0.6)):.0%})")
    # rail B: the arm servos' stall current without a limit would exceed the regulator; the torque limit must hold it
    if arms:
        d = el.d
        lim = arm_torque_limit(d)
        reg = float(d("regulator_5v.current_limit_a"))
        free = len(arms) * float(d("arm_servo.stall_current_a"))
        held = len(arms) * arm_stalled_a(d) + len(arms) * float(d("arm_servo.idle_current_a"))
        rep.notes.append(f"rail B: {len(arms)} arm servos stalled at full torque would draw {free:.1f} A; with the "
                         f"{lim:.0%} torque limit {held:.2f} A of {reg:g} A")
        if not 0.0 < lim <= 1.0:
            rep.fail("arm limit", f"the arm torque limit {lim:.2f} is not between 0 and 1")
        elif free > reg and lim >= 1.0:
            rep.fail("arm limit", f"{len(arms)} arm servos stalled draw {free:.1f} A against regulator B's {reg:g} A and no "
                                  f"torque limit is set (arm_servo.torque_limit_permille)")
        elif held > reg:
            rep.fail("arm limit", f"even at a {lim:.0%} torque limit the arm servos can draw {held:.2f} A against "
                                  f"regulator B's {reg:g} A: lower the limit")


def check(net=None, layout=None, data=None, servo_ids=None):
    """Run every check of the netlist; returns a Report."""
    net = net if net is not None else Net()
    layout = layout if layout is not None else Layout()
    el = Electrical(net, data, layout)
    rep = Report()
    check_pins(net, rep)
    check_i2c(net, rep)
    check_supply(net, rep)
    if "supply" not in rep.kinds():
        check_ratings(net, rep, el)
    check_coverage(net, rep, layout)
    check_bus(net, rep, el, servo_ids)
    return rep


def table(net=None, layout=None, data=None):
    net = net if net is not None else Net()
    el = Electrical(net, data, layout)
    rep = Report()
    check_supply(net, rep)
    check_ratings(net, rep, el)
    rows = [f"{'wire':26s} {'mm':>5s} {'mOhm':>6s} {'sustained A':>11s} {'stall A':>8s} {'rated A':>8s}  limited by"]
    for i, ln, r, s, t, rate, what in rep.edge_rows:
        rows.append(f"{i:26s} {ln:5.0f} {r * 1000:6.1f} {s:11.2f} {t:8.2f} {rate:8.1f}  {what}")
    return "\n".join(rows)


# -- the circuit ----------------------------------------------------------------------------------------------------------

def _interp(x, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.interp(x, xs, ys))


class Circuit:
    """The power network of the netlist, solved once per control tick.

    step(dt, a, b, arm_amps, torque_on) with, for the 12 leg servos in the order of `legs`:
        a, b   the servo draws  I = max(0, a * V - b) + idle  from the voltage V at its terminals, because its motor model
               gives I_motor = (duty * V - kt * w) / R and the supply current is duty * I_motor:
               a = mean(duty^2) / R, b = mean(duty * kt * w) / R over the tick
        arm_amps  the six arm servos' current at 5 V (from their torque)
    Results: v_servo[12], v_arm[6], i_pack, v_ina (what the INA219's bus input sees), i_shunt, rail voltages, soc, ...
    and the events: "fuse blown", "bms trip", "brown-out", "restart".
    """

    ITER = 10

    def __init__(self, net=None, data=None, layout=None, rng=None, soc=1.0, activity="camera", enforce_module_min=None):
        self.net = net if net is not None else Net()
        self.d = data if isinstance(data, Data) else Data(data)
        self.el = Electrical(self.net, self.d, layout)
        d = self.d
        b = "battery_v2"
        self.cells = int(d(f"{b}.cells"))
        self.cap_as = float(d(f"{b}.capacity_ah")) * float(d(f"{b}.usable_fraction")) * 3600.0
        self.ocv = np.asarray(d(f"{b}.ocv_per_cell_v"), float)
        self.r0 = float(d(f"{b}.internal_resistance_ohm"))
        self.r_rise = float(d(f"{b}.resistance_rise_at_empty"))
        self.bms_a, self.bms_s = float(d(f"{b}.bms_trip_a")), float(d(f"{b}.bms_trip_s"))
        self.soc = float(soc)
        # regulators
        r5 = "regulator_5v"
        self.reg = dict(v=float(d(f"{r5}.output_v")), limit=float(d(f"{r5}.current_limit_a")),
                        mod_min=float(d(f"{r5}.module_min_input_v")), chip_min=float(d(f"{r5}.chip_min_input_v")),
                        drop=float(d(f"{r5}.dropout_v")), eff=list(d(f"{r5}.efficiency_points")),
                        iq=float(d(f"{r5}.quiescent_a")), droop=float(d(f"{r5}.overload_droop_ohm")),
                        hiccup=float(d(f"{r5}.hiccup_factor")))
        self.enforce = bool(d(f"{r5}.enforce_module_min")) if enforce_module_min is None else bool(enforce_module_min)
        # computer
        self.pc = {"idle": float(d("computer.power_idle_w")), "camera": float(d("computer.power_camera_w")),
                   "npu": float(d("computer.power_npu_w"))}
        self.activity = activity
        self.v_min_computer = float(d("computer.min_input_v"))
        self.hold_up, self.boot_s = float(d("computer.hold_up_s")), float(d("computer.boot_s"))
        self.eta33 = float(d("computer.rail_3v3_efficiency"))
        self.v_set_rail3 = 3.3
        # the monitor
        self.q_bus, self.q_shunt = float(d("power_monitor.bus_voltage_step_v")), float(d("power_monitor.shunt_voltage_step_v"))
        self.shunt_noise = float(d("power_monitor.shunt_noise_v"))
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.gain = 1.0 + self.rng.uniform(-1, 1) * float(d("power_monitor.gain_error_frac"))
        # the fuse
        self.fuse_a = float(d("fuse.rating_a"))
        self.fuse_curve = [(float(x), float(y)) for x, y in d("fuse.curve")]
        self.fuse_cool = float(d("fuse.cooling_s"))
        self.fuse_damage, self.fuse_blown, self.bms_off, self._bms_t = 0.0, False, False, 0.0
        # topology
        self._build()
        self.servo_idle = float(d("leg_servo.idle_current_a"))
        self.t = 0.0
        self.acc = dict(a=np.zeros(len(self.legs)), b=np.zeros(len(self.legs)), arm=np.zeros(len(self.arms)), t=0.0)
        self.events = []
        self.i_pack = self.i_shunt = 0.0
        self.v_servo = np.full(len(self.legs), 7.4)
        self.v_arm = np.full(len(self.arms), 5.0)
        self.v_ina = self.v_pack_term = self.open_circuit()
        self.rail = {"A": 5.0, "B": 5.0, "3V3": 3.3}
        self.i_rail = {"A": 0.0, "B": 0.0, "3V3": 0.0}
        self.i_reg_in = {"A": 0.0, "B": 0.0}
        self.v_reg_in = {"A": 7.4, "B": 7.4}
        self.v_computer = 5.0
        self.p_computer = self.pc[activity]
        self.computer_up, self._low_s, self._boot_left = True, 0.0, 0.0
        self.resets = self.restarts = 0
        self.hiccup = {"A": 0.0, "B": 0.0}
        self.stats = dict(i_peak=0.0, i_sum=0.0, n=0, v_servo_min=99.0, v_arm_min=99.0, v_pack_min=99.0,
                          v_computer_min=99.0, load_a_max=0.0, load_b_max=0.0, below_module_min_s=0.0,
                          below_servo_spec_s=0.0, below_computer_s=0.0, i_3v3_max=0.0)
        self.step(0.0, np.zeros(len(self.legs)), np.zeros(len(self.legs)), np.zeros(len(self.arms)), True)

    # -- topology -------------------------------------------------------------------------------------------------
    def _build(self):
        net, el = self.net, self.el
        self.legs = [ld for ld in net.loads if ld["kind"] == "leg_servo"]
        self.arms = [ld for ld in net.loads if ld["kind"] == "arm_servo"]
        for i, ld in enumerate(self.legs):
            ld["_idx"] = i
        for i, ld in enumerate(self.arms):
            ld["_idx"] = i
        self.edges = {e["to"]: e for e in net.edges}
        self.R = {e["to"]: el.resistance(e) for e in net.edges}
        self.kids = {}
        for e in net.edges:
            self.kids.setdefault(e["from"], []).append(e["to"])
        self.loads_at = {}
        for ld in net.loads:
            self.loads_at.setdefault(ld["node"], []).append(ld)
        self.conv = {c["in_node"]: c for c in net.converters}
        self.conv_out = {c["out_node"]: c for c in net.converters}
        # order of nodes below each root, parents first
        self.order = {}
        for root in ["PACK"] + [c["out_node"] for c in net.converters]:
            seq, todo = [], [root]
            while todo:
                n = todo.pop(0)
                seq.append(n)
                todo += self.kids.get(n, [])
            self.order[root] = seq
        self.shunt_node = next(e["to"] for e in net.edges if e.get("element") == "shunt")
        self.fuse_node = next(e["to"] for e in net.edges if e.get("element") == "fuse")
        self.leg_nodes = [ld["node"] for ld in self.legs]
        self.arm_nodes = [ld["node"] for ld in self.arms]
        self.node_set = set(self.R) | {"PACK"} | set(self.conv_out)
        self.fixed = {}
        for ld in net.loads:
            if ld["kind"] == "fixed":
                self.fixed[ld["id"]] = float(self.d(ld["current"]))

    def open_circuit(self):
        x = np.linspace(1.0, 0.0, len(self.ocv))
        return float(np.interp(np.clip(self.soc, 0, 1), x[::-1], self.ocv[::-1])) * self.cells

    def resistance(self):
        return self.r0 * (1.0 + (self.r_rise - 1.0) * (1.0 - float(np.clip(self.soc, 0, 1))) ** 2)

    # -- one regulator ------------------------------------------------------------------------------------------------
    def _regulator(self, key, vin, iout):
        """Output volts of a 5 V module from its input volts and output current; and its input current."""
        r = self.reg
        if self.hiccup[key] > 0 or vin < r["chip_min"] or (self.enforce and vin < r["mod_min"]):
            return 0.0, r["iq"] * (vin > r["chip_min"])
        v = min(r["v"], vin - r["drop"])
        if iout > r["limit"]:
            v -= r["droop"] * (iout - r["limit"])
        v = max(v, 0.0)
        eta = max(0.3, _interp(max(iout, 0.05), r["eff"]))
        return v, v * iout / (eta * max(vin, 1.0)) + r["iq"]

    def _rail_loads(self, root, v_rail, ctx):
        """Bottom-up current on the tree below `root` (a rail or the pack side), returns {node: current into its edge}."""
        cur = {}
        for n in reversed(self.order[root]):
            i = 0.0
            for ld in self.loads_at.get(n, []):
                i += self._load_current(ld, ctx.get(n, v_rail))
            for k in self.kids.get(n, []):
                i += cur[k]
            c = self.conv.get(n)
            if c is not None:
                i += ctx["conv_in"][c["id"]]
            cur[n] = i
        return cur

    def _load_current(self, ld, v):
        k = ld["kind"]
        if k == "leg_servo":
            i = self._servo_a[ld["_idx"]] * v - self._servo_b[ld["_idx"]]
            return max(0.0, i) + self.servo_idle * (v > 1.0)
        if k == "arm_servo":
            return float(self._arm_a[ld["_idx"]]) * (v > 1.0)
        if k == "computer":
            return (self.p_computer / max(v, 2.5)) if v > 1.0 else 0.0
        if k == "amp":
            return (float(self.d("load.amp_idle_a")) + self._amp_w / max(v, 2.5)) if v > 1.0 else 0.0
        return self.fixed[ld["id"]] if v > 1.0 else 0.0

    # -- the solve ------------------------------------------------------------------------------------------------------
    def step(self, dt, a, b, arm_amps, torque_on=True):
        self._servo_a = np.where(torque_on, np.asarray(a, float), 0.0)
        self._servo_b = np.where(torque_on, np.asarray(b, float), 0.0)
        self._arm_a = np.where(torque_on, np.asarray(arm_amps, float), 0.0) + float(self.d("arm_servo.idle_current_a"))
        self._amp_w = 0.0
        if dt > 0:                                            # running means of the servo coefficients (for battery runtime)
            acc = self.acc
            acc["a"] += dt * self._servo_a
            acc["b"] += dt * self._servo_b
            acc["arm"] += dt * self._arm_a
            acc["t"] += dt
        self.t += dt
        dead = self.fuse_blown or self.bms_off
        voc, rint = self.open_circuit(), self.resistance()
        V = {n: (0.0 if dead else voc) for n in self.node_set}
        if not dead:
            for c in self.net.converters:
                V[c["out_node"]] = self.reg["v"] if c["model"] == "regulator_5v" else self.v_set_rail3
                for n in self.order[c["out_node"]][1:]:
                    V[n] = V[c["out_node"]]
        iters = 1 if dead else self.ITER
        p_computer = self.pc[self.activity] if self.computer_up or self._boot_left > 0 else self.pc["idle"]
        self.p_computer = p_computer
        i_total = 0.0
        for it in range(iters):
            conv_in = {}
            # the 3.3 V rail first (it hangs under the Radxa node, which hangs under rail A)
            for c in self.net.converters:
                if c["model"] == "rail_3v3":
                    root = c["out_node"]
                    ctx = {"conv_in": {}, **{n: V[n] for n in self.order[root]}}
                    rc = self._rail_loads(root, V[root], ctx)
                    i3 = rc[root]
                    v_in = V[c["in_node"]]
                    ok = v_in > 3.6 and self.computer_up
                    v33 = 3.3 if ok else 0.0
                    V[root] = v33
                    conv_in[c["id"]] = (3.3 * i3 / (self.eta33 * max(v_in, 1.0))) if ok else 0.0
                    self.i_rail["3V3"] = i3 if ok else 0.0
            for c in self.net.converters:
                if c["model"] != "regulator_5v":
                    continue
                key = "A" if c["id"].endswith("a") else "B"
                root = c["out_node"]
                ctx = {"conv_in": {x["id"]: conv_in.get(x["id"], 0.0) for x in self.net.converters},
                       **{n: V[n] for n in self.order[root]}}
                rc = self._rail_loads(root, V[root], ctx)
                iout = rc[root]
                v_in = V[c["in_node"]]
                if (iout > self.reg["limit"] * self.reg["hiccup"]) and self.hiccup[key] <= 0 and it == iters - 1:
                    self.hiccup[key] = 0.1
                    self.events.append((self.t, f"regulator {key} overload, shut down for 0.1 s"))
                v_out, i_in = self._regulator(key, v_in, iout)
                V[root] = v_out
                conv_in[c["id"]] = i_in
                self.i_rail[key] = iout
                self.i_reg_in[key] = i_in
                self.v_reg_in[key] = v_in
            # 2. the pack side bottom-up, then voltages top-down
            ctx = {"conv_in": conv_in, **{n: V[n] for n in self.order["PACK"]}}
            cur = self._rail_loads("PACK", 0.0, ctx)
            i_total = cur["PACK"]
            if dead:
                i_total = 0.0
            i_total = min(i_total, voc / (2 * rint))
            V["PACK"] = 0.0 if dead else voc - rint * i_total
            for n in self.order["PACK"][1:]:
                e = self.edges[n]
                Vn = V[e["from"]] - cur[n] * self.R[n]
                V[n] = 0.7 * Vn + 0.3 * V[n] if it < iters - 1 else Vn
            # rails below converters: top-down
            for c in self.net.converters:
                root = c["out_node"]
                ctx = {"conv_in": {x["id"]: conv_in.get(x["id"], 0.0) for x in self.net.converters},
                       **{n: V[n] for n in self.order[root]}}
                rc = self._rail_loads(root, V[root], ctx)
                for n in self.order[root][1:]:
                    e = self.edges[n]
                    V[n] = max(0.0, V[e["from"]] - rc[n] * self.R[n])
        # -- results
        self.V = V
        self.i_pack = float(i_total)
        self.v_pack_term = float(V["PACK"])
        self.i_shunt = float(i_total)
        self.v_ina = float(V["SHUNT_IN"])
        self.v_servo = np.array([V[n] for n in self.leg_nodes])
        self.v_arm = np.array([V[n] for n in self.arm_nodes])
        self.rail["A"], self.rail["B"] = float(V[self.conv_out_by("reg_a")]), float(V[self.conv_out_by("reg_b")])
        self.rail["3V3"] = float(V[self.conv_out_by("radxa_3v3")])
        self.v_computer = float(V["RADXA5"])
        self._housekeeping(dt, i_total, voc, dead)
        return self

    def conv_out_by(self, cid):
        return next(c["out_node"] for c in self.net.converters if c["id"] == cid)

    # -- slow things: charge, fuse, BMS, brown-out ---------------------------------------------------------------------
    def _housekeeping(self, dt, i, voc, dead):
        if dt <= 0:
            return
        self.soc = max(0.0, self.soc - i * dt / self.cap_as)
        for k in ("A", "B"):
            if self.hiccup[k] > 0:
                self.hiccup[k] = max(0.0, self.hiccup[k] - dt)
        # the fuse: damage adds up as dt / time-to-open at this current, and cools off below the rating
        if not dead:
            ratio = i / self.fuse_a
            if ratio > self.fuse_curve[0][0]:
                pts = [(math.log(x), math.log(y)) for x, y in self.fuse_curve]
                t_open = math.exp(float(np.interp(math.log(ratio), [p[0] for p in pts], [p[1] for p in pts])))
                self.fuse_damage += dt / t_open
            elif ratio < 1.0:
                self.fuse_damage = max(0.0, self.fuse_damage * math.exp(-dt / self.fuse_cool))
            if self.fuse_damage >= 1.0:
                self.fuse_blown = True
                self.events.append((self.t, f"fuse blown at {i:.1f} A"))
            if i > self.bms_a:
                self._bms_t += dt
                if self._bms_t >= self.bms_s:
                    self.bms_off = True
                    self.events.append((self.t, f"BMS over-current cut-off at {i:.1f} A"))
            else:
                self._bms_t = 0.0
        # the computer: below its minimum for longer than its hold-up time it resets, then boots
        s = self.stats
        if self.v_computer < self.v_min_computer:
            self._low_s += dt
            s["below_computer_s"] += dt
            if self.computer_up and self._low_s >= self.hold_up:
                self.computer_up = False
                self.resets += 1
                self._boot_left = self.boot_s
                self.events.append((self.t, f"brown-out: the computer's rail fell to {self.v_computer:.2f} V, reset"))
        else:
            self._low_s = 0.0
            if not self.computer_up:
                self._boot_left -= dt
                if self._boot_left <= 0:
                    self.computer_up = True
                    self.restarts += 1
                    self.events.append((self.t, "the computer is up again"))
        if dead:
            self.computer_up = False
        # statistics for reports
        s["n"] += 1
        s["i_sum"] += i
        s["i_peak"] = max(s["i_peak"], i)
        s["v_servo_min"] = min(s["v_servo_min"], float(self.v_servo.min()))
        s["v_arm_min"] = min(s["v_arm_min"], float(self.v_arm.min()))
        s["v_pack_min"] = min(s["v_pack_min"], self.v_pack_term)
        s["v_computer_min"] = min(s["v_computer_min"], self.v_computer)
        s["load_a_max"] = max(s["load_a_max"], self.i_rail["A"])
        s["load_b_max"] = max(s["load_b_max"], self.i_rail["B"])
        s["i_3v3_max"] = max(s["i_3v3_max"], self.i_rail["3V3"])
        if self.v_reg_in["A"] < self.reg["mod_min"]:
            s["below_module_min_s"] += dt
        if float(self.v_servo.min()) < 6.0:
            s["below_servo_spec_s"] += dt

    # -- what the firmware sees ------------------------------------------------------------------------------------------
    @property
    def dead(self):
        return self.fuse_blown or self.bms_off

    def measure(self):
        """(pack volts, amps) as the INA219 reports them from its shunt where the netlist wires it."""
        shunt = float(self.d("shunt.ohm"))
        shunt_v = float(quantise(self.i_shunt * shunt * self.gain + self.rng.normal(0, self.shunt_noise), self.q_shunt))
        bus_v = float(quantise(self.v_ina - shunt_v, self.q_bus))
        return bus_v + shunt_v, shunt_v / shunt

    def servo_voltage(self):
        return float(quantise(float(self.v_servo.mean()), float(self.d("leg_servo.voltage_step_v"))))

    @property
    def v_pack(self):
        return self.v_pack_term

    def mean_load(self):
        """Time-averaged (a, b, arm amps) of the run so far: what a battery-life simulation replays."""
        t = max(self.acc["t"], 1e-9)
        return self.acc["a"] / t, self.acc["b"] / t, self.acc["arm"] / t

    def summary(self, seconds):
        s = self.stats
        return dict(i_peak=s["i_peak"], i_mean=s["i_sum"] / max(1, s["n"]), v_servo_min=s["v_servo_min"],
                    v_arm_min=s["v_arm_min"], v_pack_min=s["v_pack_min"], v_computer_min=s["v_computer_min"],
                    load_a_max=s["load_a_max"], load_b_max=s["load_b_max"], i_3v3_max=s["i_3v3_max"],
                    below_module_min_s=s["below_module_min_s"], below_servo_spec_s=s["below_servo_spec_s"],
                    fuse_damage=self.fuse_damage, fuse_blown=self.fuse_blown, bms_off=self.bms_off,
                    resets=self.resets, soc=self.soc, seconds=seconds)


def main():
    net = Net()
    rep = check(net)
    print(table(net))
    print()
    for n in rep.notes:
        print("  " + n)
    if rep.failures:
        print(f"\nWIRING CHECK FAILED ({len(rep.failures)}):")
        for k, t in rep.failures:
            print(f"  [{k}] {t}")
        return 1
    print("\nWIRING CHECK PASSED: every pin once, every address free, every load powered, every wire and connector "
          "rated for what it can see, every part wired.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
