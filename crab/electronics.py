"""The robot's electronics as numbers (configs/electronics.toml) and as small models of what they do to the signals.

Part 1, the data: `load()` reads the file, `Data(path)("imu.gyro_noise_rms_dps")` returns one value, `Data.problems()`
lists every number whose source or reference is missing or wrong (tests/test_electronics.py runs it).

Part 2, the models, used by robot/backends.py SimBackend (the twin) and sim/benches/robustness.py:
  CommandDelay   servo targets reach the joint late (bus time + the servo's own delay, PWM frames for the arms)
  Imu            BNO055: sampled at its output rate, low-pass lag, noise, a fixed gyro bias and accelerometer offset
  Distance       VL53L0X: sampled at its rate, noise growing with range, 1 mm steps, dropouts at long range
  Power          battery state of charge and internal resistance, 5 V regulator limit, INA219 readout
`Realism(data, ...)` bundles them. Everything is seeded: the same seed and the same calls give the same numbers.
Nothing here imports the rest of the project (crab/ is the lowest layer)."""

import json
import math
import os
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "configs" / "electronics.toml"
SOURCES = ("measured", "datasheet", "estimate")
MODES = ("ideal", "datasheet")


# -- the data ---------------------------------------------------------------------------------------------------

class Data:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_PATH
        with open(self.path, "rb") as f:
            self.tree = tomllib.load(f)

    def __call__(self, dotted):
        """One value by dotted name: data("battery.capacity_ah"). Plain keys (battery.cells) work too."""
        node = self.tree
        for part in dotted.split("."):
            node = node[part]
        return node["value"] if isinstance(node, dict) and "value" in node else node

    def for_design(self, design):
        """The data of one design: v1 is the file as it is; another design (v2) swaps in its own battery and
        regulator sections ([battery_v2], [regulator_v2]) and leaves everything else."""
        if design == "v1":
            return self
        out = Data.__new__(Data)
        out.path = self.path
        out.tree = dict(self.tree)
        for part in ("battery", "regulator"):
            out.tree[part] = self.tree[f"{part}_{design}"]
        return out

    def parameters(self):
        """[(dotted name, {value, unit, source, ref, ...})] for every sourced number in the file."""
        out = []

        def walk(node, prefix):
            for k, v in node.items():
                if isinstance(v, dict):
                    if "value" in v:
                        out.append((prefix + k, v))
                    else:
                        walk(v, prefix + k + ".")
        walk(self.tree, "")
        return out

    def problems(self, root=ROOT):
        """What is wrong with the file: a source that is not measured/datasheet/estimate, no unit, no reference
        (a URL or a file that exists), or an estimate without its reasoning."""
        bad = []
        for name, p in self.parameters():
            if p.get("source") not in SOURCES:
                bad.append(f"{name}: source must be one of {SOURCES}, not {p.get('source')!r}")
            ref = p.get("ref")
            if not isinstance(ref, str) or not ref:
                bad.append(f"{name}: no ref")
            elif not ref.startswith("https://") and not (Path(root) / ref).exists():
                bad.append(f"{name}: ref {ref!r} is neither a URL nor a file in the repository")
            if not p.get("unit"):
                bad.append(f"{name}: no unit")
            if p.get("source") == "estimate" and len(str(p.get("why", ""))) < 20:
                bad.append(f"{name}: an estimate needs a `why`")
            if p.get("source") in ("measured", "datasheet") and ref and not ref.startswith("https://") \
                    and ref not in ("hardware/BOM.md",):
                bad.append(f"{name}: a {p['source']} value must point at the URL it was read from")
        return bad


def load(path=None):
    return Data(path)


# -- the models --------------------------------------------------------------------------------------------------

class CommandDelay:
    """Servo targets reach the joint `delay[i]` seconds late. push() is called once per physics step with the
    newest targets and returns what each servo has actually been told by now (a ring buffer, step resolution)."""

    def __init__(self, delays_s, dt, initial):
        self.n = np.maximum(0, np.round(np.asarray(delays_s, float) / dt).astype(int))
        self.size = int(self.n.max()) + 1
        self.hist = np.tile(np.asarray(initial, float), (self.size, 1))
        self.pos = 0
        self.idx = np.arange(len(self.n))

    def push(self, target):
        self.pos = (self.pos + 1) % self.size
        self.hist[self.pos] = target
        return self.hist[(self.pos - self.n) % self.size, self.idx]


def quantise(x, step):
    return np.round(np.asarray(x) / step) * step


class Imu:
    """BNO055 in IMU fusion mode. Output: (gyro rad/s, unit vector toward the ground), body frame."""

    def __init__(self, data, rng):
        g = data
        self.rate = float(g("imu.fusion_rate_hz"))
        left = float(g("imu.offset_left_after_calibration"))
        i2c = float(g("imu.i2c_read_delay_s"))
        self.tau_grav = float(g("imu.fusion_lag_s")) + i2c
        self.tau_gyro = 1.0 / (2 * math.pi * float(g("imu.gyro_bandwidth_hz"))) + i2c
        self.gyro_sigma = math.radians(float(g("imu.gyro_noise_rms_dps")))
        self.grav_sigma = float(g("imu.accel_noise_density_ug")) * 1e-6 * math.sqrt(float(g("imu.accel_bandwidth_hz")))
        self.rng = rng
        self.gyro_bias = rng.uniform(-1, 1, 3) * math.radians(float(g("imu.gyro_zero_rate_offset_dps"))) * left
        self.grav_bias = rng.uniform(-1, 1, 3) * float(g("imu.accel_zero_g_offset_mg")) * 1e-3 * left
        self.idx = None
        self.gyro_s = self.grav_s = None
        self.out = None

    def read(self, t, gyro, grav):
        idx = int(math.floor(t * self.rate + 1e-9))
        if self.idx is not None and idx == self.idx:
            return self.out[0].copy(), self.out[1].copy()                 # same 10 ms sample as the last read
        gyro, grav = np.asarray(gyro, float), np.asarray(grav, float)
        if self.idx is None:
            self.gyro_s, self.grav_s = gyro.copy(), grav.copy()
        else:
            dt = (idx - self.idx) / self.rate
            self.gyro_s += (1 - math.exp(-dt / self.tau_gyro)) * (gyro - self.gyro_s)
            self.grav_s += (1 - math.exp(-dt / self.tau_grav)) * (grav - self.grav_s)
        self.idx = idx
        g = self.gyro_s + self.gyro_bias + self.rng.normal(0, self.gyro_sigma, 3)
        v = self.grav_s + self.grav_bias + self.rng.normal(0, self.grav_sigma, 3)
        self.out = (g, v / np.linalg.norm(v))
        return self.out[0].copy(), self.out[1].copy()


class MountedImu(Imu):
    """The BNO055 where configs/layout_v2.toml mounts it. It measures angular rate and specific force (gravity plus the
    acceleration of the chip's own spot, lever arm included) in the chip's own axes; the fusion estimates the direction of
    gravity by integrating the gyro and pulling slowly toward the accelerometer's direction (a complementary filter with time
    constant imu.fusion_accel_tau_s, an estimate); the output is rotated into body axes (the chip's axis remap, `R`).
    So walking accelerations leak into the tilt a little, as they do in the real chip, instead of the exact gravity vector
    of `Imu`. read(t, gyro, accel) -> (gyro, unit vector toward the ground), body axes."""

    def __init__(self, data, rng, R):
        super().__init__(data, rng)
        self.R = np.asarray(R, float)
        self.tau_acc = float(data("imu.fusion_accel_tau_s"))
        self.u = None

    def read(self, t, gyro, accel):
        idx = int(math.floor(t * self.rate + 1e-9))
        if self.idx is not None and idx == self.idx:
            return self.out[0].copy(), self.out[1].copy()
        gyro, accel = np.asarray(gyro, float), np.asarray(accel, float)
        n = float(np.linalg.norm(accel))
        u_acc = -accel / n if n > 1e-6 else (self.u if self.u is not None else np.array([0.0, 0.0, -1.0]))
        if self.idx is None:
            self.gyro_s, self.u = gyro.copy(), u_acc.copy()
        else:
            dt = (idx - self.idx) / self.rate
            self.gyro_s += (1 - math.exp(-dt / self.tau_gyro)) * (gyro - self.gyro_s)
            w = self.gyro_s + self.gyro_bias
            self.u = self.u - dt * np.cross(w, self.u)                         # a world-fixed vector turns against the body
            self.u = self.u + (1 - math.exp(-dt / self.tau_acc)) * (u_acc - self.u)
            self.u /= np.linalg.norm(self.u)
        self.idx = idx
        g = self.gyro_s + self.gyro_bias + self.rng.normal(0, self.gyro_sigma, 3)
        v = self.u + self.grav_bias + self.rng.normal(0, self.grav_sigma, 3)
        self.out = (self.R @ g, self.R @ (v / np.linalg.norm(v)))
        return self.out[0].copy(), self.out[1].copy()


class Distance:
    """The VL53L0X beams. read(name, range_m, t) -> metres, or -1 for "no answer" (nothing hit, out of reach or
    a dropped reading). Each beam keeps its own sample clock, so a beam read twice in 33 ms gives the same value."""

    def __init__(self, data, rng):
        g = data
        self.rate = float(g("distance.rate_hz"))
        self.max_range = float(g("distance.max_range_m"))
        self.res = float(g("distance.resolution_m"))
        self.sig_frac = float(g("distance.noise_sigma_frac"))
        self.sig_floor = float(g("distance.noise_sigma_floor_m"))
        self.reliable = float(g("distance.reliable_range_m"))
        self.p_short = float(g("distance.dropout_short"))
        self.p_max = float(g("distance.dropout_at_max"))
        self.rng = rng
        self.state = {}                                                   # beam -> (sample index, value)

    def dropout_probability(self, r):
        x = np.clip((r - self.reliable) / max(self.max_range - self.reliable, 1e-9), 0.0, 1.0)
        return self.p_short + (self.p_max - self.p_short) * x

    def read(self, name, r, t):
        idx = int(math.floor(t * self.rate + 1e-9))
        last = self.state.get(name)
        if last is not None and last[0] == idx:
            return last[1]
        if r < 0 or r > self.max_range or self.rng.random() < self.dropout_probability(r):
            v = -1.0
        else:
            v = max(self.res, float(quantise(r + self.rng.normal(0, max(self.sig_floor, self.sig_frac * r)), self.res)))
        self.state[name] = (idx, v)
        return v


class Power:
    """Pack, regulator and monitor. step(dt, servo_amps) advances the state of charge and recomputes the voltages."""

    def __init__(self, data, rng, soc=1.0):
        g = data
        self.cap_as = float(g("battery.capacity_ah")) * float(g("battery.usable_fraction")) * 3600.0
        self.cells = int(g("battery.cells"))
        self.ocv = np.asarray(g("battery.ocv_per_cell_v"), float)
        self.r0 = float(g("battery.internal_resistance_ohm"))
        self.r_rise = float(g("battery.resistance_rise_at_empty"))
        self.v_set = float(g("regulator.output_v"))
        self.i_lim = float(g("regulator.current_limit_a"))
        self.eff = float(g("regulator.efficiency"))
        self.dropout = float(g("regulator.dropout_v"))
        self.r_wire = float(g("regulator.wiring_resistance_ohm"))
        self.r_over = float(g("regulator.overload_droop_ohm"))
        self.p_computer = float(g("regulator.computer_load_w"))
        self.shunt = float(g("power_monitor.shunt_ohm"))
        self.q_bus = float(g("power_monitor.bus_voltage_step_v"))
        self.q_shunt = float(g("power_monitor.shunt_voltage_step_v"))
        self.shunt_noise = float(g("power_monitor.shunt_noise_v"))
        self.q_servo_v = float(g("leg_servo.voltage_step_v"))
        self.rng = rng
        self.gain = 1.0 + rng.uniform(-1, 1) * float(g("power_monitor.gain_error_frac"))
        self.soc = float(soc)
        self.i_batt = 0.0
        self.i_servo = 0.0
        self.v_bus = self.v_set
        self.v_pack = self.open_circuit()
        self.step(0.0, 0.0)

    def open_circuit(self):
        x = np.linspace(1.0, 0.0, len(self.ocv))
        return float(np.interp(np.clip(self.soc, 0, 1), x[::-1], self.ocv[::-1])) * self.cells

    def resistance(self):
        return self.r0 * (1.0 + (self.r_rise - 1.0) * (1.0 - float(np.clip(self.soc, 0, 1))) ** 2)

    def step(self, dt, servo_amps):
        voc, r = self.open_circuit(), self.resistance()
        p_in = (self.v_bus * servo_amps + self.p_computer) / self.eff          # watts drawn from the pack
        disc = voc * voc - 4 * r * p_in
        if disc >= 0:
            i = (voc - math.sqrt(disc)) / (2 * r)
        else:
            i = voc / (2 * r)                                                   # the pack cannot deliver it: collapse
        self.i_batt, self.i_servo = i, servo_amps
        self.v_pack = voc - i * r
        v_reg = min(self.v_set, self.v_pack - self.dropout)
        v = v_reg - self.r_wire * servo_amps - self.r_over * max(0.0, servo_amps - self.i_lim)
        self.v_bus = max(0.0, v)
        self.soc = max(0.0, self.soc - i * dt / self.cap_as)

    def servo_voltage(self):
        """What a leg servo reports: the bus voltage in 0.1 V steps."""
        return float(quantise(self.v_bus, self.q_servo_v))

    def measure(self):
        """(pack volts, amps) as the INA219 reports them: shunt and bus voltage in LSB steps, a gain error."""
        shunt_v = float(quantise(self.i_batt * self.shunt * self.gain + self.rng.normal(0, self.shunt_noise),
                                 self.q_shunt))
        bus_v = float(quantise(self.v_pack - shunt_v, self.q_bus))
        return bus_v + shunt_v, shunt_v / self.shunt


class Realism:
    """All the electronics of one simulated robot. n joints, `pwm` = which of them are PWM servos, `kt` = N*m per
    amp of each joint's servo (crab/params.py kt_current), `dt` = the physics step."""

    def __init__(self, data=None, seed=0, soc=1.0, pwm=(), kt=None, dt=0.002, initial=None):
        d = data if isinstance(data, Data) else Data(data)
        self.data = d
        self.pwm = np.asarray(pwm, bool)
        n = len(self.pwm)
        bus_bytes = float(d("leg_servo.bus_overhead_bytes")) + (n - int(self.pwm.sum())) * float(
            d("leg_servo.sync_write_bytes_per_servo"))
        bus_s = float(d("leg_servo.command_delay_s")) + bus_bytes * 10.0 / float(d("leg_servo.bus_baud"))
        self.delays = np.where(self.pwm, float(d("arm_servo.command_delay_s")), bus_s)
        self.delay = CommandDelay(self.delays, dt, np.zeros(n) if initial is None else initial)
        self.steps = float(d("leg_servo.encoder_steps_per_rev"))
        self.encoder_step = 2 * math.pi / self.steps
        self.kt = None if kt is None else np.asarray(kt, float)
        self.idle = np.where(self.pwm, float(d("arm_servo.idle_current_a")), float(d("leg_servo.idle_current_a")))
        ss = np.random.SeedSequence([int(seed), 0xE1EC])
        r_imu, r_dist, r_pow = (np.random.default_rng(s) for s in ss.spawn(3))
        self.imu = Imu(d, r_imu)
        self.distance = Distance(d, r_dist)
        self.power = Power(d, r_pow, soc)
        self.circuit = None                                             # v2: crab/wiring.py Circuit, set by SimBackend

    def encoder(self, angle):
        """A leg servo's reported angle: the magnetic encoder's 4096 steps per turn."""
        return quantise(angle, self.encoder_step)

    def servo_amps(self, forces, torque_on=True):
        """Total current of all servos from their joint torques (N*m) and the servos' N*m per amp."""
        if not torque_on:
            return 0.0
        i = self.idle + (np.abs(forces) / self.kt if self.kt is not None else 0.0)
        return float(np.sum(i))


# -- the placement of every part in the body (configs/layout_v2.toml, design v2) -----------------------------------

LAYOUT_PATH = ROOT / "configs" / "layout_v2.toml"


def _rot(rpy_deg):
    """Rotation matrix of roll about x, pitch about y, yaw about z (degrees), applied as Rz @ Ry @ Rx."""
    r, p, y = (math.radians(a) for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Part:
    """One part of the placement file. Lengths in mm, mass in g, in the body frame (x front, y left, z up)."""

    def __init__(self, name, d):
        self.name, self.d = name, d
        self.kind = d.get("kind", "")
        self.size = np.asarray(d.get("size_mm", (0, 0, 0)), float)            # L, W, H along the part's x, y, z
        self.mass_g = float(d["mass_g"])
        self.pos = np.asarray(d.get("pos_mm", (0, 0, 0)), float)
        self.rpy = tuple(d.get("rpy_deg", (0, 0, 0)))
        self.R = _rot(self.rpy)
        self.on = d.get("on")                       # the part this one is mounted on (touching is allowed)
        self.mass_only = bool(d.get("mass_only", False))
        self.in_model = bool(d.get("in_model", False))      # already built into the model by its own code (servos)
        self.connectors = list(d.get("connectors", []))
        self.window = d.get("window")

    @property
    def half(self):
        return self.size / 2.0

    def world(self, local):
        """A point given in the part's own frame (mm from its centre) in the body frame."""
        return self.pos + self.R @ np.asarray(local, float)

    def corners(self):
        s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        return self.pos + (s * self.half) @ self.R.T

    def window_pose(self):
        """(position, unit direction) of the sensor window / lens / glass in the body frame."""
        w = self.window
        return self.world(w["pos_mm"]), self.R @ (np.asarray(w["dir"], float) / np.linalg.norm(w["dir"]))

    def connector_box(self, c):
        """The cable room of one connector as (centre, half sizes, rotation) in the body frame: a slab on the named
        face of the part, `reach_mm` deep, `width_mm` wide across the face (default the whole face)."""
        axis = "xyz".index(c["side"][1])
        sign = 1.0 if c["side"][0] == "+" else -1.0
        half = self.half.copy()
        centre = np.zeros(3)
        centre[axis] = sign * (half[axis] + c["reach_mm"] / 2.0)
        h = half.copy()
        h[axis] = c["reach_mm"] / 2.0
        if "width_mm" in c:
            other = [i for i in range(3) if i != axis]
            along = other[0] if c.get("across", "") != "z" else other[1]
            h[along] = c["width_mm"] / 2.0
            centre[along] = c.get("offset_mm", 0.0)
        return self.world(centre), h, self.R


def _apply_overrides(tree, text):
    """Layout candidates without editing the file (sim/situations.py): CRAB_LAYOUT_JSON = {"chassis.shell_half_mm":
    [84, 94, 40], "part.shoulder_l.joint_mm": [60, 40, 5], ...}; each dotted key sets one value."""
    if not text:
        return
    for key, value in json.loads(text).items():
        node, *path, leaf = key.split(".")
        d = tree[node]
        for p in path:
            d = d[p]
        d[leaf] = value


class Layout:
    """configs/layout_v2.toml: the chassis, every part's outline, mass and place, with where each number came from."""

    def __init__(self, path=None):
        self.path = Path(path) if path else LAYOUT_PATH
        with open(self.path, "rb") as f:
            self.tree = tomllib.load(f)
        _apply_overrides(self.tree, os.environ.get("CRAB_LAYOUT_JSON"))
        self.chassis = self.tree["chassis"]
        self.rules = self.tree["rules"]
        self.parts = {n: Part(n, d) for n, d in self.tree["part"].items()}
        if "CRAB_DISPLAY_G" in os.environ:          # the face display's mass: 300 g listed worst case, 150 g hoped for
            self.parts["oled"].mass_g = float(os.environ["CRAB_DISPLAY_G"])
        self.lid_tilt = float(os.environ.get("CRAB_LID_TILT", self.chassis.get("lid_tilt_rad", 0.0)))
        if "lid_hinge_mm" in self.chassis and abs(self.lid_tilt - float(self.chassis["lid_tilt_rad"])) > 1e-9:
            self._pose_lid(self.lid_tilt)

    def _pose_lid(self, tilt):
        """Re-pose the parts of the face lid (the display module and the lid plate) for another tilt back from
        vertical (0.35 rad standing up, 1.50 folded flat): same place on the lid, the lid turned about its hinge."""
        ch = self.chassis
        H = np.asarray(ch["lid_hinge_mm"], float)
        n = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
        u = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
        up = float(ch["lid_panel_mm"][1]) / 2 + float(ch["lid_bezel_mm"])          # hinge line to the lid's middle
        pitch = 90.0 - math.degrees(tilt)
        for name, depth in (("oled", 5.0), ("visor", 5.0)):
            p = self.parts[name]
            p.pos = H + u * up + n * depth
            p.rpy = (0.0, pitch, 0.0)
            p.R = _rot(p.rpy)

    def __getitem__(self, name):
        return self.parts[name]

    def solid(self):
        return [p for p in self.parts.values() if not p.mass_only]

    def total_mass_g(self, only=None):
        return sum(p.mass_g for p in self.parts.values() if only is None or only(p))

    def com_mm(self, only=None):
        ps = [p for p in self.parts.values() if only is None or only(p)]
        m = sum(p.mass_g for p in ps)
        return sum(p.mass_g * p.pos for p in ps) / m

    def problems(self, root=ROOT):
        """What is wrong with the file itself: sizes and masses without a source or a reference."""
        bad = []
        for n, p in self.parts.items():
            d = p.d
            for what in ("size", "mass"):
                if what == "size" and p.mass_only:
                    continue
                s, ref = d.get(what + "_source"), d.get(what + "_ref")
                if s not in SOURCES:
                    bad.append(f"{n}: {what}_source must be one of {SOURCES}, not {s!r}")
                if not isinstance(ref, str) or not ref:
                    bad.append(f"{n}: no {what}_ref")
                elif not ref.startswith("https://") and not (Path(root) / ref).exists():
                    bad.append(f"{n}: {what}_ref {ref!r} is neither a URL nor a file in the repository")
                if s == "estimate" and len(str(d.get("why", ""))) < 20:
                    bad.append(f"{n}: an estimate needs a `why`")
                if s in ("measured", "datasheet") and isinstance(ref, str) and not ref.startswith("https://"):
                    bad.append(f"{n}: a {s} {what} must point at the URL it was read from")
        return bad


def load_layout(path=None):
    return Layout(path)
