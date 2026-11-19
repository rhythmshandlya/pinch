"""Single source of truth for the crab's geometry, masses and servos.

Units: metres, kilograms, radians, newton-metres.
Body frame: +x = front (eyes/claws), +y = left, +z = up.

This is the hand-sized "pocket crab": 20 Feetech STS3032 micro serial-bus servos, the
smallest legs that still walk, everything from parts in stock. The earlier ST3215 design is
archived in variants/standard-st3215.zip. Every joint names its servo, so the simulator can
try other servos (env vars CRAB_MICRO_SERVO / CRAB_FEMUR_SERVO, bus voltage CRAB_BUS_V).
"""

import os
from dataclasses import dataclass, field
from math import radians

# ------------------------------------------------------------------ servos
@dataclass(frozen=True)
class ServoSpec:
    """One servo model: what the simulator runs and what designs are checked against.

    Simulator model (MuJoCo position actuator + joint damping, same structure as
    Rhoban's BAM): the actuator's force saturates at `model_stall`; joint damping
    `model_stall / model_noload` (+ viscous) gives the torque-speed line.
    Checks: `datasheet_stall` / `datasheet_noload` for torque-speed margin,
    `rated_torque` for heating (continuous load), `max_cmd_speed` for commands.
    """
    name: str
    bus_model: str               # register map in robot/feetech_bus.py
    source: str                  # where the numbers come from
    vin: float                   # bus voltage the numbers refer to
    model_stall: float           # N*m, simulator's max torque
    model_noload: float          # rad/s, simulator's no-load speed (sets back-EMF slope)
    viscous: float               # N*m*s/rad extra viscous friction
    frictionloss: float          # N*m Coulomb friction
    armature: float              # kg*m^2 reflected rotor inertia
    kp: float                    # N*m/rad position gain
    kt_current: float            # N*m/A (to estimate current draw)
    datasheet_stall: float       # N*m
    datasheet_noload: float      # rad/s
    rated_torque: float          # N*m continuous (heating limit)
    max_cmd_speed: float         # rad/s a controller may command
    backlash_deg: float
    command_delay: float         # s, bus + firmware latency
    mass: float                  # kg
    size_mm: tuple               # case L x W x H (shaft along H)

    @property
    def kv(self) -> float:
        return 0.0                               # damping lives in the joint (back-EMF), not the actuator

    @property
    def force_limit(self) -> float:
        return self.model_stall

    @property
    def back_emf(self) -> float:                 # N*m per rad/s
        return self.model_stall / self.model_noload

    @property
    def damping(self) -> float:
        return self.viscous + self.back_emf

    # backwards-compatible names used across the code base
    @property
    def stall_torque(self) -> float:
        return self.datasheet_stall

    @property
    def max_speed(self) -> float:
        return self.max_cmd_speed


# The servo bus voltage. 5.0 V = the compact fixed 5 V / 5 A regulator; 6.0 V = an adjustable one.
# Micro servos' torque and speed scale with the voltage (a DC motor's stall torque and no-load speed
# both do), so their models below are written at 6 V and scaled to the bus.
BUS_V = float(os.environ.get("CRAB_BUS_V", "5.0"))


def _at_bus(spec: "ServoSpec", v: float) -> "ServoSpec":
    from dataclasses import replace
    k = v / spec.vin
    return replace(spec, vin=v, source=f"{spec.source}, scaled to {v:g} V",
                   model_stall=spec.model_stall * k, model_noload=spec.model_noload * k, kp=spec.kp * k,
                   datasheet_stall=spec.datasheet_stall * k, datasheet_noload=spec.datasheet_noload * k,
                   rated_torque=spec.rated_torque * k, max_cmd_speed=spec.max_cmd_speed * k)


SERVOS = {
    # Feetech STS3032 (Feetech ST-3032-C001): the servo on EVERY joint of the crab. The only small serial
    # bus servo in stock (Evelta, 2026-11-18). Case 23 x 12 x 27.5 mm (32 mm over the ears), 20 g,
    # coreless motor, metal gears, aluminium case, 2 ball bearings, magnetic 4096 steps, STS protocol,
    # 4.8-6 V. Feetech spec @6 V: 4.5 kg*cm stall, 1.5 kg*cm rated, 0.09 s/60deg, 1.2 A stall.
    # No published system identification exists -> datasheet model; RL randomises it widely.
    "STS3032": _at_bus(ServoSpec("STS3032", "sts3215", "Feetech spec table @6 V", 6.0,
                                 model_stall=0.441, model_noload=11.6, viscous=0.006, frictionloss=0.012,
                                 armature=5e-4, kp=2.4, kt_current=0.3675,
                                 datasheet_stall=0.441, datasheet_noload=11.6, rated_torque=0.147,
                                 max_cmd_speed=9.5, backlash_deg=0.5, command_delay=0.005, mass=0.020,
                                 size_mm=(23.0, 12.0, 27.5)), BUS_V),
    # Feetech SCS0009 / Waveshare SC09 (sold out on 2026-11-18): 23.2 x 12.1 x 25.25 mm, 13.2 g,
    # potentiometer, 300 deg / 1024 steps. Datasheet @6 V: 2.3 kg*cm stall, 0.7 kg*cm rated.
    "SCS0009": _at_bus(ServoSpec("SCS0009", "scs0009", "Feetech datasheet @6 V", 6.0,
                                 model_stall=0.2256, model_noload=10.47, viscous=0.003, frictionloss=0.008,
                                 armature=7e-4, kp=1.5, kt_current=0.2256,
                                 datasheet_stall=0.2256, datasheet_noload=10.47, rated_torque=0.0687,
                                 max_cmd_speed=9.0, backlash_deg=1.0, command_delay=0.005, mass=0.0132,
                                 size_mm=(23.2, 12.1, 25.25)), BUS_V),
    # Feetech HD-1910-C001 ("for Microduck", sold out on 2026-11-18): 34 x 20 x 23 mm, 21 g. Datasheet
    # @4.8 V: 9 kg*cm stall, 2.2 kg*cm rated, 0.137 s/60deg (not scaled: conservative at 5-6 V).
    "HD1910": ServoSpec("HD1910", "sts3215", "Feetech datasheet @4.8 V (Evelta listing)", 4.8,
                        model_stall=0.883, model_noload=7.64, viscous=0.01, frictionloss=0.02,
                        armature=6e-4, kp=4.5, kt_current=0.735,
                        datasheet_stall=0.883, datasheet_noload=7.64, rated_torque=0.216,
                        max_cmd_speed=6.5, backlash_deg=0.5, command_delay=0.005, mass=0.021,
                        size_mm=(34.0, 20.0, 23.0)),
    # GOTECK GS-4308MG: 4.3 g metal-gear PWM micro servo for the arms (yaw, lift, claw). 20 x 8.5 mm
    # case. Listing: 0.58 kg*cm & 0.14 s/60deg @4.8 V, 0.65 kg*cm & 0.12 s/60deg @6 V; at our 5 V about
    # 0.6 kg*cm & 0.135 s/60deg. Hobby servo: stiff P control, ~1.5 deg gear play, 50 Hz PWM frames
    # (up to 20 ms latency), no position feedback. Continuous rating assumed 1/3 of stall.
    "GS4308": ServoSpec("GS4308", "pwm", "GOTECK listing, interpolated to 5 V", 5.0,
                        model_stall=0.059, model_noload=7.8, viscous=0.0005, frictionloss=0.002,
                        armature=6e-5, kp=0.8, kt_current=0.10,
                        datasheet_stall=0.059, datasheet_noload=7.8, rated_torque=0.020,
                        max_cmd_speed=6.5, backlash_deg=1.5, command_delay=0.020, mass=0.0058,
                        size_mm=(15.8, 8.5, 17.6)),
    # The large design's servo (variants/standard-st3215.zip). Rhoban BAM m1: published measurements.
    "ST3215": ServoSpec("ST3215", "sts3215", "Rhoban BAM m1 (measured) + Feetech datasheet", 7.4,
                        model_stall=3.516, model_noload=6.284, viscous=0.0599, frictionloss=0.0515,
                        armature=0.0261, kp=18.1, kt_current=1.1776,
                        datasheet_stall=1.91, datasheet_noload=5.45, rated_torque=0.49,
                        max_cmd_speed=4.8, backlash_deg=0.5, command_delay=0.005, mass=0.055,
                        size_mm=(45.2, 24.7, 35.0)),
}

# Which servo drives which joint: STS3032 bus servos in the legs (they carry the body), the tiny
# GS-4308MG in the arms (light work). The env vars let the simulator try others.
FEMUR_SERVO = os.environ.get("CRAB_FEMUR_SERVO", "STS3032")
MICRO_SERVO = os.environ.get("CRAB_MICRO_SERVO", "STS3032")
ARM_SERVO = os.environ.get("CRAB_ARM_SERVO", "GS4308")
JOINT_SERVO = {"coxa": MICRO_SERVO, "femur": FEMUR_SERVO, "tibia": MICRO_SERVO,
               "shoulder_yaw": ARM_SERVO, "shoulder_pitch": ARM_SERVO, "claw": ARM_SERVO}


def servo_for(joint: str) -> ServoSpec:
    return SERVOS[JOINT_SERVO[joint.split("_", 1)[1]]]


# ------------------------------------------------------------------ body and limbs
@dataclass(frozen=True)
class Body:
    half_size: tuple = (0.040, 0.055, 0.022)   # carapace ellipsoid (sim primitive)
    mass: float = 0.20                          # 6 yaw servos, computer, battery, electronics, shell


@dataclass(frozen=True)
class Leg:
    coxa_len: float = 0.022           # yaw -> femur axis: the shortest that clears the yaw servo's ears
    femur_len: float = 0.030           # short thigh: less load on the thigh servo, feet closer in
    tibia_len: float = 0.052           # the shin sets the ground clearance (and the gait scaling)
    foot_radius: float = 0.005
    coxa_mass: float = field(default_factory=lambda: 0.004 + SERVOS[FEMUR_SERVO].mass)  # bracket + thigh servo
    femur_mass: float = 0.017         # thigh link + knee servo
    tibia_mass: float = 0.004
    coxa_range: tuple = (radians(-55), radians(49))           # rear legs: swung forward, the thigh meets the body (CAD)
    front_coxa_range: tuple = (radians(-55), radians(28))     # forward swing: front leg meets the arm (CAD)
    femur_range: tuple = (radians(-40), radians(86))          # lifted higher, the knee meets the shell (CAD)
    tibia_range: tuple = (radians(-133), radians(30))         # folded further, the foot hits the hip (CAD)
    # femur + tibia (the shin's angle from horizontal): folded back past -125 deg the shin's underside
    # meets the bottom corner of the hip link (CAD sweep, sideways crawl); 3 deg margin
    shin_min: float = radians(-122)

    def coxa_range_for(self, leg_id: str) -> tuple:
        return self.front_coxa_range if leg_id[0] == "F" else self.coxa_range


@dataclass(frozen=True)
class Arm:
    """3 motors per arm: shoulder yaw (swing), shoulder pitch (lift), claw. The claw motor sits in the
    palm; its shaft is the moving finger's hinge, at the front of the palm."""
    shoulder_offset: float = 0.016    # yaw -> pitch axis
    upper_len: float = 0.030          # pitch axis -> claw hinge
    finger_len: float = 0.024         # claw hinge -> finger tips
    upper_mass: float = 0.014         # arm + palm + claw servo (primitive model only)
    jaw_mass: float = 0.0015
    shoulder_yaw_range: tuple = (radians(0), radians(20))     # out: the front leg; in: the other claw (CAD)
    shoulder_pitch_range: tuple = (radians(-40), radians(72))  # higher, the claw meets the shell (CAD)
    claw_range: tuple = (radians(0), radians(55))


@dataclass(frozen=True)
class CrabParams:
    body: Body = field(default_factory=Body)
    leg: Leg = field(default_factory=Leg)
    arm: Arm = field(default_factory=Arm)
    # Left-side mounts (right side mirrored automatically): name -> ((x, y, z), yaw)
    leg_mounts: dict = field(default_factory=lambda: {
        "F": ((0.022, 0.046, 0.0), radians(64)),
        "R": ((-0.022, 0.046, 0.0), radians(116)),
    })
    arm_mount: tuple = ((0.040, 0.019, 0.004), radians(15))   # arms point well forward: clear of the front legs
    bus_voltage: float = BUS_V        # regulated servo bus (STS3032: 4.8-6 V)
    battery_nominal: float = 7.6      # 2S LiPo, mid charge
    battery_warn: float = 6.8         # 3.4 V per cell
    battery_stop: float = 6.4         # 3.2 V per cell (under load)

    @property
    def servo(self) -> ServoSpec:
        """The most common servo (micro). Use servo_for(joint) for per-joint values."""
        return SERVOS[MICRO_SERVO]

    @property
    def scale(self) -> float:
        """Size relative to the original 100 mm-shin design (gait / test scaling)."""
        return self.leg.tibia_len / 0.100


PARAMS = CrabParams()

# Actuated joints in a fixed order: the action vector for sim, RL and real robot.
LEG_IDS = ["FL", "FR", "RL", "RR"]
ARM_IDS = ["L", "R"]
LEG_JOINTS = ["coxa", "femur", "tibia"]
ARM_JOINTS = ["shoulder_yaw", "shoulder_pitch", "claw"]

JOINT_NAMES = [f"{leg}_{j}" for leg in LEG_IDS for j in LEG_JOINTS] + \
              [f"{arm}_{j}" for arm in ARM_IDS for j in ARM_JOINTS]
JOINT_SPECS = [servo_for(j) for j in JOINT_NAMES]          # per-actuator servo specs, JOINT_NAMES order

# Sign conventions (identical on both sides thanks to mirroring):
#   coxa +           foot swings forward
#   femur +          knee lifts up
#   tibia +          foot swings outward (negative = tucked under)
#   shoulder_yaw +   arm swings outward
#   shoulder_pitch + arm lifts up
#   claw +           claw opens


# five VL53L0X cliff beams (decision 30): (name, window centre x, y, z in the body frame mm, beam direction
# (unit, pointing out and down)). Front pair forward 40 deg down, rear backward 40 deg down (they see the desk
# ~3.5 cm beyond the feet: walking forward is the fast way), sides outward 30 deg down from low on the side wall,
# under the servo deck (the desk 1.7 cm beyond the feet. Mounted at z=+5 they looked 5.4 cm out and stopped every
# sideways walk the planners allow, feet 2 cm from the edge; and the wall there ends at z~2 - the windows cut
# slivers. Decision 48). A drop is seen before a foot reaches it whichever way Pinch walks.
# Open windows (no cover): a VL53L0X behind a cover reads short over empty air without crosstalk calibration.
CLIFF_BEAMS = [
    ("front_left", 47.0, 11.0, -16.0, (0.720, 0.262, -0.643)),       # (splayed 20 deg out: covers the corners)
    ("front_right", 47.0, -11.0, -16.0, (0.720, -0.262, -0.643)),
    ("rear", -39.0, 0.0, -16.0, (-0.766, 0.0, -0.643)),
    ("side_left", -8.0, 55.0, -16.0, (0.0, 0.866, -0.5)),
    ("side_right", -8.0, -55.0, -16.0, (0.0, -0.866, -0.5)),
]
