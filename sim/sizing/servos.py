"""Servo models for the sizing study.

Two kinds, both reduced to a MuJoCo position actuator + joint damping/friction/armature:

* STS3215 7.4 V: Rhoban/bam's measured m1 / m6 fits, converted with BAM's own `bam/to_mujoco.py` formulas
  (copied below). BAM itself marks that export as a first-order approximation ("deprecated ... only an
  approximation"): it collapses the fitted model into constant kp, damping, frictionloss, armature and
  forcerange, so it cannot reproduce the load-dependent friction (m6) or the firmware's rate-limited target.
  We add the rate limit (max_velocity) and the command delay in the simulation loop, as BAM's actuator does.
* Today's servos (STS3032, GS4308): the repo's own datasheet models in crab/params.py (no measurement exists).

Units: N*m, rad, s, kg, A, ohm, V.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import SERVOS  # noqa: E402

# bam/feetech/actuator.py, class STS3215Actuator (Rhoban/bam, read with `gh api`, 2026-11-21)
BAM_VIN = 7.4
BAM_KP_FW = 32            # firmware proportional gain
BAM_ERROR_GAIN = 0.166    # duty = error_gain * kp * dq (oscilloscope)
BAM_MAX_PWM = 0.97


@dataclass(frozen=True)
class ServoModel:
    name: str
    source: str
    mass: float               # kg
    size_mm: tuple            # L x W x H, shaft along H
    vin: float                # V
    kp: float                 # N*m/rad (MuJoCo position actuator)
    forcerange: float         # N*m, +-: the voltage-limited torque (vin*kt/R)
    damping: float            # N*m*s/rad on the joint (viscous + back-EMF)
    frictionloss: float       # N*m
    armature: float           # kg*m^2
    kt: float                 # N*m/A
    R: float                  # ohm
    ke: float                 # N*m per rad/s: back-EMF slope used to turn actuator force into motor torque
    max_vel: float            # rad/s: firmware target slew limit
    delay: float              # s
    ds_stall: float           # N*m datasheet stall at vin (the conservative reference)
    rated: float              # N*m continuous (heating)
    idle_a: float = 0.006     # A per servo at rest (the repo's value; unknown for the 3215)


def bam_first_order(p: dict, vin=BAM_VIN, kp_fw=BAM_KP_FW, error_gain=BAM_ERROR_GAIN, max_pwm=BAM_MAX_PWM) -> dict:
    """bam/to_mujoco.py `voltage_controlled_to_mujoco`, verbatim formulas."""
    kt, R = p["kt"], p["R"]
    return {
        "forcerange": vin * kt / R,
        "armature": p["armature"],
        "kp": error_gain * kp_fw * vin * max_pwm * kt / R,
        "damping": p["friction_viscous"] + kt ** 2 / R,
        "frictionloss": p["friction_base"],
    }


def sts3215(variant="m1") -> ServoModel:
    p = json.loads((HERE / f"bam_sts3215_{variant}.json").read_text())
    mj = bam_first_order(p)
    return ServoModel(
        name=f"STS3215-{variant}", source=f"Rhoban/bam feetech_sts3215_7_4V/{variant}.json, to_mujoco first order",
        mass=0.055, size_mm=(45.2, 24.7, 35.0), vin=BAM_VIN, kp=mj["kp"], forcerange=mj["forcerange"],
        damping=mj["damping"], frictionloss=mj["frictionloss"], armature=mj["armature"],
        kt=p["kt"], R=p["R"], ke=p["kt"] ** 2 / p["R"], max_vel=p["max_velocity"], delay=p["command_delay"],
        # Datasheet stall 19.5 kg.cm and rated 6.5 kg.cm (Feetech STS3215 datasheet via DFRobot: 19.5 kg.cm at 7.4 V,
        # rated current 0.65 A, stall 2.5 A, idle 6 mA; Evelta lists 19.5 at 6 V: sources disagree, we take the low one).
        ds_stall=19.5 * 0.0980665, rated=6.5 * 0.0980665)


def repo_servo(name: str) -> ServoModel:
    """Today's servos from crab/params.py (at its default 5 V bus). R from the datasheet stall current
    (STS3032: 1.2 A at 6 V -> 5 ohm; GS4308: kt * stall / vin, kt = stall torque / stall current)."""
    sp = SERVOS[name]
    if name == "STS3032":
        R = 6.0 / 1.2
    else:
        R = sp.vin * sp.kt_current / sp.datasheet_stall
    return ServoModel(
        name=name, source="crab/params.py datasheet model", mass=sp.mass, size_mm=sp.size_mm, vin=sp.vin,
        kp=sp.kp, forcerange=sp.force_limit, damping=sp.damping, frictionloss=sp.frictionloss, armature=sp.armature,
        kt=sp.kt_current, R=R, ke=sp.back_emf, max_vel=sp.max_cmd_speed, delay=sp.command_delay,
        ds_stall=sp.datasheet_stall, rated=sp.rated_torque)


if __name__ == "__main__":
    for v in ("m1", "m6"):
        s = sts3215(v)
        print(f"{s.name}: forcerange {s.forcerange:.3f} kp {s.kp:.2f} damping {s.damping:.3f} "
              f"friction {s.frictionloss:.4f} armature {s.armature:.4f} max_vel {s.max_vel:.2f} "
              f"no-load {s.vin/s.kt:.2f} rad/s stall current {s.vin/s.R:.2f} A ds_stall {s.ds_stall:.2f} rated {s.rated:.3f}")
    for n in ("STS3032", "GS4308"):
        s = repo_servo(n)
        print(f"{n}: forcerange {s.forcerange:.3f} kp {s.kp:.2f} damping {s.damping:.4f} R {s.R:.2f} "
              f"kt {s.kt:.3f} rated {s.rated:.4f} max_vel {s.max_vel:.1f}")
