# The robot in the twin

Pinch's body is the v2.1 design: twelve Feetech STS3215 leg servos at 7.4 V, whose motor Rhoban's BAM project has
*measured and published*, six STS3032 arm servos, a pill-shaped shell with a hinged face lid, and every electronic
part where the build puts it. The simulator model is `sim/models/crab_v2.xml` (`PARAMS.model_file`). It is generated,
never edited by hand, and a test keeps the committed file equal to what the generator makes.

```powershell
.venv\Scripts\python cad\v2\export.py              # CAD -> sim/models/v2_cad.json and meshes (only after a CAD change)
.venv\Scripts\python sim\build_model.py            # writes sim/models/crab_v2.xml
.venv\Scripts\python sim\build_model.py --primitive --out x.xml   # hand-built shapes, no CAD (sweeps, candidates)
.venv\Scripts\python sim\fit_check.py              # every part fits inside the shell
.venv\Scripts\python sim\v2_check.py               # stand and walk on both servo models, compared
```

```python
SimBackend(servo_model="first_order")                     # the default: position actuator + firmware slew limit
SimBackend(servo_model="bam", realism="datasheet")        # the full measured model, with the electronics of the file
```

## Where the numbers live

| What | File |
|---|---|
| Legs, arms, body, mounts, stance, gait timing | `crab/params.py` (`_v2_params`) |
| Every electronic part: outline, mass, pose, windows | `configs/layout_v2.toml` (checked by `sim/fit_check.py`) |
| Wiring and the circuit | `configs/wiring_v2.toml` ([ELECTRONICS.md](ELECTRONICS.md)) |
| Servo, sensor and power numbers, each with its source | `configs/electronics.toml` |
| Printed parts' meshes, masses and inertias | `cad/v2/` -> `cad/v2/export.py` -> `sim/models/v2_cad.json` |

With the CAD exported, each link's mass, centre of mass and inertia come from the CAD solids, the meshes draw it, and
boxes round each leg part collide. Without it (`--primitive`) the same robot is built from capsules, boxes and inline
superellipsoid meshes (`crab/shapes.py`). The claw keeps its jaw and pad boxes either way: the CAD jaws are built from
them (`sim/build_model.py jaw_geoms`).

## What the robot is

| | |
|---|---|
| Shell | superellipsoid "pill", 168 x 172 x 104 mm (x forward, y, z); it is the body's collision mesh. The printed body is a tray and a cover with six M2.5 heat-set inserts at the split (`cad/v2/body.py`) |
| Legs | coxa 35.2 mm, femur 56 mm, tibia 80 mm, a 14 mm rubber ball foot; hips at +-35.2 x +-68 mm. The thigh and knee servos ride in the coxa and femur links, the hip servos in the body |
| Stance | `crab/emotes.py STAND`: shins 85 deg below the thighs; the hip plane starts 9.6 cm above the floor |
| Arms | at the front of the shell, swing (yaw) -> lift (pitch) -> claw, 45 mm upper arm. The yaw servo stands in the body, the lift servo in the shoulder, the claw servo in the palm. Swing -5 to 80 deg, lift -70 to 72 deg, claw 0 to 74 deg |
| Claw | one fixed and one moving jaw; TPU pads 22 mm tall with a pen groove, a ball cup and tip lips, and a tip hook on each pad. The jaws are tilted 41 deg, and the claw hinge with them, so both tips meet the desk level when the arm picks from a 30 mm crouch |
| Arm servos | 6 x Feetech STS3032 on the same bus as the legs, from 5 V rail B, each at a 40 % firmware torque limit (the model holds them there: `STS3032T`) |
| Face | a Waveshare 6.25 inch 720 x 1560 HDMI panel in a lid hinged at the top front, 0.35 rad back from vertical (a friction hinge set by hand, so fixed in the model). The module is modelled at 0.30 kg, the listed worst case |
| Electronics | every part of `configs/layout_v2.toml` at its place with its own mass: Radxa ZERO 3W, 2S Samsung 25R pack (2500 mAh), two 5 V regulators, BNO055 IMU, INA219, MPR121, five VL53L0X cliff sensors and a VL53L5CX, a wide camera, two I2S microphones, a speaker |
| Mass | about 2.28 kg (`sim/models/crab_v2.xml`) |
| Scripted trot | period 1.21 s; sideways stride capped at 0.6 of the forward one (heat in the thigh servo) |

The IMU site and the camera are where the layout mounts them; the five cliff beams start at the layout's windows
(`crab/params.py CLIFF_BEAMS`). The servos' joint ranges are the CAD's; each joint also has a small gear-play joint.

## The measured servo model, two levels

Source: `Rhoban/bam`, `bam/params/feetech_sts3215_7_4V/m6.json` (the most complete of six fitted models) and the actuator
class in `bam/feetech/actuator.py` (commit b9e1111, read 2026-11-21). Every number is in `configs/electronics.toml`
under `[leg_servo]` with `source = "measured"` and the m6 file as `ref`, including the stall torque (supply x kt / R =
3.43 N*m) and the no-load speed (supply / kt = 5.80 rad/s); the test fails if any is missing or differs from the BAM
copy in `sim/sizing/bam_sts3215_m6.json`. The one exception is the 0.97 maximum duty, which BAM itself marks TODO: an
estimate in the file.

**Level (a), the full model**: `crab/servo_model.py BamServos` drives BAM's own `bam.mujoco.MujocoController`. The
leg actuators become torque motors. Every physics step: the firmware moves an internal target toward the command at
most 5.1 rad/s; the position error times firmware gain 32 times 0.166 times a fitted ratio gives a PWM duty (limited to
+-0.97); the motor torque is kt * (duty * supply - kt * speed) / R (back-EMF included); and BAM writes the joint's
Coulomb friction and viscous damping from its load-, direction- and speed-dependent (Stribeck) friction model. The
supply voltage can follow the twin's sagging battery (`set_supply`, done in `realism="datasheet"`). It needs the
optional `bam` package (`requirements.txt` has the pinned commit and the install line). If it is missing the tests
skip that level and `make_drive("bam", ...)` says how to install it.

**Level (b), first order**: `crab/servo_model.py FirstOrderServos` plus the position actuator in `crab_v2.xml`: stiffness
`kt * supply * 32 * 0.166 * ratio / R` = 21.2 N*m/rad, drive limit 3.43 N*m, back-EMF damping `kt^2 / R` + viscous = 0.619
N*m*s/rad, constant Coulomb friction 0.053 N*m, rotor inertia 0.0216 kg*m^2, plus the firmware slew limit. It is
BAM's own `bam/to_mujoco.py` form (which BAM marks deprecated and approximate) with the slew limit added. Use it where
the full model cannot run: MJX and the training environments, the firmware. It cannot reproduce the load-dependent
friction, and its torque limit does not follow the supply voltage.

**Command delay** is the measured 5.0 ms (`leg_servo.command_delay_s`) and is applied outside the servo model: by
`realism="datasheet"` (plus 0.9 ms of bus time) in the twin, by `sim/latency.py` in `sim/v2_check.py`.

**The two levels agree** on where the robot goes and differ a little on what the servos do. On the v2.1 body
(`sim/v2_check.py`, 2026-11-23) the full model walks within 4 % of the first-order speed, joint angles within 0.25
deg RMS, with peak torque 3-9 points of stall higher and power up to 6 % higher.

### Limits of the measurement

- Fitted at **7.4 V with firmware gain 32**. Another voltage or gain is an extrapolation; the model scales the
  voltage, the gain does not change anything else.
- The **exact gear version of the BAM servo is inferred**, not known (BAM lists the servo as a Feetech STS3215 C001,
  sold out in India). A different gear train would have different friction and inertia.
- The fit comes from a pendulum test rig (one servo, a known load). A leg adds contact and a different load path.
  `q_offset` (the rig's zero offset) is not used.
- The maker's printed stall (19.5 kg*cm = 1.91 N*m) is far below the model's 3.43 N*m "electrical" stall. Torque
  percentages in `sim/v2_check.py` use the *maker's* 1.91 N*m (and 0.637 N*m rated), the conservative figure.
- Temperature is a one-node heat model with estimated constants ([ELECTRONICS.md](ELECTRONICS.md)); no current
  limit and no gear wear are modelled.

## Not settled yet

- The face module's real mass (0.15 kg hoped for, 0.30 kg modelled): weigh it in bench order 1.
- The arm servos' command delay and torque-limit units: an STS3032 on the bench settles them.
- Drops and impacts, and the regulators during long stalls, are not studied.
- The skill benches and behaviour reports on this body come with phases 5 and 6.
