# Design v2 in the twin: the resized robot on the measured STS3215

Pinch has two designs in simulation. **v1** is today's robot (Feetech STS3032 legs, 0.61 kg, CAD-built model) and
stays the default, bit for bit. **v2** is the robot the sizing study (`docs/research/SIZING_STS3215.md`) recommends:
legs on the Feetech STS3215 at 7.4 V, whose motor Rhoban's BAM project has *measured and published*. v2 exists only
in simulation so far: no CAD, no hardware. Its job is to let us test walking, torque, speed and power on a servo model
that rests on measurements instead of a guessed datasheet.

> **Update (electronics in the body).** The table below is the sizing study's robot (1.571 kg, lumped electronics, eyes on stalks). Since the
> placement, wiring and circuit work (`docs/dev/ELECTRONICS.md`, "Design v2") `crab_v2.xml` holds every electronic part where it really sits
> (chassis box 4.2 mm taller, **1.750 kg**), the servos run from the pack through the circuit model, and v2 has today's face visor and screen instead
> of the stalk eyes. The check results further down were taken on the lighter, lumped model.

> **Update (style B, "chubbier cartoon crab").** The owner chose a new look for the resized robot and it is now built into the model, not drawn on top of it:
> a superellipsoid pill shell, a hinged face lid with a 6.25 inch colour screen, orange thigh covers, tapered shins, dark ball feet, arms on the front of the
> shell with a gripper sized for 5-40 mm toys (flat things hold, balls do not yet). The section "Style B" below has the numbers; the older sections describe the model underneath (servos, circuit,
> checks) and where they differ the style B numbers win. The check results further down were re-measured on style B.


## Style B (the shape of the model)

`sim/build_model.py --design v2` builds all of it from primitives and inline meshes (`crab/shapes.py` makes the superellipsoids); `configs/layout_v2.toml`
is still the one place where parts live, now inside the shell.

| | style B |
|---|---|
| Shell | superellipsoid "pill", half sizes 84 x 108 x 52 mm (x forward, y, z), exponents 0.65 vertical / 0.60 horizontal, centred at (6.9, 0, 0.3) mm of the hip plane; it is the body's convex collision mesh. Printed mass = outer surface 1042 cm2 x 2.0 mm wall x 1.27 g/cm3 PETG x 1.2 (ribs, bosses) = **318 g** |
| Legs | coxa 35.2 mm, **femur 56 mm, tibia 80 mm**, dark **ball foot radius 14 mm** (collision); thigh cover radius 19 mm, shin cover tapering 20 to 14 mm (two collision capsules), masses from the covers' area by the same rule plus the servo in each link; hips at +-35.2 x **+-68** mm (73.6 before: the rear hip servos' corners stuck out of the pill) |
| Stand pose | thighs 35 deg above horizontal, **shins 85 deg below horizontal** (the first spec, 78 deg, misses the torque limit, see the table); hip plane 6.0 cm above the floor, belly about 1 cm clear |
| Arms | shoulders at the front of the shell at mid-height, (cx + 0.95 a, +-40, cz - 0.10 c) = (86.7, +-40, -14.9) mm, pointing forward, claws 39 mm clear of the floor standing; reach the desk when lowered (arm pitch about -8 deg puts the pad on the floor) |
| Arm servos | Feetech STS3032 x 6 (23 x 12 x 27.5 mm, 20 g) as a **placeholder** on rail B; the yaw servo is fixed in the body, the pitch servo in the shoulder, the claw servo in the palm. Their PWM wiring entries are unchanged |
| Claw | one fixed and one moving jaw, TPU pads 22 mm tall and about 35 mm long, a tip hook on the upper half of each pad, claw range 0-95 deg, pads parallel 4 mm apart when shut; stiff contact parameters on the pads (a 6 g toy is squashed flat by MuJoCo's default soft contact) |
| Face lid | hinged at (58.98, 0, 48.66) mm, standing 0.35 rad back from vertical (a friction hinge set by hand, so fixed in the model; `CRAB_LID_TILT=1.5` builds it folded flat). Wraps the Waveshare 6.25 inch 720 x 1560 HDMI panel (active 144.18 x 66.6 mm, 2.16:1; the module is **modelled as 158 x 76 x 8 mm, an estimate, with a listed mass of 0.30 kg worst case**, 0.15 kg hoped for: `CRAB_DISPLAY_G=150`); printed lid about 165 x 82 x 10 mm, 61 g by the shell rule. The micro-HDMI flat cable runs through the hinge. The texture mapping of `sim/desk.py set_face` is unchanged (2:1 texture on a 2.16:1 area) |
| Camera | low at the front, between the shoulders, below the lid (its window flush with the shell); the five cliff windows and the distance sensor sit on the shell surface (the front pair now looks out from the belly: their thresholds need retuning before the pet runs on v2) |

Mass **2.135 kg** (1.985 kg with the 0.15 kg screen): 12 leg servos 660 g, 6 arm servos 120 g, shell 318 g, printed legs 318 g (covers, brackets, ball feet), lid 61 g, screen module 300 g, pack
100 g, the other electronics 208 g, arm covers and jaws about 45 g. Centre of mass 1.9 cm ahead of the foot centre (0.30 kg screen; 1.7 cm with 0.15 kg), 1.4 cm above the hip
plane (0.8 cm). Footprint with the arms at rest 32.1 cm long x 34.5 cm wide, 18.9 cm tall to the lid's top edge.

Fit (`python sim/fit_check.py`): every part is inside the shell shrunk by the 2 mm wall and 1 mm clearance, nothing overlaps, every connector has its cable room, every
window sees out. To make that true: the hip y moved in, the pack, fuse, XT60, regulator B, speaker and amplifier moved a little, the microphones are glued under the top of the
shell (no dome any more), the pads lie on the inner surface, and the cliff beams, camera and distance sensor slid along their beams onto the shell.

Gripper test (`pen`, a 5 mm rod, an eraser, a 30 mm and a 36 mm ball, a 40 mm cube, a 40 x 20 mm box; five offsets along and across the jaws; claw closed on the object, arm lifted 25 mm or more
and shaken): pen, eraser, cube and box 6/6, rod 5/6, **balls 2/6**: a sphere rolls out of a wedge of flat pads even with the hooks. Balls need cupped pads; not done.

## Torque limits (the owner's R-03) on style B

Peak motor torque at most 80 % of the maker's stall (1.91 N*m) and the hottest joint's RMS at most 70 % of rated (0.637 N*m). With the owner's first stance (shins 78 deg) style B
**misses**: standing already holds the thigh at 62 % of rated (full BAM model, 300 g screen), sideways walking at 5 cm/s reaches 83 %, at the old speed cap 96 % and 85 % of stall. The cheapest fixes,
measured (full BAM model unless said, hottest joint RMS % of rated):

| change | stand | sideways cap | forward cap |
|---|---|---|---|
| none (shins 78 deg, 300 g) | 62 | 78 (and 85 % of stall) | 58 |
| sideways stride cap x0.8 / x0.65 (first-order servo, shins 78) | | 77 / 70 | 62 / 60 |
| feet pulled in 28 mm (first-order servo, shins 78) | | 121 | 64 |
| 0.15 kg screen (shins 78) | 55 | 71 | 55 |
| **shins 85 deg, 300 g (adopted, with the sideways cap x0.5)** | **53** | **64** | **60** |
| shins 90 deg | 47 | 79 | 65 |

The adopted fix is the shin angle (85 deg: the foot sits 10 mm closer under the knee, so the thigh motor carries less of the weight) and a sideways stride cap of half the forward one
(`GaitParams.side_cap`, v2 trot only; sideways about 4.2 cm/s, forward about 9.8 cm/s).

## Check results (`python sim/v2_check.py`, 8 s per case, whole gait cycles, 5 ms command delay; style B, shins 85, sideways cap 0.5)

Peak = % of the maker's stall; RMS = hottest joint, % of rated. Screen 0.30 kg (listed worst case) first, 0.15 kg in brackets.

| case | speed | first order peak / RMS | full BAM peak / RMS |
|---|---|---|---|
| stand 5 s | | 17 / 49 (15 / 43) | 21 / 53 (18 / 47) |
| forward 10 cm/s | 9.0 cm/s | 49 / 64 | 51 / 57 |
| forward at the cap | 9.8 cm/s | 54 / 65 (55 / 62) | 56 / 60 (57 / 59) |
| sideways at the cap | 4.2 cm/s | 50 / 69 (50 / 66) | 56 / 64 (55 / 62) |
| turn 0.4 rad/s | 0.34 rad/s | 32 / 59 (28 / 54) | 41 / 47 (36 / 44) |

Every gait peak is under 57 % of stall and every RMS under 70 %: R-03 holds with both servo models and both screen masses. Tilt stays under 5.3 deg, nothing falls. Joint speed at most 4.0 rad/s (78 % of the
measured limit). Power 4-8 W.

The 0.30 kg lid moves the centre of mass 0.2 cm forward and 0.6 cm up against 0.15 kg, and adds 8 % to the weight the legs carry: standing RMS +6 points (BAM, hottest thigh), walking 1-3 points, the sideways
cap +2 points. **Getting up**: the joint ranges allow v1's righting swing (`crab/righting.py`), but run on style B's back it does not turn the robot over in 12 s at either mass (the lid and the high pill need their
own search; not done).

## Select it

```powershell
$env:CRAB_DESIGN = "v2"                                   # read once, when crab/params.py is first imported
.venv\Scripts\python sim\build_model.py --design v2       # writes sim/models/crab_v2.xml (checked in; a test keeps it current)
.venv\Scripts\python sim\v2_check.py                      # stand + walks, both servo levels, compared
```

```python
# CRAB_DESIGN=v2 in the environment of this process:
SimBackend(servo_model="first_order")                     # the default: position actuator + firmware slew limit
SimBackend(servo_model="bam", realism="datasheet")        # the full measured model, with the electronics of the file
```

The switch is an environment variable on purpose. Every module that builds constants at import (`crab/kinematics.py`,
`crab/gait.py`, the servo tables, `sim/build_model.py`) reads `crab.params.PARAMS`, so one value read once keeps them
all in agreement and today's import-time constants keep working. The price: a process is v1 or v2 for life. Tests run
v2 in a child process (`tests/test_v2.py`, `tests/v2_probe.py`). `sim/check.py`, `sim/walk.py` and
`robot/backends.py` load `PARAMS.model_file`, so they follow the switch too.

## What v2 is

| | v1 (today) | v2 |
|---|---|---|
| Leg links coxa / femur / tibia | 22 / 30 / 52 mm | 35.2 / 48 / 83.2 mm (scale 1.6), foot radius 8 mm |
| Hips | x +-22, y +-46 mm | x +-35.2, y +-73.6 mm |
| Chassis | CAD | box 135 x 180 x 46 mm + dome shell (primitives) |
| Mass | 0.61 kg | **1.571 kg**: 12 leg servos 660 g, 6 arm servos 35 g, printed legs 349 g, printed body 335 g, electronics 90 g, 2S 1500 mAh pack 81 g, printed arms 17 g (the study's 1.55 kg budget plus the longer arms) |
| Standing | hip 4.9 cm up | hip 7.6 cm up; 27 x 34 cm footprint with the arms forward, 13.6 cm tall (eyes) |
| Leg servo | STS3032, guessed datasheet model | STS3215 7.4 V, **measured** BAM m6 model |
| Arms | 30 mm upper arm, 24 mm fingers | 1.5x: 45 mm / 36 mm, whole claw 1.5x, same GS4308 micro servos |
| Scripted trot | period 0.65 s | period 1.23 s (the sizing study's "slow it 1.5x"), speed cap 11.1 cm/s |
| Servo bus / pack | 5 V, 2S 850 mAh | 7.4 V buck-boost rail, 2S 1500 mAh (`[regulator_v2]`, `[battery_v2]`) |
| Cliff beam windows | CAD positions | the same positions x 1.6 (`crab/params.py CLIFF_BEAMS`), same directions |

`sim/models/crab_v2.xml` is built from primitives by `sim/build_model.py` (no CAD mesh exists). Printed links are
capsules that **collide** with the floor (so the legs can push the body up and a fallen robot rests on its shell); the
chassis box and the dome collide too. Each STS3215 is a box with its real size (45.2 x 24.7 x 35 mm) and mass (55 g)
in the link that carries it (hip servos in the body, thigh servo in the coxa link, knee servo in the femur link, as in
the CAD), so mass and inertia come out of the geometry. The gear-play joints are as in v1 (+-0.5 deg per joint).
Foot friction and every joint range are v1's. The arm's claw geometry is v1's, scaled by `PARAMS.arm_scale`.

Arms. At 1.5x the claw reaches the floor at the lowest pitch (-40 deg: 3 mm below the floor surface, v1: 3 mm), so it
can touch the desk as today. The GS4308 (0.059 N*m stall, 0.020 rated) holds the longer arm horizontally at 13 % of
stall (38 % of rated); with a 10 g pen at the grip site 25 % of stall and **75 % of rated**. So the micro servos are
*adequate* for the arm itself and *marginal for a held payload* (today's arm: 15 % / 43 %). The claw tips lose force
(0.059 N*m over a 36 mm finger: 1.6 N, today 2.5 N). I kept the servo; if the pet should hold more than a pen for a
long time, the arms need a stronger servo (the next size up in the same family), which was not studied.

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
optional `bam` package (`requirements.txt` has the pinned commit and the install line: BAM declares Python <3.13 but
runs on 3.13 with mujoco 4.0.1, so it is installed with `--no-deps --ignore-requires-python`). If it is missing the
tests skip that level and `make_drive("bam", ...)` says how to install it.

**Level (b), first order**: `crab/servo_model.py FirstOrderServos` plus the position actuator in `crab_v2.xml`: stiffness
`kt * supply * 32 * 0.166 * ratio / R` = 21.2 N*m/rad, drive limit 3.43 N*m, back-EMF damping `kt^2 / R` + viscous = 0.619
N*m*s/rad, constant Coulomb friction 0.053 N*m, rotor inertia 0.0216 kg*m^2, plus the firmware slew limit. It is
BAM's own `bam/to_mujoco.py` form (which BAM marks deprecated and approximate) with the slew limit added. Use it where
the full model cannot run: MJX and the training environments, the firmware. It cannot reproduce the load-dependent
friction (BAM m6: about 0.10 N*m at the thigh while standing against 0.053 constant), and its torque limit does not
follow the supply voltage.

**Command delay** is the measured 5.0 ms (`leg_servo.command_delay_s`) and is applied outside the servo model: by
`realism="datasheet"` (plus 0.9 ms of bus time) in the twin, by `sim/latency.py` in `sim/v2_check.py`.

### Limits of the measurement

- Fitted at **7.4 V with firmware gain 32**. Another voltage or gain is an extrapolation; the model scales the
  voltage, the gain does not change anything else.
- The **exact gear version of the BAM servo is inferred**, not known (BAM lists the servo as a Feetech STS3215 C001;
  `SERVO_CHOICE.md` found that variant sold out in India). A different gear train would have different friction and
  inertia.
- The fit comes from a pendulum test rig (one servo, a known load). A leg adds contact and a different load path.
  `q_offset` (the rig's zero offset) is not used.
- The maker's printed stall (19.5 kg*cm = 1.91 N*m) is far below the model's 3.43 N*m "electrical" stall. Every torque
  percentage below uses the *maker's* 1.91 N*m (and 0.637 N*m rated), the conservative figure, as the sizing study did.
- No temperature, no current limit and no gear wear are modelled.

## Check results of the first v2 (before style B; kept for the servo-model comparison)

Standing 5 s: hip 7.6 cm, tilt 0.05 deg; worst joint (thigh) holds 9 % of the maker's stall and 23 % of rated
(sizing study: 8 % and 23 %). `sim/check.py` (every emote, the v1 test) also passes on v2: no fall, no joint at its
drive limit.

Walking, scripted trot (period 1.23 s). Peak = peak motor torque as % of the maker's stall (1.91 N*m); RMS = RMS motor
torque of the hottest joint as % of rated (0.637 N*m); speed = fastest joint against the measured 5.10 rad/s;
power = 12 leg servos (copper loss R*I^2 + positive mechanical power + idle current, I = torque / kt):

| case | speed reached | peak / RMS, first order | peak / RMS, full BAM | joint speed | power 1st / full |
|---|---|---|---|---|---|
| forward 5 cm/s | 4.4 cm/s | 29 % / 44 % | 31 % / 47 % | 2.0 rad/s | 5.5 / 5.7 W |
| forward 10 cm/s | 9.0 cm/s | 38 % / 51 % | 40 % / 54 % | 3.8 rad/s | 7.4 / 7.7 W |
| sideways 5 cm/s | 4.9 cm/s | 43 % / 49 % | 49 % / 53 % | 2.5 rad/s | 5.6 / 5.9 W |
| sideways 10 cm/s | 9.7 cm/s | 65 % / 72 % | 71 % / 74 % | 4.0 rad/s | 8.6 / 9.1 W |
| forward at the gait's cap | **10.0 cm/s** | 42 % / 56 % | 43 % / 60 % | 4.2 rad/s | 7.9 / 8.3 W |
| sideways at the gait's cap | **10.2 cm/s** | 71 % / 81 % | 78 % / 83 % | 4.4 rad/s | 9.8 / 10.4 W |
| turn 0.4 rad/s | 0.35 rad/s | 30 % / 43 % | 33 % / 40 % | 2.3 rad/s | 5.6 / 5.7 W |

Tilt stays under 6 deg in every case, no case falls. The slowed trot tops out at **about 10 cm/s forward and 10 cm/s
sideways** (the sizing study: about 10-11); today's robot reached about 12. The fastest joint stays under 4.4 rad/s,
86 % of the measured 5.10 rad/s firmware limit, so speed is not the limit; heat is: sideways at the cap runs the
thigh at 81-83 % of rated (RMS), so long fast sideways walks would need the servo's temperature reading. About 3.6 W of
every power figure is the 12 idle currents (40 mA each at 7.4 V, an *estimate*: no number is published for a holding
STS3215), so the walking part is 2-6 W.

**Full model against first order, same walk** (`sim/v2_check.py` prints this table): speed within 0.7 %, joint angles
within 0.1-0.13 deg RMS, tracking error equal to 0.03 deg, peak torque 1-7 points of stall higher in the full model
(more for the sideways gaits, where the load-dependent friction matters most), RMS torque 2-4 points of rated
higher, power 4-6 % higher (the extra friction costs energy). The first-order model is therefore a good stand-in
for *where the robot goes*, and slightly optimistic by about 5 % for *what the servos do*.

## What still uses today's design

The simulation switch changes the body and legs; most of the pet's code does not know about it yet:

- **Emotes** are joint-angle poses and timings tuned on v1's proportions. They run on v2 (`sim/check.py` passes them)
  but were not re-tuned or judged by eye.
- **The pet's desk routines and the desk world** (`pinch/activities/`, `pinch/mind/`, `sim/desk.py`, `pinch/touch.py`
  `STAND_BODY_Z`) assume v1's heights, claw reach and desk objects. They are not v2-ready.
- **The CAD** (`cad/`) builds v1 only; v2 has no meshes, and `crab_v2.xml` is not drawn from a CAD model.
- **Learned skills** (`policies/*.npz`, the training environments) were trained on v1's body and servo and are not valid
  for v2. The training code still loads `crab.xml`.
- **The golden trace and every bench** run v1 (the default). The robustness benches have not been run on v2.
- **Firmware** register tables (`robot/feetech_bus.py` already knows the STS3215) work, but `hardware/` describes v1.
- Not studied: drops and impacts (the sizing study's largest unknown), getting up from the back (the collision shapes
  allow it, no skill exists), the regulator's behaviour during stalls.
