# Electronics in the twin

> Two chapters. The first is the part models: delays, noise, steps and the battery, switched on with `realism="datasheet"`. The second, "Every part inside the body", is where every part sits, the wiring as data with checks, and the power circuit solved from it.

The digital twin used to have perfect electronics: no command delay, an exact IMU, joint angles without steps, a
battery that never changed and distance beams that never missed. This page describes the switch that fixes that.

```python
PinchSim(realism="ideal")      # the default: exactly today's behaviour (the golden trace is pinned to it)
PinchSim(realism="datasheet")  # the electronics of configs/electronics.toml, seeded and replayable
SimBackend(realism="datasheet", seed=0, soc=1.0)       # the same switch one level down (soc = starting charge)
python sim/benches/robustness.py --realism datasheet --jobs 4
python sim/benches/righting.py --realism datasheet
```

## Where the numbers live

One file, `configs/electronics.toml`, lists every electronic part of the robot (leg servos, arm servos, IMU, five
distance beams, battery monitor, touch board, camera, microphones, battery, regulators). Every number carries a
unit, a **source** and a **ref**:

- `measured`: someone measured the part (a public data set). The leg servo's whole motor model (torque constant,
  resistance, rotor inertia, friction, speed limit, command delay), from Rhoban BAM's `feetech_sts3215_7_4V/m6.json`;
  the twin's leg servos run on it ([V2_MODEL.md](V2_MODEL.md)).
- `datasheet`: printed in the maker's datasheet or product page, and the ref is the page it was read from.
- `estimate`: our own reasoning, written in `why`. Replace it when a real number exists.

The loader is `crab/electronics.py` (`Data()("imu.fusion_rate_hz")`); `tests/test_electronics.py` fails if a number
has no valid source, no reference or an estimate without its reasoning. The datasheet and measured numbers were read
on 2026-11-21 (BNO055 datasheet rev 1.8, INA219 SBOS448G, MPR121, Feetech and Pololu pages, the BAM JSON). ST's own
VL53L0X datasheet could not be fetched, so its range, accuracy and resolution come from Pololu's product page, which
quotes ST, and the page is the ref. The leg servo's measured numbers are here (`[leg_servo]`), and so are the pack and
servo rail (`[battery_v2]`, `[regulator_v2]`, which `SimBackend` reads in place of the older `[battery]` and `[regulator]`).

## What is modelled

| Part | What the twin does |
|---|---|
| Leg servo commands | Every servo target reaches the joint late: BAM's measured 5.0 ms plus the time the 12-servo write takes on the 1 Mbaud bus (0.9 ms). Physics runs in 2 ms steps, so the delay is 3 steps (6 ms). |
| Arm servo commands | 15 ms (`arm_servo.command_delay_s`, an estimate from the earlier PWM arm servos, kept until an STS3032 is timed on the bench): 8 steps. |
| Leg encoders | Angles come in 4096 steps per turn. |
| Servo voltage report | The bus voltage in 0.1 V steps. |
| IMU (BNO055) | Sampled at 100 Hz (reads inside one sample match); gyro noise 0.3 deg/s rms; accelerometer noise from its density and 62.5 Hz bandwidth; a fixed gyro bias and accelerometer offset per robot (a quarter of the datasheet limit, because fusion mode calibrates); a first-order lag of about 34 ms on the gravity direction and 9 ms on the gyro. |
| Distance beams (VL53L0X) | 30 Hz samples per beam, noise of 1.5 % of the range (at least 2 mm), 1 mm steps, no answer beyond 2 m, and dropouts that grow from 0.2 % on a good surface to about 50 % at the end of the range. A dropout reads as "no answer", which the cliff reflex already treats as unknown. |
| Battery | (The lumped model, `power_model="lumped"`: `SimBackend`'s default is the circuit of the second chapter.) State of charge falls with the current the servos and the computer draw (servo torque divided by each servo's N*m per amp, plus idle current, plus 4 W for the computer, through a 90 % regulator). The pack voltage follows a LiPo curve minus current times internal resistance (0.1 ohm, doubling when empty). |
| Regulator | The 5 V bus holds until the pack drops within 0.5 V of it, sags by the wire resistance times the servo current, and droops steeply past the 5 A limit. |
| INA219 | Reads pack voltage and current in its 4 mV and 10 uV steps, with a gain error (0.5 % max) and a little noise. |
| Low battery | In the twin the mind's battery level comes from the INA219 reading, so hunger, the low-battery warning and the safety stop (6.4 V under load) happen by themselves. A pack at 0 % trips the stop within seconds. |
| Servo temperature | Every bus servo reports the temperature of a one-node heat model (`crab/servo_model.py` `ServoHeat`), in every realism: the copper loss of its motor torque (BAM's measured winding resistance, rising 0.393 %/K), the gearbox's friction loss and the electronics' idle draw, into a thermal resistance to the room with a time constant (`[servo_heat]`: STS3215 10 K/W and 240 s, estimated from SO-ARM100 users' hold tests; a smaller servo scales by its case area). The mind's heat need and cool-down drive, and the body's walking limit (slower from 55 C, stopped at 60 C), read it. `fit_heat` fits the two constants to a logged bench run. |

Everything is seeded: the same seed and the same calls give the same numbers (`test_datasheet_realism_replays_exactly`).
The models draw from their own random streams, so the twin's other random choices do not change.

## Not modelled yet

- **Camera image realism**: frame rate, latency, exposure, lens distortion and noise are in the file as estimates
  but the twin's "camera" is still object positions with a little noise. The 100 ms camera latency is not applied.
- **Microphone self-noise** and the real microphone placement (the two I2S microphones): the hearing code gets
  clean events.
- **MPR121 touch**: 10-bit electrode values, drift and the 16 ms sample period are listed but touch events are not
  generated from them.
- **The servo's torque limit following the bus voltage.** The battery sags and the bus voltage is reported, but
  the torque model does not weaken with it (that belongs to the servo model being redesigned).
- **Servo speed and load feedback** are still exact. Temperature is modelled (above) but its two thermal constants
  are estimates until a real servo is logged; the sensor's lag behind the winding is not modelled.
- **Bus transmission errors** (garbled replies) and I2C NACKs.
- **Real servo current in the lumped model**: current comes from torque divided by `kt_current` in `crab/params.py`, so
  it follows whichever servo is configured. When the servo model changes the current, the power budget follows.
- **Beam field of view**: the twin's beams are single rays; the 25 deg cone is not modelled.
- Gyro bias is fixed per run; it does not drift with temperature.
- In the benches, tilt is measured from the sensed gravity vector, so the IMU's bias shows up as extra "tilt".

## Values

Every number below is in `configs/electronics.toml` with its unit and reference; `source` is repeated here.
**leg_servo**

| Value | Number | Unit | Source |
|---|---|---|---|
| encoder_steps_per_rev | 4096 | steps | datasheet |
| voltage_step_v | 0.1 | V | datasheet |
| current_feedback_step_a | 0.0065 | A | datasheet |
| command_delay_s | 0.00498075 | s | measured |
| bus_baud | 1000000 | baud | estimate |
| sync_write_bytes_per_servo | 7 | bytes | estimate |
| bus_overhead_bytes | 8 | bytes | estimate |
| idle_current_a | 0.04 | A | estimate |

**arm_servo**

| Value | Number | Unit | Source |
|---|---|---|---|
| pwm_frame_s | 0.02 | s | estimate |
| command_delay_s | 0.015 | s | estimate |
| idle_current_a | 0.015 | A | estimate |

**imu**

| Value | Number | Unit | Source |
|---|---|---|---|
| fusion_rate_hz | 100 | Hz | datasheet |
| gyro_noise_rms_dps | 0.3 | deg/s | datasheet |
| gyro_bandwidth_hz | 32 | Hz | datasheet |
| gyro_zero_rate_offset_dps | 3 | deg/s | datasheet |
| accel_noise_density_ug | 190 | ug/sqrt(Hz) | datasheet |
| accel_bandwidth_hz | 62.5 | Hz | datasheet |
| accel_zero_g_offset_mg | 80 | mg | datasheet |
| offset_left_after_calibration | 0.25 | fraction | estimate |
| fusion_lag_s | 0.03 | s | estimate |
| i2c_read_delay_s | 0.004 | s | estimate |

**distance**

| Value | Number | Unit | Source |
|---|---|---|---|
| max_range_m | 2 | m | datasheet |
| accuracy_best_frac | 0.03 | fraction of range | datasheet |
| resolution_m | 0.001 | m | datasheet |
| noise_sigma_frac | 0.015 | fraction of range | estimate |
| noise_sigma_floor_m | 0.002 | m | estimate |
| rate_hz | 30 | Hz | datasheet |
| fov_deg | 25 | deg | datasheet |
| reliable_range_m | 0.4 | m | datasheet |
| dropout_short | 0.002 | probability per reading | estimate |
| dropout_at_max | 0.5 | probability per reading | estimate |

**power_monitor**

| Value | Number | Unit | Source |
|---|---|---|---|
| bus_voltage_step_v | 0.004 | V | datasheet |
| shunt_voltage_step_v | 1e-05 | V | datasheet |
| conversion_time_s | 0.000532 | s | datasheet |
| gain_error_frac | 0.005 | fraction | datasheet |
| shunt_noise_v | 2e-05 | V | estimate |

**touch**

| Value | Number | Unit | Source |
|---|---|---|---|
| adc_bits | 10 | bits | datasheet |
| electrodes | 12 | channels | datasheet |
| sample_interval_s | 0.016 | s | estimate |

**camera**

| Value | Number | Unit | Source |
|---|---|---|---|
| fov_deg | 160 | deg | estimate |
| frame_rate_hz | 30 | Hz | estimate |
| latency_s | 0.1 | s | estimate |

**microphones**

| Value | Number | Unit | Source |
|---|---|---|---|
| spacing_m | 0.07 | m | estimate |
| sample_rate_hz | 16000 | Hz | estimate |

**battery**

| Value | Number | Unit | Source |
|---|---|---|---|
| capacity_ah | 0.85 | Ah | estimate |
| usable_fraction | 0.8 | fraction | estimate |
| internal_resistance_ohm | 0.1 | ohm | estimate |
| resistance_rise_at_empty | 2 | times | estimate |
| ocv_per_cell_v | curve (11 points) | V | estimate |

**regulator**

| Value | Number | Unit | Source |
|---|---|---|---|
| output_v | 5 | V | estimate |
| current_limit_a | 5 | A | estimate |
| efficiency | 0.9 | fraction | estimate |
| dropout_v | 0.5 | V | estimate |
| wiring_resistance_ohm | 0.05 | ohm | estimate |
| overload_droop_ohm | 1 | ohm | estimate |
| computer_load_w | 4 | W | estimate |

## What it changes (benches)

Measured on the earlier, smaller robot (decision 101); the v2.1 body's numbers come with its benches.

Robustness bench (`sim/benches/robustness.py`, desk world, 40 random robots and worlds per row, the same seeds in
both modes, 4 workers). In `datasheet` mode the bench's own ad-hoc noise (0.003 rad on joints, 0.02 rad/s on the
gyro) is replaced by the models above. Falls with their 95 % ranges:

| What | Controller | Falls, ideal | Falls, datasheet |
|---|---|---|---|
| walk | scripted trot | 0/40 = 0 % (95 % range 0-9 %) | 0/40 = 0 % (95 % range 0-9 %) |
| walk | trained residual | 0/40 = 0 % (95 % range 0-9 %) | 0/40 = 0 % (95 % range 0-9 %) |
| emote | scripted | 0/40 = 0 % (95 % range 0-9 %) | 0/40 = 0 % (95 % range 0-9 %) |
| emote | trained policy | 0/40 = 0 % (95 % range 0-9 %) | 0/40 = 0 % (95 % range 0-9 %) |

`stats.compare`: every pair is "inconclusive" (p = 1.0): neither mode makes the robot fall in 40 episodes, so these
benches cannot separate them on falls. The paired differences (datasheet minus ideal, mean with 95 % range over the
40 seeds) say more:

| What | Controller | Body tilt RMS (deg) | Speed tracking |
|---|---|---|---|
| walk | scripted | 0.00 +- 0.06 | +2.7 points +- 0.1 (69.4 -> 72.1 %) |
| walk | trained residual | +0.03 +- 0.06 | +2.1 points +- 0.3 (85.8 -> 87.9 %) |
| emote | scripted | +0.33 +- 0.10 | - |
| emote | trained policy | +0.35 +- 0.11 | - |

Emote tilt rises by about a third of a degree. Most or all of that is the IMU's fixed accelerometer offset (the bench
measures tilt from the sensed vector), not a real lean. Walking tracks slightly better with the delay in
(+2 to +3 points, a small but consistent effect on every seed; we did not investigate why, and it is not a reason to
keep the delay). With the delay, encoder steps and the BNO055's noise the trained policies still pass their criteria.

Self-righting bench (`sim/benches/righting.py --starts 100`, same 100 starts):

| Controller | Upright and standing, ideal | Upright and standing, datasheet |
|---|---|---|
| swing (scripted) | 53/100 = 53 % (95 % range 43-62 %) | 49/100 = 49 % (95 % range 39-59 %) |
| trained policy | 100/100 = 100 % (95 % range 96-100 %) | 100/100 = 100 % (95 % range 96-100 %) |

The swing's 53 % against 49 % is "inconclusive" (p = 0.67; about 2400 starts per arm would be needed to tell them
apart). The policy's time to upright rises from 0.54 s to 0.56 s (mean) and its 95th-percentile commanded joint speed
falls from 18.4 to 17.5 rad/s. In short: those controllers were robust to this electronics, with a measurable but
small cost, and the benches are not sensitive enough to rank them further. The robustness bench does not model the
battery, so the power model has no effect there.

## Using it

- Test a controller against realistic sensors: pass `realism="datasheet"` to `PinchSim` or `SimBackend`, or `--realism
  datasheet` to the two benches.
- Swap a part: edit its number in `configs/electronics.toml` with its source; nothing else changes. A new servo:
  change `leg_servo.command_delay_s` (BAM's m1 fits 1.0 ms and m6 5.0 ms) and the servo model in `crab/params.py`.
- Test low battery: `PinchSim(realism="datasheet", soc=0.1)`.

# Every part inside the body, wired, and a circuit that runs

The sections above are the part models. Here the electronics are modelled as the real build ([V2_MODEL.md](V2_MODEL.md)): where every part sits, how it is wired, and a circuit solved from that wiring while the twin runs.
"Exact" here means exact to the datasheets and to the wiring file; each number says which it is (measured, datasheet, estimate) and what
only the real parts can settle is listed at the end.

```powershell
.venv\Scripts\python sim\fit_check.py            # does everything fit? (layout_v2.toml)
.venv\Scripts\python -m crab.wiring              # wiring checks and the table of wires (wiring_v2.toml)
.venv\Scripts\python sim\build_model.py                # rebuilds sim/models/crab_v2.xml from the layout
.venv\Scripts\python sim\circuit_check.py [--bam]      # the numbers below
.venv\Scripts\python hardware\wiring_diagram.py        # hardware/wiring.svg, drawn from the netlist
.venv\Scripts\python sim\render_v2_face.py             # media/review/v2/v2_face.png
```

```python
SimBackend(realism="datasheet")                          # the default: the circuit model of the netlist
SimBackend(realism="datasheet", power_model="lumped")    # the earlier battery + one regulator, kept for comparison
SimBackend(realism="datasheet", enforce_module_min=True) # the regulator modules stop below their listed 7 V input
hw.set_activity("npu")                                   # what the computer is doing: "idle", "camera" (default), "npu"
```

## 1. Placement: `configs/layout_v2.toml`

A separate file next to `electronics.toml` on purpose. `electronics.toml` says how a part *behaves* (noise, delay, resistance); the layout
says where a part *is* (outline, mass, position, tilt). The two files
change for different reasons (a new part number versus moving a board). Every part has an outline L x W x H, a mass, a centre
position and a tilt in the body frame (x front, y left, z up, hip plane z = 0), the sides where cables leave with the room they need, and for
sensors the window position and view cone. Each size and each mass says `datasheet` (printed on the maker's or seller's page, with the page),
`measured` or `estimate` (with the reasoning). 44 entries; 14 outlines are from datasheets and 28 are estimates (listings rarely
give outlines: the Radxa's 65 x 30 mm, the regulator module's 25 x 22 mm, the Waveshare driver board's 42 x 33 mm, the STS3215's
45.22 x 35 x 24.72 mm, the MAX98357A's 19.4 x 17.8 x 3 mm, the Samsung 25R cell). Replace an estimate with a caliper number when the part arrives.

`sim/build_model.py` builds every electronic part into `crab_v2.xml` as a box with its own mass inside the shell (no contact with
the world), puts the IMU site where the BNO055 is mounted and the camera at the lens, and
draws the face (section 6). `sim/fit_check.py` fails on a part outside the shell's interior (the 2 mm wall and 1 mm clearance), two
parts overlapping, less than 1 mm between parts, a connector without its cable room, and a sensor, camera or screen window that something
blocks or that is not at the shell. A test makes a layout bad on purpose in each of those ways and checks that the check fails.

The first fit, in the earlier box chassis, is recorded in decision 104. The body is now the pill shell (168 x 172 x 104 mm,
`configs/layout_v2.toml` `[chassis]`); the parts in the layout weigh 954 g and the whole model about 2.28 kg (`sim/models/crab_v2.xml`).
`crab/params.py` takes `CLIFF_BEAMS` from the five distance sensors' windows in the layout, so the twin's desk beams start exactly there.

### The placement table

From `configs/layout_v2.toml` (`python sim/fit_check.py` prints it; window cones and connector rooms are in the file).

| Part | Outline L x W x H (mm) | g | Position x, y, z (mm) | Tilt r, p, y (deg) | Size from | Mass from |
|---|---|---|---|---|---|---|
| hip_fl | 45.2 x 24.7 x 35 | 55 | 35.2, 68.0, -5.2 | 0, 0, 0 | datasheet | datasheet |
| hip_fr | 45.2 x 24.7 x 35 | 55 | 35.2, -68.0, -5.2 | 0, 0, 0 | datasheet | datasheet |
| hip_rl | 45.2 x 24.7 x 35 | 55 | -35.2, 68.0, -5.2 | 0, 0, 0 | datasheet | datasheet |
| hip_rr | 45.2 x 24.7 x 35 | 55 | -35.2, -68.0, -5.2 | 0, 0, 0 | datasheet | datasheet |
| shoulder_l | 12 x 23 x 27.5 | 20 | 61.8, 40.0, -16.7 | 0, 0, 90 | estimate | datasheet |
| shoulder_r | 12 x 23 x 27.5 | 20 | 61.8, -40.0, -16.7 | 0, 0, 90 | estimate | datasheet |
| pack | 72 x 38.5 x 19.5 | 100 | -12.0, -35.0, -26.0 | 0, 0, 0 | estimate | estimate |
| xt60 | 26 x 8 x 16 | 12 | 29.5, -33.0, -26.0 | 0, 0, 90 | estimate | estimate |
| fuse | 30 x 18 x 22 | 14 | 16.0, -30.0, 13.5 | 0, 0, 0 | estimate | estimate |
| ina | 25 x 25 x 14 | 4.5 | -20.0, 0.0, -24.0 | 0, 0, 90 | datasheet | estimate |
| reg_a | 25 x 22 x 11 | 5.5 | -30.0, -34.0, -4.0 | 0, 0, 0 | datasheet | estimate |
| reg_b | 25 x 22 x 11 | 5.5 | 29.0, 40.0, 0.0 | 0, 0, 0 | datasheet | estimate |
| hub | 55 x 36 x 14 | 24 | -27.5, 34.0, -23.0 | 0, 0, 0 | estimate | estimate |
| driver | 42 x 34 x 15 | 15 | 19.0, 14.0, -24.5 | 0, 0, 90 | datasheet | estimate |
| radxa | 65 x 32 x 13 | 12.5 | 2.0, -0.5, -6.5 | 0, 0, 0 | datasheet | estimate |
| imu | 21 x 27 x 4 | 3 | -20.0, 0.0, 10.0 | 0, 0, 0 | datasheet | estimate |
| mpr121 | 21 x 18 x 3 | 2 | 16.0, -32.0, -8.0 | 0, 0, 0 | estimate | estimate |
| amp | 19.4 x 17.8 x 3 | 1.2 | -38.0, 40.0, 0.0 | 0, 0, 0 | datasheet | datasheet |
| radxa_heatsink | 14 x 14 x 6 | 3 | -2.3, -2.3, -4.7 | 0, 0, 0 | estimate | estimate |
| camera | 1.6 x 25 x 24 | 4 | 48.3, 0.0, -19.5 | 0, 0, 0 | estimate | estimate |
| camera_lens | 12 x 17 x 17 | 6 | 55.1, 0.0, -19.5 | 0, 0, 0 | estimate | estimate |
| tof8 | 4 x 25.4 x 25.4 | 3 | 58.0, 0.0, 6.0 | 0, 0, 0 | datasheet | estimate |
| tof_fl | 25 x 13 x 3.5 | 1.4 | 51.9, 17.6, -44.2 | 131, -10, 101 | estimate | estimate |
| tof_fr | 25 x 13 x 3.5 | 1.4 | 51.9, -17.6, -44.2 | -131, -10, -101 | estimate | estimate |
| tof_rear | 25 x 13 x 3.5 | 1.4 | -66.5, 0.0, -32.1 | -130, 0, 90 | estimate | estimate |
| tof_left | 25 x 13 x 3.5 | 1.4 | 0.0, 55.0, 35.0 | -124, 0, 0 | estimate | estimate |
| tof_right | 25 x 13 x 3.5 | 1.4 | 0.0, -55.0, 35.0 | 124, 0, 0 | estimate | estimate |
| oled | 76 x 158 x 8 | 300 | 49.6, 0.0, 88.9 | 0, 70, 0 | estimate | estimate |
| visor | 82 x 165 x 10 | 61 | 49.6, 0.0, 88.9 | 0, 70, 0 | estimate | estimate |
| mic_l | 16 x 11 x 3 | 1.5 | -9.8, 34.6, 47.6 | -6, -1, 0 | datasheet | estimate |
| mic_r | 16 x 11 x 3 | 1.5 | -9.8, -34.6, 47.6 | 6, -1, 0 | datasheet | estimate |
| speaker | 28 x 28 x 5 | 4 | -46.3, 0.0, 38.9 | 0, -18, 0 | estimate | estimate |
| switch | 27 x 30 x 22 | 12 | -63.1, 0.0, -3.0 | 0, 0, 0 | estimate | estimate |
| dock_board | 18 x 12 x 4 | 3 | -48.0, -48.0, 5.0 | 0, 0, 0 | estimate | estimate |
| pogo_l | 16.35 x 6 x 2 | 0.8 | -68.3, 20.0, 3.0 | 0, 0, 0 | estimate | estimate |
| pogo_r | 16.35 x 6 x 2 | 0.8 | -68.3, -20.0, 3.0 | 0, 0, 0 | estimate | estimate |
| harness | - | 60 | 0.0, 0.0, -12.0 | 0, 0, 0 | - | estimate |
| fasteners | - | 25 | 0.0, 0.0, -5.0 | 0, 0, 0 | - | estimate |
| pad_head | 14 x 10 x 0.2 | 0.3 | 4.0, 0.0, 50.2 | 0, 0, 0 | estimate | estimate |
| pad_back_left | 14 x 10 x 0.2 | 0.3 | -45.0, 38.0, 45.4 | -3, -17, 0 | estimate | estimate |
| pad_back_right | 14 x 10 x 0.2 | 0.3 | -45.0, -38.0, 45.4 | 3, -17, 0 | estimate | estimate |
| pad_side_left | 14 x 10 x 0.2 | 0.3 | -30.0, 58.4, 36.0 | -90, 0, 0 | estimate | estimate |
| pad_side_right | 14 x 10 x 0.2 | 0.3 | -30.0, -58.4, 36.0 | 90, 0, 0 | estimate | estimate |
| pad_face | 14 x 10 x 0.2 | 0.3 | 46.0, 0.0, 48.5 | 0, 8, 0 | estimate | estimate |

## 2. Wiring: `configs/wiring_v2.toml`, checked by `crab/wiring.py`

One file lists every power connection (wire gauge, length, connectors, how many parallel conductors) and every signal: the Radxa's header
pins as printed in Radxa's table, three I2C buses with every address, I2S, USB, the camera flex, the touch pads. Wire lengths
come from the layout (straight distance x 1.25 + a 40 mm loop), except bought cables (the 900 mm servo cable is 900 mm). Resistances,
connector ratings, the fuse curve and the regulator numbers are in `electronics.toml` with their sources. `python -m crab.wiring` prints the
table and runs the checks; the wiring diagram `hardware/wiring.svg` is drawn from the same file.

The power chain: pack -> XT60 -> 15 A fuse -> INA219 with its shunt -> main switch -> the servo hub's 7.4 V rail (12 leg servos, **straight
from the pack, 6.4-8.4 V**), regulator A (5 V: the Radxa, the audio amplifier and, through the Radxa's own 3.3 V pins, all sensors) and
regulator B (5 V: the six arm servos, on their own hub headers). Signals: the servo bus (a star at the hub, IDs 1-12 as in
`robot/servos.py`), `i2c3` (INA219 0x40, MPR121 0x5A: Radxa pins 3 / 5), `i2c4` (the BNO055 alone: pins 27 / 28), `i2c5` (five
VL53L0X and the VL53L5CX: pins 31 / 29, each brought up through its XSHUT / LPn pin on pins 11, 13, 15, 22, 36, 32 and given 0x30-0x35).

What the checks cover, and they all pass on the real netlist (tests make each fail on purpose):

- **Pins**: no header pin used twice; no signal on a power, ground or unconnected pin; every function must be one Radxa lists for that pin.
- **I2C**: no clash of addresses on a bus, with the VL53 sequence modelled (all wake at 0x29; with their reset lines they are brought up one at
  a time and keep 0x30-0x35; two without a reset line would clash); the BNO055 stretches the clock and may not share its bus
  (`docs/research/audit/ext_runtime_power.md` section 6); no bus more than 60 % busy (i2c3 26 %, i2c4 4 %, i2c5 24 %).
- **Supply**: every load is reached from the pack; the netlist is a tree.
- **Ratings**: for every wire, connector, the switch and the shunt, the current it can see against its rating. Two figures per load: what it
  can draw for as long as it likes (a leg servo holding the maker's stall torque: 1.5 A) and what it draws in a stall (its locked-rotor current
  at a full pack: 3.05 A, for the 2 s its overload protection allows). Everything upstream is capped by what the fuse (15 A) and the BMS (25 A)
  let through. A wire passes if it carries the first at its rating and the second at 2x its rating. Regulator loads and the Radxa's 3.3 V
  pins are checked against their limits (regulator A at worst 1.06 A of 5 A, regulator B 4.80 A of 5 A, 3.3 V 0.20 A of an assumed 0.5 A).
- **Coverage**: every part of the layout is wired and every wired part is in the layout.

The checks found things, and the netlist now holds the fix; each is a **BOM change**, listed in section 7:
the BOM's 0.005 ohm shunt would burn 1.1 W at the fuse's 15 A in a 1 W resistor (0.002 ohm burns 0.45 W and still reads to 5 mA);
Radxa's table lists no I2C function on pins 3 and 5 (so the INA219 and MPR121 sit on a kernel `i2c-gpio` bus there); three servos on one daisy-chain cable would put 8 A through a 3 A connector (so each servo has its own hub header: 12 + 6
headers, not the BOM's 6 + 6).

## 3. The circuit model (`crab/wiring.py` `Circuit`, used by `SimBackend`)

Solved once per control tick (10-20 ms; 10 fixed-point iterations over the netlist tree), deterministic. Equations, with the source of each number:

- **Pack**: `V = V_oc(soc) - R_pack(soc) I`, `d soc/dt = -I / (Q x 0.8)`. Two Samsung 25R cells: 2.5 Ah (datasheet); resistance 0.06 ohm (estimate:
  13.2 mohm AC each from the datasheet, about 1.7x that at DC, plus the BMS); the open-circuit curve is a typical Li-ion one (estimate); the
  resistance doubles at empty (estimate).
- **Wires**: the drop is current x the loop resistance of the edge: `R = 2 rho L / n + 2 sum(contact)` with copper resistance per AWG
  (datasheet) and connector contacts (estimates, 0.5-50 mohm).
- **Leg servo** (the measured BAM motor: `kt` 1.275 N*m/A, `R` 2.75 ohm): `I_motor = (duty V - kt w) / R`, torque `kt I_motor`, and the
  supply gives `I_supply = duty x I_motor` (a chopped drive: the pack only supplies the on-time) plus 40 mA idle (estimate). The twin
  accumulates `duty^2` and `duty x w` every physics step. The torque limit and stiffness of each servo follow **the voltage at its own
  terminals** (stall torque is 3.43 N*m at 7.4 V, scaled by V / 7.4), in both the first-order and the full BAM model.
- **Regulators** (7Semi AP64500 module): `V_out = min(5, V_in - 0.6)` (dropout, estimate), minus 1 ohm x the current above 5 A (datasheet limit),
  efficiency from a light-load curve through the listing's 90 % peak (estimate), 4 mA of its own; off below the chip's 3.8 V (datasheet); switched
  off 0.1 s when asked for 1.3x its limit; optionally off below the **module's listed 7 V** (`enforce_module_min`, off by default because the
  chip itself runs lower: the owner decides when the module is measured).
- **Computer**: constant power by activity, idle 1.3 W, camera 2.1 W, NPU 3.4 W (all estimates, Radxa publishes none); below 4.5 V (estimate)
  for 20 ms it resets: the twin's firmware sees nothing for 30 s (boot), commands to the servos are ignored (the servos hold their last goal),
  then the mind restarts from its saved memory (`pinch/twin/core.py`).
- **Sensors**: current from `electronics.toml [load]` (BNO055 12.3 mA and INA219 1 mA datasheet; VL53L0X 20 mA datasheet; the rest estimates),
  fed from the Radxa's 3.3 V pins, which draw from regulator A through an 85 % converter.
- **Fuse**: it blows when `sum dt / t_open(I / 15 A) >= 1`, `t_open` interpolated on a log-log curve through the datasheet limits at 110 % and
  135 % and typical ATO values at 200, 350 and 600 % (estimates); the damage cools with a 10 s time constant below the rating. The BMS
  cuts the pack above 25 A for 0.1 s (estimate; the fuse is faster above about 1.7x). Both are permanent in a run.
- **INA219**: reads the shunt voltage where it is wired (between the fuse and the switch: the whole pack current) in 10 uV steps with its gain
  error, and the bus voltage in 4 mV steps; the mind's battery level comes from it.
- **IMU**: the chip sits at its mount, measures angular rate and specific force at that spot (the MuJoCo sensors on its site, lever arm
  included), fuses gravity from the gyro and the accelerometer with a 0.5 s time constant (estimate), and the output is turned into body axes.
  Walking accelerations therefore leak into the tilt a little, as in the real chip. `realism="ideal"` stays exact at the body centre.

## 4. Results (`sim/circuit_check.py`)

Pack at 100 %, camera on, first-order servo model, scripted trot at the gait's cap (10 cm/s), 8 s per case. Current is the pack current
averaged over each 10 ms control tick.

| Case | Pack A, peak / mean | Pack V, lowest | Leg servo V, lowest | Arm V | Computer V | Regulator A / B load, peak | Fuse damage | Trips |
|---|---|---|---|---|---|---|---|---|
| standing | 1.10 / 1.10 | 8.33 | 8.28 | 4.99 | 4.98 | 0.58 A / 0.25 A | 0.000 | none |
| walk forward, at the cap | 2.17 / 1.59 | 8.27 | 8.13 | 4.97 | 4.98 | 0.58 A / 0.49 A | 0.000 | none |
| walk sideways, at the cap | 3.07 / 1.82 | 8.21 | 7.95 | 4.97 | 4.98 | 0.58 A / 0.46 A | 0.000 | none |
| get-up swing from the back | 16.23 / 3.74 | 7.42 | 6.59 | 4.90 | 4.98 | 0.58 A / 3.56 A | 0.000 | none (the swing did not right it this time) |

The earlier lumped model on the same cases (one rail model at 7.4 V, no wires, no fuse, servo motor current counted as supply current):

| Case | Pack A, peak / mean | Pack V, lowest | Servo bus V, lowest |
|---|---|---|---|
| standing | 1.76 / 1.73 | 8.29 | 7.34 |
| walk forward | 11.51 / 8.38 | 7.70 | 6.85 |
| walk sideways | 12.26 / 9.03 | 7.65 | 6.81 |
| get-up swing | 46.55 / 14.05 | 5.60 | 0.00 (collapsed) |

Reading it:

- Standing costs 1.1 A (9 W): 12 servo idles 0.5 A, the computer 0.3 A, the rest the arms' holding current and the sensors.
- Walking draws a mean 1.6-1.8 A and peaks at 2.2-3.1 A: **far below the 10 A of the BOM's estimate and the 11-12 A of the earlier lumped
  model**. The lumped model counted a servo's motor current (torque / kt) as supply current; a chopped drive only pulls `duty x I_motor` from the pack,
  and at walking speed the duty is 0.2-0.3. Real measurements may differ (switching losses, regeneration into the hub capacitors, a real servo's
  quiescent current), so the INA219 on the first bring-up decides.
- A get-up attempt is the hard case: 16.2 A for a moment (10 ms mean), the pack dips to 7.4 V and the lowest servo to 6.6 V. The fuse did
  not blow (damage 0.000: 16 A is only 1.08 x its rating and a swing lasts under 2 s), the BMS did not trip, the computer's rail stayed at
  4.90 V or more. A longer stall would blow the fuse: at 22 A (1.5 x the rating) in about 15 s, at 25 A the BMS first.
- No brown-out anywhere on a full pack. The regulator modules' listed 7 V minimum is the one thing that can reset the robot: the loaded pack falls
  under 7 V after about 83 minutes of the mixed day (at roughly 10 % charge), so if the modules really stop at 7 V the computer resets then, 81
  minutes in, instead of the 92 minutes the pack has. Measure one module's real cut-off first (BOM_CHANGE.md risk 3).
- The full BAM servo model gives the same walk within 10 % (2.37 A peak, 1.63 A mean).

Battery life: "mixed day" is 50 % standing with the camera on, 20 % walking forward, 10 % walking sideways, 20 % standing with the computer idle
(servo loads replayed from the runs above; `python sim/power.py` runs them): **92 minutes** from full to the pack's usable 80 %
(2.5 Ah x 0.8 at a mean 1.26 A); 83 minutes until the loaded pack passes 7 V; 81 minutes to a reset if the modules obey their 7 V rating. (The 25R pack
is 18.5 Wh.)

## 5. What is still an estimate, and what only real hardware settles

Estimates in this chapter (each flagged `estimate` with its reasoning in the files; see "Source audit" below for the count and the reason each one is still open): 33 of 42 part outlines and most masses, the pack's resistance
and discharge curve, the BMS limit, every connector's rating and contact resistance, the wire ampacities, the fuse curve above 135 %, the
regulator's dropout and light-load efficiency, the computer's power per activity, its brown-out voltage and boot time, the 3.3 V pin budget,
the servo's idle current and sustained current, the IMU's fusion time constant, the 60 g harness.

Only the real parts settle: the STS3215's behaviour at 8.4 V (the model was fitted at 7.4 V; the torque and speed scaling above is the motor
equation, the firmware's PWM limit is a guess), the pack's true resistance and BMS cut-off under a 16 A pulse, the regulator modules' real
minimum input, what the Radxa does at 4.5-4.8 V, whether a 15 A fuse in this holder survives the get-up, how much of the walking supply
current is switching loss, the shunt's accuracy, whether `i2c-gpio` at 400 kHz is stable beside the BNO055 bus, clock stretching on the RK3566,
and every outline in the placement table that is an estimate.

## Source audit (2026-11-21): what each file claims, and why the rest is still an estimate

Every `estimate` now starts its `why` with a tag, so the leftovers can be sorted: `needs bench:` (only a measurement on the real part or
pack settles it), `design choice:` (we decided it, no datasheet can give it) or `unverified:` (a maker or seller figure that exists but whose
page could not be opened, or a generic class rating).

| File | measured | datasheet | estimate | needs bench | design choice | unverified |
|---|---|---|---|---|---|---|
| `configs/electronics.toml` before | 23 | 35 | 105 | - | - | - |
| `configs/electronics.toml` now | 23 | 49 | 91 | 46 | 24 | 21 |
| `configs/layout_v2.toml` before (size or mass fields) | 0 | 10 | 76 | - | - | - |
| `configs/layout_v2.toml` now (size or mass fields) | 0 | 14 | 72 | 25 parts | 10 parts | 4 parts |

(The layout file has one `why` per part for both its size and its mass, so its tags are counted per part; 39 parts still carry an estimate.)

Upgraded to `datasheet` this pass, each opened and read:

- VL53L0X (ST datasheet DocID029104 Rev 2, the Pololu-hosted copy because st.com did not answer): field of view 25 deg (sec 5.1), the 33 ms default
  timing budget = 30 Hz (p.23 note 3), reliable range 0.4 m = Table 11, grey 17 % target, outdoor overcast, minimum.
- VL53L5CX supply current 0.100 A (was 0.060 A; Pololu's carrier page quoting ST: typical 100 mA ranging, 150 mA peak).
- MAX98357A quiescent current 2.75 mA at 5 V (was 2.4 mA, the 3.7 V figure; Maxim datasheet p.4).
- I2S microphone supply current 1 mA = the MSM261S4030H0R maximum (DS-050 p.3: 750 uA nominal, 1000 uA max).
- XT60 rated 30 A and 0.55 mOhm contact resistance (was 0.5; Amass spec sheet V1.2).
- Wire ampacity from the Wikipedia AWG table, 75 C column: 14 AWG 20 A (was 32 A, an unsourced figure), 20 AWG 11 A, 22 AWG 7 A, 24 AWG 3.5 A,
  26 AWG 2.2 A, 28 AWG 1.4 A (the last five were already these numbers).
- The four hip servos' mass: 55 +- 1 g on DFRobot's ST3215 7.4 V page (Waveshare and Feetech print no weight).

Read and found NOT to settle an estimate (so they stay estimates, with the reading in `why`):

- VL53L0X ranging noise: Table 12 prints a standard deviation of 4 % (white, 120 cm) to 7 % (grey, 70 cm) of the distance, higher than our 1.5 %
  at the far end; the cliff beams work at a few cm where the datasheet gives nothing. Bench: sigma at 3, 5 and 10 cm over our floor.
- AP64500 (DS41979 Rev 5-2): Fig. 4 (12 V in, 5 V out, the chip's EVM inductor) reads about 0.86 at 50 mA and 0.2 A, 0.90 at 1 A, 0.91 at 5 A; the
  7Semi module has its own inductor and our input is 7.4-8.4 V, so the module's curve stays an estimate. The chip's quiescent current is 25 uA
  (the module's LED is most of our 4 mA), its peak current limit 9.2 A (the module is sold as 5 A), and it prints no dropout number.
- Samsung INR18650-25R (Samsung SDI spec sheet, copy at akkuman.de, p.3 and p.6): 18.33 x 64.85 mm, 43.5 g typical (45 g max), AC impedance 12.93 mOhm
  (<= 18), DC impedance 19.23 mOhm (10 A to 1 A); the only curves printed are loaded discharge curves from 1C to 25 A, not an open-circuit curve.
  The pack's resistance includes a BMS and tabs nobody published, so it stays an estimate (the cells alone are 38 mOhm for two).
- INA219 (SBOS448G): no noise figure is printed at all. Radxa ZERO 3W product brief: only a 5 V / 2 A supply, no consumption, no minimum input, no weight.
- Mass and height are missing for most boards (Waveshare, Feetech, Adafruit print sizes without weight or thickness).

Found but NOT applied (it would change `sim/models/crab_v2.xml`, which is outside this task's areas; the model must be rebuilt by its owner first):

- The IMU board we buy (7Semi, Evelta page) is 21 x 27 x 4 mm, the layout box is Adafruit's 20 x 27 x 4 mm.
- The I2S microphone we buy (DFRobot SEN0526) is 16 x 11 mm per its wiki, the layout box is the analogue board's 18 x 12.5 mm.
- Pololu's genuine VL53L0X and VL53L5CX carriers are 13 x 18 mm; the clones we buy are bigger and unmeasured.

Sources that refused or hung: robu.in and hubtronics.in (403), analog.com and st.com (no answer), Amazon (not tried).

## 6. The face

The face is a Waveshare 6.25 inch 720 x 1560 colour IPS panel over micro-HDMI (`hardware/BOM.md`; the layout's part `oled` keeps
its old id), in a lid hinged at the top front of the shell and set by hand on a printed friction hinge, so the model holds it fixed (0.35 rad
back from vertical). The module is modelled as 158 x 76 x 8 mm and 0.30 kg (an estimate and the listed worst case; 0.15 kg is hoped for:
weigh it, bench order 1); it takes about 2 W from regulator A over its USB-C lead, and its flat cable runs through the hinge. In the model
the lit area (144.18 x 66.6 mm) is a visual box named `face_screen` with material `face_mat` (emission 0.9) and texture `face_tex`
(256 x 128), which `sim/desk.py set_face()` fills from `pinch.face.Face().rgb(scale=2)`. The fit check covers the screen's window like the
camera's. `sim/render_v2_face.py` draws a picture of it.

## 7. Parts-list changes this work needs (`hardware/bom.py` and `BOM.md` are not edited here)

- **Shunt**: 0.002 ohm, 2512, 1 W, instead of 0.005 ohm (it burns 1.1 W at the fuse's limit).
- **Servo cables**: 12 leg cables of 900 mm plus 6 arm cables are needed (one per servo, no daisy chain): 3 packs of the 5264 cable, not 2; the arms'
  500 mm is a guess.
- **Servo hub board**: 18 three-pin headers, two 1000 uF capacitors and a separate arm rail need about 55 x 36 mm, not "3 x 2 cm"; two header strips are enough.
- **Dock pogo pins**: use two per polarity (4 of the pack of 5): their current rating is not listed, 1 A each is assumed.
- **Waveshare driver board**: its wiki lists a 9-12.6 V input for the servo supply; our design feeds it only USB and takes signal + GND from it. Check on the
  bench that the bus works without its DC jack powered.
- **Pack, camera, screen, switch, fuse holder, VL53L0X and VL53L5CX boards, driver board height**: measure when they arrive and edit the layout.
- **Firmware not edited here** (other owners): `robot/drivers/i2c_devices.py` still drives the old 2.42 inch OLED on I2C4 (pins 27 / 28) and puts the
  IMU on a shared bus on pins 3 / 5; the netlist has the BNO055 alone on i2c4, and the HDMI panel needs no I2C.
