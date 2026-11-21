# Electronics in the twin

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
  the twin uses it for the resized design only (CRAB_DESIGN=v2, [V2_MODEL.md](V2_MODEL.md)).
- `datasheet`: printed in the maker's datasheet or product page, and the ref is the page it was read from.
- `estimate`: our own reasoning, written in `why`. Replace it when a real number exists.

The loader is `crab/electronics.py` (`Data()("imu.fusion_rate_hz")`); `tests/test_electronics.py` fails if a number
has no valid source, no reference or an estimate without its reasoning. The datasheet and measured numbers were read
on 2026-11-21 (BNO055 datasheet rev 1.8, INA219 SBOS448G, MPR121, Feetech and Pololu pages, the BAM JSON). ST's own
VL53L0X datasheet could not be fetched, so its range, accuracy and resolution come from Pololu's product page, which
quotes ST, and the page is the ref. Today's (v1) STS3032 torque and speed numbers are not in this file: they stay in
`crab/params.py`. The v2 servo's measured numbers are (`[leg_servo]`), and so are v2's pack and servo rail
(`[battery_v2]`, `[regulator_v2]`, swapped in by `Data.for_design("v2")`).

## What is modelled

| Part | What the twin does |
|---|---|
| Leg servo commands | Every servo target reaches the joint late: BAM's measured 5.0 ms plus the time the 12-servo write takes on the 1 Mbaud bus (0.9 ms). Physics runs in 2 ms steps, so the delay is 3 steps (6 ms). |
| Arm servo commands | 15 ms (the next 50 Hz PWM frame plus the servo's own electronics): 8 steps. |
| Leg encoders | Angles come in 4096 steps per turn. |
| Servo voltage report | The bus voltage in 0.1 V steps. |
| IMU (BNO055) | Sampled at 100 Hz (reads inside one sample match); gyro noise 0.3 deg/s rms; accelerometer noise from its density and 62.5 Hz bandwidth; a fixed gyro bias and accelerometer offset per robot (a quarter of the datasheet limit, because fusion mode calibrates); a first-order lag of about 34 ms on the gravity direction and 9 ms on the gyro. |
| Distance beams (VL53L0X) | 30 Hz samples per beam, noise of 1.5 % of the range (at least 2 mm), 1 mm steps, no answer beyond 2 m, and dropouts that grow from 0.2 % on a good surface to about 50 % at the end of the range. A dropout reads as "no answer", which the cliff reflex already treats as unknown. |
| Battery | State of charge falls with the current the servos and the computer draw (servo torque divided by each servo's N*m per amp, plus idle current, plus 4 W for the computer, through a 90 % regulator). The pack voltage follows a LiPo curve minus current times internal resistance (0.1 ohm, doubling when empty). |
| Regulator | The 5 V bus holds until the pack drops within 0.5 V of it, sags by the wire resistance times the servo current, and droops steeply past the 5 A limit. |
| INA219 | Reads pack voltage and current in its 4 mV and 10 uV steps, with a gain error (0.5 % max) and a little noise. |
| Low battery | In the twin the mind's battery level comes from the INA219 reading, so hunger, the low-battery warning and the safety stop (6.4 V under load) happen by themselves. A pack at 0 % trips the stop within seconds. |

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
- **Servo speed, load and temperature feedback** are still exact (temperature is a constant 35 C).
- **Bus transmission errors** (garbled replies) and I2C NACKs.
- **Real current of the STS3032 legs**: current comes from torque divided by `kt_current` in `crab/params.py`, so
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
| rate_hz | 30 | Hz | estimate |
| fov_deg | 25 | deg | estimate |
| reliable_range_m | 0.4 | m | estimate |
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
falls from 18.4 to 17.5 rad/s. In short: today's controllers are robust to this electronics, with a measurable but
small cost, and the benches are not sensitive enough to rank them further. The robustness bench does not model the
battery, so the power model has no effect there.

## Using it

- Test a controller against realistic sensors: pass `realism="datasheet"` to `PinchSim` or `SimBackend`, or `--realism
  datasheet` to the two benches.
- Swap a part: edit its number in `configs/electronics.toml` with its source; nothing else changes. A new servo:
  change `leg_servo.command_delay_s` (BAM's m1 fits 1.0 ms and m6 5.0 ms) and the servo model in `crab/params.py`.
- Test low battery: `PinchSim(realism="datasheet", soc=0.1)`.
