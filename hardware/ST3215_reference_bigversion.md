# ST3215 (7.4 V) servo: reference facts

Same part as **Feetech STS3215 C001** (gear 1/345), the servo used by Open Duck Mini and SO-100/101.
⚠️ Waveshare also sells a **12 V / 30 kg·cm "ST3215"**. We want the **7.4 V / 19.5 kg·cm** version
(robu.in SKU R258500 is listed as "19.5kg.cm @ 7.4V").

Sources: [Feetech STS3215 spec (translated)](https://core-electronics.com.au/attachments/uploads/sts3215-smart-servo-datasheet-translated.pdf),
[Waveshare product page](https://www.waveshare.com/st3215-servo.htm), [Waveshare wiki](https://www.waveshare.com/wiki/ST3215_Servo),
[SO-ARM100 STEP](https://github.com/TheRobotStudio/SO-ARM100), [Rhoban BAM](https://github.com/Rhoban/bam),
[Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground), [rustypot](https://github.com/pollen-robotics/rustypot).

## Mechanical (used by `cad/`)

| Item | Value |
|---|---|
| Case | 45.2 × 24.7 × 35 mm (35 incl. boss; 32 main body; 29 across horn region) |
| Output shaft | centred across the 24.7 mm width, 10.11 mm from the output end |
| Spline | 25T, OD 5.9 mm, M3×6 horn screw; spline 3.4 mm proud of the 29 mm face |
| Idler (rear) shaft | Φ6 mm boss, coaxial, 4.1 mm proud; spline tip → idler boss 36.5 mm |
| Body mounting holes | 4 × M2 self-tapping (Φ1.5 pilot), rectangle 20.7 mm (along length) × 20.5 mm (across); near pair 18.41 mm from output end (8.3 mm from shaft axis) |
| Horn (aluminium, 2 incl.) | Φ19.2 mm, 4 holes on Φ14 mm PCD, M3 (links attach with M3×6) |
| Connector | 2 × Molex 5264 3-pin 2.54 mm (daisy chain): pin 1 GND, 2 V+, 3 Signal; 15 cm cable |
| Mass | 55 g (bare) |
| Backlash | ≤ 0.5° |

## Electrical

| | @ 6 V | @ 7.4 V |
|---|---|---|
| No-load speed | 4.40 rad/s | **5.45 rad/s** |
| Stall torque | 1.62 N·m | **1.91 N·m** (datasheet) |
| Stall current | 2.0 A | **2.5 A** |
| Rated torque / current | 0.39 N·m / 0.5 A | 0.49 N·m / 0.65 A |

- Working voltage **4–7.4 V** (over-voltage protection is configurable). **A 2S LiPo is 8.4 V when full, so regulate the servo bus to 7.4 V.**
- Encoder: 12-bit, 4096 counts/rev, centre 2048. No mechanical stop.
- Bus: half-duplex TTL UART, 8N1, default **1 Mbps**, IDs 0–253 (254 = broadcast), new servos ship as **ID 1**.
- Protocol: Feetech SCS (Dynamixel-1.0-style): PING, READ, WRITE, SYNC_READ (0x82), SYNC_WRITE (0x83).

### Key registers (decimal address, bytes)

| Addr | Size | Name | Notes |
|---|---|---|---|
| 5 | 1 | ID | EEPROM (unlock via 55) |
| 6 | 1 | Baud | 0 = 1 Mbps |
| 9 / 11 | 2 / 2 | Min / max angle limit | |
| 14 / 15 | 1 / 1 | Max / min input voltage | 0.1 V units |
| 16 | 2 | Max torque | 1000 = 100 % |
| 21 / 22 / 23 | 1 each | P / D / I gains | factory P 32, D 32, I 0; Open Duck runs P 32, D 0 |
| 33 | 1 | Mode | 0 = position |
| 40 | 1 | Torque enable | 128 = set current position as centre (2048) |
| 41 | 1 | Acceleration | 0 = max |
| 42 | 2 | Goal position | sign = bit 15 |
| 46 | 2 | Goal speed | steps/s |
| 48 | 2 | Torque limit | |
| 55 | 1 | EEPROM lock | |
| 56 / 58 / 60 | 2 each | Present position / speed / load | |
| 62 / 63 | 1 / 1 | Present voltage (0.1 V) / temperature (°C) | |
| 69 | 2 | Present current | 6.5 mA units |

Python driver: **rustypot** (Pollen Robotics, Apache-2.0, Pi wheels): `Sts3215PyController`, with radians where raw 2048 = 0 rad.

## Measured actuator model (Rhoban BAM, `feetech_sts3215_7_4V`, model m1)

kt = 1.1776 N·m/A, R = 2.4787 Ω, armature = 0.02608 kg·m², friction_base = 0.05152 N·m,
friction_viscous = 0.05989 N·m·s/rad, max_velocity = 5.288 rad/s, command_delay ≈ 1 ms (m3–m6 ≈ 5 ms).
Firmware: kp_fw = 32, error_gain = 0.166, max duty 0.97, vin = 7.4 V.

MuJoCo position-actuator equivalent (BAM `to_mujoco`):
**kp = 0.166·32·7.4·0.97·kt/R ≈ 18.1**, **forcerange = vin·kt/R ≈ ±3.52 N·m**,
**joint damping = viscous + kt²/R ≈ 0.62**, **frictionloss ≈ 0.052**, **armature ≈ 0.026**.
Open Duck Mini uses kp 13.4–17.8, forcerange ±3.2–3.35, damping 0.56–0.60, armature 0.027–0.028.

⚠️ BAM's force limit (3.5 N·m) is ~1.8× the datasheet stall torque (1.91 N·m). We simulate with the
measured model but **check designs against the datasheet 1.91 N·m**, and RL randomises the force
limit down to the datasheet value, so policies work in the worst case.
