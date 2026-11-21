# Servo choice for the 12 leg joints (research note, 2026-11-21)

Question from the owner: size is not a hard limit, so why not use a servo whose measured data is public?
Method: read the repo files, then fetched the sources below. Anything I could not open is marked UNVERIFIED.
No number was invented. Estimates (mine) are labelled ESTIMATE.

## 1. Short answer

Use the **Feetech STS3215, the 7.4 V version, run on a regulated 7.4 V rail**. It is the only candidate where
(a) a published, measured model exists, (b) the model's voltage is the voltage we would really run, (c) the gears
are metal, and (d) it is cheaper than today's STS3032 in India when in stock. The price is a bigger robot (about
28-32 cm, about 1.2 kg, ESTIMATE) and one real supply problem: the exact variant the model was fitted on is sold
out in India today.

Runner-up: Dynamixel XL330-M288-T, if the owner wants to keep the 20 cm body. It has a public model and a working
walker, but the model was fitted at about 7.5 V on a servo rated 6 V max, it has plastic gears, and no Indian seller
was found.

## 2. What a "public measured model" really is here

Rhoban/bam (629 stars, pushed 2026-11-10, Apache-2.0). `bam/params/` holds one folder per servo, each with six
model variants m1-m6. Exactly these exist:
`erob80_50, erob80_100, feetech_sts3215_7_4V, mx64, mx106, waveshare_st3025, xl320, xl330`.
No STS3032. No XL430, XC330, STS3046, STS3020.

| Servo in BAM | Voltage in the BAM actuator class | Firmware gain in the class | Notes (all read from the repo) |
|---|---|---|---|
| Feetech STS3215 | 7.4 V | kp 32, error_gain 0.166 (oscilloscope) | m1: kt 1.178, R 2.48 ohm, armature 0.0261, friction 0.0515 base + 0.0599 viscous, max_velocity 5.29 rad/s, command_delay 1.05 ms |
| Dynamixel XL330 | 7.5 V (class default) | kp 400 | m1: kt 0.371, R 3.45, armature 0.0016, delay 5.9 ms. The JSON files carry no voltage field; the voltage comes from the class. |
| Waveshare ST3025 | 12 V | kp 32 | 124 g servo (Waveshare): too heavy, not a candidate |

Key reading: the fit is a physical model (voltage, back-EMF, load-dependent friction), so it only holds at the
voltage and firmware gain it was recorded at. Do not assume it holds at another voltage or a different winding.

Cross-check I made: BAM max_velocity 5.29 rad/s matches the C001 (1/345 gear) speed scaled to 7.4 V
(0.238 s/60 deg at 6 V is 4.4 rad/s; times 7.4/6 is 5.4 rad/s). And armature 0.026 is large, which fits a 1/345 gear.
So the BAM STS3215 is most likely the C001 variant (my inference; the BAM files do not name the part number).

Other places checked:
- **Open_Duck_Mini** (4,763 stars, pushed 2026-01-31): docs/sim2real.md says it uses BAM and links
  `params/feetech_sts3215_7_4V`. A working sim-to-real walker on this servo and model. Its trainer
  Open_Duck_Playground has 205 stars, last push 2025-08-11 (stale).
- **pollen-robotics/microduck_rl** (2,937 stars, pushed 2026-11-17): README says servos are "the same BAM M6 XL330
  model". Real robot: Microduck (pollen-robotics/microduck, 13,957 stars, pushed 2026-11-20), about 25 cm, about
  800 g, 15 servos on an RK3566, the same chip family as our Radxa ZERO 3W. Its code randomises battery voltage
  6.5-8.2 V, sag, delay, friction, backlash, firmware kp 200. Best-maintained recipe I found.
- **MuJoCo Playground** (2,326 stars, pushed 2026-11-21): `tasks/sysid` has `so101` and `yam`. The SO101 task fits
  24 numbers (armature, frictionloss, stiffness, damping for 6 joints) to teleop logs of ONE arm. It is a method,
  not a reusable servo model.
- **MuJoCo Menagerie** (4,394 stars): has `trs_so_arm100` and others. I listed folders only and did not open the
  actuator tuning, so I cannot say whether it is a measured fit. UNVERIFIED.
- **LeRobot** (29,009 stars): drives STS3215 on SO-100/101. I found no servo simulation model there. UNVERIFIED
  (not read in depth).
- **SO-ARM100** (8,178 stars): its BOM uses STS3215 7.4 V in C001, C044 and C046 gear versions and a 12 V C018 version.

## 3. Candidates

### (1) Feetech STS3032 (today)
Numbers are from the repo's own table (crab/params.py, Evelta/Feetech), not re-fetched today.
20 g, 23 x 12 x 27.5 mm, 4.5 kg.cm stall at 6 V (about 3.75 at 5 V), 0.09 s/60 deg, magnetic encoder,
position/load/temperature, STS bus via FE-URT-1.
BOM: Evelta INR 3,327.60 each, 27 in stock on 2026-11-18. No published model. 12 + 2 spares = INR 46,586.

### (2) Feetech STS3215
Source: Evelta product pages fetched today. Price is INR 2,224.30 incl. GST (INR 1,885 ex. GST) for all 7.4 V versions.

| Version | Torque / speed (Evelta) | Voltage (Evelta) | Stock today |
|---|---|---|---|
| C001, 1/345 | 19.5 kg.cm stall and 6.5 rated at 6 V; 0.238 s/60 deg at 6 V; no 7.4 V figure given | 6-7.4 V | Sold out (Evelta). Hubtronics: INR 1,785 incl., 0 in stock |
| C044, 1/191 | 12.5 / 16 kg.cm at 6 / 7.4 V; 0.137 / 0.116 s/60 deg | input 5-8.4 V | page says "4 in stock" (ambiguous display) |
| C046, 1/147 | 9.2 / 14.4 kg.cm at 6 / 7.4 V; 0.111 / 0.094 s/60 deg | input 5-8.4 V | Sold out |
| C018, 12 V, 1/345 | 30 kg.cm stall, 10 rated, 2.7 A at 12 V; 0.222 s/60 deg | 4-14 V | "98 in stock"; INR 2,419 incl. (INR 2,050 ex.) |

Common: 55 +/- 1 g, 45.2 x 24.7 x 35 mm, 12-bit magnetic encoder, position/speed/load/voltage/temperature feedback,
same STS bus and the same FE-URT-1 adapter we already plan (robu.in INR 1,069, from the BOM). Gear material is
listed inconsistently on Evelta ("metal", "copper", "steel"); treat as metal.
Model: yes, BAM `feetech_sts3215_7_4V`, valid at 7.4 V, kp 32. The 12 V C018 is a different winding and is NOT covered.
Price 12 + 2: C001/C044/C046 14 x 2,224.30 = **INR 31,140**; C018 14 x 2,419 = INR 33,866.
Import fallback: the SO-ARM100 BOM lists C001 at $13.89 each on Alibaba (price date UNVERIFIED), so about $195 for 14
before shipping and customs; I could not check Indian customs cost.

### (3) Dynamixel XL330-M288-T
Source: ROBOTIS e-Manual, fetched today. 18 g, 20 x 34 x 26 mm, gear 288.4:1 **engineering plastic**, cored motor,
3.7-6.0 V (5.0 V recommended). Stall: 0.42 N.m at 3.7 V, 0.52 N.m (5.3 kg.cm) at 5 V / 1.47 A, 0.60 N.m at 6 V /
1.74 A. No-load: 76 / 103 / 123 rpm at 3.7 / 5 / 6 V. 12-bit absolute encoder (AS5601). Feedback: position, velocity,
current, temperature, input voltage. Bus: 3.3 V TTL half-duplex, Dynamixel protocol, up to 4 Mbps.
Rated (ROBOTIS estimate, 20 % of stall) 0.10 N.m, as quoted by OpenElab (secondary source).
Adapter: U2D2 (ROBOTIS). The Feetech FE-URT-1 does not speak this protocol (my knowledge; not checked on a page).
Model: yes, BAM `xl330` (m1-m6) and microduck_rl (m6). Conflict: the model class uses 7.5 V and the robot code
samples 6.5-8.2 V, while ROBOTIS lists 6.0 V as the top of the range. I found nothing that explains this. At our 5 V
we would be outside the fitted domain. UNVERIFIED why Pollen can run it that high.
India: no listing found. robotis.us shows $27.49 but sold out (page fetch was rate-limited; figure from a search
snippet). Europe: EUR 39.78 (Funduino) to EUR 40.20 (Generation Robots). U2D2 $43.57 (Robotshop, search snippet).
Mouser India is an authorized ROBOTIS distributor but I could not load an INR price (page timed out).
14 servos at about EUR 40 is about EUR 560 plus U2D2 plus customs, likely above INR 55,000 (my arithmetic at roughly
INR 100 per EUR; rate UNVERIFIED), import only.

### Extra candidates (not added)
ST3025 is 124 g and 12 V: heavier than needed. XC330 (metal gear) and XL430 have no BAM fit. Neither clearly beats
the three.

## 4. What it does to the robot (ESTIMATE where marked)

Today's need: standing is about 33 % of the STS3032 rated torque (1.5 kg.cm at 6 V, 1.25 at 5 V), so about
0.41 kg.cm per joint at 0.61 kg.

| | STS3032 | STS3215 (7.4 V) | XL330 (5 V) |
|---|---|---|---|
| Mass of 12 servos | 240 g | 660 g | 216 g |
| Body mass | 0.61 kg | about 1.2 kg (ESTIMATE: +420 g servos, bigger pack, heavier brackets) | about 0.59 kg |
| Robot size | 20 cm | about 28-32 cm (ESTIMATE; servo volume is about 5x) | about 20 cm (volume 2.3x, pockets change) |
| Torque need at a joint | 0.41 kg.cm | about 1.1 kg.cm (ESTIMATE: mass x 2, legs 1.4x longer) | about 0.4 kg.cm |
| Rated torque | 1.25 kg.cm | 6.5 kg.cm at 6 V (about 8 at 7.4 V, scaled) | 1.06 kg.cm |
| Standing load vs rated | 33 % | about 14-18 % | about 39 % |
| Top joint speed | 11.6 rad/s at 6 V | about 5.3 rad/s (C001 at 7.4 V, matches BAM) | 10.8 rad/s |
| Battery | 2S, servos on 5 V buck | 2S, servo rail regulated to 7.4 V, pack about 1500-2200 mAh (ESTIMATE) | 2S with 5 V buck, as today |
| Redesign | none | full: CAD (legs, body, dome, dock), servo power rail, bigger pack | modest: servo pockets, U2D2 bus, new driver |

STS3215 note: it is about half as fast, but that is the speed BAM's own fit and Open Duck Mini work with. The extra
torque margin is the main gain: righting and a heavier claw become easier. Total stall current is not verified for
C001. Size the buck and fuse after measuring (UNVERIFIED).

## 5. Comparison table

| | STS3032 | STS3215 7.4 V | XL330-M288-T |
|---|---|---|---|
| Mass / size | 20 g, 23x12x27.5 | 55 g, 45x25x35 | 18 g, 20x34x26 |
| Stall at run voltage | 3.75 kg.cm at 5 V | C001 19.5 kg.cm at 6 V; C044 16 at 7.4 V | 5.3 kg.cm at 5 V |
| Speed | 11.6 rad/s | about 5.3 rad/s | 10.8 rad/s |
| Gears | metal | metal | plastic |
| Voltage | 4.8-6 V | 6-7.4 V (C001), 5-8.4 V (C044) | 3.7-6 V |
| Public measured model | none | BAM, 7.4 V, kp 32 | BAM, about 7.5 V, kp 400 |
| Working sim-to-real robot | none | Open Duck Mini (4.8k stars) | Microduck (14k stars, active) |
| Model valid at our voltage | no model | yes (7.4 V) | no (fit 7.5 V, we would run 5 V) |
| India price x 14 | INR 46,586 | INR 31,140 (INR 33,866 for 12 V) | no Indian seller; about EUR 560 import |
| India stock today | 27 (BOM, 11-18) | C001 and C046 sold out, C044 about 4, C018 98 | none found |
| Adapter | FE-URT-1, INR 1,069 | same | U2D2, $43.57 |
| Redesign | none | large | small |

## 6. Recommendation

STS3215, 7.4 V, regulated 7.4 V servo rail, target the C001 (1/345) because that is what the BAM fit most likely
describes. Reasons:
1. Only option where the measured model, the datasheet voltage and our supply all match.
2. Metal gears and about 5x more rated torque: a pet that is handled and falls will strip plastic gears.
3. About INR 15,000 cheaper than today's servos, and the adapter and bus are already in the BOM.
4. Open Duck Mini (4.8k stars) is a working walker on this servo and model.
5. The owner said size is not a limit.

## 7. Still to measure on the real robot, even with the public model

- Command delay through our chain (Radxa, USB, FE-URT-1): BAM stored 1.05 ms on its bench; ours will differ.
- Backlash and gear play of our own units (BAM does not model it; microduck_rl adds it).
- Voltage sag under our pack and wiring (Pollen samples 6.5-8.2 V).
- Unit-to-unit spread (cheap servos differ) and encoder zero offsets.
- The firmware gain we actually set (BAM assumed kp 32); check it reads back.
- Heat after several minutes of standing, and holding current.
- Gear version: confirm ours matches BAM's with one pendulum run.

## 8. First bench step: buy list (INR, incl. GST)

| Item | Qty | Price | Note |
|---|---|---|---|
| Feetech STS3215 C001, 7.4 V | 2 | 2 x 2,224.30 = 4,448.60 | Evelta, or Hubtronics (INR 1,785 each). Both sold out today: set a stock alert, or import 2 from Alibaba (about $14 each, UNVERIFIED) |
| Feetech FE-URT-1 USB bus adapter | 1 | 1,069 | robu.in, already in BOM |
| 7.4 V supply for the bench (2S LiPo or bench PSU) | 1 | ESTIMATE 800-1,500 | not priced today |
| Pendulum arm and a few weights | 1 set | ESTIMATE 300-600 | 3D printed, own weights |

Total about INR 6,600-7,600 (estimates included). Run BAM `record`, then `fit`, and compare with the public m1-m6.
If the match is good, order the other 12 and do the redesign. If only C044 (1/191) can be had, the public model does
not apply, so do the fit yourself (about one day, per the audit note).

## 9. Main risk

Supply: the exact variant the public fit describes (C001, 1/345) is sold out at both Indian sellers, and the
STS3215 gear versions do not share one model. Buying C044 or C046 gives a good servo but no public model.
Second: the robot grows to roughly 1.2 kg and 30 cm, which means a large CAD and electronics rework (ESTIMATE, not
designed). Third: the C001 voltage rating (6-7.4 V on Evelta) is below a fully charged 2S pack (8.4 V), so the
regulated 7.4 V rail is required, not optional.
