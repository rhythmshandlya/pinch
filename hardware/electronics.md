# Electronics design (Pinch, 12 x STS3215 legs)

> **Agrees with the netlist since 2026-11-21.** The wiring below is `configs/wiring_v2.toml`, checked by `python -m crab.wiring`
> (pins, I2C addresses, current ratings, every part wired); where each part sits is `configs/layout_v2.toml`, checked by
> `python sim/fit_check.py`; `wiring.svg` is drawn from the netlist (`python hardware/wiring_diagram.py`). If this page and
> those files disagree, the files win: edit them, run the checks, then fix this page. Currents here are the twin's circuit
> simulation (`docs/dev/ELECTRONICS.md`, "Design v2"), **estimates until measured on the bench**.
> The legs run from the pack at 6.4-8.4 V, not a 5 V regulator: reasons and sources in
> [../docs/research/BOM_CHANGE.md](../docs/research/BOM_CHANGE.md).

Every board lives inside the body (a chassis 135 x 179 x 50 mm under a dome, 1.75 kg with everything). Parts and prices are
in [BOM.md](BOM.md). The CAD for v2 does not exist yet; the placement below is the layout file's, with real outlines where a datasheet gave
one and estimates (marked in the file) where not.

| Place | What |
|---|---|
| **Belly** (x -45..-5, across the body) | the 2S1P 18650 pack with its BMS (Samsung 25R, 7.4 V 2500 mAh, about 100 g), the XT60 and the fuse beside it |
| **Belly, front** | servo hub (12 leg headers + 6 arm headers, 2 x 1000 uF), Waveshare bus servo driver board, INA219 with its shunt |
| **Over the pack** | Radxa ZERO 3W with its heatsink, the two 5 V regulators, BNO055 IMU (back, under the dome), MPR121 |
| **Back wall** | main switch (20 A rocker), one VL53L0X (rear cliff beam), two pairs of dock pogo pins with the reverse guard board behind |
| **Front wall, low** | 160 degree camera (lens at z -19.5 mm flush with the wall, its 25 x 24 mm board behind), VL53L5CX 8x8 above it |
| **Front wall, corners** | two VL53L0X (front cliff beams), each in an angled pocket |
| **Side walls** | two VL53L0X (side cliff beams), at x = 0 between the hip servos |
| **Front, on top** | the face: today's orange visor with the 2.42" OLED (61.5 x 39.5 mm module), leaning back 25 degrees, lower edge just above the shoulder servos |
| **Under the dome** | speaker, two I2S microphones 70 mm apart, six copper touch pads, MAX98357A amplifier |

## Block diagram

```mermaid
flowchart LR
    BAT["2S Li-ion pack 7.4 V (6.4-8.4 V)<br/>2500 mAh, BMS inside"] --> XT["XT60"]
    XT --> FUSE["15 A fuse"]
    FUSE --> INA["INA219 + 0.002 ohm shunt<br/>reads the whole pack current"]
    INA --> SW["Main switch 20 A"]
    SW -- "7.4 V direct, 14 AWG trunk" --> HUB["Servo hub<br/>12 leg headers 7.4 V<br/>6 arm headers 5 V"]
    SW --> REGA["5 V 5 A regulator A<br/>Radxa + audio + sensors"]
    SW --> REGB["5 V 5 A regulator B<br/>6 arm servos"]
    REGB --> HUB
    URT["Waveshare driver board<br/>USB -> servo bus"] -- "signal + GND only" --> HUB
    HUB -- "12 cables, 900 mm" --> LEGS["12 leg servos, IDs 1-12"]
    HUB -- "6 cables" --> ARMS["6 arm servos GS-4308MG"]
    REGA --> PI["Radxa ZERO 3W"]
    PI -- "USB-C to USB-C" --> URT
    PI -- "6 x hardware PWM (signals)" --> ARMS
    PI -- "I2C4 (hardware), alone" --> IMU["BNO055 IMU"]
    PI -- "i2c-gpio, pins 3/5" --> INA
    PI -- "i2c-gpio, pins 3/5" --> TOUCH["MPR121 touch, 6 pads"]
    PI -- "i2c-gpio, pins 3/5" --> FACE["2.42 in OLED face"]
    PI -- "I2C5 + 6 reset pins" --> TOF["5 x VL53L0X + VL53L5CX"]
    PI -- "I2S" --> AMP["MAX98357A amp -> speaker"]
    PI -- "I2S (stereo)" --> MIC["2 x I2S microphone (L/R)"]
    PI -- "CSI" --> CAM["160 degree camera"]
    DOCK["Dock: USB-C charger module<br/>5 V in, 8.4 V out"] -. "4 pogo pins, guard, polyfuses" .-> BAT
```

## Why it's built this way

- **Legs run straight from the pack (6.4-8.4 V), no regulator.** Open Duck Mini (4,763 stars) powers 14 STS3215
  from a 2 x 18650 series pack through a BMS and a switch, with only a 5 V regulator for the computer. Rhoban/bam's fitted
  model and Pollen's training both treat battery voltage and sag as part of the motor model (Pollen randomises 6.5-8.2 V).
  A regulator for the legs would need to carry 10 A or more. In the twin the servo's torque follows the voltage at its terminals.
  **Open question:** Waveshare and DFRobot list the 7.4 V servo as 4-7.4 V, and a full pack is 8.4 V (other sellers list 6-8.4 V). Test
  one servo at 8.4 V for an hour first (BOM_CHANGE.md, question 1).
- **Two 5 V regulators, neither feeds the legs.** Regulator A feeds the Radxa, sensors and audio, so servo spikes cannot reset the computer.
  Regulator B feeds the six arm servos (GS-4308MG, rated 4.8-6 V). The AP64500 module is listed for 7-40 V input, the chip alone for 3.8 V:
  the loaded pack falls under 7 V near the end of the charge (about 83 minutes into the simulated day). Measure a module's real cut-off.
- **Pack protection.** The pack's BMS cuts off at its own low-voltage level. The 15 A fuse sits directly after the XT60; the plug cannot go in
  backwards. The twin's fuse follows a time-current curve: a 16 A get-up swing does not blow it, 22 A for 15 s does.
- **Fuse: 15 A.** Each STS3215 stalls at 2.7 A (Waveshare; 3.05 A at a full pack from the measured motor model), so twelve stalled servos would
  be 36 A, which cannot happen (a leg stalls for 2 s at most, the servo's overload protection ends it, and the BMS cuts at about 25 A). The
  simulated walking peak is 3 A and a get-up swing 16 A.
- **Every servo has its own hub header and cable (a star, not a chain).** Three stalled servos on one daisy-chain cable would put 8 A through a
  3 A connector; the netlist check refuses that. The signal pins of all headers are joined on the hub: one half-duplex bus, 1 Mbaud, IDs 1-12.
- **Power doesn't go through the signal adapter.** The driver board only converts the signal (USB to the one-wire servo bus). Servo power goes
  from the switch into the hub. The board's own DC jack stays empty in the robot (its wiki lists a 9-12.6 V input: check on the bench that the
  bus works without it).
- **Bulk capacitors.** 1000 uF at the hub (two) and on each 5 V regulator input (one each) soak up the current spikes.
- **A battery monitor with a real shunt.** The INA219 breakout's 0.1 ohm shunt clips at 3.2 A, so it is replaced by a **0.002 ohm** 2512
  resistor (the BOM said 0.005 ohm, which burns 1.1 W at the fuse's 15 A in a 1 W part; 0.002 ohm burns 0.45 W, reads up to 160 A in 5 mA steps). It sits between the
  fuse and the switch, so it sees the whole pack current: servos and both regulators. The software stop must stay above what the 5 V
  regulators need (see BOM_CHANGE.md, risk 3).
- **The BNO055 gets a bus of its own.** It holds the clock line low for long; on a shared bus it would block the others
  (`docs/research/audit/ext_runtime_power.md` section 6). Radxa's pin table lists I2C4 on pins 27/28 and I2C5 on 29/31, and **no I2C function on
  pins 3/5**: the INA219, MPR121 and OLED therefore share a kernel `i2c-gpio` bus on pins 3/5 (400 kHz, 26 % busy).
- **Arm servos straight on the Radxa's PWM pins:** six hardware PWM outputs, power from regulator B.
- **Charging dock.** The charger lives in the dock (USB-C 5 V in, boosted to 8.4 V 2 A), and four spring pins (two per polarity: the listing gives
  no current rating) carry the charge into the pack through the BMS. Needs a reverse guard and polyfuses on the robot side. Not built yet.

## Power budget (the twin's circuit simulation; ESTIMATES, to be measured on the bench)

Pack 2S 2500 mAh (18.5 Wh), camera on, servos at the measured STS3215 model, scripted trot at the gait's cap (10 cm/s):

| Activity | Pack current, mean / peak (10 ms) | Lowest leg-servo voltage |
|---|---|---|
| Standing, torque on | 1.1 A / 1.1 A | 8.28 V |
| Walking forward | 1.6 A / 2.2 A | 8.13 V |
| Walking sideways | 1.8 A / 3.1 A | 7.95 V |
| Get-up attempt (peak, short) | 3.7 A / 16 A | 6.59 V |

Regulator A carries up to 0.6 A in practice (1.06 A worst case), regulator B up to 3.6 A in the get-up (4.8 A with all six arms stalled).
Mixed day (50 % standing with the camera on, 20 % walking forward, 10 % sideways, 20 % standing with the computer idle): **92 minutes** to the
pack's usable 80 %, mean 1.26 A. The old table (the STS3032 at 5 V, `python sim/power.py`) described another robot; `python sim/power.py --v2` runs this one.

## Wiring table (the netlist, `configs/wiring_v2.toml`)

| From | To | Wire / connector | Notes |
|---|---|---|---|
| Battery pack | XT60 socket | 14 AWG, 150 mm pigtail | Red = +, black = - |
| XT60 | Fuse holder | 14 AWG | **15 A** ATC fuse |
| Fuse | INA219 VIN+ | 14 AWG | the **0.002 ohm** shunt across VIN+ / VIN-; GND goes straight through |
| INA219 VIN- | Main switch | 14 AWG | |
| Switch | Servo hub 7.4 V rail | 14 AWG trunk | **Straight pack voltage (6.4-8.4 V).** Two 1000 uF caps across the hub |
| Switch | Regulator A / B input | 20 AWG each | 1000 uF across each input |
| Regulator A output / GND | Radxa pins 2 + 4 (5 V) and 6 + 9 (GND), amp VIN | 22 AWG, two wires each way to the Radxa | **Measure 4.9-5.2 V first, with nothing connected, also with the input at 7 V** |
| Regulator B output / GND | Hub arm-servo headers (V+ and GND) | 20 AWG | Measure 4.9-5.2 V first. Never join this rail to the leg rail |
| Hub leg headers (12) | the 12 leg servos | the bought 900 mm 3-wire cable each (24 AWG, 5264 connectors) | one servo per header; signal pins joined on the hub |
| Hub arm headers (6) | the 6 arm servos | about 500 mm, 24 AWG | signal wire of each plug goes to the Radxa pin below, not to the hub |
| Waveshare driver board servo port | Hub **signal + GND only** | Servo cable | Do not connect the board's DC jack in the robot |
| Driver board USB-C | Radxa USB-C host port | short USB-C to USB-C data cable | a serial port; 1 Mbaud |
| BNO055 | Radxa pins 1 (3.3 V), 27 (SDA), 28 (SCL), GND | 28 AWG | I2C4, address 0x28, **alone on its bus** |
| INA219 | 3.3 V, GND, pin 3 (SDA), pin 5 (SCL) | 28 AWG | `i2c-gpio` bus, address 0x40 |
| MPR121 touch | 3.3 V, GND, pin 3, pin 5 | 28 AWG | address 0x5A; electrodes E0-E5 to the six pads, E6 / E7 through the arms to the claws |
| Face OLED (SSD1309) | 3.3 V, GND, pin 3, pin 5 | 28 AWG | address 0x3C; set the module to I2C |
| VL53L0X x 5 | 3.3 V, GND, pins 31 (SDA), 29 (SCL); XSHUT: front left 11, front right 13, rear 15, side left 22, side right 36 | 28 AWG | all wake at 0x29; the software gives them 0x30-0x34 one at a time |
| VL53L5CX | 3.3 V, GND, pins 31, 29, LPn pin 32 | 28 AWG, Qwiic-style | wakes at 0x29 too; given 0x35 |
| MAX98357A | 5 V (regulator A), GND, BCLK pin 12, LRC pin 35, DIN pin 40 | 28 AWG | Speaker on its + / - terminals |
| I2S microphones x 2 | both: 3.3 V, GND, SCK pin 12, WS pin 35, SD pin 38. Left SEL to GND, right to 3.3 V | 28 AWG | One stereo stream |
| Camera (160 degrees, OV5647) | Radxa 22-pin CSI connector | its own flex | Measure its board first |
| Dock pogo pins (2 pairs) | Reverse guard, polyfuses, then the pack's charge lead | 22 AWG | Contacts dead until the robot is sensed; NTC on the pack |

All header pins are used once (the netlist check enforces it). Power pins: 3.3 V on 1 and 17, 5 V in on 2 and 4, GND on 6, 9, 14, 20, 25, 30, 34, 39.

### Arm servo signals (Radxa hardware PWM)

| PWM output | Joint | Radxa header pin | Radxa function (as listed) |
|---|---|---|---|
| 0 | L_shoulder_yaw (left arm swing) | 16 | PWM8_M0 |
| 1 | L_shoulder_pitch (left arm lift) | 18 | PWM9_M0 |
| 2 | L_claw | 21 | PWM12_M1 |
| 3 | R_shoulder_yaw | 24 | PWM13_M1 |
| 4 | R_shoulder_pitch | 7 | PWM14_M0 |
| 5 | R_claw | 19 | **PWM15_IR_M1** (Radxa's table names it so; the old note said PWM15_M1) |

Enable the six overlays in `sudo rsetup` -> Overlays, then reboot. `python robot/run.py --check` lists any output it
can't find by its header pin.

Each leg servo has its own cable to the hub (a star): `hub header n` -> servo ID n. The arm servos don't chain either.

## Servo ID map (matches `crab/params.py` JOINT_NAMES order and the hub's header order)

| ID | Joint | Servo | ID | Joint | Servo |
|---|---|---|---|---|---|
| 1 | FL_coxa | STS3215 | 7 | RL_coxa | STS3215 |
| 2 | FL_femur | STS3215 | 8 | RL_femur | STS3215 |
| 3 | FL_tibia | STS3215 | 9 | RL_tibia | STS3215 |
| 4 | FR_coxa | STS3215 | 10 | RR_coxa | STS3215 |
| 5 | FR_femur | STS3215 | 11 | RR_femur | STS3215 |
| 6 | FR_tibia | STS3215 | 12 | RR_tibia | STS3215 |

IDs 13-18 are the arm servos in the software's joint list, but they have no bus ID: they are PWM outputs 0-5.

New bus servos all ship as **ID 1**. `robot/setup_servo.py --id N` assigns IDs one servo at a time. Put a small sticker
with the ID on each servo.

## Servo settings written by the setup script (to confirm on the bench; `robot/setup_servo.py` still has the old values)

| Register | Value | Why |
|---|---|---|
| P / I / D gains | 32 / 0 / 0 | Feetech default; the public BAM fit assumed kp 32 |
| Max input voltage | about 86 (8.6 V) | Above a full pack (8.4 V), so the servo trips only on a real over-voltage. Check the register units on the bench |
| Min input voltage | about 55 (5.5 V) | Stops only on a real sag |
| Return delay | 0 | Faster bus |

## Safety rules (read before first power-up)

1. **First power-up through the 15 A fuse, servos unplugged.** Then plug in one leg servo at a time, then a leg.
2. **Check polarity with the multimeter** at every connector before plugging anything in. The XT60 only fits one way,
   the DC plug cable does not care: check the centre pin is +.
3. Measure both 5 V regulators (**4.9-5.2 V**) with nothing connected.
4. Charge the pack **in the LiPo bag, in a fire-safe place, never unattended**, with the 8.4 V charger (it charges
   through the pack's main lead; the BMS balances the two cells). Never charge a pack that is warm, dented or puffy.
5. The software warns and stops at pack-voltage levels it was given for the old 2S LiPo (6.8 V and 6.4 V): these must be
   re-set after the 5 V regulator test (BOM_CHANGE.md, risk 3). The modules' listed 7 V minimum is the number to measure.
6. If a battery is **puffy, dented or hot**, stop using it and take it to an e-waste point.
