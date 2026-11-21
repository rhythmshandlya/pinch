# Bill of materials change: legs on STS3215, new power path (2026-11-21)

Written for the owner. Short sentences on purpose. "ESTIMATE" means my own number, not a measured or published one.
"UNVERIFIED" means I did not open the page or the datasheet.

## 1. What was decided and what changed

- Legs: **12 x Feetech STS3215 7.4 V + 2 spares** (replaces 14 x STS3032). Reason: docs/research/SERVO_CHOICE.md.
- Servos run **straight from a 2S Li-ion pack**, not from a 5 V regulator.
- Pack: **Samsung 25R 2S1P, 7.4 V, 2500 mAh, 8C, with a built-in BMS** (robu SKU 1483665).
- The two 5 V regulators stay, but they no longer feed the legs. One feeds the Radxa, sensors and audio. One feeds the six
  arm servos.
- New: one VL53L5CX 8x8 distance sensor, the charging-dock parts, PETG/ASA filament, a 15 A fuse, a 20 A switch, big
  capacitors, a 0.005 ohm current shunt.
- The full list, with seller, SKU, price and stock for each part, is in `hardware/BOM.md` and `hardware/BOM.csv`.

## 2. The power design, with sources

### 2.1 What other robots do

| Robot | Stars, last push (read today with `gh api`) | Battery | How servos are powered | Computer |
|---|---|---|---|---|
| Open Duck Mini v2 (apirrone/Open_Duck_Mini) | 4,763 stars, pushed 2026-01-31 | 2 x 18650 in series (Molicel P30B 3000 mAh, 30 A) + a 2S BMS | **Straight from the pack** (7.4 V) through a power switch and the servo board | UBEC 5 V regulator to a Pi Zero 2 W |
| SO-ARM100 / SO-101 arm (TheRobotStudio/SO-ARM100) | 8,178 stars, pushed 2026-11-20 | wall supply, no battery | follower arm on a **5 V** supply (README); the 12 V servo needs 12 V 5 A | Waveshare bus servo driver board |
| Microduck (pollen-robotics/microduck) | 13,957 stars, pushed 2026-11-20 | **I found no battery spec** in its software repo (hardware docs not found) | n/a | RK3566 board |
| Pollen microduck_rl | 2,937 stars (earlier note) | its training code randomises battery voltage 6.5-8.2 V and sag | n/a | n/a |
| Rhoban/bam (the servo model) | 629 stars, pushed 2026-11-10 | model is fitted at 7.4 V for the STS3215 | n/a | n/a |

Where I read it: Open Duck Mini's BOM sheet (linked from its README: "18650 cell", "18650 cell holder", "2S BMS", "5V
regulator", "Small power switch", "usb-c charger", XT30) and its wiring drawing `docs/open_duck_mini_v2_wiring_diagram.drawio`
(labels: "18650 x2 series", "BMS", "7.4V DC in", "7.4V out", "UBEC", "5V out", "power switch", "USB-C charger"); the SO-ARM100
README (the IMPORTANT note on 5 V and 12 V supplies). The Open Duck Mini BOM says "get the beefiest 18650 cells you can
find: 3000 mAh with 30 A discharge".

### 2.2 My decision

**Battery.** 2S Li-ion (two 18650 cells in series), 2500 mAh, 8C, BMS inside. Why Li-ion and not LiPo: it is what Open Duck
Mini flies on, the BMS stops over-discharge and over-current by itself, and the pack charges through its main lead. The
LiPo packs I wanted (2S 2200 mAh) are out of stock at robu. I bought a **prebuilt** pack so you do not have to weld or
solder cells (beginner-safe). 2500 mAh x 7.4 V = 18.5 Wh.

**Servos from the pack, not a regulator.** A full pack is 8.4 V. The servo is sold as a 7.4 V part. A regulator that holds
7.4 V at 10 A or more would be big and hot, and the public model (BAM) already contains battery voltage and sag as part of
the motor. Open Duck Mini does exactly this with 14 of the same servos. I accept the risk below (question 1).

**Currents.**
- Stall current: **2.5 A at 7.4 V** (2.0 A at 6 V). Sources: DFRobot's STS3215 page and Waveshare's ST3215 page, both opened
  today. Evelta's C001 page also says 2000 mA at 6 V. Evelta's C044 page lists a lower 1.0 / 1.2 A: the sellers disagree.
  I used the higher figure. No-load: 150 mA (Waveshare).
- Standing: about 2-3 A for all legs (ESTIMATE: torque need is about 6 % of stall).
- Walking: about 4 A average, 10 A peak (ESTIMATE: four servos pulling hard at once).
- Theoretical worst case 12 x 2.5 A = 30 A. It cannot happen at once. The servo also protects itself.
- **Fuse: 15 A** (ATC blade, in stock at robu, INR 19). **Pack rating: 20 A continuous** (the seller's "8C"; the BMS limit
  is not on the listing: UNVERIFIED). **Trunk wire: 14 AWG.** **Switch: 20 A.**
- **Runtime** (ESTIMATE): about 37 minutes standing, about 25 minutes walking.

**The 5 V rails.**
- Rail A (Radxa ZERO 3W, camera, sensors, amp): about 4-5 W, about 1 A. Fed by one 7Semi AP64500 (5 V 5 A).
- Rail B (six GS-4308MG arm servos): about 3 A at peak (ESTIMATE; I did not find the servo's stall current).
- This fixes the second audit's finding: the old single 5 V 5 A regulator was asked to feed the legs. It no longer is.
- Two 1000 uF capacitors at the servo hub and one at each regulator input absorb spikes.

**The INA219.** Its board has a 0.1 ohm shunt that clips at 3.2 A. It is swapped for a 0.005 ohm resistor (reads 64 A).

**Charging.** Dock side: a USB-C to 2S 8.4 V charger module (robu 1794300, INR 321, IP2326 class, in stock) and a USB-C
adapter. Robot side: two spring pins, a reverse guard, a polyfuse and a 10 k NTC. This is the parts list from
docs/research/audit/ext_runtime_power.md section 12. For the first weeks, charge with the plain 8.4 V 3 A charger (INR 449).

## 3. Price comparison, old and new (INR incl. GST)

| | Old (STS3032, 5 V rail) | New (STS3215, pack straight to servos) |
|---|---|---|
| 14 leg servos | 46,586 (14 x 3,327.60) | 33,600 (14 x 2,399.99) |
| USB bus adapter | 1,069 (FE-URT-1) | 475 (Waveshare driver board) |
| Battery, charger, protection | about 4,400 (2 x Bonka LiPo, SkyRC E3, bag, buzzer, BMS) | about 3,700 (2 Li-ion packs, 3 A charger, bag, XT60, fuse) |
| Dock parts | 0 | about 1,200 |
| VL53L5CX | 0 | 851 |
| Whole list | **88,566** | **77,412** |

The new list is cheaper by about INR 11,000 even though it adds a dock, a sensor and a second pack. The STS3215 is INR 928
cheaper per servo than the STS3032 at ThinkRobotics' price. **Not in either number:** the CAD redesign for a 30 cm robot,
the 3D prints (priced by robu after upload), shipping, customs on any import.

### The three orders

| Order | Required | With recommended | What for |
|---|---|---|---|
| 1. First order: servo bench | INR 10,859 | INR 10,859 | 2 servos, bus board, pack, charger, fuse, cables, soldering tools, multimeter, pendulum weights. About INR 3,070 of that is tools you may own |
| 2. Second order: bring-up | INR 13,214 | INR 13,214 | Radxa, SD, IMU, battery monitor, 2 regulators, VL53L5CX, capacitors, cables |
| 3. Full build | INR 50,424 | INR 53,339 | 12 more servos (INR 28,800), arms, senses, dock, spare pack, fasteners, other tools |

The bench power is the pack, not a bench supply: the Yihua bench supplies at robu are out of stock (INR 3,339-4,029).

## 4. What is in stock and what is not (opened 2026-11-21)

- **Buy this one:** ThinkRobotics, Waveshare ST3215 **19.5 kg.cm @ 7.4 V**, SKU WVSH0593, INR 2,399.99. Page says "In stock
  | Ready to ship", with no count. Its speed (0.192 s per 60 degrees at 7.4 V) matches the C001. DFRobot lists the 7.4 V
  part as 1:345. So it should be the version BAM modelled. **Not proven**: the pendulum test settles it.
- **The exact STS3215 versions:**
  - C001 (1/345, the BAM one): **Evelta sold out (0 in stock).** robu's Waveshare listing R258500 (INR 2,159): sold out.
    Hubtronics: could not open (Cloudflare check). Robocraze: not listed.
  - C044 (1/191): Evelta **4 in stock**, INR 2,224.30. Different gear: the public model does not describe it.
  - C046 (1/147): Evelta sold out.
  - C018 (12 V): Evelta 98 in stock, INR 2,419. Wrong voltage for this design.
- **Alternatives if ThinkRobotics fails:** import C001 from Alibaba (the SO-ARM100 BOM says about $13.89 each; price date
  UNVERIFIED; I could not check customs) or wait for Evelta's restock.
- Also out of stock: robu 16 AWG silicone wire (all colours), robu 2S 2200 mAh LiPo packs, Yihua bench supplies,
  Robocraze's MPR121, robu's SparkFun VL53L5CX.
- **Not opened (UNVERIFIED):** Hubtronics (blocked by a Cloudflare check, I did not bypass it), Robokits (search page not
  found), Amazon.in (not tried). Parts still on estimates: the 0.005 ohm shunt, the USB-C cable, the heatsink, VL53L0X
  cliff sensors (INR 300 each, older estimate), dock protection parts, pendulum weights.

## 5. Risks

1. **Servo overvoltage.** Waveshare and DFRobot list the 7.4 V servo at 4-7.4 V. A full pack is 8.4 V. Sigmanortec lists
   6-8.4 V and Evelta lists 5-8.4 V for the C044. Open Duck Mini runs a full 2S pack on 14 of them. I could not read
   Feetech's own datasheet (the PDF would not convert). **Test on the bench first** (hold one servo at 8.4 V for an hour,
   watch its temperature). If it fails: put an adjustable 10 A+ buck set to 7.6 V between pack and hub (not priced).
2. **Version risk.** If ThinkRobotics' servo is not the BAM version, the public model is wrong and you do the fit
   yourself (about one day, per the earlier audit).
3. **5 V rails drop out when the pack is nearly flat.** The AP64500 board is sold as "input 7-40 V". A 2S pack sags below
   7 V when low or under a heavy step. The chip itself may work lower, but I did not verify. Test on the bench with the
   input at 6.0-7.0 V. Until then the software's stop levels (6.8 V warn, 6.4 V stop, old values) are wrong for this
   design: the Radxa could reset before they trigger. If the board fails the test, use a low-dropout 5 V module.
4. **Pack limits unverified.** I did not see the pack's connector or the BMS cut-off current. If it trips at 10 A the legs
   will stop in a hard step. Measure first.
5. **Wire and connector heat.** 3 servos share the first cable in each leg (about 7.5 A if all stalled). Keep runs short.
6. **Charging at 5 V input.** The USB-C booster module cannot give 2 A from a 5 V 3 A adapter. It charges slower. Fine for a
   dock; test it.
7. **Size.** The 20 A switch is bigger than the old cut-out, and the pack and 12 servos make a bigger robot (about 30 cm,
   1.2 kg, ESTIMATE). The CAD must follow.

## 6. Open questions for the owner

1. Are you happy to run the servos at up to 8.4 V (what Open Duck Mini does) if the one-hour bench test is clean? If not, I
   add a 7.6 V regulator and the list grows.
2. Buy 14 STS3215 from ThinkRobotics (INR 2,399.99 each, one seller), or wait for Evelta's C001 (INR 2,224.30, sold out)?
3. Do you already own a soldering iron, multimeter and wire strippers? That takes about INR 3,070 off order 1.
4. Do you want the dock in the first full build, or after the robot walks? It is about INR 1,200 and a CAD feature.
5. Robot size: is about 30 cm and 1.2 kg fine for a desk pet that is picked up? It was 20 cm and 0.6 kg.

## 7. Things left stale (not changed here)

`hardware/ASSEMBLY.md` (stages 0-4: STS3032, 5 V servo regulator, FE-URT-1, ALPS switch, OTG cable, regulator test),
`hardware/wiring_diagram.py` and `hardware/wiring.svg`, `robot/setup_servo.py` (voltage limits 6.5 / 4.0 V),
`robot/crab_robot.py` battery thresholds, `sim/power.py`, `crab/params.py` (STS3032 table), `cad/` (pockets, battery
bay, switch cut-out), README and `docs/DECISIONS.md` (decision text on the servo). A decision entry is for the integrator.
