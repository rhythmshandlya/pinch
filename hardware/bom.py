"""Single source of truth for the order. Generates hardware/BOM.md and hardware/BOM.csv.

    python hardware/bom.py

Prices in INR **including GST**. Evelta shows prices ex-GST; they are converted at 18% (5% for the
two hand tools Evelta marks so). Robocraze, robu.in and ThinkRobotics show prices including GST.

`checked` says how far to trust a row:
  "2026-11-21"        the seller's page was opened that day (price and stock as shown on the page)
  "2026-11-18"        opened on the earlier check, NOT re-opened on 2026-11-21 (the part did not change)
  "UNVERIFIED"        never opened: the price (and the stock) are an estimate
Stock changes daily: re-check in the cart before paying.

order: 1 = "First order: servo bench", 2 = "Second order: bring-up", 3 = "Full build" (everything else)
need:  "required" | "recommended" | "optional"
stock: what the seller's page showed ("OUT" = sold out when checked)
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV, RB, AZ, TR, RC = "Evelta", "robu.in", "Amazon.in", "ThinkRobotics", "Robocraze"
TN, MO, LO = "Tenettech", "Mouser India / DigiKey India", "local hardware store / print service"
LIMIT = 106000                    # the owner's limit (2026-11-21) for ALL parts, prints and tools
PRINTS_MID, PRINTS_LOW, PRINTS_HIGH = 7000, 4250, 10200
T, D, U = "2026-11-21", "2026-11-18", "UNVERIFIED"
ORDER_NAMES = {1: "First order: servo bench", 2: "Second order: bring-up", 3: "Full build"}

# (order, section, item, store, SKU, qty, unit price INR incl. GST, need, stock, checked, note)
ITEMS = [
    # ================================================================ ORDER 1: servo bench
    (1, "Servos & bus", "Waveshare ST3215 serial bus servo, 7.4 V, 19.5 kg.cm (= Feetech STS3215 C001 class, 1/345)",
     TR, "WVSH0593", 2, 2399.99, "required", "in stock (\"Ready to ship\", no count shown)", T,
     "Leg servo. Same speed as the C001 (0.192 s/60 deg at 7.4 V; DFRobot lists 1:345). Evelta's C001 is sold out. "
     "Page does not say whether the price includes GST (store shows consumer prices: assumed incl.). "
     "Waveshare and DFRobot list a 4-7.4 V operating range (see BOM_CHANGE.md, open question 1)"),
    (1, "Servos & bus", "Waveshare serial bus servo driver board (USB-C, DC jack; the SO-ARM101 board)", RB, "1782919", 1,
     475.00, "required", "in stock", T,
     "USB to servo bus. Cheaper than the FE-URT-1 and what SO-ARM100/LeRobot use. Alternative: SmartElex copy, robu R255565 INR 509, in stock; Robocraze INR 509, in stock"),
    (1, "Servos & bus", "Feetech STS3032 serial bus servo, 6 V, 4.5 kg.cm (arm servo; 23.2 x 12.1 x 28.5 mm, 20 g)", EV, "501-STS3032", 1,
     3327.60, "required", "27 in stock", T,
     "The arm servo, benched before the other five are bought (ID, baud, zero, pendulum fit, torque-limit register, plug fit on the "
     "900 mm cable). Same bus protocol as the STS3215. 4.8-6 V: power it from the 5 V buck below, NEVER from the 7.4 V pack. "
     "Evelta price ex GST INR 2,820 x 1.18. The C001 listing (501-ST-3032-C001) and the SCS0009 are sold out"),
    (1, "Power", "7Semi AP64500 buck converter, 5 V 5 A (one for the bench: it feeds the arm servo; later regulator B)", EV,
     "004-BO-10188", 1, 823.64, "required", "listed (125 in stock on 11-18)", T + " price; stock " + D,
     "The plan's bench also tests the 5 V regulator at 6-7 V input, and the arm servo needs 5 V, so one buck comes in order 1 "
     "(order 2 then buys one more). Seller says input 7-40 V: see BOM_CHANGE.md risk 3"),
    (1, "Face", "Waveshare 6.25 inch 720 x 1560 IPS capacitive touch panel, HDMI (face; weigh it on the bench)", TN,
     "search 'Waveshare 6.25 inch HDMI LCD'", 1, 6403.00, "required", "listed (search snippet)", U,
     "Bought now ONLY to weigh it: the requirement is a panel mass of at most 0.15 kg (Waveshare lists 0.30 kg, probably packed); "
     "over that, send it back and use the 3.5 inch SPI panel (INR 2,345 Robocraze) instead. Price is Tenettech's snippet, page not "
     "parseable; Hubtronics snippet INR 5,617 incl GST (page blocked): UNVERIFIED both. Ships with an HDMI to micro-HDMI adapter. "
     "Bought in order 1, so it is not in order 3"),
    (1, "Power", "Samsung INR18650-25R 2S1P Li-ion pack, 7.4 V 2500 mAh, 8C (BMS inside)", RB, "1483665", 1, 1259.00,
     "required", "in stock", T,
     "Bench supply for the pendulum test AND the robot's first battery. 8C = 20 A continuous (seller's figure). "
     "Connector not shown on the listing: fit an XT60. The 2S2P 5000 mAh 8C version (1483666) is OUT"),
    (1, "Power", "Pro-Range ETC-30-8.4-3 charger, 2S Li-ion 8.4 V 3 A (DC 5.5x2.5 plug)", RB, "R259675", 1, 449.00,
     "required", "in stock", T,
     "Charges the pack through its main lead (the BMS balances). 3 A is 1.2C: inside the 25R's rapid-charge rating (from memory of the cell sheet: UNVERIFIED). "
     "The 2 A version (R151390) is OUT; the 1 A version SKU 1299276 INR 709 is in stock if you want a gentler one"),
    (1, "Power", "SafeConnect XT-60 male 14 AWG 15 cm pigtail", RB, "6783", 1, 89.00, "required", "in stock", T,
     "Goes on the pack"),
    (1, "Power", "SafeConnect XT-60 female 14 AWG 15 cm pigtail", RB, "57749", 1, 74.00, "required", "in stock", T,
     "Bench lead: to the fuse holder, then to the DC plug cable"),
    (1, "Power", "DC plug cable 5.5x2.1 mm male-to-male, 1 m", RB, "753433", 1, 89.00, "required", "in stock", T,
     "Cut in half: one half plugs into the driver board's DC jack (check polarity: centre +)"),
    (1, "Power", "15 A 32 V ATC blade fuse", RB, "R260439", 2, 19.00, "required", "in stock", T, "1 + 1 spare"),
    (1, "Power", "Medium car fuse holder, add-a-circuit type (use as ATC fuse holder)", RB, "R265459", 1, 69.00,
     "required", "in stock", T, "Cut off the tap; solder the leads. Stand-in until a plain inline ATC holder is found"),
    (1, "Assembly", "Pendulum weights and hardware (M8 nuts / washers, bearing; hardware store)", AZ,
     "local hardware store", 1, 450.00, "required", "n/a", U,
     "For Rhoban/bam's pendulum test: the arm is 3D printed. Price is an ESTIMATE"),
    # tools needed for the bench (not repeated in the full build: buy once)
    (1, "Tools", "YIHUA 926 temperature-controlled soldering station", RB, "1316303", 1, 1509.00, "required", "in stock",
     D, "Skip the tools you already own"),
    (1, "Tools", "Solder wire 0.8 mm, 50 g", RB, "1837790", 1, 269.00, "required", "in stock", D, ""),
    (1, "Tools", "Noel flux paste 10 g", RB, "762465", 1, 21.00, "required", "in stock", D, ""),
    (1, "Tools", "Mastech MAS830L digital multimeter", RB, "5981", 1, 899.00, "required", "in stock", D,
     "Check every voltage and polarity before plugging in"),
    (1, "Tools", "Stanley wire stripper / cutter 130 mm", RB, "1600571", 1, 369.00, "required", "in stock", D, ""),

    # ================================================================ ORDER 2: bring-up
    (2, "Brain & senses", "Radxa ZERO 3W, 2 GB RAM (Pi Zero size, RK3566, Wi-Fi)", EV, "498-RS107-D2E0H1W15", 1, 6460.50,
     "required", "listed (15 in stock on 11-18)", T + " price; stock " + D,
     "Replaces the sold-out Pi Zero 2 W: same size, 40-pin header and mounting holes"),
    (2, "Brain & senses", "SanDisk Ultra 32 GB microSD (A1, class 10)", RB, "R268040", 1, 1629.00, "required",
     "in stock", D, "Radxa OS (Debian)"),
    (2, "Brain & senses", "7Semi BNO055 9-DOF IMU breakout", EV, "004-ES-12149", 1, 1341.66, "required", "440 in stock",
     D, "Balance + fall detection. Robocraze has the 7Semi board at INR 1,528, in stock (2026-11-21)"),
    (2, "Brain & senses", "7Semi INA219 voltage / current monitor breakout", EV, "004-ES-12152", 1, 241.90, "required",
     "88 in stock", D, "Battery voltage for the safety stop and the pet's 'hunger'. Robocraze INR 240, in stock (2026-11-21)"),
    (2, "Power", "0.002 ohm 2512 current-sense resistor, 1 W (for the INA219)", MO, "search '2 mohm 2512 1 W current sense'", 2,
     350.00, "required", "check", U,
     "Replaces the INA219 board's 0.1 ohm shunt: that one clips at 3.2 A, and the legs draw up to ~10 A. 0.002 ohm burns 0.45 W at "
     "the 15 A fuse (0.005 ohm would burn 1.1 W in a 1 W part) and the INA219 reads up to 160 A in 5 mA steps. No Indian shop page found: "
     "Mouser India lists 2.5 mohm 1 W 2512 at INR 97.84 and DigiKey India lists 2 mohm 2512 parts (search snippets). "
     "INR 350 each is an ALLOWANCE (INR 700 for two, shipping included): ESTIMATE"),
    (2, "Power", "7Semi AP64500 buck converter, 5 V 5 A (second one; the first is in order 1)", EV, "004-BO-10188", 1, 823.64,
     "required", "listed (125 in stock on 11-18)", T + " price; stock " + D,
     "Two in total: one for the Radxa + sensors + audio + the face panel (rail A), one for the six arm servos (rail B, with the "
     "torque limit). NOT for the legs. Seller says input 7-40 V: the pack drops below 7 V when nearly empty, see BOM_CHANGE.md risk 3. "
     "Robocraze INR 823, in stock (2026-11-21)"),
    (2, "Power", "1000 uF 16 V electrolytic capacitor (Aishi, 10 x 20 mm)", RB, "R172629", 4, 9.00, "required",
     "in stock", T, "Soaks up the servos' current spikes: 2 at the servo hub, 1 on each 5 V rail"),
    (2, "Brain & senses", "SmartElex VL53L5CX 8x8 multizone ToF distance sensor", RB, "R101042", 1, 851.00, "required",
     "in stock", T, "Forward look: sees steps, walls and the dock. Robocraze INR 849, in stock. SparkFun version R268026 is OUT"),
    (2, "Brain & senses", "USB-C (male) to USB-C (male) data cable, 10-15 cm", AZ, "search 'USB C to USB C short cable 15cm'",
     1, 199.00, "required", "check", U,
     "Radxa USB-C host port to the driver board's USB-C. Replaces the old USB-C to Mini-USB cable. Price is an ESTIMATE"),
    (2, "Brain & senses", "Aluminium heatsink ~14 x 14 x 6 mm, self-adhesive", RB, "search 'heatsink 14x14'", 1, 40.00,
     "required", "check", U, "On the Radxa's RK3566 inside the closed shell. Price is an ESTIMATE"),
    (2, "Brain & senses", "40-pin female-female jumper wires, 20 cm", EV, "083-JW-254-F-F-40", 1, 68.44, "required",
     "54 in stock", D, "Peel off what you need for the sensor wiring"),
    (2, "Power", "20 AWG silicone wire, red (per metre)", RB, "1824992", 2, 40.00, "required", "in stock", D, ""),
    (2, "Power", "20 AWG silicone wire, black (per metre)", RB, "1824989", 2, 40.00, "required", "in stock", D, ""),
    (2, "Power", "28 AWG silicone wire kit, 5 colours x 5 m", RB, "R229090", 1, 499.00, "required", "in stock", D,
     "Sensor and I2S wiring. NOTE: 16 AWG silicone wire is OUT at robu today (all colours); use the XT60 14 AWG pigtails for the trunk"),

    # ================================================================ ORDER 3: full build
    (3, "Servos & bus", "Waveshare ST3215 serial bus servo, 7.4 V, 19.5 kg.cm (= STS3215 C001 class)", TR, "WVSH0593", 12,
     2399.99, "required", "in stock (\"Ready to ship\", no count shown)", T,
     "12 more: with the 2 bench servos that is 12 legs + 2 spares (the spares are the cut to make first if the total must shrink). "
     "Ask ThinkRobotics for stock of 12 before ordering. "
     "Fallback if they run out: import C001 (about $13.89 each on Alibaba per the SO-ARM100 BOM; price date UNVERIFIED)"),
    (3, "Servos & bus", "Feetech STS3032 serial bus servo, 6 V, 4.5 kg.cm (arms: swing, lift, claw x 2)", EV, "501-STS3032", 5,
     3327.60, "required", "27 in stock", T,
     "5 more: with the bench servo that is exactly the six arms, NO spare (owner's cut; Evelta had 27 in stock, so a failed one is a "
     "re-order). Bus servos with position and load feedback, IDs 13-18, 5 V rail B, torque limit 40 % in each (crab/wiring.py checks "
     "the rail). Replaces the six GS-4308MG PWM servos. Do not order until the bench servo's plug fits the 900 mm cable"),
    (3, "Servos & bus", "5264 3-pin servo cable, 900 mm (pack of 6)", RB, "R263666", 3, 359.00, "required", "in stock", D,
     "Three packs = the 18 cables, one per servo: 12 legs + 6 arms (cut the arm cables to about 500 mm). robu not re-opened on "
     "2026-11-21 (blocked). Check on arrival that the STS3032's plug is the same size (the STS3032 / SCS class is often 1.25 mm): "
     "if not, buy the matching cable then (not priced)"),
    (3, "Servos & bus", "8 x 12 cm double-sided prototype PCB (perfboard servo hub)", RB, "44139", 1, 114.00, "required", "check", D,
     "Servo hub, about 55 x 36 mm cut from it: 18 three-pin headers (12 leg on the pack rail, 6 arm on rail B), signal pins joined. "
     "Perfboard instead of a custom PCB (a 2-layer board would be about INR 1,500, not bought). The leg power runs on this board: "
     "flood the tracks with solder or use wire"),
    (3, "Servos & bus", "2.54 mm 1 x 40 male header strip", RB, "539076", 2, 15.00, "required", "check", D,
     "Hub: 18 x 3-pin headers = 54 pins"),
    (3, "Brain & senses", "160 deg wide-angle camera for Pi Zero (OV5647 fisheye, 22-pin)", AZ, "B083XMGSVP", 1, 3346.00,
     "required", "listed", D, "Pinch's eyes (decision 16)"),
    (3, "Brain & senses", "VL53L0X ToF distance module ('VL53L0X Laser Ranging Sensor')", RC, "search 'VL53L0X'", 6,
     132.00, "required", "in stock (as noted 2026-11-21)", T,
     "Five cliff beams + 1 spare, at Robocraze's price (the earlier estimate was INR 300 each). The board size is NOT measured: "
     "the layout assumes the 25 x 10.7 mm GY-VL53L0XV2; compare with the first board before cutting the pockets"),
    (3, "Brain & senses", "Adafruit MAX98357A I2S 3 W amplifier", EV, "076-3006", 1, 860.07, "required", "14 in stock", D,
     "Tones and chirps (the pet never talks)"),
    (3, "Brain & senses", "28 mm 8 ohm 0.5 W speaker (GSPK2805TN)", RB, "R210811", 1, 82.00, "required", "in stock", D,
     "Glued into the seat inside the dome"),
    (3, "Brain & senses", "DFRobot I2S MEMS microphone module", RB, "1074722", 2, 629.00, "required", "in stock", D,
     "Left + right on one I2S line"),
    (3, "Face", "Micro-HDMI to HDMI flat cable, thin, 20-30 cm (through the lid hinge)", AZ,
     "search 'micro HDMI flat cable 20cm FPC'", 1, 600.00, "required", "check", U,
     "Radxa micro-HDMI to the panel through the hinge. Must survive the hinge: test at the bench. INR 600 is an ALLOWANCE (ESTIMATE)"),
    (3, "Face", "Printed friction hinge hardware: M3 x 25 bolt, 2 wave washers, nylock nut, rubber / felt disc", LO,
     "local hardware store", 1, 150.00, "required", "n/a", U,
     "The lid's friction hinge (printed knuckles on the bolt, preload set by hand with the nut). Hang the lid, add washers until it "
     "holds at 0 deg. Price is an ESTIMATE; fallback two 0.2 N.m laptop hinges from Alibaba"),
    (3, "Brain & senses", "MPR121 12-channel capacitive touch controller (Robocraze)", AZ, "B07KF5FNBL", 1, 229.00,
     "required", "listed", D, "Touch zones under the shell. Robocraze's own MPR121 (INR 144) is OUT today"),
    (3, "Brain & senses", "Conductive copper foil tape, 10 mm x 10 m", AZ, "B0FGQCPFSP", 1, 294.00, "required", "listed", D,
     "The touch pads; also the dock's contact strips"),
    (3, "Power", "Samsung INR18650-25R 2S1P Li-ion pack, 7.4 V 2500 mAh, 8C (BMS inside), spare", RB, "1483665", 1, 1259.00,
     "recommended", "in stock", T, "Second pack: swap while one charges. Needed until the dock works"),
    (3, "Power", "LiPo explosion-proof bag", RB, "R260502", 1, 299.00, "recommended", "in stock", D,
     "Charge Li-ion packs in it too, until you trust the BMS"),
    (3, "Power", "ASW-15D 12 V 20 A rocker switch, SPST, blue", RB, "R265192", 1, 159.00, "required", "in stock", T,
     "Replaces the ALPS 8 A switch: the legs can pull more than 8 A for a moment. It is bigger than the 19.4 x 13.4 mm "
     "cutout: measure it, the belly cut-out must change (CAD is being redone). Green R265193 also in stock"),
    (3, "Power", "SafeConnect XT-60 female 14 AWG 15 cm pigtail (robot side)", RB, "57749", 1, 74.00, "required", "in stock", T,
     "Robot's battery socket. The male pigtail from order 1 goes on the pack"),
    (3, "Power", "JST-PH 2.0 2-pin lead, 30 cm", EV, "083-2P-2mm-F-Wire", 5, 10.62, "recommended", "205 in stock", D,
     "Pluggable speaker / touch leads"),
    # --- charging dock (docs/research/audit/ext_runtime_power.md section 12)
    (3, "Charging dock", "USB-C to 2S 8.4 V 2 A Li-ion charger/booster module", RB, "1794300", 1, 321.00, "required",
     "in stock", T,
     "Dock-side charger (IP2326 class). Input 5 V: at 2 A out it needs about 3.7 A in, more than a 5 V 3 A adapter gives, "
     "so it will charge slower than 2 A. 4 A version SKU 1794301 INR 329 in stock. Test on the bench first"),
    (3, "Charging dock", "Nextech 25 W USB-C fast charger + USB-C cable", RB, "R267654", 1, 669.00, "required", "in stock", T,
     "Dock input. Any good USB-C PD adapter does the same"),
    (3, "Charging dock", "P50-E2 spring test probe (pogo pin), pack of 5 (four on the robot, one spare)", RC,
     "p50-e2-spring-test-probe-pogo-pin", 1, 94.00, "required", "in stock", T,
     "Four pins on the robot: two per polarity (INR 94 for five). The current rating is NOT listed anywhere I could open: "
     "check before trusting it with 2 A"),
    (3, "Charging dock", "Reverse guard (P-MOSFET or Schottky), 10 k NTC, resistors, 2 polyfuses 1.5-3 A", RB,
     "search 'polyfuse 2A', 'NTC 10k', 'Schottky 5A'", 1, 100.00, "required", "check", U,
     "Charge-path protection on the robot side. Price is an ESTIMATE"),
    (3, "Charging dock", "PETG-HS 1.75 mm filament 1 kg, Pro-Range (dock base and prints at home)", RB, "R269355", 1, 780.00,
     "recommended", "in stock", T,
     "PETG (about 75 C) instead of PLA (softens at 55-60 C) for the shell and dock. ASA is better in heat: "
     "Bambu ASA Blue R258670 INR 1,999 in stock; Polymaker ASA White R265999 INR 2,816 in stock"),
    # --- mechanical
    (3, "Assembly", "M2 x 6 socket screw + nut (pack of 12)", RB, "56219", 3, 239.00, "required", "in stock", D,
     "Brackets and spares (the servos' own screws come in their boxes)"),
    (3, "Assembly", "M2 x 8 socket screw + nut (pack of 12)", RB, "56246", 1, 219.00, "required", "in stock", D, "Brain tray"),
    (3, "Assembly", "M2 x 10 socket screw + nut (pack of 12)", RB, "56253", 1, 239.00, "required", "in stock", D,
     "Foot tips, battery door"),
    (3, "Assembly", "M2.5 x 8 socket screw (pack of 10)", RB, "1711459", 1, 154.00, "required", "in stock", D,
     "Radxa ZERO 3W onto the tray"),
    (3, "Assembly", "Heat-shrink kit, 140 pcs", RB, "R267330", 1, 349.00, "required", "in stock", D, ""),
    (3, "Assembly", "Nylon zip ties 100 mm (pack of 100)", RB, "1081544", 1, 30.00, "required", "in stock", D, ""),
    (3, "Assembly", "N52 neodymium disc magnets 6 x 3 mm (pack of 50)", AZ, "B00Z7Z3QXQ", 1, 299.00, "required", "listed", D,
     "8 hold the shell on; 4-6 pull the robot into the dock"),
    (3, "Assembly", "3M Scotch heavy-duty double-sided foam tape", AZ, "B00N1U9AJS", 1, 171.00, "required", "listed", D,
     "Mounts the small boards"),
    (3, "Assembly", "Fevikwik Ezee Gel 3 g (non-drip super glue)", AZ, "B0CQV798H9", 1, 50.00, "required", "listed", D,
     "Magnets, foot tips"),
    # --- tools not needed for the bench
    (3, "Tools", "TE-801 helping hands with magnifier", RB, "17940", 1, 489.00, "recommended", "in stock", D, ""),
    (3, "Tools", "Noel desoldering wick", RB, "769914", 1, 34.00, "required", "in stock", D, ""),
    (3, "Tools", "Freemans FDC150 digital vernier calliper 150 mm", RB, "1608703", 1, 2349.00, "required", "in stock", D,
     "Measure one servo of each kind before printing"),
    (3, "Tools", "8-in-1 aluminium hex screwdriver set (RC)", EV, "447-A-234", 1, 903.00, "required", "in stock", D,
     "1.5 / 2 / 2.5 mm hex for the M2 / M2.5 screws"),
    (3, "Tools", "Plato 170 flush cutter", EV, "447-A-371", 1, 115.50, "required", "40 in stock", D, ""),
    (3, "Tools", "ESD curved tweezers", RB, "R232396", 1, 35.00, "recommended", "in stock", D, ""),
    # --- prints (not shop items, but inside the owner's limit)
    (3, "3D prints", "Printed body: pill shell, four legs, two arms, lid, internal trays (about 850 g PETG, print service)", LO,
     "robu.in 3D printing service, SKU 901845, or another service", 1, PRINTS_MID, "required", "n/a", U,
     f"ESTIMATE: about 850 g (shell 168 x 216 x 104 mm with 2 mm walls about 380 g, legs 160, arms 40, lid 60, trays 120, +10 % "
     f"supports and a reprint) x INR 8/g = INR 6,800, rounded to {PRINTS_MID:,}. Range INR {PRINTS_LOW:,} (INR 5/g) to {PRINTS_HIGH:,} "
     f"(INR 12/g): service rates are IndiaMART / Zbotic-guide snippets, not read from robu (price shown only after upload). "
     f"The shell may exceed a service's bed: print it in halves"),
]

REMOVED = [
    ("GOTECK GS-4308MG metal-gear micro servo, 4 packs of 2 (arms)", "INR 3,546.32",
     "PWM servos with no position or load feedback: the owner wants feedback on every joint. Replaced by 6 x Feetech STS3032 bus servos"),
    ("Spare arm servo (the 7th STS3032, INR 3,327.60)", "was planned in the parts decision",
     "Owner's cut to get under the limit: exactly six arm servos, a failure means a re-order (Evelta had 27 in stock)"),
    ("2.42 inch SSD1309 OLED, Teyleten (I2C)", "INR 2,300", "Replaced by the Waveshare 6.25 inch 720 x 1560 colour IPS panel in the hinged lid"),
    ("0.005 ohm 2512 shunt x 2 (estimate)", "INR 40", "Burns 1.1 W at the 15 A fuse in a 1 W part: 0.002 ohm instead (allowance INR 700)"),
    ("VL53L0X at an estimated INR 300 each", "6 x INR 300 = INR 1,800", "Robocraze's INR 132 board (6 x INR 132 = INR 792): size to be measured"),
    ("Custom servo hub PCB (allowance INR 1,500)", "was proposed", "Perfboard hub, about 55 x 36 mm, from the INR 114 board already in the list"),
    ("Feetech STS3032 serial bus servo, as the LEG servo", "14 x INR 3,327.60 = INR 46,586",
     "No public measured model, 5x less torque for the legs; the legs are STS3215 (docs/research/SERVO_CHOICE.md). The STS3032 is now the ARM servo"),
    ("Feetech FE-URT-1 USB bus adapter", "INR 1,069", "The Waveshare driver board does the same for INR 475 (what SO-ARM101 uses)"),
    ("Bonka 850 mAh 2S LiPo x2", "INR 2,078", "Too small for 12 STS3215; replaced by the 2500 mAh 8C Li-ion pack with BMS"),
    ("SkyRC E3 balance charger", "INR 1,689", "Needs a balance lead; the new pack has a BMS and charges through its main lead"),
    ("BX100 low-voltage buzzer", "INR 218", "Made for the LiPo balance lead; the pack's BMS and the INA219 stop do this job"),
    ("5 A mini blade fuse x3 + mini holder", "INR 76", "Legs now draw up to ~10 A: 15 A ATC fuse"),
    ("ALPS 8 A rocker switch", "INR 202", "Too small for the peaks: 20 A rocker"),
    ("2S protection board (BMS) 5 A, estimate", "INR 60", "Pack has its own BMS"),
    ("Amass T-plug pairs x3", "INR 111", "XT60 matches the new pack and carries more current"),
    ("USB-C to Mini-USB OTG cable (estimate)", "INR 199", "The Waveshare board has USB-C"),
]

PRINT_SERVICE_NOTE = (
    "The CAD scripts in cad/v2/ write the printable parts to cad/out/v2/print/ (python cad/v2/body.py: the body tray "
    "and cover; python cad/v2/coupon.py: a small fit-test coupon to print first). Upload the STL files to robu.in's 3D "
    "printing service (https://robu.in/product/3d-printing-service1/, SKU 901845). Choose PETG, and TPU for the foot "
    "tips and jaw pads. The full print set is finished with the build, so do not order the whole set before then. "
    "robu shows the price after the upload; there is no published per-gram rate.")

OLD_TOTAL = 77412  # the list before the owner's decisions of 2026-11-21 (GS-4308MG arms, 2.42 inch OLED, no prints counted)


def rows():
    out = []
    for order, sec, item, store, sku, qty, price, need, stock, checked, note in ITEMS:
        out.append(dict(order=ORDER_NAMES[order], section=sec, item=item, store=store, sku=sku, qty=qty, unit_inr=price,
                        total_inr=round(qty * price, 2), need=need, stock=stock, checked=checked, note=note))
    return out


def tot(rs, need=None):
    return sum(x["total_inr"] for x in rs if need is None or x["need"] in need)


def main():
    r = rows()
    with open(HERE / "BOM.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)

    by_order = {n: [x for x in r if x["order"] == n] for n in ORDER_NAMES.values()}
    unver = [x for x in r if x["checked"] == U]
    grand, req = tot(r), tot(r, ("required",))
    pack2 = next(x for x in r if x["need"] == "recommended" and "spare" in x["item"])["total_inr"]
    bag = next(x for x in r if "explosion" in x["item"])["total_inr"]
    filament = next(x for x in r if "PETG-HS" in x["item"])["total_inr"]
    md = ["# Bill of materials (Pinch v2 style B: STS3215 legs, STS3032 arms, 6.25 inch HDMI face)", "",
          "Generated by `python hardware/bom.py`. Prices in INR **including GST**, shipping excluded. Stores were opened on 2026-11-21; "
          "each row says in `Checked` whether its page was opened that day (`UNVERIFIED` = an estimate or a search snippet). "
          "Stock changes daily, so re-check in the cart.", "",
          f"**Everything, including prints and tools: INR {grand:,.0f} against the owner's limit of INR {LIMIT:,}** "
          f"(margin INR {LIMIT - grand:,.0f}; required rows only: INR {req:,.0f}, margin INR {LIMIT - req:,.0f}). "
          f"The earlier list (GS-4308MG arms, OLED, no prints counted) was INR {OLD_TOTAL:,}.", "",
          "## The three orders", "",
          "| Order | What it is for | Required | With recommended |", "|---|---|---|---|"]
    purpose = {
        1: "The servo bench: 2 leg servos + 1 arm servo, driver board, pack, fuse, charger, one 5 V buck, tools, pendulum weights, "
           "and the 6.25 inch panel (to weigh it: at most 0.15 kg or fall back to the 3.5 inch SPI panel).",
        2: "Computer, IMU, battery monitor, the second buck, VL53L5CX: test the electronics before the body exists.",
        3: "The other 12 leg servos, 5 arm servos, hub, senses, dock, hinge, fasteners, tools and the printed body. Order after orders 1 and 2 pass.",
    }
    cum = 0.0
    for n, name in ORDER_NAMES.items():
        rs = by_order[name]
        cum += tot(rs)
        md.append(f"| {n}. {name} | {purpose[n]} | INR {tot(rs, ('required',)):,.0f} | INR {tot(rs):,.0f} |")
    md += ["", "| | Order 1 | Order 2 | Order 3 | **All three** | Limit | Margin |", "|---|---|---|---|---|---|---|",
           "| With recommended | " + " | ".join(f"INR {tot(by_order[nm]):,.0f}" for nm in ORDER_NAMES.values())
           + f" | **INR {grand:,.0f}** | INR {LIMIT:,} | INR {LIMIT - grand:,.0f} |",
           "| Required only | " + " | ".join(f"INR {tot(by_order[nm], ('required',)):,.0f}" for nm in ORDER_NAMES.values())
           + f" | INR {req:,.0f} | INR {LIMIT:,} | INR {LIMIT - req:,.0f} |", "",
           "Each order alone is far under the limit; the limit is on the sum. Order 1 grew from INR 10,859 (the plan's figure was "
           "about INR 10,900) because the bench now also needs the arm servo, a 5 V buck to power it, and the face panel to weigh; "
           "those rows leave orders 2 and 3, so the total does not change.", "",
           "The battery and charger are in order 1, not order 2: the bench test needs a 7.4 V supply, and the bench "
           "supply sold by robu (Yihua 30 V 5 A, INR 3,339-4,029) is out of stock today. The same pack then becomes "
           "the robot's first battery.", "",
           "## Budget (owner's limit INR 1,06,000 for parts, prints and tools)", "",
           f"Total INR {grand:,.0f} with the 3D prints at their mid estimate of INR {PRINTS_MID:,} (range INR {PRINTS_LOW:,} to {PRINTS_HIGH:,}: "
           f"the total is INR {grand - PRINTS_MID + PRINTS_LOW:,.0f} at the low end and INR {grand - PRINTS_MID + PRINTS_HIGH:,.0f} at the high end). "
           f"**The margin at the mid estimate is only INR {LIMIT - grand:,.0f}**, so it is not safe: if the print quote comes in high, "
           f"drop the 'recommended' rows in this order: the PETG filament (INR {filament:,.0f}), the explosion bag (INR {bag:,.0f}) and the second "
           f"pack (INR {pack2:,.0f}), together INR {filament + bag + pack2:,.0f}. They are kept in the list because the total allows them today. "
           "Cuts already applied against the parts decision's INR 111,561: no spare arm servo (6 STS3032, not 7: -3,328); perfboard servo "
           "hub instead of a custom PCB (-1,500); Robocraze VL53L0X boards at INR 132 (-1,008); the shunt, hub, HDMI cable, hinge and "
           "prints are allowances. Still available if needed: 13 instead of 14 leg servos (-2,400), the 3.5 inch SPI panel instead of "
           "the 6.25 inch one (-4,658), printing at home (-5,500 or more).", "",
           "Not in the total, nothing to buy: the home server for heavy work (speech, a local language model, face recognition) runs on a PC "
           "the owner already has; Wi-Fi is on the Radxa.", ""]
    md += ["## Out of stock or unverified (read this before ordering)", "",
           "- **Feetech STS3032 (arm servo): Evelta 501-STS3032, 27 in stock** (the page's own data read with curl on 2026-11-21; the page text "
           "also shows a hidden '0 in stock' label). Evelta's C001 listing of the same servo and the SCS0009 are sold out. "
           "I did not confirm the servo's plug size against the 5264 cable: the bench servo settles it.",
           "- **Waveshare 6.25 inch panel: price and stock are search snippets** (Tenettech INR 6,403 'in stock', page 404 for me; Hubtronics "
           "INR 5,617 incl GST, page 403): UNVERIFIED. The listed 0.30 kg may be the packed weight; weigh it. A 720 x 1560 portrait mode may "
           "need a custom mode or rotation on the ZERO 3W (Waveshare states Linux support for Raspberry Pi only).",
           "- **Evelta STS3215 C001 (1/345): OUT (0 in stock).** Evelta C046 (1/147): OUT. Evelta C044 (1/191): 4 in stock "
           "but it is a different gear, so the public BAM model does not describe it. Evelta C018 (12 V): 98 in stock, wrong voltage.",
           "- **robu.in Waveshare ST3215 7.4 V (R258500, INR 2,159): OUT.** robu has no Feetech-branded STS3215.",
           "- **Hubtronics and robu.in sit behind a Cloudflare check** (403 today; I did not bypass it): robu prices below are the earlier readings "
           "of 2026-11-18, not re-opened.",
           "- **Robokits and Amazon.in: not opened** (Robokits search page not found; Amazon.in not tried). UNVERIFIED.",
           "- **Robocraze: no STS3215 listed** (0 results). Its own MPR121 (INR 144) is OUT. Its VL53L0X INR 132 board size is not measured.",
           "- **In stock: ThinkRobotics Waveshare ST3215 7.4 V, SKU WVSH0593, INR 2,399.99** (page says 'In stock | Ready to "
           "ship', no count). This is the part to buy. It has the C001's speed.",
           "- 16 AWG silicone wire: all OUT at robu today. 2S 2200 mAh LiPo packs (all brands): OUT at robu. Yihua bench supplies: OUT.",
           f"- Rows still UNVERIFIED (estimates, allowances and snippets; {len(unver)} rows, INR {tot(unver):,.0f} of the total): "
           + "; ".join(x["item"].split(",")[0] for x in unver) + ".", ""]
    md += ["## What changed and why", "",
           "- **Arms: 6 x Feetech STS3032 bus servos (IDs 13-18), not GOTECK GS-4308MG PWM servos** (owner, 2026-11-21): position and load "
           "feedback on every joint, same protocol and driver board as the legs, 4.8-6 V from the 5 V rail B. One STS3032 stalls at 1.2 A, "
           "six at 7.2 A: more than the 5 A regulator, so each gets a torque limit of 40 % (about 0.5 A stalled, 2.9 A for all six) that "
           "`python -m crab.wiring` enforces. 18 servo cables: three packs.",
           "- **Face: Waveshare 6.25 inch 720 x 1560 colour IPS over micro-HDMI** in a hinged lid with a printed friction hinge, not the 2.42 inch "
           "OLED. It takes about 2 W at 5 V from rail A. The panel is bought in order 1 only to weigh it (at most 0.15 kg).",
           "- **Computer: Radxa ZERO 3W plus a home server on Wi-Fi** for heavy work (nothing to buy).",
           "- **Fixes:** INA219 shunt 0.002 ohm (not 0.005); three servo cable packs; a servo hub of about 55 x 36 mm on perfboard; four pogo pins "
           "for the dock (Robocraze, INR 94 for five).",
           "- **Legs: 12 x STS3215 7.4 V (+2 spares)**: Rhoban/bam publishes a measured model for the STS3215 and Open Duck Mini walks on it. "
           "Details: docs/research/SERVO_CHOICE.md and docs/research/BOM_CHANGE.md.",
           "- **Servos on the pack, regulators for the rest:** the legs run straight from the 2S Li-ion pack (6.4-8.4 V); the two 5 V regulators feed "
           "the Radxa, sensors, audio and face panel (A) and the six arm servos (B).",
           "- **New:** VL53L5CX 8x8 sensor, charging-dock parts, PETG/ASA filament, 15 A fuse, 20 A switch, bulk capacitors, the printed "
           "body as a priced row (estimate).", ""]
    md += ["## Rows removed", "", "| Removed | Was | Why |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in REMOVED] + [""]
    for n, name in ORDER_NAMES.items():
        rs = by_order[name]
        md += [f"## Order {n}: {name}", "",
               f"Required INR {tot(rs, ('required',)):,.0f}; with recommended INR {tot(rs):,.0f}.", "",
               "| Section | Item | Store | SKU | Qty | Unit INR | Total INR | Need | Stock | Checked | Note |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in rs:
            md.append(f"| {x['section']} | {x['item']} | {x['store']} | {x['sku']} | {x['qty']} | {x['unit_inr']:,.2f} | "
                      f"{x['total_inr']:,.2f} | {x['need']} | {x['stock']} | {x['checked']} | {x['note']} |")
        md.append("")
    md += ["## Cart lists (one per store, required + recommended, all orders)", "",
           "Type each SKU into the store's search box and set the quantity. Rows marked UNVERIFIED are estimates.", ""]
    for store in dict.fromkeys(x["store"] for x in r):
        if store == LO:
            continue
        lines = [x for x in r if x["store"] == store and x["need"] in ("required", "recommended")]
        if not lines:
            continue
        md += [f"### {store} cart ({len(lines)} lines, INR {sum(x['total_inr'] for x in lines):,.0f})", "",
               "| Order | SKU | Qty | Item | Line INR |", "|---|---|---|---|---|"]
        md += [f"| {x['order'].split(':')[0]} | {x['sku']} | {x['qty']} | {x['item']} | {x['total_inr']:,.2f} |" for x in lines]
        md.append("")
    md += ["", "## 3D printed parts", "", PRINT_SERVICE_NOTE, "",
           "## Not in the list on purpose", "",
           "- **Servo horns, ear screws and horn screws** come in each servo's box.",
           "- **A bench power supply:** the pack replaces it (see above).",
           "- **A USB data cable for the Radxa's first boot and any 5 V USB-C phone charger:** you own these.",
           "- **The home server:** an existing PC; nothing to buy.", ""]
    (HERE / "BOM.md").write_text("\n".join(md), encoding="utf-8")
    print(f"BOM: {len(r)} lines, total INR {grand:,.0f} of {LIMIT:,}; "
          + "; ".join(f"order {n} INR {tot(by_order[nm]):,.0f} (required {tot(by_order[nm], ('required',)):,.0f})"
                      for n, nm in ORDER_NAMES.items()))


if __name__ == "__main__":
    main()
