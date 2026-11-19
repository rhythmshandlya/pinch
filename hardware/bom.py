"""Single source of truth for the order. Generates hardware/BOM.md and hardware/BOM.csv.

    python hardware/bom.py

Prices in INR **including GST**, checked by hand on the vendors' pages on 2026-11-18.
Evelta shows prices ex-GST; they are converted at 18% (5% for the two hand tools Evelta marks so).
Three stores: Evelta (main), robu.in (prints + what Evelta lacks), Amazon.in (the few small items
neither had in stock; SKU = Amazon's ASIN, type it into the Amazon search box).
Stock changes daily: re-check in the cart before paying.

need:  "required" | "recommended" | "optional"
stock: what the store showed on the day ("OUT" = sold out at the time of checking)
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV, RB, AZ = "Evelta", "robu.in", "Amazon.in"

# (section, item, store, SKU, qty, unit price INR incl. GST, need, stock, note)
ITEMS = [
    # ------------------------------------------------------------- servos
    ("Servos & bus", "Feetech STS3032 serial bus servo (legs)", EV, "501-STS3032", 14, 3327.60,
     "required", "27 in stock", "12 + 2 spares. All leg joints (IDs 1-12); they report position, load and heat"),
    ("Servos & bus", "GOTECK GS-4308MG metal-gear micro servo, pack of 2 (arms)", EV, "448-GS-4308MG", 4, 886.58,
     "required", "14 in stock", "4 packs = 8 servos: 6 + 2 spares (Evelta's unit is a pair: audit). Arm swing, lift and claw. Plain PWM servos (no feedback), 150 mm leads"),
    ("Servos & bus", "Feetech FE-URT-1 USB to bus-servo signal converter", RB, "1110047", 1, 1069.00,
     "required", "in stock", "Evelta's FE-URT-2 was sold out; FE-URT-1 does the same job"),
    ("Servos & bus", "5264 3-pin servo cable, 900 mm (pack of 6)", RB, "R263666", 2, 359.00,
     "required", "in stock", "Extensions from each limb to the hub; coil the extra length inside the body"),
    ("Servos & bus", "8 x 12 cm double-sided prototype PCB", RB, "44139", 1, 114.00,
     "required", "check", "Servo bus hub (cut a 3 x 2 cm piece); price from the earlier robu check"),
    ("Servos & bus", "2.54 mm 1 x 40 male header strip", RB, "539076", 2, 15.00,
     "required", "check", "Hub: 6 x 3-pin bus headers + 6 x 3-pin arm-servo headers (power from the hub, signal to the Radxa)"),
    # ------------------------------------------------------------- compute, sensing, sound
    ("Brain & senses", "Radxa ZERO 3W, 2 GB RAM (Pi Zero size, RK3566, Wi-Fi)", EV, "498-RS107-D2E0H1W15", 1, 6460.50,
     "required", "15 in stock", "Replaces the sold-out Pi Zero 2 W: same size, 40-pin header and mounting holes"),
    ("Brain & senses", "SanDisk Ultra 32 GB microSD (A1, class 10)", RB, "R268040", 1, 1629.00,
     "required", "in stock", "Radxa OS (Debian)"),
    ("Brain & senses", "160 deg wide-angle camera for Pi Zero (OV5647 fisheye, 22-pin)", AZ, "B083XMGSVP", 1, 3346.00,
     "required", "listed", "Pinch's eyes: sees the claw's workspace, the desk around it and your face (decision 16)"),
    ("Brain & senses", "USB-C (male) to Mini-USB (male) OTG cable, 10-15 cm", AZ, "search 'USB C to mini USB OTG cable short'", 1, 199.00,
     "required", "check", "Radxa's USB-C 2 (host) port -> FE-URT-1's Mini-USB socket. Short and direct: an OTG adapter + cable is ~8 cm of rigid parts inside the body (audit). Price estimated"),
    ("Brain & senses", "7Semi BNO055 9-DOF IMU breakout", EV, "004-ES-12149", 1, 1341.66,
     "required", "440 in stock", "Balance + fall detection"),
    ("Brain & senses", "7Semi INA219 voltage / current monitor breakout", EV, "004-ES-12152", 1, 241.90,
     "required", "88 in stock", "Battery voltage for the safety stop and the pet's 'hunger'"),
    ("Brain & senses", "VL53L0X ToF distance module, small (GY-VL53L0XV2, 25 x 10.7 mm)", RB, "search 'VL53L0X'", 6,
     300.00, "required", "check", "Five cliff beams (front pair, rear, both sides; decision 30) + 1 spare. Price estimated. Evelta's Grove board (20 x 40) is too long"),
    ("Brain & senses", "Adafruit MAX98357A I2S 3 W amplifier", EV, "076-3006", 1, 860.07,
     "required", "14 in stock", "Tones and chirps (the pet never talks)"),
    ("Brain & senses", "28 mm 8 ohm 0.5 W speaker (GSPK2805TN)", RB, "R210811", 1, 82.00,
     "required", "in stock", "Glued into the seat inside the dome"),
    ("Brain & senses", "DFRobot I2S MEMS microphone module", RB, "1074722", 2, 629.00,
     "required", "in stock", "Two (left + right on one I2S line): hears you, your mood, meetings, and where you are"),
    ("Brain & senses", "2.42 inch 128x64 white OLED, SSD1309, I2C/SPI (Teyleten Robot)", AZ, "B09LND6QJ1", 1, 2300.00,
     "required", "listed", "Pinch's face (animated eyes). Set to I2C (resistor jumpers on the back), own bus"),
    ("Brain & senses", "MPR121 12-channel capacitive touch controller (Robocraze)", AZ, "B07KF5FNBL", 1, 229.00,
     "required", "listed", "Touch zones under the shell: head, back, sides, face, claws (strokes, tickles, boops)"),
    ("Brain & senses", "Conductive copper foil tape, 10 mm x 10 m", AZ, "B0FGQCPFSP", 1, 294.00,
     "required", "listed", "The touch pads: strips stuck inside the shell, wired to the MPR121"),
    ("Brain & senses", "40-pin female-female jumper wires, 20 cm", EV, "083-JW-254-F-F-40", 1, 68.44,
     "required", "54 in stock", "Peel off what you need for the sensor wiring"),
    # ------------------------------------------------------------- power
    ("Power", "Bonka 7.4 V 850 mAh 25C 2S LiPo (57 x 30 mm, 46 g, T-plug)", RB, "1891489", 2, 1039.00,
     "required", "in stock", "1 + 1 spare. Thickness not listed: the bay takes up to 18 mm"),
    ("Power", "SkyRC E3 AC 2-3S balance charger", RB, "4511", 1, 1689.00,
     "required", "in stock", "Charges through the balance lead; wall plug, beginner-proof"),
    ("Power", "LiPo explosion-proof bag", RB, "R260502", 1, 299.00, "required", "in stock", "Charge and store in it"),
    ("Power", "BX100 1-8S LiPo low-voltage buzzer", RB, "835784", 1, 218.00,
     "recommended", "in stock", "Second line of defence; plugs into the balance lead"),
    ("Power", "7Semi AP64500 buck converter, 5 V 5 A", EV, "004-BO-10188", 2, 823.64,
     "required", "125 in stock", "One for the servo bus, one for the Pi + audio. Fixed 5 V: nothing to adjust"),
    ("Power", "Mini blade fuse holder (PCB type; solder two leads on)", RB, "R260426", 1, 19.00,
     "required", "in stock", ""),
    ("Power", "5 A mini blade fuse", RB, "R260430", 3, 19.00, "required", "in stock", "1 + 2 spares"),
    ("Power", "ALPS SPST rocker switch, 8 A (SDDJF1A100)", EV, "341-SDDJF1A100", 1, 201.78,
     "required", "28 in stock", "In the belly; check it fits the 19.4 x 13.4 mm cutout. VERIFY it latches: one listing calls it momentary / discontinued (audit) - if so, any latching mini rocker that fits"),
    ("Power", "2S Li-ion/LiPo protection board (BMS), 5 A, e.g. HX-2S", RB, "search '2S BMS protection board'", 1, 60.00,
     "required", "check", "Hard low-voltage floor: the software's 6.4 V stop leaves the Radxa and OLED draining the pack (audit). Price estimated"),
    ("Power", "0.01 ohm 2512 current-sense resistor (for the INA219)", RB, "search '0.01 ohm 2512 resistor'", 2, 20.00,
     "required", "check", "Replaces the INA219 board's 0.1 ohm shunt: that one clips at 3.2 A and drops 0.5 V at 5 A (audit). Price estimated"),
    ("Brain & senses", "Aluminium heatsink ~14 x 14 x 6 mm, self-adhesive", RB, "search 'heatsink 14x14'", 1, 40.00,
     "required", "check", "On the Radxa's RK3566 inside the closed shell (audit: no cooling). Price estimated"),
    ("Power", "Amass T-plug (Deans) connector pair with cap", RB, "R264629", 3, 37.00,
     "required", "in stock", "Matches the Bonka pack"),
    ("Power", "20 AWG silicone wire, red (per metre)", RB, "1824992", 2, 40.00, "required", "in stock", ""),
    ("Power", "20 AWG silicone wire, black (per metre)", RB, "1824989", 2, 40.00, "required", "in stock", ""),
    ("Power", "28 AWG silicone wire kit, 5 colours x 5 m", RB, "R229090", 1, 499.00, "required", "in stock",
     "Sensor and I2S wiring"),
    ("Power", "JST-PH 2.0 2-pin lead, 30 cm", EV, "083-2P-2mm-F-Wire", 5, 10.62, "recommended", "205 in stock",
     "Pluggable speaker / touch leads"),
    # ------------------------------------------------------------- mechanical
    ("Assembly", "M2 x 6 socket screw + nut (pack of 12)", RB, "56219", 3, 239.00, "required", "in stock",
     "Brackets and spares (the servos' own ear and horn screws come in their boxes)"),
    ("Assembly", "M2 x 8 socket screw + nut (pack of 12)", RB, "56246", 1, 219.00, "required", "in stock",
     "Brain tray"),
    ("Assembly", "M2 x 10 socket screw + nut (pack of 12)", RB, "56253", 1, 239.00, "required", "in stock",
     "Foot tips, battery door"),
    ("Assembly", "M2.5 x 8 socket screw (pack of 10)", RB, "1711459", 1, 154.00, "required", "in stock",
     "Radxa ZERO 3W onto the tray"),
    ("Assembly", "Heat-shrink kit, 140 pcs", RB, "R267330", 1, 349.00, "required", "in stock", ""),
    ("Assembly", "Nylon zip ties 100 mm (pack of 100)", RB, "1081544", 1, 30.00, "required", "in stock", ""),
    ("Assembly", "N52 neodymium disc magnets 6 x 3 mm (pack of 50)", AZ, "B00Z7Z3QXQ", 1, 299.00, "required", "listed",
     "8 hold the shell on; the rest are for swappable shells and hats later"),
    ("Assembly", "3M Scotch heavy-duty double-sided foam tape", AZ, "B00N1U9AJS", 1, 171.00, "required", "listed",
     "Mounts the small boards"),
    ("Assembly", "Fevikwik Ezee Gel 3 g (non-drip super glue)", AZ, "B0CQV798H9", 1, 50.00, "required", "listed",
     "Magnets, foot tips"),
    # ------------------------------------------------------------- tools
    ("Tools", "YIHUA 926 temperature-controlled soldering station", RB, "1316303", 1, 1509.00, "required", "in stock", ""),
    ("Tools", "Solder wire 0.8 mm, 50 g", RB, "1837790", 1, 269.00, "required", "in stock", ""),
    ("Tools", "Noel flux paste 10 g", RB, "762465", 1, 21.00, "required", "in stock", ""),
    ("Tools", "Noel desoldering wick", RB, "769914", 1, 34.00, "required", "in stock", ""),
    ("Tools", "TE-801 helping hands with magnifier", RB, "17940", 1, 489.00, "recommended", "in stock", ""),
    ("Tools", "Mastech MAS830L digital multimeter", RB, "5981", 1, 899.00, "required", "in stock",
     "Check every voltage and polarity before plugging in"),
    ("Tools", "Freemans FDC150 digital vernier calliper 150 mm", RB, "1608703", 1, 2349.00, "required", "in stock",
     "Measure one servo of each kind before printing (ASSEMBLY.md stage 0)"),
    ("Tools", "8-in-1 aluminium hex screwdriver set (RC)", EV, "447-A-234", 1, 903.00, "required", "in stock",
     "1.5 / 2 / 2.5 mm hex for the M2 / M2.5 screws"),
    ("Tools", "Stanley wire stripper / cutter 130 mm", RB, "1600571", 1, 369.00, "required", "in stock", ""),
    ("Tools", "Plato 170 flush cutter", EV, "447-A-371", 1, 115.50, "required", "40 in stock", ""),
    ("Tools", "ESD curved tweezers", RB, "R232396", 1, 35.00, "recommended", "in stock", ""),
]


PRINT_SERVICE_NOTE = (
    "Upload the 14 STL files from cad/out/print/ to robu.in's 3D printing service "
    "(https://robu.in/product/3d-printing-service1/, SKU 901845) and set the quantities from "
    "cad/out/print_sheet.csv (31 pieces, incl. the desk tray). Choose PETG, and TPU for the 4 foot tips and 4 jaw pads. robu shows the price "
    "(material, service, shipping and tax) after the upload; there is no published per-gram rate.")


def rows():
    out = []
    for sec, item, store, sku, qty, price, need, stock, note in ITEMS:
        out.append(dict(section=sec, item=item, store=store, sku=sku, qty=qty, unit_inr=price,
                        total_inr=round(qty * price, 2), need=need, stock=stock, note=note))
    return out


def main():
    r = rows()
    with open(HERE / "BOM.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r[0]))
        w.writeheader()
        w.writerows(r)

    req = [x for x in r if x["need"] == "required"]
    total = sum(x["total_inr"] for x in r)
    total_req = sum(x["total_inr"] for x in req)
    by_store = {}
    for x in r:
        by_store[x["store"]] = by_store.get(x["store"], 0) + x["total_inr"]
    out_now = [x for x in r if x["stock"] == "OUT"]

    md = ["# Bill of materials (hand-sized crab)", "",
          "Generated by `python hardware/bom.py`. Prices in INR **including GST**, checked 2026-11-18 "
          "(by hand, on the vendors' pages). Stock changes daily, so re-check in the cart.", "",
          f"**Total: ₹{total:,.0f}** (required items: ₹{total_req:,.0f}), plus shipping and the 3D prints.", ""]
    md += ["| Store | Subtotal |", "|---|---|"] + [f"| {k} | ₹{v:,.0f} |" for k, v in by_store.items()] + [""]
    if out_now:
        md += ["## ⚠ Sold out on the day of checking", "",
               "These were sold out on 2026-11-18: " + ", ".join(x["item"] for x in out_now) + ".", ""]
    md += ["## Why these parts", "",
           "Every part was in stock on 2026-11-18. The first design's HD-1910 and SCS0009 servos and the Pi Zero 2 W "
           "were sold out everywhere, so the crab was redesigned around what was available:", "",
           "- **Legs: 12 × Feetech STS3032.** Same bus and adapter as before. They report position, load and "
           "temperature, so the robot can calibrate and protect itself.",
           "- **Arms: 6 × GOTECK GS-4308MG (5.8 g each).** A big STS3032 in each claw made the arms heavy and "
           "overloaded the front thighs. These tiny servos cut the arm weight by four-fifths. Their signal "
           "wires go straight to six hardware PWM pins on the Radxa, so no extra board is needed. They can't "
           "report their position, so the trim tool (robot/calibrate.py --trim-arms) lines them up by eye.",
           "- **Brain: Radxa ZERO 3W.** The same size and pin header as the Pi Zero, and faster.", ""]
    md += ["## Parts and tools", "",
           "| Section | Item | Store | SKU | Qty | Unit ₹ | Total ₹ | Need | Stock | Note |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for x in r:
        md.append(f"| {x['section']} | {x['item']} | {x['store']} | {x['sku']} | {x['qty']} | {x['unit_inr']:,.2f} | "
                  f"{x['total_inr']:,.2f} | {x['need']} | {x['stock']} | {x['note']} |")
    md += ["", "## Cart lists (one per store, required + recommended)", "",
           "Type each SKU into the store's search box and set the quantity. Items marked OUT were sold out on "
           "the day of checking.", ""]
    for store in (EV, RB, AZ):
        lines = [x for x in r if x["store"] == store and x["need"] in ("required", "recommended")]
        md += [f"### {store} cart ({len(lines)} lines, ₹{sum(x['total_inr'] for x in lines):,.0f})", "",
               "| SKU | Qty | Item | Line ₹ |", "|---|---|---|---|"]
        md += [f"| {x['sku']} | {x['qty']} | {x['item']}{' **(OUT)**' if x['stock'] == 'OUT' else ''} | "
               f"{x['total_inr']:,.2f} |" for x in lines]
        md.append("")
    md += ["", "## 3D printed parts", "", PRINT_SERVICE_NOTE, "",
           "## Not in the list on purpose", "",
           "- **Servo horns, ear screws and horn screws** come in each servo's box.",
           "- **A bench power supply** isn't needed. The first power-up uses the battery, with no servos connected "
           "(ASSEMBLY.md stage 2).",
           "- **A 5 V USB-C charger** for the Radxa on the bench (Stage 1): any phone charger with a USB-C cable "
           "works; inside the robot it runs from the regulator.", ""]
    (HERE / "BOM.md").write_text("\n".join(md), encoding="utf-8")
    print(f"BOM: {len(r)} lines, total INR {total:,.0f} (required INR {total_req:,.0f}); "
          + ", ".join(f"{k} INR {v:,.0f}" for k, v in by_store.items()))


if __name__ == "__main__":
    main()
