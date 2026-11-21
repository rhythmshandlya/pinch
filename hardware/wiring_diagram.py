"""Draw the v2 robot's wiring diagram as an SVG (no extra libraries), from the netlist.

    python hardware/wiring_diagram.py      -> hardware/wiring.svg

Everything on it comes from configs/wiring_v2.toml (through crab/wiring.py): the power chain with each wire's gauge, length and the
current it can see, the servo cables, the three I2C buses with their addresses and pins, the PWM pins, I2S, USB and the camera.
Edit the netlist, run `python -m crab.wiring` to check it, then run this again. Every board also shares GND; ground wires are
left out of the drawing (each power wire is a loop, the netlist counts both conductors)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.wiring import Electrical, Net, check  # noqa: E402

OUT = Path(__file__).resolve().parent / "wiring.svg"
W, H = 1500, 1120
COL = {"batt": "#d62728", "7v": "#ff7f0e", "5v": "#9467bd", "3v3": "#8c564b", "sig": "#e6b800",
       "i2c": "#1f77b4", "i2s": "#2ca02c", "gpio": "#7f7f7f", "usb": "#17becf", "csi": "#333333", "pwm": "#bcbd22"}

net = Net()
el = Electrical(net)
rep = check(net)
rows = {r[0]: r for r in rep.edge_rows}
parts, wires, labels = [], [], []


def box(x, y, w, h, title, lines=(), fill="#ffffff", stroke="#333"):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    parts.append(f'<text x="{x + w / 2}" y="{y + 20}" class="t">{title}</text>')
    for i, ln in enumerate(lines):
        parts.append(f'<text x="{x + w / 2}" y="{y + 38 + 15 * i}" class="s">{ln}</text>')
    return dict(x=x, y=y, w=w, h=h, l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y), b=(x + w / 2, y + h))


def wire(pts, color, label=None, at=0.5, dash=False, lx=0, ly=-6):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.0f},{y:.0f}" for i, (x, y) in enumerate(pts))
    extra = ' stroke-dasharray="6,4"' if dash else ""
    wires.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3"{extra} marker-end="url(#a)"/>')
    if label:
        k = min(int(at * (len(pts) - 1)), len(pts) - 2)
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        labels.append(f'<text x="{(x0 + x1) / 2 + lx:.0f}" y="{(y0 + y1) / 2 + ly:.0f}" class="w" fill="{color}">{label}</text>')


def wl(edge_id):
    """'14 AWG 156 mm, 15 A' for an edge of the netlist."""
    e = next(x for x in net.edges if x["id"] == edge_id)
    r = rows[edge_id]
    awg = f"{e['awg']} AWG " if "awg" in e else ""
    return f"{awg}{r[1]:.0f} mm, up to {r[3]:.1f} A" if "awg" in e else f"{r[3]:.0f} A max"


# ---- the power chain, left to right along the top ----------------------------------------------------------------------
pack = box(30, 60, 170, 70, "2S Li-ion pack", ["2 x Samsung 25R, 2500 mAh", "BMS inside, 7.4 V (6.4-8.4)"], "#fdecea")
xt = box(260, 60, 120, 70, "XT60", ["pack plug + robot", "socket"], "#fdecea")
fuse = box(440, 60, 130, 70, "15 A fuse", ["ATC blade, in a holder", f"{el.d('fuse.resistance_mohm')} mOhm"], "#fdecea")
ina = box(630, 60, 150, 70, "INA219 + shunt", [f"{float(el.d('shunt.ohm')) * 1000:g} mOhm, I2C 0x40", "reads the whole pack"], "#fdecea")
sw = box(840, 60, 130, 70, "Main switch", ["20 A rocker", "back wall"], "#fdecea")
wire([pack["r"], xt["l"]], COL["batt"], "14 AWG 150 mm", lx=0, ly=-26)
wire([xt["r"], fuse["l"]], COL["batt"], f"{rows['xt60_fuse'][1]:.0f} mm", ly=-8)
wire([fuse["r"], ina["l"]], COL["batt"], f"{rows['fuse_ina'][1]:.0f} mm", ly=-8)
wire([ina["r"], sw["l"]], COL["batt"], f"{rows['ina_switch'][1]:.0f} mm", ly=-8)

# ---- hub and the two regulators --------------------------------------------------------------------------------------
hub = box(900, 260, 250, 110, "Servo hub", ["12 leg headers: 7.4 V from the pack",
                                            "6 arm headers: 5 V from regulator B (not joined)",
                                            "bus signal pins joined; 2 x 1000 uF"], "#fff8e1")
rega = box(440, 260, 210, 70, "5 V regulator A", ["AP64500 module, 5 A", "computer, audio, sensors"], "#f3e5f5")
regb = box(600, 400, 210, 70, "5 V regulator B", ["AP64500 module, 5 A", "the six arm servos"], "#fff3e0")
wire([(905, 130), (905, 195), (1025, 195), hub["t"]], COL["7v"], f"trunk {wl('trunk')}", at=0.5, lx=-30, ly=-8)
wire([(905, 130), (905, 160), (545, 160), rega["t"]], COL["batt"], f"20 AWG {rows['sw_rega'][1]:.0f} mm", at=0.5, lx=-5, ly=-6)
wire([(700, 160), (700, 400)], COL["batt"], f"20 AWG {rows['sw_regb'][1]:.0f} mm, up to {rows['sw_regb'][3]:.1f} A", at=0.5, lx=70, ly=-6)
wire([(705, 470), (705, 520), (1025, 520), (1025, 370)], COL["5v"], f"20 AWG {rows['regb_hub5'][1]:.0f} mm", at=0.5, lx=0, ly=-6)

# ---- the servos -------------------------------------------------------------------------------------------------------------
legs = [("Front-left leg", "IDs 1, 2, 3", "FL"), ("Front-right leg", "IDs 4, 5, 6", "FR"), ("Rear-left leg", "IDs 7, 8, 9", "RL"),
        ("Rear-right leg", "IDs 10, 11, 12", "RR")]
for i, (name, ids, _k) in enumerate(legs):
    y = 40 + 74 * i
    b = box(1230, y, 240, 62, name, [ids + ": 3 x STS3215", "one 900 mm 24 AWG cable each"], "#fffde7")
    wire([hub["r"], (1190, 315), (1190, y + 31), b["l"]], COL["7v"])
b = box(1230, 350, 240, 74, "Arms: 6 x GS-4308MG", ["500 mm 24 AWG cable each", "signal wires to the Radxa PWM pins"], "#fffde7")
wire([(1150, 345), (1190, 345), (1190, 387), b["l"]], COL["5v"])
labels.append(f'<text x="1205" y="334" class="w" fill="{COL["7v"]}">7.4 V</text>')

# ---- the Radxa and what hangs on it ---------------------------------------------------------------------------------------------
pi = box(440, 640, 300, 150, "Radxa ZERO 3W", ["5 V in on pins 2 + 4, GND 6 + 9 (two wires each)",
                                               "3.3 V out on pins 1 + 17 to the sensors",
                                               f"regulator A to Radxa: {wl('rega_radxa')}",
                                               "USB-C host to the servo driver board", "CSI flex to the camera"], "#e8f5e9")
wire([(470, 330), (470, 640)], COL["5v"], "5 V", lx=-26, ly=40)
drv = box(130, 640, 220, 70, "Servo driver board", ["Waveshare, USB to the 1 Mbaud bus", "signal + GND to the hub only"], "#e0f7fa")
wire([(440, 690), (350, 690)], COL["usb"], "USB-C", lx=0, ly=-8)
wire([(240, 640), (240, 560), (960, 560), (960, 370)], COL["sig"], "bus signal", at=0.4, lx=0, ly=-6)
cam = box(130, 760, 220, 56, "Camera (front, low)", ["OV5647 160 deg, 22-pin flex"], "#f1f8e9")
wire([(440, 760), (350, 788)], COL["csi"], "CSI", lx=-6, ly=-8)

y0 = 600
for k, bus in enumerate(net.i2c):
    devs = ", ".join(f"{d['part']} {d['addr']:#04x}" for d in bus["devices"])
    note = {"i2c3": "kernel i2c-gpio, pins 3 / 5", "i2c4": "hardware I2C4, pins 27 / 28", "i2c5": "hardware I2C5, pins 31 / 29"}[bus["name"]]
    xs = [d for d in bus["devices"] if "reset_pin" in d]
    extra = ("XSHUT/LPn pins " + ", ".join(str(d["reset_pin"]) for d in xs)) if xs else "its own bus (clock stretching)" if bus["name"] == "i2c4" else "shared, slow devices"
    b = box(820, y0 + 105 * k, 380, 92, f"{bus['name']}: {note}", [devs[:64], devs[64:128], extra, f"{bus['speed_khz']} kHz"], "#e3f2fd")
    wire([pi["r"], (780, 700), (780, y0 + 105 * k + 46), b["l"]], COL["i2c"], bus["name"], at=0.9, lx=-18, ly=-6)
i2s = net.tree["i2s"]
b = box(1230, 600, 240, 70, "Audio (I2S3 pins 12, 35, 38, 40)", ["amp + speaker out, 2 mics in", "mics: SEL to GND / 3.3 V"], "#e8f5e9")
wire([(740, 740), (780, 740), (780, 720), (1215, 720), (1215, 635), b["l"]], COL["i2s"], "I2S", at=0.5, lx=0, ly=-8)
pw = ", ".join(f"{p['pin']}" for p in sorted(net.pwm, key=lambda p: p["channel"]))
b = box(1230, 700, 240, 70, "Arm PWM outputs", [f"pins {pw}", "channel 0-5 = arm servos"], "#f9fbe7")
wire([(740, 770), (1000, 770), (1000, 760), b["l"]], COL["pwm"], "6 x PWM", at=0.5, lx=0, ly=-8)
b = box(1230, 800, 240, 70, "Touch + dock", ["MPR121 on i2c3: 6 copper pads", "pogo pins -> pack charge lead"], "#f1f8e9")

# ---- legend ---------------------------------------------------------------------------------------------------------------------
lg = [("pack 6.4-8.4 V", COL["batt"]), ("servo rail", COL["7v"]), ("5 V", COL["5v"]), ("bus signal", COL["sig"]), ("USB", COL["usb"]),
      ("I2C", COL["i2c"]), ("I2S", COL["i2s"]), ("PWM", COL["pwm"]), ("camera flex", COL["csi"])]
for i, (name, col) in enumerate(lg):
    x = 30 + 150 * i
    labels.append(f'<line x1="{x}" y1="{H - 56}" x2="{x + 26}" y2="{H - 56}" stroke="{col}" stroke-width="4"/>'
                  f'<text x="{x + 32}" y="{H - 51}" class="lg">{name}</text>')
state = "passes" if rep.ok else "FAILS"
labels.append(f'<text x="30" y="{H - 28}" class="lg">Generated from configs/wiring_v2.toml; the netlist check {state} '
              f'(python -m crab.wiring). "up to" = the current a wire can carry at worst: sustained, capped by the fuse.</text>')
labels.append(f'<text x="30" y="{H - 10}" class="lg">Every board also shares GND (black wires, not drawn). '
              f'Check polarity with the multimeter before plugging anything in.</text>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Arial, sans-serif">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>
<style>.t{{font-size:15px;font-weight:600;text-anchor:middle;fill:#222}} .s{{font-size:12px;text-anchor:middle;fill:#444}}
.w{{font-size:12px;font-weight:600}} .lg{{font-size:12px;fill:#333}} .h{{font-size:20px;font-weight:700;fill:#222}}</style></defs>
<rect width="{W}" height="{H}" fill="#fafafa"/>
<text x="30" y="34" class="h">Pinch v2 - wiring (generated from the netlist)</text>
{chr(10).join(wires)}
{chr(10).join(parts)}
{chr(10).join(labels)}
</svg>
"""

if __name__ == "__main__":
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}")
