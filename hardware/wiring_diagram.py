"""Draw the crab's wiring diagram as an SVG (no extra libraries).

    python hardware/wiring_diagram.py      -> hardware/wiring.svg

Matches hardware/electronics.md. Every board also shares GND; ground wires are left out of the
drawing to keep it readable (each power arrow carries its own black return wire).
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "wiring.svg"
W, H = 1280, 900

COL = {"batt": "#d62728", "6v": "#ff7f0e", "5v": "#9467bd", "3v3": "#8c564b", "sig": "#e6b800",
       "i2c": "#1f77b4", "i2s": "#2ca02c", "gpio": "#7f7f7f", "usb": "#17becf", "csi": "#333333"}

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


# ---------------------------------------------------------------- power chain (left)
bat = box(30, 60, 170, 70, "2S LiPo 7.4 V", ["850 mAh, battery bay", "T-plug"], "#fdecea")
fuse = box(30, 175, 170, 50, "5 A blade fuse", ["right after the battery"], "#fdecea")
ina = box(30, 270, 170, 60, "INA219", ["battery monitor", "I2C 0x40"], "#fdecea")
sw = box(30, 375, 170, 50, "Main switch", ["belly, 8 A rocker"], "#fdecea")
reg6 = box(30, 480, 170, 60, "5 V regulator #1", ["5 A, servo power", "measure 4.9-5.2 V"], "#fff3e0")
reg5 = box(30, 600, 170, 60, "5 V regulator #2", ["5 A, Radxa + audio", "measure 4.9-5.2 V"], "#f3e5f5")
wire([bat["b"], fuse["t"]], COL["batt"], "7.4 V", lx=28)
wire([fuse["b"], ina["t"]], COL["batt"])
wire([ina["b"], sw["t"]], COL["batt"])
wire([sw["b"], reg6["t"]], COL["batt"])
wire([(115, 425), (15, 425), (15, 630), reg5["l"]], COL["batt"])

# ---------------------------------------------------------------- servo bus (middle / right)
hub = box(330, 250, 190, 110, "Servo hub", ["V+ from regulator #1", "bus signal from FE-URT-1",
                                             "4 leg branches + 6 arm plugs"], "#fff8e1")
urt = box(330, 470, 190, 60, "FE-URT-1", ["USB -> servo bus", "/dev/ttyUSB0, 1 Mbps"], "#e0f7fa")
wire([reg6["r"], (265, 510), (265, 330), (330, 330)], COL["6v"], "5 V", at=0.3, lx=-24)
wire([urt["t"], hub["b"]], COL["sig"], "signal", lx=30)

limbs = [("Front-left leg", "IDs 1 coxa, 2 thigh, 3 knee"), ("Front-right leg", "IDs 4, 5, 6"),
         ("Rear-left leg", "IDs 7, 8, 9"), ("Rear-right leg", "IDs 10, 11, 12"),
         ("Left arm (PWM 0-2)", "swing, lift, claw: Radxa pins 16, 18, 21"),
         ("Right arm (PWM 3-5)", "swing, lift, claw: Radxa pins 24, 7, 19")]
for i, (name, ids) in enumerate(limbs):
    y = 40 + 62 * i
    b = box(640, y, 270, 50, name, [ids], "#fffde7")
    wire([hub["r"], (580, 305), (580, y + 25), b["l"]], COL["6v"])
labels.append('<text x="775" y="420" class="s">legs: STS3032 bus servos, daisy-chained inside each leg</text>')
labels.append('<text x="775" y="437" class="s">arms: GS-4308MG, power from the hub, signal wire to the Radxa</text>')

# ---------------------------------------------------------------- Radxa + sensors (bottom)
pi = box(330, 640, 280, 120, "Radxa ZERO 3W", ["brain tray, under the dome",
                                                "5 V in on pin 2 / GND pin 6",
                                                "I2C pins 3/5, I2S pins 12/35/38/40",
                                                "arm PWM pins 7, 16, 18, 19, 21, 24"], "#e8f5e9")
wire([reg5["r"], (260, 630), (260, 680), (330, 680)], COL["5v"], "5 V", at=0.6, lx=-26)
wire([(400, 640), (400, 530)], COL["usb"], "USB-C", lx=-36)
wire([(560, 640), (560, 590), (610, 590), (610, 368), (640, 368)], COL["sig"], "arm PWM", at=0.4, lx=-34)
wire([(440, 640), (440, 595), (230, 595), (230, 300), ina["r"]], COL["i2c"], "I2C", at=0.5, lx=-22, dash=True)

sens = [("BNO055 IMU", "I2C 0x28, deck (foam tape)", COL["i2c"], "I2C"),
        ("2 x VL53L0X", "I2C 0x29 -> re-addressed, XSHUT pins 11/13", COL["i2c"], "I2C + 2 GPIO"),
        ("MAX98357A amp + speaker", "I2S out, 5 V", COL["i2s"], "I2S"),
        ("I2S microphone", "I2S in, 3.3 V, SEL to GND", COL["i2s"], "I2S"),
        ("2 x TTP223 touch", "head / back, pins 29 / 31", COL["gpio"], "GPIO"),
        ("Camera (front face)", "22-pin CSI flex", COL["csi"], "CSI")]
for i, (name, desc, col, wl) in enumerate(sens):
    y = 440 + 64 * i
    b = box(960, y, 290, 52, name, [desc], "#f1f8e9")
    wire([pi["r"], (780, 700), (780, y + 26), b["l"]], col, wl, at=0.9, lx=-10)

# ---------------------------------------------------------------- legend
lg = [("battery 7.4 V", COL["batt"]), ("servo 5 V", COL["6v"]), ("Radxa 5 V", COL["5v"]),
      ("servo signal", COL["sig"]), ("USB", COL["usb"]), ("I2C", COL["i2c"]), ("I2S audio", COL["i2s"]),
      ("GPIO", COL["gpio"]), ("camera flex", COL["csi"])]
for i, (name, col) in enumerate(lg):
    x = 30 + 136 * (i % 9)
    labels.append(f'<line x1="{x}" y1="{H - 40}" x2="{x + 26}" y2="{H - 40}" stroke="{col}" stroke-width="4"/>'
                  f'<text x="{x + 32}" y="{H - 35}" class="lg">{name}</text>')
labels.append(f'<text x="30" y="{H - 12}" class="lg">Every board also shares GND (black wires, not drawn). '
              f'Check polarity with the multimeter before plugging anything in.</text>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Arial, sans-serif">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>
<style>.t{{font-size:15px;font-weight:600;text-anchor:middle;fill:#222}} .s{{font-size:12px;text-anchor:middle;fill:#444}}
.w{{font-size:12px;font-weight:600}} .lg{{font-size:12px;fill:#333}} .h{{font-size:20px;font-weight:700;fill:#222}}</style></defs>
<rect width="{W}" height="{H}" fill="#fafafa"/>
<text x="30" y="34" class="h">Robotic crab (hand-sized) - wiring</text>
{chr(10).join(wires)}
{chr(10).join(parts)}
{chr(10).join(labels)}
</svg>
"""

if __name__ == "__main__":
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}")
