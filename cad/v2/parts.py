"""Real shapes for the bought electronics, in place of plain boxes (cad/v2/assembly.py electronics()).

Each builder returns [(suffix, solid, material)] in the part's own frame: the layout box centred at the origin with its
size_mm along x, y, z (configs/layout_v2.toml), so every shape stays inside the room the fit checks reserved. Materials:
"pcb" (green board), "dark" (chips, plastic, cells' wrap ends), "metal" (connector shells, heatsink), "battery" (the
pack's wrap), "white" (connector bodies).

Makers' 3D models (cad/vendor/fetch.py downloads them; the raw files are not committed): Radxa ZERO 3W, Waveshare
Bus Servo Adapter (A), DFRobot SEN0526 microphone, Adafruit MAX98357A.

Sources: Samsung INR18650-25R data sheet (18.33 x 64.85 mm cell); Radxa ZERO 3W product brief (65 x 30 mm, Pi Zero
footprint: holes 3.5 mm from the edges, 58 x 23 mm apart); Adafruit MAX98357A (product 3006) official STEP from
Adafruit_CAD_Parts (MIT); Waveshare Bus Servo Adapter (A) wiki (42 x 33 mm, holes 37 x 28 mm); GSPK2805TN data sheet
(28 mm, 5 mm); the rest from the sellers' listings and photos (estimates, flagged in the layout's size_source).
"""

import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Box, Cylinder, Pos, Rot, import_step  # noqa: E402

VENDOR = Path(__file__).resolve().parent.parent / "vendor"
PCB_T = 1.6


def _box(c, s):
    return Pos(*c) * Box(*s)


def _xcyl(c, r, length):
    return Pos(*c) * Rot(0, 90, 0) * Cylinder(r, length)


def _zcyl(c, r, h):
    return Pos(*c) * Cylinder(r, h)


def _ycyl(c, r, length):
    return Pos(*c) * Rot(90, 0, 0) * Cylinder(r, length)


def _board_z(size, parts):
    """A board lying in xy (thickness along z): the PCB on the -z face, `parts` = [(x, y, l, w, h, material)] standing
    on the board's top (x, y from the centre)."""
    sx, sy, sz = size
    z0 = -sz / 2
    out = [("pcb", _box((0, 0, z0 + PCB_T / 2), (sx, sy, PCB_T)), "pcb")]
    for i, (x, y, lx, ly, h, mat) in enumerate(parts):
        h = min(h, sz - PCB_T)
        out.append((f"c{i}", _box((x, y, z0 + PCB_T + h / 2), (lx, ly, h)), mat))
    return out


def battery(size):
    """2S1P Samsung 25R: two cells side by side along x in a blue PVC wrap, the BMS board across one end, leads out."""
    sx, sy, sz = size
    r = 18.33 / 2 + 0.25                                       # cell + wrap
    L = 64.85 + 0.5
    x0 = -sx / 2 + L / 2
    out = []
    for k, y in enumerate((-r - 0.1, r + 0.1)):
        out.append((f"cell{k}", _xcyl((x0, y, -sz / 2 + r), r, L), "battery"))
        out.append((f"cap{k}", _xcyl((x0 - L / 2 + 0.3, y, -sz / 2 + r), r - 1.5, 0.6), "dark"))
    bx = x0 + L / 2 + 1.2                                      # the BMS stands on the + end
    out.append(("bms", _box((bx, 0, 0), (1.6, min(sy - 1, 34.0), min(sz - 1, 16.0))), "pcb"))
    out.append(("fet", _box((bx + 1.3, 6, 2), (1.0, 6.0, 5.0)), "dark"))
    return out


def radxa(size):
    """Radxa ZERO 3W: 65 x 30 mm board, RK3566 + RAM in the middle, two USB-C and the micro-HDMI on one long edge,
    the camera FPC at the -x end, the microSD at the +x end, the 40-pin header holes along the other long edge."""
    sx, sy, sz = size
    e = -sy / 2                                                # connector edge
    parts = [(2.0, 1.0, 14.0, 14.0, 1.3, "dark"),              # RK3566
             (-12.0, 2.0, 10.0, 12.0, 1.0, "dark"),            # RAM
             (16.0, -3.0, 8.0, 8.0, 0.9, "dark"),              # Wi-Fi module
             (-22.0, e + 3.6, 8.9, 7.2, 3.2, "metal"),         # USB-C (power)
             (-10.0, e + 3.6, 8.9, 7.2, 3.2, "metal"),         # USB-C (host, to the servo driver)
             (6.0, e + 3.0, 6.6, 6.0, 3.0, "metal"),           # micro-HDMI (to the face panel)
             (-sx / 2 + 2.5, 0.0, 5.0, 17.0, 1.5, "white"),    # camera FPC (22-pin)
             (sx / 2 - 7.5, 3.0, 12.0, 11.5, 1.4, "metal"),    # microSD
             (0.0, sy / 2 - 3.5, 51.0, 5.0, 0.3, "metal")]     # 40-pin header pads (jumper wires plug onto it)
    return _board_z(size, parts)


def heatsink(size):
    sx, sy, sz = size
    out = [("base", _box((0, 0, -sz / 2 + 1.0), (sx, sy, 2.0)), "metal")]
    n = 6
    for k in range(n):
        x = -sx / 2 + (k + 0.5) * sx / n
        out.append((f"fin{k}", _box((x, 0, 1.0), (1.0, sy, sz - 2.0)), "metal"))
    return out


def driver(size):
    """Waveshare Bus Servo Adapter (A): 42 x 33 mm; DC jack and screw terminal on one end, USB-C on the side, the
    servo bus headers along the other end."""
    sx, sy, sz = size
    parts = [(-sx / 2 + 7.0, -8.0, 14.0, 9.0, 10.5, "dark"),   # 5.5 x 2.1 DC jack
             (-sx / 2 + 5.0, 8.0, 10.0, 7.5, 9.0, "white"),    # power screw terminal
             (0.0, -sy / 2 + 3.6, 8.9, 7.2, 3.2, "metal"),     # USB-C
             (sx / 2 - 4.0, 0.0, 5.0, 20.0, 6.0, "white"),     # servo bus headers
             (4.0, 6.0, 6.0, 6.0, 1.2, "dark")]                # USB-serial chip
    return _board_z(size, parts)


def hub(size):
    """The perfboard servo hub (55 x 36 mm): two 1000 uF bulk capacitors, rows of 3-pin servo headers, power posts."""
    sx, sy, sz = size
    parts = [(-sx / 2 + 7.0, -sy / 2 + 7.0, 10.0, 10.0, 12.0, "dark"),
             (-sx / 2 + 7.0, sy / 2 - 7.0, 10.0, 10.0, 12.0, "dark")]
    for k in range(6):
        parts.append((-6.0 + k * 6.0, 0.0, 2.5, 7.5, 8.5, "white"))
    parts.append((sx / 2 - 5.0, 0.0, 6.0, 12.0, 8.0, "metal"))
    return _board_z(size, parts)


def buck(size):
    """AP64500 5 V 5 A buck board: the inductor block, input/output capacitors, terminals."""
    sx, sy, sz = size
    parts = [(0.0, 2.0, 10.0, 10.0, 5.0, "dark"), (-7.0, -9.0, 6.3, 6.3, 7.5, "metal"),
             (7.0, -9.0, 6.3, 6.3, 7.5, "metal"), (-sx / 2 + 3.0, 10.0, 4.0, 8.0, 8.0, "white"),
             (sx / 2 - 3.0, 10.0, 4.0, 8.0, 8.0, "white"), (5.0, 9.0, 3.0, 3.0, 1.0, "dark")]
    return _board_z(size, parts)


def ina219(size):
    sx, sy, sz = size
    parts = [(0.0, -3.0, 3.0, 3.0, 1.0, "dark"), (6.0, 4.0, 6.4, 3.2, 0.8, "dark"),
             (-sx / 2 + 4.0, 3.0, 7.5, 10.0, 8.5, "white"), (0.0, sy / 2 - 2.5, 15.0, 2.5, 8.5, "dark")]
    return _board_z(size, parts)


def small_board(chip, header=True):
    def build(size):
        sx, sy, sz = size
        parts = [(0.0, 0.0, chip[0], chip[1], chip[2], "dark")]
        if header:
            parts.append((0.0, -sy / 2 + 1.5, min(sx - 2, 15.0), 2.5, 0.3, "metal"))
        return _board_z(size, parts)
    return build


def tof(size):
    """VL53L0X blue board: the sensor at one end facing out (+z), the 6-pin header pads at the other."""
    sx, sy, sz = size
    return _board_z(size, [(-sx / 2 + 5.0, 0.0, 4.4, 2.4, 1.0, "dark"), (2.0, 0.0, 3.0, 3.0, 1.0, "dark"),
                           (sx / 2 - 2.0, 0.0, 2.5, min(sy - 1, 15.0), 0.3, "metal")])


def tof8(size):
    """VL53L5CX 8x8 board standing up (thickness along x, facing +x)."""
    sx, sy, sz = size
    out = [("pcb", _box((-sx / 2 + PCB_T / 2, 0, 0), (PCB_T, sy, sz)), "pcb"),
           ("sensor", _box((-sx / 2 + PCB_T + 1.2, 0, 1.0), (2.4, 6.4, 3.0)), "dark")]
    return out


def camera(size):
    """OV5647 board standing up (thickness along x), the sensor/lens holder in front: the lens barrel is its own part."""
    sx, sy, sz = size
    return [("pcb", _box((0, 0, 0), (sx, sy, sz)), "pcb")]


def camera_lens(size):
    sx, sy, sz = size
    holder = min(sy, sz)
    return [("holder", _box((-sx / 2 + 3.0, 0, 0), (6.0, holder, holder)), "dark"),
            ("barrel", _xcyl((sx / 2 - (sx - 6.0) / 2, 0, 0), min(holder, 14.0) / 2 - 0.5, sx - 6.0), "dark"),
            ("glass", _xcyl((sx / 2 - 0.2, 0, 0), min(holder, 14.0) / 2 - 2.5, 0.4), "metal")]


def speaker(size):
    """GSPK2805TN: a 28 mm round frame 5 mm deep with its magnet behind."""
    sx, sy, sz = size
    r = min(sx, sy) / 2
    return [("frame", _zcyl((0, 0, sz / 2 - 0.75), r, 1.5), "dark"),
            ("cone", _zcyl((0, 0, sz / 2 - 1.75), r - 2.0, 0.5), "dark"),
            ("magnet", _zcyl((0, 0, -sz / 2 + 1.75), r * 0.55, 3.5), "metal")]


def mic(size):
    sx, sy, sz = size
    return _board_z(size, [(-sx / 2 + 4.0, 0.0, 4.0, 3.0, 1.1, "metal"), (sx / 2 - 2.0, 0.0, 2.5, 10.0, 0.3, "metal")])


def switch(size):
    """ASW-15D rocker: the bezel at the outside face (+x is out through the rear wall), the body and lugs inside."""
    sx, sy, sz = size
    return [("bezel", _box((sx / 2 - 1.5, 0, 0), (3.0, sy, sz)), "dark"),
            ("rocker", _box((sx / 2 - 0.5, 0, 0), (1.0, sy - 6, sz - 6)), "dark"),
            ("body", _box((-1.5, 0, 0), (sx - 6.0, sy - 6.0, sz - 4.0)), "dark"),
            ("lugs", _box((-sx / 2 + 1.5, 0, 0), (3.0, sy - 12.0, 6.0)), "metal")]


def xt60(size):
    sx, sy, sz = size
    return [("body", _box((0, 0, 0), (sx - 6.0, sy, sz)), "battery"),
            ("lead", _xcyl((sx / 2 - 3.0, 0, 0), min(sy, sz) / 2 - 1.0, 6.0), "dark")]


def fuse(size):
    sx, sy, sz = size
    return [("holder", _box((0, 0, -2.0), (sx, sy, sz - 4.0)), "dark"),
            ("blade", _box((0, 0, sz / 2 - 3.0), (19.1, 5.1, 6.0)), "battery")]


def pogo(size):
    sx, sy, sz = size
    out = [("strip", _box((0, 0, 0), (sx, sy, sz)), "dark")]
    for k in range(2):
        out.append((f"pin{k}", _zcyl((-sx / 4 + k * sx / 2, 0, sz / 2 + 0.5), 1.0, 1.0), "metal"))
    return out


@cache
def max98357():
    """Adafruit's official STEP of the MAX98357A breakout (Adafruit_CAD_Parts, MIT), centred: [(suffix, solid, mat)]."""
    s = import_step(str(VENDOR / "adafruit" / "3006_MAX98357.step"))
    bb = s.bounding_box()
    c = bb.center()
    out = []
    for k, sol in enumerate(s.solids()):
        b = sol.bounding_box()
        mat = "pcb" if b.size.Z <= PCB_T + 0.1 and b.size.X * b.size.Y > 100 else "dark"
        out.append((f"s{k}", Pos(-c.X, -c.Y, -c.Z) * sol, mat))
    return out


VENDOR_STEP = {              # part -> (the maker's STEP, unpacked by cad/vendor/fetch.py)
    "radxa": "radxa/x/radxa.step", "driver": "waveshare/x/waveshare.step",
    "mic_l": "dfrobot/x/dfrobot.step", "mic_r": "dfrobot/x/dfrobot.step",
}


@cache
def _vendor(rel):
    """A maker's STEP, centred on its bounding box: [(suffix, solid, material)] (the board's flat body green, the
    rest dark), or None when it has not been fetched."""
    f = VENDOR / rel
    if not f.exists():
        return None
    s = import_step(str(f))
    c = s.bounding_box().center()
    out = []
    for k, sol in enumerate(s.solids()):
        b = sol.bounding_box()
        if sol.volume < 0.5:
            continue
        flat = b.size.Z <= 2.0 and b.size.X * b.size.Y > 60.0
        out.append((f"s{k}", Pos(-c.X, -c.Y, -c.Z) * sol, "pcb" if flat or sol.volume > 2000 else "dark"))
    return out


def amp(size):
    """The STEP is 17.8 (x) x 19.1 (y); the layout box is 19.4 x 17.8: turned 90 deg about z to fit."""
    return [(n, Rot(0, 0, 90) * s, m) for n, s, m in max98357()]


BUILDERS = {
    "pack": battery, "radxa": radxa, "radxa_heatsink": heatsink, "driver": driver, "hub": hub,
    "reg_a": buck, "reg_b": buck, "ina": ina219, "imu": small_board((5.2, 3.8, 1.1)), "mpr121": small_board((3.0, 3.0, 1.0)),
    "tof_fl": tof, "tof_fr": tof, "tof_rear": tof, "tof_left": tof, "tof_right": tof, "tof8": tof8,
    "camera": camera, "camera_lens": camera_lens, "speaker": speaker, "mic_l": mic, "mic_r": mic,
    "switch": switch, "xt60": xt60, "fuse": fuse, "dock_board": small_board((3.0, 3.0, 1.0), header=False),
    "pogo_l": pogo, "pogo_r": pogo, "amp": amp,
}


def shapes(name, size):
    """[(suffix, solid, material)] for a layout part: the maker's 3D model where it is fetched, else a shape drawn
    from its data, or None (a plain box is used)."""
    if name in VENDOR_STEP:
        v = _vendor(VENDOR_STEP[name])
        if v:
            return v
    b = BUILDERS.get(name)
    return b(tuple(float(v) for v in size)) if b else None
