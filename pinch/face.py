"""Pinch's face: animated eyes on the 2.42" 128x64 OLED (SSD1309).

The same frames go to the real screen (robot/drivers/ssd1309.py), the laptop app and the 3D digital
twin's face texture. Eyes are Vector-style rounded rectangles: emotion comes from their size, eyelids
(slanted for angry / sad, a curved lower lid for happy), gaze, blinking, and a few glyph eyes
(hearts, X, spirals, sleepy lines, stars, squeezed > <) plus little overlays (zzz, !, ?, ...).

    face = Face()
    img = face.update(dt, expression="happy", gaze=(0.3, 0.0))     # PIL "1" image, 128 x 64
    face.ssd1309_pages()                                           # 1024 bytes for the screen
"""

from dataclasses import dataclass, field, replace

import numpy as np
from PIL import Image, ImageDraw

W, H = 128, 64
EYE_CX = (36, 92)                  # eye centres (x)
EYE_CY = 32
EYE_W, EYE_H, EYE_R = 32, 36, 9    # neutral eye size and corner radius


@dataclass
class Eyes:
    """Target look for both eyes. Numbers are blended smoothly toward each new target."""
    open: float = 1.0          # 0 closed .. 1 normal .. 1.2 wide
    w: float = 1.0             # width scale
    h: float = 1.0             # height scale
    lid_angle: float = 0.0     # deg; + = inner corners down (angry), - = outer corners down (sad)
    lid_top: float = 0.0       # 0..1 of the eye covered from the top
    lid_bottom: float = 0.0    # 0..1 covered from the bottom by an upward arc (happy cheeks)
    left_scale: float = 1.0    # asymmetry (curious: one eye bigger)
    right_scale: float = 1.0
    y: float = 0.0             # whole-face vertical offset in pixels (bounce, droop)
    glyph: str = ""            # "" | heart | x | spiral | closed | star | squeeze | arc
    overlay: str = ""          # "" | zzz | ! | ? | ... | shh | note | tear | sweat | anger


EXPRESSIONS = {
    "neutral":  Eyes(),
    "happy":    Eyes(lid_bottom=0.38, h=0.95),
    "joy":      Eyes(glyph="arc", y=-2),
    "excited":  Eyes(open=1.15, w=1.08, h=1.08, lid_bottom=0.25, y=-2),
    "love":     Eyes(glyph="heart"),
    "content":  Eyes(lid_top=0.30, lid_bottom=0.30),
    "curious":  Eyes(left_scale=1.12, right_scale=0.85, lid_top=0.08, overlay="?"),
    "surprised": Eyes(open=1.2, w=0.85, h=1.15, overlay="!"),
    "sad":      Eyes(lid_angle=-22, lid_top=0.30, h=0.9, y=4, overlay="tear"),
    "lonely":   Eyes(lid_angle=-14, lid_top=0.22, h=0.92, y=3),
    "angry":    Eyes(lid_angle=24, lid_top=0.34, w=1.05, overlay="anger"),
    "grumpy":   Eyes(lid_angle=12, lid_top=0.42, lid_bottom=0.15),
    "scared":   Eyes(open=1.05, w=0.75, h=0.85, lid_angle=-10, overlay="sweat"),
    "shy":      Eyes(lid_bottom=0.30, lid_top=0.15, w=0.9, y=3),
    "sleepy":   Eyes(lid_top=0.55, h=0.9, y=2),
    "asleep":   Eyes(glyph="closed", y=3, overlay="zzz"),
    "bored":    Eyes(lid_top=0.48, h=0.85),
    "focused":  Eyes(lid_top=0.25, lid_bottom=0.20, w=1.05),
    "quiet":    Eyes(lid_top=0.30, lid_bottom=0.25, h=0.8, overlay="shh"),
    "dizzy":    Eyes(glyph="spiral"),
    "dead":     Eyes(glyph="x"),
    "proud":    Eyes(glyph="star"),
    "squeeze":  Eyes(glyph="squeeze"),
    "thinking": Eyes(lid_top=0.2, overlay="..."),
    "music":    Eyes(glyph="arc", overlay="note"),
}


def _mix(a, b, k):
    return a + (b - a) * k


class Face:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.cur = Eyes()
        self.target = Eyes()
        self.expression = "neutral"
        self.gaze = np.zeros(2)                # current, -1..1
        self.gaze_target = np.zeros(2)
        self.t = 0.0
        self.next_blink = 2.0
        self.blink_t = -1.0                    # time since a blink started (<0: not blinking)
        self.double = False
        self.next_saccade = 1.0
        self.img = Image.new("1", (W, H))

    # -- control -------------------------------------------------------------
    def set(self, expression=None, gaze=None):
        if expression is not None and expression in EXPRESSIONS:
            self.expression = expression
            self.target = EXPRESSIONS[expression]
        if gaze is not None:
            self.gaze_target = np.clip(np.asarray(gaze, float), -1, 1)

    def blink(self):
        self.blink_t = 0.0

    # -- animation -----------------------------------------------------------
    def update(self, dt, expression=None, gaze=None):
        self.set(expression, gaze)
        self.t += dt
        k = 1 - np.exp(-dt / 0.08)             # ~80 ms blend toward the target look
        c, tg = self.cur, self.target
        self.cur = Eyes(**{f: (_mix(getattr(c, f), getattr(tg, f), k) if isinstance(getattr(tg, f), float)
                               else getattr(tg, f)) for f in c.__dataclass_fields__})
        # gaze: small saccades around the target, like a living eye
        if self.t > self.next_saccade:
            jitter = self.rng.normal(0, 0.08, 2)
            self.gaze = np.clip(self.gaze_target + jitter, -1, 1)
            self.next_saccade = self.t + self.rng.uniform(0.4, 2.2)
        else:
            self.gaze += (self.gaze_target - self.gaze) * (1 - np.exp(-dt / 0.15)) * 0.3
        # blinking: every 2-6 s, sometimes twice
        if self.blink_t < 0 and self.t > self.next_blink and not self.cur.glyph:
            self.blink_t, self.double = 0.0, self.rng.random() < 0.25
            self.next_blink = self.t + self.rng.uniform(2.0, 6.0)
        blink_open = 1.0
        if self.blink_t >= 0:
            self.blink_t += dt
            dur = 0.14
            ph = self.blink_t / dur
            blink_open = abs(1 - 2 * min(ph, 1.0)) if ph < 1 else 1.0
            if ph >= 1:
                if self.double:
                    self.double, self.blink_t = False, 0.0
                else:
                    self.blink_t = -1.0
        self.img = self.render(self.cur, self.gaze, blink_open, self.t)
        return self.img

    # -- drawing -------------------------------------------------------------
    def render(self, e: Eyes, gaze, blink=1.0, t=0.0):
        img = Image.new("1", (W, H), 0)
        d = ImageDraw.Draw(img)
        gx, gy = gaze[0] * 9, gaze[1] * 6 + e.y
        for side, (cx, s) in enumerate(((EYE_CX[0], e.left_scale), (EYE_CX[1], e.right_scale))):
            x, y = cx + gx, EYE_CY + gy
            if e.glyph:
                self._glyph(d, e.glyph, x, y, side, t)
                continue
            w = EYE_W * e.w * s
            h = EYE_H * e.h * s * max(0.06, min(e.open, 1.25)) * blink
            if h < 3:                                           # closed: a line
                d.line([(x - w / 2, y), (x + w / 2, y)], fill=1, width=2)
                continue
            r = min(EYE_R * s, h / 2, w / 2)
            d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], radius=r, fill=1)
            # upper lid: a slanted black polygon covering lid_top of the eye
            if e.lid_top > 0.01 or abs(e.lid_angle) > 0.5:
                top = y - h / 2 + e.lid_top * h
                inner = 1 if side == 0 else -1                 # inner corner is toward the middle
                slope = np.tan(np.radians(e.lid_angle)) * inner
                y_l, y_r = top - slope * w / 2, top + slope * w / 2
                d.polygon([(x - w / 2 - 2, y - h), (x + w / 2 + 2, y - h), (x + w / 2 + 2, y_r), (x - w / 2 - 2, y_l)],
                          fill=0)
            # lower lid: an upward arc (cheeks pushing up when happy)
            if e.lid_bottom > 0.01:
                rr = w * 0.9
                cy = y + h / 2 + rr - e.lid_bottom * h
                d.ellipse([x - rr, cy - rr, x + rr, cy + rr], fill=0)
            # a glint makes eyes look alive (kept clear of the lid edge)
            visible_top = y - h / 2 + max(e.lid_top * h, 0) + abs(np.tan(np.radians(e.lid_angle))) * w / 2
            if h > 12 and e.lid_top < 0.4 and e.lid_bottom < 0.3:
                gxp, gyp = x - w * 0.22, max(y - h * 0.22, visible_top + 5)
                if gyp < y + h / 2 - 5:
                    d.ellipse([gxp - 2, gyp - 2, gxp + 2, gyp + 2], fill=0)
        self._overlay(d, e.overlay, t)
        return img

    def _glyph(self, d, g, x, y, side, t):
        if g == "heart":
            s = 13 + 1.5 * np.sin(t * 6)                       # beating
            d.ellipse([x - s, y - s * 0.75, x, y + s * 0.15], fill=1)
            d.ellipse([x, y - s * 0.75, x + s, y + s * 0.15], fill=1)
            d.polygon([(x - s, y - s * 0.2), (x + s, y - s * 0.2), (x, y + s)], fill=1)
        elif g == "x":
            for dx in (-1, 1):
                d.line([(x - 11, y - 11 * dx), (x + 11, y + 11 * dx)], fill=1, width=4)
        elif g == "spiral":
            pts = [(x + 1.1 * a * np.cos(a + t * 8 * (1 if side else -1)), y + 1.1 * a * np.sin(a + t * 8))
                   for a in np.linspace(0, 13, 60)]
            d.line(pts, fill=1, width=2)
        elif g == "closed":
            d.arc([x - 14, y - 10, x + 14, y + 6], start=20, end=160, fill=1, width=3)
        elif g == "arc":                                      # ^ ^ happy closed eyes
            d.arc([x - 14, y - 4, x + 14, y + 20], start=200, end=340, fill=1, width=4)
        elif g == "star":
            pts = []
            for i in range(10):
                a = -np.pi / 2 + i * np.pi / 5
                r = 15 if i % 2 == 0 else 6
                pts.append((x + r * np.cos(a), y + r * np.sin(a)))
            d.polygon(pts, fill=1)
        elif g == "squeeze":                                   # > <
            k = 1 if side == 0 else -1
            d.line([(x - 10 * k, y - 10), (x + 8 * k, y), (x - 10 * k, y + 10)], fill=1, width=4)

    def _overlay(self, d, o, t):
        if not o:
            return
        if o == "zzz":
            for i, s in enumerate((6, 8, 10)):
                ph = (t * 0.8 + i * 0.33) % 1.0
                x, y = 104 + i * 6, 14 - i * 5 - ph * 4
                d.line([(x, y), (x + s, y), (x, y + s), (x + s, y + s)], fill=1, width=1)
        elif o == "!":
            d.rectangle([118, 4, 121, 16], fill=1)
            d.rectangle([118, 19, 121, 22], fill=1)
        elif o == "?":
            d.arc([112, 2, 124, 14], start=180, end=90, fill=1, width=2)
            d.line([(118, 14), (118, 18)], fill=1, width=2)
            d.rectangle([117, 21, 119, 23], fill=1)
        elif o == "...":
            for i in range(3):
                if (t * 3) % 3 >= i:
                    d.ellipse([52 + i * 10, 56, 56 + i * 10, 60], fill=1)
        elif o == "shh":                                       # a little claw over a small mouth
            d.line([(58, 56), (70, 56)], fill=1, width=2)
            d.rounded_rectangle([62, 46, 66, 62], radius=2, fill=1)
        elif o == "note":
            y = 8 + 3 * np.sin(t * 4)
            d.ellipse([110, y + 8, 116, y + 13], fill=1)
            d.line([(115, y + 10), (115, y)], fill=1, width=2)
            d.line([(115, y), (121, y + 3)], fill=1, width=2)
        elif o == "tear":
            y = 44 + (t * 20) % 16
            d.ellipse([22, y, 27, y + 6], fill=1)
        elif o == "sweat":
            d.polygon([(117, 6), (114, 14), (120, 14)], fill=1)
            d.ellipse([113, 11, 121, 18], fill=1)
        elif o == "anger":
            for a, b in (((4, 4), (10, 10)), ((10, 4), (4, 10))):
                d.line([a, b], fill=1, width=2)

    # -- output --------------------------------------------------------------
    def ssd1309_pages(self) -> bytes:
        """Frame in the SSD1309/SSD1306 page layout: 8 pages x 128 columns, LSB = top pixel."""
        a = np.array(self.img, dtype=np.uint8).reshape(8, 8, W)        # page, bit row, column
        return bytes((a * (1 << np.arange(8))[None, :, None]).sum(axis=1).astype(np.uint8).ravel())

    def rgb(self, scale=4, on=(235, 245, 255), off=(6, 8, 12)):
        """Upscaled colour image for the laptop app / 3D twin (OLED white on black)."""
        a = np.array(self.img, dtype=bool)
        out = np.where(a[..., None], np.array(on, np.uint8), np.array(off, np.uint8))
        return np.kron(out, np.ones((scale, scale, 1), np.uint8))
