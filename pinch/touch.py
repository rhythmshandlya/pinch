"""Touch: what kind of contact was that?

Inputs (all on the robot already):
  * MPR121 capacitive zones behind the shell (copper tape): head, back_left, back_right, side_left,
    side_right, face (the bezel below the screen), claw_left, claw_right
  * IMU: being lifted, shaken, bumped, tipped over
  * leg servo load (STS3032 reports it): someone holding or pushing a leg

Gestures: tap, double_tap, hold, stroke (head -> back = petting, back -> head = ruffled), boop (face
tap), tickle (fast alternating side touches), claw_hold (a "handshake" on a claw pad), plus lifted,
shaken, bumped, upside_down, held_leg.
"""

from collections import deque

import numpy as np

ZONES = ["head", "back_left", "back_right", "side_left", "side_right", "face", "claw_left", "claw_right"]
ORDER = {"face": 0, "head": 1, "back_left": 2, "back_right": 2}        # front -> back, for strokes


class TouchGestures:
    def __init__(self):
        self.down = {}                        # zone -> time it was touched
        self.history = deque(maxlen=24)       # (t, zone, "on"/"off")
        self.last_tap = {}
        self.t = 0.0

    def update(self, t, touched: set):
        """touched: zones currently touched. Returns gesture names that just completed."""
        self.t = t
        out = []
        for z in touched - set(self.down):
            self.down[z] = t
            self.history.append((t, z, "on"))
        for z in set(self.down) - touched:
            t0 = self.down.pop(z)
            self.history.append((t, z, "off"))
            dur = t - t0
            if dur < 0.35:
                if z == "face":
                    out.append("boop")
                elif t - self.last_tap.get(z, -9) < 0.6:
                    out.append("double_tap")
                else:
                    out.append("tap")
                self.last_tap[z] = t
            stroke = self._stroke(t)
            if stroke:
                out.append(stroke)
        for z, t0 in self.down.items():
            if t - t0 > 1.2 and (z, t0) not in getattr(self, "_held", set()):
                self._held = getattr(self, "_held", set()) | {(z, t0)}
                out.append("claw_hold" if z.startswith("claw") else "hold")
        if self._tickle(t):
            out.append("tickle")
        return out

    def _stroke(self, t):
        ons = [(tt, z) for tt, z, kind in self.history if kind == "on" and t - tt < 1.5 and z in ORDER]
        if len(ons) < 2:
            return None
        seq = [ORDER[z] for _, z in sorted(ons)]
        if seq[-1] - seq[0] >= 1 and all(b >= a for a, b in zip(seq, seq[1:])):
            self.history.clear()
            return "stroke"              # head -> back: petting the right way
        if seq[0] - seq[-1] >= 1 and all(b <= a for a, b in zip(seq, seq[1:])):
            self.history.clear()
            return "ruffle"
        return None

    def _tickle(self, t):
        ons = [z for tt, z, kind in self.history if kind == "on" and t - tt < 1.2 and z.startswith("side")]
        flips = sum(1 for a, b in zip(ons, ons[1:]) if a != b)
        if flips >= 3:
            self.history.clear()
            return True
        return False


class BodySense:
    """IMU + leg loads -> being handled. Call every control tick."""

    def __init__(self):
        self.gyro_hist = deque(maxlen=25)
        self.lifted = False
        self.load_dev_s = np.zeros(4)
        self.t = 0.0

    def update(self, dt, gyro, gravity, accel_norm=None, leg_loads=None, standing_still=True, feet_loaded=None):
        """gravity: unit vector in the body frame (0,0,-1 level). accel_norm: |a|/g if the IMU gives it.
        feet_loaded: how many feet carry weight (from leg loads) - 0 means lifted."""
        self.t += dt
        out = []
        w = float(np.linalg.norm(gyro))
        self.gyro_hist.append(w)
        if len(self.gyro_hist) == self.gyro_hist.maxlen and np.mean(self.gyro_hist) > 3.0:
            out.append("shaken")
            self.gyro_hist.clear()
        if gravity[2] > 0.5:
            out.append("upside_down")
        if accel_norm is not None and accel_norm > 2.2:
            out.append("bumped")
        if feet_loaded is not None:
            lifted = feet_loaded == 0
            if lifted and not self.lifted:
                out.append("lifted")
            if not lifted and self.lifted:
                out.append("put_down")
            self.lifted = lifted
        if leg_loads is not None and standing_still:
            dev = np.abs(np.asarray(leg_loads)) > 0.35
            self.load_dev_s = np.where(dev, self.load_dev_s + dt, 0.0)
            if np.any(self.load_dev_s > 0.6):
                out.append("held_leg")
                self.load_dev_s[:] = -5.0               # don't repeat for a while
        return out


GESTURE_EVENT = {                   # gesture -> affect event (pinch/affect.py)
    "tap": "tapped", "double_tap": "tapped", "hold": "petted", "stroke": "stroked", "ruffle": "tapped",
    "boop": "booped", "tickle": "tickled", "claw_hold": "petted", "lifted": "lifted", "shaken": "shaken",
    "bumped": "bumped", "upside_down": "fell", "held_leg": "held_leg",
}
