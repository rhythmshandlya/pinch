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


class StuckSense:
    """Is Pinch stuck? While standing: tilted on what should be a flat desk, or a foot carrying no weight
    (hanging over an edge, or on top of something). While walking: commands but no progress. Debounced;
    emits "stuck" once, then "unstuck" when it has cleared for a while. Inputs exist on the robot too: IMU
    gravity, feet carrying load (servo loads), the walking command, and a *measured* desk position: the
    camera's marker fixes and ground flow (pinch_main.Localizer.measured_xy), never the position integrated
    from the walking command - that one keeps moving while the crab is stuck, which is the very thing to catch.
    No measurement in the window = no judgement on progress (the tilt / feet checks still work)."""

    def __init__(self, tilt_deg=7.0, bad_s=1.5, ok_s=2.0):     # belly on an edge tilts 7.7-11 deg; trotting <= 4.8
        self.tilt_deg, self.need_bad, self.need_ok = tilt_deg, bad_s, ok_s
        self.stuck, self.why = False, ""
        self.bad_s = self.ok_s = 0.0
        self.hist = deque(maxlen=400)
        self.t = 0.0

    def update(self, dt, gravity, feet_loaded, pose, cmd_speed, lifted=False, standing=True, walking=False,
               feet_margin=None):
        """feet_margin: how far the feet are inside the desk (m, from the desk position and the leg layout;
        negative = a foot is over the edge) - known at once, whatever Pinch is doing."""
        self.t += dt
        mx, my = (None, None) if pose is None else (pose[0], pose[1])
        self.hist.append((self.t, mx, my, cmd_speed if walking else 0.0))
        why = ""
        if not lifted and feet_margin is not None and feet_margin < -0.003:
            why = "a foot over the edge"
        elif not lifted and standing:
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
            if tilt > self.tilt_deg:
                why = f"tilted {tilt:.0f} deg"
            elif feet_loaded is not None and 0 < feet_loaded <= 2:
                why = f"only {feet_loaded} feet down"
        if not lifted and walking:
            win = [r for r in self.hist if r[0] > self.t - 3.0]
            meas = [r for r in win if r[1] is not None]
            if win and win[-1][0] - win[0][0] > 2.5 and np.mean([r[3] for r in win]) > 0.012 \
                    and len(meas) >= 4 and meas[-1][0] - meas[0][0] > 2.0:
                k = max(1, len(meas) // 3)                    # mean of the first and last third: robust to jitter
                a = np.mean([(r[1], r[2]) for r in meas[:k]], axis=0)
                b = np.mean([(r[1], r[2]) for r in meas[-k:]], axis=0)
                if np.hypot(*(b - a)) < 0.008:
                    why = "walking but not moving"
        out = []
        if why:
            self.bad_s += dt
            self.ok_s = 0.0
            if not self.stuck and self.bad_s >= self.need_bad:
                self.stuck, self.why = True, why
                out.append("stuck")
        else:
            self.ok_s += dt
            self.bad_s = 0.0
            if self.stuck and self.ok_s >= self.need_ok:
                self.stuck = False
                out.append("unstuck")
        return out


class CliffReflex:
    """The two cliff sensors under the front corners -> the edge reflex (shared by the firmware and the twin).

      * either beam sees no desk (max of the two, not min: one corner over the edge is enough) while the
        body is moving at all (its walking command, not the task's name) -> stop and back off, once:
        the reflex latches until both beams see desk again, so its own backing-off can't re-trigger it;
      * a beam that is missing, failing or stale is "don't know", which means no walking forward (the
        direction the sensors guard) - never "cliff", which would back off forever.
    Forward walking also stays blocked while latched. Body reads `forward_ok` through its edge guard."""

    BACK = ("walk", -0.03, 0.0, 0.0, 1.0)

    def __init__(self, cliff_mm=110, stale_s=0.3, clear_s=0.2):
        self.cliff_mm, self.stale_s, self.clear_s = cliff_mm, stale_s, clear_s
        self.state = "unknown"              # ok | cliff | unknown
        self.latched = False
        self.clear_since = None
        self.stops = 0

    @property
    def forward_ok(self):
        return self.state == "ok" and not self.latched

    def update(self, t, readings_mm, age_s, moving):
        """readings_mm: the two beams (mm; None = no answer) or None if there are no sensors; age_s: how
        old they are. Returns the motion to start ("walk" backwards) or None."""
        if readings_mm is None or age_s is None or age_s > self.stale_s or any(r is None for r in readings_mm):
            self.state = "unknown"
        else:
            self.state = "cliff" if max(readings_mm) > self.cliff_mm else "ok"
        if self.latched:
            if self.state == "ok":
                self.clear_since = t if self.clear_since is None else self.clear_since
                if t - self.clear_since >= self.clear_s:
                    self.latched = False
            else:
                self.clear_since = None
            return None
        if self.state == "cliff" and moving:
            self.latched, self.clear_since = True, None
            self.stops += 1
            return self.BACK
        return None


class BodySense:
    """IMU + leg loads -> being handled. Call every control tick."""

    def __init__(self):
        self.gyro_hist = deque(maxlen=25)
        self.lifted = False
        self.air_s = self.ground_s = 0.0
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
            # debounced: mid-stride a trotting crab can have no foot down for an instant
            self.air_s = self.air_s + dt if feet_loaded == 0 else 0.0
            self.ground_s = self.ground_s + dt if feet_loaded >= 3 else 0.0
            if not self.lifted and self.air_s >= 0.3:
                out.append("lifted")
                self.lifted = True
            elif self.lifted and self.ground_s >= 0.3:
                out.append("put_down")
                self.lifted = False
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
