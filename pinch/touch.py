"""Touch: what kind of contact was that?

Inputs (all on the robot already):
  * MPR121 capacitive zones behind the shell (copper tape): head, back_left, back_right, side_left,
    side_right, face (the bezel below the screen), claw_left, claw_right
  * IMU: being lifted, shaken, bumped, tipped over
  * leg servo load (STS3032 reports it): someone holding or pushing a leg

Gestures: tap, double_tap, hold, stroke (head -> back = petting, back -> head = ruffled), boop (face
tap), tickle (fast alternating side touches), claw_hold (a "handshake" on a claw pad), plus lifted,
shaken, bumped, upside_down, held_leg.

Pad gestures come out as Touch objects: the gesture name (so the mind takes them unchanged) plus the pads it
involved; touch_place() turns those into a side, a region and a direction for the reaction (ReactTouch).
TouchHabituation makes repeated touch wear off and recover; WakeRules decide which touches wake it from sleep.
"""

import math
from collections import deque

import numpy as np

ZONES = ["head", "back_left", "back_right", "side_left", "side_right", "face", "claw_left", "claw_right"]
ORDER = {"face": 0, "head": 1, "back_left": 2, "back_right": 2}        # front -> back, for strokes
# where each pad is: side (+1 left, -1 right, 0 middle), region, and its direction in the body frame (rad, 0 ahead,
# + to the left)
ZONE_INFO = {
    "face": (0, "front", 0.0), "head": (0, "top", 0.0),
    "back_left": (1, "back", 2.6), "back_right": (-1, "back", -2.6),
    "side_left": (1, "side", 1.57), "side_right": (-1, "side", -1.57),
    "claw_left": (1, "claw", 0.5), "claw_right": (-1, "claw", -0.5),
}
REGION_RANK = ("claw", "side", "back", "front", "top")       # a gesture over several pads: the most telling one


class Touch(str):
    """A gesture that also says where it was: compares, hashes and prints as its name ("stroke"), so everything
    that takes a gesture name takes it unchanged, and carries the pads it involved. A plain string is a touch
    whose place is unknown (the twin's buttons, the HTTP API)."""

    zones: tuple = ()

    def __new__(cls, name, zones=()):
        s = super().__new__(cls, name)
        s.zones = tuple(z for z in zones if z in ZONE_INFO)
        return s

    def __reduce__(self):                       # (pickles and copies keep the pads)
        return (Touch, (str(self), self.zones))


def touch_place(gesture):
    """(side, region, direction) of a gesture: side +1 left / -1 right / 0 middle or unknown; region one of
    claw, side, back, front, top, or "" if unknown; direction in the body frame (rad, + left) or None."""
    zones = getattr(gesture, "zones", ())
    if not zones:
        return 0, "", None
    info = [ZONE_INFO[z] for z in zones]
    s = sum(i[0] for i in info)
    side = 0 if s == 0 else (1 if s > 0 else -1)
    region = min((i[1] for i in info), key=REGION_RANK.index)
    dirs = [i[2] for i in info if i[1] == region]
    d = math.atan2(sum(math.sin(a) for a in dirs), sum(math.cos(a) for a in dirs))
    return side, region, d


class TouchHabituation:
    """Repeated touch wears off, rests back, and a rough touch sharpens everything again. After the behavioural
    characteristics of habituation (Thompson & Spencer 1966, revised by Rankin et al. 2009) in the discrete form of
    Stanley's model as used on robots (Marsland, Nehmzow & Shapiro 2000):

      * each touch on a channel: h <- h + alpha (1 - h)  - an exponential decrement toward an asymptote (the
        response never drops below FLOOR)
      * between touches: h <- h exp(-dt / TAU_RECOVER)   - spontaneous recovery, several times slower than the
        decrement (more frequent touches therefore habituate more: less time to recover)
      * weaker stimuli habituate faster (a tap's alpha > a stroke's > a tickle's)
      * an intense touch (tickle, ruffle, being lifted or shaken) dishabituates every channel (h halves) and adds a
        sensitisation term that fades in ~20 s (the dual-process view, Groves & Thompson 1970)
    response = FLOOR + (1 - FLOOR)(1 - h) + sensitisation, at most 1: 1 = full reaction, FLOOR = barely."""

    ALPHA = {"pet": 0.25, "tap": 0.35, "tickle": 0.15, "claw": 0.25, "ruffle": 0.2}
    CHANNEL = {"stroke": "pet", "hold": "pet", "tap": "tap", "double_tap": "tap", "boop": "tap",
               "tickle": "tickle", "claw_hold": "claw", "ruffle": "ruffle"}
    INTENSE = {"tickle", "ruffle", "lifted", "shaken", "upside_down", "held_leg"}
    TAU_RECOVER, TAU_SENSITISE = 60.0, 20.0
    FLOOR, SENSITISE, DISHABITUATE = 0.15, 0.3, 0.5

    def __init__(self):
        self.h = {c: 0.0 for c in self.ALPHA}
        self.s = 0.0
        self.t = None
        self.count = {c: 0 for c in self.ALPHA}

    def _relax(self, t):
        if self.t is not None and t > self.t:
            k = math.exp(-(t - self.t) / self.TAU_RECOVER)
            self.h = {c: v * k for c, v in self.h.items()}
            self.s *= math.exp(-(t - self.t) / self.TAU_SENSITISE)
        self.t = t if self.t is None else max(self.t, t)

    def response(self, gesture, t):
        """How strongly it would react to this gesture now (no change)."""
        save = (dict(self.h), self.s, self.t)
        self._relax(t)
        c = self.CHANNEL.get(str(gesture))
        r = 1.0 if c is None else min(1.0, self.FLOOR + (1 - self.FLOOR) * (1 - self.h[c]) + self.s)
        self.h, self.s, self.t = save
        return r

    def stimulate(self, gesture, t):
        """A touch arrives: returns the response strength to it (0..1), then habituates / sensitises."""
        r = self.response(gesture, t)
        self._relax(t)
        g = str(gesture)
        c = self.CHANNEL.get(g)
        if c is not None:
            self.h[c] += self.ALPHA[c] * (1 - self.h[c])
            self.count[c] += 1
        if g in self.INTENSE:
            self.h = {k: v * (1 - self.DISHABITUATE) if k != c else v for k, v in self.h.items()}
            self.s = min(0.6, self.s + self.SENSITISE)
        return r


# Asleep (resting with its eyes shut), which touches wake it. "always": wakes at once. Others add wake pressure
# (fading with WAKE_TAU s); it wakes when the pressure reaches 1. A stroke while asleep soothes (lowers it).
# Chosen by the owner's setting memory.data["settings"]["wake_on_touch"]; "any" is the old behaviour.
WAKE_RULES = {
    "any": dict(always="*", weights={}),
    "gentle": dict(always={"lifted", "shaken", "upside_down", "held_leg", "tickle", "boop"},
                   weights={"tap": 0.4, "double_tap": 0.7, "claw_hold": 0.5, "hold": 0.25, "ruffle": 0.6,
                            "put_down": 1.0, "stroke": -0.2}),
    "deep": dict(always={"lifted", "shaken", "upside_down"},
                 weights={"tap": 0.2, "double_tap": 0.35, "claw_hold": 0.25, "boop": 0.4, "tickle": 0.5,
                          "ruffle": 0.3, "held_leg": 0.5, "put_down": 1.0, "hold": 0.1, "stroke": -0.2}),
}
WAKE_DEFAULT, WAKE_TAU = "gentle", 30.0


class WakeRules:
    """Decides, once per touch, whether a touch while asleep wakes Pinch."""

    def __init__(self):
        self.pressure, self.t = 0.0, None
        self.decided = {}                        # touch event -> bool (the mind asks again every 0.5 s)

    def wakes(self, event, mode=WAKE_DEFAULT):
        if event in self.decided:
            return self.decided[event]
        gesture, t = event
        rule = WAKE_RULES.get(mode, WAKE_RULES[WAKE_DEFAULT])
        if self.t is not None:
            self.pressure *= math.exp(-max(0.0, t - self.t) / WAKE_TAU)
        self.t = t
        g = str(gesture)
        if rule["always"] == "*" or g in rule["always"]:
            woke = True
        else:
            self.pressure = max(0.0, self.pressure + rule["weights"].get(g, 0.0))
            woke = self.pressure >= 1.0
        if woke:
            self.pressure = 0.0
        self.decided = {event: woke}
        return woke


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
                    out.append(Touch("boop", [z]))
                elif t - self.last_tap.get(z, -9) < 0.6:
                    out.append(Touch("double_tap", [z]))
                else:
                    out.append(Touch("tap", [z]))
                self.last_tap[z] = t
            stroke = self._stroke(t)
            if stroke:
                out.append(stroke)
        for z, t0 in self.down.items():
            if t - t0 > 1.2 and (z, t0) not in getattr(self, "_held", set()):
                self._held = getattr(self, "_held", set()) | {(z, t0)}
                out.append(Touch("claw_hold" if z.startswith("claw") else "hold", [z]))
        tickle = self._tickle(t)
        if tickle:
            out.append(tickle)
        return out

    def _stroke(self, t):
        ons = [(tt, z) for tt, z, kind in self.history if kind == "on" and t - tt < 1.5 and z in ORDER]
        if len(ons) < 2:
            return None
        seq = [ORDER[z] for _, z in sorted(ons)]
        zones = sorted({z for _, z in ons})
        if seq[-1] - seq[0] >= 1 and all(b >= a for a, b in zip(seq, seq[1:])):
            self.history.clear()
            return Touch("stroke", zones)          # head -> back: petting the right way
        if seq[0] - seq[-1] >= 1 and all(b <= a for a, b in zip(seq, seq[1:])):
            self.history.clear()
            return Touch("ruffle", zones)
        return None

    def _tickle(self, t):
        ons = [z for tt, z, kind in self.history if kind == "on" and t - tt < 1.2 and z.startswith("side")]
        flips = sum(1 for a, b in zip(ons, ons[1:]) if a != b)
        if flips >= 3:
            self.history.clear()
            return Touch("tickle", sorted(set(ons)))
        return None


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


STAND_BODY_Z = 0.0487          # m: body origin above the desk when standing (the twin's stand keyframe)


def cliff_beams(beams=None, body_z=STAND_BODY_Z, margin=1.6):
    """The cliff ring from crab/params.py CLIFF_BEAMS: [(name, guard, nominal_mm, cliff_mm)].
    guard: unit vector (body frame, x forward, y left) of the horizontal direction the beam looks - the way
    of walking it vouches for. nominal_mm: what it reads standing on a flat desk; cliff_mm: margin x that,
    beyond which there's no desk under it (front/rear look 40 deg down from low on the wall: ~51 mm -> 82;
    the sides look 30 deg down from low on the wall: ~65 mm -> 104)."""
    if beams is None:
        from crab.params import CLIFF_BEAMS as beams
    out = []
    for name, _x, _y, z_mm, d in beams:
        d = np.asarray(d, float) / np.linalg.norm(d)
        g = d[:2] / np.linalg.norm(d[:2])
        nominal = (body_z + z_mm / 1000.0) / -d[2] * 1000.0
        out.append((name, (float(g[0]), float(g[1])), float(nominal), float(margin * nominal)))
    return out


def twin_sensor(name):
    """The twin's rangefinder for a beam (sim/desk.py keeps the old names for the front pair)."""
    return "cliff_" + {"front_left": "left", "front_right": "right"}.get(name, name)


def limit_velocity(v, blocked):
    """The closest velocity to v (vx, vy) that doesn't move toward any blocked direction (unit vectors):
    v . d <= 0 for every d. Each constraint is a half-plane through the origin, so the answer is v itself,
    v slid along one boundary, or standing still."""
    v = np.asarray(v, float)
    ok = lambda u: all(u @ d <= 1e-9 for d in blocked)          # noqa: E731
    if ok(v):
        return v
    best = np.zeros(2)
    for d in blocked:
        u = v - (v @ d) * np.asarray(d)
        if ok(u) and u @ u > best @ best:
            best = u
    return best


class CliffReflex:
    """The cliff ring (five VL53L0X: front pair, rear, both sides) -> the edge reflex, shared by the firmware
    and the twin. Each beam guards the direction it looks in:

      * a beam that sees no desk (beyond its own threshold) while Pinch walks toward its side (from the
        walking command, whatever the task) -> stop and back off away from it, once: the reflex latches
        until the beams that fired see desk again, so its own backing-off can't re-trigger it;
      * a beam that sees no desk, or is missing, failing or stale ("don't know"), blocks walking toward its
        side - never "cliff" for a sensor that's just broken, which would back off forever.
    `blocked` (unit vectors) goes to Body.cliff_blocked; its edge guard removes any motion toward them."""

    SPEED, BACK_S = 0.03, 1.0

    def __init__(self, beams=None, stale_s=0.3, clear_s=0.2):
        self.beams = cliff_beams() if beams is None else beams
        self.stale_s, self.clear_s = stale_s, clear_s
        self.states = {b[0]: "unknown" for b in self.beams}      # ok | cliff | unknown
        self.latched = set()                                     # beams whose back-off is still in effect
        self.clear_since = None
        self.stops = 0

    @property
    def state(self):
        """One word for the whole ring: cliff > unknown > ok."""
        v = set(self.states.values())
        return "cliff" if "cliff" in v else "unknown" if "unknown" in v else "ok"

    @property
    def blocked(self):
        return [np.array(g) for name, g, _, _ in self.beams if self.states[name] != "ok" or name in self.latched]

    @property
    def forward_ok(self):
        return not any(g[0] > 0.5 for g in self.blocked)

    def update(self, t, readings_mm, ages_s, cmd_vel):
        """readings_mm: {beam: mm or None (no answer)}, or None if there are no sensors; ages_s: {beam: s} or
        one age for all; cmd_vel: the walking command (vx, vy, wz). Returns a walk to start, or None."""
        readings_mm = readings_mm or {}
        for name, _, _, cliff in self.beams:
            r = readings_mm.get(name)
            age = ages_s.get(name) if isinstance(ages_s, dict) else ages_s
            if r is None or age is None or age > self.stale_s:
                self.states[name] = "unknown"
            else:
                self.states[name] = "cliff" if r > cliff else "ok"
        if self.latched:
            if all(self.states[n] == "ok" for n in self.latched):
                self.clear_since = t if self.clear_since is None else self.clear_since
                if t - self.clear_since >= self.clear_s:
                    self.latched = set()
            else:
                self.clear_since = None
            return None
        v = np.asarray(cmd_vel[:2], float)
        fired = [(n, np.array(g)) for n, g, _, _ in self.beams if self.states[n] == "cliff" and v @ g > 1e-4]
        if not fired:
            return None
        self.latched, self.clear_since = {n for n, _ in fired}, None
        self.stops += 1
        away = -sum(g for _, g in fired)
        away = away / (np.linalg.norm(away) + 1e-9) * self.SPEED
        return ("walk", float(away[0]), float(away[1]), 0.0, self.BACK_S)


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
