"""What Pinch believes about its little world, and the situations it notices in it.

Desk frame (metres): x to the right, y away from you, z up; the desk top is z = 0. You sit at the
front edge (y < 0). Pinch's own spot and the desk tray are in WorldModel.desk.
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Desk:
    half_x: float = 0.35           # desk top is 0.70 x 0.45 m (sim); the robot learns its own with the cliff sensors
    half_y: float = 0.225
    owner_xy: tuple = (0.0, -0.30)            # where you sit
    tray_xy: tuple = (0.18, 0.08)             # desk tray (pens, small things): Pinch's tidy target
    tray_half: tuple = (0.08, 0.035)          # 16 x 7 cm inside, open toward you; Pinch needs ~7 cm around it
    home_xy: tuple = (-0.05, 0.08)            # Pinch's rest spot, left of the tray
    edge_margin: float = 0.07                 # Pinch keeps its body centre this far from any edge

    def inside(self, xy, margin=0.0):
        return abs(xy[0]) < self.half_x - margin and abs(xy[1]) < self.half_y - margin

    def edge_distance(self, xy):
        return min(self.half_x - abs(xy[0]), self.half_y - abs(xy[1]))

    # Pinch's feet when standing (body frame, m; a little generous): they reach ~10 cm from the body centre,
    # so keeping the centre edge_margin from the edge is not enough on its own (a rear foot ended up over
    # the front edge in a 10-minute twin run, and Pinch stood there, lopsided, without noticing)
    FEET = ((0.047, 0.094), (0.047, -0.094), (-0.047, 0.094), (-0.047, -0.094))   # measured stance (twin)

    def fit_feet(self, xy, heading, margin=0.035):
        """The nearest spot to xy (moving straight toward the desk's middle) where Pinch, facing `heading`, has
        every foot at least `margin` inside the desk - so planners never aim at a spot the edge guard forbids."""
        p = np.array(xy, float)
        for _ in range(60):
            if self.feet_margin(p, heading) >= margin:
                break
            n = np.linalg.norm(p)
            if n < 1e-3:
                break
            p -= p / n * min(0.005, n)
        return (float(p[0]), float(p[1]))

    def feet_margin(self, xy, heading):
        """How far Pinch's feet stay inside the desk (m; negative: a foot is over the edge)."""
        c, s = np.cos(heading), np.sin(heading)
        return min(self.edge_distance((xy[0] + c * fx - s * fy, xy[1] + s * fx + c * fy)) for fx, fy in self.FEET)

    def in_tray(self, xy):
        return (abs(xy[0] - self.tray_xy[0]) < self.tray_half[0] and abs(xy[1] - self.tray_xy[1]) < self.tray_half[1])

    def clamp(self, xy, margin=None):
        m = self.edge_margin if margin is None else margin
        return (float(np.clip(xy[0], -self.half_x + m, self.half_x - m)),
                float(np.clip(xy[1], -self.half_y + m, self.half_y - m)))


@dataclass
class ObjectBelief:
    name: str
    xy: tuple
    yaw: float = 0.0
    seen_t: float = 0.0
    still_since: float = 0.0
    graspable: bool = True
    on_desk: bool = True


@dataclass
class OwnerBelief:
    present: bool = False
    last_seen: float = -1e9
    last_voice: float = -1e9
    last_touch: float = -1e9
    last_attention: float = -1e9      # spoke to Pinch / touched it / looked at it
    busy: str = ""                    # "" | meeting | focused | away
    mood: str = "calm"                # calm | happy | excited | sad | stressed | tired | angry
    mood_t: float = -1e9
    voice_dir: float = 0.0            # radians, + = left of Pinch


@dataclass
class WorldModel:
    desk: Desk = field(default_factory=Desk)
    owner: OwnerBelief = field(default_factory=OwnerBelief)
    objects: dict = field(default_factory=dict)
    pose: tuple = (0.0, 0.0, 0.0)     # Pinch x, y, heading
    battery: float = 1.0
    t: float = 0.0
    clock: callable = time.time

    def see(self, name, xy, yaw=0.0, graspable=True, on_desk=True):
        o = self.objects.get(name)
        moved = o is None or np.hypot(xy[0] - o.xy[0], xy[1] - o.xy[1]) > 0.01
        if o is None:
            o = self.objects[name] = ObjectBelief(name, tuple(xy), yaw, self.t, self.t, graspable, on_desk)
        o.xy, o.yaw, o.seen_t, o.on_desk, o.graspable = tuple(xy), yaw, self.t, on_desk, graspable
        if moved:
            o.still_since = self.t
        return o

    def owner_attending(self, within=60.0):
        return self.t - self.owner.last_attention < within

    def owner_here(self, within=300.0):
        return self.owner.present or self.t - max(self.owner.last_seen, self.owner.last_voice,
                                                  self.owner.last_touch) < within


@dataclass
class Situation:
    kind: str
    strength: float = 1.0
    target: str = ""


def situations(world: WorldModel, affect, memory) -> list:
    """Everything Pinch notices right now, strongest first."""
    out = []
    w, o, d = world, world.owner, world.desk
    if o.busy == "meeting":
        out.append(Situation("owner_in_meeting", 1.0))
    if o.present and o.busy != "meeting":
        if w.t - o.mood_t < 600 and o.mood in ("sad", "stressed", "tired"):
            out.append(Situation("owner_down", 0.9 if o.mood == "sad" else 0.7, o.mood))
        if w.t - o.mood_t < 120 and o.mood in ("happy", "excited"):
            out.append(Situation("owner_happy", 0.7, o.mood))
        if affect.ignored_s > 90 and affect.needs.social > 0.5:
            out.append(Situation("ignored", min(1.0, affect.ignored_s / 600)))
    if not w.owner_here():
        out.append(Situation("owner_away", 1.0))
    for name, ob in w.objects.items():
        if not ob.on_desk:
            continue
        if d.edge_distance(ob.xy) < 0.035:
            out.append(Situation("object_near_edge", 1.0, name))
        home = memory.home_of(name) if memory else None
        settled = w.t - ob.still_since > 20.0          # don't snatch a pen you're using
        if home is not None and settled and np.hypot(ob.xy[0] - home[0], ob.xy[1] - home[1]) > 0.06:
            out.append(Situation("object_out_of_place", 0.6, name))
    if w.battery < 0.2:
        out.append(Situation("battery_low", 1.0 - w.battery))
    if affect.needs.safety > 0.5:
        out.append(Situation("scared", affect.needs.safety))
    out.sort(key=lambda s: -s.strength)
    return out
