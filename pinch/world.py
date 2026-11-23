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
        """The nearest spot to xy where Pinch, facing `heading`, has every foot at least `margin` inside the desk
        and clear of the tray - so planners never aim at a spot the edge guard forbids. (Searches outward in 16
        directions: sliding toward the desk's middle alone drags a spot beside the tray far away.)"""
        p = np.array(xy, float)
        if self.feet_clear(p, heading) >= margin:
            return (float(p[0]), float(p[1]))
        dirs = [np.array((np.cos(a), np.sin(a))) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        for r in np.arange(0.005, 0.25, 0.005):
            fits = [p + r * d for d in dirs if self.feet_clear(p + r * d, heading) >= margin]
            if fits:
                best = max(fits, key=lambda q: self.feet_clear(q, heading))
                return (float(best[0]), float(best[1]))
        return (float(p[0]), float(p[1]))

    def feet_margin(self, xy, heading):
        """How far Pinch's feet stay inside the desk (m; negative: a foot is over the edge)."""
        return min(self.edge_distance(p) for p in self._feet(xy, heading))

    def feet_clear(self, xy, heading):
        """feet_margin for planning a spot to stand: the tray counts as off the desk too, but nearer - a foot
        TRAY_ALLOW closer to it than to an edge reads the same (its 12 mm walls snag a foot; brushing past it is
        fine, and a push into it brings the front feet up to its mouth). The reflexes use feet_margin."""
        pts = self._feet(xy, heading)       # (feet only: the chassis clears the tray's 12 mm walls by ~7 mm, and
        # also keeping the shell off it cost tidies in the bench - 13/32 vs 16/32 - for no physical gain)
        return min(min(self.edge_distance(p) for p in pts), min(self.tray_gap(p) for p in pts) + self.TRAY_ALLOW)

    def feet(self, xy, heading):
        """Where Pinch's feet are, standing at xy facing heading."""
        return self._feet(xy, heading)

    def _feet(self, xy, heading):
        c, s = np.cos(heading), np.sin(heading)
        return [(xy[0] + c * fx - s * fy, xy[1] + s * fx + c * fy) for fx, fy in self.FEET]

    TRAY_ALLOW = 0.015

    def tray_gap(self, p):
        """Distance from p to the tray's footprint (negative inside it)."""
        dx = abs(p[0] - self.tray_xy[0]) - self.tray_half[0]
        dy = abs(p[1] - self.tray_xy[1]) - self.tray_half[1]
        if dx <= 0 and dy <= 0:
            return max(dx, dy)
        return float(np.hypot(max(dx, 0.0), max(dy, 0.0)))

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
    rest_xy: tuple = None             # where it came to rest (still_since counts from then)


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
    servo_temp: float = None          # C, the hottest leg servo's reading (None: nothing reports it, e.g. no physics)
    t: float = 0.0
    clock: callable = time.time

    def things_on_desk(self):
        """[(name, xy)] of everything believed to be on the desk (what the body keeps its feet off)."""
        return [(n, o.xy) for n, o in self.objects.items() if o.on_desk]

    def see(self, name, xy, yaw=0.0, graspable=True, on_desk=True):
        """A sighting. Camera fixes jitter by several mm, so "it moved" means 2 cm from where it came to rest
        (comparing one noisy fix with the last, it would never count as still). The position is the latest fix -
        a thing being pushed must not lag (the push steers on it) - averaged only once it has been still 0.5 s."""
        o = self.objects.get(name)
        if o is None:
            o = self.objects[name] = ObjectBelief(name, tuple(xy), yaw, self.t, self.t, graspable, on_desk)
        rest = o.rest_xy if o.rest_xy is not None else o.xy
        moved = o.rest_xy is None or np.hypot(xy[0] - rest[0], xy[1] - rest[1]) > 0.02
        if moved:
            o.still_since, o.rest_xy, o.xy = self.t, tuple(xy), tuple(xy)
        elif self.t - o.still_since > 0.5:
            o.xy = (0.7 * o.xy[0] + 0.3 * xy[0], 0.7 * o.xy[1] + 0.3 * xy[1])
        else:
            o.xy = tuple(xy)
        o.yaw, o.seen_t, o.on_desk, o.graspable = yaw, self.t, on_desk, graspable
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
