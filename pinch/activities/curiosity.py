"""Curious but careful: watching what moves, and going to look at a new thing on the desk (pinch/novelty.py).

Investigate is one approach-retreat excursion (Gordon, Fonio & Ahissar 2014): freeze and look (risk assessment),
creep closer in stop-go steps to a distance set by how wary it is, pause, side-step round it to see it from
another side, back off, and come back later - closer and bolder each time - until the thing is familiar. A
fright (sudden motion, the thing moving, fear) or a failed step cuts the look short: it backs off at once and
waits longer before trying again. It never plans a spot near an edge, in the tray or on a feared spot of the
danger map (pinch/safety.py), and doesn't wander about during quiet hours or meetings (Watch still does).
"""

import math

from pinch.activities.base import (  # noqa: F401
    Activity,
)
from pinch.attention import direction_of
from pinch.novelty import MOVE_V
from pinch.safety import CAREFUL

SIDE_ANGLES = (1.05, 0.85, 0.7)   # rad round the thing to the next viewpoint: 60 deg, less where the desk is tight
SIDE_SEP = 0.5               # rad: viewpoints this far apart (seen from the thing) count as different sides
ARC_STEP = 0.55              # rad: side-step round it in steps this big (keeps the path off the thing)
FEET_CLEAR = 0.025           # m: as the body's goto (Locomotion.FEET_EDGE + 5 mm)
NO_RESUME = ("idle", "watch", "orient", "investigate", "startle")


def _view(center, pose):
    """The side it is looking from: the angle from the thing to Pinch (rad)."""
    return math.atan2(pose[1] - center[1], pose[0] - center[0])


def _sides(views, sep=SIDE_SEP):
    """How many different sides these views cover (views at least sep apart)."""
    out = []
    for v in views:
        if all(abs((v - o + math.pi) % (2 * math.pi) - math.pi) >= sep for o in out):
            out.append(v)
    return len(out)


def _dist(m, xy):
    x, y, _ = m.world.pose
    return math.hypot(xy[0] - x, xy[1] - y)


def _spot(m, center, r, ang):
    """A place r from the thing at angle ang (from the thing), facing it (or nearly) - or None if Pinch shouldn't
    stand there: near an edge (every foot FEET_CLEAR inside), in or by the tray, or somewhere it has learned to fear."""
    p = (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang))
    face = math.atan2(center[1] - p[1], center[0] - p[0])
    d = m.world.desk
    if not d.inside(p, 0.05) or m.danger(p) >= CAREFUL or d.tray_gap(p) < 0.04:
        return None
    for off in (0.0, 0.35, -0.35, 0.7, -0.7):       # turned a little away is fine: the camera sees +-0.9 rad
        if d.feet_clear(p, face + off) >= FEET_CLEAR:
            return p, face + off
    return None


class Watch(Activity):
    """Something is moving: the eyes follow it, and Pinch turns to keep it in view, until it stops."""
    state, max_s = "alert", 20.0
    name, min_s, quiet_ok, own_jitter = "watch", 1.0, True, True

    def utility(self, m):
        th = m.novelty.moving(m.t) if "_novelty" in m.__dict__ else None
        return 0.0 if th is None else 0.6 + 0.3 * th.confidence

    def run(self, m):
        th = m.novelty.moving(m.t)
        if th is None:
            return
        yield ("phase", "watch")
        yield ("face", "curious")
        last_move, end = m.t, m.t + 15.0
        while m.t < end:
            cur = m.novelty.get(th.id)
            if cur is None:
                break
            if cur.speed > MOVE_V:
                last_move = m.t
            elif m.t - last_move > 1.5:
                break
            yield ("look_at", cur.xy)
            if abs(direction_of(cur.xy, m.world.pose)) > 0.75:
                yield ("turn_to", cur.xy)
                yield ("look_at", cur.xy)
            else:
                yield ("wait", 0.2)
        m.attend_key(f"motion:{th.id}")


class Investigate(Activity):
    """A new thing on the desk: go and have a careful look at it (see the module docstring)."""
    state, max_s = "explore", 120.0
    name, min_s, own_jitter = "investigate", 3.0, True

    def utility(self, m):
        if "_novelty" not in m.__dict__:
            return 0.0
        th, interest = m.novelty.most_interesting(m.t, m.world.pose)
        if th is None or interest < 0.05:
            return 0.0
        n = m.affect.needs
        fresh = th.bouts == 0 and m.t - th.born_t < 60.0
        base = 0.5 if fresh else 0.3 + 0.4 * n.curiosity
        u = (base + 0.3 * interest) * m.affect.personality.curiosity / 0.8
        return u * (1 - n.sleep) * (1 - m.hunger()) * (1 - n.safety)

    @staticmethod
    def _abort(m, th, start):
        cur = m.novelty.get(th.id)
        if cur is None:
            return "gone"
        if math.hypot(cur.xy[0] - start[0], cur.xy[1] - start[1]) > 0.04 or cur.speed > MOVE_V:
            return "it moved"
        if m.affect.needs.safety > 0.4:
            return "scared"
        return None

    def run(self, m):
        th, _ = m.novelty.most_interesting(m.t, m.world.pose)
        if th is None:
            return
        prev = m.decisions[-1]["prev"] if m.decisions else None
        if prev and prev not in NO_RESUME and m.suspended is None:
            m.suspended = (prev, m.t)                   # whatever it was doing comes back afterwards
        m.attend_key(th.key)
        start, first = th.xy, th.bouts == 0
        d0, closest, views, why = _dist(m, th.xy), [], [], None

        # 1. freeze and look: how risky is it?
        yield ("phase", "notice")
        yield ("face", "surprised" if first else "curious")
        if first and not m.affect.quiet:
            yield ("sound", "curious_chirp")
        yield ("look_at", th.xy)
        yield ("turn_to", th.xy)
        yield ("look_at", th.xy)
        yield ("wait", 0.8 + 1.5 * th.threat)
        why = self._abort(m, th, start)
        r = m.novelty.standoff(th)
        x, y, _ = m.world.pose
        a0 = math.atan2(y - th.xy[1], x - th.xy[0])
        side = next(((a0 + da, s) for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6)
                     if (s := _spot(m, th.xy, r, a0 + da)) is not None), None)
        if why is None and side is None:              # nowhere safe to stand near it: look from here
            yield ("phase", "watch_from_afar")
            yield ("wait", 2.0)
            views.append(_view(th.xy, m.world.pose))
            m.novelty.end_bout(th, d0, views, True, m.t, close=False)
            m._say("curious", f"can't get near thing {th.id} safely - looked from {d0 * 100:.0f} cm")
            return
        slow = 0.35 + 0.35 * (1.0 - th.threat)          # creeps while wary

        # 2. approach, stop-go
        if why is None:
            ang, (p, face) = side
            yield ("phase", "approach")
            if _dist(m, th.xy) > r + 0.08 and (mid := _spot(m, th.xy, r + 0.06, ang)) is not None:
                yield ("goto", mid[0], 0.03, mid[1], 25.0, slow)
                closest.append(_dist(m, th.xy))
                yield ("look_at", th.xy)
                yield ("wait", 0.6 + 1.2 * th.threat)
                why = self._abort(m, th, start)
            if why is None:
                yield ("goto", p, 0.02, face, 25.0, slow)
                closest.append(_dist(m, th.xy))
                why = self._abort(m, th, start)
                if why is None and closest[-1] > r + 0.06:
                    why = "couldn't get there"

        # 3. pause and look
        if why is None:
            yield ("phase", "pause")
            yield ("face", "curious")
            yield ("look_at", th.xy)
            yield ("wait", 1.0 + th.threat)
            views.append(_view(th.xy, m.world.pose))
            why = self._abort(m, th, start)

        # 4. side-step round it to see it from another side (both sides once it is less wary)
        if why is None:
            yield ("phase", "inspect")
            turn = 1.0 if th.bouts % 2 == 0 else -1.0
            want = 2 if th.threat > 0.45 else 3         # sides in all: wary, one more; bolder, both
            rel, rr = 0.0, r
            options = [(sgn * a, rad) for a in SIDE_ANGLES for sgn in (turn, -turn) for rad in (r, r + 0.03)]
            for target, rad in options:
                if _sides(views) >= want:
                    break
                if _sides(views + [ang + target]) == _sides(views):     # (not a new side)
                    continue
                n = max(1, math.ceil(abs(target - rel) / ARC_STEP))
                arc = [_spot(m, th.xy, rr + (rad - rr) * k / n, ang + rel + (target - rel) * k / n)
                       for k in range(1, n + 1)]
                if any(s is None for s in arc):         # (an edge, the tray or a feared spot that way)
                    continue
                for s in arc:
                    yield ("goto", s[0], 0.02, s[1], 20.0, slow)
                    closest.append(_dist(m, th.xy))
                    why = self._abort(m, th, start)
                    if why:
                        break
                if why:
                    break
                rel, rr = target, rad
                yield ("look_at", th.xy)
                yield ("wait", 1.0)
                views.append(_view(th.xy, m.world.pose))

        # 5. back off (a fright: at once and quickly), then a last look from a distance
        cur = m.novelty.get(th.id)
        yield ("phase", "retreat")
        if why is not None:
            yield ("face", "scared" if why in ("scared", "it moved") else "confused")
        where = cur.xy if cur is not None else th.xy
        x, y, _ = m.world.pose
        a = math.atan2(y - where[1], x - where[0])
        back = next((s for da in (0.0, 0.5, -0.5, 1.0, -1.0)
                     if (s := _spot(m, where, _dist(m, where) + 0.10, a + da)) is not None), None)
        if back is not None:
            yield ("goto", back[0], 0.03, back[1], 15.0, 1.0 if why else 0.8)
        yield ("look_at", where)
        yield ("wait", 1.0)
        if cur is None:
            m._say("curious", f"thing {th.id} is gone")
            return
        safe = why is None or why == "couldn't get there"
        near = min(closest, default=d0)
        gained = m.novelty.end_bout(th, near, views, safe, m.t)
        m.memory.changed()
        m.affect.feel("novelty", min(1.0, 2.0 * gained))
        m._say("curious", f"looked at thing {th.id} from {_sides(views)} side(s), closest {near * 100:.0f} cm"
                          + (f", cut short: {why}" if why else "")
                          + f"; novelty {th.novelty:.2f}, wariness {th.threat:.2f}")
        yield ("face", "happy" if th.familiar else "curious")
