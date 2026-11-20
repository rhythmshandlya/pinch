"""Desk work: tidying things back where they live, raising the alarm about things near the edge."""

from pinch.activities.base import (  # noqa: F401
    GRASP_ENABLED,
    Activity,
    np,
)


class Tidy(Activity):
    """Put things back where they live (pens to the tray, ...). Spontaneous or asked."""
    state, max_s = "task", 400.0
    name, min_s, quiet_ok = "tidy", 5.0, True

    def utility(self, m):
        req = m.tidy_request
        if req and m.t - req[2] < 120:
            return 1.5
        todo = [s for s in m.situations if s.kind == "object_out_of_place" and m.tidy_ok(s.target)]
        if not todo:
            return 0.0
        return 0.75 * m.affect.personality.tidiness * (0.6 + 0.6 * m.affect.needs.tidy)

    def run(self, m):
        kind, obj, _ = m.tidy_request or ("tidy", "", 0)
        m.tidy_request = None
        todo = [obj] if obj else [s.target for s in m.situations
                                  if s.kind == "object_out_of_place" and m.tidy_ok(s.target)][:1]
        if not todo:
            yield ("face", "curious")
            yield ("emote", "look_around")
            return
        for name in todo:
            ob = m.world.objects.get(name)
            if ob is None:
                yield ("face", "curious")
                yield ("emote", "look_around")
                continue
            goal = m.world.desk.owner_xy if kind == "fetch" else (m.memory.home_of(name) or m.world.desk.tray_xy)
            if kind == "fetch":
                goal = (goal[0], m.world.desk.owner_xy[1] + 0.29)
            if m.danger(ob.xy) >= 0.5:                           # a spot it learned to fear: don't go
                yield ("face", "scared")
                yield ("turn_to", m.world.desk.owner_xy)
                yield ("emote", "point")
                m.tidy_attempt(name, False)
                continue
            yield ("face", "focused")
            yield ("look_at", ob.xy)
            picked = False
            if ob.graspable and GRASP_ENABLED:
                yield ("phase", "pick")
                yield ("pick", name)                    # the claw first (decision 24) ...
                picked = m.last_step_ok
                if picked:
                    yield ("phase", "carry")
                    yield ("drop", goal)
            if not picked:
                yield ("phase", "push")
                yield ("push", name, goal)              # ... and pushing when it can't get hold of it
            ok = m.last_step_ok
            # (right by an edge it can't get behind it - no point trying again)
            hopeless = m.world.desk.edge_distance(ob.xy) < 0.08
            if not ok and (hopeless or len([1 for t, k in m.tidy_tries.get(name, []) if not k and m.t - t < 600]) >= 1):
                # second failure (or out of reach): ask my person for help instead of thrashing about
                yield ("phase", "ask_help")
                yield ("face", "sad")
                yield ("turn_to", ob.xy)
                yield ("emote", "point")
                if not m.affect.quiet:
                    yield ("sound", "uh_oh")
                yield ("turn_to", m.world.desk.owner_xy)
                yield ("emote", "beg")
                m.memory.remember("help", f"asked for help with the {name}")
            m.tidy_attempt(name, ok)
            yield ("face", "proud" if ok else "sad")
            if not m.affect.quiet:
                yield ("sound", "fanfare" if ok else "uh_oh")
            if ok:
                m.affect.feel("tidied")
                m.memory.count("tidies")
                m.memory.remember("tidy", f"put the {name} back")
        if kind == "fetch":
            yield ("turn_to", m.world.desk.owner_xy)
            yield ("emote", "beg")
            yield ("praise_wait", 6.0)


class GuardEdge(Activity):
    """Something is about to fall off the desk. Pinch can't get behind it without leaving the desk itself,
    so it raises the alarm: hurries over, points at it and looks at you, worried."""
    state, max_s = "task", 120.0
    name, min_s, quiet_ok = "guard_edge", 4.0, True

    def utility(self, m):
        s = m.situation("object_near_edge")
        return 1.35 if s and m.t - m.edge_alert_t.get(s.target, -1e9) > 120 else 0.0

    def run(self, m):
        s = m.situation("object_near_edge")
        if not s:
            return
        m.edge_alert_t[s.target] = m.t
        ob = m.world.objects[s.target]
        near = m.world.desk.clamp(np.array(ob.xy) * 0.6)
        yield ("face", "surprised")
        if not m.affect.quiet:
            yield ("sound", "uh_oh")
        yield ("goto", near, 0.05)
        yield ("turn_to", ob.xy)
        yield ("face", "scared")
        yield ("emote", "point")
        yield ("turn_to", m.world.desk.owner_xy)
        if not m.affect.quiet:
            yield ("sound", "attention_chirp")
        yield ("emote", "beg")
        m.memory.remember("edge", f"the {s.target} was about to fall off the desk")
