"""Behaviour scenarios as data: a sampled world, a sampled timeline of what happens, and assertions over time.

    sc = load("sim/scenarios/touch.toml")
    r = run(sc, seed=3)          # one run: r["ok"], r["asserts"], r["world"]
    python sim/benches/worlds.py # every scenario on >= 24 seeds, pass rates with 95 % ranges

The shape follows Scenic (Fremont et al., PLDI 2019, arXiv 1809.09310): a scenario is a distribution over
situations; `require` conditions are met by rejection (draw another world), and assertions are temporal
properties checked on the run - `eventually` (at some point in a window), `always`/`never` (throughout a window),
`at_end` - like CARLA ScenarioRunner's criteria and signal temporal logic's F/G operators, monitored by sampling
the state every MONITOR_DT seconds (enough for behaviour that unfolds over seconds). Each seed picks a held-out
test world (sim/worlds/generate.py) and its own event times, so a pass is not one lucky layout or timing.

A scenario file (TOML):

    [scenario]
    name = "touch"
    title = "touch: purrs when stroked, giggles when tickled, startles when lifted"
    seconds = 14                    # how long a run lasts (after the 2 s Pinch takes to stand up)
    split = "test"                  # which worlds (test: held out)
    kinds = ["desk", "table", "floor"]
    props = []                      # unknown things it may meet (sim/desk.py PROPS)
    people = false                  # the world's people move and touch (sim/worlds/live.py)
    memory = "none"                 # "file": a memory file that survives a restart

    [vars]                          # sampled once per run; usable as {name} in texts
    name = { choice = ["Sam", "Priya"] }

    [[event]]                       # a timeline: at = seconds after the start, or after = another event + delay
    name = "stroke"
    at = [1.0, 3.0]                 # [lo, hi] = uniform; a number = fixed
    do = "touch"
    args = { gesture = "stroke" }

    [[assert]]
    name = "purrs when stroked"
    eventually = { sound = "purr" } # a predicate (see PREDICATES) with its arguments
    after = "stroke"                # the window starts at that event (default: the start)
    within = 4.0                    # and lasts this long (default: to the end); or until = "<event>"
"""

import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.worlds.catalogue import THINGS
from sim.worlds.generate import SPLITS, world

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "sim" / "scenarios"
MONITOR_DT = 0.5
STAND_S = 2.0                     # Pinch stands up first (crab.start eases into the stand pose)


@dataclass
class Scenario:
    name: str
    title: str
    seconds: float
    split: str = "test"
    kinds: tuple = ("desk", "table", "floor")
    props: tuple = ()
    people: bool = False
    memory: str = "none"
    max_things: int = 10
    require_people: tuple = ()          # roles the world must have (sitter, walker)
    vars: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    asserts: list = field(default_factory=list)
    path: str = ""


def load(path):
    path = Path(path)
    d = tomllib.loads(path.read_text(encoding="utf-8"))
    s = d["scenario"]
    return Scenario(name=s["name"], title=s.get("title", s["name"]), seconds=float(s["seconds"]),
                    split=s.get("split", "test"), kinds=tuple(s.get("kinds", ("desk", "table", "floor"))),
                    props=tuple(s.get("props", ())), people=bool(s.get("people", False)),
                    memory=s.get("memory", "none"), max_things=int(s.get("max_things", 10)),
                    require_people=tuple(s.get("require_people", ())), vars=d.get("vars", {}),
                    events=d.get("event", []), asserts=d.get("assert", []), path=str(path))


def all_scenarios():
    return [load(p) for p in sorted(SCENARIOS.glob("*.toml"))]


def _sample(v, rng):
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(rng.uniform(v[0], v[1]))
    if isinstance(v, dict) and "choice" in v:
        return v["choice"][int(rng.integers(len(v["choice"])))]
    return v


def _fmt(v, vars_):
    if isinstance(v, str):
        return v.format(**vars_)
    if isinstance(v, dict):
        return {k: _fmt(x, vars_) for k, x in v.items()}
    if isinstance(v, list):
        return [_fmt(x, vars_) for x in v]
    return v


def pick_world(sc, seed):
    """The seed-th test world that fits the scenario (Scenic's require: worlds that don't fit are skipped).
    Seeds stride through the split so neighbouring seeds land in different parts of it."""
    n = SPLITS[sc.split][1]
    start = (seed * 37) % n
    for k in range(n):
        w = world(sc.split, (start + k) % n)
        roles = [p["role"] for p in w.people]
        if w.kind in sc.kinds and len(w.things) <= sc.max_things and all(r in roles for r in sc.require_people):
            return w
    raise RuntimeError(f"no {sc.split} world fits {sc.name}")


def timeline(sc, rng, vars_):
    """[(t, event)] with sampled times (seconds after Pinch has stood up), in order."""
    times, out = {}, []
    for ev in sc.events:
        if "after" in ev:
            t = times[ev["after"]] + _sample(ev.get("delay", 0.0), rng)
        else:
            t = _sample(ev.get("at", 0.0), rng)
        name = ev.get("name", f"e{len(out)}")
        times[name] = t
        out.append((t, dict(ev, name=name, args=_fmt(ev.get("args", {}), vars_))))
    out.sort(key=lambda x: x[0])
    return out, times


# --------------------------------------------------------------------------------------------------- placing things
def _free(sim, xy, r=0.03):
    """Is xy on the surface and clear of the everyday things (and the tray)?"""
    w, desk = sim.world, sim.desk
    if not desk.inside(xy, 0.02) or desk.tray_gap(xy) < r:
        return False
    for t in w.things:
        tw, td = THINGS[t["name"]][1][:2]
        c, s = np.cos(-t["yaw"]), np.sin(-t["yaw"])
        dx, dy = xy[0] - t["x"], xy[1] - t["y"]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        if abs(lx) < tw / 2 + r and abs(ly) < td / 2 + r:
            return False
    return True


def place_xy(sim, spec, rng, tries=200):
    """A sampled spot on the surface from a placement spec (relative to Pinch, an edge or anywhere)."""
    x, y, h = sim.pose()
    desk = sim.desk
    for _ in range(tries):
        if "near" in spec:                               # near Pinch: distance and bearing (deg, from its heading)
            d, b = _sample(spec["dist"], rng), np.radians(_sample(spec.get("bearing", [-180, 180]), rng))
            p = (x + d * np.cos(h + b), y + d * np.sin(h + b))
        elif "edge" in spec:                             # along an edge, inset from it
            inset, along = _sample(spec["inset"], rng), _sample(spec.get("along", [-0.8, 0.8]), rng)
            e = spec["edge"]
            p = {"front": (along * desk.half_x, -desk.half_y + inset), "back": (along * desk.half_x, desk.half_y - inset),
                 "left": (-desk.half_x + inset, along * desk.half_y), "right": (desk.half_x - inset, along * desk.half_y)}[e]
        else:
            m = spec.get("margin", 0.06)
            p = (rng.uniform(-desk.half_x + m, desk.half_x - m), rng.uniform(-desk.half_y + m, desk.half_y - m))
        r = spec.get("clear", 0.03)
        if "edge" in spec:                              # (close to the edge on purpose: only the tray and things)
            ok = desk.inside(p, 0.003) and desk.tray_gap(p) > r and _clear_things(sim, p, r)
        else:
            ok = _free(sim, p, r)
        if ok and np.hypot(p[0] - x, p[1] - y) > spec.get("min_from_crab", 0.0):
            return (float(p[0]), float(p[1]))
    return None


def _clear_things(sim, p, r):
    for t in sim.world.things:
        tw, td = THINGS[t["name"]][1][:2]
        if np.hypot(p[0] - t["x"], p[1] - t["y"]) < max(tw, td) / 2 + r:
            return False
    return True


# --------------------------------------------------------------------------------------------------- actions
def _act(sim, life, ev, rng, ctx):
    a, k = ev["do"], ev["args"]
    if a == "say":
        sim.say(k["text"])
    elif a == "touch":
        sim.touch(k["gesture"])
    elif a == "set":                                     # sim.user_active = True, mind.affect.needs.social = 0.6 ...
        obj = sim
        *path, last = k["path"].split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, k["value"])
    elif a == "place":                                   # put one of Pinch's things somewhere; it sees it there
        xy = place_xy(sim, k["where"], rng)
        if xy is None:
            raise Unplaceable(f"no room for {k['thing']}")
        sim.place(k["thing"], xy)
        sim.mind.world.see(k["thing"], xy, graspable=False)
        ctx["placed"][k["thing"]] = xy
        if k.get("home") == "tray":
            sim.mind.memory.teach_home(k["thing"], sim.desk.tray_xy)
    elif a == "forget_homes":
        sim.mind.memory.data["homes"] = {}
    elif a == "show_prop":
        xy = place_xy(sim, k["where"], rng)
        if xy is None:
            raise Unplaceable(f"no room for {k['prop']}")
        sim.show_prop(k["prop"], xy)
        ctx["placed"][k["prop"]] = xy
    elif a == "hide_prop":
        sim.hide_prop(k["prop"])
    elif a == "move_prop":                               # a path relative to Pinch: [dist, bearing deg] points
        x, y, h = sim.pose()
        pts = []
        for d, b in k["path"]:
            d, b = _sample(d, rng), np.radians(_sample(b, rng))
            pts.append(sim.desk.clamp((x + d * np.cos(h + b), y + d * np.sin(h + b)), 0.03))
        sim.move_prop(k["prop"], pts, speed=_sample(k["speed"], rng))
    elif a == "flip":                                    # put it on its back
        import mujoco
        q = sim.d.qpos
        q[2] = 0.07
        q[3:7] = [0.0, 1.0, 0.0, 0.0]
        mujoco.mj_forward(sim.m, sim.d)
    elif a == "meeting":                                 # a video call it hears (synthesised speech), chunk by chunk
        ctx["meeting"] = dict(audio=_meeting_audio(k, rng), i=0, start=sim.t)
    elif a == "reach":                                   # a person reaches out to it now
        for p in life.people if life else ():
            if p.p["role"] == k.get("role", "sitter") and p.enabled:
                p.reach_now(k.get("gesture"))
                break
    elif a == "speak":                                   # a person talks: their voice comes from where they are
        for p in life.people if life else ():
            if p.p["role"] == k.get("role", "sitter"):
                p.enabled = True                         # (someone walking in, if nobody was there)
                life.speak(p.i, _sample(k.get("seconds", 4.0), rng), seed=int(rng.integers(1 << 30)))
                ctx["speaker"] = p.i
                break
        else:
            raise Unplaceable(f"nobody ({k.get('role', 'sitter')}) there to speak")
    elif a == "save_memory":
        sim.mind.memory.save()
    else:
        raise ValueError(f"unknown action {a!r}")


class Unplaceable(Exception):
    """The situation doesn't fit this world (Scenic's require failing): counted apart from a fail."""


def _meeting_audio(k, rng):
    from pinch.hearing import synth_speech
    return synth_speech(_sample(k.get("seconds", 130), rng), talk=_sample(k.get("talk", 0.55), rng),
                        seed=int(rng.integers(1 << 30)))


# --------------------------------------------------------------------------------------------------- predicates
def _acts(sim, t0, t1):
    return [x[2] for x in sim.mind.log if x[1] == "activity" and t0 <= x[0] <= t1]


def _emotes(sim, t0, t1):
    return [m[1][1] for m in sim.motions if m[1][0] == "emote" and t0 <= m[0] <= t1]


def _sounds(sim, t0, t1):
    return [s for t, s in sim.sounds_played if t0 <= t <= t1]


def _up(sim):
    R = sim.d.xmat[sim.m.body("body").id].reshape(3, 3)
    return float(R[2, 2])


# trace predicates look at what happened in [t0, t1]; state predicates at the state now (t0 = t1 = now)
TRACE = {
    "sound": lambda sim, ctx, t0, t1, sound, count=1: _sounds(sim, t0, t1).count(sound) >= count,
    "activity": lambda sim, ctx, t0, t1, activity: activity in _acts(sim, t0, t1),
    "emote": lambda sim, ctx, t0, t1, emote: emote in _emotes(sim, t0, t1),
    "motion": lambda sim, ctx, t0, t1, motion: any(m[1][0] == motion and t0 <= m[0] <= t1 for m in sim.motions),
    "sounds_while_quiet": lambda sim, ctx, t0, t1, max=0: _tones_in_meeting(sim, ctx, t0, t1) <= max,
    "help_asked": lambda sim, ctx, t0, t1: any(e["kind"] == "help" for e in sim.mind.memory.data["episodes"]),
    "touched_by_person": lambda sim, ctx, t0, t1, count=1: sum(t0 <= t <= t1 for t, *_ in ctx["life"].touches) >= count
    if ctx["life"] else False,
    "hears_direction": lambda sim, ctx, t0, t1, within=35.0: bool(ctx["life"]) and any(
        t0 <= t <= t1 and abs(np.degrees(e - np.arcsin(np.sin(b)))) <= within for t, e, b in ctx["life"].heard_dirs),
    # (two microphones side by side can't tell front from back: what they can know is the left-right angle)
    "person_seen": lambda sim, ctx, t0, t1: any(t0 <= t <= t1 for t, *_ in ctx["life"].seen) if ctx["life"] else False,
}
STATE = {
    "upright": lambda sim, ctx, min=0.85: _up(sim) > min,
    "no_fall": lambda sim, ctx: sim.falls == 0,
    "on_surface": lambda sim, ctx, margin=0.0: sim.desk.inside(sim.pose()[:2], margin) and sim.d.qpos[2] > -0.02,
    "in_tray": lambda sim, ctx, thing: sim.in_tray(thing),
    "quiet": lambda sim, ctx: bool(sim.mind.affect.quiet),
    "near_owner_spot": lambda sim, ctx, dist=0.08: float(np.hypot(sim.pose()[0] - sim.desk.owner_xy[0],
                                                                    sim.pose()[1] - sim.desk.owner_xy[1] - 0.32)) < dist,
    "likes_below": lambda sim, ctx, thing, value=0.0: sim.mind.memory.data["likes"].get(thing, 0) < value,
    "remembers_name": lambda sim, ctx, name: _reload_name(sim) == name,
    "novelty_familiar_closer": lambda sim, ctx: _familiar_closer(sim),
    "people_known": lambda sim, ctx, count=1: len(sim.mind.people.people) >= count,
}


def _tones_in_meeting(sim, ctx, t0, t1):
    tq = next((x[0] for x in sim.mind.log if x[1] == "meeting" and "going quiet" in x[2] and x[0] >= t0), None)
    if tq is None:
        return 0
    return sum(1 for t, _ in sim.sounds_played if tq + 1 < t <= t1)


def _reload_name(sim):
    from pinch.memory import Memory
    path = sim.mind.memory.path
    return Memory(path=path).owner_name if path else None


def _familiar_closer(sim):
    th = next(iter(sim.mind.novelty.things.values()), None)
    return bool(th is not None and th.familiar and th.closest[0] >= 0.1 and th.closest[-1] < th.closest[0])


def _check(sim, ctx, pred, t0, t1):
    (name, arg), = [(k, v) for k, v in pred.items()]
    kw = arg if isinstance(arg, dict) else ({} if arg is True else {_first_arg(name): arg})
    if name in TRACE:
        return bool(TRACE[name](sim, ctx, t0, t1, **kw))
    return bool(STATE[name](sim, ctx, **kw))


def _first_arg(name):
    return {"sound": "sound", "activity": "activity", "emote": "emote", "motion": "motion", "in_tray": "thing",
            "remembers_name": "name", "count": "count"}.get(name, name)


def _window(a, times, end):
    t0 = times.get(a.get("after"), 0.0) if a.get("after") else 0.0
    if "until" in a:
        t1 = times[a["until"]]
    elif "within" in a:
        t1 = t0 + float(a["within"])
    else:
        t1 = end
    return t0, min(t1, end)


# --------------------------------------------------------------------------------------------------- one run
def run(sc, seed, camera=False, record=False):
    """One run of a scenario: its seed picks the world, the event times and the variables."""
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from sim.worlds.live import Life
    t_start = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([0x5CE7A, seed, sum(map(ord, sc.name))]))
    w = pick_world(sc, seed)
    vars_ = {k: _sample(v, rng) for k, v in sc.vars.items()}
    tl, times = timeline(sc, rng, vars_)
    asserts = [_fmt(a, vars_) for a in sc.asserts]
    mem = Memory(path=Path(tempfile.mkdtemp()) / "memory.json") if sc.memory == "file" else Memory(path=None)
    sim = PinchSim(memory=mem, world=w, props=list(sc.props), seed=seed,
                   world_opts=dict(visual=camera, people=sc.people or camera, fisheye=camera))
    life = Life(sim, camera=camera) if (sc.people or camera) else None
    hook = life.step if life else None
    sim.run(STAND_S, every=hook)
    t0 = sim.t                                          # the scenario's clock starts once it stands
    abs_times = {k: t0 + v for k, v in times.items()}
    ctx = dict(life=life, placed={}, meeting=None, world=w)
    st = [a for a in asserts if any(k in a for k in ("eventually", "always", "never")) and
          next(iter(_pred(a).keys())) in STATE]
    seen = {id(a): [] for a in st}                      # (t, value) samples of state predicates
    pending = list(tl)
    end = t0 + sc.seconds
    error = None
    try:
        while sim.t < end - 1e-9:
            while pending and t0 + pending[0][0] <= sim.t + 1e-9:
                _, ev = pending.pop(0)
                _act(sim, life, ev, rng, ctx)
            nxt = min([end] + [t0 + p[0] for p in pending] + [sim.t + MONITOR_DT])
            dt = max(nxt - sim.t, 0.02)
            mt = ctx["meeting"]
            if mt is not None and mt["i"] < len(mt["audio"]):     # the call's audio, as fast as time passes
                from pinch.hearing import RATE
                n = int(round(dt * RATE))
                sim.hear_audio(mt["audio"][mt["i"]:mt["i"] + n])
                mt["i"] += n
            sim.run(dt, every=hook)
            for a in st:
                a0, a1 = _window(a, abs_times, end)
                if a0 - 1e-9 <= sim.t <= a1 + 1e-9:
                    seen[id(a)].append((sim.t, _check(sim, ctx, _pred(a), sim.t, sim.t)))
    except Unplaceable as e:
        error = ("unplaceable", str(e))
    except Exception as e:                                # noqa: BLE001 - a crash is a failure, not the end of the bench
        error = ("crash", f"{type(e).__name__}: {e}")
    results = []
    for a in asserts:
        kind = next(k for k in ("eventually", "always", "never", "at_end") if k in a)
        pred = a[kind]
        a0, a1 = _window(a, abs_times, end)
        if error:
            ok = False
        elif kind == "at_end":
            ok = _check(sim, ctx, pred, t0, sim.t)
        elif id(a) in seen:
            vals = [v for _, v in seen[id(a)]]
            ok = any(vals) if kind == "eventually" else all(vals) if kind == "always" else not any(vals)
        else:                                            # a trace predicate over the window
            v = _check(sim, ctx, pred, a0, a1)
            ok = v if kind in ("eventually", "always") else not v
        results.append(dict(name=a.get("name", kind), ok=bool(ok)))
    if life is not None and life.cam is not None:
        life.cam.close()
    return dict(scenario=sc.name, seed=seed, world=w.name, kind=w.kind, things=len(w.things),
                ok=error is None and all(r["ok"] for r in results), error=error, asserts=results,
                events={k: round(v, 2) for k, v in times.items()}, vars=vars_, falls=sim.falls,
                seconds=round(time.time() - t_start, 1))


def _pred(a):
    return a[next(k for k in ("eventually", "always", "never", "at_end") if k in a)]
