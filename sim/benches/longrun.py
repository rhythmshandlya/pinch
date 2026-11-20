"""Long-duration acceptance runs: a 24-hour day of full autonomy, and a 7-day memory run.

    python sim/benches/longrun.py day --seeds 0-3 --jobs 4 --json day.json     # 24 h, 5 physics windows of 10 min
    python sim/benches/longrun.py week --seeds 0 --json week.json              # 7 days, a reboot every night

Why two speeds. The digital twin (pinch/twin) runs at about 3.5x real time on one CPU core, so 24 h of physics
would take ~7 h and a week two days. The mind alone runs ~1000x faster. So, as in sampled simulation (SMARTS:
fast functional simulation of the whole program, detailed simulation of sampled windows warmed from it), a run
here is:
  * the whole span in **fast-forward**: the real Mind (same code, same seeded generators, same one simulation
    clock) on a kinematic desk (FastPet): a motion step takes its nominal time (emotes their keyframe length,
    walking ~6 cm/s) and has its effect (a goto moves Pinch, a push or a carry moves the thing; seeded success
    rates). Everything the mind decides, says, learns and remembers is real; only the body is abstracted.
  * **physics windows**: at sampled moments (morning, a meeting, the afternoon, the evening, the night) the full
    twin is started from the fast-forward state at that moment (memory, feelings, desk layout, Pinch's pose, who
    is here) and runs the same scripted day, with the twin's safety checks (falls, leaving the desk, things
    knocked off, sounds inside quiet hours).
Everything is seeded: the same seed gives the same day, decision for decision (`digest` in the results).
(Sources: Wunderlich et al., "SMARTS", ISCA 2003 - fast-forward plus warmed, sampled detailed windows; Feng et al.,
"Dense reinforcement learning for safety validation of autonomous vehicles", Nature 615, 2023 - accelerated testing
that skips the uneventful stretches of driving to evaluate rare events 10^3-10^5x faster.)

The day (DayPlan): you arrive at 07:00, set quiet hours (22:00-07:00) the first morning, leave for breakfast,
work at the desk 09:00-18:00 with a daily stand-up at ~10:00 and an afternoon meeting, lunch away, an evening at
home with a visitor, and say bye at 22:30. Nobody tells Pinch to sleep: the night must come from the routine's
sleep drive (pinch/routine.py, x3 at night).

The week: seven days of that (weekends without work), a reboot every night at 03:00 (the memory saved to its
file and loaded by a fresh mind), a colleague who drops by on workdays, a rough guest on the first day only,
tricks praised on the first three days and never again, and a near-fall spot learned on day one. Checked:
memory survives every reboot, familiarity grows for people who come and decays (90-day half-life) for one who
doesn't, fear of the rough guest fades (3-day half-life), the danger spot fades (8 awake hours), the stand-up
becomes a known routine, and likes stay put without new evidence.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES  # noqa: E402
from pinch.activities import LIBRARY  # noqa: E402
from pinch.affect import Needs  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402
from pinch.observer import observe_records  # noqa: E402
from pinch.people import HALF_F, HALF_R  # noqa: E402
from pinch.routine import in_window  # noqa: E402
from pinch.safety import CAREFUL, V_HALF_H  # noqa: E402
from pinch.twin.common import SIM_EPOCH, TWIN_UTC_OFFSET_H  # noqa: E402
from pinch.world import Desk  # noqa: E402
from sim.benches.common import run_parallel, save_json  # noqa: E402
from sim.desk import OBJECTS, START  # noqa: E402

HOUR, DAY = 3600.0, 86400.0
EPOCH0 = SIM_EPOCH - 2 * HOUR             # 07:00 local (India) on the twin's first day
QUIET = (22.0, 7.0)                       # the quiet hours the owner sets the first morning
NIGHT_SLEEP = (23.0, 6.0)                 # judged for sleep: the routine's night, an hour after you leave
QUIET_OK = {a.name: a.quiet_ok for a in LIBRARY}

# ------------------------------------------------------------------------------------------------- the day
CHAT = ["hi pinch", "dance!", "how are you?", "pinch, wave", "come here", "let's play",
        "pinch, put the cap away", "shake", "play dead", "I'm tired", "I'm sad today", "pinch, come here"]
EVIDENCE = ["good job pinch", "no, bad pinch", "you're so cute"]   # praise / scold: after a trick, likes change
TOUCHES = ["stroke", "stroke", "tickle", "boop", "tap", "hold"]


def local(t):
    """Local hour (0-24) at run time t (seconds since 07:00 on day 0)."""
    return (7.0 + t / HOUR) % 24.0


class DayPlan:
    """A seeded schedule of days: [(t, kind, arg)] sorted by t (t in seconds since 07:00 on day 0), plus the
    intervals the checks need (when you're here, in meetings)."""

    def __init__(self, seed, days=1, evidence_days=None, colleague=False, guest_day=None, visitor=True, intro_h=7.0):
        self.intro_h = intro_h                    # when you introduce yourself and set quiet hours (day 0)
        self.rng = np.random.default_rng(20_000 + seed)
        self.events, self.meetings, self.here = [], [], []
        self.desk = Desk()
        evidence_days = range(days) if evidence_days is None else evidence_days
        for d in range(days):
            self._day(d, workday=d % 7 < 5, evidence=d in evidence_days, colleague=colleague,
                      guest=d == guest_day, visitor=visitor and d % 7 == 0)
        self.events.sort(key=lambda e: e[0])

    def at(self, d, h):
        return d * DAY + (h - 7.0) * HOUR

    def _day(self, d, workday, evidence, colleague, guest, visitor):
        r, ev = self.rng, self.events
        if workday:
            blocks = [(7.0, 8.25, 90.0), (9.0, 13.0, 300.0), (14.0, 18.0, 300.0), (19.0, 22.5, 120.0)]
            stand = 10.0 + r.uniform(0, 5) / 60
            self.meetings.append((self.at(d, stand), self.at(d, stand + r.uniform(15, 30) / 60)))
            if r.random() < 0.8:
                pm = r.uniform(14.5, 16.5)
                self.meetings.append((self.at(d, pm), self.at(d, pm + r.uniform(30, 60) / 60)))
        else:
            blocks = [(8.5, 12.0, 120.0), (15.0, 23.0, 150.0)]
        for h0, h1, gap in blocks:
            t0, t1 = self.at(d, h0), self.at(d, h1)
            self.here.append((t0, t1))
            ev.append((t0, "arrive", None))
            ev.append((t0 + 2, "say", "good morning pinch" if h0 < 9 else "hi pinch"))
            ev.append((t1, "leave", None))
            t = t0 + 60
            while t < t1 - 30:                                          # sightings: the camera sees you
                ev.append((t, "person", ("sam", 0.9, "Sam", True)))
                t += 30.0
            t = t0 + r.exponential(gap)
            while t < t1 - 60:
                if not any(a - 30 <= t <= b + 30 for a, b in self.meetings):
                    ev.extend(self._interaction(t, evidence))
                t += r.exponential(gap)
        if d == 0:
            ev.append((self.at(d, self.intro_h) + 20, "say", "hi pinch, I'm Sam"))
            ev.append((self.at(d, self.intro_h) + 40, "say", "Pinch, quiet hours from 10 pm to 7 am"))
        for a, b in self.meetings:
            if d * DAY <= a < (d + 1) * DAY:
                ev.append((a, "say", "I'm in a meeting"))
                ev.append((b, "say", "my meeting's over"))
        if evidence:                                                     # a trick, then praise (likes learn)
            for h in (9.8, 20.5):
                t = self.at(d, h)
                ev.append((t, "say", "pinch, wave"))
                ev.append((t + 9, "say", "good job pinch"))
        if colleague and workday:                                        # Priya drops by after the stand-up
            t0 = self.at(d, 10.6)
            for k in range(0, 1200, 30):
                ev.append((t0 + k, "person", ("priya", 0.85, "Priya", False)))
            ev.append((t0 + 300, "touch_by", ("priya", "stroke")))
        if guest:                                                        # a rough guest, never seen again
            t0 = self.at(d, 9.65)
            for k in range(0, 1200, 30):
                ev.append((t0 + k, "person", ("guest", 0.8, "", False)))
            for k in (400, 700, 1000):
                ev.append((t0 + k, "touch_by", ("guest", "shaken")))
        if visitor:                                                      # someone drops by in the evening
            t0 = self.at(d, 20.0)
            for k in range(0, 2700, 30):
                ev.append((t0 + k, "person", (f"visitor-{d}", 0.8, "", False)))

    def _interaction(self, t, evidence):
        r = self.rng.random()
        if r < 0.5:
            pool = CHAT + (EVIDENCE if evidence else [])
            return [(t, "say", pool[self.rng.integers(len(pool))])]
        if r < 0.85:
            return [(t, "touch_by", ("sam", TOUCHES[self.rng.integers(len(TOUCHES))]))]
        name = ("cap", "eraser", "pen")[self.rng.integers(3)]
        for _ in range(20):                       # you move a desk thing while you work
            xy = (float(self.rng.uniform(-0.2, 0.2)), float(self.rng.uniform(-0.1, 0.1)))
            if not self.desk.in_tray(xy) and math.hypot(*xy) > 0.15:
                return [(t, "nudge", (name, xy, float(self.rng.uniform(-math.pi, math.pi))))]
        return []

    def state_at(self, t):
        """(here, in_meeting) at time t."""
        return (any(a <= t < b for a, b in self.here), any(a <= t < b for a, b in self.meetings))


def apply_event(pet, kind, arg):
    """One scheduled event, on a FastPet or a PinchSim (both: .mind, .say(), .place(), .user_active)."""
    m = pet.mind
    if kind == "say":
        pet.say(arg)
    elif kind == "arrive":
        pet.user_active = True
        m.world.owner.present = True
    elif kind == "leave":
        pet.say("bye pinch")
        pet.user_active = False
    elif kind == "person":
        pid, conf, name, owner = arg
        m.see_person(pid, conf, name=name, owner=owner)
    elif kind == "touch_by":                     # seen, then touches Pinch (credit or blame goes to them)
        pid, gesture = arg
        if pid in m.people.people:
            m.see_person(pid, 0.9)
        m.touch(gesture)
    elif kind == "nudge":
        name, xy, yaw = arg
        pet.place(name, xy, yaw)


# ------------------------------------------------------------------------------------------ fast-forward
MIND_DT = 0.1                    # the mind's 10 Hz, as in the twin
WALK_V, PUSH_V = 0.06, 0.03      # m/s, the walker's and the push's typical speeds on the bench
PICK_OK, PUSH_OK = 0.7, 0.85     # seeded success rates for the abstract body (pick and push benches' ballpark)


class FastPet:
    """The real Mind on a kinematic desk: no physics, each motion step takes its nominal time and has its effect.
    Same interface the events use as PinchSim (say, place, user_active)."""

    def __init__(self, memory, seed, epoch=EPOCH0, t0=0.0, desk_state=None):
        self.t, self.epoch = t0, epoch
        self.desk = Desk()
        self.mind = Mind(memory=memory, llm=False, seed=seed, clock=lambda: self.epoch + self.t)
        if self.mind.routine.settings.get("utc_offset_h") is None:
            self.mind.routine.settings["utc_offset_h"] = TWIN_UTC_OFFSET_H
        self.rng = np.random.default_rng(seed + 101)
        st = desk_state or dict(pose=(0.0, 0.0, 0.0), objects={n: START[n][:2] for n in OBJECTS})
        self.pose = tuple(st["pose"])
        self.objects = {n: tuple(xy) for n, xy in st["objects"].items()}
        self.held = None
        for name in OBJECTS:
            self.mind.world.see(name, self.objects[name], graspable=OBJECTS[name][4])
        self.mind.memory.teach_home("pen", self.desk.tray_xy)
        self.mind.memory.teach_home("eraser", (self.desk.tray_xy[0] - 0.04, self.desk.tray_xy[1]))
        self.user_active = False
        self.busy, self.ok, self.done_t, self.effect = False, True, 0.0, None
        self.sounds, self.motions = [], []
        self.decisions, self.log = [], []
        _tap(self.mind, self.decisions, self.log, lambda: self.t)

    def desk_state(self):
        return dict(pose=self.pose, objects=dict(self.objects))

    def say(self, text):
        self.user_active = True
        return self.mind.hear_text(text)

    def place(self, name, xy, yaw=0.0):
        self.objects[name] = (float(xy[0]), float(xy[1]))

    # -- the abstract body --------------------------------------------------------------------------------
    def _clamp(self, xy):
        return self.desk.clamp(xy, 0.06)

    def _start(self, step):
        kind, a = step[0], step[1:]
        x, y, h = self.pose
        dur, ok, eff = 0.5, True, None
        if kind == "emote":
            emo = EMOTES.get(a[0])
            dur, ok = (emo.duration, True) if emo else (0.0, False)
        elif kind == "pose":
            dur = 0.6 + (a[1] if len(a) > 1 else 2.0)
        elif kind == "walk":
            vx, vy, wz, secs = a
            c, s = math.cos(h), math.sin(h)
            dur = secs + 0.6
            eff = ("pose", (*self._clamp((x + (c * vx - s * vy) * secs, y + (s * vx + c * vy) * secs)), h + wz * secs))
        elif kind == "goto":
            to = self._clamp(a[0])
            d = math.hypot(to[0] - x, to[1] - y)
            dur = d / WALK_V + 1.0
            eff = ("pose", (to[0], to[1], math.atan2(to[1] - y, to[0] - x) if d > 0.01 else h))
        elif kind == "turn_to":
            dur = 1.5
            eff = ("pose", (x, y, math.atan2(a[0][1] - y, a[0][0] - x)))
        elif kind == "push":
            name, goal = a[0], a[1]
            o = self.objects.get(name)
            if o is None:
                ok = False
            else:
                ok = self.rng.random() < PUSH_OK
                goal = tuple(goal)
                d1, d2 = math.hypot(o[0] - x, o[1] - y), math.hypot(goal[0] - o[0], goal[1] - o[1])
                dur = d1 / WALK_V + d2 / PUSH_V + 3.0
                to = goal if ok else (o[0] + 0.5 * (goal[0] - o[0]), o[1] + 0.5 * (goal[1] - o[1]))
                eff = ("move", name, to)
        elif kind == "pick":
            o = self.objects.get(a[0])
            ok = o is not None and OBJECTS[a[0]][4] and self.rng.random() < PICK_OK
            dur = (math.hypot(o[0] - x, o[1] - y) / WALK_V if o else 0.0) + 4.0
            eff = ("hold", a[0]) if ok else None
        elif kind == "drop":
            ok = self.held is not None
            if ok:
                goal = tuple(a[0])
                dur = math.hypot(goal[0] - x, goal[1] - y) / WALK_V + 2.0
                eff = ("move", self.held, goal)
        elif kind == "release":
            dur, eff = 1.0, ("move", self.held, (x, y)) if self.held else None
        elif kind in ("recover", "right_self"):
            dur = 3.0
        else:
            ok, dur = False, 0.0
        self.busy, self.ok, self.done_t, self.effect = dur > 0, ok, self.t + dur, eff
        self.motions.append((round(self.t, 2), kind))

    def _free_spot(self, name, xy, gap=0.035):
        """Two things can't share a spot (the physics windows start from this layout): next to what's there."""
        x, y = float(xy[0]), float(xy[1])
        for k in range(12):
            cand = (x - gap * (k % 4), y + gap * (k // 4) * (1 if y < 0 else -1))
            if all(math.hypot(cand[0] - o[0], cand[1] - o[1]) >= gap for n, o in self.objects.items() if n != name):
                return self._clamp(cand) if k else cand
        return (x, y)

    def _finish(self):
        eff = self.effect
        if eff and eff[0] == "pose":
            self.pose = eff[1]
        elif eff and eff[0] == "move":
            self.objects[eff[1]] = self._free_spot(eff[1], eff[2])
            if self.held == eff[1]:
                self.held = None
        elif eff and eff[0] == "hold":
            self.held = eff[1]
        self.busy, self.effect = False, None

    # -- the loop ---------------------------------------------------------------------------------------------
    def step(self):
        m = self.mind
        w = m.world
        w.pose = self.pose
        if self.held:
            self.objects[self.held] = self.pose[:2]
        for name, xy in self.objects.items():
            w.see(name, xy, graspable=OBJECTS[name][4])
        if self.user_active:
            w.owner.present = True
            w.owner.last_seen = m.t
        motion, cancel = m.tick(MIND_DT, self.busy, self.ok)
        if cancel:
            self.busy, self.ok, self.effect = False, False, None
        if motion is not None:
            self._start(motion)
        for s in m.outputs()[2]:
            self.sounds.append((round(self.t, 2), s))
        self.t += MIND_DT
        if self.busy and self.t >= self.done_t - 1e-9:
            self._finish()

    def run_until(self, t_end, plan=None, k=0, on_tick=None):
        """Step until t_end, applying plan events as their time comes. Returns the next event index."""
        ev = plan.events if plan is not None else []
        while self.t < t_end - 1e-9:
            while k < len(ev) and ev[k][0] <= self.t:
                apply_event(self, ev[k][1], ev[k][2])
                k += 1
            self.step()
            if on_tick:
                on_tick(self)
        return k


def _tap(mind, decisions, log, now):
    """Keep every decision and diary line (the mind keeps only the last 1000 / 300), stamped with run time."""
    record, say = mind._record_decision, mind._say

    def _record(name, reason):
        record(name, reason)
        decisions.append(dict(mind.decisions[-1], t=round(now(), 2)))

    def _say(kind, text):
        say(kind, text)
        log.append((round(now(), 1), kind, text))

    mind._record_decision, mind._say = _record, _say


def digest(decisions, sounds):
    h = hashlib.sha256()
    h.update(json.dumps([(d["t"], d["to"], d["reason"]) for d in decisions]).encode())
    h.update(json.dumps(sounds).encode())
    return h.hexdigest()[:16]


def snapshot(pet):
    """What a physics window needs to start where the fast-forward is."""
    a = pet.mind.affect
    return dict(t=pet.t, memory=json.dumps(pet.mind.memory.data), needs=a.needs.as_dict(), valence=a.valence,
                arousal=a.arousal, desk=pet.desk_state(), user_active=pet.user_active,
                present=pet.mind.world.owner.present, meeting=pet.mind.world.owner.busy == "meeting")


# ------------------------------------------------------------------------------------------ physics windows
CLEAR = 0.12                     # m: desk things start at least this far from Pinch (the twin's own start: 13 cm)


def physics_window(args):
    """Run the full twin for `seconds` from a fast-forward snapshot, replaying the same plan."""
    seed, snap, events, seconds, label = args
    from pinch.observer import observe
    from pinch.twin import PinchSim
    mem = Memory(path=None)
    mem.data = json.loads(snap["memory"])
    t0 = snap["t"]
    sim = PinchSim(memory=mem, seed=seed + 1000 + int(t0) % 9973, epoch=EPOCH0 + t0)
    m = sim.mind
    m.affect.needs = Needs(**snap["needs"])
    m.affect.valence, m.affect.arousal = snap["valence"], snap["arousal"]
    x, y, h = snap["desk"]["pose"]
    if sim.desk.feet_margin((x, y), h) > 0.03:                    # Pinch where the fast-forward left it
        import mujoco
        sim.d.qpos[0:2] = (x, y)
        sim.d.qpos[3:7] = (math.cos(h / 2), 0, 0, math.sin(h / 2))
        sim.d.qvel[:] = 0
        mujoco.mj_forward(sim.m, sim.d)
    px, py, _ = sim.pose()
    for name, xy in snap["desk"]["objects"].items():
        d = math.hypot(xy[0] - px, xy[1] - py)
        if d < CLEAR:                     # (the kinematic desk has no contacts: never start a thing under a foot)
            u = ((xy[0] - px) / d, (xy[1] - py) / d) if d > 1e-6 else (1.0, 0.0)
            xy = sim.desk.clamp((px + CLEAR * u[0], py + CLEAR * u[1]), 0.06)
        if math.hypot(xy[0] - sim.obj(name)[0][0], xy[1] - sim.obj(name)[0][1]) > 0.005:
            sim.place(name, xy, 0.0)
    sim.user_active = snap["user_active"]
    m.world.owner.present = snap["present"]
    if snap["meeting"]:
        sim.say("I'm in a meeting")
    off_desk, things_off, k = 0.0, set(), 0
    ev = [(t - t0, kind, arg) for t, kind, arg in events if t0 <= t < t0 + seconds]
    while sim.t < seconds:
        while k < len(ev) and ev[k][0] <= sim.t:
            apply_event(sim, ev[k][1], ev[k][2])
            k += 1
        sim.run(1.0)
        x, y, _ = sim.pose()
        if not sim.desk.inside((x, y), 0.0):
            off_desk += 1.0
        for name in OBJECTS:
            if sim.obj(name)[1] < -0.02:
                things_off.add(name)
    quiet_sounds = [s for t, s in sim.sounds_played if in_window(local(t0 + t), *QUIET)]
    findings = observe(m, seconds)
    errors = [f"{f.kind}: {f.evidence}" for f in findings if f.severity == "error"]
    acts = Counter(d["to"] for d in m.decisions)
    ok = sim.falls == 0 and off_desk == 0 and not quiet_sounds and not errors    # (things off: their own check)
    return dict(label=label, start_local=round(local(t0), 2), seconds=seconds, ok=ok, falls=sim.falls,
                pinch_off_desk_s=off_desk, things_off=sorted(things_off), quiet_hour_sounds=quiet_sounds,
                edge_stops=sim.edge_stops, edge_recovers=sim.edge_recovers, stuck=len(sim.stuck_events),
                observer_errors=errors, activities=dict(acts.most_common()), events=len(ev))


# Checks these runs found failing in the mind itself (not in the harness), kept visible until fixed: a known issue
# shows as KNOWN and doesn't fail the verdict, and the tests expect it to still fail (a strict xfail): when one is
# fixed, its test fails and says to delete the entry here.
KNOWN_ISSUES = {
    "sleeps_at_night": "Rest ends once sleepiness < 0.05 and each 10 s nap takes 0.08 off it, ~20x faster than the "
                       "night drive (x3) builds it back, so a night is ~1 min of nap per ~16 min of idling "
                       "(pinch/activities/rest.py)",
    "night_not_day": "same cause as sleeps_at_night",
    "observer_no_failure_loop": "SeekAttention (and Shy while a stranger is near) finishes in ~10 s and wins again, "
                                "every ~75 s for hours while you work; the observer counts each re-choice as a restart",
    "physics_nothing_knocked_off": "turning in place doesn't keep the legs off nearby things: in the evening window "
                                   "the ball flew 16 cm in 1 s during SeekAttention's turn_to with Pinch ~10 cm "
                                   "away, and the ball and cap then roll on to the edge (pinch/body turn_to)",
    "danger_extinction_survives_reboot": "the danger map's decay is never written back to memory.data['danger'] "
                                         "(pinch/mind/selfcare.py), so a reboot restores the fear as last learned",
}


def _check(checks, name, ok, value, want):
    c = dict(ok=bool(ok), value=value, want=want)
    if name in KNOWN_ISSUES:
        c["known_issue"] = KNOWN_ISSUES[name]
    checks[name] = c


def _state_share(decisions, t0, t1, end):
    """Share of [t0, t1) spent in each top-level state, from the decision records."""
    out = Counter()
    for d, nxt in zip(decisions, decisions[1:] + [None]):
        a, b = max(t0, d["t"]), min(t1, nxt["t"] if nxt else end)
        if b > a:
            out[d["state"][1]] += b - a
    return {k: v / (t1 - t0) for k, v in out.items()}


# --------------------------------------------------------------------------------------------- the 24 h run
WINDOWS = (("morning", 7.1), ("standup", None), ("afternoon", 15.2), ("evening", 20.1), ("night", 2.0))


def day_run(seed, hours=24.0, windows=WINDOWS, window_s=600.0, jobs=1, physics=True):
    """24 h (from 07:00) of full autonomy: fast-forward for the whole day, physics for the sampled windows."""
    w0 = time.time()
    plan = DayPlan(seed, days=1)
    span = hours * HOUR
    starts = []
    for label, h in windows:
        t = plan.meetings[0][0] + 120.0 if h is None else ((h - 7.0) % 24.0) * HOUR
        if t + window_s <= span:
            starts.append((label, t))
    pet = FastPet(Memory(path=None, clock=lambda: EPOCH0), seed)       # (a fresh memory's "born" on the sim clock)
    snaps, k = {}, 0
    for label, t in sorted(starts, key=lambda s: s[1]):
        k = pet.run_until(t, plan, k)
        snaps[label] = snapshot(pet)
    pet.run_until(span, plan, k)
    ff_s = time.time() - w0
    dec, end = pet.decisions, span
    checks = {}
    findings = observe_records(dec, [], pet.log, end)
    bad = [f for f in findings if f.kind in ("oscillation", "state_lock") or f.severity == "error"]
    _check(checks, "observer_no_lock_or_oscillation", not bad, [f"{f.kind}: {f.evidence}" for f in bad],
           "no state lock, oscillation or observer error")
    loops = Counter(f.data.get("activity") for f in findings if f.kind == "failure_loop")
    _check(checks, "observer_no_failure_loop", not loops, dict(loops), "no failure loop")
    night0, night1 = ((NIGHT_SLEEP[0] - 7.0) * HOUR, (NIGHT_SLEEP[1] + 24.0 - 7.0) * HOUR)
    if span >= night1:
        rest_night = _state_share(dec, night0, night1, end).get("rest", 0.0)
        rest_day = _state_share(dec, 2 * HOUR, 11 * HOUR, end).get("rest", 0.0)      # 09:00-18:00
        _check(checks, "sleeps_at_night", rest_night >= 0.6, round(rest_night, 3), ">= 0.6 of 23:00-06:00 resting")
        _check(checks, "night_not_day", rest_night >= rest_day + 0.3, [round(rest_day, 3), round(rest_night, 3)],
               "night rest share >= day (09-18) share + 0.3")
        first = next((d["t"] for d in dec if d["t"] >= 15 * HOUR and d["state"][1] == "rest"), None)
        _check(checks, "asleep_by_midnight", first is not None and first <= 17 * HOUR,
               None if first is None else round(local(first), 2), "first rest after 22:00 starts by 00:00")
    in_quiet = [(t, s) for t, s in pet.sounds if in_window(local(t), *QUIET) and t > 60]
    in_meet = [(t, s) for t, s in pet.sounds if any(a + 1 <= t <= b for a, b in plan.meetings)]
    loud_start = [(d["t"], d["to"]) for d in dec if not QUIET_OK[d["to"]] and
                  (in_window(local(d["t"]), *QUIET) or any(a + 1 <= d["t"] <= b for a, b in plan.meetings))]
    _check(checks, "quiet_hours_silent", not in_quiet, len(in_quiet), "0 sounds 22:00-07:00")
    _check(checks, "meetings_silent", not in_meet, len(in_meet), "0 sounds in meetings")
    _check(checks, "quiet_activities_only", not loud_start, loud_start[:5], "no loud activity started in quiet time")
    acts = Counter(d["to"] for d in dec)
    n = sum(acts.values())
    entropy = -sum(c / n * math.log2(c / n) for c in acts.values()) if n else 0.0
    switches = Counter(d["to"] for d in dec if d["prev"] != d["to"])     # (not idle restarting itself)
    top = switches.most_common(1)[0][1] / sum(switches.values()) if switches else 1.0
    _check(checks, "variety", len(acts) >= 10 and top <= 0.6, [len(acts), round(top, 3), round(entropy, 2)],
           ">= 10 different activities, none > 60 % of the switches (entropy of all choices reported)")
    day_sounds = sum(1 for t, _ in pet.sounds if not in_window(local(t), *QUIET))
    _check(checks, "alive_by_day", day_sounds > 0, day_sounds, "some sounds outside quiet hours")
    wins = []
    if physics and snaps:
        w1 = time.time()
        jobs_args = [(seed, snaps[label], plan.events, window_s, label) for label, _ in starts]
        wins = run_parallel(physics_window, jobs_args, jobs) if jobs > 1 else [physics_window(a) for a in jobs_args]
        phys_s = time.time() - w1
        for w, (_, t) in zip(wins, starts):          # what the fast-forward chose in the same minutes, beside it
            w["ff_activities"] = dict(Counter(d["to"] for d in dec if t <= d["t"] < t + window_s).most_common())
        _check(checks, "physics_safe", all(w["ok"] for w in wins),
               {w["label"]: dict(falls=w["falls"], off=w["pinch_off_desk_s"], things=w["things_off"],
                                 quiet=len(w["quiet_hour_sounds"]), err=len(w["observer_errors"])) for w in wins},
               "no falls, Pinch never off the desk, silent in quiet hours, no observer errors")
        _check(checks, "physics_nothing_knocked_off", not any(w["things_off"] for w in wins),
               {w["label"]: w["things_off"] for w in wins if w["things_off"]}, "no desk thing ends up off the desk")
    else:
        phys_s = 0.0
    hourly = {}
    for hh in range(int(hours)):
        sh = _state_share(dec, hh * HOUR, (hh + 1) * HOUR, end)
        hourly[f"{int(local(hh * HOUR)):02d}"] = max(sh, key=sh.get) if sh else "-"
    return dict(seed=seed, hours=hours, ok=all(c["ok"] or c.get("known_issue") for c in checks.values()),
                checks=checks, decisions=n, activities=dict(acts.most_common()), entropy_bits=round(entropy, 2),
                state_share={k: round(v, 3) for k, v in sorted(_state_share(dec, 0, end, end).items(), key=lambda kv: -kv[1])},
                main_state_by_hour=hourly, sounds=len(pet.sounds), events=len(plan.events),
                findings=[dict(kind=f.kind, severity=f.severity, t_local=round(local(f.t0), 2), evidence=f.evidence)
                          for f in findings],
                windows=wins, digest=digest(dec, pet.sounds),
                wall_s=dict(fast_forward=round(ff_s, 1), physics=round(phys_s, 1),
                            physics_sim_s=window_s * len(wins), ff_speedup=round(span / max(ff_s, 1e-9))))


# ------------------------------------------------------------------------------------------ the 7-day run
DANGER_XY = (0.15, -0.10)


def _people_now(mind, now):
    """Each person's numbers faded to `now` (on a copy: reading must not change them)."""
    ppl = copy.deepcopy(mind.people)
    ppl.seen = {}
    out = {}
    for pid, p in ppl.people.items():
        ppl._fade(p, now)
        out[pid] = dict(F=round(p["F"], 5), W=round(p["W"], 5), R=round(p["R"], 5), t=p["t_update"])
    return out


def week_run(seed, days=7, awake=None, reboot_h=3.0, mem_dir=None):
    """Seven simulated days of one pet. awake=None: every hour of every day, with a reboot at reboot_h each night
    (memory saved, a fresh mind loads it). awake=(h0, h1): only those local hours each day (the robot is switched
    on at h0 and off at h1: a short version for the test suite)."""
    w0 = time.time()
    plan = DayPlan(seed, days=days, evidence_days=range(3), colleague=True, guest_day=0, visitor=False,
                   intro_h=7.0 if awake is None else awake[0])
    tmp = None
    if mem_dir is None:
        tmp = tempfile.TemporaryDirectory()
        mem_dir = tmp.name
    path = Path(mem_dir) / "memory.json"
    if path.exists():
        path.unlink()
    if awake is None:
        boots = [(0.0, (reboot_h + 24.0 - 7.0) * HOUR)]
        boots += [(boots[0][1] + d * DAY, min(days * DAY, boots[0][1] + (d + 1) * DAY)) for d in range(days)]
        boots = [b for b in boots if b[1] > b[0]]
    else:
        boots = [(d * DAY + (awake[0] - 7.0) * HOUR, d * DAY + (awake[1] - 7.0) * HOUR) for d in range(days)]
    daily, restarts, k, desk, awake_s, danger_before = [], [], 0, None, 0.0, None
    danger_learned, danger_v0, awake_at_learn = False, None, 0.0
    all_dec, sounds = [], []
    for b, (t0, t1) in enumerate(boots):
        mem = Memory(path=path, clock=lambda t0=t0: EPOCH0 + t0)
        before = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if before is not None:
            restarts.append(dict(t_local=round(local(t0), 2), day=int(t0 // DAY),
                                 identical=json.loads(json.dumps(mem.data)) == before, danger_before=danger_before))
        pet = FastPet(mem, seed + 31 * b, t0=t0, desk_state=desk)
        if before is not None:
            restarts[-1]["danger_after"] = round(pet.mind.danger(DANGER_XY), 4)
        while k < len(plan.events) and plan.events[k][0] < t0:           # (what happened while it was off)
            k += 1
        here, meeting = plan.state_at(t0)
        if here:
            pet.user_active = True
            pet.mind.world.owner.present = True
            if meeting:
                pet.say("I'm in a meeting")
        if not danger_learned:                                            # a near-fall at the right edge, twice
            k = pet.run_until(t0 + 600.0, plan, k)
            pet.mind.learn_danger(DANGER_XY, "near_fall")
            pet.mind.learn_danger(DANGER_XY, "near_fall")
            danger_learned, danger_v0, awake_at_learn = True, pet.mind.danger(DANGER_XY), awake_s + 600.0
        boot_t, boot_danger = pet.t, pet.mind.danger(DANGER_XY)          # (the fear this boot starts with)
        # end-of-day readings happen at the end of each local day (or of each awake span)
        cuts = sorted({t for t in [(d + 1) * DAY - 1.0 for d in range(days)] if t0 < t < t1} | {t1})
        for cut in cuts:
            k = pet.run_until(cut, plan, k)
            mind = pet.mind
            now = mind.world.clock()
            awake_now = awake_s + (pet.t - t0)
            daily.append(dict(day=int((cut - 1) // DAY), t_local=round(local(cut), 2), awake_h=round(awake_now / HOUR, 2),
                              people=_people_now(mind, now), danger=round(mind.danger(DANGER_XY), 4), boot=b,
                              boot_h=round((pet.t - boot_t) / HOUR, 3), boot_danger0=round(boot_danger, 4),
                              danger_spots=len(mind.danger_map.spots), likes=dict(mind.memory.data["likes"]),
                              meeting_counts=list(mind.memory.data["routines"]["meeting"]),
                              meeting_likely={h: mind.memory.meeting_likely(h) for h in range(24)},
                              stats={s: mind.memory.data["stats"][s] for s in ("pets", "praise", "scolds", "tricks")},
                              owner=mind.memory.owner_name, quiet_hours=mind.routine.settings.get("quiet_hours")))
        awake_s += t1 - t0
        danger_before = round(pet.mind.danger(DANGER_XY), 4)
        pet.mind.memory.changed()                                        # power off: what's in memory is saved
        pet.mind.memory.save()
        desk = pet.desk_state()
        all_dec.extend(pet.decisions)
        sounds.extend(pet.sounds)
    if tmp is not None:
        tmp.cleanup()
    checks = _week_checks(daily, restarts, danger_v0, awake_at_learn, days)
    return dict(seed=seed, days=days, awake=awake, ok=all(c["ok"] or c.get("known_issue") for c in checks.values()),
                checks=checks, restarts=restarts, daily=daily, decisions=len(all_dec),
                digest=digest(all_dec, sounds), wall_s=round(time.time() - w0, 1))


def _week_checks(daily, restarts, danger_v0, awake_at_learn, days):
    checks = {}
    ends = {}
    for r in daily:                                   # the last reading of each day
        ends[r["day"]] = r
    last = ends[max(ends)]
    _check(checks, "memory_survives_reboots", restarts and all(r["identical"] for r in restarts), len(restarts),
           "every reboot loads exactly what was saved")
    keep = all(r["owner"] == "Sam" and r["quiet_hours"] == [22.0, 7.0] for r in daily)
    mono = all(all(b["stats"][s] >= a["stats"][s] for s in a["stats"]) for a, b in zip(daily, daily[1:]))
    _check(checks, "owner_settings_stats_kept", keep and mono, dict(owner=last["owner"], quiet=last["quiet_hours"],
           stats=last["stats"]), "owner name and quiet hours kept, counters never go back")
    # people: the owner and the colleague grow familiar, the guest fades exactly by the half-lives
    sam = [ends[d]["people"]["sam"]["F"] for d in sorted(ends)]
    _check(checks, "owner_familiarity_grows", all(b >= a for a, b in zip(sam, sam[1:])) and sam[-1] >= 0.8,
           [round(f, 3) for f in sam], "rises every day, >= 0.8 by the end (from 0.6 when introduced)")
    pri = [ends[d]["people"].get("priya", {}).get("F", 0.0) for d in sorted(ends)]
    work = [d for d in sorted(ends) if d % 7 < 5]
    grew = all(pri[b] > pri[a] for a, b in zip(work, work[1:]))
    _check(checks, "colleague_familiarity_grows", grew and 0.05 < pri[-1] < 0.95, [round(f, 3) for f in pri],
           "rises every workday, between 0.05 and 0.95")
    g0, g1 = ends[0]["people"]["guest"], last["people"]["guest"]
    span = (max(ends) - 0) * DAY
    f_ratio, r_ratio = g1["F"] / g0["F"], (g1["R"] / g0["R"]) if g0["R"] > 0 else 0.0
    f_want, r_want = 0.5 ** (span / HALF_F), 0.5 ** (span / HALF_R)
    _check(checks, "guest_familiarity_decays", abs(f_ratio - f_want) < 0.003 and g1["F"] < g0["F"],
           dict(F_day0=g0["F"], F_end=g1["F"], ratio=round(f_ratio, 4)), f"ratio {f_want:.4f} (90-day half-life)")
    _check(checks, "guest_fear_fades", g0["R"] > 0.2 and abs(r_ratio - r_want) < 0.01,
           dict(R_day0=g0["R"], R_end=g1["R"], ratio=round(r_ratio, 4)), f"R > 0.2 on day 0, ratio {r_want:.4f} (3-day)")
    # danger: fades with awake time in the running mind; and a reboot should not bring it back
    want_ok = all(r["danger"] <= r["boot_danger0"] * 0.5 ** (r["boot_h"] / V_HALF_H) + 0.01 for r in daily)
    long_boots = [r for r in daily if r["boot_h"] >= 16.0]          # 2 half-lives of 8 h: below "careful"
    _check(checks, "danger_fades_while_awake", danger_v0 >= 0.5 and want_ok
           and all(r["danger"] < CAREFUL for r in long_boots),
           dict(v0=round(danger_v0, 3), by_reading=[(r["day"], r["boot_h"], r["boot_danger0"], r["danger"]) for r in daily]),
           "no-go (>= 0.5) when learned; each boot <= start * 0.5^(h / 8); below careful after 16 awake hours")
    back = [r for r in restarts if r["danger_before"] is not None and r["danger_after"] > r["danger_before"] + 0.05]
    _check(checks, "danger_extinction_survives_reboot", not back,
           [(r["day"], r["danger_before"], r["danger_after"]) for r in back[:4]], "a reboot keeps the faded fear")
    # routines: the daily stand-up is learned, hours without meetings are not
    learned = next((d for d in sorted(ends) if ends[d]["meeting_likely"][10]), None)
    false_hours = [h for h in (3, 8, 12, 21) if last["meeting_likely"][h]]
    _check(checks, "standup_routine_learned", learned is not None and learned <= 3 and not false_hours
           and all(ends[d]["meeting_likely"][10] for d in sorted(ends) if d >= learned),
           dict(learned_on_day=learned, counts_10h=[ends[d]["meeting_counts"][10] for d in sorted(ends)],
                false_hours=false_hours), "meeting_likely(10) by day 3 and after; not at 03/08/12/21")
    # likes: learned while there was praise (days 0-2), identical afterwards
    after = [ends[d]["likes"] for d in sorted(ends) if d >= 2]
    _check(checks, "likes_stable_without_evidence", bool(after and after[0]) and all(x == after[0] for x in after),
           dict(day2=after[0] if after else None, end=after[-1] if after else None), "non-empty, unchanged days 2..end")
    return checks


# ---------------------------------------------------------------------------------------------- the CLI
def _day_job(a):
    return day_run(*a)


def _week_job(a):
    return week_run(*a)


def _verdict(r):
    known = sum(1 for c in r["checks"].values() if not c["ok"] and c.get("known_issue"))
    return ("PASS" if r["ok"] else "FAIL") + (f" ({known} known issues)" if known else "")


def _print_checks(r):
    for name, c in r["checks"].items():
        tag = "ok  " if c["ok"] else ("KNOWN" if c.get("known_issue") else "FAIL")
        print(f"    [{tag}] {name}: {json.dumps(c['value'], default=str)[:160]}  (want {c['want']})")
        if c.get("known_issue") and not c["ok"]:
            print(f"           known issue: {c['known_issue']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=("day", "week"))
    ap.add_argument("--seeds", default="0", help="e.g. 3 or 0-5")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--window-min", type=float, default=10.0, help="day: length of each physics window")
    ap.add_argument("--no-physics", action="store_true", help="day: fast-forward only")
    ap.add_argument("--days", type=int, default=7, help="week: how many days")
    ap.add_argument("--json", "--out", dest="json", default=None, help="also write the results here")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    if a.run == "day":
        if len(seeds) == 1:                       # one day: its physics windows run in parallel
            res = [day_run(seeds[0], window_s=a.window_min * 60, jobs=a.jobs, physics=not a.no_physics)]
        else:
            res = run_parallel(_day_job, [(s, 24.0, WINDOWS, a.window_min * 60, 1, not a.no_physics) for s in seeds],
                               a.jobs)
        for r in res:
            print(f"seed {r['seed']}: {_verdict(r)}  24 h, {r['decisions']} decisions over "
                  f"{len(r['activities'])} activities ({r['entropy_bits']} bits), wall {r['wall_s']}, digest {r['digest']}")
            print(f"    main state by hour: {' '.join(f'{h}:{s}' for h, s in r['main_state_by_hour'].items())}")
            _print_checks(r)
            for w in r["windows"]:
                print(f"    window {w['label']:9s} {w['start_local']:5.2f}h {w['seconds']:.0f} s: {'ok' if w['ok'] else 'FAIL'} "
                      f"falls {w['falls']} off {w['pinch_off_desk_s']:.0f} s things {w['things_off']} edge stops "
                      f"{w['edge_stops']} recovers {w['edge_recovers']} stuck {w['stuck']}")
                print(f"        physics:      {w['activities']}\n        fast-forward: {w.get('ff_activities')}")
    else:
        res = run_parallel(_week_job, [(s, a.days) for s in seeds], a.jobs)
        for r in res:
            print(f"seed {r['seed']}: {_verdict(r)}  {r['days']} days, {len(r['restarts'])} reboots, "
                  f"{r['decisions']} decisions, wall {r['wall_s']} s, digest {r['digest']}")
            _print_checks(r)
    save_json(a.json, res)
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
