"""The behaviour states as data (configs/mind/states.toml, PRD state requirements): every state has its sub-states,
allowed motions, intent for every body part, interruption rule and timeout behaviour; a motion its state does not
allow is refused and logged; resting turns into the sleep state while asleep."""

import inspect
import re

from pinch.activities import LIBRARY
from pinch.activities.base import STATES, Activity
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.mind.behaviour import CHANNELS, allowed, states
from pinch.mind.common import MOTION_STEPS


def test_every_state_is_fully_defined():
    s = states()
    assert set(s) == set(STATES), f"states.toml {sorted(s)} vs STATES {sorted(STATES)}"
    for name, st in s.items():
        assert st["sub"] and st["on_timeout"], name
        assert set(st["intent"]) == set(CHANNELS), name
        assert set(st["allow"]) <= MOTION_STEPS, f"{name}: unknown steps {set(st['allow']) - MOTION_STEPS}"
        assert 1 <= st["interrupt"] <= 8, name


def test_every_activity_only_uses_motions_its_state_allows():
    for a in LIBRARY:
        used = set(re.findall(r'yield \(\s*"([a-z_]+)"', inspect.getsource(type(a)))) & MOTION_STEPS
        bad = {k for k in used if not allowed(a.state, k)}
        assert not bad, f"{a.name} ({a.state}) uses {sorted(bad)}"


class _Rogue(Activity):
    name, state = "rogue", "idle"

    def run(self, m):
        yield ("pick", "cap", "right")                   # idle may not pick things up
        yield ("emote", "wave")


def test_a_disallowed_motion_is_refused_and_logged():
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    rogue = _Rogue()
    m.activity, m.gen, m.started = rogue, rogue.run(m), 0.0
    step = m._advance(body_busy=False, body_ok=True)
    assert step == ("emote", "wave"), step
    assert any(k == "refused" and "pick" in txt for _, k, txt in m.log), m.log[-3:]


def test_resting_asleep_is_the_sleep_state():
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    m._start("rest", "test")
    assert m.state == "rest"
    for _ in range(8):                                   # face, goto (motion) ... until the asleep phase
        m.t += 0.5                                       # (time passes: the eyes lead a walk by 0.4 s)
        m._advance(body_busy=False, body_ok=True)
        if m.phase == "asleep":
            break
    assert m.phase == "asleep" and m.state == "sleep"


def test_the_state_decides_who_may_cut_in_urgently():
    from pinch.mind.behaviour import rank
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    m._start("rest", "test")
    m.phase = "asleep"                                   # the sleep state: only rank <= 4 may wake it urgently
    m.started = m.t - 100.0
    assert m.state == "sleep" and rank("greet") > 4 >= rank("react_touch")
    old = {a.name: a.utility for a in LIBRARY}
    try:
        for a in LIBRARY:
            a.utility = (lambda self_, mm, n=a.name: 3.0 if n == "greet" else 0.0).__get__(a)
        m.start_score = 2.9                              # (the rest it is in still counts nearly as much)
        m._choose()
        assert m.activity.name == "rest", m.activity.name
        for a in LIBRARY:
            a.utility = (lambda self_, mm, n=a.name: 3.0 if n == "react_touch" else 0.0).__get__(a)
        m._choose()
        assert m.activity.name == "react_touch", m.activity.name
    finally:
        for a in LIBRARY:
            a.utility = old[a.name]


def test_every_step_name_belongs_to_a_sub_state():
    """Each ("phase", name) an activity yields maps to one of its state's sub-states (states.toml phases), so the
    sub-states are what the mind reports, not just a list (dynamic names: game verbs, quirks, orient_<kind>)."""
    from pinch.activities.play import Quirk
    from pinch.mind.behaviour import substate
    from pinch.play import VERBS
    for a in LIBRARY:
        names = set(re.findall(r'\("phase", "([a-z_]+)"\)', inspect.getsource(type(a))))
        names |= set(VERBS) if a.name == "play" else set(Quirk.QUIRKS) if a.name == "quirk" else set()
        if a.name == "orient":
            names.add("orient_loud")
        if "disengage" in inspect.getsource(type(a)):
            names.add("disengage")
        for n in names:
            st = "sleep" if (a.state, n) == ("rest", "asleep") else a.state
            assert substate(st, n) is not None, f"{a.name}: step '{n}' is no sub-state of {st}"


def test_the_mind_reports_its_sub_state_and_social_exchanges_close():
    from pinch.activities.social import Celebrate
    from pinch.observer import _no_closing
    m = Mind(memory=Memory(path=None), llm=False, seed=0, clock=lambda: 1.8e9)
    m._start("celebrate", "test")
    assert m.substate == "notice_person"                  # every activity starts in its state's first sub-state
    steps = list(Celebrate().run(m))
    assert ("phase", "disengage") in steps and steps.index(("phase", "disengage")) == len(steps) - 3
    m.phase = "disengage"
    assert m.substate == "disengage"
    decs = [dict(t=0.0, prev="idle", to="celebrate", reason="x"),
            dict(t=9.0, prev="celebrate", to="idle", reason="previous activity finished")]
    assert _no_closing(decs, [(8.0, "celebrate", "disengage")]) == []
    assert [f.kind for f in _no_closing(decs, [])] == ["no_closing"]
