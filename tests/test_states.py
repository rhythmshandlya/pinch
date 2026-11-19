"""The explicit state layer (decision 63): every activity belongs to a top-level state and has a time limit;
the mind reports the state and the activity's phase, ends anything that overruns, and comes back to an
activity that something urgent cut short."""

from pinch.activities import LIBRARY
from pinch.activities.base import STATES
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.twin import PinchSim


def test_every_activity_declares_a_state_and_limits():
    for a in LIBRARY:
        assert a.state in STATES, f"{a.name}: state {a.state!r} not in {STATES}"
        assert 0 < a.min_s < a.max_s, f"{a.name}: min_s {a.min_s} / max_s {a.max_s}"
    assert len({a.name for a in LIBRARY}) == len(LIBRARY), "two activities share a name"


def test_an_overrunning_activity_ends_with_a_timeout():
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    rest = next(a for a in LIBRARY if a.name == "rest")
    old = rest.max_s
    try:
        rest.max_s = 3.0
        m._start("rest", "test")
        busy = True                                       # the body stays busy: the activity never finishes
        for _ in range(100):
            m.tick(0.1, body_busy=busy, body_ok=True)
        assert any(kind == "timeout" for _, kind, _ in m.log), "no timeout logged"
        assert m.timed_out and m.timed_out[0] == "rest"
    finally:
        rest.max_s = old


def test_state_phase_and_resume_in_the_twin():
    sim = PinchSim(memory=Memory(path=None), seed=7)
    sim.place("cap", (-0.06, 0.0))
    sim.mind.world.see("cap", (-0.06, 0.0), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)
    sim.run(3.0)
    sim.say("pinch, put the cap away")
    states = set()
    for k in range(120):
        sim.run(0.5)
        states.add(sim.mind.state)
        if k == 60:
            sim.mind.touch("stroke")                      # an interruption while it tidies
    recs = list(sim.mind.decisions)
    assert all(r["state"][1] in STATES for r in recs)
    assert "task" in states and "social" in states, f"states seen: {states}"
    tidy_phases = {p for _, a, p in sim.mind.phases if a == "tidy"}
    assert tidy_phases & {"push", "pick"}, f"tidy phases: {tidy_phases}"
    assert any(r["to"] == "tidy" and "(resumed)" in r["reason"] for r in recs), \
        f"tidy was not resumed after the interruption: {[(r['to'], r['reason']) for r in recs]}"
