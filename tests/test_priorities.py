"""The declared priority order (configs/mind/behaviour.toml, PRD behaviour priority hierarchy): every activity has
a category in it, and only the top categories may cut short an activity that is not interruptible."""

from pinch.activities import LIBRARY
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.mind.behaviour import category, overrides_uninterruptible, rank, settings, version


def test_every_activity_has_a_priority_category():
    placed = set(settings()["activity"])
    names = {a.name for a in LIBRARY}
    assert names <= placed, f"activities without a priority category: {sorted(names - placed)}"
    assert placed <= names, f"priorities for activities that do not exist: {sorted(placed - names)}"
    assert version()


def test_the_order_puts_safety_first():
    assert rank("right_self") == 1
    assert rank("low_power") == 2
    assert all(rank(n) <= 3 for n in ("startle", "unstick", "hide", "guard_edge"))
    assert rank("obey") < rank("greet") < rank("explore") < rank("idle")


def test_only_the_top_three_categories_override_uninterruptible_activities():
    for a in LIBRARY:
        assert overrides_uninterruptible(a.name) == (rank(a.name) <= 3), a.name
    assert not overrides_uninterruptible("greet") and not overrides_uninterruptible("explore")


def _mind_running(name):
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    m._start(name, "test")
    m.started = m.t - 10.0                       # past its minimum time
    return m


def _offer(m, best, score):
    """Make `best` the clear favourite and let the mind choose; the activity running afterwards."""
    old = {a.name: a.utility for a in LIBRARY}
    try:
        for a in LIBRARY:
            a.utility = (lambda self_, mm, s=score: s) if a.name == best else (lambda self_, mm: 0.0)
            a.utility = a.utility.__get__(a)
        m.start_score = 0.0
        m._choose()
    finally:
        for a in LIBRARY:
            a.utility = old[a.name]
    return m.activity.name


def test_an_uninterruptible_activity_gives_way_only_to_the_top_categories():
    # obey (asked by its person) is not interruptible: an urgent greeting waits, a startle does not
    m = _mind_running("obey")
    assert _offer(m, "greet", 3.0) == "obey"
    m = _mind_running("obey")
    assert _offer(m, "startle", 3.0) == "startle"
    m = _mind_running("obey")
    assert _offer(m, "right_self", 3.0) == "right_self"


def test_the_decision_log_names_the_categories():
    m = _mind_running("explore")
    _offer(m, "startle", 3.0)
    rec = m.decisions[-1]
    assert rec["to"] == "startle" and rec["priority"] == [category("explore"), "hazard"]
