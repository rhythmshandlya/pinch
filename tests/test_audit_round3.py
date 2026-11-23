"""Third audit round (phase 4): a reflex can't be replaced by a lower layer, a slow turn at the goal is not "no
progress", letting go never snaps the claw, every decision says what made its scores, a recording starts with what
the mind starts from, expectation shapes how strongly an outcome is felt, and how close it comes depends on who."""

import numpy as np

from pinch.memory import Memory
from pinch.mind import Mind
from pinch.twin import PinchSim


def _sim(seed=3):
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    sim.autonomous = False
    sim.run(1.0)
    return sim


def test_a_reflex_is_never_replaced_by_a_lower_layer():
    sim = _sim()
    b = sim.body
    b.reflex(("recover",))
    b.start(("goto", (0.05, 0.0), 0.03), owner="tidy", layer="activity")       # the mind asks for something else
    assert b.claim.layer == "reflex" and b.pending is not None                 # it waits behind the reflex
    b.cancel()                                                                  # the mind changing its mind
    assert b.claim.layer == "reflex" and b.pending is None                     # ... drops what waited, not the reflex
    b.start(("emote", "wave"), owner="greet", layer="activity")
    for _ in range(600):
        sim.step()
        if b.claim is None or b.claim.layer != "reflex":
            break
    assert b.status == "emote"                                                  # what waited starts after it


def test_a_slow_turn_at_the_goal_is_not_no_progress():
    sim = _sim()
    b = sim.body
    b.grip_fn, b.held = None, ("pen", "right")                                  # carrying: turns slowly
    x, y, h = sim.pose()
    b.log = []
    b.start(("goto", (x + 0.02, y), 0.012, h + 2.5, 50.0, 0.5, 0.15))
    t0 = sim.t
    while b.busy and sim.t - t0 < 40:
        sim.run(0.1)
    assert b.ok and not b.blocked
    assert not [e for e in b.log if e[1] == "goto_no_progress"]


def test_letting_go_opens_the_claw_smoothly():
    sim = _sim()
    b = sim.body
    b.grip_fn, b.held = None, ("pen", "right")
    b.start(("release",))
    from crab.params import JOINT_NAMES
    j = list(JOINT_NAMES).index("R_claw")
    last, worst = None, 0.0
    while b.busy:
        q = b.tick()
        if last is not None:
            worst = max(worst, abs(q[j] - last))
        last = q[j]
    assert worst < 0.2, f"the claw jumped {worst:.2f} rad in one tick"


def test_every_decision_says_what_made_its_scores():
    m = Mind(memory=Memory(path=None), llm=False, seed=0)
    m.affect.needs.play = 0.9
    m.affect.needs.safety = 0.5
    m._choose()
    m._start("idle", "test")
    rec = m.decisions[-1]
    assert rec["why"] and all(isinstance(v, dict) for v in rec["why"].values())
    assert any("mood" in v for v in m.score_parts.values())                    # the fear damped something


def test_a_recording_starts_with_what_the_mind_starts_from(tmp_path=None):
    import tempfile
    from pathlib import Path

    from pinch.recorder import load_session
    with tempfile.TemporaryDirectory() as tmp:
        sim = PinchSim(memory=Memory(path=None), seed=4, record_dir=Path(tmp) / "s", record=dict(state=False))
        sim.run(2.0)
        sim.stop_recording()
        s = load_session(Path(tmp) / "s")
    start = [e for e in s["events"] if e["kind"] == "mind_start"]
    assert start and start[0]["seed"] == 4 and "learned" in start[0] and "personality" in start[0]
    assert "mood" in s["samples"] and "emotion" in s["samples"]


def test_an_unexpected_success_delights_more_than_an_expected_one():
    def proud_after(prior_wins):
        m = Mind(memory=Memory(path=None), llm=False, seed=0)
        m._start("tidy", "test")
        for _ in range(prior_wins):
            m._expectation("tidy", "pick", True)
        m.affect.emotions.clear()
        m.react_to_outcome(True, "pick")
        return m.affect.emotions.get("proud", 0.0)
    surprised = Mind(memory=Memory(path=None), llm=False, seed=0)
    surprised._start("tidy", "test")
    for _ in range(10):
        surprised._expectation("tidy", "pick", False)                          # it has learned picks usually fail
    surprised.affect.emotions.clear()
    surprised.react_to_outcome(True, "pick")
    assert surprised.affect.emotions.get("proud", 0.0) > 1.5 * proud_after(10)


def test_how_close_it_comes_depends_on_who():
    from pinch.activities.base import _owner_near

    def near(w, f, r, owner=False):
        m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 1.8e9)
        m.see_person("p", owner=owner)
        m.people.people["p"].update(W=w, F=f, R=r)
        return _owner_near(m)[1] - m.world.desk.owner_xy[1]
    loved, known, stranger = near(0.9, 1.0, 0.0, owner=True), near(0.3, 0.6, 0.0), near(0.0, 0.0, 0.0)
    assert loved < known < stranger


def test_idle_life_settles_after_a_long_quiet_spell():
    from pinch.body.life import IdleLife
    lf = IdleLife(np.random.default_rng(0))
    lf.liveliness = 0.8

    def amp(quiet):
        lf.quiet_s = quiet
        return float(np.mean([np.abs(lf.offsets(t * 0.02, 0.02)).max() for t in range(2000)]))
    assert amp(600.0) < amp(0.0)
