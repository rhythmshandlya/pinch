"""Touch nuance: where it was touched shapes the reaction (left / right, claw, back), repeated petting habituates
and recovers after a pause, and touches while asleep follow the wake rules (pinch/touch.py, ReactTouch)."""

import numpy as np

from pinch.activities.social import ReactTouch
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.touch import Touch, TouchGestures, TouchHabituation, WakeRules, touch_place


def _mind():
    return Mind(memory=Memory(path=None), llm=False, seed=0)


def _react(m, gesture):
    """Run ReactTouch on a fresh touch; returns its steps."""
    m.touch_event = (gesture, m.t)
    return list(ReactTouch().run(m))


def _emotes(steps):
    return [s for s in steps if s[0] == "emote"]


def test_gestures_carry_their_pads():
    g = TouchGestures()
    g.update(0.0, {"side_left"})
    out = g.update(0.1, set())
    assert out == ["tap"] and out[0].zones == ("side_left",) and touch_place(out[0])[:2] == (1, "side")
    out = []
    for t, pads in ((1.0, {"head"}), (1.2, {"head", "back_right"}), (1.4, {"back_right"}), (1.6, set())):
        out += g.update(t, pads)
    stroke = next(x for x in out if x == "stroke")
    assert touch_place(stroke)[:2] == (-1, "back")
    assert touch_place("stroke") == (0, "", None)            # a plain name: place unknown
    assert {Touch("tap", ["claw_left"]): 1}["tap"] == 1        # still a gesture name everywhere


def test_left_and_right_touch_turn_toward_the_side():
    m = _mind()
    left = _react(m, Touch("tap", ["side_left"]))
    m.t += 300
    right = _react(m, Touch("tap", ["side_right"]))
    (el,), (er,) = _emotes(left), _emotes(right)
    assert el[2]["gaze"] > 0 and er[2]["gaze"] < 0, "the body turns toward the touched side (+ = left)"
    assert el[2]["side"] == 1 and er[2]["side"] == -1
    looks = [s for s in left if s[0] == "look_at"], [s for s in right if s[0] == "look_at"]
    assert looks[0] and looks[1] and looks[0][0][1][1] > 0 > looks[1][0][1][1], "eyes to the touched side"
    # the body really mirrors the claw it answers with
    from crab.emotes import EmoteStyler
    s = EmoteStyler(np.random.default_rng(0))
    assert s.style("curious", side=-1).mirror != s.style("curious", side=1).mirror


def test_claw_touch_gets_a_claw_response():
    m = _mind()
    steps = _react(m, Touch("tap", ["claw_right"]))
    (e,) = _emotes(steps)
    assert e[1] == "wave" and e[2]["side"] == -1                 # the right claw waves back (wave mirrored)
    m.t += 300
    (e,) = _emotes(_react(m, Touch("claw_hold", ["claw_left"])))
    assert e[1] == "handshake" and e[2]["side"] == 1


def test_back_stroke_settles():
    m = _mind()
    steps = _react(m, Touch("stroke", ["head", "back_left"]))
    assert ("sound", "purr") in steps and any(s[0] == "pose" and s[1] == "low" for s in steps)


def test_repeated_strokes_habituate_and_recover():
    m = _mind()
    lows, emotes = [], []
    for _ in range(6):
        steps = _react(m, Touch("stroke", ["head", "back_left"]))
        lows.append(next(s[2] for s in steps if s[0] == "pose"))
        emotes.append(len(_emotes(steps)))
        m.t += 4.0
    assert emotes[0] == 1 and emotes[-1] == 0, "the excited wiggle wears off"
    assert lows[-1] > lows[0] + 1.0, "it settles longer as it relaxes into the petting"
    h = m.__dict__["touch_state"]["habit"]
    worn = h.response("stroke", m.t)
    assert worn < 0.5
    assert h.response("stroke", m.t + 240) > 0.9, "after a few minutes' pause it is fresh again"
    m.t += 240
    assert len(_emotes(_react(m, Touch("stroke", ["head", "back_left"])))) == 1


def test_habituation_characteristics():
    h = TouchHabituation()
    r = [h.stimulate("tap", t) for t in range(0, 20, 2)]
    assert all(b <= a for a, b in zip(r, r[1:])) and r[-1] >= h.FLOOR, "decrement toward an asymptote"
    slow = TouchHabituation()
    rs = [slow.stimulate("tap", t) for t in range(0, 200, 20)]
    assert rs[-1] > r[-1], "less frequent touch habituates less"
    weak, strong = TouchHabituation(), TouchHabituation()
    for t in range(5):
        weak.stimulate("tap", t)
        strong.stimulate("tickle", t)
    assert weak.response("tap", 5) < strong.response("tickle", 5), "weaker stimuli habituate faster"
    before = h.response("tap", 20)
    h.stimulate("ruffle", 20)
    assert h.response("tap", 20.5) > before + 0.2, "a rough touch dishabituates (and sensitises)"


def _asleep_mind(mode=None):
    m = _mind()
    if mode:
        m.memory.data.setdefault("settings", {})["wake_on_touch"] = mode
    from pinch.activities import LIBRARY
    m.activity = next(a for a in LIBRARY if a.name == "rest")
    m.face_expr = "asleep"
    return m


def test_sleep_wake_rules():
    m = _asleep_mind()                                 # default "gentle"
    react = ReactTouch()
    m.touch_event = ("stroke", m.t)
    assert react.utility(m) == 0.0, "a stroke while asleep doesn't wake it"
    m.touch_event = ("boop", m.t)
    assert react.utility(m) > 1.5, "a boop does"
    taps = []
    for _ in range(3):
        m.t += 2.0
        m.touch_event = ("tap", m.t)
        taps.append(react.utility(m) > 0)
        assert react.utility(m) == (taps[-1] and 1.7 or 0.0), "asked twice, the same answer"
    assert taps == [False, False, True], "one tap doesn't wake it, a few in a row do"
    m.touch_event = ("lifted", m.t)
    assert react.utility(m) > 0
    any_m = _asleep_mind("any")
    any_m.touch_event = ("stroke", any_m.t)
    assert react.utility(any_m) > 0, "'any': the old behaviour"
    deep = _asleep_mind("deep")
    deep.touch_event = ("boop", deep.t)
    assert react.utility(deep) == 0.0
    deep.touch_event = ("shaken", deep.t + 1)
    deep.t += 1
    assert react.utility(deep) > 0


def test_woken_by_touch_is_groggy_and_wake_pressure_fades():
    m = _asleep_mind()
    m.touch_event = ("boop", m.t)
    assert ReactTouch().utility(m) > 0
    steps = list(ReactTouch().run(m))
    assert steps[0] == ("face", "sleepy")
    w = WakeRules()
    assert not w.wakes(("tap", 0.0)) and not w.wakes(("tap", 2.0))
    assert not w.wakes(("tap", 200.0)), "taps far apart don't add up"


def test_awake_unchanged_without_zones():
    """A plain gesture name (twin buttons, HTTP API) still gets the reaction it always had."""
    m = _mind()
    steps = _react(m, "claw_hold")
    assert ("emote", "handshake") == _emotes(steps)[0][:2] and "side" not in _emotes(steps)[0][2]
    m.t += 300
    steps = _react(m, "stroke")
    assert ("face", "love") in steps and ("sound", "purr") in steps
