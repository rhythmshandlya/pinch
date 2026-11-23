"""Each state's timeout rule is carried out (configs/mind/states.toml timeout_do), "stop" and "go away" cancel any
resume, a startle backs away from where the fright came from, and a held leg is only played with by a friend in a
good mood."""

from pinch.activities.defensive import Startle
from pinch.activities.social import Obey, ReactTouch
from pinch.attention import Stimulus
from pinch.memory import Memory
from pinch.mind import Mind


def _mind():
    return Mind(memory=Memory(path=None), llm=False, seed=0, clock=lambda: 1.8e9)


def _overrun(m, name):
    m._start(name, "test")
    m.started = m.t - 10_000.0
    assert m._timeout()


def test_a_timed_out_job_leaves_its_object_alone_a_while():
    m = _mind()
    m._start("tidy", "test")
    m.tidy_current = "pen"
    m.started = m.t - 10_000.0
    assert m.state == "task" and m._timeout()
    assert not m.tidy_ok("pen")
    m.t += m.DROP_OBJECT_S + 31.0
    assert m.tidy_ok("pen")


def test_a_timed_out_recovery_asks_for_help():
    m = _mind()
    _overrun(m, "unstick")
    assert any(e for e in m.memory.data.get("episodes", []) if "help" in str(e)) or \
        any("couldn't manage" in str(v) for v in m.memory.data.values())


def test_without_resume_ok_nothing_comes_back_after_a_timeout():
    m = _mind()
    m.suspended, m.resume_stack = ("explore", m.t), [("tidy", m.t)]
    _overrun(m, "play")                                   # play: "choose again" only
    assert m.suspended is None and not m.resume_stack
    m.suspended = ("idle", m.t)
    _overrun(m, "explore")                                # explore: the interrupted activity may resume
    assert m.suspended is not None


def test_stop_and_go_away_cancel_any_resume():
    for word in ("stop", "go_away"):
        m = _mind()
        m.suspended, m.resume_stack = ("tidy", m.t), [("explore", m.t)]
        m.request = (word, "")
        list(Obey().run(m))
        assert m.suspended is None and not m.resume_stack, word


def test_a_startle_backs_away_from_the_fright():
    m = _mind()
    m.attention.focus = Stimulus("loud", m.t, 1.0, direction=1.2)    # off to its left
    walk = next(s for s in Startle().run(m) if s[0] == "walk")
    assert walk[1] < 0 and walk[2] < 0                                # back and to the right: away from it


def test_a_held_leg_is_a_game_only_for_a_friend_in_a_good_mood():
    m = _mind()
    m.affect.feel("praised")
    m.touch_event = ("held_leg", m.t)
    happy = list(ReactTouch().run(m))
    assert ("emote", "handshake") in [s[:2] for s in happy if s[0] == "emote"]
    m2 = _mind()
    m2.affect.feel("scolded")
    m2.touch_event = ("held_leg", m2.t)
    sad = list(ReactTouch().run(m2))
    assert not any(s[0] == "emote" and s[1] == "handshake" for s in sad) and any(s[0] == "pose" for s in sad)
