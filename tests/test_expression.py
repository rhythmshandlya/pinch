"""One inner state drives what Pinch shows (phase 4, part 2; configs/mind/expression.toml): emotions fade at
their own pace into a mood, the face shows the feeling (an activity's look only when it fits), sounds, walking
tempo and wants follow the feeling, and touch, recovery and the daily rhythm answer to it."""

from types import SimpleNamespace

from pinch.activities.social import ReactTouch
from pinch.body.common import POSES, STAND
from pinch.body.locomotion import Locomotion
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.mind.expression import face_now, fits, sound_for
from pinch.routine import NIGHT


def _mind():
    return Mind(memory=Memory(path=None), llm=False, seed=0)


def _age(m, seconds, dt=0.5):
    for _ in range(int(seconds / dt)):
        m.t += dt
        m.affect.tick(dt)


def test_emotions_fade_at_their_own_pace():
    m = _mind()
    m.affect.feel("loud_noise")                      # surprised (3 s half-life) and scared
    m.affect.feel("scolded")                         # sad (60 s)
    s0, d0 = m.affect.emotions["surprised"], m.affect.emotions["sad"]
    _age(m, 6.0)
    assert m.affect.emotions.get("surprised", 0) < 0.3 * s0      # startle is gone in seconds
    assert m.affect.emotions["sad"] > 0.85 * d0                  # sadness lingers


def test_a_feeling_stays_on_screen_long_enough():
    m = _mind()
    m.affect.feel("praised")
    first = m.affect.feeling()
    m.affect.emotions["curious"] = m.affect.emotions[first] + 0.02    # barely stronger, right away
    assert m.affect.feeling() == first                                 # no flicker


def test_the_face_shows_the_feeling_not_the_script():
    m = _mind()
    m._start("idle", "test")
    m.affect.feel("scolded")
    m.affect.feel("scolded")
    feeling = m.affect.feeling()
    assert feeling in ("sad", "shy")
    m.face_intent = ("happy", m.t, m.started)        # an activity asks for a happy look while it is sad
    assert face_now(m) != "happy" and fits(face_now(m), feeling)
    m.affect.emotions.clear()
    m.affect.feel("praised")
    m.face_intent = ("happy", m.t, m.started)
    assert face_now(m) == "happy"                     # fits a proud feeling: shown
    m.t += 3.0
    assert face_now(m) == m.affect.feeling() or fits(face_now(m), m.affect.feeling())   # then the feeling again


def test_play_dead_and_sleep_keep_their_faces():
    m = _mind()
    m._start("idle", "test")
    m.affect.feel("scolded")
    m.face_intent = ("dead", m.t, m.started)
    m.t += 10.0
    assert face_now(m) == "dead"


def test_sounds_follow_the_feeling():
    assert sound_for("giggle", "scared") != "giggle"
    assert sound_for("whimper", "happy") != "whimper"
    assert sound_for("happy_trill", "happy") == "happy_trill"
    assert sound_for("happy_trill", "sad") == "sad_tone"


def test_walking_tempo_follows_the_mood():
    def tempo(**mood):
        return Locomotion.walk_tempo(SimpleNamespace(mood_fn=lambda: mood))
    happy = tempo(arousal=0.6, valence=0.6)
    scared = tempo(arousal=0.8, valence=-0.4, fear=0.8)
    tired = tempo(arousal=0.2, valence=0.1, fatigue=0.8, sleepy=0.6)
    sad = tempo(arousal=0.2, valence=-0.5)
    assert scared >= happy > sad > tired
    assert happy / tired >= 1.3 and scared / tired >= 1.3


def test_no_play_or_showing_off_while_scared():
    m = _mind()
    from pinch.mind.selection import WANTS
    calm = m._mood_gate(WANTS["play"])
    m.affect.needs.safety = 0.6
    assert m._mood_gate(WANTS["play"]) < 0.3 * calm
    assert m._mood_gate(WANTS["shy"]) > 1.2


def test_one_line_for_fear():
    from pinch.activities.defensive import FEAR, Startle
    m = _mind()
    m.affect.needs.safety = FEAR + 0.01
    assert m.affect.feeling() == "scared" and Startle().utility(m) > 0


def test_a_pet_from_someone_feared_is_shrunk_from():
    m = _mind()
    now = m.world.clock()
    m.people.ensure("p1", name="", owner=False, now=now)
    m.people.people["p1"]["R"] = 0.9
    m.people.seen["p1"] = (now, 1.0)
    m.touch("stroke")
    steps = list(ReactTouch().run(m))
    faces = [s[1] for s in steps if s[0] == "face"]
    assert "love" not in faces and ("shy" in faces or "grumpy" in faces)
    assert any(s[0] == "walk" and s[1] < 0 for s in steps)        # backs away from the hand


def test_dizzy_after_a_shake():
    m = _mind()
    m.affect.feel("shaken")
    assert m.affect.feeling() == "dizzy"


def test_hiding_raises_the_claws():
    g = POSES["guard"]
    from pinch.body.common import J
    assert g[J["L_shoulder_pitch"]] > STAND[J["L_shoulder_pitch"]] + 0.3


def test_bed_and_wake_times_vary_by_day():
    m = _mind()
    wakes = {round(m.routine.night_window(d * 86400.0 + 12 * 3600)[1], 2) for d in range(20000, 20010)}
    assert len(wakes) > 5 and all(abs(w - NIGHT[1]) <= 0.5 for w in wakes)
