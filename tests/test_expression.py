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
    m.face_intent = ("excited", m.t, m.started)      # fits a proud feeling too: shown for a while ...
    assert face_now(m) == "excited"
    m.t += 3.0
    assert face_now(m) == m.affect.feeling()          # ... then the feeling's own face again


def test_asleep_shows_asleep_whatever_it_feels():
    m = _mind()
    m._start("rest", "test")
    m.phase = "asleep"
    m.affect.feel("praised")
    assert m.state == "sleep" and face_now(m) == "asleep"


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


def test_its_own_person_stroking_is_welcome_even_with_a_feared_visitor_about():
    m = _mind()
    now = m.world.clock()
    for pid, w, r in (("visitor", -0.5, 0.9), ("me", 0.9, 0.0)):
        m.people.ensure(pid, now=now)
        m.people.people[pid].update(W=w, R=r, F=1.0 if pid == "me" else 0.5)
    m.people.seen["visitor"] = (now - 5, 1.0)
    m.people.seen["me"] = (now, 1.0)                              # its person came up to it last
    m.touch("stroke")
    faces = [s[1] for s in ReactTouch().run(m) if s[0] == "face"]
    assert "shy" not in faces and ("love" in faces or "content" in faces)


def test_lifted_near_a_feared_person_is_not_a_shrink_back():
    m = _mind()
    now = m.world.clock()
    m.people.ensure("p1", now=now)
    m.people.people["p1"]["R"] = 0.9
    m.people.seen["p1"] = (now, 1.0)
    m.touch_event = ("lifted", m.t)
    m.touch_from = "feared"
    assert not any(s[0] == "walk" for s in ReactTouch().run(m))   # no walking backwards in the air


def test_how_are_you_is_not_a_goal_outcome():
    m = _mind()
    m._start("tidy", "test")
    m._motion_kind = "drop"                                      # (left over from the tidy)
    m._start("obey", "test")
    m.gen = iter([("say_mood",)])
    m._advance(body_busy=False, body_ok=True)
    before = dict(m.affect.emotions)
    m._advance(body_busy=False, body_ok=True)                   # the reply emote finished
    assert m.affect.emotions.get("proud", 0) <= before.get("proud", 0)


def test_an_emergency_stop_is_not_a_success():
    m = _mind()
    m._start("tidy", "test")
    m.waiting_motion, m._motion_kind = True, "pick"
    m.motion_cancelled()
    assert not m.waiting_motion and not m.last_step_ok and m._lead_step is None


def test_dizzy_after_a_shake_or_a_fall():
    for ev in ("shaken", "fell"):
        m = _mind()
        m.affect.feel(ev)
        assert m.affect.feeling() == "dizzy", ev


def test_hiding_raises_the_claws():
    g = POSES["guard"]
    from pinch.body.common import J
    assert g[J["L_shoulder_pitch"]] > STAND[J["L_shoulder_pitch"]] + 0.3


def test_bed_and_wake_times_vary_by_day_and_each_morning_wakes_once():
    m = _mind()
    for off in (5.5, -8.0, 0.0):
        m.routine.settings["utc_offset_h"] = off
        wakes, beds = [], []
        for d in range(60):
            t0 = 1.8e9 + d * 86400.0
            flips = []
            prev = None
            for k in range(24 * 60):                               # every minute of a day
                now = m.routine.is_night(t0 + 60 * k)
                if prev is not None and now != prev:
                    flips.append((now, round(m.routine.hour(t0 + 60 * k), 2)))
                prev = now
            assert sorted(f[0] for f in flips) == [False, True], (off, d, flips)   # one wake, one bedtime
            wakes += [h for n, h in flips if not n]
            beds += [h for n, h in flips if n]
        assert len({round(w, 2) for w in wakes}) > 30 and all(abs(w - NIGHT[1]) <= 0.51 for w in wakes)
        assert all(abs(b - NIGHT[0]) <= 0.51 for b in beds)


def test_posture_and_freeze_follow_the_mood():
    import numpy as np

    from pinch.body.common import J as JJ
    from pinch.body.life import IdleLife
    life = IdleLife(np.random.default_rng(0))
    life.mood_in = (1.0, -0.2)                         # afraid
    for _ in range(500):
        q = life.posture(0.02)
    assert q[JJ["FL_femur"]] > 0.08 and q[JJ["L_claw"]] < -0.05      # crouched, claws drawn in
    life.mood_in = (0.0, 0.8)                          # happy
    for _ in range(1000):
        q = life.posture(0.02)
    assert q[JJ["FL_femur"]] < 0.0                     # a little taller

    def moves(fear):
        lf = IdleLife(np.random.default_rng(1))
        lf.liveliness, lf.mood_in = 0.8, (fear, 0.0)
        for _ in range(300):
            lf.posture(0.02)
        amp = [np.abs(lf.offsets(t * 0.02, 0.02)).max() for t in range(3000)]
        return float(np.mean(amp)), len(lf.events)
    calm, scared = moves(0.0), moves(0.9)
    assert scared[1] < calm[1] and scared[0] < calm[0]                 # it freezes: fewer, smaller movements


def test_repeated_harmless_bangs_scare_less_and_it_recovers():
    m = _mind()
    a = m.affect
    jumps = []
    for _ in range(5):
        s0 = a.needs.safety
        a.feel("loud_noise")
        jumps.append(a.needs.safety - s0)
        _age(m, 20.0)
    assert jumps[-1] < 0.6 * jumps[0] and jumps[-1] > 0.2 * jumps[0]     # wears off, never to nothing
    _age(m, 2400.0, dt=5.0)                                               # 40 quiet minutes
    s0 = a.needs.safety
    a.feel("loud_noise")
    assert a.needs.safety - s0 > 0.85 * jumps[0]                          # fresh again
    s0 = a.needs.safety
    a.feel("fell")
    a.feel("fell")
    assert a.needs.safety > s0                                            # a fall never habituates


def test_hello_depends_on_who_and_how_long_away():
    from pinch.activities.social import Greet

    def hello(who, away_h, warm):
        m = _mind()
        m.greet_who, m.greet_away_h, m.greet_warmth = who, away_h, warm
        return [s[1] for s in Greet().run(m) if s[0] in ("emote", "goto")][:3]
    quick, long_loved = hello("ana", 0.5, 0.3), hello("ana", 10.0, 0.9)
    assert quick == ["curious"] and long_loved[0] != "curious" and len(long_loved) >= 3
    sigs = {hello(p, 3.0, 0.4)[0] for p in ("ana", "ben", "chen", "dia", "eli", "fay")}
    assert len(sigs) >= 2                                            # people get their own hello


def test_asking_for_the_same_emote_again_gets_a_sibling():
    m = _mind()
    m._start("idle", "test")
    names = [m._vary_emote(("emote", "look_around"))[1] for _ in range(4)]
    assert names[0] == "look_around" and "curious" in names
    assert [m._vary_emote(("emote", "wave"))[1] for _ in range(3)] == ["wave"] * 3     # no family: as asked


def test_after_a_long_absence_it_remembers_what_you_like():
    from pinch.activities.social import Greet
    m = _mind()
    m.memory.data.setdefault("likes", {})["dance"] = 0.5
    m.greet_who, m.greet_away_h, m.greet_warmth = "ana", 30.0, 0.9
    steps = list(Greet().run(m))
    assert ("phase", "remember") in steps and steps[steps.index(("phase", "remember")) + 1] == ("emote", "dance")


def test_rare_quirks_are_rare_and_never_the_same_as_the_last_few():
    from pinch.activities.play import Quirk
    m = _mind()
    q = Quirk()
    assert q.utility(m) == 0.0                                  # not due straight away
    seen = []
    for _ in range(9):
        m.memory.data["quirk_due"] = m.world.clock() - 1
        m.affect.valence = 0.5
        assert q.utility(m) > 0
        steps = list(q.run(m))
        seen.append(steps[0][1])
        assert m.memory.data["quirk_due"] - m.world.clock() >= 8 * 3600     # the next one is hours away
    assert all(seen[i] not in seen[max(0, i - 3):i] for i in range(len(seen)))
    assert len(set(seen)) >= 5
