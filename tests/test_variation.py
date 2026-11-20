"""Emote variation (crab/emotes.py EmoteStyler, played by pinch/body/expression.py): every play is a variant of
the keyframes, never outside the joint limits or the servo speed limit, seeded (exact replay), and steered by the
mood in the right direction."""

import numpy as np

from crab.emotes import (
    EMOTES,
    STAND,
    EmoteStyler,
    Style,
    VariedEmote,
    arms_ok,
    joint_limits,
    peak_speed,
)
from crab.params import JOINT_NAMES, PARAMS

LEGS = ("FL", "FR", "RL", "RR")


def _q(emo, dt=0.02):
    return np.array([emo.target(t) for t in np.arange(0, emo.duration + dt, dt)])


def test_same_emote_twice_differs():
    s = EmoteStyler(np.random.default_rng(1))
    for name in EMOTES:
        a, b = s.vary(name), s.vary(name)
        st_a, st_b = a.style, b.style
        assert abs(st_a.tempo - st_b.tempo) >= 0.05 or abs(st_a.amplitude - st_b.amplitude) >= 0.05             or st_a.mirror != st_b.mirror, \
            f"{name} back to back: {a.style} vs {b.style}"
        n = min(len(_q(a)), len(_q(b)))
        assert np.abs(_q(a)[:n] - _q(b)[:n]).max() > 0.02, f"{name}: two plays look the same"


def test_sequences_do_not_repeat():
    s = EmoteStyler(np.random.default_rng(2))
    seq = ["wave", "dance", "wave", "dance", "wave", "dance"]
    styles = [s.style(n) for n in seq]
    assert styles[0:2] != styles[2:4] and styles[2:4] != styles[4:6]


def test_variants_stay_within_limits_and_speed():
    lo, hi = joint_limits()
    shin = PARAMS.leg.shin_min
    s = EmoteStyler(np.random.default_rng(3))
    moods = [dict(arousal=1, valence=1, side=1, gaze=0.15), dict(arousal=1, valence=1, side=-1, gaze=-0.15),
             dict(arousal=0, valence=-1, sleepy=1), dict(), dict(arousal=0.8, gain=0.2)]
    for name in EMOTES:
        for mood in moods:
            emo = s.vary(name, **mood)
            Q = _q(emo)
            assert (Q >= lo - 1e-9).all() and (Q <= hi + 1e-9).all(), f"{name} {emo.style} leaves a joint limit"
            fem = Q[:, [JOINT_NAMES.index(f"{lg}_femur") for lg in LEGS]]
            tib = Q[:, [JOINT_NAMES.index(f"{lg}_tibia") for lg in LEGS]]
            assert (fem + tib >= shin - 1e-9).all(), f"{name}: shin against the hip link"
            p = peak_speed(emo)
            assert p <= 1.0 and p <= s.allowed_speed(name) + 1e-6, f"{name} {emo.style}: {p:.2f} of the servo limit"
            assert np.allclose(Q[0], STAND, atol=1e-6) and np.allclose(Q[-1], STAND, atol=1e-6) or emo.loop, \
                f"{name}: a variant must start and end in the stand pose"
            assert arms_ok(emo, dt=0.08), f"{name} {emo.style}: the arms touch"


def test_seeded_replay_identical():
    def run(seed):
        s = EmoteStyler(np.random.default_rng(seed))
        return [s.style(n, arousal=0.6) for n in ("wave", "wave", "happy", "shy", "wave")]
    assert run(5) == run(5)
    assert run(5) != run(6)


def test_affect_modulates_tempo_and_amplitude():
    s = EmoteStyler(np.random.default_rng(4))
    t_lo, a_lo = s.mean(arousal=0.0)
    t_hi, a_hi = s.mean(arousal=1.0)
    assert t_hi > t_lo and a_hi > a_lo, "arousal: faster and bigger"
    t_sl, a_sl = s.mean(arousal=0.4, sleepy=1.0)
    t_aw, a_aw = s.mean(arousal=0.4, sleepy=0.0)
    assert t_sl < t_aw and a_sl < a_aw, "sleepy: slower and smaller"
    assert s.mean(valence=1.0)[1] > s.mean(valence=-1.0)[1], "happier: a little bigger"
    # through the whole chooser (jitter, anti-repetition, speed cap), averaged over many plays
    names = [n for n in EMOTES if n != "applause"] * 4
    hi = [s.style(n, arousal=1.0, valence=0.5) for n in names]
    lo = [s.style(n, arousal=0.0, sleepy=1.0) for n in names]
    assert np.mean([x.tempo for x in hi]) > np.mean([x.tempo for x in lo]) + 0.15
    assert np.mean([x.amplitude for x in hi]) > np.mean([x.amplitude for x in lo]) + 0.15
    assert s.mean(gain=0.2)[1] < s.mean(gain=1.0)[1], "a habituated reaction is smaller"


def test_mirror_and_side():
    s = EmoteStyler(np.random.default_rng(7))
    assert s.style("wave", side=-1).mirror and not s.style("wave", side=1).mirror      # wave is a left-claw emote
    assert s.style("handshake", side=1).mirror and not s.style("handshake", side=-1).mirror
    plain = _q(VariedEmote(EMOTES["wave"]))
    mirrored = _q(VariedEmote(EMOTES["wave"], Style(mirror=True)))
    from crab.emotes import MIRROR
    assert np.allclose(plain[:, MIRROR], mirrored)


def test_body_plays_variants_with_seeded_replay():
    """Through the twin's body: the same emote twice plays differently; the same seed replays exactly."""
    from pinch.memory import Memory
    from pinch.twin import PinchSim

    def styles(seed):
        sim = PinchSim(memory=Memory(path=None), seed=seed)
        sim.autonomous = False
        out = []
        for _ in range(2):
            sim.body.start(("emote", "wave"))
            while sim.body.busy:
                sim.step()
            out.append(sim.body.last_style)
        return out, sim.d.qpos.copy()
    a, qa = styles(11)
    b, qb = styles(11)
    assert a == b and np.array_equal(qa, qb)
    assert a[0][1] != a[1][1]
