"""The mind-body link shared by the twin (pinch/twin/core.py) and the robot (robot/firmware/core.py): one place for
how the mind's choices reach the body and how the body's state reaches the mind, so a new sense or style input
(phases 6, 8, 9) is wired once. The hosts keep what differs: their sensors, the cliff beams, localisation.
The call order inside a tick is fixed here; the golden trace (tests/test_golden.py) guards it.
"""


def mind_step(mind, body, dt, t, recorder=None, on_motion=None, on_sound=None):
    """Advance the mind by dt and hand what it chose to the body: cancel, start a motion with its owner and layer,
    and the sounds to play (on_sound(name) for each). Returns the motion started or None."""
    motion, cancel = mind.tick(dt, body.busy, body.ok)
    if cancel:
        body.cancel()
        if recorder is not None:
            recorder.step_cancelled(t, "mind")
    if motion is not None:
        body.start(motion, *mind.motion_owner())
        if on_motion is not None:
            on_motion(motion)
        if recorder is not None:
            if body.pending is not None and body.pending[0] is motion:   # (waits behind a running reflex)
                recorder.step_queued(t, motion, "mind")
            else:
                recorder.step_started(t, motion, "mind")
    for s in mind.outputs()[2]:
        if on_sound is not None:
            on_sound(s)
    return motion


def foot_over_edge(body, margin, lifted, t, last_recover, recorder=None):
    """Reflex: a foot over the desk edge -> drop everything and shuffle back in (before anything else). Returns the
    time of this recovery if it started one, else last_recover."""
    if margin < -0.003 and not lifted and body.status != "recover" and t - last_recover > 3.0:
        body.reflex(("recover",))
        if recorder is not None:
            recorder.step_started(t, ("recover",), "edge reflex")
        return t
    return last_recover


def sync(mind, body):
    """Every tick, before the body moves: the mind's state that styles the body (liveliness, sleep, how long
    nothing has happened, fear and valence for the posture, the whole mood for emotes and walking), and the
    body's state the mind needs (a reflex running)."""
    life = body.life
    life.liveliness = mind.affect.arousal
    life.asleep = mind.state == "sleep"
    mind.body_reflex = body.claim is not None and body.claim.layer == "reflex"
    o = mind.world.owner                                   # nothing happening for a while: idle life settles
    life.quiet_s = mind.t - max(o.last_attention, o.last_voice, o.last_touch, 0.0)
    life.mood_in = (mind.affect.needs.safety, mind.affect.valence)
    if body.mood_fn is None:                               # the emote style follows the whole mood (decision 92)
        a = mind.affect
        body.mood_fn = lambda: dict(arousal=a.arousal, valence=a.valence, sleepy=a.needs.sleep,
                                    fear=a.needs.safety, fatigue=a.needs.fatigue)
