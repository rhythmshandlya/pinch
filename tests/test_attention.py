"""Attention (decision 72): habituation, inhibition of return and hysteresis in the model; in the twin, a voice
off to the left turns Pinch toward it and its eyes follow, after which it goes back to what it was doing; and
a voice doesn't interrupt a job you asked for."""

import numpy as np

from pinch.attention import Attention
from pinch.memory import Memory
from pinch.sim_runner import PinchSim


def test_habituation_and_recovery():
    a = Attention()
    s1 = a.notice("voice", 0.0, 1.0, direction=1.2)
    s2 = a.notice("voice", 5.0, 1.0, direction=1.2)
    s3 = a.notice("voice", 10.0, 1.0, direction=1.2)
    assert s1 > s2 > s3, (s1, s2, s3)
    s4 = a.notice("voice", 400.0, 1.0, direction=1.2)      # long quiet: it's interesting again
    assert s4 > 0.9 * s1


def test_inhibition_of_return_and_hysteresis():
    a = Attention()
    a.notice("voice", 0.0, 1.0, direction=1.2)
    a.notice("new_thing", 0.0, 1.0, direction=-1.0)
    f, _ = a.update(0.1)
    assert f.kind == "voice"
    a.attended(f, 0.2)
    f2, _ = a.update(0.3)
    assert f2.kind == "new_thing", "after looking at the voice it should move on"
    a.notice("motion", 0.4, 0.95, direction=0.0)            # a bit weaker than the current focus: no flicker
    f3, _ = a.update(0.5)
    assert f3.kind == "new_thing"


def _turned(sim, seconds):
    h0 = sim.pose()[2]
    sim.run(seconds)
    d = (sim.pose()[2] - h0 + np.pi) % (2 * np.pi) - np.pi
    return d


def test_voice_from_the_left_turns_it_and_it_goes_back():
    sim = PinchSim(memory=Memory(path=None), seed=2)
    sim.run(3.0)
    sim.mind.attention.notice("voice", sim.mind.t, 1.0, direction=1.4)
    gaze = sim.mind.outputs()[1]
    turned = _turned(sim, 6.0)
    acts = [r["to"] for r in sim.mind.decisions]
    assert "orient" in acts, f"no orienting: {acts}"
    assert gaze[0] < -0.5, f"the eyes should look left first: gaze {gaze}"
    assert turned > 0.6, f"turned only {turned:.2f} rad toward the voice"
    sim.run(10.0)
    assert sim.mind.activity is None or sim.mind.activity.name != "orient", "it should move on after looking"


def test_a_voice_doesnt_interrupt_a_job_you_asked_for():
    sim = PinchSim(memory=Memory(path=None), seed=2)
    sim.place("cap", (-0.06, 0.0))
    sim.mind.world.see("cap", (-0.06, 0.0), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)
    sim.run(2.0)
    sim.say("pinch, put the cap away")
    sim.run(6.0)
    assert sim.mind.activity.name == "tidy"
    sim.mind.attention.notice("voice", sim.mind.t, 0.8, direction=1.4)
    sim.run(3.0)
    assert sim.mind.activity.name == "tidy", f"interrupted by a voice: now {sim.mind.activity.name}"
