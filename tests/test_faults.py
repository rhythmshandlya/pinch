"""Failure injection (decision 71): with each fault, two minutes of pet life with you there (talking, touching,
a thing out of place) stay safe - no falls, Pinch stays on the desk, the mind keeps choosing, the observer
finds no errors. A dead cliff beam blocks walking toward its side instead of failing open."""

import numpy as np

from pinch.memory import Memory
from pinch.observer import observe
from pinch.twin import PinchSim


def _run(fault, **kw):
    sim = PinchSim(memory=Memory(path=None), seed=4)
    sim.inject(fault, **kw)
    sim.place("eraser", (-0.05, 0.02))
    sim.mind.memory.teach_home("eraser", sim.desk.tray_xy)
    for _k, (t, what) in enumerate([(5, "hi pinch"), (20, "dance!"), (45, "stroke"), (70, "pinch, come here"),
                                   (95, "tickle")]):
        sim.run(t - sim.t)
        sim.mind.touch(what) if what in ("stroke", "tickle") else sim.say(what)
    sim.run(120.0 - sim.t)
    x, y, _ = sim.pose()
    errors = [f.evidence for f in observe(sim.mind) if f.severity == "error"]
    return sim, (sim.falls, sim.desk.inside((x, y), 0.0), len(sim.mind.decisions), errors)


def _safe(fault, **kw):
    sim, (falls, on_desk, n, errors) = _run(fault, **kw)
    assert falls == 0, f"{fault}: {falls} falls"
    assert on_desk, f"{fault}: Pinch left the desk"
    assert n >= 2, f"{fault}: the mind stopped choosing ({n} decisions)"
    assert not errors, f"{fault}: observer errors {errors}"
    return sim


def test_vision_lost():
    sim = _safe("vision")
    assert "eraser" not in sim.mind.world.objects or sim.mind.world.objects["eraser"].seen_t < 1.0


def test_flaky_detection():
    _safe("flaky_detection")


def test_audio_lost():
    sim = _safe("audio")
    assert not any(r["to"] == "obey" for r in sim.mind.decisions), "it obeyed words it couldn't have heard"


def test_dead_cliff_beam_blocks_its_side():
    sim = _safe("cliff_dead", beam="side_left")
    assert any(np.allclose(g, (0.0, 1.0)) for g in sim.cliff.blocked), \
        f"the dead left beam should block walking left; blocked {sim.cliff.blocked}"


def test_weak_leg():
    _safe("weak_leg", leg="RL", strength=0.5)
