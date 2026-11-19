"""Exact replay: the twin with the same seed and the same inputs gives the same run, to the last decimal; a
different seed gives a different (valid) run. Everything that dates or randomises things in the brain reads the
simulation's clock and seeded generators (decision 61)."""

import numpy as np

from pinch.memory import Memory
from pinch.sim_runner import PinchSim


def _session(seed, seconds=40.0):
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    script = {5.0: "hi pinch, I'm Sam", 15.0: "dance!", 28.0: "I'm sad today"}
    acts = []
    while sim.t < seconds:
        for t_say, text in script.items():
            if abs(sim.t - t_say) < 1e-9 or (sim.t < t_say <= sim.t + 0.5):
                sim.say(text)
        sim.run(0.5)
        acts.append(sim.mind.activity.name if sim.mind.activity else None)
    episodes = [(round(e["t"], 6), e["kind"], e["text"]) for e in sim.mind.memory.data["episodes"]]
    return dict(acts=acts, motions=[(t, str(m)) for t, m in sim.motions], sounds=list(sim.sounds_played),
                episodes=episodes, qpos=sim.d.qpos.copy(), feeling=sim.mind.affect.feeling())


def test_same_seed_same_run():
    a, b = _session(3), _session(3)
    for k in ("acts", "motions", "sounds", "episodes", "feeling"):
        assert a[k] == b[k], f"{k} differs between two runs with the same seed"
    assert np.array_equal(a["qpos"], b["qpos"]), "the body ended in a different place with the same seed"


def test_different_seed_different_run():
    a, b = _session(3), _session(4)
    assert not np.array_equal(a["qpos"], b["qpos"]) or a["motions"] != b["motions"], \
        "two different seeds gave identical runs"
    assert a["episodes"] and all(t >= 1795059000.0 for t, *_ in a["episodes"]), \
        "memories should be dated on the simulation's calendar"
