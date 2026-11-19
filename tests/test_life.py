"""The idle life layer (decision 74): a still Pinch keeps moving a little - varied micro-actions, never the same
one twice running, livelier when aroused - and none of it leaks into a motion step; three minutes idle in the
twin stay safe."""

import numpy as np

from pinch.body.life import ACTIONS, IdleLife
from pinch.memory import Memory
from pinch.sim_runner import PinchSim


def _events(liveliness, seconds=600.0, seed=0):
    life = IdleLife(np.random.default_rng(seed))
    life.liveliness = liveliness
    t, dt = 0.0, 0.02
    peak = 0.0
    while t < seconds:
        peak = max(peak, float(np.max(np.abs(life.offsets(t, dt)))))
        t += dt
    return life.events, peak


def test_varied_and_never_twice_running():
    ev, peak = _events(0.5)
    kinds = [a for _, a in ev]
    assert len(set(kinds)) >= 5, f"only {set(kinds)} in ten minutes"
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "the same micro-action twice running"
    assert peak <= 0.16, f"offsets too big: {peak:.3f} rad"
    assert set(kinds) <= set(ACTIONS)


def test_livelier_when_aroused():
    calm, _ = _events(0.1)
    excited, _ = _events(0.9)
    assert len(excited) > 1.3 * len(calm), (len(calm), len(excited))


def test_quiet_in_the_twin_and_off_during_steps():
    sim = PinchSim(memory=Memory(path=None), seed=3)
    sim.autonomous = False
    sim.run(180.0)
    assert sim.falls == 0
    assert len(sim.body.life.events) >= 8, f"only {len(sim.body.life.events)} micro-actions in 3 minutes"
    sim.body.start(("pose", "stand", 1.0))
    q_before = sim.body.crab.q_cmd.copy()
    sim.run(0.5)                                     # during a step the targets are the step's own
    assert sim.body.busy
    n = len(sim.body.life.events)
    sim.run(0.4)
    assert len(sim.body.life.events) == n, "micro-actions started during a motion step"
