"""The JAX gait (crab/gait_jax.py) matches the numpy gait the robot runs (crab/gait.py) over random commands,
for the trot and the crawl (decision 66). Needs JAX (.venv-rl); skipped where it isn't installed."""

import numpy as np


def _compare(name, steps=1500, dt=0.002, tol=2e-4):
    try:
        import jax
    except ImportError:                          # (the plain .venv has no JAX: run with .venv-rl)
        print("    (skipped: JAX not installed)")
        return None
    jax.config.update("jax_enable_x64", True)
    from crab.gait import GAITS, Gait
    from crab.gait_jax import GaitJax
    com = np.array([0.0078, 0.0])
    a = Gait(GAITS[name], com_offset=com)
    b = GaitJax(GAITS[name], com_offset=com)
    step = jax.jit(lambda s, c: b.step(s, dt, c))
    s = b.init()
    rng = np.random.default_rng(3)
    cmd = np.zeros(3)
    worst = 0.0
    for k in range(steps):
        if k % 300 == 0:
            cmd = rng.uniform([-0.06, -0.05, -0.8], [0.06, 0.05, 0.8]) * (rng.random() > 0.2)
        qa = a.step(dt, tuple(cmd))
        s, qb = step(s, np.asarray(cmd))
        worst = max(worst, float(np.max(np.abs(qa - np.asarray(qb)))))
    jax.config.update("jax_enable_x64", False)
    assert worst < tol, f"{name}: JAX gait differs by {worst:.2e} rad"
    return worst


def test_trot_matches():
    _compare("trot")


def test_crawl_matches():
    _compare("crawl")
