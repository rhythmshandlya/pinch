"""Eyes on the attention target while walking (phase 4 exit check, PRD ARC-005/009): a toy moves slowly beside
Pinch while it walks about; the eyes follow it, re-aimed from the current pose every tick."""

import math

from pinch.memory import Memory
from pinch.twin import PinchSim

LIMIT = 0.15                    # mean gaze error (the eye's -1..1 range) while walking with the target in view


def _walking_gaze_error(seed):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=["toy"])
    sim.mind.memory.data["homes"] = {}
    sim.run(2.0)
    sim.move_prop("toy", [(0.30, 0.20), (-0.20, 0.20), (0.30, 0.20)], speed=0.03)
    errs = []
    for _ in range(300):
        sim.run(0.1)
        if sim.body.busy and sim.body.status in ("goto", "walk", "turn_to", "detour"):
            p, b = sim.pose(), sim.prop_xy("toy")
            br = (math.atan2(b[1] - p[1], b[0] - p[0]) - p[2] + math.pi) % (2 * math.pi) - math.pi
            if abs(br) < 0.8:
                errs.append(abs(sim.mind.outputs()[1][0] - max(-1.0, min(1.0, -br / 0.8))))
    return sum(errs) / len(errs), len(errs)


def test_the_eyes_stay_on_the_target_while_it_walks():
    for seed in (0, 2, 4):
        err, n = _walking_gaze_error(seed)
        assert n >= 50, f"seed {seed}: only {n} walking samples with the toy in view"
        assert err < LIMIT, f"seed {seed}: gaze off by {err:.3f} on average while walking (limit {LIMIT})"
