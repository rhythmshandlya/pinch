"""The idle arm swing (pinch/body/life.py look) lets the camera see past the arms (sim/benches/arm_view.py): straight,
the arms hide the face band 10-30 deg to the side; over one swing those bearings are seen, the desk ahead stays in
view, and the arms keep clear of the front legs over the gait's hip range. It stops when asleep and shrinks when
afraid (the freeze)."""

import numpy as np

from crab.params import JOINT_NAMES


def test_the_swing_clears_the_view_beside_the_claws():
    from sim.benches.arm_view import measure
    r = measure(step_deg=2.0, swing_samples=4)
    straight, swing = r["straight"]["by_bearing"], r["swing"]["by_bearing"]
    assert straight[10] > 0.2 and straight[20] > 0.8, straight            # the problem as measured
    assert swing[10] < 0.05 and swing[20] < 0.35, swing                     # seen during the swing
    assert r["straight"]["desk_blocked"] <= 3 and r["out"]["desk_blocked"] <= 3
    assert r["arm_leg_gap_mm"] >= 3.0, f"the arms come within {r['arm_leg_gap_mm']:.1f} mm of the front legs"


def test_the_swing_rests_when_asleep_and_freezes_when_afraid():
    from pinch.body.life import LK, IdleLife
    yaw = [JOINT_NAMES.index("L_shoulder_yaw"), JOINT_NAMES.index("R_shoulder_yaw")]
    life = IdleLife(np.random.default_rng(0))
    ts = np.arange(0.0, LK["period_s"], 0.05)
    calm = max(life.look(t)[yaw].max() for t in ts)
    assert abs(calm - LK["arm_yaw_rad"]) < 0.01 and min(life.look(t)[yaw].min() for t in ts) >= 0.0
    life.asleep = True
    assert max(np.abs(life.look(t)).max() for t in ts) == 0.0
    life.asleep, life.mood_in = False, (1.0, 0.0)
    for _ in range(500):
        life.posture(0.02)
    assert max(life.look(t)[yaw].max() for t in ts) < 0.3 * calm
