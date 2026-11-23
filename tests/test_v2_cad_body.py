"""The printable v2 body (cad/v2/body.py): the cheap checks of cad/v2/body_check.py. The full build (tray and cover,
about two minutes) runs with `python cad/v2/body_check.py`. Skipped (passes empty) when build123d is not installed."""

import functools
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAVE_CAD = importlib.util.find_spec("build123d") is not None


def _isolated(fn):
    """Runs the test only where the CAD library (build123d) is installed; passes empty elsewhere."""
    @functools.wraps(fn)
    def run():
        if HAVE_CAD:
            return fn()
    return run


def _modules():
    from cad.v2 import body as B
    from cad.v2 import body_check as C
    return B, C


@_isolated
def test_joint_ranges_come_from_the_model():
    B, _ = _modules()
    j = B.joints()
    assert set(j) == {"FL", "FR", "RL", "RR", "L", "R"}
    from crab.params import PARAMS
    lo, hi = j["FL"]["range"]
    want = PARAMS.leg.coxa_range_for("FL")
    assert math.isclose(lo, want[0], abs_tol=1e-3) and math.isclose(hi, want[1], abs_tol=1e-3)
    assert all(a < b for a, b in (v["world"] for v in j.values()))


@_isolated
def test_servo_interfaces():
    B, C = _modules()
    lo, hi = B.hip_case_aabb("FL")
    assert abs(lo[0] - 1.25) < 0.01 and abs(hi[0] - 46.65) < 0.01 and abs(lo[1] - 55.6) < 0.01
    lo_r, hi_r = B.hip_case_aabb("RL")
    assert hi_r[0] < 0 < lo[0]
    sh_lo, sh_hi = B.shoulder_case_aabb("L")
    assert abs(sh_lo[2] - (B.SHOULDER_Z0 - 13.75)) < 0.01 and abs(sh_hi[2] - (B.SHOULDER_Z0 + 13.75)) < 0.01
    assert sh_hi[2] < B.SPLIT_Z
    r = C.Report()
    C.check_interface(r)
    bad = [row for row in r.rows if row[2] == "FAIL"]
    assert not bad, bad


@_isolated
def test_boards_are_held():
    B, C = _modules()
    for name in B.STANDOFFS:
        posts, _ = B.standoffs(name)
        assert len(posts) >= 2 or C.stacked_on(name), f"{name}: {len(posts)} supports and nothing under it"


@_isolated
def test_finger_gaps():
    B, C = _modules()
    for key in B.LEGS + B.ARMS:
        side, face, slot = C.finger_gaps(key)
        assert side < C.SAFE_GAP and face < C.OPEN_GAP and (slot is None or slot >= C.OPEN_GAP), (key, slot)
