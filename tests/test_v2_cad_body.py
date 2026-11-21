"""The printable v2 body (cad/v2/body.py): the cheap checks of cad/v2/body_check.py. The full build (tray and cover,
about two minutes) runs with `python cad/v2/body_check.py`. Skipped (passes empty) when build123d is not installed."""

import importlib.util
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAVE_CAD = importlib.util.find_spec("build123d") is not None


def _modules():
    os.environ["CRAB_DESIGN"] = "v2"
    from cad.v2 import body as B
    from cad.v2 import body_check as C
    return B, C


def test_joint_ranges_come_from_the_model():
    if not HAVE_CAD:
        return
    B, _ = _modules()
    j = B.joints()
    assert set(j) == {"FL", "FR", "RL", "RR", "L", "R"}
    lo, hi = j["FL"]["range"]
    assert math.isclose(math.degrees(lo), -55.0, abs_tol=0.1) and math.isclose(math.degrees(hi), 28.0, abs_tol=0.1)
    assert all(a < b for a, b in (v["world"] for v in j.values()))


def test_servo_interfaces():
    if not HAVE_CAD:
        return
    B, C = _modules()
    lo, hi = B.hip_case_aabb("FL")
    assert abs(lo[0] - 1.25) < 0.01 and abs(hi[0] - 46.65) < 0.01 and abs(lo[1] - 55.6) < 0.01
    lo_r, hi_r = B.hip_case_aabb("RL")
    assert hi_r[0] < 0 < lo[0]
    sh_lo, sh_hi = B.shoulder_case_aabb("L")
    assert abs(sh_lo[2] - 0.05) < 0.01 and abs(sh_hi[2] - 27.55) < 0.01 and sh_hi[2] < B.SPLIT_Z
    r = C.Report()
    C.check_interface(r)
    bad = [row for row in r.rows if row[2] == "FAIL"]
    assert not bad, bad


def test_boards_are_held():
    if not HAVE_CAD:
        return
    B, C = _modules()
    for name in B.STANDOFFS:
        posts, _ = B.standoffs(name)
        assert len(posts) >= 2 or C.stacked_on(name), f"{name}: {len(posts)} supports and nothing under it"


def test_finger_gaps():
    if not HAVE_CAD:
        return
    B, C = _modules()
    for key in B.LEGS + B.ARMS:
        side, face, slot = C.finger_gaps(key)
        assert side < C.SAFE_GAP and face < C.OPEN_GAP and (slot is None or slot >= C.OPEN_GAP), (key, slot)
