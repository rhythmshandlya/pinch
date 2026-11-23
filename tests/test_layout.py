"""Where every electronic part of the v2 robot sits (configs/layout_v2.toml) and the fit check (sim/fit_check.py): the real
layout fits, and a layout made bad on purpose fails in the way it should. Design v2's own process is in test_circuit.py."""

import copy

import numpy as np

from crab.electronics import Layout
from sim.fit_check import check


def _layout():
    return Layout()


def _kinds(layout):
    return check(layout).kinds()


def test_every_size_and_mass_has_a_source():
    L = _layout()
    assert L.problems() == [], "\n".join(L.problems())
    assert {p.d.get("size_source") for p in L.parts.values() if not p.mass_only} >= {"datasheet", "estimate"}
    # the parts the owner listed are all there
    for name in ("pack", "reg_a", "reg_b", "fuse", "switch", "driver", "radxa", "radxa_heatsink", "camera", "imu", "tof_fl",
                 "tof_fr", "tof_rear", "tof_left", "tof_right", "tof8", "mpr121", "pad_head", "oled", "mic_l", "mic_r", "amp",
                 "speaker", "pogo_l", "pogo_r", "dock_board", "hip_fl", "xt60", "ina", "hub", "visor"):
        assert name in L.parts, name


def test_the_real_layout_fits():
    rep = check(_layout())
    assert rep.ok, "\n".join(f"[{k}] {t}" for k, t in rep.failures)
    assert rep.min_gap[0] >= float(_layout().rules["clearance_mm"]) - 1e-6


def test_the_check_catches_overlap_clearance_and_parts_outside():
    L = _layout()
    L.parts["pack"].pos = L.parts["radxa"].pos.copy()                   # a battery inside the computer
    assert "overlap" in _kinds(L)
    L = _layout()
    L.parts["amp"].pos = L.parts["speaker"].pos + np.array([14.0 + 9.7 + 0.4, 0.0, 0.0])      # 0.4 mm apart: less than 1 mm
    assert "clearance" in _kinds(L)
    L = _layout()
    L.parts["hub"].pos = np.array([200.0, 0.0, -20.0])                  # a metre of wire away, outside the body
    assert "outside" in _kinds(L)
    L = _layout()
    L.chassis = copy.deepcopy(L.chassis)
    L.chassis["shell_half_mm"] = [84.0, 108.0, 38.0]                    # a flatter pill: the pack and the hip servos stand through it
    assert "outside" in _kinds(L)


def test_the_check_catches_a_connector_with_no_cable_room():
    L = _layout()
    r = L.parts["radxa"]
    L.parts["amp"].pos = r.pos + np.array([-(32.5 + 5.0), 0.0, 0.0])    # right where the USB-C cable bends
    rep = check(L)
    assert any(k == "cable room" and "radxa.usb_host" in t for k, t in rep.failures), rep.failures


def test_the_check_catches_blocked_windows_the_camera_a_beam_and_the_screen():
    L = _layout()
    lens = L.parts["camera_lens"]
    L.parts["pad_face"].pos = lens.pos + np.array([20.0, 0.0, 0.0])      # a part a few cm in front of the camera
    L.parts["pad_face"].size = np.array([2.0, 14.0, 14.0])
    assert any(k == "window" and "camera_lens" in t for k, t in check(L).failures)
    L = _layout()
    L.parts["speaker"].pos = L.parts["tof_left"].pos + np.array([0.0, 30.0, -10.0])   # in the way of the side beam
    assert any(k == "window" and "tof_left" in t for k, t in check(L).failures)
    L = _layout()
    o = L.parts["oled"]
    n = o.R @ np.array([0.0, 0.0, 1.0])
    L.parts["mic_l"].pos = o.pos + n * 60.0                             # something standing in front of the face
    L.parts["mic_l"].size = np.array([30.0, 30.0, 30.0])
    assert any(k == "window" and "oled" in t for k, t in check(L).failures)


def test_the_face_is_the_lid_with_the_6_inch_screen():
    L = _layout()
    o, v = L.parts["oled"], L.parts["visor"]
    assert sorted(o.size[:2]) == [76.0, 158.0] and o.mass_g == 300.0          # the 6.25 inch HDMI panel module (estimate), worst case
    assert o.d["size_source"] == "estimate" and "0.15 kg" in o.d["why"]
    assert sorted(v.size[:2]) == [82.0, 165.0]                                 # the lid wraps the module: about 165 x 82 mm
    n = o.R @ np.array([0.0, 0.0, 1.0])
    ch = L.chassis
    tilt = float(ch["lid_tilt_rad"])
    assert abs(np.degrees(np.arcsin(n[2])) - np.degrees(tilt)) < 0.05          # the screen looks up by the lid's tilt back from vertical
    hinge = np.asarray(ch["lid_hinge_mm"])
    along = (o.pos - hinge) @ np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    assert 35 < along < 50                                                     # the module's middle is above the hinge line
    assert o.d["window"]["exposed"] and o.d["connectors"][0]["through_wall"]    # the HDMI flat cable goes through the hinge


def test_the_lid_can_be_folded_flat():
    import os
    import subprocess
    import sys
    env = dict(os.environ, CRAB_LID_TILT="1.5")
    code = ("import sys; sys.path.insert(0, '.'); from crab.electronics import Layout; import numpy as np; L = Layout(); "
            "n = L['oled'].R @ np.array([0, 0, 1.0]); print(L.lid_tilt, float(n[2]))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    tilt, nz = (float(x) for x in out.stdout.split())
    assert tilt == 1.5 and nz > 0.99                                           # folded flat: the screen looks straight up


def test_the_shell_is_a_pill_and_the_windows_are_on_it():
    from crab.shapes import Shell
    L = _layout()
    sh = Shell.from_chassis(L.chassis)
    assert sh.f(np.array([6.9, 0.0, 0.3])) < 0.01 and sh.f(np.array([6.9 + 84.0, 0.0, 0.3])) == 1.0
    for name in ("tof_fl", "tof_fr", "tof_rear", "tof_left", "tof_right", "camera_lens", "tof8"):
        pos, _ = L[name].window_pose()
        deep = float(L[name].window.get("max_depth_mm", 4.5))       # a sensor in a bay's inner wall sits deeper
        assert -2.5 <= sh.depth(pos) <= deep, (name, sh.depth(pos))
    assert 0.08 < sh.area_m2() < 0.12                    # the v2.1 172 mm pill: 0.087 m2


def test_masses_and_the_centre_of_mass_are_sane():
    L = _layout()
    electronics = L.total_mass_g(lambda p: not p.in_model)
    assert 600 < electronics < 700, electronics                                # with the 300 g screen module and the 61 g lid
    com = L.com_mm(lambda p: not p.in_model)
    assert abs(com[0]) < 40 and abs(com[1]) < 8 and -25 < com[2] < 40, com
