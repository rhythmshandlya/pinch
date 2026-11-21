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
    L.chassis["box_half_mm"] = [67.7, 89.7, 22.9]                       # the old, lower box: the hip servos stand through the lid
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


def test_the_face_is_todays_visor_at_real_size():
    L = _layout()
    o, v = L.parts["oled"], L.parts["visor"]
    assert sorted(o.size[:2]) == [39.5, 61.5]                           # cad/parts.py FACE_MODULE, not scaled
    assert sorted(v.size[:2]) == [45.0, 67.5]                           # FACE_HOUSING
    n = o.R @ np.array([0.0, 0.0, 1.0])
    assert abs(np.degrees(np.arcsin(n[2])) - 25.0) < 1e-6               # FACE_TILT: the screen looks 25 degrees up
    assert o.pos[0] > 55.0 and o.pos[0] < 75.0                          # on the front, not the roof
    lower_edge = o.pos + (o.R @ np.array([o.half[0], 0.0, 0.0]))        # local x runs down the screen
    shoulder_top = L.parts["shoulder_l"].pos[2] + L.parts["shoulder_l"].half[2]
    assert shoulder_top < lower_edge[2] < shoulder_top + 6.0, (lower_edge, shoulder_top)    # just above the shoulder servos


def test_masses_and_the_centre_of_mass_are_sane():
    L = _layout()
    electronics = L.total_mass_g(lambda p: not p.in_model)
    assert 300 < electronics < 360, electronics
    com = L.com_mm(lambda p: not p.in_model)
    assert abs(com[0]) < 20 and abs(com[1]) < 8 and -25 < com[2] < 12, com
