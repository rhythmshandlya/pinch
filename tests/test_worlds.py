"""Phase 5's worlds: seeded and reproducible, train and test kept apart, built into the twin, people, the fisheye
lens, scenario files and the training terrains (sim/worlds, sim/scenarios, training/mjlab_pinch/terrains.py)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.worlds import SPLITS, world, worlds  # noqa: E402
from sim.worlds.generate import TRAY_HALF  # noqa: E402


def test_splits_are_the_planned_sizes_and_reproducible():
    assert SPLITS["train"][1] == 1000 and SPLITS["test"][1] == 200
    a, b = world("test", 7), world("test", 7)
    assert a.to_dict() == b.to_dict(), "the same seed must give the same world"
    assert world("train", 7).layout_hash() != a.layout_hash()
    for bad in (("test", 200), ("train", -1), ("nope", 0)):
        try:
            world(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be refused")


def test_test_worlds_are_held_out():
    """No held-out test world is the same place as any training world (layout hashes, rounded to cm)."""
    train = {w.layout_hash() for w in worlds("train")}
    test = [w.layout_hash() for w in worlds("test")]
    assert len(train) == 1000 and len(set(test)) == 200
    assert not train & set(test), "a test world is also a training world"


def test_worlds_vary_in_what_matters():
    from sim.worlds.generate import use_procthor
    ws = [world("train", i) for i in range(120)]
    kinds = {w.kind for w in ws}
    assert kinds == {"desk", "table", "floor"}, kinds
    assert {1, 2, 3} <= {len(w.rooms) for w in ws}       # (ready-made ProcTHOR homes: up to 10 rooms)
    things = [len(w.things) for w in ws]
    assert min(things) == 0 and max(things) >= 8
    sizes = {round(w.surface["half_x"], 2) for w in ws if w.kind == "desk"}
    assert len(sizes) > (8 if use_procthor() else 20), f"desks of many sizes: {sorted(sizes)}"
    assert {p["style"] for w in ws for p in w.people} == {"indifferent", "curious", "fearful", "playful"}
    assert any(p["child"] for w in ws for p in w.people)
    levels = [w.extra["level"] for w in ws]
    assert min(levels) < 0.45 and max(levels) > 1.1, "dim evenings and bright days"
    from pinch.world import Desk
    from sim.worlds.catalogue import SURFACE_MIN
    stance = Desk.body_radius()
    for w in ws:                                        # Pinch's start, the tray and its things fit on its surface
        hx, hy = w.surface["half_x"], w.surface["half_y"]
        if w.kind != "floor":                           # room for the stance, the cliff beams clear of the edge
            assert 2 * min(hx, hy) >= SURFACE_MIN[1] - 1e-6 and 2 * max(hx, hy) >= SURFACE_MIN[0] - 1e-6
        assert abs(w.crab[0]) < hx - stance - 0.05 and abs(w.crab[1]) < hy - stance - 0.05
        for x, y, _ in w.named.values():                # its desk things not under its legs
            assert np.hypot(x - w.crab[0], y - w.crab[1]) > stance
        assert abs(w.tray_xy[0]) + TRAY_HALF[0] < hx and abs(w.tray_xy[1]) + TRAY_HALF[1] < hy
        for x, y, _ in w.named.values():
            assert abs(x) < hx and abs(y) < hy


def test_cliff_beams_see_a_drawn_desk_top_headless():
    # a ready-made desk drawn instead of the twin's desk top: headless, the top must still be there for the cliff
    # beams (it was made transparent, and rangefinders skip alpha-0 geoms: the edge guard then held Pinch still)
    import copy

    from pinch.memory import Memory
    from pinch.twin import PinchSim
    w = copy.deepcopy(next(world("test", i) for i in range(200) if world("test", i).kind == "desk"))
    w.surface["model"] = dict(asset="none", z=0.0, off=[0.0, 0.0], yaw=0.0)     # (not drawn headless: no fetch needed)
    sim = PinchSim(memory=Memory(path=None), world=w, world_opts=dict(visual=False, people=False, fisheye=False))
    sim.run(0.5)
    front = [v for k, v in sim.cliff_mm().items() if k.startswith("front")]
    assert all(v < 150 for v in front), f"the front cliff beams see no desk under them: {front} mm"


def test_a_world_runs_in_the_twin():
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    w = next(world("test", i) for i in range(200) if world("test", i).kind == "floor" and world("test", i).things)
    sim = PinchSim(memory=Memory(path=None), world=w, world_opts=dict(visual=False, people=False, fisheye=False))
    assert sim.desk.half_x == w.surface["half_x"] and sim.mind.world.desk is sim.desk
    sim.run(2.5)
    x, y, _ = sim.pose()
    assert sim.falls == 0 and abs(x - w.crab[0]) < 0.15 and abs(y - w.crab[1]) < 0.15
    assert sim.d.qpos[2] > 0.02, "standing on its surface"


def test_people_move_and_touch():
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from sim.worlds.live import Life
    w = next(world("test", i) for i in range(200) if world("test", i).kind == "desk"
             and any(p["role"] == "sitter" for p in world("test", i).people))
    sim = PinchSim(memory=Memory(path=None), world=w, world_opts=dict(visual=False, fisheye=False))
    life = Life(sim)
    sitter = next(p for p in life.people if p.p["role"] == "sitter")
    q = sim.d.qpos                                      # Pinch near the front edge, within the person's reach
    q[0], q[1] = sitter.base[0], -w.surface["half_y"] + 0.12
    q[3:7] = (np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4))          # facing them (-y)
    sim.autonomous = False
    life.run(1.0)
    sitter.reach_now("stroke")
    life.run(8.0)
    assert any(g == "stroke" for _, i, g in life.touches if i == sitter.i), f"no stroke: {life.touches}"
    assert life.seen, "a person facing Pinch counts as seen"


def test_fisheye_lens_geometry():
    from sim.worlds.fisheye import build_maps, default_K
    K = default_K()
    assert abs(K[0, 0] - 286.5) < 0.5, "160 degrees across the 640 x 480 diagonal"
    mx, my = build_maps(K, None, (640, 480), 64)
    assert mx[240, 320] >= 0 and 0 <= mx[240, 320] < 64, "the image centre comes from the front face"
    assert (mx >= 0).mean() > 0.99, "the whole 160-degree image is covered by the cube faces"
    assert 64 * 3 <= mx[240, 2] < 64 * 4 or 64 <= mx[240, 2] < 64 * 3, "the far left/right edge comes from a side face"


def test_scenario_files():
    from sim.worlds.scenario import STATE, TRACE, _pred, all_scenarios, timeline
    scs = all_scenarios()
    assert len(scs) >= 13, "the behaviour scenarios, converted"
    rng = np.random.default_rng(0)
    for sc in scs:
        tl, times = timeline(sc, rng, {k: "x" for k in sc.vars})
        assert [t for t, _ in tl] == sorted(t for t, _ in tl)
        assert all(0 <= t <= sc.seconds for t in times.values()), sc.name
        for a in sc.asserts:
            (name,) = _pred(a).keys()
            assert name in STATE or name in TRACE, f"{sc.name}: unknown predicate {name}"
            for k in ("after", "until"):
                assert k not in a or a[k] in times, f"{sc.name}: {a[k]} is not an event"


def test_a_scenario_runs():
    from sim.worlds.scenario import load, run
    r = run(load(ROOT / "sim" / "scenarios" / "memory_restart.toml"), seed=1)
    assert r["error"] is None and r["ok"], r
    assert r["world"].startswith("test-")


def test_training_terrains_are_pinch_sized():
    from training.mjlab_pinch import terrains as T
    assert abs(sum(p for p, _, _ in T.FAMILIES.values()) - 1.0) < 1e-9
    assert T.FAMILIES["flat"][0] == max(p for p, _, _ in T.FAMILIES.values()), "flat ground stays the biggest share"
    assert T.FAMILIES["thresholds"][2]["step_height_range"][1] <= 0.04, "thresholds a 15 cm robot can step"
    assert T.FAMILIES["slopes"][2]["slope_range"][1] <= 0.4, "ramps up to ~20 degrees"
    assert T.TILE[0] / 2 <= 0.06 * 20, "half a tile is walkable in one episode at top speed"


def test_ready_made_people_fit_the_posing():
    """Rocketbox's Biped skeleton folds into the bones the people's motion drives (humans.pose / lengths)."""
    from sim.worlds import rocketbox as R
    assert set(R.NAMES.values()) == set(R.PARENT) and all(v is None or v in R.PARENT for v in R.PARENT.values())
    for b in ("thigh_l", "calf_l", "foot_l", "upperarm_r", "lowerarm_r", "hand_r", "neck_01", "head", "pelvis"):
        assert b in R.PARENT
    assert R._fold("Bip01 L Finger21") == "hand_l" and R._fold("Bip01 R Toe0") == "foot_r"
    assert R._fold("Bip01 MJaw") == "head" and R._fold("Bip01 Footsteps") is None and R._fold("Bip01") is None
    if R.available():                                   # (when fetched) left is +x, facing -y, scaled to height
        g = R.geometry(R.ADULTS[0], 1.6)
        assert g["head"]["upperarm_l"][0] > 0 > g["head"]["upperarm_r"][0]
        top = max(m["V"][:, 2].max() for m in g["meshes"])
        assert abs(top - 1.6) < 0.01 and 0.05 < g["face_up"] < 0.2


def test_ready_made_homes_convert():
    """ProcTHOR houses -> rooms, walls (each once, doors and windows along it), Pinch's frame; no download needed."""
    from sim.worlds import procthor as P
    pt = lambda x, z, y=0.0: {"x": x, "y": y, "z": z}
    L = [pt(0, 0), pt(4, 0), pt(4, 2), pt(2, 2), pt(2, 4), pt(0, 4)]           # an L-shaped room (Unity x, z)
    h = dict(rooms=[dict(roomType="Kitchen", floorPolygon=L)],
             walls=[dict(id="w1", polygon=[pt(0, 0), pt(4, 0), pt(0, 0, 2.5), pt(4, 0, 2.5)]),
                    dict(id="w1b", polygon=[pt(4, 0), pt(0, 0), pt(4, 0, 2.5), pt(0, 0, 2.5)])],
             doors=[dict(wall0="w1", holePolygon=[pt(1.0, 0, 0), pt(2.0, 0, 2.0)])],
             windows=[dict(wall0="w1b", holePolygon=[pt(0.5, 0, 0.9), pt(1.5, 0, 2.0)])],
             objects=[dict(id="Desk|1|0", assetId="Desk_1", position=pt(1, 3, 0.4), rotation={"x": 0, "y": 90, "z": 0})])
    c = P._compact(h)
    o = c["objects"][0]
    assert (o["type"], o["x"], o["y"], o["z"]) == ("Desk", 1, 3, 0.4) and abs(o["yaw"] + np.pi / 2) < 1e-4
    walls = P._walls(c)
    assert len(walls) == 1, "a wall listed from both sides is built once"
    (door,), (win,) = walls[0]["doors"], walls[0]["windows"]
    assert abs(door[0] - 1.5) < 1e-6 and abs(door[1] - 1.0) < 1e-6
    assert abs(win[0] - 3.0) < 1e-6 and win[2:] == (0.9, 2.0), "the window, seen from the other side, is at 3 m"
    cells, rect = P._cells(c["rooms"][0]["poly"])
    assert abs(sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in cells) - 12.0) < 1e-6
    assert rect in ((0, 0, 4, 2), (0, 0, 2, 4)), "the largest rectangle of the L"
