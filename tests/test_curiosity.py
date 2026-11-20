"""Curiosity about unknown things and motion (pinch/novelty.py, pinch/activities/curiosity.py): the model on its own
(habituation, wariness, the Wundt curve, proportionate startle), then in the twin: a new thing on the desk makes
Pinch look, creep closer slowly, keep its distance at first, see it from more than one side, back off, come back
bolder and get used to it; its eyes follow a moving thing; a thing rushing at it startles it and a slow one far
away doesn't; it never puts a foot over the edge or plans a spot it fears; quiet hours keep it in place."""

import math

from pinch.memory import Memory
from pinch.novelty import NoveltyTracker, Observation, interest
from pinch.safety import CAREFUL
from pinch.twin import PinchSim

POSE0 = (0.0, 0.0, 0.0)


def _sim(props, seed=3):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=props)
    sim.mind.memory.data["homes"] = {}          # (no tidying in these tests: nothing is out of place)
    sim.run(2.0)
    return sim


def _dist(sim, name):
    p, b = sim.pose(), sim.prop_xy(name)
    return math.hypot(p[0] - b[0], p[1] - b[1])


def _bouts(sim):
    return [ln for ln in sim.mind.log if ln[1] == "curious" and ln[2].startswith("looked at")]


# -- the model ------------------------------------------------------------------------------------
def test_habituation_wariness_and_memory():
    nt = NoveltyTracker(timidity=0.5)
    ev = nt.observe(Observation("unknown", (0.2, 0.0), (0.0, 0.0), 0.04, 0.9, "box:40"), 1.0, POSE0, now=0.0)
    assert ev[0][0] == "new"
    th = ev[0][1]
    stand = [nt.standoff(th)]
    for k in range(6):
        nt.end_bout(th, 0.2, [0.0, 1.0], True, 10.0 + k)
        stand.append(nt.standoff(th))
        if th.familiar:
            break
    assert 3 <= th.bouts <= 4, f"familiar after {th.bouts} looks (want 3-4)"
    assert all(a > b for a, b in zip(stand, stand[1:])), f"it should dare closer each time: {stand}"
    assert stand[0] - stand[-1] > 0.06
    # the same thing taken away and brought back: still familiar; a day later partly; a different thing: new
    nt2 = NoveltyTracker(familiar=nt.familiar, timidity=0.5)
    again = nt2.observe(Observation("unknown", (-0.1, 0.1), None, 0.04, 0.9, "box:40"), 0.0, POSE0, now=60.0)[0][1]
    assert again.familiar
    nt3 = NoveltyTracker(familiar=nt.familiar, timidity=0.5)
    later = nt3.observe(Observation("unknown", (-0.1, 0.1), None, 0.04, 0.9, "box:40"), 0.0, POSE0,
                        now=10 * 86400.0)[0][1]
    assert later.novelty > 0.5, "after ten days it should be fairly new again (spontaneous recovery)"
    other = nt2.observe(Observation("unknown", (0.1, -0.1), None, 0.07, 0.9, "cylinder:30"), 0.1, POSE0,
                        now=60.0)[0][1]
    assert other.novelty == 1.0 and other is not again


def test_wundt_curve():
    assert interest(1.0, 0.6) > 0.9                       # new, a little wary: go and look
    assert interest(0.1, 0.1) < 0.2                       # familiar: not interesting
    assert interest(1.0, 0.98) < 0.1                      # far too scary: only watch it


def test_startle_is_proportionate():
    def peak(xy0, vel, steps=8):
        nt = NoveltyTracker(timidity=0.5)
        total, xy = 0.0, list(xy0)
        for k in range(steps):
            for ev in nt.observe(Observation("unknown", tuple(xy), vel, 0.04, 0.9, "toy"), 0.1 * k, POSE0):
                if ev[0] == "sudden":
                    total += ev[2]
            xy[0] += vel[0] * 0.1
            xy[1] += vel[1] * 0.1
        return total
    rushing = peak((0.30, 0.0), (-0.4, 0.0), steps=5)      # comes at it fast, ends 10 cm away
    passing = peak((0.15, -0.15), (0.0, 0.3))              # fast, close, but passing by
    slow_far = peak((0.35, -0.15), (0.0, 0.05))            # slow, far
    assert rushing > 0.7, rushing
    assert 0.0 < passing < rushing, (passing, rushing)
    assert slow_far == 0.0


# -- in the twin --------------------------------------------------------------------------------
def test_new_thing_cautious_look_and_habituation():
    sim = _sim(["box"])
    t_show = sim.t
    sim.show_prop("box", (0.15, 0.0))                    # 15 cm in front: closer than it likes at first
    speeds, near_edge = [], 1.0
    last, n_mo, slows = sim.pose(), len(sim.motions), []
    while sim.t < t_show + 200.0:
        sim.run(0.2)
        p = sim.pose()
        for mo in sim.motions[n_mo:]:
            if mo[1][0] == "goto" and sim.mind.phase in ("approach", "inspect"):
                slows.append(mo[1][5])
        n_mo = len(sim.motions)
        if sim.mind.activity and sim.mind.activity.name == "investigate" and sim.mind.phase == "approach":
            speeds.append(math.hypot(p[0] - last[0], p[1] - last[1]) / 0.2)
        last = p
        near_edge = min(near_edge, sim.desk.feet_margin(p[:2], p[2]))
        th = next(iter(sim.mind.novelty.things.values()), None)
        if th is not None and th.familiar and sim.mind.activity.name != "investigate":
            break
    recs = [(r["t"], r["to"]) for r in sim.mind.decisions]
    first = next((t for t, a in recs if a == "investigate"), None)
    assert first is not None and first - t_show < 2.0, f"it should go and look at once: {recs}"
    phases = [p for _, a, p in sim.mind.phases if a == "investigate"]
    order = [p for p in ("notice", "approach", "pause", "inspect", "retreat") if p in phases]
    assert order == ["notice", "approach", "pause", "inspect", "retreat"], f"phases {phases}"
    assert slows and max(slows) <= 0.7, f"approach should be slow: {slows}"
    assert speeds and max(speeds) < 0.045, f"crept at up to {max(speeds):.3f} m/s"
    th = next(iter(sim.mind.novelty.things.values()))
    assert th.closest[0] >= 0.15, f"first look from {th.closest[0]:.3f} m: it should keep its distance"
    assert th.bouts >= 3 and th.familiar, f"{th.bouts} looks, novelty {th.novelty:.2f}"
    assert th.closest[-1] < th.closest[0] - 0.02, f"closer as it gets used to it: {th.closest}"
    sides = max(int(s.split(" side")[0].split()[-1]) for _, _, s in _bouts(sim))
    assert sides >= 2, f"seen from {sides} side(s): {_bouts(sim)}"
    assert near_edge > 0.0 and sim.edge_recovers == 0 and sim.falls == 0, (near_edge, sim.edge_recovers, sim.falls)
    # familiar now: it leaves it alone; taken away and put back elsewhere it is still familiar
    t1 = sim.t
    sim.run(30.0)
    again = [(r["t"], r["to"], r["reason"]) for r in sim.mind.decisions if r["t"] > t1]
    assert "investigate" not in [a for _, a, _ in again], f"familiar, yet looked again: {again}"
    sim.hide_prop("box")
    sim.run(5.0)
    p = sim.pose()
    a = p[2] + (0.5 if p[1] < 0 else -0.5)              # somewhere else, in view
    t2 = sim.t - 0.1
    sim.show_prop("box", sim.desk.clamp((p[0] + 0.2 * math.cos(a), p[1] + 0.2 * math.sin(a)), 0.06))
    sim.run(20.0)
    assert not [r for r in sim.mind.decisions if r["t"] > t2 and r["to"] == "investigate"], "it knows this box"
    assert any("a familiar" in ln[2] for ln in sim.mind.log if ln[1] == "curious" and ln[0] >= t2)


def test_it_goes_back_to_what_it_was_doing():
    sim = _sim(["box"])
    sim.mind.affect.needs.curiosity = 0.95
    sim.mind._start("explore", "test")
    sim.run(1.0)
    sim.show_prop("box", (0.15, 0.0))
    sim.run(60.0)
    recs = [(r["to"], r["reason"]) for r in sim.mind.decisions]
    i = next((k for k, (a, _) in enumerate(recs) if a == "investigate"), None)
    assert i is not None, recs
    assert any(a == "explore" and "(resumed)" in why for a, why in recs[i:]), f"not resumed: {recs}"


def test_eyes_follow_a_moving_thing():
    sim = _sim(["toy"])
    sim.move_prop("toy", [(0.30, 0.16), (0.30, -0.16), (0.30, 0.16)], speed=0.06)
    errs, gaze = [], []
    for _ in range(100):
        sim.run(0.1)
        p, b = sim.pose(), sim.prop_xy("toy")
        br = (math.atan2(b[1] - p[1], b[0] - p[0]) - p[2] + math.pi) % (2 * math.pi) - math.pi
        g = sim.mind.outputs()[1][0]
        errs.append(abs(g - max(-1.0, min(1.0, -br / 0.8))))
        gaze.append(g)
    assert "watch" in [r["to"] for r in sim.mind.decisions]
    err = sum(errs[10:]) / len(errs[10:])
    assert err < 0.12, f"gaze off by {err:.2f} on average"
    assert min(gaze) < -0.3 and max(gaze) > 0.3, f"the eyes should sweep with it: {min(gaze):.2f}..{max(gaze):.2f}"


def test_sudden_motion_close_startles_slow_far_does_not():
    sim = _sim(["toy"])
    sim.move_prop("toy", [(0.33, -0.15), (0.33, 0.15)], speed=0.04)       # slow, far
    sim.run(8.0)
    assert "startle" not in [r["to"] for r in sim.mind.decisions]
    assert sim.mind.affect.needs.safety < 0.2
    sim.hide_prop("toy")
    sim.run(8.0)
    p = sim.pose()
    c, s = math.cos(p[2]), math.sin(p[2])
    start = sim.desk.clamp((p[0] + 0.32 * c, p[1] + 0.32 * s), 0.03)
    t0 = sim.t
    sim.move_prop("toy", [start, (p[0] + 0.12 * c, p[1] + 0.12 * s)], speed=0.45)   # rushes at it
    sim.run(1.2)
    d_after_rush = _dist(sim, "toy")
    sim.run(4.0)
    recs = [r["to"] for r in sim.mind.decisions if r["t"] >= t0]
    assert "startle" in recs, f"no startle: {recs}, safety {sim.mind.affect.needs.safety:.2f}"
    assert _dist(sim, "toy") > d_after_rush + 0.02, "it should back away from it"
    th = max(sim.mind.novelty.things.values(), key=lambda t: t.id)
    assert th.threat > sim.mind.novelty.threat0, "the thing that scared it is more worrying now"


def test_edges_and_feared_spots_are_respected():
    sim = _sim(["box"])
    sim.mind.learn_danger((0.0, -0.12), "near_fall")
    sim.mind.learn_danger((0.0, -0.12), "near_fall")
    sim.show_prop("box", (0.17, -0.16))                  # 6.5 cm from the front edge, the feared spot nearby
    worst, targets = 1.0, []
    for _ in range(300):
        sim.run(0.2)
        p = sim.pose()
        worst = min(worst, sim.desk.feet_margin(p[:2], p[2]))
    targets = [m[1][1] for m in sim.motions if m[1][0] == "goto" and len(m[1]) > 5]
    assert "investigate" in [r["to"] for r in sim.mind.decisions]
    assert worst > 0.0 and sim.edge_recovers == 0, f"feet margin {worst:.3f}, edge recoveries {sim.edge_recovers}"
    assert all(sim.mind.danger(t) < CAREFUL for t in targets), "it planned a spot it fears"


def test_quiet_hours_keep_it_in_place():
    sim = _sim(["box", "toy"])
    sim.mind.quiet_until = 1e9
    sim.run(4.0)
    p = sim.pose()
    sim.show_prop("box", sim.desk.clamp((p[0] + 0.2 * math.cos(p[2]), p[1] + 0.2 * math.sin(p[2])), 0.06))
    sim.run(20.0)
    assert "investigate" not in [r["to"] for r in sim.mind.decisions]
    assert any(ln[1] == "curious" for ln in sim.mind.log), "it should still notice the new thing"
