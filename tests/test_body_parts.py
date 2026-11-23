"""Body parts and their owners (phase 4: pinch/body/arbiter.py, configs/mind/channels.toml, PRD orchestration
invariants): one owner per part, the higher layer wins, a claim is all or nothing, and ambient life keeps moving
the parts no one uses - the claws while it walks - but never the legs under a walk or a claw that carries."""

import numpy as np

from pinch.body.arbiter import Claim, ambient_parts, arbitrate, claim_for, config, layer_rank
from pinch.memory import Memory
from pinch.mind import Mind


def test_no_part_ever_has_two_owners_and_the_higher_layer_wins():
    rng = np.random.default_rng(0)
    parts, layers = config()["channels"], config()["layers"][:-1]          # (ambient never claims)
    for _ in range(2000):
        claims = []
        for k in range(rng.integers(1, 5)):
            p = frozenset(rng.choice(parts, size=rng.integers(1, 4), replace=False).tolist())
            claims.append(Claim(f"c{k}", str(rng.choice(layers)), p, since=float(rng.random())))
        owners, refused = arbitrate(claims)
        granted = {c for c in owners.values() if c is not None}
        for c in granted:                                       # all or nothing
            assert all(owners[p] is c for p in c.parts)
        for c, _part, holder in refused:
            assert c not in granted
            assert (layer_rank(holder.layer), holder.since) <= (layer_rank(c.layer), c.since)


def test_a_reflex_takes_the_legs_from_an_activity():
    walk = claim_for("walk", "explore", "activity", since=0.0)
    back = claim_for("walk", "cliff reflex", "reflex", since=1.0)
    owners, refused = arbitrate([walk, back])
    assert owners["legs"] is back and refused[0][0] is walk


def test_ambient_life_moves_only_the_parts_no_one_uses():
    owners, _ = arbitrate([claim_for("walk", "explore", "activity")])
    assert ambient_parts(owners) >= {"claw_L", "claw_R"} and "legs" not in ambient_parts(owners)
    owners, _ = arbitrate([claim_for("emote", "greet", "activity")])
    assert not ambient_parts(owners) & {"legs", "claw_L", "claw_R"}
    owners, _ = arbitrate([])
    assert {"legs", "claw_L", "claw_R"} <= ambient_parts(owners)


def test_the_claws_stay_alive_while_it_walks_in_the_twin():
    from pinch.twin import PinchSim
    sim = PinchSim(memory=Memory(path=None), seed=3)
    sim.run(1.0)
    sim.mind.gen = None                                         # hold the mind: drive the body directly
    sim.mind.estopped = True
    body = sim.body
    n0 = len(body.life.events)
    body.start(("walk", 0.03, 0.0, 0.0, 12.0), "test", "activity")
    mask = body._ambient_mask()
    legs = [i for i, m in enumerate(mask) if m == 0.0]
    assert mask.sum() == 6 and len(legs) == 12                  # both arms, no leg joint
    for _ in range(int(10.0 / 0.02)):
        body.tick()
    claw_moves = [a for _, a in body.life.events[n0:] if a.startswith(("claw_flex", "shoulder"))]
    assert claw_moves, f"no ambient claw movement in 10 s of walking: {body.life.events[n0:]}"


def test_carrying_keeps_the_ambient_life_off_that_claw():
    from pinch.twin import PinchSim
    sim = PinchSim(memory=Memory(path=None), seed=3)
    body = sim.body
    body.start(("walk", 0.0, 0.0, 0.0, 1.0), "test", "activity")
    body.held = ("cap", "left")
    m = body._ambient_mask()
    from pinch.body.core import PART_JOINTS
    assert m[PART_JOINTS["claw_L"]].sum() == 0 and m[PART_JOINTS["claw_R"]].sum() == 3


def test_the_mind_says_what_layer_a_motion_comes_from():
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0)
    for name, layer in (("right_self", "reflex"), ("obey", "command"), ("startle", "reaction"),
                        ("react_touch", "reaction"), ("explore", "activity"), ("idle", "activity")):
        m._start(name, "test")
        assert m.motion_owner() == (name, layer), (name, m.motion_owner())
