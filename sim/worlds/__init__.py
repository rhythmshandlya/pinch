"""Pinch's worlds: seeded homes, desks, tables and floors to train skills in and to judge them on.

    from sim.worlds import world
    w = world("train", 17)              # the 18th of 1,000 training worlds (always the same)
    w = world("test", 3)                # one of 200 held-out test worlds, never trained or tuned on
    sim = PinchSim(world=w)              # the full pet in it (pinch/twin)

A world is plain data (sim/worlds/generate.py: a home of 1-3 rooms with doorways and furniture, the surface Pinch
stands on - a desk, a table or the floor - its size, material and friction, a rug or desk mat, 0-10 everyday things,
lights, and the people around), built into MuJoCo by sim/worlds/build.py. People (sim/worlds/people.py) move as a
hand, a face and walking legs with a style (indifferent, curious, fearful, playful). The camera's real fisheye view
is rendered by sim/worlds/fisheye.py. Behaviour scenarios with sampled timelines and assertions live in
sim/scenarios/*.toml and run with sim/worlds/scenario.py.
"""

from sim.worlds.generate import SPLITS, World, split_of, world, worlds  # noqa: F401
