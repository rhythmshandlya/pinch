"""The benches: each runs Pinch many times in the twin and reports how often it works.

    python sim/benches/robustness.py     walking and emotes over random robots and worlds
    python sim/benches/hurdles.py        bars and ramps, forward and sideways
    python sim/benches/obstacles.py      walking into things on the desk
    python sim/benches/tidy.py           whole-pet tidying over 48 desk layouts
    python sim/benches/pick.py           picking things up
    python sim/benches/righting.py       getting off its back: swing vs learned policy
    python sim/benches/soak.py           hours of pet life with a verdict
    python sim/benches/scenarios.py      the behaviour scenarios (docs/BEHAVIOUR_TESTS.md)

Shared: common.run_parallel (one process pool for all) and common.save_json (--json on every bench).
"""
