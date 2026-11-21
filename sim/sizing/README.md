Sizing study: STS3215 crab vs today (see docs/research/SIZING_STS3215.md).

Rerun (about 5 minutes, 4 processes): `python sim/sizing/sweep.py`, then `variants.py`, then `recommend.py` (all need the .venv python; they write sim/sizing/out/).
Files: servos.py (BAM first-order servo models), model.py (MjSpec crab), worker.py (one design in a process, reuses crab/gait.py), sweep.py, variants.py, recommend.py.
