"""Pinch in the digital twin: desk world + robot runtime + body + mind + face, one 50 Hz loop.

    sim = PinchSim()
    sim.say("Pinch, put the pen back")
    sim.run(60)                         # simulated seconds
    sim.in_tray("pen")

Used by the virtual pet app (pinch/app.py) and the scenario tests (sim/pinch_scenarios.py).

Assembled from its parts: core.py (set-up and the loop), world.py (where things are), sensors.py
(camera, cliff beams, grip), stimuli.py (talk, touch, sound) and faults.py (injected failures)."""

from pinch.twin.common import DT, FACE_EVERY, MIND_EVERY, SIM_EPOCH  # noqa: F401
from pinch.twin.core import TwinCore
from pinch.twin.faults import TwinFaults
from pinch.twin.sensors import TwinSensors
from pinch.twin.stimuli import TwinStimuli
from pinch.twin.world import TwinWorld


class PinchSim(TwinStimuli, TwinFaults, TwinSensors, TwinWorld, TwinCore):
    pass
