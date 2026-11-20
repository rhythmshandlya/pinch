"""Pinch's mind: senses in, one activity at a time out, with feelings, needs and memory in between.

    mind = Mind(memory=Memory())
    mind.hear_text("Pinch, put the pen back")       # speech (typed in the app, or speech-to-text)
    mind.hear_summary(voice_summary)                # once a second from pinch/hearing.py
    mind.touch("stroke")                            # from pinch/touch.py
    mind.world.see("pen", (0.1, 0.0))              # from vision
    mind.observe(Observation("unknown", xy, velocity, size, confidence, signature))   # unknown things, motion
    motion, cancel = mind.tick(dt, body_busy, body_ok)
    # motion: the next body step to run ("emote", "wave") / ("goto", xy, tol) / ... or None
    # mind.face_expr, mind.gaze, mind.sounds (queue) drive the face screen and the speaker

Same code on the laptop (virtual pet, simulator) and on the robot.

Assembled from its parts in pinch/mind/: core, senses, selection, people, selfcare, curiosity; shared constants in common.py.
"""

from pinch.mind.common import *  # noqa: F401,F403 - constants, as before the split
from pinch.mind.core import MindCore
from pinch.mind.curiosity import Curious
from pinch.mind.people import PeopleAware
from pinch.mind.privacy import Privacy
from pinch.mind.selection import Selection
from pinch.mind.selfcare import SelfCare
from pinch.mind.senses import Senses


class Mind(Senses, Selection, PeopleAware, Privacy, SelfCare, Curious, MindCore):
    pass
