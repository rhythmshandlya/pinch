"""Pinch's activities: everything it can decide to do, and how much it wants to right now.

Each activity has
  utility(m)  how much Pinch wants to do it now (m = the Mind: needs, feelings, world, memory, requests)
  run(m)      a generator of steps the body carries out one after another:
                ("face", expr) ("sound", name) ("emote", name) ("walk", vx, vy, wz, s) ("goto", xy, tol)
                ("turn_to", xy) ("pose", name, s) ("push", obj, goal_xy) ("pick", obj) ("drop", goal_xy)
                ("wait", s) ("quiet", on) ("say_mood",) ("remember", kind, text) ("praise_wait", s)
The Mind (pinch/mind.py) re-scores every activity twice a second and switches when something else
matters clearly more, or at once for urgent things (a scare, a command, a meeting starting).

The activities live in families (pinch/activities/): rest, social, defensive, recovery, tasks, play,
curiosity;
the Activity base class and shared helpers in base.py. LIBRARY keeps its order (ties go to the first).
"""

from pinch.activities.base import *  # noqa: F401,F403 - shared constants and helpers, as before the split
from pinch.activities.curiosity import Investigate, Watch
from pinch.activities.defensive import Hide, Orient, Shy, Startle
from pinch.activities.play import Explore, Play, ShowOff
from pinch.activities.recovery import HeadHome, LowPower, RightSelf, Unstick
from pinch.activities.rest import Idle, Rest
from pinch.activities.social import (
           AfterMeeting,
           Celebrate,
           Comfort,
           GoQuiet,
           Greet,
           Obey,
           ReactTouch,
           SeekAttention,
           Sulk,
)
from pinch.activities.tasks import GuardEdge, Tidy

LIBRARY = [Idle(), Obey(), GoQuiet(), AfterMeeting(), SeekAttention(), Comfort(), Celebrate(), ReactTouch(),
           Startle(), Orient(), RightSelf(), Unstick(), Hide(), Shy(), HeadHome(), LowPower(), Tidy(), GuardEdge(), Play(), ShowOff(), Explore(), Rest(), Sulk(), Greet(),
           Watch(), Investigate()]
