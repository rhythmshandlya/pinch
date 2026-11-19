"""Pinch's body: turns the mind's steps into joint targets every 20 ms (robot.crab_robot.Crab does the
safety checks and talks to the servos, simulated or real).

Motion skills:
  emote(name)            keyframed motion (optionally balanced by the trained emote policy)
  walk(vx, vy, wz, s)    the gait (scripted, or the trained walking policy)
  goto(xy, tol)          walk to a spot on the desk (crab-style: sideways is fine), never past the edge margin
  turn_to(xy)            face something
  pose(name, s)          hold a pose (stand, low, nap)
  push(obj, goal)        bulldoze a thing to a goal with the claws held low
  pick(obj)              line a claw up with the thing, open, reach down, close, lift
  drop(goal)             carry what it holds to the goal, put it down, let go

Positions come from pose_fn() -> (x, y, heading) and objects_fn(name) -> ((x, y, yaw), z) in the desk
frame: ground truth in the simulator; on the robot, odometry + the camera's view of the desk.

The body is assembled from its parts (pinch/body/): core.py (stepping, reflexes, basic motions),
locomotion.py, manipulation.py, expression.py; constants and helpers in common.py.
"""

from pinch.body.common import *  # noqa: F401,F403 - constants and helpers, as before the split
from pinch.body.core import BodyCore
from pinch.body.expression import Expression
from pinch.body.locomotion import Locomotion
from pinch.body.manipulation import Manipulation


class Body(Locomotion, Manipulation, Expression, BodyCore):
    """Pinch's body: give it a step (start), tick it at 50 Hz, read busy/ok."""
