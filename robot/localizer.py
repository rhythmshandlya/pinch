"""Where Pinch is on the desk between camera fixes (dead reckoning + marker fixes + ground flow)."""

import numpy as np


class Localizer:
    """Where am I on the desk. Two estimates, kept apart on purpose:

      pose()         what the body steers by: integrated between camera fixes - from the ground flow when
                     the camera measures it, else from the walking command - and pulled to each marker fix.
      measured_xy()  only what the camera measured: the last marker fix, carried forward by ground flow.
                     The stuck check (no progress while walking) uses this one: the integrated pose keeps
                     moving when the feet slip on the spot, which is exactly the case to catch.

    sigma() is how far pose() may have drifted since the last fix; the edge guard keeps that much further
    from the edge. While Pinch is stuck the walking command is not integrated at all."""

    SIGMA0, SIGMA_MAX = 0.005, 0.06          # m
    DRIFT_CMD, DRIFT_FLOW = 0.3, 0.1         # drift per metre integrated: commanded (gait tracks 50-110 %), flow

    def __init__(self, start=(0.0, 0.0, 0.0), tracking=0.8):
        self.x, self.y, self.yaw = start
        self.tracking = tracking             # the gait achieves ~80 % of the commanded speed
        self.last_fix = -1e9
        self.fixes = 0
        self.frozen = False                  # True while stuck: don't integrate commands that go nowhere
        self.drift = 0.0                     # m of possible error accumulated since the last fix
        self.flow_v, self.flow_t = None, -1e9
        self.meas, self.meas_t = None, -1e9

    def predict(self, cmd, dt, t=None):
        if self.frozen:
            return
        vx, vy, wz = cmd
        if t is not None and self.flow_v is not None and t - self.flow_t < 0.5:
            (vx, vy), k = self.flow_v, self.DRIFT_FLOW          # measured over the desk: trust it
        else:
            vx, vy, k = self.tracking * vx, self.tracking * vy, self.DRIFT_CMD
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw += self.tracking * wz * dt
        self.drift += k * np.hypot(vx, vy) * dt

    GATE_M, GATE_YAW = 0.05, 0.5             # a fix further than max(this, 3 sigma) / this many rad from the estimate ...
    CONFIRM, AGREE_M = 3, 0.03               # ... is believed only once this many in a row agree within this (gating:
    #                                          a far, small marker has a second, mirrored pose solution that can jump the
    #                                          estimate 30 cm; the robot then guards the wrong edge)

    def fix(self, pose, t, weight=0.6):
        """A camera fix. Returns False when it was held back as an outlier (see GATE_M)."""
        x, y, yaw = pose
        far = (np.hypot(x - self.x, y - self.y) > max(self.GATE_M, 3.0 * self.sigma())
               or abs((yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi) > self.GATE_YAW)
        if far and self.fixes:
            c = self._cands = [q for q in getattr(self, "_cands", []) if t - q[2] < 2.0] + [(x, y, t)]
            agree = len(c) >= self.CONFIRM and all(np.hypot(q[0] - x, q[1] - y) < self.AGREE_M for q in c[-self.CONFIRM:])
            if not agree:
                self.rejected = getattr(self, "rejected", 0) + 1
                return False
            weight = 1.0                                     # (several fixes agree: it really is somewhere else)
        self._cands = []
        self.x += weight * (x - self.x)
        self.y += weight * (y - self.y)
        self.yaw += weight * ((yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi)
        self.last_fix = t
        self.fixes += 1
        self.drift = 0.0
        self.meas, self.meas_t = (float(x), float(y)), t
        return True

    FLOW_MAX = 0.1                           # m/s: faster than Pinch walks = a bad flow reading (a claw moving)

    def flow(self, v_body, t, dt):
        """Ground flow measured the body moving at v_body (m/s, body frame) over the last dt seconds.
        Marker fixes outrank it: while one is recent, flow doesn't move the measured track."""
        if np.hypot(*v_body) > self.FLOW_MAX:
            return False
        self.flow_v, self.flow_t = v_body, t
        if t - self.last_fix < 0.6:
            return True
        if self.meas is None or t - self.meas_t > 1.0:      # no recent anchor: start a new measured track
            self.meas = (self.x, self.y)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        vx, vy = v_body
        self.meas = (self.meas[0] + (c * vx - s * vy) * dt, self.meas[1] + (s * vx + c * vy) * dt)
        self.meas_t = t
        return True

    def measured_xy(self, t, max_age=0.6):
        return self.meas if self.meas is not None and t - self.meas_t <= max_age else None

    def sigma(self):
        return min(self.SIGMA0 + self.drift, self.SIGMA_MAX)

    def pose(self):
        return self.x, self.y, self.yaw
