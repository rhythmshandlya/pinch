"""The scripted gait (crab/gait.py) as pure JAX functions, so training environments can run Pinch's real walking
controller inside the simulation batch (decision 66): learned skills on top of it (pushing, walking on
continuous commands) train against exactly what the robot runs.

    g = GaitJax(GAITS["trot"])                 # constants (numpy, once)
    s = g.init()                               # state: a dict of arrays
    s, q = g.step(s, dt, cmd)                  # q: 18 joint targets (legs from the gait, arms from `arms`)

Same algorithm as Gait.step, vectorised over the four legs: phase clock, stance feet sliding with the body,
Raibert-style swing placement, the speed governor, weight shift and the blend-in from STAND; then the knee-up
leg IK with the same joint limits. Checked against crab/gait.py in tests/test_gait_jax.py.
"""

import jax.numpy as jp
import numpy as np

from .emotes import STAND
from .gait import LEG_IDX, Gait, GaitParams
from .kinematics import _mount
from .params import LEG_IDS, PARAMS

L = PARAMS.leg


class GaitJax:
    def __init__(self, params: GaitParams, com_offset=(0.0, 0.0), arms=None):
        self.P = params
        ref = Gait(params, com_offset=np.asarray(com_offset, float))
        self.neutral = jp.array(np.array([ref.neutral[lg] for lg in LEG_IDS]))          # (4, 3)
        self.offsets = jp.array([params.offsets[lg] for lg in LEG_IDS])                  # (4,)
        self.com = jp.array(com_offset, dtype=jp.float32)
        self.stand = jp.array(STAND)
        self.base = jp.array(STAND if arms is None else arms)
        self.leg_idx = jp.array(np.array(LEG_IDX))                                        # (4, 3)
        mounts = [_mount(lg) for lg in LEG_IDS]
        self.mount = jp.array(np.array([m for m, _ in mounts]))                           # (4, 3), left-mirrored
        self.yaw = jp.array([y for _, y in mounts])
        self.mirror = jp.array([1.0 if lg[1] == "L" else -1.0 for lg in LEG_IDS])
        self.lo = jp.array([[L.coxa_range_for(lg)[0], L.femur_range[0], L.tibia_range[0]] for lg in LEG_IDS])
        self.hi = jp.array([[L.coxa_range_for(lg)[1], L.femur_range[1], L.tibia_range[1]] for lg in LEG_IDS])

    def init(self):
        z2 = jp.zeros((4, 2))
        return dict(t=jp.array(0.0), phase=jp.array(0.0), cmd=jp.zeros(3), offset=z2, lift_start=z2,
                    swinging=jp.zeros(4, bool), shift=jp.zeros(2), lower=jp.array(0.0))

    def _feasible_scale(self, cmd):
        P = self.P
        vx, vy, wz = cmd[0], cmd[1], cmd[2]
        n = self.neutral
        need = jp.max(jp.hypot(vx - wz * n[:, 1], vy + wz * n[:, 0])) * P.duty * P.period / 2
        return jp.where(need <= P.max_step, 1.0, P.max_step / jp.maximum(need, 1e-12))

    def step(self, s, dt, cmd, lower=None):
        P = self.P
        target = jp.asarray(cmd, dtype=jp.float32)
        target = target * self._feasible_scale(target)
        lim = jp.array(P.cmd_accel) * dt
        c = s["cmd"] + jp.clip(target - s["cmd"], -lim, lim)
        idle = (jp.linalg.norm(c) < 1e-3) & (jp.linalg.norm(target) < 1e-3) & \
            jp.all(jp.linalg.norm(s["offset"], axis=1) < 2e-3) & ~jp.any(s["swinging"])
        phase = jp.where(idle, s["phase"], (s["phase"] + dt / P.period) % 1.0)
        ph = (phase + self.offsets) % 1.0                                                  # (4,)
        shift = s["shift"]
        px = self.neutral[:, 0] - shift[0]
        py = self.neutral[:, 1] - shift[1]
        v = jp.stack([c[0] - c[2] * py, c[1] + c[2] * px], axis=1)                          # (4, 2)
        speed = jp.linalg.norm(v, axis=1)
        stance = ph < P.duty
        # speed governor: no stance foot may run out of travel before it lifts
        travelled = -jp.sum(s["offset"] * v, axis=1) / jp.maximum(speed, 1e-12)
        budget = jp.maximum(1.25 * P.max_step - travelled, 0.0)
        needed = speed * (P.duty - ph) * P.period
        active = (~idle) & stance & (speed >= 1e-6)
        govern = jp.min(jp.where(active & (needed > budget), budget / jp.maximum(needed, 1e-12), 1.0))
        # stance: slide with the body; swing: lift and place (Raibert)
        off_stance = s["offset"] - govern * v * dt
        sw = (ph - P.duty) / (1.0 - P.duty)
        starting = (~stance) & (~s["swinging"])
        lift_start = jp.where(starting[:, None], s["offset"], s["lift_start"])
        land = v * P.duty * P.period / 2
        ln = jp.linalg.norm(land, axis=1, keepdims=True)
        land = jp.where(ln > P.max_step, land * P.max_step / jp.maximum(ln, 1e-12), land)
        cyc = sw - jp.sin(2 * jp.pi * sw) / (2 * jp.pi)
        off_swing = lift_start + (land - lift_start) * cyc[:, None]
        offset = jp.where(idle, s["offset"], jp.where(stance[:, None], off_stance, off_swing))
        z = jp.where(idle | stance, 0.0, P.step_height * 0.5 * (1 - jp.cos(2 * jp.pi * sw)))
        swinging = jp.where(idle, s["swinging"], ~stance)
        on = jp.linalg.norm(offset, axis=1, keepdims=True)
        offset = jp.where(on > 1.4 * P.max_step, offset * 1.4 * P.max_step / jp.maximum(on, 1e-12), offset)
        low = s["lower"] if lower is None else lower
        feet = jp.concatenate([self.neutral[:, :2] + offset, (self.neutral[:, 2] + (z + low))[:, None]], axis=1)
        # weight shift
        if P.weight_shift > 0:
            if P.shift_mode == "support":
                lead = P.shift_lead
                sm = lambda x: x * x * (3 - 2 * x)  # noqa: E731
                w = jp.where(ph >= P.duty, 0.0, jp.where(ph < lead, sm(ph / lead),
                             jp.where(ph > P.duty - lead, 1.0 - sm((ph - (P.duty - lead)) / lead), 1.0)))
                pts = feet[:, :2]
            else:
                w = jp.ones(4)
                pts = self.neutral[:, :2]
            centroid = jp.sum(w[:, None] * pts, axis=0) / jp.maximum(jp.sum(w), 1e-6)
            desired = P.weight_shift * (centroid - self.com)
            dn = jp.linalg.norm(desired)
            desired = jp.where(dn > 0.06, desired * 0.06 / jp.maximum(dn, 1e-12), desired)
            shift = shift + min(1.0, dt / 0.06) * (desired - shift)
        else:
            shift = jp.zeros(2)
        t = s["t"] + dt
        q = self._ik(feet - jp.concatenate([shift, jp.zeros(1)])[None, :])
        out = self.base.at[self.leg_idx.reshape(-1)].set(q.reshape(-1))
        blend = jp.clip(t / P.blend_time, 0.0, 1.0)
        blend = blend * blend * (3 - 2 * blend)
        out = jp.where(t < P.blend_time, self.stand + (out - self.stand) * blend, out)
        ns = dict(t=t, phase=phase, cmd=c, offset=offset, lift_start=lift_start, swinging=swinging, shift=shift,
                  lower=low)
        return ns, out

    def _ik(self, p):
        """(4, 3) foot targets in the body frame -> (4, 3) joint angles (knee-up, clipped like crab/kinematics)."""
        pm = p * jp.stack([jp.ones(4), self.mirror, jp.ones(4)], axis=1)
        d = pm - self.mount
        c = self.yaw - jp.arctan2(d[:, 1], d[:, 0])
        c = (c + jp.pi) % (2 * jp.pi) - jp.pi
        r = jp.hypot(d[:, 0], d[:, 1]) - L.coxa_len
        h = d[:, 2]
        D = jp.clip(jp.hypot(r, h), abs(L.femur_len - L.tibia_len) + 1e-4, L.femur_len + L.tibia_len - 1e-4)
        cos_t = (D ** 2 - L.femur_len ** 2 - L.tibia_len ** 2) / (2 * L.femur_len * L.tibia_len)
        t = -jp.arccos(jp.clip(cos_t, -1, 1))
        f = jp.arctan2(h, r) - jp.arctan2(L.tibia_len * jp.sin(t), L.femur_len + L.tibia_len * jp.cos(t))
        q = jp.clip(jp.stack([c, f, t], axis=1), self.lo, self.hi)
        tib = jp.minimum(jp.maximum(q[:, 2], L.shin_min - q[:, 1]), self.hi[:, 2])
        return q.at[:, 2].set(tib)
