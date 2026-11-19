"""The hurdle / ramp course Pinch trains and is tested on (decision 58), without JAX: shared by
training/terrain_env.py (jax, one course per training robot) and sim/robustness.py --world terrain (numpy)."""

import mujoco

OBSTACLES = [  # name, box half sizes (x radial, y tangent, z)
    ("hurdle_a", (0.006, 0.15, 0.010)),             # (tall boxes sunk into the floor: up to 20 mm can show)
    ("hurdle_b", (0.006, 0.15, 0.010)),
    ("ramp", (0.05, 0.12, 0.003)),
    ("ramp_top", (0.06, 0.12, 0.015)),
]
N_U = 10                     # uniforms per layout


def _quat_yaw_pitch(xp, yaw, pitch):
    """q = yaw about z, then pitch about the (new) y axis; (w, x, y, z)."""
    cy, sy, cp, sp = xp.cos(yaw / 2), xp.sin(yaw / 2), xp.cos(pitch / 2), xp.sin(pitch / 2)
    return xp.stack([cy * cp, -sy * sp, cy * sp, sy * cp])


def terrain_layout(xp, u, hard=1.0):
    """u: N_U uniforms in [0, 1). Returns pos (4, 3) and quat (4, 4) for OBSTACLES, around the origin.
    hard scales the hurdle heights and the ramp angle (training: 1; the bench's terrain_hard world: 1.5)."""
    two_pi = 2 * xp.pi
    a1 = u[0] * two_pi
    r1 = 0.13 + 0.06 * u[1]
    h1 = (0.003 + 0.007 * u[2]) * hard
    a2 = a1 + xp.pi + (u[3] - 0.5) * 1.2
    r2 = 0.13 + 0.06 * u[4]
    h2 = (0.003 + 0.007 * u[5]) * hard
    a3 = a1 + xp.pi / 2 + (u[6] - 0.5) * 1.0
    r3 = 0.12 + 0.04 * u[7]
    th = xp.radians(14.0) * u[8] * hard
    hx, hz = OBSTACLES[2][1][0], OBSTACLES[2][1][2]
    top_h = 2 * hx * xp.sin(th)                                    # the ramp's high end
    ramp_c = r3 + hx * xp.cos(th)
    top_c = r3 + 2 * hx * xp.cos(th) + OBSTACLES[3][1][0]
    pos = xp.stack([
        xp.stack([r1 * xp.cos(a1), r1 * xp.sin(a1), h1 - OBSTACLES[0][1][2]]),
        xp.stack([r2 * xp.cos(a2), r2 * xp.sin(a2), h2 - OBSTACLES[1][1][2]]),
        xp.stack([ramp_c * xp.cos(a3), ramp_c * xp.sin(a3), hx * xp.sin(th) - hz * xp.cos(th)]),
        xp.stack([top_c * xp.cos(a3), top_c * xp.sin(a3), top_h - OBSTACLES[3][1][2]]),
    ])
    zero = a1 * 0.0
    quat = xp.stack([_quat_yaw_pitch(xp, a1, zero), _quat_yaw_pitch(xp, a2, zero),
                     _quat_yaw_pitch(xp, a3, -th), _quat_yaw_pitch(xp, a3, zero)])
    return pos, quat


def add_obstacles(spec, pos=None, quat=None):
    """Add the obstacle boxes to a MjSpec's world body (at the given layout, or parked out of the way)."""
    for i, (name, size) in enumerate(OBSTACLES):
        p = [5.0 + i, 5.0, -0.5] if pos is None else [float(v) for v in pos[i]]
        q = [1, 0, 0, 0] if quat is None else [float(v) for v in quat[i]]
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=p, quat=q,
                                rgba=[0.55, 0.45, 0.35, 1], friction=[0.8, 0.02, 0.001])
