"""Leg forward / inverse kinematics in the body frame (+x front, +y left, +z up).

Right legs are mirror images of left legs, and the joint sign conventions are
mirrored too, so a right leg's IK is the left leg's IK on the y-mirrored target.
The same code runs in sim and on the real robot.
"""

import numpy as np

from .params import LEG_IDS, PARAMS

L = PARAMS.leg


def _mount(leg_id):
    (x, y, z), yaw = PARAMS.leg_mounts[leg_id[0]]
    return np.array([x, y, z]), yaw


def _mirror(p, leg_id):
    return p if leg_id[1] == "L" else p * np.array([1, -1, 1])


def leg_fk(leg_id: str, q) -> np.ndarray:
    """(coxa, femur, tibia) -> foot centre position in body frame."""
    c, f, t = q
    m, yaw = _mount(leg_id)
    r = L.coxa_len + L.femur_len * np.cos(f) + L.tibia_len * np.cos(f + t)
    h = L.femur_len * np.sin(f) + L.tibia_len * np.sin(f + t)
    a = yaw - c                      # coxa + swings the foot forward
    return _mirror(m + np.array([r * np.cos(a), r * np.sin(a), h]), leg_id)


def leg_ik(leg_id: str, p) -> tuple[np.ndarray, bool]:
    """Foot target in body frame -> (coxa, femur, tibia), reachable flag.

    Knee-up solution (tibia bends down). Unreachable targets are projected to
    the nearest reachable distance and joints are clipped to their ranges;
    the flag tells you that happened.
    """
    m, yaw = _mount(leg_id)
    d = _mirror(np.asarray(p, float), leg_id) - m
    c = yaw - np.arctan2(d[1], d[0])
    c = (c + np.pi) % (2 * np.pi) - np.pi
    r = np.hypot(d[0], d[1]) - L.coxa_len
    h = d[2]
    D = np.hypot(r, h)
    lo, hi = abs(L.femur_len - L.tibia_len) + 1e-4, L.femur_len + L.tibia_len - 1e-4
    ok = lo <= D <= hi
    D = np.clip(D, lo, hi)
    cos_t = (D**2 - L.femur_len**2 - L.tibia_len**2) / (2 * L.femur_len * L.tibia_len)
    t = -np.arccos(np.clip(cos_t, -1, 1))
    f = np.arctan2(h, r) - np.arctan2(L.tibia_len * np.sin(t), L.femur_len + L.tibia_len * np.cos(t))
    q = np.array([c, f, t])
    rng = np.array([L.coxa_range_for(leg_id), L.femur_range, L.tibia_range])
    qc = np.clip(q, rng[:, 0], rng[:, 1])
    qc[2] = min(max(qc[2], L.shin_min - qc[1]), rng[2, 1])     # shin never folds back into the hip link
    return qc, ok and np.allclose(q, qc)


def feet_fk(q_legs) -> dict:
    """q_legs: (4, 3) in LEG_IDS order -> {leg_id: foot pos}."""
    return {lg: leg_fk(lg, q) for lg, q in zip(LEG_IDS, q_legs)}


# ------------------------------------------------------------------ arms
A = PARAMS.arm


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Boxes around each arm's parts, in the frame of the link that carries them (m; from cad/parts.py,
# left arm; the right arm is the mirror image). The moving finger is boxed closed. Used to keep the
# two arms from ever being driven into each other (with 3-motor arms they can't reach each other
# inside their joint limits; robot/selftest.py checks that).
ARM_BOXES = {
    "shoulder": [((0.0087, 0.0224), (-0.0121, 0.0061), (-0.0141, 0.0073))],      # bracket + lift servo
    "upper":    [((-0.0060, 0.0230), (0.0130, 0.0155), (-0.0060, 0.0060)),       # plate on the lift horn
                 ((-0.0055, 0.0055), (0.0061, 0.0130), (-0.0055, 0.0055)),       # lift servo's horn
                 ((0.0160, 0.0230), (0.0057, 0.0135), (-0.0138, 0.0026)),        # web plate -> palm
                 ((0.0160, 0.0373), (-0.0064, 0.0064), (-0.0138, 0.0026)),       # palm cup
                 ((0.0185, 0.0355), (-0.0055, 0.0055), (0.0026, 0.0130)),        # claw servo top + horn
                 ((0.0362, 0.0540), (-0.0059, -0.0005), (-0.0090, -0.0010)),     # fixed finger
                 ((0.0240, 0.0425), (-0.0060, 0.0060), (0.0130, 0.0150)),        # moving finger's plate
                 ((0.0403, 0.0540), (0.0005, 0.0045), (-0.0090, 0.0150))],       # ... its drop + finger
}


def arm_frames(side: str, q):
    """{link: (origin, rotation)} of the shoulder and upper (arm) frames in the body frame.
    q = (shoulder_yaw, shoulder_pitch[, claw]) - same chain as sim/build_model.py."""
    yaw, pitch = q[0], q[1]
    s = 1 if side == "L" else -1
    (x, y, z), mount_yaw = PARAMS.arm_mount
    o_s, R_s = np.array([x, s * y, z]), _rz(s * mount_yaw) @ _rz(s * yaw)
    o_u, R_u = o_s + R_s @ np.array([A.shoulder_offset, 0, 0]), R_s @ _ry(-pitch)
    return {"shoulder": (o_s, R_s), "upper": (o_u, R_u)}


def arm_points(side: str, q) -> np.ndarray:
    """Hinge and finger-tip positions of one arm, body frame (m)."""
    o, R = arm_frames(side, q)["upper"]
    return np.array([o + R @ np.array([d, 0, -0.005]) for d in (A.upper_len, A.upper_len + A.finger_len)])


def arm_obbs(side: str, q):
    """Oriented boxes (link, centre, rotation, half sizes) around one arm, body frame."""
    s = 1 if side == "L" else -1
    out = []
    for link, (o, R) in arm_frames(side, q).items():
        for (x0, x1), (y0, y1), (z0, z1) in ARM_BOXES[link]:
            if s < 0:
                y0, y1 = -y1, -y0
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
            out.append((link, o + R @ c, R, np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])))
    return out


def _obb_overlap(c1, R1, h1, c2, R2, h2, margin):
    """Separating-axis test for two oriented boxes (grown by margin)."""
    h1, h2 = h1 + margin / 2, h2 + margin / 2
    d = c2 - c1
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    axes += [np.cross(R1[:, i], R2[:, j]) for i in range(3) for j in range(3)]
    for a in axes:
        n = np.linalg.norm(a)
        if n < 1e-9:
            continue
        a = a / n
        r1 = sum(h1[i] * abs(a @ R1[:, i]) for i in range(3))
        r2 = sum(h2[i] * abs(a @ R2[:, i]) for i in range(3))
        if abs(a @ d) > r1 + r2:
            return False
    return True


def arms_clear(q_left, q_right, margin=0.0) -> bool:
    """True if the two arms keep at least `margin` apart (q = (yaw, pitch) per arm).
    The boxes already sit slightly outside the real parts, so no extra margin is needed."""
    L_, R_ = arm_obbs("L", q_left), arm_obbs("R", q_right)
    for la, c1, R1, h1 in L_:
        for lb, c2, R2, h2 in R_:
            if la == "shoulder" and lb == "shoulder":
                continue                                   # fixed to the body, far apart
            if _obb_overlap(c1, R1, h1, c2, R2, h2, margin):
                return False
    return True
