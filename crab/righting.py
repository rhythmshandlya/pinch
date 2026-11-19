"""Getting off its back (audit A11, decision 59).

On its back the dome rocks; swinging between two whole-body poses at the right rhythm rolls Pinch over. The
poses were found by search in the twin, not designed: 3000 random pose pairs and periods from on-its-back,
51 turned it over; each of those was re-tried from 8 different starts (rolled +-15 deg, any heading, desk
friction x0.4-1.2, masses +-15 %) and this one - search seed 844 - got upright in all 8, after
0.55 s on average. Joint targets are within every servo's range (they were drawn from it).
"""

import numpy as np

from crab.params import JOINT_NAMES

PERIOD = 1.379            # s for one swing A -> B -> A
POSES = {                  # joint: (pose A, pose B), rad
    "FL_coxa": (+0.001, +0.154),
    "FL_femur": (-0.456, +0.859),
    "FL_tibia": (-1.367, -0.479),
    "FR_coxa": (-0.563, +0.281),
    "FR_femur": (+1.486, +0.198),
    "FR_tibia": (-0.512, -1.774),
    "RL_coxa": (+0.736, -0.447),
    "RL_femur": (+1.406, +1.429),
    "RL_tibia": (-0.305, -2.203),
    "RR_coxa": (+0.430, -0.866),
    "RR_femur": (+0.820, +1.318),
    "RR_tibia": (+0.454, +0.345),
    "L_shoulder_yaw": (+0.138, +0.341),
    "L_shoulder_pitch": (+0.044, -0.647),
    "L_claw": (+0.683, +0.401),
    "R_shoulder_yaw": (+0.101, +0.062),
    "R_shoulder_pitch": (+0.522, +0.673),
    "R_claw": (+0.528, +0.468),
}
_A = np.array([POSES[j][0] for j in JOINT_NAMES])
_B = np.array([POSES[j][1] for j in JOINT_NAMES])


def target(t):
    """Joint targets t seconds into the righting motion (smooth swing between the two poses)."""
    s = 0.5 - 0.5 * np.cos(2 * np.pi * ((t / PERIOD) % 1.0))
    return _A + (_B - _A) * s
