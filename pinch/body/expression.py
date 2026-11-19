"""Emotes: keyframed tricks and feelings, balanced live by the trained emote policy."""

from pinch.body.common import (  # noqa: F401
    DT,
    EMOTES,
    JOINT_NAMES,
    SERVO_IDS,
    STAND,
    arm_pose,
    copy,
    np,
    pio,
)


class Expression:
    def _emote(self, name):
        """Keyframes; with the trained emote policy (policies/emote.npz) the legs get small live
        corrections that keep the emote balanced under shoves, slippery spots or a heavier battery."""
        emo = EMOTES.get(name)
        if emo is None:
            return False
        t, last = 0.0, np.zeros(12)
        while t < emo.duration:
            q = emo.target(t)
            if self.emote_policy is not None and not self.held:
                ahead = emo.target(min(t + 0.1, emo.duration))
                gyro, grav = self.crab.hw.read_imu()
                qj, st = self.crab.read_joints()
                qd = np.array([self.crab.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
                obs = pio.emote_observation(np, gyro, grav, qj[pio.LEG_ACT_IDX], qd[pio.LEG_ACT_IDX], last, q, ahead,
                                            None, t / emo.duration)
                last = np.clip(self.emote_policy(obs), -1, 1)
                q = pio.emote_targets(np, q, last)
            if self.held:
                q = arm_pose(q, self.held[1], claw=0.0)
            yield q
            t += DT
        self.hold = STAND.copy()
        return True
