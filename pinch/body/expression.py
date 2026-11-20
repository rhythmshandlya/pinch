"""Emotes: keyframed tricks and feelings, balanced live by the trained emote policy.

Every play is a variant (crab/emotes.py EmoteStyler): the keyframes are the base clip, and a style chosen per play
from a seeded generator layers tempo, size, a little asymmetry and arm lag, a mirror and a body turn on top, set
by the mood (arousal: faster and bigger; sleepy: slower and smaller). The same emote twice running differs.
"""

from crab.emotes import EmoteStyler
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

STYLE_KEYS = ("arousal", "valence", "sleepy", "side", "gaze", "gain")


class Expression:
    vary_emotes = True          # False: the plain keyframes (diagnostics)
    mood_fn = None              # () -> dict(arousal=, valence=, sleepy=): set by the host if it has the mind's affect
    last_style = None           # the style of the emote played last (tests, the observer)

    @property
    def styler(self):
        """Seeded from the idle-life generator's seed (a spawned child stream: drawing from it doesn't change the
        idle life's own draws), so the twin replays exactly and the robot varies from boot to boot."""
        st = self.__dict__.get("_styler")
        if st is None:
            seq = self.life.rng.bit_generator.seed_seq.spawn(1)[0]
            st = self.__dict__["_styler"] = EmoteStyler(np.random.default_rng(seq))
        return st

    def _mood(self, style):
        mood = {"arousal": float(getattr(self.life, "liveliness", 0.4))}
        if self.mood_fn is not None:
            mood.update(self.mood_fn())
        mood.update({k: v for k, v in (style or {}).items() if k in STYLE_KEYS and v is not None})
        return mood

    def _emote(self, name, style=None):
        """Keyframes; with the trained emote policy (policies/emote.npz) the legs get small live
        corrections that keep the emote balanced under shoves, slippery spots or a heavier battery.
        style: optional overrides from the mind ({"side": +1 left / -1 right, "gaze": rad, "gain": 0..1,
        "arousal", "valence", "sleepy"})."""
        if name not in EMOTES:
            return False
        emo = EMOTES[name]
        if self.vary_emotes:
            emo = self.styler.vary(name, **self._mood(style))
            self.last_style = (name, emo.style)
            self.trace("emote", name=name, **emo.style.short())
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
