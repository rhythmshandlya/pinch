"""The servo-level supervisor (robot/crab_robot.Crab.tick) and the emergency stop (decision 69).

Fuzzing: thousands of random targets - huge, tiny, NaN, infinite, the wrong length - and every command that
reaches the servos is finite, inside the joint limits, within the speed limit of the previous command, and the
shin never folds into the hip. The emergency stop: nothing at all is sent while it is on; resuming starts from
where the joints are."""

import numpy as np

from crab.params import JOINT_NAMES
from robot.crab_robot import MAX_STEP, SHIN_MIN, Crab
from robot.servos import SERVO_IDS


class FakeServos:
    """Just enough backend: records what is sent; reports the last command back as the position."""
    def __init__(self):
        self.sent, self.torque_on, self.last = [], True, {}

    def write_positions(self, goals):
        self.sent.append(dict(goals))
        self.last.update(goals)

    def read_state(self):
        from robot.backends import ServoState
        return {sid: ServoState(position=self.last.get(sid, 2048), speed=0, load=0, voltage=7.4, temperature=35)
                for sid in SERVO_IDS.values()}

    def read_imu(self):
        return np.zeros(3), np.array([0, 0, -1.0])

    def torque(self, on):
        self.torque_on = on

    def wait(self, dt):
        pass

    def battery_voltage(self):
        return 7.6


def _crab():
    hw = FakeServos()
    c = Crab(hw)
    return c, hw


def test_fuzzed_targets_never_reach_the_servos_unsafe():
    c, hw = _crab()
    rng = np.random.default_rng(0)
    fem = [JOINT_NAMES.index(f"{lg}_femur") for lg in ("FL", "FR", "RL", "RR")]
    tib = [JOINT_NAMES.index(f"{lg}_tibia") for lg in ("FL", "FR", "RL", "RR")]
    prev = c.q_cmd.copy()
    for k in range(3000):
        kind = k % 6
        if kind == 0:
            q = rng.normal(0, 3.0, len(prev))                  # wild
        elif kind == 1:
            q = prev + rng.normal(0, 0.05, len(prev))          # plausible
        elif kind == 2:
            q = prev.copy()
            q[rng.integers(len(q))] = np.nan
        elif kind == 3:
            q = np.full(len(prev), np.inf * rng.choice([-1, 1]))
        elif kind == 4:
            q = rng.normal(0, 1.0, len(prev) - 3)              # the wrong length
        else:
            q = rng.uniform(-50, 50, len(prev))
        out = c.tick(q)
        assert np.all(np.isfinite(out)), f"non-finite command sent at step {k}"
        assert np.all(out >= c.lo - 1e-9) and np.all(out <= c.hi + 1e-9), f"outside the joint limits at step {k}"
        assert np.all(np.abs(out - prev) <= MAX_STEP + 1e-9), f"faster than the servos at step {k}"
        assert np.all(out[tib] >= SHIN_MIN - out[fem] - 1e-9), f"shin into the hip at step {k}"
        prev = out.copy()
    assert c.bad_targets > 0


def test_emergency_stop_sends_nothing_until_resumed():
    c, hw = _crab()
    for _ in range(5):
        c.tick(c.q_cmd + 0.01)
    c.pause("test")
    assert not hw.torque_on
    n = len(hw.sent)
    for _ in range(50):
        c.tick(c.q_cmd + 0.5)                                  # whatever is asked, nothing goes out
    assert len(hw.sent) == n, "commands were sent during an emergency stop"
    assert c.resume() and hw.torque_on and not c.paused
    c.tick(c.q_cmd + 0.01)
    assert len(hw.sent) == n + 1
