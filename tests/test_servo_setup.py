"""robot/setup_servo.py against the byte-level bus emulator: the settings land in the servo's memory, the claw gets
the stalled-grip protection and the legs keep the factory overload behaviour."""

from robot.bus_emulator import EmulatedPort
from robot.feetech_bus import STS3215, FeetechBus
from robot.setup_servo import setup


def _servo(target, claw=False, torque=None):
    bus = FeetechBus(EmulatedPort(STS3215, [1]), STS3215)
    sid = setup(bus, target, 1, log=lambda *_: None, torque_permille=torque, claw=claw)
    return bus, sid


def test_claw_gets_grip_protection():
    bus, sid = _servo(15, claw=True, torque=400)
    assert bus.read(sid, "overload_torque") == 25
    assert bus.read(sid, "torque_limit") == 400 and bus.read(sid, "max_torque") == 400
    assert bus.read(sid, "return_delay") == 0


def test_leg_keeps_factory_overload():
    bus, sid = _servo(2)
    assert sid == 2 and bus.ping(2)
    assert bus.read(sid, "overload_torque") == 80
    assert bus.read(sid, "protective_torque") == 20
