"""The daily routine (decision 82): quiet hours said in plain words, kept in the memory, silence inside them,
and sleepiness that builds faster at night."""

from pinch.language import understand_rules
from pinch.memory import Memory
from pinch.routine import Routine, in_window, local_hour, parse_quiet_hours
from pinch.twin import SIM_EPOCH, PinchSim

HOUR = 3600.0


def test_parsing_quiet_hours():
    assert parse_quiet_hours("Pinch, quiet hours from 10 pm to 7 am") == (22.0, 7.0)
    assert parse_quiet_hours("quiet hours 22:30 to 6") == (22.5, 6.0)
    assert parse_quiet_hours("set quiet hours from eleven p.m. until six a.m.") == (23.0, 6.0)
    assert parse_quiet_hours("no more quiet hours please") == "clear"
    assert parse_quiet_hours("be quiet pinch") is None
    assert understand_rules("quiet hours from 10 pm to 7 am")["intent"] == "quiet_hours"
    assert understand_rules("shh pinch, be quiet")["intent"] == "quiet"


def test_hours_and_windows():
    assert in_window(23.0, 22, 7) and in_window(6.5, 22, 7) and not in_window(8.0, 22, 7)
    assert in_window(13.0, 12, 14) and not in_window(14.0, 12, 14)
    assert abs(local_hour(SIM_EPOCH, 5.5) - 9.0) < 1e-9          # the twin starts at 09:00 India time
    r = Routine({"utc_offset_h": 5.5})
    assert r.part_of_day(SIM_EPOCH) == "morning" and r.part_of_day(SIM_EPOCH + 14 * HOUR) == "night"
    assert r.sleep_drive(SIM_EPOCH + 14 * HOUR) > r.sleep_drive(SIM_EPOCH) == 1.0


def _night_sim(hours_after_nine, seed=4):
    sim = PinchSim(memory=Memory(path=None), seed=seed, epoch=SIM_EPOCH + hours_after_nine * HOUR)
    sim.say("Pinch, quiet hours from 10 pm to 7 am")
    sim.run(1.0)
    return sim


def test_quiet_hours_are_kept_and_quiet():
    night, day = _night_sim(14.0), _night_sim(3.0)                  # 23:00 and 12:00, same quiet hours set
    assert night.mind.memory.data["settings"]["quiet_hours"] == [22.0, 7.0]
    assert night.mind.affect.quiet and not day.mind.affect.quiet
    for sim in (night, day):
        t0 = sim.t
        for k in range(6):                               # stroked and talked to for two minutes
            sim.touch("stroke")
            sim.say("hi pinch, good crab")
            sim.run(20.0)
        sim.loud = [s for t, s in sim.sounds_played if t > t0]
    assert day.loud, "no sounds by day: the test can't tell quiet from silent"
    assert not night.loud, f"sounds inside quiet hours: {night.loud}"
    assert night.falls == 0


def test_sleepier_at_night():
    gain = {}
    for name, h in (("day", 3.0), ("night", 14.0)):
        sim = PinchSim(memory=Memory(path=None), seed=5, epoch=SIM_EPOCH + h * HOUR)
        s0 = sim.mind.affect.needs.sleep
        sim.run(240.0)
        gain[name] = sim.mind.affect.needs.sleep - s0
    assert gain["night"] > 2.5 * gain["day"] > 0, gain
