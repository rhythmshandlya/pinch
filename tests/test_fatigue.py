"""ARC-002 fatigue: tiredness from moving builds while the body works and recovers at rest; a tired Pinch wants
to rest more (it is not sleepiness: a tired crab at noon rests, it doesn't sleep)."""

from pinch.activities import LIBRARY
from pinch.memory import Memory
from pinch.mind import Mind


def test_fatigue_builds_with_work_and_recovers_at_rest():
    noon = 20776 * 86400 + 12 * 3600                       # midday UTC (no night sleep drive)
    mem = Memory(path=None)
    mem.data["settings"]["utc_offset_h"] = 0
    m = Mind(memory=mem, llm=False, clock=lambda: noon + m.t)
    rest = next(a for a in LIBRARY if a.name == "rest")
    fresh = rest.utility(m)
    for _ in range(3000):                                  # 50 minutes of non-stop motion
        m.tick(1.0, body_busy=True, body_ok=True)
    tired = m.affect.needs.fatigue
    assert tired > 0.5, tired
    assert rest.utility(m) > fresh + 0.2, (fresh, rest.utility(m))
    assert m.affect.needs.sleep < 0.6                      # tired, not sleepy
    for _ in range(900):                                   # 15 minutes still
        m.tick(1.0, body_busy=False, body_ok=True)
    assert m.affect.needs.fatigue < tired - 0.3, (tired, m.affect.needs.fatigue)
