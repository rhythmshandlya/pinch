"""The decision log: every activity switch is recorded with where it came from, why, the strongest candidates,
what Pinch had noticed and how it felt - in memory and as JSON lines (decision 61)."""

import json
import tempfile
from pathlib import Path

from pinch.memory import Memory
from pinch.twin import PinchSim


def test_every_switch_is_logged_with_a_reason():
    log = Path(tempfile.mkdtemp()) / "decisions.jsonl"
    sim = PinchSim(memory=Memory(path=None), seed=1, decision_log=str(log))
    sim.run(4.0)
    sim.say("dance!")
    sim.run(12.0)
    sim.say("I'm sad today")
    sim.run(20.0)
    switches = [text for _, kind, text in sim.mind.log if kind == "activity"]
    recs = list(sim.mind.decisions)
    assert len(recs) >= 3, f"only {len(recs)} decisions in 36 s"
    assert [r["to"] for r in recs][-len(switches):] == switches[-len(recs):], "log and activities disagree"
    for r in recs:
        assert r["reason"], f"a switch to {r['to']} without a reason"
        assert r["top"] and isinstance(r["needs"], dict) and r["feeling"]
        if r["reason"] in ("first", "previous activity finished"):
            assert r["top"][0][0] == r["to"], f"chose {r['to']} but {r['top'][0][0]} scored highest"
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines == recs, "the JSON lines file should hold the same records"
    assert any(r["to"] == "obey" for r in recs) and any(r["to"] == "comfort" for r in recs), \
        f"expected obey and comfort among {[r['to'] for r in recs]}"
