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


def test_the_log_holds_what_a_learned_chooser_would_need():
    """Every switch carries all candidate scores and what made them, the feelings and the situation; every
    activity that ends leaves how it went (steps, feedback, the change in feeling and needs); every choice, not
    only the switches, can be kept (the "keep going" examples)."""
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    sim = PinchSim(memory=Memory(path=None), seed=7)
    m = sim.mind
    m.choice_log = []
    sim.run(20.0)
    sim.say("good crab")
    sim.run(20.0)
    d = m.decisions[-1]
    for k in ("id", "scores", "parts", "affect", "context"):
        assert k in d, k
    assert len(d["scores"]) >= 2 and "emotions" in d["affect"] and "part_of_day" in d["context"]
    assert m.episodes, "no activity ended"
    e = m.episodes[-1]
    for k in ("id", "activity", "end", "seconds", "steps_ok", "steps_failed", "feedback", "dvalence", "dneeds"):
        assert k in e, k
    assert any(ep["feedback"] for ep in m.episodes) or (m.episode and m.episode["feedback"])
    assert len(m.choice_log) > 2 * len(m.decisions)          # (choices to keep going are there too)


def test_a_new_score_stage_plugs_in_and_explains_itself():
    """The seam a learned scorer or the language-model director will use: one more stage, logged like the rest."""
    from pinch.memory import Memory
    from pinch.mind import Mind
    m = Mind(memory=Memory(path=None), llm=False, seed=0)

    def director(a, u, why, env):
        if a.name == "explore" and u > 0:
            why["director"] = 0.5
            return u + 0.5
        return u
    m.affect.needs.curiosity = 0.9
    m.extra_stages = (director,)
    m._choose()
    assert m.score_parts["explore"].get("director") == 0.5
    assert max(m.scores, key=m.scores.get) == "explore"
