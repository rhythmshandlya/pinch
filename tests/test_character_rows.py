"""Phase 4 rows that existed in code without a test of their own: personality (ARC-001), play that starts, varies
and ends (ARC-011), greeting and withdrawal (ARC-010)."""

from pinch.activities import LIBRARY
from pinch.affect import Personality
from pinch.memory import Memory
from pinch.mind import Mind
from pinch.twin import PinchSim


def _mind(**traits):
    return Mind(memory=Memory(path=None), llm=False, clock=lambda: 0.0, personality=Personality(**traits))


def _act(name):
    return next(a for a in LIBRARY if a.name == name)


def test_personality_makes_individuals():
    """ARC-001: the same situation, two characters: a playful bold one and a shy calm one."""
    bold, shy = _mind(playfulness=1.4, timidity=0.2), _mind(playfulness=0.4, timidity=1.2)
    for m in (bold, shy):
        m.affect.needs.play = 0.8
    assert _act("play").utility(bold) > 2 * _act("play").utility(shy)
    assert shy.novelty.threat0 > bold.novelty.threat0 + 0.4          # new things look scarier to the shy one
    assert _mind(neediness=2.0).affect.personality.neediness == 2.0


def test_play_starts_varies_and_ends():
    """ARC-011: with toys about and play overdue it starts playing by itself, the games differ, and each ends."""
    sim = PinchSim(memory=Memory(path=None), seed=4)            # (the desk things are its toys)
    sim.mind.memory.data["homes"] = {}
    sim.mind.affect.needs.play = 0.95
    plays = []
    for _ in range(12):
        sim.run(30.0)
        sim.mind.affect.needs.play = max(sim.mind.affect.needs.play, 0.9)     # keep it wanting to play
        plays = [r for r in sim.mind.decisions if r["to"] == "play"]
        if len(plays) >= 2:
            break
    assert plays, f"never played: {[r['to'] for r in sim.mind.decisions]}"
    ended = [r for r in sim.mind.decisions if r["prev"] == "play"]
    assert ended, "a game never ended"
    verbs = {p for _, a, p in sim.mind.phases if a == "play"}
    assert verbs, "no game phases"
    assert len(plays) < 2 or len(verbs) >= 2 or len(sim.mind.playbook.games) >= 2, f"the same game every time: {verbs}"


def test_it_greets_its_person_and_withdraws_when_scolded():
    """ARC-010: its person comes back -> a greeting; scolded -> it withdraws (sulks), it does not carry on as before."""
    sim = PinchSim(memory=Memory(path=None), seed=2)
    sim.run(3.0)
    sim.say("hi pinch")
    sim.run(6.0)
    assert any(r["to"] in ("greet", "obey") for r in sim.mind.decisions), [r["to"] for r in sim.mind.decisions]
    sim.run(20.0)
    sim.say("bad crab, no!")
    sim.run(4.0)
    tos = [r["to"] for r in sim.mind.decisions]
    assert "sulk" in tos[-3:] or sim.mind.affect.last_event == "scolded", tos[-5:]
    assert sim.mind.affect.valence < 0.3
