"""The behaviour-quality observer (decision 64): it catches planted problems, and ten minutes of pet life alone
produce no errors (oscillation, unexplained switches)."""

from pinch.memory import Memory
from pinch.observer import observe, observe_records
from pinch.twin import PinchSim


def _d(t, prev, to, reason="x", state=("idle", "idle")):
    return dict(t=t, prev=prev, to=to, reason=reason, state=list(state))


def test_it_catches_planted_problems():
    osc = [_d(i * 2.0, None, "a" if i % 2 == 0 else "b") for i in range(9)]
    kinds = {f.kind for f in observe_records(osc, [], [], 20.0)}
    assert "oscillation" in kinds
    loop = [_d(i * 20.0, "tidy", "tidy", "previous activity finished") for i in range(5)]
    assert "failure_loop" in {f.kind for f in observe_records(loop, [], [], 120.0)}
    assert "unexplained" in {f.kind for f in observe_records([_d(0, None, "idle", "")], [], [], 1.0)}
    slow = [_d(i * 60.0, None, "idle" if i % 2 == 0 else "tidy") for i in range(9)]
    assert "alternation" in {f.kind for f in observe_records(slow, [], [], 600.0)}
    rep = [_d(i * 10.0, None, n) for i, n in enumerate(["a", "b", "c"] * 5)]
    assert "repetition" in {f.kind for f in observe_records(rep, [], [], 150.0)}
    lock = [_d(0.0, None, "idle", state=("idle", "idle"))]
    assert "state_lock" in {f.kind for f in observe_records(lock, [], [], 1000.0)}


def test_ten_minutes_alone_has_no_errors():
    sim = PinchSim(memory=Memory(path=None), seed=5)
    sim.run(600.0)
    findings = observe(sim.mind)
    errors = [f for f in findings if f.severity == "error"]
    assert not errors, "; ".join(f"{f.kind}: {f.evidence}" for f in errors)
    assert len(sim.mind.decisions) >= 5, "barely any decisions in ten minutes"
