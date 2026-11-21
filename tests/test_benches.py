"""The benches are the adoption gate, so a bench that fails must say so in its exit code. The robustness bench
used to print FAIL and exit 0, and `ops eval` (which combines exit codes) then reported success."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.benches import robustness  # noqa: E402

GOOD = dict(n=40, falls=0, track=0.82)
SCRIPTED = dict(n=40, falls=1, track=0.70)


def test_robustness_pass_rule():
    assert robustness.passes("walk", GOOD, SCRIPTED)
    assert robustness.passes("walk", dict(GOOD, falls=2), dict(SCRIPTED, falls=2))        # 5 % of 40 is still a pass
    assert not robustness.passes("walk", dict(GOOD, falls=3), dict(SCRIPTED, falls=3))    # more than 5 % fall
    assert not robustness.passes("walk", dict(GOOD, falls=2), SCRIPTED)                   # worse than scripted
    assert not robustness.passes("walk", dict(GOOD, track=0.55), SCRIPTED)                # too slow
    assert robustness.passes("emote", dict(GOOD, track=float("nan")), None)               # emotes have no speed


def test_a_failed_verdict_fails_the_bench():
    ok = {("walk", "policy"): GOOD, ("walk", "scripted"): SCRIPTED}
    bad = {("walk", "policy"): dict(GOOD, falls=9), ("walk", "scripted"): SCRIPTED, ("emote", "policy"): GOOD}
    assert robustness.verdict_lines(ok) == ["- **walk (policy): PASS**"]
    assert robustness.exit_code(robustness.verdict_lines(ok)) == 0
    lines = robustness.verdict_lines(bad)
    assert lines == ["- **walk (policy): FAIL**", "- **emote (policy): PASS**"], lines
    assert robustness.exit_code(lines) == 1
    assert robustness.exit_code([]) == 0                         # scripted-only runs have no verdict to fail


def test_benches_with_pass_rules_return_their_result():
    """Every bench script hands main()'s return value to sys.exit, so a failure cannot be dropped on the way out."""
    root = Path(__file__).resolve().parent.parent / "sim" / "benches"
    for f in sorted(root.glob("*.py")):
        if f.stem in ("__init__", "common", "stats"):
            continue
        src = f.read_text(encoding="utf-8")
        assert "sys.exit(main())" in src, f"{f.name} must end with sys.exit(main())"
