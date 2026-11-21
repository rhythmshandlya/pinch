"""The bench statistics give the textbook answers. Reference values are from SciPy 1.18.1
(binomtest(...).proportion_ci(method="wilson"), fisher_exact, and binomtest(min, n, 0.5) for McNemar)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.benches import stats  # noqa: E402

TOL = 1e-6


def test_wilson_interval_matches_reference():
    ref = {(11, 24): (0.278913, 0.649251), (2, 24): (0.023159, 0.258488), (0, 10): (0.0, 0.277533),
           (10, 10): (0.722467, 1.0), (19, 48): (0.270208, 0.536896), (131, 200): (0.586756, 0.717402)}
    for (k, n), (lo, hi) in ref.items():
        got = stats.wilson(k, n)
        assert abs(got[0] - lo) < TOL and abs(got[1] - hi) < TOL, (k, n, got)
    assert stats.wilson(0, 0) == (0.0, 1.0)                      # nothing run: nothing known


def test_fisher_exact_matches_reference():
    ref = {(19, 48, 14, 48): 0.390241, (11, 24, 2, 24): 0.007841, (2, 24, 10, 24): 0.017337,
           (10, 24, 9, 24): 1.0, (0, 5, 5, 5): 0.007937, (11, 24, 16, 24): 0.244315}
    for args, p in ref.items():
        assert abs(stats.fisher(*args) - p) < TOL, (args, stats.fisher(*args))
    assert stats.fisher(11, 24, 2, 24) == stats.fisher(2, 24, 11, 24)        # order does not matter


def test_mcnemar_exact_matches_reference():
    ref = {(5, 1): 0.21875, (3, 3): 1.0, (8, 0): 0.0078125, (1, 9): 0.021484, (12, 4): 0.076813}
    for (b, c), p in ref.items():
        assert abs(stats.mcnemar(b, c) - p) < TOL, (b, c, stats.mcnemar(b, c))
    assert stats.mcnemar(0, 0) == 1.0


def test_small_benches_are_called_inconclusive():
    """The cases behind the audit: 19 against 14 of 48 is chance; 11 against 2 of 24 is a real difference."""
    assert stats.compare(19, 48, 14, 48)["verdict"] == "inconclusive"
    assert stats.compare(10, 24, 9, 24)["verdict"] == "inconclusive"
    assert stats.compare(11, 24, 2, 24)["verdict"] == "different"
    assert stats.compare(2, 24, 10, 24)["verdict"] == "different"
    assert 80 <= stats.needed(0.46, 0.67) <= 90 and 380 <= stats.needed(0.46, 0.56) <= 400


def test_paired_layouts_count_only_disagreements():
    a = [True, True, True, False, False, True, True, True]
    b = [False, False, True, False, False, True, False, False]
    r = stats.paired(a, b)
    assert (r["only_a"], r["only_b"]) == (4, 0) and abs(r["p"] - 0.125) < TOL and r["verdict"] == "inconclusive"
    assert stats.paired([True] * 8 + [False] * 2, [False] * 8 + [False] * 2)["verdict"] == "different"
    try:
        stats.paired([True], [True, False])
    except ValueError:
        pass
    else:
        raise AssertionError("unequal lengths must be refused")


def test_rate_reads_plainly():
    assert stats.rate(11, 24) == "11/24 = 46 % (95 % range 28-65 %)"
    assert stats.rate(0, 0) == "0/0"
