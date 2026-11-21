"""Small-sample statistics for the benches: how sure is a success count, and do two counts really differ?

    python sim/benches/stats.py 11/24             # one rate with its 95 % range
    python sim/benches/stats.py 11/24 2/24        # two rates: do they differ? (Fisher's exact test)
    python sim/benches/stats.py --need 0.46 0.67  # layouts per policy needed to tell two rates apart

Why this exists: with 24 layouts, 11 successes has a 95 % range of 28-65 %, so most differences between two
policies are chance. Every bench prints its counts through `rate()`, and a candidate is compared with the policy
in use through `compare()` (or `paired()` when both ran on the same layouts and the per-layout results are known).

Methods (all standard, pure Python so they also run in CI and on the robot):
- `wilson`: the Wilson score interval for one proportion, the usual choice for small samples.
- `fisher`: Fisher's exact test, two-sided, for two independent success counts.
- `mcnemar`: the exact McNemar test for paired yes/no outcomes (only the layouts where the two disagree count).
- `needed`: the normal-approximation sample size for comparing two proportions.
tests/test_stats.py pins them to reference values from SciPy.
"""

import argparse
import math
import sys

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """(low, high) of the Wilson score interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k, n):
    """'11/24 = 46 % (95 % range 28-65 %)': how every bench reports a success count."""
    if n <= 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f} % (95 % range {100 * lo:.0f}-{100 * hi:.0f} %)"


def fisher(k1, n1, k2, n2):
    """Two-sided p-value of Fisher's exact test for k1 of n1 against k2 of n2 (independent trials)."""
    total, row, col = n1 + n2, n1, k1 + k2

    def prob(x):
        return math.comb(col, x) * math.comb(total - col, row - x) / math.comb(total, row)

    p_obs = prob(k1)
    lo, hi = max(0, row - (total - col)), min(row, col)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def mcnemar(only_a, only_b):
    """Two-sided exact McNemar p-value: `only_a` layouts where A succeeded and B failed, `only_b` the reverse."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def needed(p1, p2, power_z=0.8416212335729143, z=Z95):
    """Trials per policy to tell rate p1 from p2 (two-sided 5 %, 80 % power by default)."""
    if p1 == p2:
        return math.inf
    mean = (p1 + p2) / 2
    num = z * math.sqrt(2 * mean * (1 - mean)) + power_z * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num * num / (p1 - p2) ** 2)


def compare(k1, n1, k2, n2, alpha=0.05):
    """Are k1/n1 and k2/n2 different? Returns dict(p, verdict, text); verdict is 'different' or 'inconclusive'."""
    p = fisher(k1, n1, k2, n2)
    verdict = "different" if p < alpha else "inconclusive"
    text = f"{rate(k1, n1)}  against  {rate(k2, n2)}:  p = {p:.3f}, {verdict}"
    if verdict == "inconclusive" and n1 and n2 and k1 / n1 != k2 / n2:
        text += f" (about {needed(k1 / n1, k2 / n2)} trials each would be needed to tell these rates apart)"
    return dict(p=p, verdict=verdict, text=text)


def paired(a_ok, b_ok, alpha=0.05):
    """The same question when A and B ran on the same layouts: a_ok and b_ok are equal-length lists of bools."""
    if len(a_ok) != len(b_ok):
        raise ValueError("paired results need the same layouts in the same order")
    only_a = sum(1 for a, b in zip(a_ok, b_ok) if a and not b)
    only_b = sum(1 for a, b in zip(a_ok, b_ok) if b and not a)
    p = mcnemar(only_a, only_b)
    verdict = "different" if p < alpha else "inconclusive"
    return dict(p=p, verdict=verdict, only_a=only_a, only_b=only_b,
                text=f"{rate(sum(a_ok), len(a_ok))}  against  {rate(sum(b_ok), len(b_ok))} on the same layouts: "
                     f"{only_a} won only by the first, {only_b} only by the second, p = {p:.3f}, {verdict}")


def _count(text):
    k, n = text.split("/")
    return int(k), int(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("counts", nargs="*", help="one or two results as successes/trials, e.g. 11/24 2/24")
    ap.add_argument("--need", nargs=2, type=float, metavar=("P1", "P2"), help="trials needed to tell two rates apart")
    a = ap.parse_args()
    if a.need:
        print(f"about {needed(*a.need)} trials per policy to tell {a.need[0]:.0%} from {a.need[1]:.0%}")
        return 0
    if len(a.counts) == 1:
        print(rate(*_count(a.counts[0])))
    elif len(a.counts) == 2:
        print(compare(*_count(a.counts[0]), *_count(a.counts[1]))["text"])
    else:
        ap.error("give one or two counts, or --need P1 P2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
