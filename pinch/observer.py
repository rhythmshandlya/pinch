"""Behaviour-quality observer: reads what Pinch decided (Mind.decisions, Mind.phases, Mind.log) and reports
patterns that look unnatural - it never changes behaviour (decision 64).

    findings = observe(mind)            # or observe_records(decisions, phases, log, duration_s)
    for f in findings: print(f.kind, f.severity, f.evidence)

Checks:
  oscillation       two activities swapping back and forth (A B A B ...) within a short window
  failure_loop      the same activity started again and again after failing / timing out
  unexplained       a switch with no recorded reason
  state_lock        one top-level state holding more than its share of a long run (e.g. 90 % "idle")
  repetition        an identical sequence of activities repeating like an animation loop
  timeouts          activities ending by their time limit (each one is a design smell)
Each finding has an id, the time interval, evidence, a severity and a suggested next step, so it can be
turned into a regression test.
"""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Finding:
    kind: str
    severity: str                  # "info" | "warn" | "error"
    t0: float
    t1: float
    evidence: str
    suggestion: str
    data: dict = field(default_factory=dict)

    @property
    def id(self):
        return f"{self.kind}@{self.t0:.0f}"


def observe(mind, duration_s=None):
    return observe_records(list(mind.decisions), list(getattr(mind, "phases", [])), list(mind.log),
                           duration_s if duration_s is not None else mind.t)


def observe_records(decisions, phases, log, duration_s):
    out = []
    out += _oscillation(decisions)
    out += _failure_loops(decisions)
    out += _unexplained(decisions)
    out += _state_lock(decisions, duration_s)
    out += _repetition(decisions)
    out += _timeouts(log)
    return out


def _oscillation(decs, window=30.0, flips=3):
    out = []
    for i in range(len(decs) - 2 * flips):
        seq = decs[i:i + 2 * flips + 1]
        names = [d["to"] for d in seq]
        a, b = names[0], names[1]
        if a != b and all(n == (a if k % 2 == 0 else b) for k, n in enumerate(names)) \
                and seq[-1]["t"] - seq[0]["t"] < window:
            out.append(Finding("oscillation", "error", seq[0]["t"], seq[-1]["t"],
                               f"{a} <-> {b} {len(seq)} times in {seq[-1]['t'] - seq[0]['t']:.0f} s",
                               "add hysteresis (min_s / a larger 'better' margin) or fix the utility that flips",
                               dict(pair=(a, b))))
            break
    return out


def _failure_loops(decs, n=4, window=300.0):
    out = []
    restarts = [d for d in decs if d["prev"] == d["to"] or "timeout" in d["reason"]]
    by = Counter()
    first = {}
    for d in restarts:
        if d["t"] - first.get(d["to"], d["t"]) > window:
            by[d["to"]], first[d["to"]] = 0, d["t"]
        first.setdefault(d["to"], d["t"])
        by[d["to"]] += 1
        if by[d["to"]] == n:
            out.append(Finding("failure_loop", "warn", first[d["to"]], d["t"],
                               f"{d['to']} restarted {n} times within {window:.0f} s",
                               "make it give up sooner (ask for help, or a cool-down) or fix why it fails",
                               dict(activity=d["to"])))
    return out


def _unexplained(decs):
    bad = [d for d in decs if not d.get("reason")]
    return [Finding("unexplained", "error", bad[0]["t"], bad[-1]["t"], f"{len(bad)} switches without a reason",
                    "every _start must pass a reason")] if bad else []


def _state_lock(decs, duration, share=0.9, min_run=300.0):
    if duration < min_run or not decs:
        return []
    time_in = Counter()
    for d, nxt in zip(decs, decs[1:] + [None]):
        st = d.get("state", [None, None])[1]
        time_in[st] += (nxt["t"] if nxt else duration) - d["t"]
    st, s = time_in.most_common(1)[0]
    if s / duration > share and st not in ("rest",):            # (sleeping through a quiet night is fine)
        return [Finding("state_lock", "warn", 0.0, duration, f"'{st}' for {100 * s / duration:.0f} % of {duration:.0f} s",
                        "check that other activities can win (needs grow, boredom) in this situation",
                        dict(state=st, share=s / duration))]
    return []


def _repetition(decs, k=3, repeats=4):
    names = [d["to"] for d in decs]
    for i in range(len(names) - k * repeats + 1):
        pat = names[i:i + k]
        if len(set(pat)) > 1 and all(names[i + j * k:i + (j + 1) * k] == pat for j in range(repeats)):
            t0, t1 = decs[i]["t"], decs[i + k * repeats - 1]["t"]
            return [Finding("repetition", "info", t0, t1, f"{' > '.join(pat)} repeated {repeats} times",
                            "vary the choice (personality noise, boredom) so it doesn't look like a loop",
                            dict(pattern=pat))]
    return []


def _timeouts(log):
    tos = [(t, text) for t, kind, text in log if kind == "timeout"]
    return [Finding("timeouts", "info", tos[0][0], tos[-1][0], f"{len(tos)} activities ended by their time limit: "
                    + "; ".join(x for _, x in tos[:3]), "an activity that times out usually has a missing exit")] \
        if tos else []
