"""The behaviour scenarios (sim/scenarios/*.toml) on held-out test worlds: how often does Pinch behave as it should?

    python sim/benches/worlds.py                    # every scenario on 24 seeds (its own world and timeline each)
    python sim/benches/worlds.py touch tricks -n 48 # some scenarios, more seeds
    python sim/benches/worlds.py --json out.json  # runs are also saved as they finish (out.json.runs.jsonl): run
                                                  # the same command again after a crash and it carries on

Each run picks a held-out test world (sim/worlds: homes, desks, tables, floors, clutter, light), samples the
scenario's event times and words, runs the full pet (mind + body + robot runtime + physics) and checks the
scenario's assertions over time (sim/worlds/scenario.py). Reported per scenario: passes / runs, the 95 % range
(Wilson), the failure count per assertion and the worlds it failed in (to replay: sim/worlds/scenario.py run).
A run whose situation doesn't fit its world (no room to place a thing) is counted apart, not as a fail.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402
from sim.benches.stats import wilson  # noqa: E402
from sim.worlds.scenario import all_scenarios, load, run  # noqa: E402

SEEDS = 24


def _job(args):
    path, seed = args
    return run(load(path), seed)


def summarise(results):
    out = {}
    for name in dict.fromkeys(r["scenario"] for r in results):
        rs = [r for r in results if r["scenario"] == name]
        counted = [r for r in rs if not (r["error"] and r["error"][0] == "unplaceable")]
        k, n = sum(r["ok"] for r in counted), len(counted)
        lo, hi = wilson(k, n) if n else (0.0, 0.0)
        fails = Counter(a["name"] for r in counted for a in r["asserts"] if not a["ok"])
        crashes = [r["error"][1] for r in counted if r["error"]]
        out[name] = dict(passed=k, runs=n, rate=k / n if n else 0.0, lo=lo, hi=hi, unplaceable=len(rs) - n,
                         failed_asserts=dict(fails), crashes=crashes[:3],
                         failed_worlds=[r["world"] for r in counted if not r["ok"]][:8],
                         by_kind={kd: (sum(r["ok"] for r in counted if r["kind"] == kd),
                                       sum(r["kind"] == kd for r in counted)) for kd in ("desk", "table", "floor")})
    return out


def table(summary):
    lines = ["| Scenario | Passed | Rate | 95 % range | Desk / table / floor | Failing checks |", "|---|---|---|---|---|---|"]
    for name, s in summary.items():
        kinds = " / ".join(f"{a}/{b}" if b else "-" for a, b in s["by_kind"].values())
        fails = ", ".join(f"{k} ({v})" for k, v in s["failed_asserts"].items()) or "-"
        lines.append(f"| {name} | {s['passed']}/{s['runs']} | {s['rate']:.0%} | {s['lo']:.0%}-{s['hi']:.0%} | {kinds} | {fails} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("-n", "--seeds", type=int, default=SEEDS)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json")
    a = ap.parse_args()
    scs = [s for s in all_scenarios() if not a.only or s.name in a.only]
    # longest first, so the pool doesn't end on one long run
    jobs = sorted([(s.path, seed) for s in scs for seed in range(a.first_seed, a.first_seed + a.seeds)],
                  key=lambda j: -next(s.seconds for s in scs if s.path == j[0]))
    t0 = time.time()
    print(f"{len(scs)} scenarios x {a.seeds} seeds = {len(jobs)} runs on held-out test worlds")
    log = Path(a.json + ".runs.jsonl") if a.json else None
    done = {}
    if log is not None and log.exists():               # carry on after a crash: keep the runs already finished
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["scenario"], r["seed"])] = r
        print(f"{len(done)} runs already done ({log})")
    name = {s.path: s.name for s in scs}
    todo = [j for j in jobs if (name[j[0]], j[1]) not in done]

    def keep(job, r):
        if log is not None:
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "\n")

    for _ in range(2):                                 # a run lost to the machine (out of memory) is tried again
        res = run_parallel(_job, todo, workers=a.workers, fresh_every=5, on_result=keep, keep_going=True)
        for j, r in zip(todo, res):
            if r is not None:
                done[(name[j[0]], j[1])] = r
        todo = [j for j, r in zip(todo, res) if r is None]
        if not todo:
            break
    if todo:
        print(f"{len(todo)} runs lost twice and left out: {[(name[p], sd) for p, sd in todo]}")
    results = [done[(name[p], sd)] for p, sd in jobs if (name[p], sd) in done]
    summary = summarise(results)
    print(table(summary))
    print(f"\n{sum(s['passed'] for s in summary.values())}/{sum(s['runs'] for s in summary.values())} runs passed "
          f"in {(time.time() - t0) / 60:.1f} min")
    save_json(a.json, dict(summary=summary, runs=results))
    return 0                                  # a report: the pass rates are the result, there is no pass rule yet


if __name__ == "__main__":
    sys.exit(main())
