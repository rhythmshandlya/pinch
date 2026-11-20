"""`python -m ops <command>`: the one entry point for checks, training jobs, hosts and worktrees.

    ops check                         lint + tests + layering + private-file guard (what CI and pre-commit run)
    ops envs                          the training environments (training/registry.py)
    ops hosts [check|bootstrap NAME]  the SSH/Tailscale GPU machines
    ops submit EXP [--host H|any] [--backend local|ssh|modal|skypilot]   start or queue a job
    ops jobs [--refresh]              the ledger, with live state from every backend
    ops dispatch [--watch]            start queued jobs on whichever host is free
    ops logs RUN | fetch RUN | stop RUN
    ops export RUN NAME               training/checkpoints/RUN -> policies/NAME.npz (.venv-rl)
    ops eval EXP NAME                 the experiment's benches on candidate policies/NAME
    ops worktree new TASK | list | remove TASK
    ops skills sync                   vendor the official MuJoCo agent skills for the installed mujoco
"""

import argparse
import os
import subprocess
import sys
import time

from ops import ROOT, gitutil, jobs
from ops.backends import get as backend

PAID = ("modal", "skypilot")


def _print_jobs(js):
    if not js:
        print("no jobs")
        return
    print(f"{'run':28} {'state':8} {'backend':8} {'host':10} {'step':>12} {'reward':>9} {'success':>8}  commit")
    for j in js:
        last = j.last or {}
        rew, suc = last.get("eval/episode_reward"), last.get("success")
        print(f"{j.run:28} {j.state:8} {j.backend:8} {j.host or '-':10} {last.get('step', 0):>12,} "
              f"{'' if rew is None else f'{rew:9.2f}':>9} {'' if suc is None else f'{suc:8.2f}':>8}  {j.sha[:8]}")


def cmd_envs(a):
    from training.registry import ENVS
    for s in ENVS.values():
        if a.all or not s.retired:
            print(f"{s.name:16} {s.skill:9} {'(retired) ' if s.retired else ''}{s.description}")


def cmd_hosts(a):
    from ops.hosts import BOOTSTRAP_SCRIPT, check, load_hosts, run_bash
    hosts = load_hosts()
    if not hosts:
        print("no hosts: copy ops/hosts.example.toml to ops/hosts.local.toml")
        return 1
    if a.action == "bootstrap":
        h = hosts[a.name]
        print(run_bash(h, BOOTSTRAP_SCRIPT.format(repo=h.repo, python=h.python, remote=gitutil.remote_url(),
                                                   branch="master"), timeout=3600))
        return 0
    for h in hosts.values():
        facts = check(h) if a.action == "check" else {}
        line = f"{h.name:12} {h.os:12} {h.vram_gb:>4} GB  {h.ssh}"
        if facts:
            line += ("  UNREACHABLE: " + facts["error"]) if "error" in facts else (
                f"\n    gpu: {facts.get('gpu', '?')}\n    repo: {facts.get('repo')}  python: {facts.get('python')}"
                f"\n    running: {' '.join(facts['running']) or 'nothing'}")
        print(line)


def _make_job(a):
    exp_path, exp = jobs.read_experiment(a.experiment)
    e, c = exp["experiment"], exp.get("compute", {})
    backend_name = a.backend or ("ssh" if a.host else "local")
    sha = gitutil.head()
    dirty = gitutil.dirty()
    if backend_name in ("ssh", "skypilot") and not gitutil.pushed(sha):
        raise SystemExit(f"commit {sha[:8]} is not on the remote: push it first (workers fetch it)")
    if dirty and backend_name != "local" and not a.allow_dirty:
        raise SystemExit("uncommitted changes: commit (and push) first, so the run is reproducible "
                         "(--allow-dirty to override; Modal would upload the dirty files)")
    if backend_name in PAID and not c.get("cloud_ok", False):
        raise SystemExit(f"{exp_path} has [compute] cloud_ok = false: not starting a paid {backend_name} job")
    run = a.run or jobs.unique_run(e.get("name", e["env"]))
    return jobs.Job(run=run, experiment=exp_path, backend=backend_name, host=a.host or "", sha=sha,
                    dirty=dirty and backend_name in ("local", "modal"), num_envs=a.num_envs or 0,
                    timesteps=a.timesteps or 0, timeout_min=a.timeout or int(c.get("timeout_min", 90)),
                    gpu=a.gpu or c.get("gpu", ""), cloud_ok=bool(c.get("cloud_ok", False)) and a.cloud_fallback)


def _small_gpu_envs(job, host):
    _, exp = jobs.read_experiment(job.experiment)
    c = exp.get("compute", {})
    if not job.num_envs and host.vram_gb < 12 and c.get("small_gpu_num_envs"):
        job.num_envs = int(c["small_gpu_num_envs"])


def _start(job):
    if job.backend == "ssh":
        from ops.hosts import load_hosts
        _small_gpu_envs(job, load_hosts()[job.host])
    if job.backend in PAID:
        print(f"paid backend: {job.backend} {job.gpu or 'A10G'}, last-resort limit {job.timeout_min} min; check it ~4 h in")
    backend(job.backend).submit(job)
    jobs.upsert(job)
    print(f"started {job.run} on {job.backend}{(' ' + job.host) if job.host else ''} at {job.sha[:8]}")


def cmd_submit(a):
    job = _make_job(a)
    if job.host == "any":
        job.state = "queued"
        jobs.upsert(job)
        print(f"queued {job.run}; `ops dispatch` starts it on the first free host"
              + (" (or Modal)" if job.cloud_ok else ""))
        return cmd_dispatch(argparse.Namespace(watch=False, interval=0))
    _start(job)


def _free_hosts(min_vram, active):
    from ops.hosts import check, load_hosts
    free = []
    for h in load_hosts().values():
        if h.vram_gb < min_vram:
            continue
        facts = check(h)
        if "error" in facts:
            continue
        busy = len(facts["running"]) + sum(1 for j in active if j.host == h.name and j.state == "queued")
        if busy < h.max_jobs:
            free.append(h)
    return free


def cmd_dispatch(a):
    while True:
        queued = [j for j in jobs.load() if j.state == "queued"]
        for job in queued:
            _, exp = jobs.read_experiment(job.experiment)
            need = float(exp.get("compute", {}).get("min_vram_gb", 0))
            free = _free_hosts(need, [])
            if free:
                job.backend, job.host = "ssh", free[0].name
                _start(job)
            elif job.cloud_ok:
                job.backend, job.host = "modal", ""
                _start(job)
            else:
                print(f"{job.run}: no free host with >= {need} GB yet")
        if not a.watch:
            return
        time.sleep(a.interval)


def _refresh(js):
    for j in js:
        if j.state != "running":
            continue
        try:
            state, last, _ = backend(j.backend).status(j)
            j.state, j.last = state, last or j.last
        except Exception as e:                    # noqa: BLE001 - one unreachable backend must not hide the rest
            j.note = f"status failed: {str(e).splitlines()[-1][:120]}"
        jobs.upsert(j)
    return jobs.load()


def cmd_jobs(a):
    js = jobs.load()
    if a.refresh:
        js = _refresh(js)
    if not a.all:
        js = [j for j in js if j.state in jobs.ACTIVE] + [j for j in js if j.state not in jobs.ACTIVE][-8:]
    _print_jobs(js)


def cmd_logs(a):
    j = jobs.get(a.run)
    print(backend(j.backend).logs(j, a.n))


def cmd_fetch(a):
    j = jobs.get(a.run)
    dest = ROOT / "training" / "checkpoints" / j.run
    backend(j.backend).fetch(j, dest)
    print(f"{j.run} -> {dest}")


def cmd_stop(a):
    j = jobs.get(a.run)
    print(backend(j.backend).stop(j))
    j.state = "stopped"
    jobs.upsert(j)


def _rl_python():
    for p in (ROOT / ".venv-rl" / "Scripts" / "python.exe", ROOT / ".venv-rl" / "bin" / "python"):
        if p.exists():
            return str(p)
    raise SystemExit("no .venv-rl: export needs the training environment (requirements-train.txt)")


def cmd_export(a):
    return subprocess.call([_rl_python(), "training/export_policy.py", a.run, a.name], cwd=ROOT)


def cmd_eval(a):
    _, exp = jobs.read_experiment(a.experiment)
    ev = exp.get("eval")
    if not ev:
        raise SystemExit("this experiment has no [eval]")
    out_dir = ROOT / ".ops" / "eval" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{f"PINCH_{ev['skill'].upper()}_POLICY": a.name})
    code = 0
    for i, c in enumerate(ev["commands"]):
        argv = c.format(candidate=a.name).split()
        js = out_dir / f"{i}_{os.path.basename(argv[0]).removesuffix('.py')}.json"
        print(f"== {' '.join(argv)}", flush=True)
        code |= subprocess.call([sys.executable, *argv, "--json", str(js)], cwd=ROOT, env=env)
    print(f"results in {out_dir}")
    return code


def cmd_check(a):
    from ops.guard import check_tree
    steps = [("guard", lambda: check_tree()),
             ("ruff", lambda: subprocess.call([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT)),
             ("tests", lambda: subprocess.call([sys.executable, "tests/run.py", *a.only], cwd=ROOT))]
    failed = []
    for name, fn in steps:
        print(f"== {name}", flush=True)
        if fn():
            failed.append(name)
    print("all checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_worktree(a):
    from ops import worktree
    if a.action == "new":
        print(worktree.new(a.task, a.base))
    elif a.action == "remove":
        worktree.remove(a.task)
    else:
        print(worktree.list_())


def cmd_skills(a):
    from ops import skills
    for line in skills.sync():
        print(line)


def build_parser():
    ap = argparse.ArgumentParser(prog="ops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="guard + ruff + tests")
    p.add_argument("only", nargs="*", help="test file filters")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("envs", help="training environments")
    p.add_argument("--all", action="store_true", help="include retired ones")
    p.set_defaults(fn=cmd_envs)
    p = sub.add_parser("hosts", help="SSH/Tailscale GPU hosts")
    p.add_argument("action", nargs="?", default="list", choices=["list", "check", "bootstrap"])
    p.add_argument("name", nargs="?")
    p.set_defaults(fn=cmd_hosts)
    p = sub.add_parser("submit", help="start (or queue) a training job")
    p.add_argument("experiment")
    p.add_argument("--backend", choices=["local", "ssh", "modal", "skypilot"])
    p.add_argument("--host", help="an ops/hosts.local.toml host, or 'any' to queue for the first free one")
    p.add_argument("--run")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--timesteps", type=int)
    p.add_argument("--timeout", type=int, help="minutes (default: the experiment's)")
    p.add_argument("--gpu", help="Modal/SkyPilot GPU type (default: the experiment's)")
    p.add_argument("--cloud-fallback", action="store_true", help="with --host any: use Modal if no host is free")
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("jobs", help="the job ledger")
    p.add_argument("--refresh", "-r", action="store_true", help="ask every backend for live state")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_jobs)
    p = sub.add_parser("dispatch", help="start queued jobs on free hosts")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=300)
    p.set_defaults(fn=cmd_dispatch)
    for name, fn in (("logs", cmd_logs), ("fetch", cmd_fetch), ("stop", cmd_stop)):
        p = sub.add_parser(name)
        p.add_argument("run")
        if name == "logs":
            p.add_argument("-n", type=int, default=40)
        p.set_defaults(fn=fn)
    p = sub.add_parser("export", help="checkpoint -> policies/NAME.npz")
    p.add_argument("run")
    p.add_argument("name")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("eval", help="run an experiment's benches on a candidate policy")
    p.add_argument("experiment")
    p.add_argument("name", help="policies/<name>.npz")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("worktree", help="one git worktree per parallel task")
    p.add_argument("action", choices=["new", "list", "remove"])
    p.add_argument("task", nargs="?")
    p.add_argument("--base", default="master")
    p.set_defaults(fn=cmd_worktree)
    p = sub.add_parser("skills", help="agent skills")
    p.add_argument("action", choices=["sync"])
    p.set_defaults(fn=cmd_skills)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
