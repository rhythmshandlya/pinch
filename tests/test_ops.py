"""The operations layer (docs/ARCHITECTURE.md sections 3-4): the env registry, presets and experiments agree;
job scripts are well-formed for both host kinds; the guard catches private files, tailnet addresses and
attribution lines. None of this needs JAX, a GPU or the network."""

import base64
import tomllib
from pathlib import Path

from ops import guard
from ops.backends import jobscript, skypilot
from ops.hosts import Host, remote_command
from ops.jobs import Job, unique_run

ROOT = Path(__file__).resolve().parents[1]


def _presets():
    return tomllib.loads((ROOT / "configs/train/presets.toml").read_text(encoding="utf-8"))


def test_registry_is_consistent():
    from training.registry import ENVS
    assert len(ENVS) >= 10
    for s in ENVS.values():
        mod, attr = s.factory.split(":")
        src = (ROOT / (mod.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert f"class {attr}" in src or f"def {attr}" in src, f"{s.name}: {s.factory} not found"
        rmod, rattr = s.randomize.split(":")
        assert f"def {rattr}" in (ROOT / (rmod.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        assert s.skill in ("gait", "emote", "righting", "grasp", "push"), s.name


def test_every_experiment_resolves():
    from training.registry import ENVS
    presets = _presets()
    exps = sorted((ROOT / "configs/experiments").glob("*.toml"))
    assert len(exps) >= 5
    for p in exps:
        exp = tomllib.loads(p.read_text(encoding="utf-8"))
        e = exp["experiment"]
        assert e["env"] in ENVS and not ENVS[e["env"]].retired, f"{p.name}: env {e['env']}"
        assert e["preset"] in presets, f"{p.name}: preset {e['preset']}"
        for k in exp.get("overrides", {}):
            assert k in presets[e["preset"]], f"{p.name}: override {k} is not a preset key"
        c = exp.get("compute", {})
        assert c.get("timeout_min", 0) > 0, f"{p.name}: every job needs a hard timeout"
        ev = exp.get("eval")
        if ev:
            for cmd in ev["commands"]:
                assert (ROOT / cmd.split()[0]).exists(), f"{p.name}: {cmd}"


def test_presets_match_ppo_keys():
    keys = {"num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length", "num_updates_per_batch",
            "episode_length", "num_evals", "num_eval_envs", "learning_rate", "entropy_cost", "discounting",
            "policy_layers", "value_layers"}
    for name, cfg in _presets().items():
        assert set(cfg) == keys, f"preset {name}: {set(cfg) ^ keys}"


def test_remote_command_survives_both_shells():
    script = "echo \"quotes 'and' $HOME\"\nexit 0\n"
    lin = remote_command(Host("a", "u@h", os="linux"), script)
    win = remote_command(Host("b", "u@h", os="windows-wsl"), script)
    for cmd in (lin, win):
        b64 = cmd.split("echo ")[1].split(" |")[0]
        assert base64.b64decode(b64).decode() == script
        assert "'" not in cmd and "$" not in cmd            # nothing for PowerShell or a login shell to expand
    assert win.startswith("wsl -d Ubuntu -- bash -c ")


def test_job_scripts_are_complete():
    job = Job(run="r1", experiment="configs/experiments/smoke.toml", backend="ssh", host="h", sha="a" * 40,
              timeout_min=7, num_envs=512)
    kw = dict(job_root=jobscript.JOB_ROOT, stop_when_done=1, run=job.run, repo="~/pinch", python=".venv-rl/bin/python", sha=job.sha,
              remote="https://example.invalid/pinch.git", experiment=job.experiment, timeout_min=7,
              extra="--num-envs 512")
    prep = jobscript.PREPARE.format(**kw)
    assert "git -C \"$REPO\" archive " + "a" * 40 in prep and "timeout 7m" in prep and "--num-envs 512" in prep
    assert "{" not in prep.replace("${", "").replace("^{commit}", "")       # every placeholder filled
    assert 'runpodctl pod stop "$RUNPOD_POD_ID"' in prep and '[ "1" = "1" ]' in prep
    assert jobscript.parse_status("exit_code=0\nlast={\"step\": 5}") == ("done", {"step": 5})
    assert jobscript.parse_status("exit_code=1")[0] == "failed"
    assert jobscript.parse_status("exit_code=143")[0] == "stopped"
    assert jobscript.parse_status("exit_code=124")[0] == "timeout"
    assert jobscript.parse_status("state=running")[0] == "running"
    assert jobscript.parse_status("state=lost")[0] == "lost"
    y = skypilot.task_yaml(job)
    assert "accelerators: A10G:1" in y and "timeout 7m" in y and "PINCH_RESULT_B64" in y


def test_unique_run_never_reuses_a_name():
    assert unique_run("zz-test", taken=set()) == "zz-test"
    assert unique_run("zz-test", taken={"zz-test", "zz-test-2"}) == "zz-test-3"


def test_guard():
    assert guard.check_files(["docs/PRD.md"]) and guard.check_files(["ops/hosts.local.toml"])
    assert not guard.check_files(["pinch/mind/core.py"])
    assert guard.check_message("fix\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    assert not guard.check_message("fix the gait")
    assert guard.TAILNET_IP.search("ssh u@" + ".".join(["100", "101", "2", "3"])) and not guard.TAILNET_IP.search("v 100.2.3.4")
    assert guard.check_tree() == 0, "a tracked file is private or holds a tailnet address"


def test_paid_host_budget_caps_the_job():
    from ops.hosts import budget_timeout_min
    assert budget_timeout_min(Host("free", "u@h")) == 0
    h = Host("pod", "root@1.2.3.4", port=40122, price_per_h=0.89, budget_usd=13.0)
    m = budget_timeout_min(h)
    assert 0.89 * m / 60 <= 13.0 * 0.85 + 1e-9 and m > 700        # ~12.4 h for $13 at $0.89/h, with slack


def test_seed_is_recorded_and_reaches_every_backend():
    """One training seed per policy cannot show whether a change helped (the audit of 2026-11-21): the seed is part
    of the resolved config, so run.json names it, and `ops submit --seed N` carries it to every backend."""
    from ops.backends import ssh
    try:
        from training.train import resolve_config
    except ModuleNotFoundError as e:                           # the commit hook's bare Python has no numpy
        print(f"    (trainer part skipped: {e.name} not installed)")
    else:
        assert resolve_config("smoke")["seed"] == 0
        assert resolve_config("smoke", seed=3)["seed"] == 3
        try:
            resolve_config("smoke", overrides=dict(seed=1))
        except ValueError:
            pass                                               # the seed belongs in [experiment] or --seed
        else:
            raise AssertionError("a seed among the preset overrides must be refused")
    job = Job(run="r1", experiment="configs/experiments/smoke.toml", backend="ssh", host="h", sha="a" * 40, seed=3)
    host = Host(name="h", ssh="u@h", repo="~/pinch", python=".venv-rl/bin/python")
    assert "--seed 3" in ssh._fmt(jobscript.PREPARE, job, host)
    assert "--seed 3" in skypilot.task_yaml(job)
    plain = Job(run="r2", experiment="configs/experiments/smoke.toml", backend="ssh", host="h", sha="a" * 40)
    assert "--seed" not in ssh._fmt(jobscript.PREPARE, plain, host) and "--seed" not in skypilot.task_yaml(plain)
    for backend in ("modal", "local"):                          # the same three-line pattern as --num-envs
        src = (ROOT / "ops" / "backends" / f"{backend}.py").read_text(encoding="utf-8")
        assert '"--seed", str(job.seed)' in src, backend


def test_installer_copies_only_folders_that_exist():
    """deploy/install.sh once copied a folder that had been removed, and failed at its second step on a clean
    checkout. Every folder it copies must exist, and it must take every package the robot imports."""
    sh = (ROOT / "deploy" / "install.sh").read_text(encoding="utf-8")
    line = next(ln for ln in sh.splitlines() if ln.startswith("cp -r ") and '"$DEST"/' in ln)
    folders = line.split()[2:-1]
    assert folders, line
    for f in folders:
        assert (ROOT / f).is_dir(), f"deploy/install.sh copies {f}/, which does not exist"
    assert {"crab", "pinch", "robot", "policies"} <= set(folders), folders


def test_training_python_is_found_from_a_task_worktree():
    """A task worktree has no .venv-rl of its own; local jobs and exports must use the main checkout's."""
    import tempfile

    from ops import gitutil
    with tempfile.TemporaryDirectory() as tmp:
        wt, main = Path(tmp) / "wt", Path(tmp) / "main"
        exe = main / ".venv-rl" / "bin" / "python"
        exe.parent.mkdir(parents=True)
        wt.mkdir()
        exe.write_text("")
        assert gitutil.training_python((wt, main)) == str(exe)
        assert gitutil.training_python((wt,)) is None
    assert gitutil.main_checkout().is_dir()


def test_an_experiment_chooses_its_robot_design_before_anything_loads_it():
    """`design = "v2"` in an experiment sets CRAB_DESIGN for the trainer; a process already running the other
    design refuses instead of training the wrong robot."""
    import os
    try:
        from training import train
    except ModuleNotFoundError:                       # the commit hook's bare Python has no numpy
        return
    import crab.params as cp
    assert train.apply_design(["--experiment", "configs/experiments/gait-v2.toml"]) == os.environ.get("CRAB_DESIGN", "v1")
    if cp.DESIGN == "v2":
        assert train.apply_design(["--experiment", "configs/experiments/gait-v2-sts3215.toml"]) == "v2"
        return
    try:
        train.apply_design(["--experiment", "configs/experiments/gait-v2-sts3215.toml"])
    except SystemExit as e:
        assert "v1" in str(e)
    else:
        raise AssertionError("a v1 process accepted a v2 experiment")
    assert os.environ.get("CRAB_DESIGN", "v1") == "v1"
