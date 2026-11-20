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
    kw = dict(job_root=jobscript.JOB_ROOT, run=job.run, repo="~/pinch", python=".venv-rl/bin/python", sha=job.sha,
              remote="https://example.invalid/pinch.git", experiment=job.experiment, timeout_min=7,
              extra="--num-envs 512")
    prep = jobscript.PREPARE.format(**kw)
    assert "git -C \"$REPO\" archive " + "a" * 40 in prep and "timeout 7m" in prep and "--num-envs 512" in prep
    assert "{" not in prep.replace("${", "").replace("^{commit}", "")       # every placeholder filled
    assert jobscript.parse_status("exit_code=0\nlast={\"step\": 5}") == ("done", {"step": 5})
    assert jobscript.parse_status("exit_code=1")[0] == "failed"
    assert jobscript.parse_status("exit_code=143")[0] == "stopped"
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
