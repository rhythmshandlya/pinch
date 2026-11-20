"""GPU machines reachable over SSH (normally over Tailscale): msi-1, a friend's rented GPU, a future server.

Hosts live in ops/hosts.local.toml (gitignored: addresses are private); ops/hosts.example.toml is the template.

    [hosts.msi-1]
    ssh = "user@100.x.y.z"          # anything `ssh` accepts, including a ~/.ssh/config alias
    os = "windows-wsl"              # "linux" or "windows-wsl" (jobs run inside WSL)
    wsl_distro = "Ubuntu"
    repo = "~/pinch"                # the worker's clone (inside WSL on Windows)
    python = ".venv-rl/bin/python"  # relative to repo
    vram_gb = 6
    max_jobs = 1
"""

import base64
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass

from ops import ROOT

HOSTS_FILE = ROOT / "ops" / "hosts.local.toml"
EXAMPLE_FILE = ROOT / "ops" / "hosts.example.toml"


@dataclass
class Host:
    name: str
    ssh: str
    os: str = "linux"
    wsl_distro: str = "Ubuntu"
    repo: str = "~/pinch"
    python: str = ".venv-rl/bin/python"
    vram_gb: float = 8
    max_jobs: int = 1
    gpu: str = ""
    notes: str = ""
    port: int = 22                  # a Runpod pod's direct SSH is on a mapped port
    price_per_h: float = 0.0        # paid hosts (a rented pod): jobs get a budget-derived timeout
    budget_usd: float = 0.0         # spend at most this on one job here (0: no budget limit)
    job_root: str = "$HOME/pinch-jobs"   # a Runpod pod: /workspace/pinch-jobs (the container disk is wiped on stop)
    stop_when_done: bool = False    # a Runpod pod: stop itself when the job ends, so an idle GPU doesn't bill

    def __post_init__(self):
        if self.os not in ("linux", "windows-wsl"):
            raise ValueError(f"host {self.name}: os must be 'linux' or 'windows-wsl', not {self.os!r}")


def load_hosts(path=None):
    p = path or HOSTS_FILE
    if not p.exists():
        return {}
    data = tomllib.loads(p.read_text(encoding="utf-8")).get("hosts", {})
    return {name: Host(name=name, **cfg) for name, cfg in data.items()}


def ssh_exe():
    """ssh on PATH, else Git for Windows' copy (Windows' own OpenSSH client is often not on PATH)."""
    found = os.environ.get("PINCH_SSH") or shutil.which("ssh")
    if found:
        return found
    for p in (r"C:\Program Files\Git\usr\bin\ssh.exe", r"C:\Windows\System32\OpenSSH\ssh.exe"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no ssh client found; set PINCH_SSH")


def remote_command(host, script):
    """The command line that runs a bash script on the host (inside WSL on Windows). The script travels as
    base64 in the argument, so no quoting survives PowerShell, cmd or a login shell differently."""
    b64 = base64.b64encode(script.encode()).decode()
    inner = f"echo {b64} | base64 -d | bash"
    if host.os == "windows-wsl":
        return f'wsl -d {host.wsl_distro} -- bash -c "{inner}"'
    return inner


def _ssh_base(host):
    return [ssh_exe(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(host.port), host.ssh]


def budget_timeout_min(host, margin=0.85):
    """Minutes a job may run on a paid host before its budget is spent (85 % of it, the rest is slack for setup
    and fetching). 0 when the host has no budget."""
    if not (host.price_per_h and host.budget_usd):
        return 0
    return int(60 * margin * host.budget_usd / host.price_per_h)


def run_bash(host, script, timeout=60, check=True):
    """Run script on host; returns stdout. Batch mode: never prompts (key login only)."""
    cmd = [*_ssh_base(host), remote_command(host, script)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{host.name}: exit {r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


def run_powershell(host, ps, timeout=60):
    """A PowerShell command on a Windows host (its SSH shell is PowerShell). Encoded, so quotes survive."""
    enc = base64.b64encode(ps.encode("utf-16-le")).decode()
    cmd = [*_ssh_base(host), f"powershell -NoProfile -NonInteractive -EncodedCommand {enc}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{host.name}: exit {r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


CHECK_SCRIPT = r"""
set -u
echo "uname=$(uname -sr)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)"
R={repo}
if [ -d "$R/.git" ]; then echo "repo=$(git -C $R rev-parse --short HEAD)"; else echo "repo=missing"; fi
if [ -x "$R/{python}" ]; then echo "python=$($R/{python} -c 'import jax,brax,mujoco;print(jax.__version__,brax.__version__,mujoco.__version__,jax.default_backend())' 2>&1 | tail -1)"; else echo "python=missing"; fi
echo "jobs=$(ls -d $HOME/pinch-jobs/*/ 2>/dev/null | while read d; do if [ -f "$d/pid" ] && [ ! -f "$d/exit_code" ]; then basename $d; fi; done | tr '\n' ' ')"
"""


def check(host, timeout=40):
    """{key: value} health facts, or {"error": ...} when the host can't be reached."""
    try:
        out = run_bash(host, CHECK_SCRIPT.format(repo=host.repo, python=host.python), timeout=timeout)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        return {"error": str(e).splitlines()[-1] if str(e) else type(e).__name__}
    facts = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    facts["running"] = facts.get("jobs", "").split()
    return facts


BOOTSTRAP_SCRIPT = r"""
set -euo pipefail
R={repo}
if [ ! -d "$R/.git" ]; then git clone -q {remote} "$R"; fi
cd "$R" && git fetch -q origin && git checkout -q {branch} && git pull -q --ff-only || true
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
if [ ! -x "$R/{python}" ]; then uv venv -q --python 3.12 "$R/.venv-rl"; fi
uv pip install -q --python "$R/{python}" -r requirements-train.txt "jax[cuda12]==0.9.2"
"$R/{python}" -c 'import jax;print("jax", jax.__version__, jax.devices())'
"""
