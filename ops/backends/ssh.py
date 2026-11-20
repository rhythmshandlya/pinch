"""Any GPU machine reachable over SSH (normally over Tailscale): Linux directly, Windows through WSL.

Windows detail: a process started from an OpenSSH session dies when the session closes (the session's job
object), and WSL stops a distro when its last wsl.exe exits. So on Windows the job runs under a one-shot
Scheduled Task (S4U: runs without the user logged in, no stored password), which keeps its wsl.exe alive for the
whole run. On Linux a setsid'd background process is enough.
"""

import base64
import io
import tarfile

from ops import gitutil
from ops.backends import jobscript as js
from ops.hosts import load_hosts, run_bash, run_powershell
from ops.jobs import now


def _host(job):
    hosts = load_hosts()
    if job.host not in hosts:
        raise KeyError(f"unknown host {job.host!r}; hosts: {', '.join(hosts) or 'none (ops/hosts.local.toml)'}")
    return hosts[job.host]


def _fmt(tmpl, job, host, **kw):
    extra = []
    if job.num_envs:
        extra += ["--num-envs", str(job.num_envs)]
    if job.timesteps:
        extra += ["--timesteps", str(job.timesteps)]
    return tmpl.format(job_root=host.job_root, stop_when_done=int(host.stop_when_done), run=job.run, repo=host.repo, python=host.python, sha=job.sha,
                       remote=gitutil.remote_url(), experiment=job.experiment, timeout_min=job.timeout_min,
                       extra=" ".join(extra), **kw)


def _task_name(job):
    return f"PinchJob-{job.run}"


def launch_windows(job, host):
    # PowerShell single-quoted string: nothing inside is expanded ($HOME must reach bash, not PowerShell);
    # a literal ' is written ''
    wsl_cmd = f"-d {host.wsl_distro} -- bash -lc 'bash {host.job_root}/{job.run}/job.sh'".replace("'", "''")
    ps = f"""
$a = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument '{wsl_cmd}'
$p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$s = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName '{_task_name(job)}' -Action $a -Principal $p -Settings $s -Force | Out-Null
Start-ScheduledTask -TaskName '{_task_name(job)}'
'launched'
"""
    return run_powershell(host, ps)


def submit(job):
    host = _host(job)
    run_bash(host, _fmt(js.PREPARE, job, host), timeout=300)
    if host.os == "windows-wsl":
        launch_windows(job, host)
    else:
        run_bash(host, _fmt(js.LAUNCH_LINUX, job, host))
    job.state, job.started = "running", now()
    job.handle = dict(host=host.name, job_dir=f"{host.job_root}/{job.run}")
    return job


def status(job):
    host = _host(job)
    out = run_bash(host, _fmt(js.STATUS, job, host), timeout=40)
    state, last = js.parse_status(out)
    return state, last, out.strip()


def logs(job, n=40):
    host = _host(job)
    return run_bash(host, _fmt(js.LOGS, job, host, n=n), timeout=40)


def fetch(job, dest):
    host = _host(job)
    b64 = run_bash(host, _fmt(js.FETCH, job, host), timeout=600)
    data = base64.b64decode(b64.strip())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():                        # files of this run only, nothing outside dest
            if m.isfile() and m.name.startswith(job.run + "/") and ".." not in m.name:
                m.name = m.name[len(job.run) + 1:]
                tf.extract(m, dest, filter="data")
    return dest


def stop(job):
    host = _host(job)
    out = run_bash(host, _fmt(js.STOP, job, host), timeout=40)
    if host.os == "windows-wsl":
        try:
            run_powershell(host, f"Unregister-ScheduledTask -TaskName '{_task_name(job)}' -Confirm:$false")
        except RuntimeError:
            pass
    return out
