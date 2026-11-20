# Compute: every GPU behind one command

| Backend | For | Cost | Set up |
|---|---|---|---|
| `local` | smoke runs, CPU checks | free | `.venv-rl` |
| `ssh` | your machines and friends' machines on Tailscale (msi-1, a rented GPU, a server) | free | one entry in `ops/hosts.local.toml` + `ops hosts bootstrap` |
| `modal` | burst GPUs (A10G default) | paid; runs end when done, 24 h last-resort limit, checked ~4 h in | `pip install modal`, `modal token new` |
| `skypilot` | the cheapest cloud GPU across RunPod, Lambda, Vast.ai, AWS, GCP...; Linux servers as SSH Node Pools | paid (cloud) | `pip install "skypilot[runpod,lambda]"` on Linux/WSL, `sky check` |

All four run the same contract: snapshot a pushed commit, `python -m training.train --experiment ...`, write
`<run>/{params.pkl, progress.jsonl, config.json, run.json}`, fetch back to `training/checkpoints/<run>`.

## Add a GPU machine (for example a friend's)

1. **Network.** Both machines on Tailscale. For a friend's machine, they share it with you (Tailscale node
   sharing) instead of joining your tailnet; you get a stable `100.x` address or a MagicDNS name. Restrict it with
   a Tailscale ACL to SSH (port 22) if you like.
2. **SSH.** Key login from this laptop (`ssh-copy-id`, or append your public key to the remote
   `authorized_keys`; on Windows admin accounts: `C:\ProgramData\ssh\administrators_authorized_keys`). `ops` never
   types passwords (BatchMode).
3. **Describe it** in `ops/hosts.local.toml` (template: `ops/hosts.example.toml`): `os = "linux"` or
   `"windows-wsl"`, its VRAM, how many jobs at once.
4. **Prepare it:** `python -m ops hosts bootstrap <name>` clones the repo, installs uv and builds `.venv-rl` with
   CUDA JAX. On Windows, first install WSL Ubuntu and the NVIDIA driver (CUDA in WSL comes from the Windows driver).
5. **Check it:** `python -m ops hosts check` shows the GPU, the commit, the JAX/Brax/MuJoCo versions and any job.

## How a job runs on a host

- The worker fetches the repo and extracts the job's commit with `git archive` into `~/pinch-jobs/<run>/src`:
  jobs never share a working tree, so several can run at once or alongside development.
- Linux: a `setsid` background process. Windows: a one-shot S4U Scheduled Task running `wsl.exe` (an OpenSSH
  session kills its child processes on disconnect, and WSL stops a distro when its last `wsl.exe` exits).
- `timeout <minutes>` wraps the trainer; the exit code lands in `exit_code`; results come back as a tar stream.

## Keeping every GPU busy

`python -m ops submit <exp> --host any [--cloud-fallback]` queues a job; `python -m ops dispatch --watch` starts
queued jobs on the first host with enough free VRAM, or on Modal when the experiment allows it
(`cloud_ok = true`) and you asked for the fallback. `python -m ops jobs -r` is the board.

## Why these tools

- **Tailscale** (39k stars): a private network between your machines and shared ones without opening ports.
- **Plain SSH for own machines:** SkyPilot's SSH Node Pools deploy k3s with NVIDIA's GPU Operator, whose platform
  list has no WSL2, so the Windows laptops get a small SSH backend instead.
- **SkyPilot** (10.8k stars, UC Berkeley, NSDI '23): the standard way to reach the cheapest cloud GPU across
  providers, with managed jobs that survive preemption.
- **Modal**: already in use, per-second billing, a 24 h last-resort limit per run.
