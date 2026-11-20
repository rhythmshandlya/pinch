---
name: pinch-add-gpu-host
description: Connect a new GPU machine to Pinch's training fleet over Tailscale and SSH - a friend's rented GPU, another laptop, a server - describe it in ops/hosts.local.toml, bootstrap the training environment and verify it. Use when the user offers a new machine for training, or when a host shows as unreachable or broken in `ops hosts check`.
---

# Add a GPU host

Reference: `docs/dev/COMPUTE.md`.

## What the user must do (never do these for them)

- Put the machine on Tailscale (a friend shares the node with them; no ports opened).
- Give this laptop key-based SSH access: add its public key (`~/.ssh/id_ed25519.pub`) to the host's
  `authorized_keys` (Windows admin accounts: `C:\ProgramData\ssh\administrators_authorized_keys`). Never type or
  ask for passwords in the session.
- Windows hosts: install WSL Ubuntu and the NVIDIA driver.

## What you do

1. Test: `ssh -o BatchMode=yes <user>@<host> echo ok` (Windows' shell answers in PowerShell).
2. Add a `[hosts.<name>]` entry to `ops/hosts.local.toml` (gitignored - addresses never go in tracked files):
   `ssh`, `os` ("linux" or "windows-wsl"), `repo = "~/pinch"`, `python = ".venv-rl/bin/python"`, `vram_gb`,
   `max_jobs`.
3. `python -m ops hosts bootstrap <name>` - clones the repo, installs uv, builds `.venv-rl` with CUDA JAX.
4. `python -m ops hosts check` - expect the GPU line, a commit, `jax ... gpu`.
5. Smoke it: `python -m ops submit configs/experiments/smoke.toml --host <name>` then `ops jobs -r`.

## Gotchas

- `jax.default_backend()` printing `cpu` means CUDA JAX is missing: re-run bootstrap; on WSL check `nvidia-smi`
  works inside the distro.
- 6-8 GB GPUs: experiments set `small_gpu_num_envs`; set `vram_gb` honestly so it is applied.
- SkyPilot SSH Node Pools are for Linux servers only (they install k3s + the NVIDIA GPU Operator, which does
  not list WSL2); Windows machines use the plain ssh backend.
