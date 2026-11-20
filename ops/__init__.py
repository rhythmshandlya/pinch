"""Operations layer: run training jobs on any GPU (this laptop, Tailscale machines over SSH, Modal, SkyPilot
clouds), keep a ledger of them, check the repo, and manage worktrees for parallel work.

    python -m ops --help

See docs/ARCHITECTURE.md (sections 4-5) and docs/dev/. Nothing in pinch/, robot/, crab/ or training/ imports
this package; it only reads configs/ and training/registry.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".ops"                       # local, gitignored: the job ledger and logs
