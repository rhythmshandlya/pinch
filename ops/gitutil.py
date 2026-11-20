"""Git facts a job needs: which commit, whether the tree is clean, whether the commit is on the remote."""

import subprocess

from ops import ROOT


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


def head():
    return git("rev-parse", "HEAD")


def branch():
    return git("rev-parse", "--abbrev-ref", "HEAD")


def dirty(paths=()):
    """Uncommitted changes to tracked files (optionally only under paths)."""
    return bool(git("status", "--porcelain", "--untracked-files=no", "--", *paths))


def remote_url():
    return git("remote", "get-url", "origin")


def pushed(sha):
    """Is sha reachable from some remote-tracking branch (so a worker can fetch it)?"""
    out = git("branch", "-r", "--contains", sha, check=False)
    return bool(out.strip())
