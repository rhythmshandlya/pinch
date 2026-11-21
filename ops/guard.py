"""Keep private things out of the public repository. Run by pre-commit (staged files, commit message) and by
`ops check` / CI (the whole tracked tree).

    python -m ops.guard files <paths...>     # pre-commit: these staged files
    python -m ops.guard msg <file>           # commit-msg hook
    python -m ops.guard tree                 # every tracked file
"""

import fnmatch
import re
import subprocess
import sys

from ops import ROOT

# local-only documents and machine-specific config (mirrors .gitignore; the guard catches `git add -f`)
PRIVATE = ["docs/PRD.md", "docs/PLAN.md", "docs/CHARACTER.md", "docs/MOVES.md", "docs/HW_REQUIREMENTS.md",
           "docs/phase1/*", "docs/phase2/*", "docs/phase3/*", "docs/TRACEABILITY.md", "docs/MORNING_REPORT.md", "docs/TODO.md", "docs/LIFE_PLAN.md",
           "docs/PET_MIND.md", "hardware/*_prices_raw*", "ops/hosts.local.toml", ".ops/*", "CLAUDE.local.md",
           "*.env", ".env*"]
# Tailscale CGNAT addresses (100.64.0.0/10) other than the documentation placeholder
TAILNET_IP = re.compile(r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")
PLACEHOLDER_IPS = {"100.64.0.0", "100.64.0.1"}   # the range itself and the docs placeholder
PRD_WORD = re.compile(r"PRD")          # the private requirements document is never mentioned in public files
ATTRIBUTION = re.compile(r"co-authored-by:.*(claude|anthropic)|generated with \[?claude", re.I)
TEXT_SUFFIXES = (".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".sh", ".service", ".cfg", ".ini")


def _private(path):
    p = path.replace("\\", "/")
    return any(fnmatch.fnmatch(p, pat) for pat in PRIVATE)


def check_files(paths):
    problems = []
    for p in paths:
        rel = str(p).replace("\\", "/")
        if _private(rel):
            problems.append(f"{rel}: local-only file (see ops/guard.py PRIVATE); unstage it")
            continue
        f = ROOT / rel
        if f.is_file() and rel.endswith(TEXT_SUFFIXES) and f.stat().st_size < 2_000_000:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if rel.endswith((".md", ".txt")) and not rel.startswith(".claude/") and PRD_WORD.search(text):
                problems.append(f"{rel}: mentions the private requirements document (PRD); reword it")
            ips = {m.group(0) for m in TAILNET_IP.finditer(text)} - PLACEHOLDER_IPS
            if ips:
                problems.append(f"{rel}: private Tailscale address {sorted(ips)}; put hosts in ops/hosts.local.toml")
    return problems


def check_message(text):
    return ["commit message: no AI attribution lines in this repository"] if ATTRIBUTION.search(text) else []


def check_tree():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    problems = check_files(files)
    for p in problems:
        print("guard:", p)
    return 1 if problems else 0


def main(argv):
    if not argv or argv[0] == "tree":
        return check_tree()
    if argv[0] == "msg":
        problems = check_message(open(argv[1], encoding="utf-8").read())
    else:
        problems = check_files(argv[1:])
    for p in problems:
        print("guard:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
