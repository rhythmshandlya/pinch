"""Vendor the official MuJoCo agent skills (google-deepmind/mujoco, Apache-2.0) into .claude/skills/.

They ship inside the mujoco pip package (mujoco/doc/skills/), so the copy always matches the mujoco version this
project pins. Relative links into MuJoCo's manual (`../../XMLreference.rst`) are rewritten to the published
manual, so the copies work anywhere. Re-run after upgrading mujoco.
"""

import re
import shutil
from pathlib import Path

from ops import ROOT

DEST = ROOT / ".claude" / "skills"
WANTED = ("accelerated", "parity", "python", "spec_editing", "rendering")   # what this project uses
DOCS = "https://mujoco.readthedocs.io/en/stable/"
REPO = "https://github.com/google-deepmind/mujoco/tree/main/"
LINK = re.compile(r"\]\((\.\./)+([^)#\s]+?)(\.rst)?(#[^)]*)?\)")


def _rewrite(text):
    def sub(m):
        path, rst, anchor = m.group(2), m.group(3), m.group(4) or ""
        if rst:
            return f"]({DOCS}{path}.html{anchor.lower()})"
        return f"]({REPO}{path}{anchor})"
    return LINK.sub(sub, text)


def sync():
    import mujoco
    src = Path(mujoco.__file__).parent / "doc" / "skills"
    if not src.exists():
        raise SystemExit(f"mujoco {mujoco.__version__} ships no skills at {src}")
    out = []
    for name in WANTED:
        s = src / name
        if not (s / "SKILL.md").exists():
            out.append(f"skip {name}: not in mujoco {mujoco.__version__}")
            continue
        d = DEST / f"mujoco-{name}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__"))
        for md in d.rglob("*.md"):
            md.write_text(_rewrite(md.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        (d / "SOURCE.md").write_text(
            f"Vendored from the mujoco {mujoco.__version__} package (google-deepmind/mujoco, doc/skills/{name}),\n"
            f"Apache License 2.0. Regenerate with `python -m ops skills sync`; do not edit by hand.\n",
            encoding="utf-8", newline="\n")
        out.append(f"mujoco-{name} <- mujoco {mujoco.__version__}")
    return out
