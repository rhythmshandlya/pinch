"""The layers stay layered (decision 68): crab/ is the robot's geometry and controllers, robot/ its firmware and
drivers, pinch/ the pet, training/ the learning. crab/ depends on nothing of ours above it; training/ may use
crab/ but not the pet or the firmware (it ships to the GPU without them)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {"crab": {"pinch", "robot", "training", "sim"}, "training": {"pinch", "robot"}}


def _imports(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def test_layers():
    bad = []
    for pkg, forbidden in RULES.items():
        for f in (ROOT / pkg).rglob("*.py"):
            hit = _imports(f) & forbidden
            if hit:
                bad.append(f"{f.relative_to(ROOT)} imports {sorted(hit)}")
    assert not bad, "layering broken: " + "; ".join(bad)
