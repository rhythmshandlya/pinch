"""The layers stay layered (decision 68): crab/ is the robot's geometry and controllers, robot/ its firmware and
drivers, pinch/ the pet, training/ the learning. crab/ depends on nothing of ours above it; training/ may use
crab/ but not the pet or the firmware (it ships to the GPU without them); the pet and the firmware never use
training/ (the robot runs without JAX: learned skills reach it as exported policies/*.npz, decision 77). ops/ is
the controller-side operations layer: it may read training/'s registry, and nothing imports it (decision 90)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {"crab": {"pinch", "robot", "training", "sim", "ops"}, "training": {"pinch", "robot", "ops"},
         "pinch": {"training", "ops"}, "robot": {"training", "ops"}, "sim": {"ops"},
         "ops": {"pinch", "robot", "sim"}}


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
