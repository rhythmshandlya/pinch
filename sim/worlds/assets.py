"""Everyday things for the worlds: Google Scanned Objects in MuJoCo, fetched on demand (not in git).

    python -m sim.worlds.assets fetch       # the things in sim/worlds/catalogue.py THINGS -> models/objects/ (~70 MB)
    python -m sim.worlds.assets measure     # print each thing's measured size (width, depth, height) for the catalogue

Source: kevinzakka/mujoco_scanned_objects (MIT), pinned to one commit, a MuJoCo conversion of Google Scanned
Objects (Downs et al. 2022, arXiv 2204.11918; the scans are CC-BY 4.0 by Google LLC). Each model has a textured
visual mesh and a convex decomposition for contacts. Worlds use them when present and fall back to a box or a
cylinder of the same size when not (CI, a fresh checkout).
"""

import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.worlds.catalogue import THINGS  # noqa: E402

REPO = "kevinzakka/mujoco_scanned_objects"
COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
DIR = ROOT / "models" / "objects"
N_HULLS = 32                                     # each model ships model_collision_0..31.obj


def folder(gso):
    return DIR / gso


def available(gso):
    f = folder(gso)
    return (f / "model.xml").exists() and (f / "model.obj").exists() and (f / "texture.png").exists()


def _get(gso, fname):
    out = folder(gso) / fname
    if out.exists() and out.stat().st_size > 0:
        return True
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/models/{gso}/{fname}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception:                                    # noqa: BLE001 - a hull that doesn't exist (fewer than 32)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return True


def fetch(names=None):
    names = names or list(THINGS)
    jobs = []
    for n in names:
        gso = THINGS[n][0]
        jobs += [(gso, f) for f in ("model.xml", "model.obj", "texture.png")]
        jobs += [(gso, f"model_collision_{i}.obj") for i in range(N_HULLS)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: _get(*j), jobs))
    have = [n for n in names if available(THINGS[n][0])]
    print(f"{len(have)}/{len(names)} things in {DIR}")
    return have


def hulls(gso):
    return sorted(folder(gso).glob("model_collision_*.obj"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def _verts(path):
    v = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(v, float)


def bounds(gso):
    """(min xyz, max xyz) of the thing's collision hulls, in its own frame."""
    v = np.concatenate([_verts(p) for p in hulls(gso)])
    return v.min(0), v.max(0)


def measure():
    for n, (gso, size, *_) in THINGS.items():
        if not available(gso):
            print(f"{n:14s} not fetched")
            continue
        lo, hi = bounds(gso)
        s = hi - lo
        print(f"{n:14s} catalogue {tuple(round(x, 3) for x in size)}  measured {tuple(round(float(x), 3) for x in s)}"
              f"  base z {lo[2]:+.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    fetch() if cmd == "fetch" else measure()
