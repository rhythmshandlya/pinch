"""Find which feature group breaks a shell half's booleans: assemble it with each group left out, side by side.

    python cad/v2/bisect_shell.py [--half tray]

Uses the features cad/v2/export.py saves when a half comes out empty (or export_parallel.py --keep --stage1-only). Prints, per feature family left out (hip, tie, liner, waist, opening, ...), whether the half
came out whole and its volume; the group whose removal makes it whole is the one to look at.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def family(name):
    """A feature's family from its name: hip_fl_seat -> hip, tie12 -> tie, opening_FL -> opening, ..."""
    import re
    return re.split(r"[_0-9]", name, maxsplit=1)[0]


def _rows(keep):
    m = json.loads((Path(keep) / "manifest.json").read_text())
    return m["features"] if "features" in m else [r for g in m["feats"].values() for r in g]


def _try(keep, half, skip):
    import functools

    from cad.v2 import body as B
    from cad.v2.export_parallel import _empty, _load
    fs = []
    for name, owner, h, op, clip, trimmed, fn in _rows(keep):
        if family(name) == skip:
            continue
        fs.append(B.Feature(name, owner, h, op, _load(Path(keep) / fn), clip, trimmed))
    fs = tuple(fs)
    B.features = functools.cache(lambda: fs)
    t0 = time.time()
    try:
        s = getattr(B, half)()
    except Exception:                                      # noqa: BLE001 (a failed boolean raises or returns empty)
        return skip, False, 0, time.time() - t0
    ok = not _empty(s)
    return skip, ok, (round(B.volume(s)) if ok else 0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=str(ROOT / ".ops" / "cad_parts"))
    ap.add_argument("--half", default="tray")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    fams = sorted({family(r[0]) for r in _rows(a.keep)})
    print("families:", ", ".join(fams), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_try, a.keep, a.half, g) for g in [None] + fams]
        for f in as_completed(futs):
            skip, ok, vol, secs = f.result()
            print(f"without {skip or '(nothing)':14s} {'WHOLE' if ok else 'empty'}  {vol:>8} mm3  {secs:5.0f} s", flush=True)


if __name__ == "__main__":
    main()
