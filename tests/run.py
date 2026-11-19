"""Pinch's test suite: every tests/test_*.py function named test_* is a test (pytest-compatible, but this
runner needs nothing installed).

    python tests/run.py                 # all
    python tests/run.py replay          # files whose name contains "replay"
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    pick = sys.argv[1:]
    results = []
    for f in sorted(HERE.glob("test_*.py")):
        if pick and not any(p in f.stem for p in pick):
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                getattr(mod, name)()
                ok, why = True, ""
            except AssertionError as e:
                ok, why = False, str(e) or traceback.format_exc(limit=1)
            except Exception:                                    # noqa: BLE001 - a crash is a failure, not a stop
                ok, why = False, traceback.format_exc(limit=3)
            results.append((f.stem, name, ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] {f.stem}.{name}  ({time.time() - t0:.1f} s)" + (f"\n    {why}" if why else ""))
    n = sum(ok for *_, ok in results)
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
