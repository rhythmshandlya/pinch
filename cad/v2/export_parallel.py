"""The CAD export with its heavy parts built side by side, one process each (cad/v2/export.py does the rest).

    python cad/v2/export_parallel.py [--workers 8]      # same output as cad/v2/export.py

The export is CPU-bound OpenCascade work (a GPU does not help) and ran in one process: the tray, the cover, the
battery door, every leg and arm link, the electronics' real shapes and the cables were built one after another.
Here the body's features (seats, mounts, walls, liners, tie anchors, joint openings) are built first in groups
side by side, then the tray and the cover from them, while every other part is built at once, saved as an exact BRep (OpenCascade's own format: the volume and
every face survive the round trip), loaded back, and handed to export.main() in place of being rebuilt, which then
writes the meshes, the masses and the JSON as before. A part that comes out empty is named with its job.
"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# job -> (module, function); each returns a solid, or a list of (label, solid, mass, material)
JOBS = {
    "tray": ("cad.v2.body", "tray"), "cover": ("cad.v2.body", "cover"),
    "battery_door": ("cad.v2.body", "battery_door"),
    "coxa_link": ("cad.v2.legs", "coxa_link"), "femur_link": ("cad.v2.legs", "femur_link"),
    "tibia": ("cad.v2.legs", "tibia"), "foot": ("cad.v2.legs", "foot"),
    "shoulder_link": ("cad.v2.arms", "shoulder_link"), "upper_arm": ("cad.v2.arms", "upper_arm"),
    "dactyl": ("cad.v2.arms", "dactyl"),
    "electronics": ("cad.v2.assembly", "electronics"), "wiring": ("cad.v2.wiring", "solids"),
}


# the body's features (cad/v2/body.py features()), in its order, as groups built side by side: each is trimmed
# round the joint zones and the other parts' room in its own process; the joint openings are cuts (not trimmed)
FEATURE_GROUPS = {
    "f_hips": lambda B: sum((B.hip_seat(x) for x in B.LEGS), []),
    "f_shoulders": lambda B: sum((B.shoulder_seat(x) for x in B.ARMS), []),
    "f_standoffs": lambda B: sum((B.standoffs(x)[0] for x in B.STANDOFFS), []),
    "f_wallboards": lambda B: sum((B.wall_board(x) for x in B.WALL_BOARDS), []),
    "f_mounts": lambda B: sum((B.cradle(x) for x in B.CRADLES), []) + B.battery_cradle() + B.camera_mount()
    + B.speaker_mount() + B.grille()[0] + B.mic_mount("mic_l") + B.mic_mount("mic_r"),
    "f_pads": lambda B: sum((B.pad_recess(x) for x in B.PADS), []) + B.switch_cut() + B.pogo_mount("pogo_l")
    + B.pogo_mount("pogo_r"),
    "f_bosses": lambda B: B.split_bosses() + B.side_vents() + B.lid_knuckle(),
    "f_liners": lambda B: B.bay_liners(),
    "f_ties": lambda B: B.cable_ties(),
    "f_openings": None,                                   # joint_openings(): cuts, not trimmed
}


def _features(group, outdir):
    """In a worker: one group of the body's features, trimmed; saved as BRep + a manifest."""
    from build123d import export_brep
    sys.path.insert(0, str(ROOT))
    from cad.v2 import body as B
    t0 = time.time()
    fs = B.joint_openings() if FEATURE_GROUPS[group] is None else [B._trim(x) for x in FEATURE_GROUPS[group](B)]
    out = []
    for k, f in enumerate(fs):
        if _empty(f.solid):
            continue                                       # trimmed away entirely (as in the serial path: no solid)
        fn = Path(outdir) / f"{group}_{k}.brep"
        export_brep(f.solid, str(fn))
        out.append((f.name, f.owner, f.half, f.op, f.clip, f.trimmed, fn.name))
    _peak(group)
    return group, time.time() - t0, out


def _peak(job):
    """Print this worker's peak memory (Windows), to size the pool to the machine."""
    try:
        import ctypes
        import ctypes.wintypes as w

        class PMC(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        print(f"    {job}: peak memory {c.PeakWorkingSetSize / 1e9:.1f} GB", flush=True)
    except Exception:                                      # noqa: BLE001 (not Windows)
        pass


def _install_features(outdir, manifests):
    from cad.v2 import body as B
    fs = []
    for group in FEATURE_GROUPS:
        for name, owner, half, op, clip, trimmed, fn in manifests[group]:
            fs.append(B.Feature(name, owner, half, op, _load(Path(outdir) / fn), clip, trimmed))
    fs = tuple(fs)
    import functools
    B.features = functools.cache(lambda: fs)


def _empty(s):
    return s is None or getattr(s, "_wrapped", None) is None


def _build(job, outdir, feats=None):
    """In a worker: build one job, save its solids as BRep files, return (job, seconds, manifest). `feats`: the
    feature groups already built (stage 1), installed so the tray and cover do not rebuild them."""
    import importlib

    from build123d import export_brep
    sys.path.insert(0, str(ROOT))
    t0 = time.time()
    if feats is not None:
        _install_features(outdir, feats)
    mod, fn = JOBS[job]
    r = getattr(importlib.import_module(mod), fn)()
    out = Path(outdir)
    if isinstance(r, (list, tuple)):                       # electronics: (label, solid, mass, material)
        manifest = []                                      # wiring: (name, kind, solid, mass)
        for k, row in enumerate(r):
            solid = row[1] if job == "electronics" else row[2]
            if _empty(solid):
                raise ValueError(f"{job}: {row[0]} is an empty solid")
            f = out / f"{job}_{k}.brep"
            export_brep(solid, str(f))
            meta = list(row)
            meta[1 if job == "electronics" else 2] = f.name
            manifest.append(meta)
        return job, time.time() - t0, manifest
    if _empty(r):
        raise ValueError(f"{job}: the CAD produced an empty solid (a boolean failed upstream)")
    f = out / f"{job}.brep"
    export_brep(r, str(f))
    _peak(job)
    return job, time.time() - t0, f.name


def _load(path):
    from build123d import import_brep
    s = import_brep(str(path))
    sol = s.solids() if hasattr(s, "solids") else []
    return sol[0] if len(sol) == 1 else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(len(JOBS), max(2, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--keep", default=None, help="keep the built parts in this folder (stage 1 manifest.json)")
    ap.add_argument("--stage1-only", action="store_true", help="build the features and parts, then stop")
    ap.add_argument("--reuse", action="store_true",
                    help="skip building: take the parts a finished run left in --keep (retry the export step)")
    a = ap.parse_args()
    outdir = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="pinch_cad_"))
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results, feats = {}, {}
    shell = ("tray", "cover")
    import json
    if a.reuse:
        if not a.keep or not (outdir / "parts.json").exists():
            raise SystemExit("--reuse needs --keep <folder> from a run that built every part (parts.json)")
        m = json.loads((outdir / "parts.json").read_text())
        feats, results = m["feats"], m["parts"]
    else:
        feats, results = _build_all(a, outdir, shell)
    _install_features(outdir, feats)
    _export(outdir, results, t0)


def _build_all(a, outdir, shell):
    """Build every feature group and part side by side; returns (feats, results) and saves them in parts.json."""
    import json
    t0 = time.time()
    results, feats = {}, {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        # stage 1: the body's feature groups and every part that does not need them, side by side
        futs = [ex.submit(_features, g, str(outdir)) for g in FEATURE_GROUPS]
        futs += [ex.submit(_build, j, str(outdir)) for j in JOBS if j not in shell]
        for f in as_completed(futs):
            try:
                job, secs, manifest = f.result()           # an empty part stops here, named
            except Exception as e:
                raise RuntimeError(f"stage 1 job {futs.index(f)} failed: {e}") from e
            (feats if job in FEATURE_GROUPS else results)[job] = manifest
            print(f"  built {job:15s} {secs:6.0f} s", flush=True)
        print(f"stage 1 in {time.time() - t0:.0f} s", flush=True)
        (outdir / "manifest.json").write_text(json.dumps({"feats": feats, "parts": results}))
        if a.stage1_only:
            raise SystemExit(0)
        # stage 2: the tray and the cover from those features, side by side
        for f in as_completed([ex.submit(_build, j, str(outdir), feats) for j in shell]):
            job, secs, manifest = f.result()
            results[job] = manifest
            print(f"  built {job:15s} {secs:6.0f} s", flush=True)
    print(f"parts built in parallel in {time.time() - t0:.0f} s ({a.workers} processes)", flush=True)
    (outdir / "parts.json").write_text(json.dumps({"feats": feats, "parts": results}))
    return feats, results


def _export(outdir, results, t0):
    """Hand the built parts to the export (meshes, masses, JSON) in place of rebuilding them."""
    # hand the built parts to the export in place of rebuilding them
    import functools

    from cad.v2 import arms, assembly, body, export, legs, wiring
    for job, (mod, fn) in JOBS.items():
        target = {"cad.v2.body": body, "cad.v2.legs": legs, "cad.v2.arms": arms, "cad.v2.assembly": assembly,
                  "cad.v2.wiring": wiring}[mod]
        m = results[job]
        if job == "electronics":
            rows = [(lab, _load(outdir / f), mass, mat) for lab, f, mass, mat in m]
            setattr(target, fn, functools.cache(lambda rows=rows: rows))
        elif job == "wiring":
            rows = [(name, kind, _load(outdir / f), mass) for name, kind, f, mass in m]
            setattr(target, fn, functools.cache(lambda rows=rows: rows))
        else:
            s = _load(outdir / m)
            setattr(target, fn, functools.cache(lambda s=s: s))
    t1 = time.time()
    export.main()
    print(f"meshes, masses and JSON in {time.time() - t1:.0f} s; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
