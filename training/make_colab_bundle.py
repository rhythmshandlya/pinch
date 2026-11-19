"""Pack what Colab needs into dist/crab_bundle.zip (upload it in the notebook).

    python training/make_colab_bundle.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["crab/*.py", "sim/*.py", "sim/models/*.xml", "sim/models/**/*.stl", "requirements-train.txt",
           "training/*.py", "requirements.txt"]


def main():
    out = ROOT / "dist" / "crab_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    files = sorted({p for pat in INCLUDE for p in ROOT.glob(pat) if p.is_file()})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"wrote {out.relative_to(ROOT)}  ({len(files)} files, {out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
