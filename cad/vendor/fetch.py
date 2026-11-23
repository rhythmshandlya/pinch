"""Download the makers' 3D models of the bought electronics (used by cad/v2/parts.py; not committed).

    python cad/vendor/fetch.py

Each archive goes to its own folder and only its STEP file is unpacked (as data: nothing in it is run).
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    "radxa": "https://dl.radxa.com/zero3/docs/hw/3w/radxa_zero_3w_3d_stp.zip",
    "waveshare": "https://files.waveshare.com/wiki/Bus-Servo-Adapter-(A)/Bus%20Servo%20Adapter%20(A)_3D.zip",
    "dfrobot": "https://dfimg.dfrobot.com/wiki/19409/SEN0526_i2s-mems-microphone_stpfile_v1.0.zip",
}
FILES = {   # single STEP files (Adafruit_CAD_Parts is MIT-licensed; Pololu's carrier is kept as an alternative part)
    "adafruit/3006_MAX98357.step":
        "https://raw.githubusercontent.com/adafruit/Adafruit_CAD_Parts/main/3006%20MAX98357/3006%20MAX98357.step",
    "pololu/vl53l0x_carrier_2490.step": "https://www.pololu.com/file/0J1195/vl53l0x-vl53l1x-vl53l3cx-vl53l4cd-carrier-model.step",
}


def main():
    for name, url in MODELS.items():
        out = HERE / name / "x" / f"{name}.step"
        if out.exists():
            continue
        data = urllib.request.urlopen(url, timeout=120).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        step = [i for i in z.infolist() if i.filename.lower().endswith((".stp", ".step")) and ".." not in i.filename]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(step[0]))
        print(name, "->", out, file=sys.stderr)
    for rel, url in FILES.items():
        out = HERE / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(urllib.request.urlopen(url, timeout=120).read())
            print(rel, file=sys.stderr)


if __name__ == "__main__":
    main()
