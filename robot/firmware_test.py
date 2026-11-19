"""Firmware self-test: the robot's own program (robot/pinch_main.py) run against emulated hardware.

    python robot/firmware_test.py

Checks the deployable firmware end to end, with the I2C drivers talking to fake chips and the desk
twin as the body: the face screen gets frames, the camera localises Pinch from the tray marker,
speech through the HTTP API makes it perform, a stroke on the touch pads makes it purr, a meeting
heard through the stereo mics makes it go quiet, and it never falls.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.drivers.i2c_devices import pages_to_image  # noqa: E402
from robot.hw_emulated import EmulatedHardware  # noqa: E402
from robot.pinch_main import PinchFirmware, serve  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def main():
    print("Pinch firmware self-test (emulated hardware)")
    hw = EmulatedHardware(start={"pen": (-0.18, -0.02, 0.6), "eraser": (0.10, -0.12, 0.3), "cap": (-0.12, 0.13, 0.0),
                                 "ball": (0.16, 0.0, 0.0)})                 # the ball right in front of Pinch
    fw = PinchFirmware(hw, llm=False, memory_path=False)
    port = 18080
    srv = serve(fw, port)
    fw.crab.start()

    def run(seconds):
        for _ in range(int(seconds / 0.02)):
            fw.step()

    run(6.0)
    img = pages_to_image(bytes(hw.oled_chip.frame))
    check("face screen: the OLED driver sends frames, eyes are drawn", hw.oled_chip.on and fw.stats["oled_frames"] > 50
          and 200 < img.sum() < 4000, f"{fw.stats['oled_frames']} frames, {int(img.sum())} lit pixels")
    x, y, _ = fw.loc.pose()
    true = hw.d.qpos[:2]
    check("camera + tray marker: knows where it is on the desk", fw.stats["vision_fixes"] > 0
          and np.hypot(x - true[0], y - true[1]) < 0.03,
          f"{fw.stats['vision_fixes']} fixes, error {np.hypot(x - true[0], y - true[1]) * 1000:.0f} mm")
    check("camera: sees its desk things", fw.stats["things_seen"] > 0, f"{fw.stats['things_seen']} sightings")

    status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5).read())
    check("HTTP API: status", "feeling" in status and "needs" in status, status["feeling"])
    post(port, "/say", {"text": "hi pinch, can you dance?"})
    run(10.0)
    emotes = [m[1] for m in fw.stats["motions"] if m[1] == "emote" or m[2] == "dance"]
    check("speech through the API -> it dances", any(m[2] == "dance" for m in fw.stats["motions"]),
          f"motions {[m[2] for m in fw.stats['motions']][-5:]}")

    n = len(fw.stats["sounds"])
    hw.touch_chip.touching = {"head"}
    run(0.15)
    hw.touch_chip.touching = {"head", "back_left"}
    run(0.15)
    hw.touch_chip.touching = {"back_left"}
    run(0.3)
    hw.touch_chip.touching = set()
    run(4.0)
    check("touch pads (MPR121 driver): a stroke -> purr", "purr" in fw.stats["sounds"][n:],
          f"sounds {fw.stats['sounds'][n:]}")

    hw.audio_mode = "meeting"
    quiet = False
    for _ in range(150):
        run(1.0)
        if fw.mind.affect.quiet:
            quiet = True
            break
    n = len(fw.stats["sounds"])
    run(15.0)
    check("stereo mics: a meeting is heard -> goes quiet, no tones", quiet and len(fw.stats["sounds"]) == n,
          f"quiet after {fw.t:.0f} s, tones during the call: {fw.stats['sounds'][n:]}")
    hw.audio_mode = "quiet"
    check("never fell, servo safety never tripped", fw.bodysense is not None and hw.d.qpos[2] > 0.02
          and fw.stats["errors"] == 0, f"body height {hw.d.qpos[2] * 1000:.0f} mm, I2C errors {fw.stats['errors']}")
    srv.shutdown()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
