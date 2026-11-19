"""Give the crab a brain: see -> think -> act, in the simulator or on the robot.

    python brain/run_brain.py --sim --mock --script "hello crab" "come here" "go left" "goodnight"
    python brain/run_brain.py --sim                       # Gemma 4 via Ollama, type to talk
    python brain/run_brain.py --sim --model gemma4:12b    # bigger model, slower, smarter
    python brain/run_brain.py                             # real robot (camera + servos on the Pi;
                                                          #  point --host at the PC running Ollama)
Saves what the crab saw at each step to docs/images/brain_eye_<n>.png.
"""

import argparse
import io
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.brain import MockBrain, OllamaBrain  # noqa: E402
from brain.sounds import play  # noqa: E402
from robot.crab_robot import Crab, SafetyStop  # noqa: E402

IMG_DIR = ROOT / "docs" / "images"


def sim_eye(hw, renderer):
    renderer.update_scene(hw.d, camera="eye_cam")
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(renderer.render()).save(buf, format="PNG")
    return buf.getvalue()


def pi_eye(device=0):
    """One PNG from the face camera. On the Radxa ZERO 3W the OV5647 is a plain V4L2 camera
    (vendor kernel, camera overlay enabled in rsetup), so OpenCV reads it; picamera2 is
    Raspberry-Pi-only."""
    import cv2                            # pip install opencv-python-headless
    cam = cv2.VideoCapture(device)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):                    # let auto-exposure settle
        ok, frame = cam.read()
    cam.release()
    if not ok:
        raise RuntimeError("no picture from the camera - is its overlay enabled and the flex seated?")
    return cv2.imencode(".png", frame)[1].tobytes()


def act(crab, action):
    if action["sound"] != "none":                         # pet tones, never words
        where = play(action["sound"])
        print(f"  crab sounds: {action['sound']}" + (f" (saved {where.name})" if where else ""))
    if action["emote"] not in ("none", ""):
        crab.play_emote(action["emote"])
    w = action["walk"]
    if w["seconds"] > 0 and any(abs(w[k]) > 1e-3 for k in ("vx", "vy", "wz")):
        crab.walk((w["vx"], w["vy"], w["wz"]), w["seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--mock", action="store_true", help="keyword rules instead of a model")
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--script", nargs="*", help="things to say, in order (otherwise type them)")
    ap.add_argument("--view", action="store_true", help="show the simulator viewer")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    brain = MockBrain() if a.mock else OllamaBrain(a.model, a.host)
    if a.sim:
        import mujoco
        from robot.backends import SimBackend
        hw = SimBackend(render=a.view)
        renderer = mujoco.Renderer(hw.m, 480, 640)
        eye = lambda: sim_eye(hw, renderer)  # noqa: E731
    else:
        from robot.backends import RealBackend
        hw = RealBackend()
        eye = pi_eye

    crab = Crab(hw)
    crab.start()
    lines = iter(a.script) if a.script is not None else None
    log = []
    try:
        for n in range(1000):
            if lines is not None:
                heard = next(lines, None)
                if heard is None:
                    break
            else:
                heard = input("\nyou> ").strip()
                if heard in ("quit", "exit"):
                    break
            img = eye()
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            (IMG_DIR / f"brain_eye_{n}.png").write_bytes(img)
            action = brain.decide(img, heard)
            print(f"you: {heard!r}\n  brain thinks: {action['thought']}\n  -> emote={action['emote']} walk={action['walk']}")
            act(crab, action)
            log.append((heard, action, tuple(hw.d.qpos[:2]) if a.sim else None))
        crab.shutdown()
    except SafetyStop as e:
        logging.error("safety stop: %s", e)
    return log


if __name__ == "__main__":
    main()
