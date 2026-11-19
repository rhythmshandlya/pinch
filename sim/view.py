"""Interactive MuJoCo viewer: watch the crab perform its emotes.

    python sim/view.py                 # cycle through all emotes
    python sim/view.py --emote wave    # loop one emote

Keys:  Up / Down arrow = next / previous emote.
Mouse: left-drag rotate, right-drag pan, scroll zoom, double-click a part to select,
       Ctrl+right-drag on a part to push the crab.
(The standard MuJoCo viewer panel on the left shows joints, actuators, sensors.)
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.params import PARAMS  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
KEY_UP, KEY_DOWN = 265, 264
HOLD_AFTER = 1.0   # seconds of STAND between emotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emote", choices=list(EMOTES), help="loop a single emote")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)

    names = [args.emote] if args.emote else list(EMOTES)
    state = {"idx": 0, "t0": 0.0}

    def switch(step):
        state["idx"] = (state["idx"] + step) % len(names)
        state["t0"] = d.time
        print(f"> {names[state['idx']]}")

    def on_key(key):
        if key == KEY_UP:
            switch(+1)
        elif key == KEY_DOWN:
            switch(-1)

    print(__doc__)
    print(f"> {names[0]}")
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = m.body("body").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.75 * PARAMS.scale, 135, -22   # sized to the crab

        while v.is_running():
            frame_start = time.time()
            emo = EMOTES[names[state["idx"]]]
            t = d.time - state["t0"]
            d.ctrl[:] = emo.target(t) if (emo.loop or t <= emo.duration) else STAND
            if not args.emote and not emo.loop and t > emo.duration + HOLD_AFTER:
                switch(+1)
            elif not args.emote and emo.loop and t > 2 * emo.duration:
                switch(+1)
            elif args.emote and not emo.loop and t > emo.duration + HOLD_AFTER:
                state["t0"] = d.time

            # advance physics by one 60 Hz frame, then sleep to stay real-time
            for _ in range(int((1 / 60) / m.opt.timestep)):
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - frame_start)))


if __name__ == "__main__":
    main()
