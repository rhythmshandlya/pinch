"""Headless sanity check: does the crab stand, survive every emote, and stay within
its servos' limits (each joint against its own servo)? Renders snapshots to docs/images/.

    python sim/check.py            # report + snapshots
    python sim/check.py --no-render

Pass criteria per emote: no fall; drive saturation <= 3% of the time; datasheet
torque-speed line exceeded <= 3%; commanded speed <= the servo's limit.
Standing must keep every servo under 70% of its continuous rated torque (heating).
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.params import JOINT_NAMES, JOINT_SPECS, LEG_IDS  # noqa: E402
from sim.latency import DelayLine  # noqa: E402
from sim.servo_check import MAX_CMD, ServoMonitor  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
IMG_DIR = ROOT / "docs" / "images"
DELAY = max(s.command_delay for s in JOINT_SPECS)
STAND_THERMAL_LIMIT = 70.0       # % of rated torque while standing still


def tilt_deg(d):
    zz = d.xmat[1].reshape(3, 3)[2, 2]
    return np.degrees(np.arccos(np.clip(zz, -1, 1)))


def run(m, d, ctrl_fn, seconds, snap_times=(), renderer=None, cam="track"):
    dt = m.opt.timestep
    n = int(seconds / dt)
    mon = ServoMonitor(m.nu)
    min_h, max_tilt, cmd_ratio = 1e9, 0.0, 0.0
    snaps, snap_steps = [], {int(t / dt) for t in snap_times}
    prev = np.asarray(ctrl_fn(0.0), float)
    delay = DelayLine(DELAY, dt, prev)
    for i in range(n):
        q = np.asarray(ctrl_fn(i * dt), float)
        cmd_ratio = max(cmd_ratio, float((np.abs(q - prev) / dt / MAX_CMD).max()))
        prev = q
        d.ctrl[:] = delay(q)
        mujoco.mj_step(m, d)
        mon.update(d.actuator_force, d.actuator_velocity)
        min_h = min(min_h, d.xpos[1, 2])
        max_tilt = max(max_tilt, tilt_deg(d))
        if renderer is not None and i in snap_steps:
            renderer.update_scene(d, camera=cam)
            snaps.append(renderer.render().copy())
    return dict(mon=mon, min_h=min_h, max_tilt=max_tilt, cmd_ratio=cmd_ratio, snaps=snaps)


def grid(images, cols):
    h, w, _ = images[0].shape
    rows = -(-len(images) // cols)
    out = np.full((rows * h, cols * w, 3), 255, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    renderer = None if args.no_render else mujoco.Renderer(m, 360, 640)
    h_fall = 0.25 * P_HEIGHT(m, d)

    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    run(m, d, lambda t: STAND, 2.0)
    stand = run(m, d, lambda t: STAND, 3.0)
    th = stand["mon"].thermal_pct
    print(f"total mass {sum(m.body_mass)*1000:.0f} g | stand height {d.xpos[1, 2]*100:.1f} cm | tilt {tilt_deg(d):.1f} deg")
    print(f"\nStanding: motor torque as % of each servo's continuous rated torque (limit {STAND_THERMAL_LIMIT:.0f}%):")
    for leg in LEG_IDS:
        cells = []
        for j in ("coxa", "femur", "tibia"):
            i = JOINT_NAMES.index(f"{leg}_{j}")
            cells.append(f"{j} {stand['mon'].rms[i]*100:5.1f} N*cm = {th[i]:4.0f}% of {JOINT_SPECS[i].name}")
        print(f"  {leg}: " + " | ".join(cells))
    ok_all = bool(th.max() <= STAND_THERMAL_LIMIT)
    if not ok_all:
        print(f"  !! standing overheats {JOINT_NAMES[int(np.argmax(th))]} ({th.max():.0f}% of rated)")

    print("\n(peak = peak motor torque as % of that servo's datasheet stall; vsat% = time at drive limit;"
          " ds% = time beyond datasheet torque-speed line; heat = RMS torque % of rated; cmd = commanded speed % of limit)")
    print(f"{'emote':8s} {'dur':>5s} {'min h':>7s} {'tilt':>8s} {'peak':>5s} {'worst joint':>16s} "
          f"{'vsat%':>5s} {'ds%':>5s} {'heat':>5s} {'cmd':>5s}  result")
    snaps_all, labels = [], []
    if renderer:
        renderer.update_scene(d, camera="track")
        snaps_all.append(renderer.render().copy())
        labels.append("stand")
    for name, emo in EMOTES.items():
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        run(m, d, lambda t: STAND, 1.0)
        dur = emo.duration if not emo.loop else 2 * emo.duration
        r = run(m, d, emo.target, dur + 0.8, snap_times=[dur * 0.45], renderer=renderer)
        mon = r["mon"]
        worst = mon.worst_joint()
        peak_pct = 100 * mon.peak_motor[worst] / JOINT_SPECS[worst].datasheet_stall
        fell = r["max_tilt"] > 45 or r["min_h"] < h_fall
        too_hard = mon.volt_sat_pct > 3 or mon.ds_over_pct > 3 or r["cmd_ratio"] > 1.0
        ok_all &= not (fell or too_hard)
        status = "FELL" if fell else ("TOO FAST/STRONG" if too_hard else "ok")
        print(f"{name:8s} {dur:5.1f} {r['min_h']*100:6.1f}cm {r['max_tilt']:5.1f}deg {peak_pct:4.0f}% "
              f"{JOINT_NAMES[worst]:>16s} {mon.volt_sat_pct:5.1f} {mon.ds_over_pct:5.1f} "
              f"{mon.thermal_pct.max():4.0f}% {100*r['cmd_ratio']:4.0f}%  {status}")
        snaps_all += r["snaps"]
        labels += [name] * len(r["snaps"])

    if renderer:
        from PIL import Image, ImageDraw
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        tiles = []
        for im, lab in zip(snaps_all, labels):
            pil = Image.fromarray(im)
            ImageDraw.Draw(pil).text((12, 10), lab, fill=(20, 20, 20))
            tiles.append(np.asarray(pil))
        Image.fromarray(grid(tiles, 4)).save(IMG_DIR / "emotes.png")
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        run(m, d, lambda t: STAND, 1.0)
        big = mujoco.Renderer(m, 720, 1280)
        big.update_scene(d, camera="track")
        Image.fromarray(big.render()).save(IMG_DIR / "stand.png")
        big.update_scene(d, camera="eye_cam")
        Image.fromarray(big.render()).save(IMG_DIR / "eye_cam.png")
        print(f"\nsnapshots -> {IMG_DIR.relative_to(ROOT)}/ (stand.png, emotes.png, eye_cam.png)")

    sys.exit(0 if ok_all else 1)


def P_HEIGHT(m, d):
    """Standing body height from the stand keyframe (scale-independent fall threshold)."""
    return float(m.key("stand").qpos[2])


if __name__ == "__main__":
    main()
