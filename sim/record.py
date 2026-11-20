"""Record a session of Pinch in the twin, judge it, and film it (pinch/recorder.py, pinch/observer.py).

    python sim/record.py --minutes 1 --seed 5 --video                     # a minute of pet life -> media/sessions/...
    python sim/record.py --minutes 1.5 --seed 3 --scenario curiosity --video --out media/review/curiosity
    python sim/record.py --render media/sessions/life-seed5 [--speed 2] [--gif]   # film an existing recording
    python sim/record.py --report media/sessions/life-seed5                # observer + realism metrics again

Scenarios (what you do during the session; all on the twin's seeded clock, so a recording replays exactly):
  life       a normal stretch of pet life: you say hello, later stroke it
  alone      nobody around: what it does by itself
  tidy       the pen is left out on the desk and you ask Pinch to tidy up
  curiosity  an unknown box is put down in front of it, later a toy rolls past
  golden     the golden trace's script (tests/test_golden.py)

The video is the twin from a camera that follows Pinch, at 400x300, with the timeline over it: time, activity and
phase, feeling, face, the step it is doing, sounds, what you did, a strip of the whole session coloured by
activity, and the observer's findings as red marks.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinch.recorder import load_session  # noqa: E402

W, H = 400, 300
SESSION_WIDE = ("activity_mix", "state_lock", "timeouts")
OUT = ROOT / "media" / "sessions"            # (media/ is not in git)


# -- scenarios ------------------------------------------------------------------------------------------------------
def _setup_life(sim):
    pass


def _setup_tidy(sim):
    sim.place("pen", (-0.05, -0.08), 0.9)
    sim.mind.world.see("pen", (-0.05, -0.08), graspable=True)


def _setup_curiosity(sim):
    sim.mind.memory.data["homes"] = {}                  # (nothing to tidy: the new things are the story)


SCENARIOS = {
    "life": dict(setup=_setup_life, script={4.0: ("say", "hi pinch, I'm Sam"), 30.0: ("touch", "stroke"),
                                            50.0: ("say", "good crab")}),
    "alone": dict(setup=_setup_life, script={}),
    "tidy": dict(setup=_setup_tidy, script={3.0: ("say", "pinch, tidy up the desk please")}),
    "curiosity": dict(setup=_setup_curiosity, props=("box", "toy"),
                      script={3.0: ("show_prop", ("box", (0.15, 0.0))),
                              55.0: ("move_prop", ("toy", [(0.30, 0.16), (0.30, -0.16)], 0.05))}),
    "golden": dict(setup=None, script={3.0: ("say", "hi pinch, I'm Sam"), 12.0: ("say", "dance!"),
                                       30.0: ("say", "pinch, put the cap away"), 70.0: ("touch", "stroke"),
                                       90.0: ("say", "I'm sad today")}),
}


def _golden_setup(sim):
    sim.place("cap", (-0.06, 0.0))
    sim.mind.world.see("cap", (-0.06, 0.0), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)


SCENARIOS["golden"]["setup"] = _golden_setup


def record(seconds, seed, scenario="life", out=None, state=True, live_every=None, quiet=False):
    """Run a scenario with the recorder on. Returns (session dir, report)."""
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    sc = SCENARIOS[scenario]
    out = Path(out) if out else OUT / f"{scenario}-seed{seed}"
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=sc.get("props", ()), record_dir=out,
                   record=dict(header=dict(scenario=scenario, seconds=seconds, script=sc["script"]), state=state,
                               live_every=live_every))
    if sc["setup"]:
        sc["setup"](sim)
    done = set()
    while sim.t < seconds:
        for t0, (kind, arg) in sc["script"].items():
            if t0 not in done and sim.t >= t0:
                done.add(t0)
                if kind == "say":
                    sim.say(arg)
                elif kind == "touch":
                    sim.touch(arg)
                elif kind == "show_prop":
                    sim.show_prop(*arg)
                elif kind == "move_prop":
                    sim.move_prop(*arg)
        sim.run(0.5)
        if not quiet and int(sim.t) % 30 == 0 and sim.t - int(sim.t) < 0.5:
            print(f"  t={sim.t:5.0f}s  {sim.mind.activity.name if sim.mind.activity else ''}", flush=True)
    rep = sim.stop_recording()
    return out, rep


# -- video ------------------------------------------------------------------------------------------------------------
PALETTE = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0),
           (204, 121, 167), (160, 160, 160), (120, 200, 120), (200, 120, 120), (150, 150, 230), (230, 200, 150)]


def _color(name, table):
    if name not in table:
        table[name] = PALETTE[len(table) % len(PALETTE)]
    return table[name]


def render(session, out=None, speed=1.0, gif=False, dist=0.42, elev=-32.0, azim=135.0):
    """Film a recording (needs state: the samples' qpos and model.mjb). Returns the video's path."""
    import mujoco
    from PIL import Image, ImageDraw

    from pinch.face import Face
    from sim.desk import set_face
    s = load_session(session)
    sm, ev, hd = s["samples"], s["events"], s["header"]
    if "qpos" not in sm:
        raise SystemExit(f"{session}: recorded without state (state=False): nothing to film")
    m = mujoco.MjModel.from_binary_path(str(Path(session) / "model.mjb"))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
    face = Face(seed=hd.get("config", {}).get("seed") or 0)
    tid = m.texture("face_tex").id
    body = m.body("body").id
    ts = sm["t"]
    T = float(ts[-1]) if len(ts) else 1.0
    sdt = float(hd.get("sample_dt", 0.06))
    every = max(1, int(round(speed)))
    fps = 1.0 / sdt * every / speed
    findings = []
    fp = Path(session) / "findings.json"
    if fp.exists():          # (the whole-session ones, like the activity mix, are in the report, not on frames)
        findings = [f for f in json.loads(fp.read_text(encoding="utf-8")) if f["kind"] not in SESSION_WIDE]
    # the activity strip, once
    colors = {}
    acts = sm.get("activity", np.array([""] * len(ts)))
    strip = Image.new("RGB", (W, 8), (30, 30, 30))
    sd = ImageDraw.Draw(strip)
    for i in range(len(ts)):
        x = int(ts[i] / T * (W - 1))
        sd.line([(x, 0), (x, 7)], fill=_color(str(acts[i]), colors))
    for f in findings:
        x = int(min(f["t0"], T) / T * (W - 1))
        sd.line([(x, 0), (x, 3)], fill=(255, 40, 40) if f["severity"] != "info" else (255, 160, 160))
    # what to show for a while after it happened
    shows = []
    for e in ev:
        if e["kind"] == "sound":
            shows.append((e["t"], 2.0, f"sound: {e['name']}"))
        elif e["kind"] in ("say", "touch", "show_prop", "move_prop", "hide_prop", "place"):
            what = e.get("text") or e.get("gesture") or e.get("thing")
            shows.append((e["t"], 3.5, f"you: {e['kind'].replace('_', ' ')} {what}"))
    steps = [(e["t"], e["step"]) for e in ev if e["kind"] == "step_start"]
    frames, writer = [], None
    out = Path(out) if out else Path(session) / ("session.gif" if gif else "session.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not gif:
        import cv2
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    si = 0
    for i in range(len(ts)):
        face.update(sdt, str(sm["face"][i]) if "face" in sm else None,
                    (float(sm["gaze_x"][i]), float(sm["gaze_y"][i])) if "gaze_x" in sm else None)
        if i % every:
            continue
        t = float(ts[i])
        d.qpos[:] = sm["qpos"][i]
        if "mocap" in sm:
            d.mocap_pos[:] = sm["mocap"][i]
        mujoco.mj_forward(m, d)
        set_face(m, face.rgb(scale=2))
        r._gl_context.make_current()
        mujoco.mjr_uploadTexture(m, r._mjr_context, tid)
        target = d.xpos[body].copy()
        target[2] = max(target[2], 0.02)
        cam.lookat[:] = target
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, W, 30], fill=(0, 0, 0, 150))
        act, ph = (str(sm["activity"][i]), str(sm["phase"][i])) if "activity" in sm else ("", "")
        dr.text((5, 3), f"t={t:5.1f}s  {act}{': ' + ph if ph else ''}", fill=(255, 255, 255))
        if "feeling" in sm:
            dr.text((5, 16), f"feeling {sm['feeling'][i]}  face {sm['face'][i]}  v={float(sm['valence'][i]):+.2f}",
                    fill=(200, 230, 255))
        while si + 1 < len(steps) and steps[si + 1][0] <= t:
            si += 1
        if steps and steps[si][0] <= t:
            dr.text((W - 5 - 6 * len(steps[si][1][:22]), 3), steps[si][1][:22], fill=(255, 220, 120))
        y = H - 24
        for t0, dur, text in shows:
            if t0 <= t < t0 + dur:
                dr.rectangle([0, y - 13, W, y], fill=(0, 0, 0, 140))
                dr.text((5, y - 12), text[:60], fill=(255, 255, 160))
                y -= 13
        for f in findings:
            if f["t0"] - 0.5 <= t <= max(f["t1"], f["t0"] + 3.0):
                dr.rectangle([0, y - 13, W, y], fill=(120, 0, 0, 160))
                dr.text((5, y - 12), f"observer: {f['kind']}  {f['evidence']}"[:62], fill=(255, 210, 210))
                y -= 13
                break
        img.paste(strip, (0, H - 10))
        x = int(t / T * (W - 1))
        dr.rectangle([x - 1, H - 12, x + 1, H], fill=(255, 255, 255, 255))
        if writer is not None:
            writer.write(np.asarray(img)[:, :, ::-1])
        else:
            frames.append(img)
    if writer is not None:
        writer.release()
    else:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    r.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--scenario", default="life", choices=sorted(SCENARIOS))
    ap.add_argument("--out", help="session directory (default media/sessions/<scenario>-seed<seed>)")
    ap.add_argument("--video", action="store_true", help="film it afterwards (MP4)")
    ap.add_argument("--gif", action="store_true", help="a GIF instead of an MP4")
    ap.add_argument("--video-out", help="where to put the video (default <session>/session.mp4)")
    ap.add_argument("--speed", type=float, default=1.0, help="video speed-up (whole numbers)")
    ap.add_argument("--no-state", action="store_true", help="don't keep the simulator state (no video later)")
    ap.add_argument("--quiet", action="store_true", help="no progress lines")
    ap.add_argument("--live", type=float, default=None, help="run the observer live every N seconds")
    ap.add_argument("--render", metavar="SESSION", help="film an existing recording")
    ap.add_argument("--report", metavar="SESSION", help="re-run the observer and metrics on a recording")
    a = ap.parse_args()
    if a.render:
        print(render(a.render, a.video_out, speed=a.speed, gif=a.gif))
        return 0
    if a.report:
        from pinch.observer import write_report
        print(write_report(a.report)["text"])
        return 0
    out, rep = record(a.minutes * 60.0, a.seed, a.scenario, a.out, state=not a.no_state, live_every=a.live,
                      quiet=a.quiet)
    print(rep["text"])
    print("session:", out)
    if a.video or a.gif:
        print("video:", render(out, a.video_out, speed=a.speed, gif=a.gif))
    return 0


if __name__ == "__main__":
    sys.exit(main())
