"""Let the crab live for a few minutes in the simulator with its pet mind.

    python brain/run_pet.py                 # stand-in mind, 180 s, writes docs/images/pet_timeline.png
    python brain/run_pet.py --gemma         # Gemma 4 (Ollama) as the slow mind
    python brain/run_pet.py --view          # watch it
    python brain/run_pet.py --gemma --host http://<pc-ip>:11434   # Gemma on another computer

Scripted world events: petted at 40 s, "hello" at 70 s, a loud bang at 110 s, then quiet.
Between events nobody does anything - whatever the crab does then, it does on its own.
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.brain import MockBrain, OllamaBrain  # noqa: E402
from brain.pet_mind import Event, Heart, Mind, Novelty, Reflexes  # noqa: E402
from brain.sounds import play  # noqa: E402
from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from robot.crab_robot import DT, Crab  # noqa: E402

WORLD = [(40.0, Event("touch", 1.0)), (70.0, Event("voice", 1.0, {"text": "hello little crab"})),
         (72.0, Event("face", 1.0)), (110.0, Event("loud_sound", 1.0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--host", default="http://localhost:11434", help="where Ollama runs")
    a = ap.parse_args()

    import mujoco
    from PIL import Image
    from robot.backends import SimBackend
    hw = SimBackend(render=a.view)
    ren = mujoco.Renderer(hw.m, 120, 160)
    crab = Crab(hw)
    crab.start()
    heart, nov, refl = Heart(), Novelty(), Reflexes(JOINT_NAMES)
    mind = Mind(OllamaBrain("gemma4:e4b", host=a.host) if a.gemma else MockBrain(), period=45.0)
    gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * _S, 0.0]))
    V, W = _S / _T, 1 / _T                                          # speeds scaled to the crab's size
    rng = np.random.default_rng(1)
    pending = list(WORLD)
    trace, sounds = [], []
    explore_cmd, chunk_end, current = (0, 0, 0), 0.0, None
    t = 0.0
    last_heard = ""

    while t < a.seconds:
        events = []
        while pending and pending[0][0] <= t:
            ev = pending.pop(0)[1]
            events.append(ev)
            if ev.kind == "voice":
                last_heard = ev.data.get("text", "")
        if int(t / DT) % 25 == 0:                                   # look around ~2x per second
            ren.update_scene(hw.d, camera="eye_cam")
            n = nov(ren.render())
            if n > 0.5 and t > 3.0:                                 # not while it is still getting up
                events.append(Event("novelty", n))
        beh = heart.tick(DT, events)
        if int(t / DT) % 50 == 0:                                   # ask the slow mind now and then
            ren.update_scene(hw.d, camera="eye_cam")
            buf = io.BytesIO()
            Image.fromarray(ren.render()).save(buf, format="PNG")
            d = heart.d
            ctx = (last_heard or "") + f" (you feel: curiosity {d.curiosity:.1f}, boredom {d.boredom:.1f}, " \
                f"energy {d.energy:.1f}, lonely {d.social:.1f}, fear {d.fear:.1f})"
            mind.maybe_think(t, buf.getvalue(), ctx)
            last_heard = ""
        sug = mind.take()
        if sug:
            heart.suggest(Mind.MAP.get(sug["emote"], "explore" if sug["walk"]["seconds"] > 0 else "idle"))
            if sug["sound"] != "none":
                sounds.append((t, sug["sound"]))
                play(sug["sound"], out_dir=ROOT / "docs" / "sounds")

        # turn the chosen behaviour into motion, in short chunks so the heart can change its mind
        if beh != current or t >= chunk_end:
            current = beh
            if beh in ("play", "seek_attention", "hide", "rest", "startle"):
                emo = {"play": "happy", "seek_attention": "wave", "hide": "shy", "rest": "sleep", "startle": "angry"}[beh]
                snd = {"play": "happy_trill", "seek_attention": "greeting_beep", "hide": "curious_chirp",
                       "rest": "sleepy_hum", "startle": "startled_squeak"}[beh]
                sounds.append((t, snd))
                t0 = t
                while t - t0 < EMOTES[emo].duration and t < a.seconds:
                    crab.tick(EMOTES[emo].target(t - t0))
                    t += DT
                    heart.tick(DT)
                    if int(t / DT) % 25 == 0:
                        d = heart.d
                        trace.append((t, d.curiosity, d.boredom, d.energy, d.social, d.fear, beh,
                                      hw.d.qpos[0], hw.d.qpos[1]))
                chunk_end = t
                continue
            if beh == "explore":
                explore_cmd = [(0, 0.08 * V, 0), (0, -0.08 * V, 0), (0.06 * V, 0, 0), (0, 0, 0.5 * W)][rng.integers(4)]
            elif beh == "approach":
                explore_cmd = (0.06 * V, 0, 0)
            else:
                explore_cmd = (0, 0, 0)
            chunk_end = t + 3.0
        q = STAND
        for _ in range(int(DT / 0.002)):
            q = gait.step(0.002, explore_cmd if beh in ("explore", "approach") else (0, 0, 0))
        crab.tick(refl.overlay(q, t, heart.d, beh))
        t += DT
        if int(t / DT) % 25 == 0:
            d = heart.d
            trace.append((t, d.curiosity, d.boredom, d.energy, d.social, d.fear, beh, hw.d.qpos[0], hw.d.qpos[1]))

    crab.shutdown()
    plot(trace, sounds, heart.log)
    travelled = sum(np.hypot(b[7] - a_[7], b[8] - a_[8]) for a_, b in zip(trace, trace[1:]))
    beh_time = {}
    for i in range(1, len(trace)):
        beh_time[trace[i][6]] = beh_time.get(trace[i][6], 0) + (trace[i][0] - trace[i - 1][0])
    print("time per behaviour (s):", {k: round(v) for k, v in sorted(beh_time.items(), key=lambda x: -x[1])})
    quiet = [s for s in heart.log if s[1] == "behaviour" and not any(abs(s[0] - w[0]) < 8 for w in WORLD)]
    print(f"self-started behaviour changes (no event within 8 s): {len(quiet)}; walked {travelled * 100:.0f} cm in total")
    print("sounds:", sounds[:12])


def plot(trace, sounds, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = [r[0] for r in trace]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, (name, col) in enumerate([("curiosity", "#2a9d8f"), ("boredom", "#e9c46a"), ("energy", "#457b9d"),
                                     ("lonely (social)", "#e76f51"), ("fear", "#6d597a")]):
        ax.plot(tt, [r[1 + i] for r in trace], label=name, color=col, lw=2)
    last = -1e9
    for tw, ev in WORLD:
        ax.axvline(tw, color="#999", ls=":", lw=1)
        if tw - last > 6:                                       # don't print labels on top of each other
            ax.text(tw, 1.02, ev.kind.replace("_", " "), ha="center", fontsize=8, color="#555")
            last = tw
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("drive level")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title("The crab left alone: drives and what it chose to do")
    names = ["idle", "explore", "approach", "play", "seek_attention", "rest", "hide", "startle"]
    bx.scatter(tt, [names.index(r[6]) for r in trace], c=[names.index(r[6]) for r in trace], cmap="tab10", s=8)
    bx.set_yticks(range(len(names)), names, fontsize=7)
    bx.set_xlabel("time (s)")
    fig.tight_layout()
    out = ROOT / "docs" / "images" / "pet_timeline.png"
    fig.savefig(out, dpi=110)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
