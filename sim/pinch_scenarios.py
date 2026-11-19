"""Pinch's behaviour, tested in the digital twin: does it act like a pet should?

    python sim/pinch_scenarios.py          # all scenarios, writes docs/BEHAVIOUR_TESTS.md

Each scenario sets up a situation (you say something, a meeting starts, the desk is messy, you ignore
it, you're sad, you pet it ...) and checks what Pinch does in the full simulation: brain + body +
robot runtime + physics.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinch.hearing import RATE, synth_speech  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.sim_runner import DT, PinchSim  # noqa: E402

RESULTS = []


def scenario(name):
    def wrap(fn):
        def run():
            t0 = time.time()
            try:
                ok, detail = fn()
            except Exception as e:                     # a crash is a failure, not the end of the run
                ok, detail = False, f"crashed: {type(e).__name__}: {e}"
            RESULTS.append((name, ok, detail, time.time() - t0))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})", flush=True)
        run.__name__ = fn.__name__
        return run
    return wrap


def acts(sim, since=0.0):
    return [x[2] for x in sim.mind.log if x[1] == "activity" and x[0] >= since]


def emotes(sim, since=0.0):
    return [m[1][1] for m in sim.motions if m[1][0] == "emote" and m[0] >= since]


def fresh(**kw):
    sim = PinchSim(memory=Memory(path=None), **kw)
    sim.run(2.0)
    return sim


@scenario("obeys tricks: dance, wave, play dead, handshake")
def tricks():
    sim = fresh()
    done = []
    for word, emote in (("dance!", "dance"), ("pinch, wave", "wave"), ("play dead", "play_dead"), ("shake", "handshake")):
        t = sim.t
        sim.say(word)
        sim.run(9.0)
        done.append(emote in emotes(sim, t))
    return all(done), f"{sum(done)}/4 performed, falls {sim.falls}"


@scenario("learns your name and remembers it after a restart")
def memory_restart():
    path = Path(tempfile.mkdtemp()) / "memory.json"
    sim = PinchSim(memory=Memory(path=path))
    sim.run(1.0)
    sim.say("hi! my name is Sam")
    sim.say("I love green tea")
    sim.run(3.0)
    sim.mind.memory.save()
    again = Memory(path=path)
    return again.owner_name == "Sam", f"remembered '{again.owner_name}'"


@scenario("meeting from the sound alone: goes quiet (shh, no tones), then celebrates")
def meeting():
    sim = fresh()
    sim.run(20.0)
    t_start = sim.t
    audio = synth_speech(130, talk=0.55, seed=4)
    sounds_before = len(sim.sounds_played)
    quiet_seen = False
    for i in range(0, len(audio), RATE):
        sim.hear_audio(audio[i:i + RATE])
        sim.run(1.0)
        quiet_seen |= sim.mind.affect.quiet
    t_quiet = next((x[0] for x in sim.mind.log if x[1] == "meeting" and "going quiet" in x[2]), None)
    in_meeting_sounds = [s for t, s in sim.sounds_played[sounds_before:] if t_quiet and t > t_quiet + 1]
    shh = "shh" in emotes(sim, t_start)
    t_end = sim.t
    sim.say("ok, the meeting is over")
    sim.run(10.0)
    celebrated = "after_meeting" in acts(sim, t_end)
    ok = quiet_seen and shh and not in_meeting_sounds and celebrated
    return ok, (f"quiet after {t_quiet - t_start:.0f} s of talking, shh={shh}, tones during the call: "
                f"{len(in_meeting_sounds)}, celebrated after: {celebrated}" if t_quiet else "never went quiet")


@scenario("tidies: pushes the cap into the desk tray when asked")
def tidy_cap():
    sim = fresh()
    sim.place("cap", (-0.06, 0.0))
    sim.mind.world.see("cap", (-0.06, 0.0), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)
    sim.say("pinch, put the cap away")
    for _ in range(200):
        sim.run(1.0)
        if sim.in_tray("cap"):
            break
    return sim.in_tray("cap"), f"cap in tray after {sim.t:.0f} s, falls {sim.falls}, edge stops {sim.edge_stops}"


@scenario("notices on its own that something is out of place, and tidies it")
def tidy_spontaneous():
    sim = fresh()
    sim.place("eraser", (-0.05, 0.02))
    sim.mind.world.see("eraser", (-0.05, 0.02), graspable=False)
    sim.mind.memory.teach_home("eraser", sim.desk.tray_xy)
    for _ in range(240):
        sim.run(1.0)
        if sim.in_tray("eraser"):
            break
    return sim.in_tray("eraser"), f"eraser in tray after {sim.t:.0f} s without being asked"


@scenario("can't reach it (too close to the desk's back edge): asks you for help")
def ask_help():
    sim = fresh()
    sim.place("cap", (-0.12, 0.16))
    sim.mind.world.see("cap", (-0.12, 0.16), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)
    sim.say("pinch, put the cap away")
    sim.run(90.0)
    asked = any(e["kind"] == "help" for e in sim.mind.memory.data["episodes"])
    return asked and "beg" in emotes(sim), f"asked for help: {asked}, pointed: {'point' in emotes(sim)}"


@scenario("falls on its back: rolls itself back over")
def right_self():
    import mujoco
    sim = fresh()
    q = sim.d.qpos
    q[2] = 0.07
    q[3:7] = [0.0, 1.0, 0.0, 0.0]                               # upside down
    mujoco.mj_forward(sim.m, sim.d)
    upright_at = None
    for k in range(200):
        sim.run(0.1)
        R = sim.d.xmat[sim.m.body("body").id].reshape(3, 3)
        if upright_at is None and R[2, 2] > 0.85 and k > 5:
            upright_at = sim.t
    tried = any(mo[1][0] == "right_self" for mo in sim.motions)
    R = sim.d.xmat[sim.m.body("body").id].reshape(3, 3)
    ok = tried and R[2, 2] > 0.85 and sim.desk.inside(sim.pose()[:2], 0.0)
    return ok, f"tried righting: {tried}, upright again after {upright_at - 2.0 if upright_at else None} s, on desk {sim.desk.inside(sim.pose()[:2], 0.0)}"


@scenario("something about to fall off the desk: raises the alarm")
def edge_alert():
    sim = fresh()
    sim.place("ball", (0.10, -0.205))
    sim.mind.world.see("ball", (0.10, -0.205), graspable=False)
    sim.run(25.0)
    ok = "guard_edge" in acts(sim) and "point" in emotes(sim)
    return ok, f"activities {sorted(set(acts(sim)))[:6]}"


@scenario("you're sad: it comes close, offers a claw, purrs")
def comfort():
    sim = fresh()
    sim.run(5.0)
    t = sim.t
    sim.say("I'm feeling really sad today")
    sim.run(25.0)
    ok = "comfort" in acts(sim, t) and "handshake" in emotes(sim, t) and any(s == "purr" for tt, s in sim.sounds_played if tt > t)
    x, y, _ = sim.pose()
    d = np.hypot(x - sim.desk.owner_xy[0], y - (sim.desk.owner_xy[1] + 0.32))
    return ok and d < 0.08, f"comfort={ok}, ended {d * 100:.0f} cm from its spot near you"


@scenario("a bit needy: ignored while you're there, it seeks attention")
def needy():
    sim = fresh()
    sim.user_active = True                            # you're at your desk...
    sim.mind.affect.needs.social = 0.6
    t = sim.t
    sim.run(240.0)                                    # you're at the desk but never talk to it or touch it
    seek = "seek_attention" in acts(sim, t)
    chirps = sum(1 for tt, s in sim.sounds_played if s == "attention_chirp" and tt > t)
    return seek and chirps >= 1, f"seek_attention={seek}, attention chirps {chirps}"


@scenario("touch: purrs when stroked, giggles when tickled, startles when lifted")
def touch():
    sim = fresh()
    t = sim.t
    sim.touch("stroke")
    sim.run(4.0)
    sim.touch("tickle")
    sim.run(4.0)
    sim.touch("lifted")
    sim.run(4.0)
    s = [x for tt, x in sim.sounds_played if tt > t]
    ok = "purr" in s and "giggle" in s and "startled_squeak" in s
    return ok, f"sounds {s}"


@scenario("scolded: sulks, and stops doing the trick you didn't like")
def scolded():
    sim = fresh()
    sim.say("dance")
    sim.run(8.0)
    sim.say("no! bad crab")
    sim.run(6.0)
    sulk = "sulk" in acts(sim)
    return sulk and sim.mind.memory.data["likes"].get("dance", 0) < 0, \
        f"sulked={sulk}, likes dance={sim.mind.memory.data['likes'].get('dance', 0)}"


@scenario("ten minutes of pet life on its own: never falls, never leaves the desk")
def long_life():
    sim = fresh()
    sim.mess_seed = 1
    for k in range(60):
        sim.run(10.0)
        if k == 20:
            sim.say("let's play!")
        if k == 40:
            sim.touch("stroke")
    x, y, _ = sim.pose()
    on_desk = sim.desk.inside((x, y)) and sim.d.qpos[2] > 0.0
    kinds = sorted(set(acts(sim)))
    return sim.falls == 0 and on_desk, f"falls {sim.falls}, on desk {on_desk}, did: {', '.join(kinds)}"


def main():
    print("Pinch behaviour scenarios (digital twin)")
    only = sys.argv[1:]
    for fn in (tricks, memory_restart, meeting, tidy_cap, tidy_spontaneous, ask_help, edge_alert, comfort, needy,
               touch, scolded, right_self, long_life):
        if not only or fn.__name__ in only:
            fn()
    n = sum(ok for _, ok, _, _ in RESULTS)
    lines = ["# Behaviour tests (digital twin)", "", "Generated by `python sim/pinch_scenarios.py`: the full Pinch "
             "(brain + body + robot runtime + physics) in the desk simulation.", "",
             "| Scenario | Result | Details |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |" for name, ok, detail, _ in RESULTS]
    lines += ["", f"**{n}/{len(RESULTS)} passed.**", ""]
    if not only:
        (ROOT / "docs" / "BEHAVIOUR_TESTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{n}/{len(RESULTS)} passed")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
