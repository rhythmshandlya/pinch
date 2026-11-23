"""Congruence bench (phase 4): do Pinch's face, sounds and actions fit what it feels and what it is doing?

    python sim/benches/congruence.py              # three recorded scenarios (pet life, a scare, a tidy job)

Measured on every 0.1 s of awake time:
  face_feeling   the face says the opposite of the feeling (a happy face while sad or scared, a sad face while happy)
  face_hides_feeling  the face is not one that can show the current feeling (expression.toml [shows]), <= 10 %
  face_state     the face is not one that fits the current state (configs/mind/states.toml: e.g. a joyful face while
                 hiding, a sleepy face while playing)
  act_feeling    the activity contradicts the feeling (playing, celebrating or showing off while sad, scared or
                 angry; hiding or cowering while happy and not scared)
  stale_face     after an activity switch, the old activity's face still showing more than 3 s into the new one when
                 it doesn't fit the new state
  sound_feeling  a sound that says the opposite of the feeling (a giggle while scared, a whimper while happy)
Targets: face_feeling <= 2 %, face_state <= 10 %, act_feeling <= 2 %, stale_face <= 2 %, sound_feeling 0 (starting
values; the owner's review of the clips is the judge).
"""

import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinch.memory import Memory  # noqa: E402
from pinch.mind.expression import BRIGHT as POS_SOUNDS  # noqa: E402  (one table: configs/mind/expression.toml)
from pinch.mind.expression import DARK as NEG_SOUNDS  # noqa: E402
from pinch.mind.expression import NEGATIVE as NEG  # noqa: E402
from pinch.mind.expression import POSITIVE as POS  # noqa: E402
from pinch.mind.expression import SHOWS as EXPR_SHOWS  # noqa: E402
from pinch.mind.expression import STATE_FACES as FAMILY  # noqa: E402  (configs/mind/expression.toml)
from pinch.twin import PinchSim  # noqa: E402

# faces that can honestly show each feeling: the same table the face engine uses (configs/mind/expression.toml)
SHOWS = EXPR_SHOWS
FUN = {"play", "celebrate", "show_off"}
COWER = {"hide", "startle", "shy"}


def scenario(name, seed):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=["toy"] if name == "scare" else ())
    m = sim.mind
    if name != "tidy":
        m.memory.data["homes"] = {}
    script = {"life": [(5, "say", "hi pinch"), (40, "touch", "stroke"), (80, "say", "good crab"), (120, "say", "bad crab!"),
                       (170, "touch", "tickle"), (220, "say", "dance!"), (260, "say", "how are you?")],
              "scare": [(30, "rush", None), (90, "touch", "stroke"), (150, "rush", None), (200, "say", "hi pinch")],
              "tidy": [(5, "say", "pinch, tidy up"), (120, "touch", "stroke"), (200, "say", "good crab")]}[name]
    return sim, list(script)


def rush(sim):
    import math
    x, y, h = sim.pose()
    a = h + 0.2
    start = sim.desk.clamp((x + 0.30 * math.cos(a), y + 0.30 * math.sin(a)), 0.03)
    sim.show_prop("toy", start)
    sim.move_prop("toy", [start, (x + 0.10 * math.cos(h), y + 0.10 * math.sin(h))], speed=0.45)


def run(name, seed, seconds=300.0):
    sim, script = scenario(name, seed)
    m = sim.mind
    n = Counter()
    last_act, switch_t, face_at_switch = None, 0.0, None
    sounds_seen = 0
    examples = {}
    while sim.t < seconds:
        if script and sim.t >= script[0][0]:
            _, what, arg = script.pop(0)
            if what == "say":
                sim.say(arg)
            elif what == "touch":
                sim.touch(arg)
            else:
                rush(sim)
        sim.run(0.1)
        state = m.state
        if state == "sleep":
            continue
        n["t"] += 1
        f, face, act = m.affect.feeling(), m.face_expr, m.activity.name if m.activity else "idle"
        neg_feel = f in NEG
        pos_feel = f in POS
        if (face in POS and neg_feel) or (face in NEG and pos_feel):
            n["face_feeling"] += 1
            examples.setdefault("face_feeling", f"{sim.t:.0f}s face {face} while feeling {f} ({act})")
        if face not in SHOWS.get(f, {face}):
            n["face_hides_feeling"] += 1
            examples.setdefault("face_hides_feeling", f"{sim.t:.0f}s face {face} while feeling {f} ({act})")
        if face != "neutral" and face not in FAMILY.get(state, set()):
            n["face_state"] += 1
            examples.setdefault("face_state", f"{sim.t:.0f}s face {face} in state {state} ({act})")
        if (act in FUN and neg_feel) or (act in COWER and pos_feel and m.affect.needs.safety < 0.3):
            n["act_feeling"] += 1
            examples.setdefault("act_feeling", f"{sim.t:.0f}s {act} while feeling {f}")
        if act != last_act:
            last_act, switch_t, face_at_switch = act, sim.t, face
        elif sim.t - switch_t > 3.0 and face == face_at_switch and face != "neutral" \
                and face not in FAMILY.get(state, set()):
            n["stale_face"] += 1
            examples.setdefault("stale_face", f"{sim.t:.0f}s face {face} kept {sim.t - switch_t:.0f}s into {act}")
        for _, s in sim.sounds_played[sounds_seen:]:
            n["sounds"] += 1
            if (s in POS_SOUNDS and neg_feel) or (s in NEG_SOUNDS and pos_feel):
                n["sound_feeling"] += 1
                examples.setdefault("sound_feeling", f"{sim.t:.0f}s sound {s} while feeling {f}")
        sounds_seen = len(sim.sounds_played)
    return n, examples


LIMITS = dict(face_feeling=0.02, face_hides_feeling=0.10, face_state=0.10, act_feeling=0.02, stale_face=0.02)


def main():
    total = Counter()
    allex = {}
    for name, seed in (("life", 5), ("scare", 6), ("tidy", 7)):
        n, ex = run(name, seed)
        total.update(n)
        for k, v in ex.items():
            allex.setdefault(k, []).append(f"[{name}] {v}")
        print(f"{name:6s} " + "  ".join(f"{k} {100 * n[k] / max(1, n['t']):.1f}%" for k in LIMITS)
              + f"  sound_feeling {n['sound_feeling']}/{n['sounds']}")
    ok = True
    for k, lim in LIMITS.items():
        share = total[k] / max(1, total["t"])
        good = share <= lim
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {k:13s} {100 * share:5.1f} % of awake time (limit {100 * lim:.0f} %)")
        for e in allex.get(k, [])[:3]:
            print(f"        e.g. {e}")
    good = total["sound_feeling"] == 0
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  sound_feeling {total['sound_feeling']} of {total['sounds']} sounds")
    for e in allex.get("sound_feeling", [])[:3]:
        print(f"        e.g. {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PINCH_QUIET", "1")
    sys.exit(main())
