"""The crab's "brain": a local vision-language model (Gemma 4 via Ollama) that
looks through the crab's eye camera, listens to what you say, and picks what
to do next. The fast stuff (balance, walking, emote motion) stays in the robot
runtime; the brain only chooses *which* behaviour, a few times a minute.

    brain = OllamaBrain(model="gemma4:e4b")     # needs `ollama serve` + `ollama pull gemma4:e4b`
    action = brain.decide(image_png, heard="hello crab!")
    # -> {"thought": ..., "say": ..., "emote": "wave", "walk": {"vx":0,"vy":0,"wz":0,"seconds":0}}

MockBrain gives the same interface without a model, for tests.
Everything the brain returns goes through `sanitize()` before the robot acts.
"""

import base64
import json
import urllib.request

from brain.sounds import NAMES as SOUND_NAMES
from crab.emotes import EMOTES
from crab.gait import _S, _T

EMOTE_NAMES = sorted(EMOTES)
_V, _W = _S / _T, 1 / _T                       # speeds scaled to the crab's size (hand-sized)
LIMITS = {"vx": round(0.10 * _V, 3), "vy": round(0.12 * _V, 3), "wz": round(0.7 * _W, 2), "seconds": 5.0}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "description": "one short sentence: what you notice and why you act"},
        "sound": {"type": "string", "enum": SOUND_NAMES + ["none"],
                  "description": "a pet tone that shows your feeling (you never speak words)"},
        "emote": {"type": "string", "enum": EMOTE_NAMES + ["none"]},
        "walk": {
            "type": "object",
            "properties": {
                "vx": {"type": "number", "description": f"forward m/s, -{LIMITS['vx']}..{LIMITS['vx']}"},
                "vy": {"type": "number", "description": f"sideways (left +) m/s, -{LIMITS['vy']}..{LIMITS['vy']}; "
                                                       "crabs love sideways"},
                "wz": {"type": "number", "description": f"turn rad/s (left +), -{LIMITS['wz']}..{LIMITS['wz']}"},
                "seconds": {"type": "number", "description": "0..5"},
            },
            "required": ["vx", "vy", "wz", "seconds"],
        },
    },
    "required": ["thought", "sound", "emote", "walk"],
}

SYSTEM_PROMPT = f"""You are the brain of a small robot crab (hand-sized: about 20 cm wide, 0.65 kg).
You see through one wide camera low in the front of your body, and you sometimes hear a person.
Your body can: play one emote ({", ".join(EMOTE_NAMES)}), or walk for up to 5 seconds
(forward/back up to {LIMITS['vx']} m/s, sideways up to {LIMITS['vy']} m/s - you are a crab, you prefer
sideways - and turn up to {LIMITS['wz']} rad/s). You cannot pick things up yet.
Personality: curious, friendly, a bit dramatic, easily startled. You are a pet: you NEVER talk,
you express yourself with body language (emotes, walking) and short tones ({", ".join(SOUND_NAMES)}).
You understand people a little, like a clever pet does.
Approaching a person who invites you is good - walk toward them slowly (2-3 seconds at a time).
Safety: never walk toward a table edge, stairs, water or an animal you don't know; if unsure,
stay still and use an emote instead. Walking 0 seconds means stay put.
Reply ONLY with JSON matching the schema."""


def sanitize(action: dict) -> dict:
    """Clamp whatever a model says into something the robot can safely do."""
    emote = action.get("emote", "none")
    if emote not in EMOTE_NAMES:
        emote = "none"
    walk = action.get("walk") or {}
    clean = {}
    for k, lim in LIMITS.items():
        try:
            v = float(walk.get(k, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        clean[k] = max(0.0, min(lim, v)) if k == "seconds" else max(-lim, min(lim, v))
    sound = action.get("sound", "none")
    if sound not in SOUND_NAMES:
        sound = "none"
    return {"thought": str(action.get("thought", ""))[:200], "sound": sound, "emote": emote, "walk": clean}


class OllamaBrain:
    def __init__(self, model="gemma4:e4b", host="http://localhost:11434", timeout=120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.history = []                       # last few exchanges, for a sense of continuity

    def decide(self, image_png: bytes | None, heard: str = "") -> dict:
        user = {"role": "user",
                "content": (f'The person said: "{heard}". ' if heard else "Nobody is talking right now. ")
                + "Here is what your eye camera sees. What do you do?"}
        if image_png:
            user["images"] = [base64.b64encode(image_png).decode()]
        body = {"model": self.model, "stream": False, "format": ACTION_SCHEMA,
                "options": {"temperature": 0.7},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-6:] + [user]}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            reply = json.loads(r.read())["message"]["content"]
        action = sanitize(json.loads(reply))
        self.history += [{"role": "user", "content": user["content"]},
                         {"role": "assistant", "content": json.dumps(action)}]
        return action


class MockBrain:
    """Keyword rules, same interface. For tests and for running without a model."""

    RULES = [
        (("hello", "hi ", "hey"), dict(sound="greeting_beep", emote="wave")),
        (("come", "here"), dict(sound="curious_chirp", walk=dict(vx=0.8 * LIMITS["vx"], vy=0, wz=0, seconds=3))),
        (("left",), dict(sound="happy_trill", walk=dict(vx=0, vy=0.8 * LIMITS["vy"], wz=0, seconds=3))),
        (("right",), dict(sound="happy_trill", walk=dict(vx=0, vy=-0.8 * LIMITS["vy"], wz=0, seconds=3))),
        (("turn", "spin"), dict(sound="happy_trill", walk=dict(vx=0, vy=0, wz=0.8 * LIMITS["wz"], seconds=3))),
        (("angry", "fight", "mad"), dict(sound="angry_buzz", emote="angry")),
        (("dance", "happy", "yay"), dict(sound="happy_trill", emote="happy")),
        (("shy", "cute"), dict(sound="curious_chirp", emote="shy")),
        (("night", "sleep", "bye"), dict(sound="sleepy_hum", emote="sleep")),
    ]

    def decide(self, image_png: bytes | None, heard: str = "") -> dict:
        h = f" {heard.lower()} "
        for keys, act in self.RULES:
            if any(k in h for k in keys):
                return sanitize({"thought": f"heard '{heard}'", "emote": "none", "walk": {}, **act})
        return sanitize(dict(thought="nothing to react to", sound="none", emote="idle", walk={}))
