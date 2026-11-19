"""Voice understanding: what did my person just say, and what does it mean for me?

Two layers, so Pinch reacts at once and still understands more than keywords:
  1. rules  instant, offline: greetings, praise/scolding, tricks, "come here", "be quiet", tidy,
            "put the pen back", your name, your mood, meeting start/end ...
  2. LLM    a local model through Ollama (default gemma4:e4b; set PINCH_LLM=llama3.2:3b to switch)
            reads the same sentence plus Pinch's memory and returns the same structure, a few
            seconds later. It can only fill fields with values from fixed lists, so whatever it
            says is safe to act on.

    u = understand_rules("Pinch, put the pen back please")
    # {'intent': 'put_away', 'object': 'pen', 'addressed': True, ...}
"""

import json
import os
import re
import threading
import urllib.request

from crab.emotes import EMOTES

INTENTS = ["estop", "greet", "goodbye", "praise", "scold", "trick", "come", "go_away", "stop", "quiet", "tidy",
           "fetch", "put_away", "play", "sleep", "wake", "tell_name", "teach_home", "owner_feeling",
           "meeting_start", "meeting_end", "how_are_you", "chat", "none"]
OBJECTS = ["pen", "pencil", "eraser", "cap", "ball", "marker", "coin", "clip", "note"]
OWNER_EMOTIONS = ["", "happy", "excited", "sad", "stressed", "tired", "angry", "calm"]
TRICKS = {                                  # spoken word -> what Pinch does
    "wave": "wave", "hi five": "wave", "dance": "dance", "boogie": "dance", "beg": "beg", "please": "beg",
    "play dead": "play_dead", "bang": "play_dead", "shake": "handshake", "handshake": "handshake",
    "paw": "handshake", "claw": "handshake", "stretch": "stretch", "yawn": "yawn", "clap": "applause",
    "applause": "applause", "point": "point", "look around": "look_around", "spin": "spin", "turn around": "spin",
    "sneeze": "sneeze", "be shy": "shy", "angry face": "angry", "happy dance": "happy", "excited": "excited",
}
TRICK_NAMES = sorted(set(TRICKS.values()))
FEELINGS = {
    "sad": ("sad", "down", "upset", "depressed", "lonely", "crying", "miserable", "heartbroken", "bad day"),
    "stressed": ("stressed", "anxious", "worried", "overwhelmed", "deadline", "panicking", "so much work"),
    "tired": ("tired", "sleepy", "exhausted", "drained", "worn out"),
    "happy": ("happy", "great day", "awesome", "glad", "so good", "wonderful", "yay", "i did it"),
    "excited": ("excited", "can't wait", "pumped", "thrilled"),
    "angry": ("angry", "annoyed", "furious", "pissed", "frustrated"),
    "calm": ("relaxed", "calm", "chill"),
}
MEETING_START = ("in a meeting", "on a call", "meeting now", "call now", "joining a call", "joining the call",
                 "meeting starts", "meeting is starting", "got a meeting", "have a meeting", "hop on a call")
MEETING_END = ("meeting is over", "meeting's over", "meeting over", "meeting done", "call is over", "call's over",
               "call ended", "done with my call", "done with the meeting", "finished my meeting", "out of the meeting")
# phrases that give away a call even when nobody tells Pinch (used by the meeting detector)
CALL_CUES = ("can you hear me", "you're on mute", "you are on mute", "let's get started", "next slide",
             "share my screen", "see my screen", "any questions", "thanks everyone", "bye everyone",
             "let's sync", "can everyone", "i'll share", "you're breaking up", "talk to you later", "agenda")
NOT_NAMES = {w for ws in FEELINGS.values() for w in ws} | {"busy", "back", "here", "home", "fine", "ok", "okay",
             "good", "in", "on", "going", "so", "not", "just", "sorry", "done", "a", "the", "very", "really"}


def blank():
    return dict(intent="none", trick="", object="", name="", fact="", owner_emotion="", sentiment=0.0,
                addressed=False, meeting="", call_cue=False, text="", source="rules")


def _has(t, words):
    return any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", t) for w in words)


def understand_rules(text: str, pet_name="pinch") -> dict:
    u = blank()
    t = " " + text.lower().strip() + " "
    t = re.sub(r"[^a-z' ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    u["text"] = text
    u["addressed"] = pet_name in t or bool(re.match(r"^ (come|go|stop|sit|wave|dance|be|put|bring|fetch|get|"
                                                    r"tidy|clean|play|shake|beg|look|good|bad|no) ", t))
    u["call_cue"] = _has(t, CALL_CUES)
    for emo, words in FEELINGS.items():
        if _has(t, words) and re.search(r"\b(i'm|i am|i feel|feeling|i've been|today|my day|so)\b", t):
            u["owner_emotion"] = emo
            u["intent"] = "owner_feeling"
            break
    m = re.search(r"\b(?:my name is|call me|i am|i'm)\s+([a-z]+)", t)
    if m and m.group(1) not in NOT_NAMES and u["intent"] == "none":
        u["intent"], u["name"] = "tell_name", m.group(1).title()
    obj = next((o for o in OBJECTS if _has(t, [o, o + "s"])), "")
    u["object"] = obj

    def pick(intent, cond):
        if u["intent"] in ("none", "owner_feeling") and cond:
            u["intent"] = intent
            return True
        return False

    pick("meeting_end", _has(t, MEETING_END))
    pick("meeting_start", _has(t, MEETING_START))
    if u["intent"] in ("meeting_start", "meeting_end"):
        u["meeting"] = "start" if u["intent"] == "meeting_start" else "end"
    pick("estop", re.search(r"\bemergency stop\b", t))          # (a safety stop: only these exact words)
    pick("teach_home", obj and re.search(r"\b(goes here|lives here|belongs here|this is where|is its home|put it here)\b", t))
    pick("put_away", obj and re.search(r"\b(put|place|return|move)\b.*\b(back|away|tray|stand|home)\b", t))
    pick("fetch", obj and re.search(r"\b(bring|fetch|get|give|pass)\b", t))
    pick("tidy", re.search(r"\b(tidy|clean up|clean the desk|organi[sz]e|declutter|put (?:things|stuff|everything) away)\b", t))
    pick("quiet", re.search(r"\b(quiet|shh+|hush|silence|shut up|keep it down)\b", t))
    pick("scold", re.search(r"\b(no+ ?!?|bad|naughty|stop that|don't do that|cut it out)\b", t) and not re.search(r"\bnot bad\b", t))
    pick("praise", re.search(r"\b(good (?:boy|girl|crab|job|pinch)|well done|clever|smart|love you|cute|adorable|"
                             r"great job|nice job|attaboy|you're the best|thank you|thanks pinch|bravo)\b", t))
    for word, trick in sorted(TRICKS.items(), key=lambda kv: -len(kv[0])):
        if _has(t, [word]) and pick("trick", True):
            u["trick"] = trick
            break
    pick("come", re.search(r"\b(come here|come to me|come on|here pinch|over here|come)\b", t))
    pick("go_away", re.search(r"\b(go away|shoo|leave me alone|not now|go play)\b", t))
    pick("stop", re.search(r"\b(stop|freeze|stay|wait)\b", t))
    pick("play", re.search(r"\b(let's play|play with me|wanna play|want to play|play time|playtime|play)\b", t))
    pick("sleep", re.search(r"\b(go to sleep|good night|goodnight|nap time|bed ?time|sleep)\b", t))
    pick("wake", re.search(r"\b(wake up|good morning|rise and shine)\b", t))
    pick("goodbye", re.search(r"\b(bye|see you|see ya|gotta go|i'm leaving|later pinch)\b", t))
    pick("greet", re.search(r"\b(hi|hello|hey|hiya|yo|good afternoon|good evening|i'm back|i am back)\b", t))
    pick("how_are_you", re.search(r"\b(how are you|you ok|you okay|how do you feel|how's it going)\b", t))
    if u["intent"] == "none" and (u["addressed"] or len(t.split()) > 2):
        u["intent"] = "chat"
    pos = sum(_has(t, [w]) for w in ("good", "love", "great", "nice", "happy", "thank", "yay", "awesome", "cute"))
    neg = sum(_has(t, [w]) for w in ("bad", "no", "stop", "hate", "sad", "angry", "stupid", "annoying", "ugh"))
    u["sentiment"] = max(-1.0, min(1.0, 0.4 * (pos - neg)))
    if u["intent"] in ("praise", "greet"):
        u["sentiment"] = max(u["sentiment"], 0.6)
    if u["intent"] == "scold":
        u["sentiment"] = min(u["sentiment"], -0.6)
    if u["intent"] == "owner_feeling" and u["owner_emotion"] in ("sad", "stressed", "tired", "angry"):
        u["fact"] = f"felt {u['owner_emotion']}"
    return u


SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "trick": {"type": "string", "enum": [""] + TRICK_NAMES},
        "object": {"type": "string", "enum": [""] + OBJECTS},
        "name": {"type": "string", "description": "the person's own first name if they just told it, else empty"},
        "fact": {"type": "string", "description": "a short personal fact worth remembering (likes, pets, plans), else empty"},
        "owner_emotion": {"type": "string", "enum": OWNER_EMOTIONS},
        "sentiment": {"type": "number", "description": "-1 harsh .. 0 neutral .. 1 warm, toward Pinch"},
        "addressed": {"type": "boolean", "description": "is the person talking to Pinch (not to someone else)?"},
        "meeting": {"type": "string", "enum": ["", "start", "end"]},
    },
    "required": ["intent", "trick", "object", "name", "fact", "owner_emotion", "sentiment", "addressed", "meeting"],
}

SYSTEM = (
    "You are the ears of Pinch, a small robot pet crab on a desk. Pinch never talks; it understands its "
    "person like a clever pet. For each sentence you hear, fill the JSON fields.\n"
    f"intent: one of {', '.join(INTENTS)}. 'trick' when asked to perform one of: {', '.join(TRICK_NAMES)}. "
    "'put_away' when asked to put an object back where it belongs; 'fetch' when asked to bring it; "
    "'tidy' for cleaning up in general; 'teach_home' when the person says where an object belongs; "
    "'quiet' when asked to be quiet; 'owner_feeling' when they say how they feel; 'meeting_start'/'meeting_end' "
    "when they say a call or meeting starts or ends; 'chat' for anything else said to Pinch; 'none' if the "
    "sentence is not for Pinch (for example talking to colleagues on a call).\n"
    "Only use values from the lists. Leave fields empty when unsure. Reply with JSON only."
)


class LLMUnderstanding:
    """Asynchronous LLM parser: submit() returns at once, poll() gives the result when it's ready."""

    def __init__(self, model=None, host=None, timeout=90):
        self.model = model or os.environ.get("PINCH_LLM", "gemma4:e4b")
        self.host = (host or os.environ.get("PINCH_OLLAMA", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._result, self._busy = None, False
        self.lock = threading.Lock()
        self.available = None                     # unknown until the first call

    def check(self):
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read())["models"]]
            self.available = any(n.split(":")[0] == self.model.split(":")[0] for n in names)
        except Exception:
            self.available = False
        return self.available

    def parse(self, text, memory_summary=""):
        body = {"model": self.model, "stream": False, "format": SCHEMA, "think": False,
                "options": {"temperature": 0.2, "num_predict": 160},
                "messages": [{"role": "system", "content": SYSTEM + (" " + memory_summary if memory_summary else "")},
                             {"role": "user", "content": text}]}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            reply = json.loads(json.loads(r.read())["message"]["content"])
        return sanitize(reply, text)

    def submit(self, text, memory_summary=""):
        if self._busy:
            return False
        self._busy = True

        def work():
            try:
                res = self.parse(text, memory_summary)
            except Exception as e:                # offline, slow, malformed: the rules already reacted
                res = dict(blank(), text=text, source=f"llm-error: {type(e).__name__}")
            with self.lock:
                self._result = res
            self._busy = False
        threading.Thread(target=work, daemon=True).start()
        return True

    def poll(self):
        with self.lock:
            r, self._result = self._result, None
        return r


def sanitize(d: dict, text="") -> dict:
    u = blank()
    u.update(text=text, source="llm")
    u["intent"] = d.get("intent") if d.get("intent") in INTENTS else "none"
    u["trick"] = d.get("trick") if d.get("trick") in TRICK_NAMES else ""
    u["object"] = d.get("object") if d.get("object") in OBJECTS else ""
    name = re.sub(r"[^A-Za-z]", "", str(d.get("name", "")))[:20]
    u["name"] = name.title() if name.lower() not in NOT_NAMES else ""
    u["fact"] = str(d.get("fact", ""))[:120]
    u["owner_emotion"] = d.get("owner_emotion") if d.get("owner_emotion") in OWNER_EMOTIONS else ""
    try:
        u["sentiment"] = max(-1.0, min(1.0, float(d.get("sentiment", 0.0))))
    except (TypeError, ValueError):
        u["sentiment"] = 0.0
    u["addressed"] = bool(d.get("addressed", False))
    u["meeting"] = d.get("meeting") if d.get("meeting") in ("start", "end") else ""
    if u["trick"] and u["trick"] not in EMOTES and u["trick"] != "spin":
        u["trick"] = ""
    return u
