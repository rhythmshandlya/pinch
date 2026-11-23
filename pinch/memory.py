"""Pinch's long-term memory: a small JSON file that survives restarts.

What it keeps (never audio, never images):
  owner       name and the little facts you tell it ("I like tea", "my cat is Momo")
  homes       where each desk object belongs (learned from where it usually sits, or taught)
  routines    when you tend to be in meetings / away / asleep (counts per hour of the day)
  stats       pets, praise, scolds, tricks, tidy-ups, last time you played
  episodes    the last few hundred notable moments, for "what happened today" and the LLM's context
  likes       how much it enjoys each activity (it learns your praise)
"""

import json
import os
import time
from pathlib import Path

from pinch.routine import DEFAULT_SETTINGS, local_hour

DEFAULT_FILE = Path(os.environ.get("PINCH_MEMORY", Path.home() / ".pinch" / "memory.json"))
MAX_EPISODES = 400


def _blank(now=None):
    return dict(owner=dict(name="", facts=[]), homes={}, sightings={}, routines=dict(meeting=[0] * 24,
                away=[0] * 24, active=[0] * 24), stats=dict(pets=0, praise=0, scolds=0, tricks=0, tidies=0,
                last_play=0.0, last_seen=0.0, born=time.time() if now is None else now), episodes=[], likes={},
                known_words={}, settings=dict(DEFAULT_SETTINGS))


class Memory:
    def __init__(self, path=DEFAULT_FILE, clock=time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.data = _blank(clock())
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in _blank().items():                    # keep new keys after upgrades
                    loaded.setdefault(k, v)
                self.data = loaded
            except (OSError, json.JSONDecodeError):
                pass
        self._dirty = False

    # -- writing ------------------------------------------------------------
    def changed(self):
        """Something in data was changed directly (the settings): save it next time."""
        self._dirty = True

    def snapshot(self):
        """The memory as text if it changed since the last save, else None. Cheap enough for the control
        loop; the slow part (writing it to the SD card) is write() and can run in another thread."""
        if not self.path or not self._dirty:
            return None
        self._dirty = False
        return json.dumps(self.data, indent=1)

    def write(self, text):
        """Atomic and durable: a power cut leaves either the old file or the new one, never half of one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())                                 # data on the card before the rename
        os.replace(tmp, self.path)
        if hasattr(os, "O_DIRECTORY"):                           # and the rename itself (Linux)
            fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def save(self):
        text = self.snapshot()
        if text is not None:
            self.write(text)

    def remember(self, kind, text, **extra):
        self.data["episodes"].append(dict(t=self.clock(), kind=kind, text=text, **extra))
        self.data["episodes"] = self.data["episodes"][-MAX_EPISODES:]
        self._dirty = True

    def count(self, stat, n=1):
        self.data["stats"][stat] = self.data["stats"].get(stat, 0) + n
        self._dirty = True

    def set_owner_name(self, name):
        name = name.strip().strip(".!?,").title()[:30]
        if name:
            self.data["owner"]["name"] = name
            self.remember("owner", f"my person's name is {name}")

    def add_fact(self, fact):
        """Something it heard about its person. With the setting ask_before_remembering (on by default) it waits in
        `pending` until the person approves it (approve_facts; the app and HTTP /approve); otherwise it is kept."""
        fact = (fact or "")[:120]
        facts = self.data["owner"]["facts"]
        pending = self.data["owner"].setdefault("pending", [])
        if not fact or fact in facts or fact in pending:
            return
        if self.data["settings"].get("ask_before_remembering", True):
            pending.append(fact)
            self.data["owner"]["pending"] = pending[-40:]
        else:
            facts.append(fact)
            self.data["owner"]["facts"] = facts[-40:]
        self._dirty = True

    def approve_facts(self, which=None, keep=True):
        """The person's answer about pending facts (all of them, or the ones named): keep them, or drop them."""
        pending = self.data["owner"].setdefault("pending", [])
        chosen = list(pending) if which is None else [f for f in pending if f in which]
        self.data["owner"]["pending"] = [f for f in pending if f not in chosen]
        if keep:
            facts = self.data["owner"]["facts"]
            facts += [f for f in chosen if f not in facts]
            self.data["owner"]["facts"] = facts[-40:]
        self._dirty = True
        return chosen

    def mark_routine(self, kind, hour=None):
        hour = self.hour() if hour is None else hour
        self.data["routines"][kind][hour] += 1
        self._dirty = True

    def like(self, activity, delta):
        likes = self.data["likes"]
        likes[activity] = round(min(1.0, max(-1.0, likes.get(activity, 0.0) + delta)), 3)
        self._dirty = True

    # -- object homes ---------------------------------------------------------
    def see_object(self, name, xy, settled=True):
        """Where an object sits when it's been still for a while: its home is the place it rests most."""
        if not settled:
            return
        s = self.data["sightings"].setdefault(name, [])
        s.append([round(float(xy[0]), 3), round(float(xy[1]), 3)])
        self.data["sightings"][name] = s[-50:]
        if name not in self.data["homes"] or len(s) % 10 == 0:
            if not self.data["homes"].get(name, {}).get("taught"):
                pts = sorted(s, key=lambda p: sum(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) < 0.03 ** 2 for q in s))
                self.data["homes"][name] = dict(xy=pts[-1], taught=False)
        self._dirty = True

    def teach_home(self, name, xy):
        self.data["homes"][name] = dict(xy=[round(float(xy[0]), 3), round(float(xy[1]), 3)], taught=True)
        self.remember("taught", f"{name} lives at ({xy[0]:.2f}, {xy[1]:.2f})")

    def home_of(self, name):
        h = self.data["homes"].get(name)
        return tuple(h["xy"]) if h else None

    # -- reading ---------------------------------------------------------------
    @property
    def owner_name(self):
        return self.data["owner"]["name"]

    def hour(self):
        """The local hour now (0-23), by the routine's clock (pinch/routine.py)."""
        return int(local_hour(self.clock(), self.data["settings"].get("utc_offset_h")))

    def meeting_likely(self, hour=None):
        hour = self.hour() if hour is None else hour
        m = self.data["routines"]["meeting"]
        return m[hour] >= 3 and m[hour] >= 0.3 * max(1, max(m))

    def recent(self, n=8, kinds=None):
        eps = [e for e in self.data["episodes"] if kinds is None or e["kind"] in kinds]
        return eps[-n:]

    def summary(self, n=6):
        """A few lines for the language model's context."""
        o = self.data["owner"]
        lines = []
        if o["name"]:
            lines.append(f"Your person is {o['name']}.")
        if o["facts"]:
            lines.append("Things you know about them: " + "; ".join(o["facts"][-6:]) + ".")
        if self.data["homes"]:
            lines.append("You know where these belong: " + ", ".join(sorted(self.data["homes"])) + ".")
        rec = self.recent(n)
        if rec:
            lines.append("Recently: " + "; ".join(e["text"] for e in rec) + ".")
        return " ".join(lines)
