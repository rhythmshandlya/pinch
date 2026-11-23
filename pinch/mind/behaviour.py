"""The behaviour settings as data (configs/mind/behaviour.toml): the declared priority order and every activity's
category in it. Selection asks rank() when deciding whether one activity may interrupt another; the decision log
records the categories; the version goes into every session recording."""

import tomllib
from functools import cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "configs" / "mind" / "behaviour.toml"


@cache
def settings(path=None):
    with open(path or PATH, "rb") as f:
        s = tomllib.load(f)
    order = s["priority"]["order"]
    unknown = {c for c in s["activity"].values() if c not in order}
    if unknown:
        raise ValueError(f"{PATH.name}: categories not in the priority order: {sorted(unknown)}")
    return s


def version():
    return settings()["version"]


def category(name):
    """The priority category of an activity (KeyError for an activity the file does not place)."""
    return settings()["activity"][name]


def rank(name):
    """1 = highest priority (physical safety) .. 9 (background)."""
    return settings()["priority"]["order"].index(category(name)) + 1


def overrides_uninterruptible(name):
    """May this activity cut short one that is not interruptible (an emergency stop is separate: no motion at all)?"""
    return rank(name) <= settings()["priority"]["overrides_uninterruptible"]


STATES_PATH = PATH.with_name("states.toml")
CHANNELS = ("legs", "claws", "gaze", "face", "sound")


@cache
def states():
    with open(STATES_PATH, "rb") as f:
        s = tomllib.load(f)
    s.pop("version", None)
    for name, st in s.items():
        missing = [k for k in ("sub", "allow", "intent", "interrupt", "on_timeout") if k not in st]
        if missing or set(st["intent"]) != set(CHANNELS):
            raise ValueError(f"{STATES_PATH.name} [{name}]: missing {missing or 'intent for ' + str(CHANNELS)}")
    return s


def allowed(state, step_kind):
    """May a motion step of this kind run in this state?"""
    return step_kind in states()[state]["allow"]
