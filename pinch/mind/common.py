"""Shared by the mind's parts: the motion step kinds, mood replies and imports."""

from collections import deque

import numpy as np

from pinch.activities import LIBRARY
from pinch.affect import Affect, Personality
from pinch.hearing import MeetingDetector, OwnerVoiceMood
from pinch.language import LLMUnderstanding, understand_rules
from pinch.memory import Memory
from pinch.people import People
from pinch.play import PlayBook
from pinch.routine import Routine
from pinch.safety import NO_GO, DangerMap, hunger
from pinch.touch import GESTURE_EVENT
from pinch.world import WorldModel, situations

MOTION_STEPS = {"emote", "walk", "goto", "turn_to", "pose", "push", "pick", "drop", "release", "recover",
                "right_self"}
MOOD_REPLY = {   # "how are you?" -> face, sound, emote
    "happy": ("happy", "happy_trill", "happy"), "excited": ("excited", "giggle", "excited"),
    "content": ("content", "purr", "fidget"), "love": ("love", "purr", "handshake"),
    "lonely": ("lonely", "whimper", "beg"), "sad": ("sad", "sad_tone", "sad"), "bored": ("bored", "sleepy_hum", "yawn"),
    "sleepy": ("sleepy", "yawn", "yawn"), "scared": ("scared", "startled_squeak", "scared"),
    "curious": ("curious", "curious_chirp", "curious"), "grumpy": ("grumpy", "angry_buzz", "angry"),
    "angry": ("angry", "angry_buzz", "angry"), "neutral": ("happy", "greeting_beep", "wave"),
}
