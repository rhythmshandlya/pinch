"""What Pinch shows comes from one inner state (phase 4, part 2; configs/mind/expression.toml).

The face: the feeling is the base (affect.feeling()); an activity asks for a look (an intent, e.g. "focused" while
lining up a pick, "happy" when a game starts), which shows for intent_hold_s only if it can honestly show the
feeling now ([shows] table); otherwise, and afterwards, the feeling's own face shows. Asleep shows asleep.
Research: Kismet blends one affect state into the face; Loyall/Oz: emotion shown through all behaviour, the
activity never overrides what the character feels (docs/research/v2_1/expression, believable).
"""

from pinch.settings import load

EXPR = load("expression")
SHOWS = {k: set(v) for k, v in EXPR["shows"].items()}
FEELING_FACE = {"neutral": "neutral"}              # a feeling's own face is the face of the same name
PERFORM = set(EXPR["face"]["perform"])


def fits(face, feeling):
    """Can this face honestly show this feeling?"""
    return face in SHOWS.get(feeling, {feeling, "neutral"})


def face_now(mind):
    """The face to draw now."""
    if mind.state == "sleep":
        return "asleep"
    feeling = mind.affect.feeling()
    out = mind.outcome_intent       # how the move it just made went: shown first, briefly
    fresh = out is not None and out[2] == mind.started and mind.t - out[1] <= EXPR["face"]["outcome_hold_s"]
    if fresh and fits(out[0], feeling):
        return out[0]
    intent = mind.face_intent
    if intent is not None and intent[2] == mind.started and mind.activity is not None:   # this activity asked
        face, t0, _ = intent
        if face in PERFORM:                            # a body state or a trick: shown while the activity lasts
            return face
        if mind.t - t0 <= EXPR["face"]["intent_hold_s"] and fits(face, feeling):
            return face
    return FEELING_FACE.get(feeling, feeling)


STATE_FACES = {k: set(v) | {"neutral"} for k, v in EXPR["state_faces"].items()}   # faces that fit each state
SND = EXPR["sound"]
BRIGHT, DARK, SLEEPY_SOUNDS = set(SND["bright"]), set(SND["dark"]), set(SND["sleepy"])
POSITIVE, NEGATIVE = set(EXPR["valence"]["positive"]), set(EXPR["valence"]["negative"])
GOOD, BAD = POSITIVE, NEGATIVE                     # (feelings a sound must not contradict)


def sound_ok(sound, feeling):
    """Does this sound fit the feeling? (a giggle while scared or sad does not; a whimper while happy does not)"""
    return not ((sound in BRIGHT and feeling in BAD) or (sound in DARK and feeling in GOOD))


def sound_for(sound, feeling):
    """The sound to play when an activity asks for this one: itself if it fits, else the feeling's own sound if
    that fits, else none."""
    if sound_ok(sound, feeling):
        return sound
    own = SND["of"].get(feeling)
    return own if own and sound_ok(own, feeling) else None
