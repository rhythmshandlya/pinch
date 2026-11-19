"""The owner's controls (decision 83): what Pinch remembers is readable, each kind can be forgotten (people
take their face ids with them), "everything" keeps the settings, and face recognition can be turned off."""

from pinch.memory import Memory
from pinch.mind import Mind


def _mind_that_knows_things():
    m = Mind(memory=Memory(path=None), llm=False, seed=1, clock=lambda: 1795059000.0 + m.t)
    m.hear_text("my name is Sam")
    m.memory.add_fact("likes tea")
    m.see_person("p1", 0.9, name="Sam", owner=True)
    m.see_person("p2", 0.8)
    m.memory.teach_home("pen", (0.05, 0.10))
    m.danger_map.learn((0.1, 0.1), "fell")
    m.memory.remember("note", "a test episode")
    m.routine.set_quiet_hours(22, 7)
    return m


def test_summary_shows_what_it_knows():
    s = _mind_that_knows_things().memory_summary()
    assert s["owner"]["name"] == "Sam" and "likes tea" in s["owner"]["facts"]
    assert {p["id"] for p in s["people"]} == {"p1", "p2"}
    assert s["homes"] == ["pen"] and s["danger_spots"] == 1 and s["episodes"] >= 1
    assert "embedding" not in str(s) and "gallery" not in str(s)


def test_forgetting_each_kind():
    m = _mind_that_knows_things()
    assert sorted(m.forget("people")) == ["p1", "p2"] and m.memory_summary()["people"] == []
    m2 = _mind_that_knows_things()
    assert m2.forget("person:p2") == ["p2"] and [p["id"] for p in m2.memory_summary()["people"]] == ["p1"]
    m2.forget("homes")
    m2.forget("places")
    assert m2.memory_summary()["homes"] == [] and m2.memory_summary()["danger_spots"] == 0
    for bad in ("passwords", "person:nobody"):
        try:
            m2.forget(bad)
            raise AssertionError(f"forget({bad!r}) should fail")
        except (KeyError, ValueError):
            pass


def test_forget_everything_keeps_settings():
    m = _mind_that_knows_things()
    faces = m.forget("everything")
    s = m.memory_summary()
    assert sorted(faces) == ["p1", "p2"]
    assert s["owner"] == {"name": "", "facts": []} and s["people"] == [] and s["episodes"] == 0
    assert s["homes"] == [] and s["danger_spots"] == 0 and sum(s["routines"].values()) == 0
    assert s["settings"]["quiet_hours"] == [22.0, 7.0], "the owner's settings must survive a forget"


def test_face_recognition_switch():
    m = _mind_that_knows_things()
    assert m.face_recognition
    m.set_face_recognition(False)
    assert not m.face_recognition and m.memory.data["settings"]["face_recognition"] is False
