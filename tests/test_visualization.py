"""SIM-019 behaviour visualization: the clip recorder (sim/clips.py) films the twin with what Pinch is doing written
on every frame - its state, activity, sub-state and feeling - into a GIF a person can watch."""

import tempfile
from pathlib import Path

from PIL import Image


def test_a_clip_shows_the_behaviour_it_films():
    import sim.clips as clips
    with tempfile.TemporaryDirectory() as tmp:
        old = clips.OUT
        clips.OUT = Path(tmp)
        try:
            sim = clips._sim(seed=5)
            rec = clips.Recorder(sim.m, sim.d, dist=0.6, elev=-45.0)
            labels = []
            for _ in range(6):
                sim.run(0.5)
                a = sim.mind.activity
                labels.append(f"{sim.mind.state}: {a.name if a else ''} {sim.mind.phase} feeling {sim.mind.affect.feeling()}")
                rec.grab(labels[-1])
            path = rec.save("viz_test")
            with Image.open(path) as im:
                assert im.n_frames == 6 and im.size == (clips.W, clips.H)
        finally:
            clips.OUT = old
    assert all(lb.split(":")[0] in ("idle", "explore", "social", "play", "alert", "defensive", "rest", "sleep",
                                    "recover", "task") for lb in labels)
