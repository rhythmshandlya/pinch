"""Seeing unknown things and motion (pinch/novelty.py): sightings in, attention, startle and wariness out."""

from pinch.attention import IOR_S, direction_of
from pinch.novelty import FAMILIAR_N, NoveltyTracker, Observation  # noqa: F401 - Observation for callers


class Curious:
    @property
    def novelty(self):
        """The unknown things it is tracking and how familiar each is (created on first use; familiarity is
        remembered per look in memory["familiar"])."""
        nt = self.__dict__.get("_novelty")
        if nt is None:
            nt = self._novelty = NoveltyTracker(self.memory.data.setdefault("familiar", {}),
                                                timidity=self.affect.personality.timidity)
        return nt

    def observe(self, obs):
        """A sighting from vision (an Observation): something it can't name, or anything moving."""
        pose = self.world.pose
        for ev in self.novelty.observe(obs, self.t, pose, self.world.clock()):
            kind, th = ev[0], ev[1]
            d = direction_of(th.xy, pose)
            if kind == "new":
                fam = th.novelty < FAMILIAR_N
                if th.kind == "unknown":
                    self.attention.notice("new_thing", self.t, th.confidence * (0.3 if fam else 1.0), direction=d,
                                          xy=th.xy, key=th.key)
                    self._say("curious", f"{'a familiar' if fam else 'something new'} on the desk at "
                                         f"({th.xy[0]:.2f}, {th.xy[1]:.2f}), {th.size * 100:.0f} cm"
                                         f" (confidence {th.confidence:.2f})")
            elif kind == "moving":
                key = f"motion:{th.id}"
                if not self.attention.track(key, self.t, direction=d, xy=th.xy):
                    self.attention.notice("motion", self.t, min(1.0, ev[2] / 0.1) * th.confidence, direction=d,
                                          xy=th.xy, key=key)
            elif kind == "sudden":
                s = min(1.0, ev[2] * (0.5 + self.affect.personality.timidity))
                self.attention.notice("motion", self.t, 1.0, direction=d, xy=th.xy, key=f"motion:{th.id}")
                self.affect.feel("sudden_motion", s)
                self._say("startle", f"something moved at me ({th.speed:.2f} m/s, startle {s:.2f})")
                if s > 0.5:
                    self._interrupt()

    def unknown_things(self):
        """[(name, xy)] of the unknown things believed on the desk (the body keeps its feet off them too)."""
        if "_novelty" not in self.__dict__:
            return []
        return [(f"thing{th.id}", th.xy) for th in self.novelty.unknown(self.t)]

    def attend_key(self, key):
        """It has looked at the stimulus with this key: attention moves on (inhibition of return)."""
        self.attention.inhibited[key] = self.t + IOR_S
        if self.attention.focus is not None and self.attention.focus.key == key:
            self.attention.focus = None
