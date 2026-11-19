"""The owner's controls over what Pinch keeps (decision 83): see what it remembers, make it forget, turn face
recognition off. Reached through the robot's HTTP API (robot/http_api.py) and the app."""

FORGETTABLE = ("people", "owner", "facts", "episodes", "homes", "places", "games", "routines", "everything")


class Privacy:
    def memory_summary(self):
        """What Pinch remembers, readable: names and counts, never face fingerprints or raw episodes."""
        d = self.memory.data
        return dict(
            owner=dict(name=d["owner"]["name"], facts=list(d["owner"]["facts"])),
            people=[dict(id=pid, name=p.get("name", ""), owner=bool(p.get("owner")),
                         feeling=self.people.describe(pid)) for pid, p in list(self.people.people.items())],
            homes=sorted(d["homes"]), episodes=len(d["episodes"]), likes=dict(d["likes"]),
            danger_spots=len(self.danger_map.spots), games=sorted(self.playbook.games),
            routines={k: sum(v) for k, v in d["routines"].items()},
            settings=dict(self.routine.settings))

    def forget(self, what):
        """Forget one kind of memory (FORGETTABLE) or one person ("person:<id>"). Returns the face ids the
        firmware must also drop from its face gallery (the fingerprints live there, not here)."""
        d = self.memory.data
        faces = []
        if what.startswith("person:"):
            pid = what.split(":", 1)[1]
            if pid not in self.people.people:
                raise KeyError(pid)
            self.forget_person(pid)
            faces = [pid]
        elif what not in FORGETTABLE:
            raise ValueError(f"can forget {', '.join(FORGETTABLE)} or person:<id>, not {what!r}")
        everything = what == "everything"
        if what in ("people", "everything"):
            faces = list(self.people.people)
            for pid in faces:
                self.people.forget(pid)
            d["people"] = self.people.to_dict()
        if what in ("owner", "everything"):
            d["owner"]["name"] = ""
            d["owner"]["facts"] = []
        if what == "facts":
            d["owner"]["facts"] = []
        if what in ("episodes", "everything"):
            d["episodes"] = []
        if what in ("homes", "everything"):
            d["homes"], d["sightings"] = {}, {}
        if what in ("places", "everything"):                  # the desk spots it learned to fear
            self.danger_map.spots = []
            d["danger"] = self.danger_map.to_list()
        if what in ("games", "everything"):
            self.playbook.games, self.playbook.toys = {}, {}
            d["play"] = self.playbook.to_dict()
            d["likes"] = {}
        if what in ("routines", "everything"):
            d["routines"] = {k: [0] * 24 for k in d["routines"]}
        self.memory.changed()
        self._say("privacy", f"forgot {what}" + ("" if not everything else " (settings kept)"))
        return faces

    @property
    def face_recognition(self):
        return bool(self.routine.settings.get("face_recognition", True))

    def set_face_recognition(self, on):
        """Off: faces are no longer matched to people (the firmware stops fingerprinting them)."""
        self.routine.settings["face_recognition"] = bool(on)
        self.memory.changed()
        self._say("privacy", "face recognition " + ("on" if on else "off"))
