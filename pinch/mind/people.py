"""Who is around: sightings (faces), forgetting someone, and blaming the person who did something."""



class PeopleAware:
    # ------------------------------------------------------------------ people (people.py)
    def see_person(self, pid, confidence=1.0, name="", owner=False):
        """A person seen (on the robot: a recognised face; in the twin and the app: simulated)."""
        now = self.world.clock()
        known = pid in self.people.people
        last = self.people.get(pid)["last_seen"] if known else None
        if last is None or now - last > 60.0:                # someone (re)appearing catches its eye
            self.attention.notice("person", self.t, confidence, key=f"person:{pid}")
        away_h = self.people.see(pid, confidence, now, name=name, owner=owner)
        p = self.people.get(pid)
        if p["R"] > 0.45:                                     # someone it's afraid of
            self.affect.feel("loud_noise", 0.6)
            self._say("people", f"{p['name'] or pid} is here - scared ({self.people.describe(pid)})")
            self._interrupt()
        elif p["W"] > 0.5 and (not known or away_h > 0.25):   # a favourite back: greet, harder after longer
            self.greet_warmth = min(1.0, p["W"] * min(1.0, away_h / 8.0) + 0.2)
            self.owner_returned_t = self.t
            self._say("people", f"{p['name'] or pid} is back after {away_h:.1f} h!")
        self.memory.data["people"] = self.people.to_dict()

    def forget_person(self, pid):
        self.people.forget(pid)
        self.memory.data["people"] = self.people.to_dict()

    def _blame(self, event):
        pid = self.people.blame(event, self.world.clock())
        if pid:
            self.memory.data["people"] = self.people.to_dict()
