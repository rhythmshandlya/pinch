"""Failures injected on purpose for tests (decision 71)."""



class TwinFaults:
    FAULTS = ("vision", "flaky_detection", "audio", "cliff_dead", "weak_leg")

    def inject(self, fault, on=True, **kw):
        """Break something on purpose, for failure tests:
          vision            the camera sees nothing (no things, no faces)
          flaky_detection   half the sightings are lost and the rest are 3x noisier
          audio             the microphones hear nothing (speech and the room)
          cliff_dead        one cliff beam stops answering (beam="rear" ...)
          weak_leg          one leg's servos at a fraction of their strength (leg="FL", strength=0.5)"""
        assert fault in self.FAULTS, fault
        self._rec("fault", fault=fault, on=on, **kw)
        if not on:
            self.faults.pop(fault, None)
            if fault == "weak_leg" and "_forcerange0" in self.__dict__:
                self.m.actuator_forcerange[:] = self._forcerange0
            return
        self.faults[fault] = kw
        if fault == "weak_leg":
            self._forcerange0 = self.m.actuator_forcerange.copy()
            leg, k = kw.get("leg", "FL"), kw.get("strength", 0.5)
            for j in ("coxa", "femur", "tibia"):
                self.m.actuator_forcerange[self.m.actuator(f"{leg}_{j}").id] *= k
