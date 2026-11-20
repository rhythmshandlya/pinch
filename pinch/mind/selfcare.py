"""Looking after itself: the learned danger map, battery hunger, and the self-preservation checks."""

from pinch.mind.common import (  # noqa: F401
    hunger,
)


class SelfCare:
    # ------------------------------------------------------------------ self-preservation (safety.py)
    def danger(self, xy):
        return self.danger_map.danger(xy)

    def hunger(self):
        return hunger(self.world.battery)

    def learn_danger(self, xy, what):
        spot = self.danger_map.learn(xy, what)
        self.danger_map.cap(self.world.desk)
        self.memory.data["danger"] = self.danger_map.to_list()
        self.memory.remember("danger", f"{what} near ({spot['x']:.2f}, {spot['y']:.2f}): fear {spot['v']:.2f}")
        self._say("danger", f"learned: {what} here is dangerous (fear {spot['v']:.2f})")

    def _self_preservation(self, dt):
        self.danger_map.decay(dt / 3600.0)
        if self.t - getattr(self, "_danger_saved_t", -1e9) > 60.0:    # faded fear survives a reboot (decision 94)
            self._danger_saved_t = self.t
            self.memory.data["danger"] = self.danger_map.to_list()
        if self.t - self._safe_check_t > 5.0:                 # walking through a feared spot safely unlearns it
            self._safe_check_t = self.t
            if not self.stuck and self.danger(self.world.pose[:2]) > 0.05:
                self.danger_map.safe_pass(self.world.pose[:2])
        # held up in the air: fear grows the longer it lasts (fear of heights), calmed by being petted
        if self.lifted_since is not None and self.t - self._lift_fear_t > 2.0:
            self._lift_fear_t = self.t
            held = self.t - self.lifted_since
            calm = 0.6 if self.t - self.world.owner.last_touch < 5.0 else 0.0   # a trusted hand on it
            self.affect.feel("high_up", (1 - calm) * min(1.0, 0.25 + held / 20.0))
        b = self.world.battery
        if self._last_battery is not None and b > self._last_battery + 1e-5 and not self.charging:
            self.charging = True                              # put on charge: relief, scaled by how hungry it was
            self.affect.feel("fed", 0.5 + 1.5 * hunger(self._last_battery))
            self.memory.remember("charged", f"charged from {self._last_battery:.0%}")
            self._say("battery", "charging - ahh")
        elif self._last_battery is not None and b < self._last_battery - 1e-5:
            self.charging = False
        self._last_battery = b
