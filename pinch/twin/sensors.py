"""Pinch's simulated senses: the camera's sightings, the cliff beams, the close-up grasp view and the claw's grip."""

from pinch.novelty import FOV_HALF, FOV_RANGE, Observation
from pinch.twin.common import (  # noqa: F401
    OBJECTS,
    np,
    twin_sensor,
)


class TwinSensors:
    # what the learned grab sees (on the robot: the camera); kinds and half-extents as in training/grasp_env.py
    GRASP_SHAPES = {"pen": (0, (0.0695, 0.0045, 0.0045), 0.055), "eraser": (1, (0.010, 0.006, 0.0045), 0.006)}

    def grasp_view(self, name, side):
        """(rel, ang, kind, size) for the learned grab: the thing's grab point (the point of its long axis
        nearest the open claw's gap) relative to the gap, and its direction, in Pinch's body frame."""
        if name not in self.GRASP_SHAPES or side != "right":
            return None
        kind, size, half = self.GRASP_SHAPES[name]
        m, d = self.m, self.d
        site = m.site("R_grip_site").id
        gap = d.site_xpos[site] + d.site_xmat[site].reshape(3, 3) @ np.array([0.0, -0.006, 0.0])
        b = m.body(name).id
        c, ax = d.xpos[b], d.xmat[b].reshape(3, 3)[:, 0]
        grab = c + np.clip(np.dot(gap - c, ax), -half, half) * ax
        R = d.xmat[m.body("body").id].reshape(3, 3)
        rel = R.T @ (grab - gap) + self.view_rng.normal(0, 0.002, 3)
        ax_b = R.T @ ax
        return rel, float(np.arctan2(ax_b[1], ax_b[0])), kind, np.array(size)

    def gripping(self, name, side):
        """Both claw fingers touch the thing (on the robot: the camera sees it between the fingers)."""
        s = "L" if side == "left" else "R"
        g = self.m.geom(f"{name}_geom").id
        pollex = self.m.geom(f"{s}_pollex").id
        dactyl = self.m.body(f"{s}_dactyl").id
        touch_p = touch_d = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if g not in (c.geom1, c.geom2):
                continue
            o = c.geom2 if c.geom1 == g else c.geom1
            touch_p |= o == pollex
            touch_d |= self.m.geom_bodyid[o] == dactyl
        return touch_p and touch_d

    def perceive(self):
        """What Pinch's camera sees: things within its field of view (with a little noise)."""
        x, y, h = self.pose()
        w = self.mind.world
        w.pose = (x, y, h)
        flaky = "flaky_detection" in self.faults
        for name in (() if "vision" in self.faults else OBJECTS):
            (ox, oy, oyaw), oz = self.obj(name)
            on_desk = oz > -0.02
            bearing = np.arctan2(oy - y, ox - x) - h
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            dist = np.hypot(ox - x, oy - y)
            held = self.body.held and self.body.held[0] == name
            if flaky and self.rng.random() < 0.5:
                continue
            if (abs(bearing) < 0.9 and 0.05 < dist < 0.6) or held or not on_desk:
                noise = self.rng.normal(0, 0.012 if flaky else 0.004, 2)
                w.see(name, (ox + noise[0], oy + noise[1]), oyaw, graspable=OBJECTS[name][4], on_desk=on_desk)
                if not held and self.mind.world.objects[name].still_since < self.t - 30:
                    pass
        if self.user_active:
            w.owner.present = True
            w.owner.last_seen = self.mind.t
        if self.props:
            self.perceive_props()

    def perceive_props(self):
        """What the camera reports of things it can't name (pinch/novelty.py Observation): where (4 mm noise), how
        big, how fast (from successive frames, ~1 cm/s noise) and how sure, within its field of view; the same
        failures as the named things (vision dead, flaky detection). On the robot this comes from the camera."""
        x, y, h = self.pose()
        flaky = "flaky_detection" in self.faults
        for name, (shape, size, _) in self.props.items():
            p = self.prop_xy(name)
            last = self._prop_last.get(name)
            self._prop_last[name] = (p, self.t)
            if p is None or "vision" in self.faults:
                continue
            vel = (0.0, 0.0)
            if last is not None and last[0] is not None and self.t - last[1] > 1e-6:
                vel = ((p[0] - last[0][0]) / (self.t - last[1]), (p[1] - last[0][1]) / (self.t - last[1]))
            bearing = (np.arctan2(p[1] - y, p[0] - x) - h + np.pi) % (2 * np.pi) - np.pi
            dist = float(np.hypot(p[0] - x, p[1] - y))
            if not (abs(bearing) < FOV_HALF and 0.05 < dist < FOV_RANGE):
                continue
            if flaky and self.prop_rng.random() < 0.5:
                continue
            k = 3.0 if flaky else 1.0
            n = self.prop_rng.normal(0, 1, 4)
            conf = float(np.clip(0.95 - 0.8 * max(0.0, dist - 0.25), 0.3, 0.95)) * (0.7 if flaky else 1.0)
            across = 2 * max(size[0], size[1] if shape == "box" else size[0])
            self.mind.observe(Observation("unknown", (p[0] + 0.004 * k * n[0], p[1] + 0.004 * k * n[1]),
                                          (vel[0] + 0.01 * k * n[2], vel[1] + 0.01 * k * n[3]),
                                          across + 0.003 * n[0], conf, f"{shape}:{round(across * 1000)}"))

    def cliff_mm(self):
        """The cliff ring's readings (mm; the rangefinder's -1 = nothing within reach = 9999)."""
        out = {}
        for name, *_ in self.cliff.beams:
            v = float(self.d.sensor(twin_sensor(name)).data[0])
            if self.hw.elec is not None:                        # VL53L0X: noise, 1 mm steps, 30 Hz, dropouts
                v = self.hw.elec.distance.read(name, v, self.d.time)
            out[name] = 9999 if v < 0 else v * 1000
        dead = self.faults.get("cliff_dead")
        if dead is not None:
            out[dead.get("beam", "rear")] = None                # (no answer: the reflex treats it as unknown)
        return out
