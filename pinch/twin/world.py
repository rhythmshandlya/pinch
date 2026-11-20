"""Where things are in the twin: Pinch's pose, the desk things, the claw tip, the feet, and moving things by hand."""

from pinch.twin.common import (  # noqa: F401
    DT,
    JOINT_NAMES,
    OBJECTS,
    PROP_PARK,
    mujoco,
    np,
    object_pose,
)


class TwinWorld:
    def pose(self):
        q = self.d.qpos
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(q[0]), float(q[1]), float(yaw)

    def obj(self, name):
        return object_pose(self.m, self.d, name)

    def in_tray(self, name):
        (x, y, _), z = self.obj(name)
        return self.desk.in_tray((x, y)) and z < 0.03

    GAP = 0.006          # aim a thing this far off the fixed jaw, into the open gap (mid-jaw)

    def tip(self, side, q):
        """Centre of the open claw's gap, in Pinch's body frame, for joint targets q (forward kinematics).
        The fixed finger is on one side of the grip line and the moving finger swings in from the other,
        so a thing to grab has to sit a few mm toward the moving finger."""
        dd = self._tip_d
        dd.qpos[:] = self.m.qpos0
        dd.qpos[3:7] = (1, 0, 0, 0)
        for i, n in enumerate(JOINT_NAMES):
            dd.qpos[self.m.jnt_qposadr[self.m.joint(n).id]] = q[i]
        mujoco.mj_kinematics(self.m, dd)
        s = "L" if side == "left" else "R"
        site = self.m.site(f"{s}_grip_site").id
        p = dd.site_xpos[site] - dd.qpos[:3]
        R = dd.site_xmat[site].reshape(3, 3)
        # the open claw's gap: 4 mm back from the grip line's end, 8 mm off the fixed finger (measured from the
        # finger geometry: at full open the moving finger is 14-20 mm out there, the fixed one 0.5 mm in)
        off = R @ np.array([0.0, (1 if side == "left" else -1) * self.GAP, 0.0])
        return float(p[0] + off[0]), float(p[1] + off[1])

    def feet_down(self, any_surface=False):
        """How many feet touch the desk (the real robot reads this from its leg servos' load). any_surface:
        also count a foot standing on the tray or on a desk thing (it carries weight there too)."""
        if not hasattr(self, "_feet"):
            self._feet = {self.m.geom(f"{lg}_foot").id for lg in ("FL", "FR", "RL", "RR")}
            self._desk = self.m.geom("desk_top").id
            self._robot_root = self.m.body_rootid[self.m.body("body").id]
        n = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for f, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if f not in self._feet:
                    continue
                if o == self._desk or any_surface and self.m.body_rootid[self.m.geom_bodyid[o]] != self._robot_root:
                    n.add(f)
        return len(n)

    def place(self, name, xy, yaw=0.0):
        """Put a desk thing somewhere (you moving the pen while you work)."""
        b = self.m.body(name)
        adr = self.m.jnt_qposadr[b.jntadr[0]]
        z = OBJECTS[name][1][0] + 0.001
        self.d.qpos[adr:adr + 3] = (xy[0], xy[1], z)
        self.d.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))   # lying flat, turned by yaw
        self.d.qvel[self.m.jnt_dofadr[b.jntadr[0]]:self.m.jnt_dofadr[b.jntadr[0]] + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    # -- things Pinch doesn't know (PinchSim(props=[...]); kinematic: they move only when told to) -----------
    def _prop_height(self, name):
        shape, size, _ = self.props[name]
        return size[1] if shape == "cylinder" else size[-1] if shape == "box" else size[0]

    def show_prop(self, name, xy):
        """Put an unknown thing on the desk (you set something down)."""
        self.prop_paths.pop(name, None)
        self.d.mocap_pos[self.m.body(name).mocapid[0]] = (xy[0], xy[1], self._prop_height(name) + 0.0005)

    def hide_prop(self, name):
        """Take it away."""
        self.prop_paths.pop(name, None)
        self.d.mocap_pos[self.m.body(name).mocapid[0]] = PROP_PARK

    def prop_xy(self, name):
        p = self.d.mocap_pos[self.m.body(name).mocapid[0]]
        return (float(p[0]), float(p[1])) if p[2] > -0.1 else None

    def move_prop(self, name, waypoints, speed=0.05):
        """Slide it along waypoints at speed (m/s), from where it is (shown at the first waypoint if hidden)."""
        if self.prop_xy(name) is None:
            self.show_prop(name, waypoints[0])
        self.prop_paths[name] = ([tuple(w) for w in waypoints], float(speed))

    def _advance_props(self):
        for name, (pts, v) in list(self.prop_paths.items()):
            p = np.array(self.prop_xy(name))
            step = v * DT
            while pts and step > 0:
                d = np.array(pts[0]) - p
                n = float(np.linalg.norm(d))
                if n <= step:
                    p, step = np.array(pts[0]), step - n
                    pts.pop(0)
                else:
                    p, step = p + d / n * step, 0.0
            self.d.mocap_pos[self.m.body(name).mocapid[0]][:2] = p
            if not pts:
                del self.prop_paths[name]
