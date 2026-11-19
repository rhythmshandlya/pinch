"""Where things are in the twin: Pinch's pose, the desk things, the claw tip, the feet, and moving things by hand."""

from pinch.twin.common import (  # noqa: F401
    JOINT_NAMES,
    OBJECTS,
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
