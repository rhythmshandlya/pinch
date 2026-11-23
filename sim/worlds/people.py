"""Simulated people: a body Pinch can see (a face, a reaching hand, walking legs) moved by a style of behaviour.

Each person is a set of kinematic (mocap) bodies - torso, head with a face texture, arms, a hand with fingers, legs -
posed every step by a small motion model, the way Habitat 3.0 drives its humanoids kinematically rather than with
physics (Puig et al. 2023, arXiv 2310.13724; their simulated people ranked robot policies like real people did).
People don't collide with anything: when a hand reaches Pinch's shell, the twin is told about the touch (the
gesture follows the style), as the touch pads would report it.

Styles (how people treat a small pet robot; numbers anchored on proxemics studies, the rest are priors):
  indifferent  passes by at a normal walking pace (1.2-1.4 m/s), 0.8-2 m away, glances, rarely reaches out
  curious      comes to 0.4-0.6 m (people stop at 48 +- 11 cm from a small robot, arXiv 2009.01818; Walters et
               al. 2005 ~57 cm), looks at it, reaches slowly (0.15-0.35 m/s) and strokes it, stays 10-60 s
  fearful      stops 0.9-1.5 m away (Hall's social zone), reaches only part way and pulls back fast when Pinch
               moves toward it; people who dislike a robot back away from its gaze (Mumm & Mutlu 2011)
  playful      comes close (0.3-0.5 m), quick taps and tickles (0.5-1 m/s hand, 2-3 Hz), often
Touch kinds follow Yohanan & MacLean 2012 (stroke, pat/tap, poke/boop, tickle).
"""

import numpy as np

STYLE = {     # walk speed m/s, stop distance m, hand speed m/s, dwell s, reaches per minute, touch gestures
    "indifferent": dict(walk=(1.2, 1.4), stop=(0.8, 2.0), hand=(0.2, 0.3), dwell=(0.0, 2.0), reach=0.3,
                        gestures=("tap",), hover=0.06, withdraw=0.4),
    "curious":     dict(walk=(0.6, 1.0), stop=(0.37, 0.59), hand=(0.15, 0.35), dwell=(10.0, 60.0), reach=3.0,
                        gestures=("stroke", "stroke", "tap"), hover=0.02, withdraw=0.3),
    "fearful":     dict(walk=(0.4, 0.8), stop=(0.9, 1.5), hand=(0.1, 0.2), dwell=(3.0, 10.0), reach=1.0,
                        gestures=("boop",), hover=0.22, withdraw=0.9),
    "playful":     dict(walk=(0.8, 1.2), stop=(0.3, 0.5), hand=(0.5, 1.0), dwell=(20.0, 90.0), reach=6.0,
                        gestures=("tap", "tickle", "boop"), hover=0.0, withdraw=0.6),
}
PARTS = ("torso", "head", "uarm_l", "farm_l", "uarm_r", "farm_r", "hand", "thigh_l", "shin_l", "thigh_r", "shin_r")


def dims(p):
    """Segment sizes for a person of height p["height"] (adult proportions; children's heads are bigger)."""
    s = p["height"] / 1.75
    head = 0.115 * s * (1.25 if p["child"] else 1.0)
    return dict(s=s, hip=0.92 * s, thigh=0.45 * s, shin=0.45 * s, torso=0.52 * s, shoulder=0.19 * s,
                uarm=0.30 * s, farm=0.27 * s, head=head, limb_r=0.055 * s, arm_r=0.04 * s)


def add_person(spec, i, p, face_material):
    """Mocap bodies for person i (named person{i}_<part>) in an MjSpec; visual only (they collide with nothing)."""
    import mujoco
    g = mujoco.mjtGeom
    D = dims(p)
    vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
    skin, shirt, trousers = list(p["skin"]) + [1], list(p["shirt"]) + [1], list(p["trousers"]) + [1]
    park = [50.0 + 3 * i, 50.0, -5.0]
    w = spec.worldbody

    def body(part):
        return w.add_body(name=f"person{i}_{part}", mocap=True, pos=park)

    b = body("torso")
    b.add_geom(type=g.mjGEOM_BOX, size=[0.10 * D["s"], 0.17 * D["s"], D["torso"] / 2], rgba=shirt, **vis)
    b = body("head")
    h = D["head"]
    b.add_geom(type=g.mjGEOM_ELLIPSOID, size=[0.85 * h, 0.75 * h, h], rgba=skin, **vis)
    b.add_geom(type=g.mjGEOM_ELLIPSOID, size=[0.88 * h, 0.8 * h, 0.75 * h], pos=[-0.08 * h, 0, 0.3 * h],
               rgba=[0.12, 0.08, 0.05, 1], **vis)                                                      # hair
    b.add_geom(name=f"person{i}_face", type=g.mjGEOM_BOX, size=[0.002, 0.62 * h, 0.78 * h], pos=[0.87 * h, 0, -0.05 * h],
               material=face_material, **vis)
    b.add_geom(type=g.mjGEOM_CAPSULE, size=[0.35 * h, 0.3 * h], pos=[0, 0, -1.1 * h], rgba=skin, **vis)  # neck
    for part, r, L, rgba in (("uarm_l", D["arm_r"], D["uarm"], shirt), ("uarm_r", D["arm_r"], D["uarm"], shirt),
                             ("farm_l", D["arm_r"] * 0.85, D["farm"], skin), ("farm_r", D["arm_r"] * 0.85, D["farm"], skin),
                             ("thigh_l", D["limb_r"], D["thigh"], trousers), ("thigh_r", D["limb_r"], D["thigh"], trousers),
                             ("shin_l", D["limb_r"] * 0.85, D["shin"], trousers), ("shin_r", D["limb_r"] * 0.85, D["shin"], trousers)):
        b = body(part)
        b.add_geom(type=g.mjGEOM_CAPSULE, size=[r, L / 2], rgba=rgba, **vis)
        if part.startswith("shin"):
            b.add_geom(type=g.mjGEOM_BOX, size=[0.12 * D["s"], 0.045 * D["s"], 0.03 * D["s"]],
                       pos=[0.06 * D["s"], 0, -L / 2 - 0.02 * D["s"]], rgba=[0.15, 0.15, 0.18, 1], **vis)   # shoe
    b = body("hand")                                                    # palm in the x-y plane, fingers along +x
    s = D["s"]
    b.add_geom(type=g.mjGEOM_BOX, size=[0.045 * s, 0.042 * s, 0.013 * s], rgba=skin, **vis)
    for k, y in enumerate((-0.03, -0.01, 0.01, 0.03)):
        L = (0.035, 0.042, 0.04, 0.032)[k] * s
        b.add_geom(type=g.mjGEOM_CAPSULE, size=[0.008 * s, L / 2], pos=[0.045 * s + L / 2, y * s, 0],
                   euler=[0, np.pi / 2, 0], rgba=skin, **vis)
    b.add_geom(type=g.mjGEOM_CAPSULE, size=[0.009 * s, 0.018 * s], pos=[0.015 * s, 0.05 * s, 0], euler=[np.pi / 2.5, 0, 0],
               rgba=skin, **vis)


def _quat_z_to(v):
    """Quaternion rotating +z onto v."""
    v = np.asarray(v, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    a = np.cross([0, 0, 1.0], v)
    c = v[2]
    if np.linalg.norm(a) < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0, 1.0, 0, 0])
    a = a / np.linalg.norm(a)
    ang = np.arccos(np.clip(c, -1, 1))
    return np.array([np.cos(ang / 2), *(np.sin(ang / 2) * a)])


def _quat_yaw_pitch(yaw, pitch=0.0, roll=0.0):
    cy, sy, cp, sp, cr, sr = (np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2),
                              np.cos(roll / 2), np.sin(roll / 2))
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])


class Person:
    """One person's motion in a world (surface frame: Pinch's surface top is z = 0, the room floor is z = -height).

    role "sitter": seated at the desk/table front (or on the floor near Pinch), face toward Pinch or the desk,
    reaching to Pinch now and then; role "walker": walks through, and (by style) stops to look and reaches.
    step(t, dt, crab_xyz, crab_vel) poses the bodies and returns a touch gesture when the hand reaches the shell."""

    def __init__(self, i, p, world):
        self.i, self.p, self.w = i, p, world
        self.rng = np.random.default_rng(p["seed"])
        self.st = STYLE[p["style"]]
        self.D = dims(p)
        self.floor_z = -world.surface["height"]
        self.hand_speed = self.rng.uniform(*self.st["hand"])
        self.stop = self.rng.uniform(*self.st["stop"])
        self.walk_speed = self.rng.uniform(*self.st["walk"])
        self.phase = 0.0
        self.hand_state, self.hand_t = "rest", 0.0
        self.next_reach = self.rng.exponential(60.0 / max(self.st["reach"], 1e-3))
        self.touched_at = -99.0
        self.enabled = p["role"] == "sitter"
        self.reach_enabled = True
        self.script = []                                   # scenario overrides: (t, action)
        hy = world.surface["half_y"]
        if p["role"] == "sitter":
            ox, oy = world.owner_xy
            if world.kind == "floor":
                self.base = np.array([ox, oy])
                self.seat = self.floor_z + 0.0                 # sitting on the floor
            elif world.surface["height"] > 0.6:              # at a desk or dining table: knees under it
                self.base = np.array([ox, -hy - 0.38])
                self.seat = self.floor_z + 0.46
            else:                                            # at a coffee table: on a sofa behind it
                self.base = np.array([ox, -hy - 0.62])
                self.seat = self.floor_z + 0.42
            self.heading = None
        else:
            self._new_walk(0.0)
        self.hand = None
        self.pose = {}

    # -- walking paths ------------------------------------------------------------------------------------------
    def _new_walk(self, t):
        """A path past Pinch's surface: from one side of the room to the other, passing at the style's distance
        (indifferent people go by; curious and playful people come up to it; fearful ones keep away)."""
        r = self.rng
        ang = r.uniform(-np.pi, np.pi)
        stop = self.stop if self.p["style"] != "indifferent" else r.uniform(0.8, 2.0)
        side = np.array([np.cos(ang), np.sin(ang)])
        surf = max(self.w.surface["half_x"], self.w.surface["half_y"])
        across = np.array([-side[1], side[0]])
        d0 = surf + stop
        self.path = [side * d0 + across * 3.0, side * d0, side * d0 - across * 3.0]
        self.path_t0, self.dwell = t, r.uniform(*self.st["dwell"])
        self.base = self.path[0].copy()
        self.leg = 0
        self.wait_until = None

    def walk(self, dt, t, crab_xy):
        if self.wait_until is not None:
            if t < self.wait_until:
                if self.p["style"] == "fearful" and np.hypot(*(crab_xy - self.base)) < self.stop * 0.8:
                    self.wait_until = t            # it came toward them: they leave now
                return 0.0
            self.wait_until = None
            self.leg += 1
        if self.leg >= len(self.path) - 1:
            self._new_walk(t)
            self.enabled = self.rng.random() < 0.7
            return 0.0
        tgt = self.path[self.leg + 1]
        d = tgt - self.base
        n = float(np.linalg.norm(d))
        step = self.walk_speed * dt
        if n <= step:
            self.base = tgt.copy()
            if self.leg == 0 and self.p["style"] != "indifferent":
                self.wait_until = t + self.dwell
            else:
                self.leg += 1
            return 0.0
        self.base = self.base + d / n * step
        self.heading = float(np.arctan2(d[1], d[0]))
        return self.walk_speed

    # -- the hand ------------------------------------------------------------------------------------------------
    def _hand_rest(self, face_yaw):
        c, s = np.cos(face_yaw), np.sin(face_yaw)
        if self.p["role"] == "sitter" and self.w.kind != "floor":             # on the desk, in front of the right shoulder
            return np.array([self.base[0] + 0.18 * c + 0.15 * s, self.w.surface["half_y"] * -1 + 0.05, 0.015])
        sh = self._shoulder(face_yaw, right=True)
        return sh + np.array([0.05 * c, 0.05 * s, -self.D["uarm"] - self.D["farm"] + 0.05])

    def _shoulder(self, yaw, right):
        c, s = np.cos(yaw), np.sin(yaw)
        side = -1.0 if right else 1.0
        return np.array([self.base[0] - side * self.D["shoulder"] * s, self.base[1] + side * self.D["shoulder"] * c,
                         self.shoulder_z])

    def reach_now(self, gesture=None):
        """Scenario hook: reach out to Pinch at once (with this gesture, or the style's)."""
        self.hand_state, self.hand_t, self.forced = "reach", 0.0, gesture

    def _hand(self, dt, t, crab, crab_moving_toward, face_yaw):
        rest = self._hand_rest(face_yaw)
        if self.hand is None:
            self.hand = rest.copy()
        target, speed, touch = rest, self.hand_speed, None
        crab_top = np.array([crab[0], crab[1], crab[2] + 0.05])
        reach_d = np.hypot(*(crab[:2] - self.base))
        can_reach = reach_d < self.D["uarm"] + self.D["farm"] + 0.25
        if self.hand_state == "rest":
            self.next_reach -= dt
            if self.next_reach <= 0 and can_reach and self.reach_enabled:
                self.hand_state, self.hand_t, self.forced = "reach", 0.0, None
        if self.hand_state == "reach":
            self.hand_t += dt
            target = crab_top + np.array([0, 0, self.st["hover"]])
            if crab_moving_toward and self.p["style"] == "fearful":
                self.hand_state = "withdraw"
            elif np.linalg.norm(self.hand - target) < 0.012 or self.hand_t > 8.0:
                self.hand_state, self.hand_t = "touch", 0.0
        elif self.hand_state == "touch":
            self.hand_t += dt
            g = self.forced or self.st["gestures"][int(self.rng.integers(len(self.st["gestures"])))]
            self.forced = g
            if self.st["hover"] > 0.05:                  # fearful/indifferent: they don't quite touch
                target = crab_top + np.array([0, 0, self.st["hover"]])
            else:
                wig = 0.03 * np.sin(2 * np.pi * (1.0 if g == "stroke" else 2.5) * self.hand_t)
                target = crab_top + np.array([wig, 0.0, 0.0])
                if self.hand_t > 0.15 and t - self.touched_at > 1.5:
                    touch, self.touched_at = g, t
            if self.hand_t > (2.0 if g == "stroke" else 0.8):
                self.hand_state, self.hand_t = "withdraw", 0.0
        elif self.hand_state == "withdraw":
            speed = self.hand_speed / max(self.st["withdraw"], 0.1) * 0.5
            if np.linalg.norm(self.hand - rest) < 0.02:
                self.hand_state = "rest"
                self.next_reach = self.rng.exponential(60.0 / max(self.st["reach"], 1e-3))
        d = target - self.hand
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            self.hand = self.hand + d / n * min(n, speed * dt)
        return touch

    # -- pose ----------------------------------------------------------------------------------------------------
    def step(self, t, dt, crab_xyz, crab_vel=(0.0, 0.0)):
        crab = np.asarray(crab_xyz, float)
        crab_xy = crab[:2]
        if not self.enabled:
            self.pose = {}
            if self.p["role"] == "walker":
                self.walk(dt, t, crab_xy) if self.rng.random() < dt / 20 else None
                self.enabled = self.enabled or self.rng.random() < dt / 15
            return None
        speed = self.walk(dt, t, crab_xy) if self.p["role"] == "walker" else 0.0
        to_me = self.base - crab_xy
        toward = float(np.dot(crab_vel, to_me / (np.linalg.norm(to_me) + 1e-9))) > 0.02
        look = float(np.arctan2(crab_xy[1] - self.base[1], crab_xy[0] - self.base[0]))
        if self.p["role"] == "walker":
            body_yaw = self.heading if speed > 0 else look
        else:
            body_yaw = look if self.w.kind == "floor" else np.pi / 2 + 0.3 * np.clip(look - np.pi / 2, -1, 1)
        if self.p["style"] == "indifferent" and int(t / 4.0) % 3 != 0:
            look = body_yaw + 0.3 * np.sin(0.4 * t)           # mostly not looking at it
        D = self.D
        sitting = self.p["role"] == "sitter"
        floor_sit = sitting and self.w.kind == "floor"
        hip_z = self.seat + 0.10 * D["s"] if sitting else self.floor_z + D["hip"]
        if floor_sit:
            hip_z = self.floor_z + 0.12 * D["s"]
        self.shoulder_z = hip_z + D["torso"] * 0.95
        P = {}
        P["torso"] = (np.array([*self.base, hip_z + D["torso"] / 2]), _quat_yaw_pitch(body_yaw))
        head_c = np.array([*self.base, hip_z + D["torso"] + 0.04 * D["s"] + D["head"]])
        dz = crab[2] - head_c[2]
        pitch = float(np.clip(-np.arctan2(dz, np.hypot(*(crab_xy - self.base))), -0.2, 0.7))
        P["head"] = (head_c, _quat_yaw_pitch(look, pitch))
        # legs: a sinusoidal walk (~25 deg hip swing) or sitting poses
        self.phase += 2 * np.pi * (speed / (1.2 * D["thigh"] + 1e-9)) * dt * 0.5
        c, s = np.cos(body_yaw), np.sin(body_yaw)
        fwd = np.array([c, s, 0.0])
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            hip = np.array([self.base[0] - sgn * 0.09 * D["s"] * s, self.base[1] + sgn * 0.09 * D["s"] * c, hip_z])
            if floor_sit:
                knee = hip + fwd * D["thigh"] * 0.9 + np.array([-sgn * s, sgn * c, 0]) * 0.15 + np.array([0, 0, 0.1 * D["s"]])
                foot = knee + fwd * -0.1 + np.array([sgn * s, -sgn * c, 0]) * 0.3 * D["s"] + np.array([0, 0, -0.1 * D["s"]])
            elif sitting:
                knee = hip + fwd * D["thigh"]
                foot = np.array([knee[0], knee[1], self.floor_z + 0.08 * D["s"]])
            else:
                a = 0.45 * np.sin(self.phase + (0 if sgn > 0 else np.pi)) if speed > 0 else 0.0
                bend = 0.5 * max(0.0, np.sin(self.phase + (0 if sgn > 0 else np.pi) + 1.2)) if speed > 0 else 0.05
                thigh_dir = np.array([np.sin(a) * c, np.sin(a) * s, -np.cos(a)])
                knee = hip + thigh_dir * D["thigh"]
                b = a - bend
                foot = knee + np.array([np.sin(b) * c, np.sin(b) * s, -np.cos(b)]) * D["shin"]
            P["thigh_" + side] = ((hip + knee) / 2, _quat_z_to(knee - hip))
            P["shin_" + side] = ((knee + foot) / 2, _quat_z_to(foot - knee))
        touch = self._hand(dt, t, crab, toward, body_yaw) if self.p["role"] == "sitter" or self.wait_until else None
        if self.hand is None or (self.p["role"] == "walker" and not self.wait_until):
            self.hand = self._hand_rest(body_yaw)
        for side, right in (("l", False), ("r", True)):
            sh = self._shoulder(body_yaw, right)
            if right:
                wrist = self.hand - np.array([0.05 * c, 0.05 * s, 0.0])
            else:
                swing = 0.35 * np.sin(self.phase) if speed > 0 else 0.0
                wrist = sh + np.array([np.sin(swing) * c * D["farm"], np.sin(swing) * s * D["farm"],
                                       -D["uarm"] - D["farm"] * np.cos(swing)])
                if sitting and not floor_sit:                   # the other hand rests on the desk, left of it
                    wrist = np.array([self.base[0] + 0.18 * c - 0.15 * s, -self.w.surface["half_y"] + 0.04, 0.03])
            v = wrist - sh
            L = np.linalg.norm(v)
            reach = D["uarm"] + D["farm"]
            u = v / (L + 1e-9)
            if L >= reach * 0.999:
                elbow = sh + u * D["uarm"]
            else:                                         # two-link: elbow out and down
                a = (D["uarm"] ** 2 - D["farm"] ** 2 + L ** 2) / (2 * L)
                hgt = np.sqrt(max(D["uarm"] ** 2 - a ** 2, 0.0))
                out = np.array([s, -c, 0.0]) if right else np.array([-s, c, 0.0])    # away from the body
                bend = 0.5 * out + np.array([0, 0, -0.85])                             # and down
                bend = bend - u * (bend @ u)
                elbow = sh + u * a + bend / (np.linalg.norm(bend) + 1e-9) * hgt
            P["uarm_" + side] = ((sh + elbow) / 2, _quat_z_to(elbow - sh))
            P["farm_" + side] = ((elbow + wrist) / 2, _quat_z_to(wrist - elbow))
        hand_yaw = float(np.arctan2(self.hand[1] - self._shoulder(body_yaw, True)[1],
                                    self.hand[0] - self._shoulder(body_yaw, True)[0]))
        P["hand"] = (self.hand.copy(), _quat_yaw_pitch(hand_yaw, 0.25))
        self.pose = P
        return touch

    def apply(self, m, d):
        """Write the pose into the model's mocap bodies (parked far away when the person is not around)."""
        for part in PARTS:
            mid = m.body(f"person{self.i}_{part}").mocapid[0]
            if part in self.pose:
                d.mocap_pos[mid], d.mocap_quat[mid] = self.pose[part]
            else:
                d.mocap_pos[mid] = (50.0 + 3 * self.i, 50.0, -5.0)

    def face_xyz(self):
        return self.pose["head"][0] if "head" in self.pose else None
