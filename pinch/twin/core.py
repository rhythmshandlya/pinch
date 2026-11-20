"""The twin's frame: building the world, robot, body and mind, and the 50 Hz loop."""

from pinch.twin.common import (  # noqa: F401
    DT,
    FACE_EVERY,
    MIND_EVERY,
    OBJECTS,
    PROP_PARK,
    PROPS,
    SIM_EPOCH,
    SKILLS,
    START,
    TWIN_UTC_OFFSET_H,
    Body,
    BodySense,
    Calibration,
    CliffReflex,
    Crab,
    Desk,
    Face,
    Hearing,
    Memory,
    Mind,
    SimBackend,
    StuckSense,
    build_desk_model,
    load_policies,
    mujoco,
    np,
    set_face,
)


class TwinCore:
    def __init__(self, memory=None, llm=False, seed=0, start=None, render=False, policies=True, extras=(),
                 epoch=SIM_EPOCH, decision_log=None, props=(), record_dir=None, record=None):
        """epoch: the wall-clock time the run starts at (what the mind's clock reads at t = 0).
        decision_log: a file to write every activity switch and its reason to (JSON lines).
        props: things Pinch doesn't know (names from sim.desk.PROPS), parked off the desk until shown.
        record_dir: record the session there (pinch/recorder.py; off by default, and it changes nothing);
        record: options for the recorder (header=dict(scenario=...), state=False, live_every=60 ...)."""
        self.t = 0.0
        self.faults = {}                                  # injected failures (inject())
        self.desk = Desk()
        self.props = {p: PROPS[p] for p in props}
        self.prop_paths, self._prop_last = {}, {}
        prop_extras = [dict(name=p, type=sh, size=sz, rgba=c, pos=list(PROP_PARK), mocap=True)
                       for p, (sh, sz, c) in self.props.items()]
        self.m = build_desk_model(self.desk, start=start or START, extras=tuple(extras) + tuple(prop_extras))
        self.hw = SimBackend(model=self.m, render=render)
        self.d = self.hw.d
        self.crab = Crab(self.hw, calibration=Calibration())
        self.crab.start()                                  # ease into the stand pose (2 s)
        self.mind = Mind(memory=memory if memory is not None else Memory(path=None), llm=llm, seed=seed,
                         clock=lambda: epoch + self.t)
        self.mind.decision_log_path = decision_log
        if self.mind.routine.settings.get("utc_offset_h") is None:
            self.mind.routine.settings["utc_offset_h"] = TWIN_UTC_OFFSET_H
        self.face = Face(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.senses = BodySense()
        self.stuck_sense = StuckSense()
        self.cliff = CliffReflex()
        self.stuck_events = []            # (t, "stuck"/"unstuck", why)
        self.last_recover, self.edge_recovers = -99.0, 0
        self.hearing = Hearing()
        self._tip_d = mujoco.MjData(self.m)
        pols = load_policies() if policies else dict.fromkeys(SKILLS)
        self.body = Body(self.crab, self.pose, self.obj, self.desk, tip_fn=self.tip, emote_policy=pols["emote"],
                         walk_policy=pols["walk"], push_policy=pols["push"], grasp_policy=pols["grasp"],
                         righting_policy=pols["righting"], life_seed=seed)
        self.body.grip_fn = self.gripping
        self.body.bystanders_fn = lambda: self.mind.world.things_on_desk() + self.mind.unknown_things()
        self.body.grasp_view_fn = self.grasp_view
        self.view_rng = np.random.default_rng(seed + 7)
        self.prop_rng = np.random.default_rng(seed + 11)     # the props' sightings (own stream: replay unchanged)
        self.k = 0
        self.t = 0.0
        self.sounds_played = []           # (t, name)
        self.motions = []                 # (t, step)
        self.falls = 0
        self.edge_stops = 0
        for name in OBJECTS:              # Pinch knows its desk things where they start (homes are learned)
            self.mind.world.see(name, self.obj(name)[0][:2], graspable=OBJECTS[name][4])
        self.mind.memory.teach_home("pen", self.desk.tray_xy)
        self.mind.memory.teach_home("eraser", (self.desk.tray_xy[0] - 0.04, self.desk.tray_xy[1]))
        self.user_active = False          # the app sets this while you interact
        self.autonomous = True            # False: the mind doesn't move the body (skill tests)
        if record_dir is not None:
            self.start_recording(record_dir, seed=seed, props=list(props), epoch=epoch, **(record or {}))

    # -- recording (opt-in: pinch/recorder.py) ---------------------------------------------------------------------
    recorder = None

    def start_recording(self, path, header=None, seed=None, **config):
        """Record from now on into the directory path (header: more to put in header.json)."""
        from pinch.recorder import SessionRecorder, run_header
        from robot.crab_robot import MAX_STEP
        opts = {k: config.pop(k) for k in ("sample_every", "state", "live_every", "chunk") if k in config}
        self.recorder = SessionRecorder(path, header=run_header(seed=seed, **dict(config, **(header or {}))),
                                        dt=DT, max_step=MAX_STEP, **opts)
        if self.recorder.state:
            mujoco.mj_saveModel(self.m, str(self.recorder.dir / "model.mjb"), None)
        self._qadr = [(self.hw.qadr[j], self.hw.bl_qadr.get(j)) for j in self.hw.qadr]
        return self.recorder

    def stop_recording(self, analyse=True):
        """Close the recording (flush, observer, report). Returns the report, or None."""
        rec, self.recorder = self.recorder, None
        return rec.close(self.t, analyse=analyse) if rec is not None else None

    def _rec(self, kind, **data):
        if self.recorder is not None:
            self.recorder.event(self.t, kind, **data)

    def _rec_sample(self):
        q = self.d.qpos
        R = self.d.xmat[1].reshape(3, 3)
        out = dict(pose=self.pose(), z=float(q[2]), up=float(R[2, 2]),
                   q_meas=[q[a] + (q[b] if b is not None else 0.0) for a, b in self._qadr])
        if self.recorder.state:
            out["qpos"] = q
            if self.m.nmocap:
                out["mocap"] = self.d.mocap_pos
        return out

    def step(self):
        if self.prop_paths:
            self._advance_props()
        if self.k % MIND_EVERY == 0 and not self.autonomous:
            self.perceive()
        elif self.k % MIND_EVERY == 0:
            self.perceive()
            motion, cancel = self.mind.tick(DT * MIND_EVERY, self.body.busy, self.body.ok)
            if cancel:
                self.body.cancel()
                if self.recorder is not None:
                    self.recorder.step_cancelled(self.t, "mind")
            if motion is not None:
                self.body.start(motion)
                self.motions.append((round(self.t, 2), motion))
                if self.recorder is not None:
                    self.recorder.step_started(self.t, motion, "mind")
            for s in self.mind.outputs()[2]:
                self.sounds_played.append((round(self.t, 2), s))
                self._rec("sound", name=s, activity=self.mind.activity.name if self.mind.activity else None)
            # reflex: a cliff beam sees no desk while walking toward its side -> back off (latched)
            back = self.cliff.update(self.t, self.cliff_mm(), 0.0, self.body.cmd_vel)
            self.body.cliff_blocked = self.cliff.blocked
            if back is not None:
                self.body.reflex(back)
                self.edge_stops += 1
                if self.recorder is not None:
                    self.recorder.step_started(self.t, back, "cliff reflex")
            gyro, grav = self.hw.read_imu()
            for ev in self.senses.update(DT * MIND_EVERY, gyro, grav, feet_loaded=self.feet_down()):
                self.mind.touch(ev)
            walking = self.body.busy and self.body.status in ("goto", "walk", "push", "recover", "turn_to")
            x, y, h = self.pose()
            margin = self.desk.feet_margin((x, y), h)
            # reflex: a foot over the desk edge -> drop everything and shuffle back in (before anything else)
            if margin < -0.003 and not self.senses.lifted and self.body.status != "recover" \
                    and self.t - self.last_recover > 3.0:
                self.body.reflex(("recover",))
                self.last_recover = self.t
                self.edge_recovers += 1
                if self.recorder is not None:
                    self.recorder.step_started(self.t, ("recover",), "edge reflex")
            for ev in self.stuck_sense.update(DT * MIND_EVERY, grav, self.feet_down(any_surface=True), (x, y),
                                              float(np.hypot(*self.body.cmd_vel[:2])), lifted=self.senses.lifted,
                                              standing=not self.body.busy, walking=walking, feet_margin=margin):
                self.mind.touch(ev)
                self.stuck_events.append((round(self.t, 1), ev, self.stuck_sense.why))
                self._rec("body_sense", sense=ev, why=self.stuck_sense.why)
            if grav[2] > -0.5:
                self.falls += 1
        if self.k % FACE_EVERY == 0:
            expr, gaze, _ = self.mind.face_expr, self.mind.gaze, None
            self.face.update(DT * FACE_EVERY, expr, gaze)
            set_face(self.m, self.face.rgb(scale=2))
        self.body.life.liveliness = self.mind.affect.arousal
        if self.body.mood_fn is None:          # the emote style follows the whole mood (decision 92)
            a = self.mind.affect
            self.body.mood_fn = lambda: dict(arousal=a.arousal, valence=a.valence, sleepy=a.needs.sleep)
        q_req = self.body.tick()                           # 20 ms of physics through the robot runtime
        if self.recorder is not None:
            self.recorder.tick(self.t + DT, self.k, self.mind, self.body, q_req, self.crab.q_cmd, self._rec_sample)
        self.k += 1
        self.t += DT

    def run(self, seconds, every=None):
        for _ in range(int(seconds / DT)):
            self.step()
            if every:
                every(self)
