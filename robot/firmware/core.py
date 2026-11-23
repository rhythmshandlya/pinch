"""The firmware's frame: start-up, the 50 Hz loop, emergency stop, saving, status."""

from pinch import runtime
from robot.firmware.common import (  # noqa: F401
    DT,
    SAVE_EVERY,
    AudioDirection,
    Body,
    BodySense,
    Calibration,
    CliffReflex,
    Counter,
    Crab,
    Desk,
    Face,
    Hearing,
    Memory,
    Mind,
    Path,
    SafetyStop,
    StuckSense,
    TouchGestures,
    cliff_beams,
    deque,
    json,
    load_policies,
    log,
    np,
    queue,
    signal,
    threading,
    time,
)
from robot.localizer import Localizer  # noqa: F401


class FirmwareCore:
    def __init__(self, hw, llm=True, memory_path=None, record_dir=None, mind_seed=None, clock=None):
        self.hw = hw
        self.desk = hw.desk if hasattr(hw, "desk") else Desk()
        self.crab = Crab(hw.backend, calibration=hw.calibration if hasattr(hw, "calibration") else Calibration.load())
        # memory_path: None = the default file (PINCH_MEMORY), False = don't keep a memory file (tests)
        memory = Memory() if memory_path is None else Memory(path=memory_path or None)
        # a fresh seed each boot, so its days don't repeat (audit LIFE-008); written to the diary so a day can be
        # replayed; tests pass mind_seed for a fixed run
        if mind_seed is None:
            import secrets
            mind_seed = secrets.randbits(31)
        self.mind_seed = mind_seed
        self.mind = Mind(memory=memory, llm=llm, seed=mind_seed, clock=clock)   # clock: wall time unless a test fixes it
        self.mind._say("boot", f"mind seed {mind_seed}")
        if getattr(memory, "path", None):                    # why it did what it did, next to its memory
            self.mind.decision_log_path = str(Path(memory.path).parent / "decisions.jsonl")
        self.face = Face()
        self.loc = Localizer(getattr(hw, "start_pose", (0.0, 0.0, 0.0)))
        pols = load_policies()
        self.body = Body(self.crab, self.loc.pose, self.object_belief, self.desk, tip_fn=hw.tip, emote_policy=pols["emote"],
                         walk_policy=pols["walk"], push_policy=pols["push"], grasp_policy=pols["grasp"],
                         righting_policy=pols["righting"], life_seed=None)   # (the robot: fresh every boot)
        self.body.pose_sigma_fn = self.loc.sigma
        self.body.bystanders_fn = lambda: self.mind.world.things_on_desk() + self.mind.unknown_things()
        self.gestures = TouchGestures()
        self.bodysense = BodySense()
        self.stuck_sense = StuckSense()
        # the ring the robot is built with (crab/params.py CLIFF_BEAMS): one that didn't come up stays in the
        # list as "don't know"; emulated hardware may model fewer (hw.cliff_beams)
        names = getattr(hw, "cliff_beams", None)
        beams = cliff_beams()
        self.cliff_reflex = CliffReflex([b for b in beams if names is None or b[0] in names])
        self.last_recover = -99.0
        self.hearing = Hearing()
        self.direction = AudioDirection()
        self.inbox = queue.Queue()           # things from the HTTP API / threads, applied in the loop
        self.cliff = None                    # {beam: mm or None (no answer)}, from the cliff thread
        self.cliff_at = {}                   # beam -> when it last answered (self._now() clock)
        self.face_frame = None
        self.running = True
        self.autonomous = True               # False: the mind doesn't move the body (tests drive it)
        self.t = 0.0
        self.k = 0
        self.stats = dict(oled_frames=0, sounds=[], motions=[], vision_fixes=0, flow_updates=0, things_seen=0,
                          touches=0, errors=0, edge_stops=0)
        self.errors = Counter()              # where the transient errors came from
        self.tick_ms = deque(maxlen=500)     # how long each tick's own work took (not the sleep)
        self.last_save = 0.0
        self._saving = None
        self.vision = self.flow = None
        self.faceid = None
        # record_dir: keep a session recording there (pinch/recorder.py): the timeline and 5 Hz samples, no video
        self.recorder, self._grav = None, None
        if record_dir:
            from pinch.recorder import SessionRecorder, run_header
            from robot.crab_robot import MAX_STEP
            self.recorder = SessionRecorder(record_dir, header=run_header(where="robot", hw=type(hw).__name__),
                                            sample_every=10, state=False, dt=DT, max_step=MAX_STEP)
        if hw.camera is not None:
            from robot.vision import GroundFlow, Vision
            self.vision = Vision(hw.camera_model, self.desk)
            from pinch.faces import FaceID  # who is that (decision 57); without the models: no one
            if FaceID.available():
                mp = getattr(self.mind.memory, "path", None)
                self.faceid = FaceID(gallery_path=(Path(mp).parent / "faces.json") if mp else None)
            self.flow = GroundFlow(hw.camera_model)
            if not hw.camera_model.calibrated:
                log.warning("camera lens not calibrated (robot/calibrate_camera.py): it won't be used to "
                            "tell where Pinch is - only for things and faces")
        missing = [b[0] for b in self.cliff_reflex.beams if not hw.cliff or hw.cliff.get(b[0]) is None]
        if missing:
            log.warning("cliff sensors missing: %s - Pinch won't walk toward those sides", missing)

    # -- what the world tells us ---------------------------------------------------------------
    def object_belief(self, name):
        ob = self.mind.world.objects.get(name)
        if ob is None:
            return None
        return (ob.xy[0], ob.xy[1], ob.yaw), 0.0045

    def say(self, text):
        self.inbox.put(("say", text))

    def touch(self, gesture):
        self.inbox.put(("touch", gesture))

    def forget(self, what):
        """Make Pinch forget something (pinch/mind/privacy.py FORGETTABLE, or "person:<id>"); applied in the loop."""
        self.inbox.put(("forget", what))

    def approve_facts(self, which=None, keep=True):
        """The person's yes or no to facts it heard about them (pinch/memory.py approve_facts); applied in the loop."""
        self.inbox.put(("approve", (which, keep)))

    def set_face_recognition(self, on):
        self.inbox.put(("face_recognition", bool(on)))

    def set_quiet_hours(self, hours):
        """(start, end) local hours, or None to clear them (pinch/routine.py); applied in the loop."""
        self.inbox.put(("quiet_hours", hours))

    def _now(self):
        return time.monotonic() if getattr(self.hw, "threaded", True) else self.t

    def _error(self, source, e=None):
        self.stats["errors"] += 1
        self.errors[source] += 1
        if self.errors[source] in (1, 10, 100, 1000):
            log.warning("%s: transient error #%d (%s) - carrying on", source, self.errors[source], e)

    def _safe(self, source, fn, default):
        """A sensor read that may hiccup (I2C NACK, a servo reply lost): count it, use `default`."""
        try:
            return fn()
        except OSError as e:
            self._error(source, e)
            return default

    def _play(self, s):
        self.hw.play(s)
        self.stats["sounds"].append(s)
        if self.recorder is not None:
            self.recorder.event(self.t, "sound", name=s, activity=self.mind.activity.name if self.mind.activity else None)

    def estop(self, why="user"):
        """Emergency stop (HTTP /stop, or "freeze" said aloud): cancel what the body is doing, go limp, and do
        nothing that moves until resume(). Senses, memory and the face keep working."""
        self.body.cancel(force=True)
        self.crab.pause(why)
        self.mind.estopped = True
        self.mind.motion_cancelled()                          # (the step it was waiting on did not happen)
        self.stats["estops"] = self.stats.get("estops", 0) + 1

    def resume(self):
        if self.crab.resume():
            self.mind.estopped = False
            self.body.start(("pose", "stand", 1.0))           # ease back up from wherever it settled

    def step(self):
        while not self.inbox.empty():
            self._apply(*self.inbox.get())
        if self.mind.estop_request:                           # "emergency stop" said aloud
            self.mind.estop_request = False
            self.estop("voice")
        if not getattr(self.hw, "threaded", True):
            self._poll()
        t_work = time.perf_counter()
        if self.k % 5 == 0:
            m = self.mind
            m.world.pose = self.loc.pose()
            m.world.battery = self._safe("battery", self.hw.battery_fraction, m.world.battery)
            m.world.servo_temp = self.body.servo_temp = self.crab.hottest_leg_c    # the servos' present_temperature
            if self.autonomous:
                runtime.mind_step(m, self.body, DT * 5, self.t, self.recorder,
                                  on_motion=lambda mo: self.stats["motions"].append(
                                      (round(self.t, 1), mo[0], mo[1] if len(mo) > 1 else "")),
                                  on_sound=self._play)
            # reflex: a beam sees no desk while walking toward its side -> back off away from it (latched,
            # so its own backing-off can't re-trigger it); a beam missing or stale -> no walking that way
            now = self._now()
            ages = {n: now - t for n, t in self.cliff_at.items()}
            back = self.cliff_reflex.update(self.t, self.cliff, ages, self.body.cmd_vel)
            self.body.cliff_blocked = self.cliff_reflex.blocked
            if back is not None:
                self.body.reflex(back)
                self.stats["edge_stops"] += 1
                if self.recorder is not None:
                    self.recorder.step_started(self.t, back, "cliff reflex")
            gyro, grav = self.hw.backend.read_imu()          # the backend rides out IMU hiccups itself
            self._grav = grav
            feet = self._safe("feet", lambda: self.hw.feet_down(self.crab.state), None)
            for ev in self.bodysense.update(DT * 5, gyro, grav, feet_loaded=feet):
                m.touch(ev)
            walking = self.body.busy and self.body.status in ("goto", "walk", "push", "recover", "turn_to")
            x, y, h = self.loc.pose()
            margin = self.desk.feet_margin((x, y), h)
            # reflex: a foot over the desk edge (from the marker position + leg layout) -> shuffle back in
            self.last_recover = runtime.foot_over_edge(self.body, margin, self.bodysense.lifted, self.t,
                                                       self.last_recover, self.recorder)
            # stuck: judged on what the camera measured, never on the integrated command
            for ev in self.stuck_sense.update(DT * 5, grav, feet, self.loc.measured_xy(self.t),
                                              float(np.hypot(*self.body.cmd_vel[:2])),
                                              lifted=self.bodysense.lifted, standing=not self.body.busy,
                                              walking=walking, feet_margin=margin):
                m.touch(ev)
                if self.recorder is not None:
                    self.recorder.event(self.t, "body_sense", sense=ev, why=self.stuck_sense.why)
            self.loc.frozen = self.stuck_sense.stuck
        if self.k % 2 == 0 and self.hw.touch is not None:
            try:
                for g in self.gestures.update(self.t, self.hw.touch.touched()):
                    self.stats["touches"] += 1
                    self.mind.touch(g)
            except OSError as e:
                self._error("touch", e)
        if self.k % 3 == 0:
            self.face.update(DT * 3, self.mind.face_expr, self.mind.gaze)
            self.face_frame = self.face.ssd1309_pages()
        if self.t - self.last_save >= SAVE_EVERY:
            self.save_memory()
        runtime.sync(self.mind, self.body)                 # the mind's state that styles the body, and back
        q = self.body.tick()                                 # 20 ms: servos (and physics when emulated)
        if self.recorder is not None:
            self.recorder.tick(self.t + DT, self.k, self.mind, self.body, q, self.crab.q_cmd, self._rec_sample)
        cmd = getattr(self.body, "cmd_vel", (0.0, 0.0, 0.0))
        self.loc.predict(cmd, DT, self.t)
        # the tick's own work: everything above except the wait for the next tick (a sleep on the robot, the
        # physics in the twin); on the robot the senses' threads run beside it
        self.tick_ms.append(1000 * (time.perf_counter() - t_work - self.crab.last_wait))
        self.t += DT
        self.k += 1
        return q

    def save_memory(self, wait=False):
        """Snapshot in the loop (fast), write + fsync to the SD card in a thread (can take 50+ ms)."""
        self.last_save = self.t
        if self.faceid is not None:
            try:
                self.faceid.save()
            except OSError as e:
                self._error("faces", e)
        text = self.mind.memory.snapshot()
        if text is None:
            return
        if self._saving is not None and self._saving.is_alive():
            self._saving.join()

        def write():
            try:
                self.mind.memory.write(text)
            except OSError as e:
                self._error("memory", e)
        self._saving = threading.Thread(target=write, daemon=True)
        self._saving.start()
        if wait:
            self._saving.join()

    def _on_sigterm(self, signum, frame):
        log.info("SIGTERM: sitting down, servos off, saving memory")
        self.running = False

    def run(self, seconds=None, realtime=True):
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._on_sigterm)
        if getattr(self.hw, "threaded", True):
            for f in (self._face_thread, self._cliff_thread, self._vision_thread, self._audio_thread):
                threading.Thread(target=f, daemon=True).start()
        t0 = time.perf_counter()
        last_report = 0.0
        try:
            self.crab.start()
            while self.running and (seconds is None or self.t < seconds):
                self.step()
                if realtime and getattr(self.hw, "realtime", True):
                    lag = t0 + self.t - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.5:                          # badly behind: don't try to catch up in a burst
                        t0 -= lag
                if self.t - last_report >= 60.0:
                    last_report = self.t
                    ms = self.timing()
                    if ms["p95"] > 0.75 * DT * 1000:
                        log.warning("control loop slow: %s ms per tick (budget %d)", ms, DT * 1000)
                    cpu = self._cpu_temp()
                    if cpu is not None:
                        (log.warning if cpu > 80 else log.info)("CPU %.0f C", cpu)
        except SafetyStop as e:
            log.error("safety stop: %s", e)
            self.stop_kind = e.kind
            self.face.update(0.1, "sleep" if e.kind == "battery" else "dizzy")
            if self.hw.oled is not None:
                try:
                    self.hw.oled.show(self.face.ssd1309_pages())
                except OSError:
                    pass
        finally:
            self.running = False
            try:
                self.crab.shutdown()                           # sits down unless a safety stop happened
            except Exception as e:                             # noqa: BLE001 - still save the memory
                log.error("shutdown: %s", e)
            self.save_memory(wait=True)
            if self.recorder is not None:
                try:
                    self.recorder.close(self.t, stop=self.stop_kind)
                except Exception as e:                         # noqa: BLE001 - a recording must not stop a shutdown
                    log.error("recording: %s", e)
            if self.stop_kind == "battery" and hasattr(self.hw, "poweroff"):
                # empty pack: say so, then power the computer down instead of draining the cells further
                log.error("battery empty: shutting the computer down")
                self.hw.play("low_battery")
                time.sleep(1.5 if getattr(self.hw, "realtime", True) else 0.0)
                self.hw.poweroff()

    stop_kind = None

    def _rec_sample(self):
        g = self._grav
        return dict(pose=self.loc.pose(), up=float(-g[2]) if g is not None else float("nan"),
                    eye=float(self.face.gaze[0]), blink=bool(self.face.blink_t >= 0))

    def _cpu_temp(self):
        return self.hw.cpu_temp() if hasattr(self.hw, "cpu_temp") else None

    def timing(self):
        a = np.array(self.tick_ms) if self.tick_ms else np.zeros(1)
        return dict(mean=round(float(a.mean()), 2), p95=round(float(np.percentile(a, 95)), 2),
                    max=round(float(a.max()), 2))

    def status(self):
        m = self.mind
        return dict(feeling=m.affect.feeling(), activity=m.activity.name if m.activity else None,
                    needs=m.affect.needs.as_dict(), valence=round(m.affect.valence, 3), owner=m.memory.owner_name,
                    busy=m.world.owner.busy, pose=[round(v, 3) for v in self.loc.pose()], t=round(self.t, 1),
                    pose_sigma_m=round(self.loc.sigma(), 3), stuck=self.stuck_sense.stuck,
                    battery=round(self._safe("battery", self.hw.battery_fraction, 0.0), 2), cliff_mm=self.cliff,
                    cliff=dict(self.cliff_reflex.states), tick_ms=self.timing(), cpu_c=self._cpu_temp(),
                    errors=dict(self.errors) | dict(getattr(self.hw.backend, "errors", {})),
                    diary=[f"{t:.0f}s {k}: {x}" for t, k, x in m.log[-8:]],
                    quiet_hours=m.routine.settings.get("quiet_hours"), quiet=m.affect.quiet,
                    part_of_day=m.routine.part_of_day(m.world.clock()))
