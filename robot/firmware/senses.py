"""The senses: face, cliff, vision and audio threads, the emulated poll, and how their results reach the mind."""

from robot.firmware.common import (  # noqa: F401
    DIR_N,
    time,
)


class FirmwareSenses:
    # -- threads ---------------------------------------------------------------------------------
    def _face_thread(self):
        while self.running:
            frame = self.face_frame
            if frame is not None and self.hw.oled is not None:
                try:
                    self.hw.oled.show(frame)
                    self.stats["oled_frames"] += 1
                except OSError as e:
                    self._error("face", e)
            time.sleep(0.06)                                 # ~15 frames a second is plenty for eyes

    def _read_cliff(self):
        from robot.drivers.i2c_devices import read_ring
        vals = read_ring(self.hw.cliff)                      # all beams measure at once
        now = self._now()
        for name, v in vals.items():
            if v is None:
                if self.hw.cliff.get(name) is not None:
                    self._error("cliff", f"{name} didn't answer")
            else:
                self.cliff_at[name] = now
        self.cliff = vals

    def _cliff_thread(self):
        while self.running and self.hw.cliff is not None:
            self._read_cliff()
            time.sleep(0.03)

    def see(self, bgr, t):
        """All the camera work for one frame (in the vision thread on the robot): returns the answers only."""
        out = dict(t=t, pose=None, flow=None, things={}, face=False)
        out["pose"] = self.vision.locate(bgr)
        if self.flow is not None:
            prev_t = self.flow.prev_t
            v = self.flow.update(bgr, t)
            if v is not None:
                out["flow"] = (v, t - prev_t)
        out["things"] = self.vision.things(bgr, out["pose"] or self.loc.pose())
        out["face"], _ = self.vision.face(bgr)
        if self.faceid is not None and self.mind.face_recognition:     # (the owner can turn it off: decision 83)
            out["people"] = self.faceid.see(bgr, t)
            out["face"] = out["face"] or bool(out["people"])
        return out

    def _vision_thread(self):
        while self.running and self.vision is not None:
            t0 = time.monotonic()
            bgr = self.hw.camera()
            if bgr is not None:
                try:
                    self.inbox.put(("vision", self.see(bgr, self.t)))
                except Exception as e:                       # noqa: BLE001 - one bad frame must not blind Pinch
                    self._error("vision", e)
            time.sleep(max(0.0, 0.2 - (time.monotonic() - t0)))

    def _audio_thread(self):
        while self.running and self.hw.audio is not None:
            try:
                chunk = self.hw.audio()                       # (n, 2) stereo float, 16 kHz
            except OSError as e:
                self._error("audio", e)
                chunk = None
            if chunk is None:
                time.sleep(0.05)
                continue
            mono = chunk.mean(axis=1)
            for s in self.hearing.feed(mono):
                ang = self.direction.angle(chunk[-DIR_N:, 0], chunk[-DIR_N:, 1]) if len(chunk) >= DIR_N else None
                self.inbox.put(("voice", (s, ang)))
            for ev, _ in self.hearing.take_events():
                self.inbox.put(("sound_event", ev))

    def _apply(self, kind, data):
        m = self.mind
        if kind == "say":
            m.hear_text(data)
        elif kind == "touch":
            self.stats["touches"] += 1
            m.touch(data)
        elif kind == "forget":
            try:
                faces = m.forget(data)
            except (KeyError, ValueError) as e:
                self._error("forget", e)
                return
            if self.faceid is not None and faces:            # the fingerprints go too, and at once
                for pid in faces:
                    self.faceid.forget(pid)
                self.faceid.save()
        elif kind == "face_recognition":
            m.set_face_recognition(data)
        elif kind == "quiet_hours":
            if data is None:
                m.routine.clear_quiet_hours()
            else:
                m.routine.set_quiet_hours(*data)
            m.memory.changed()
        elif kind == "voice":
            m.hear_summary(*data)
        elif kind == "sound_event":
            m.hear_event(data)
        elif kind == "vision":
            if data["pose"] is not None:
                self.loc.fix(data["pose"], self.t)
                self.stats["vision_fixes"] += 1
            if data["flow"] is not None:
                if self.loc.flow(data["flow"][0], self.t, data["flow"][1]):
                    self.stats["flow_updates"] += 1
                else:
                    self.stats["flow_rejected"] = self.stats.get("flow_rejected", 0) + 1
            for name, xy in data["things"].items():
                if self.desk.inside(xy, -0.01):
                    m.world.see(name, xy, graspable=False)
                    self.stats["things_seen"] += 1
            if data["face"]:
                m.world.owner.present, m.world.owner.last_seen = True, m.t
            for obs in data.get("observations", ()):        # unknown things and motion (pinch/novelty.py), once
                m.observe(obs)                               # the vision pipeline reports them
            for pid, conf, _box in data.get("people", ()):
                if pid is not None:
                    m.see_person(pid, conf)

    # -- the loop --------------------------------------------------------------------------------
    def _poll(self):
        """Emulated hardware: read the senses inside the loop (deterministic) instead of in threads."""
        hw = self.hw
        if self.k % 3 == 0 and self.face_frame is not None and hw.oled is not None:
            hw.oled.show(self.face_frame)
            self.stats["oled_frames"] += 1
        if self.k % 5 == 0 and hw.cliff is not None:
            hw.update_cliff()
            self._read_cliff()
        if self.k % 10 == 0 and self.vision is not None:
            self._apply("vision", self.see(hw.camera(), self.t))
        if self.k % 5 == 0 and hw.audio is not None:
            chunk = hw.audio()
            for s in self.hearing.feed(chunk.mean(axis=1)):
                self._apply("voice", (s, self.direction.angle(chunk[-DIR_N:, 0], chunk[-DIR_N:, 1])))
            for ev, _ in self.hearing.take_events():
                self._apply("sound_event", ev)
