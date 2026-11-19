"""Pinch's firmware: the program that runs on the robot (Radxa ZERO 3W), started by systemd at boot.

    python robot/pinch_main.py                 # the real robot
    python robot/pinch_main.py --emulate 60    # the same firmware against emulated hardware (self-test)

One 50 Hz loop runs the body (robot.crab_robot.Crab: limits, speed, battery and bus checks), the mind at
10 Hz (pinch/mind.py - the same brain as the digital twin and the laptop app) and the reflexes. The senses
run in their own threads and only hand results to the loop, so nothing slow ever sits in the 20 ms tick:
  touch    MPR121 zones -> gestures (in the loop: one 2-byte read)
  cliff    5 x VL53L0X ring, own thread          ears   2 I2S mics (stereo) -> hearing, own thread
  eyes     wide camera, own thread: where am I (tray marker), how fast am I really moving (ground flow),
           my things, your face - the loop only receives the answers
  face     SSD1309 OLED, own I2C bus, own thread  sound  tones through the I2S amp (aplay)
A small HTTP API on port 8080 lets the laptop or a phone talk to Pinch (typed speech, status):
    POST /say {"text": "..."}   POST /touch {"gesture": "stroke"}   GET /status
Speech understanding uses the laptop's Ollama when it's reachable (PINCH_OLLAMA=http://laptop:11434),
and the instant keyword rules otherwise.

Hardware hiccups (a NACK on I2C, a garbled servo reply) are counted and ridden out (stats["errors"]);
only a fault that persists becomes a safety stop. `systemctl stop` (SIGTERM) sits Pinch down, turns the
servos and the arm pulses off and saves its memory; the memory is also saved every minute.
"""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.policy_np import load_policies  # noqa: E402
from pinch.body import Body  # noqa: E402
from pinch.face import Face  # noqa: E402
from pinch.hearing import AudioDirection, Hearing  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402
from pinch.touch import BodySense, CliffReflex, StuckSense, TouchGestures, cliff_beams  # noqa: E402
from pinch.world import Desk  # noqa: E402
from robot.crab_robot import DT, Crab, SafetyStop  # noqa: E402
from robot.servos import Calibration  # noqa: E402

log = logging.getLogger("pinch")
DIR_N = 1024              # samples per mic for the voice direction (GCC-PHAT); a 0.1 s chunk has 1600
SAVE_EVERY = 60.0         # s between memory saves (plus one at shutdown)


class Localizer:
    """Where am I on the desk. Two estimates, kept apart on purpose:

      pose()         what the body steers by: integrated between camera fixes - from the ground flow when
                     the camera measures it, else from the walking command - and pulled to each marker fix.
      measured_xy()  only what the camera measured: the last marker fix, carried forward by ground flow.
                     The stuck check (no progress while walking) uses this one: the integrated pose keeps
                     moving when the feet slip on the spot, which is exactly the case to catch.

    sigma() is how far pose() may have drifted since the last fix; the edge guard keeps that much further
    from the edge. While Pinch is stuck the walking command is not integrated at all."""

    SIGMA0, SIGMA_MAX = 0.005, 0.06          # m
    DRIFT_CMD, DRIFT_FLOW = 0.3, 0.1         # drift per metre integrated: commanded (gait tracks 50-110 %), flow

    def __init__(self, start=(0.0, 0.0, 0.0), tracking=0.8):
        self.x, self.y, self.yaw = start
        self.tracking = tracking             # the gait achieves ~80 % of the commanded speed
        self.last_fix = -1e9
        self.fixes = 0
        self.frozen = False                  # True while stuck: don't integrate commands that go nowhere
        self.drift = 0.0                     # m of possible error accumulated since the last fix
        self.flow_v, self.flow_t = None, -1e9
        self.meas, self.meas_t = None, -1e9

    def predict(self, cmd, dt, t=None):
        if self.frozen:
            return
        vx, vy, wz = cmd
        if t is not None and self.flow_v is not None and t - self.flow_t < 0.5:
            (vx, vy), k = self.flow_v, self.DRIFT_FLOW          # measured over the desk: trust it
        else:
            vx, vy, k = self.tracking * vx, self.tracking * vy, self.DRIFT_CMD
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw += self.tracking * wz * dt
        self.drift += k * np.hypot(vx, vy) * dt

    def fix(self, pose, t, weight=0.6):
        x, y, yaw = pose
        self.x += weight * (x - self.x)
        self.y += weight * (y - self.y)
        self.yaw += weight * ((yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi)
        self.last_fix = t
        self.fixes += 1
        self.drift = 0.0
        self.meas, self.meas_t = (float(x), float(y)), t

    FLOW_MAX = 0.1                           # m/s: faster than Pinch walks = a bad flow reading (a claw moving)

    def flow(self, v_body, t, dt):
        """Ground flow measured the body moving at v_body (m/s, body frame) over the last dt seconds.
        Marker fixes outrank it: while one is recent, flow doesn't move the measured track."""
        if np.hypot(*v_body) > self.FLOW_MAX:
            return False
        self.flow_v, self.flow_t = v_body, t
        if t - self.last_fix < 0.6:
            return True
        if self.meas is None or t - self.meas_t > 1.0:      # no recent anchor: start a new measured track
            self.meas = (self.x, self.y)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        vx, vy = v_body
        self.meas = (self.meas[0] + (c * vx - s * vy) * dt, self.meas[1] + (s * vx + c * vy) * dt)
        self.meas_t = t
        return True

    def measured_xy(self, t, max_age=0.6):
        return self.meas if self.meas is not None and t - self.meas_t <= max_age else None

    def sigma(self):
        return min(self.SIGMA0 + self.drift, self.SIGMA_MAX)

    def pose(self):
        return self.x, self.y, self.yaw


class PinchFirmware:
    def __init__(self, hw, llm=True, memory_path=None):
        self.hw = hw
        self.desk = hw.desk if hasattr(hw, "desk") else Desk()
        self.crab = Crab(hw.backend, calibration=hw.calibration if hasattr(hw, "calibration") else Calibration.load())
        # memory_path: None = the default file (PINCH_MEMORY), False = don't keep a memory file (tests)
        memory = Memory() if memory_path is None else Memory(path=memory_path or None)
        self.mind = Mind(memory=memory, llm=llm)
        self.face = Face()
        self.loc = Localizer(getattr(hw, "start_pose", (0.0, 0.0, 0.0)))
        emote_pol, walk_pol = load_policies()
        self.body = Body(self.crab, self.loc.pose, self.object_belief, self.desk, tip_fn=hw.tip, emote_policy=emote_pol,
                         walk_policy=walk_pol)
        self.body.pose_sigma_fn = self.loc.sigma
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
        if hw.camera is not None:
            from robot.vision import GroundFlow, Vision
            self.vision = Vision(hw.camera_model, self.desk)
            from pinch.faces import FaceID                  # who is that (decision 57); without the models: no one
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
        if self.faceid is not None:
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

    def step(self):
        while not self.inbox.empty():
            self._apply(*self.inbox.get())
        if not getattr(self.hw, "threaded", True):
            self._poll()
        t_work = time.perf_counter()
        if self.k % 5 == 0:
            m = self.mind
            m.world.pose = self.loc.pose()
            m.world.battery = self._safe("battery", self.hw.battery_fraction, m.world.battery)
            if self.autonomous:
                motion, cancel = m.tick(DT * 5, self.body.busy, self.body.ok)
                if cancel:
                    self.body.cancel()
                if motion is not None:
                    self.body.start(motion)
                    self.stats["motions"].append((round(self.t, 1), motion[0], motion[1] if len(motion) > 1 else ""))
                for s in m.outputs()[2]:
                    self.hw.play(s)
                    self.stats["sounds"].append(s)
            # reflex: a beam sees no desk while walking toward its side -> back off away from it (latched,
            # so its own backing-off can't re-trigger it); a beam missing or stale -> no walking that way
            now = self._now()
            ages = {n: now - t for n, t in self.cliff_at.items()}
            back = self.cliff_reflex.update(self.t, self.cliff, ages, self.body.cmd_vel)
            self.body.cliff_blocked = self.cliff_reflex.blocked
            if back is not None:
                self.body.reflex(back)
                self.stats["edge_stops"] += 1
            gyro, grav = self.hw.backend.read_imu()          # the backend rides out IMU hiccups itself
            feet = self._safe("feet", lambda: self.hw.feet_down(self.crab.state), None)
            for ev in self.bodysense.update(DT * 5, gyro, grav, feet_loaded=feet):
                m.touch(ev)
            walking = self.body.busy and self.body.status in ("goto", "walk", "push", "recover", "turn_to")
            x, y, h = self.loc.pose()
            margin = self.desk.feet_margin((x, y), h)
            # reflex: a foot over the desk edge (from the marker position + leg layout) -> shuffle back in
            if margin < -0.003 and not self.bodysense.lifted and self.body.status != "recover" \
                    and self.t - self.last_recover > 3.0:
                self.body.reflex(("recover",))
                self.last_recover = self.t
            # stuck: judged on what the camera measured, never on the integrated command
            for ev in self.stuck_sense.update(DT * 5, grav, feet, self.loc.measured_xy(self.t),
                                              float(np.hypot(*self.body.cmd_vel[:2])),
                                              lifted=self.bodysense.lifted, standing=not self.body.busy,
                                              walking=walking, feet_margin=margin):
                m.touch(ev)
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
        q = self.body.tick()                                 # 20 ms: servos (and physics when emulated)
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
            if self.stop_kind == "battery" and hasattr(self.hw, "poweroff"):
                # empty pack: say so, then power the computer down instead of draining the cells further
                log.error("battery empty: shutting the computer down")
                self.hw.play("low_battery")
                time.sleep(1.5 if getattr(self.hw, "realtime", True) else 0.0)
                self.hw.poweroff()

    stop_kind = None

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
                    diary=[f"{t:.0f}s {k}: {x}" for t, k, x in m.log[-8:]])


def serve(fw: PinchFirmware, port=8080):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(fw.status() if self.path.startswith("/status") else {"error": "try /status"},
                       200 if self.path.startswith("/status") else 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send({"error": "bad json"}, 400)
            if self.path == "/say" and isinstance(data.get("text"), str):
                fw.say(data["text"][:300])
            elif self.path == "/touch" and data.get("gesture") in ("stroke", "hold", "tickle", "boop", "tap", "claw_hold"):
                fw.touch(data["gesture"])
            else:
                return self._send({"error": "unknown"}, 400)
            self._send({"ok": True})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emulate", type=float, default=0, help="run against emulated hardware for N seconds")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PINCH_PORT", 8080)))
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.emulate:
        from robot.hw_emulated import EmulatedHardware
        hw = EmulatedHardware()
        fw = PinchFirmware(hw, llm=False, memory_path=False)
        serve(fw, a.port)
        fw.run(a.emulate, realtime=False)
        print(json.dumps(fw.status(), indent=1))
        return 0
    from robot.hw_real import RealHardware
    from robot.servos import NotCalibrated
    try:
        hw = RealHardware()
    except NotCalibrated as e:
        log.error("%s", e)
        return 2                         # systemd retries a few times, then gives up (StartLimitBurst)
    fw = PinchFirmware(hw, llm=not a.no_llm)
    serve(fw, a.port)
    fw.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
