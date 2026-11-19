"""Pinch's firmware: the program that runs on the robot (Radxa ZERO 3W), started by systemd at boot.

    python robot/pinch_main.py                 # the real robot
    python robot/pinch_main.py --emulate 60    # the same firmware against emulated hardware (self-test)

One 50 Hz loop runs the body (robot.crab_robot.Crab: limits, speed, battery and bus checks), the mind at
10 Hz (pinch/mind.py - the same brain as the digital twin and the laptop app) and the senses:
  touch    MPR121 zones -> gestures              face   SSD1309 OLED, own I2C bus, own thread
  cliff    2 x VL53L0X, own thread               ears   2 I2S mics (stereo) -> hearing, own thread
  eyes     wide camera -> where am I (tray marker), my things, your face; own thread
  sound    tones through the I2S amp (aplay)
A small HTTP API on port 8080 lets the laptop or a phone talk to Pinch (typed speech, status):
    POST /say {"text": "..."}   POST /touch {"gesture": "stroke"}   GET /status
Speech understanding uses the laptop's Ollama when it's reachable (PINCH_OLLAMA=http://laptop:11434),
and the instant keyword rules otherwise.
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
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
from pinch.touch import BodySense, StuckSense, TouchGestures  # noqa: E402
from pinch.world import Desk  # noqa: E402
from robot.crab_robot import DT, Crab, SafetyStop  # noqa: E402
from robot.servos import Calibration  # noqa: E402

log = logging.getLogger("pinch")
CLIFF_MM = 110            # more than this below the front corners = no desk there


class Localizer:
    """Where am I on the desk: dead reckoning from the walking commands, corrected by the camera
    whenever it sees the tray marker."""

    def __init__(self, start=(0.0, 0.0, 0.0), tracking=0.8):
        self.x, self.y, self.yaw = start
        self.tracking = tracking             # the gait achieves ~80 % of the commanded speed
        self.last_fix = -1e9

    def predict(self, cmd, dt):
        vx, vy, wz = cmd
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += self.tracking * (c * vx - s * vy) * dt
        self.y += self.tracking * (s * vx + c * vy) * dt
        self.yaw += self.tracking * wz * dt

    def fix(self, pose, t, weight=0.6):
        x, y, yaw = pose
        self.x += weight * (x - self.x)
        self.y += weight * (y - self.y)
        self.yaw += weight * ((yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi)
        self.last_fix = t

    def pose(self):
        return self.x, self.y, self.yaw


class PinchFirmware:
    def __init__(self, hw, llm=True, memory_path=None):
        self.hw = hw
        self.desk = hw.desk if hasattr(hw, "desk") else Desk()
        self.crab = Crab(hw.backend, calibration=hw.calibration if hasattr(hw, "calibration") else Calibration.load())
        self.mind = Mind(memory=Memory(path=memory_path) if memory_path is not False else Memory(path=None), llm=llm)
        self.face = Face()
        self.loc = Localizer(getattr(hw, "start_pose", (0.0, 0.0, 0.0)))
        emote_pol, _ = load_policies()
        self.body = Body(self.crab, self.loc.pose, self.object_belief, self.desk, tip_fn=hw.tip, emote_policy=emote_pol)
        self.gestures = TouchGestures()
        self.bodysense = BodySense()
        self.stuck_sense = StuckSense()
        self.last_recover = -99.0
        self.hearing = Hearing()
        self.direction = AudioDirection()
        self.inbox = queue.Queue()           # things from the HTTP API / threads, applied in the loop
        self.cliff = [9999, 9999]
        self.face_frame = None
        self.running = True
        self.t = 0.0
        self.k = 0
        self.stats = dict(oled_frames=0, sounds=[], motions=[], vision_fixes=0, things_seen=0, touches=0, errors=0)
        self.vision = None
        if hw.camera is not None:
            from robot.vision import Camera, Vision
            self.vision = Vision(hw.camera_model, self.desk)

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

    # -- threads ---------------------------------------------------------------------------------
    def _face_thread(self):
        while self.running:
            frame = self.face_frame
            if frame is not None and self.hw.oled is not None:
                try:
                    self.hw.oled.show(frame)
                    self.stats["oled_frames"] += 1
                except OSError:
                    self.stats["errors"] += 1
            time.sleep(0.06)                                 # ~15 frames a second is plenty for eyes

    def _cliff_thread(self):
        while self.running and self.hw.cliff is not None:
            try:
                self.cliff = [r.range_mm() or 9999 for r in self.hw.cliff]
            except OSError:
                self.stats["errors"] += 1
            time.sleep(0.05)

    def _vision_thread(self):
        while self.running and self.vision is not None:
            bgr = self.hw.camera()
            if bgr is not None:
                self.inbox.put(("frame", bgr))
            time.sleep(0.2)

    def _audio_thread(self):
        while self.running and self.hw.audio is not None:
            chunk = self.hw.audio()                           # (n, 2) stereo float, 16 kHz
            if chunk is None:
                time.sleep(0.05)
                continue
            mono = chunk.mean(axis=1)
            for s in self.hearing.feed(mono):
                ang = self.direction.angle(chunk[-2048:, 0], chunk[-2048:, 1]) if len(chunk) >= 2048 else None
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
        elif kind == "frame" and self.vision is not None:
            pose = self.vision.locate(data)
            if pose is not None:
                self.loc.fix(pose, self.t)
                self.stats["vision_fixes"] += 1
            for name, xy in self.vision.things(data, self.loc.pose()).items():
                if self.desk.inside(xy, -0.01):
                    m.world.see(name, xy, graspable=False)
                    self.stats["things_seen"] += 1
            seen, _ = self.vision.face(data)
            if seen:
                m.world.owner.present, m.world.owner.last_seen = True, m.t

    # -- the loop --------------------------------------------------------------------------------
    def _poll(self):
        """Emulated hardware: read the senses inside the loop (deterministic) instead of in threads."""
        hw = self.hw
        if self.k % 3 == 0 and self.face_frame is not None and hw.oled is not None:
            hw.oled.show(self.face_frame)
            self.stats["oled_frames"] += 1
        if self.k % 5 == 0 and hw.cliff is not None:
            hw.update_cliff()
            self.cliff = [r.range_mm() or 9999 for r in hw.cliff]
        if self.k % 10 == 0 and self.vision is not None:
            self._apply("frame", hw.camera())
        if self.k % 5 == 0 and hw.audio is not None:
            chunk = hw.audio()
            for s in self.hearing.feed(chunk.mean(axis=1)):
                self._apply("voice", (s, self.direction.angle(chunk[-1024:, 0], chunk[-1024:, 1])))
            for ev, _ in self.hearing.take_events():
                self._apply("sound_event", ev)

    def step(self):
        while not self.inbox.empty():
            self._apply(*self.inbox.get())
        if not getattr(self.hw, "threaded", True):
            self._poll()
        if self.k % 5 == 0:
            m = self.mind
            m.world.pose = self.loc.pose()
            m.world.battery = self.hw.battery_fraction()
            motion, cancel = m.tick(DT * 5, self.body.busy, self.body.ok)
            if cancel:
                self.body.cancel()
            if motion is not None:
                self.body.start(motion)
                self.stats["motions"].append((round(self.t, 1), motion[0], motion[1] if len(motion) > 1 else ""))
            for s in m.outputs()[2]:
                self.hw.play(s)
                self.stats["sounds"].append(s)
            # reflex: no desk under the front corners -> stop and back off
            if min(self.cliff) > CLIFF_MM and self.body.busy and self.body.status in ("goto", "walk", "push"):
                self.body.cancel()
                self.body.start(("walk", -0.03, 0.0, 0.0, 1.0))
            gyro, grav = self.hw.backend.read_imu()
            feet = self.hw.feet_down()
            for ev in self.bodysense.update(DT * 5, gyro, grav, feet_loaded=feet):
                m.touch(ev)
            walking = self.body.busy and self.body.status in ("goto", "walk", "push", "recover", "turn_to")
            x, y, h = self.loc.pose()
            margin = self.desk.feet_margin((x, y), h)
            # reflex: a foot over the desk edge (from the marker position + leg layout) -> shuffle back in
            if margin < -0.003 and not self.bodysense.lifted and self.body.status != "recover" \
                    and self.t - self.last_recover > 3.0:
                self.body.cancel()
                self.body.start(("recover",))
                self.last_recover = self.t
            for ev in self.stuck_sense.update(DT * 5, grav, feet, (x, y), float(np.hypot(*self.body.cmd_vel[:2])),
                                              lifted=self.bodysense.lifted, standing=not self.body.busy,
                                              walking=walking, feet_margin=margin):
                m.touch(ev)
        if self.k % 2 == 0 and self.hw.touch is not None:
            try:
                for g in self.gestures.update(self.t, self.hw.touch.touched()):
                    self.stats["touches"] += 1
                    self.mind.touch(g)
            except OSError:
                self.stats["errors"] += 1
        if self.k % 3 == 0:
            self.face.update(DT * 3, self.mind.face_expr, self.mind.gaze)
            self.face_frame = self.face.ssd1309_pages()
        q = self.body.tick()                                 # 20 ms: servos (and physics when emulated)
        cmd = getattr(self.body, "cmd_vel", (0.0, 0.0, 0.0))
        self.loc.predict(cmd, DT)
        self.t += DT
        self.k += 1
        return q

    def run(self, seconds=None, realtime=True):
        if getattr(self.hw, "threaded", True):
            for f in (self._face_thread, self._cliff_thread, self._vision_thread, self._audio_thread):
                threading.Thread(target=f, daemon=True).start()
        self.crab.start()
        t0 = time.perf_counter()
        try:
            while self.running and (seconds is None or self.t < seconds):
                self.step()
                if realtime and getattr(self.hw, "realtime", True):
                    lag = t0 + self.t - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
        except SafetyStop as e:
            log.error("safety stop: %s", e)
            self.face.update(0.1, "dizzy")
            if self.hw.oled is not None:
                self.hw.oled.show(self.face.ssd1309_pages())
        finally:
            self.running = False
            self.mind.memory.save()
            try:
                self.crab.shutdown()
            except Exception:
                pass

    def status(self):
        m = self.mind
        return dict(feeling=m.affect.feeling(), activity=m.activity.name if m.activity else None,
                    needs=m.affect.needs.as_dict(), valence=round(m.affect.valence, 3), owner=m.memory.owner_name,
                    busy=m.world.owner.busy, pose=[round(v, 3) for v in self.loc.pose()], t=round(self.t, 1),
                    battery=round(self.hw.battery_fraction(), 2), cliff_mm=self.cliff,
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
        return
    from robot.hw_real import RealHardware
    fw = PinchFirmware(RealHardware(), llm=not a.no_llm)
    serve(fw, a.port)
    fw.run()


if __name__ == "__main__":
    main()
