"""Firmware self-test: the robot's own program (robot/pinch_main.py) run against emulated hardware.

    python robot/firmware_test.py

Checks the deployable firmware end to end, with the I2C drivers talking to fake chips and the desk
twin as the body: the face screen gets frames, the camera localises Pinch from the tray marker,
speech through the HTTP API makes it perform, a stroke on the touch pads makes it purr, a meeting
heard through the stereo mics makes it go quiet, and it never falls.

Then the robustness checks from the firmware audit (docs/LIFE_PLAN.md A7, A8): the control tick's own cost,
transient I2C errors ridden out, the cliff reflex (latched; a failing sensor means "don't walk forward",
never "back off forever"), a stuck Pinch noticed from what the camera measures, SIGTERM -> servos off and
memory saved, the real-hardware glue against fake smbus2 / libgpiod v1 and v2 / camera, and the fisheye
lens model (undistort before solvePnP).
"""

import json
import signal
import sys
import tempfile
import threading
import types
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.drivers.i2c_devices import pages_to_image  # noqa: E402
from robot.hw_emulated import EmulatedHardware  # noqa: E402
from robot.pinch_main import PinchFirmware, serve  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def main():
    print("Pinch firmware self-test (emulated hardware)")
    # Pinch starts turned a little toward the tray, so its claws don't hide the marker from the camera
    hw = EmulatedHardware(start=START, pose=(0.0, 0.0, 0.4))
    fw = PinchFirmware(hw, llm=False, memory_path=False)
    port = 18080
    srv = serve(fw, port)
    fw.crab.start()

    def run(seconds):
        for _ in range(int(seconds / 0.02)):
            fw.step()

    run(6.0)
    img = pages_to_image(bytes(hw.oled_chip.frame))
    check("face screen: the OLED driver sends frames, eyes are drawn", hw.oled_chip.on and fw.stats["oled_frames"] > 50
          and 200 < img.sum() < 4000, f"{fw.stats['oled_frames']} frames, {int(img.sum())} lit pixels")
    x, y, _ = fw.loc.pose()
    true = hw.d.qpos[:2]
    check("camera + tray marker: knows where it is on the desk", fw.stats["vision_fixes"] > 0
          and np.hypot(x - true[0], y - true[1]) < 0.03,
          f"{fw.stats['vision_fixes']} fixes, error {np.hypot(x - true[0], y - true[1]) * 1000:.0f} mm")
    check("camera: sees its desk things", fw.stats["things_seen"] > 0, f"{fw.stats['things_seen']} sightings")

    status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5).read())
    check("HTTP API: status", "feeling" in status and "needs" in status, status["feeling"])
    post(port, "/say", {"text": "hi pinch, can you dance?"})
    run(10.0)
    emotes = [m[1] for m in fw.stats["motions"] if m[1] == "emote" or m[2] == "dance"]
    check("speech through the API -> it dances", any(m[2] == "dance" for m in fw.stats["motions"]),
          f"motions {[m[2] for m in fw.stats['motions']][-5:]}")

    n = len(fw.stats["sounds"])
    hw.touch_chip.touching = {"head"}
    run(0.15)
    hw.touch_chip.touching = {"head", "back_left"}
    run(0.15)
    hw.touch_chip.touching = {"back_left"}
    run(0.3)
    hw.touch_chip.touching = set()
    run(4.0)
    check("touch pads (MPR121 driver): a stroke -> purr", "purr" in fw.stats["sounds"][n:],
          f"sounds {fw.stats['sounds'][n:]}")

    hw.audio_mode = "meeting"
    quiet = False
    for _ in range(150):
        run(1.0)
        if fw.mind.affect.quiet:
            quiet = True
            break
    n = len(fw.stats["sounds"])
    run(15.0)
    check("stereo mics: a meeting is heard -> goes quiet, no tones", quiet and len(fw.stats["sounds"]) == n,
          f"quiet after {fw.t:.0f} s, tones during the call: {fw.stats['sounds'][n:]}")
    hw.audio_mode = "quiet"
    check("never fell, servo safety never tripped", fw.bodysense is not None and hw.d.qpos[2] > 0.02
          and fw.stats["errors"] == 0, f"body height {hw.d.qpos[2] * 1000:.0f} mm, I2C errors {fw.stats['errors']}")
    ms = fw.timing()
    check("control tick: its own work fits the 20 ms budget with room for a 5x slower CPU (RK3566)",
          ms["p95"] < 4.0, f"{ms} ms on this computer")
    srv.shutdown()

    reflexes_and_errors()
    stuck_from_measured_motion()
    sigterm_and_memory()
    real_hardware_glue()
    fisheye_geometry()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


START = {"pen": (-0.18, -0.02, 0.6), "eraser": (0.10, -0.12, 0.3), "cap": (-0.12, 0.13, 0.0),
         "ball": (0.16, 0.0, 0.0)}                                         # the ball in front of Pinch


def quiet_firmware(pose=(0.0, 0.0, 0.4), memory_path=False):
    """A firmware whose mind doesn't move the body: the test drives it."""
    hw = EmulatedHardware(start=START, pose=pose)
    fw = PinchFirmware(hw, llm=False, memory_path=memory_path)
    fw.autonomous = False
    fw.crab.start()
    return hw, fw


def steps(fw, seconds, each=None):
    for _ in range(int(round(seconds / 0.02))):
        fw.step()
        if each:
            each()


class Flaky:
    """Wraps an I2C bus (or anything): every n-th call raises the OSError a NACK gives."""

    def __init__(self, inner, n=4):
        self.inner, self.n, self.calls = inner, n, 0

    def __getattr__(self, name):
        f = getattr(self.inner, name)

        def call(*a, **k):
            self.calls += 1
            if self.calls % self.n == 0:
                raise OSError(121, "Remote I/O error")
            return f(*a, **k)
        return call


class Dead:
    """A device that stopped answering."""

    def __getattr__(self, name):
        def fail(*a, **k):
            raise OSError(121, "Remote I/O error")
        return fail


def reflexes_and_errors():
    print("robustness: errors, cliff reflex")
    hw, fw = quiet_firmware()
    hw.touch.bus = Flaky(hw.touch.bus, 3)
    for r in hw.cliff:
        r.bus = Flaky(r.bus, 7)
    feet = hw.feet_down
    n = [0]

    def flaky_feet(state_fn=None):
        n[0] += 1
        if n[0] % 5 == 0:
            raise OSError(5, "Input/output error")
        return feet(state_fn)
    hw.feet_down = flaky_feet
    try:
        steps(fw, 3.0)
        ok = True
    except OSError:
        ok = False
    check("transient I2C / bus errors are counted and ridden out (the loop keeps running)",
          ok and fw.errors["touch"] > 0 and fw.errors["cliff"] > 0 and fw.errors["feet"] > 0, dict(fw.errors))

    # the desk ends ahead: one beam sees nothing. The reflex backs off once; its own backing-off (the beam
    # still sees nothing) must not trigger it again
    hw2, fw2 = quiet_firmware()
    hw2.update_cliff = lambda: None
    fw2.body.start(("walk", 0.03, 0.0, 0.0, 6.0))
    steps(fw2, 0.5)
    hw2.rangers[0].mm, hw2.rangers[1].mm = 55, 9999                       # max of the two, not min
    steps(fw2, 0.5)
    backing = fw2.body.status == "walk" and fw2.body.cmd_vel[0] < 0
    steps(fw2, 1.5)
    check("cliff reflex: one corner over the edge -> backs off once (latched, no re-trigger while backing)",
          fw2.stats["edge_stops"] == 1 and backing and fw2.cliff_reflex.latched,
          f"{fw2.stats['edge_stops']} back-offs, state {fw2.cliff_reflex.state}")
    fw2.body.start(("walk", 0.03, 0.0, 0.0, 1.0))
    before = fw2.body.forward_blocked
    steps(fw2, 0.5)
    check("...and while it still sees no desk ahead, it won't walk forward", fw2.body.forward_blocked > before
          and fw2.stats["edge_stops"] == 1 and abs(fw2.body.cmd_vel[0]) < 1e-9)
    hw2.rangers[1].mm = 55
    steps(fw2, 0.6)
    check("...desk seen again: the reflex re-arms", not fw2.cliff_reflex.latched and fw2.cliff_reflex.forward_ok)

    # a cliff sensor that stops answering is "don't know": no forward walking - and no backing off
    hw3, fw3 = quiet_firmware()
    steps(fw3, 0.3)
    hw3.cliff[1].bus = Dead()
    fw3.body.start(("walk", 0.03, 0.0, 0.0, 2.0))
    x0 = hw3.d.qpos[0]
    steps(fw3, 1.5)
    check("a dead cliff sensor -> 'don't know': no walking forward, and no backing off either",
          fw3.cliff_reflex.state == "unknown" and fw3.stats["edge_stops"] == 0 and fw3.body.forward_blocked > 0
          and abs(hw3.d.qpos[0] - x0) < 0.01,
          f"moved {(hw3.d.qpos[0] - x0) * 1000:.0f} mm, state {fw3.cliff_reflex.state}")


def stuck_from_measured_motion():
    print("A8: stuck noticed from what the camera measures")
    from robot.pinch_main import Localizer
    hw, fw = quiet_firmware()
    p0 = hw.d.qpos[0:2].copy()

    def pin():                       # something holds Pinch's body: the feet paddle, the body stays put
        hw.d.qpos[0:2] = p0
        hw.d.qvel[0:2] = 0.0
    fw.body.start(("walk", 0.04, 0.0, 0.0, 8.0))
    t0, t_stuck = fw.t, None
    for _ in range(int(8.0 / 0.02)):
        fw.step()
        pin()
        if fw.stuck_sense.stuck and t_stuck is None:
            t_stuck = fw.t
    x, y, _ = fw.loc.pose()
    drift = float(np.hypot(x - p0[0], y - p0[1]))
    check("held in place while walking -> 'stuck', from the marker fixes (not the walking command)",
          t_stuck is not None and fw.stuck_sense.why == "walking but not moving" and fw.stats["vision_fixes"] > 10,
          f"stuck after {None if t_stuck is None else round(t_stuck - t0, 1)} s, "
          f"{fw.stats['vision_fixes']} fixes")
    check("...its position estimate stays where it really is (commands not integrated while stuck)",
          drift < 0.02, f"{drift * 1000:.0f} mm off; dead reckoning alone would say {0.8 * 0.04 * 8 * 1000:.0f} mm")

    # walking freely with the marker in view: never called stuck
    hw1, fw1 = quiet_firmware()
    fw1.body.start(("walk", 0.0, 0.03, 0.0, 5.0))
    steps(fw1, 5.0)
    check("...and walking freely is not 'stuck'", not fw1.stuck_sense.stuck and fw1.stats["vision_fixes"] > 5,
          f"moved {np.hypot(*(hw1.d.qpos[0:2] - p0)) * 1000:.0f} mm, {fw1.stats['vision_fixes']} fixes")

    loc = Localizer()
    loc.frozen = True
    loc.predict((0.05, 0.0, 0.0), 1.0, 0.0)
    frozen_ok = loc.pose() == (0.0, 0.0, 0.0)
    loc.frozen = False
    for k in range(50):
        loc.predict((0.04, 0.0, 0.0), 0.1, k * 0.1)
    grown = loc.sigma()
    loc.fix((0.2, 0.0, 0.0), 5.0)
    check("localizer: frozen while stuck; its uncertainty grows between fixes, a fix resets it",
          frozen_ok and grown > 0.04 and loc.sigma() < 0.01 and loc.measured_xy(5.1) == (0.2, 0.0)
          and loc.measured_xy(7.0) is None, f"sigma {grown * 1000:.0f} mm after 16 cm unmeasured")

    # the edge guard keeps the uncertainty further from the edge
    hw2, fw2 = quiet_firmware(pose=(0.0, 0.0, 0.0))
    fw2.loc.x, fw2.loc.drift = fw2.desk.half_x - 0.16, 0.0
    a = fw2.body._edge_guard((0.04, 0.0, 0.0))
    fw2.loc.drift = 0.06
    b = fw2.body._edge_guard((0.04, 0.0, 0.0))
    check("edge guard: an uncertain position keeps Pinch further from the edge", a[0] > 0 and b[0] == 0.0,
          f"forward allowed when sure: {a[0] > 0}, when 6 cm unsure: {b[0] > 0}")


def sigterm_and_memory():
    print("shutdown and memory")
    from pinch.memory import DEFAULT_FILE
    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp) / "memory.json"
        hw = EmulatedHardware(start=START, pose=(0.0, 0.0, 0.4))
        fw = PinchFirmware(hw, llm=False, memory_path=mem)
        fw.mind.memory.remember("test", "the firmware test was here")
        timer = threading.Timer(4.0, signal.raise_signal, [signal.SIGTERM])
        timer.start()
        fw.run(seconds=600, realtime=False)
        timer.cancel()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        saved = mem.exists() and "the firmware test was here" in mem.read_text(encoding="utf-8")
        check("SIGTERM (systemctl stop): loop ends, servos torque off, memory saved",
              fw.t < 600 and not hw.backend.torque_on and saved, f"stopped at t={fw.t:.1f} s")
        m2 = Path(tmp) / "m2.json"
        hw2, fw2 = quiet_firmware(memory_path=m2)
        fw2.mind.memory.remember("test", "periodic")
        steps(fw2, 61.0)
        if fw2._saving is not None:
            fw2._saving.join()
        check("memory is saved every minute (fsync'd), not only at shutdown", m2.exists())
        fw3 = PinchFirmware(hw2, llm=False)              # what main() does on the robot
        check("the robot's memory goes to its file (PINCH_MEMORY), not nowhere", fw3.mind.memory.path == DEFAULT_FILE)


class FakeGpiodV2:
    """libgpiod v2's API surface (gpiod.request_lines)."""

    def __init__(self, on_set):
        self.on_set = on_set
        self.line = types.SimpleNamespace(Direction=types.SimpleNamespace(OUTPUT="out"),
                                          Value=types.SimpleNamespace(ACTIVE=1, INACTIVE=0))
        self.LineSettings = lambda **k: k

    def request_lines(self, path, consumer, config):
        (line, settings), = config.items()
        self.on_set(path, line, settings["output_value"])
        return types.SimpleNamespace(set_value=lambda ln, v: self.on_set(path, ln, v))


class FakeGpiodV1:
    """libgpiod v1.6's API surface (Debian 12 python3-libgpiod): Chip / get_line / request."""
    LINE_REQ_DIR_OUT = 3

    def __init__(self, on_set):
        on = on_set

        class Line:
            def __init__(self, path, n):
                self.path, self.n = path, n

            def request(self, consumer, type, default_vals):
                assert type == FakeGpiodV1.LINE_REQ_DIR_OUT
                on(self.path, self.n, default_vals[0])

            def set_value(self, v):
                on(self.path, self.n, v)

        class Chip:
            def __init__(self, path):
                self.path = path

            def get_line(self, n):
                return Line(self.path, n)
        self.Chip = Chip


def real_hardware_glue():
    print("real-hardware glue (fake smbus2, libgpiod, camera)")
    import robot.servos as servos
    from robot.drivers.fakes import FakeBus, FakeOLED, FakeRanger, FakeTouch
    from robot.hw_real import RealHardware
    from robot.backends import SimBackend
    from robot.servos import Calibration, NotCalibrated

    for api in ("v1", "v2"):
        rangers = {1: FakeRanger(0x29), 2: FakeRanger(0x29)}
        powered = {1: True, 2: True}

        def on_set(path, line, v):
            powered[line] = bool(v)

        class SensorBus(FakeBus):                       # bus 3: touch + two rangers behind XSHUT lines
            def _chip(self, addr):
                for k, r in rangers.items():
                    if powered[k] and r.addr == addr:
                        return r
                return super()._chip(addr)

            def write_byte_data(self, addr, reg, val):
                self._chip(addr).write(reg, [val])

        handles = []

        def smbus(n):
            b = SensorBus(FakeTouch()) if n == 3 else FakeBus(FakeOLED())
            handles.append((n, b))
            return b

        class NoCamera:
            def __init__(self, i):
                pass

            def isOpened(self):
                return False

            def release(self):
                pass

        gpiod = FakeGpiodV1(on_set) if api == "v1" else FakeGpiodV2(on_set)
        with tempfile.TemporaryDirectory() as tmp:
            old = servos.CAL_FILE
            servos.CAL_FILE = Path(tmp) / "calibration.json"
            try:
                try:
                    RealHardware(smbus=smbus, backend=SimBackend(), gpiod=gpiod, video=NoCamera)
                    refused = False
                except NotCalibrated:
                    refused = True
                Calibration().save()
                handles.clear()
                hw = RealHardware(smbus=smbus, backend=SimBackend(), gpiod=gpiod, video=NoCamera)
            finally:
                servos.CAL_FILE = old
        sensor_handles = [b for k, b in handles if k == 3]
        ok = (hw.cliff is not None and sorted(r.addr for r in rangers.values()) == [0x29, 0x30]
              and hw.cliff[0].range_mm() == 55 and hw.cliff[1].range_mm() == 55)
        check(f"libgpiod {api} API: both cliff sensors brought up (XSHUT), one moved to 0x30", ok,
              f"addresses {sorted(hex(r.addr) for r in rangers.values())}")
        if api == "v1":
            check("no calibration file -> the firmware refuses to start", refused)
            check("touch and cliff sensors get their own I2C handles (no cross-thread address race)",
                  hw.touch.bus is not hw.cliff[0].bus and len(sensor_handles) == 2)
            check("camera that doesn't open -> no camera (checked with isOpened), the rest carries on",
                  hw.camera is None and hw.oled is not None)


def fisheye_geometry():
    print("fisheye camera")
    import cv2
    from robot.calibrate_camera import selftest
    from robot.vision import Camera, Vision
    from pinch.world import Desk
    K0, D0, K, D, rms = selftest()
    check("camera calibration recovers a known fisheye lens", abs(K[0, 0] - K0[0, 0]) < 2 and
          abs(K[0, 2] - K0[0, 2]) < 2 and rms < 0.5, f"focal {K[0, 0]:.1f} vs {K0[0, 0]:.1f}, rms {rms:.2f} px")
    cam = Camera.load(intrinsics="nonexistent.json")
    check("without measured intrinsics the camera is 'uncalibrated' and never gives a position",
          not cam.calibrated and Vision(cam, Desk()).locate(np.zeros((480, 640, 3), np.uint8)) is None)
    # a fisheye view of the tray marker from a known pose: undistorting first gives the pose back
    cam.K, cam.D, cam.calibrated = K0.copy(), D0.copy(), True
    v = Vision(cam, Desk())
    x, y, yaw = (-0.02, 0.01, 0.5)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    R = (Rz @ cam.R).T                                  # desk -> camera
    p_cam = np.array([x, y, 0.0]) + Rz @ cam.pos        # camera origin in the desk frame (body origin at z=0)
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ p_cam
    pts, _ = cv2.fisheye.projectPoints(v.marker_corners_desk().reshape(-1, 1, 3).astype(np.float64), rvec,
                                       tvec.reshape(3, 1), cam.K, cam.D)
    est = v.pose_from_corners(pts.reshape(4, 2))
    err = np.hypot(est[0] - x, est[1] - y)
    cam_pin = Camera.load(intrinsics="nonexistent.json")
    cam_pin.K, cam_pin.calibrated = K0.copy(), True     # same lens, distortion ignored (the old code)
    bad = Vision(cam_pin, Desk()).pose_from_corners(pts.reshape(4, 2))
    err_bad = np.hypot(bad[0] - x, bad[1] - y)
    check("marker pose through the fisheye lens: undistorted before solvePnP", err < 0.002,
          f"{err * 1000:.1f} mm off (ignoring the distortion: {err_bad * 1000:.0f} mm)")


if __name__ == "__main__":
    sys.exit(main())
