"""A world, alive around the twin: people move, touch Pinch, and are seen through its fisheye camera.

    sim = PinchSim(world=w)
    life = Life(sim)                     # the world's people; camera=True: the fisheye + face recognition in the loop
    life.run(30.0)                       # = sim.run(30.0) with people and the camera

What reaches the mind is what reaches it on the robot (robot/firmware/senses.py): a touch on the shell when a hand
gets there (the gesture by the person's style), and, from rendered camera frames, the faces YuNet finds and SFace
recognises (pinch/faces.py) - mind.see_person(pid, confidence) and "someone is there". Without the camera (the
default for fast, headless runs) a person facing Pinch within 1.2 m counts as seen, with the confidence falling off
with distance and the field of view (pinch/novelty.py FOV_HALF), which is what the camera path measures.
"""

import numpy as np

from pinch.twin import DT
from sim.worlds.people import Person

CAMERA_HZ = 5.0                      # faces at 5 Hz, as on the robot (pinch/faces.py)


def _has(m, body):
    try:
        m.body(body)
        return True
    except KeyError:
        return False


class Life:
    def __init__(self, sim, camera=False, people=True):
        self.sim, self.w = sim, sim.world
        self.people = []
        if people and self.w:
            from sim.worlds import humans
            for i, p in enumerate(self.w.people):
                body = None
                if _has(sim.m, f"person{i}_pelvis"):     # built as a MakeHuman body
                    body = humans.geometry(*humans.params(p), round(p["height"], 2))
                self.people.append(Person(i, p, self.w, body=body))
        self.touches = []                 # (t, person index, gesture)
        self.seen = []                    # (t, pid, confidence)
        self.frames = 0
        self.cam = self.faceid = None
        self.last_frame = None
        if camera:
            from sim.worlds.fisheye import FisheyeCamera
            self.cam = FisheyeCamera(sim.m)
            from pinch.faces import FaceID
            if FaceID.available():
                self.faceid = FaceID()
        self._next_cam = 0.0
        self._prev_xy = None
        self.voices = []                  # [person index, audio, position] someone talking (speak())
        self.heard_dirs = []              # (t, estimated direction, true direction) of voices
        from pinch.hearing import AudioDirection
        self._dir = AudioDirection()
        self._audio_t = 0.0

    def crab(self):
        q = self.sim.d.qpos
        return np.array([q[0], q[1], q[2]])

    def step(self, sim):
        c = self.crab()
        vel = (0.0, 0.0) if self._prev_xy is None else tuple((c[:2] - self._prev_xy) / DT)
        self._prev_xy = c[:2].copy()
        for p in self.people:
            g = p.step(sim.t, DT, c, vel)
            p.apply(sim.m, sim.d)
            if g is not None:
                self.touches.append((round(sim.t, 2), p.i, g))
                sim.touch(g)
        if self.voices and sim.t - self._audio_t >= 0.5 - 1e-9:
            self._audio_t = sim.t
            self.hear(sim, 0.5)
        if sim.t + 1e-9 >= self._next_cam:
            self._next_cam = sim.t + 1.0 / CAMERA_HZ
            self.look(sim)

    def look(self, sim):
        if self.cam is not None:
            import mujoco
            mujoco.mj_forward(sim.m, sim.d)
            bgr = self.cam.render(sim.d)
            self.last_frame = bgr
            self.frames += 1
            if self.faceid is not None:
                for pid, conf, _box in self.faceid.see(bgr, sim.t):
                    sim.mind.world.owner.present, sim.mind.world.owner.last_seen = True, sim.mind.t
                    if pid is not None:
                        self.seen.append((round(sim.t, 2), pid, conf))
                        sim.mind.see_person(pid, conf)
            return
        x, y, h = sim.pose()
        for p in self.people:                             # no camera: the geometry the camera would see
            f = p.face_xyz()
            if f is None:
                continue
            d = float(np.hypot(f[0] - x, f[1] - y))
            bearing = (np.arctan2(f[1] - y, f[0] - x) - h + np.pi) % (2 * np.pi) - np.pi
            if d < 1.2 and abs(bearing) < 1.1:
                conf = float(np.clip(1.0 - (d - 0.3) / 0.9, 0.3, 1.0))
                pid = f"face-sim{p.i}"
                sim.mind.world.owner.present, sim.mind.world.owner.last_seen = True, sim.mind.t
                self.seen.append((round(sim.t, 2), pid, conf))
                sim.mind.see_person(pid, conf, owner=p.p["role"] == "sitter")

    def speak(self, i, seconds=4.0, seed=0, talk=0.6):
        """Person i talks for a while: their voice reaches Pinch's two microphones from where their head is."""
        from pinch.hearing import synth_speech
        p = self.people[i]
        pitch = 230.0 if p.p["child"] else (120.0 if p.p.get("male", True) else 200.0)
        self.voices.append([i, synth_speech(seconds, pitch=pitch, talk=talk, seed=seed), 0])

    def hear(self, sim, dt):
        """Half a second of the voices, as the two microphones (~70 mm apart, pinch/hearing.py) would hear them:
        the far microphone hears each voice a little later. The robot's own hearing then estimates the direction
        (GCC-PHAT) and the mind gets the summary with it, as on the robot (robot/firmware/senses.py)."""
        from pinch.hearing import RATE
        n = int(dt * RATE)
        x, y, h = sim.pose()
        left, right, best = np.zeros(n), np.zeros(n), None
        for v in list(self.voices):
            i, audio, k = v
            chunk = audio[k:k + n]
            v[2] += n
            if len(chunk) < n:                             # (the end of what they said)
                self.voices.remove(v)
                continue
            f = self.people[i].face_xyz()
            if f is None:
                continue
            bearing = (np.arctan2(f[1] - y, f[0] - x) - h + np.pi) % (2 * np.pi) - np.pi
            d = float(np.hypot(f[0] - x, f[1] - y))
            lag = int(round(self._dir.spacing * np.sin(bearing) / 343.0 * RATE))     # + : the left mic first
            g = 1.0 / max(d, 0.3)                          # (louder when closer)

            def late(a, k):
                return np.concatenate([np.zeros(k), a[:n - k]]) if k > 0 else a
            left += g * late(chunk, max(-lag, 0))          # to the right: the left mic hears it later
            right += g * late(chunk, max(lag, 0))          # to the left: the right mic hears it later
            best = bearing if best is None else best
        if best is None:
            return
        est = self._dir.angle(left, right)
        mono = (left + right) / 2
        for s in sim.hearing.feed(mono * 0.5):
            sim.mind.hear_summary(s, voice_dir=est)
            self.heard_dirs.append((round(sim.t, 2), est, best))
        for kind, _ in sim.hearing.take_events():
            sim.mind.hear_event(kind)

    def run(self, seconds):
        self.sim.run(seconds, every=self.step)
