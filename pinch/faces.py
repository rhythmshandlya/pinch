"""Who is that? Faces from Pinch's camera -> a person id for pinch/people.py (decision 57).

YuNet finds faces (and five landmarks); SFace turns each aligned face into a 128-number fingerprint. A face
whose fingerprint is close enough (cosine >= MATCH, SFace's published threshold) to someone Pinch has met is
them; otherwise it is someone new, who gets an id ("face-3") and is remembered after a few steady sightings -
so a stranger who keeps coming back becomes familiar. Names come from voice ("I'm Sam" while one face is in
view: Mind attaches the name to that person).

Models (not in git, ~39 MB): models/faces/face_detection_yunet_2023mar.onnx and
face_recognition_sface_2021dec.onnx from the OpenCV model zoo (deploy/install.sh fetches them; SHA-256 below).
Without them FaceID.available() is False and Pinch falls back to "someone is there" (the Haar detector).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "faces"
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
SHA256 = {YUNET: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
          SFACE: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"}
MATCH = 0.363            # SFace: cosine similarity at or above this = the same person
NEW_AFTER = 3            # sightings of an unknown face (within NEW_WINDOW s) before it becomes a person
NEW_WINDOW = 10.0
KEEP = 8                 # fingerprints kept per person (varied: lighting, angle)


class FaceID:
    def __init__(self, model_dir=MODEL_DIR, gallery_path=None, min_score=0.8):
        import cv2
        self.cv2 = cv2
        # one thread: OpenCV's DNN worker pool kept spinning after each frame and took CPU from the 50 Hz
        # control loop (firmware test: tick p95 4.6 ms -> see decision 57); faces at 5 Hz don't need more
        cv2.setNumThreads(1)
        self.det = cv2.FaceDetectorYN.create(str(Path(model_dir) / YUNET), "", (320, 240), min_score, 0.3, 20)
        self.rec = cv2.FaceRecognizerSF.create(str(Path(model_dir) / SFACE), "")
        self.gallery = {}                     # pid -> list of unit fingerprints
        self.pending = []                     # (t, fingerprint) of unknown faces not yet a person
        self.next_id = 1
        self.path = Path(gallery_path) if gallery_path else None
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.gallery = {k: [np.array(f, np.float32) for f in v] for k, v in d["gallery"].items()}
            self.next_id = d.get("next_id", len(self.gallery) + 1)

    @staticmethod
    def available(model_dir=MODEL_DIR):
        return all((Path(model_dir) / f).exists() for f in (YUNET, SFACE))

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        return [] if faces is None else list(faces)

    def fingerprint(self, bgr, face):
        f = self.rec.feature(self.rec.alignCrop(bgr, face)).ravel().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def match(self, f):
        """(pid, cosine) of the best-matching person, or (None, best cosine)."""
        best, score = None, -1.0
        for pid, fs in self.gallery.items():
            s = max(float(f @ g) for g in fs)
            if s > score:
                best, score = pid, s
        return (best, score) if score >= MATCH else (None, score)

    def remember(self, pid, f):
        fs = self.gallery.setdefault(pid, [])
        if all(float(f @ g) < 0.9 for g in fs):          # only fingerprints that add something
            fs.append(f)
            del fs[:-KEEP]

    def see(self, bgr, t):
        """[(pid, confidence 0..1, (x, y, w, h))] for the faces in this frame. Unknown faces are pid None until
        they have been seen NEW_AFTER times in NEW_WINDOW seconds - then they become a new person."""
        out = []
        for face in self.detect(bgr):
            f = self.fingerprint(bgr, face)
            pid, score = self.match(f)
            if pid is None:
                self.pending = [(tp, g) for tp, g in self.pending if t - tp < NEW_WINDOW]
                same = [g for _, g in self.pending if float(f @ g) >= MATCH]
                self.pending.append((t, f))
                if len(same) + 1 >= NEW_AFTER:
                    pid = f"face-{self.next_id}"
                    self.next_id += 1
                    for g in same + [f]:
                        self.remember(pid, g)
                    self.pending = [(tp, g) for tp, g in self.pending if float(f @ g) < MATCH]
                    score = 1.0
            else:
                self.remember(pid, f)
            conf = float(np.clip((score - MATCH) / (1 - MATCH) * 0.5 + 0.5, 0, 1)) if pid else 0.0
            out.append((pid, conf, tuple(int(v) for v in face[:4])))
        return out

    def forget(self, pid):
        self.gallery.pop(pid, None)

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(dict(next_id=self.next_id, gallery={
                k: [[round(float(x), 5) for x in f] for f in v] for k, v in self.gallery.items()})))
