"""Pinch's eyes: the wide-angle face camera -> where am I on the desk, where are my things, are you there?

  * where am I: the ArUco marker on the desk tray's tab (OpenCV's ArUco detector + solvePnP) gives the
    camera's pose relative to the tray, so Pinch's position and heading on the desk.
  * where are my things: Pinch's desk things are distinctly coloured (blue pen, pink eraser, red cap,
    yellow ball); each colour blob's lowest point is projected onto the desk plane.
  * are you there / looking at me: OpenCV's bundled face detector (Haar cascade).

Same code on the robot (camera frames) and in the digital twin (renders of the eye camera), so the
twin tests the real vision pipeline. Camera geometry: robot/camera.json (from the CAD / twin).
"""

import json
from pathlib import Path

import cv2
import numpy as np

CAMERA_FILE = Path(__file__).resolve().parent / "camera.json"
MARKER_ID = 7
MARKER_TAB_M = 0.040                    # printed marker incl. its white border (sim/desk.py MARKER_M)
MARKER_BLACK_M = MARKER_TAB_M * 0.75    # the black square ArUco measures
COLOURS = {                      # HSV ranges (OpenCV: H 0..180)
    "pen": [((100, 120, 60), (125, 255, 255))],
    "eraser": [((150, 40, 120), (175, 200, 255))],
    "cap": [((0, 150, 80), (8, 255, 255)), ((172, 150, 80), (180, 255, 255))],
    "ball": [((20, 120, 120), (35, 255, 255))],
}
HALF_HEIGHT = {"pen": 0.0045, "eraser": 0.0045, "cap": 0.006, "ball": 0.015}


class Camera:
    """Pinhole model of the (undistorted) wide camera, mounted in the body frame."""

    def __init__(self, width, height, fovy_deg, pos_body, R_body):
        self.w, self.h = width, height
        f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
        self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        self.pos = np.asarray(pos_body, float)          # camera origin in the body frame
        self.R = np.asarray(R_body, float)               # columns: camera x (right), y (down), z (forward), body frame

    @classmethod
    def from_mujoco(cls, m, width=640, height=480, name="eye_cam"):
        import mujoco
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[0:3] = 0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)                          # camera poses are only computed here
        c = m.camera(name).id
        R_mj = d.cam_xmat[c].reshape(3, 3)               # MuJoCo camera: x right, y up, looks along -z
        return cls(width, height, float(m.cam_fovy[c]), d.cam_xpos[c].copy(), R_mj @ np.diag([1, -1, -1]))

    def save(self, path=CAMERA_FILE):
        Path(path).write_text(json.dumps(dict(width=self.w, height=self.h, K=self.K.tolist(), pos=self.pos.tolist(),
                                              R=self.R.tolist()), indent=1))

    @classmethod
    def load(cls, path=CAMERA_FILE):
        j = json.loads(Path(path).read_text())
        cam = cls(j["width"], j["height"], 60, j["pos"], j["R"])
        cam.K = np.array(j["K"])
        return cam

    def ray_body(self, u, v):
        d = np.linalg.solve(self.K, np.array([u, v, 1.0]))
        return self.R @ (d / np.linalg.norm(d))

    def to_desk(self, u, v, pose, z=0.0, body_z=0.045, body_R=None):
        """Pixel -> point on the plane at height z, in the desk frame. pose = (x, y, yaw) of the body;
        body_R: the body's full orientation (from the IMU) if it's tilted."""
        x, y, yaw = pose
        R = body_R if body_R is not None else np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                                                         [0, 0, 1.0]])
        o = np.array([x, y, body_z]) + R @ self.pos
        r = R @ self.ray_body(u, v)
        if r[2] > -1e-3:
            return None                                  # above the horizon
        t = (z - o[2]) / r[2]
        return o + t * r


class Vision:
    def __init__(self, camera: Camera, desk=None):
        self.cam = camera
        self.desk = desk
        self.aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                             cv2.aruco.DetectorParameters())
        # OpenCV 4 (the Radxa's apt package) ships the Haar face detector; OpenCV 5 moved it out
        self.faces = None
        if hasattr(cv2, "CascadeClassifier"):
            self.faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def marker_corners_desk(self):
        """The marker's four corners in the desk frame, in ArUco's order (top-left, top-right, bottom-right,
        bottom-left as seen from the front). The tab stands at the back of the tray facing you (-y)."""
        tx, ty = self.desk.tray_xy
        y = ty + self.desk.tray_half[1] + 0.0011
        cz, s = MARKER_TAB_M / 2 + 0.006, MARKER_BLACK_M / 2
        return np.array([[tx - s, y, cz + s], [tx + s, y, cz + s], [tx + s, y, cz - s], [tx - s, y, cz - s]], np.float32)

    def locate(self, bgr):
        """Body pose (x, y, yaw) on the desk from the tray marker, or None if it isn't in view."""
        corners, ids, _ = self.aruco.detectMarkers(bgr)
        if ids is None or MARKER_ID not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(self.marker_corners_desk(), c, self.cam.K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)                       # desk -> camera
        p_cam = -R.T @ tvec.ravel()                      # camera origin in the desk frame
        R_db = R.T @ self.cam.R.T                        # body -> desk (camera axes known in the body frame)
        yaw = float(np.arctan2(R_db[1, 0], R_db[0, 0]))
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
        p_body = p_cam - Rz @ self.cam.pos
        return float(p_body[0]), float(p_body[1]), yaw

    def things(self, bgr, pose, body_z=0.045):
        """Desk things by colour: {name: (x, y) on the desk}."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out = {}
        for name, ranges in COLOURS.items():
            mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 25:
                continue
            M = cv2.moments(c)
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            p = self.cam.to_desk(u, v, pose, z=HALF_HEIGHT[name], body_z=body_z)
            if p is not None:
                out[name] = (float(p[0]), float(p[1]))
        return out

    def face(self, bgr):
        """(found, how centred -1..1) - someone looking toward Pinch."""
        if self.faces is None:
            return False, 0.0
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        f = self.faces.detectMultiScale(g, 1.2, 5, minSize=(30, 30))
        if len(f) == 0:
            return False, 0.0
        x, y, w, h = max(f, key=lambda r: r[2] * r[3])
        return True, float((x + w / 2) / g.shape[1] * 2 - 1)
