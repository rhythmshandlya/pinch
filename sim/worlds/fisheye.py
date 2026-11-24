"""Pinch's real camera view in the twin: a 160-degree fisheye (OV5647 with an M12 fisheye lens), rendered in the loop.

MuJoCo cameras are pinhole only (no lens or distortion option, MuJoCo XML reference, <camera>), and one pinhole
render can't reach 160 degrees (it would need ~3,250 px across). So, as Gazebo's wide-angle camera and CARLA's
fisheye do, the eye camera renders five 90-degree cube faces (front, left, right, up, down - added to the model by
sim/worlds/build.py as eye_front ... eye_down) and a precomputed lookup table remaps them into the fisheye image
(cv2.remap, ~2 ms). The lens is the Kannala-Brandt model OpenCV's cv2.fisheye uses and robot/calibrate_camera.py
fits on the real camera: theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), r = f theta_d.
Default: equidistant (k = 0), 160 degrees across the 640 x 480 diagonal, so f = 400 px / 80 deg = 286.5 px
(~128 x 96 degrees across the width and height); once the real lens is calibrated, pass its K and D.

    cam = FisheyeCamera(sim.m)
    bgr = cam.render(sim.d)          # (480, 640, 3) uint8, BGR like the real camera
"""

import numpy as np

FACES = ("front", "left", "right", "up", "down")
# each face's rotation from the eye camera's frame (MuJoCo camera: x right, y up, looking along -z)
_ROT = {"front": ("y", 0.0), "left": ("y", 90.0), "right": ("y", -90.0), "up": ("x", 90.0), "down": ("x", -90.0)}


def _R(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def default_K(width=640, height=480, fov_diag_deg=160.0):
    f = np.hypot(width, height) / 2 / np.radians(fov_diag_deg / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def theta_from_distorted(td, D, iters=8):
    """Invert theta_d = theta (1 + k1 t^2 + ...) by Newton's method (D = k1..k4)."""
    k1, k2, k3, k4 = np.ravel(D) if D is not None else (0, 0, 0, 0)
    th = td.copy()
    for _ in range(iters):
        t2 = th * th
        f = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4) - td
        df = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
        th = th - f / df
    return th


def build_maps(K, D, size, face_px):
    """Lookup tables (map_x, map_y) from fisheye pixels into the atlas of the five faces laid side by side."""
    w, h = size
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    td = np.hypot(x, y)
    th = theta_from_distorted(td, D)
    phi = np.arctan2(y, x)
    # the ray in OpenCV camera axes (x right, y down, z forward) -> MuJoCo camera axes (x right, y up, -z forward)
    ray = np.stack([np.sin(th) * np.cos(phi), -np.sin(th) * np.sin(phi), -np.cos(th)], -1)
    best = np.full((h, w), -np.inf)
    mx, my = np.full((h, w), -1.0, np.float32), np.full((h, w), -1.0, np.float32)
    n = face_px
    for k, name in enumerate(FACES):
        r = ray @ _R(*_ROT[name])                        # the ray in this face's frame
        fwd = -r[..., 2]
        use = (fwd > best) & (fwd > 1e-6) & (th < np.radians(95))
        px = n / 2 + n / 2 * r[..., 0] / np.maximum(fwd, 1e-6)
        py = n / 2 - n / 2 * r[..., 1] / np.maximum(fwd, 1e-6)
        inside = use & (px >= 0) & (px <= n) & (py >= 0) & (py <= n)
        mx[inside] = (k * n + np.clip(px[inside], 0, n - 1.001)).astype(np.float32)
        my[inside] = np.clip(py[inside], 0, n - 1.001).astype(np.float32)
        best[inside] = fwd[inside]
    return mx, my


class FisheyeCamera:
    def __init__(self, m, size=(640, 480), K=None, D=None, face_px=576):
        import mujoco
        self.m, self.size = m, size
        self.K = default_K(*size) if K is None else np.asarray(K, float)
        self.D = None if D is None else np.asarray(D, float)
        # the renderer needs an offscreen buffer as big as a face
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, face_px)
        m.vis.global_.offheight = max(m.vis.global_.offheight, face_px)
        self.n = face_px
        self.renderer = mujoco.Renderer(m, face_px, face_px)
        self.cams = [m.camera(f"eye_{f}").id for f in FACES]
        self.map_x, self.map_y = build_maps(self.K, self.D, size, face_px)
        self.atlas = np.zeros((face_px, face_px * len(FACES), 3), np.uint8)
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False      # the cliff beams are invisible light
        self.opt = opt

    def render(self, d):
        """The fisheye frame (BGR, like the real camera through OpenCV)."""
        import cv2
        n = self.n
        for k, cid in enumerate(self.cams):
            self.renderer.update_scene(d, camera=cid, scene_option=self.opt)
            self.atlas[:, k * n:(k + 1) * n] = self.renderer.render()
        img = cv2.remap(self.atlas, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return np.ascontiguousarray(img[:, :, ::-1])

    def project(self, p_cam):
        """Where a point in the eye camera's MuJoCo frame (x right, y up, -z forward) lands in the image (u, v)."""
        x, y, z = p_cam[0], -p_cam[1], -p_cam[2]
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        k = np.ravel(self.D) if self.D is not None else np.zeros(4)
        td = th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6 + k[3] * th ** 8)
        s = td / r if r > 1e-12 else 0.0
        return self.K[0, 0] * x * s + self.K[0, 2], self.K[1, 1] * y * s + self.K[1, 2]

    def close(self):
        self.renderer.close()
