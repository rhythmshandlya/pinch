"""Calibrate Pinch's wide (fisheye) camera once, after assembly or after replacing the camera.

    sudo systemctl stop pinch                                   # the firmware holds the camera
    sudo /opt/pinch/venv/bin/python /opt/pinch/robot/calibrate_camera.py
    sudo systemctl start pinch

Print a checkerboard with 10 x 7 squares (9 x 6 inner corners) of 25 mm (OpenCV's standard pattern;
any printer at 100 % scale; tape it to something flat). Hold it in front of Pinch's face and move it
slowly: near and far, tilted, and especially into the corners of the view, where a fisheye bends the most.
Every time the board is found somewhere new a view is kept (it prints a line). 25 views take
about a minute. Then it solves for the lens (focal length, centre and the 4 fisheye distortion terms) and
saves them to /var/lib/pinch/camera_intrinsics.json (PINCH_CAMERA_INTRINSICS overrides).

    --images DIR    calibrate from photos you took instead (any .png/.jpg of the board)
    --selftest      check the solver on synthetic views of a known lens (no camera needed)

Without this file the firmware never uses the camera to say where Pinch is on the desk (robot/vision.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.vision import INTRINSICS_FILE  # noqa: E402

BOARD = (9, 6)              # inner corners
SQUARE_M = 0.025
WIDTH, HEIGHT = 640, 480    # the firmware's capture size (robot/hw_real.py)


def board_points(board=BOARD, square=SQUARE_M):
    obj = np.zeros((1, board[0] * board[1], 3), np.float64)
    obj[0, :, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    return obj


def solve(img_points, size, board=BOARD, square=SQUARE_M):
    """Fisheye intrinsics from the corner sets of several views -> (K, D, rms reprojection error in px)."""
    obj = [board_points(board, square)] * len(img_points)
    imgs = [np.asarray(p, np.float64).reshape(1, -1, 2) for p in img_points]
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_CHECK_COND
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj, imgs, size, K, D, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7))
    return K, D, float(rms)


def save(K, D, rms, views, size, path=None):
    p = Path(path or INTRINSICS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(model="fisheye", width=size[0], height=size[1], K=K.tolist(),
                                 D=D.ravel().tolist(), rms_px=rms, views=views,
                                 date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    return p


def find_corners(bgr, board=BOARD):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(g, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def capture(n_views=25, timeout=300.0, device=0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit("camera not found (is the firmware still running? sudo systemctl stop pinch)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    views, centres, t0 = [], [], time.monotonic()
    print("show the checkerboard; move it around, into the corners too")
    try:
        while len(views) < n_views and time.monotonic() - t0 < timeout:
            ok, frame = cap.read()
            if not ok:
                continue
            c = find_corners(frame)
            if c is None:
                continue
            centre, spread = c.reshape(-1, 2).mean(0), np.ptp(c.reshape(-1, 2), 0).mean()
            if any(np.hypot(*(centre - m)) < 40 and abs(spread - s) < 20 for m, s in centres):
                continue                                        # too like a view we have
            views.append(c)
            centres.append((centre, spread))
            print(f"  view {len(views)}/{n_views} at ({centre[0]:.0f}, {centre[1]:.0f})")
        size = (frame.shape[1], frame.shape[0])
    finally:
        cap.release()
    return views, size


def selftest(seed=0):
    """Synthetic views of a known fisheye lens -> the solver must recover it."""
    rng = np.random.default_rng(seed)
    K_true = np.array([[230.0, 0, 322.0], [0, 231.0, 238.0], [0, 0, 1]])
    D_true = np.array([[0.05], [-0.02], [0.01], [-0.004]])
    obj = board_points()
    views = []
    while len(views) < 25:
        rvec = rng.normal(0, 0.35, 3)
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.03), rng.uniform(0.15, 0.35)])
        pts, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K_true, D_true)
        pts = pts.reshape(-1, 2)
        if (pts.min() < 5) or (pts[:, 0].max() > WIDTH - 5) or (pts[:, 1].max() > HEIGHT - 5):
            continue
        views.append(pts + rng.normal(0, 0.15, pts.shape))
    K, D, rms = solve(views, (WIDTH, HEIGHT))
    return K_true, D_true, K, D, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="calibrate from photos in this folder")
    ap.add_argument("--views", type=int, default=25)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        K0, D0, K, D, rms = selftest()
        print(f"focal {K[0, 0]:.1f} (true {K0[0, 0]:.1f}), centre ({K[0, 2]:.1f}, {K[1, 2]:.1f}), rms {rms:.2f} px")
        return 0
    if a.images:
        views, size = [], None
        for f in sorted(Path(a.images).iterdir()):
            img = cv2.imread(str(f))
            if img is None:
                continue
            size = (img.shape[1], img.shape[0])
            c = find_corners(img)
            print(f"  {f.name}: {'board found' if c is not None else 'no board'}")
            if c is not None:
                views.append(c)
    else:
        views, size = capture(a.views)
    if len(views) < 10:
        print(f"only {len(views)} usable views - need at least 10")
        return 1
    K, D, rms = solve(views, size)
    print(f"focal {K[0, 0]:.1f} px, centre ({K[0, 2]:.1f}, {K[1, 2]:.1f}), distortion {D.ravel().round(4)}, "
          f"rms {rms:.2f} px")
    if rms > 1.0:
        print("  ! rms above 1 px: some views were blurred or the board isn't flat - consider redoing it")
    print(f"saved {save(K, D, rms, len(views), size)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
