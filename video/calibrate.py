'''
Camera calibration for top-down videos of the vine on its board.

The board is the reference: its four corners have known positions in the board frame (mm, origin
at one corner, x along the long edge, y up, the same frame as envs/*.txt and the sim scenes). Their
pixel positions give a homography from the image to the board plane. Extra correspondences (wall
or obstacle corners whose board position is known) refine it and check it: with only the four
corners the fit is exact and has no residual to report.

Two optional extras:
  * lens distortion: `intrinsics` (from `python video/calibrate.py intrinsics ...`, a checkerboard
    video) undistorts every frame before the homography is applied.
  * height: the vine's centerline and the boxes' top faces are above the board, so a camera that
    is not straight above them sees them shifted outward (parallax). Calibration.lift() undoes
    that for points at a known height, exactly when the intrinsics are known (the camera pose then
    follows from the board corners) and approximately from `camera_height_mm` otherwise.

On a pegboard (`pegboard_pitch_mm`), the holes replace the corners: every hole is a known grid
point, so the calibration needs no clicking and reports how well a homography fits hundreds of
points (lens distortion shows up there).

The four corners need not be the board's own: `corners_mm` names any rectangle (or quadrilateral)
of known board coordinates, e.g. the inner corners of the boundary walls.

Pixel coordinates are (u, v) = (column, row) of the raw video frame.

Run from the repo root (actvine env):
    python video/calibrate.py grid  trial.json            # frame with a pixel grid, to read corners off
    python video/calibrate.py check trial.json            # fit, report residuals, draw the check image
    python video/calibrate.py intrinsics board.mp4 --pattern 9 6 --square-mm 25 --out intrinsics.json
'''

import argparse, json, os
from dataclasses import dataclass, field

import cv2
import numpy as np


#------------------------------------------------------------------ calibration

@dataclass
class Calibration:
    H: np.ndarray                          # 3x3, undistorted pixels -> board mm (z = 0)
    board_mm: tuple                        # (width, height) of the board
    image_size: tuple                      # (width, height) of the frames [px]
    K: np.ndarray = None                   # 3x3 camera matrix (optional)
    dist: np.ndarray = None                # distortion coefficients (optional)
    camera_height_mm: float = None         # used by lift() when K is unknown
    residuals_mm: list = field(default_factory=list)   # per correspondence, after the fit

    # ---- points

    def undistort_px(self, px):
        px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        if self.K is None:
            return px
        return cv2.undistortPoints(px[:, None], self.K, self.dist, P=self.K).reshape(-1, 2)

    def px_to_mm(self, px):
        '''Raw pixel (u, v) -> board (x, y) mm on the board plane.'''
        return _apply_h(self.H, self.undistort_px(px))

    def mm_to_px(self, mm):
        '''Board (x, y) mm on the board plane -> undistorted pixel (u, v).'''
        return _apply_h(np.linalg.inv(self.H), mm)

    def lift(self, mm, z_mm):
        '''
        Points seen at board position `mm` (what px_to_mm gives) that are really at height z_mm
        above the board -> their true (x, y) [mm]. A no-op when neither the intrinsics nor the
        camera height are known.
        '''
        mm = np.asarray(mm, dtype=np.float64).reshape(-1, 2)
        if not z_mm:
            return mm
        if self.K is not None:
            # the plane z = z_mm images through K [r1 r2 z*r3 + t]
            R, t = self.camera_pose()
            Hz = self.K @ np.column_stack([R[:, 0], R[:, 1], z_mm * R[:, 2] + t])
            return _apply_h(np.linalg.inv(Hz), self.mm_to_px(mm))
        if self.camera_height_mm:
            # nearly overhead camera: points at height z appear pushed away from the point below the
            # camera (the image of the principal point) by Zc / (Zc - z)
            nadir = self.px_to_mm([[self.image_size[0] / 2, self.image_size[1] / 2]])[0]
            return nadir + (mm - nadir) * (self.camera_height_mm - z_mm) / self.camera_height_mm
        return mm

    def camera_pose(self):
        '''(R, t) of the board in the camera frame, from the homography and K.'''
        A = np.linalg.inv(self.K) @ np.linalg.inv(self.H)   # board mm -> normalized camera
        s = 1.0 / np.linalg.norm(A[:, 0])
        if A[2, 2] < 0:                                      # board in front of the camera
            s = -s
        r1, r2, t = s * A[:, 0], s * A[:, 1], s * A[:, 2]
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)                          # nearest rotation
        return U @ Vt, t

    # ---- images

    def rect_matrix(self, mm_per_px):
        '''Board mm -> rectified-image pixel: row 0 is the far (y = height) edge, so y points up.'''
        W, Hb = self.board_mm
        return np.array([[1 / mm_per_px, 0, 0], [0, -1 / mm_per_px, Hb / mm_per_px], [0, 0, 1]])

    def rectify(self, frame, mm_per_px):
        '''Top-down image of the board plane at mm_per_px.'''
        if self.K is not None:
            frame = cv2.undistort(frame, self.K, self.dist)
        W, Hb = self.board_mm
        size = (int(round(W / mm_per_px)), int(round(Hb / mm_per_px)))
        return cv2.warpPerspective(frame, self.rect_matrix(mm_per_px) @ self.H, size,
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---- io

    def to_dict(self):
        return {"H": self.H.tolist(), "board_mm": list(self.board_mm), "image_size": list(self.image_size),
                "K": None if self.K is None else self.K.tolist(),
                "dist": None if self.dist is None else self.dist.ravel().tolist(),
                "camera_height_mm": self.camera_height_mm, "residuals_mm": self.residuals_mm}

    @staticmethod
    def from_dict(d):
        return Calibration(H=np.array(d["H"]), board_mm=tuple(d["board_mm"]), image_size=tuple(d["image_size"]),
                           K=None if d.get("K") is None else np.array(d["K"]),
                           dist=None if d.get("dist") is None else np.array(d["dist"]),
                           camera_height_mm=d.get("camera_height_mm"), residuals_mm=d.get("residuals_mm", []))


def _apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H).T
    return q[:, :2] / q[:, 2:3]


def board_corners_mm(board_mm):
    '''The board's corners in the order corners_px lists them: (0,0), (W,0), (W,H), (0,H).'''
    W, H = board_mm
    return np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)


def detect_board(frame):
    '''
    Pixel corners of the board, found as the largest 4-sided contour of the image's brighter (or
    darker) region. Ordered as board_corners_mm assumes the camera sees it: board (0,0) at the
    image's bottom left, x to the right, y up the image. Use `corner_rotation` in the trial if the
    camera is turned. Returns None if nothing board-like is found.
    '''
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    best, best_area = None, 0.1 * gray.size
    for invert in (False, True):
        _, bw = cv2.threshold(gray, 0, 255, (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            hull = cv2.convexHull(c)
            quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            area = cv2.contourArea(quad)
            # a board fills a good part of the frame but does not touch every image edge
            if len(quad) == 4 and best_area < area < 0.98 * gray.size:
                best, best_area = quad.reshape(4, 2).astype(np.float64), area
    if best is None:
        return None
    # order: bottom-left, bottom-right, top-right, top-left in the image
    c = best.mean(axis=0)
    ang = np.arctan2(-(best[:, 1] - c[1]), best[:, 0] - c[0])    # image y points down
    best = best[np.argsort(ang)]                                  # counterclockwise from -pi
    start = np.argmin(best[:, 0] - best[:, 1])                    # bottom left: small u, large v
    return np.roll(best, -start, axis=0)


def detect_holes(gray):
    '''Centers [px] of small, round, dark blobs (pegboard holes).'''
    d = cv2.subtract(cv2.GaussianBlur(gray, (0, 0), 6), gray)          # darker than the surroundings
    _, bw = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw)
    area = st[1:, cv2.CC_STAT_AREA].astype(float)
    w, h = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    med = np.median(area[area >= 3]) if (area >= 3).any() else 0
    round_ = (np.maximum(w, h) <= 1.6 * np.minimum(w, h) + 1) & (area >= 0.5 * w * h)
    keep = (area >= max(3, 0.4 * med)) & (area <= 2.0 * med) & round_
    return cen[1:][keep]


def pegboard_lattice(frame, pitch_mm=25.4):
    '''
    Correspondences (px, grid index) from the holes of a pegboard. Every hole gets its integer
    (column, row) by stepping hole to hole from a well-surrounded hole near the image center: a
    neighbor is accepted only where the step from its predecessor predicts it, so marks off the
    board (a tape measure, the frame) are not reached across the gap. Holes that a homography over
    all of them does not fit are dropped.
    Board frame: the bottom-left hole of the rows and columns that are at least 3/4 in view is at
    (pitch, pitch), x along the columns to the right, y up the rows, so the board is (columns + 1,
    rows + 1) pitches. Holes of partly hidden outer rows are still used, outside that range.
    Returns (px (n, 2), mm (n, 2), board_mm, (columns, rows)).
    '''
    from scipy.spatial import cKDTree
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = detect_holes(gray)
    if len(pts) < 50:
        raise RuntimeError(f"only {len(pts)} pegboard holes found")
    tree = cKDTree(pts)
    dist, nb = tree.query(pts, k=5)
    pitch_px = float(np.median(dist[:, 1]))
    # grid orientation from the nearest-neighbor steps (angle mod 90 degrees)
    steps = (pts[nb[:, 1:]] - pts[:, None]).reshape(-1, 2)
    steps = steps[np.abs(np.linalg.norm(steps, axis=1) - pitch_px) < 0.2 * pitch_px]
    ang = np.arctan2(steps[:, 1], steps[:, 0])
    th = np.angle(np.mean(np.exp(4j * ang))) / 4
    col = pitch_px * np.array([np.cos(th), np.sin(th)])        # one column right in the image
    row = pitch_px * np.array([np.sin(th), -np.cos(th)])       # one row up in the image (v down)

    # seed: near the center, with all four neighbors where the grid puts them
    full = np.array([all(tree.query(p + v)[0] < 0.2 * pitch_px for v in (col, -col, row, -row)) for p in pts])
    center = np.array(gray.shape[::-1]) / 2
    seed = int(np.argmin(np.linalg.norm(pts - center, axis=1) + 1e9 * ~full))
    index = {seed: (0, 0)}
    local = {seed: (col, row)}
    queue = [seed]
    while queue:
        k = queue.pop(0)
        i, j = index[k]
        c, r = local[k]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d, m = tree.query(pts[k] + di * c + dj * r)
            if d < 0.25 * pitch_px and m not in index:
                index[m] = (i + di, j + dj)
                step = pts[m] - pts[k]
                local[m] = (step * di, r) if di else (c, step * dj)
                queue.append(m)
    ids = np.array(list(index))
    ij = np.array([index[k] for k in ids], dtype=np.float64)
    px = pts[ids]
    H, inl = cv2.findHomography(ij, px, cv2.RANSAC, 0.3 * pitch_px)
    ok = inl.ravel() > 0
    ij, px = ij[ok], px[ok]
    # the outermost rows and columns can hide under a frame's lip, partly and differently in each
    # video: anchor on the rows and columns that are at least 3/4 visible
    lo, hi = [], []
    for a in (0, 1):
        idx, count = np.unique(ij[:, a], return_counts=True)
        full = idx[count >= 0.75 * count.max()]
        lo.append(full.min()); hi.append(full.max())
    lo, hi = np.array(lo), np.array(hi)
    shape = tuple(int(v) for v in hi - lo + 1)
    mm = (ij - lo + 1) * pitch_mm
    return px, mm, ((shape[0] + 1) * pitch_mm, (shape[1] + 1) * pitch_mm), shape


def fit_radial_distortion(px, mm, image_size):
    '''
    (K, dist) of a two-term radial distortion about the image center that makes the board points
    fit one homography best. K is only a normalization here (focal length = image width), so the
    result is for undistorting this video, not a real camera calibration.
    '''
    from scipy.optimize import least_squares
    w, h = image_size
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1.0]])

    def residual(k):
        dist = np.array([k[0], k[1], 0, 0, 0])
        u = cv2.undistortPoints(px[:, None], K, dist, P=K).reshape(-1, 2)
        H, _ = cv2.findHomography(u, mm, 0)
        return (_apply_h(H, u) - mm).ravel()

    k = least_squares(residual, [0.0, 0.0], x_scale=0.1).x
    return K, np.array([k[0], k[1], 0, 0, 0])


def calibrate(frame, trial):
    '''
    Calibration for one trial (see video/trial_example.json for the fields):
      board_mm, corners_px ("auto" or 4 points in board_corners_mm order), corner_rotation,
      corners_mm (what the 4 corners are in board mm, if not the board's own corners: e.g. the
      inner corners of the boundary walls, which is what "auto" finds when the walls look like the
      floor), extra_points [{"px": [u, v], "mm": [x, y]}], intrinsics (json path), camera_height_mm.
    With pegboard_pitch_mm set, the corners are not used: every hole of the pegboard is a
    correspondence (pegboard_lattice), and residuals_mm is [median, 95th percentile, max].
    '''
    K = dist = None
    if trial.get("intrinsics"):
        with open(trial["intrinsics"]) as f:
            d = json.load(f)
        K, dist = np.array(d["K"]), np.array(d["dist"])

    if trial.get("pegboard_pitch_mm"):
        px, mm, board_mm, shape = pegboard_lattice(frame, trial["pegboard_pitch_mm"])
        if trial.get("origin_shift_mm"):     # move the board frame's origin (when the lattice anchors
            mm = mm + np.asarray(trial["origin_shift_mm"], dtype=np.float64)   # above things of interest)
        if K is None and trial.get("fit_distortion", True):
            K, dist = fit_radial_distortion(px, mm, (frame.shape[1], frame.shape[0]))
        cal = Calibration(H=np.eye(3), board_mm=tuple(trial.get("board_mm") or board_mm),
                          image_size=(frame.shape[1], frame.shape[0]), K=K, dist=dist,
                          camera_height_mm=trial.get("camera_height_mm"))
        cal.H, _ = cv2.findHomography(cal.undistort_px(px), mm, 0)
        res = np.linalg.norm(cal.px_to_mm(px) - mm, axis=1)
        cal.residuals_mm = [round(float(np.median(res)), 2), round(float(np.percentile(res, 95)), 2),
                            round(float(res.max()), 2)]
        cal.holes = (len(px), shape)
        cal.corners_px = cal.mm_to_px(board_corners_mm(cal.board_mm))
        return cal

    corners = trial.get("corners_px", "auto")
    if isinstance(corners, str):
        corners = detect_board(cv2.undistort(frame, K, dist) if K is not None else frame)
        if corners is None:
            raise RuntimeError("no board found: give corners_px in the trial (see `calibrate.py grid`)")
        corners = np.roll(corners, -int(trial.get("corner_rotation", 0)), axis=0)
        undistorted = True
    else:
        corners = np.asarray(corners, dtype=np.float64)
        undistorted = False

    cal = Calibration(H=np.eye(3), board_mm=tuple(trial["board_mm"]), image_size=(frame.shape[1], frame.shape[0]),
                      K=K, dist=dist, camera_height_mm=trial.get("camera_height_mm"))
    src = corners if undistorted else cal.undistort_px(corners)
    dst = np.asarray(trial["corners_mm"], dtype=np.float64) if trial.get("corners_mm") else board_corners_mm(cal.board_mm)
    extra = trial.get("extra_points", [])
    if extra:
        src = np.vstack([src, cal.undistort_px([p["px"] for p in extra])])
        dst = np.vstack([dst, [p["mm"] for p in extra]])
    cal.H, _ = cv2.findHomography(src, dst, 0)
    cal.residuals_mm = np.linalg.norm(_apply_h(cal.H, src) - dst, axis=1).round(2).tolist()
    cal.corners_px = cal.mm_to_px(board_corners_mm(cal.board_mm))
    return cal


#------------------------------------------------------------------ intrinsics

def calibrate_intrinsics(video_path, pattern, square_mm, max_views=40):
    '''Camera matrix and distortion from a video of a checkerboard (inner corners `pattern`).'''
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size = [], [], None
    for idx in np.linspace(0, max(n - 1, 0), min(n, 4 * max_views)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj_pts.append(obj)
            img_pts.append(corners)
            if len(obj_pts) >= max_views:
                break
    cap.release()
    if len(obj_pts) < 5:
        raise RuntimeError(f"checkerboard found in only {len(obj_pts)} frames")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms),
            "views": len(obj_pts), "image_size": list(size)}


#------------------------------------------------------------------ trials and frames

def load_trial(path):
    '''A trial json, with relative paths in it resolved against the json's folder.'''
    with open(path) as f:
        trial = json.load(f)
    here = os.path.dirname(os.path.abspath(path))
    for key in ("video", "intrinsics", "env"):
        if trial.get(key) and not os.path.isabs(trial[key]):
            trial[key] = os.path.normpath(os.path.join(here, trial[key]))
    trial.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return trial


def read_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video_path}")
    return frame


def walls_mm(trial):
    '''Static boxes [x1, y1, x2, y2] in board mm: the trial's own list, else its env file's.'''
    if "walls_mm" in trial:
        walls = trial["walls_mm"]
    elif trial.get("env"):
        from kinodynamic.env_loader import load_box_config
        walls = load_box_config(trial["env"])["obstacles"]
    else:
        walls = []
    # the env files list some corners as (x1, y2, x2, y1)
    return [[min(w[0], w[2]), min(w[1], w[3]), max(w[0], w[2]), max(w[1], w[3])] for w in walls]


def grid_image(frame, step=50):
    '''The frame with a labelled pixel grid, to read corner coordinates off in an image viewer.'''
    img = frame.copy()
    h, w = img.shape[:2]
    for u in range(0, w, step):
        cv2.line(img, (u, 0), (u, h), (0, 255, 255) if u % (4 * step) == 0 else (0, 160, 160), 1)
        if u % (2 * step) == 0:
            cv2.putText(img, str(u), (u + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for v in range(0, h, step):
        cv2.line(img, (0, v), (w, v), (0, 255, 255) if v % (4 * step) == 0 else (0, 160, 160), 1)
        if v % (2 * step) == 0:
            cv2.putText(img, str(v), (2, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return img


def check_image(frame, cal, walls, mm_per_px=2.0):
    '''
    Two panels: the raw frame with the board outline and walls projected into it, and the
    rectified board with a 100 mm grid and the walls. Misaligned walls mean a bad calibration.
    '''
    raw = frame.copy()
    if cal.K is not None:
        raw = cv2.undistort(raw, cal.K, cal.dist)
    outline = cal.mm_to_px(board_corners_mm(cal.board_mm))
    cv2.polylines(raw, [outline.round().astype(np.int32)], True, (0, 0, 255), 2)
    for w in walls:
        pts = cal.mm_to_px([[w[0], w[1]], [w[2], w[1]], [w[2], w[3]], [w[0], w[3]]])
        cv2.polylines(raw, [pts.round().astype(np.int32)], True, (255, 0, 255), 2)
    for k, (u, v) in enumerate(outline):
        cv2.putText(raw, ["(0,0)", "(W,0)", "(W,H)", "(0,H)"][k], (int(u) + 4, int(v) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    rect = cal.rectify(frame, mm_per_px)
    A = cal.rect_matrix(mm_per_px)
    W, Hb = cal.board_mm
    for x in np.arange(0, W + 1, 100):
        (u, _), = _apply_h(A, [[x, 0]])
        cv2.line(rect, (int(u), 0), (int(u), rect.shape[0]), (200, 200, 0), 1)
    for y in np.arange(0, Hb + 1, 100):
        (_, v), = _apply_h(A, [[0, y]])
        cv2.line(rect, (0, int(v)), (rect.shape[1], int(v)), (200, 200, 0), 1)
    for w in walls:
        p = _apply_h(A, [[w[0], w[1]], [w[2], w[3]]]).round().astype(int)
        cv2.rectangle(rect, tuple(p[0]), tuple(p[1]), (255, 0, 255), 2)
    # same height for both panels
    scale = raw.shape[0] / rect.shape[0]
    rect = cv2.resize(rect, (int(rect.shape[1] * scale), raw.shape[0]))
    return np.hstack([raw, rect])


def calibrate_trial(trial, out_dir):
    '''Calibrate on the trial's reference frame; writes calib.json and calib_check.png.'''
    frame = read_frame(trial["video"], trial.get("reference_frame", 0))
    cal = calibrate(frame, trial)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calib.json"), "w") as f:
        json.dump(cal.to_dict(), f, indent=1)
    cv2.imwrite(os.path.join(out_dir, "calib_check.png"), check_image(frame, cal, walls_mm(trial)))
    return cal


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid", help="write the reference frame with a pixel grid")
    g.add_argument("trial")
    g.add_argument("--frame", type=int, default=None)
    g.add_argument("--out", default=None)
    c = sub.add_parser("check", help="calibrate and write calib.json + calib_check.png")
    c.add_argument("trial")
    c.add_argument("--out", default=None)
    i = sub.add_parser("intrinsics", help="camera matrix from a checkerboard video")
    i.add_argument("video")
    i.add_argument("--pattern", type=int, nargs=2, default=(9, 6), help="inner corners (cols rows)")
    i.add_argument("--square-mm", type=float, required=True)
    i.add_argument("--out", default="intrinsics.json")
    args = parser.parse_args()

    if args.cmd == "intrinsics":
        d = calibrate_intrinsics(args.video, tuple(args.pattern), args.square_mm)
        with open(args.out, "w") as f:
            json.dump(d, f, indent=1)
        print(f"rms reprojection {d['rms_px']:.3f} px over {d['views']} views -> {args.out}")
        return

    trial = load_trial(args.trial)
    out = args.out or os.path.join("out", "video", trial["name"])
    os.makedirs(out, exist_ok=True)
    if args.cmd == "grid":
        frame = read_frame(trial["video"], args.frame if args.frame is not None else trial.get("reference_frame", 0))
        path = os.path.join(out, "grid.png")
        cv2.imwrite(path, grid_image(frame))
        corners = detect_board(frame)
        print("wrote", path)
        if corners is not None:
            print("auto-detected board corners (BL, BR, TR, TL):", corners.round(1).tolist())
    else:
        cal = calibrate_trial(trial, out)
        print("residuals [mm] (corners first, then extra_points):", cal.residuals_mm)
        print("board corners [px]:", cal.corners_px.round(1).tolist())
        print("wrote", os.path.join(out, "calib.json"), "and calib_check.png")


if __name__ == "__main__":
    main()
