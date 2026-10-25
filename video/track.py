'''
Vine centerlines and box poses from a top-down video of one trial.

Following DiffVine's pipeline: the vine is whatever differs from a reference frame taken before it
grows, cleaned up with blurring and morphological opening/closing, then skeletonized into a
curve from the base to the tip and resampled. Two additions for pushing:
  * the movable boxes are tracked by template matching (their appearance in the reference frame,
    searched over small translations and rotations around their last pose), and their footprints
    are removed from the vine's mask;
  * the background the vine is compared against is the reference frame with the boxes painted
    out, so the table a box uncovers when it moves is not mistaken for vine.

Everything is in the board frame of video/calibrate.py (mm, y up), and the vine is measured on a
rectified top-down image of the board.

Run from the repo root (actvine env):
    python video/track.py trial.json                 # writes out/video/<name>/
    python video/track.py trial.json --stride 3 --end 900
Outputs: tracks.npz (load with load_tracks), overlay.mp4 (what was tracked, on the rectified video),
summary.png (centerlines over time, length vs time with the growth rate, box paths), calib.json,
calib_check.png.
'''

import argparse, json, math, os, sys, time

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import Calibration, calibrate_trial, load_trial, read_frame, walls_mm, _apply_h

# defaults for the trial fields that tune the tracking (see trial_example.json)
DEFAULTS = dict(
    mm_per_px=2.0,          # resolution of the rectified board image the vine is found in
    diff_mode="abs",        # "abs", "brighter" (light vine on a dark table: ignores shadows), "darker",
                            # "saturation" (a pale or clear vine on a colored board)
    diff_threshold=30,      # 0..255, on the blurred per-channel difference to the background (on the
                            # saturation drop for "saturation")
    vine_radius_mm=50.0,    # a first guess, only used to size filters; the measured one is reported
    vine_height_mm=None,    # height of the seen centerline for parallax (None: the measured radius)
    smooth_mm=10.0,         # Gaussian smoothing of the centerline along its length
    spacing_mm=5.0,         # spacing of the saved centerline points
    base_tol_mm=80.0,       # the vine's mask must come this close to the base
    gap_mm=60.0,            # pieces of the vine's mask this close are joined (a clear tube's dropouts,
                            # the part a box hides)
    base_cut_mm=150.0,      # mask behind the base (against its heading) within this distance is
                            # dropped: the mount
    box_margin_mm=6.0,      # removed around each box footprint from the vine's mask
    box_search_mm=25.0,     # box tracking: translation searched per processed frame
    box_search_deg=8.0,     #               rotation searched per processed frame
    box_res_mm=1.0,         #               resolution of the box templates
    box_min_score=0.5,      # below this the box keeps its last pose and the frame is flagged
    box_channel="gray",     # what box templates match on: "gray", or "hue" (the box's hue differs from
                            # the board's: a clean silhouette without the board's texture)
    box_margin_tpl_mm=15.0, # board around the box in its template (its edges carry the match)
    bg_paint_mm=20.0,       # boxes are painted out of the background this far past their footprint
                            # (their sides show in an oblique view)
    bg_rate=0.05,           # "saturation" mode: the background follows slow lighting/exposure changes
                            # at this rate per processed frame, away from the vine and the boxes
    roi_mm=None,            # polygon [[x, y], ...] in board mm: the vine is only looked for inside it
                            # (when the board_mm of the trial extends the top-down image past the board)
    exclude_obb_mm=None,    # oriented boxes [cx, cy, theta_rad, hw, hh] in board mm where the vine cannot
                            # be (a standing panel and what is behind it), dropped from its mask
)


#------------------------------------------------------------------ trial geometry

def trial_boxes(trial):
    '''
    Movable boxes: [{"size_mm": (w, h), "pose": (cx, cy, theta_rad), "mass_kg", "height_mm"}].
    From the trial's "boxes" list (pose_mm = [cx, cy, theta_deg], or "auto" (pose None): found by
    its hue on the reference frame), else the env file's
    dynamic_obstacles (axis aligned).
    '''
    out = []
    if "boxes" in trial:
        for b in trial["boxes"]:
            pose = None
            if b.get("pose_mm", "auto") != "auto":
                cx, cy, deg = b["pose_mm"]
                pose = (cx, cy, math.radians(deg))
            out.append(dict(size_mm=tuple(b["size_mm"]), pose=pose,
                            mass_kg=b.get("mass_kg"), height_mm=b.get("height_mm", 0.0)))
    elif trial.get("env"):
        from kinodynamic.env_loader import load_box_config
        cfg = load_box_config(trial["env"])
        for (x1, y1, x2, y2), (m,) in zip(cfg["dynamic_obstacles"], cfg["dynamic_object_masses"]):
            out.append(dict(size_mm=(abs(x2 - x1), abs(y2 - y1)), pose=((x1 + x2) / 2, (y1 + y2) / 2, 0.0),
                            mass_kg=m, height_mm=0.0))
    return out


def trial_base(trial, cal):
    '''(x, y, heading_rad) of the vine's base in board mm. heading None = measure it from the vine.'''
    if "base" in trial:
        b = trial["base"]
        xy = cal.px_to_mm([b["px"]])[0] if "px" in b else np.array(b["mm"], dtype=float)
        heading = math.radians(b["heading_deg"]) if b.get("heading_deg") is not None else None
        return float(xy[0]), float(xy[1]), heading
    if trial.get("env"):
        from kinodynamic.env_loader import load_box_config
        x, y, h = load_box_config(trial["env"])["start"]
        return x, y, h
    raise ValueError("the trial needs a base (or an env file with a start)")


def box_corners(pose, size):
    cx, cy, th = pose
    hw, hh = size[0] / 2, size[1] / 2
    c, s = math.cos(th), math.sin(th)
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


#------------------------------------------------------------------ boxes

def box_patch(gray_ud, cal, center, theta, size_mm, res):
    '''
    The board around `center`, sampled in a frame turned by `theta`: a (size_mm[1]/res,
    size_mm[0]/res) image whose middle is `center`, straight from the undistorted frame.
    '''
    w, h = int(round(size_mm[0] / res)), int(round(size_mm[1] / res))
    c, s = math.cos(theta), math.sin(theta)
    # patch pixel (j, i) -> local mm (x right, y up) -> board mm -> frame px
    to_local = np.array([[res, 0, -res * (w - 1) / 2], [0, -res, res * (h - 1) / 2], [0, 0, 1]])
    to_board = np.array([[c, -s, center[0]], [s, c, center[1]], [0, 0, 1]])
    M = np.linalg.inv(cal.H) @ to_board @ to_local
    return cv2.warpPerspective(gray_ud, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def refine_box(gray_ud, cal, pose, size, res):
    '''
    The box's pose in the reference frame, from its outline near the given pose: the region that
    contrasts with the table around the given center, fitted with a rectangle. Keeps the given
    pose if that region is not about the right size.
    '''
    cx, cy, th = pose
    span = 1.8 * max(size)
    patch = cv2.GaussianBlur(box_patch(gray_ud, cal, (cx, cy), 0.0, (span, span), res), (5, 5), 0)
    mid = (patch.shape[0] // 2, patch.shape[1] // 2)
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, bw = cv2.threshold(patch, 0, 255, flag + cv2.THRESH_OTSU)
        n, lab = cv2.connectedComponents(bw)
        k = lab[mid]
        if k == 0:
            continue
        ys, xs = np.nonzero(lab == k)
        area_mm2 = len(xs) * res * res
        if not 0.7 < area_mm2 / (size[0] * size[1]) < 1.3:
            continue
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        corners = cv2.boxPoints(rect)                      # patch px
        # patch px -> board mm
        loc = np.column_stack([(corners[:, 0] - (patch.shape[1] - 1) / 2) * res,
                               -(corners[:, 1] - (patch.shape[0] - 1) / 2) * res]) + [cx, cy]
        center = loc.mean(axis=0)
        edge = loc[1] - loc[0]
        a = math.atan2(edge[1], edge[0])
        a = th + (a - th + math.pi / 4) % (math.pi / 2) - math.pi / 4   # the 90-degree turn nearest the given one
        return (float(center[0]), float(center[1]), float(a)), True
    return pose, False


class BoxTracker:
    '''Template tracking of one box: its reference-frame appearance, searched around its last pose.'''

    def __init__(self, gray_ud, cal, pose, size, p):
        self.cal, self.size, self.p = cal, size, p
        self.margin = p["box_margin_tpl_mm"]
        self.pose = pose
        self.template = self._patch(gray_ud, pose, extra=0.0)

    def _patch(self, gray_ud, pose, extra):
        s = (self.size[0] + 2 * self.margin + 2 * extra, self.size[1] + 2 * self.margin + 2 * extra)
        return cv2.GaussianBlur(box_patch(gray_ud, self.cal, pose[:2], pose[2], s, self.p["box_res_mm"]), (3, 3), 0)

    def update(self, gray_ud):
        p, res = self.p, self.p["box_res_mm"]
        best = (-2.0, None)
        for dth in np.radians(np.arange(-p["box_search_deg"], p["box_search_deg"] + 1e-6, 1.0)):
            th = self.pose[2] + dth
            window = self._patch(gray_ud, (self.pose[0], self.pose[1], th), extra=p["box_search_mm"])
            score = cv2.matchTemplate(window, self.template, cv2.TM_CCOEFF_NORMED)
            _, val, _, loc = cv2.minMaxLoc(score)
            if val > best[0]:
                best = (val, (th, score, loc))
        val, (th, score, (j, i)) = best
        # sub-pixel peak, then the offset from the window's middle in the turned frame -> board mm
        dj = _parabola_peak(score[i, j - 1], score[i, j], score[i, j + 1]) if 0 < j < score.shape[1] - 1 else 0.0
        di = _parabola_peak(score[i - 1, j], score[i, j], score[i + 1, j]) if 0 < i < score.shape[0] - 1 else 0.0
        ox = (j + dj - (score.shape[1] - 1) / 2) * res
        oy = -(i + di - (score.shape[0] - 1) / 2) * res
        c, s = math.cos(th), math.sin(th)
        if val >= p["box_min_score"]:
            self.pose = (self.pose[0] + c * ox - s * oy, self.pose[1] + s * ox + c * oy, th)
        return self.pose, float(val)


def hue_feature(bgr, board_hue):
    '''How far each pixel's hue is from the board's, smoothed so a pegboard's holes vanish (uint8).'''
    hsv = cv2.cvtColor(cv2.medianBlur(bgr, 5), cv2.COLOR_BGR2HSV)
    d = np.abs(hsv[:, :, 0].astype(np.int16) - board_hue)
    d = np.minimum(d, 180 - d)                                   # hue is circular (0..180 in OpenCV)
    d[hsv[:, :, 1] < 60] = 0                                     # gray things have no reliable hue
    d = cv2.morphologyEx(np.clip(d * 25, 0, 255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(d, (0, 0), 1.5)


def detect_box_by_hue(feature_ud, cal, size, res=2.0):
    '''
    Pose (cx, cy, theta) [board mm, as seen on the board plane] of the largest region whose hue
    differs from the board's, with theta along its long side (size[0] >= size[1] assumed).
    '''
    W, H = cal.board_mm
    A = cal.rect_matrix(res)
    img = cv2.warpPerspective(feature_ud, A @ cal.H, (int(W / res), int(H / res)))
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        raise RuntimeError("no box found by hue")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    corners = _apply_h(np.linalg.inv(A), cv2.boxPoints(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))))
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    th = math.atan2(e[1], e[0])
    th = (th + math.pi / 2) % math.pi - math.pi / 2               # a box looks the same turned 180 degrees
    c = corners.mean(axis=0)
    return (float(c[0]), float(c[1]), th)


def _parabola_peak(a, b, c):
    d = a - 2 * b + c
    return 0.0 if d >= 0 else 0.5 * (a - c) / d


#------------------------------------------------------------------ vine

def board_saturation(img, sigma):
    '''Saturation with a pegboard's holes closed and smoothed away.'''
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    return cv2.GaussianBlur(cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), (0, 0), sigma)


def saturation_background(ref_rect, paint, p):
    '''
    The board's saturation without the vine: the reference frame with the boxes (paint) and every
    unsaturated region (a vine already out, the mount) painted over. A clear tube shows up as a
    drop in the board's saturation even where it barely changes brightness.
    '''
    sigma = max(1.0, 4.5 / p["mm_per_px"])
    s = board_saturation(ref_rect, sigma)
    low = (s < 0.7 * np.median(s)).astype(np.uint8) * 255
    low = cv2.dilate(low, np.ones((int(20 / p["mm_per_px"]) | 1,) * 2, np.uint8))
    return cv2.inpaint(s, cv2.bitwise_or(low, paint), 7, cv2.INPAINT_TELEA)


def vine_mask(rect, background, p, roi=None):
    '''Pixels of the rectified frame that differ from the background (uint8 0/255), inside roi (a
    0/255 image) if given.'''
    if p["diff_mode"] == "saturation":
        sigma = max(1.0, 4.5 / p["mm_per_px"])
        diff = background.astype(np.int16) - board_saturation(rect, sigma).astype(np.int16)
        m = (diff > p["diff_threshold"]).astype(np.uint8) * 255
        return _clean(m if roi is None else cv2.bitwise_and(m, roi), p)
    a = cv2.GaussianBlur(rect, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(background, (5, 5), 0).astype(np.int16)
    d = a - b
    if p["diff_mode"] == "brighter":
        diff = d.clip(min=0).max(axis=2)
    elif p["diff_mode"] == "darker":
        diff = (-d).clip(min=0).max(axis=2)
    else:
        diff = np.abs(d).max(axis=2)
    m = (diff > p["diff_threshold"]).astype(np.uint8) * 255
    return _clean(m if roi is None else cv2.bitwise_and(m, roi), p)


def _clean(mask, p):
    r_px = p["vine_radius_mm"] / p["mm_per_px"]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(3, int(r_px / 2) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255


def component_at(mask, point_px, tol_px):
    '''The connected part of the mask that comes within tol_px of point_px (None if none does).'''
    n, lab = cv2.connectedComponents(mask)
    if n <= 1:
        return None
    ys, xs = np.nonzero(lab)
    d = np.hypot(xs - point_px[0], ys - point_px[1])
    i = int(np.argmin(d))
    if d[i] > tol_px:
        return None
    return (lab == lab[ys[i], xs[i]]).astype(np.uint8) * 255


def join_pieces(mask, comp, gap_px, r_px, min_area_px):
    '''
    comp plus every other piece of the mask that comes within gap_px of it (repeatedly), each
    joined by a tube-wide bridge between their nearest points, so a skeleton runs through gaps
    where the vine is hidden or barely visible.
    '''
    from scipy.spatial import cKDTree
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    out = comp.copy()
    pieces = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area_px]
    edge = lambda m: np.column_stack(np.nonzero(cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))[::-1])
    while True:
        tree = cKDTree(edge(out))
        best = None
        for k in pieces:
            piece = (lab == k).astype(np.uint8) * 255
            if (piece & out).any():
                continue
            pe = edge(piece)
            d, i = tree.query(pe)
            j = int(np.argmin(d))
            if d[j] <= gap_px and (best is None or d[j] < best[0]):
                best = (d[j], piece, pe[j], tree.data[i[j]])
        if best is None:
            return out
        _, piece, a, b = best
        out = cv2.bitwise_or(out, piece)
        cv2.line(out, tuple(np.int32(a)), tuple(np.int32(b)), 255, max(1, int(r_px)))


def centerline_px(comp, base_px, r_px):
    '''
    Base-to-tip curve through a vine mask [rectified px]: the longest geodesic path along the
    skeleton from the point nearest the base, with the forked ends of the skeleton cut off and the
    tip end extended along the vine to the center of its rounded cap (a radius short of the mask's
    edge, like the last body center of the sim's vine). Also returns the median radius [px].
    '''
    skel = skeletonize(comp > 0)
    ys, xs = np.nonzero(skel)
    if len(xs) < 3:
        return None, None
    index = -np.ones(skel.shape, dtype=np.int64)
    index[ys, xs] = np.arange(len(xs))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        j = np.full(len(xs), -1)
        j[ok] = index[y2[ok], x2[ok]]
        ok = j >= 0
        rows += [np.arange(len(xs))[ok]]; cols += [j[ok]]; w += [np.full(ok.sum(), math.hypot(dx, dy))]
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    graph = coo_matrix((w, (rows, cols)), shape=(len(xs), len(xs))).tocsr()
    start = int(np.argmin(np.hypot(xs - base_px[0], ys - base_px[1])))
    dist, pred = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    dist[~np.isfinite(dist)] = -1
    node = int(np.argmax(dist))
    path = []
    while node >= 0:
        path.append(node)
        node = pred[node]
    path = np.array(path[::-1])
    pts = np.column_stack([xs[path], ys[path]]).astype(np.float64)

    edt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    radius = float(np.median(edt[ys[path], xs[path]])) if len(path) else r_px

    # the skeleton forks within about a radius of a blunt end: cut that, extend straight to the edge
    s = _arc(pts)
    keep = s <= s[-1] - radius
    if keep.sum() >= 3:
        pts = pts[keep]
        back = pts[_arc(pts) >= _arc(pts)[-1] - radius]
        direction = back[-1] - back[0]
        n = np.linalg.norm(direction)
        if n > 0:
            direction /= n
            tip = pts[-1].copy()
            for _ in range(int(4 * radius) + 2):
                nxt = tip + direction
                u, v = int(round(nxt[0])), int(round(nxt[1]))
                if not (0 <= v < comp.shape[0] and 0 <= u < comp.shape[1]) or comp[v, u] == 0:
                    break
                tip = nxt
            # the tip is a rounded cap: the centerline ends at its center, a radius short of the edge
            tip = tip - radius * direction
            if np.dot(tip - pts[-1], direction) > 0:
                pts = np.vstack([pts, tip])
    # start exactly at the base, dropping the skeleton's fork there
    near = np.hypot(*(pts - base_px).T) < radius
    pts = np.vstack([base_px, pts[~near] if (~near).sum() >= 2 else pts])
    return pts, radius


def _arc(pts):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def resample(pts, spacing):
    s = _arc(pts)
    if s[-1] <= 0:
        return pts[:1]
    q = np.append(np.arange(0.0, s[-1], spacing), s[-1])
    return np.column_stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])])


def smooth(pts, sigma):
    '''Gaussian smoothing along the curve, ends kept in place.'''
    if sigma <= 0 or len(pts) < 5:
        return pts
    out = ndimage.gaussian_filter1d(pts, sigma, axis=0, mode="nearest")
    out[0], out[-1] = pts[0], pts[-1]
    return out


#------------------------------------------------------------------ the whole video

def track(trial, out_dir, stride=1, start=None, end=None, overlay=True, log_every=100):
    p = dict(DEFAULTS)
    p.update({k: trial[k] for k in DEFAULTS if k in trial})
    res = p["mm_per_px"]

    cal = calibrate_trial(trial, out_dir)
    walls = walls_mm(trial)
    ref_idx = trial.get("reference_frame", 0)
    ref_raw = read_frame(trial["video"], ref_idx)
    ref_ud = cv2.undistort(ref_raw, cal.K, cal.dist) if cal.K is not None else ref_raw
    if p["box_channel"] == "hue":
        board_hue = int(np.median(cv2.cvtColor(ref_ud, cv2.COLOR_BGR2HSV)[:, :, 0]))
        box_image = lambda ud: hue_feature(ud, board_hue)
    else:
        box_image = lambda ud: cv2.cvtColor(ud, cv2.COLOR_BGR2GRAY)
    ref_gray = box_image(ref_ud)
    ref_rect = cal.rectify(ref_raw, res)
    A = cal.rect_matrix(res)
    to_rect = lambda mm: _apply_h(A, mm)
    to_mm = lambda px: _apply_h(np.linalg.inv(A), px)

    # boxes: refine the given poses on the reference frame, then paint them out of the background
    boxes = trial_boxes(trial)
    trackers, refined = [], []
    for b in boxes:
        if b["pose"] is None:                       # "auto": found by its hue
            pose, ok = detect_box_by_hue(box_image(ref_ud) if p["box_channel"] == "hue"
                                         else hue_feature(ref_ud, int(np.median(cv2.cvtColor(ref_ud, cv2.COLOR_BGR2HSV)[:, :, 0]))),
                                         cal, b["size_mm"]), True
        else:
            pose, ok = refine_box(ref_gray, cal, b["pose"], b["size_mm"], p["box_res_mm"])
        refined.append(ok)
        b["pose_ref"] = pose
        trackers.append(BoxTracker(ref_gray, cal, pose, b["size_mm"], p))
    paint = np.zeros(ref_rect.shape[:2], np.uint8)
    for b in boxes:
        cv2.fillPoly(paint, [to_rect(box_corners(b["pose_ref"], b["size_mm"])).round().astype(np.int32)], 255)
    paint = cv2.dilate(paint, np.ones((int(2 * p["bg_paint_mm"] / res) | 1,) * 2, np.uint8))
    if p["diff_mode"] == "saturation":
        background = saturation_background(ref_rect, paint, p).astype(np.float32)
    else:
        background = cv2.inpaint(ref_rect, paint, 5, cv2.INPAINT_TELEA) if boxes else ref_rect

    walls_px = np.zeros(ref_rect.shape[:2], np.uint8)
    for w in walls:
        cv2.fillPoly(walls_px, [to_rect([[w[0], w[1]], [w[2], w[1]], [w[2], w[3]], [w[0], w[3]]]).round().astype(np.int32)], 255)
    for cx, cy, th, hw, hh in p["exclude_obb_mm"] or []:
        cv2.fillPoly(walls_px, [to_rect(box_corners((cx, cy, th), (2 * hw, 2 * hh))).round().astype(np.int32)], 255)

    roi = None
    if p["roi_mm"]:
        roi = np.zeros(ref_rect.shape[:2], np.uint8)
        cv2.fillPoly(roi, [to_rect(p["roi_mm"]).round().astype(np.int32)], 255)

    bx, by, heading = trial_base(trial, cal)
    base_px = to_rect([[bx, by]])[0]
    # behind the base, against the direction the vine leaves it in (the mount is there)
    behind = np.zeros(ref_rect.shape[:2], np.uint8)
    cut_deg = trial.get("base", {}).get("cut_dir_deg")
    a = math.radians(cut_deg) if cut_deg is not None else heading
    if a is not None:
        yy, xx = np.mgrid[0:behind.shape[0], 0:behind.shape[1]]
        pts = to_mm(np.column_stack([xx.ravel(), yy.ravel()])) - [bx, by]
        along = pts @ [math.cos(a), math.sin(a)]
        behind[((along < 0) & (np.hypot(*pts.T) < p["base_cut_mm"])).reshape(behind.shape)] = 255

    cap = cv2.VideoCapture(trial["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or trial.get("fps", 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start is None:
        start = trial["start_frame"] if trial.get("start_frame") is not None else ref_idx
    end = min(n_frames, (trial.get("end_frame") or n_frames) if end is None else end)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    writer = None
    if overlay:
        writer = cv2.VideoWriter(os.path.join(out_dir, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / stride), (ref_rect.shape[1], ref_rect.shape[0]))

    rec = dict(frame=[], t_s=[], valid=[], centerline=[], radius_mm=[], boxes=[], box_score=[])
    t0 = time.time()
    for idx in range(start, end):
        ok, raw = cap.read()
        if not ok:
            break
        if (idx - start) % stride:
            continue
        ud = cv2.undistort(raw, cal.K, cal.dist) if cal.K is not None else raw
        gray = box_image(ud) if boxes else None
        rect = cal.rectify(raw, res)

        poses, scores = [], []
        for tr in trackers:
            pose, score = tr.update(gray)
            poses.append(pose); scores.append(score)

        mask = vine_mask(rect, background, p, roi)
        mask[walls_px > 0] = 0
        mask[behind > 0] = 0
        cut = np.zeros_like(mask)
        if boxes:
            for pose, b in zip(poses, boxes):
                cv2.fillPoly(cut, [to_rect(box_corners(pose, b["size_mm"])).round().astype(np.int32)], 255)
            mask[cv2.dilate(cut, np.ones((int(2 * p["box_margin_mm"] / res) | 1,) * 2, np.uint8)) > 0] = 0
        if p["diff_mode"] == "saturation" and p["bg_rate"] > 0:
            # follow slow changes of the board, but not where the vine or a box is (or just was)
            busy = cv2.bitwise_or(mask, cv2.dilate(cut, np.ones((int(2 * p["bg_paint_mm"] / res) | 1,) * 2, np.uint8)))
            busy = cv2.dilate(busy, np.ones((int(2 * p["vine_radius_mm"] / res) | 1,) * 2, np.uint8))
            free = busy == 0
            now = board_saturation(rect, max(1.0, 4.5 / res)).astype(np.float32)
            background[free] += p["bg_rate"] * (now[free] - background[free])
        comp = component_at(mask, base_px, p["base_tol_mm"] / res)
        if comp is not None and p["gap_mm"]:
            r_px = p["vine_radius_mm"] / res
            comp = join_pieces(mask, comp, p["gap_mm"] / res, r_px, (r_px ** 2) * 2)
        line, radius = (None, None) if comp is None else centerline_px(comp, base_px, p["vine_radius_mm"] / res)

        line_mm = None
        if line is not None and len(line) >= 2:
            line_mm = to_mm(line)
            radius_mm = radius * res
            z = p["vine_height_mm"] if p["vine_height_mm"] is not None else radius_mm
            line_mm = cal.lift(line_mm, z)
            line_mm = resample(smooth(resample(line_mm, 1.0), p["smooth_mm"]), p["spacing_mm"])
        lifted = [cal.lift([pose[:2]], b["height_mm"])[0].tolist() + [pose[2]] for pose, b in zip(poses, boxes)]

        rec["frame"].append(idx)
        rec["t_s"].append((idx - start) / fps)
        rec["valid"].append(line_mm is not None)
        rec["centerline"].append(line_mm if line_mm is not None else np.zeros((0, 2)))
        rec["radius_mm"].append(radius * res if line_mm is not None else np.nan)
        rec["boxes"].append(lifted)
        rec["box_score"].append(scores)

        if writer is not None:
            writer.write(draw_overlay(rect, mask, line, poses, boxes, scores, to_rect, base_px, idx, fps, p))
        if log_every and len(rec["frame"]) % log_every == 0:
            L = _arc(line_mm)[-1] if line_mm is not None else 0.0
            print(f"frame {idx}/{end}: vine {L:.0f} mm, box scores {np.round(scores, 2).tolist()} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    cap.release()
    if writer is not None:
        writer.release()

    if heading is None:
        heading = measured_heading(rec["centerline"])
    meta = dict(trial=trial, params=p, fps=fps, stride=stride, base=[bx, by, heading],
                boxes=[dict(size_mm=list(b["size_mm"]), mass_kg=b["mass_kg"], height_mm=b["height_mm"],
                            pose_given=None if b["pose"] is None else list(b["pose"]), pose_ref=list(b["pose_ref"]), refined=ok)
                       for b, ok in zip(boxes, refined)],
                walls_mm=walls, calibration=cal.to_dict())
    return save_tracks(os.path.join(out_dir, "tracks.npz"), rec, meta)


def measured_heading(centerlines, length_mm=60.0):
    '''Launch heading from the first centerline at least length_mm long.'''
    for line in centerlines:
        if len(line) and _arc(line)[-1] >= length_mm:
            q = line[np.searchsorted(_arc(line), length_mm)]
            return float(math.atan2(q[1] - line[0, 1], q[0] - line[0, 0]))
    return 0.0


def draw_overlay(rect, mask, line, poses, boxes, scores, to_rect, base_px, idx, fps, p):
    img = rect.copy()
    tint = img.copy()
    tint[mask > 0] = (0.5 * tint[mask > 0] + [0, 90, 0]).astype(np.uint8)
    img = tint
    for pose, b, sc in zip(poses, boxes, scores):
        color = (0, 140, 255) if sc >= p["box_min_score"] else (0, 0, 255)
        cv2.polylines(img, [to_rect(box_corners(pose, b["size_mm"])).round().astype(np.int32)], True, color, 2)
    if line is not None:
        cv2.polylines(img, [line.round().astype(np.int32)], False, (255, 80, 0), 2)
        cv2.circle(img, tuple(line[-1].round().astype(int)), 4, (255, 0, 255), -1)
    cv2.circle(img, tuple(base_px.round().astype(int)), 5, (0, 0, 0), -1)
    cv2.putText(img, f"frame {idx}  t {idx / fps:.2f} s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f"frame {idx}  t {idx / fps:.2f} s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


#------------------------------------------------------------------ io

def save_tracks(path, rec, meta):
    F = len(rec["frame"])
    P = max([len(c) for c in rec["centerline"]] + [1])
    lines = np.full((F, P, 2), np.nan, np.float32)
    for k, c in enumerate(rec["centerline"]):
        lines[k, :len(c)] = c
    K = len(meta["boxes"])
    np.savez_compressed(path, frame=np.array(rec["frame"]), t_s=np.array(rec["t_s"]),
                        valid=np.array(rec["valid"]), centerline=lines,
                        n_points=np.array([len(c) for c in rec["centerline"]]),
                        radius_mm=np.array(rec["radius_mm"]),
                        boxes=np.array(rec["boxes"], dtype=np.float64).reshape(F, K, 3),
                        box_score=np.array(rec["box_score"]).reshape(F, K),
                        meta=np.array(json.dumps(meta)))
    return load_tracks(path)


def load_tracks(path):
    '''
    dict: frame, t_s, valid, n_points, radius_mm, boxes (F, K, 3) [mm, mm, rad], box_score (F, K),
    centerlines (list of (n, 2) mm arrays, base first), length_mm (F,), meta (dict).
    '''
    d = np.load(path)
    out = {k: d[k] for k in d.files if k not in ("meta", "centerline")}
    out["meta"] = json.loads(str(d["meta"]))
    out["centerlines"] = [d["centerline"][k, :n].astype(np.float64) for k, n in enumerate(d["n_points"])]
    out["length_mm"] = np.array([_arc(c)[-1] if len(c) > 1 else 0.0 for c in out["centerlines"]])
    return out


def growth_rate(tracks, min_mm=100.0, window_s=0.5, quantile=0.9):
    '''
    Free growth rate [mm/s]: a high quantile of the least-squares rates over short windows, so the
    time the vine spends stalled against a box or wall does not drag it down (dvsim's grow rate is
    the speed of unobstructed growth). Only frames where the vine is longer than min_mm count.
    '''
    ok = tracks["valid"] & (tracks["length_mm"] >= min_mm)
    t, L = tracks["t_s"][ok], tracks["length_mm"][ok]
    if len(t) < 3:
        return None
    rates = []
    for i in range(len(t)):
        w = (t >= t[i]) & (t <= t[i] + window_s)
        if w.sum() >= 3 and t[w][-1] - t[w][0] >= 0.5 * window_s:
            rates.append(np.polyfit(t[w], L[w], 1)[0])
    if not rates:
        return float(np.polyfit(t, L, 1)[0])
    return float(np.quantile(rates, quantile))


def summary_figure(tracks, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    meta = tracks["meta"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for w in meta["walls_mm"]:
        ax.add_patch(Rectangle((w[0], w[1]), w[2] - w[0], w[3] - w[1], color="0.6"))
    valid = np.nonzero(tracks["valid"])[0]
    cmap = plt.get_cmap("viridis")
    for n, k in enumerate(valid[:: max(1, len(valid) // 25)]):
        c = tracks["centerlines"][k]
        ax.plot(c[:, 0], c[:, 1], color=cmap(k / max(len(tracks["frame"]) - 1, 1)), lw=1.2)
    for j, b in enumerate(meta["boxes"]):
        for k, alpha in ((0, 0.25), (-1, 0.8)):
            if len(tracks["frame"]):
                ax.add_patch(Polygon(box_corners(tracks["boxes"][k, j], b["size_mm"]), closed=True, fill=False,
                                     ec="tab:orange", alpha=alpha, lw=2))
        ax.plot(tracks["boxes"][:, j, 0], tracks["boxes"][:, j, 1], "-", color="tab:orange", lw=1)
    ax.plot(*meta["base"][:2], "ko")
    W, H = meta["calibration"]["board_mm"]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title(f"{meta['trial']['name']}: centerlines (color = time), boxes first/last")

    rate = growth_rate(tracks)
    ax2.plot(tracks["t_s"][tracks["valid"]], tracks["length_mm"][tracks["valid"]], ".", ms=3, label="vine length [mm]")
    for j in range(len(meta["boxes"])):
        disp = np.linalg.norm(tracks["boxes"][:, j, :2] - tracks["boxes"][0, j, :2], axis=1)
        ax2.plot(tracks["t_s"], disp, "-", label=f"box {j} displacement [mm]")
    ax2.set_xlabel("t [s]")
    ax2.set_title("growth rate " + (f"{rate:.1f} mm/s" if rate else "n/a"))
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial")
    parser.add_argument("--out", default=None, help="default out/video/<trial name>")
    parser.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    trial = load_trial(args.trial)
    out = args.out or os.path.join("out", "video", trial["name"])
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    tracks = track(trial, out, stride=args.stride, start=args.start, end=args.end, overlay=not args.no_overlay)
    summary_figure(tracks, os.path.join(out, "summary.png"))
    rate = growth_rate(tracks)
    radius = np.nanmedian(tracks["radius_mm"]) if tracks["valid"].any() else float("nan")
    print(f"{len(tracks['frame'])} frames ({tracks['valid'].sum()} with a vine) in {time.time() - t0:.0f} s; "
          f"final length {tracks['length_mm'][-1]:.0f} mm, growth {rate if rate else float('nan'):.1f} mm/s, "
          f"radius {radius:.1f} mm")
    print("calibration residuals [mm]:", tracks["meta"]["calibration"]["residuals_mm"])
    print("wrote", out)


if __name__ == "__main__":
    main()
