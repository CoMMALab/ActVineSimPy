'''
Trial jsons for the simple-contact videos in video/simple_contact/<type>/<type>_NN.mov, written next to
this file as <type>_NN.json, plus video/manifest.json (every video: new path, original name, type,
dataset split) and, per video, a check image out/video/_detect/<name>.png of what was detected.

Setup, read off the videos: the same 1 in pegboard as the brick demos (calibrate.pegboard_lattice),
the vine's mount clamped to the board (it moves between videos), a clear tube that grows straight
(no actuation) out of a white stub, and the real 98 mm, 0.503 kg boxes (real_measurements.md).
    vine_block   one box
    block_block  two boxes, the vine pushes one into the other
    block_wall   one box, pushed into a slanted cardboard panel held up by a black weight

Detected on the reference frame, in a top-down image of the board plane that extends past the pegboard's
visible hole lattice (the lattice is cut short where the wall or the frame hides holes):
    base      the stub's end at the black mount; heading along the stub (track.py measures the
              launch heading from the vine anyway; this one is a first guess and the mount cut)
    boxes     the best matches of a box template (box_template.png, cropped from vine_block_01) over
              position and rotation; refined by track.py on the reference frame
    walls     (block_wall) the cardboard panel: a long dark-brown region; its face toward the camera's
              nadir is where it stands on the board (its top edge is seen pushed outward), so the wall
              is a 10 mm thick oriented box on that face. The black weight behind it is a wall too.
              Oriented walls go in "walls_obb_mm" ([cx, cy, theta_rad, hw, hh], what dvsim.si takes);
              track.py ignores them, to_reference.py adds them to the scenario's walls.
The reference frame is taken just before growth starts (a first tracking pass finds when).

Long-horizon videos (video/long/long_NN.mov, --long; a test set): the vine grows up from the mount,
turns along a top cardboard panel and a slanted one, runs up a channel between the slanted panel and
two bricks (one is grey in some videos) and pushes the box out of its end. The panels' feet are
refined per video from the pegboard holes next to rough lines read off long_02 (LONG_PANELS_MM); the
bricks are dark red or grey brick-sized regions; both are walls_obb_mm, and the regions behind the
panels and on the bricks are excluded from the vine's mask. Reference frame 0; growth speed measured.

    python video/trials/make_contact_trials.py              # detect, write trials and the manifest
    python video/trials/make_contact_trials.py --growth     # also find growth starts (runs track.py)
    python video/trials/make_contact_trials.py --long       # the long-horizon trials
'''

import argparse, glob, json, math, os, subprocess, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.dirname(HERE)
REPO = os.path.dirname(VIDEO)
sys.path.insert(0, VIDEO)
from calibrate import Calibration, calibrate, pegboard_lattice, read_frame, _apply_h

TYPES = ("vine_block", "block_block", "block_wall")
N_BOXES = {"vine_block": 1, "block_block": 2, "block_wall": 1}
BOX_MM, BOX_KG = 98.0, 0.503
BOX_HEIGHT_MM = 98.0          # not measured; a cube is assumed for the parallax of the box tops
CAMERA_HEIGHT_MM = 1100       # as for the brick demos
RES = 2.0                     # mm per px of the detection image
# original names, in the order the _NN suffixes were given (video/manifest.json has them too)
ORIGINAL = {
    "vine_block": [3689, 3690, 3692, 3693, 3695, 3696, 3697, 3698, 3699, 3700, 3702],
    "block_block": [3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713],
    "block_wall": [3716, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3731, 3732, 3733, 3735],
}
BRICK_ORIGINAL = [3497, 3499, 3500, 3501, 3509, 3511, 3512, 3515, 3516, 3518, 3520, 3521, 3522, 3523, 3524]
# left out of the dataset, with the reason (kept in the manifest)
EXCLUDED = {"block_wall_01": "a different, zoomed-in camera view: the board is partly out of frame, the "
                             "pegboard lattice anchors above the base, and the panel is seen obliquely, so it "
                             "and the weight are not detected"}
# held out, one per type: typical, cleanly tracked pushes (the box moves 100-300 mm and turns 20-35
# deg; the block_wall one has the most slanted panel)
VALIDATION = ("vine_block_06", "block_block_08", "block_wall_07")
# free growth speed: the median over the vine_block videos (160-191 mm/s each, from track.growth_rate).
# Used for every video: in block_wall the vine stalls against the box and panel, so its own estimate
# is not the free speed
GROW_RATE_MPS = 0.177
GROWTH_FILE = os.path.join(HERE, "contact_growth_start.json")
LEAD = 10                     # reference frame this many frames before growth starts


def extended(cal, frame):
    '''The calibration with its board extended up and right to everything the frame shows (the
    origin stays at the lattice's lower left), and that board's top-down image at RES.'''
    h, w = frame.shape[:2]
    corners = cal.px_to_mm([[0, 0], [w, 0], [w, h], [0, h]])
    W = float(min(corners[:, 0].max(), cal.board_mm[0] + 600))
    H = float(min(corners[:, 1].max(), cal.board_mm[1] + 600))
    ext = Calibration(H=cal.H, board_mm=(W, H), image_size=cal.image_size, K=cal.K, dist=cal.dist,
                      camera_height_mm=cal.camera_height_mm, residuals_mm=cal.residuals_mm)
    return ext, ext.rectify(frame, RES)


FULL_BOARD_MM = (46 * 25.4, 33 * 25.4)      # the pegboard's hole lattice when all of it is seen


def masks(ext, rect, holes_mm):
    '''white (the stub), black (mount, weight) and dark (boxes, the panel) masks, relative to the
    board's local brightness (the light falls off toward the bottom), inside the wooden frame.'''
    hsv = cv2.cvtColor(cv2.medianBlur(rect, 7), cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1].astype(np.float32), hsv[..., 2]
    Vbg = cv2.medianBlur(V, int(200 / RES) | 1).astype(np.float32)
    V = V.astype(np.float32)
    white = (V > Vbg + 20) & (S < 70)
    black = V < 70
    dark = (V < 0.78 * Vbg) & ~black
    # the cardboard panel: more saturated and darker than the board (a box is only a little more
    # saturated)
    panel = (S > np.median(S) + 30) & (V < np.median(V) - 50) & ~black
    # on the board: the convex hull of the pegboard holes found (any that were seen, not only the
    # lattice's anchoring rows), 30 mm past them
    A = ext.rect_matrix(RES)
    roi = np.zeros(V.shape, np.uint8)
    cv2.fillPoly(roi, [cv2.convexHull(_apply_h(A, holes_mm).astype(np.float32)).round().astype(np.int32)], 1)
    roi = cv2.dilate(roi, np.ones((int(60 / RES) | 1,) * 2, np.uint8))
    k = np.ones((5, 5), np.uint8)
    clean = lambda m: cv2.morphologyEx(cv2.morphologyEx((m & (roi > 0)).astype(np.uint8) * 255, cv2.MORPH_OPEN, k),
                                       cv2.MORPH_CLOSE, k)
    return clean(white), clean(black), clean(dark), clean(panel), roi


def components(mask, min_area_mm2):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] * RES * RES >= min_area_mm2:
            ys, xs = np.nonzero(lab == i)
            out.append(np.column_stack([xs, ys]).astype(np.float32))
    return out


def to_mm(ext, pts_px):
    A = ext.rect_matrix(RES)
    return _apply_h(np.linalg.inv(A), pts_px)


def rect_mm(ext, pts_px):
    '''minAreaRect of pixel points, in board mm: (center, (side_a, side_b), angle_rad of side_a).'''
    (cx, cy), (a, b), ang = cv2.minAreaRect(pts_px)
    c = to_mm(ext, [[cx, cy]])[0]
    th = -math.radians(ang)                     # image y is down
    return c, (a * RES, b * RES), th


TEMPLATE = os.path.join(HERE, "box_template.png")   # a box's top, axis aligned, at RES (from vine_block_01)


def make_template():
    '''Crop the (axis-aligned, isolated) box of vine_block_01 as the matching template.'''
    video = os.path.join(VIDEO, "simple_contact", "vine_block", "vine_block_01.mov")
    frame = read_frame(video, 0)
    ext, rect = extended(calibrate(frame, {"pegboard_pitch_mm": 25.4}), frame)
    _, _, dark = masks(ext, rect, ext.px_to_mm(pegboard_lattice(frame, 25.4)[0]))[:3]
    comps = [c for c in components(dark, 0.6 * BOX_MM ** 2) if 0.8 < np.ptp(c[:, 0]) / max(np.ptp(c[:, 1]), 1) < 1.25]
    c = min(comps, key=lambda c: abs(np.ptp(c[:, 0]) * RES - 1.08 * BOX_MM))
    cx, cy = (c.min(0) + c.max(0)) / 2
    h = int(round(1.08 * BOX_MM / RES / 2))
    cv2.imwrite(TEMPLATE, rect[int(cy) - h:int(cy) + h + 1, int(cx) - h:int(cx) + h + 1])


def detect_boxes(ext, rect, n_boxes, roi, min_sep_mm=0.8 * BOX_MM):
    '''The n_boxes best matches of the box template over position and rotation (normalized
    correlation on gray, the template masked to its square): poses (cx, cy, theta_deg) [board mm, as
    seen] and scores.'''
    if not os.path.exists(TEMPLATE):
        make_template()
    tpl = cv2.cvtColor(cv2.imread(TEMPLATE), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    side = tpl.shape[0]
    pad = int(side * 0.25)
    big = cv2.copyMakeBorder(tpl, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    sq = np.zeros_like(big); sq[pad:pad + side, pad:pad + side] = 255
    c = (big.shape[1] / 2 - 0.5, big.shape[0] / 2 - 0.5)
    best = None
    for deg in np.arange(0, 90, 2.0):
        M = cv2.getRotationMatrix2D(c, deg, 1.0)
        t = cv2.warpAffine(big, M, big.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(sq, M, big.shape[::-1], flags=cv2.INTER_NEAREST)
        score = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED, mask=m)
        score = np.nan_to_num(score, nan=-1, posinf=-1, neginf=-1)
        if best is None:
            best, ang = score.copy(), np.zeros(score.shape)
        better = score > best
        best[better], ang[better] = score[better], deg
    found = []
    work = best.copy()
    off = big.shape[0] // 2                         # score (y, x) is the template centered at (y + off, x + off)
    inside = roi[off:off + work.shape[0], off:off + work.shape[1]] > 0
    work[~inside[:work.shape[0], :work.shape[1]]] = -1
    r = int(min_sep_mm / RES)
    for _ in range(n_boxes):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        found.append((x + big.shape[1] / 2 - 0.5, y + big.shape[0] / 2 - 0.5, ang[y, x], float(work[y, x])))
        work[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
    out = []
    for x, y, deg, score in found:
        cmm = to_mm(ext, [[x, y]])[0]
        th = deg                                    # counterclockwise on the top-down image = in the board frame
        th = (th + 45) % 90 - 45
        out.append(dict(pose_mm=[round(float(cmm[0]), 1), round(float(cmm[1]), 1), round(float(th), 1)],
                        score=round(score, 3)))
    return out


def detect_base(ext, white, black, boxes):
    '''(x, y, heading_deg) of the base: the white stub next to the black mount, and its end away from
    the boxes (the vine is launched at them in every video); the heading along the stub.'''
    big = np.zeros(black.shape, np.uint8)
    for c in components(black, 2000):
        big[c[:, 1].astype(int), c[:, 0].astype(int)] = 255
    if not big.any():
        raise RuntimeError("no mount found")
    to_black = cv2.distanceTransform(255 - big, cv2.DIST_L2, 5) * RES       # mm to the nearest big black blob
    def stub_like(c):                                                       # 60-300 mm long, 25-80 mm wide
        _, (a, b), _ = rect_mm(ext, c)
        return 60 <= max(a, b) <= 300 and 25 <= min(a, b) <= 80
    stubs = [c for c in components(white, 1500) if stub_like(c)]
    if not stubs:
        raise RuntimeError("no stub found")
    near = lambda c: float(to_black[c[:, 1].astype(int), c[:, 0].astype(int)].min())
    stub = min(stubs, key=near)                                              # the one at the mount
    pts = to_mm(ext, stub)
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    t = (pts - c) @ d
    ends_mm = c + np.outer([t.min(), t.max()], d)
    centers = np.array([b["pose_mm"][:2] for b in boxes])
    to_box = [np.linalg.norm(centers - e, axis=1).min() for e in ends_mm]
    base, tip = (ends_mm[0], ends_mm[1]) if to_box[0] >= to_box[1] else (ends_mm[1], ends_mm[0])
    h = math.degrees(math.atan2(*(tip - base)[::-1]))
    return [round(float(base[0]), 1), round(float(base[1]), 1)], round(h, 1), float(np.linalg.norm(tip - base)), base


def detect_walls(ext, panel, black, boxes, mount):
    '''Oriented walls [cx, cy, theta, hw, hh]: the cardboard panel (its face toward the box, where it
    stands on the board, 10 mm thick) and the black weight beside it.'''
    walls, drawn, exclude = [], [], []
    cands = []
    for comp in components(panel, 3000):
        c, (a, b), th = rect_mm(ext, comp)
        if max(a, b) > 150 and min(a, b) < 0.45 * max(a, b):
            cands.append((max(a, b), comp, c, (a, b), th))
    if not cands:
        return walls, drawn, exclude
    _, comp, c, (a, b), th = max(cands, key=lambda p: p[0])
    if b > a:                                       # theta along the long side
        a, b, th = b, a, th + math.pi / 2
    u = np.array([math.cos(th), math.sin(th)])
    n = np.array([-u[1], u[0]])
    box = np.array(boxes[0]["pose_mm"][:2]) if boxes else c
    if (box - c) @ n < 0:
        n = -n                                      # n points toward the box
    face = (to_mm(ext, comp) - c) @ n
    off_face = np.percentile(face, 97)               # the face toward the box, robust to stray pixels
    T = 10.0
    wc = c + (off_face - T / 2) * n
    th = math.atan2(u[1], u[0])
    walls.append([round(float(wc[0]), 1), round(float(wc[1]), 1), round(th, 4), round(a / 2, 1), T / 2])
    drawn.append(("panel", wc, u, a / 2, T / 2))
    # no vine on or behind the panel (its top edge and face are seen there): 150 mm back from its face
    back = wc - (75 - T / 2) * n
    exclude.append([round(float(back[0]), 1), round(float(back[1]), 1), round(th, 4), round(a / 2 + 20, 1), 75.0])
    for comp in components(black, 6000):
        pts = to_mm(ext, comp)
        if np.linalg.norm(pts.mean(0) - mount) < 200 or np.linalg.norm(pts.mean(0) - c) > 250:
            continue                                # the vine's mount, or not beside the panel
        cw, (wa, wb), wth = rect_mm(ext, comp)
        walls.append([round(float(cw[0]), 1), round(float(cw[1]), 1), round(wth, 4), round(wa / 2, 1), round(wb / 2, 1)])
        drawn.append(("weight", cw, np.array([math.cos(wth), math.sin(wth)]), wa / 2, wb / 2))
        exclude.append(walls[-1][:3] + [round(wa / 2 + 15, 1), round(wb / 2 + 15, 1)])
    return walls, drawn, exclude


def detect_panel_lines(ext, panel, max_lines=3, tol_mm=6.0, min_pts=40, min_len_mm=150.0, seed=0):
    '''Standing panels as segments on the board: the edge of each panel's visible face nearest the
    camera's nadir (a vertical panel's face is seen pushed away from the nadir, its foot is not), split
    into straight pieces by sequential RANSAC. Returns [(a, b)] segment ends [board mm].'''
    nadir = ext.px_to_mm([[ext.image_size[0] / 2, ext.image_size[1] / 2]])[0]
    pts_px = np.column_stack(np.nonzero(panel)[::-1]).astype(np.float32)
    if len(pts_px) == 0:
        return []
    pts = to_mm(ext, pts_px)
    A = ext.rect_matrix(RES)
    toward = (nadir - pts) / (np.linalg.norm(nadir - pts, axis=1, keepdims=True) + 1e-9)
    probe = _apply_h(A, pts + 6.0 * toward).round().astype(int)
    h, w = panel.shape
    inside = (probe[:, 0] >= 0) & (probe[:, 0] < w) & (probe[:, 1] >= 0) & (probe[:, 1] < h)
    edge = np.ones(len(pts), bool)
    edge[inside] = panel[probe[inside, 1], probe[inside, 0]] == 0
    P = pts[edge]
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(max_lines):
        if len(P) < min_pts:
            break
        best = None
        for _ in range(400):
            i, j = rng.choice(len(P), 2, replace=False)
            d = P[j] - P[i]
            L = np.linalg.norm(d)
            if L < 30:
                continue
            n = np.array([-d[1], d[0]]) / L
            inl = np.abs((P - P[i]) @ n) < tol_mm
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < min_pts:
            break
        Q = P[best]
        c = Q.mean(0)
        u = np.linalg.svd(Q - c)[2][0]
        t = (Q - c) @ u
        # the longest run of inliers without a gap over 40 mm (one panel, not two in line)
        order = np.sort(t)
        gaps = np.nonzero(np.diff(order) > 40)[0]
        bounds = np.split(order, gaps + 1)
        run = max(bounds, key=lambda r: r[-1] - r[0])
        if run[-1] - run[0] >= min_len_mm:
            segs.append((c + run[0] * u, c + run[-1] * u))
        P = P[~best]
    return segs


def detect_bricks(ext, rect, roi, cal, height_mm=57.0):
    '''The largest region of red bricks (static here) as oriented boxes [cx, cy, theta, hw, hh] [board mm]: regions of red hue,
    split where two touch end to end (each a 203 x 92 mm brick top); centers lifted from the top face.'''
    hsv = cv2.cvtColor(cv2.medianBlur(rect, 7), cv2.COLOR_BGR2HSV).astype(np.int16)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    Vb = np.median(V)
    red = ((Hh <= 10) | (Hh >= 172)) & (S > 60) & (V > 50) & (V < 0.85 * Vb) & (roi > 0)
    k = np.ones((5, 5), np.uint8)
    red = cv2.morphologyEx(cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    out = []
    comps = components(red, 8000)
    for comp in sorted(comps, key=len)[-1:]:       # the pair in the channel (others stand behind panels)
        c, (a, b), th = rect_mm(ext, comp)
        if b > a:
            a, b, th = b, a, th + math.pi / 2
        n = max(1, int(round(a / 210)))             # bricks end to end
        u = np.array([math.cos(th), math.sin(th)])
        for k_ in range(n):
            ck = c + ((k_ + 0.5) / n - 0.5) * a * u
            ck = cal.lift([ck], height_mm)[0]
            out.append([round(float(ck[0]), 1), round(float(ck[1]), 1), round(float(th), 4), round(a / n / 2, 1), round(b / 2, 1)])
    return out


def obb_corners(c, u, hw, hh):
    n = np.array([-u[1], u[0]])
    return np.array([c - hw * u - hh * n, c + hw * u - hh * n, c + hw * u + hh * n, c - hw * u + hh * n])


def shifted_calibration(frame, calib):
    '''The pegboard calibration with its origin moved so the whole frame has x, y >= 0 (the lattice
    anchors on the rows and columns that are 3/4 in view, which can be above the base or the boxes;
    track.py's top-down image starts at the origin). Returns (calibration, shift [mm]).'''
    cal = calibrate(frame, calib)
    h, w = frame.shape[:2]
    corners = cal.px_to_mm([[0, 0], [w, 0], [w, h], [0, h]])
    shift = [round(float(max(0.0, -corners[:, 0].min())), 1), round(float(max(0.0, -corners[:, 1].min())), 1)]
    if max(shift) > 0:
        cal = calibrate(frame, dict(calib, origin_shift_mm=shift))
    return cal, shift


def frame_walls(cal, shift=(0.0, 0.0)):
    '''The wooden frame, on the sides that are in view and where the lattice spans the full board
    across them (46 x 33 pitches), so the side's position is known.'''
    LIP, WALL = 12.7, 40.0
    W, H = cal.board_mm
    w_img, h_img = cal.image_size
    full_w, full_h = abs(W - FULL_BOARD_MM[0]) < 12, abs(H - FULL_BOARD_MM[1]) < 12

    def seen(pts):
        px = cal.mm_to_px(pts)
        return np.mean((px[:, 0] > 0) & (px[:, 0] < w_img) & (px[:, 1] > 0) & (px[:, 1] < h_img)) > 0.8

    xs, ys = np.linspace(0, W, 20), np.linspace(0, H, 20)          # lattice frame (before the shift)
    sides = [(full_h, np.column_stack([xs, np.full(20, -LIP - 10)]), [-LIP - WALL, -LIP - WALL, W + LIP + WALL, -LIP]),
             (full_h, np.column_stack([xs, np.full(20, H + LIP + 10)]), [-LIP - WALL, H + LIP, W + LIP + WALL, H + LIP + WALL]),
             (full_w, np.column_stack([np.full(20, -LIP - 10), ys]), [-LIP - WALL, -LIP, -LIP, H + LIP]),
             (full_w, np.column_stack([np.full(20, W + LIP + 10), ys]), [W + LIP, -LIP, W + LIP + WALL, H + LIP])]
    sx, sy = shift
    return [[round(w[0] + sx, 1), round(w[1] + sy, 1), round(w[2] + sx, 1), round(w[3] + sy, 1)]
            for full, pts, w in sides if full and seen(np.asarray(pts) + [sx, sy])]


def make_trial(kind, name, video, growth, n_boxes=None):
    ref = max(0, growth.get(name, LEAD) - LEAD)
    # detected on frame 0 (the camera's exposure changes once the vine moves), expressed in the
    # calibration of the reference frame (what track.py calibrates on)
    frame = read_frame(video, 0)
    calib = {"pegboard_pitch_mm": 25.4, "camera_height_mm": CAMERA_HEIGHT_MM}
    cal, _ = shifted_calibration(frame, calib)
    ext, rect = extended(cal, frame)
    holes_px = pegboard_lattice(frame, 25.4)[0]
    white, black, _, panel, roi = masks(ext, rect, cal.px_to_mm(holes_px))
    boxes = detect_boxes(ext, rect, n_boxes or N_BOXES[kind], roi)
    # the boxes' dark rims are not the mount, and their tops are not the stub
    A = ext.rect_matrix(RES)
    foot = np.zeros(black.shape, np.uint8)
    for b in boxes:
        cx, cy, deg = b["pose_mm"]
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        cv2.fillPoly(foot, [_apply_h(A, obb_corners(np.array([cx, cy]), u, 0.6 * BOX_MM, 0.6 * BOX_MM)).round().astype(np.int32)], 255)
    black[foot > 0] = 0
    white[foot > 0] = 0
    panel[foot > 0] = 0                       # the boxes' brown sides
    base, heading, stub_mm, mount = detect_base(ext, white, black, boxes)
    walls_obb, drawn, exclude = detect_walls(ext, panel, black, boxes, mount) if kind == "block_wall" else ([], [], [])
    # frame 0 -> reference frame board coordinates: the two lattice fits agree up to a shift (a whole
    # pitch if the anchoring rows differ)
    ref_frame = read_frame(video, ref)
    cal_ref, shift = shifted_calibration(ref_frame, calib)
    # where the vine is looked for: the board, as the hull of its holes plus 8 mm (the frame's lip
    # starts 12.7 mm past the outer holes)
    hull = cv2.convexHull(cal_ref.px_to_mm(pegboard_lattice(ref_frame, 25.4)[0]).astype(np.float32)).reshape(-1, 2)
    out = hull - hull.mean(0)
    roi_mm = (hull + 8.0 * out / np.linalg.norm(out, axis=1, keepdims=True)).round(1).tolist()
    h, w = frame.shape[:2]
    grid = np.stack(np.meshgrid(np.linspace(0.2 * w, 0.8 * w, 7), np.linspace(0.2 * h, 0.8 * h, 7)), -1).reshape(-1, 2)
    d = cal_ref.px_to_mm(grid) - cal.px_to_mm(grid)
    delta = np.median(d, axis=0)
    if np.abs(d - delta).max() > 5:
        print(f"warning: {name}: the reference frame's calibration differs from frame 0's by "
              f"{np.abs(d - delta).max():.1f} mm beyond a shift")
    move = lambda xy: [round(float(xy[0] + delta[0]), 1), round(float(xy[1] + delta[1]), 1)]
    base = move(base)
    for b in boxes:
        b["pose_mm"] = move(b["pose_mm"][:2]) + [b["pose_mm"][2]]
    walls_obb = [move(wl[:2]) + wl[2:] for wl in walls_obb]
    exclude = [move(wl[:2]) + wl[2:] for wl in exclude]
    trial = {
        "name": name, "type": kind,
        "video": os.path.relpath(video, HERE),
        "pegboard_pitch_mm": 25.4,
        "camera_height_mm": CAMERA_HEIGHT_MM,
        # tracked on the board extended past the lattice (the wall and the boxes' paths are off it)
        "board_mm": [round(v, 1) for v in extended(cal_ref, frame)[0].board_mm],
        "origin_shift_mm": shift,
        "walls_mm": frame_walls(cal_ref, shift),
        "walls_obb_mm": walls_obb,
        "base": {"mm": base, "heading_deg": None, "cut_dir_deg": heading},
        "boxes": [{"size_mm": [BOX_MM, BOX_MM], "pose_mm": b["pose_mm"], "mass_kg": BOX_KG,
                   "height_mm": BOX_HEIGHT_MM} for b in boxes],
        "schedule": [[0.0, 1500]],
        "grow_rate_mps": GROW_RATE_MPS,
        "reference_frame": ref,
        "roi_mm": roi_mm,
        "exclude_obb_mm": exclude,
        "mm_per_px": 1.5,
        "diff_mode": "saturation",
        "diff_threshold": 12,
        "vine_radius_mm": 24,
        "box_channel": "gray",
        "box_search_mm": 15,
        "box_search_deg": 5,
        # boxes are painted out of the background only 8 mm past their footprint (20 by default): a
        # wider paint is mistaken for vine where a box uncovers it, and bends the tip toward it
        "bg_paint_mm": 8,
        "box_margin_mm": 4,
        "detected": {"stub_mm": round(stub_mm), "box_scores": [b["score"] for b in boxes],
                     "lattice_board_mm": [round(v, 1) for v in cal.board_mm],
                     "calibration_residuals_mm": cal_ref.residuals_mm,
                     "frame0_to_reference_mm": [round(float(v), 1) for v in delta]},
    }
    # check image: detections on the extended top-down frame
    A = ext.rect_matrix(RES)
    img = rect.copy()
    for b in boxes:
        cx, cy, deg = b["pose_mm"]
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        cv2.polylines(img, [_apply_h(A, obb_corners(np.array([cx, cy]), u, BOX_MM / 2, BOX_MM / 2)).round().astype(np.int32)],
                      True, (0, 0, 255), 2)
    for _, c, u, hw, hh in drawn:
        cv2.polylines(img, [_apply_h(A, obb_corners(np.array(c), u, hw, hh)).round().astype(np.int32)], True, (255, 0, 0), 2)
    for w in trial["walls_mm"]:
        cv2.rectangle(img, tuple(_apply_h(A, [[w[0], w[1]]])[0].round().astype(int)),
                      tuple(_apply_h(A, [[w[2], w[3]]])[0].round().astype(int)), (255, 0, 255), 2)
    bp = _apply_h(A, [base])[0]
    tip = np.array(base) + 120 * np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    cv2.arrowedLine(img, tuple(bp.round().astype(int)), tuple(_apply_h(A, [tip])[0].round().astype(int)), (0, 255, 0), 2)
    cv2.putText(img, f"{name} ({kind})", (10, 25), 0, 0.7, (0, 255, 0), 2)
    os.makedirs(os.path.join(REPO, "out", "video", "_detect"), exist_ok=True)
    cv2.imwrite(os.path.join(REPO, "out", "video", "_detect", name + ".png"), img)
    return trial


#------------------------------------------------------------------------ long-horizon videos

LONG_ORIGINAL = [3679, 3736, 3738, 3739, 3741, 3744, 3745, 3747, 3749, 3750]    # long_01 ...
# The cardboard panels' feet (where they stand on the board), read off long_02 (frame 0, the hole rows
# next to each panel) in long_02's board frame: the rough lines every video's panels are refined from
# (make_long_trial; the panels are the same in all ten, but shift by a few cm between them).
LONG_REF = "long_02"
LONG_PANELS_MM = [((70.0, 560.0), (70.0, 810.0)),          # left, along the board's edge
                  ((70.0, 810.0), (465.0, 876.0)),         # top, slanted ~10 deg
                  ((465.0, 876.0), (725.0, 741.0))]        # slanted ~27 deg, down to the box
LONG_INSIDE_MM = (600.0, 450.0)                            # a point on the board side of every panel


def panel_obbs(panels, inside, thickness=10.0, behind=150.0):
    '''Walls (10 mm thick, standing on each segment, on its far side from `inside`) and the regions
    behind them where the vine cannot be (from 12 mm past the foot: the face and the top edge are seen
    there).'''
    walls, exclude = [], []
    for a, b in panels:
        a, b = np.array(a), np.array(b)
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        if (np.array(inside) - a) @ n > 0:
            n = -n                                  # n points away from the board side
        L = np.linalg.norm(b - a)
        th = math.atan2(u[1], u[0])
        c = (a + b) / 2 + thickness / 2 * n
        walls.append([round(float(c[0]), 1), round(float(c[1]), 1), round(th, 4), round(L / 2, 1), thickness / 2])
        ce = (a + b) / 2 + (12 + behind / 2) * n
        exclude.append([round(float(ce[0]), 1), round(float(ce[1]), 1), round(th, 4), round(L / 2 + 20, 1), behind / 2])
    return walls, exclude


def topdown_gray(cal, frame, res=2.0, size=(1400, 1100)):
    ext = Calibration(H=cal.H, board_mm=size, image_size=cal.image_size, K=cal.K, dist=cal.dist)
    g = cv2.cvtColor(ext.rectify(frame, res), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g - cv2.GaussianBlur(g, (0, 0), 15), (0, 0), 2)


def refine_panel(holes, a, b, inside, band=90.0, max_turn_deg=10.0, foot_mm=10.0):
    '''A panel's foot from the holes seen next to it: of the lines near a-b (turned up to max_turn_deg),
    the one along which the outermost visible holes line up best (most holes within half a pitch of
    the envelope); the foot is foot_mm past that envelope (the next row of holes is hidden under the
    panel's face). Returns the refined (a, b), or None if too few holes are seen along it.'''
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid, L = (a + b) / 2, np.linalg.norm(b - a)
    u0 = (b - a) / L
    best = None
    for turn in np.radians(np.arange(-max_turn_deg, max_turn_deg + 0.25, 0.5)):
        c, s_ = math.cos(turn), math.sin(turn)
        u = np.array([c * u0[0] - s_ * u0[1], s_ * u0[0] + c * u0[1]])
        n = np.array([-u[1], u[0]])
        if (np.asarray(inside) - mid) @ n > 0:
            n = -n                                  # outward
        rel = holes - mid
        t, d = rel @ u, rel @ n
        near = (np.abs(t) < L / 2 - 15) & (d > -band) & (d < band)
        if near.sum() < 10:
            continue
        env = d[near].max()
        score = int((d[near] > env - 12.7).sum())
        if best is None or score > best[0]:
            best = (score, u, n, env)
    if best is None or best[0] < 5:
        return None
    _, u, n, env = best
    foot = mid + (env + foot_mm) * n
    # keep the original extent along the panel, projected on the refined line
    ta, tb = (a - mid) @ u, (b - mid) @ u
    return foot + ta * u, foot + tb * u


_long_ref = None


def make_long_trial(name, video):
    global _long_ref
    calib = {"pegboard_pitch_mm": 25.4, "camera_height_mm": CAMERA_HEIGHT_MM}
    if _long_ref is None:
        f = read_frame(os.path.join(VIDEO, "long", LONG_REF + ".mov"), 0)
        _long_ref = topdown_gray(shifted_calibration(f, calib)[0], f)
    trial = make_trial("long", name, video, {}, n_boxes=1)
    frame = read_frame(video, trial["reference_frame"])
    cal, _ = shifted_calibration(frame, calib)
    # rough: long_02's panels moved by the phase correlation of the two top-down images (good to a few
    # cm: the panels' faces and tops shift with the camera); then each foot from the holes next to it
    g = topdown_gray(cal, frame)
    (dx, dy), _ = cv2.phaseCorrelate(_long_ref, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
    off = np.array([dx * 2.0, -dy * 2.0])
    holes = cal.px_to_mm(pegboard_lattice(frame, 25.4)[0])
    inside = np.array(LONG_INSIDE_MM) + off
    panels, refined = [], []
    for a, b in LONG_PANELS_MM:
        a, b = np.array(a) + off, np.array(b) + off
        r = refine_panel(holes, a, b, inside)
        refined.append(r is not None)
        panels.append(r if r is not None else (a, b))
    # consecutive panels meet: join each corner at the intersection of the refined lines
    for i in range(len(panels) - 1):
        (a1, b1), (a2, b2) = panels[i], panels[i + 1]
        d1, d2 = b1 - a1, b2 - a2
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) > 1e-6:
            t = ((a2 - a1)[0] * d2[1] - (a2 - a1)[1] * d2[0]) / den
            x = a1 + t * d1
            panels[i], panels[i + 1] = (a1, x), (x, b2)
    walls, exclude = panel_obbs(panels, inside)
    # bricks: dark red or grey brick-sized regions, not a panel, the box, the mount or the stub
    ext, rect = extended(cal, frame)
    _, black, _, _, roi = masks(ext, rect, holes)
    A = ext.rect_matrix(RES)
    # darker than the board as a whole (two bricks end to end fill too much of masks()' local window)
    hsv = cv2.cvtColor(cv2.medianBlur(rect, 7), cv2.COLOR_BGR2HSV).astype(np.float32)
    Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # red (hue <= 10, not the panels' saturated brown) or grey bricks
    brickish = (((Hh <= 10) | (Hh >= 172)) & (S <= 125)) | (S < 60)
    dark = ((V < 0.8 * np.median(V[roi > 0])) & brickish & (roi > 0) & (black == 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    keep = dark.copy()
    for cx, cy, th, hw, hh in exclude + walls:
        cv2.fillPoly(keep, [_apply_h(A, obb_corners(np.array([cx, cy]), np.array([math.cos(th), math.sin(th)]), hw, hh)).round().astype(np.int32)], 0)
    for bx in trial["boxes"]:
        cx, cy, deg = bx["pose_mm"]
        u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        cv2.fillPoly(keep, [_apply_h(A, obb_corners(np.array([cx, cy]), u, 70, 70)).round().astype(np.int32)], 0)
    base = np.array(trial["base"]["mm"])
    cv2.circle(keep, tuple(_apply_h(A, [base])[0].round().astype(int)), int(150 / RES), 0, -1)
    bricks = []
    for comp in components(keep, 8000):
        c, (la, lb), th = rect_mm(ext, comp)
        if lb > la:
            la, lb, th = lb, la, th + math.pi / 2
        if not (150 <= la <= 470 and 60 <= lb <= 140):
            continue
        k = max(1, int(round(la / 205)))
        u = np.array([math.cos(th), math.sin(th)])
        for j in range(k):
            ck = cal.lift([c + ((j + 0.5) / k - 0.5) * la * u], 57.0)[0]
            bricks.append([round(float(ck[0]), 1), round(float(ck[1]), 1), round(float(th), 4), round(la / k / 2, 1), round(lb / 2, 1)])
    # the two bricks lie end to end: give them a brick's footprint (203 x 92 mm) on the line through both;
    # with one found, the other is the darker of the two places next to it along its length
    if len(bricks) == 1:
        c, th = np.array(bricks[0][:2]), bricks[0][2]
        u = np.array([math.cos(th), math.sin(th)])
        def darkness(p):
            x, y = _apply_h(A, [p])[0].round().astype(int)
            return float(V[max(0, y - 15):y + 15, max(0, x - 30):x + 30].mean())
        other = min((c + 205 * u, c - 205 * u), key=darkness)
        bricks.append([round(float(other[0]), 1), round(float(other[1]), 1), th, bricks[0][3], bricks[0][4]])
        trial["detected"]["brick_added"] = True
    if len(bricks) == 2:
        c1, c2 = np.array(bricks[0][:2]), np.array(bricks[1][:2])
        th = math.atan2(*(c2 - c1)[::-1])
        bricks = [[b[0], b[1], round(th, 4), 101.5, 46.0] for b in bricks]
    trial["walls_obb_mm"] = walls + bricks
    # no vine on the bricks (a grey brick looks like the clear tube to the saturation test): their
    # footprints and where their tops are seen (57 mm up, pushed away from the nadir)
    nadir = cal.px_to_mm([[cal.image_size[0] / 2, cal.image_size[1] / 2]])[0]
    zc = CAMERA_HEIGHT_MM
    for cx, cy, th, hw, hh in bricks:
        seen = nadir + (np.array([cx, cy]) - nadir) * zc / (zc - 57.0)
        # the footprint and the top as seen, with margins that also cover the side faces seen beyond them
        # (the vine runs up the channel between the bricks and the slanted panel without touching them)
        for c, margin in ((np.array([cx, cy]), 12.0), (seen, 20.0)):
            exclude.append([round(float(c[0]), 1), round(float(c[1]), 1), th, hw + margin, hh + margin])
    trial["exclude_obb_mm"] = exclude
    trial.pop("grow_rate_mps")                      # measured per video (long_01 grows slower)
    trial["detected"]["panels"] = {"rough_offset_mm": [round(float(v), 1) for v in off], "refined_from_holes": refined,
                                   "feet_mm": [[[round(float(v), 1) for v in a], [round(float(v), 1) for v in b]] for a, b in panels]}
    trial["detected"]["bricks"] = len(bricks)
    img = cv2.imread(os.path.join(REPO, "out", "video", "_detect", name + ".png"))
    for w in trial["walls_obb_mm"]:
        u = np.array([math.cos(w[2]), math.sin(w[2])])
        cv2.polylines(img, [_apply_h(A, obb_corners(np.array(w[:2]), u, w[3], w[4])).round().astype(np.int32)], True, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(REPO, "out", "video", "_detect", name + ".png"), img)
    return trial


def manifest():
    rows = []
    for k, n in enumerate(BRICK_ORIGINAL):
        rows.append(dict(path=f"video/demos/brick_{k + 1:02d}.mov", original=f"IMG_{n}", type="brick",
                         dataset="brick", split="train"))
    for kind in TYPES:
        for k, n in enumerate(ORIGINAL[kind]):
            name = f"{kind}_{k + 1:02d}"
            rows.append(dict(path=f"video/simple_contact/{kind}/{name}.mov", original=f"IMG_{n}", type=kind,
                             dataset="simple_contact",
                             split="excluded" if name in EXCLUDED else "val" if name in VALIDATION else "train",
                             **({"excluded": EXCLUDED[name]} if name in EXCLUDED else {})))
    for k, n in enumerate(LONG_ORIGINAL):
        rows.append(dict(path=f"video/long/long_{k + 1:02d}.mov", original=f"IMG_{n}", type="long", dataset="long",
                         split="test"))
    with open(os.path.join(VIDEO, "manifest.json"), "w") as f:
        json.dump({"note": "every video: new path, original camera file name, type, dataset and split "
                           "(simple_contact: one video per type held out for validation)", "videos": rows}, f, indent=1)


def find_growth_starts(names, jobs=4):
    '''First frame where the vine is 30 mm longer than at the start, from a stride-2 tracking pass
    with the reference at frame 0 (the trials must exist).'''
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, VIDEO)
    from track import load_tracks

    def one(name):
        out = os.path.join(REPO, "out", "video", "_growth", name)
        os.makedirs(out, exist_ok=True)
        trial = os.path.join(HERE, name + ".json")
        t = json.load(open(trial))
        t["reference_frame"] = 0
        tmp = os.path.join(out, "trial.json")
        t["video"] = os.path.abspath(os.path.join(HERE, t["video"]))
        json.dump(t, open(tmp, "w"))
        subprocess.run([sys.executable, "-B", os.path.join(VIDEO, "track.py"), tmp, "--out", out, "--stride", "2",
                        "--no-overlay"], check=True, capture_output=True)
        tr = load_tracks(os.path.join(out, "tracks.npz"))
        L = np.where(tr["valid"], tr["length_mm"], np.nan)
        L0 = np.nanmedian(L[:5])
        grown = np.nonzero(L > L0 + 30)[0]
        return name, int(tr["frame"][grown[0]]) if len(grown) else 0

    with ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(one, names))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="only these (e.g. block_wall_03)")
    parser.add_argument("--growth", action="store_true", help="find growth starts first (tracking pass)")
    parser.add_argument("--long", action="store_true", help="the long-horizon videos (video/long/) instead")
    args = parser.parse_args()
    manifest()
    if args.long:
        for p in sorted(glob.glob(os.path.join(VIDEO, "long", "long_*.mov"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if args.names and name not in args.names:
                continue
            trial = make_long_trial(name, p)
            with open(os.path.join(HERE, name + ".json"), "w") as f:
                json.dump(trial, f, indent=1)
            print(f"wrote {name}.json: base {trial['base']['mm']} at {trial['base']['cut_dir_deg']} deg, box "
                  f"{trial['boxes'][0]['pose_mm']} score {trial['detected']['box_scores']}, {trial['detected']['bricks']} "
                  f"bricks, panels {trial['detected']['panels']}")
        return
    videos = {os.path.splitext(os.path.basename(p))[0]: (kind, p) for kind in TYPES
              for p in sorted(glob.glob(os.path.join(VIDEO, "simple_contact", kind, "*.mov")))}
    names = args.names or sorted(videos)
    growth = json.load(open(GROWTH_FILE)) if os.path.exists(GROWTH_FILE) else {}
    for name in names:
        kind, video = videos[name]
        trial = make_trial(kind, name, video, growth)
        with open(os.path.join(HERE, name + ".json"), "w") as f:
            json.dump(trial, f, indent=1)
        print(f"wrote {name}.json: base {trial['base']['mm']} at {trial['base']['cut_dir_deg']} deg, "
              f"boxes {[b['pose_mm'] for b in trial['boxes']]} scores {trial['detected']['box_scores']}, "
              f"{len(trial['walls_obb_mm'])} oriented walls, residuals {trial['detected']['calibration_residuals_mm']}")
    if args.growth:
        growth.update(find_growth_starts(names))
        with open(GROWTH_FILE, "w") as f:
            json.dump(growth, f, indent=1, sort_keys=True)
        for name in names:
            kind, video = videos[name]
            with open(os.path.join(HERE, name + ".json"), "w") as f:
                json.dump(make_trial(kind, name, video, growth), f, indent=1)
        print("growth starts:", growth)


if __name__ == "__main__":
    main()
