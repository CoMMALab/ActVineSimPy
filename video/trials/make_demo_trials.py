'''
Trial jsons for the live 5b run, video/puzzle/demo_5b_push{1,2,3}.mov (one clip per push of the straight-push
plan, xnew_rrt/straight_puzzle.py: up, right, down), written next to this file as demo_5b_push{1,2,3}.json,
plus a check image out/video/_detect/<name>.png. A test set: never fit to.

Setup as in the block puzzle videos (make_puzzle_trials.py): the same pegboard, the same cream block (taken as
BLOCK_MM square), and brown boxes that do not move, here all N_WALL_BOXES of 5b's wall cells. The bricks
stand behind the boxes and are left out. The mount moves between pushes (bottom, left, top edge), each clip
starts where the last one left the block, and the reference frame is frame 0 (stub out, vine not grown).

    python video/trials/make_demo_trials.py
'''

import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import make_puzzle_trials as mp
import make_contact_trials as mc
from calibrate import Calibration

ORIGINAL = [3834, 3835, 3837]          # push 1, 2, 3
DIRECTION = {3834: "up", 3835: "right", 3837: "down"}


def video_path(n):
    return os.path.join(mp.VIDEO, "puzzle", f"demo_5b_push{ORIGINAL.index(n) + 1}.mov")


def extended(cal, frame):
    '''make_contact_trials.extended without its 600 mm cap: the lattice anchors only on the 10-27 hole rows
    below the boxes, and the top row and the block are further up than that.'''
    h, w = frame.shape[:2]
    corners = cal.px_to_mm([[0, 0], [w, 0], [w, h], [0, h]])
    ext = Calibration(H=cal.H, board_mm=(float(corners[:, 0].max()), float(corners[:, 1].max())),
                      image_size=cal.image_size, K=cal.K, dist=cal.dist, camera_height_mm=cal.camera_height_mm,
                      residuals_mm=cal.residuals_mm)
    return ext, ext.rectify(frame, mc.RES)


_detect_base = mc.detect_base


def detect_base(ext, white, black, boxes):
    '''make_contact_trials.detect_base, or with the mount off the board (push 3: clamped past the frame's
    top edge) the stub nearest the board's top, launched down from its top end.'''
    try:
        return _detect_base(ext, white, black, boxes)
    except RuntimeError:
        pass
    stubs = []
    for c in mc.components(white, 1500):
        _, (a, b), _ = mc.rect_mm(ext, c)
        if 60 <= max(a, b) <= 300 and 25 <= min(a, b) <= 80:
            stubs.append(mc.to_mm(ext, c))
    pts = max(stubs, key=lambda p: p[:, 1].max())
    c = pts.mean(0)
    d = np.linalg.svd(pts - c)[2][0]
    if d[1] > 0:
        d = -d                                      # downward
    t = (pts - c) @ d
    base, tip = c + t.min() * d, c + t.max() * d
    h = math.degrees(math.atan2(*(tip - base)[::-1]))
    return [round(float(base[0]), 1), round(float(base[1]), 1)], round(h, 1), float(np.linalg.norm(tip - base)), base


WEAK_MIN_SCORE = 0.45
_detect_puzzle_boxes = mp.detect_puzzle_boxes


def detect_demo_boxes(ext, rect, n_boxes, roi):
    '''make_puzzle_trials.detect_puzzle_boxes, then the weaker brown matches (WEAK_MIN_SCORE: the top-left box
    scores 0.46-0.47, under the stub and the board's edge) up to N_WALL_BOXES, each only one cell (95-125 mm,
    in line) from a box already taken and clear of the block.'''
    out = _detect_puzzle_boxes(ext, rect, n_boxes, roi)
    block, walls = out[0], out[1:]
    bxy = np.array(block["pose_mm"][:2])
    for b in mp._detect(ext, rect, n_boxes + 6, roi):
        if len(walls) >= mp.N_WALL_BOXES or b["score"] < WEAK_MIN_SCORE:
            break
        p = np.array(b["pose_mm"][:2])
        taken = np.array([w["pose_mm"][:2] for w in walls])
        if np.linalg.norm(taken - p, axis=1).min() < 50 or np.linalg.norm(p - bxy) < (mp.BLOCK_MM + mc.BOX_MM) / 2:
            continue
        d = np.abs(taken - p)
        dist = np.linalg.norm(taken - p, axis=1)
        if ((dist >= 95) & (dist <= 125) & (d.min(axis=1) < 20)).any():
            walls.append(b)
    return [block] + walls


def main():
    names = sys.argv[1:]
    mp.N_WALL_BOXES = 14               # 5b's wall cells
    mp.detect_puzzle_boxes = detect_demo_boxes
    mp.video_path = video_path
    mp.DIRECTION = DIRECTION
    mc.extended = mp.extended = extended
    mc.detect_base = detect_base
    for k, n in enumerate(ORIGINAL):
        name = f"demo_5b_push{k + 1}"
        if names and name not in names:
            continue
        trial = mp.make_puzzle_trial(n, name)
        trial["type"] = "demo_5b_" + DIRECTION[n]
        with open(os.path.join(HERE, name + ".json"), "w") as f:
            json.dump(trial, f, indent=1)
        print(f"wrote {name}.json (IMG_{n}, {DIRECTION[n]}): base {trial['base']['mm']} at "
              f"{trial['base']['cut_dir_deg']} deg, block {trial['boxes'][0]['pose_mm']} score "
              f"{trial['detected']['box_scores']}, {len(trial['walls_obb_mm'])} walls {trial['detected']['wall_scores']}")


if __name__ == "__main__":
    main()
