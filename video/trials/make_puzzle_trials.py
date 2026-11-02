'''
Trial jsons for the block puzzle videos video/puzzle/puzzle_NN.mov (originally IMG_NNNN.mov, in the order
of PUZZLE_ORIGINAL), written next to this file as puzzle_NN.json, plus a check image out/video/_detect/puzzle_NN.png.

Setup, read off the videos: the pegboard and vine mount of the simple-contact videos, six fixed
brown boxes (the 98 mm boxes of real_measurements.md) laid out as a puzzle cell, and one cream block
in a one-cell slot between two of them, which the vine pushes straight out of the slot:
    up     puzzle_01..05: from below, out of the slot and up against a row of four boxes
    left   puzzle_06..10: from the right, out of the slot between the row of four and two boxes below,
           into open board
This is the move the block puzzles (envs/5*.json) are made of, and the one where the simulated
block turned too much on leaving the slot.

Detected on frame 0 (also the reference frame: the stub is already out but the vine has not grown):
    brown boxes   the best matches of make_contact_trials' box template, score >= WALL_MIN_SCORE; they
                  do not move, so they are walls (walls_obb_mm, the footprint under the top face seen;
                  the top and side faces seen are kept out of the vine's mask)
    cream block   the best match of a cream block template (puzzle_block_template.png, cropped from
                  puzzle_01), the one movable box; refined by track.py on the reference frame
    base          as for the simple-contact videos, with the boxes' footprints masked out
The cream block's size, height and mass were not measured. Its top is seen ~143 mm wide, next to the
brown boxes' ~107 mm (98 mm boxes, 98 mm tall), but the slot it sits in is only ~126 mm wide between
the brown boxes' footprints, so it is narrower and taller than a brown box. It is taken as BLOCK_MM
square, which leaves it ~5 mm a side in the slot: the tracked start poses are up to 17 mm off the
slot's center and turned up to 2 deg, and a block that starts inside a wall is thrown out of it by the
sim and jams. BLOCK_HEIGHT_MM tall (for the parallax of its top), as heavy as a brown box.

    python video/trials/make_puzzle_trials.py
'''

import glob, json, math, os, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.dirname(HERE)
REPO = os.path.dirname(VIDEO)
sys.path.insert(0, HERE)
sys.path.insert(0, VIDEO)
import make_contact_trials as mc
from calibrate import read_frame, _apply_h

PUZZLE_ORIGINAL = [3760, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770]   # puzzle_01 ...
DIRECTION = {n: "up" if n <= 3765 else "left" for n in PUZZLE_ORIGINAL}
N_WALL_BOXES = 6
WALL_MIN_SCORE = 0.49
BLOCK_MM = 116.0           # estimated (see above)
BLOCK_HEIGHT_MM = 140.0    # estimated (see above)
BLOCK_KG = mc.BOX_KG       # assumed
TEMPLATE = os.path.join(HERE, "puzzle_block_template.png")
TEMPLATE_FROM = ("puzzle_01", (810.7, 524.7))     # the block's top as seen in puzzle_01, frame 0 [board mm]


def video_path(n):
    return os.path.join(VIDEO, "puzzle", f"puzzle_{PUZZLE_ORIGINAL.index(n) + 1:02d}.mov")


def setup(frame):
    cal, _ = mc.shifted_calibration(frame, {"pegboard_pitch_mm": 25.4, "camera_height_mm": mc.CAMERA_HEIGHT_MM})
    ext, rect = mc.extended(cal, frame)
    return cal, ext, rect


def make_template():
    '''Crop the cream block (axis aligned in puzzle_01) as its matching template, the size of the
    brown template scaled to the block's seen top.'''
    name, (cx, cy) = TEMPLATE_FROM
    frame = read_frame(os.path.join(VIDEO, "puzzle", name + ".mov"), 0)
    _, ext, rect = setup(frame)
    x, y = _apply_h(ext.rect_matrix(mc.RES), [[cx, cy]])[0]
    h = int(round(1.08 * 143.0 / mc.RES / 2))
    cv2.imwrite(TEMPLATE, rect[int(y) - h:int(y) + h + 1, int(x) - h:int(x) + h + 1])


def detect_puzzle_boxes(ext, rect, n_boxes, roi):
    '''The cream block first (its template), then the brown boxes (the box template), as
    make_contact_trials.detect_boxes returns them.'''
    if not os.path.exists(TEMPLATE):
        make_template()
    brown_tpl = mc.TEMPLATE
    try:
        mc.TEMPLATE = TEMPLATE
        block = _detect(ext, rect, 1, roi, min_sep_mm=0.8 * BLOCK_MM)
    finally:
        mc.TEMPLATE = brown_tpl
    brown = [b for b in _detect(ext, rect, N_WALL_BOXES + 1, roi) if b["score"] >= WALL_MIN_SCORE]
    # a brown match on the block (its dark inner box) is not a wall
    bxy = np.array(block[0]["pose_mm"][:2])
    brown = [b for b in brown if np.linalg.norm(np.array(b["pose_mm"][:2]) - bxy) > 0.6 * BLOCK_MM]
    return block + brown[:N_WALL_BOXES]


_detect = mc.detect_boxes


def make_puzzle_trial(n, name):
    video = video_path(n)
    mc.detect_boxes = detect_puzzle_boxes
    try:
        trial = mc.make_trial("puzzle", name, video, {}, n_boxes=1 + N_WALL_BOXES)
    finally:
        mc.detect_boxes = _detect
    frame = read_frame(video, trial["reference_frame"])
    cal, ext, rect = setup(frame)
    block, walls = trial["boxes"][0], trial["boxes"][1:]
    block.update(size_mm=[BLOCK_MM, BLOCK_MM], height_mm=BLOCK_HEIGHT_MM, mass_kg=BLOCK_KG)
    trial["boxes"] = [block]
    trial["type"] = "puzzle_" + DIRECTION[n]
    trial["original"] = f"IMG_{n}"
    # the brown boxes: walls on their footprints (the top face is seen pushed away from the nadir), and
    # neither the footprint nor the top and sides seen are vine
    nadir = cal.px_to_mm([[cal.image_size[0] / 2, cal.image_size[1] / 2]])[0]
    zc = mc.CAMERA_HEIGHT_MM
    obbs, exclude = [], []
    for w in walls:
        seen = np.array(w["pose_mm"][:2])
        foot = nadir + (seen - nadir) * (zc - mc.BOX_HEIGHT_MM) / zc
        th = math.radians(w["pose_mm"][2])
        hw = mc.BOX_MM / 2
        obbs.append([round(float(foot[0]), 1), round(float(foot[1]), 1), round(th, 4), hw, hw])
        mid = (foot + seen) / 2
        ext_ = float(np.linalg.norm(seen - foot)) / 2
        exclude.append([round(float(mid[0]), 1), round(float(mid[1]), 1), round(th, 4), round(hw + ext_ + 4, 1),
                        round(hw + ext_ + 4, 1)])
    trial["walls_obb_mm"] = obbs
    trial["exclude_obb_mm"] = exclude
    # the simple-contact videos' free growth speed: these are too short to measure it (the vine pushes
    # the block from almost the start; track.py's estimates scatter 100-300 mm/s, the pushing phase
    # grows 150-190 mm/s)
    trial["grow_rate_mps"] = mc.GROW_RATE_MPS
    # the block is pale like the clear vine and tall: its side faces must come out of the vine's mask too
    trial["box_silhouette"] = True
    trial["box_margin_mm"] = 10      # the block is seen ~143 mm wide, more than BLOCK_MM at BLOCK_HEIGHT_MM gives
    trial["detected"]["wall_scores"] = trial["detected"]["box_scores"][1:]
    trial["detected"]["box_scores"] = trial["detected"]["box_scores"][:1]
    trial["detected"]["block_assumed"] = dict(size_mm=BLOCK_MM, height_mm=BLOCK_HEIGHT_MM, mass_kg=BLOCK_KG)
    # check image: make_trial's, with the block at its size and the walls' footprints
    path = os.path.join(REPO, "out", "video", "_detect", name + ".png")
    img = cv2.imread(path)
    A = mc.extended(cal, frame)[0].rect_matrix(mc.RES)
    cx, cy, deg = block["pose_mm"]
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    cv2.polylines(img, [_apply_h(A, mc.obb_corners(np.array([cx, cy]), u, BLOCK_MM / 2, BLOCK_MM / 2)).round().astype(np.int32)],
                  True, (0, 255, 255), 2)
    for w in obbs:
        u = np.array([math.cos(w[2]), math.sin(w[2])])
        cv2.polylines(img, [_apply_h(A, mc.obb_corners(np.array(w[:2]), u, w[3], w[4])).round().astype(np.int32)],
                      True, (255, 0, 0), 2)
    cv2.imwrite(path, img)
    return trial


def main():
    names = sys.argv[1:]
    for k, n in enumerate(PUZZLE_ORIGINAL):
        name = f"puzzle_{k + 1:02d}"
        if names and name not in names:
            continue
        trial = make_puzzle_trial(n, name)
        with open(os.path.join(HERE, name + ".json"), "w") as f:
            json.dump(trial, f, indent=1)
        print(f"wrote {name}.json (IMG_{n}, {DIRECTION[n]}): base {trial['base']['mm']} at "
              f"{trial['base']['cut_dir_deg']} deg, block {trial['boxes'][0]['pose_mm']} score "
              f"{trial['detected']['box_scores']}, {len(trial['walls_obb_mm'])} walls {trial['detected']['wall_scores']}")


if __name__ == "__main__":
    main()
