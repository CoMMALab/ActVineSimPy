'''
Trial jsons for the brick-pushing demos in video/demos/ (one per video), written next to this file.

Setup, read off the videos: pegboard with 1 in hole pitch, the vine's mount bolted at the top
right (base at about (1041, 737) mm in the board frame, the same in every video), a clear tube
that grows straight (no actuation) and glances off a brick. Brick: a standard 8 x 3 5/8 x 2 1/4 in
clay brick (203 x 92 x 57 mm) by its top face in the video; its mass is not known (2.0-2.7 kg for
such a brick, 2.3 kg assumed: only mass x friction matters for when it slides, and the fit
identifies the friction for that mass). Camera height ~1.1 m, from how much larger the brick's
top face looks than its footprint.

    python video/trials/make_trials.py
'''
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
# Frame where the vine starts to grow (its length first exceeds the starting stub by 30 mm, from a
# first pass of track.py). The reference frame is taken just before it, so the background is as
# fresh as possible (in IMG_3497 the light at the top of the board changes while the vine waits).
GROWTH_START = {"IMG_3497": 126, "IMG_3499": 162, "IMG_3500": 256, "IMG_3501": 215, "IMG_3509": 172,
                "IMG_3511": 197, "IMG_3512": 182, "IMG_3515": 175, "IMG_3516": 134, "IMG_3518": 172,
                "IMG_3520": 50, "IMG_3521": 158, "IMG_3522": 133, "IMG_3523": 160, "IMG_3524": 128}
LEAD = 15   # frames before growth starts
BOARD_MM = (45 + 1) * 25.4, (33 + 1) * 25.4      # see calibrate.pegboard_lattice
LIP = 12.7                                         # the frame's inner edge, half a pitch past the outer holes
WALL = 40.0

for video in sorted(glob.glob(os.path.join(HERE, "..", "demos", "*.mov"))):
    name = os.path.splitext(os.path.basename(video))[0]
    W, H = BOARD_MM
    trial = {
        "name": name,
        "video": os.path.relpath(video, HERE),
        "pegboard_pitch_mm": 25.4,
        "camera_height_mm": 1100,
        # the wooden frame around the board
        "walls_mm": [[-LIP - WALL, -LIP - WALL, W + LIP + WALL, -LIP], [-LIP - WALL, H + LIP, W + LIP + WALL, H + LIP + WALL],
                     [-LIP - WALL, -LIP, -LIP, H + LIP], [W + LIP, -LIP, W + LIP + WALL, H + LIP]],
        "base": {"mm": [1041, 737], "heading_deg": None, "cut_dir_deg": -141},
        "boxes": [{"size_mm": [203, 92], "pose_mm": "auto", "mass_kg": 2.3, "height_mm": 57}],
        "schedule": [[0.0, 1500]],
        "reference_frame": max(0, GROWTH_START[name] - LEAD),
        "mm_per_px": 1.5,
        "diff_mode": "saturation",
        "diff_threshold": 12,
        "vine_radius_mm": 23,
        "box_channel": "hue",
        "box_search_mm": 15,
        "box_search_deg": 5,
    }
    with open(os.path.join(HERE, name + ".json"), "w") as f:
        json.dump(trial, f, indent=1)
    print("wrote", name + ".json")
