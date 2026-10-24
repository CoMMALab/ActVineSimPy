'''
A synthetic trial with known answers, to test the video pipeline before there are real videos.

`make` simulates a vine pushing a box in dvsim, draws it as an overhead camera would see it (board
on a darker floor, seen at an angle, with noise and flicker), and writes the video, a trial json
that only says what a real trial would (board size, walls, base, box size and roughly where it
starts; the board corners are found automatically) and the true centerlines and box poses.
`check` compares what video/track.py extracted with the truth.

Run from the repo root (actvine env):
    python video/synthetic.py make  --out out/video_synth
    python video/track.py out/video_synth/trial.json --out out/video_synth/track
    python video/synthetic.py check out/video_synth
'''

import argparse, json, math, os, sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "xnew_rrt"))
sys.path.insert(0, HERE)

FPS = 30
IMAGE = (1280, 800)
# where the board's corners (0,0), (W,0), (W,H), (0,H) land in the image: an oblique view
CORNERS_PX = [(150, 735), (1175, 700), (1110, 70), (210, 95)]
PRE_FRAMES = 10   # frames of the empty board before the vine grows (the reference frame is 0)


def simulate(steps, controls, base_y_m):
    '''
    dvsim rollout of the default scene with the vine launched straight at the movable box. A push
    at the box's center line is a bifurcation (which way the vine buckles off it depends on a
    millimeter), so the default base is 20 mm off it.
    '''
    from scene import Sim, SimConfig, default_scene
    scene = default_scene()
    scene.start = (0.4, base_y_m, 0.0)
    sim = Sim(scene, SimConfig())
    states = [sim.initial_state()]
    for curvature, n in controls:
        new, ok = sim.rollout(states[-1], curvature, n)
        states += new
        if not ok:
            break
    return sim, states[:steps]


def render_board(sim, state, rng, board_mm, draw_vine=True):
    '''Top-down board image at 1 mm/px, y up (row 0 is y = H).'''
    W, H = board_mm
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (170, 190, 200)                                  # table (BGR)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    img = cv2.GaussianBlur(img, (0, 0), 3)
    flip = lambda pts: np.column_stack([pts[:, 0], H - pts[:, 1]]).round().astype(np.int32)
    for w in sim.scene.walls_m:
        x1, y1, x2, y2 = [1000 * v for v in w]
        cv2.rectangle(img, (int(x1), int(H - y2)), (int(x2), int(H - y1)), (70, 70, 80), -1)
    bx, by = [1000 * v for v in sim.scene.start[:2]]
    cv2.rectangle(img, (int(bx - 60), int(H - by - 70)), (int(bx - 5), int(H - by + 70)), (40, 40, 40), -1)  # base housing
    for (pose, hw, hh) in sim.object_boxes(state):
        cx, cy, th = pose
        c, s = math.cos(th), math.sin(th)
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        cv2.fillPoly(img, [flip(local @ np.array([[c, s], [-s, c]]) + [cx, cy])], (60, 110, 150))
        inner = 0.6 * local @ np.array([[c, s], [-s, c]]) + [cx, cy]
        cv2.polylines(img, [flip(inner)], True, (50, 90, 125), 3)   # a printed mark on the lid
    if draw_vine:
        line = truth_centerline(sim, state)
        r = int(round(1000 * sim.cfg.vine_radius_m))
        cv2.polylines(img, [flip(line)], False, (225, 230, 232), 2 * r)
        for p in flip(line[1:]):
            cv2.circle(img, tuple(p), r, (225, 230, 232), -1)
        cv2.polylines(img, [flip(line)], False, (205, 212, 215), r // 2)   # shading along the middle
    return img


def truth_centerline(sim, state):
    '''Base, then every body center [mm].'''
    import viz
    xs, ys = viz.vine_xy_mm(state)
    base = sim.base_of(state)
    return np.array([[float(base[0]), float(base[1])]] + list(zip(xs, ys)))


def camera(board_img, board_mm, rng, flicker):
    W, H = board_mm
    src = np.array([[0, H], [W, H], [W, 0], [0, 0]], np.float32)   # board image px of (0,0), (W,0), (W,H), (0,H)
    M = cv2.getPerspectiveTransform(src, np.array(CORNERS_PX, np.float32))
    frame = np.empty((IMAGE[1], IMAGE[0], 3), np.uint8)
    frame[:] = (45, 50, 55)                                          # floor
    frame = cv2.warpPerspective(board_img, M, IMAGE, dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
    frame = cv2.convertScaleAbs(frame, alpha=flicker)
    noise = rng.normal(0, 3, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def make(out, steps, controls, base_y_m=0.43, seed=0):
    rng = np.random.default_rng(seed)
    sim, states = simulate(steps, controls, base_y_m)
    board_mm = (1525, 1000)
    os.makedirs(out, exist_ok=True)
    video = os.path.join(out, "synthetic.mp4")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE)
    # one sim step is 1/90 s: every third one makes real-time 30 fps video
    shown = states[::3]
    truth_lines, truth_boxes = [], []
    for k in range(PRE_FRAMES + len(shown)):
        st = shown[max(0, k - PRE_FRAMES)]
        board = render_board(sim, st, rng, board_mm, draw_vine=k >= PRE_FRAMES)
        writer.write(camera(board, board_mm, rng, 1.0 + 0.02 * math.sin(k / 7)))
        truth_lines.append(truth_centerline(sim, st) if k >= PRE_FRAMES else np.zeros((0, 2)))
        truth_boxes.append([[b[0][0], b[0][1], b[0][2]] for b in sim.object_boxes(st)])
    writer.release()

    P = max(len(l) for l in truth_lines)
    lines = np.full((len(truth_lines), P, 2), np.nan)
    for k, l in enumerate(truth_lines):
        lines[k, :len(l)] = l
    np.savez_compressed(os.path.join(out, "truth.npz"), centerline=lines,
                        n_points=[len(l) for l in truth_lines], boxes=np.array(truth_boxes))

    (x1, y1, x2, y2), = [[1000 * v for v in o] for o in sim.scene.objects_m]
    trial = {
        "name": "synthetic",
        "video": "synthetic.mp4",
        "board_mm": list(board_mm),
        "corners_px": "auto",
        # the dark walls look like the floor, so what "auto" finds is the table inside them
        "corners_mm": [[100, 100], [1425, 100], [1425, 900], [100, 900]],
        "walls_mm": [[1000 * v for v in w] for w in sim.scene.walls_m],
        "base": {"mm": [1000 * sim.scene.start[0], 1000 * sim.scene.start[1]], "heading_deg": 0.0},
        # roughly where the box is: refine_box finds it exactly on the reference frame
        "boxes": [{"size_mm": [x2 - x1, y2 - y1], "pose_mm": [(x1 + x2) / 2 + 6, (y1 + y2) / 2 - 4, 3.0],
                   "mass_kg": sim.scene.masses_kg[0]}],
        "schedule": [[c, n * 3.33] for c, n in controls],
        "reference_frame": 0,
        "vine_radius_mm": 1000 * sim.cfg.vine_radius_m,
        "vine_height_mm": 0,
    }
    with open(os.path.join(out, "trial.json"), "w") as f:
        json.dump(trial, f, indent=1)
    print(f"wrote {video} ({PRE_FRAMES + len(shown)} frames), trial.json, truth.npz")


def check(out, track_dir=None):
    from track import load_tracks, _arc
    track_dir = track_dir or os.path.join(out, "track")
    tracks = load_tracks(os.path.join(track_dir, "tracks.npz"))
    truth = np.load(os.path.join(out, "truth.npz"))
    shape, tip, length, box_pos, box_ang = [], [], [], [], []
    for k, f in enumerate(tracks["frame"]):
        n = int(truth["n_points"][f])
        if n >= 2 and tracks["valid"][k]:
            ref, got = truth["centerline"][f, :n], tracks["centerlines"][k]
            L_ref, L_got = _arc(ref)[-1], _arc(got)[-1]
            if L_ref > 100:
                s = np.linspace(0, min(L_ref, L_got), 40)
                at = lambda p, s: np.column_stack([np.interp(s, _arc(p), p[:, 0]), np.interp(s, _arc(p), p[:, 1])])
                shape.append(np.linalg.norm(at(ref, s) - at(got, s), axis=1).mean())
                tip.append(np.linalg.norm(ref[-1] - got[-1]))
                length.append(L_got - L_ref)
        tb = truth["boxes"][f]
        box_pos.append(np.linalg.norm(tracks["boxes"][k, :, :2] - tb[:, :2], axis=1).max())
        box_ang.append(np.degrees(np.abs(tracks["boxes"][k, :, 2] - tb[:, 2])).max())
    report = {
        "frames compared": len(shape),
        "centerline error [mm] mean / max": (np.mean(shape), np.max(shape)),
        "tip error [mm] mean / max": (np.mean(tip), np.max(tip)),
        "length error [mm] mean / max abs": (np.mean(length), np.max(np.abs(length))),
        "box center error [mm] mean / max": (np.mean(box_pos), np.max(box_pos)),
        "box angle error [deg] mean / max": (np.mean(box_ang), np.max(box_ang)),
        "box moved [mm] (truth)": float(np.linalg.norm(truth["boxes"][-1, 0, :2] - truth["boxes"][0, 0, :2])),
        "radius [mm] measured (true 50)": float(np.nanmedian(tracks["radius_mm"])),
    }
    for k, v in report.items():
        print(f"{k:40s} " + (" / ".join(f"{x:.1f}" for x in v) if isinstance(v, tuple) else f"{v:.1f}" if isinstance(v, float) else str(v)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default="out/video_synth")
    m.add_argument("--steps", type=int, default=200, help="sim steps (90 per second)")
    m.add_argument("--controls", default="0:200", help="curvature[1/m]:steps, ...")
    m.add_argument("--base-y", type=float, default=0.43, help="vine base y [m]; the box's center is at 0.45")
    c = sub.add_parser("check")
    c.add_argument("out")
    c.add_argument("--track", default=None, help="track.py output folder (default <out>/track)")
    args = parser.parse_args()
    if args.cmd == "make":
        controls = [(float(a), int(b)) for a, b in (part.split(":") for part in args.controls.split(","))]
        make(args.out, args.steps, controls, args.base_y)
    else:
        check(args.out, args.track)


if __name__ == "__main__":
    main()
