'''
Turn tracked trials (video/track.py) into a reference for system identification.

The output has the same format as xnew_rrt/data/master_ref.npz, so xnew_rrt/fit_to_master.py fits
dvsim's stiffness and damping to the real vine with `--ref`:
    python video/to_reference.py out/video/trial01 out/video/trial02 --out xnew_rrt/data/video_ref.npz
    python xnew_rrt/fit_to_master.py --ref xnew_rrt/data/video_ref.npz --out out/fit_video

Per trial (one scenario each):
  frames      the centerline every `record_mm` of growth (the fit compares shapes at equal length)
  walls_mm    the trial's static boxes [x1, y1, x2, y2]; walls_obb_mm its oriented ones; start = the base (x, y [mm], heading [rad]), the heading
              fitted to the whole straight vine just before it touches a box (launch_heading)
  schedule    [(curvature [1/m], length grown with it [mm]), ...] from the trial's "schedule";
              what the vine was fabricated or actuated with, counted from the first recorded frame
  objects     the movable boxes (size, mass, pose at the first frame), with the box poses stored
              at every recorded frame, so the fit can also score where the boxes went
  grow_rate_mps  the trial's "grow_rate_mps" if given, else estimated from the video
'''

import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track import load_tracks, growth_rate, _arc

RECORD_MM = 20.0


def launch_heading(tracks, still_mm=2.0):
    '''
    Heading [rad] of the straight line through the base that best fits the vine on the last frame
    before any box moves (or the last frame, without boxes): what the vine was launched along.
    More reliable than the first few centimeters the tracker measures it from.
    '''
    base = np.array(tracks["meta"]["base"][:2])
    ok = tracks["valid"].copy()
    if len(tracks["meta"]["boxes"]):
        disp = np.linalg.norm(tracks["boxes"][:, :, :2] - tracks["boxes"][:1, :, :2], axis=2).max(axis=1)
        moved = np.nonzero(disp >= still_mm)[0]
        if len(moved):
            ok[moved[0]:] = False
    if not ok.any():
        return tracks["meta"]["base"][2]
    v = tracks["centerlines"][np.nonzero(ok)[0][-1]] - base
    d = np.linalg.svd(np.vstack([v, -v]))[2][0]
    if d @ v[-1] < 0:
        d = -d
    return float(np.arctan2(d[1], d[0]))


def scenario_from_tracks(tracks, record_mm=RECORD_MM, min_start_mm=30.0, min_box_score=None):
    '''(meta, frames, boxes): a scenario of the fit's reference from one trial's tracks. With
    min_box_score, the scenario ends before the first frame where a box's track scores below it (the
    tracker lost the box: hidden by a panel, under the vine).'''
    meta = tracks["meta"]
    trial = meta["trial"]
    usable = tracks["valid"] & (tracks["length_mm"] >= min_start_mm)
    if min_box_score is not None and len(meta["boxes"]):
        lost = np.nonzero(np.asarray(tracks["box_score"]).min(axis=1) < min_box_score)[0]
        if len(lost):
            usable[lost[0]:] = False
    ok = np.nonzero(usable)[0]
    if len(ok) == 0:
        raise ValueError(f"{trial['name']}: no frame with a vine longer than {min_start_mm} mm")
    first = ok[0]
    L0 = float(tracks["length_mm"][first])

    # a length far above its neighbors' is a tracking glitch (the skeleton jumping across a fold
    # or a hand): it would make every frame after it look short, so skip it
    lengths = tracks["length_mm"][ok]
    local = np.array([np.median(lengths[max(0, i - 3):i + 4]) for i in range(len(ok))])
    ok = ok[lengths <= local + 2 * record_mm]

    frames, boxes, grown = [], [], []
    target = L0
    longest = L0
    for k in ok:
        L = float(tracks["length_mm"][k])
        if L > longest:
            longest = L
        # growth is monotone; a length below the running maximum is tracking noise (or the vine
        # was pushed back), so only frames that reach the next record length count
        if L >= target:
            frames.append(tracks["centerlines"][k].astype(np.float32))
            boxes.append(tracks["boxes"][k].astype(np.float32))
            grown.append(L - L0)
            target = L0 + record_mm * (np.floor((L - L0) / record_mm) + 1)

    schedule = trial.get("schedule")
    if not schedule:
        print(f"warning: {trial['name']} has no schedule; assuming a straight (unactuated) vine")
        schedule = [[0.0, longest - L0 + record_mm]]
    # a measured free growth speed in the trial beats the estimate from the video
    rate = trial["grow_rate_mps"] * 1e3 if trial.get("grow_rate_mps") else growth_rate(tracks)
    start = list(meta["base"][:2]) + [launch_heading(tracks)]
    sc = {"walls_mm": meta["walls_mm"], "start": start, "schedule": schedule,
          # oriented walls [cx, cy, theta_rad, hw, hh] (mm, rad), e.g. a slanted panel: kept apart from the
          # axis-aligned walls_mm, which fit_to_master scales and draws as [x1, y1, x2, y2]
          "walls_obb_mm": trial.get("walls_obb_mm", []),
          "num_frames": len(frames), "initial_body_mm": L0,
          "grow_rate_mps": None if rate is None else rate * 1e-3,
          "objects": [{"size_mm": b["size_mm"], "mass_kg": b["mass_kg"],
                       "pose": tracks["boxes"][first, j].tolist()} for j, b in enumerate(meta["boxes"])],
          "source": {"video": trial.get("video"), "frames": [int(tracks["frame"][first]), int(tracks["frame"][ok[-1]])]}}
    return sc, frames, boxes, grown


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="output folders of video/track.py (or their tracks.npz)")
    parser.add_argument("--out", default="xnew_rrt/data/video_ref.npz")
    parser.add_argument("--record-mm", type=float, default=RECORD_MM)
    parser.add_argument("--min-box-score", type=float, default=None,
                        help="end each scenario before a box's track scores below this")
    args = parser.parse_args()

    out, meta = {}, {}
    for run in args.runs:
        path = run if run.endswith(".npz") else os.path.join(run, "tracks.npz")
        tracks = load_tracks(path)
        name = tracks["meta"]["trial"]["name"]
        sc, frames, boxes, grown = scenario_from_tracks(tracks, args.record_mm, min_box_score=args.min_box_score)
        for k, (f, b) in enumerate(zip(frames, boxes)):
            out[f"{name}/frame{k}"] = f
            out[f"{name}/boxes{k}"] = b
        out[f"{name}/grown_mm"] = np.array(grown, dtype=np.float32)
        meta[name] = sc
        moved = [float(np.linalg.norm(boxes[-1][j, :2] - boxes[0][j, :2])) for j in range(len(sc["objects"]))]
        rate = sc["grow_rate_mps"]
        print(f"{name}: {len(frames)} frames from {sc['initial_body_mm']:.0f} mm, grew {grown[-1]:.0f} mm, "
              f"growth {'n/a' if rate is None else f'{rate * 1e3:.1f} mm/s'}, boxes moved {np.round(moved, 1).tolist()} mm")

    out["meta"] = np.array(json.dumps({"record_mm": args.record_mm, "source": "video", "scenarios": meta}))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
