'''
Move each scenario's movable boxes out of the walls they overlap at the start, by the least translation (SAT over
the oriented boxes, a few passes). The tracked start pose can put a block a few mm inside a wall it sits next
to (the live 5b run: a 116 mm block in ~128 mm gaps, its pose measured from its top 140 mm up, the walls'
footprints from theirs); the sim then throws it out of the wall, it jams, and the vine stalls. Only the start
pose changes; the tracked poses the sim is scored against do not. The shift is printed and stored in the
scenario ("start_shift_mm").

    python video/clear_start.py xnew_rrt/data/demo_5b.npz      # in place
'''

import argparse, json, math

import numpy as np

CLEARANCE_MM = 1.0


def corners(cx, cy, th, hw, hh):
    u, v = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    c = np.array([cx, cy])
    return np.array([c + hw * u + hh * v, c - hw * u + hh * v, c - hw * u - hh * v, c + hw * u - hh * v]), (u, v)


def push_out(a, b):
    '''Least translation of box a (corners, axes) out of box b, or None if they do not overlap.'''
    best = None
    for ax in a[1] + b[1]:
        pa, pb = a[0] @ ax, b[0] @ ax
        over = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if over <= 0:
            return None
        sign = 1.0 if pa.mean() > pb.mean() else -1.0
        if best is None or over < best[0]:
            best = (over, sign * ax)
    return (best[0] + CLEARANCE_MM) * best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("--out", default=None, help="default: in place")
    args = ap.parse_args()
    data = dict(np.load(args.ref, allow_pickle=False))
    meta = json.loads(str(data["meta"]))
    for name, sc in meta["scenarios"].items():
        walls = [corners(*w) for w in sc.get("walls_obb_mm", [])]
        walls += [corners((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, 0.0, (w[2] - w[0]) / 2, (w[3] - w[1]) / 2)
                  for w in sc.get("walls_mm", [])]
        for o in sc.get("objects", []):
            x, y, th = o["pose"]
            start = np.array([x, y])
            for _ in range(10):
                moves = [m for w in walls if (m := push_out(corners(x, y, th, *np.divide(o["size_mm"], 2)), w)) is not None]
                if not moves:
                    break
                x, y = np.array([x, y]) + max(moves, key=np.linalg.norm)
            shift = np.array([x, y]) - start
            o["pose"] = [float(x), float(y), th]
            o["start_shift_mm"] = [round(float(v), 1) for v in shift]
            print(f"{name}: block moved {np.linalg.norm(shift):.1f} mm {o['start_shift_mm']}")
    data["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(args.out or args.ref, **data)
    print("wrote", args.out or args.ref)


if __name__ == "__main__":
    main()
