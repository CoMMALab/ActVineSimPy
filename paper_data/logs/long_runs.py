'''
Long-horizon real runs (test split, never fit to), scored only over the frames the sim vine reaches: shape and block
error there, how many frames it reaches, and whether it gets as far as the frame where the real block first moves
(i.e. it reaches the block instead of stalling at a wall).   xnew_rrt/py.sh out/fit/long_reached.py
'''
import json, os, sys

import numpy as np

REPO = "/home/yitian/vine-push"
sys.path.insert(0, os.path.join(REPO, "xnew_rrt"))
import fit_sim  # noqa: E402
import launch_fix  # noqa: E402
import checkpoint  # noqa: E402
from fit_to_master import load_reference, arc_length, MIN_COMPARE_MM  # noqa: E402

MOVED_MM = 5.0
CKPTS = ["contact_envfric_mu2", "contact_launchfix", "base_pbd_brick", "pbd_only"]

scs = launch_fix.apply(load_reference(os.path.join(REPO, "xnew_rrt", "data", "long.npz")))
contact = {}
for n, sc in scs.items():
    comp = [i for i, f in enumerate(sc["frames"]) if arc_length(f)[-1] >= MIN_COMPARE_MM]
    b = np.asarray([sc["boxes"][i] for i in comp])
    disp = np.linalg.norm(b[:, :, :2] - b[:1, :, :2], axis=2).max(1)
    contact[n] = int(np.argmax(disp >= MOVED_MM)) if (disp >= MOVED_MM).any() else len(comp)

out, lines = {}, []
for name in CKPTS:
    cfg = fit_sim.cfg_of(checkpoint.load(os.path.join(REPO, "checkpoints", name + ".json")))
    per = fit_sim.evaluate(scs, cfg, {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT})
    rows = {}
    for n, (r,) in sorted(per.items()):
        reached = r.get("reached", 0)
        rows[n] = dict(frames=r.get("frames"), reached=reached, contact_frame=contact[n],
                       reaches_block=reached > contact[n], shape_reached_mm=r.get("shape_reached_mm"),
                       box_reached_mm=r["box_mm"] if reached else None)
        print(f"{name:20s} {n}: {reached}/{r.get('frames')} frames, real block moves at {contact[n]}, "
              f"shape {r.get('shape_reached_mm', float('nan')):.1f} mm, box {r['box_mm']:.1f} mm", flush=True)
    ok = [v for v in rows.values() if v["reached"]]
    m = dict(shape_mm=float(np.mean([v["shape_reached_mm"] for v in ok])),
             box_mm=float(np.mean([v["box_reached_mm"] for v in ok])),
             reached_frac=float(np.sum([v["reached"] for v in rows.values()]) / np.sum([v["frames"] for v in rows.values()])),
             reaches_block=sum(v["reaches_block"] for v in rows.values()), runs=len(rows))
    out[name] = dict(summary=m, runs=rows)
    lines.append(f"| {name} | {m['shape_mm']:.1f} | {m['box_mm']:.1f} | {100 * m['reached_frac']:.0f}% | "
                 f"{m['reaches_block']}/{m['runs']} |")
with open(os.path.join(REPO, "out", "fit", "long_reached.json"), "w") as f:
    json.dump(out, f, indent=1)
table = ["| Checkpoint | vine [mm] | block [mm] | frames reached | reaches block |", "|---|---|---|---|---|"] + lines
with open(os.path.join(REPO, "out", "fit", "long_reached.md"), "w") as f:
    f.write("\n".join(table) + "\n")
print("\n".join(table))
