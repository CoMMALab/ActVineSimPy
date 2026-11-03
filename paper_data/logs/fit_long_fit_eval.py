'''long_fit (fit on long_train, 9 runs) against its start contact_envfric_mu2: long_train and the held-out long_02
(scored once), as long_reached.py scores (over the frames the sim vine reaches; runs that reach the block).
    xnew_rrt/py.sh out/fit/long_fit_eval.py'''
import os, sys
import numpy as np
REPO = "/home/yitian/vine-push"
sys.path.insert(0, os.path.join(REPO, "xnew_rrt"))
import fit_sim, launch_fix, checkpoint
from fit_to_master import load_reference, arc_length, MIN_COMPARE_MM
for ref in ("long_train", "long_val"):
    scs = launch_fix.apply(load_reference(os.path.join(REPO, "xnew_rrt", "data", ref + ".npz")))
    for name in ("contact_envfric_mu2", "long_fit"):
        cfg = fit_sim.cfg_of(checkpoint.load(os.path.join(REPO, "checkpoints", name + ".json")))
        per = fit_sim.evaluate(scs, cfg, {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT})
        rows = []
        for n, (r,) in sorted(per.items()):
            sc = scs[n]
            comp = [i for i, f in enumerate(sc["frames"]) if arc_length(f)[-1] >= MIN_COMPARE_MM]
            b = np.asarray([sc["boxes"][i] for i in comp])
            disp = np.linalg.norm(b[:, :, :2] - b[:1, :, :2], axis=2).max(1)
            cf = int(np.argmax(disp >= 5)) if (disp >= 5).any() else len(comp)
            rows.append((n, r.get("reached", 0), r.get("frames"), r.get("shape_reached_mm"), r["box_mm"], r.get("reached", 0) > cf))
            print(f"  {ref} {name:20s} {n}: {rows[-1][1]}/{rows[-1][2]} frames, vine {rows[-1][3] or float('nan'):.1f}, block {rows[-1][4]:.1f}, reaches block {rows[-1][5]}", flush=True)
        ok = [x for x in rows if x[1]]
        print(f"{ref} {name}: vine {np.mean([x[3] for x in ok]):.1f} mm, block {np.mean([x[4] for x in ok]):.1f} mm, "
              f"reaches block {sum(x[5] for x in rows)}/{len(rows)}", flush=True)
