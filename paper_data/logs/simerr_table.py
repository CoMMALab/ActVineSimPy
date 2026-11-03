'''
Numbers for the paper's combined simulator-error table (old Tables II and III, plus the live 5b run), per checkpoint:
    simple contact   per type, train mean and the held-out trial: fit_sim's equal-length vine / block error (as Table II)
    long horizon     the 10 runs (never fit to, but by long_fit, fit to the other 9), and long_02 alone (held out by
                     long_fit, scored once): over the frames the sim vine reaches, averaged over the runs that
                     reach any; runs that reach the block (as long_reached.py, Table III)
    puzzle 5b        the 3 live pushes, never fit to: as the long runs (xnew_rrt/data/demo_5b.npz, start poses cleared
                     out of the walls by video/clear_start.py)
    xnew_rrt/py.sh out/fit/combined_table.py  -> out/fit/combined_table.{json,md}
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
CKPTS = ["pbd_only", "base_pbd_brick", "contact_launchfix", "contact_envfric_mu2", "spin8_manual", "long_fit"]
TYPES = ("vine_block", "block_block", "block_wall")
data = lambda f: launch_fix.apply(load_reference(os.path.join(REPO, "xnew_rrt", "data", f)))
sets = dict(train=data("contact_train.npz"), val=data("contact_val.npz"), long=data("long.npz"), long02=data("long_val.npz"), demo=data("demo_5b.npz"))


def contact_frame(sc):
    comp = [i for i, f in enumerate(sc["frames"]) if arc_length(f)[-1] >= MIN_COMPARE_MM]
    b = np.asarray([sc["boxes"][i] for i in comp])
    disp = np.linalg.norm(b[:, :, :2] - b[:1, :, :2], axis=2).max(1)
    return int(np.argmax(disp >= MOVED_MM)) if (disp >= MOVED_MM).any() else len(comp)


def reached(per, scs):
    n_comp = lambda sc: sum(arc_length(f)[-1] >= MIN_COMPARE_MM for f in sc["frames"])
    rows = {n: dict(reached=r.get("reached", 0), frames=n_comp(scs[n]),     # a run whose sim fails reaches none
                    shape=r.get("shape_reached_mm"), box=r["box_mm"],
                    reaches_block=r.get("reached", 0) > contact_frame(scs[n])) for n, (r,) in per.items()}
    ok = [v for v in rows.values() if v["reached"]]
    mean = lambda k: float(np.mean([v[k] for v in ok])) if ok else float("nan")
    return dict(vine_mm=mean("shape"), box_mm=mean("box"),
                reaches_block=sum(v["reaches_block"] for v in rows.values()), runs=len(rows),
                reached_frames=f"{sum(v['reached'] for v in rows.values())}/{sum(v['frames'] for v in rows.values())}",
                per_run=rows)


out = {}
for name in CKPTS:
    cfg = fit_sim.cfg_of(checkpoint.load(os.path.join(REPO, "checkpoints", name + ".json")))
    vals = {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT}
    res = {}
    for split in ("train", "val"):
        per = fit_sim.evaluate(sets[split], cfg, vals)
        m = fit_sim.summary(per)
        res[split] = dict(vine_mm=m["shape_mm"], box_mm=m["box_mm"])
        for t in TYPES:
            rows = [r for n, r in m["per_scenario"].items() if n.startswith(t)]
            res[f"{split}_{t}"] = dict(vine_mm=float(np.mean([r["shape_mm"] for r in rows])),
                                       box_mm=float(np.mean([r["box_mm"] for r in rows])), n=len(rows))
    for split in ("long", "long02", "demo"):
        res[split] = reached(fit_sim.evaluate(sets[split], cfg, vals), sets[split])
    out[name] = res
    print(name, json.dumps({k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_run"}
                            for k, v in res.items()}), flush=True)

with open(os.path.join(REPO, "out", "fit", "combined_table.json"), "w") as f:
    json.dump(out, f, indent=1, default=int)
f1 = lambda r: f"{r['vine_mm']:.1f} / {r['box_mm']:.1f}"
lines = ["| Checkpoint | " + " | ".join(f"{t} train / val" for t in TYPES) + " | contact train | contact val | long (reach) | long_02 (reach) | 5b (reach) |",
         "|---" * 10 + "|"]
for n, r in out.items():
    lines.append(f"| {n} | " + " | ".join(f"{f1(r['train_' + t])} ; {f1(r['val_' + t])}" for t in TYPES)
                 + f" | {f1(r['train'])} | {f1(r['val'])} | {f1(r['long'])} ({r['long']['reaches_block']}/{r['long']['runs']})"
                 + f" | {f1(r['long02'])} ({r['long02']['reaches_block']}/1) | {f1(r['demo'])} ({r['demo']['reaches_block']}/{r['demo']['runs']}, frames {r['demo']['reached_frames']}) |")
with open(os.path.join(REPO, "out", "fit", "combined_table.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
