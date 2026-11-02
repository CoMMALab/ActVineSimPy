# contact_envfric with its box-wall friction capped at 2 (the fit ran away to 18.6; at <= 1 two block_wall training
# trials fail, and the Anitescu relaxation lifts sliding contacts off by mu |v_t| dt). Chosen on contact_train only;
# then the held-out contact_val is scored once.     xnew_rrt/py.sh out/fit/envfric_mu2.py   (CPU)
import copy, datetime, os, sys
sys.path.insert(0, "/home/yitian/vine-push/xnew_rrt")
os.chdir("/home/yitian/vine-push")
import fit_sim, checkpoint, launch_fix
from fit_to_master import load_reference

ck = copy.deepcopy(checkpoint.load("checkpoints/contact_envfric.json"))
ck.update(name="contact_envfric_mu2", parent="contact_envfric", created=datetime.datetime.now().isoformat(timespec="seconds"),
          description="contact_envfric with wall_friction capped at 2 by hand (fit: 18.6), chosen on contact_train",
          metrics={})
ck["sim"]["wall_friction"] = 2.0
cfg = fit_sim.cfg_of(ck)
cfg.device, cfg.qp_backend = "cpu", "TORCH_IPM"
for ref, key in (("xnew_rrt/data/contact_train.npz", "contact_train"), ("xnew_rrt/data/contact_val.npz", "contact_val")):
    scs = launch_fix.apply(load_reference(ref))
    m = fit_sim.summary(fit_sim.evaluate(scs, cfg, {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT}))
    ck["metrics"][key] = m
    print(f"{key}: shape {m['shape_mm']:.1f} box {m['box_mm']:.1f} turn {m['box_deg']:.1f} deg", flush=True)
    for n, r in m["per_scenario"].items():
        print(f"    {n:22s} shape {r['shape_mm']:6.1f}  box {r['box_mm']:6.1f}", flush=True)
checkpoint.save(ck, "checkpoints/contact_envfric_mu2.json")
print("wrote checkpoints/contact_envfric_mu2.json")
