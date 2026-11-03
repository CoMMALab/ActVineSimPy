#!/bin/bash
# Baseline: dvsim with only the stiffness/damping fitted to master's PBD vine (no real-video fitting), scored on
# the contact sets with the same launch fix as the fitted checkpoints.   out/fit/pbd_only.sh
cd /home/yitian/vine-push
until [ -f out/fit_master_pbdonly/fit.json ]; do sleep 30; done
until grep -q "^done" out/fit/per_type_train.log 2>/dev/null; do sleep 30; done
xnew_rrt/py.sh - <<'PY' 2>&1 | grep -v -i warn
import json, sys
sys.path.insert(0, "xnew_rrt")
import checkpoint
f = json.load(open("out/fit_master_pbdonly/fit.json"))
base = checkpoint.load("checkpoints/base_pbd_brick.json")
r = 0.018 / 0.03                  # 18 mm fit segments -> 30 mm (same EI), as run_envs.DEFAULT_SIM
sim = dict(base["sim"], joint_stiffness=f["stiffness_val"] * r, ang_damp=f["damping"] * 1e-4 * r)
ck = checkpoint.make("pbd_only", sim, description="unfitted to real data: joint stiffness and damping fitted to master's "
                     "PBD vine only (fit_to_master.py, 20 N stall force), rescaled to 30 mm segments; everything else "
                     "as base_pbd_brick (obj_friction 0.08 nominal, no wrinkle model)", parent=None,
                     trained_on=[{"ref": "xnew_rrt/data/master_ref.npz"}],
                     fit=dict(search="cross-entropy (fit_to_master.py)", fitted=["joint_stiffness", "ang_damp"],
                              source="out/fit_master_pbdonly/fit.json"))
print("wrote", checkpoint.save(ck, "checkpoints/pbd_only.json"), sim["joint_stiffness"], sim["ang_damp"])
PY
xnew_rrt/py.sh xnew_rrt/fit_sim.py eval checkpoints/pbd_only.json checkpoints/base_pbd_brick.json \
  --ref xnew_rrt/data/contact_train.npz xnew_rrt/data/contact_val.npz --launch-fix --per-scenario \
  --table out/fit/baselines.md 2>&1 | grep -v -i warn > out/fit/baselines.log
echo done >> out/fit/baselines.log
