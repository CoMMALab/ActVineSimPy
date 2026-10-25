'''
Simulator checkpoints: the physical parameters of a fitted sim, kept like model weights.

A checkpoint is a json file in checkpoints/:
    name, description, created
    parent          the checkpoint the fit started from (None for a hand-set one)
    trained_on      [{"ref": reference npz, "scenarios": [...]}]: the data the parameters were fitted to
    sim             the SimConfig fields that are physical parameters (FIELDS); anything not listed
                    stays SimConfig's / run_envs.DEFAULT_SIM's
    fit             how it was fitted (search, population, generations, which fields were fitted)
    metrics         {dataset name: {"loss", "shape_mm", "box_mm", "per_scenario": {...}}}, filled by
                    `fit_sim.py eval`; several checkpoints evaluated on one dataset are comparable

    python xnew_rrt/checkpoint.py base                 # checkpoints/base_pbd_brick.json from run_envs.DEFAULT_SIM
    python xnew_rrt/checkpoint.py show checkpoints/*.json

run_envs.py, block_puzzle.py and fit_sim.py take --checkpoint <path>.
'''

import argparse, datetime, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, "checkpoints")
FIELDS = ("max_bodies", "grow_rate_mps", "vine_radius_m", "seg_len_m", "stiffness_mode", "grow_force_n",
          "seg_mass_kg", "joint_stiffness", "ang_damp", "seg_inertia_kgm2", "obj_friction")


def make(name, sim, description="", parent=None, trained_on=(), fit=None, metrics=None):
    return {"name": name, "description": description,
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "parent": parent,
            "trained_on": list(trained_on), "sim": {k: sim[k] for k in FIELDS if k in sim},
            "fit": fit or {}, "metrics": metrics or {}}


def save(ckpt, path=None):
    path = path or os.path.join(DIR, ckpt["name"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ckpt, f, indent=1)
    return path


def load(path):
    with open(path) as f:
        return json.load(f)


def apply(cfg, ckpt):
    '''Set a SimConfig's physical fields from a checkpoint (dict or path); returns cfg.'''
    if isinstance(ckpt, str):
        ckpt = load(ckpt)
    for k, v in ckpt["sim"].items():
        if k == "max_bodies":           # an env may need more bodies than the fit did
            cfg.max_bodies = max(cfg.max_bodies, v)
        else:
            setattr(cfg, k, v)
    return cfg


def base():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_envs
    return make("base_pbd_brick", run_envs.DEFAULT_SIM,
                description="run_envs.DEFAULT_SIM: joint stiffness and damping fitted to master's PBD vine "
                            "(fit_to_master.py) and then to the 15 brick demos (video/demos, 2.3 kg brick assumed, "
                            "20 N stall force), rescaled from 18 mm to 30 mm segments; vine radius and box size "
                            "from real_measurements.md. obj_friction 0.08 is not fitted for the 0.503 kg box.",
                trained_on=[{"ref": "xnew_rrt/data/master_ref.npz"}, {"ref": "xnew_rrt/data/video_ref.npz (brick demos)"}],
                fit={"search": "cross-entropy (fit_to_master.py)", "fitted": ["joint_stiffness", "ang_damp"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("base")
    p = sub.add_parser("show")
    p.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.cmd == "base":
        print("wrote", save(base()))
    else:
        for path in args.paths:
            c = load(path)
            print(f"{c['name']}  (parent {c['parent']}; trained on {[t.get('ref') for t in c['trained_on']]})")
            print("  sim:", ", ".join(f"{k}={v}" for k, v in c["sim"].items()))
            for ds, m in c.get("metrics", {}).items():
                print(f"  {ds}: loss {m['loss']:.1f}, shape {m['shape_mm']:.1f} mm, box {m['box_mm']:.1f} mm")


if __name__ == "__main__":
    main()
