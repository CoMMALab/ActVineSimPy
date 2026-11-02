'''
Simulator throughput for the paper: how many vine steps per second the planner's own rollout path
(Sim.rollout_batch -> batched_rollout, CUDA graphs, batched QP on the GPU) gets at each batch size, against
the per-element CPU solver (cvxpylayers/diffcp, the sim before batching). Each timed call grows `--steps`
steps from a vine already grown to a working length, with a spread of curvatures, as planner edges do.

    xnew_rrt/py.sh xnew_rrt/bench_throughput.py envs/4a.json envs/1a.json envs/4d.json
    -> out/paper_figs/throughput/throughput.{json,pdf,png}
'''

import argparse, json, os, time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import run_envs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_sim(env, ckpt, backend, device):
    e, scene, cfg, _ = run_envs.load_env(env)
    checkpoint.apply(cfg, checkpoint.load(ckpt))
    cfg.qp_backend, cfg.device = backend, device
    return e, Sim(scene, cfg)


def grown(sim, steps):
    '''A vine grown `steps` steps straight from the env's start (the last state of a rollout).'''
    states, _ = sim.rollout(sim.initial_state(), 0.0, steps)
    return states[-1]


def time_batch(sim, start, B, steps, reps, sync):
    curv = [float(c) for c in np.linspace(-2.0, 2.0, B)]
    sim.rollout_batch([start] * B, curv, steps)            # warm up: graph capture, solver set-up
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        sim.rollout_batch([start] * B, curv, steps)
    sync()
    return (time.perf_counter() - t) / reps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("envs", nargs="+")
    p.add_argument("--checkpoint", default=os.path.join(REPO, "checkpoints", "spin8_manual.json"))
    p.add_argument("--gpu-batch", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    p.add_argument("--cpu-batch", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--steps", type=int, default=25, help="steps per timed rollout (a planner edge)")
    p.add_argument("--grow", type=int, default=60, help="steps grown first")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=os.path.join(REPO, "out", "paper_figs", "throughput"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for env in args.envs:
        for backend, device, batches in (("TORCH_IPM", "cuda", args.gpu_batch), ("DIFFCP", "cpu", args.cpu_batch)):
            e, sim = make_sim(env, args.checkpoint, backend, device)
            sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
            start = grown(sim, args.grow)
            n_obj = len(e.get("boxes", e.get("moveable_objects", [])) or [])
            for B in batches:
                dt = time_batch(sim, start, B, args.steps, args.reps, sync)
                r = dict(env=e["name"], boxes=n_obj, backend=backend, device=device, batch=B,
                         seconds_per_rollout=dt, element_steps_per_s=B * args.steps / dt)
                rows.append(r)
                print(f"{e['name']:4s} {backend:9s} {device:4s} B {B:5d}: {dt:7.3f} s per {args.steps}-step rollout, "
                      f"{r['element_steps_per_s']:9.0f} element-steps/s", flush=True)
    with open(os.path.join(args.out, "throughput.json"), "w") as f:
        json.dump(dict(checkpoint=os.path.relpath(args.checkpoint, REPO), steps=args.steps, rows=rows), f, indent=1)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
    for c, env in zip(colors, dict.fromkeys(r["env"] for r in rows)):
        for backend, ls, mk in (("TORCH_IPM", "-", "o"), ("DIFFCP", "--", "s")):
            rr = [r for r in rows if r["env"] == env and r["backend"] == backend]
            if rr:
                ax.plot([r["batch"] for r in rr], [r["element_steps_per_s"] for r in rr], ls=ls, marker=mk, ms=3,
                        color=c, lw=1.2, label=f"{env} ({rr[0]['boxes']} boxes), {'GPU batched' if backend == 'TORCH_IPM' else 'CPU'}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size"); ax.set_ylabel("vine steps / s")
    ax.grid(True, which="major", lw=0.3, color="#cccccc")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"throughput.{ext}"), dpi=300)
    print("wrote", os.path.join(args.out, "throughput.{json,pdf,png}"))


if __name__ == "__main__":
    main()
