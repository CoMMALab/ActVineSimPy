'''
Paper figure of the RRT search around one solution: the solution replayed on the GPU (as paper_figs.py rrt),
and from every node on its branch a small random subtree (--root-branch children, each with --branch
children, --depth edges deep; controls drawn as the planner draws them), drawn as tip paths that fade with
depth. With resets (retractable envs) one panel per push: the blocks where the earlier pushes left them, that
push's vine, and the subtrees off its nodes. The subtrees are fresh samples, not the run's own tree (bench
runs keep no tree log).

Run from the repo root (GPU):
    python xnew_rrt/tree_fig.py envs/1c.json out/bench_rrt_mu03/runs/1c/s000/results/1c.result.json \
        --checkpoint checkpoints/wrinkle_mu03.json --out out/paper_figs/tree/1c
    -> <out>.pdf, <out>.png, <out>.json (the drawn paths, to redraw without the sim)
'''

import argparse, json, math, os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import paper_figs
import run_envs
import viz
from paper_figs import COLORS, draw_sequence, snapshot
from reset_action_rrt import RESET
from reset_rrt import reset_state

TREE = "#6f6d66"
CHUNK = 256                     # rollouts per batched call


def tip(s):
    return [float(v) for v in viz._tip_mm(s)]


def subtrees(sim, roots, args, steps, controls):
    '''Random subtrees off every root state: [(root index, depth 1.., tip path)], one per edge.'''
    edges, frontier = [], list(enumerate(roots))
    for depth in range(1, args.depth + 1):
        k = args.root_branch if depth == 1 else args.branch
        starts = [(i, s) for i, s in frontier for _ in range(k)]
        frontier = []
        for c0 in range(0, len(starts), CHUNK):
            chunk = starts[c0:c0 + CHUNK]
            out = sim.rollout_batch([s for _, s in chunk], controls(len(chunk)), steps)
            for (i, s), (states, ok) in zip(chunk, out):
                if not states:
                    continue
                edges.append((i, depth, [tip(s)] + [tip(x) for x in states]))
                if ok and len(states) == steps:          # not failed, not fully grown: expand further
                    frontier.append((i, states[-1]))
        print(f"  depth {depth}: {len(starts)} rollouts", flush=True)
    return edges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env"); ap.add_argument("result")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root-branch", type=int, default=4, help="subtree children of each solution node")
    ap.add_argument("--branch", type=int, default=2, help="children of each subtree node below that")
    ap.add_argument("--depth", type=int, default=4, help="subtree depth [edges]")
    ap.add_argument("--every", type=int, default=1, help="subtrees off every n-th solution node")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--width", type=float, default=3.5, help="panel width [in]")
    ap.add_argument("--cols", type=int, default=3, help="panels per row (with resets)")
    ap.add_argument("--crop", type=float, default=None, metavar="MM",
                    help="zoom every panel to the drawn tree, solution and blocks, with this margin [mm]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--redraw", action="store_true", help="redraw from <out>.json (no sampling; CPU)")
    args = ap.parse_args()
    if args.redraw:
        args.device = "cpu"

    e, scene, cfg, _ = run_envs.load_env(args.env)
    with open(args.result) as f:
        result = json.load(f)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.max_bodies = result.get("sim", {}).get("max_bodies", cfg.max_bodies)
    cfg.device, cfg.qp_backend = args.device, "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.7)
    sim = Sim(scene, cfg)
    goal_test, goal_distance = run_envs.goal_fns(sim, e)
    pl = result["planner"]
    steps = pl["steps_per_edge"]
    rng = np.random.default_rng(args.seed)

    def controls(n):
        return [0.0 if rng.random() < pl["straight_prob"] else float(rng.uniform(-pl["max_curvature"], pl["max_curvature"]))
                for _ in range(n)]

    if args.redraw:
        with open(args.out + ".json") as f:
            saved = json.load(f)
        segs, reached = saved["pushes"], saved["reached"]
        args.depth = saved["args"]["depth"]
    else:
        segs, reached = sample(sim, e, result, goal_test, goal_distance, args, steps, controls)
    draw_fig(e, sim, cfg, segs, reached, args)


def sample(sim, e, result, goal_test, goal_distance, args, steps, controls):
    # replay the solution: each push's start snapshot, its nodes (state after every edge) and its end snapshot
    state = sim.initial_state()
    segs = [dict(start=snapshot(sim, state), nodes=[state])]
    ctl = result["controls"]
    for k, (c, _) in enumerate(ctl):
        if c == RESET:
            segs[-1]["end"] = snapshot(sim, state)
            state = reset_state(sim, state)
            segs.append(dict(start=snapshot(sim, state), nodes=[state]))
            continue
        new, ok = sim.rollout(state, float(c), steps, stop_fn=goal_test if k == len(ctl) - 1 else None)
        state = new[-1] if new else state
        segs[-1]["nodes"].append(state)
        if not ok:
            break
    segs[-1]["end"] = snapshot(sim, state)
    reached = bool(goal_test(state))
    print(f"{e['name']}: {len(segs)} pushes, replay {'reaches' if reached else 'MISSES'} the goal "
          f"({goal_distance(state) * 1000:.0f} mm)", flush=True)

    for seg in segs:
        roots = seg["nodes"][:-1][::args.every]                # not from the push's last node
        seg["edges"] = subtrees(sim, roots, args, steps, controls)
        seg.pop("nodes")
    return segs, reached


def draw_fig(e, sim, cfg, segs, reached, args):
    n = len(segs)
    cols = min(n, args.cols)
    rows = math.ceil(n / cols)
    fig0, ax0 = viz.new_figure(sim.scene, max_dim_in=args.width)
    viz.draw_scene(ax0, sim)
    xlim, ylim = ax0.get_xlim(), ax0.get_ylim()                     # the board
    plt.close(fig0)
    if args.crop is not None:                   # one view for every panel: all trees, vines and blocks
        pts = np.array([q for seg in segs for _, _, p in seg["edges"] for q in p]
                       + [q for seg in segs for snap in (seg["start"], seg["end"])
                          for q in snap["vine"] + [c for b in snap["boxes"] for c in b[0]]])
        lo, hi = pts.min(0) - args.crop, pts.max(0) + args.crop
        xlim = (max(lo[0], min(xlim)), min(hi[0], max(xlim)))
        ylim = (max(lo[1], min(ylim)), min(hi[1], max(ylim)))
    w = args.width
    h = w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, (h + (0.25 if n > 1 else 0)) * rows), squeeze=False)
    for i, seg in enumerate(segs):
        ax = axes[i // cols][i % cols]
        viz.draw_scene(ax, sim)
        draw_tree(ax, seg["edges"], args.depth)
        col = COLORS[i % len(COLORS)]
        paper_figs.COLORS = [col]                   # draw_sequence colors its one push as push 1
        draw_sequence(ax, seg["start"], [seg["end"]], cfg.vine_radius_m * 1000)
        paper_figs.COLORS = COLORS
        if n > 1:
            ax.set_title(f"push {i + 1} of {n}", fontsize=9, color=col)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(pad=0.2)
    handles = [Line2D([], [], color=COLORS[0], lw=1.6, label="solution"),
               Line2D([], [], color=TREE, lw=0.8, label=f"random subtrees ({args.depth} edges deep)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    with open(args.out + ".json", "w") as f:
        json.dump(dict(env=e["name"], result=args.result, reached=reached, args=vars(args), pushes=segs), f)
    print("wrote", args.out + ".pdf", args.out + ".png")


def draw_tree(ax, edges, depth):
    '''Subtree edges as tip paths, fainter with depth, a dot at each edge's end.'''
    for _, d, p in edges:
        a = 0.75 - 0.45 * (d - 1) / max(depth - 1, 1)
        ax.plot([q[0] for q in p], [q[1] for q in p], "-", color=TREE, lw=0.6, alpha=a, zorder=5)
        ax.plot(*p[-1], "o", color=TREE, ms=1.2, alpha=a, zorder=5)


if __name__ == "__main__":
    main()
