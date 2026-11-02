'''
Funneling test: replay a planned solution open loop under perturbations and see where the block ends up.

Takes an env and a solution (result.json from run_envs.py), and replays the solution's controls from
perturbed launches, in batches on the GPU. Perturbation families, each at several scales:
  launch     base position, Gaussian, --pos-mm per axis
  heading    base heading, Gaussian, --heading-deg
  actuation  every control's curvature, independent Gaussian, --curv (1/m)
  all        the three together
A sample counts as reached if box 0 ends in the env's goal (the same test the planner used).

Run from the repo root (GPU):
    python xnew_rrt/funnel_test.py envs/2a.json out/bench_overnight/runs/2a/s000/results/2a.result.json \
        --checkpoint checkpoints/spin8_manual.json --out out/funnel/2a
    -> funnel.json (every sample: perturbation, final box pose, reached), funnel.png, a table on stdout
'''

import argparse, json, math, os, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import run_envs
import tip_block_rrt
import viz
from reset_action_rrt import RESET

FAMILIES = ("launch", "heading", "actuation", "all")


def edge_len(steps, stride=3):
    '''len(node.edge) kino.plan stores for an edge of `steps` sim steps (every stride-th state, plus the last).'''
    return steps // stride + (0 if steps % stride == 0 else 1)


def schedule(sim, controls, steps_per_edge, goal_test):
    '''Sim steps per control. result.json stores each edge's thinned length, so full edges map back
    to steps_per_edge; the last (goal-reaching) edge is replayed with the goal stop to find its length.
    Returns ([(curvature, steps)], nominal final state).'''
    if any(c == RESET for c, _ in controls):
        raise ValueError("solutions with resets are not supported")
    state, out = sim.initial_state(), []
    for k, (c, n) in enumerate(controls):
        last = k == len(controls) - 1
        if not last and n != edge_len(steps_per_edge):
            raise ValueError(f"control {k}: stored length {n} is not a full {steps_per_edge}-step edge")
        states, ok = sim.rollout(state, c, steps_per_edge, stop_fn=goal_test if last else None)
        if not ok:
            raise RuntimeError(f"nominal replay failed at control {k}")
        if last and edge_len(len(states)) != n:
            print(f"  warning: last edge replayed in {len(states)} steps, stored length {n}")
        out.append((float(c), len(states)))
        state = states[-1] if states else state
    return out, state


def replay(sim, bases, curvs, sched):
    '''Batched open-loop replay. bases (N, 3) [m, m, rad], curvs (N, K). Returns [final state or None].'''
    cur = [sim.initial_state(base_m=tuple(float(v) for v in b)) for b in bases]
    alive = [True] * len(cur)
    for k, (_, steps) in enumerate(sched):
        idx = [i for i in range(len(cur)) if alive[i]]
        res = sim.rollout_batch([cur[i] for i in idx], [float(curvs[i, k]) for i in idx], steps)
        for i, (states, ok) in zip(idx, res):
            if not ok:
                alive[i] = False
            elif states:
                cur[i] = states[-1]
    return [s if a else None for s, a in zip(cur, alive)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("result", help="result.json of a solved run (run_envs.py)")
    parser.add_argument("--checkpoint", default="checkpoints/spin8_manual.json")
    parser.add_argument("--n", type=int, default=256, help="samples per family and scale")
    parser.add_argument("--scales", default="0.5,1,2,4", help="multipliers of the base sigmas")
    parser.add_argument("--pos-mm", type=float, default=5.0)
    parser.add_argument("--heading-deg", type=float, default=2.0)
    parser.add_argument("--curv", type=float, default=0.5, help="[1/m], the planner samples +-3.33")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="out/funnel")
    args = parser.parse_args()
    os.chdir(run_envs.REPO)
    os.makedirs(args.out, exist_ok=True)

    e, scene, cfg, planner = run_envs.load_env(args.env)
    with open(args.result) as f:
        result = json.load(f)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.max_bodies = result["sim"]["max_bodies"]
    cfg.device = args.device
    cfg.qp_backend = "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.7)
    sim = Sim(scene, cfg)
    goal_test, _ = run_envs.goal_fns(sim, e)

    sched, nominal = schedule(sim, result["controls"], result["planner"]["steps_per_edge"], goal_test)
    nom_box = tip_block_rrt.box_positions_mm(nominal)[0]
    print(f"{e['name']}: {len(sched)} controls, {sum(s for _, s in sched)} steps; nominal replay "
          f"{'reaches' if sim.box_reached_goal(nominal) else 'DOES NOT reach'} the goal, box at "
          f"({nom_box[0]:.0f}, {nom_box[1]:.0f}) mm")

    rng = np.random.default_rng(args.seed)
    x0, y0, h0 = scene.start
    c0 = np.array([c for c, _ in sched])
    scales = [float(s) for s in args.scales.split(",")]
    samples, t0 = [], time.time()
    for fam in FAMILIES:
        for sc in scales:
            n = args.n
            dpos = rng.normal(0, args.pos_mm * sc, (n, 2)) if fam in ("launch", "all") else np.zeros((n, 2))
            dh = rng.normal(0, math.radians(args.heading_deg) * sc, n) if fam in ("heading", "all") else np.zeros(n)
            dc = rng.normal(0, args.curv * sc, (n, len(c0))) if fam in ("actuation", "all") else np.zeros((n, len(c0)))
            bases = np.stack([x0 + dpos[:, 0] / 1000, y0 + dpos[:, 1] / 1000, h0 + dh], 1)
            finals = replay(sim, bases, c0[None] + dc, sched)
            for i, s in enumerate(finals):
                row = dict(family=fam, scale=sc, dx_mm=float(dpos[i, 0]), dy_mm=float(dpos[i, 1]),
                           dheading_deg=float(math.degrees(dh[i])), dcurv=dc[i].tolist(), ok=s is not None)
                if s is not None:
                    bx, by = tip_block_rrt.box_positions_mm(s)[0]
                    row.update(box_x_mm=bx, box_y_mm=by, box_deg=math.degrees(float(s.moveable_obj_pose[0, 0, 2])),
                               reached=bool(sim.box_reached_goal(s)))
                samples.append(row)
            print(f"  {fam} x{sc}: {n} rollouts, {time.time() - t0:.0f} s", flush=True)

    # table: per family and scale
    g = e["goal"]
    lines = ["family  scale  sigma                       reached  box spread [mm]  to nominal [mm]  failed"]
    stats = []
    for fam in FAMILIES:
        for sc in scales:
            rs = [r for r in samples if r["family"] == fam and r["scale"] == sc]
            ok = [r for r in rs if r["ok"]]
            P = np.array([[r["box_x_mm"], r["box_y_mm"]] for r in ok]).reshape(-1, 2)
            spread = float(np.sqrt(P.var(0).sum())) if len(P) else float("nan")
            to_nom = float(np.linalg.norm(P - np.array(nom_box), axis=1).mean()) if len(P) else float("nan")
            sig = {"launch": f"{args.pos_mm * sc:.1f} mm", "heading": f"{args.heading_deg * sc:.1f} deg",
                   "actuation": f"{args.curv * sc:.2f} 1/m",
                   "all": f"{args.pos_mm * sc:.1f}mm {args.heading_deg * sc:.1f}deg {args.curv * sc:.2f}/m"}[fam]
            reached = sum(r.get("reached", False) for r in rs) / len(rs)
            stats.append(dict(family=fam, scale=sc, sigma=sig, reached=reached, spread_mm=spread,
                              to_nominal_mm=to_nom, failed=len(rs) - len(ok)))
            lines.append(f"{fam:9s} {sc:4.1f}  {sig:26s}  {100 * reached:5.1f}%  {spread:14.1f}  {to_nom:15.1f}  {len(rs) - len(ok):6d}")
    print("\n".join(lines))
    with open(os.path.join(args.out, "funnel.json"), "w") as f:
        json.dump(dict(env=e["name"], result=args.result, checkpoint=args.checkpoint, args=vars(args),
                       schedule=sched, nominal_box_mm=nom_box, stats=stats, samples=samples), f)

    # figure: one panel per family, final box centers colored by scale, over the nominal final state
    import matplotlib.pyplot as plt
    # top row: the whole board; bottom row: zoomed on the goal (the block drawing hides the points up top)
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5 * len(FAMILIES), 9.4))
    cmap = plt.get_cmap("viridis")
    zoom = max(150.0, 1.5 * g["radius"])
    for row in range(2):
        for ax, fam in zip(axes[row], FAMILIES):
            viz.draw_scene(ax, sim, nominal if row == 0 else None)
            if row == 0:
                viz.draw_vine(ax, sim, nominal)
            ax.add_patch(plt.Circle(g["center"], g["radius"], fill=False, ls="--", color=viz.INK, lw=1, zorder=8))
            for j, sc in enumerate(reversed(scales)):
                ok = [r for r in samples if r["family"] == fam and r["scale"] == sc and r["ok"]]
                ax.scatter([r["box_x_mm"] for r in ok], [r["box_y_mm"] for r in ok], s=5 if row == 0 else 9,
                           alpha=0.6, zorder=9, color=cmap(1 - (j + 0.5) / len(scales)), label=f"x{sc:g}")
            ax.plot(*nom_box, "r+", ms=10, mew=2, zorder=10)
            if row == 0:
                s1 = [s for s in stats if s["family"] == fam and s["scale"] == 1.0]
                ax.set_title(f"{fam}" + (f" (x1: {s1[0]['sigma']}, {100 * s1[0]['reached']:.0f}% reach)" if s1 else ""),
                             fontsize=9)
                ax.set_xlim(0, e["board_mm"][0]); ax.set_ylim(0, e["board_mm"][1])
            else:
                cx, cy = g["center"]
                ax.set_xlim(cx - zoom, cx + zoom); ax.set_ylim(cy - zoom, cy + zoom)
                ax.set_title(f"{fam}, zoomed on the goal", fontsize=9)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="lower right", markerscale=2)
    fig.suptitle(f"{e['name']}: final block centers, open-loop replay of the solution under perturbation "
                 f"(+ nominal, dashed goal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "funnel.png"), dpi=110)
    print("wrote", os.path.join(args.out, "funnel.json"), os.path.join(args.out, "funnel.png"))


if __name__ == "__main__":
    main()
