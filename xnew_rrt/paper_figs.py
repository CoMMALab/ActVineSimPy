'''
Paper figures of multi-push solutions: the vine at the end of each push and the blocks where that push
left them (filled, numbered by push), over the start pose of the blocks (grey, dashed).

  rrt     replay an RRT result.json (run_envs.py) on the GPU and split it at the resets
          (the GPU replay matches the planner; a CPU replay drifts over several pushes)
  puzzle  re-run block_puzzle.py with rendering off and draw its pushes
  log     the solution branch from a run's tree_log.pkl (the planner's own states; for runs whose replay drifts)
  grid    an overview of several rrt figures (from their .json), one panel per env

Run from the repo root (GPU):
    python xnew_rrt/paper_figs.py rrt envs/3a.json out/bench_rrt_mu03/runs/3a/s000/results/3a.result.json \
        --checkpoint checkpoints/wrinkle_mu03.json --out out/paper_figs/pushes/3a_s000
    python xnew_rrt/paper_figs.py puzzle envs/5c.json --checkpoint checkpoints/spin8_manual.json --seed 0 \
        --extra "--max-final-angle 15" --out out/paper_figs/pushes/5c_a15_s000
    -> <out>.pdf, <out>.png, <out>.json (the drawn poses, to redraw without the sim)
'''

import argparse, json, math, os, shlex, sys

import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})     # TrueType, not Type 3 (submission check)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import run_envs
import viz
import obstacle_style
from reset_action_rrt import RESET

PUZZLE_CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "spin8_manual.json")
from reset_rrt import reset_state

COLORS = ["#2a78d6", "#6250d6", "#1baf7a", "#d6a02a", "#d62a78", "#2ab8d6", "#898781", "#e0662a", "#4a4a4a"]
START_FACE, START_EDGE = "#e4e2dc", "#6f6d66"
LAUNCH_L, LAUNCH_W = 90.0, 48.0          # overview launch marker [mm]: behind the base, long side along the heading


def snapshot(sim, state):
    xs, ys = viz.vine_xy_mm(state)
    bx, by = sim.base_of(state)[:2]
    return dict(vine=[[viz.MM(float(bx)), viz.MM(float(by))]] + [[x, y] for x, y in zip(xs, ys)],
                boxes=[[list(viz._box_corners_mm(pose, hw, hh))] for pose, hw, hh in sim.object_boxes(state)])


def moved(a, b, tol_mm=3.0):
    '''Did a box's corners move more than tol_mm between two snapshots?'''
    return max(math.dist(p, q) for p, q in zip(a[0], b[0])) > tol_mm


def draw_sequence(ax, start, pushes, vine_radius_mm, label_all=False, fontsize=11, numbers=True):
    '''start: snapshot before push 1; pushes: snapshot at the end of each push. A push's blocks are drawn
    only if they moved; the numbers only when there is more than one push.'''
    for k, c in enumerate(start["boxes"]):
        ax.add_patch(Polygon(c[0], closed=True, facecolor=START_FACE, edgecolor=START_EDGE, lw=1.2, ls="--", zorder=3))
    prev = start
    for n, s in enumerate(pushes):
        col = COLORS[n % len(COLORS)]
        v = s["vine"]
        ax.add_collection(matplotlib.collections.PatchCollection(
            [matplotlib.patches.Circle(p, vine_radius_mm) for p in v[1:]], facecolor=col, edgecolor="none",
            alpha=0.13, zorder=4))
        ax.plot([p[0] for p in v], [p[1] for p in v], "-", color=col, lw=1.6, alpha=0.9, zorder=6)
        ax.add_patch(matplotlib.patches.Circle(v[-1], vine_radius_mm, fill=False, edgecolor=col, lw=1.6, zorder=6))
        for k, c in enumerate(s["boxes"]):
            if not (label_all or moved(c, prev["boxes"][k])):
                continue
            ax.add_patch(Polygon(c[0], closed=True, facecolor=col, edgecolor=col, lw=2.0, alpha=0.28, zorder=7))
            ax.add_patch(Polygon(c[0], closed=True, fill=False, edgecolor=col, lw=2.0, zorder=8))
            cx = sum(p[0] for p in c[0]) / 4; cy = sum(p[1] for p in c[0]) / 4
            if numbers and len(pushes) > 1:
                ax.annotate(str(n + 1), (cx, cy), ha="center", va="center", fontsize=fontsize, weight="bold",
                            color=col, zorder=9)
        prev = s
    handles = [Patch(facecolor=START_FACE, edgecolor=START_EDGE, ls="--", label="start")] + \
              [Patch(facecolor=COLORS[n % len(COLORS)], alpha=0.5, label=f"push {n + 1}") for n in range(len(pushes))]
    return handles


def finish(fig, ax, handles, out, data):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(len(handles), 6),
              fontsize=8, frameon=False, handlelength=1.2, columnspacing=1.0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    with open(out + ".json", "w") as f:
        json.dump(data, f)
    print("wrote", out + ".pdf", out + ".png")


def rrt(args):
    e, scene, cfg, planner = run_envs.load_env(args.env)
    with open(args.result) as f:
        result = json.load(f)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.max_bodies = result.get("sim", {}).get("max_bodies", cfg.max_bodies)
    cfg.device = args.device
    cfg.qp_backend = "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.7)
    sim = Sim(scene, cfg)
    goal_test, goal_distance = run_envs.goal_fns(sim, e)
    steps = result["planner"]["steps_per_edge"]
    state = sim.initial_state()
    start, pushes = snapshot(sim, state), []
    controls = result["controls"]
    for k, (c, _) in enumerate(controls):
        if c == RESET:
            pushes.append(snapshot(sim, state))
            state = reset_state(sim, state)
            continue
        if args.batch > 1:          # through the planner's batched path (cuda graphs), copies of one edge
            new, ok = sim.rollout_batch([state] * args.batch, [float(c)] * args.batch, steps, stop_fn=goal_test)[0]
        else:
            new, ok = sim.rollout(state, float(c), steps, stop_fn=goal_test if k == len(controls) - 1 else None)
        state = new[-1] if new else state
        if not ok:
            break
    pushes.append(snapshot(sim, state))
    reached = bool(goal_test(state))
    print(f"{e['name']}: {len(pushes)} pushes, replay {'reaches' if reached else 'MISSES'} the goal "
          f"({goal_distance(state) * 1000:.0f} mm)")
    fig, ax = viz.new_figure(scene, max_dim_in=args.width)
    viz.draw_scene(ax, sim)
    handles = draw_sequence(ax, start, pushes, cfg.vine_radius_m * 1000)
    finish(fig, ax, handles, args.out, dict(env=e["name"], result=args.result, reached=reached, start=start, pushes=pushes))


def puzzle(args):
    import block_puzzle
    captured = {}
    orig = block_puzzle.draw_pushes

    def capture(e, grid, sim, pushes, path, title):
        captured.update(e=e, grid=grid, sim=sim, pushes=pushes, title=title)
        return orig(e, grid, sim, pushes, path, title)

    block_puzzle.draw_pushes = capture
    if args.straight:
        import straight_puzzle  # noqa: F401  (straight pushes only: patches block_puzzle.PushSearch.candidates)
    tmp = args.out + "_run"
    sys.argv = ["block_puzzle.py", args.env, "--checkpoint", args.checkpoint, "--no-render", "--seed", str(args.seed),
                "--max-attempts", "400", "--out", tmp] + shlex.split(args.extra)
    block_puzzle.main()
    e, grid, sim, pushes = captured["e"], captured["grid"], captured["sim"], captured["pushes"]
    start = snapshot(sim, pushes[0]["states"][0])
    snaps = [snapshot(sim, p["settle_states"][-1] if p["settle_states"] else p["states"][-1]) for p in pushes]
    fig, ax = block_puzzle.new_fig(grid, scale=args.width / grid.n_cols)
    block_puzzle.draw_board(ax, e, grid)
    handles = draw_sequence(ax, start, snaps, sim.cfg.vine_radius_m * 1000)
    ax.set_aspect("equal")
    finish(fig, ax, handles, args.out, dict(env=e["name"], title=captured["title"], start=start, pushes=snaps))


def from_log(args):
    '''The solution branch from a run's tree_log.pkl (run_envs.py with rendering): the planner's own states.'''
    import tree_stages
    log = tree_stages.load_log(args.log)
    nodes = {n["id"]: n for n in log["nodes"]}
    goal = min(log["nodes"], key=lambda n: n["goal_dist_mm"])
    branch = [goal]
    while branch[-1]["parent"] is not None:
        branch.append(nodes[branch[-1]["parent"]])
    branch.reverse()
    snap = lambda n: dict(vine=n["vine"].tolist(), boxes=[[[list(map(float, p)) for p in c]] for c in n["boxes"]])
    start, pushes = snap(branch[0]), []
    for prev, n in zip(branch, branch[1:]):
        if n.get("reset"):
            pushes.append(snap(prev))
    pushes.append(snap(branch[-1]))
    e, scene, cfg, _ = run_envs.load_env(args.env)
    cfg.device, cfg.qp_backend = "cpu", "DIFFCP"
    sim = Sim(scene, cfg)
    print(f"{e['name']}: {len(pushes)} pushes, branch ends {goal['goal_dist_mm']:.0f} mm from the goal (log)")
    fig, ax = viz.new_figure(scene, max_dim_in=args.width)
    viz.draw_scene(ax, sim)
    handles = draw_sequence(ax, start, pushes, cfg.vine_radius_m * 1000)
    finish(fig, ax, handles, args.out, dict(env=e["name"], log=args.log, start=start, pushes=pushes))


def grid(args):
    '''Panels of rrt figures already written (their .json), one per env; the scenes are drawn on the CPU.'''
    n = len(args.figs)
    cols = args.cols
    drop = {}
    for ec in args.drop_col:
        e, c = ec.split(":")
        drop.setdefault(e, []).append(int(c))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(args.width, args.width / cols * rows * 0.8), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, path in zip(axes.flat, args.figs):
        with open(path) as f:
            d = json.load(f)
        with open(os.path.join("envs", d["env"] + ".json")) as f:
            env = json.load(f)
        if "grid" in env:                     # a block puzzle (5x): its own board drawing
            import block_puzzle
            for c in sorted(drop.get(d["env"], []), reverse=True):   # drawing only: an empty grid column cut out
                assert all(r[c] in ".o" for r in env["grid"]["rows"])
                x = (c + 1) * env["grid"]["cell_mm"]    # (anything right of it would need shifting left)
                assert all(w["rect"][2] <= x - env["grid"]["cell_mm"] for w in env["walls"] if "rect" in w)
                env["grid"]["rows"] = [r[:c] + r[c + 1:] for r in env["grid"]["rows"]]
                env["board_mm"][0] -= env["grid"]["cell_mm"]
            block_puzzle.draw_board(ax, env, block_puzzle.Grid(env))
            radius = checkpoint.load(PUZZLE_CK)["sim"]["vine_radius_m"] * 1000
            draw_sequence(ax, d["start"], d["pushes"], radius, fontsize=6, numbers=args.numbers)
            (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        else:
            e, scene, cfg, _ = run_envs.load_env(os.path.join("envs", d["env"] + ".json"))
            cfg.device, cfg.qp_backend = "cpu", "DIFFCP"
            sim = Sim(scene, cfg)
            viz.draw_scene(ax, sim, walls=False)
            obstacle_style.draw_static(ax, sim, lw=0.8, borders=False)
            for t in list(ax.texts):
                t.remove()
            for l in list(ax.lines):          # the start triangle -> a small bar along the launch heading
                if l.get_marker() == "^":
                    l.remove()
            sx, sy, th = scene.start[0] * 1000, scene.start[1] * 1000, scene.start[2]
            u, w = (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))
            ax.add_patch(Polygon([(sx + a * LAUNCH_L * u[0] + b * LAUNCH_W / 2 * w[0],
                                   sy + a * LAUNCH_L * u[1] + b * LAUNCH_W / 2 * w[1])
                                  for a, b in ((-1, -1), (0, -1), (0, 1), (-1, 1))],
                                 closed=True, facecolor=viz.INK, edgecolor="none", zorder=12))
            draw_sequence(ax, d["start"], d["pushes"], cfg.vine_radius_m * 1000, fontsize=6, numbers=args.numbers)
            xmin, xmax, ymin, ymax = obstacle_style.board_bounds(e)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        label = args.labels[args.figs.index(path)] if args.labels else d["env"]
        ax.set_title(label, fontsize=9, pad=2)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.6)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", args.out + ".pdf", args.out + ".png")


def steps(args):
    '''One puzzle solution (a `puzzle` figure's .json): one panel per push, the board after that push (its vine,
    the blocks where it left them, filled) over where the push found them (grey, dashed).'''
    import block_puzzle
    with open(args.fig) as f:
        d = json.load(f)
    with open(os.path.join("envs", d["env"] + ".json")) as f:
        env = json.load(f)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    g = block_puzzle.Grid(env)
    radius = checkpoint.load(PUZZLE_CK)["sim"]["vine_radius_m"] * 1000
    n = len(d["pushes"])
    rows = math.ceil(n / args.cols)
    pw = args.width / args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.width, rows * pw * g.n_rows / g.n_cols + (0.15 * rows + 0.3 if args.titles else 0.04 * rows)),
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    prev = d["start"]
    for i, (ax, s) in enumerate(zip(axes.flat, d["pushes"])):
        block_puzzle.draw_board(ax, env, g)
        draw_sequence(ax, prev, [s], radius, label_all=True)
        for p in ax.patches:                  # one push per panel: color it as in the overview
            if p.get_facecolor()[:3] == matplotlib.colors.to_rgb(COLORS[0]):
                p.set_facecolor(matplotlib.colors.to_rgba(COLORS[i % len(COLORS)], p.get_facecolor()[3]))
            if p.get_edgecolor()[:3] == matplotlib.colors.to_rgb(COLORS[0]):
                p.set_edgecolor(matplotlib.colors.to_rgba(COLORS[i % len(COLORS)], p.get_edgecolor()[3]))
        for c in ax.collections:
            c.set_facecolor(matplotlib.colors.to_rgba(COLORS[i % len(COLORS)], 0.13))
        for l in ax.lines:
            l.set_color(COLORS[i % len(COLORS)])
            l.set_linewidth(0.9)
        for p in ax.patches:                  # thinner outlines at this size (draw_sequence is drawn for 7 in)
            p.set_linewidth(0.5 * p.get_linewidth())
        if args.titles:
            ax.set_title(f"push {i + 1}", fontsize=8, pad=2)
        prev = s
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.4 if args.titles else 0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", args.out + ".pdf", args.out + ".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    gp = sub.add_parser("grid", help="panels of rrt figures already written (their .json)")
    gp.add_argument("--no-numbers", dest="numbers", action="store_false", help="no push numbers (push order by color only)")
    gp.add_argument("figs", nargs="+")
    gp.add_argument("--labels", nargs="*", default=None)
    gp.add_argument("--cols", type=int, default=4)
    gp.add_argument("--width", type=float, default=7.0)
    gp.add_argument("--drop-col", nargs="*", default=[], metavar="ENV:COL",
                    help="puzzle panels: leave out an empty grid column (drawing only, e.g. 5c:13)")
    gp.add_argument("--out", required=True)
    p = sub.add_parser("rrt")
    p.add_argument("env"); p.add_argument("result")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch", type=int, default=1, help="replay each edge as this many copies through rollout_batch, "
                   "as the planner does (try the planner's batch size when the single replay misses)")
    sp = sub.add_parser("steps", help="one panel per push of a puzzle solution (a puzzle figure's .json)")
    sp.add_argument("fig")
    sp.add_argument("--cols", type=int, default=3)
    sp.add_argument("--no-titles", dest="titles", action="store_false", help="no \"push k\" over the panels")
    sp.add_argument("--width", type=float, default=3.5)
    sp.add_argument("--out", required=True)
    lp = sub.add_parser("log", help="the solution branch from a tree_log.pkl")
    lp.add_argument("env"); lp.add_argument("log")
    lp.add_argument("--width", type=float, default=7.0)
    lp.add_argument("--out", required=True)
    q = sub.add_parser("puzzle")
    q.add_argument("env")
    q.add_argument("--checkpoint", default="checkpoints/spin8_manual.json")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--straight", action="store_true", help="straight pushes only (xnew_rrt/straight_puzzle.py)")
    q.add_argument("--extra", default="", help="more block_puzzle.py flags, e.g. \"--max-final-angle 15\"")
    for a in (p, q):
        a.add_argument("--device", default="cuda")
        a.add_argument("--width", type=float, default=7.0, help="figure width [in]")
        a.add_argument("--out", required=True, help="output path without extension")
    args = parser.parse_args()
    os.chdir(run_envs.REPO)
    {"rrt": rrt, "puzzle": puzzle, "grid": grid, "log": from_log, "steps": steps}[args.cmd](args)


if __name__ == "__main__":
    main()
