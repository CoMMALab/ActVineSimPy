'''
Staged view of an RRT run: how the tree expanded, stage by stage.

Every panel / frame shows the tree up to the end of one stage (a range of RRT iterations):
  - edges added in that stage in blue, earlier edges in grey (the path the vine TIP took)
  - where the movable boxes were at the nodes: a small dot per node, and the box outline only
    for poses at least --box-spread mm from every outline already drawn (faint), so a cloud of
    boxes that barely moved does not hide everything
  - the path a box was pushed along during an edge (orange), and its start pose (dashed)
  - the node closest to the goal so far: its branch (purple), its vine (faint) and its boxes
The goal is box 0's goal disc when the run planned for a box (tip_block_rrt.py), otherwise the
tip goal.

sim_rrt.py / tip_block_rrt.py / batch_tip_block_rrt.py write, under their --out:
    tree_log.pkl        what is needed to redraw (no sim state)
    tree_stages.png     all stages side by side
    tree_stages.gif     one frame per stage (and tree_stages.html: player with a scrubber)
Redraw an old run with other settings (no simulation, fast):
    python xnew_rrt/tree_stages.py out/batch_tip_block_rrt --stages 6 --box-spread 40
'''

import argparse, math, os, pickle, shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon

import viz

OLD_EDGE = "#c3c2b7"
BOX_SPREAD_MM = 30.0


#----------------------------------------------------------------------- Recording

class TreeLog:
    '''What the staged view needs from the tree, recorded as nodes are added (mm, no torch).'''

    def __init__(self, sim, goal_distance):
        scene = sim.scene
        self.goal_distance = goal_distance  # state -> m; not saved
        self.sim = sim                      # not saved
        self.scene = {
            "name": scene.name,
            "bounds_mm": [v * 1000 for v in scene.bounds_m],
            "walls_mm": [viz._box_corners_mm(pose, hw, hh) for pose, hw, hh in sim.static_boxes()],
            "start_mm": (scene.start[0] * 1000, scene.start[1] * 1000),
            "goal_mm": (scene.goal_m[0] * 1000, scene.goal_m[1] * 1000, scene.goal_radius_m * 1000),
            "box_goal_mm": None if scene.box_goal_m is None else [v * 1000 for v in scene.box_goal_m],
            "vine_radius_mm": sim.cfg.vine_radius_m * 1000,
        }
        self.nodes = []      # one dict per tree node, in node_id order
        self.iter_end = []   # iter_end[i]: number of nodes after iteration i + 1

    def update(self, tree, it):
        for n in tree.all_nodes[len(self.nodes):]:
            self.nodes.append(self._summarize(n, 0 if n.parent is None else it))
        self.iter_end.append(len(self.nodes))

    def _summarize(self, node, it):
        sim = self.sim
        boxes = sim.object_boxes(node.info)
        edge_states = ([node.parent.info] if node.parent is not None else []) + node.edge
        xs, ys = viz.vine_xy_mm(node.info)
        base = sim.base_of(node.info)
        return {
            "id": node.node_id,
            "parent": None if node.parent is None else node.parent.node_id,
            "iter": it,
            "control": node.control,
            "tip_trace": np.array(viz.edge_trace_mm(node), dtype=np.float32).reshape(-1, 2),
            "boxes": [viz._box_corners_mm(pose, hw, hh) for pose, hw, hh in boxes],
            "box_centers": np.array([(viz.MM(p[0]), viz.MM(p[1])) for p, _, _ in boxes], dtype=np.float32).reshape(-1, 2),
            # (num edge states, num boxes, 2)
            "box_traces": np.array([[(viz.MM(p[0]), viz.MM(p[1])) for p, _, _ in sim.object_boxes(s)]
                                    for s in edge_states], dtype=np.float32).reshape(len(edge_states), len(boxes), 2),
            "vine": np.array([(viz.MM(float(base[0])), viz.MM(float(base[1])))] + list(zip(xs, ys)), dtype=np.float32),
            "goal_dist_mm": self.goal_distance(node.info) * 1000,
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"scene": self.scene, "nodes": self.nodes, "iter_end": self.iter_end}, f)


def load_log(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def log_dict(log):
    return {"scene": log.scene, "nodes": log.nodes, "iter_end": log.iter_end} if isinstance(log, TreeLog) else log


#----------------------------------------------------------------------- Drawing

def _new_axes(ax, scene):
    xmin, xmax, ymin, ymax = scene["bounds_mm"]
    ax.set_facecolor(viz.SURFACE)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.tick_params(colors=viz.MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(OLD_EDGE)


def spread_out(poses, centers, min_dist_mm, kept_centers=()):
    '''Indices of the poses whose center is >= min_dist_mm from every pose kept before it.'''
    kept = list(kept_centers)
    picked = []
    for i, c in enumerate(centers):
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 >= min_dist_mm ** 2 for k in kept):
            kept.append(c)
            picked.append(i)
    return picked


def draw_stage(ax, log, it_lo, it_hi, box_spread_mm=BOX_SPREAD_MM, legend=True, title=None):
    '''The tree after iteration it_hi; nodes from iterations (it_lo, it_hi] are this stage's.'''
    log = log_dict(log)
    scene, all_nodes = log["scene"], log["nodes"]
    nodes = [n for n in all_nodes if n["iter"] <= it_hi]
    by_id = {n["id"]: n for n in all_nodes}
    root = all_nodes[0]
    box_goal = scene["box_goal_mm"]
    _new_axes(ax, scene)

    # scene
    for corners in scene["walls_mm"]:
        ax.add_patch(Polygon(corners, closed=True, facecolor=viz.WALL_FACE, edgecolor=viz.WALL_EDGE, lw=1, zorder=1))
    if box_goal is not None:
        gx, gy, gr = box_goal
        label = "box goal"
    else:
        gx, gy, gr = scene["goal_mm"]
        label = "tip goal"
    ax.add_patch(Circle((gx, gy), gr, facecolor=viz.GOAL, alpha=0.15, zorder=2))
    ax.add_patch(Circle((gx, gy), gr, fill=False, edgecolor=viz.GOAL, lw=2, ls="--", zorder=2))
    ax.annotate(label, (gx, gy + gr), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                color=viz.INK_2, fontsize=8, zorder=2)
    sx, sy = scene["start_mm"]
    ax.plot([sx], [sy], marker="^", ms=8, color=viz.INK, zorder=9, ls="none")

    # tree edges: earlier stages grey, this stage blue
    old = [n for n in nodes if n["parent"] is not None and n["iter"] <= it_lo]
    new = [n for n in nodes if n["parent"] is not None and n["iter"] > it_lo]
    if old:
        ax.add_collection(LineCollection([n["tip_trace"] for n in old], colors=OLD_EDGE, linewidths=0.8, zorder=5))
    if new:
        ax.add_collection(LineCollection([n["tip_trace"] for n in new], colors=viz.VINE, linewidths=1.3, zorder=6))
        ends = np.array([n["tip_trace"][-1] for n in new])
        ax.scatter(ends[:, 0], ends[:, 1], s=7, color=viz.VINE, edgecolors=viz.SURFACE, linewidths=0.3, zorder=7)

    # boxes: start pose, a dot per node, faint outlines only where they spread out, push paths
    num_boxes = len(root["boxes"])
    for k in range(num_boxes):
        ax.add_patch(Polygon(root["boxes"][k], closed=True, fill=False, edgecolor=viz.INK_2, lw=1, ls="--", zorder=3))
        centers = np.array([n["box_centers"][k] for n in nodes])
        ax.scatter(centers[:, 0], centers[:, 1], s=3, color=viz.BOX, alpha=0.35, linewidths=0, zorder=3)
        picked = spread_out([n["boxes"][k] for n in nodes], centers, box_spread_mm, kept_centers=[root["box_centers"][k]])
        polys = [nodes[i]["boxes"][k] for i in picked]
        if polys:
            ax.add_collection(PolyCollection(polys, facecolors=viz.BOX, edgecolors=viz.BOX, alpha=0.12,
                                             linewidths=0.8, zorder=3))
        pushes = [n["box_traces"][:, k] for n in nodes if len(n["box_traces"]) > 1 and
                  np.linalg.norm(n["box_traces"][-1, k] - n["box_traces"][0, k]) > 2.0]
        if pushes:
            ax.add_collection(LineCollection(pushes, colors=viz.BOX, linewidths=1.0, alpha=0.6, zorder=4))

    # closest to the goal so far
    best = min(nodes, key=lambda n: n["goal_dist_mm"])
    branch, n = [], best
    while n is not None:
        branch.append(n)
        n = by_id[n["parent"]] if n["parent"] is not None else None
    trace = np.concatenate([n["tip_trace"] for n in reversed(branch) if len(n["tip_trace"])] or [np.zeros((0, 2))])
    if len(trace):
        ax.plot(trace[:, 0], trace[:, 1], "-", color=viz.BEST, lw=2.2, zorder=8)
    vine = best["vine"]
    ax.plot(vine[:, 0], vine[:, 1], "-", color=viz.BEST, lw=1, alpha=0.5, zorder=7)
    ax.add_collection(PolyCollection([_circle(x, y, scene["vine_radius_mm"]) for x, y in vine[1:]],
                                     facecolors=viz.BEST, edgecolors="none", alpha=0.06, zorder=2))
    for corners in best["boxes"]:
        ax.add_patch(Polygon(corners, closed=True, facecolor=viz.BOX, edgecolor=viz.SURFACE, lw=1.5, alpha=0.9, zorder=8))

    if title is None:
        what = "box 0" if box_goal is not None else "tip"
        its = f"iteration {it_hi}" if it_hi == it_lo + 1 else f"iterations {it_lo + 1}-{it_hi}"
        title = (f"{its}  |  {len(nodes)} nodes (+{len(new)})  |  "
                 f"closest {what} to goal: {best['goal_dist_mm']:.0f} mm")
    ax.set_title(title, color=viz.INK, fontsize=9, loc="left")

    if legend:
        ax.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=8, labelcolor=viz.INK_2)


def _circle(x, y, r, n=16):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([x + r * np.cos(t), y + r * np.sin(t)], axis=1)


def legend_handles():
    return [Line2D([], [], color=viz.VINE, lw=1.3, marker="o", ms=3, label="tip paths added this stage"),
            Line2D([], [], color=OLD_EDGE, lw=1, label="earlier tip paths"),
            Line2D([], [], color=viz.BEST, lw=2.2, label="branch closest to goal"),
            Patch(facecolor=viz.BOX, alpha=0.9, label="boxes at that node"),
            Patch(facecolor=viz.BOX, edgecolor=viz.BOX, alpha=0.2, label="box poses reached (spread out)"),
            Line2D([], [], color=viz.BOX, alpha=0.6, lw=1, label="box pushed along an edge"),
            Patch(facecolor="none", edgecolor=viz.INK_2, ls="--", label="box start pose"),
            Patch(facecolor=viz.GOAL, alpha=0.3, edgecolor=viz.GOAL, ls="--", label="goal")]


def stage_bounds(num_iters, num_stages):
    '''[(it_lo, it_hi)]: num_stages consecutive iteration ranges covering 1..num_iters.'''
    ends = sorted(set(int(round(e)) for e in np.linspace(0, num_iters, num_stages + 1)))
    return list(zip(ends[:-1], ends[1:]))


def render_stages(log, out_dir, num_stages=8, box_spread_mm=BOX_SPREAD_MM, frame_ms=900):
    '''Writes tree_stages.png (grid), tree_stages.gif and tree_stages.html (one frame per stage).'''
    log = log_dict(log)
    stages = stage_bounds(len(log["iter_end"]), num_stages)
    xmin, xmax, ymin, ymax = log["scene"]["bounds_mm"]
    aspect = (ymax - ymin) / (xmax - xmin)

    # all stages side by side
    ncols = min(len(stages), 3 if len(stages) in (3, 5, 6, 9) else 4)
    nrows = math.ceil(len(stages) / ncols)
    panel_w = 5.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_w * aspect * nrows + 1.2),
                             facecolor=viz.SURFACE, layout="constrained", squeeze=False)
    for ax, (lo, hi) in zip(axes.flat, stages):
        draw_stage(ax, log, lo, hi, box_spread_mm, legend=False)
        ax.title.set_fontsize(8)
    for ax in list(axes.flat)[len(stages):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=4, frameon=False, fontsize=8,
               labelcolor=viz.INK_2)
    fig.suptitle(f"{log['scene']['name']}: tree expansion in {len(stages)} stages", color=viz.INK, fontsize=11)
    grid_path = os.path.join(out_dir, "tree_stages.png")
    viz.save_figure(fig, grid_path, dpi=90)

    # one frame per stage
    frame_dir = os.path.join(out_dir, "_stage_frames")
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir)
    frames = []
    for k, (lo, hi) in enumerate(stages):
        fig, ax = plt.subplots(figsize=(12, 12 * aspect / 1.3), facecolor=viz.SURFACE, layout="constrained")
        draw_stage(ax, log, lo, hi, box_spread_mm, legend=True)
        ax.set_title(f"stage {k + 1}/{len(stages)}  |  " + ax.get_title(loc="left"), color=viz.INK,
                     fontsize=10, loc="left")
        frames.append(os.path.join(frame_dir, f"s{k:03d}.png"))
        viz.save_figure(fig, frames[-1])
    viz.write_gif(frames, os.path.join(out_dir, "tree_stages.gif"), frame_ms=frame_ms, hold_last_ms=3000)
    viz.write_html_player(frames, os.path.join(out_dir, "tree_stages.html"), title="tree stages", frame_ms=frame_ms)
    shutil.copyfile(frames[-1], os.path.join(out_dir, "tree_final.png"))
    shutil.rmtree(frame_dir)
    return grid_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="an RRT --out directory containing tree_log.pkl")
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--box-spread", type=float, default=BOX_SPREAD_MM,
                        help="draw a box outline only if its center is this far (mm) from every one already drawn")
    args = parser.parse_args()
    log = load_log(os.path.join(args.run_dir, "tree_log.pkl"))
    print("wrote", render_stages(log, args.run_dir, args.stages, args.box_spread))


if __name__ == "__main__":
    main()
