'''
Load the environments in envs/*.json (written by envs/make_envs.py), draw them, and run the basic
planner on each.

    b2g_unretractable   batched RRT over (tip, box positions) to box 0's goal
    vine_to_goal        the same RRT, to the tip goal
    b2g_retractable     the same RRT with vine resets as an action in the one tree
All three are reset_action_rrt.plan (resets off for the first two), run until the goal, --iters
batched iterations, or the time limit (DEFAULT_PLANNER / CATEGORY_PLANNER, minutes).

Sim settings (DEFAULT_SIM): the real vine and boxes (real_measurements.md) and the stiff vine fitted
to the brick videos. Joint stiffness and damping are rescaled from the fit's 18 mm segments to 30 mm
ones (same bending stiffness EI = k * segment length); 30 mm segments keep the long vines of these
boards under ~90 bodies. The env's "planner" block can override max_bodies and any DEFAULT_PLANNER key.

The sim runs on the CPU only (see the README's "How well the sim batches"); CUDA is capped at half
the GPU's memory anyway, in case anything does touch it.
Run from the repo root (actvine env):
    python xnew_rrt/run_envs.py --draw-only                 # envs/vis/<name>.png for every env
    python xnew_rrt/run_envs.py envs/1a.json envs/4b.json   # plan on some
    python xnew_rrt/run_envs.py                             # plan on all
Per env, under out/envs/<name>/:
    env.png          the environment
    tree.gif         the tree, --tree-frames frames spread over the run (or one per --tree-every iterations)
    solution.gif/.mp4  the best branch: the solution, or the closest attempt (title says which)
    result.json      reached or not, distance to goal, nodes, resets, controls; summary.json for all envs
'''

import argparse, json, math, os, random, sys
from dataclasses import asdict

import numpy as np
import torch

from scene import Scene, Sim, SimConfig, scene_to_dict  # (puts the repo root on sys.path)
import reset_action_rrt
import tree_stages
import viz
from rollout import render_states

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALL_OUT_MM = 50.0     # boundary walls, outside the board

# real vine and boxes, stiffness from the video fit (see the module docstring)
DEFAULT_SIM = dict(max_bodies=60, grow_rate_mps=0.25, vine_radius_m=0.02426, seg_len_m=0.03,
                   stiffness_mode="curvature", grow_force_n=20.0, seg_mass_kg=0.005,
                   joint_stiffness=40.0 * 0.018 / 0.03, ang_damp=1.73e-3 * 0.018 / 0.03,
                   seg_inertia_kgm2=1.0e-3, obj_friction=0.08)
DEFAULT_PLANNER = dict(iters=100000, time_limit_min=45, batch_size=32, steps_per_edge=25, max_curvature=3.33,
                       goal_bias=0.1, box_weight=1.0,
                       # resets (b2g_retractable only)
                       unlock_mm=50.0, straight_prob=0.3)
CATEGORY_PLANNER = {"b2g_retractable": dict(time_limit_min=90)}


#----------------------------------------------------------------------- loading

def load_env(path):
    '''(env dict, Scene, SimConfig, planner settings) of one envs/*.json.'''
    with open(path) as f:
        e = json.load(f)
    mm = 1e-3
    W, H = e["board_mm"]
    t = WALL_OUT_MM
    walls = [[-t * mm, -t * mm, (W + t) * mm, 0.0], [-t * mm, H * mm, (W + t) * mm, (H + t) * mm],
             [-t * mm, 0.0, 0.0, H * mm], [W * mm, 0.0, (W + t) * mm, H * mm]]
    for w in e["walls"]:
        if "rect" in w:
            walls.append([v * mm for v in w["rect"]])
        else:                                         # a segment -> oriented box, ends capped
            (ax, ay), (bx, by) = w["from"], w["to"]
            L = math.hypot(bx - ax, by - ay)
            walls.append([(ax + bx) / 2 * mm, (ay + by) / 2 * mm, math.atan2(by - ay, bx - ax),
                          (L + w["thickness"]) / 2 * mm, w["thickness"] / 2 * mm])
    objects = [[(b["center"][0] - b["size"][0] / 2) * mm, (b["center"][1] - b["size"][1] / 2) * mm,
                (b["center"][0] + b["size"][0] / 2) * mm, (b["center"][1] + b["size"][1] / 2) * mm] for b in e["boxes"]]
    g = e["goal"]
    start = (e["start"]["xy"][0] * mm, e["start"]["xy"][1] * mm, math.radians(e["start"]["heading_deg"]))
    kw = dict(walls_m=walls, objects_m=objects, masses_kg=[b["mass_kg"] for b in e["boxes"]], start=start,
              bounds_m=(-t * mm, (W + t) * mm, -t * mm, (H + t) * mm), name=e["name"],
              objects_theta=[math.radians(b["angle_deg"]) for b in e["boxes"]])
    if g["type"] == "box":
        if g.get("box", 0) != 0:
            raise ValueError("the box goal is for box 0: list the goal's box first")
        kw.update(goal_m=(g["center"][0] * mm, g["center"][1] * mm), goal_radius_m=0.0,
                  box_goal_m=(g["center"][0] * mm, g["center"][1] * mm, g["radius"] * mm))
        if "angle_deg" in g:
            kw["box_goal_theta"] = (math.radians(g["angle_deg"]), math.radians(g.get("angle_tol_deg", 5.0)))
    else:
        kw.update(goal_m=(g["center"][0] * mm, g["center"][1] * mm), goal_radius_m=g["radius"] * mm)
    planner = dict(DEFAULT_PLANNER, **CATEGORY_PLANNER.get(e["category"], {}),
                   **{k: v for k, v in e.get("planner", {}).items() if k in DEFAULT_PLANNER})
    cfg = SimConfig(**dict(DEFAULT_SIM, **{k: v for k, v in e.get("planner", {}).items() if k == "max_bodies"}))
    return e, Scene(**kw), cfg, planner


#----------------------------------------------------------------------- drawing

def draw_env(e, sim, path):
    import matplotlib
    from matplotlib.patches import Polygon
    fig, ax = viz.new_figure(sim.scene, max_dim_in=9)
    start = sim.initial_state()
    viz.draw_scene(ax, sim, start)
    viz.draw_vine(ax, sim, start)
    for k, (pose, hw, hh) in enumerate(sim.object_boxes(start)):
        ax.annotate(str(k), (viz.MM(pose[0]), viz.MM(pose[1])), ha="center", va="center", color="w",
                    fontsize=8, weight="bold", zorder=6)
    sx, sy, h = sim.scene.start
    ax.annotate("", (sx * 1000 + 90 * math.cos(h), sy * 1000 + 90 * math.sin(h)), (sx * 1000, sy * 1000),
                arrowprops=dict(arrowstyle="->", color=viz.INK, lw=1.5), zorder=9)
    g = e["goal"]
    if g["type"] == "box" and "angle_deg" in g:            # the goal pose
        size = e["boxes"][0]["size"]
        a = math.radians(g["angle_deg"])
        c, s = math.cos(a), math.sin(a)
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(size) / 2
        ax.add_patch(Polygon(local @ np.array([[c, s], [-s, c]]) + g["center"], closed=True, fill=False,
                             ec=viz.BOX, lw=1.5, ls=":", zorder=3))
    ax.set_title(f"{e['name']} ({e['category']}): {e['description']}", color=viz.INK, fontsize=9, loc="left")
    viz.save_figure(fig, path)


def gif_to_mp4(gif_path, mp4_path, fps=15):
    import cv2, imageio.v2 as imageio
    frames = imageio.mimread(gif_path, memtest=False)
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
    for f in frames:
        f = np.asarray(f)[:h2, :w2, :3]
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


#----------------------------------------------------------------------- planning

def goal_fns(sim, e):
    if e["goal"]["type"] == "box":
        return sim.box_reached_goal, sim.box_goal_distance_m
    return sim.reached_goal, sim.goal_distance_m


def run(e, sim, planner, out, seed, tree_frames, tree_every=None):
    '''Plan with reset_action_rrt: resets on for b2g_retractable, off (the batched RRT) otherwise.'''
    goal_test, goal_distance = goal_fns(sim, e)
    retractable = e["category"] == "b2g_retractable"
    log = reset_action_rrt.ResetTreeLog(sim, goal_distance)
    tree, best, stats = reset_action_rrt.plan(
        sim, planner["iters"], planner["batch_size"], planner["steps_per_edge"], planner["max_curvature"],
        planner["goal_bias"], random.Random(seed), goal_test, goal_distance,
        unlock_mm=planner["unlock_mm"] if retractable else None,
        straight_prob=planner["straight_prob"] if retractable else 0.0,
        box_weight=planner["box_weight"], time_limit_s=60 * planner["time_limit_min"],
        on_iteration=lambda it, t, *_: log.update(t, it))
    log.save(os.path.join(out, "tree_log.pkl"))
    iters = len(log.iter_end)
    stages = math.ceil(iters / tree_every) if tree_every else min(iters, tree_frames)
    tree_stages.render_stages(log, out, max(1, stages), grid=stages <= 12)
    os.replace(os.path.join(out, "tree_stages.gif"), os.path.join(out, "tree.gif"))
    os.remove(os.path.join(out, "tree_stages.html"))

    reached = tree.goal_node is not None
    dist = goal_distance(best.info)
    states, labels = reset_action_rrt.branch_states(best)
    tag = "SOLVED" if reached else f"FAILED, closest {dist * 1000:.0f} mm from the goal"
    render_states(sim, states, [f"{e['name']} {tag} | {l}" for l in labels], out, name="solution", every=3)
    controls = reset_action_rrt.branch_controls(best)
    return dict(reached=reached, goal_distance_mm=dist * 1000, pushes=1 + sum(c == reset_action_rrt.RESET for c, _ in controls),
                controls=controls, **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="*", help="envs/*.json (default: all)")
    parser.add_argument("--draw-only", action="store_true")
    parser.add_argument("--iters", type=int, default=None, help="override every env's batched iterations")
    parser.add_argument("--time-limit", type=float, default=None, help="override every env's time limit [min]")
    parser.add_argument("--tree-frames", type=int, default=40, help="tree.gif frames, spread over the run")
    parser.add_argument("--tree-every", type=int, default=None, help="instead one tree.gif frame per this many iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/envs")
    args = parser.parse_args()

    paths = args.envs or sorted(p for p in (os.path.join(REPO, "envs", f) for f in os.listdir(os.path.join(REPO, "envs")))
                                if p.endswith(".json"))
    os.makedirs(os.path.join(REPO, "envs", "vis"), exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)
    for path in paths:
        e, scene, cfg, planner = load_env(path)
        if args.iters:
            planner["iters"] = args.iters
        if args.time_limit:
            planner["time_limit_min"] = args.time_limit
        sim = Sim(scene, cfg)
        out = os.path.join(args.out, e["name"])
        os.makedirs(out, exist_ok=True)
        draw_env(e, sim, os.path.join(REPO, "envs", "vis", e["name"] + ".png"))
        draw_env(e, sim, os.path.join(out, "env.png"))
        if args.draw_only:
            print("drew", e["name"])
            continue
        print(f"=== {e['name']} ({e['category']})", flush=True)
        result = run(e, sim, planner, out, args.seed, args.tree_frames, args.tree_every)
        gif_to_mp4(os.path.join(out, "solution.gif"), os.path.join(out, "solution.mp4"))
        result.update(env=e["name"], category=e["category"], scene=scene_to_dict(scene), sim=asdict(cfg), planner=planner)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)
        print(f"{e['name']}: {'reached the goal' if result['reached'] else 'did not reach the goal'}, "
              f"{result['goal_distance_mm']:.0f} mm, {result['nodes']} nodes, {result['resets']} resets, "
              f"{result['iterations']} iterations in {result['seconds'] / 60:.0f} min", flush=True)
        write_summary(args.out)


def write_summary(out_dir):
    '''summary.json from every result.json under out_dir (several runs can share it).'''
    summary = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name, "result.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            summary[name] = {k: r.get(k) for k in ("category", "reached", "goal_distance_mm", "pushes", "resets",
                                                    "nodes", "iterations", "seconds", "stopped")}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
