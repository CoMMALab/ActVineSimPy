'''
Launch-pose search: where to put the vine's base, and which way to point it, so that a
passive (unactuated) vine pushes a box to a goal.

For the live demo we cannot fabricate an actuated vine per plan, so the only things we choose are
the base pose (x, y, heading) and how long to grow. That is a 3-4 dimensional search, small
enough for guess and check:

    1. Guess --num-poses launch poses (headings spread over --heading-range, base positions
       spread along --base-line if given, otherwise the scene's start position).
    2. Simulate all of them together (one batched dvsim rollout, 'linear' stiffness mode).
    3. Score each by the closest the box center ever got to its goal; the step where that
       happened is how long to grow.
    4. Repeat --rounds times, each time guessing again in a window half as wide around the
       best pose so far.

Run from the repo root:
    python xnew_rrt/pose_search.py --box-goal 0.85 0.60 0.06
    python xnew_rrt/pose_search.py --box-goal 0.85 0.60 0.06 --base-line 0.2 0.2 0.2 0.8 --num-poses 96

Writes under out/pose_search/:
    candidates.png   every pose tried, shaded by score, with the best rollout drawn in full
    best.gif         the best rollout, stopped where the box is closest to its goal
    result.json      the best pose, how long to grow, and the score of every candidate
'''

import argparse, json, math, os, random, time
from dataclasses import asdict

from scene import Sim, SimConfig, default_scene, load_scene  # (puts the repo root on sys.path)
import viz
from viz import plt, LineCollection, LinearSegmentedColormap, Normalize
from rollout import render_states

# closest = dark, farthest = light (one hue)
SCORE_CMAP = LinearSegmentedColormap.from_list("score", ["#0d366b", "#2a78d6", "#cde2fb"])


def guess_poses(center, heading_half_range, line, line_half_range, num_poses, rng):
    '''
    Launch poses (x_m, y_m, heading_rad) around `center` = (position along the base line in
    [0, 1], heading). Evenly spread in heading; random along the line.
    '''
    t0, heading0 = center
    poses = []
    for k in range(num_poses):
        frac = k / (num_poses - 1) if num_poses > 1 else 0.5
        heading = heading0 + (2 * frac - 1) * heading_half_range
        t = min(1.0, max(0.0, t0 + rng.uniform(-line_half_range, line_half_range)))
        x = line[0] + t * (line[2] - line[0])
        y = line[1] + t * (line[3] - line[1])
        poses.append(((x, y, heading), t))
    return poses


def evaluate(sim: Sim, poses, max_steps):
    '''
    Simulate a passive vine from every pose. Returns one dict per pose with the states of the
    rollout up to the step where the box was closest to its goal, and that distance [m].
    '''
    starts = [sim.initial_state(base_m=pose) for pose, _ in poses]
    results = sim.rollout_batch(starts, [0.0] * len(starts), max_steps, stop_fn=sim.box_reached_goal)

    candidates = []
    for (pose, t), start, (states, ok) in zip(poses, starts, results):
        states = [start] + states
        distances = [sim.box_goal_distance_m(s) for s in states]
        best_step = min(range(len(states)), key=lambda i: distances[i])
        candidates.append({"pose": pose, "line_t": t, "ok": ok,
                           "distance_m": distances[best_step], "grow_steps": best_step,
                           "states": states[:best_step + 1]})
    return candidates


def draw_candidates(sim: Sim, candidates, best, out_path):
    fig, ax = viz.new_figure(sim.scene, max_dim_in=13)
    viz.draw_scene(ax, sim)

    distances_mm = [c["distance_m"] * 1000 for c in candidates]
    norm = Normalize(vmin=min(distances_mm), vmax=max(max(distances_mm), min(distances_mm) + 1))

    # tip path of every candidate, shaded by how close it got the box
    order = sorted(range(len(candidates)), key=lambda i: -distances_mm[i])  # best drawn last (on top)
    ax.add_collection(LineCollection([[viz._tip_mm(s) for s in candidates[i]["states"]] for i in order],
                                     colors=[SCORE_CMAP(norm(distances_mm[i])) for i in order],
                                     linewidths=1.2, zorder=6))
    arrow = 0.04 * (sim.scene.bounds_m[1] - sim.scene.bounds_m[0]) * 1000
    for i in order:
        x, y, heading = candidates[i]["pose"]
        ax.annotate("", xy=(x * 1000 + arrow * math.cos(heading), y * 1000 + arrow * math.sin(heading)),
                    xytext=(x * 1000, y * 1000), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=SCORE_CMAP(norm(distances_mm[i])), lw=1.2))

    # the best one in full
    final = best["states"][-1]
    viz.draw_objects(ax, sim, best["states"][0], alpha=0.5, outline_only=True, zorder=3)
    viz.draw_objects(ax, sim, final)
    viz.draw_vine(ax, sim, final, alpha=0.25, zorder=4)

    handles = [viz.Line2D([], [], color=SCORE_CMAP(0.4), lw=1.2, label="tip path of one launch pose"),
               viz.Patch(facecolor=viz.VINE, alpha=0.3, label="best pose: vine when the box is closest"),
               viz.Patch(facecolor=viz.BOX, label="box at that moment"),
               viz.Patch(facecolor="none", edgecolor=viz.BOX, alpha=0.6, label="box at the start")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=8, labelcolor=viz.INK_2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=SCORE_CMAP); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, anchor=(0.0, 0.0))
    cbar.set_label("closest the box got to its goal (mm)", color=viz.INK_2, fontsize=8)
    cbar.ax.tick_params(colors=viz.MUTED, labelsize=7)
    cbar.outline.set_visible(False)

    x, y, heading = best["pose"]
    ax.set_title(f"{sim.scene.name}  |  {len(candidates)} launch poses tried  |  best: base ({x * 1000:.0f}, "
                 f"{y * 1000:.0f}) mm, heading {math.degrees(heading):.1f} deg, grow {best['grow_steps']} steps  ->  "
                 f"box {best['distance_m'] * 1000:.0f} mm from goal center", color=viz.INK, fontsize=10, loc="left")
    viz.save_figure(fig, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=None, help="envs/*.txt file (default: built-in table-top scene)")
    parser.add_argument("--box-goal", type=float, nargs=3, required=True, metavar=("X_M", "Y_M", "RADIUS_M"),
                        help="movable box 0's center should end inside this disc")
    parser.add_argument("--base-line", type=float, nargs=4, default=None, metavar=("X1_M", "Y1_M", "X2_M", "Y2_M"),
                        help="segment the base may be placed on (default: fixed at the scene's start position)")
    parser.add_argument("--heading-range", type=float, default=90.0,
                        help="headings are tried within +- this many degrees of the scene's start heading")
    parser.add_argument("--num-poses", type=int, default=64, help="launch poses per round (one batch)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=250, help="longest rollout, in sim steps (dt = 1/90 s)")
    parser.add_argument("--max-bodies", type=int, default=SimConfig.max_bodies)
    parser.add_argument("--grow-rate", type=float, default=SimConfig.grow_rate_mps)
    parser.add_argument("--vine-radius", type=float, default=SimConfig.vine_radius_m)
    parser.add_argument("--grow-force", type=float, default=SimConfig.grow_force_n,
                        help="stall force of growth [N]; 0 = hard growth constraint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/pose_search")
    args = parser.parse_args()

    scene = load_scene(args.env) if args.env else default_scene()
    scene.box_goal_m = tuple(args.box_goal)
    sim = Sim(scene, SimConfig(max_bodies=args.max_bodies, grow_rate_mps=args.grow_rate,
                               vine_radius_m=args.vine_radius, stiffness_mode="linear",
                               grow_force_n=args.grow_force or None))
    if sim.num_objs == 0:
        raise SystemExit("this scene has no movable box to push")

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    line = args.base_line or (scene.start[0], scene.start[1], scene.start[0], scene.start[1])
    center = (0.5, scene.start[2])
    heading_half_range = math.radians(args.heading_range)
    line_half_range = 0.5 if args.base_line else 0.0

    all_candidates, best = [], None
    t0 = time.time()
    for round_idx in range(1, args.rounds + 1):
        poses = guess_poses(center, heading_half_range, line, line_half_range, args.num_poses, rng)
        candidates = [c for c in evaluate(sim, poses, args.max_steps) if c["ok"]]
        all_candidates += candidates

        best = min(all_candidates, key=lambda c: c["distance_m"])
        print(f"round {round_idx}/{args.rounds}: {len(candidates)}/{len(poses)} rollouts ok, "
              f"heading window +-{math.degrees(heading_half_range):.1f} deg  ->  best so far: box "
              f"{best['distance_m'] * 1000:.0f} mm from goal center "
              f"(heading {math.degrees(best['pose'][2]):.1f} deg, {best['grow_steps']} steps)")

        if best["distance_m"] < scene.box_goal_m[2]:
            print("box is inside its goal region")
            break

        # guess again, closer around the best
        center = (best["line_t"], best["pose"][2])
        heading_half_range /= 2
        line_half_range /= 2

    print(f"search took {time.time() - t0:.1f} s")

    candidates_path = os.path.join(args.out, "candidates.png")
    draw_candidates(sim, all_candidates, best, candidates_path)
    print("wrote", candidates_path)

    dt = float(sim.params.dt)
    result_path = os.path.join(args.out, "result.json")
    with open(result_path, "w") as f:
        json.dump({"scene": scene.name, "sim": asdict(sim.cfg), "box_goal_m": scene.box_goal_m,
                   "best": {"base_x_m": best["pose"][0], "base_y_m": best["pose"][1],
                            "heading_rad": best["pose"][2], "heading_deg": math.degrees(best["pose"][2]),
                            "grow_steps": best["grow_steps"], "grow_time_s": best["grow_steps"] * dt,
                            "grow_length_m": best["grow_steps"] * dt * sim.cfg.grow_rate_mps,
                            "box_goal_distance_m": best["distance_m"],
                            "reached": best["distance_m"] < scene.box_goal_m[2]},
                   "candidates": [{"pose": c["pose"], "grow_steps": c["grow_steps"],
                                   "box_goal_distance_m": c["distance_m"]} for c in all_candidates]}, f, indent=1)
    print("wrote", result_path)

    labels = [f"best launch pose: heading {math.degrees(best['pose'][2]):.1f} deg"] * len(best["states"])
    outputs = render_states(sim, best["states"], labels, args.out, name="best")
    print("wrote", outputs["gif"])
