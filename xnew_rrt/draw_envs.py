'''
Draw every environment file to a picture: envs/<name>.txt -> envs/vis/<name>.png

Shows the static obstacles, the movable boxes (labelled with their mass), the start pose
(with its heading) and the goal region, in the file's units times its `scale:`.

This has its own small parser instead of kinodynamic/env_loader.py, because the older env files
(one-number `bound:`, circle / line obstacles) are in formats that loader does not read.

Run from the repo root:
    python xnew_rrt/draw_envs.py                  # all of envs/*.txt
    python xnew_rrt/draw_envs.py envs/divider.txt # just some
'''

import glob, math, os, sys

from scene import REPO  # (puts the repo root on sys.path)
import viz
from viz import plt, Circle, Polygon


def parse_env(path):
    env = {"name": os.path.splitext(os.path.basename(path))[0], "ob_type": "box", "scale": 1.0,
           "goal_radius": None, "obstacles": [], "dynamic": []}
    section = None

    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue

            key, _, rest = line.partition(":")
            values = rest.split()
            if key == "bound":
                nums = [float(v) for v in values]
                env["bound"] = (nums[0], nums[1] if len(nums) > 1 else nums[0])
            elif key in ("start", "goal"):
                env[key] = [float(v) for v in values]
            elif key in ("goal_radius", "scale"):
                env[key] = float(values[0])
            elif key == "ob_type":
                env["ob_type"] = values[0]
            elif key == "obstacles":
                section = "obstacles"
            elif key == "dynamic_obstacles":
                section = "dynamic"
            elif section is not None:
                env[section].append([float(v) for v in line.split()])

    return env


def draw_env(env_path, out_dir):
    env = parse_env(env_path)
    k = env["scale"]
    bound_x, bound_y = env["bound"][0] * k, env["bound"][1] * k
    span = max(bound_x, bound_y)
    goal_radius = (env["goal_radius"] * k) if env["goal_radius"] is not None else 100.0  # env_loader's default
    # the default is meant for mm-scale files; in the small unit-less ones it is bigger than the scene
    show_goal_disc = goal_radius < 0.3 * span
    if not show_goal_disc:
        goal_radius = 0.0

    xs, ys = [0.0, bound_x], [0.0, bound_y]  # some files put obstacles outside the bound
    for ob in env["obstacles"] + env["dynamic"]:
        if env["ob_type"] == "circle" and ob in env["obstacles"]:
            xs += [(ob[0] - ob[2]) * k, (ob[0] + ob[2]) * k]; ys += [(ob[1] - ob[2]) * k, (ob[1] + ob[2]) * k]
        else:
            xs += [ob[0] * k, ob[2] * k]; ys += [ob[1] * k, ob[3] * k]
    xs += [env["start"][0] * k, env["goal"][0] * k - goal_radius, env["goal"][0] * k + goal_radius]
    ys += [env["start"][1] * k, env["goal"][1] * k - goal_radius, env["goal"][1] * k + goal_radius]
    pad = 0.03 * span
    xmin, xmax, ymin, ymax = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    aspect = (ymax - ymin) / (xmax - xmin)
    size = (9, max(9 * aspect, 2.5)) if aspect <= 1 else (max(9 / aspect, 4), 9)
    fig, ax = plt.subplots(figsize=size, facecolor=viz.SURFACE, layout="constrained")
    ax.set_facecolor(viz.SURFACE)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.tick_params(colors=viz.MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")

    def rect(x1, y1, x2, y2):
        return [(x1 * k, y1 * k), (x2 * k, y1 * k), (x2 * k, y2 * k), (x1 * k, y2 * k)]

    # the declared bound
    ax.add_patch(Polygon([(0, 0), (bound_x, 0), (bound_x, bound_y), (0, bound_y)], closed=True, fill=False,
                         edgecolor=viz.MUTED, lw=1, ls=":", zorder=1))

    for ob in env["obstacles"]:
        if env["ob_type"] == "circle":
            ax.add_patch(Circle((ob[0] * k, ob[1] * k), ob[2] * k, facecolor=viz.WALL_FACE,
                                edgecolor=viz.WALL_EDGE, lw=1, zorder=2))
        elif env["ob_type"] == "line":
            ax.plot([ob[0] * k, ob[2] * k], [ob[1] * k, ob[3] * k], "-", color=viz.WALL_EDGE, lw=3, zorder=2)
        else:
            ax.add_patch(Polygon(rect(*ob[:4]), closed=True, facecolor=viz.WALL_FACE,
                                 edgecolor=viz.WALL_EDGE, lw=1, zorder=2))

    for ob in env["dynamic"]:
        ax.add_patch(Polygon(rect(*ob[:4]), closed=True, facecolor=viz.BOX, edgecolor=viz.SURFACE, lw=2, zorder=4))
        if len(ob) > 4:
            ax.annotate(f"{ob[4]:g} kg", ((ob[0] + ob[2]) / 2 * k, (ob[1] + ob[3]) / 2 * k), ha="center",
                        va="center", color=viz.INK, fontsize=8, zorder=5)

    gx, gy = env["goal"][0] * k, env["goal"][1] * k
    if show_goal_disc:
        ax.add_patch(Circle((gx, gy), goal_radius, facecolor=viz.GOAL, alpha=0.15, zorder=3))
        ax.add_patch(Circle((gx, gy), goal_radius, fill=False, edgecolor=viz.GOAL, lw=2, ls="--", zorder=3))
        ax.annotate("goal", (gx, gy), ha="center", va="center", color=viz.INK_2, fontsize=9, zorder=3)
    else:
        ax.plot([gx], [gy], marker="*", ms=14, color=viz.GOAL, ls="none", zorder=9)
        ax.annotate("goal", (gx, gy), xytext=(0, 10), textcoords="offset points", ha="center",
                    color=viz.INK_2, fontsize=9, zorder=9)

    sx, sy, heading = env["start"][0] * k, env["start"][1] * k, env["start"][2]
    ax.plot([sx], [sy], marker="o", ms=7, color=viz.INK, ls="none", zorder=9)
    ax.annotate("", xy=(sx + 0.08 * span * math.cos(heading), sy + 0.08 * span * math.sin(heading)),
                xytext=(sx, sy), arrowprops=dict(arrowstyle="-|>", color=viz.INK, lw=1.5), zorder=9)
    ax.annotate("start", (sx, sy), xytext=(0, -14), textcoords="offset points", ha="center",
                color=viz.INK_2, fontsize=9, zorder=9)

    notes = f"{len(env['obstacles'])} {env['ob_type']} obstacles"
    if env["dynamic"]:
        notes += f", {len(env['dynamic'])} movable"
    if env["ob_type"] == "tube":
        notes += " (the tube itself is hard-coded in pbd_vine.py, not in this file)"
    ax.set_title(f"{env['name']}   |   {notes}   |   start heading {math.degrees(heading):.0f} deg, "
                 + (f"goal radius {goal_radius:g}" if show_goal_disc else "no goal radius in file"),
                 color=viz.INK, fontsize=10, loc="left")

    out_path = os.path.join(out_dir, f"{env['name']}.png")
    viz.save_figure(fig, out_path)
    return out_path


if __name__ == "__main__":
    env_paths = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO, "envs", "*.txt")))
    out_dir = os.path.join(REPO, "envs", "vis")
    os.makedirs(out_dir, exist_ok=True)

    for env_path in env_paths:
        try:
            print("wrote", os.path.relpath(draw_env(env_path, out_dir), REPO))
        except Exception as e:
            print(f"FAILED {env_path}: {type(e).__name__}: {e}")
