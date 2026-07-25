"""Render the dvsim dynamic-obstacle demo GIFs (vine pushing / object-wall / object-object /
spin / sPAM curve). Everything is specified in SI (meters, kg, m/s) via the dvsim.si layer;
the sim runs in internal non-dim units and the plots are drawn in millimeters.

Run with the project's env, from anywhere:
    /home/zak/micromamba/envs/vine/bin/python dvsim/demos/render_demos.py
GIFs are written next to this script (dvsim/demos/*.gif).
"""
import os, sys, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from PIL import Image
import dvsim.solver as solver
from dvsim.vine import create_state_batched, init_state_batched
from dvsim.dynamic_vine import step
from dvsim import si

MM = lambda x: si.len_to_mm(x)          # internal length -> mm (for display)


def _obb_corners_mm(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c * lx - s * ly), MM(cy + s * lx + c * ly))
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def render(name, gifname, objs_m, masses_kg, frames, xlim_mm, ylim_mm=(-45, 45),
           obstacles_m=None, grow_rate_mps=0.3):
    """objs_m: AABB boxes [x1,y1,x2,y2] in METERS; masses_kg in kg; obstacles_m: static walls
    (AABB, meters) or None; grow_rate_mps in m/s. Plot limits are in mm."""
    solver.cvxpylayer = None                       # reset cached QP layer (shapes differ per scene)
    params = si.vine_params_si(max_bodies=40, obstacles_m=obstacles_m, grow_rate_mps=grow_rate_mps)
    obj_pose = si.set_objects_si(params, objs_m, masses_kg)   # sets obj_hw/hh/mass, returns pose (internal)
    n_obj = len(objs_m)
    # obj_damp / velocity caps now come from VineParams defaults; override params.* here to tune.

    B = 1
    ih = torch.zeros(B, 1); ix = torch.zeros(B, 1); iy = torch.zeros(B, 1)
    state, dstate = create_state_batched(B, 40); bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih)
    obj_dstate = torch.zeros(B, n_obj, 3)
    R_mm = MM(float(params.radius))
    # Walls are oriented boxes too (params.obstacle_*); drawn as OBB polygons like the movable
    # objects. The far "no obstacles" dummy is offscreen and simply clipped by the plot limits.
    walls_obb = [([float(v) for v in params.obstacle_pose[k]],
                  float(params.obstacle_hw[k]), float(params.obstacle_hh[k]))
                 for k in range(params.obstacle_pose.shape[0])]

    fdir = os.path.join(HERE, "_frames"); os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    saved = []

    def draw(i):
        fig, ax = plt.subplots(figsize=(8, 3.0))
        for (pose, hw, hh) in walls_obb:
            corners = _obb_corners_mm(pose[0], pose[1], pose[2], hw, hh)
            ax.add_patch(Polygon(corners, closed=True, facecolor=(1, .89, .71), edgecolor="k", zorder=1))
        for k in range(n_obj):
            cx, cy, th = [float(v) for v in obj_pose[0, k]]
            corners = _obb_corners_mm(cx, cy, th, float(params.obj_hw[k]), float(params.obj_hh[k]))
            ax.add_patch(Polygon(corners, closed=True, facecolor=(.55, .8, .95), edgecolor="b", lw=2, zorder=3))
        n = int(bodies[0])
        xs = [MM(float(state[0, 3 * j])) for j in range(n)]; ys = [MM(float(state[0, 3 * j + 1])) for j in range(n)]
        ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), R_mm, facecolor=(.3, .5, .95, .4), edgecolor=(.1, .2, .6), zorder=4))
        ax.plot([0], [0], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim_mm); ax.set_ylim(*ylim_mm); ax.set_aspect("equal")
        ax.set_xlabel("mm"); ax.set_title(f"{name}  frame {i}")
        fig.savefig(os.path.join(fdir, f"f{i:04d}.png"), dpi=70, bbox_inches="tight"); plt.close(fig)

    i = 0
    for i in range(frames):
        if i % 2 == 0:
            draw(i); saved.append(i)
        try:
            state, dstate, bodies, obj_pose, obj_dstate = step(
                params, ih, ix, iy, state, dstate, bodies, obj_pose, obj_dstate)
        except Exception as e:
            print(f"  {name}: stopped at frame {i} ({type(e).__name__})"); break
        if not torch.isfinite(state).all():
            break
    draw(i); saved.append(i)

    imgs = [Image.open(os.path.join(fdir, f"f{s:04d}.png")) for s in saved]
    out = os.path.join(HERE, gifname)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=110, loop=0)
    print("wrote", out, f"({len(imgs)} frames)")


def render_vine_spam(name, gifname, p, l0, scale, frames=45, xlim_mm=(-15, 160), ylim_mm=(-15, 95),
                     grow_rate_mps=0.3):
    """A vine with no obstacles that CURVES under sPAM actuation (the ActVine design model, Gao et al.
    2025). `l0` sign sets the curl direction; `scale` is the actuation-moment SI->non-dim knob
    (spam_moment_scale, pending sysid) -- its ratio to spam_restore_scale sets the equilibrium curl."""
    solver.cvxpylayer = None
    mb = 40
    params = si.vine_params_si(max_bodies=mb, grow_rate_mps=grow_rate_mps)
    params.stiffness_mode = 'spam'                        # auto-loads params.spam_moment_fn (Eq. 1d actuation)
    params.bend_length_scale = torch.tensor(0.018)        # segment length in METERS (sPAM input)
    params.spam_moment_scale = scale
    params.spam_p = torch.full((mb,), float(p))           # actuator pressure (Pa)
    params.spam_l0 = torch.full((mb,), float(l0))         # rest length (m); sign = curl direction

    B = 1
    ih = torch.zeros(B, 1); ix = torch.zeros(B, 1); iy = torch.zeros(B, 1)
    state, dstate = create_state_batched(B, mb); bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih)
    R_mm = MM(float(params.radius))

    fdir = os.path.join(HERE, "_frames"); os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    saved = []

    def draw(i):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        n = int(bodies[0])
        xs = [MM(float(state[0, 3 * j])) for j in range(n)]; ys = [MM(float(state[0, 3 * j + 1])) for j in range(n)]
        ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), R_mm, facecolor=(.9, .6, .3, .45), edgecolor=(.6, .35, .1), zorder=4))
        ax.plot([0], [0], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim_mm); ax.set_ylim(*ylim_mm); ax.set_aspect("equal")
        ax.set_xlabel("mm"); ax.set_title(f"{name}  frame {i}")
        fig.savefig(os.path.join(fdir, f"f{i:04d}.png"), dpi=70, bbox_inches="tight"); plt.close(fig)

    i = 0
    for i in range(frames):
        if i % 2 == 0:
            draw(i); saved.append(i)
        try:
            # vine-only: no objects, so step() runs the same QP degraded to zero object DOFs
            state, dstate, bodies, _, _ = step(params, ih, ix, iy, state, dstate, bodies)
        except Exception as e:
            print(f"  {name}: stopped at frame {i} ({type(e).__name__})"); break
        if not torch.isfinite(state).all():
            break
    draw(i); saved.append(i)

    imgs = [Image.open(os.path.join(fdir, f"f{s:04d}.png")) for s in saved]
    out = os.path.join(HERE, gifname)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=110, loop=0)
    print("wrote", out, f"({len(imgs)} frames)")


if __name__ == "__main__":
    # All geometry in METERS, masses in kg, growth in m/s. Plot windows in mm.
    render_vine_spam("vine curves (sPAM actuation)", "vine_spam_curve.gif",
                     p=8000.0, l0=-0.04, scale=22000.0, frames=45)
    render("vine pushes box", "vine_pushes_box.gif",
           objs_m=[[0.045, -0.014, 0.073, 0.014]], masses_kg=[0.05], frames=46, xlim_mm=(-15, 230))
    render("object stops at wall", "obj_wall.gif",
           obstacles_m=[[0.200, -0.045, 0.220, 0.045]],
           objs_m=[[0.045, -0.014, 0.073, 0.014]], masses_kg=[0.05], frames=34, xlim_mm=(-15, 245))
    render("object pushes object", "obj_obj.gif",
           objs_m=[[0.045, -0.014, 0.073, 0.014], [0.120, -0.014, 0.148, 0.014]],
           masses_kg=[0.05, 0.05], frames=50, xlim_mm=(-15, 290))
    render("vine spins a box (off-center push)", "obj_spin.gif",
           objs_m=[[0.041, -0.002, 0.069, 0.026]], masses_kg=[0.05],
           frames=40, xlim_mm=(-15, 150), ylim_mm=(-30, 60))
    print("done")
