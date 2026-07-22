import os, sys, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from PIL import Image
from functools import partial
import dvsim.solver as solver
from dvsim.vine import (VineParams, create_state_batched, init_state_batched,
                        forward_batched_part, solve)
from dvsim.dynamic_vine import dynamic_step


def _obb_corners(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(cx + c * lx - s * ly, cy + s * lx + c * ly)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def render(name, gifname, obstacles, objs, masses, frames, xlim, ylim=(-45, 45)):
    solver.cvxpylayer = None                       # reset cached QP layer (shapes differ per scene)
    mb = 40
    params = VineParams(max_bodies=mb, obstacles=[list(map(float, o)) for o in obstacles],
                        grow_rate=0.3, stiffness_mode='linear')
    n_obj = len(objs)
    # scene objects are given as AABB specs [x1,y1,x2,y2]; convert to oriented-box pose + extents
    aabb = torch.tensor(objs, dtype=torch.float32)
    obj_pose = torch.stack([(aabb[:, 0] + aabb[:, 2]) / 2,
                            (aabb[:, 1] + aabb[:, 3]) / 2,
                            torch.zeros(n_obj)], dim=1)[None]         # (1, n_obj, 3) = (cx,cy,theta)
    params.obj_hw = (aabb[:, 2] - aabb[:, 0]) / 2
    params.obj_hh = (aabb[:, 3] - aabb[:, 1]) / 2
    params.obj_mass = torch.tensor(masses)          # inertia is derived physically from mass+size
    params.obj_damp = 0.2                           # viscous friction
    params.obj_vel_cap = 600.0                      # linear velocity cap
    params.obj_ang_vel_cap = 5.0                    # angular velocity cap (rad/s)

    B = 1
    ih = torch.zeros(B, 1); ix = torch.zeros(B, 1); iy = torch.zeros(B, 1)
    state, dstate = create_state_batched(B, mb); bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih)
    obj_dstate = torch.zeros(B, n_obj, 3)
    R = float(params.radius)
    walls = [o for o in obstacles if o[0] < 1e3]

    fdir = os.path.join(HERE, "_frames"); os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    saved = []

    def draw(i):
        fig, ax = plt.subplots(figsize=(8, 3.0))
        for (x1, y1, x2, y2) in walls:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=(1, .89, .71), edgecolor="k", zorder=1))
        for k in range(n_obj):
            cx, cy, th = [float(v) for v in obj_pose[0, k]]
            corners = _obb_corners(cx, cy, th, float(params.obj_hw[k]), float(params.obj_hh[k]))
            ax.add_patch(Polygon(corners, closed=True, facecolor=(.55, .8, .95), edgecolor="b", lw=2, zorder=3))
        n = int(bodies[0])
        xs = [float(state[0, 3 * j]) for j in range(n)]; ys = [float(state[0, 3 * j + 1]) for j in range(n)]
        ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), R, facecolor=(.3, .5, .95, .4), edgecolor=(.1, .2, .6), zorder=4))
        ax.plot([0], [0], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_title(f"{name}  frame {i}")
        fig.savefig(os.path.join(fdir, f"f{i:04d}.png"), dpi=70, bbox_inches="tight"); plt.close(fig)

    i = 0
    for i in range(frames):
        if i % 2 == 0:
            draw(i); saved.append(i)
        try:
            state, dstate, bodies, obj_pose, obj_dstate = dynamic_step(
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


_MOMENT_FN = None   # lazily-loaded sPAM surrogate (shared across curve renders)


def render_vine_spam(name, gifname, p, l0, scale, frames=45, xlim=(-15, 160), ylim=(-15, 95)):
    """A vine with NO obstacles that CURVES under sPAM actuation (the ActVine design model).
    `l0` sign sets curl direction; `scale` is the sPAM-moment visualization knob (see vine.py)."""
    global _MOMENT_FN
    if _MOMENT_FN is None:
        from dvsim.spam import make_moment_fn
        _MOMENT_FN = make_moment_fn()
    solver.cvxpylayer = None
    mb = 40
    params = VineParams(max_bodies=mb, obstacles=[[1e4, -1, 1e4 + 1, 1]], grow_rate=0.3, stiffness_mode='linear')
    params.stiffness_mode = 'spam'
    params.bend_length_scale = torch.tensor(0.018)
    params.spam_moment_fn = _MOMENT_FN
    params.spam_moment_scale = scale
    params.spam_p = torch.full((mb,), float(p))       # actuator pressure (per body)
    params.spam_l0 = torch.full((mb,), float(l0))     # rest length; sign = curl direction

    B = 1
    ih = torch.zeros(B, 1); ix = torch.zeros(B, 1); iy = torch.zeros(B, 1)
    state, dstate = create_state_batched(B, mb); bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih)
    fwd = torch.func.vmap(partial(forward_batched_part, params))
    R = float(params.radius)

    fdir = os.path.join(HERE, "_frames"); os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    saved = []

    def draw(i):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        n = int(bodies[0])
        xs = [float(state[0, 3 * j]) for j in range(n)]; ys = [float(state[0, 3 * j + 1]) for j in range(n)]
        ax.plot(xs, ys, "-", color=(.15, .3, .8), lw=2, zorder=5)
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), R, facecolor=(.9, .6, .3, .45), edgecolor=(.6, .35, .1), zorder=4))
        ax.plot([0], [0], "g^", ms=9, zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_title(f"{name}  frame {i}")
        fig.savefig(os.path.join(fdir, f"f{i:04d}.png"), dpi=70, bbox_inches="tight"); plt.close(fig)

    i = 0
    for i in range(frames):
        if i % 2 == 0:
            draw(i); saved.append(i)
        try:
            bodies, forces, growth, sdf_now, dev, L, J, gws, gwd = fwd(ih, ix, iy, state, dstate, bodies)
            nds = solve(params, dstate, forces, growth, sdf_now, dev, L, J, gws, gwd).detach().float()
        except Exception as e:
            print(f"  {name}: stopped at frame {i} ({type(e).__name__})"); break
        state = state + nds * params.dt; dstate = nds
        if not torch.isfinite(state).all():
            break
    draw(i); saved.append(i)

    imgs = [Image.open(os.path.join(fdir, f"f{s:04d}.png")) for s in saved]
    out = os.path.join(HERE, gifname)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=110, loop=0)
    print("wrote", out, f"({len(imgs)} frames)")


if __name__ == "__main__":
    render_vine_spam("vine curves (sPAM actuation)", "vine_spam_curve.gif",
                     p=8000.0, l0=-0.04, scale=20.0, frames=45)
    render("vine pushes box", "vine_pushes_box.gif",
           obstacles=[[1e4, -1, 1e4 + 1, 1]], objs=[[45, -14, 73, 14]], masses=[0.05],
           frames=46, xlim=(-15, 230))
    render("object stops at wall", "obj_wall.gif",
           obstacles=[[200, -45, 220, 45]], objs=[[45, -14, 73, 14]], masses=[0.05],
           frames=34, xlim=(-15, 245))
    render("object pushes object", "obj_obj.gif",
           obstacles=[[1e4, -1, 1e4 + 1, 1]], objs=[[45, -14, 73, 14], [120, -14, 148, 14]],
           masses=[0.05, 0.05], frames=50, xlim=(-15, 290))
    render("vine spins a box (off-center push)", "obj_spin.gif",
           obstacles=[[1e4, -1, 1e4 + 1, 1]], objs=[[41, -2, 69, 26]], masses=[0.05],
           frames=40, xlim=(-15, 150), ylim=(-30, 60))
    print("done")
