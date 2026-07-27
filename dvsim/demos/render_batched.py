"""Batched dvsim demo: fan out the vine over 64 launch angles evenly spaced in [-pi/4, pi/4],
each in its own world with a movable box, and render the paths of all vines and objects.

Run: /home/zak/micromamba/envs/vine/bin/python dvsim/demos/render_batched.py
"""
import os, sys, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
import dvsim.solver as solver
from dvsim.vine import create_state_batched, init_state_batched
from dvsim.dynamic_vine import step
from dvsim import si

MM = lambda x: si.len_to_mm(x)


def obb_corners_mm(cx, cy, th, hw, hh):
    c, s = math.cos(th), math.sin(th)
    return [(MM(cx + c*lx - s*ly), MM(cy + s*lx + c*ly)) for lx, ly in ((-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh))]


def main(B=64, frames=44, use_qpth=True):
    solver.USE_QPTH = use_qpth          # solve the batched per-step QP on the GPU via qpth
    solver.cvxpylayer = None
    params = si.vine_params_si(max_bodies=40, obstacles_m=None, grow_rate_mps=0.3)
    obj_pose = si.set_objects_si(params, [[0.085, -0.014, 0.113, 0.014]], [0.05]).repeat(B, 1, 1)
    hw, hh = float(params.obj_hw[0]), float(params.obj_hh[0])

    angles = torch.linspace(-math.pi/4, math.pi/4, B).unsqueeze(1)      # (B,1) launch headings
    ih, ix, iy = angles, torch.zeros(B, 1), torch.zeros(B, 1)
    
    state, dstate = create_state_batched(B, 40)
    bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih)
    obj_dstate = torch.zeros(B, 1, 3)

    tip_paths = [[] for _ in range(B)]        # per-vine tip trajectory
    box_paths = [[] for _ in range(B)]        # per-vine box-center trajectory
    def record():
        for i in range(B):
            n = int(bodies[i])
            tip_paths[i].append((MM(float(state[i, 3*(n-1)])), MM(float(state[i, 3*(n-1)+1]))))
            box_paths[i].append((MM(float(obj_pose[i, 0, 0])), MM(float(obj_pose[i, 0, 1]))))
    record()
    import warnings, time
    t0 = time.time()
    for _ in range(frames):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state, dstate, bodies, obj_pose, obj_dstate = step(
                params, ih, ix, iy, state, dstate, bodies, obj_pose, obj_dstate)
        if not torch.isfinite(state).all():
            break
        record()
    sim_dt = time.time() - t0

    # ---- render ----
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = cm.turbo(torch.linspace(0, 1, B).numpy())
    for i in range(B):
        col = colors[i]
        tp = tip_paths[i]; bp = box_paths[i]
        ax.plot([p[0] for p in tp], [p[1] for p in tp], "-", color=col, lw=1.2, alpha=0.9, zorder=3)
        ax.plot([p[0] for p in bp], [p[1] for p in bp], ":", color=col, lw=1.0, alpha=0.5, zorder=2)
        # final vine shape
        n = int(bodies[i])
        xs = [MM(float(state[i, 3*j])) for j in range(n)]; ys = [MM(float(state[i, 3*j+1])) for j in range(n)]
        ax.plot(xs, ys, "-", color=col, lw=2.0, alpha=0.55, zorder=4)
        # final box
        cx, cy, th = [float(v) for v in obj_pose[i, 0]]
        ax.add_patch(Polygon(obb_corners_mm(cx, cy, th, hw, hh), closed=True,
                             facecolor=(*col[:3], 0.10), edgecolor=(*col[:3], 0.5), lw=0.8, zorder=1))
    ax.plot([0], [0], "k^", ms=11, zorder=6)
    ax.set_aspect("equal"); ax.set_xlabel("mm"); ax.set_ylabel("mm")
    engine = "qpth (GPU)" if use_qpth else "cvxpylayers/Clarabel (CPU)"
    ax.set_title(f"dvsim batched: {B} vines fanned over launch angles [-45°, +45°]  —  solver: {engine}\n"
                 f"solid = vine tip paths, dotted = box paths, thick = final vine shape")
    sm = cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(-45, 45)); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="launch angle (deg)", shrink=0.7)
    out = os.path.join(HERE, "batched_fan.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, f"| B={B}, frames={len(tip_paths[0])}, bodies [{int(bodies.min())},{int(bodies.max())}]"
          f", solver={engine}, sim {sim_dt:.2f}s")


if __name__ == "__main__":
    main()
