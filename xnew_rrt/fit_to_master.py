'''
Fit dvsim's vine stiffness and damping so its bending matches master's quasi-static PBD vine.

The reference is xnew_rrt/data/master_ref.npz (made by master_reference.py): master's vine grown
in free space and along walls under master's own sPAM control. Every scenario is grown again in
dvsim ('curvature' stiffness mode, same dynamics as 'spam') with the same control schedule, and
the loss is the mean distance [mm] between the two centerlines, compared at equal vine length,
plus a penalty on joint oscillation (see JITTER_MM_PER_RADPS).

The fitted values are VineParams.stiffness_val and VineParams.damping (all candidates of one
generation run as one batch, see dynamic_vine.step's per_batch). Segment mass/inertia and the
growth force enter the QP itself, so they are fixed per run (--seg-mass, --grow-force). The search
is a cross-entropy method over (log stiffness, log damping).

The same fit runs on real trials: video/to_reference.py writes their tracked centerlines in this
reference format. A scenario from video also carries its measured growth rate (used instead of
SimConfig's) and its movable boxes, whose tracked centers are scored too (BOX_WEIGHT; the boxes'
table friction is --obj-friction, fixed per run like the growth force).

Run from the repo root (actvine env):
    python xnew_rrt/fit_to_master.py
    python xnew_rrt/fit_to_master.py --grow-force 0 --generations 8
    python xnew_rrt/fit_to_master.py --ref xnew_rrt/data/video_ref.npz --out out/fit_video
Writes <out>/fit.json and <out>/compare.png (master vs dvsim with the fitted values).
'''

import argparse, json, math, os, time, warnings

import numpy as np
import torch

from scene import Scene, SimConfig  # (puts the repo root on sys.path)
from dvsim import si
import dvsim.solver as solver
from dvsim.dynamic_vine import step
from dvsim.vine import create_state_batched, init_state_batched

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "master_ref.npz")
MIN_COMPARE_MM = 150.0   # skip the first frames (both vines are still nearly straight stubs)
FAIL_LOSS = 1000.0       # mm, for a candidate whose QP failed or blew up
# Penalty [mm per rad/s] on the mean relative angular velocity of the settled joints (all but the
# TIP_JOINTS nearest the tip). Without it the search runs to the explicit-stability edge, where
# the joints oscillate but the averaged shape still matches.
JITTER_MM_PER_RADPS = 20.0
TIP_JOINTS = 4
BOX_WEIGHT = 1.0   # weight of the mean box-center error [mm] (scenarios with tracked boxes)


def load_reference(path=REF):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    scenarios = {}
    for name, m in meta["scenarios"].items():
        frames = [data[f"{name}/frame{k}"] for k in range(m["num_frames"])]
        boxes = [data[f"{name}/boxes{k}"] for k in range(m["num_frames"])] if m.get("objects") else None
        scenarios[name] = dict(m, frames=frames, boxes=boxes)
    return scenarios


def arc_length(poly):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def resample(poly, s):
    '''Points at arc lengths s along a polyline (clamped to its ends).'''
    a = arc_length(poly)
    return np.stack([np.interp(s, a, poly[:, 0]), np.interp(s, a, poly[:, 1])], axis=1)


def shape_error(ref, poly, n=40):
    '''Mean distance [mm] between two centerlines, both resampled at the same arc lengths.'''
    s = np.linspace(0.0, min(arc_length(ref)[-1], arc_length(poly)[-1]), n)
    return float(np.linalg.norm(resample(ref, s) - resample(poly, s), axis=1).mean())


def dvsim_polylines(params, state, bodies, base):
    '''(B,) list of centerlines [mm]: base, every body center, the tip end.'''
    st = state.detach().numpy()
    hl = float(params.half_len)
    out = []
    for k in range(st.shape[0]):
        n = int(bodies[k])
        xyt = st[k, :3 * n].reshape(n, 3)
        tip = xyt[-1, :2] + hl * np.array([math.cos(xyt[-1, 2]), math.sin(xyt[-1, 2])])
        out.append(np.vstack([base[None, :2], xyt[:, :2], tip[None]]))
    return out


def run_scenario(sc, cfg, stiffness, damping, grow_force_n, seg_inertia, obj_friction=None):
    '''
    Grow one scenario for every (stiffness, damping) candidate in one batch. Returns (B,) losses and,
    per candidate, its centerline at the last compared length (None if it never got there).
    '''
    B = len(stiffness)
    walls_m = [[v * 1e-3 for v in w] for w in sc["walls_mm"]]
    x0, y0, h0 = sc["start"]

    grow_rate_mps = sc.get("grow_rate_mps") or cfg.grow_rate_mps

    solver.cvxpylayer = None
    params = si.vine_params_si(max_bodies=cfg.max_bodies, obstacles_m=walls_m or None,
                               grow_rate_mps=grow_rate_mps, radius_m=cfg.vine_radius_m,
                               seg_len_m=cfg.seg_len_m, seg_mass_kg=cfg.seg_mass_kg,
                               seg_inertia_kgm2=seg_inertia, grow_force_n=grow_force_n)
    params.stiffness_mode = "curvature"

    base = np.array([si.len_to_nd(x0 * 1e-3), si.len_to_nd(y0 * 1e-3), h0])
    ih, ix, iy = (torch.full((B, 1), float(v)) for v in (base[2], base[0], base[1]))
    state, dstate = create_state_batched(B, cfg.max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih, ix, iy)

    objects = sc.get("objects") or []
    obj_pose = obj_dstate = None
    friction_per_batch = obj_friction is not None and np.ndim(obj_friction) > 0
    if objects:
        if obj_friction is not None and not friction_per_batch:
            params.obj_friction = obj_friction
        aabbs = [[(o["pose"][0] - o["size_mm"][0] / 2) * 1e-3, (o["pose"][1] - o["size_mm"][1] / 2) * 1e-3,
                  (o["pose"][0] + o["size_mm"][0] / 2) * 1e-3, (o["pose"][1] + o["size_mm"][1] / 2) * 1e-3]
                 for o in objects]
        pose0 = si.set_objects_si(params, aabbs, [o["mass_kg"] for o in objects])
        pose0[0, :, 2] = torch.tensor([float(o["pose"][2]) for o in objects])
        obj_pose = pose0.repeat(B, 1, 1)
        obj_dstate = torch.zeros_like(obj_pose)
    box_errors = [[] for _ in range(B)]

    bend_target = torch.zeros(B, cfg.max_bodies)
    per_batch = {"stiffness_val": torch.tensor(stiffness, dtype=torch.float32),
                 "damping": torch.tensor(damping, dtype=torch.float32),
                 "bend_target": bend_target}
    if objects and friction_per_batch:
        per_batch["obj_friction"] = torch.tensor(obj_friction, dtype=torch.float32)

    # control schedule in total vine length (master starts with one 68 mm body)
    bounds, length = [], sc["initial_body_mm"]
    for curvature, seg_mm in sc["schedule"]:
        length += seg_mm
        bounds.append((length, curvature))
    phase = [-1] * B

    compared = [i for i, f in enumerate(sc["frames"]) if arc_length(f)[-1] >= MIN_COMPARE_MM]
    ref_frames = [sc["frames"][i] for i in compared]
    ref_box_frames = [sc["boxes"][i] for i in compared] if objects else None
    ref_lengths = [arc_length(f)[-1] for f in ref_frames]
    errors = [[] for _ in range(B)]
    jitter_sum, jitter_n = np.zeros(B), np.zeros(B)
    last_poly = [None] * B
    last_boxes = [None] * B
    next_frame = [0] * B
    alive = np.ones(B, dtype=bool)

    max_steps = int(1.6 * (bounds[-1][0] - 2 * cfg.seg_len_m * 1e3) / (grow_rate_mps * 1e3 * float(params.dt) * si.T0))
    for _ in range(max_steps):
        polys = dvsim_polylines(params, state, bodies, base)
        for k in range(B):
            if not alive[k]:
                continue
            L = arc_length(polys[k])[-1]
            # switch control when this vine has grown past the next boundary
            while phase[k] + 1 < len(bounds) and (phase[k] < 0 or L >= bounds[phase[k]][0]):
                phase[k] += 1
                n = int(bodies[k])
                bend_target[k, n:] = bounds[phase[k]][1] * cfg.seg_len_m
            while next_frame[k] < len(ref_frames) and L >= ref_lengths[next_frame[k]]:
                errors[k].append(shape_error(ref_frames[next_frame[k]], polys[k]))
                if objects:
                    ref_boxes = ref_box_frames[next_frame[k]]
                    sim_boxes = si.len_to_mm(obj_pose[k, :, :2].detach().numpy())
                    box_errors[k].append(float(np.linalg.norm(sim_boxes - ref_boxes[:, :2], axis=1).mean()))
                next_frame[k] += 1
                if next_frame[k] == len(ref_frames):
                    last_poly[k] = polys[k]
                    if objects:
                        last_boxes[k] = np.column_stack([si.len_to_mm(obj_pose[k, :, :2].detach().numpy()),
                                                         obj_pose[k, :, 2].detach().numpy()])
        if all(next_frame[k] >= len(ref_frames) or not alive[k] for k in range(B)):
            break
        if int(bodies.max()) >= cfg.max_bodies - 1:
            break
        state, dstate, bodies, obj_pose, obj_dstate, ok = step(params, ih, ix, iy, state, dstate, bodies,
                                                               obj_pose, obj_dstate,
                                                               return_ok=True, per_batch=per_batch)
        bad = ~(ok & torch.isfinite(state).all(dim=1) & (state.abs().amax(dim=1) < 1e5)).numpy()
        if objects:
            bad |= ~torch.isfinite(obj_pose).flatten(1).all(dim=1).numpy()
        alive &= ~bad
        omega = dstate[:, 2::3].detach().numpy()
        for k in range(B):
            n = int(bodies[k]) - TIP_JOINTS
            if alive[k] and n > 2:
                jitter_sum[k] += np.abs(np.diff(omega[k, :n])).mean()
                jitter_n[k] += 1

    losses = []
    for k in range(B):
        if not alive[k] or not errors[k]:
            losses.append(FAIL_LOSS)
        else:
            # frames the vine never reached (stalled) count as failures
            missing = len(ref_frames) - len(errors[k])
            jitter = jitter_sum[k] / max(jitter_n[k], 1)
            box = BOX_WEIGHT * np.mean(box_errors[k]) if box_errors[k] else 0.0
            losses.append((sum(errors[k]) + missing * FAIL_LOSS) / len(ref_frames)
                          + JITTER_MM_PER_RADPS * jitter + box)
    return np.array(losses), last_poly, last_boxes


def evaluate(scenarios, cfg, stiffness, damping, grow_force_n, seg_inertia, obj_friction=None):
    per = {name: run_scenario(sc, cfg, stiffness, damping, grow_force_n, seg_inertia, obj_friction)[0]
           for name, sc in scenarios.items()}
    return np.mean(np.stack(list(per.values())), axis=0), per


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=REF)
    parser.add_argument("--grow-force", type=float, default=SimConfig.grow_force_n,
                        help="growth stall force [N]; 0 = hard growth constraint")
    parser.add_argument("--seg-mass", type=float, default=SimConfig.seg_mass_kg)
    parser.add_argument("--seg-inertia", type=float, default=1.0e-5)
    parser.add_argument("--obj-friction", type=float, default=None,
                        help="box-table friction coefficient for scenarios with boxes; without it, it is "
                             "fitted along with stiffness and damping")
    parser.add_argument("--vine-radius", type=float, default=SimConfig.vine_radius_m, help="[m]")
    parser.add_argument("--max-bodies", type=int, default=SimConfig.max_bodies,
                        help="segment slots: the longest vine is about 18 mm times this")
    parser.add_argument("--start", type=float, nargs="+", default=None,
                        help="where the search starts: stiffness damping [box friction] (default SimConfig's, 0.5)")
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario names")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/fit_master")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    scenarios = load_reference(args.ref)
    if args.scenarios:
        scenarios = {k: v for k, v in scenarios.items() if k in args.scenarios}
    cfg = SimConfig(seg_mass_kg=args.seg_mass, vine_radius_m=args.vine_radius, max_bodies=args.max_bodies)
    grow_force = args.grow_force if args.grow_force > 0 else None
    rng = np.random.default_rng(args.seed)
    fit_friction = args.obj_friction is None and any(sc.get("objects") for sc in scenarios.values())

    # cross-entropy method over (log stiffness_val, log damping[, log box friction]); start around
    # the current defaults
    default_damping = cfg.ang_damp / 1.0e-4   # si ang_damp -> VineParams.damping
    start = list(args.start) if args.start else [cfg.joint_stiffness, default_damping, 0.5]
    mean = np.log(start[:2] + ((start[2:3] or [0.5]) if fit_friction else []))
    default_damping = start[1]
    std = np.array([1.5, 1.5] + ([0.7] if fit_friction else []))
    best = (float("inf"), None, None)
    history = []
    t0 = time.time()
    for gen in range(args.generations):
        samples = rng.normal(mean, std, size=(args.population, len(mean)))
        if gen == 0:
            samples[0] = mean   # always score the defaults
        k, c = np.exp(samples[:, 0]), np.exp(samples[:, 1])
        mu = np.exp(samples[:, 2]) if fit_friction else args.obj_friction
        loss, per = evaluate(scenarios, cfg, k, c, grow_force, args.seg_inertia, mu)
        order = np.argsort(loss)
        elite = samples[order[:max(2, args.population // 4)]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.1)
        i = order[0]
        mu_i = float(mu[i]) if fit_friction else args.obj_friction
        if loss[i] < best[0]:
            best = (float(loss[i]), (float(k[i]), float(c[i]), mu_i), {n: float(v[i]) for n, v in per.items()})
        history.append({"generation": gen, "stiffness": k.tolist(), "damping": c.tolist(), "loss": loss.tolist(),
                        "obj_friction": mu.tolist() if fit_friction else args.obj_friction})
        if gen == 0:
            print(f"start (stiffness {start[0]:g}, damping {start[1]:g}"
                  + (f", box friction {np.exp(samples[0, 2]):g}" if fit_friction else "") + f"): {loss[0]:.1f} mm")
        print(f"gen {gen}: best {loss[i]:.1f} mm at stiffness {k[i]:.3g}, damping {c[i]:.3g}"
              + (f", box friction {mu_i:.3g}" if fit_friction else "")
              + f"  (all-time best {best[0]:.1f} mm, {time.time() - t0:.0f} s)", flush=True)

    loss, (k, c, mu_best), per = best
    print(f"\nbest: stiffness_val {k:.4g}, damping {c:.4g} (si ang_damp {c * 1e-4:.3g})"
          + (f", box friction {mu_best:.3g}" if mu_best is not None else "") + f" -> {loss:.1f} mm mean")
    for name, v in per.items():
        print(f"  {name:16s} {v:7.1f} mm")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fit.json"), "w") as f:
        json.dump({"grow_force_n": grow_force, "seg_mass_kg": args.seg_mass, "seg_inertia_kgm2": args.seg_inertia,
                   "vine_radius_m": args.vine_radius, "max_bodies": args.max_bodies, "obj_friction": mu_best, "stiffness_val": k, "damping": c, "ang_damp_si": c * 1e-4, "loss_mm": loss,
                   "per_scenario_mm": per, "history": history}, f, indent=1)
    draw_comparison(scenarios, cfg, k, c, grow_force, args.seg_inertia, os.path.join(args.out, "compare.png"),
                    mu_best)
    print("wrote", os.path.join(args.out, "fit.json"), "and compare.png")


def draw_comparison(scenarios, cfg, stiffness, damping, grow_force_n, seg_inertia, path, obj_friction=None):
    '''Final shapes: master (black) vs dvsim with the given values (blue), one panel per scenario.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = list(scenarios)
    cols = min(5, len(names))
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        sc = scenarios[name]
        _, finals, final_boxes = run_scenario(sc, cfg, [stiffness], [damping], grow_force_n, seg_inertia, obj_friction)
        final, final_box = finals[0], final_boxes[0]
        for w in sc["walls_mm"]:
            ax.add_patch(Rectangle((w[0], w[1]), w[2] - w[0], w[3] - w[1], color="0.6"))
        ref = sc["frames"][-1]
        ax.plot(ref[:, 0], ref[:, 1], "k-", lw=2, label="reference")
        if final is not None:
            ax.plot(final[:, 0], final[:, 1], "-", color="tab:blue", lw=2, label="dvsim")
        ax.set_title(name, fontsize=9)
        if sc.get("boxes"):   # tracked boxes at the first and last frame
            from matplotlib.patches import Polygon
            for o, first, last in zip(sc["objects"], sc["boxes"][0], sc["boxes"][-1]):
                for pose, alpha in ((first, 0.3), (last, 0.9)):
                    ax.add_patch(Polygon(_corners(pose, o["size_mm"]), closed=True, fill=False,
                                         ec="tab:orange", alpha=alpha))
            if final_box is not None:   # where the sim's boxes ended up
                for o, pose in zip(sc["objects"], final_box):
                    ax.add_patch(Polygon(_corners(pose, o["size_mm"]), closed=True, fill=False,
                                         ec="tab:blue", ls="--"))
        ax.set_aspect("equal")
        pts = np.vstack([ref] + ([final] if final is not None else []))
        cx, cy = (pts.min(axis=0) + pts.max(axis=0)) / 2
        half = max(650.0, 0.6 * float((pts.max(axis=0) - pts.min(axis=0)).max()))
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"final shapes: reference (black; tracked boxes orange) vs dvsim (blue; boxes dashed), "
                 f"stiffness {stiffness:.3g}, damping {damping:.3g}"
                 + (f", box friction {obj_friction:.3g}" if obj_friction is not None else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _corners(pose, size):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(size) / 2
    return local @ np.array([[c, s], [-s, c]]) + np.array(pose[:2])


if __name__ == "__main__":
    main()
