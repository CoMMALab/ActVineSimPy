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

Run from the repo root (vine env):
    python xnew_rrt/fit_to_master.py
    python xnew_rrt/fit_to_master.py --grow-force 0 --generations 8
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


def load_reference(path=REF):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    scenarios = {}
    for name, m in meta["scenarios"].items():
        frames = [data[f"{name}/frame{k}"] for k in range(m["num_frames"])]
        scenarios[name] = dict(m, frames=frames)
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


def run_scenario(sc, cfg, stiffness, damping, grow_force_n, seg_inertia):
    '''
    Grow one scenario for every (stiffness, damping) candidate in one batch. Returns (B,) losses and,
    per candidate, its centerline at the last compared length (None if it never got there).
    '''
    B = len(stiffness)
    walls_m = [[v * 1e-3 for v in w] for w in sc["walls_mm"]]
    x0, y0, h0 = sc["start"]

    solver.cvxpylayer = None
    params = si.vine_params_si(max_bodies=cfg.max_bodies, obstacles_m=walls_m or None,
                               grow_rate_mps=cfg.grow_rate_mps, radius_m=cfg.vine_radius_m,
                               seg_len_m=cfg.seg_len_m, seg_mass_kg=cfg.seg_mass_kg,
                               seg_inertia_kgm2=seg_inertia, grow_force_n=grow_force_n)
    params.stiffness_mode = "curvature"

    base = np.array([si.len_to_nd(x0 * 1e-3), si.len_to_nd(y0 * 1e-3), h0])
    ih, ix, iy = (torch.full((B, 1), float(v)) for v in (base[2], base[0], base[1]))
    state, dstate = create_state_batched(B, cfg.max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, ih, ix, iy)

    bend_target = torch.zeros(B, cfg.max_bodies)
    per_batch = {"stiffness_val": torch.tensor(stiffness, dtype=torch.float32),
                 "damping": torch.tensor(damping, dtype=torch.float32),
                 "bend_target": bend_target}

    # control schedule in total vine length (master starts with one 68 mm body)
    bounds, length = [], sc["initial_body_mm"]
    for curvature, seg_mm in sc["schedule"]:
        length += seg_mm
        bounds.append((length, curvature))
    phase = [-1] * B

    ref_frames = [f for f in sc["frames"] if arc_length(f)[-1] >= MIN_COMPARE_MM]
    ref_lengths = [arc_length(f)[-1] for f in ref_frames]
    errors = [[] for _ in range(B)]
    jitter_sum, jitter_n = np.zeros(B), np.zeros(B)
    last_poly = [None] * B
    next_frame = [0] * B
    alive = np.ones(B, dtype=bool)

    max_steps = int(1.6 * (bounds[-1][0] - 2 * cfg.seg_len_m * 1e3) / (cfg.grow_rate_mps * 1e3 * float(params.dt) * si.T0))
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
                next_frame[k] += 1
                if next_frame[k] == len(ref_frames):
                    last_poly[k] = polys[k]
        if all(next_frame[k] >= len(ref_frames) or not alive[k] for k in range(B)):
            break
        if int(bodies.max()) >= cfg.max_bodies - 1:
            break
        state, dstate, bodies, _, _, ok = step(params, ih, ix, iy, state, dstate, bodies,
                                               return_ok=True, per_batch=per_batch)
        bad = ~(ok & torch.isfinite(state).all(dim=1) & (state.abs().amax(dim=1) < 1e5)).numpy()
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
            losses.append((sum(errors[k]) + missing * FAIL_LOSS) / len(ref_frames)
                          + JITTER_MM_PER_RADPS * jitter)
    return np.array(losses), last_poly


def evaluate(scenarios, cfg, stiffness, damping, grow_force_n, seg_inertia):
    per = {name: run_scenario(sc, cfg, stiffness, damping, grow_force_n, seg_inertia)[0]
           for name, sc in scenarios.items()}
    return np.mean(np.stack(list(per.values())), axis=0), per


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=REF)
    parser.add_argument("--grow-force", type=float, default=SimConfig.grow_force_n,
                        help="growth stall force [N]; 0 = hard growth constraint")
    parser.add_argument("--seg-mass", type=float, default=SimConfig.seg_mass_kg)
    parser.add_argument("--seg-inertia", type=float, default=1.0e-5)
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
    cfg = SimConfig(seg_mass_kg=args.seg_mass)
    grow_force = args.grow_force if args.grow_force > 0 else None
    rng = np.random.default_rng(args.seed)

    # cross-entropy method over (log stiffness_val, log damping); start around the current defaults
    default_damping = cfg.ang_damp / 1.0e-4   # si ang_damp -> VineParams.damping
    mean = np.log([cfg.joint_stiffness, default_damping])
    std = np.array([1.5, 1.5])
    best = (float("inf"), None, None)
    history = []
    t0 = time.time()
    for gen in range(args.generations):
        samples = rng.normal(mean, std, size=(args.population, 2))
        if gen == 0:
            samples[0] = mean   # always score the defaults
        k, c = np.exp(samples[:, 0]), np.exp(samples[:, 1])
        loss, per = evaluate(scenarios, cfg, k, c, grow_force, args.seg_inertia)
        order = np.argsort(loss)
        elite = samples[order[:max(2, args.population // 4)]]
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.1)
        i = order[0]
        if loss[i] < best[0]:
            best = (float(loss[i]), (float(k[i]), float(c[i])), {n: float(v[i]) for n, v in per.items()})
        history.append({"generation": gen, "stiffness": k.tolist(), "damping": c.tolist(), "loss": loss.tolist()})
        if gen == 0:
            print(f"defaults (stiffness {cfg.joint_stiffness:g}, damping {default_damping:g}): {loss[0]:.1f} mm")
        print(f"gen {gen}: best {loss[i]:.1f} mm at stiffness {k[i]:.3g}, damping {c[i]:.3g}  "
              f"(all-time best {best[0]:.1f} mm, {time.time() - t0:.0f} s)", flush=True)

    loss, (k, c), per = best
    print(f"\nbest: stiffness_val {k:.4g}, damping {c:.4g} (si ang_damp {c * 1e-4:.3g}) -> {loss:.1f} mm mean")
    for name, v in per.items():
        print(f"  {name:16s} {v:7.1f} mm")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fit.json"), "w") as f:
        json.dump({"grow_force_n": grow_force, "seg_mass_kg": args.seg_mass, "seg_inertia_kgm2": args.seg_inertia,
                   "stiffness_val": k, "damping": c, "ang_damp_si": c * 1e-4, "loss_mm": loss,
                   "per_scenario_mm": per, "history": history}, f, indent=1)
    draw_comparison(scenarios, cfg, k, c, grow_force, args.seg_inertia, os.path.join(args.out, "compare.png"))
    print("wrote", os.path.join(args.out, "fit.json"), "and compare.png")


def draw_comparison(scenarios, cfg, stiffness, damping, grow_force_n, seg_inertia, path):
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
        final = run_scenario(sc, cfg, [stiffness], [damping], grow_force_n, seg_inertia)[1][0]
        for w in sc["walls_mm"]:
            ax.add_patch(Rectangle((w[0], w[1]), w[2] - w[0], w[3] - w[1], color="0.6"))
        ref = sc["frames"][-1]
        ax.plot(ref[:, 0], ref[:, 1], "k-", lw=2, label="master")
        if final is not None:
            ax.plot(final[:, 0], final[:, 1], "-", color="tab:blue", lw=2, label="dvsim")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-250, 1050); ax.set_ylim(-650, 650)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"final shapes: master (black) vs dvsim (stiffness {stiffness:.3g}, damping {damping:.3g})")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
