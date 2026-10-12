'''
Sweep the vine's dynamic parameters and score how well a free-growing vine follows a commanded
curvature.

The parameters (segment inertia and mass, joint stiffness, joint damping) are not measured, and
they are codependent: many combinations behave alike, and many oscillate or blow up. This tries
a grid of them. For every combination it grows a vine in free space at a few commanded
curvatures ('curvature' stiffness mode, which has the same dynamics as 'spam') and reports

    achieved / commanded   mean joint angle over the commanded one (1 = follows the command)
    spread                 joint-to-joint variation along the vine, relative to the command
    jitter                 variation of each joint over the last 20 steps, relative to the command

All stiffness / damping values of one (inertia, mass) pair run as one batch.

Run from the repo root:
    python xnew_rrt/sweep_vine_params.py
    python xnew_rrt/sweep_vine_params.py --grow-rate 0.1 --out out/sweep_slow
Writes <out>/results.json and prints the ranking.
'''

import argparse, itertools, json, os, warnings

import torch

from scene import SimConfig  # (puts the repo root on sys.path)
from dvsim import si
import dvsim.solver as solver
from dvsim.dynamic_vine import step
from dvsim.vine import create_state_batched, init_state_batched

INERTIAS = [1e-5, 3e-5, 1e-4, 3e-4]   # segment inertia [kg m^2]
MASSES = [0.02, 0.005]                # segment mass [kg]
STIFFNESSES = [0.1, 0.3, 1.0]         # VineParams.stiffness_val (internal units)
DAMPINGS = [0.5, 2.0, 8.0, 32.0]      # VineParams.damping (internal units)
CURVATURES = [1.0, 3.33]              # commanded [1/m]
SEG_LEN_M = 0.018
MAX_BODIES = 45
ANG_DAMP_PER_UNIT = 1.0e-4            # dvsim.si ang_damp that gives VineParams.damping = 1


def run_batch(inertia, mass, grow_rate):
    '''One batched free-growth run over STIFFNESSES x DAMPINGS x CURVATURES.'''
    grid = list(itertools.product(STIFFNESSES, DAMPINGS, CURVATURES))
    B = len(grid)

    solver.cvxpylayer = None
    params = si.vine_params_si(max_bodies=MAX_BODIES, grow_rate_mps=grow_rate, seg_len_m=SEG_LEN_M,
                               seg_inertia_kgm2=inertia, seg_mass_kg=mass)
    params.stiffness_mode = 'curvature'
    per_batch = {'stiffness_val': torch.tensor([g[0] for g in grid]),
                 'damping': torch.tensor([g[1] for g in grid]),
                 'bend_target': torch.tensor([[g[2] * SEG_LEN_M] * MAX_BODIES for g in grid])}

    zeros = torch.zeros(B, 1)
    state, dstate = create_state_batched(B, MAX_BODIES)
    bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, zeros, zeros, zeros)

    num_steps = int(0.62 / grow_rate / float(params.dt))  # grow to ~0.62 m
    alive = torch.ones(B, dtype=torch.bool)
    last_states = []
    for i in range(num_steps):
        state, dstate, bodies, _, _, ok = step(params, zeros, zeros, zeros, state, dstate, bodies,
                                               return_ok=True, per_batch=per_batch)
        alive &= ok & torch.isfinite(state).all(dim=1) & (state.abs().amax(dim=1) < 1e5)
        if i >= num_steps - 20:
            last_states.append(state.clone())

    history = torch.stack(last_states)  # (20, B, 3 * max_bodies)
    rows = []
    for k, (stiffness, damping, curvature) in enumerate(grid):
        n = int(bodies[k])
        theta = history[:, k, 2::3][:, :n]
        joint_angles = (theta[:, 1:] - theta[:, :-1])[:, 2:-3]  # skip the base and the sliding tip
        commanded = curvature * SEG_LEN_M
        rows.append({"inertia": inertia, "mass": mass, "stiffness": stiffness, "damping": damping,
                     "curvature": curvature, "bodies": n,
                     "ok": bool(alive[k]) and n > 0.8 * 0.62 / SEG_LEN_M,
                     "ratio": float(joint_angles.mean()) / commanded,
                     "spread": float(joint_angles.mean(dim=0).std()) / commanded,
                     "jitter": float(joint_angles.std(dim=0).mean()) / commanded})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grow-rate", type=float, default=SimConfig.grow_rate_mps, help="growth speed [m/s]")
    parser.add_argument("--out", default="out/sweep")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    rows = []
    for inertia, mass in itertools.product(INERTIAS, MASSES):
        rows += run_batch(inertia, mass, args.grow_rate)
        print(f"\r inertia {inertia:g} kg m^2, mass {mass:g} kg done ({len(rows)} runs)", end="", flush=True)
    print()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"grow_rate_mps": args.grow_rate, "runs": rows}, f, indent=1)

    # one score per parameter combination: the worst over the commanded curvatures
    combos = {}
    for r in rows:
        combos.setdefault((r["inertia"], r["mass"], r["stiffness"], r["damping"]), []).append(r)
    ranked = []
    for key, runs in combos.items():
        runs.sort(key=lambda r: r["curvature"])
        usable = all(r["ok"] for r in runs)
        score = (max(abs(r["ratio"] - 1) for r in runs) + max(r["spread"] for r in runs)
                 + max(r["jitter"] for r in runs)) if usable else float("inf")
        ranked.append((score, key, runs))
    ranked.sort(key=lambda item: item[0])

    def show(score, key, runs):
        inertia, mass, stiffness, damping = key
        ratios = ", ".join(f"{r['ratio']:.2f}" for r in runs)
        print(f"  {inertia:8g} {mass:7g} {stiffness:9g} {damping:7g} ({damping * ANG_DAMP_PER_UNIT:.0e})   "
              f"{ratios:>12}   {max(r['spread'] for r in runs):6.2f}  {max(r['jitter'] for r in runs):6.2f}   {score:5.2f}")

    usable = sum(1 for item in ranked if item[0] != float("inf"))
    print(f"\n{usable} of {len(ranked)} combinations stayed finite and grew to full length "
          f"(growth {args.grow_rate} m/s). Best 10:")
    print(f"   inertia    mass stiffness damping (si ang_damp)   achieved/commanded @ {CURVATURES}   spread  jitter   score")
    for item in ranked[:10]:
        show(*item)
    print("dvsim defaults:")
    show(*next(item for item in ranked if item[1] == (1e-5, 0.02, 0.3, 0.5)))
    wrapper = (1e-5, SimConfig.seg_mass_kg, SimConfig.joint_stiffness, SimConfig.ang_damp / ANG_DAMP_PER_UNIT)
    match = [item for item in ranked if item[1] == wrapper]
    if match:
        print("Sim wrapper defaults (scene.SimConfig):")
        show(*match[0])
