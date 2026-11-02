'''
funnel_test.py with the blocks left to come to rest. funnel_test stops every replay after the solution's own
schedule, whose last edge ends the step the nominal block enters the goal, so its blocks are still sliding
and their final centers spread along the sliding path. Here every sample (the same perturbations, read from
funnel.json) and the unperturbed solution are replayed in the same batches on the GPU, then settled with
growth off (block_puzzle.PushSearch's settle: grow force x 1e-3, until every block has been slow for 5 steps):
where the block ends up when the vine is stopped after the planned schedule.

Run from the repo root (GPU):
    python xnew_rrt/funnel_settle.py out/funnel/2a/funnel.json     # -> out/funnel/2a/funnel_settled.json
'''

import argparse, copy, json, math, os, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
from batched_rollout import BatchRollout
import checkpoint
import funnel_test
import run_envs
import tip_block_rrt
import viz

CHUNK = 512


def settle(sim, params, starts, max_steps):
    '''Hold each vine (growth off) until its blocks have been slow for 5 steps. Returns the final states.'''
    designs = [(s.p_control, s.l0_control, s.moment_control, s.bend_target) for s in starts]
    roll = BatchRollout(sim, starts, designs, params=params)
    still = torch.zeros(len(starts) + 1, dtype=torch.long, device=sim.device)
    for t in range(max_steps):
        if t % 10 == 0 and not roll.any_alive():
            break
        roll.step()
        el = roll.element_ids()
        el = torch.where(el >= 0, el, len(starts))
        slow = roll.x["obj_dstate"][:, :, :2].norm(dim=2).amax(1) < 2.0
        n = torch.where(slow, still[el] + 1, 0)
        still[el] = n
        roll.stop((n >= 5) & roll.last_moved)
    roll.finish()
    out = []
    for i, s in enumerate(starts):
        n = int(roll.length[i])
        out.append(roll.state_at(i, n - 1) if n else s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("funnel")
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="default: funnel_settled.json next to the input")
    args = ap.parse_args()
    os.chdir(run_envs.REPO)
    with open(args.funnel) as f:
        d = json.load(f)
    a = d["args"]
    e, scene, cfg, _ = run_envs.load_env(a["env"])
    with open(d["result"]) as f:
        result = json.load(f)
    checkpoint.apply(cfg, d["checkpoint"])
    cfg.max_bodies = result["sim"]["max_bodies"]
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    sim = Sim(scene, cfg)
    params = copy.copy(sim.device_params)
    params.grow_force = sim.device_params.grow_force * 1e-3

    sched = [tuple(s) for s in d["schedule"]]
    c0 = np.array([c for c, _ in sched])
    x0, y0, h0 = scene.start
    rows = [dict(family="unperturbed", scale=0.0, dx_mm=0.0, dy_mm=0.0, dheading_deg=0.0, dcurv=[0.0] * len(c0))] \
        + [{k: r[k] for k in ("family", "scale", "dx_mm", "dy_mm", "dheading_deg", "dcurv")} for r in d["samples"]]
    bases = np.array([[x0 + r["dx_mm"] / 1000, y0 + r["dy_mm"] / 1000, h0 + math.radians(r["dheading_deg"])] for r in rows])
    curvs = c0[None] + np.array([r["dcurv"] for r in rows])
    t0 = time.time()
    for i0 in range(0, len(rows), CHUNK):
        sl = slice(i0, i0 + CHUNK)
        grown = funnel_test.replay(sim, bases[sl], curvs[sl], sched)
        ok = [s is not None for s in grown]
        rested = settle(sim, params, [s for s in grown if s is not None], args.settle_steps)
        it = iter(rested)
        for r, g, k in zip(rows[sl], grown, ok):
            r["ok"] = k
            if not k:
                continue
            s = next(it)
            bx, by = tip_block_rrt.box_positions_mm(s)[0]
            gx, gy = tip_block_rrt.box_positions_mm(g)[0]
            r.update(box_x_mm=bx, box_y_mm=by, box_deg=math.degrees(float(s.moveable_obj_pose[0, 0, 2])),
                     slid_mm=math.hypot(bx - gx, by - gy), reached=bool(sim.box_reached_goal(s)))
            if r["family"] == "unperturbed":
                n = s.num_bodies
                r["vine_mm"] = [[viz.MM(float(s.state[0, 3 * j])), viz.MM(float(s.state[0, 3 * j + 1]))] for j in range(n)] \
                    + [list(viz._tip_mm(s))]
                r["box_pose"] = [bx, by, float(s.moveable_obj_pose[0, 0, 2])]
        print(f"  {min(i0 + CHUNK, len(rows))}/{len(rows)} replayed and settled, {time.time() - t0:.0f} s", flush=True)

    nom = rows[0]
    print(f"unperturbed: block at ({nom['box_x_mm']:.0f}, {nom['box_y_mm']:.0f}) mm after sliding {nom['slid_mm']:.0f} mm, "
          f"{'in' if nom['reached'] else 'NOT in'} the goal")
    out = args.out or os.path.join(os.path.dirname(args.funnel), "funnel_settled.json")
    with open(out, "w") as f:
        json.dump(dict(d, samples=rows[1:], nominal=nom, nominal_box_mm=[nom["box_x_mm"], nom["box_y_mm"]],
                       settled=dict(settle_steps=args.settle_steps, grow_force_scale=1e-3)), f)
    print("wrote", out)


if __name__ == "__main__":
    main()
