'''
How well does one dvsim step batch? Grows a vine in an env to a working length, then times
batched steps of copies of that state at several batch sizes, split into
    forward   the per-element constraint measures and Jacobians (torch.func.vmap)
    qp        the QP solve (cvxpylayers -> diffcp -> Clarabel, one solve per element on a thread pool)
    rest      building the QP matrices, stacking states, integrating
Run from the repo root (CPU; the sim never uses the GPU):
    python xnew_rrt/bench_batch.py envs/1a.json envs/4d.json --batch 1 4 16 64 --threads 4
Writes out/bench_batch.json and prints a table.

For comparison, --pbd times master's soft-constraint JAX sim (pbd_vine.py, what kinodynamic/sst.py
plans with) on the GPU instead, capped at half its memory: one jitted, vmapped step of the whole
batch, a vine of the same max_bodies among a few walls, no movable boxes (it has none), with a
linear stand-in for the sPAM bending model:
    python xnew_rrt/bench_batch.py --pbd --batch 1 64 1024 4096
'''

import argparse, json, os, time

import torch

from scene import Sim  # (puts the repo root on sys.path)
import dvsim.dynamic_vine as dv
import run_envs


def time_steps(sim, start, B, steps):
    clock = {"qp": 0.0, "forward": 0.0}
    solve, forward = dv.solve_layers, dv.dynamic_forward_part

    def timed(name, fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            out = fn(*a, **k)
            clock[name] += time.perf_counter() - t
            return out
        return wrapper

    dv.solve_layers = timed("qp", solve)
    # the forward part runs inside vmap, so time the vmapped call around it instead
    vmap = torch.func.vmap
    outer = ("forward_one", "object_env_constraints")
    torch.func.vmap = lambda f, *a, **k: (timed("forward", vmap(f, *a, **k))
                                          if getattr(getattr(f, "func", f), "__name__", "") in outer else vmap(f, *a, **k))
    try:
        sim.rollout_batch([start] * B, [0.0] * B, 1)          # build the QP layer for this shape
        clock.update(qp=0.0, forward=0.0)
        t = time.perf_counter()
        sim.rollout_batch([start] * B, [1.0 * (k % 5 - 2) for k in range(B)], steps)
        total = time.perf_counter() - t
    finally:
        dv.solve_layers, torch.func.vmap = solve, vmap
    per = {k: v / steps for k, v in clock.items()}
    per["total"] = total / steps
    per["rest"] = per["total"] - per["qp"] - per["forward"]
    return per


def bench_pbd(batches, steps, max_bodies=60):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    import jax, jax.numpy as jnp
    from pbd_vine import VineParams, step_vine_batched
    walls = [[-20, 20, 10, 80], [40, -40, 70, 30], [130, -20, 160, 80], [220, -40, 250, 30],
             [10, 60, 450, 80], [10, -60, 450, -40]]
    params = VineParams(max_bodies=max_bodies, body_length=12.0, radius=8.0, dt=1 / 10, grow_rate=20.0,
                        grow_force=5.0, stiffness=12.0, damping=50.0, substeps=15, alpha=1e-2,
                        obstacle_rects=__import__("numpy").array(walls, dtype="float32"))
    bend = lambda r, a, b: a * jnp.where(r == 0, 0.0, 1e-3 / jnp.where(r == 0, 1.0, r))
    forward = jax.jit(step_vine_batched, static_argnames=["params", "bend_energy_func"])
    rows = []
    for B in batches:
        cs = jnp.zeros((B, max_bodies + 1)).at[:, max_bodies].set(params.body_length)
        nb = jnp.full((B,), 1, dtype=jnp.int32)
        bp = jnp.zeros((B, max_bodies, 2)).at[:, :, 0].set(jnp.linspace(-1, 1, B)[:, None])
        x0, y0, h0 = 0.0, -30.0, 0.4
        for _ in range(150):                       # compile, and grow to a working length
            cs, nb = forward(params, cs, nb, bp, x0, y0, h0, bend)
        cs.block_until_ready()
        t = time.perf_counter()
        for _ in range(steps):
            cs, nb = forward(params, cs, nb, bp, x0, y0, h0, bend)
        cs.block_until_ready()
        per = (time.perf_counter() - t) / steps
        rows.append(dict(sim="pbd_vine (jax, gpu)", batch=B, total_ms=1000 * per, element_steps_per_s=B / per,
                         bodies=int(nb[0])))
        print(f"pbd  B {B:5d}  step {1000 * per:7.2f} ms  {B / per:9.0f} element-steps/s  ({int(nb[0])} bodies)", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="*")
    parser.add_argument("--pbd", action="store_true", help="time master's JAX PBD sim on the GPU instead")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--grow-steps", type=int, default=100, help="grow the vine this long first")
    parser.add_argument("--steps", type=int, default=5, help="timed steps per batch size")
    parser.add_argument("--threads", type=int, default=4, help="torch threads (diffcp's pool uses all cores)")
    parser.add_argument("--out", default="out/bench_batch.json")
    args = parser.parse_args()
    if args.pbd:
        rows = bench_pbd(args.batch, max(args.steps, 20))
        with open(args.out.replace(".json", "_pbd.json"), "w") as f:
            json.dump(rows, f, indent=1)
        return
    torch.set_num_threads(args.threads)

    rows = []
    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        sim = Sim(scene, cfg)
        states, _ = sim.rollout(sim.initial_state(), 0.0, args.grow_steps)
        start = states[-1]
        bodies = int(start.bodies.item())
        for B in args.batch:
            per = time_steps(sim, start, B, args.steps)
            rows.append(dict(env=e["name"], boxes=len(e["boxes"]), bodies=bodies, max_bodies=cfg.max_bodies, batch=B,
                             **{k + "_ms": 1000 * v for k, v in per.items()},
                             element_steps_per_s=B / per["total"]))
            r = rows[-1]
            print(f"{r['env']:>3} boxes {r['boxes']:2d} bodies {bodies:2d}  B {B:3d}  step {r['total_ms']:7.1f} ms  "
                  f"(qp {r['qp_ms']:6.1f}, forward {r['forward_ms']:6.1f}, rest {r['rest_ms']:6.1f})  "
                  f"{r['element_steps_per_s']:6.1f} element-steps/s", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
