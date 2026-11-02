'''
Where a sim step's time goes, at several batch sizes, on the planner's real code path.

Part 1 (eager, CUDA-synchronized timers around each stage; the timers add syncs, so the stage sum is a
bit above the plain eager step):
    forward      vine measures (forward_batched_part) + vine-object proximity and Jacobians, under vmap
    obj_env      object-wall and object-object separations and Jacobians
    assembly     building M, p, G, h, A, b in dynamic_solve
    QP: friction linearization, row culling, equality elimination (LU), IPM iterations
Part 2: plain step time, eager vs CUDA graph (what BatchRollout runs).
Part 3: the sPAM part. In 'curvature' mode (the default checkpoint) actuation is a per-joint bend
target; in 'spam' mode each rollout first designs its actuator (pressure, rest length) with the MLP
surrogate, once per rollout, not per step. Timed per design.

    python xnew_rrt/profile_sim.py envs/3a.json envs/5c.json --batch 32 128 512
'''

import argparse, json, os, random, time
from collections import defaultdict

import torch

from scene import Sim  # (puts the repo root on sys.path)
import run_envs, checkpoint
import batched_rollout
import dvsim.dynamic_vine as dv
import dvsim.batched_qp as bq

T = defaultdict(float)
ON = [False]


def timed(name, fn):
    def wrap(*a, **k):
        if ON[0] == "kernel":
            with torch.profiler.record_function(name):
                return fn(*a, **k)
        if not ON[0]:
            return fn(*a, **k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t
        return out
    return wrap


def install():
    dv.forward_batched_part = timed("forward: vine", dv.forward_batched_part)
    dv.proximity_measure = timed("forward: vine-object gap", dv.proximity_measure)
    dv.proximity_jacobians = timed("forward: vine-object Jacobian", dv.proximity_jacobians)
    dv.object_env_constraints = timed("object-wall/object", dv.object_env_constraints)
    dv.dynamic_solve = timed("_dynamic_solve", dv.dynamic_solve)
    dv.solve_layers = timed("_qp", dv.solve_layers)
    bq.linearize_friction = timed("qp: friction linearization", bq.linearize_friction)
    bq.cull_rows = timed("qp: row culling", bq.cull_rows)
    bq.eliminate_equalities = timed("qp: equality elimination (LU)", bq.eliminate_equalities)
    bq.ipm = timed("qp: IPM iterations", bq.ipm)


def inputs(sim, B, steps_in):
    '''A batch of mid-rollout states (grown `steps_in` steps with random curvatures), so contacts exist.'''
    rng = random.Random(B)
    curv = [rng.uniform(-3.3, 3.3) for _ in range(B)]
    res = sim.rollout_batch([sim.initial_state()] * B, curv, steps_in)
    states = [r[0][-1] if r[0] else sim.initial_state() for r in res]
    designs = [sim.apply_control(sim.initial_state(), c) for c in curv]
    return states, designs


def time_steps(gs, kw, n):
    for _ in range(3):
        gs(**kw)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        gs(**kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--batch", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--steps-in", type=int, default=120, help="grow this far first so there is contact")
    parser.add_argument("--n", type=int, default=20, help="timed steps per setting")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="out/bench/profile_sim.json")
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.7)
    install()
    rows = []
    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        if args.checkpoint:
            checkpoint.apply(cfg, args.checkpoint)
        sim = Sim(scene, cfg)
        n_obj = int(sim.params.obj_mass.shape[0])
        for B in args.batch:
            states, designs = inputs(sim, B, args.steps_in)
            roll = batched_rollout.BatchRollout(sim, states, designs, record=False)
            kwi = dict(bases=roll.bases, bend_target=roll.bend, spam_moment=roll.spam, **roll.x)
            row = {"env": e["name"], "batch": B, "objects": n_obj, "walls": int(sim.params.obstacle_hw.shape[0])}
            for mode in ("eager", "graph"):
                if mode == "graph" and B > batched_rollout.graph_max_batch(sim.device_params):
                    continue
                gs = batched_rollout.GraphStep(sim.device_params, B, sim.device, use_graph=mode == "graph",
                                               pool=None)
                row[f"{mode}_ms"] = 1000 * time_steps(gs, kwi, args.n)
            # stage breakdown, eager
            gs = batched_rollout.GraphStep(sim.device_params, B, sim.device, use_graph=False)
            gs(**kwi)
            T.clear()
            ON[0] = True
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.n):
                gs(**kwi)
            torch.cuda.synchronize()
            ON[0] = False
            total = (time.perf_counter() - t) / args.n
            st = {k: 1000 * v / args.n for k, v in T.items()}
            qp = st.pop("_qp")
            ds = st.pop("_dynamic_solve")
            st["assembly (dynamic_solve minus QP)"] = ds - qp
            st["QP other"] = qp - sum(v for k, v in st.items() if k.startswith("qp:"))
            st["rest of step"] = 1000 * total - ds - sum(v for k, v in st.items()
                                                         if k.startswith("forward") or k.startswith("object"))
            row["stages_ms"] = st
            # GPU kernel time per stage (what a CUDA graph replay pays: no launch or Python overhead)
            ON[0] = "kernel"
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(args.n):
                    gs(**kwi)
                torch.cuda.synchronize()
            ON[0] = False
            names = set(T) | {"_qp", "_dynamic_solve"}
            gpu, all_k = defaultdict(float), 0.0
            for ev in prof.profiler.function_events:      # each kernel goes to every stage it was launched in
                k_us = sum(k.duration for k in ev.kernels)
                if not k_us:
                    continue
                all_k += k_us
                seen, par = set(), ev
                while par is not None:
                    if par.name in names and par.name not in seen:
                        gpu[par.name] += k_us
                        seen.add(par.name)
                    par = par.cpu_parent
            gpu = {k: v / 1000 / args.n for k, v in gpu.items()}
            all_k /= 1000 * args.n
            if gpu:
                gq, gd = gpu.pop("_qp", 0), gpu.pop("_dynamic_solve", 0)
                gpu["assembly (dynamic_solve minus QP)"] = gd - gq
                gpu["QP other"] = gq - sum(v for k, v in gpu.items() if k.startswith("qp:"))
                gpu["rest of step"] = all_k - gd - sum(v for k, v in gpu.items() if k.startswith("forward") or k.startswith("object"))
            row["gpu_ms"], row["gpu_total_ms"] = gpu, all_k
            row["timed_total_ms"] = 1000 * total
            rows.append(row)
            print(f"\n{e['name']} B={B} ({n_obj} boxes, {row['walls']} walls): eager {row['eager_ms']:.1f} ms/step"
                  + (f", graph {row['graph_ms']:.1f} ms/step" if "graph_ms" in row else "")
                  + f"  -> {B / row.get('graph_ms', row['eager_ms']) * 1000:.0f} element-steps/s", flush=True)
            gpu = row["gpu_ms"]
            print(f"    {'stage':38s} {'wall (eager)':>14s} {'GPU kernels':>14s}   (GPU total {row['gpu_total_ms']:.1f} ms)")
            for k, v in sorted(st.items(), key=lambda kv: -kv[1]):
                print(f"    {k:38s} {v:8.2f} ms {100 * v / (1000 * total):4.0f}% {gpu.get(k, 0):8.2f} ms "
                      f"{100 * gpu.get(k, 0) / max(row['gpu_total_ms'], 1e-9):4.0f}%")
    # sPAM design cost (spam mode only)
    from dvsim import spam
    from dvsim.vine import STIFF_FACTOR
    spam._load()
    rng = random.Random(0)
    curv = [rng.uniform(0.3, 3.3) for _ in range(50)]
    t = time.perf_counter()
    for c in curv:
        p, l0 = spam.design_actuator(0.01 * c, 1.0 / c)
        spam.actuation_moment(p, l0, 1.0 / c)
    per = (time.perf_counter() - t) / len(curv)
    print(f"\nsPAM design (spam mode, once per rollout): {1000 * per:.1f} ms per rollout "
          f"({len(spam.L0_CANDIDATES) + 1} MLP calls on the CPU); per step it is a constant moment: 0 ms")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "spam_design_ms": 1000 * per}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
