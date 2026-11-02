'''
torch.compile for the part of a sim step before the QP (dvsim.dynamic_vine.forward_all: vine
measures, contact Jacobians by jacrev under vmap, object-wall/object separations, contact tangents).
Eagerly that part is thousands of tiny kernels, each with Python dispatch; compiled it is a few
fused ones. The QP (batched_qp) stays as it is. enable() swaps it in for every step.

Run as a script to check it against the eager version on real states and time both:
    python xnew_rrt/compiled_forward.py envs/1b.json --checkpoint checkpoints/wrinkle_mu03.json --batch 256 1024
'''

import argparse, random, time

import torch

from scene import Sim  # (puts the repo root on sys.path)
import dvsim.dynamic_vine as dv

_compiled = None


def enable(mode=None):
    '''Make dvsim's step use the compiled forward part (mode: torch.compile's, None = default).'''
    global _compiled
    # one compiled graph per batch size (the planner's batches are powers of two, compacted as
    # rollouts end), so allow more than torch's default 8 before it falls back to eager
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)
    if _compiled is None:
        _compiled = torch.compile(dv.forward_all, mode=mode, dynamic=False)
    dv.FORWARD = _compiled


def disable():
    dv.FORWARD = dv.forward_all


def main():
    import run_envs, checkpoint
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--steps", type=int, default=250, help="rollout length the test states are drawn from")
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.7)
    e, scene, cfg, _ = run_envs.load_env(args.env)
    if args.checkpoint:
        checkpoint.apply(cfg, args.checkpoint)
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    sim = Sim(scene, cfg)
    params, dev = sim.device_params, sim.device

    for B in args.batch:
        rng = random.Random(B)
        curv = [rng.uniform(-3.33, 3.33) for _ in range(B)]
        starts = [sim.initial_state()] * B
        res = sim.rollout_batch(starts, curv, args.steps)
        # one state per element from anywhere along its rollout: free growth and contact mixed
        picks = [states[rng.randrange(len(states))] for states, _ in res]
        designs = [sim.apply_control(s, c) for s, c in zip(picks, curv)]
        cat = lambda get: torch.cat([get(s) for s in picks]).to(dev)
        bases = torch.stack([sim.base_of(s) for s in picks]).to(dev)
        x = (params, bases[:, 2:3], bases[:, 0:1], bases[:, 1:2], cat(lambda s: s.state), cat(lambda s: s.dstate),
             cat(lambda s: s.bodies).long(), cat(lambda s: s.moveable_obj_pose))
        per = {"spam_moment": torch.stack([d[2] for d in designs]).to(dev),
               "bend_target": torch.stack([d[3] for d in designs]).to(dev)}
        fresh = lambda: (*x[:4], *(t.clone() for t in x[4:]))    # (forward promotes the tip in place)
        with torch.device(dev):
            ex = torch._dynamo.explain(dv.forward_all)(*fresh(), per)
            print(f"dynamo: {ex.graph_count} graphs, {ex.graph_break_count} breaks, {ex.op_count} ops")
            for r in ex.break_reasons[:8]:
                print("   break:", str(r.reason).split(chr(10))[0][:200])
            torch._dynamo.reset()
            ref = dv.forward_all(*fresh(), per)
            fn = torch.compile(dv.forward_all, mode=args.mode, dynamic=False)
            t = time.perf_counter()
            out = fn(*fresh(), per)
            torch.cuda.synchronize()
            compile_s = time.perf_counter() - t
            err, names = 0.0, ("bodies forces growth sdf dev L J gws gwd prox pjs pjp ow owJ oo ooJ tangents").split()
            for nm, a, b in zip(names, ref, out):
                if a is None:
                    continue
                if isinstance(a, tuple):
                    pairs = zip(a, b)
                else:
                    pairs = [(a, b)]
                for u, v in pairs:
                    if nm in ("oo", "ooJ"):           # a box with itself: masked out in dynamic_solve
                        keep = ~torch.eye(u.shape[1], dtype=torch.bool, device=u.device)
                        u, v = u[:, keep], v[:, keep]
                    d = float(((u - v).abs() / (1 + u.abs())).max()) if u.is_floating_point() else float((u != v).any())
                    if d > 1e-6:
                        print(f"   {nm}: max relative difference {d:.2e}")
                    if nm == "ooJ" and d > 1e-3:
                        gap = ref[names.index("oo")][:, keep].reshape(len(u), -1)                # (B, pairs)
                        bad = ((u - v).abs() / (1 + u.abs())).reshape(len(u), gap.shape[1], -1).amax(2) > 1e-3
                        print(f"   ooJ differs in {int(bad.sum())} of {bad.numel()} box pairs, gaps there "
                              f"{gap[bad].min():.3g} to {gap[bad].max():.3g} (internal units, mm)")
                    err = max(err, d)

            def timed(f, n=20):
                f(*fresh(), per); torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(n):
                    f(*fresh(), per)
                torch.cuda.synchronize()
                return 1000 * (time.perf_counter() - t) / n

            te, tc = timed(dv.forward_all), timed(fn)
        print(f"{e['name']} B={B}: forward eager {te:.1f} ms, compiled {tc:.1f} ms ({te / tc:.1f}x), "
              f"first compile {compile_s:.0f} s, max relative difference {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
