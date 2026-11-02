'''
Why the throughput benchmark looks wrong (out/fit/throughput.log): GPU batch 1-4 slower than 16, and 4d at batch
1024 taking 410 s per 25-step rollout (2.8 s at 256). Same timed call as xnew_rrt/bench_throughput.py, but for each
(env, batch) also: graph or eager path, padded bucket, peak GPU memory, allocator retries / OOMs, interior point
iterations (eager), and for the small batches the same batch at curvature 0 vs the extreme +-2 the benchmark uses.
    xnew_rrt/py.sh out/fit/throughput_diag.py > out/fit/throughput_diag.log
'''
import json, os, sys, time

import numpy as np
import torch

REPO = "/home/yitian/vine-push"
sys.path.insert(0, os.path.join(REPO, "xnew_rrt"))
from bench_throughput import make_sim, grown  # noqa: E402
import batched_rollout  # noqa: E402
import dvsim.batched_qp as bq  # noqa: E402

CK = os.path.join(REPO, "checkpoints", "spin8_manual.json")
RUNS = {"4d": [1, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024], "1a": [1, 4, 8, 16, 64, 256, 1024]}
STEPS, REPS = 25, 3


def timed(sim, start, curv, reps):
    sim.rollout_batch([start] * len(curv), curv, STEPS)        # warm up / graph capture
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s0 = torch.cuda.memory_stats()
    bq.STATS = []
    t = time.perf_counter()
    for _ in range(reps):
        res = sim.rollout_batch([start] * len(curv), curv, STEPS)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    s1 = torch.cuda.memory_stats()
    its = torch.cat(bq.STATS) if bq.STATS else None
    bq.STATS = None
    return dict(seconds=dt, steps_done=float(np.mean([len(r[0]) for r in res])), failed=sum(not r[1] for r in res),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                alloc_retries=s1.get("num_alloc_retries", 0) - s0.get("num_alloc_retries", 0),
                ooms=s1.get("num_ooms", 0) - s0.get("num_ooms", 0),
                ipm_iters_mean=float(its[its >= 0].float().mean()) if its is not None and (its >= 0).any() else None,
                ipm_never=int((its < 0).sum()) if its is not None else None)


def parts(sim, start, curv):
    '''The same rollout split: GPU stepping (BatchRollout.step x STEPS) vs bringing the history to the CPU
    (finish) vs building the per-step StateInfo lists rollout_batch returns.'''
    designs = [sim.apply_control(start, c) for c in curv]
    roll = batched_rollout.BatchRollout(sim, [start] * len(curv), designs)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(STEPS):
        roll.step()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    roll.finish()
    t2 = time.perf_counter()
    for i in range(len(curv)):
        roll.states(i)
    t3 = time.perf_counter()
    return dict(step_s=t1 - t0, finish_s=t2 - t1, stateinfo_s=t3 - t2)


rows = []
for env, batches in RUNS.items():
    e, sim = make_sim(os.path.join(REPO, "envs", env + ".json"), CK, "TORCH_IPM", "cuda")
    start = grown(sim, 60)
    gmax = batched_rollout.graph_max_batch(sim.device_params)
    for B in batches:
        cases = [("spread", [float(c) for c in np.linspace(-2.0, 2.0, B)])]
        if B <= 16:
            cases.append(("zero", [0.0] * B))
        for label, curv in cases:
            try:
                r = timed(sim, start, curv, 1 if B >= 512 else REPS)
            except torch.cuda.OutOfMemoryError as err:
                r = dict(error=f"OOM: {err}"[:200])
                torch.cuda.empty_cache()
            if "error" not in r:
                r.update(parts(sim, start, curv))
            r.update(env=env, batch=B, curvature=label, bucket=batched_rollout.bucket(B), graph_max=gmax)
            rows.append(r)
            print(json.dumps(r), flush=True)
with open(os.path.join(REPO, "out", "fit", "throughput_diag.json"), "w") as f:
    json.dump(rows, f, indent=1)
print("done")
