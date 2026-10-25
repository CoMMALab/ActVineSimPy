'''
Batched rollouts that stay on the GPU, each sim step recorded once as a CUDA graph and replayed.

    GraphStep     one dvsim step for a fixed batch size and params. The first call records it as a
                  CUDA graph (dvsim/batched_qp.py in graph mode: fixed row count, fixed iteration
                  count); later calls copy the inputs in and replay it: one launch instead of the
                  several hundred small kernels a step is made of. Used up to GRAPH_MAX_BATCH;
                  larger batches step eagerly (still on the device).
    BatchRollout  the states of a whole batch kept on the device between steps. Every step runs
                  the whole (padded) batch; elements that are done or failed are frozen with masks
                  instead of being dropped in Python, and the batch is compacted to a smaller
                  power of two once half of it is done. Stop tests are tensor ops (see
                  Sim.rollout_batch and block_puzzle.py). The history comes back to the CPU once,
                  at the end (finish()).

Sim.rollout_batch uses this on CUDA (SimConfig.cuda_graphs, on by default); on the CPU, or with
cuda_graphs off, the step runs eagerly through the same code.
'''

import math

import torch

import dvsim.batched_qp as bq
from dvsim.dynamic_vine import step
from rrt_util import StateInfo

KEYS = ("state", "dstate", "bodies", "obj_pose", "obj_dstate")
MIN_BUCKET = 8
# Graph steps for batches up to this size, eager steps above it. A graph step always runs every
# interior point iteration on padded rows, so it does ~4x the GPU work of an eager step, which stops
# once the batch has converged; it wins while launches dominate (1a on a shared 4090, ms per step):
#   batch      16    32    64   128   256   512
#   eager      63    62    64    65    69    84
#   graph      23    26    33    46    78   147
# With many boxes the padded rows grow (25 friction rows per box): 4d (13 boxes) is faster as a
# graph at batch 32 (56 vs 95 ms) but slower at 128 (204 vs 144 ms).
GRAPH_MAX_BATCH = 128
GRAPH_MAX_BATCH_MANY_BOXES = 32      # more than 2 boxes


def graph_max_batch(params):
    return GRAPH_MAX_BATCH if int(params.obj_mass.shape[0]) <= 2 else GRAPH_MAX_BATCH_MANY_BOXES


def bucket(n):
    return max(MIN_BUCKET, 1 << max(0, math.ceil(math.log2(max(n, 1)))))


CONTACT_ROWS = 40


def static_rows(params):
    '''Fixed row count for graph mode: the friction rows of every box (always live) and the
    CONTACT_ROWS nearest contact rows. Live contact rows within the 60 mm cull measured 8-40 for 1-2
    boxes; only those within a few mm can bind in one step (batched_qp.SAFE).'''
    n_obj = int(params.obj_mass.shape[0])
    return n_obj * (bq.N_XY + bq.N_RC) + CONTACT_ROWS


class GraphStep:
    '''dvsim.step for batch size B with these params, on `device`, as a CUDA graph (use_graph) or eager.'''

    def __init__(self, params, B, device, use_graph=True, pool=None, extra=()):
        '''extra: names of more per-element VineParams values, each (B,) (e.g. stiffness_val,
        damping, obj_friction; see dynamic_vine.step's per_batch).'''
        self.params, self.B, self.device = params, B, torch.device(device)
        self.extra = tuple(extra)
        self.use_graph = use_graph and self.device.type == "cuda"
        self.pool = pool
        self.graph = None
        mb, n_obj = params.max_bodies, int(params.obj_mass.shape[0])
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=self.device)
        self.inputs = dict(bases=z(B, 3), state=z(B, 3 * mb), dstate=z(B, 3 * mb),
                           bodies=z(B, 1, dtype=torch.long), obj_pose=z(B, n_obj, 3), obj_dstate=z(B, n_obj, 3),
                           bend_target=z(B, mb), spam_moment=z(B, mb), **{k: z(B) for k in self.extra})

    def _run(self):
        i = self.inputs
        with torch.device(self.device):
            out = step(self.params, i["bases"][:, 2:3], i["bases"][:, 0:1], i["bases"][:, 1:2], i["state"],
                       i["dstate"], i["bodies"], i["obj_pose"], i["obj_dstate"], return_ok=True,
                       per_batch={k: i[k] for k in ("spam_moment", "bend_target") + self.extra})
            over = bq.OVERFLOW if (bq.OVERFLOW is not None and bq.STATIC_ROWS is not None) \
                else torch.zeros(1, dtype=torch.bool)
        return (*out, over)

    def _capture(self):
        saved = bq.STATIC_ROWS, bq.GRAPH, torch.backends.cuda.preferred_linalg_library()
        bq.STATIC_ROWS, bq.GRAPH = static_rows(self.params), True
        torch.backends.cuda.preferred_linalg_library("cusolver")   # MAGMA's batched LU syncs with the host
        try:
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):                                   # warm up (allocations, cuBLAS/cuSOLVER handles)
                    self._run()
            torch.cuda.current_stream(self.device).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.outputs = self._run()
        finally:
            bq.STATIC_ROWS, bq.GRAPH = saved[:2]
            torch.backends.cuda.preferred_linalg_library(saved[2])

    def __call__(self, **inputs):
        '''Returns (state, dstate, bodies, obj_pose, obj_dstate, ok, overflow), fresh tensors.'''
        for k, v in inputs.items():
            self.inputs[k].copy_(v)
        if not self.use_graph:
            return self._run()
        if self.graph is None:
            self._capture()
        self.graph.replay()
        return tuple(t.clone() for t in self.outputs)


class BatchRollout:
    '''
    A batch of rollouts kept on the device. starts: [StateInfo] (CPU); designs: [(p, l0, moment,
    bend_target)] per element, as Sim.apply_control returns them; params: dvsim params on the device
    (default sim.device_params). Call step() until done; stop(mask) ends elements early (after the
    state of this step, which is kept); finish() brings the history back to the CPU.
    '''

    def __init__(self, sim, starts, designs, params=None, record=True):
        self.sim, self.params = sim, params if params is not None else sim.device_params
        self.dev = sim.device
        self.N = len(starts)
        self.designs, self.bases_cpu = designs, [s.base for s in starts]
        self.record = record
        with torch.device(self.dev):
            cat = lambda get: torch.cat([get(s) for s in starts], 0).to(self.dev)
            self.x = dict(state=cat(lambda s: s.state), dstate=cat(lambda s: s.dstate),
                          bodies=cat(lambda s: s.bodies).long(), obj_pose=cat(lambda s: s.moveable_obj_pose),
                          obj_dstate=cat(lambda s: s.moveable_obj_dstate))
            self.bases = torch.stack([sim.base_of(s) for s in starts]).to(self.dev)
            self.bend = torch.stack([d[3] for d in designs]).to(self.dev)
            self.spam = torch.stack([d[2] for d in designs]).to(self.dev)
            self.ids = torch.arange(self.N, device="cpu")           # row -> element (CPU)
            self.alive = torch.tensor([sim.can_grow(s) for s in starts], device=self.dev)
            self.failed = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
            self.failed_el = torch.zeros(self.N, dtype=torch.long, device=self.dev)   # per element
            self.overflow = torch.zeros(1, dtype=torch.bool, device=self.dev)
        self.segments = []                                          # [(ids, [per-step dict])]
        self.steps = 0
        self._pad()

    # -- batch layout

    def _pad(self):
        '''Pad the rows to a power of two by repeating row 0, frozen.'''
        n = len(self.ids)
        b = bucket(n)
        if b > n:
            rep = lambda t: torch.cat([t, t[:1].expand(b - n, *t.shape[1:])], 0)
            self.x = {k: rep(v) for k, v in self.x.items()}
            self.bases, self.bend, self.spam = rep(self.bases), rep(self.bend), rep(self.spam)
            self.alive = torch.cat([self.alive, torch.zeros(b - n, dtype=torch.bool, device=self.dev)])
            self.failed = torch.cat([self.failed, torch.zeros(b - n, dtype=torch.bool, device=self.dev)])
            self.ids = torch.cat([self.ids, torch.full((b - n,), -1, dtype=torch.long, device="cpu")])
        self.ids_dev = self.ids.to(self.dev)
        self.segments.append((self.ids.clone(), []))

    def _compact(self):
        keep = self.alive.nonzero().squeeze(1)
        if len(keep) == 0 or bucket(len(keep)) >= len(self.ids):
            return
        kc = keep.cpu()
        self.x = {k: v[keep] for k, v in self.x.items()}
        self.bases, self.bend, self.spam = self.bases[keep], self.bend[keep], self.spam[keep]
        self.alive, self.failed = self.alive[keep], self.failed[keep]
        self.ids = self.ids[kc]
        self._pad()

    def stepper(self):
        sim = self.sim
        cache = sim.__dict__.setdefault("_graph_steps", {})
        key = (id(self.params), len(self.ids))
        if key not in cache:
            if sim.device.type == "cuda" and "_graph_pool" not in sim.__dict__:
                sim._graph_pool = torch.cuda.graph_pool_handle()
            cache[key] = GraphStep(self.params, len(self.ids), self.dev,
                                   sim.cfg.cuda_graphs and len(self.ids) <= graph_max_batch(self.params),
                                   pool=sim.__dict__.get("_graph_pool"))
        return cache[key]

    # -- stepping

    def any_alive(self):
        return bool(self.alive.any())

    def step(self):
        # compact first (not after the step), so last_moved and the rows always match
        if self.steps and self.steps % 10 == 0 and len(self.ids) > MIN_BUCKET:
            if int(self.alive.sum()) <= len(self.ids) // 2:
                self._compact()
        st, ds, bo, op, od, ok, over = self.stepper()(
            bases=self.bases, state=self.x["state"], dstate=self.x["dstate"], bodies=self.x["bodies"],
            obj_pose=self.x["obj_pose"], obj_dstate=self.x["obj_dstate"], bend_target=self.bend,
            spam_moment=self.spam)
        good = ok & torch.isfinite(st).all(1) & torch.isfinite(op).flatten(1).all(1)
        moved = self.alive & good
        fr = self.alive & ~good & (self.ids_dev >= 0)
        self.failed |= fr
        self.failed_el.index_add_(0, self.ids_dev.clamp(min=0), fr.long())
        new = dict(state=st, dstate=ds, bodies=bo, obj_pose=op, obj_dstate=od)
        for k in KEYS:
            m = moved.view(-1, *[1] * (new[k].dim() - 1))
            self.x[k] = torch.where(m, new[k], self.x[k])
        self.alive = moved & (self.x["bodies"][:, 0] < self.sim.cfg.max_bodies - 1)   # Sim.can_grow
        self.overflow |= over
        self.last_moved = moved
        if self.record:
            self.segments[-1][1].append(dict(self.x, moved=moved))
        self.steps += 1

    def stop(self, mask):
        '''End the elements (rows) in mask after this step.'''
        self.alive &= ~mask

    def rows_of(self, elements):
        '''Current row of each element (CPU long tensor; -1 if it is not in the batch any more).'''
        where = {int(e): r for r, e in enumerate(self.ids) if e >= 0}
        return torch.tensor([where.get(int(e), -1) for e in elements])

    def element_ids(self):
        return self.ids_dev

    # -- results

    def finish(self):
        '''
        History on the CPU: self.hist[key] (T, N, ...) for every key in KEYS, with each element's
        own value at every step it was stepped, and self.valid (T, N): stepped at that step.
        self.length[i]: how many steps element i made (its states are steps 0..length-1);
        self.ok[i]: False if its QP failed.
        '''
        T = sum(len(s) for _, s in self.segments)
        ref = self.segments[0][1][0] if T else {k: self.x[k] for k in KEYS}
        self.hist = {k: torch.zeros(T, self.N, *ref[k].shape[1:], dtype=ref[k].dtype) for k in KEYS}
        self.valid = torch.zeros(T, self.N, dtype=torch.bool)
        t = 0
        for ids, steps in self.segments:
            if not steps:
                continue
            rows = (ids >= 0).nonzero().squeeze(1)
            el = ids[rows]
            for k in KEYS:
                self.hist[k][t:t + len(steps), el] = torch.stack([s[k] for s in steps]).cpu()[:, rows]
            self.valid[t:t + len(steps), el] = torch.stack([s["moved"] for s in steps]).cpu()[:, rows]
            t += len(steps)
        self.length = self.valid.sum(0)
        self.ok = self.failed_el.cpu() == 0
        self.segments = []
        return self

    def state_at(self, i, t):
        '''StateInfo of element i after step t (CPU).'''
        return StateInfo(self.hist["state"][t, i:i + 1], self.hist["dstate"][t, i:i + 1],
                         self.hist["bodies"][t, i:i + 1], self.hist["obj_pose"][t, i:i + 1],
                         self.hist["obj_dstate"][t, i:i + 1], *self.designs[i], base=self.bases_cpu[i])

    def states(self, i, upto=None):
        '''[StateInfo] of element i (CPU), steps 0..length-1 (or ..upto-1).'''
        n = int(self.length[i]) if upto is None else upto
        d = self.designs[i]
        return [StateInfo(self.hist["state"][t, i:i + 1], self.hist["dstate"][t, i:i + 1],
                          self.hist["bodies"][t, i:i + 1], self.hist["obj_pose"][t, i:i + 1],
                          self.hist["obj_dstate"][t, i:i + 1], *d, base=self.bases_cpu[i]) for t in range(n)]
