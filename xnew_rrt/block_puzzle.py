'''
Hierarchical planner for the block puzzles (envs/5*.json, written by envs/make_puzzles.py).

High level: a discrete planner over the grid cells of every block (block 0 has the goal; the
others, which may span several cells, are obstacles that can be pushed out of the way). A move
pushes one block k cells along one axis. It needs the k cells ahead free (no wall, spawn-ring cell
or other block), and room for a vine behind the block: a cell behind its back face free and either
    straight    a free straight corridor from there back to the spawn ring, or
    deflected   a wall beside that cell for the vine to slide along, and the cell diagonally
                behind it free, so a diagonal vine can hit the wall and turn into the push
                (5c needs these: a stub wall blocks the straight approach)
Dijkstra over the blocks' cells (pushes, plus a penalty for deflected moves and for moves that no
wall stops).

Low level: realizes the plan's first move with the batched sim (dvsim/batched_qp.py on the GPU).
Candidates are every spawn-ring cell x heading (the env's N uniform headings plus headings aimed at
the cells behind the block, its back face and the guide walls; --no-aim drops these) whose straight
ray hits the block from behind, or hits a wall next to a cell behind it (--seed instead draws the
uniform headings at random from the 180 deg facing the push, and jitters the aimed ones at random,
so repeated runs differ); all of them grow straight
(no actuation) in one batched rollout. For each, the stop time whose predicted resting pose (the
block slides on after the vine stops: v^2 / 2 mu g, cut short by walls and the other blocks) is
nearest a planned cell is settled in a second batched run with growth switched off. A push counts
only if the block comes to rest in a cell the plan puts it in: the first move's target, or a later
one if the plan keeps pushing the same block in the same direction (one vine may push a block
several cells along an axis, but turning it onto another axis takes another vine). It is invalid
if any block touches a spawn-ring cell, another block is moved, or the block drifts more than
LATERAL_MM off the push axis, on the way.

Search: depth first, with backtracking, so nothing is committed until a whole sequence reaches the
goal. From the current node (the blocks' simulated poses) the high level plans; a push that fails
bans that block's moves in that direction from that node and the high level replans there; a node
with no plan left is abandoned, and the move that led to it is banned at its parent. Every push in
the final answer was simulated from the exact pose the previous push left (open loop).

A push whose block turns more than 15 deg (FLAG_DEG) on the way is flagged. One that leaves it
turned more than the jam angle is not accepted: past it the block no longer fits through a
one-cell corridor (7.5 deg for a 98 mm block in 110 mm cells, 9.8 deg with 10 mm rounded corners;
--max-final-angle overrides).

Run from the repo root:
    python xnew_rrt/block_puzzle.py envs/5a.json envs/5b.json envs/5c.json envs/5d.json
Per env, in out/solutions/<kind>/<name>.* (xnew_rrt/outputs.py; replaced on every run): plan.png (the
first high-level plan), pushes.png (the solution's pushes), solution.gif/.mp4, result.json.
'''

import argparse, copy, heapq, json, math, os, shutil, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
from batched_rollout import BatchRollout
import checkpoint
import outputs
import run_envs
import viz

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}   # grid (column right, row down)
FLAG_DEG = 15.0
G_MM = 9810.0
OTHER_MOVED_MM = 15.0      # another block that moves more than this invalidates a push
LATERAL_MM = 30.0          # a pushed block that drifts more than this off the push axis invalidates it
ALT_MM = 3.0               # a fallback push must leave some block at least this far from the ones kept before it


#----------------------------------------------------------------------- grid

class Grid:
    def __init__(self, e):
        g = e["grid"]
        self.rows, self.cell = g["rows"], g["cell_mm"]
        self.n_cols, self.n_rows = len(self.rows[0]), len(self.rows)
        self.goal = tuple(g["goal"])
        self.n_headings = g["n_headings"]
        # each block: its cells' offsets from its anchor (its top-left cell), and its start anchor
        blocks = g.get("blocks") or [{"cells": [g["block"]]}]
        self.shapes, self.start = [], []
        for b in blocks:
            cells = [tuple(c) for c in b["cells"]]
            a = (min(c[0] for c in cells), min(c[1] for c in cells))
            self.shapes.append(tuple((c[0] - a[0], c[1] - a[1]) for c in cells))
            self.start.append(a)
        self.start = tuple(self.start)

    def inside(self, c):
        return 0 <= c[0] < self.n_cols and 0 <= c[1] < self.n_rows

    def wall(self, c):
        return not self.inside(c) or self.rows[c[1]][c[0]] in "#q"

    def ring(self, c):
        return self.inside(c) and self.rows[c[1]][c[0]] == "o"

    def center(self, c):
        return np.array([(c[0] + 0.5) * self.cell, (self.n_rows - c[1] - 0.5) * self.cell])

    def cell_of(self, xy):
        return (int(xy[0] // self.cell), int(self.n_rows - 1 - xy[1] // self.cell))

    def ring_cells(self):
        return [(i, j) for j in range(self.n_rows) for i in range(self.n_cols) if self.ring((i, j))]

    def cells(self, b, anchor):
        return [add(anchor, o) for o in self.shapes[b]]

    def block_center(self, b, anchor):
        '''World center [mm] of block b with its anchor in cell `anchor`.'''
        return np.mean([self.center(c) for c in self.cells(b, anchor)], axis=0)

    def anchor_offset(self, b):
        '''Anchor cell center minus block center [mm], to find the anchor from a pose.'''
        return self.center((0, 0)) - self.block_center(b, (0, 0))

    def anchor_of(self, b, xy):
        return self.cell_of(np.asarray(xy) + self.anchor_offset(b))

    def occupied(self, state, skip=None):
        '''Cells taken by the blocks of `state` (anchors), except block `skip`.'''
        return {c for b, a in enumerate(state) if b != skip for c in self.cells(b, a)}

    @staticmethod
    def world(d):
        '''Grid direction -> unit vector in the board frame (y up).'''
        return np.array([d[0], -d[1]], dtype=float)


#----------------------------------------------------------------------- high level

def add(a, b, k=1):
    return (a[0] + k * b[0], a[1] + k * b[1])


def behind_cells(grid, b, anchor, d):
    '''The cells just behind block b's back face (pushing along d).'''
    own = set(grid.cells(b, anchor))
    return [add(c, d, -1) for c in own if add(c, d, -1) not in own]


def access(grid, occ, b, anchor, d):
    '''How a vine can get behind block b to push it along d: "straight", "deflected" or None.'''
    blocked = lambda c: grid.wall(c) or c in occ
    kinds = set()
    for p in behind_cells(grid, b, anchor, d):
        if blocked(p):
            continue
        q = p
        while not grid.ring(q):
            q = add(q, d, -1)
            if blocked(q):
                break
        else:
            kinds.add("straight")
            continue
        for n in ((d[1], d[0]), (-d[1], -d[0])):           # the two perpendiculars
            if blocked(add(p, n)) and not blocked(add(add(p, d, -1), n, -1)):
                kinds.add("deflected")
    return "straight" if "straight" in kinds else "deflected" if kinds else None


def moves(grid, state):
    '''[(cost, block, d_name, k, kind, stopped)] from `state` (one anchor per block).'''
    out = []
    for b, anchor in enumerate(state):
        occ = grid.occupied(state, skip=b)
        free = lambda c: not grid.wall(c) and not grid.ring(c) and c not in occ
        for name, d in DIRS.items():
            kind = access(grid, occ, b, anchor, d)
            if kind is None:
                continue
            for k in range(1, max(grid.n_cols, grid.n_rows)):
                t = add(anchor, d, k)
                if not all(free(c) for c in grid.cells(b, t)):
                    break
                stopped = not all(free(add(c, d)) for c in grid.cells(b, t))
                out.append((1.0 + (0.5 if kind == "deflected" else 0.0) + (0.0 if stopped else 0.3) + 0.02 * k,
                            b, name, k, kind, stopped))
    return out


def moved(state, b, name, k):
    return tuple(add(a, DIRS[name], k) if i == b else a for i, a in enumerate(state))


def plan_cells(grid, start, banned=frozenset()):
    '''Dijkstra over the blocks' anchors until block 0 is at the goal. Returns
    [(state, block, d_name, k, kind, stopped)], or None. banned: {(state, block, d_name, k)}.'''
    best = {start: 0.0}
    heap = [(0.0, 0, start, [])]
    tie = 0
    while heap:
        cost, _, s, path = heapq.heappop(heap)
        if s[0] == grid.goal:
            return path
        if cost > best.get(s, math.inf):
            continue
        for mc, b, name, k, kind, stopped in moves(grid, s):
            if (s, b, name, k) in banned:
                continue
            t = moved(s, b, name, k)
            if cost + mc < best.get(t, math.inf):
                best[t] = cost + mc
                tie += 1
                heapq.heappush(heap, (cost + mc, tie, t, path + [(s, b, name, k, kind, stopped)]))
    return None


#----------------------------------------------------------------------- low level

def wrap_quarter(theta):
    '''A square's heading folded into (-45, 45] deg, in rad.'''
    return (theta + math.pi / 4) % (math.pi / 2) - math.pi / 4


def wrap_half(theta):
    '''A rectangle's heading folded into (-90, 90] deg, in rad.'''
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def obb(pose, hw, hh):
    return [pose[0], pose[1], pose[2], hw, hh]


class PushSearch:
    def __init__(self, e, sim, args):
        self.e, self.sim, self.args = e, sim, args
        self.grid = Grid(e)
        self.r = sim.cfg.vine_radius_m * 1000
        self.half = [(b["size"][0] / 2, b["size"][1] / 2) for b in e["boxes"]]       # (hw, hh) per block, mm
        self.square = [abs(hw - hh) < 1e-6 for hw, hh in self.half]
        from dvsim.vine import _obstacle_to_obb
        self.walls = np.array([[v * 1000 if k != 2 else v for k, v in enumerate(_obstacle_to_obb(w))]
                               for w in sim.scene.walls_m])                       # (cx, cy, theta, hw, hh), mm
        p = copy.copy(sim.device_params)
        p.grow_force = sim.device_params.grow_force * 1e-3     # growth off: the vine just holds still
        self.settle_params = p
        self.mu = sim.cfg.obj_friction
        # a push may leave a block turned at most this much: past it the block no longer fits
        # through a one-cell corridor ((w - 2 rho) cos a + (l - 2 rho) sin a + 2 rho > cell, w <= l its
        # sides, rho its corner radius)
        rho = e.get("corner_radius_mm", {}).get("boxes", 0.0)
        cell = self.grid.cell
        self.jam = []
        for hw, hh in self.half:
            w, l = 2 * min(hw, hh) - 2 * rho, 2 * max(hw, hh) - 2 * rho
            a = np.radians(np.arange(0, 45, 0.05))
            fits = w * np.cos(a) + l * np.sin(a) + 2 * rho <= cell
            self.jam.append(float(np.degrees(a[fits].max())) if fits.any() else 0.0)
        self.max_final_angle = [args.max_final_angle if args.max_final_angle is not None else j for j in self.jam]
        self.step_mm = sim.cfg.grow_rate_mps * 1000 * float(sim.params.dt)
        self.min_ray = 2 * sim.cfg.seg_len_m * 1000             # the starting stub is two segments
        self.rng = np.random.default_rng(args.seed) if args.seed is not None else None
        g = self.grid
        self.ring_mask = np.array([[g.ring((i, j)) for j in range(g.n_rows)] for i in range(g.n_cols)])

    def turn_deg(self, b, theta):
        '''|block b's heading| folded by its symmetry [deg] (numpy).'''
        t = np.asarray(theta, dtype=float)
        sym = math.pi / 2 if self.square[b] else math.pi
        return np.degrees(np.abs((t + sym / 2) % sym - sym / 2))

    # -- geometry

    def candidates(self, poses, state, b, d):
        '''Ring cell x heading pairs whose straight ray reaches behind block b: hits it moving along
        d, or first hits a wall (or another block) near a cell behind it.'''
        g, r = self.grid, self.r
        dw = Grid.world(d)
        n = np.array([-dw[1], dw[0]])
        box = poses[b]
        hw, hh = self.half[b]
        along = abs(dw[0]) * hw + abs(dw[1]) * hh                 # half extent along d (block at 0 deg)
        across = abs(n[0]) * hw + abs(n[1]) * hh
        occ = g.occupied(state, skip=b)
        behind = [g.center(c) for c in behind_cells(g, b, state[b], d) if not g.wall(c) and c not in occ]
        # the uniform headings, plus (args.aim) headings aimed from each ring cell at the cells behind
        # the block, points on its back face and points along a guide wall's face (the heading is
        # continuous: narrow gaps between wall corners need exact aim)
        offs = np.linspace(-(across - 25), across - 25, 3 if across < 80 else 5)
        aims = list(behind) + [np.array(box[:2]) - (along + r + 2) * dw + o * n for o in offs]
        for c in behind_cells(g, b, state[b], d):
            for side in (1, -1):
                nd = (int(round(side * n[0])), int(round(-side * n[1])))                 # grid direction
                if g.wall(add(c, nd)) or add(c, nd) in occ:
                    aims += [g.center(c) + side * (g.cell / 2 - r - 2) * n + o * dw for o in (-40, -20, 0, 20, 40)]
        bases, thetas = [], []
        for cell in g.ring_cells():
            bc = g.center(cell)
            if self.rng is None:
                hs = {round(2 * math.pi * k / g.n_headings, 4) for k in range(g.n_headings)}
            else:                                   # random, within 90 deg of the push direction
                th_d = math.atan2(dw[1], dw[0])
                hs = {round((th_d + x) % (2 * math.pi), 4)
                      for x in self.rng.uniform(-math.pi / 2, math.pi / 2, g.n_headings // 2 + 1)}
            if self.args.aim:
                for p in aims:
                    th = math.atan2(p[1] - bc[1], p[0] - bc[0])
                    jit = (-1.5, 0, 1.5) if self.rng is None else self.rng.uniform(-1.5, 1.5, 3)
                    hs |= {round((th + math.radians(j)) % (2 * math.pi), 4) for j in jit}
            for th in sorted(hs):
                bases.append(bc); thetas.append(th)
        bases, thetas = np.array(bases), np.array(thetas)
        u = np.stack([np.cos(thetas), np.sin(thetas)], 1)
        keep = u @ dw > -0.01
        bases, thetas, u = bases[keep], thetas[keep], u[keep]

        s = np.arange(0, 3000, 4.0)
        pts = bases[:, None, :] + s[None, :, None] * u[:, None, :]                     # (N, S, 2)
        x, y = pts[..., 0:1], pts[..., 1:2]
        W = np.concatenate([self.walls] + [np.array([obb(poses[o], *self.half[o])]) for o in range(len(poses))
                                           if o != b], 0)                              # walls and the other blocks
        c, sn = np.cos(W[:, 2]), np.sin(W[:, 2])
        lx = (x - W[:, 0]) * c + (y - W[:, 1]) * sn                                    # wall-local
        ly = -(x - W[:, 0]) * sn + (y - W[:, 1]) * c
        dx = np.maximum(np.abs(lx) - W[:, 3], 0)
        dy = np.maximum(np.abs(ly) - W[:, 4], 0)
        wall_hit = (np.hypot(dx, dy) < r).any(-1)                                        # (N, S)
        bx, by, bt = box
        lx = (x[..., 0] - bx) * math.cos(bt) + (y[..., 0] - by) * math.sin(bt)
        ly = -(x[..., 0] - bx) * math.sin(bt) + (y[..., 0] - by) * math.cos(bt)
        block_hit = np.hypot(np.maximum(np.abs(lx) - hw, 0), np.maximum(np.abs(ly) - hh, 0)) < r
        S = len(s)
        first_wall = np.where(wall_hit.any(1), wall_hit.argmax(1), S)
        first_block = np.where(block_hit.any(1), block_hit.argmax(1), S)
        out = []
        for i in range(len(bases)):
            if first_block[i] < first_wall[i]:
                # a vine whose starting stub would already touch the block (a ring cell next to it)
                # starts overlapping it, and the contact solve flings the block: not a push
                if s[first_block[i]] < self.min_ray:
                    continue
                if u[i] @ dw > math.cos(math.radians(55)):
                    out.append((0, -float(u[i] @ dw), i, s[first_block[i]]))
            elif first_wall[i] < S and behind:
                hit = pts[i, first_wall[i]]
                dist = min(np.linalg.norm(hit - p) for p in behind)
                # (a vine square to the push still counts: a slope can turn it into the push, as 5c_tri's does)
                if dist < 1.3 * g.cell and u[i] @ dw > -0.01:
                    out.append((1, float(dist), i, s[first_wall[i]]))
        out.sort()
        out = out[:self.args.max_candidates]
        return [(bases[i], thetas[i], ray, "block" if kind == 0 else "wall") for kind, _, i, ray in out]

    def limits(self, state, b):
        '''limit[i, j, a]: the farthest block b's center can get along axis direction a (0 +x, 1 -x,
        2 +y, 3 -y) with its anchor in cell (i, j) before it meets a wall or another block.'''
        g = self.grid
        occ = g.occupied(state, skip=b)
        hw, hh = self.half[b]
        limit = np.full((g.n_cols, g.n_rows, 4), np.inf)
        for j in range(g.n_rows):
            for i in range(g.n_cols):
                own = set(g.cells(b, (i, j)))
                for a, d in enumerate(((1, 0), (-1, 0), (0, -1), (0, 1))):             # grid steps for +x, -x, +y, -y
                    faces = []
                    for c in own:
                        q = c
                        while (not g.wall(add(q, d)) and add(q, d) not in occ) or add(q, d) in own:
                            q = add(q, d)
                            if not g.inside(q):
                                break
                        cq = g.center(q)
                        faces.append(cq[0] + g.cell / 2 - hw if a == 0 else -(cq[0] - g.cell / 2 + hw) if a == 1
                                     else cq[1] + g.cell / 2 - hh if a == 2 else -(cq[1] - g.cell / 2 + hh))
                    limit[i, j, a] = min(faces)
        return limit

    # -- simulation

    def start_states(self, cands, poses):
        sim = self.sim
        starts = []
        for base, th, _, _ in cands:
            s = sim.initial_state(base_m=(base[0] / 1000, base[1] / 1000, th))
            s.moveable_obj_pose = torch.tensor(np.asarray(poses)[None], dtype=s.moveable_obj_pose.dtype)
            s.moveable_obj_dstate = torch.zeros_like(s.moveable_obj_dstate)
            starts.append(s)
        return starts

    def rollout(self, params, starts, max_steps, stop):
        """Batched rollout on the device (batched_rollout.BatchRollout). stop(roll, t, buf) -> (rows,)
        bool on the device; buf is a dict of per-element device buffers it may keep (index N is a
        dummy slot for padding rows). Returns the finished rollout."""
        sim = self.sim
        design = sim.apply_control(starts[0], 0.0)                      # straight: no actuation
        roll = BatchRollout(sim, starts, [design] * len(starts), params=params)
        buf = {}
        for t in range(max_steps):
            if t % 10 == 0 and not roll.any_alive():
                break
            roll.step()
            roll.stop(stop(roll, t, buf) & roll.last_moved)
        return roll.finish()

    def touches_ring(self, P):
        '''(…, n_obj, 3) poses -> (…) bool: some block has a corner in a spawn-ring cell.'''
        g = self.grid
        hit = np.zeros(P.shape[:-2], dtype=bool)
        for o, (hw, hh) in enumerate(self.half):
            x, y, t = P[..., o, 0], P[..., o, 1], P[..., o, 2]
            c, s = np.cos(t), np.sin(t)
            for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
                cx, cy = x + c * lx - s * ly, y + s * lx + c * ly
                i = np.clip((cx // g.cell).astype(int), 0, g.n_cols - 1)
                j = np.clip((g.n_rows - 1 - cy // g.cell).astype(int), 0, g.n_rows - 1)
                hit |= self.ring_mask[i, j]
        return hit

    def push(self, poses, state, plan):
        """Search for one vine that does plan[0] (block b along d), or several of the plan's next moves
        of that block at once. The block must come to rest in a planned cell, no block may touch the
        spawn ring and the other blocks must stay put. Returns a dict (chosen push, its states, stats)
        or a dict with ok=False."""
        g, a = self.grid, self.args
        _, b, d_name, k, kind, stopped = plan[0]
        d = DIRS[d_name]
        targets = []                                         # the block's planned cells, in plan order
        for (_, bb, name, kk, _, _) in plan:
            if bb != b or name != d_name:            # one vine pushes one block along one direction
                break
            prev = targets[-1] if targets else state[b]
            targets.append(add(prev, DIRS[name], kk))
        tc = np.array([g.block_center(b, t) for t in targets])                      # (J, 2)
        cands = self.candidates(poses, state, b, d)
        info = dict(block=b, cell=list(state[b]), direction=d_name, k=k, planned_cell=list(targets[0]),
                    candidates=len(cands))
        if not cands:
            return dict(info, ok=False, why="no spawn cell and heading reaches behind the block")
        max_ray = max(ray for _, _, ray, _ in cands)
        dist = sum(abs(t[0] - s[0]) + abs(t[1] - s[1]) for s, t in zip([state[b]] + targets[:-1], targets))
        max_steps = min(int((max_ray + (dist + 1.5) * g.cell) / self.step_mm) + 40, a.max_grow_steps)
        N = len(cands)
        sym = math.pi / 2 if self.square[b] else math.pi

        def grow_stop(roll, t, buf):
            el = roll.element_ids()
            el = torch.where(el >= 0, el, N)
            if "tip" not in buf:
                buf["tip"] = torch.zeros(60, N + 1, 2, device=el.device)
            x = roll.x
            nb = x["bodies"].reshape(-1, 1) - 1
            tip = torch.cat([x["state"].gather(1, 3 * nb), x["state"].gather(1, 3 * nb + 1)], 1)
            then = buf["tip"][t % 60, el]
            buf["tip"][t % 60, el] = tip
            stalled = (t >= 60) & ((tip - then).norm(dim=1) < 6.0)
            ang = torch.remainder(x["obj_pose"][:, b, 2] + sym / 2, sym) - sym / 2
            return stalled | (ang.abs() > math.radians(35))

        t0 = time.time()
        grow = self.rollout(self.sim.device_params, self.start_states(cands, poses), max_steps, grow_stop)
        grow_s = time.time() - t0

        # every (candidate, step): block b's predicted resting pose (slides v^2 / 2 mu g on, until a
        # wall or block), its distance to each planned cell, and whether the push is still clean
        PA = grow.hist["obj_pose"].double().numpy()                                # (T, N, n_obj, 3)
        P = PA[:, :, b]
        V = grow.hist["obj_dstate"][:, :, b, :2].double().numpy()
        valid = grow.valid.numpy()
        speed = np.linalg.norm(V, axis=-1)
        pred = P[..., :2] + (speed ** 2 / (2 * self.mu * G_MM))[..., None] * V / (speed[..., None] + 1e-9)
        limit = self.limits(state, b)
        anc = P[..., :2] + g.anchor_offset(b)
        ci = np.clip((anc[..., 0] // g.cell).astype(int), 0, g.n_cols - 1)
        cj = np.clip((g.n_rows - 1 - anc[..., 1] // g.cell).astype(int), 0, g.n_rows - 1)
        lim = limit[ci, cj]                                                          # (T, N, 4)
        pred[..., 0] = np.minimum(np.maximum(pred[..., 0], -lim[..., 1]), lim[..., 0])
        pred[..., 1] = np.minimum(np.maximum(pred[..., 1], -lim[..., 3]), lim[..., 2])
        err_all = np.linalg.norm(pred[..., None, :] - tc, axis=-1)                   # (T, N, J)
        ang = self.turn_deg(b, P[..., 2])
        ring = np.logical_or.accumulate(self.touches_ring(PA), axis=0)               # touched so far
        others = [o for o in range(PA.shape[2]) if o != b]
        if others:
            moved_mm = np.linalg.norm(PA[:, :, others, :2] - PA[:1, :, others, :2], axis=-1).max(-1)
            moved_mm = np.maximum.accumulate(moved_mm, axis=0)
        else:
            moved_mm = np.zeros_like(ang)
        # the block must travel along the push axis: its center may not leave the axis line through
        # its start by more than LATERAL_MM (a push that slides it around a corner is not one push)
        n_ax = np.array([-Grid.world(d)[1], Grid.world(d)[0]])
        lateral = np.maximum.accumulate(np.abs((P[..., :2] - P[:1, :, :2]) @ n_ax), axis=0)     # (T, N)
        clean = (valid & ~ring & (moved_mm < OTHER_MOVED_MM) & (ang <= self.max_final_angle[b] + 5)
                 & (lateral < LATERAL_MM))
        J = len(targets)
        # score: the farthest planned cell first, then the error
        score = np.full(P.shape[:2], np.inf)
        for j in range(J):
            sj = np.where(clean & (err_all[..., j] < 1.5 * a.tol), (J - 1 - j) * 1000 + err_all[..., j], np.inf)
            score = np.minimum(score, sj)
        best_t = score.argmin(0)
        best = score[best_t, np.arange(N)]
        order = [i for i in np.argsort(best) if np.isfinite(best[i])][:a.settle_candidates]
        trials = [(i, int(best_t[i]) + dt) for i in order for dt in (0, -8, 8)
                  if 0 <= best_t[i] + dt < int(grow.length[i])]
        if not trials:
            # of the candidates that ever had the block headed for a planned cell, what spoiled each
            near = (err_all.min(-1) < 1.5 * a.tol) & valid
            got = near.any(0)
            why = "no vine moved the block toward a planned cell cleanly"
            if got.any():
                spoiled = {"touched ring": ring, "moved another block": moved_mm >= OTHER_MOVED_MM,
                           "turned": ang > self.max_final_angle[b] + 5, "drifted off axis": lateral >= LATERAL_MM}
                why += f" ({int(got.sum())}/{N} got near one; then " + ", ".join(
                    f"{nm} {int((near & m).any(0).sum())}" for nm, m in spoiled.items()) + ")"
            return dict(info, ok=False, why=why, grow_s=grow_s, steps=int(grow.length.max()))

        def settle_stop(roll, t, buf):
            el = roll.element_ids()
            el = torch.where(el >= 0, el, len(trials))
            if "still" not in buf:
                buf["still"] = torch.zeros(len(trials) + 1, dtype=torch.long, device=el.device)
            slow = roll.x["obj_dstate"][:, :, :2].norm(dim=2).amax(1) < 2.0
            n = torch.where(slow, buf["still"][el] + 1, 0)
            buf["still"][el] = n
            return n >= 5

        t0 = time.time()
        settle = self.rollout(self.settle_params, [grow.state_at(i, t) for i, t in trials], a.settle_steps,
                              settle_stop)
        settle_s = time.time() - t0

        results = []
        for jt, (i, t) in enumerate(trials):
            n = int(settle.length[jt])
            SA = settle.hist["obj_pose"][:n, jt].double().numpy() if n else PA[t:t + 1, i]      # (n, n_obj, 3)
            p = SA[-1, b]
            anchor = g.anchor_of(b, p[:2])
            j = targets.index(anchor) if anchor in targets else -1
            e_ = float(np.linalg.norm(p[:2] - tc[j])) if j >= 0 else math.inf
            angles = np.concatenate([ang[:t + 1, i], self.turn_deg(b, SA[:, b, 2])])
            ring_hit = bool(ring[t, i]) or bool(self.touches_ring(SA).any())
            other = float(max(moved_mm[t, i], np.linalg.norm(SA[:, others, :2] - PA[0, i, others, :2], axis=-1).max())
                          if others else 0.0)
            lat = max(float(lateral[t, i]), float(np.abs((SA[:, b, :2] - P[0, i, :2]) @ n_ax).max()))
            results.append(dict(i=i, t=t, jt=jt, j=j, cell=anchor, err=e_, final=SA[-1].tolist(), lateral=lat,
                                max_angle=float(angles.max()), final_angle=float(self.turn_deg(b, p[2])),
                                ring=ring_hit, other_moved=other))
        hits = [r for r in results if r["j"] >= 0 and r["err"] < a.tol and r["final_angle"] <= self.max_final_angle[b]
                and not r["ring"] and r["other_moved"] < OTHER_MOVED_MM and r["lateral"] < LATERAL_MM]
        info.update(grow_s=grow_s, settle_s=settle_s, steps=int(grow.length.max()), trials=len(trials),
                    reached=len({r["i"] for r in hits}),
                    reached_flagged=len({r["i"] for r in hits if r["max_angle"] > FLAG_DEG}),
                    reached_crooked=len({r["i"] for r in results if r["j"] >= 0 and r["err"] < a.tol
                                         and r["final_angle"] > self.max_final_angle[b]}),
                    touched_ring=len({r["i"] for r in results if r["ring"]}),
                    best_err_mm=min(r["err"] for r in results))
        if not hits:
            why = f"no settled pose within {a.tol:.0f} mm of a planned cell (closest {info['best_err_mm']:.0f} mm"
            if info["reached_crooked"]:
                why += f"; {info['reached_crooked']} got there turned > {self.max_final_angle[b]:.1f} deg"
            if info["touched_ring"]:
                why += f"; {info['touched_ring']} touched the spawn ring"
            return dict(info, ok=False, why=why + ")")
        hits.sort(key=lambda r: (-r["j"], r["max_angle"] > FLAG_DEG, r["err"] + 0.5 * r["max_angle"]))
        # the best hit, then up to --retries runners-up from other vines that leave the blocks somewhere else
        # (solve() falls back on these when every continuation from the best one dead-ends)
        picked = [hits[0]]
        for r in hits[1:]:
            if len(picked) > a.retries:
                break
            F = np.array(r["final"])
            if all(r["i"] != q["i"] and np.abs(F[:, :2] - np.array(q["final"])[:, :2]).max() > ALT_MM for q in picked):
                picked.append(r)

        def realize(best):
            base, th, ray, first = cands[best["i"]]
            new_state = tuple(best["cell"] if o == b else s for o, s in enumerate(state))
            return dict(info, ok=True, spawn_cell=list(g.cell_of(base)), heading_deg=round(math.degrees(th), 1),
                        first_contact=first, landed_cell=list(best["cell"]), moves_done=best["j"] + 1,
                        grow_steps=best["t"] + 1, err_mm=best["err"], final=best["final"],
                        max_angle_deg=best["max_angle"], final_angle_deg=best["final_angle"],
                        flagged=best["max_angle"] > FLAG_DEG, new_state=new_state,
                        states=grow.states(best["i"], upto=best["t"] + 1), settle_states=settle.states(best["jt"]))
        return dict(realize(picked[0]), alternatives=[realize(r) for r in picked[1:]])


#----------------------------------------------------------------------- top level

def fmt_state(state):
    return " ".join(f"{c}" for c in state) if len(state) > 1 else f"{state[0]}"


def fmt_move(b, name, k, kind, n_blocks):
    return (f"block {b} " if n_blocks > 1 else "") + f"{name} {k}" + ("*" if kind == "deflected" else "")


def solve(e, sim, args, log=print):
    '''Depth-first search with backtracking (see the module docstring). Returns the pushes of the
    first full sequence found (or of the deepest branch, if none), every push search tried, ...'''
    if args.n_headings:
        e["grid"]["n_headings"] = args.n_headings
    grid = Grid(e)
    search = PushSearch(e, sim, args)
    nb = len(grid.shapes)
    poses0 = sim.initial_state().moveable_obj_pose[0].numpy().astype(float)        # (n_obj, 3)
    root = dict(state=grid.start, poses=poses0, push=None, via=None, alts=[], banned=frozenset())
    stack, banned, attempts, first_plan, deepest = [root], set(), [], None, [root]
    t0 = time.time()
    while stack and len(attempts) < args.max_attempts:
        node = stack[-1]
        if node["state"][0] == grid.goal:
            break
        plan = plan_cells(grid, node["state"], frozenset(banned))
        if first_plan is None:
            first_plan = plan
        if not plan:
            log(f"  {'  ' * (len(stack) - 1)}dead end at {fmt_state(node['state'])}: no plan left, back up")
            stack.pop()
            if stack and node["alts"]:
                # the same move by another vine leaves the blocks elsewhere: retry from there, forgetting the
                # bans made below this node (they held for the old poses)
                alt = node["alts"][0]
                log(f"  {'  ' * (len(stack) - 1)}retry {fmt_move(*node['via'], alt['kind'], nb)} "
                    f"with another vine: spawn {alt['spawn_cell']} at {alt['heading_deg']:.0f} deg, {alt['err_mm']:.0f} mm off")
                banned = set(node["banned"])
                stack.append(dict(node, poses=np.array(alt["final"]), push=alt, alts=node["alts"][1:]))
                continue
            if stack:
                banned.add((stack[-1]["state"], *node["via"]))
            continue
        log(f"  {'  ' * (len(stack) - 1)}plan from {fmt_state(node['state'])}: "
            + ", ".join(fmt_move(b, n, k, kind, nb) for _, b, n, k, kind, _ in plan))
        s, b, name, k, kind, stopped = plan[0]
        res = search.push(node["poses"], node["state"], plan)
        alts = res.pop("alternatives", [])
        for r in [res] + alts:
            r.update(kind=kind, wall_stopped=stopped, depth=len(stack) - 1)
        attempts.append({key: v for key, v in res.items() if key not in ("states", "settle_states", "new_state")})
        pad = "  " * (len(stack) - 1)
        if not res["ok"]:
            log(f"  {pad}push {fmt_move(b, name, k, kind, nb)} from {s[b]} FAILED: {res['why']}")
            # the search scored every outcome of pushing this block this way, whatever k: ban them all
            banned.update((s, b, name, kk) for kk in range(1, max(grid.n_cols, grid.n_rows)))
            continue
        if any(n["state"] == res["new_state"] for n in stack):      # back where the branch has been
            log(f"  {pad}push {fmt_move(b, name, k, kind, nb)} from {s[b]} loops back to {fmt_state(res['new_state'])}")
            banned.add((s, b, name, k))
            continue
        flag = "  FLAG: block turned %.0f deg" % res["max_angle_deg"] if res["flagged"] else ""
        log(f"  {pad}push {fmt_move(b, name, k, kind, nb)} from {s[b]} -> {tuple(res['landed_cell'])} "
            f"({res['moves_done']} planned move{'s' if res['moves_done'] > 1 else ''}): spawn {res['spawn_cell']} at "
            f"{res['heading_deg']:.0f} deg, {res['grow_steps']} steps, {res['err_mm']:.0f} mm off, max turn "
            f"{res['max_angle_deg']:.1f} deg; {res['reached']}/{res['candidates']} candidates reached "
            f"({res['reached_flagged']} flagged), {res['grow_s'] + res['settle_s']:.0f} s{flag}")
        stack.append(dict(state=res["new_state"], poses=np.array(res["final"]), push=res, via=(b, name, k),
                          alts=[r for r in alts if r["new_state"] == res["new_state"]], banned=frozenset(banned)))
        if len(stack) > len(deepest):
            deepest = list(stack)
    solved = bool(stack) and stack[-1]["state"][0] == grid.goal
    branch = stack if solved else deepest
    pushes = [n["push"] for n in branch[1:]]
    return dict(solved=solved, pushes=pushes, attempts=attempts, first_plan=first_plan,
                seconds=time.time() - t0, final=branch[-1]["poses"].tolist(),
                backtracks=sum(1 for _ in attempts) - len(pushes))


#----------------------------------------------------------------------- drawing

def draw_board(ax, e, grid):
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
    cell = grid.cell
    for j, r in enumerate(grid.rows):
        for i, ch in enumerate(r):
            x, y = i * cell, (grid.n_rows - j - 1) * cell
            if ch == "o":
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="#dceaf9", edgecolor="none", zorder=0))
            ax.add_patch(Rectangle((x, y), cell, cell, fill=False, edgecolor="#d6d5cf", lw=0.6, zorder=1))
    for w in e["walls"]:
        if "rect" in w and "part" not in w:
            x0, y0, x1, y1 = w["rect"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=viz.WALL_FACE, edgecolor=viz.INK_2,
                                   lw=1.2, hatch="///", zorder=2))
    for tri in e.get("triangles", []):                     # drawn as the triangle; simulated as its OBBs
        ax.add_patch(Polygon(tri, closed=True, facecolor=viz.WALL_FACE, edgecolor=viz.INK_2, lw=1.2,
                             hatch="///", zorder=2))
    # the goal: block 0's outline in the goal cell (the pose it must end in, within --tol)
    gx, gy = e["goal"]["center"]
    w, h = e["boxes"][0]["size"]
    rb = e.get("corner_radius_mm", {}).get("boxes", 0.0)
    ax.add_patch(FancyBboxPatch((gx - w / 2, gy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rb}",
                                fill=False, edgecolor=viz.GOAL, lw=2, ls="--", zorder=3))
    ax.set_xlim(0, grid.n_cols * cell); ax.set_ylim(0, grid.n_rows * cell)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def new_fig(grid, scale=0.55):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(scale * grid.n_cols + 0.4, scale * grid.n_rows + 0.7), facecolor=viz.SURFACE,
                           layout="constrained")
    ax.set_facecolor(viz.SURFACE)
    return fig, ax


def draw_plan(e, grid, plan, path, title):
    from matplotlib.patches import Rectangle
    fig, ax = new_fig(grid)
    draw_board(ax, e, grid)
    if plan:
        state = plan[0][0]
        for b, a in enumerate(state):
            for c in grid.cells(b, a):
                ax.add_patch(Rectangle(grid.center(c) - 49, 98, 98, facecolor=viz.BOX if b == 0 else "#d6a02a",
                                       edgecolor=viz.SURFACE, zorder=4))
        for n, (s, b, name, k, kind, stopped) in enumerate(plan, 1):
            a, t = grid.block_center(b, s[b]), grid.block_center(b, add(s[b], DIRS[name], k))
            ax.annotate("", t, a, arrowprops=dict(arrowstyle="-|>", color=viz.BEST, lw=2,
                                                  ls="-" if kind == "straight" else "--"), zorder=6)
            mid = (a + t) / 2
            ax.annotate(str(n), mid, xytext=(6, 6), textcoords="offset points", color=viz.BEST, fontsize=10,
                        weight="bold", zorder=7)
            for c in behind_cells(grid, b, s[b], DIRS[name]):
                ax.plot(*grid.center(c), marker="o", color=viz.BEST, ms=5, alpha=0.6, zorder=6)
    ax.set_title(title, fontsize=8, loc="left", color=viz.INK)
    viz.save_figure(fig, path, dpi=110)


def draw_frame(e, grid, sim, state, title, path, trace=None):
    fig, ax = new_fig(grid)
    draw_board(ax, e, grid)
    viz.draw_objects(ax, sim, state)
    viz.draw_vine(ax, sim, state)
    if trace:
        ax.plot([p[0] for p in trace], [p[1] for p in trace], "-", color=viz.BOX, lw=1, alpha=0.7, zorder=3)
    ax.set_title(title, fontsize=8, loc="left", color=viz.INK)
    viz.save_figure(fig, path, dpi=80)


def draw_pushes(e, grid, sim, pushes, path, title):
    fig, ax = new_fig(grid)
    draw_board(ax, e, grid)
    colors = ["#2a78d6", "#6250d6", "#1baf7a", "#d6a02a", "#d62a78", "#2ab8d6", "#898781"]
    for n, p in enumerate(pushes):
        s = p["settle_states"][-1] if p["settle_states"] else p["states"][-1]
        viz.draw_vine(ax, sim, s, color=colors[n % len(colors)], alpha=0.25)
        bx, by = s.moveable_obj_pose[0, p["block"], :2].tolist()
        viz.draw_objects(ax, sim, s, outline_only=True)
        ax.annotate(str(n + 1), (bx, by), ha="center", va="center", fontsize=9, weight="bold",
                    color=colors[n % len(colors)], zorder=8)
    ax.set_title(title, fontsize=8, loc="left", color=viz.INK)
    viz.save_figure(fig, path, dpi=110)


def render_solution(e, grid, sim, result, out, every=4):
    frame_dir = os.path.join(out, "_frames")
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir)
    frames, trace = [], []
    n = len(result["pushes"])
    status = "SOLVED" if result["solved"] else "NOT SOLVED"
    nb = len(grid.shapes)
    for m, p in enumerate(result["pushes"], 1):
        b = p["block"]
        seq = [(s, "growing") for s in p["states"]] + [(s, "stopped, block settling") for s in p["settle_states"]]
        picked = list(range(0, len(seq), every)) + [len(seq) - 1]
        for idx in sorted(set(picked)):
            s, phase = seq[idx]
            if b == 0:
                trace.append(s.moveable_obj_pose[0, 0, :2].tolist())
            th = float(s.moveable_obj_pose[0, b, 2])
            ang = math.degrees(wrap_quarter(th) if abs(e["boxes"][b]["size"][0] - e["boxes"][b]["size"][1]) < 1e-6
                               else wrap_half(th))
            flag = "  >15 deg!" if abs(ang) > FLAG_DEG else ""
            title = (f"{e['name']} {status} | push {m}/{n}: {fmt_move(b, p['direction'], p['k'], p['kind'], nb)} from "
                     f"cell {tuple(p['cell'])}, spawn {tuple(p['spawn_cell'])} at {p['heading_deg']:.0f} deg\n"
                     f"{phase} | block turned {ang:+.1f} deg{flag}")
            path = os.path.join(frame_dir, f"f{len(frames):05d}.png")
            draw_frame(e, grid, sim, s, title, path, trace)
            frames.append(path)
    if not frames:
        return
    viz.write_gif(frames, os.path.join(out, "solution.gif"), frame_ms=int(1000 * every * float(sim.params.dt)),
                  hold_last_ms=int(1000 * run_envs.HOLD_LAST_S))
    run_envs.gif_to_mp4(os.path.join(out, "solution.gif"), os.path.join(out, "solution.mp4"),
                        fps=round(1 / (every * float(sim.params.dt))))
    shutil.rmtree(frame_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-fraction", type=float, default=0.7)
    parser.add_argument("--tol", type=float, default=25.0, help="settled block center to target cell center [mm]")
    parser.add_argument("--max-candidates", type=int, default=256, help="vines simulated per push")
    parser.add_argument("--settle-candidates", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--max-grow-steps", type=int, default=700, help="cap on one vine's growth steps")
    parser.add_argument("--max-attempts", type=int, default=40, help="low-level push searches per env")
    parser.add_argument("--retries", type=int, default=3,
                        help="fallback vines kept per push: when everything after a push dead-ends, the same move is "
                             "retried from where another vine left the blocks before the move is banned")
    parser.add_argument("--max-final-angle", type=float, default=None,
                        help="[deg] a push may leave the block turned at most this (default: the jam angle)")
    parser.add_argument("--no-aim", dest="aim", action="store_false",
                        help="only the env's uniform headings (no headings aimed behind the block)")
    parser.add_argument("--n-headings", type=int, default=None, help="override the env's headings per spawn cell")
    parser.add_argument("--seed", type=int, default=None,
                        help="random headings (180 deg facing the push) instead of the uniform ones; default: deterministic")
    parser.add_argument("--checkpoint", default=None, help="sim parameters from checkpoints/<name>.json (default run_envs.DEFAULT_SIM)")
    parser.add_argument("--box-mass-scale", type=float, default=1.0, help="multiply every block's mass (a stand-in for the real puzzle block)")
    parser.add_argument("--obj-friction", type=float, default=None, help="block-table friction (overrides the checkpoint's)")
    parser.add_argument("--out", default="out/solutions", help="files go to <out>/<kind>/<env>.*, replaced on every run")
    parser.add_argument("--no-render", dest="render", action="store_false", help="skip the solution gif and video (for timing runs)")
    args = parser.parse_args()

    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        if args.checkpoint:
            checkpoint.apply(cfg, args.checkpoint)
        if args.obj_friction is not None:
            cfg.obj_friction = args.obj_friction
        scene.masses_kg = [m * args.box_mass_scale for m in scene.masses_kg]
        cfg.device, cfg.qp_backend = args.device, "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
        sim = Sim(scene, cfg)
        if args.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        grid = Grid(e)
        name = e["name"] + (f"_h{args.n_headings}" if args.n_headings else "") + ("" if args.aim else "_noaim")
        out = outputs.work_dir(args.out, name)                 # published to <out>/<kind>/<name>.* at the end
        print(f"=== {e['name']}", flush=True)
        result = solve(e, sim, args, log=lambda m: print(m, flush=True))
        fp = result["first_plan"]
        nb = len(grid.shapes)
        draw_plan(e, grid, fp, os.path.join(out, "plan.png"),
                  f"{e['name']} first high-level plan ({len(fp or [])} pushes; dashed = deflected vine, "
                  f"dot = cell the vine must reach)")
        flagged = sum(p["flagged"] for p in result["pushes"])
        title = (f"{e['name']} {'SOLVED' if result['solved'] else 'NOT SOLVED'}: {len(result['pushes'])} pushes, "
                 f"{len(result['attempts'])} push searches ({result['backtracks']} failed or backed out), "
                 f"{flagged} flagged (>15 deg), {result['seconds'] / 60:.1f} min")
        draw_pushes(e, grid, sim, result["pushes"], os.path.join(out, "pushes.png"), title)
        if args.render:
            render_solution(e, grid, sim, result, out)
        summary = dict(env=e["name"], solved=result["solved"], seconds=result["seconds"], final=result["final"],
                       first_plan=[[[list(c) for c in s], b, n, k, kind, st] for s, b, n, k, kind, st in (fp or [])],
                       pushes=[{k: v for k, v in p.items() if k not in ("states", "settle_states", "new_state")}
                               for p in result["pushes"]],
                       attempts=result["attempts"], checkpoint=args.checkpoint, args=vars(args))
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(summary, f, indent=1)
        outputs.publish(out, args.out, name)
        run_envs.write_summary(args.out)
        print(title, flush=True)


if __name__ == "__main__":
    main()
