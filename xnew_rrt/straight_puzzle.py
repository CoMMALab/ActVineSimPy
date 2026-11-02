'''
block_puzzle.py with only straight pushes: the vine launches from the spawn-ring cell in line with the block
along the push axis and grows exactly along the push (no diagonal or wall-deflected vines). The launch point
may sit off the cell center across the push (the block's actual center line, +-OFFSETS_MM), staying inside
the cell, since an earlier push leaves the block a few mm off its cell center. Same arguments as block_puzzle.py.

    python xnew_rrt/straight_puzzle.py envs/5b.json --checkpoint checkpoints/spin8_manual.json --out out/straight
'''
import math

import numpy as np

import block_puzzle
from block_puzzle import Grid, PushSearch, obb

OFFSETS_MM = (-20, -10, -5, 0, 5, 10, 20)        # launch point across the push, from the block's center line


def obb_dist(p, W):
    '''Distance [mm] from points p (S, 2) to boxes W (K, 5: cx, cy, theta, hw, hh) -> (S, K).'''
    c, s = np.cos(W[:, 2]), np.sin(W[:, 2])
    x, y = p[:, :1] - W[:, 0], p[:, 1:] - W[:, 1]
    lx, ly = x * c + y * s, -x * s + y * c
    return np.hypot(np.maximum(np.abs(lx) - W[:, 3], 0), np.maximum(np.abs(ly) - W[:, 4], 0))


def straight_candidates(self, poses, state, b, d):
    g = self.grid
    dw = np.array(Grid.world(d), dtype=float)
    n = np.array([-dw[1], dw[0]])
    planned = g.block_center(b, state[b])
    actual = np.array(poses[b][:2])
    th = math.atan2(dw[1], dw[0])
    others = np.array(list(self.walls) + [obb(poses[o], *self.half[o]) for o in range(len(poses)) if o != b])
    target = np.array([obb(poses[b], *self.half[b])])
    s = np.arange(0, 3000, 2.0)
    out = []
    for cell in g.ring_cells():
        bc = np.array(g.center(cell))
        if abs((bc - planned) @ n) > 1.0 or (planned - bc) @ dw <= 0:      # in line with the block, behind it
            continue
        for o in OFFSETS_MM:
            off = (actual - bc) @ n + o
            if abs(off) > g.cell / 2 - self.r:                             # the launch point stays in the cell
                continue
            base = bc + off * n
            pts = base + s[:, None] * dw
            hit_b = np.flatnonzero(obb_dist(pts, target)[:, 0] < self.r)
            hit_w = np.flatnonzero((obb_dist(pts, others) < self.r).any(1))
            if len(hit_b) and (not len(hit_w) or hit_b[0] < hit_w[0]) and s[hit_b[0]] >= self.min_ray:
                out.append((base, th, s[hit_b[0]], "block"))
    return out


PushSearch.candidates = straight_candidates

if __name__ == "__main__":
    block_puzzle.main()
