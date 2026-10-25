'''
Hierarchical planner for the block puzzles (envs/5*.json, written by envs/make_puzzles.py).

High level: a discrete planner over the block's grid cell. A move pushes the block k cells along
one axis. It needs the k cells ahead free, and room for a vine behind the block: the cell behind
it free and either
    straight    a free straight corridor from there back to the spawn ring, or
    deflected   a wall beside that cell for the vine to slide along, and the cell diagonally
                behind it free, so a diagonal vine can hit the wall and turn into the push
                (5c needs these: a stub wall blocks the straight approach)
Dijkstra on (pushes, plus a penalty for deflected moves and for moves that no wall stops).

Low level: realizes one move with the batched sim (dvsim/batched_qp.py on the GPU). Candidates
are every spawn-ring cell x heading (the env's N uniform headings plus headings aimed at the cell
behind the block, its back face and the guide wall; --no-aim drops these) whose straight ray hits
the block from behind, or hits a wall next to the cell behind it; all of them grow straight (no actuation) in one batched
rollout. For each, the stop time whose predicted resting pose (the block slides on after the
vine stops: v^2 / 2 mu g, cut short by walls) is nearest the target cell is settled in a second
batched run with growth switched off. The best settled pose within --tol of the target cell's
center is taken; the vine is retracted (removed) and the next move starts from that pose. A
move with no such candidate is banned and the high level replans.

A push whose block turns more than 15 deg (FLAG_DEG) on the way is flagged. One that leaves it
turned more than the jam angle is not accepted: past it the block no longer fits through a
one-cell corridor (7.5 deg for a 98 mm block in 110 mm cells, 9.8 deg with 10 mm rounded corners;
--max-final-angle overrides).

Run from the repo root:
    python xnew_rrt/block_puzzle.py envs/5a.json envs/5b.json envs/5c.json
Per env, under out/puzzle/<name>/: plan.png (the first high-level plan), pushes.png (what was
executed), solution.gif/.mp4, result.json.
'''

import argparse, copy, heapq, json, math, os, time

import numpy as np
import torch

from scene import Sim  # (puts the repo root on sys.path)
from batched_rollout import BatchRollout
import checkpoint
import run_envs
import viz

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}   # grid (column right, row down)
FLAG_DEG = 15.0
G_MM = 9810.0


#----------------------------------------------------------------------- grid

class Grid:
    def __init__(self, e):
        g = e["grid"]
        self.rows, self.cell = g["rows"], g["cell_mm"]
        self.n_cols, self.n_rows = len(self.rows[0]), len(self.rows)
        self.goal = tuple(g["goal"])
        self.n_headings = g["n_headings"]

    def inside(self, c):
        return 0 <= c[0] < self.n_cols and 0 <= c[1] < self.n_rows

    def wall(self, c):
        return not self.inside(c) or self.rows[c[1]][c[0]] == "#"

    def ring(self, c):
        return self.inside(c) and self.rows[c[1]][c[0]] == "o"

    def center(self, c):
        return np.array([(c[0] + 0.5) * self.cell, (self.n_rows - c[1] - 0.5) * self.cell])

    def cell_of(self, xy):
        return (int(xy[0] // self.cell), int(self.n_rows - 1 - xy[1] // self.cell))

    def ring_cells(self):
        return [(i, j) for j in range(self.n_rows) for i in range(self.n_cols) if self.ring((i, j))]

    @staticmethod
    def world(d):
        '''Grid direction -> unit vector in the board frame (y up).'''
        return np.array([d[0], -d[1]], dtype=float)


#----------------------------------------------------------------------- high level

def add(a, b, k=1):
    return (a[0] + k * b[0], a[1] + k * b[1])


def access(grid, c, d):
    '''How a vine can get behind block cell c to push it along d: "straight", "deflected" or None.'''
    p = add(c, d, -1)
    if grid.wall(p):
        return None
    q = p
    while not grid.ring(q):
        q = add(q, d, -1)
        if grid.wall(q):
            break
    else:
        return "straight"
    for n in ((d[1], d[0]), (-d[1], -d[0])):           # the two perpendiculars
        if grid.wall(add(p, n)) and not grid.wall(add(add(p, d, -1), n, -1)):
            return "deflected"
    return None


def moves(grid, c):
    '''[(cost, d_name, k, kind, stopped_by_wall)] from block cell c.'''
    out = []
    for name, d in DIRS.items():
        kind = access(grid, c, d)
        if kind is None:
            continue
        for k in range(1, max(grid.n_cols, grid.n_rows)):
            t = add(c, d, k)
            if grid.wall(t) or grid.ring(t):
                break
            stopped = grid.wall(add(t, d))
            out.append((1.0 + (0.5 if kind == "deflected" else 0.0) + (0.0 if stopped else 0.3) + 0.02 * k,
                        name, k, kind, stopped))
    return out


def plan_cells(grid, start, banned=frozenset()):
    '''Dijkstra over block cells. Returns [(cell, d_name, k, kind, stopped)] to the goal, or None.'''
    best = {start: 0.0}
    heap = [(0.0, start, [])]
    while heap:
        cost, c, path = heapq.heappop(heap)
        if c == grid.goal:
            return path
        if cost > best.get(c, math.inf):
            continue
        for mc, name, k, kind, stopped in moves(grid, c):
            if (c, name, k) in banned:
                continue
            t = add(c, DIRS[name], k)
            if cost + mc < best.get(t, math.inf):
                best[t] = cost + mc
                heapq.heappush(heap, (cost + mc, t, path + [(c, name, k, kind, stopped)]))
    return None


#----------------------------------------------------------------------- low level

def wrap_quarter(theta):
    '''A square's heading folded into (-45, 45] deg, in rad.'''
    return (theta + math.pi / 4) % (math.pi / 2) - math.pi / 4


class PushSearch:
    def __init__(self, e, sim, args):
        self.e, self.sim, self.args = e, sim, args
        self.grid = Grid(e)
        self.r = sim.cfg.vine_radius_m * 1000
        self.half = e["boxes"][0]["size"][0] / 2
        self.walls = np.array([[v * 1000 for v in w] for w in sim.scene.walls_m])       # all AABBs, mm
        p = copy.copy(sim.device_params)
        p.grow_force = sim.device_params.grow_force * 1e-3     # growth off: the vine just holds still
        self.settle_params = p
        self.mu = sim.cfg.obj_friction
        # a push may leave the block turned at most this much: past it the block no longer fits
        # through a one-cell corridor ((size - 2 rho) (cos a + sin a) + 2 rho > cell, rho = its corner radius)
        rho = e.get("corner_radius_mm", {}).get("boxes", 0.0)
        jam = math.degrees(math.asin((self.grid.cell - 2 * rho) / ((2 * self.half - 2 * rho) * math.sqrt(2)))) - 45
        self.max_final_angle = args.max_final_angle if args.max_final_angle is not None else jam
        self.step_mm = sim.cfg.grow_rate_mps * 1000 * float(sim.params.dt)

    # -- geometry

    def candidates(self, box, c, d):
        '''Ring cell x heading pairs whose straight ray reaches behind the block: hits the block
        moving along d, or first hits a wall near the cell behind it.'''
        g, r = self.grid, self.r
        dw = Grid.world(d)
        behind = g.center(add(c, d, -1))
        # the uniform headings, plus (args.aim) headings aimed from each ring cell at the cell behind
        # the block, points on the block's back face and points along a guide wall's face (the heading
        # is continuous: narrow gaps between wall corners need exact aim)
        n = np.array([-dw[1], dw[0]])
        aims = [behind] + [np.array(box[:2]) - (self.half + r + 2) * dw + o * n for o in (-25, 0, 25)]
        for side in (1, -1):
            nd = (int(round(side * n[0])), int(round(-side * n[1])))                     # grid direction
            if g.wall(add(add(c, d, -1), nd)):
                aims += [behind + side * (g.cell / 2 - r - 2) * n + o * dw for o in (-40, -20, 0, 20, 40)]
        bases, thetas = [], []
        for cell in g.ring_cells():
            b = g.center(cell)
            hs = {round(2 * math.pi * k / g.n_headings, 4) for k in range(g.n_headings)}
            if self.args.aim:
                for p in aims:
                    th = math.atan2(p[1] - b[1], p[0] - b[0])
                    hs |= {round((th + math.radians(j)) % (2 * math.pi), 4) for j in (-1.5, 0, 1.5)}
            for th in sorted(hs):
                bases.append(b); thetas.append(th)
        bases, thetas = np.array(bases), np.array(thetas)
        u = np.stack([np.cos(thetas), np.sin(thetas)], 1)
        keep = u @ dw > 0.2
        bases, thetas, u = bases[keep], thetas[keep], u[keep]

        s = np.arange(0, 3000, 4.0)
        pts = bases[:, None, :] + s[None, :, None] * u[:, None, :]                     # (N, S, 2)
        x, y = pts[..., 0:1], pts[..., 1:2]
        W = self.walls
        dx = np.maximum(np.maximum(W[:, 0] - x, 0), x - W[:, 2])
        dy = np.maximum(np.maximum(W[:, 1] - y, 0), y - W[:, 3])
        wall_hit = (np.hypot(dx, dy) < r).any(-1)                                        # (N, S)
        bx, by, bt = box
        lx = (x[..., 0] - bx) * math.cos(bt) + (y[..., 0] - by) * math.sin(bt)
        ly = -(x[..., 0] - bx) * math.sin(bt) + (y[..., 0] - by) * math.cos(bt)
        block_hit = np.hypot(np.maximum(np.abs(lx) - self.half, 0), np.maximum(np.abs(ly) - self.half, 0)) < r
        S = len(s)
        first_wall = np.where(wall_hit.any(1), wall_hit.argmax(1), S)
        first_block = np.where(block_hit.any(1), block_hit.argmax(1), S)
        out = []
        for i in range(len(bases)):
            if first_block[i] < first_wall[i]:
                if u[i] @ dw > math.cos(math.radians(55)):
                    out.append((0, -float(u[i] @ dw), i, s[first_block[i]]))
            elif first_wall[i] < S:
                hit = pts[i, first_wall[i]]
                if np.linalg.norm(hit - behind) < 1.0 * g.cell and u[i] @ dw > 0.3:
                    out.append((1, float(np.linalg.norm(hit - behind)), i, s[first_wall[i]]))
        out.sort()
        out = out[:self.args.max_candidates]
        return [(bases[i], thetas[i], ray, "block" if kind == 0 else "wall") for kind, _, i, ray in out]

    # -- simulation

    def start_states(self, cands, box):
        sim = self.sim
        starts = []
        for base, th, _, _ in cands:
            s = sim.initial_state(base_m=(base[0] / 1000, base[1] / 1000, th))
            s.moveable_obj_pose = torch.tensor([[[box[0], box[1], box[2]]]], dtype=s.moveable_obj_pose.dtype)
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

    def tables(self, banned):
        """Cost to go from every block cell (Dijkstra over reversed moves), and for every cell and
        axis direction the farthest the block's center can get before a wall stops it."""
        g = self.grid
        rev = {}
        for j in range(g.n_rows):
            for i in range(g.n_cols):
                c = (i, j)
                if g.wall(c) or g.ring(c):
                    continue
                for mc, name, k, kind, stopped in moves(g, c):
                    if (c, name, k) not in banned:
                        rev.setdefault(add(c, DIRS[name], k), []).append((c, mc))
        ctg = np.full((g.n_cols, g.n_rows), np.inf)
        ctg[g.goal] = 0.0
        heap = [(0.0, g.goal)]
        while heap:
            d0, c = heapq.heappop(heap)
            if d0 > ctg[c]:
                continue
            for p, mc in rev.get(c, []):
                if d0 + mc < ctg[p]:
                    ctg[p] = d0 + mc
                    heapq.heappush(heap, (d0 + mc, p))
        # limit[i, j, a]: the block center's farthest coordinate along axis direction a
        # (0 +x, 1 -x, 2 +y, 3 -y) from cell (i, j) before it meets a wall
        limit = np.full((g.n_cols, g.n_rows, 4), np.inf)
        slack = g.cell / 2 - self.half
        for j in range(g.n_rows):
            for i in range(g.n_cols):
                for a, d in enumerate(((1, 0), (-1, 0), (0, -1), (0, 1))):     # grid steps for +x, -x, +y, -y
                    q = (i, j)
                    while not g.wall(add(q, d)):
                        q = add(q, d)
                    cq = g.center(q)
                    limit[i, j, a] = (cq[0] + slack if a == 0 else -(cq[0] - slack) if a == 1 else
                                      cq[1] + slack if a == 2 else -(cq[1] - slack))
        return ctg, limit

    def push(self, box, plan, banned):
        """Search for one vine that moves the block from plan[0]'s cell as far along to the goal as it
        can: every outcome is scored by its cell's cost to go, so a vine that does several planned
        moves (or a k-cell push) at once wins over one that does only the first.
        Returns a dict (chosen push, its states, stats) or a dict with ok=False."""
        g, a = self.grid, self.args
        c, d_name, k, kind, stopped = plan[0]
        d = DIRS[d_name]
        ctg, limit = self.tables(banned)
        here = ctg[c]
        cands = self.candidates(box, c, d)
        info = dict(cell=list(c), direction=d_name, k=k, planned_cell=list(add(c, d, k)), candidates=len(cands))
        if not cands:
            return dict(info, ok=False, why="no spawn cell and heading reaches behind the block")
        cells_left = sum(m[2] for m in plan)
        max_ray = max(ray for _, _, ray, _ in cands)
        max_steps = min(int((max_ray + (cells_left + 1.5) * g.cell) / self.step_mm) + 40, a.max_grow_steps)
        N = len(cands)

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
            ang = torch.remainder(x["obj_pose"][:, 0, 2] + math.pi / 4, math.pi / 2) - math.pi / 4
            return stalled | (ang.abs() > math.radians(35))

        t0 = time.time()
        grow = self.rollout(self.sim.device_params, self.start_states(cands, box), max_steps, grow_stop)
        grow_s = time.time() - t0

        # every (candidate, step): predicted resting pose (slides v^2 / 2 mu g on, until a wall),
        # the cell it lands in, its cost to go and its distance from that cell's center
        P = grow.hist["obj_pose"][:, :, 0].double().numpy()                       # (T, N, 3)
        V = grow.hist["obj_dstate"][:, :, 0, :2].double().numpy()
        valid = grow.valid.numpy()
        speed = np.linalg.norm(V, axis=-1)
        pred = P[..., :2] + (speed ** 2 / (2 * self.mu * G_MM))[..., None] * V / (speed[..., None] + 1e-9)
        ci = np.clip((P[..., 0] // g.cell).astype(int), 0, g.n_cols - 1)
        cj = np.clip((g.n_rows - 1 - P[..., 1] // g.cell).astype(int), 0, g.n_rows - 1)
        lim = limit[ci, cj]                                                          # (T, N, 4)
        pred[..., 0] = np.minimum(np.maximum(pred[..., 0], -lim[..., 1]), lim[..., 0])
        pred[..., 1] = np.minimum(np.maximum(pred[..., 1], -lim[..., 3]), lim[..., 2])
        pi_ = np.clip((pred[..., 0] // g.cell).astype(int), 0, g.n_cols - 1)
        pj = np.clip((g.n_rows - 1 - pred[..., 1] // g.cell).astype(int), 0, g.n_rows - 1)
        to_go = ctg[pi_, pj]
        cx, cy = (pi_ + 0.5) * g.cell, (g.n_rows - pj - 0.5) * g.cell
        err = np.hypot(pred[..., 0] - cx, pred[..., 1] - cy)
        ang = np.degrees(np.abs((P[..., 2] + math.pi / 4) % (math.pi / 2) - math.pi / 4))
        good = valid & (to_go < here - 1e-9) & (err < 1.5 * a.tol) & (ang <= self.max_final_angle + 5)
        score = np.where(good, to_go * 1000 + err, np.inf)                           # nearest the goal first
        best_t = score.argmin(0)
        best = score[best_t, np.arange(N)]
        order = [i for i in np.argsort(best) if np.isfinite(best[i])][:a.settle_candidates]
        trials = [(i, int(best_t[i]) + dt) for i in order for dt in (0, -8, 8)
                  if 0 <= best_t[i] + dt < int(grow.length[i])]
        if not trials:
            return dict(info, ok=False, why="no vine moved the block to a cell nearer the goal", grow_s=grow_s,
                        steps=int(grow.length.max()))

        def settle_stop(roll, t, buf):
            el = roll.element_ids()
            el = torch.where(el >= 0, el, len(trials))
            if "still" not in buf:
                buf["still"] = torch.zeros(len(trials) + 1, dtype=torch.long, device=el.device)
            slow = roll.x["obj_dstate"][:, 0, :2].norm(dim=1) < 2.0
            n = torch.where(slow, buf["still"][el] + 1, 0)
            buf["still"][el] = n
            return n >= 5

        t0 = time.time()
        settle = self.rollout(self.settle_params, [grow.state_at(i, t) for i, t in trials], a.settle_steps,
                              settle_stop)
        settle_s = time.time() - t0

        results = []
        for j, (i, t) in enumerate(trials):
            n = int(settle.length[j])
            p = (settle.hist["obj_pose"][n - 1, j, 0] if n else grow.hist["obj_pose"][t, i, 0]).double().numpy()
            cell = g.cell_of(p[:2])
            e = float(np.linalg.norm(p[:2] - g.center(cell)))
            angles = np.concatenate([ang[:t + 1, i], np.degrees(np.abs(
                (settle.hist["obj_pose"][:n, j, 0, 2].double().numpy() + math.pi / 4) % (math.pi / 2) - math.pi / 4))])
            fa = abs(math.degrees(wrap_quarter(p[2])))
            results.append(dict(i=i, t=t, j=j, cell=cell, to_go=float(ctg[cell]) if g.inside(cell) else np.inf,
                                err=e, final=p.tolist(), max_angle=float(angles.max()), final_angle=fa))
        hits = [r for r in results if r["err"] < a.tol and r["final_angle"] <= self.max_final_angle
                and r["to_go"] < here - 1e-9]
        info.update(grow_s=grow_s, settle_s=settle_s, steps=int(grow.length.max()), trials=len(trials),
                    reached=len({r["i"] for r in hits}),
                    reached_flagged=len({r["i"] for r in hits if r["max_angle"] > FLAG_DEG}),
                    reached_crooked=len({r["i"] for r in results if r["err"] < a.tol and r["to_go"] < here - 1e-9
                                         and r["final_angle"] > self.max_final_angle}),
                    best_err_mm=min(r["err"] for r in results))
        if not hits:
            return dict(info, ok=False, why=f"no settled pose within {a.tol:.0f} mm of a cell nearer the goal "
                        f"(closest {info['best_err_mm']:.0f} mm"
                        + (f"; {info['reached_crooked']} got there turned > {self.max_final_angle:.1f} deg)"
                           if info["reached_crooked"] else ")"))
        best = min(hits, key=lambda r: (r["to_go"], r["max_angle"] > FLAG_DEG, r["err"] + 0.5 * r["max_angle"]))
        base, th, ray, kind = cands[best["i"]]
        return dict(info, ok=True, spawn_cell=list(g.cell_of(base)), heading_deg=round(math.degrees(th), 1),
                    first_contact=kind, landed_cell=list(best["cell"]), cost_to_go=best["to_go"],
                    grow_steps=best["t"] + 1, err_mm=best["err"], final=best["final"],
                    max_angle_deg=best["max_angle"], final_angle_deg=best["final_angle"],
                    flagged=best["max_angle"] > FLAG_DEG,
                    states=grow.states(best["i"], upto=best["t"] + 1), settle_states=settle.states(best["j"]))


#----------------------------------------------------------------------- top level

def solve(e, sim, args, log=print):
    if args.n_headings:
        e["grid"]["n_headings"] = args.n_headings
    grid = Grid(e)
    search = PushSearch(e, sim, args)
    box = sim.initial_state().moveable_obj_pose[0, 0].numpy().astype(float)
    cell = grid.cell_of(box[:2])
    banned, pushes, attempts, first_plan = set(), [], [], None
    t0 = time.time()
    while cell != grid.goal and len(attempts) < args.max_attempts:
        plan = plan_cells(grid, cell, frozenset(banned))
        if first_plan is None:
            first_plan = plan
        if not plan:
            log(f"  no high-level plan from {cell} with {len(banned)} banned moves")
            break
        log(f"  plan from {cell}: " + ", ".join(f"{n} {k}{'*' if kind == 'deflected' else ''}"
                                                 for _, n, k, kind, _ in plan))
        c, name, k, kind, stopped = plan[0]
        res = search.push(box, plan, frozenset(banned))
        res.update(kind=kind, wall_stopped=stopped)
        attempts.append({key: v for key, v in res.items() if key not in ("states", "settle_states")})
        if not res["ok"]:
            log(f"  push {name} {k} from {c} ({kind}) FAILED: {res['why']}")
            banned.add((c, name, k))
            continue
        flag = "  FLAG: block turned %.0f deg" % res["max_angle_deg"] if res["flagged"] else ""
        log(f"  push {name} {k} from {c} ({kind}) -> landed in {tuple(res['landed_cell'])} (cost to go "
            f"{res['cost_to_go']:.2f}): spawn {res['spawn_cell']} at {res['heading_deg']:.0f} deg, "
            f"{res['grow_steps']} steps, {res['err_mm']:.0f} mm off, max turn {res['max_angle_deg']:.1f} deg; "
            f"{res['reached']}/{res['candidates']} candidates reached ({res['reached_flagged']} flagged), "
            f"{res['grow_s'] + res['settle_s']:.0f} s{flag}")
        pushes.append(res)
        box = np.array(res["final"])
        cell = grid.cell_of(box[:2])
    return dict(solved=cell == grid.goal, pushes=pushes, attempts=attempts, first_plan=first_plan,
                seconds=time.time() - t0, final=box.tolist())


#----------------------------------------------------------------------- drawing

def draw_board(ax, e, grid):
    from matplotlib.patches import Circle, Rectangle
    cell = grid.cell
    for j, r in enumerate(grid.rows):
        for i, ch in enumerate(r):
            x, y = i * cell, (grid.n_rows - j - 1) * cell
            if ch == "o":
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="#dceaf9", edgecolor="none", zorder=0))
            ax.add_patch(Rectangle((x, y), cell, cell, fill=False, edgecolor="#d6d5cf", lw=0.6, zorder=1))
    for w in e["walls"]:
        x0, y0, x1, y1 = w["rect"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=viz.WALL_FACE, edgecolor=viz.INK_2,
                               lw=1.2, hatch="///", zorder=2))
    gx, gy = e["goal"]["center"]
    ax.add_patch(Circle((gx, gy), 0.4 * cell, fill=False, edgecolor=viz.GOAL, lw=2, ls="--", zorder=3))
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
        c0 = plan[0][0]
        b = grid.center(c0)
        ax.add_patch(Rectangle(b - 49, 98, 98, facecolor=viz.BOX, edgecolor=viz.SURFACE, zorder=4))
        for n, (c, name, k, kind, stopped) in enumerate(plan, 1):
            a, t = grid.center(c), grid.center(add(c, DIRS[name], k))
            ax.annotate("", t, a, arrowprops=dict(arrowstyle="-|>", color=viz.BEST, lw=2,
                                                  ls="-" if kind == "straight" else "--"), zorder=6)
            mid = (a + t) / 2
            ax.annotate(str(n), mid, xytext=(6, 6), textcoords="offset points", color=viz.BEST, fontsize=10,
                        weight="bold", zorder=7)
            behind = grid.center(add(c, DIRS[name], -1))
            ax.plot(*behind, marker="o", color=viz.BEST, ms=5, alpha=0.6, zorder=6)
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
        bx, by = s.moveable_obj_pose[0, 0, :2].tolist()
        viz.draw_objects(ax, sim, s, outline_only=True)
        ax.annotate(str(n + 1), (bx, by), ha="center", va="center", fontsize=9, weight="bold",
                    color=colors[n % len(colors)], zorder=8)
    ax.set_title(title, fontsize=8, loc="left", color=viz.INK)
    viz.save_figure(fig, path, dpi=110)


def render_solution(e, grid, sim, result, out, every=4):
    import shutil
    frame_dir = os.path.join(out, "_frames")
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir)
    frames, trace = [], []
    n = len(result["pushes"])
    status = "SOLVED" if result["solved"] else "NOT SOLVED"
    for m, p in enumerate(result["pushes"], 1):
        seq = [(s, "growing") for s in p["states"]] + [(s, "stopped, block settling") for s in p["settle_states"]]
        picked = list(range(0, len(seq), every)) + [len(seq) - 1]
        for idx in sorted(set(picked)):
            s, phase = seq[idx]
            trace.append(s.moveable_obj_pose[0, 0, :2].tolist())
            ang = math.degrees(wrap_quarter(float(s.moveable_obj_pose[0, 0, 2])))
            flag = "  >15 deg!" if abs(ang) > FLAG_DEG else ""
            title = (f"{e['name']} {status} | push {m}/{n}: {p['direction']} {p['k']} from cell {tuple(p['cell'])}, "
                     f"spawn {tuple(p['spawn_cell'])} at {p['heading_deg']:.0f} deg\n{phase} | block turned "
                     f"{ang:+.1f} deg{flag}")
            path = os.path.join(frame_dir, f"f{len(frames):05d}.png")
            draw_frame(e, grid, sim, s, title, path, trace)
            frames.append(path)
    if not frames:
        return
    viz.write_gif(frames, os.path.join(out, "solution.gif"), frame_ms=int(1000 * every * float(sim.params.dt)))
    run_envs.gif_to_mp4(os.path.join(out, "solution.gif"), os.path.join(out, "solution.mp4"),
                        fps=round(1 / (every * float(sim.params.dt))))
    shutil.rmtree(frame_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("envs", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-fraction", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=25.0, help="settled block center to target cell center [mm]")
    parser.add_argument("--max-candidates", type=int, default=256, help="vines simulated per push")
    parser.add_argument("--settle-candidates", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--max-grow-steps", type=int, default=700, help="cap on one vine's growth steps")
    parser.add_argument("--max-attempts", type=int, default=20, help="low-level push searches per env")
    parser.add_argument("--max-final-angle", type=float, default=None,
                        help="[deg] a push may leave the block turned at most this (default: the jam angle)")
    parser.add_argument("--no-aim", dest="aim", action="store_false",
                        help="only the env's uniform headings (no headings aimed behind the block)")
    parser.add_argument("--n-headings", type=int, default=None, help="override the env's headings per spawn cell")
    parser.add_argument("--checkpoint", default=None, help="sim parameters from checkpoints/<name>.json (default run_envs.DEFAULT_SIM)")
    parser.add_argument("--out", default="out/puzzle")
    args = parser.parse_args()

    for path in args.envs:
        e, scene, cfg, _ = run_envs.load_env(path)
        if args.checkpoint:
            checkpoint.apply(cfg, args.checkpoint)
        cfg.device, cfg.qp_backend = args.device, "TORCH_IPM" if args.device.startswith("cuda") else "DIFFCP"
        sim = Sim(scene, cfg)
        if args.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        grid = Grid(e)
        out = os.path.join(args.out, e["name"] + (f"_h{args.n_headings}" if args.n_headings else "")
                           + ("" if args.aim else "_noaim"))
        os.makedirs(out, exist_ok=True)
        print(f"=== {e['name']}", flush=True)
        result = solve(e, sim, args, log=lambda m: print(m, flush=True))
        fp = result["first_plan"]
        draw_plan(e, grid, fp, os.path.join(out, "plan.png"),
                  f"{e['name']} first high-level plan ({len(fp or [])} pushes; dashed = deflected vine, "
                  f"dot = cell the vine must reach)")
        flagged = sum(p["flagged"] for p in result["pushes"])
        title = (f"{e['name']} {'SOLVED' if result['solved'] else 'NOT SOLVED'}: {len(result['pushes'])} pushes, "
                 f"{len(result['attempts'])} push searches, {flagged} flagged (>15 deg), "
                 f"{result['seconds'] / 60:.1f} min")
        draw_pushes(e, grid, sim, result["pushes"], os.path.join(out, "pushes.png"), title)
        render_solution(e, grid, sim, result, out)
        summary = dict(env=e["name"], solved=result["solved"], seconds=result["seconds"], final=result["final"],
                       first_plan=[[list(c), n, k, kind, st] for c, n, k, kind, st in (fp or [])],
                       pushes=[{k: v for k, v in p.items() if k not in ("states", "settle_states")}
                               for p in result["pushes"]],
                       attempts=result["attempts"], args=vars(args))
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(title, flush=True)


if __name__ == "__main__":
    main()
