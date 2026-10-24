'''
High-level planner for pushing square boxes with a vine: a block-pushing puzzle on a grid.

The workspace is cut into cells the size of one box. A cell is free, a wall, or holds a box.
One action is one deployment of the vine (grow, push, reset):

    1. The vine starts in a grow zone (a free cell on the border, see below) and grows inward.
    2. It follows a path of free cells (4-connected; at most --max-turns turns if given) to a
       cell next to a box.
    3. It grows into the box, which moves one cell the way the vine is heading. Every box lined
       up ahead of it moves too (several blocks at once). The push is blocked if the cell past
       the last box is a wall, off the grid, or the vine's own body.
    4. With --max-push k it can keep growing straight and push up to k cells in one deployment.

The vine then resets (retracts; not modeled), so its body never stays in the way.

Goals (any combination):
    box at        box X must end in a given cell
    clear         given cells must end up empty
    reach         the vine must be able to reach a given cell (a corridor of free cells to it)

The search is A* over the box positions. An action costs one push, and among plans with the
fewest pushes the shortest vines win. The heuristic counts, per box, the pushes it still needs
(one per axis it has to move along, more if it is farther than --max-push), and takes the max
over boxes, since one push can move several. Boxes stuck in a wall corner away from their goal
are pruned.

Grow zones: the free cells in the outermost ring of the grid. A map framed by walls has its
frame stripped first, so the zones are the ring just inside the frame. Cells marked S replace
them when there are any.

Input, either
    an ASCII map (--map file, or --example name):
        #  wall        .  free         A-Z  box         a-z  goal cell of box A-Z
        x  must end up clear           G    the vine must be able to reach it
        S  grow zone (if any S is given, only those)
    or a scene (default scene, or --env envs/*.txt), rasterized with cells the size of its first
    movable box (all boxes square and that size). The grid is aligned to box 0; a cell is a wall
    if walls cover more than half of it. Goal: --box-goal X_M Y_M (box 0), or the scene's.

Run from the repo root:
    python xnew_rrt/hierarchical_planner.py --example corridor
    python xnew_rrt/hierarchical_planner.py --example chain --max-push 3
    python xnew_rrt/hierarchical_planner.py --map my_puzzle.txt --max-turns 2
    python xnew_rrt/hierarchical_planner.py --box-goal 1.0625 0.45           # the default scene
Writes <out>/plan.json (every push: the vine's cells and launch, the boxes before and after, and
the world coordinates of those cells for the low-level planner), <out>/plan.png (every step) and
<out>/plan.gif.
'''

import argparse, heapq, itertools, json, math, os
from dataclasses import dataclass, field

DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}   # (dx, dy), y up

EXAMPLES = {
    # the vine can only start at S. B blocks the shaft down to A, so it is pushed along the top
    # row first; then A is pushed down the shaft and along the bottom row onto a
    "corridor": """
        ###########
        S...B.....#
        ####.######
        ####.######
        #...A.....#
        #.##.######
        #......a..#
        ###########
    """,
    # one entry, boxes in a row: the vine pushes all three at once to get C onto c
    "chain": """
        ##########
        S.ABC...c#
        ##########
    """,
    # G must become reachable. A blocks the only row to it; pushing it along the row would put it
    # on G, so the vine has to come around underneath and push it up into the pocket (3 turns)
    "reach": """
        ##########
        ####.#####
        S...A...G#
        #.##.#####
        #....#####
        ##########
    """,
    # border grow zones; the x cells (a lane the vine needs later) have to end up empty, and A
    # has to be parked on a
    "clear": """
        #########
        #.......#
        #.#x#.#.#
        #..xA...#
        #.#x#.#.#
        #.......#
        #...a...#
        #########
    """,
}


#----------------------------------------------------------------------- the puzzle

@dataclass
class Puzzle:
    width: int
    height: int
    walls: frozenset                         # {(x, y)}
    boxes: tuple                             # positions, in label order
    labels: tuple                            # one letter per box
    grow_zones: tuple                        # {(x, y)} where the vine can start
    box_goals: dict = field(default_factory=dict)   # box index -> cell
    clear: frozenset = frozenset()
    reach: tuple = ()
    # world coordinates of the grid (scene input): cell (x, y) spans origin + cell * (x, y) .. + cell
    origin_m: tuple = None
    cell_m: float = None

    def inside(self, c):
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def free(self, c, boxes):
        return self.inside(c) and c not in self.walls and c not in boxes

    def center_m(self, c):
        return (self.origin_m[0] + (c[0] + 0.5) * self.cell_m, self.origin_m[1] + (c[1] + 0.5) * self.cell_m)


def parse_map(text):
    rows = [r.strip() for r in text.strip("\n").splitlines() if r.strip()]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]
    # strip a full frame of walls: the grow zones are the ring inside it
    if all(c == "#" for c in rows[0] + rows[-1]) and all(r[0] == "#" and r[-1] == "#" for r in rows):
        rows = [r[1:-1] for r in rows[1:-1]]
    height, width = len(rows), len(rows[0])
    walls, boxes, goals, clear, reach, marked = set(), {}, {}, set(), [], []
    for j, row in enumerate(rows):
        y = height - 1 - j                    # first text row is the top
        for x, ch in enumerate(row):
            c = (x, y)
            if ch == "#":
                walls.add(c)
            elif ch.isupper() and ch not in "GS":
                boxes[ch] = c
            elif ch.islower() and ch != "x":
                goals[ch.upper()] = c
            elif ch == "x":
                clear.add(c)
            elif ch == "G":
                reach.append(c)
            elif ch == "S":
                marked.append(c)
    labels = tuple(sorted(boxes))
    zones = marked or [c for c in itertools.product(range(width), range(height))
                       if (c[0] in (0, width - 1) or c[1] in (0, height - 1)) and c not in walls]
    unknown = set(goals) - set(labels)
    if unknown:
        raise SystemExit(f"goal for a box that is not on the map: {sorted(unknown)}")
    return Puzzle(width, height, frozenset(walls), tuple(boxes[l] for l in labels), labels,
                  tuple(zones), {labels.index(l): c for l, c in goals.items()}, frozenset(clear), tuple(reach))


def puzzle_from_scene(scene, box_goal_m=None):
    '''Rasterize a Scene (meters) with cells the size of its first movable box.'''
    if not scene.objects_m:
        raise SystemExit("the scene has no movable boxes")
    boxes = [(min(o[0], o[2]), min(o[1], o[3]), max(o[0], o[2]), max(o[1], o[3])) for o in scene.objects_m]
    cell = boxes[0][2] - boxes[0][0]
    for b in boxes:
        if abs((b[2] - b[0]) - cell) > 1e-3 or abs((b[3] - b[1]) - cell) > 1e-3:
            raise SystemExit(f"boxes must be square and {cell * 1000:.0f} mm: {b}")
    xmin, xmax, ymin, ymax = scene.bounds_m
    # align the grid with box 0
    ox = boxes[0][0] - math.floor((boxes[0][0] - xmin) / cell + 1e-6) * cell
    oy = boxes[0][1] - math.floor((boxes[0][1] - ymin) / cell + 1e-6) * cell
    width = int(math.floor((xmax - ox) / cell + 1e-6))
    height = int(math.floor((ymax - oy) / cell + 1e-6))

    def covered(x, y):
        x0, y0 = ox + x * cell, oy + y * cell
        area = 0.0
        for w in scene.walls_m:
            wx0, wy0, wx1, wy1 = min(w[0], w[2]), min(w[1], w[3]), max(w[0], w[2]), max(w[1], w[3])
            area += max(0.0, min(x0 + cell, wx1) - max(x0, wx0)) * max(0.0, min(y0 + cell, wy1) - max(y0, wy0))
        return area / cell ** 2

    walls = {(x, y) for x in range(width) for y in range(height) if covered(x, y) > 0.5}
    to_cell = lambda px, py: (int(math.floor((px - ox) / cell)), int(math.floor((py - oy) / cell)))
    box_cells = tuple(to_cell((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes)
    # grow zones: free cells next to the frame (walls touching the grid's edge) or on the edge
    frame = set()
    stack = [c for c in walls if c[0] in (0, width - 1) or c[1] in (0, height - 1)]
    while stack:
        c = stack.pop()
        if c in frame:
            continue
        frame.add(c)
        stack += [(c[0] + dx, c[1] + dy) for dx, dy in DIRS.values() if (c[0] + dx, c[1] + dy) in walls]
    zones = [c for c in itertools.product(range(width), range(height)) if c not in walls and c not in box_cells
             and (c[0] in (0, width - 1) or c[1] in (0, height - 1)
                  or any((c[0] + dx, c[1] + dy) in frame for dx, dy in DIRS.values()))]
    goals = {}
    goal = box_goal_m or scene.box_goal_m
    if goal is not None:
        goals[0] = to_cell(goal[0], goal[1])
    labels = tuple(chr(ord("A") + k) for k in range(len(boxes)))
    return Puzzle(width, height, frozenset(walls), box_cells, labels, tuple(zones), goals,
                  origin_m=(ox, oy), cell_m=cell)


#----------------------------------------------------------------------- the vine

def launch_headings(p: Puzzle, c):
    '''
    Directions the vine can start growing in from grow-zone cell c: away from the grid's edge or a
    wall right behind it (the vine comes in from the border). Any direction if neither is there.
    '''
    out = [h for h, (dx, dy) in DIRS.items()
           if not p.inside((c[0] - dx, c[1] - dy)) or (c[0] - dx, c[1] - dy) in p.walls]
    return out or list(DIRS)


def vine_paths(p: Puzzle, boxes, max_turns=None):
    '''
    Every (cell, heading) the vine can get to, with the shortest path there (fewest cells, then
    fewest turns): {(cell, heading): (path cells, turns, launch cell, launch heading)}.
    '''
    occupied = set(boxes)
    best = {}
    queue = []
    for z in p.grow_zones:
        if not p.free(z, occupied):
            continue
        for h in launch_headings(p, z):
            heapq.heappush(queue, (1, 0, next(_tie), z, h, (z,), z, h))
    while queue:
        n, turns, _, c, h, path, z0, h0 = heapq.heappop(queue)
        key = (c, h)
        if key in best:
            continue
        best[key] = (path, turns, z0, h0)
        for h2, (dx, dy) in DIRS.items():
            if DIRS[h2] == (-DIRS[h][0], -DIRS[h][1]):
                continue                     # a vine does not fold back on itself
            t2 = turns + (h2 != h)
            if max_turns is not None and t2 > max_turns:
                continue
            c2 = (c[0] + dx, c[1] + dy)
            if p.free(c2, occupied) and c2 not in path and (c2, h2) not in best:
                heapq.heappush(queue, (n + 1, t2, next(_tie), c2, h2, path + (c2,), z0, h0))
    return best


_tie = itertools.count()


def reachable_cells(p: Puzzle, boxes, max_turns=None):
    return {c for c, _ in vine_paths(p, boxes, max_turns)}


#----------------------------------------------------------------------- pushes

@dataclass
class Push:
    launch: tuple            # grow-zone cell
    launch_heading: str
    path: tuple              # cells the vine grows through before touching the first box
    direction: str
    distance: int            # cells the boxes move
    moved: tuple             # indices of the boxes that move
    boxes_before: tuple
    boxes_after: tuple

    def vine_cells(self):
        '''Everything the vine occupies by the end of the push: its path and the cells it pushed into.'''
        dx, dy = DIRS[self.direction]
        tip = self.path[-1]
        return self.path + tuple((tip[0] + k * dx, tip[1] + k * dy) for k in range(1, self.distance + 1))


def pushes(p: Puzzle, boxes, max_turns=None, max_push=1):
    '''Every push possible from this box configuration, one (the shortest vine) per outcome.'''
    where = {c: i for i, c in enumerate(boxes)}
    out = {}
    for (c, h), (path, turns, z0, h0) in vine_paths(p, boxes, max_turns).items():
        dx, dy = DIRS[h]
        first = (c[0] + dx, c[1] + dy)
        if first not in where:
            continue
        current = list(boxes)
        body = set(path)
        for k in range(1, max_push + 1):
            # the line of boxes ahead of the tip, which now sits one cell behind the first one
            tip = (c[0] + (k - 1) * dx, c[1] + (k - 1) * dy)
            occ = {cc: i for i, cc in enumerate(current)}
            chain, cc = [], (tip[0] + dx, tip[1] + dy)
            while cc in occ:
                chain.append(occ[cc])
                cc = (cc[0] + dx, cc[1] + dy)
            if not chain or not p.inside(cc) or cc in p.walls or cc in body:
                break
            for i in chain:
                current[i] = (current[i][0] + dx, current[i][1] + dy)
            body.add((tip[0] + dx, tip[1] + dy))
            after = tuple(current)
            moved = tuple(i for i in range(len(boxes)) if after[i] != boxes[i])
            cost = len(path) + k
            if after not in out or cost < out[after][0]:
                out[after] = (cost, Push(z0, h0, path, h, k, moved, tuple(boxes), after))
    return [push for _, push in out.values()]


#----------------------------------------------------------------------- search

def satisfied(p: Puzzle, boxes, max_turns=None):
    if any(boxes[i] != c for i, c in p.box_goals.items()):
        return False
    if any(b in p.clear for b in boxes):
        return False
    if p.reach:
        cells = reachable_cells(p, boxes, max_turns)
        if any(r not in cells for r in p.reach):
            return False
    return True


def dead(p: Puzzle, boxes):
    '''
    A box that still has to move (off its goal, or on a cell that must end up clear) but sits in a
    corner of walls / the grid's edge: no push can move it again.
    '''
    blocked = lambda c: not p.inside(c) or c in p.walls
    for i, b in enumerate(boxes):
        must_move = (i in p.box_goals and p.box_goals[i] != b) or b in p.clear
        if not must_move:
            continue
        ns = [blocked((b[0] + dx, b[1] + dy)) for dx, dy in DIRS.values()]   # E, W, N, S
        if (ns[0] or ns[1]) and (ns[2] or ns[3]):
            return True
    return False


def heuristic(p: Puzzle, boxes, max_push):
    '''Pushes still needed: per box with a goal, one per axis it must move along (more if farther
    than max_push); the max over boxes, as one push can move several. Plus 1 for open clear/reach goals.'''
    h = 0
    for i, g in p.box_goals.items():
        dx, dy = abs(g[0] - boxes[i][0]), abs(g[1] - boxes[i][1])
        h = max(h, math.ceil(dx / max_push) + math.ceil(dy / max_push))
    if h == 0 and (any(b in p.clear for b in boxes)):
        h = 1
    return h


def plan(p: Puzzle, max_turns=None, max_push=1, max_expansions=200000, vine_weight=1e-3):
    '''
    A* over box configurations. Cost = pushes + vine_weight * cells the vine grows (so the fewest
    pushes win, then the shortest vines). Returns ([Push], stats) or (None, stats).
    '''
    start = tuple(p.boxes)
    queue = [(heuristic(p, start, max_push), 0.0, next(_tie), start)]
    came = {start: (None, None)}
    cost = {start: 0.0}
    expanded = 0
    while queue:
        _, g, _, boxes = heapq.heappop(queue)
        if g > cost[boxes] + 1e-9:
            continue
        if satisfied(p, boxes, max_turns):
            steps = []
            while came[boxes][0] is not None:
                prev, push = came[boxes]
                steps.append(push)
                boxes = prev
            return steps[::-1], {"expanded": expanded, "seen": len(came)}
        expanded += 1
        if expanded > max_expansions:
            break
        for push in pushes(p, boxes, max_turns, max_push):
            nxt = push.boxes_after
            if dead(p, nxt):
                continue
            g2 = g + 1 + vine_weight * len(push.vine_cells())
            if g2 < cost.get(nxt, math.inf) - 1e-9:
                cost[nxt] = g2
                came[nxt] = (boxes, push)
                heapq.heappush(queue, (g2 + heuristic(p, nxt, max_push), g2, next(_tie), nxt))
    return None, {"expanded": expanded, "seen": len(came)}


#----------------------------------------------------------------------- output

def ascii_grid(p: Puzzle, boxes, vine=()):
    lines = []
    for y in range(p.height - 1, -1, -1):
        row = ""
        for x in range(p.width):
            c = (x, y)
            if c in p.walls:
                ch = "#"
            elif c in boxes:
                ch = p.labels[boxes.index(c)]
            elif c in vine:
                ch = "~"
            elif c in p.box_goals.values():
                ch = p.labels[[i for i, g in p.box_goals.items() if g == c][0]].lower()
            elif c in p.reach:
                ch = "G"
            elif c in p.clear:
                ch = "x"
            elif c in p.grow_zones:
                ch = ","
            else:
                ch = "."
            row += ch
        lines.append(row)
    return "\n".join(lines)


def draw(p: Puzzle, steps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow

    states = [(tuple(p.boxes), None)] + [(s.boxes_after, s) for s in steps]
    cols = min(4, len(states))
    rows = math.ceil(len(states) / cols)
    s = max(2.4, 9.0 / max(p.width, p.height) * 1.0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * s * p.width / max(p.width, p.height) + 1,
                                                  rows * s * p.height / max(p.width, p.height) + 1), squeeze=False)
    colors = plt.get_cmap("tab10")
    frames = []
    for ax, (k, (boxes, push)) in zip(axes.flat, enumerate(states)):
        for c in p.grow_zones:
            ax.add_patch(Rectangle(c, 1, 1, color="#e6f2e6"))
        for c in p.walls:
            ax.add_patch(Rectangle(c, 1, 1, color="0.45"))
        for c in p.clear:
            ax.add_patch(Rectangle(c, 1, 1, fill=False, hatch="//", ec="tab:red", lw=0))
        for c in p.reach:
            ax.plot(c[0] + 0.5, c[1] + 0.5, "*", color="tab:green", ms=14)
        for i, g in p.box_goals.items():
            ax.add_patch(Rectangle((g[0] + 0.1, g[1] + 0.1), 0.8, 0.8, fill=False, ec=colors(i), lw=2, ls="--"))
        if push is not None:
            cells = push.vine_cells()
            xs = [c[0] + 0.5 for c in cells]
            ys = [c[1] + 0.5 for c in cells]
            ax.plot(xs, ys, "-", color="#c9a3e6", lw=14 * s / 3, solid_capstyle="round", alpha=0.8)
            ax.plot(xs, ys, "-", color="#6a3d9a", lw=1.5)
            ax.plot(xs[0], ys[0], "o", color="#6a3d9a")
            dx, dy = DIRS[push.direction]
            ax.add_patch(FancyArrow(xs[-1], ys[-1], 0.6 * dx, 0.6 * dy, width=0.08, color="k"))
        for i, b in enumerate(boxes):
            ax.add_patch(Rectangle((b[0] + 0.08, b[1] + 0.08), 0.84, 0.84, color=colors(i)))
            ax.text(b[0] + 0.5, b[1] + 0.5, p.labels[i], ha="center", va="center", color="w", weight="bold")
        ax.set_xlim(0, p.width); ax.set_ylim(0, p.height); ax.set_aspect("equal")
        ax.set_xticks(range(p.width + 1)); ax.set_yticks(range(p.height + 1))
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.grid(True, color="0.85", lw=0.5)
        ax.set_title("start" if push is None else
                     f"push {k}: {''.join(p.labels[i] for i in push.moved)} {push.direction} x{push.distance}",
                     fontsize=10)
    for ax in axes.flat[len(states):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_plan(p: Puzzle, steps, stats, out):
    os.makedirs(out, exist_ok=True)
    world = p.origin_m is not None

    def cells(cs):
        d = {"cells": [list(c) for c in cs]}
        if world:
            d["centers_m"] = [list(map(lambda v: round(v, 4), p.center_m(c))) for c in cs]
        return d

    data = {"grid": {"width": p.width, "height": p.height, "walls": sorted(map(list, p.walls)),
                     "grow_zones": [list(c) for c in p.grow_zones], "labels": list(p.labels),
                     "origin_m": p.origin_m, "cell_m": p.cell_m},
            "goals": {"box_at": {p.labels[i]: list(c) for i, c in p.box_goals.items()},
                      "clear": sorted(map(list, p.clear)), "reach": [list(c) for c in p.reach]},
            "solved": steps is not None, "search": stats, "pushes": []}
    for k, s in enumerate(steps or []):
        data["pushes"].append({
            "launch": cells([s.launch]), "launch_heading": s.launch_heading,
            # what the low-level planner needs: the vine grows through these cells, all free, then
            # keeps going straight into the box
            "vine_path": cells(s.path), "push_direction": s.direction, "push_cells": s.distance,
            "moved": [p.labels[i] for i in s.moved],
            "boxes_before": {p.labels[i]: cells([c]) for i, c in enumerate(s.boxes_before)},
            "boxes_after": {p.labels[i]: cells([c]) for i, c in enumerate(s.boxes_after)},
        })
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(data, f, indent=1)
    if steps is not None:
        draw(p, steps, os.path.join(out, "plan.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--map", default=None, help="ASCII map file")
    src.add_argument("--example", choices=sorted(EXAMPLES), default=None)
    src.add_argument("--env", default=None, help="envs/*.txt scene (default: the built-in scene)")
    parser.add_argument("--box-goal", type=float, nargs=2, default=None, metavar=("X_M", "Y_M"),
                        help="scene input: box 0's goal")
    parser.add_argument("--max-turns", type=int, default=None, help="turns the vine may make (default any)")
    parser.add_argument("--max-push", type=int, default=1, help="cells one deployment may push")
    parser.add_argument("--max-expansions", type=int, default=200000)
    parser.add_argument("--out", default="out/hierarchical")
    args = parser.parse_args()

    if args.map or args.example:
        text = EXAMPLES[args.example] if args.example else open(args.map).read()
        p = parse_map(text)
    else:
        from scene import default_scene, load_scene    # (puts the repo root on sys.path)
        scene = load_scene(args.env) if args.env else default_scene()
        p = puzzle_from_scene(scene, args.box_goal)
        if p.cell_m:
            print(f"grid {p.width} x {p.height} cells of {p.cell_m * 1000:.0f} mm, origin {p.origin_m} m")

    print(ascii_grid(p, tuple(p.boxes)))
    print("(',' grow zone, lowercase = goal of that box, x = keep clear, G = vine must reach)\n")
    steps, stats = plan(p, args.max_turns, args.max_push, args.max_expansions)
    if steps is None:
        print(f"no plan ({stats['expanded']} configurations expanded)")
    else:
        print(f"{len(steps)} pushes ({stats['expanded']} configurations expanded):")
        for k, s in enumerate(steps, 1):
            print(f"\npush {k}: launch at {s.launch} heading {s.launch_heading}, vine through {len(s.path)} cells, "
                  f"pushes {''.join(p.labels[i] for i in s.moved)} {s.direction} by {s.distance}")
            print(ascii_grid(p, s.boxes_after, s.vine_cells()))
    write_plan(p, steps, stats, args.out)
    print("\nwrote", os.path.join(args.out, "plan.json") + ("" if steps is None else " and plan.png"))


if __name__ == "__main__":
    main()
