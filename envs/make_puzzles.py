'''
Writes the block-puzzle environments envs/5a.json ... 5d.json (sketches of 2026-10-24 and -25) and
draws them to envs/vis/<name>.png.

The board is a grid of CELL_MM cells. Walls fill whole cells, the block (98 mm) starts centered in
its cell, and the goal is a cell. Around the drawn map is a ring of spawn cells: a vine can grow
from the center of any of them at one of N_HEADINGS headings, straight (no actuation). The vine is
retracted between pushes. xnew_rrt/block_puzzle.py plans on these.

    5a  easy    2 pushes: up the channel (stopped by the top bar), right into the pocket
    5b  medium  3 pushes: up, right, down into the pocket; every push is stopped by a wall
    5c  hard    6 pushes; two of them need a diagonal vine that a wall turns into a horizontal
                push (a stub wall blocks the straight approach), and two stop mid-corridor

    5d          two blocks: the long block R must be pushed out of the row before the block can reach the goal

Maps are written top row first: '#' wall, '.' free, 'B' block, 'G' goal (free), 'R' (also 'S', 'T') the
cells of another movable block, which may span several cells (n cells: n * CELL_MM - 12 mm long, the
mass scaled by its area), 'q' a triangular
wall filling the cell's upper-left half (right angle at the top left, the slope facing down-right).
The sim has only rectangles, so a triangle is a 24 mm plate whose face is the slope (trimmed so it
stays inside the triangle) plus a 12 mm strip along its left leg; its top leg must lie on a wall. The spawn ring is
added around them. Walls are merged into as few rectangles as possible (merge_walls), which cuts
the sim's contact rows; --per-cell writes one rectangle per wall cell instead (for comparison).

    python envs/make_puzzles.py [--cell 110] [--per-cell]
    python envs/make_puzzles.py --only 5c --corner-radius 10 --suffix _round      # envs/5c_round.json
'''

import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_MM = 110.0        # block 98 mm + 12 mm tolerance
BOX = 98.0
BOX_KG = 0.503
N_HEADINGS = 16

MAPS = {
    "5a": ("easy: push the block up its channel, then right into the pocket under the top bar",
           [".......",
            "..####.",
            "....G#.",
            "..#.##.",
            "..#B##.",
            "......."]),
    "5b": ("medium: up, right, then down into the pocket; every push ends against a wall",
           ["............",
            "............",
            "..###.#.....",
            "......#.....",
            "..#.#.#.....",
            "..#B#G#.....",
            "....###.....",
            "............"]),
    # 5c (sketch of 2026-10-27, the old map is envs/archive/5c_v1.json): the pocket's mouth one cell
    # further right, and a stub under the goal row so the last vine is guided along it
    "5c": ("hard: six pushes; diagonal vines that walls turn into horizontal pushes, two pushes stop mid-corridor",
           ["..#..........",
            ".######......",
            ".#...........",
            "....###.##...",
            "..#.#...G#...",
            ".B#...#.##...",
            "..#..........",
            "..#.........."]),
    "5d": ("two blocks: the long block up into the pocket to clear the row, then the block right into the goal",
           [".........",
            "..###....",
            "..#.#....",
            "..#.####.",
            "...RB.G#.",
            "..#R####.",
            "........."]),
}
# Variants of the old 5c (envs/archive/5c_v1.json), archived 2026-10-27 with their envs in envs/archive/
# (5c_round, 5c_120 were 5c with --corner-radius 10 / --cell 120). Not written by default.
ARCHIVED_MAPS = {
    "5c_fill": ("hard, fewer ways around: 5c with grid cells (6, 7), (7, 7) and (7, 6) filled",
                ["..#..........",
                 ".#####..#....",
                 ".#......#....",
                 "....##.###...",
                 "..#.#...G#...",
                 ".B#...####...",
                 "..#..##......",
                 "..#.........."]),
    "5c_tri": ("hard, with a ramp: 5c with rounded corners, and the stub under the T made a triangle whose slope "
               "turns a vine growing up from below into the push along the goal row",
               ["..#..........",
                ".#####..#....",
                ".#......#....",
                "....##.###...",
                "..#.q...G#...",
                ".B#....###...",
                "..#..........",
                "..#.........."]),
}
CORNER_MM = {"5c_tri": 10.0}       # per-map corner radius (at least --corner-radius)
OBSTACLE_BLOCKS = "RST"            # map letters of extra movable blocks; each letter's cells form one block
PLATE_MM, LEG_MM = 24.0, 12.0


def with_ring(rows):
    '''The map with the ring of spawn cells ('o') around it.'''
    w = len(rows[0])
    assert all(len(r) == w for r in rows)
    return ["o" * (w + 2)] + ["o" + r + "o" for r in rows] + ["o" * (w + 2)]


def cell_center(i, j, n_rows, cell=CELL_MM):
    '''Center [mm] of grid cell (column i, row j counted from the top), y up.'''
    return ((i + 0.5) * cell, (n_rows - j - 0.5) * cell)


def merge_walls(grid):
    '''Cover the '#' cells with few axis-aligned rectangles: greedy, widest run first then as many
    rows down as it spans, tried row-major and column-major; the smaller cover wins.
    Returns [(i0, j0, i1, j1)] inclusive cell ranges.'''
    def greedy(cells):
        left, rects = set(cells), []
        for (i, j) in sorted(cells, key=lambda c: (c[1], c[0])):
            if (i, j) not in left:
                continue
            i1 = i
            while (i1 + 1, j) in left:
                i1 += 1
            j1 = j
            while all((k, j1 + 1) in left for k in range(i, i1 + 1)):
                j1 += 1
            for k in range(i, i1 + 1):
                for l in range(j, j1 + 1):
                    left.discard((k, l))
            rects.append((i, j, i1, j1))
        return rects

    cells = [(i, j) for j, r in enumerate(grid) for i, ch in enumerate(r) if ch == "#"]
    row_major = greedy(cells)
    col_major = [(j0, i0, j1, i1) for (i0, j0, i1, j1) in greedy([(j, i) for (i, j) in cells])]
    return min(row_major, col_major, key=len)


def puzzle_env(name, description, rows, cell=CELL_MM, per_cell=False, corner=0.0):
    grid = with_ring(rows)
    n_cols, n_rows = len(grid[0]), len(grid)
    find = lambda ch: next((i, j) for j, r in enumerate(grid) for i, c in enumerate(r) if c == ch)
    block, goal = find("B"), find("G")
    rects = ([(i, j, i, j) for j, r in enumerate(grid) for i, ch in enumerate(r) if ch == "#"] if per_cell
             else merge_walls(grid))
    walls, triangles = [], []
    for (i0, j0, i1, j1) in rects:
        x0, y1 = i0 * cell, (n_rows - j0) * cell
        x1, y0 = (i1 + 1) * cell, (n_rows - j1 - 1) * cell
        walls.append({"rect": [x0, y0, x1, y1], "cells": [i0, j0, i1, j1]})
    for (i, j) in [(i, j) for j, r in enumerate(grid) for i, ch in enumerate(r) if ch == "q"]:
        x0, x1, ytop = i * cell, (i + 1) * cell, (n_rows - j) * cell
        ybot = ytop - cell
        triangles.append([[x0, ytop], [x1, ytop], [x0, ybot]])
        L = math.hypot(cell, cell)
        n_in = (-1 / math.sqrt(2), 1 / math.sqrt(2))                                 # into the triangle
        mid = ((x1 + x0) / 2 + PLATE_MM / 2 * n_in[0], (ytop + ybot) / 2 + PLATE_MM / 2 * n_in[1])
        walls.append({"obb": [round(mid[0], 2), round(mid[1], 2), 45.0, round((L - 2 * PLATE_MM) / 2, 2),
                              PLATE_MM / 2], "cells": [i, j, i, j], "part": "triangle slope"})
        walls.append({"rect": [x0, ybot + LEG_MM, x0 + LEG_MM, ytop], "cells": [i, j, i, j], "part": "triangle leg"})
    bc, gc = cell_center(*block, n_rows, cell), cell_center(*goal, n_rows, cell)
    first_ring = cell_center(block[0], n_rows - 1, n_rows, cell)          # ring cell under the block
    boxes = [{"center": list(bc), "angle_deg": 0.0, "size": [BOX, BOX], "mass_kg": BOX_KG}]
    blocks = [{"cells": [list(block)]}]                                   # block 0 is the one with the goal
    for ch in OBSTACLE_BLOCKS:
        cells = sorted((i, j) for j, r in enumerate(grid) for i, c in enumerate(r) if c == ch)
        if not cells:
            continue
        i0, j0 = min(i for i, _ in cells), min(j for _, j in cells)
        i1, j1 = max(i for i, _ in cells), max(j for _, j in cells)
        assert len(cells) == (i1 - i0 + 1) * (j1 - j0 + 1), f"block {ch} is not a rectangle of cells"
        w, h = (i1 - i0 + 1) * cell - (cell - BOX), (j1 - j0 + 1) * cell - (cell - BOX)
        c0, c1 = cell_center(i0, j0, n_rows, cell), cell_center(i1, j1, n_rows, cell)
        boxes.append({"center": [(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2], "angle_deg": 0.0, "size": [w, h],
                      "mass_kg": round(BOX_KG * w * h / BOX ** 2, 3)})
        blocks.append({"cells": [list(c) for c in cells]})
    return {"name": name, "category": "block_puzzle", "description": description,
            "board_mm": [n_cols * cell, n_rows * cell],
            "walls": walls,
            "boxes": boxes,
            "start": {"xy": list(first_ring), "heading_deg": 90.0},        # unused: every push picks its own
            "goal": {"type": "box", "box": 0, "center": list(gc), "radius": 25.0},
            "grid": {"cell_mm": cell, "rows": grid, "block": list(block), "blocks": blocks, "goal": list(goal),
                     "n_headings": N_HEADINGS, "spawn": "ring"},
            "planner": {"max_bodies": min(70, int((n_cols ** 2 + n_rows ** 2) ** 0.5 * cell / 30) + 4)},
            **({"corner_radius_mm": {"walls": corner, "boxes": corner}} if corner > 0 else {}),
            **({"triangles": triangles} if triangles else {}),
            "notes": f"{cell:.0f} mm cells ({cell - BOX:.0f} mm tolerance around the {BOX:.0f} mm block); "
                     f"{len(walls)} wall rectangles for {sum(r.count('#') + r.count('q') for r in grid)} wall cells"
                     + (f"; wall and block corners rounded to {corner:.0f} mm" if corner > 0 else "")}


def draw(e, path):
    '''The grid, spawn ring, walls, block and goal.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
    g = e["grid"]
    cell, grid = g["cell_mm"], g["rows"]
    n_cols, n_rows = len(grid[0]), len(grid)
    fig, ax = plt.subplots(figsize=(0.62 * n_cols + 1, 0.62 * n_rows + 0.8), facecolor="#fcfcfb",
                           layout="constrained")
    ax.set_facecolor("#fcfcfb")
    for j, r in enumerate(grid):
        for i, ch in enumerate(r):
            x, y = i * cell, (n_rows - j - 1) * cell
            if ch == "o":
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="#dceaf9", edgecolor="none", zorder=0))
            ax.add_patch(Rectangle((x, y), cell, cell, fill=False, edgecolor="#d6d5cf", lw=0.6, zorder=1))
    rw = e.get("corner_radius_mm", {}).get("walls", 0.0)
    rb = e.get("corner_radius_mm", {}).get("boxes", 0.0)
    for w in e["walls"]:
        if "part" in w:                                        # a triangle's pieces: drawn as the triangle
            continue
        x0, y0, x1, y1 = w["rect"]
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rw}",
                                    facecolor="#e1e0d9", edgecolor="#52514e", lw=1.2, hatch="///", zorder=2))
    for tri in e.get("triangles", []):
        ax.add_patch(Polygon(tri, closed=True, facecolor="#e1e0d9", edgecolor="#52514e", lw=1.2, hatch="///",
                             zorder=2))
    gx, gy = e["goal"]["center"]
    ax.add_patch(FancyBboxPatch((gx - BOX / 2, gy - BOX / 2), BOX, BOX, boxstyle=f"round,pad=0,rounding_size={rb}",
                                fill=False, edgecolor="#1baf7a", lw=2, ls="--", zorder=3))
    ax.annotate("goal", (gx, gy), ha="center", va="center", color="#1baf7a", fontsize=8, zorder=3)
    for k, b in enumerate(e["boxes"]):
        (bx, by), (w, h) = b["center"], b["size"]
        ax.add_patch(FancyBboxPatch((bx - w / 2, by - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rb}",
                                    facecolor="#eb6834" if k == 0 else "#d6a02a", edgecolor="#fcfcfb", lw=1.5,
                                    zorder=4))
    for i in range(n_cols):
        ax.annotate(str(i), ((i + 0.5) * cell, n_rows * cell + 12), ha="center", va="bottom", fontsize=7,
                    color="#898781", annotation_clip=False)
    for j in range(n_rows):
        ax.annotate(str(j), (-12, (n_rows - j - 0.5) * cell), ha="right", va="center", fontsize=7,
                    color="#898781", annotation_clip=False)
    ax.set_xlim(0, n_cols * cell); ax.set_ylim(0, n_rows * cell)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{e['name']}: {e['description']}\n{cell:.0f} mm cells, blue ring = spawn cells "
                 f"({g['n_headings']} headings each), {len(e['walls'])} wall rectangles"
                 + (f", corners rounded {rw:.0f} mm" if rw or rb else ""),
                 fontsize=8, loc="left", color="#0b0b0b", pad=16)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=float, default=CELL_MM)
    parser.add_argument("--per-cell", action="store_true", help="one wall rectangle per wall cell")
    parser.add_argument("--suffix", default="", help="written as envs/<name><suffix>.json")
    parser.add_argument("--corner-radius", type=float, default=0.0, help="[mm] round every wall and block corner")
    parser.add_argument("--only", nargs="*", default=None, help="these puzzles only (e.g. 5c)")
    args = parser.parse_args()
    os.makedirs(os.path.join(HERE, "vis"), exist_ok=True)
    for name, (desc, rows) in MAPS.items():
        if args.only and name not in args.only:
            continue
        e = puzzle_env(name + args.suffix, desc, rows, args.cell, args.per_cell,
                       max(args.corner_radius, CORNER_MM.get(name, 0.0)))
        with open(os.path.join(HERE, e["name"] + ".json"), "w") as f:
            json.dump(e, f, indent=1)
        draw(e, os.path.join(HERE, "vis", e["name"] + ".png"))
        print("wrote", e["name"], e["notes"])
