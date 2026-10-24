'''
Writes the sim environments envs/<name>.json, from the sketches of 2026-10-20. The json files are
the source of truth; this script only makes the geometry easier to write down (walls as segments,
boxes placed along a launch ray or against a wall).

Categories:
  b2g_unretractable   block to goal with one vine: the goal is close to the block, or a wall
                      guides it there (1a-1c simple, 2a-2b complex)
  b2g_retractable     block to goal with several pushes, resetting the vine between them (3a-3b)
  vine_to_goal        the vine's tip to a goal region, with movable blocks in the way (4a-4d)
(Sketch 5, the basic contact experiments, comes from real data instead.)

Units: mm and degrees. Board 1200 x 900 mm, origin at the bottom left, y up; the boundary walls
lie outside it. Boxes are the real ones (98 mm, 0.503 kg, real_measurements.md). Where a sketch
left the geometry open, the choice is noted in the env's "notes".

    python envs/make_envs.py          # then: python xnew_rrt/run_envs.py --draw-only
'''

import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = (1200.0, 900.0)
BOX = 98.0
BOX_KG = 0.503
WALL_T = 20.0


def along(p, heading_deg, d, lateral=0.0):
    '''The point d along the ray from p at heading_deg, shifted `lateral` to its left.'''
    h = math.radians(heading_deg)
    return [round(p[0] + d * math.cos(h) - lateral * math.sin(h), 1),
            round(p[1] + d * math.sin(h) + lateral * math.cos(h), 1)]


def seg(a, b, t=WALL_T):
    return {"from": list(a), "to": list(b), "thickness": t}


def box(center, angle_deg=0.0, size=BOX, mass=BOX_KG):
    return {"center": [round(v, 1) for v in center], "angle_deg": round(angle_deg, 1), "size": [size, size], "mass_kg": mass}


def against_wall(a, b, t_frac, side=-1, gap=1.0, t=WALL_T, size=BOX):
    '''A box flush against segment a-b at fraction t_frac along it, on its right (side -1) or left (+1).'''
    ux, uy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    nx, ny = -uy * side, ux * side
    off = size / 2 + t / 2 + gap
    c = [a[0] + t_frac * (b[0] - a[0]) + off * nx, a[1] + t_frac * (b[1] - a[1]) + off * ny]
    return c, math.degrees(math.atan2(uy, ux))


def corners(center, size, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    hw, hh = size[0] / 2, size[1] / 2
    return [(center[0] + x * c - y * s, center[1] + x * s + y * c) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def wall_corners(w):
    (ax, ay), (bx, by) = w["from"], w["to"]
    L = math.hypot(bx - ax, by - ay)
    return corners(((ax + bx) / 2, (ay + by) / 2), (L + w["thickness"], w["thickness"]),
                   math.degrees(math.atan2(by - ay, bx - ax)))


def _sat(p, q):
    '''Largest separation of two convex polygons along their edge normals (negative = overlap depth).'''
    best = -math.inf
    for poly in (p, q):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            n = (y2 - y1, x1 - x2)
            L = math.hypot(*n)
            n = (n[0] / L, n[1] / L)
            pp = [v[0] * n[0] + v[1] * n[1] for v in p]
            qq = [v[0] * n[0] + v[1] * n[1] for v in q]
            best = max(best, max(min(qq) - max(pp), min(pp) - max(qq)))
    return best


def _point_segment(v, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((v[0] - a[0]) * ax + (v[1] - a[1]) * ay) / (ax * ax + ay * ay)))
    return math.hypot(v[0] - a[0] - t * ax, v[1] - a[1] - t * ay)


def gap(p, q):
    '''Distance between two convex polygons [mm]; negative = they overlap (by about that much).'''
    sat = _sat(p, q)
    if sat < 0:
        return sat
    return min(_point_segment(v, poly[i], poly[(i + 1) % len(poly)])
               for (a, poly) in ((p, q), (q, p)) for v in a for i in range(len(poly)))


def gap_ends(apex, width, a_dir_deg, b_dir_deg, t=WALL_T):
    '''Ends of two walls that would meet at `apex`, each pulled back along its own line until their
    faces are `width` apart (so the walls stay on their lines). The walls leave the apex at the given
    headings; returns (end_a, end_b), the wall ends nearest the apex.'''
    ha, hb = math.radians(a_dir_deg), math.radians(b_dir_deg)
    ua, ub = (math.cos(ha), math.sin(ha)), (math.cos(hb), math.sin(hb))
    lo, hi = 0.0, 500.0
    for _ in range(60):
        h = (lo + hi) / 2
        ea, eb = (apex[0] + h * ua[0], apex[1] + h * ua[1]), (apex[0] + h * ub[0], apex[1] + h * ub[1])
        wa = wall_corners(seg(ea, (ea[0] + 300 * ua[0], ea[1] + 300 * ua[1]), t))
        wb = wall_corners(seg(eb, (eb[0] + 300 * ub[0], eb[1] + 300 * ub[1]), t))
        lo, hi = (h, hi) if gap(wa, wb) < width else (lo, h)
    return [round(v, 1) for v in ea], [round(v, 1) for v in eb]


def slide_to_contact(walls, start, direction_deg, clearance=2.0, size=BOX, angle_deg=0.0):
    '''Slide a box from `start` (clear of every wall) along direction_deg; return the last spot
    still `clearance` clear of every wall, i.e. where it comes to rest against them.'''
    h = math.radians(direction_deg)
    polys = [wall_corners(w) for w in walls]
    last = None
    for k in range(4000):
        c = (start[0] + 0.5 * k * math.cos(h), start[1] + 0.5 * k * math.sin(h))
        if min(gap(corners(c, (size, size), angle_deg), w) for w in polys) < clearance:
            if last is None:
                raise ValueError("the start is not clear of the walls")
            return [round(v, 1) for v in last]
        last = c
    raise ValueError("never reached a wall")


def env(name, category, description, start, goal, walls=(), boxes=(), planner=None, notes=""):
    return {"name": name, "category": category, "description": description, "board_mm": list(BOARD),
            "walls": list(walls), "boxes": list(boxes),
            "start": {"xy": list(start[:2]), "heading_deg": start[2]},
            "goal": goal, "planner": planner or {}, "notes": notes}


def box_goal(center, radius, box_index=0, **extra):
    return dict({"type": "box", "box": box_index, "center": [round(v, 1) for v in center], "radius": radius}, **extra)


def tip_goal(center, radius):
    return {"type": "tip", "center": list(center), "radius": radius}


ENVS = []

# ---- 1. block to goal, one vine, simple
S1, H1 = (150.0, 150.0), 37.0
ENVS.append(env("1a", "b2g_unretractable", "one block, straight push into a goal just past it",
                (*S1, H1), box_goal(along(S1, H1, 680), 60), boxes=[box(along(S1, H1, 520), H1)]))
ENVS.append(env("1b", "b2g_unretractable", "two blocks in a row, pushed together; the front one to the goal",
                (*S1, H1), box_goal(along(S1, H1, 770), 60),
                boxes=[box(along(S1, H1, 620), H1), box(along(S1, H1, 520), H1)],
                notes="box 0 is the front block (the one with the goal); the vine meets box 1 first"))
ENVS.append(env("1c", "b2g_unretractable", "precise block pose: a short push that also turns the block 20 deg",
                (*S1, H1), box_goal(along(S1, H1, 590, lateral=15), 20, angle_deg=H1 + 20, angle_tol_deg=7),
                boxes=[box(along(S1, H1, 520), H1)],
                notes="goal: center within 20 mm and heading within 7 deg (mod 90) of the goal pose"))

# ---- 2. block to goal, one vine, complex: walls guide the block
A, B = (430.0, 800.0), (1010.0, 320.0)
c2a, ang2a = against_wall(A, B, 0.55, side=-1)
g2a, _ = against_wall(A, B, 0.08, side=-1)
ENVS.append(env("2a", "b2g_unretractable", "a diagonal wall guides the pushed block up-left to the goal",
                (720.0, 60.0, 90.0), box_goal(g2a, 70), walls=[seg(A, B)], boxes=[box(c2a, ang2a)],
                planner={"max_bodies": 50}))
ENVS.append(env("2b", "b2g_unretractable",
                "one block inside a channel; one push drives it into the curved back wall, which guides it up "
                "and left along the top wall to the goal",
                (200.0, 80.0, 0.0), box_goal((300.0, 640.0), 80),
                walls=[seg((230, 720), (820, 720)),                       # top wall, goal under its left end
                       # back guiding wall: up the right side and curving into the top wall
                       seg((1010, 150), (990, 420)), seg((990, 420), (950, 590)),
                       seg((950, 590), (890, 680)), seg((890, 680), (820, 720)),
                       seg((520, 130), (760, 520))],                      # diagonal wall, the channel's left side
                boxes=[box((760, 300), 0)],
                planner={"max_bodies": 70},
                notes="corrected 2026-10-20: one block, inside the channel (right of the diagonal wall); "
                      "the back wall is a chain of segments so the block slides around the bend instead of jamming"))

# ---- 3. block to goal with resets
ENVS.append(env("3a", "b2g_retractable", "open board, the goal far from the block: several pushes",
                (150.0, 120.0, 40.0), box_goal((980, 760), 80), boxes=[box((520, 400), 20)]))
ENVS.append(env("3b", "b2g_retractable",
                "maze; small openings (narrower than a block, wider than the vine) let the vine reach blocks in corners",
                (396.0, 809.0, 180.0), box_goal((1060, 183), 90),
                walls=[seg((210, 652), (660, 652)),                         # under the block's row
                       seg((548, 678), (548, 740)), seg((548, 810), (548, 900)),   # opening 740-810
                       seg((659, 339), (659, 665)),
                       seg((925, 222), (925, 587)),
                       seg((443, 203), (925, 203)),
                       seg((408, 300), (408, 457)), seg((408, 183), (408, 230)),   # opening 230-300
                       seg((687, 0), (687, 183))],
                boxes=[box((210, 770), 0)],
                planner={"max_bodies": 70},
                notes="geometry measured off the sketch; openings are 70 mm (vine 48.5 mm wide, block 98 mm)"))

# ---- 4. vine to goal
ENVS.append(env("4a", "vine_to_goal", "no blocks: S-shaped path around two interleaved walls",
                (280.0, 159.0, 90.0), tip_goal((973, 741), 93),
                walls=[seg((453, 0), (453, 503)), seg((760, 304), (760, 900))],
                planner={"max_curvature": 7.0, "max_bodies": 90}))
ENVS.append(env("4b", "vine_to_goal", "basic push: a block plugs the only gap in a dividing wall",
                (153.0, 143.0, 60.0), tip_goal((983, 562), 80),
                walls=[seg((472, 0), (472, 430)), seg((472, 573), (472, 900))],
                boxes=[box((472, 501), 0)],
                notes="gap 143 mm; the block leaves 22 mm on each side, too little for the vine"))
# 4c: a zig-zag barrier with a gap at each bend, narrower than a block but wider than the vine.
# Each bend is a 90 deg corner and each block sits flush in the corner beside its gap (turned 45
# deg). The upper corner opens toward the start: every push the vine can make there drives that
# block into a wall, so it stays jammed in its gap. The lower block is in the corner on the goal
# side: the vine grows through the lower gap and pushes it straight out of its corner.
GAP_4C = 70.0
P_UP, P_LO = (790.0, 600.0), (490.0, 300.0)
top_end, mid_top = gap_ends(P_UP, GAP_4C, 135.0, 225.0)
mid_lo, bot_end = gap_ends(P_LO, GAP_4C, 45.0, -45.0)
walls_4c = [seg((490, 900), top_end), seg(mid_top, mid_lo), seg(bot_end, (790, 0))]
ENVS.append(env("4c", "vine_to_goal",
                "zig-zag barrier, two 70 mm gaps; the upper block is jammed in its corner, only the lower one can be pushed clear",
                (150.0, 780.0, -90.0), tip_goal((1020, 480), 90), walls=walls_4c,
                boxes=[box(slide_to_contact(walls_4c, (P_LO[0] + 250, P_LO[1]), 180.0, angle_deg=45.0), 45.0),
                       box(slide_to_contact(walls_4c, (P_UP[0] - 250, P_UP[1]), 0.0, angle_deg=45.0), 45.0)],
                planner={"max_bodies": 60},
                notes=f"gaps {GAP_4C:.0f} mm between the walls' faces (vine 48.5 mm, block 98 mm); both bends are 90 deg "
                      "corners with a block flush in them, 2 mm clear. Box 1 (upper) is in the corner that opens toward "
                      "the start, so any push drives it into a wall; box 0 (lower) is in the corner on the goal side, so "
                      "the vine grows through the lower gap and pushes it out"))
left = [(470.0, 64.3 + 128.6 * k) for k in range(7)]
right = [(575.0, 128.6 * (k + 1)) for k in range(6)]
ENVS.append(env("4d", "vine_to_goal", "clutter: two staggered columns of blocks wall off the goal",
                (150.0, 150.0, 30.0), tip_goal((1000, 760), 90),
                boxes=[box(c) for c in left + right], planner={"max_bodies": 60},
                notes="13 blocks; gaps between them (31 mm) and to the board's edges (15 mm) are narrower than the vine"))


def check(e, min_gap=1.0):
    '''Boxes must not touch walls, each other or the board's edge at the start (the sim's contact
    constraints are infeasible from an overlapping start).'''
    W, H = e["board_mm"]
    boxes = [corners(b["center"], b["size"], b["angle_deg"]) for b in e["boxes"]]
    walls = [wall_corners(w) for w in e["walls"]]
    problems = []
    for i, bx in enumerate(boxes):
        if min(x for x, _ in bx) < min_gap or max(x for x, _ in bx) > W - min_gap or \
           min(y for _, y in bx) < min_gap or max(y for _, y in bx) > H - min_gap:
            problems.append(f"box {i} touches the board's edge")
        for j, wc in enumerate(walls):
            if gap(bx, wc) < min_gap:
                problems.append(f"box {i} touches wall {j} ({gap(bx, wc):.1f} mm)")
        for k in range(i):
            if gap(bx, boxes[k]) < min_gap:
                problems.append(f"box {i} touches box {k} ({gap(bx, boxes[k]):.1f} mm)")
    return problems


if __name__ == "__main__":
    bad = {e["name"]: check(e) for e in ENVS}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f"overlaps at the start: {bad}")
    for e in ENVS:
        path = os.path.join(HERE, e["name"] + ".json")
        with open(path, "w") as f:
            json.dump(e, f, indent=1)
        print("wrote", os.path.relpath(path))
