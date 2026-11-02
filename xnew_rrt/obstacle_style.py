'''Paper-figure obstacles in the puzzle boards' style (block_puzzle.draw_board): a dark outline with hatching at 45 deg
to the obstacle's own axes (so an angled brick's hatching turns with it); thin walls outlined only (no hatching); the
board's border walls (touching the scene bounds) filled solid in the outline color, so they read as one frame rather
than separate rectangles.'''

import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Polygon

import viz

THIN_MM = 30.0       # walls at most this thick are outlined only


def add_obstacle(ax, corners, thickness_mm, angle=0.0, border=False, lw=1.2, spacing_mm=15.0, hatch_lw=0.35,
                 zorder=1):
    '''An obstacle with corners [mm] and orientation angle [rad]; hatch lines spacing_mm apart.'''
    if border:
        ax.add_patch(Polygon(corners, closed=True, facecolor=viz.INK_2, edgecolor=viz.INK_2, lw=lw, zorder=zorder))
        return
    face = ax.add_patch(Polygon(corners, closed=True, facecolor=viz.WALL_FACE, edgecolor="none", zorder=zorder))
    if thickness_mm > THIN_MM:
        P = np.asarray(corners, float)
        c = P.mean(0)
        R = np.linalg.norm(P - c, axis=1).max()
        a = angle + math.pi / 4
        u, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        offs = np.arange(-R, R + 1e-9, spacing_mm)
        segs = [[c + o * n - R * u, c + o * n + R * u] for o in offs]
        lc = ax.add_collection(LineCollection(segs, colors=viz.INK_2, linewidths=hatch_lw, zorder=zorder + 0.01))
        lc.set_clip_path(face)
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=viz.INK_2, lw=lw, zorder=zorder + 0.02))


def draw_static(ax, sim, lw=1.2, spacing_mm=15.0, borders=True):
    '''The scene's static boxes (viz.draw_scene's walls, drawn with walls=False); without borders, the border walls
    are left out (for a view cropped to the board, board_bounds).'''
    x0, x1, y0, y1 = [v * 1000 for v in sim.scene.bounds_m]
    for pose, hw, hh in sim.static_boxes():
        c = viz._box_corners_mm(pose, hw, hh)
        xs, ys = [float(p[0]) for p in c], [float(p[1]) for p in c]
        border = min(xs) <= x0 + 1 or max(xs) >= x1 - 1 or min(ys) <= y0 + 1 or max(ys) >= y1 - 1
        if border and not borders:
            continue
        add_obstacle(ax, c, 2 * viz.MM(min(float(hw), float(hh))), float(pose[2]), border, lw, spacing_mm)


def board_bounds(env):
    '''[x0, x1, y0, y1] of the board inside its border walls [mm].'''
    W, H = env["board_mm"]
    return [0.0, W, 0.0, H]


def legend_patch(label="wall"):
    return Patch(facecolor=viz.WALL_FACE, edgecolor=viz.INK_2, hatch="///", label=label)
