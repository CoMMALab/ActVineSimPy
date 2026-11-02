'''
Corrected launch (start) of the video reference scenarios.

video/to_reference.py fits the launch heading as the line THROUGH the detected base point that best fits
the vine on the last frame before a box moves. The detected base sits about 10 mm to the side of where
the tracked centerline starts, and when the vine touches a block almost at once (block_block, many
block_wall trials) only ~80-150 mm of vine is visible, so forcing the line through that point tilts it by
5-10 deg. Here the heading is the vine's own line fit on that frame (before any box moved still_mm) and
the base is moved onto that line (its distance along the line is kept).

    import launch_fix
    scs = launch_fix.apply(load_reference("xnew_rrt/data/contact_val.npz"))     # in place, also returned

apply() also replaces walls the video detection got wrong (WALL_FIX, read off the rectified frames by hand).
'''

import math, os, sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "video"))
from track import load_tracks  # noqa: E402


# trial -> {index in walls_obb_mm: [cx, cy, theta_rad, hw, hh]}
# block_wall_07: the panel was detected 8 deg off vertical; its face toward the box is a vertical line at x ~ 1012 mm
WALL_FIX = {"block_wall_07": {0: [1017.0, 304.0, math.pi / 2, 163.2, 5.0]}}


def _arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def corrected_start(tracks, still_mm=2.0, bent_mm=40.0, fit_mm=150.0):
    '''(x [mm], y [mm], heading [rad]) from the tracked vine on the last frame before any box moved still_mm.'''
    base = np.array(tracks["meta"]["base"][:2], float)
    ok = np.array(tracks["valid"], bool).copy()
    if len(tracks["meta"]["boxes"]):
        disp = np.linalg.norm(tracks["boxes"][:, :, :2] - tracks["boxes"][:1, :, :2], axis=2).max(axis=1)
        moved = np.nonzero(disp >= still_mm)[0]
        if len(moved):
            ok[moved[0]:] = False
    if not ok.any():
        return None
    P = np.asarray(tracks["centerlines"][np.nonzero(ok)[0][-1]], float)
    c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    if np.linalg.norm((P - c) - np.outer((P - c) @ u, u), axis=1).max() > bent_mm:
        # the vine has already bent (around a wall, in the long runs): fit only its first fit_mm, on the first
        # frame that long, instead (every simple-contact and puzzle trial is straight within 17 mm)
        k = next((i for i, cl in enumerate(tracks["centerlines"])
                  if tracks["valid"][i] and _arc(np.asarray(cl, float))[-1] >= fit_mm), None)
        if k is None:
            return None
        P = np.asarray(tracks["centerlines"][k], float)
        P = P[_arc(P) <= fit_mm]
        c = P.mean(0)
        u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    if u @ (P[-1] - P[0]) < 0:
        u = -u
    b = c + ((base - c) @ u) * u                     # the base, moved onto the line
    return float(b[0]), float(b[1]), float(math.atan2(u[1], u[0]))


def apply(scenarios, tracks_dir=os.path.join(REPO, "out", "video"), log=None):
    for name, sc in scenarios.items():
        for i, w in WALL_FIX.get(name, {}).items():
            if log:
                log(f"{name}: wall {i} {list(np.round(sc['walls_obb_mm'][i], 3))} -> {w}")
            sc["walls_obb_mm"] = [list(v) for v in sc["walls_obb_mm"]]
            sc["walls_obb_mm"][i] = list(w)
        path = os.path.join(tracks_dir, name, "tracks.npz")
        if not os.path.exists(path):
            continue
        s = corrected_start(load_tracks(path))
        if s is None:
            continue
        if log:
            x0, y0, h0 = sc["start"]
            log(f"{name}: start ({x0:.1f}, {y0:.1f}, {math.degrees(h0):+.1f} deg) -> "
                f"({s[0]:.1f}, {s[1]:.1f}, {math.degrees(s[2]):+.1f} deg)")
        sc["start"] = s
    return scenarios
