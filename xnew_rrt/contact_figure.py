'''
Paper figure: simulation vs. real video on the held-out "simple contact" trials (split "val" in
video/manifest.json: vine_block_06, block_block_08, block_wall_07).

For each trial, a few snapshots by vine length (just before contact, evenly spaced in length, end): the rectified top-down
video frame on top, the simulated state below, both cropped to the same board-frame mm window. The
simulation is fit_sim.run_scenario with the checkpoint's parameters (same scenario setup and the same
equal-vine-length comparison as the fit); the sim state shown under a video frame is the one whose
vine has the same length as the tracked one in that frame. The tracked centerline and box outlines
are drawn dashed on the sim panels.

    xnew_rrt/py.sh xnew_rrt/contact_figure.py                       # CPU, writes out/paper_figs/contact/
    xnew_rrt/py.sh xnew_rrt/contact_figure.py --device cuda --ckpt checkpoints/spin8_manual.json
'''

import argparse, json, math, os, pickle, sys, warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "video"))

import cv2
import fit_sim
import launch_fix
import checkpoint
from fit_to_master import arc_length, load_reference, shape_error, MIN_COMPARE_MM
from track import load_tracks
from calibrate import Calibration, read_frame

TRIALS = ["vine_block_06", "block_block_08", "block_wall_07"]
LABEL = {"vine_block_06": "Vine pushes block", "block_block_08": "Block pushes block",
         "block_wall_07": "Block hits wall", "puzzle_03": "Puzzle push (val 1)", "puzzle_08": "Puzzle push (val 2)"}
VINE, BOX, WALL_FACE, WALL_EDGE, REF = "#2a78d6", "#eb6834", "#d6d4cc", "#898781", "#0b0b0b"
RES_MM = 1.0           # rectified image resolution
LAUNCH_L, LAUNCH_W = 90.0, 48.0     # base marker [mm], as paper_figs.LAUNCH_L, LAUNCH_W
WINDOW_MM, WINDOW_FRAC = 40.0, 0.15   # sim states within max(these) of the tracked vine length are matched (match_sim)


# ------------------------------------------------------------------ data

def simulate(sc, cfg, device):
    '''Sim centerline [mm] and box poses at every reference frame (equal vine length), recorded by
    hooking the comparison functions fit_sim.run_scenario calls, plus its summary dict.'''
    frame_of = {id(f): k for k, f in enumerate(sc["frames"])}
    box_of = {id(b): k for k, b in enumerate(sc["boxes"] or [])}
    polys, boxes = {}, {}
    orig_shape, orig_box = fit_sim.shape_error, fit_sim.box_heading_error

    def hook_shape(ref, poly):
        polys.setdefault(frame_of[id(ref)], np.array(poly, float))
        return orig_shape(ref, poly)

    def hook_box(sim_pose, ref_pose, sizes):
        boxes.setdefault(box_of[id(ref_pose)], np.array(sim_pose, float))
        return orig_box(sim_pose, ref_pose, sizes)

    fit_sim.shape_error, fit_sim.box_heading_error = hook_shape, hook_box
    try:
        res = fit_sim.run_scenario(sc, cfg, {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT},
                                   device=device, use_graph=device != "cpu")[0]
    finally:
        fit_sim.shape_error, fit_sim.box_heading_error = orig_shape, orig_box
    return polys, boxes, res


def simulate_dense(sc, cfg, device, step_mm=5.0, extra_mm=150.0):
    '''Sim centerline [mm] and box poses every step_mm of sim vine length, by running fit_sim.run_scenario
    against a dummy reference of straight frames of those lengths (it only uses their lengths to decide when
    to compare). Returns {sim length: (poly, boxes)}.'''
    x, y, h = sc["start"]
    top = max(arc_length(f)[-1] for f in sc["frames"]) + extra_mm
    Ls = np.arange(MIN_COMPARE_MM, top, step_mm)
    dense = dict(sc, frames=[np.array([[x, y], [x + L * math.cos(h), y + L * math.sin(h)]]) for L in Ls],
                 boxes=[np.array(sc["boxes"][0], float).copy() for _ in Ls] if sc["objects"] else sc["boxes"])
    polys, boxes, _ = simulate(dense, cfg, device)
    return {float(Ls[k]): (polys[k], boxes.get(k)) for k in polys}


def curve_error(ref, poly, step_mm=5.0):
    '''Symmetric mean distance [mm] between two centerlines (each one's points to the other curve, averaged),
    so a sim vine shorter or longer than the tracked one pays for the missing or extra part (shape_error only
    compares up to the shorter of the two lengths).'''
    def pts(c):
        s = arc_length(c)
        t = np.arange(0.0, s[-1] + 1e-9, step_mm)
        return np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1)

    def to_curve(P, c):
        a, b = c[:-1], c[1:]
        ab = b - a
        t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12)[None], 0, 1)
        return np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)

    ref, poly = np.asarray(ref, float), np.asarray(poly, float)
    return float(0.5 * (to_curve(pts(ref), poly).mean() + to_curve(pts(poly), ref).mean()))


def match_sim(d, k):
    '''The sim state shown with (and scored against) reference frame k: among the dense sim states whose vine
    length is within the window around the tracked length (the tracked length itself depends on the
    segmentation, which can over- or under-shoot), the one with the least vine + block error.'''
    sc, Lk = d["sc"], vine_length(d, k)
    w = max(WINDOW_MM, WINDOW_FRAC * Lk)
    best = None
    for L, (poly, boxes) in d["dense"].items():
        if abs(L - Lk) > w:
            continue
        ve = curve_error(sc["frames"][k], poly)
        be = float(np.mean(np.linalg.norm(boxes[:, :2] - sc["boxes"][k][:, :2], axis=1))) if sc["objects"] else 0.0
        if best is None or ve + be < best[0]:
            best = (ve + be, L, poly, boxes, ve, be)
    if best is None:                       # nothing in the window: fall back to the equal-length state
        poly, boxes = d["polys"][k], d["boxes"].get(k)
        ve = curve_error(sc["frames"][k], poly)
        be = float(np.mean(np.linalg.norm(boxes[:, :2] - sc["boxes"][k][:, :2], axis=1))) if sc["objects"] else 0.0
        return dict(L=Lk, poly=poly, boxes=boxes, vine_mm=ve, box_mm=be)
    _, L, poly, boxes, ve, be = best
    return dict(L=L, poly=poly, boxes=boxes, vine_mm=ve, box_mm=be)


def suspect_frames(d, jump=2.5):
    '''Reference frames whose tracked length jumps by more than `jump` x the median frame-to-frame growth
    (the segmentation lost or hallucinated part of the vine); these and every later frame are left out.'''
    L = np.array([arc_length(f)[-1] for f in d["sc"]["frames"]])
    dL = np.diff(L)
    med = np.median(dL[dL > 0])
    bad = [i + 1 for i in np.nonzero(np.abs(dL) > jump * med)[0] if i + 1 >= min(d["polys"])]
    return set(range(bad[0], len(L))) if bad else set()


def matched_errors(d):
    '''Trial means of the window-matched vine and block errors over the compared, unsuspect frames.'''
    ks = [k for k in sorted(d["polys"]) if k not in d["suspect"]]
    m = [match_sim(d, k) for k in ks]
    return dict(vine_mm=float(np.mean([r["vine_mm"] for r in m])), box_mm=float(np.mean([r["box_mm"] for r in m])),
                frames=len(ks))


def frame_times(sc, tracks):
    '''Video frame index and time [s] of every reference frame (matched by centerline).'''
    out = []
    for f in sc["frames"]:
        hit = next(i for i, c in enumerate(tracks["centerlines"])
                   if c.shape == f.shape and np.allclose(c.astype(np.float32), f, atol=1e-3))
        out.append((int(tracks["frame"][hit]), float(tracks["t_s"][hit])))
    return out


def pick_snapshots(d, n=4, moved_mm=8.0, end_frac=0.8):
    '''Reference frame indices among those the sim reached (fit_sim compares only frames with a vine of
    at least MIN_COMPARE_MM) and not suspect: the last before any box has moved moved_mm, evenly spaced
    ones, and one end_frac of the way (in vine length) from that to the last.'''
    ks = sorted(set(d["polys"]) & (set(d["boxes"]) if d["sc"]["objects"] else set(d["polys"])))
    ks = [k for k in ks if k not in d["suspect"]]
    B = np.array(d["sc"]["boxes"])
    disp = np.linalg.norm(B[:, :, :2] - B[:1, :, :2], axis=2).max(1)
    first, last = ks[0], ks[-1]
    moved = [k for k in ks if disp[k] >= moved_mm]
    onset = max(first, ks[ks.index(moved[0]) - 1]) if moved else ks[len(ks) // 2]
    L = {k: vine_length(d, k) for k in ks}
    near = lambda x: min(ks, key=lambda k: abs(L[k] - x))     # evenly spaced in vine length, not time
    last = near(L[onset] + end_frac * (L[last] - L[onset]))
    picks = [onset] + [near(L[onset] + (L[last] - L[onset]) * i / (n - 1)) for i in range(1, n - 1)] + [last]
    if len(set(picks)) < len(picks):         # contact at the first compared frame: spread evenly instead
        picks = [ks[int(round(i))] for i in np.linspace(0, len(ks) - 1, n)]
    return sorted(set(picks))


def pick_regions(d, n=4, spike=0.15, halfwidth=0.3):
    '''Long runs: n evenly spaced times over the compared span (first to last compared frame, in video time), and
    in the region within halfwidth x their spacing of each, pick the reference frame whose window-matched sim
    state has the least vine + block error.
    Only single frames whose tracked length is off the median of their +-2 neighbours by more than `spike`
    are dropped (not, as suspect_frames does, everything after the first jump).'''
    ks = sorted(set(d["polys"]) & (set(d["boxes"]) if d["sc"]["objects"] else set(d["polys"])))
    L = np.array([vine_length(d, k) for k in ks])
    ks = [k for i, k in enumerate(ks) if abs(L[i] - np.median(L[max(0, i - 2):i + 3])) <= spike * L[i]]
    t = np.array([d["times"][k][1] for k in ks])
    centers = np.linspace(t[0], t[-1], n)
    half = halfwidth * (centers[1] - centers[0])
    picks = []
    for c in centers:
        inside = [k for k, tk in zip(ks, t) if abs(tk - c) <= half and k not in picks]
        if inside:
            picks.append(min(inside, key=lambda k: (lambda m: m["vine_mm"] + m["box_mm"])(match_sim(d, k))))
    return picks


PICK = [pick_snapshots]                   # main() swaps in pick_regions with --pick region


def vine_length(d, k):
    '''Tracked vine length [mm] at reference frame k (the sim is compared at the same length).'''
    return float(arc_length(d["sc"]["frames"][k])[-1])


def box_corners(pose, size):
    cx, cy, th = pose
    c, s = math.cos(th), math.sin(th)
    return np.array([(cx + c * lx - s * ly, cy + s * lx + c * ly)
                     for lx, ly in ((-size[0] / 2, -size[1] / 2), (size[0] / 2, -size[1] / 2),
                                    (size[0] / 2, size[1] / 2), (-size[0] / 2, size[1] / 2))])


def valid_mask(d):
    '''Rectified-board pixels the camera saw (undistortion and the view leave the rest black).'''
    cal = d["cal"]
    w, h = cal.image_size
    white = np.full((int(h), int(w)), 255, np.uint8)
    if cal.K is not None:
        white = cv2.undistort(white, cal.K, cal.dist)
    W, Hb = cal.board_mm
    size = (int(round(W / RES_MM)), int(round(Hb / RES_MM)))
    m = cv2.warpPerspective(white, cal.rect_matrix(RES_MM) @ cal.H, size, flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.erode(m, np.ones((5, 5), np.uint8)) > 0


def clamp_window(win, d, aspect):
    '''Shift, then shrink (keeping the aspect) the window until it lies in what the camera saw.'''
    ok = valid_mask(d)
    Hb = d["cal"].board_mm[1]
    def bad(w):
        c0, c1 = int(math.floor(w[0] / RES_MM)), int(math.ceil(w[1] / RES_MM))
        r0, r1 = int(math.floor((Hb - w[3]) / RES_MM)), int(math.ceil((Hb - w[2]) / RES_MM))
        if c0 < 0 or r0 < 0 or c1 > ok.shape[1] or r1 > ok.shape[0]:
            return 1.0
        return 1.0 - ok[r0:r1, c0:c1].mean()
    win = list(win)
    for _ in range(400):
        if bad(win) < 1e-3:
            return win
        best = None
        for dx, dy in ((5, 0), (-5, 0), (0, 5), (0, -5)):
            w2 = [win[0] + dx, win[1] + dx, win[2] + dy, win[3] + dy]
            if best is None or bad(w2) < best[0]:
                best = (bad(w2), w2)
        if best[0] < bad(win):
            win = best[1]
        else:                                  # shrink about the center
            cx, cy = (win[0] + win[1]) / 2, (win[2] + win[3]) / 2
            hw, hh = 0.98 * (win[1] - win[0]) / 2, 0.98 * (win[3] - win[2]) / 2
            win = [cx - hw, cx + hw, cy - hh, cy + hh]
    return win


def window(sc, polys, boxes, margin=70.0, aspect=None):
    '''[x0, x1, y0, y1] around everything that happens (real and sim), padded to `aspect` (h / w).'''
    pts = [np.array(sc["start"][:2])[None]]
    pts += list(sc["frames"]) + list(polys.values())
    for B in list(sc["boxes"]) + list(boxes.values()):
        for pose, o in zip(B, sc["objects"]):
            pts.append(box_corners(pose, o["size_mm"]))
    P = np.vstack(pts)
    x0, y0 = P.min(0) - margin
    x1, y1 = P.max(0) + margin
    if aspect is not None:
        w, h = x1 - x0, y1 - y0
        if h / w < aspect:
            d = (aspect * w - h) / 2; y0 -= d; y1 += d
        else:
            d = (h / aspect - w) / 2; x0 -= d; x1 += d
    return [x0, x1, y0, y1]


def load_trial_data(name, cfg, device, ref_path, cache=None, fix_launch=True):
    sc = load_reference(ref_path)[name]
    if fix_launch:                          # the launch from the vine's own line before contact (launch_fix.py)
        launch_fix.apply({name: sc})
    tracks = load_tracks(os.path.join(REPO, "out", "video", name, "tracks.npz"))
    if cache is not None and os.path.exists(cache):
        with open(cache, "rb") as f:
            polys, boxes, res, dense = pickle.load(f)
    else:
        polys, boxes, res = simulate(sc, cfg, device)
        dense = simulate_dense(sc, cfg, device)
        if cache is not None:
            with open(cache, "wb") as f:
                pickle.dump((polys, boxes, res, dense), f)
    times = frame_times(sc, tracks)
    cal = Calibration.from_dict(tracks["meta"]["calibration"])
    d = dict(name=name, sc=sc, tracks=tracks, polys=polys, boxes=boxes, res=res, dense=dense, times=times, cal=cal,
             video=sc["source"]["video"], radius_mm=cfg.vine_radius_m * 1e3)
    d["suspect"] = suspect_frames(d)
    d["matched"] = matched_errors(d)
    return d


# ------------------------------------------------------------------ drawing

def tube(poly, r):
    '''Outline of a tube of radius r around a polyline (offset normals; round tip cap separately).'''
    P = np.asarray(poly, float)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
    P = P[keep]
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return np.vstack([P + r * nrm, (P - r * nrm)[::-1]])


# trials whose parsed board-frame walls (walls_mm) are a parsing artifact: not drawn (the sim never reaches them)
HIDE_FRAME = {"vine_block_06", "block_block_08"}


def draw_walls(ax, sc, name=None):
    import obstacle_style
    for w in ([] if name in HIDE_FRAME else sc["walls_mm"]):
        obstacle_style.add_obstacle(ax, [(w[0], w[1]), (w[2], w[1]), (w[2], w[3]), (w[0], w[3])],
                                    min(w[2] - w[0], w[3] - w[1]), lw=0.6)
    for cx, cy, th, hw, hh in sc.get("walls_obb_mm", []):
        obstacle_style.add_obstacle(ax, box_corners((cx, cy, th), (2 * hw, 2 * hh)), 2 * min(hw, hh), th, lw=0.6)


def draw_sim(ax, d, k, overlay=True):
    sc = d["sc"]
    ax.set_facecolor("white")
    draw_walls(ax, sc, d["name"])
    r = d["radius_mm"]
    m = match_sim(d, k)
    if m["boxes"] is not None:
        for pose, o in zip(m["boxes"], sc["objects"]):
            ax.add_patch(Polygon(box_corners(pose, o["size_mm"]), closed=True, facecolor=BOX, alpha=0.85,
                                 edgecolor="#9c3d14", lw=0.6, zorder=3))
    if m["poly"] is not None:
        P = m["poly"]
        ax.add_patch(Polygon(tube(P, r), closed=True, facecolor=VINE, alpha=0.45, edgecolor=VINE, lw=0.6, zorder=4))
        ax.add_patch(Circle(P[-1], r, facecolor=VINE, alpha=0.45, edgecolor="none", zorder=4))
        ax.plot(*P.T, color=VINE, lw=0.8, zorder=5)
    if overlay:
        ax.plot(*sc["frames"][k].T, color=REF, lw=0.8, ls=(0, (2.5, 1.5)), zorder=6)
        for pose, o in zip(sc["boxes"][k], sc["objects"]):
            ax.add_patch(Polygon(box_corners(pose, o["size_mm"]), closed=True, fill=False, edgecolor=REF,
                                 lw=0.8, ls=(0, (2.5, 1.5)), zorder=6))
    # the base: a bar behind it along the launch heading (as paper_figs' overview)
    x, y, th = sc["start"]
    u, w = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    ax.add_patch(Polygon([np.array([x, y]) + a * LAUNCH_L * u + b * LAUNCH_W / 2 * w
                          for a, b in ((-1, -1), (0, -1), (0, 1), (-1, 1))], closed=True, facecolor=REF,
                         edgecolor="none", zorder=7))


_frame_cache = {}


def draw_real(ax, d, k, win):
    idx, _ = d["times"][k]
    key = (d["name"], idx)
    if key not in _frame_cache:
        raw = read_frame(d["video"], idx)
        rect = d["cal"].rectify(raw, RES_MM)
        _frame_cache[key] = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
    img = _frame_cache[key]
    W, H = d["cal"].board_mm
    ax.imshow(img, extent=[0, W, 0, H], origin="upper", interpolation="bilinear", zorder=0)


def style(ax, win):
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#888888")


def scale_bar(ax, win, mm=100, fs=6, color="black"):
    x = win[0] + 0.05 * (win[1] - win[0])
    y = win[2] + 0.1 * (win[3] - win[2])
    ax.plot([x, x + mm], [y, y], color=color, lw=1.5, solid_capstyle="butt", zorder=10)
    ax.text(x + mm / 2, y + 0.03 * (win[3] - win[2]), f"{mm / 10:.0f} cm", color=color, fontsize=fs,
            ha="center", va="bottom", zorder=10)


def trial_block(fig, gs_rows, d, snaps, win, fs, show_row_labels=True, title=True):
    '''Two rows of axes (real, sim) for one trial in the gridspec rows given.'''
    axes = []
    for j, k in enumerate(snaps):
        ar = fig.add_subplot(gs_rows[0, j]); asim = fig.add_subplot(gs_rows[1, j])
        draw_real(ar, d, k, win); draw_sim(asim, d, k)
        style(ar, win); style(asim, win)
        m = match_sim(d, k)
        ar.set_title(f"L = {vine_length(d, k):.0f} mm", fontsize=fs, pad=2)
        asim.set_xlabel(f"L = {m['L']:.0f} mm: vine {m['vine_mm']:.0f}, block {m['box_mm']:.0f} mm",
                        fontsize=fs - 1.5, labelpad=1.5, color="#333333")
        if j == 0 and show_row_labels:
            ar.set_ylabel("Real", fontsize=fs, labelpad=2)
            asim.set_ylabel("Sim", fontsize=fs, labelpad=2)
        if j == 0:
            scale_bar(asim, win, fs=fs - 1.5)
        axes.append((ar, asim))
    return axes


def legend_handles():
    return [Patch(facecolor=VINE, alpha=0.45, edgecolor=VINE, label="sim vine"),
            Patch(facecolor=BOX, alpha=0.85, edgecolor="#9c3d14", label="sim block"),
            Line2D([], [], color=REF, lw=0.8, ls=(0, (2.5, 1.5)), label="tracked (video)"),
            Patch(facecolor=WALL_FACE, edgecolor="#52514e", hatch="///", label="wall")]


def legend_below(fig, axes_rows, fs, gap_in=0.22, **kw):
    '''Figure legend just under the lowest axes (past their x labels).'''
    y0 = min(a.get_position().y0 for a in fig.axes)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, y0 - gap_in / fig.get_figheight()), bbox_transform=fig.transFigure, **kw)


def save(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path + ".{pdf,png}")


def combined(data, out, width_in, n_snaps, fs):
    '''All trials: per trial a real row and a sim row, columns = snapshots.'''
    aspect = 0.55
    ncol = n_snaps
    pw = (width_in - 0.25) / ncol
    ph = pw * aspect
    nt = len(data)
    height = nt * (2 * ph + 0.55)
    fig = plt.figure(figsize=(width_in, height))
    outer = fig.add_gridspec(nt, 1, hspace=0.3, left=0.04, right=0.995, top=0.96, bottom=0.03)
    for i, d in enumerate(data):
        snaps = PICK[0](d, n_snaps)
        win = clamp_window(window(d["sc"], d["polys"], d["boxes"], aspect=aspect), d, aspect)
        gs = outer[i].subgridspec(2, len(snaps), wspace=0.04, hspace=0.04 if width_in > 5 else 0.12)
        axes = trial_block(fig, gs, d, snaps, win, fs)
        r = d["matched"]
        axes[0][0].annotate(f"({chr(97 + i)}) {LABEL.get(d["name"], d["name"])}" + (f" (mean error: vine {r['vine_mm']:.0f} mm, "
                            f"block {r['box_mm']:.0f} mm)" if width_in > 5 else ""),
                            xy=(0, 1), xycoords="axes fraction", xytext=(0, 3 + fs * 1.3),
                            textcoords="offset points", fontsize=fs, fontweight="bold", ha="left", va="bottom")
    legend_below(fig, None, fs - 0.5, gap_in=0.05 + fs / 72 * 1.2)
    save(fig, out)


def single(d, out, width_in, n_snaps, fs):
    aspect = 0.6
    snaps = PICK[0](d, n_snaps)
    pw = (width_in - 0.2) / len(snaps)
    H = 2 * pw * aspect + 0.75
    fig = plt.figure(figsize=(width_in, H))
    gs = fig.add_gridspec(2, len(snaps), wspace=0.04, hspace=0.04 if width_in > 5 else 0.12,
                          left=0.05, right=0.995, top=1 - 0.2 / H, bottom=0.45 / H)
    win = clamp_window(window(d["sc"], d["polys"], d["boxes"], aspect=aspect), d, aspect)
    trial_block(fig, gs, d, snaps, win, fs)
    legend_below(fig, None, fs - 1, gap_in=0.05 + fs / 72 * 1.2, handlelength=1.5, columnspacing=0.8)
    save(fig, out)


SHORT = {"vine_block_06": "vine-block", "block_block_08": "block-block", "block_wall_07": "block-wall"}


def pair_block(fig, gs_pairs, d, snaps, win, fs):
    '''One trial as (real | sim) pairs side by side, one pair per snapshot; error in the sim panel's title.'''
    axes = []
    for j, k in enumerate(snaps):
        g = gs_pairs[j].subgridspec(1, 2, wspace=0.03)
        ar, asim = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        draw_real(ar, d, k, win); draw_sim(asim, d, k)
        style(ar, win); style(asim, win)
        m = match_sim(d, k)
        ar.set_title(f"real, L = {vine_length(d, k):.0f} mm", fontsize=fs - 1, pad=1.5)
        asim.set_title(f"sim: vine {m['vine_mm']:.0f}, block {m['box_mm']:.0f} mm", fontsize=fs - 1, pad=1.5)
        axes.append((ar, asim))
    scale_bar(axes[0][1], win, fs=fs - 2)
    return axes


def combined_pairs(data, out, width_in, n_snaps, fs, aspect=0.6):
    '''Shorter double-column layout: one row per trial, n_snaps (real | sim) pairs across.'''
    nt = len(data)
    pw = (width_in - 0.3) / (2 * n_snaps)
    height = nt * (pw * aspect + 0.2) + 0.3
    fig = plt.figure(figsize=(width_in, height))
    outer = fig.add_gridspec(nt, n_snaps, hspace=0.2 / (pw * aspect), wspace=0.08,
                             left=0.035, right=0.998, top=1 - 0.2 / height, bottom=0.25 / height)
    for i, d in enumerate(data):
        snaps = PICK[0](d, n_snaps)
        win = clamp_window(window(d["sc"], d["polys"], d["boxes"], aspect=aspect), d, aspect)
        axes = pair_block(fig, [outer[i, j] for j in range(len(snaps))], d, snaps, win, fs)
        axes[0][0].set_ylabel(f"({chr(97 + i)}) {SHORT.get(d['name'], d['name'])}", fontsize=fs - 0.5, labelpad=2)
    legend_below(fig, None, fs - 0.5, gap_in=0.02)
    save(fig, out)


def combined_columns(data, out, width_in, n_snaps, fs, aspect=0.6, gap_in=0.04):
    '''Double column, bare: one column per trial (its label on top), rows = snapshots, each a (real | sim) pair;
    no per-panel text and no legend (the caption says what is drawn).'''
    nt = len(data)
    pw = (width_in - 0.1 * (nt - 1)) / (2 * nt)
    ph = pw * aspect
    head = fs / 72 * 1.6
    height = n_snaps * ph + (n_snaps - 1) * gap_in + head
    fig = plt.figure(figsize=(width_in, height))
    outer = fig.add_gridspec(n_snaps, nt, hspace=gap_in / ph, wspace=0.1 / (2 * pw),
                             left=0, right=1, top=1 - head / height, bottom=0)
    for i, d in enumerate(data):
        snaps = PICK[0](d, n_snaps)
        win = clamp_window(window(d["sc"], d["polys"], d["boxes"], aspect=aspect), d, aspect)
        for j, k in enumerate(snaps):
            g = outer[j, i].subgridspec(1, 2, wspace=0.03)
            ar, asim = fig.add_subplot(g[0]), fig.add_subplot(g[1])
            draw_real(ar, d, k, win); draw_sim(asim, d, k)
            style(ar, win); style(asim, win)
            if j == 0:
                scale_bar(asim, win, fs=fs - 2)
                box = outer[0, i].get_position(fig)
                fig.text((box.x0 + box.x1) / 2, box.y1 + 0.3 * head / height, SHORT.get(d["name"], d["name"]),
                         ha="center", va="bottom", fontsize=fs)
    save(fig, out)


def single_vertical(d, out, width_in, n_snaps, fs, aspect=0.6, bare=False):
    '''Single column: rows = snapshots in time, columns = real | sim.'''
    snaps = PICK[0](d, n_snaps)
    pw = (width_in - 0.2) / 2
    gap = 0.05 if bare else 0.17                    # [in] between rows: room for the titles unless bare
    H = len(snaps) * (pw * aspect + gap) + (0.05 if bare else 0.45)
    fig = plt.figure(figsize=(width_in, H))
    gs = fig.add_gridspec(len(snaps), 2, wspace=0.03, hspace=gap / (pw * aspect) * 1.1,
                          left=0.06, right=0.995, top=1 - (0.02 if bare else 0.2) / H, bottom=(0.02 if bare else 0.3) / H)
    win = clamp_window(window(d["sc"], d["polys"], d["boxes"], aspect=aspect), d, aspect)
    for i, k in enumerate(snaps):
        ar, asim = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        draw_real(ar, d, k, win); draw_sim(asim, d, k)
        style(ar, win); style(asim, win)
        m = match_sim(d, k)
        if not bare:
            ar.set_title(f"real, L = {vine_length(d, k):.0f} mm", fontsize=fs - 1, pad=1.5)
            asim.set_title(f"sim, L = {m['L']:.0f} mm: vine {m['vine_mm']:.0f}, block {m['box_mm']:.0f} mm",
                           fontsize=fs - 1, pad=1.5)
        ar.set_ylabel(f"t = {d['times'][k][1] - d['times'][0][1]:.0f} s", fontsize=fs - 1, labelpad=2)
        if i == 0:
            scale_bar(asim, win, fs=fs - 2)
    if not bare:
        legend_below(fig, None, fs - 1, gap_in=0.02, handlelength=1.5, columnspacing=0.8)
    save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "spin8_manual.json"))
    p.add_argument("--ref", default=os.path.join(HERE, "data", "contact_val.npz"))
    p.add_argument("--trials", nargs="+", default=TRIALS)
    p.add_argument("--prefix", default="contact_val", help="name of the combined figures")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=os.path.join(REPO, "out", "paper_figs", "contact"))
    p.add_argument("--no-cache", action="store_true", help="re-run the sims (default: reuse <out>/cache/*.pkl)")
    p.add_argument("--no-launch-fix", dest="fix_launch", action="store_false",
                   help="use the reference's launch as is (launch_fix.py: forced through the detected base, it "
                        "tilts 5-10 deg on short pre-contact vines)")
    p.add_argument("--pick", choices=["length", "region"], default="length",
                   help="snapshot frames: evenly spaced in vine length (pick_snapshots) or the best-matching frame "
                        "in each of n equal time regions (pick_regions, for the long runs)")
    p.add_argument("--layout", choices=["rows", "pairs", "vertical", "columns"], default="rows",
                   help="also write <prefix>_pairs (one row of real|sim pairs per trial), contact_<trial>_vert "
                        "(single column, real|sim per row) or <prefix>_columns (one column per trial, bare)")
    p.add_argument("--bare", action="store_true", help="vertical layout: no per-panel titles and no legend")
    args = p.parse_args()
    if args.pick == "region":
        PICK[0] = pick_regions
    warnings.simplefilter("ignore")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    cfg = fit_sim.cfg_of(checkpoint.load(args.ckpt))
    data = []
    errors = {}
    for name in args.trials:
        cache = None if args.no_cache else os.path.join(
            args.out, "cache", f"{os.path.splitext(os.path.basename(args.ckpt))[0]}_{name}"
                               f"{'_launchfix' if args.fix_launch else ''}_dense.pkl")
        d = load_trial_data(name, cfg, args.device, args.ref, cache, args.fix_launch)
        r = d["res"]
        print(f"{name}: shape {r['shape_mm']:.1f} mm, box {r['box_mm']:.1f} mm, turn {r['box_deg']:.1f} deg, "
              f"{len(d['polys'])}/{len(d['sc']['frames'])} frames reached", flush=True)
        errors[name] = {k: float(r[k]) for k in ("shape_mm", "box_mm", "box_deg")}
        errors[name]["frames_compared"] = len(d["polys"])
        errors[name]["suspect_frames"] = sorted(d["suspect"])
        errors[name]["window_matched"] = d["matched"]
        print(f"   window-matched over {d['matched']['frames']} unsuspect frames: vine {d['matched']['vine_mm']:.1f} mm, "
              f"block {d['matched']['box_mm']:.1f} mm (suspect: {sorted(d['suspect'])})", flush=True)
        errors[name]["launch_fix"] = args.fix_launch
        errors[name]["snapshots"] = [dict(ref_frame=k, video_frame=d["times"][k][0], vine_mm=vine_length(d, k),
                                          sim_vine_mm=match_sim(d, k)["L"],
                                          t_s=d["times"][k][1] - d["times"][0][1]) for k in PICK[0](d, 4)]
        data.append(d)
        single(d, os.path.join(args.out, f"contact_{name}_col"), 3.5, 3, 7)
        single(d, os.path.join(args.out, f"contact_{name}_wide"), 7.16, 4, 8)
    if args.layout == "pairs":
        combined_pairs(data, os.path.join(args.out, f"{args.prefix}_pairs"), 7.16, 3, 7)
    if args.layout == "vertical":
        for d in data:
            single_vertical(d, os.path.join(args.out, f"contact_{d['name']}_vert" + ("_bare" if args.bare else "")),
                            3.5, 4, 7, bare=args.bare)
    if args.layout == "columns":
        combined_columns(data, os.path.join(args.out, f"{args.prefix}_columns"), 7.16, 3, 8)
    combined(data, os.path.join(args.out, f"{args.prefix}_wide"), 7.16, 4, 8)
    combined(data, os.path.join(args.out, f"{args.prefix}_col"), 3.5, 3, 7)
    with open(os.path.join(args.out, "errors.json"), "w") as f:
        json.dump(dict(checkpoint=os.path.relpath(args.ckpt, REPO), ref=os.path.relpath(args.ref, REPO),
                       trials=errors), f, indent=1)
    print("wrote", os.path.join(args.out, "errors.json"))


if __name__ == "__main__":
    main()
