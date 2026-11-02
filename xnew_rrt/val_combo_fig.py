'''
Single-column paper figure of the two held-out multi-contact validations side by side, one row per time:
(a) the long-horizon run long_02 at the last three of contact_figure's four time regions (the first, before the
vine reaches a wall, is dropped), (b) the live 5b run, one row per push, at the push's end: the first frame where the real block is within
REST_MM of where it comes to rest, with the dense sim state that best agrees within the length window. Each cell is
a (real | sim) pair drawn as in contact_figure.py, cropped to that snapshot's vine and block. The sims come from
contact_figure.py's caches (run it first for both).

    xnew_rrt/py.sh xnew_rrt/val_combo_fig.py      # -> out/paper_figs/val_combo/val_combo.{pdf,png}
'''

import argparse, math, os, warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import contact_figure as cf
import fit_sim
import checkpoint
from contact_figure import box_corners, clamp_window, draw_real, draw_sim, match_sim, style, REPO, HERE

LONG = ("long_02", "long_fit", "long.npz", "long_fit")
REST_MM = 5.0
PUZZLE = [(f"demo_5b_push{k}", "spin8_manual", "demo_5b.npz", "demo_5b_spin8_manual") for k in (1, 2, 3)]


def load(name, ckpt, ref, out):
    cfg = fit_sim.cfg_of(checkpoint.load(os.path.join(REPO, "checkpoints", ckpt + ".json")))
    cache = os.path.join(REPO, "out", "paper_figs", out, "cache", f"{ckpt}_{name}_launchfix_dense.pkl")
    if not os.path.exists(cache):
        raise SystemExit(f"no cache {cache}: run contact_figure.py for {name} with {ckpt} first")
    return cf.load_trial_data(name, cfg, "cpu", os.path.join(HERE, "data", ref), cache)


def crop(d, k, aspect, margin=45.0, width=None):
    '''[x0, x1, y0, y1] around snapshot k's tracked and simulated vine and block (and the base), padded to aspect; or,
    with width [mm], that wide (at aspect) about its center.'''
    m = match_sim(d, k)
    sc = d["sc"]
    pts = [np.array(sc["start"][:2])[None], sc["frames"][k], m["poly"]]
    for B in (sc["boxes"][k], m["boxes"]):
        pts += [box_corners(p, o["size_mm"]) for p, o in zip(B, sc["objects"])]
    P = np.vstack(pts)
    x0, y0 = P.min(0) - margin
    x1, y1 = P.max(0) + margin
    w, h = x1 - x0, y1 - y0
    if h / w < aspect:
        e = (aspect * w - h) / 2; y0 -= e; y1 += e
    else:
        e = (h / aspect - w) / 2; x0 -= e; x1 += e
    if width is None:
        return clamp_window([x0, x1, y0, y1], d, aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return shift_into_view([cx - width / 2, cx + width / 2, cy - aspect * width / 2, cy + aspect * width / 2], d)


def shift_into_view(win, d, step=5.0, max_mm=300.0):
    '''Move a fixed-size window (keeping its size) toward the board's center until it lies in what the camera
    saw, or as far in as max_mm allows.'''
    ok = cf.valid_mask(d)
    Hb = d["cal"].board_mm[1]
    def bad(w):
        c0, c1 = int(math.floor(w[0] / cf.RES_MM)), int(math.ceil(w[1] / cf.RES_MM))
        r0, r1 = int(math.floor((Hb - w[3]) / cf.RES_MM)), int(math.ceil((Hb - w[2]) / cf.RES_MM))
        c0, r0 = max(c0, 0), max(r0, 0)
        seen = ok[r0:r1, c0:c1].mean() if r1 > r0 and c1 > c0 else 0.0
        out = (c1 - c0) * (r1 - r0)
        return 1.0 - seen * out / ((w[1] - w[0]) * (w[3] - w[2]) / cf.RES_MM ** 2)
    win = list(win)
    for _ in range(int(max_mm / step)):
        if bad(win) < 1e-3:
            break
        moves = [[win[0] + dx, win[1] + dx, win[2] + dy, win[3] + dy] for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]
        best = min(moves, key=bad)
        if bad(best) >= bad(win):
            break
        win = best
    return win


def bar(ax, win, fs, mm=100):
    '''A 10 cm scale bar in the cell's lower right corner.'''
    x1 = win[0] + 0.85 * (win[1] - win[0]) + mm / 2          # centered at 85% of the width (the label fits)
    y = win[2] + 0.07 * (win[3] - win[2])
    ax.plot([x1 - mm, x1], [y, y], color="black", lw=1.5, solid_capstyle="butt", zorder=10)
    ax.text(x1 - mm / 2, y + 0.03 * (win[3] - win[2]), f"{mm / 10:.0f} cm", fontsize=fs, ha="center", va="bottom",
            zorder=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=3.5)
    ap.add_argument("--aspect", type=float, default=0.95, help="cell height / width")
    ap.add_argument("--fs", type=float, default=7)
    ap.add_argument("--puzzle-width", type=float, default=900.0, help="(b) cells: this wide [mm], the same for every push")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "paper_figs", "val_combo", "val_combo"))
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

    dl = load(*LONG)
    cf.PICK[0] = cf.pick_regions
    long_snaps = cf.pick_regions(dl, 4)[1:]
    cells = [[(dl, k, f"t = {dl['times'][k][1] - dl['times'][0][1]:.0f} s")] for k in long_snaps]
    for i, spec in enumerate(PUZZLE):
        d = load(*spec)
        B = np.array(d["sc"]["boxes"])[:, 0, :2]
        end = int(np.argmax(np.linalg.norm(B - B[-1], axis=1) <= REST_MM))
        cells[i].append((d, end, f"push {i + 1}"))

    gap, mid, fs = 0.03, 0.12, args.fs
    head = fs / 72 * 1.7
    pw = (args.width - mid - 2 * gap) / 4
    ph = pw * args.aspect
    rows = len(cells)
    H = rows * ph + (rows - 1) * gap * 2 + head
    fig = plt.figure(figsize=(args.width, H))
    for r, row in enumerate(cells):
        y1 = 1 - (head + r * (ph + 2 * gap)) / H
        for c, (d, k, label) in enumerate(row):
            x0 = c * (2 * pw + gap + mid)
            win = crop(d, k, args.aspect, width=None if c == 0 else args.puzzle_width)
            ar = fig.add_axes([x0 / args.width, y1 - ph / H, pw / args.width, ph / H])
            asim = fig.add_axes([(x0 + pw + gap) / args.width, y1 - ph / H, pw / args.width, ph / H])
            draw_real(ar, d, k, win); draw_sim(asim, d, k)
            style(ar, win); style(asim, win)
            ar.text(0.04, 0.96, label, transform=ar.transAxes, fontsize=fs - 1, color="white", ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"))
            if r == 0:
                bar(asim, win, fs - 2)
                fig.text((x0 + pw + gap / 2) / args.width, 1 - 0.35 * head / H,
                         "(a) long-horizon run" if c == 0 else "(b) puzzle 5b",
                         ha="center", va="center", fontsize=fs)
            m = match_sim(d, k)
            print(f"{d['name']} frame {k} ({label}): vine {m['vine_mm']:.1f} mm, block {m['box_mm']:.1f} mm, "
                  f"window {[round(v) for v in win]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, pad_inches=0.0)
    print("wrote", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
