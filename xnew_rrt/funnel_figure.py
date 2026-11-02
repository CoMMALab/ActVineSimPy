'''
Paper figure and table of the funneling test (funnel_test.py's funnel.json): where the block ends up when
the solution is replayed open loop under perturbed launch position, launch heading and actuation.

  left   the board around the goal: the nominal solution (vine and block) and the final block centers
         under all three perturbations together, colored by scale
  right  median distance of the final block center from the nominal one, per perturbation and scale
  table  that distance and the goal rate per perturbation and scale (the nominal block ends near the
         goal's edge, so the goal rate is a strict measure; the distance is the funneling)

The nominal replay runs on the CPU (a single push replays faithfully there).

Run from the repo root:
    python xnew_rrt/funnel_figure.py out/funnel/2a/funnel.json --out out/paper_figs/robustness/2a_funnel
    -> <out>.pdf, <out>.png, <out>.tex, <out>.md
'''

import argparse, json, math, os, statistics as st

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})     # TrueType, not Type 3 (submission check)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.collections

from scene import Sim  # (puts the repo root on sys.path)
import checkpoint
import funnel_test
import run_envs
import viz
import obstacle_style

FAMILIES = [("launch", "launch position"), ("heading", "launch heading"), ("actuation", "actuation"), ("all", "all three")]
FAMILY_COLORS = {"launch": "#2a78d6", "heading": "#1baf7a", "actuation": "#d6a02a", "all": "#d62a78"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("funnel")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-titles", dest="titles", action="store_false", help="no panel titles (sigmas go in the caption)")
    parser.add_argument("--fs", type=float, default=7, help="font size (labels one point larger)")
    parser.add_argument("--right-label-fs", type=float, default=None, help="right panel's axis titles (default fs + 1)")
    parser.add_argument("--right-legend-fs", type=float, default=None, help="right panel's legend (default fs)")
    parser.add_argument("--zoom", type=float, default=160.0, help="half width of the left panel around the unperturbed block [mm]")
    args = parser.parse_args()
    os.chdir(run_envs.REPO)
    with open(args.funnel) as f:
        d = json.load(f)
    a = d["args"]
    scales = [float(s) for s in a["scales"].split(",")]
    nom = d["nominal_box_mm"]

    e, scene, cfg, _ = run_envs.load_env(a["env"])
    with open(d["result"]) as f:
        result = json.load(f)
    checkpoint.apply(cfg, d["checkpoint"])
    cfg.max_bodies = result["sim"]["max_bodies"]
    cfg.device, cfg.qp_backend = "cpu", "DIFFCP"
    sim = Sim(scene, cfg)
    goal_test, _ = run_envs.goal_fns(sim, e)
    settled = "nominal" in d            # funnel_settle.py: blocks at rest, the unperturbed run from the same GPU batch
    if not settled:
        _, nominal = funnel_test.schedule(sim, result["controls"], result["planner"]["steps_per_edge"], goal_test)

    stats = {}
    for fam, _ in FAMILIES:
        for sc in scales:
            rs = [r for r in d["samples"] if r["family"] == fam and r["scale"] == sc]
            dist = [math.dist((r["box_x_mm"], r["box_y_mm"]), nom) for r in rs if r["ok"]]
            stats[fam, sc] = dict(median=st.median(dist), p90=sorted(dist)[int(0.9 * (len(dist) - 1))],
                                  reached=sum(r.get("reached", False) for r in rs) / len(rs), n=len(rs))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(width_ratios=[1, 1.15]))
    viz.draw_scene(ax, sim, walls=False)
    obstacle_style.draw_static(ax, sim)
    for t in list(ax.texts):            # the "box goal" / "start" labels
        t.remove()
    if settled:
        n = d["nominal"]
        r = cfg.vine_radius_m * 1000
        V = n["vine_mm"]
        ax.add_collection(matplotlib.collections.PatchCollection([Circle(q, r) for q in V], facecolor="#2a78d6",
                                                                 edgecolor="none", alpha=0.18, zorder=4))
        ax.plot([q[0] for q in V], [q[1] for q in V], "-", color="#2a78d6", lw=1.2, alpha=0.8, zorder=5)
        bx_, by_, th = n["box_pose"]
        hw, hh = e["boxes"][0]["size"][0] / 2, e["boxes"][0]["size"][1] / 2
        c, s_ = math.cos(th), math.sin(th)
        ax.add_patch(Polygon([(bx_ + c * lx - s_ * ly, by_ + s_ * lx + c * ly)
                              for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))],
                             closed=True, fill=False, edgecolor="#eb6834", lw=1.2, zorder=6))
    else:
        viz.draw_objects(ax, sim, nominal, outline_only=True, alpha=1.0)
        viz.draw_vine(ax, sim, nominal, alpha=0.2)
    g = e["goal"]
    cmap = plt.get_cmap("viridis")
    for j, sc in enumerate(reversed(scales)):
        rs = [r for r in d["samples"] if r["family"] == "all" and r["scale"] == sc and r["ok"]]
        ax.scatter([r["box_x_mm"] for r in rs], [r["box_y_mm"] for r in rs], s=4, alpha=0.7, zorder=9,
                   color=cmap(0.85 * (1 - j / max(1, len(scales) - 1))), label=f"{sc:g}$\\sigma$", linewidths=0)
    ax.plot(*nom, "*", color="black", ms=9, mec="white", mew=0.6, zorder=10, label="unperturbed")
    pts = [(r["box_x_mm"], r["box_y_mm"]) for r in d["samples"] if r["family"] == "all" and r["ok"]] + [tuple(nom)]
    cx, cy = nom
    out = sum(abs(p[0] - cx) > args.zoom or abs(p[1] - cy) > args.zoom for p in pts)
    print(f"{out} of {len(pts)} block centers outside the left panel")
    ax.set_xlim(cx - args.zoom, cx + args.zoom); ax.set_ylim(cy - args.zoom, cy + args.zoom)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    y = cy - 0.88 * args.zoom                   # 10 cm scale bar, bottom, between the legend and the vine
    ax.plot([cx + 15, cx + 115], [y, y], color="black", lw=1.5, solid_capstyle="butt", zorder=11)
    ax.text(cx + 65, y + 0.03 * args.zoom, "10 cm", fontsize=args.fs, ha="center", va="bottom", zorder=11)
    leg = ax.legend(fontsize=args.fs, loc="lower left", frameon=False, handletextpad=0.3)
    for h in leg.legend_handles:
        if isinstance(h, matplotlib.collections.PathCollection):
            h.set_sizes([16])
    if args.titles:
        ax.set_title("final block centers, all perturbations", fontsize=args.fs + 1)

    xs = range(len(scales))                 # the scales double: evenly spaced levels
    for fam, label in FAMILIES:
        bx.plot(xs, [stats[fam, sc]["median"] for sc in scales], "o-", color=FAMILY_COLORS[fam], label=label, lw=1.6, ms=4)
    bx.set_xticks(list(xs)); bx.set_xticklabels([f"{s:g}$\\sigma$" for s in scales])
    bx.set_xlim(-0.25, len(scales) - 0.75)
    bx.set_xlabel("perturbation scale", fontsize=args.right_label_fs or args.fs + 1)
    bx.set_ylabel("median error [mm]", fontsize=args.right_label_fs or args.fs + 1)
    bx.tick_params(labelsize=args.fs); bx.grid(alpha=0.3); bx.legend(fontsize=args.right_legend_fs or args.fs, frameon=False)
    if args.titles:
        bx.set_title(f"$\\sigma$: {a['pos_mm']:g} mm, {a['heading_deg']:g}$^\\circ$, {a['curv']:g} m$^{{-1}}$", fontsize=args.fs + 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight"); fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")

    sig = {"launch": lambda s: f"{a['pos_mm'] * s:g} mm", "heading": lambda s: f"{a['heading_deg'] * s:g}$^\\circ$",
           "actuation": lambda s: f"{a['curv'] * s:g} m$^{{-1}}$", "all": lambda s: "all"}
    tex = [r"\begin{tabular}{l" + "r" * len(scales) + "}", r"\toprule",
           "Perturbation & " + " & ".join(f"{s:g}$\\sigma$" for s in scales) + r" \\", r"\midrule"]
    md = ["| perturbation | " + " | ".join(f"{s:g}σ: median dev [mm] / goal %" for s in scales) + " |",
          "|---" * (len(scales) + 1) + "|"]
    for fam, label in FAMILIES:
        tex.append(label + " & " + " & ".join(f"{stats[fam, s]['median']:.1f} ({100 * stats[fam, s]['reached']:.0f}\\%)"
                                             for s in scales) + r" \\")
        md.append(f"| {label} | " + " | ".join(f"{stats[fam, s]['median']:.1f} / {100 * stats[fam, s]['reached']:.0f}%"
                                              for s in scales) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(args.out + ".tex", "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", args.out + ".pdf")


if __name__ == "__main__":
    main()
