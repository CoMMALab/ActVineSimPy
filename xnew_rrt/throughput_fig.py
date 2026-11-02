'''
Paper figure of simulator throughput against batch size: GPU batched (out/fit/throughput_diag.json, the
diagnosed rerun) against the per-element CPU solver (the DIFFCP rows of out/paper_figs/throughput/throughput.json).
Batches where the GPU solve ran out of memory (4d above 512: dvsim then re-solves element by element) are left out.

    python xnew_rrt/throughput_fig.py -> out/paper_figs/throughput/batch_efficiency.{pdf,png}
'''

import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = {"1a": ("1a", "#2a78d6"), "4d": ("4d", "#eb6834")}       # 1 and 13 blocks (in the caption)


def main():
    with open(os.path.join(REPO, "out", "fit", "throughput_diag.json")) as f:
        gpu = [r for r in json.load(f) if r["curvature"] == "spread" and "error" not in r and not r["ooms"]]
    with open(os.path.join(REPO, "out", "paper_figs", "throughput", "throughput.json")) as f:
        d = json.load(f)
    cpu = [r for r in d["rows"] if r["device"] == "cpu"]
    steps = d["steps"]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for env, (label, c) in ENVS.items():
        g = sorted((r["batch"], r["batch"] * steps / r["seconds"]) for r in gpu if r["env"] == env)
        p = sorted((r["batch"], r["element_steps_per_s"]) for r in cpu if r["env"] == env)
        ax.plot(*zip(*g), "-o", color=c, lw=1.5, ms=3.5, label=f"{label} GPU batched")
        ax.plot(*zip(*p), "--s", color=c, lw=1.2, ms=3.5, mfc="white", label=f"{label} CPU")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ticks = [1, 4, 16, 64, 256, 1024]
    ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
    ax.set_xlabel("batch size"); ax.set_ylabel("vine steps / s")
    ax.grid(True, which="major", lw=0.3, color="#d9d9d9")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.5, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, loc="lower right",
              bbox_to_anchor=(1.03, 0.06))    # the empty corner under the GPU curves
    fig.tight_layout(pad=0.3)
    out = os.path.join(REPO, "out", "paper_figs", "throughput", "batch_efficiency")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print("wrote", out + ".{pdf,png}")


if __name__ == "__main__":
    main()
