'''
LaTeX tables of the benchmark runs (bench_envs.py runs.jsonl) for the paper.

Each env row comes from one benchmark directory: 3a and 1c from the obj_spin 1 runs (wrinkle_mu03), the
others from the obj_spin 8 runs (spin8_manual). 4b is the variant with the block 50 mm right (4b_r50).
Times over successful runs: median and mean +- std of the planner time to the goal.

Run from the repo root:
    python3 xnew_rrt/paper_tables.py --out out/paper_figs/tables
    -> rrt.tex, puzzle.tex, and the same numbers as tables.md
'''

import argparse, json, os, statistics as st

SPIN8, MU03 = "out/bench_overnight", "out/bench_rrt_mu03"
RRT = [  # (paper name, job, benchmark dir, category)
    ("1a", "1a", SPIN8, "no retraction"), ("1b", "1b", SPIN8, "no retraction"), ("1c", "1c", MU03, "no retraction"),
    ("2a", "2a", SPIN8, "no retraction"), ("2b", "2b", SPIN8, "no retraction"),
    ("3a", "3a", MU03, "retraction"), ("3b", "3b", SPIN8, "retraction"),
    ("4a", "4a", SPIN8, "vine to goal"), ("4b", "4b_r50", SPIN8, "vine to goal"), ("4c", "4c", SPIN8, "vine to goal"),
    ("4d", "4d", SPIN8, "vine to goal")]
PUZZLES = [("5a", "5a"), ("5b", "5b"), ("5c", "5c_a15"), ("5d", "5d")]


def rows(d, job):
    with open(os.path.join(d, "runs.jsonl")) as f:
        return [r for r in (json.loads(l) for l in f if l.strip()) if r["job"] == job]


def mstd(v, fmt="{:.0f}"):
    if not v:
        return "--"
    return f"{fmt.format(st.fmean(v))} $\\pm$ {fmt.format(st.stdev(v) if len(v) > 1 else 0)}"


def med(v, fmt="{:.0f}"):
    return fmt.format(st.median(v)) if v else "--"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="out/paper_figs/tables")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md = []

    tex = [r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Env & Category & Success & Time [s] (median) & Time [s] (mean $\pm$ std) & Edges [k] & Pushes \\", r"\midrule"]
    md += ["| env | category | success | time median [s] | time mean ± std [s] | edges [k] | pushes |", "|---|---|---|---|---|---|---|"]
    for name, job, d, cat in RRT:
        rs = rows(d, job)
        ok = [r for r in rs if r.get("success")]
        t = [r["seconds_to_goal"] for r in ok]
        edges = [r["edges_to_goal"] / 1000 for r in ok]
        pushes = [r["pushes"] for r in ok]
        cells = [name, cat, f"{len(ok)}/{len(rs)}", med(t), mstd(t), mstd(edges, "{:.1f}"), mstd(pushes, "{:.1f}")]
        tex.append(" & ".join(cells) + r" \\")
        md.append("| " + " | ".join(c.replace(r"$\pm$", "±") for c in cells) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, "rrt.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Puzzle & Solved & Time [s] (median) & Time [s] (mean $\pm$ std) & Pushes & Final error [mm] & Max. turn [deg] \\",
           r"\midrule"]
    md += ["", "| puzzle | solved | time median [s] | time mean ± std [s] | pushes | final error [mm] | max turn [deg] |",
           "|---|---|---|---|---|---|---|"]
    for name, job in PUZZLES:
        rs = rows(SPIN8, job)
        ok = [r for r in rs if r.get("success")]
        t = [r["seconds"] for r in ok]
        cells = [name, f"{len(ok)}/{len(rs)}", med(t), mstd(t), mstd([r["pushes"] for r in ok], "{:.1f}"),
                 mstd([r["final_err_mm"] for r in ok], "{:.1f}"), mstd([r["max_angle_deg"] for r in ok], "{:.1f}")]
        tex.append(" & ".join(cells) + r" \\")
        md.append("| " + " | ".join(c.replace(r"$\pm$", "±") for c in cells) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(args.out, "puzzle.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
