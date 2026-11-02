#!/bin/bash
# Copy the run data behind the paper's tables and figures out of out/ (gitignored) into paper_data/, grouped by
# media type. Rerun from anywhere; it overwrites. See paper_data/README.md for what each file feeds.
set -e
cd "$(dirname "$0")/.."
D=paper_data
mkdir -p $D/{videos,images,json,npz,logs,tables}

# Table I benchmarks: (problem, job, benchmark dir); same mapping as xnew_rrt/paper_tables.py, 5c from the retry rerun
JOBS="1a:1a:bench_overnight 1b:1b:bench_overnight 1c:1c:bench_rrt_mu03 2a:2a:bench_overnight 2b:2b:bench_overnight
3a:3a:bench_rrt_mu03 3b:3b:bench_overnight 4a:4a:bench_overnight 4b:4b_r50:bench_overnight 4c:4c:bench_overnight
4d:4d:bench_overnight 5a:5a:bench_overnight 5b:5b:bench_overnight 5c:5c_a15:bench_puzzle_retry 5d:5d:bench_overnight"
: > $D/json/planning_runs.jsonl
for e in $JOBS; do
    IFS=: read p j b <<< "$e"
    python3 - "$p" "$j" "out/$b/runs.jsonl" >> $D/json/planning_runs.jsonl <<'EOF'
import json, sys
p, j, f = sys.argv[1:]
for l in open(f):
    if l.strip() and (r := json.loads(l))["job"] == j:
        print(json.dumps(dict(problem=p, bench=f.split("/")[1], **r)))
EOF
    for s in out/$b/runs/$j/s*; do
        n=$(basename $s)
        cp $s/results/*.result.json $D/json/planning_${p}_$n.result.json
        cp $s/run.log $D/logs/planning_${p}_$n.run.log
    done
    cp envs/$j.json $D/json/env_$p.json 2>/dev/null || cp envs/$p.json $D/json/env_$p.json
done
for b in bench_overnight bench_rrt_mu03 bench_puzzle_retry; do cp out/$b/bench.log $D/logs/planning_$b.bench.log; done
cp out/paper_figs/tables/{rrt.tex,puzzle.tex} $D/tables/
cp out/paper_figs/tables/tables.md $D/tables/planning_generated.md

# Overview figure: seed-0 solution of every problem, replayed
for p in 1a 1b 1c 2a 2b 3a 4a 4c 4d; do
    cp out/paper_figs/overview/$p.json $D/json/overview_$p.json
    cp out/paper_figs/overview/$p.png $D/images/overview_$p.png
done
# 3b: seed 12, a two-push solution that passes the block on both sides (seed 0 is a single push)
cp out/paper_figs/overview3b/3b_s012.json $D/json/overview_3b.json
cp out/paper_figs/overview3b/3b_s012.png $D/images/overview_3b.png
cp out/paper_figs/overview/4b_r50.json $D/json/overview_4b.json
cp out/paper_figs/overview/4b_r50.png $D/images/overview_4b.png
for p in 5a 5d; do
    cp out/paper_figs/pushes/${p}_s000.json $D/json/overview_$p.json
    cp out/paper_figs/pushes/${p}_s000.png $D/images/overview_$p.png
done
# 5b: straight pushes only (paper_figs.py puzzle --straight, xnew_rrt/straight_puzzle.py)
cp out/paper_figs/pushes/5b_straight.json $D/json/overview_5b.json
cp out/paper_figs/pushes/5b_straight.png $D/images/overview_5b.png
cp out/straight_5b/videos/5b.solution.mp4 $D/videos/sim_5b_straight.mp4
# the live 5b demo, one clip per push (video/puzzle, straight pushes as in the overview)
for k in 1 2 3; do cp video/puzzle/demo_5b_push$k.mov $D/videos/real_demo_5b_push$k.mov; done
cp out/paper_figs/pushes/5c_a15_s000.json $D/json/overview_5c.json
cp out/paper_figs/pushes/5c_a15_s000.png $D/images/overview_5c.png

# seed-0 solution videos (CPU replays of the same results)
for p in 1a 1b 2a 2b 3b 4a 4c 4d 5a 5b 5d; do
    cp out/videos_spin8_s000/$p.solution.mp4 $D/videos/sim_${p}_s000.mp4
done
cp out/videos_spin8_s000/4b_r50.solution.mp4 $D/videos/sim_4b_s000.mp4
cp out/videos_spin8_s000/5c_a15.solution.mp4 $D/videos/sim_5c_s000.mp4
cp out/videos_mu03/3a_s000/solution.mp4 $D/videos/sim_3a_s000.mp4
[ -f out/videos_mu03/1c_s000/solution.mp4 ] && cp out/videos_mu03/1c_s000/solution.mp4 $D/videos/sim_1c_s000.mp4

# RRT tree figures (sampled subtrees off the seed-0 solutions; xnew_rrt/tree_fig.py)
for p in 1c 3a 4d; do
    cp out/paper_figs/tree/$p.png $D/images/tree_$p.png
    cp out/paper_figs/tree/$p.pdf $D/images/tree_$p.pdf
    cp out/paper_figs/tree/$p.json $D/json/tree_$p.json
done

# Robustness figure (2a)
cp out/funnel/2a/funnel.json $D/json/robustness_2a.json
cp out/funnel/2a/run.log $D/logs/robustness_2a.log
cp out/paper_figs/robustness/2a_funnel.png $D/images/robustness_2a.png
cp out/paper_figs/robustness/2a_funnel.md $D/tables/robustness_2a.md

# Contact figure + Table II (simple contact validation)
F=out/paper_figs/contact_envfric_mu2
cp $F/errors.json $D/json/contact_val_errors.json
cp $F/contact_val_columns.png $D/images/contact_val.png
for t in vine_block_06 block_block_08 block_wall_07; do cp $F/contact_${t}_wide.png $D/images/contact_$t.png; done
cp out/fit/contact_fig_mu2.log $D/logs/contact_val.log
cp out/fit/baselines.md $D/tables/simerr_baselines.md
cp out/fit/final_val.md $D/tables/simerr_contact_launchfix.md
cp out/fit/baselines.log $D/logs/simerr_baselines.log
cp out/fit/final_val.log $D/logs/simerr_contact_launchfix.log
cp out/fit/envfric_mu2.log $D/logs/simerr_contact_envfric_mu2.log
cp out/fit/envfric_mu2.py $D/logs/simerr_contact_envfric_mu2.py
cp out/paper_figs/contact_launchfix/errors.json $D/json/simerr_contact_launchfix_val_errors.json
for c in pbd_only base_pbd_brick contact_launchfix contact_envfric_mu2 spin8_manual wrinkle_mu03; do
    cp checkpoints/$c.json $D/json/checkpoint_$c.json
done
cp xnew_rrt/data/contact_train.npz xnew_rrt/data/contact_val.npz xnew_rrt/data/long.npz $D/npz/
cp video/manifest.json $D/json/video_manifest.json
python3 - <<'EOF' > $D/json/.contact_trials
import json
for v in json.load(open("video/manifest.json"))["videos"]:
    if v["dataset"] == "simple_contact" and v["split"] in ("train", "val"):
        print(v["path"])
EOF
for v in $(cat $D/json/.contact_trials); do
    t=$(basename $v .mov)
    cp $v $D/videos/real_$t.mov
    cp out/video/$t/tracks.npz $D/npz/tracks_$t.npz
done
rm $D/json/.contact_trials

# Puzzle figure (5c, sim seed 0)
cp out/paper_figs/pushes/5c_steps.png $D/images/puzzle_5c.png

# Long-horizon figure (long_02) + Table III (all 10 long runs, never fit to)
cp out/paper_figs/long_mu2/contact_long_02_vert_bare.png $D/images/long_02.png
cp out/paper_figs/long_mu2/errors.json $D/json/long_02_errors.json
cp out/fit/long_fig_mu2.log $D/logs/long_02.log
cp out/fit/long_reached.json $D/json/long_runs.json
cp out/fit/long_reached.md $D/tables/long_runs.md
cp out/fit/long_reached.log $D/logs/long_runs.log
cp out/fit/long_reached.py $D/logs/long_runs.py
for v in video/long/long_*.mov; do
    t=$(basename $v .mov)
    cp $v $D/videos/real_$t.mov
    cp out/video/$t/tracks.npz $D/npz/tracks_$t.npz
done
cp out/video/long_02/overlay.mp4 $D/videos/real_long_02_tracked.mp4

# Throughput figure (batch efficiency)
cp out/fit/throughput_diag.json $D/json/throughput_gpu.json
cp out/paper_figs/throughput/throughput.json $D/json/throughput_cpu.json
cp out/fit/throughput_diag.log $D/logs/throughput_gpu.log
cp out/fit/throughput_diag.py $D/logs/throughput_gpu.py
cp out/paper_figs/throughput/batch_efficiency.png $D/images/throughput.png

# the figures and tables exactly as they are in the paper
cp overleaf/images/{overview,robustness_2a,batch_efficiency,contact_val,puzzle_5c,long_val}.pdf $D/images/
cp out/paper_figs/overview_15.png $D/images/overview.png
echo done
