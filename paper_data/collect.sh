#!/bin/bash
# Copy the run data behind the paper (overleaf/sections/05-experiment.tex) out of out/ (gitignored) into paper_data/,
# grouped by media type. Only what the paper uses: Table I (all 15 problems x 20 runs), the overview, the puzzle
# figure, Table II and the two validation figures, the two ablations (throughput, robustness), the supplement videos,
# and the real videos and simulator training data behind them. Rerun from anywhere; it starts from empty folders.
# See paper_data/README.md for what each file feeds.
set -e
cd "$(dirname "$0")/.."
D=paper_data
rm -rf $D/{videos,images,json,npz,logs,tables}
mkdir -p $D/{videos,images,json,npz,logs,tables}

# ---------------------------------------------------------------- Table I (planning), 20 runs per problem
# (problem, job, benchmark dir); same mapping as xnew_rrt/paper_tables.py, 5c from the retry rerun
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
        cp $s/run.log $D/logs/planning_${p}_$n.run.log
        for f in $s/results/*.result.json; do [ -f $f ] && cp $f $D/json/planning_${p}_$n.result.json; done
        for f in $s/images/*.{plan,pushes}.png; do    # the puzzles' high-level plan and pushes
            [ -f $f ] && cp $f $D/images/planning_${p}_${n}_$(basename $f | cut -d. -f2).png
        done
    done
    cp envs/$j.json $D/json/env_$p.json 2>/dev/null || cp envs/$p.json $D/json/env_$p.json
done
for b in bench_overnight bench_rrt_mu03 bench_puzzle_retry; do cp out/$b/bench.log $D/logs/planning_$b.bench.log; done

# ---------------------------------------------------------------- Overview figure
# the panels exactly as gridded into overleaf/images/overview.pdf (paper_figs.py grid ... --out out/paper_figs/overview_15)
O=out/paper_figs/overview; P=out/paper_figs/pushes
for e in 1a:$O/1a 1b:$O/1b 1c:$O/1c 2a:$O/2a 2b:$O/2b_s017_plan 3a:$O/3a 3b:$P/3b_s016 4a:$O/4a 4b:$O/4b_r50 4c:$O/4c \
         4d:$O/4d_s006 5a:$P/5a_s000 5b:$P/5b_straight 5c:$P/5c_a15_s000 5d:$P/5d_s000; do
    IFS=: read p f <<< "$e"
    cp $f.json $D/json/overview_$p.json
    [ -f $f.png ] && cp $f.png $D/images/overview_$p.png
done
cp out/paper_figs/overview_15.png $D/images/overview.png

# seed-0 solution videos (CPU replays of the result.json, or GPU reruns where the replay missed: see status.txt)
for p in 1a 1b 2a 2b 4a 4c 4d 5a 5b 5d; do
    cp out/videos_spin8_s000/$p.solution.mp4 $D/videos/sim_${p}_s000.mp4
done
cp out/videos_spin8_s000/4b_r50.solution.mp4 $D/videos/sim_4b_s000.mp4
cp out/videos_spin8_s000/5c_a15.solution.mp4 $D/videos/sim_5c_s000.mp4
cp out/videos_final/videos/3b.solution.mp4 $D/videos/sim_3b_s000.mp4    # GPU rerun: the CPU replay misses
cp out/videos_mu03/3a_s000/solution.mp4 $D/videos/sim_3a_s000.mp4
cp out/videos_mu03/1c_s000/solution.mp4 $D/videos/sim_1c_s000.mp4
# 2b: every seed, re-planned from its own states (the replays miss the goal; out/paper_videos/record_2b.sh)
for f in out/paper_videos/2b_options/2b_s*.mp4; do
    cp $f $D/videos/sim_2b_replan_s0$(basename $f .mp4 | cut -c5-).mp4
done
# 5b: straight pushes only (paper_figs.py puzzle --straight), and the live run of it, one clip per push
cp out/straight_5b/videos/5b.solution.mp4 $D/videos/sim_5b_straight.mp4
for k in 1 2 3; do cp video/puzzle/demo_5b_push$k.mov $D/videos/real_demo_5b_push$k.mov; done

# ---------------------------------------------------------------- Puzzle figure (5c, sim seed 0)
cp out/paper_figs/pushes/5c_steps.png $D/images/puzzle_5c.png

# ---------------------------------------------------------------- Table II (simulator error)
# out/fit/combined_table.json was cut short by a crash after all rows were computed; its log has every row
python3 - out/fit/combined_table.log > $D/json/simerr_table.json <<'EOF'
import json, sys
rows = {}
for l in open(sys.argv[1]):
    name, _, rest = l.partition(" ")
    if rest.startswith("{"):
        rows[name] = json.loads(rest)
print(json.dumps(rows, indent=1))
EOF
cp out/fit/combined_table.py $D/logs/simerr_table.py
cp out/fit/combined_table.log $D/logs/simerr_table.log
# the checkpoints: the table's rows and the fits they come from
for c in pbd_only base_pbd_brick contact_from_base wrinkle_mu03 spin8_manual contact_launchfix contact_envfric \
         contact_envfric_mu2 long_fit; do
    cp checkpoints/$c.json $D/json/checkpoint_$c.json
done
for f in pbd_only.log pbd_only.sh wrinkle_mu03.log grid_spin.log contact_launchfix.log contact_envfric.log \
         envfric_mu2.log envfric_mu2.py long_fit.log long_fit_eval.log long_fit_eval.py; do
    cp out/fit/$f $D/logs/fit_$f
done
# simulator references: master's PBD vine (pbd_only), and the tracked real videos as fit / scored
for n in master_ref video_ref contact_train contact_train_vine_block contact_train_block_block contact_train_block_wall \
         contact_val long long_train long_val demo_5b puzzle_train puzzle_val; do
    cp xnew_rrt/data/$n.npz $D/npz/
done

# ---------------------------------------------------------------- Real videos and their tracks
# brick demos (base_pbd_brick), simple contact (29 train, 3 held out), puzzle videos (obj_spin of spin8_manual),
# long runs, and the live 5b demo (above); the split is in json/video_manifest.json
cp video/manifest.json $D/json/video_manifest.json
python3 - <<'EOF' > $D/.videos
import json
for v in json.load(open("video/manifest.json"))["videos"]:
    if v["split"] != "excluded":
        print(v["path"])
EOF
for v in $(cat $D/.videos); do
    t=$(basename $v .mov)
    [ -f $D/videos/real_$t.mov ] || cp $v $D/videos/real_$t.mov
    cp out/video/$t/tracks.npz $D/npz/tracks_$t.npz
done
rm $D/.videos

# ---------------------------------------------------------------- Contact figure (held-out simple contact)
F=out/paper_figs/contact_envfric_mu2
cp $F/errors.json $D/json/contact_val_errors.json
cp $F/contact_val_columns.png $D/images/contact_val.png
for t in vine_block_06 block_block_08 block_wall_07; do cp $F/contact_${t}_wide.png $D/images/contact_$t.png; done
cp out/fit/contact_fig_mu2.log $D/logs/contact_val.log

# ---------------------------------------------------------------- Multi-contact figure (long_02 with long_fit, live 5b)
cp out/paper_figs/val_combo/val_combo.png $D/images/multi_val.png
cp out/paper_figs/long_fit/errors.json $D/json/multi_val_long_02_errors.json
cp out/paper_figs/long_fit/contact_long_02_wide.png $D/images/multi_val_long_02.png
cp out/paper_figs/demo_5b_spin8_manual/errors.json $D/json/multi_val_demo_5b_errors.json
cp out/paper_figs/demo_5b_spin8_manual/demo_5b_columns.png $D/images/multi_val_demo_5b.png
cp out/fit/demo_5b_spin8_manual.log $D/logs/multi_val_demo_5b.log
cp out/video/long_02/overlay.mp4 $D/videos/real_long_02_tracked.mp4
for k in 1 2 3; do cp out/video/demo_5b_push$k/overlay.mp4 $D/videos/real_demo_5b_push${k}_tracked.mp4; done

# ---------------------------------------------------------------- Ablation: throughput
cp out/fit/throughput_diag.json $D/json/throughput_gpu.json
cp out/paper_figs/throughput/throughput.json $D/json/throughput_cpu.json
cp out/fit/throughput_diag.log $D/logs/throughput_gpu.log
cp out/fit/throughput_diag.py $D/logs/throughput_gpu.py
cp out/paper_figs/throughput/batch_efficiency.png $D/images/batch_efficiency.png

# ---------------------------------------------------------------- Ablation: robustness (2a)
cp out/funnel/2a/funnel.json $D/json/robustness_2a.json
cp out/funnel/2a/run.log $D/logs/robustness_2a.log
cp out/paper_figs/robustness/2a_funnel.png $D/images/robustness_2a.png
cp out/paper_figs/robustness/2a_funnel.md $D/tables/robustness_2a.md

# ---------------------------------------------------------------- Supplement videos (xnew_rrt/paper_videos.py)
for v in 1 2 3 4 5 val_contact val_multi val_wall; do cp out/paper_videos/$v.mp4 $D/videos/supplement_$v.mp4; done
for f in out/paper_videos/data/*.json; do cp $f $D/json/supplement_$(basename $f); done
cp video/robosoft_27_video_compressed.mp4 $D/videos/supplement_robosoft_27.mp4

# ---------------------------------------------------------------- the figures exactly as they are in the paper
cp overleaf/images/{overview,contact_val,multi_val,puzzle_5c,batch_efficiency,robustness_2a}.pdf $D/images/
du -sh $D/*/ | sed 's/^/  /'
echo done
