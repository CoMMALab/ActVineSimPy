# Paper data

The run data behind the paper's tables and figures, copied out of `out/` (gitignored) by `collect.sh`, grouped
by media type. File names start with what they feed (`planning_`, `overview_`, `robustness_`, `throughput_`,
`contact_` / `simerr_`, `puzzle_`, `long_`), not figure numbers, which shift as figures move. Rerun
`paper_data/collect.sh` from anywhere after a rerun; it overwrites.

| folder | contents |
|---|---|
| `videos/` | `real_<trial>.mov`: the 32 simple-contact trials (29 train, 3 held out; split in `json/video_manifest.json`) and the 10 long runs; `real_long_02_tracked.mp4`: its tracking overlay; `sim_<problem>_s000.mp4`: the seed-0 solution of each planning problem (the one drawn in the overview) |
| `images/` | the paper figures as they are in `overleaf/images/` (`.pdf`) and the panels / PNGs they were built from |
| `json/` | per-run planner results, run tables, solutions, error summaries, checkpoints, env definitions |
| `npz/` | simulator reference sets (`contact_train`, `contact_val`, `long`) and per-trial video tracks (`tracks_<trial>.npz`) |
| `logs/` | benchmark, evaluation, and fitting logs, and the one-off scripts that produced them |
| `tables/` | generated LaTeX / markdown tables |

## Table I (planning)

- `json/planning_runs.jsonl`: one line per run, all 15 problems x 20 seeds, with `problem` and `bench` added.
  Problem -> source benchmark: 1c, 3a from `bench_rrt_mu03` (checkpoint `wrinkle_mu03`); 5c from
  `bench_puzzle_retry` (job `5c_a15`, high-level retries); everything else from `bench_overnight`
  (`spin8_manual`; 4b is job `4b_r50`).
- `json/planning_<problem>_s<seed>.result.json`, `logs/planning_<problem>_s<seed>.run.log`: each run's result and log.
- `logs/planning_<bench>.bench.log`: the benchmark drivers.
- `json/env_<problem>.json`: the env definitions as run (5c is `envs/5c.json` run with `--max-final-angle 15`).
- `tables/rrt.tex`, `puzzle.tex`, `planning_generated.md`: `xnew_rrt/paper_tables.py` output. These predate the
  5c retry rerun (they show 5c at 18/20) and the Edges column; the paper's numbers come from `planning_runs.jsonl`.

## Overview figure

`images/overview.pdf` (as in the paper), `images/overview_<problem>.png` (single panels), `json/overview_<problem>.json`
(the drawn snapshots of each seed-0 solution; 3b is seed 12, a two-push solution, since seed 0 is a single push), `videos/sim_<problem>_s000.mp4` (3b: still seed 0). 5b is a run restricted to straight pushes (`paper_figs.py puzzle envs/5b.json --straight`: each vine launched from the ring cell in line with the block, exactly along the push); its video is `videos/sim_5b_straight.mp4` (`sim_5b_s000.mp4` is the benchmark's seed 0), and `videos/real_demo_5b_push{1,2,3}.mov` is the live run of it, one clip per push. The 5c panel leaves out grid column 13 (empty) for layout (`paper_figs.py grid --drop-col 5c:13`); the env as run has it.

## RRT tree figures (1c, 3a, 4d)

`images/tree_<problem>.{pdf,png}`, `json/tree_<problem>.json`: the seed-0 solution replayed on the GPU with a random
subtree off every node (4 children, then 2 each, 4 edges deep; controls drawn as the planner draws them), made by
`xnew_rrt/tree_fig.py`. The subtrees are fresh samples, not the run's own tree. 3a: one panel per push.

## Robustness figure (2a)

`json/robustness_2a.json`, `tables/robustness_2a.md` (median deviation / goal rate per perturbation and scale),
`images/robustness_2a.pdf`, `logs/robustness_2a.log`.

## Throughput figure (batch efficiency)

`images/batch_efficiency.pdf` (made by `xnew_rrt/throughput_fig.py`). GPU: `json/throughput_gpu.json` from
`logs/throughput_gpu.py` (per batch: time, peak memory, out-of-memory count, time split); batches with
`ooms > 0` (4d above 512) are left out of the figure: there the GPU solve runs out of memory and dvsim
re-solves element by element. CPU: the DIFFCP rows of `json/throughput_cpu.json`.

## Table II and the contact figure (simple-contact validation)

- Rows of Table II: `pbd_only`, `base_pbd_brick` (`tables/simerr_baselines.md`, `logs/simerr_baselines.log`);
  `contact_launchfix` (`json/simerr_contact_launchfix_val_errors.json`, the per-trial held-out errors behind
  19.0 / 45.8; `tables/simerr_contact_launchfix.md` is the CPU eval of the same checkpoint, 19.1 / 45.5);
  `contact_envfric_mu2` (`logs/simerr_contact_envfric_mu2.log`, made by `logs/simerr_contact_envfric_mu2.py`).
  Checkpoints: `json/checkpoint_<name>.json`.
- Figure: `images/contact_val.pdf`, `images/contact_<trial>.png`, `json/contact_val_errors.json` (per held-out
  trial errors and the frames shown), `logs/contact_val.log`.
- Real data: `videos/real_{vine_block,block_block,block_wall}_*.mov`, `npz/tracks_*.npz`,
  `npz/contact_train.npz`, `npz/contact_val.npz`.

## Puzzle figure (5c)

`images/puzzle_5c.pdf`, `images/puzzle_5c.png`: the seed-0 simulated solution (`json/overview_5c.json`,
`videos/sim_5c_s000.mp4`). No real 5c run yet.

## Table III and the long-horizon figure

- Table III: `json/long_runs.json` (per checkpoint and run: frames reached, frame where the real block first moves,
  vine and block error over reached frames), from `logs/long_runs.py`. The paper's numbers average over the runs
  that reach the block; `tables/long_runs.md` averages over all runs (its block column counts the two stalled
  runs as ~0) and is not what the paper reports.
- Figure (long_02, `contact_envfric_mu2`): `images/long_val.pdf`, `images/long_02.png`, `json/long_02_errors.json`,
  `logs/long_02.log`, `videos/real_long_02_tracked.mp4`.
- Real data: `videos/real_long_*.mov`, `npz/tracks_long_*.npz`, `npz/long.npz`.
