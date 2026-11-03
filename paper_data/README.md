# Paper data

The run data behind the paper (`overleaf/sections/05-experiment.tex`), copied out of `out/` (gitignored) by
`collect.sh` and grouped by media type. Only what the paper uses is here. File names start with what they feed
(`planning_`, `overview_`, `puzzle_`, `simerr_` / `fit_`, `contact_`, `multi_val_`, `throughput_` / `batch_efficiency`,
`robustness_`, `supplement_`, `real_` / `sim_` for videos). Rerun `paper_data/collect.sh` from anywhere after a rerun;
it empties the folders and copies everything again.

| folder | contents |
|---|---|
| `videos/` | real videos (`real_*.mov`, tracking overlays `real_*_tracked.mp4`), planned solutions (`sim_*.mp4`), the supplement videos (`supplement_*.mp4`) |
| `images/` | the paper figures as they are in `overleaf/images/` (`.pdf`), the panels / PNGs they were built from, and the puzzle runs' plan drawings |
| `json/` | per-run planner results, run table, figure data, error summaries, checkpoints, env definitions, video manifest |
| `npz/` | simulator references (fit / scoring sets) and per-video tracks (`tracks_<video>.npz`) |
| `logs/` | per-run planner logs, benchmark drivers, fitting and evaluation logs, and the one-off scripts that made them |
| `tables/` | generated markdown tables |

## Table I (planning, 15 problems x 20 runs)

- `json/planning_runs.jsonl`: one line per run, all 300, with `problem` and `bench` added. Problem -> source
  benchmark: 1c, 3a from `bench_rrt_mu03` (checkpoint `wrinkle_mu03`); 5c from `bench_puzzle_retry` (job `5c_a15`,
  high-level retries); everything else from `bench_overnight` (`spin8_manual`; 4b is job `4b_r50`).
- `json/planning_<problem>_s<seed>.result.json`, `logs/planning_<problem>_s<seed>.run.log`: each run's result and log.
  RRT results hold the solution's controls (replay with `xnew_rrt/render_result.py`).
- `images/planning_<5x>_s<seed>_{plan,pushes}.png`: each puzzle run's high-level plan and its pushes.
- `logs/planning_<bench>.bench.log`: the benchmark drivers. `json/env_<problem>.json`: the env definitions as run
  (5c is `envs/5c.json` run with `--max-final-angle 15`).
- Videos: the benchmark ran without rendering, so only seed 0 has a video: `videos/sim_<problem>_s000.mp4`
  (CPU replays of the result, or GPU reruns where the replay missed the goal; a CPU replay that still misses says
  so in its title). 2b has all 20: `videos/sim_2b_replan_s<seed>.mp4`, each seed re-planned from its own states
  (`out/paper_videos/record_2b.sh`).

## Overview figure

`images/overview.pdf` (as in the paper), `images/overview.png`, and per problem `json/overview_<problem>.json` (the
drawn snapshots) with `images/overview_<problem>.png` (no single panel was saved for 2b). The panels are seed 0
except 2b (seed 17, re-planned), 3b (seed 16, a two-push solution) and 4d (seed 6). 5b is a run restricted to straight
pushes (`paper_figs.py puzzle envs/5b.json --straight`: each vine launched from the ring cell in line with the block):
`videos/sim_5b_straight.mp4` (`sim_5b_s000.mp4` is the benchmark's seed 0). The 5c panel leaves out the empty grid
column 13 (`--drop-col 5c:13`); the env as run has it.

## Puzzle figure (5c)

`images/puzzle_5c.pdf`, `images/puzzle_5c.png`: the seed-0 simulated solution (`json/overview_5c.json`,
`videos/sim_5c_s000.mp4`).

## Table II (simulator error)

- `json/simerr_table.json`: per checkpoint, vine / block error on every set (contact train and held-out per type,
  the 10 long runs, `long02` alone, the three live 5b pushes as `demo`), from `logs/simerr_table.{py,log}`. Rows:
  Sim only = `pbd_only`; + general videos = `base_pbd_brick`; + scene videos = `contact_envfric_mu2` (contact
  columns), `long_fit` (Long, held out of its fit), `spin8_manual` (Puzzle 5b, the model 5b was planned with).
- Checkpoints, `json/checkpoint_<name>.json`, with each one's parent: `pbd_only` (master's PBD vine only),
  `base_pbd_brick` -> `contact_from_base` -> `wrinkle_mu03` (planning 1c, 3a) -> `spin8_manual` (other planning
  problems; `obj_spin` set by hand on the puzzle videos, `logs/fit_grid_spin.log`) -> `contact_launchfix` ->
  `contact_envfric` -> `contact_envfric_mu2` -> `long_fit`. Their fit logs: `logs/fit_*`.
- Training / scoring data, `npz/`: `master_ref` (master's sPAM-actuated PBD vine), `video_ref` (15 brick demos),
  `contact_train*` (29 videos) / `contact_val` (3 held out), `long` (all 10), `long_train` (9) / `long_val` (long_02),
  `puzzle_train` / `puzzle_val` (10 puzzle videos), `demo_5b` (the 3 live 5b pushes, never fit to).

## Real videos

`videos/real_<name>.mov` with `npz/tracks_<name>.npz`, every video in `json/video_manifest.json` except the one
excluded: `brick_*` (15), `vine_block_*` / `block_block_*` / `block_wall_*` (32: 29 train, 3 held out), `puzzle_*` (10),
`long_*` (10), `demo_5b_push{1,2,3}` (the live 5b run, base moved by hand between pushes).

## Contact figure (held-out simple contact)

`images/contact_val.pdf`, `images/contact_val.png`, `images/contact_<trial>.png`, `json/contact_val_errors.json`
(per held-out trial errors and the frames shown), `logs/contact_val.log`. Checkpoint `contact_envfric_mu2`.

## Multi-contact figure (long_02, live 5b)

`images/multi_val.pdf` (as in the paper; `xnew_rrt/val_combo_fig.py`), `images/multi_val.png`, and its two sources:
long_02 with `long_fit` (`json/multi_val_long_02_errors.json`, `images/multi_val_long_02.png`,
`videos/real_long_02_tracked.mp4`) and the live 5b pushes with `spin8_manual` (`json/multi_val_demo_5b_errors.json`,
`images/multi_val_demo_5b.png`, `logs/multi_val_demo_5b.log`, `videos/real_demo_5b_push*_tracked.mp4`).

## Ablation: simulator throughput

`images/batch_efficiency.pdf` (`xnew_rrt/throughput_fig.py`). GPU: `json/throughput_gpu.json` from
`logs/throughput_gpu.py` (per batch: time, peak memory, out-of-memory count, time split); batches with `ooms > 0`
(4d above 512) are left out of the figure. CPU: the DIFFCP rows of `json/throughput_cpu.json`.

## Ablation: robustness to execution error (2a)

`json/robustness_2a.json`, `tables/robustness_2a.md` (median deviation / goal rate per perturbation and scale),
`images/robustness_2a.pdf`, `logs/robustness_2a.log`.

## Supplement videos

`videos/supplement_robosoft_27.mp4`: the submitted video (compressed; the 115 MB original, `video/robosoft_27_video.mp4`,
is over GitHub's file limit). `videos/supplement_{1..5}.mp4`: the planning solutions by category,
`supplement_val_{contact,wall,multi}.mp4`: the validation runs, all from `xnew_rrt/paper_videos.py` with the recorded
solutions in `json/supplement_<job>.json`.
