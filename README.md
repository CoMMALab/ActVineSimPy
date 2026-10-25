# vine-push: planning for soft growing robots that push movable objects

Work in progress. This repo started as the code release for *Parallel Simulation of Contact and
Actuation for Soft Growing Robots* (ActVine: a JAX position-based-dynamics vine simulator plus an
SST planner). We are extending it so the vine can **push movable objects**. PBD has no notion of
mass, so for this we go back to the velocity/impulse contact formulation of *Physics-Grounded
Differentiable Simulation for Soft Growing Robots* (DiffVine): every step solves one QP over body
velocities, with the mass matrix as the quadratic term. Movable boxes are extra rows and columns
of that QP, so their mass and inertia enter the dynamics directly.

## Branches

| Branch | What it is |
|---|---|
| `master` | The ActVine paper release (plus one merged PR). Leave it alone for now. |
| `dynamic_obstacles`, `dysim_fix` | Earlier attempts at movable objects inside the PBD simulator. |
| `creating_rrt` | The DiffVine-style simulator (`dvsim/`) and a first RRT on top of it (`xnew_rrt/`). |
| `yitian` | This branch: `creating_rrt` plus bug fixes, a working planner loop and headless visualizers. See [What changed on this branch](#what-changed-on-this-branch). |

## Ground rules

- **Keep changes minimal.** `dvsim/` and `xnew_rrt/` are the undergrad's work and should stay
  recognisably so: fix bugs in place, keep names and structure, and put new functionality in new
  files rather than rewriting existing ones.
- **The sPAM actuator gives a constant moment per joint**, set by its design and independent of
  the joint's current angle; only the vine's own bending moment varies with the angle. The MLP
  surrogate is fine. (`master`'s `solve_fwd` re-evaluates the actuator at the current angle, so
  it is not the reference for this.) See [dvsim/README.md](dvsim/README.md), section 7.

## Layout

There are two simulators and two planners in the tree. Only the first row of each table is
the live code for the pushing work.

| Path | Status | What it is |
|---|---|---|
| `dvsim/` | **live** | Torch impulse/QP simulator with movable boxes. See below. |
| `xnew_rrt/` | **live** | RRT over `dvsim`, tree visualizer, single-rollout viewer. |
| `video/` | **live** | Vine centerlines and box poses from top-down videos, for system identification. See [Video tracking](#video-tracking). |
| `sPAM/` | used | sPAM actuator model. `torch_*.py` are the Torch ports (`torch_nns.py` = neural surrogate, `torch_nns_usage.py` = bend radius -> pressure / actuator length). The non-`torch_` files are the JAX originals and need `jax`. |
| `envs/` | **live** | The sim environments, `envs/<name>.json` (written by `envs/make_envs.py`; the block puzzles 5a–5c by `envs/make_puzzles.py`), pictures in `envs/vis/`. See [Environments](#environments). The old `envs/*.txt` files are in `envs/archive/`; master's scripts that default to them (`kinodynamic/`, `experiments/`, `geometric/`) need `--env envs/archive/...`. |
| `pbd_vine.py`, `render.py`, `kinodynamic/`, `geometric/`, `experiments/` | from master | ActVine PBD simulator, pygame renderer, SST planner, biarc RRT*, paper experiments. Identical to `master` on this branch, except `kinodynamic/env_loader.py`, which also parses `dynamic_obstacles:`. Needs `jax`, which the current `vine` env does not have. |
| `cache/`, `pics/`, `solution.png`, `title.gif`, `x_progress_log.txt` | leftovers | Outputs and notes from earlier runs. |

### `dvsim/`: the simulator

[dvsim/README.md](dvsim/README.md) explains the solver in detail: why a step is a QP, how that
relates to SQP, how it is solved and batched, box friction, and how the sPAM model fits in.

| File | Contents |
|---|---|
| `vine.py` | `VineParams`, state layout, growth (`extend`), the constraint functions (`sdf`, `joint_deviation`, `growth_rate`), bending torques (`bending_energy`), and `forward_batched_part`, which evaluates them and their Jacobians for one batch element. |
| `dynamic_vine.py` | Movable boxes: vine-box, box-wall and box-box gap functions, assembly of the full QP (`dynamic_solve`), and `step()`, the single entry point. |
| `solver.py` | The QP solve: cvxpylayers + Clarabel on CPU (differentiable, default), `BACKEND = "TORCH_IPM"` for `batched_qp.py`, or `qpth` on GPU if installed. |
| `batched_qp.py` | The whole batch's QP in plain torch on CPU or GPU: friction cone linearized, far rows dropped, equalities eliminated by LU, Mehrotra interior point. See [The batched simulator](#the-batched-simulator). |
| `si.py` | SI <-> internal units. Internally lengths are **mm**, time is s, mass is kg, angles are rad. |
| `spam.py` | sPAM actuation for the `'spam'` stiffness mode: actuator moment from (pressure, rest length, design radius), and its inverse. |
| `demos/` | `render_demos.py` (pushing / box-wall / box-box / spin / sPAM curl gifs), `render_batched.py` (64 vines in parallel). |

State is one row per batch element: `state = [x0, y0, th0, x1, y1, th1, ...]` for `max_bodies`
rigid segments (only the first `bodies` are active), plus `obj_pose = (cx, cy, theta)` per box.
One `step()` does:

1. **Grow**: when the sliding tip body is more than one segment away, insert a new body behind it.
2. **Linearize** all constraints about the current state with `torch.func.jacrev`.
3. **Solve** for the next velocities `v = [vine (3 per body), boxes (3 per box)]`:

   ```
   minimize    1/2 v' M v + p' v            M = diag(vine masses/inertias, box masses/inertias)
   subject to  J v dt = -deviation           joints stay connected
               growth(v) = grow_rate         the tip link lengthens at the growth speed
               gap + dt * dgap/dq * v >= 0   vine-wall, vine-box, box-wall, box-box
   ```
   `p` holds the previous momentum and the bending/damping torques. Contact forces are the
   multipliers of the gap constraints, so a heavier box (larger entries in `M`) moves less.
   Each box also has a Coulomb friction term against the table, `dt * mu * m * g * ||v_box||`.
4. **Integrate** `q += v dt` with `dt = 1/90 s`, after clamping velocity spikes.

Bending is one torque per joint, a function of the joint's angle and angular velocity, that is
added to `p`. `params.stiffness_mode` picks the function:

- `'linear'`: passive vine. Spring toward straight plus damping. No actuation.
- `'spam'`: the linear spring plus a constant actuator moment per joint (`params.spam_moment`,
  from `dvsim/spam.py`).
- `'curvature'`: the linear spring toward a per-joint rest angle `params.bend_target`. Same
  dynamics as `'spam'` (a constant moment against a spring is a spring about a shifted angle),
  without going through the actuator model.

Growth is either a hard constraint (the tip link lengthens at exactly `grow_rate`, as in
DiffVine; `dvsim`'s default) or a force with a stall limit (`params.grow_force`).

`step(..., per_batch={...})` lets `bend_target`, `spam_moment` and other `VineParams` fields differ
between batch elements.

### `xnew_rrt/`: planning and visualization

| File | Contents |
|---|---|
| `scene.py` | `Scene` (walls, movable boxes, start, goal), `default_scene()`, `load_scene(env file)`, and `Sim`, a small wrapper that owns the `dvsim` params and exposes `initial_state()` / `rollout(state, curvature, steps)`. |
| `rrt_util.py` | `StateInfo` (one full sim state), `Node`, `RRTTree`. |
| `sim_rrt.py` | The RRT, over the tip position. |
| `tip_block_rrt.py` | The same RRT over (tip position, box positions), with a goal for a box (or `--tip-goal`). |
| `batch_tip_block_rrt.py` | That planner with a batch of edges simulated together per iteration. |
| `reset_action_rrt.py` | Several pushes in one tree: a vine reset is an action, unlocked once the boxes have moved far enough from every earlier reset. Used by `run_envs.py`. |
| `reset_rrt.py` | Older: several pushes as a greedy chain of batched-RRT episodes, one tree per push. |
| `run_envs.py` | Load, draw and plan on the environments in `envs/*.json`. |
| `bench_batch.py` | Times one batched dvsim step against batch size, either backend and device (and master's PBD sim on the GPU, `--pbd`). |
| `check_batched_qp.py` | Checks the batched torch QP solver against Clarabel and the exact friction cone, and compares rollouts. |
| `bench_jax_qp.py` | The batched torch QP solver against qpax and MPAX (JAX) on the same problems. |
| `kino.py` | Batched kinodynamic planners: rrt, rrt_grid, kpiece, kinopax (+ `_shape` variants), with resets. |
| `ablation.py` | Runs every env x planner x seed on the GPU sim; `--summarize` writes the table. |
| `funneling.py` | Measures how often tree nodes land on an earlier node, in contact and in free space. |
| `block_puzzle.py` | Hierarchical planner for the block puzzles 5a–5c: Dijkstra over the block's grid cell, each push realized by a batched search over straight vines from the spawn ring. |
| `hierarchical_planner.py` | High-level block-pushing puzzle on a grid of box-sized cells: which pushes, from where, in what order. |
| `pose_search.py` | Launch-pose search for a passive vine pushing a box (for the live demo). |
| `rollout.py` | Simulate one rollout from a list of controls (or replay an RRT plan) and render it. |
| `draw_envs.py` | Draw the old `envs/archive/*.txt` files (`run_envs.py --draw-only` draws the new ones). |
| `sweep_vine_params.py` | Sweep vine inertia, mass, stiffness and damping; score curvature tracking. |
| `master_reference.py` | Run `master`'s PBD vine (jax) on a set of scenarios and save the shapes to `data/master_ref.npz`. |
| `fit_to_master.py` | Fit dvsim's joint stiffness and damping to those shapes. |
| `viz.py` | Drawing (matplotlib, Agg backend) and gif / html writers. |
| `tree_stages.py` | The tree expansion drawn in stages (used by the RRTs; also redraws a finished run). |
| `test_rrt.py` | Unit tests for the tree. |
| `just_sim.py`, `scrap.py` | Older interactive viewer (needs a display) and a cleanup snippet. Superseded by `rollout.py`. |

A tree node is a complete sim state: vine, boxes, and the curvature each existing body was grown
with. The control for an edge is one curvature (1/m) that applies to the part of the vine grown
during that edge; bodies that already exist keep theirs, the way a fabricated vine keeps its
actuators. `sim_rrt.py` samples and measures distance on the tip position only;
`tip_block_rrt.py` adds the box positions to both.

## Environments

The sim environments come from the sketches of 2026-10-20 and use the real dimensions in
`real_measurements.md`: 98 mm, 0.503 kg boxes; a 24.26 mm vine radius; a 1200 x 900 mm board.
There are three categories:

| Category | Envs | Task |
|---|---|---|
| `b2g_unretractable` | 1a one block, 1b two blocks in a row, 1c precise block pose; 2a wall-guided, 2b channel | Block to goal with one vine: the goal is close, or a wall guides the block. |
| `b2g_retractable` | 3a open board, 3b maze with vine-sized openings | Block to goal with several pushes, resetting the vine between them (`reset_action_rrt.py`). |
| `vine_to_goal` | 4a S-maze, no blocks; 4b block plugging a gap; 4c zig-zag barrier, two 70 mm gaps, only the lower block can be pushed clear; 4d clutter of 13 blocks | Tip to a goal region, past movable blocks. |

In 4c both gaps are narrower than a block (98 mm) and wider than the vine (48.5 mm). The upper
block sits in the pocket that opens toward the start, so a push only jams it in its gap. The
lower block sits in the pocket on the goal side, so the vine grows through the lower gap and
pushes it away. In 2b one block sits inside the channel; one push drives it into the curved back
wall, which guides it up and left along the top wall to the goal.

The sketches' basic contact experiments (one block, two blocks, block and wall) will come from
real data, not sim envs.

```bash
python envs/make_envs.py                              # write envs/*.json
python xnew_rrt/run_envs.py --draw-only               # envs/vis/<name>.png
python xnew_rrt/run_envs.py                           # plan on all; or list envs/<name>.json
```

`run_envs.py` writes, per env, under `out/envs/<name>/`:
- `env.png`;
- `tree.gif`, 40 frames spread over the run (`--tree-frames`, or `--tree-every` iterations);
- `solution.gif` and `solution.mp4`, the solution or the closest attempt (the title says which);
- `result.json`; and `out/envs/summary.json` over all envs.

Every env runs `kino.plan` with `--planner` (default `rrt`, the batched RRT over (tip, box
positions)), until the goal, the edge budget or a time limit (45 min; 90 min for the retractable
envs). Resets are on only for `b2g_retractable`. Use `--device cuda --batch-size 256` for the GPU
sim; the default is batch 32 on the CPU. Override with `--max-edges` and `--time-limit` (minutes).

The sim settings are in `run_envs.DEFAULT_SIM`:
- the real vine radius;
- 30 mm segments;
- the stiffness and damping fitted to the brick videos, rescaled to 30 mm segments;
- 1e-3 kg m^2 segment inertia;
- box friction 0.08;
- 20 N growth force;
- 0.25 m/s growth.

The JSON format:
- walls are segments with a thickness (oriented boxes in the sim);
- boxes have a center, an angle, a size and a mass;
- the goal is a box goal (box 0, a disc, optionally an angle and tolerance) or a tip goal;
- an optional `planner` block sets `max_bodies` and any key of `run_envs.DEFAULT_PLANNER`.

Where a sketch left the geometry open, the env's `notes` say what was chosen.

Results, 2026-10-23 (`out/envs/summary.json`, seed 0, 3 envs at a time on the shared 12-core
machine). "Off" is the box-to-goal or tip-to-goal distance at the end; the goal radii are 60–90 mm.

| Env | Result | Off (mm) | Pushes | Resets in tree | Nodes | Time (min) |
|---|---|---|---|---|---|---|
| 1a | solved | 59 | 1 | – | 257 | 1 |
| 1b | solved | 58 | 1 | – | 225 | 1 |
| 1c | solved (pose within 7 deg) | 21 | 1 | – | 7553 | 26 |
| 2a | solved | 70 | 1 | – | 2817 | 8 |
| 2b | failed: pins the block against the diagonal wall, never pushes it into the back wall | 517 | 1 | – | 13697 | 45 |
| 3a | failed, close | 95 | 5 | 83 | 22228 | 90 |
| 3b | failed: Euclidean distance leads the box into the dead-end bottom strip | 432 | 4 | 76 | 19533 | 90 |
| 4a | solved | 116 | 1 | – | 1025 | 4 |
| 4b | failed, close: pushed the block through the gap, the tip stopped short | 123 | 1 | – | 25249 | 45 |
| 4c | solved, through the lower gap | 113 | 1 | – | 6593 | 34 |
| 4d | solved | 112 | 1 | – | 449 | 19 |

With 20 iterations of 16 (2026-10-20) only 1b and 4d were solved, and the old reset planner left
3a at 550 mm and 3b at 713 mm.

### Block puzzles (5a–5c)

The puzzles come from the sketch of 2026-10-24 and are written by `envs/make_puzzles.py`; images are in `envs/vis/5*.png`.

- **Grid:** 110 mm cells, i.e. the 98 mm block plus 12 mm tolerance (`--cell`).
- **Walls** fill whole cells. They are merged into rectangles: 21 wall cells become 9 in 5c. That makes a step about 4% faster than one rectangle per cell (106 vs 111 ms at B=256). Merging helps little because the solver already drops far rows.
- **Spawn:** a ring of cells around the map. A vine grows straight (no actuation) from a ring cell's center, at any heading, and is retracted between pushes.
- **Puzzles:**
  - 5a: 2 pushes;
  - 5b: 3 pushes;
  - 5c: 6 pushes, two of which need a diagonal vine that a wall turns into a horizontal push.

`xnew_rrt/block_puzzle.py` is a hierarchical planner.

**High level.** Dijkstra over the block's cell.
- A move pushes the block k cells along an axis.
- It needs the k cells ahead free.
- It needs room for the vine behind the block, in one of two forms:
  - a straight free corridor back to the ring;
  - or a wall beside the cell behind the block, with the cell diagonally behind it free (a deflected vine).
- Deflected moves and moves that no wall stops cost more.

**Low level.** Realizes one move with the batched GPU sim.
1. **Candidates:** every ring cell × heading whose ray reaches behind the block. The headings are 16 uniform ones plus continuous headings aimed at the cell behind the block, the block's back face and the guide wall.
2. **Rollout:** all candidates grow in one batched rollout.
3. **Stop time:** per candidate, the step where the predicted resting pose is nearest the target cell. The block keeps sliding after the vine stops, v²/2μg, unless a wall stops it.
4. **Settle:** the best stop times are settled in a second batched run with growth off.
5. **Accept:** the move succeeds if the settled block is within 25 mm of the target cell's center and turned at most the jam angle. The jam angle is the most a 98 mm block can turn and still pass a one-cell corridor: 7.5° at 110 mm.
6. **Failure:** a failed move is banned and the high level replans.

A push that turns the block more than 15° on the way is flagged.

```bash
python envs/make_puzzles.py                                   # envs/5{a,b,c}.json + envs/vis/
python xnew_rrt/block_puzzle.py envs/5a.json envs/5b.json envs/5c.json   # out/puzzle/<name>/
```

Each run writes these under `out/puzzle/<name>/`:
- `plan.png`, the first high-level plan;
- `pushes.png`, the pushes executed;
- `solution.mp4`;
- `result.json`, with every push search, how many candidates reached the target, and how many turned past the flag or jam angle.

Results, 2026-10-24:

| Env | Result | Pushes | Push searches | Flagged > 15° | Time |
|---|---|---|---|---|---|
| 5a | solved | 2 | 2 | 0 | 1.5 min |
| 5b | solved | 3 | 3 | 0 | 2.9 min |
| 5c | not solved: 5 of 6 pushes | 5 | 9 | 0 | 6.4 min |

**5c, what fails.**
- **Rejected shortcut.** The high level first tried a 4-push shortcut: down from the start, with a vine deflected along the pillar. The sim rejected it, 46 mm short. It then followed the sketch's route.
- **Uniform headings can't reach behind the block.** The diagonal push has to thread a 62 mm window between the stub's corner and the pillar's corner, measured on the vine centerline. No ray from a ring-cell center hits it at 16, 32 or 64 uniform headings, because 45° rays pass exactly through grid corners. Aimed headings (38–44° from ring cell (0, 6)) thread it.
- **The block ends up turned.** Single-cell pushes in a corridor leave the block wedged at the jam angle. Any push that ends where corridors meet leaves it turned more than that, so it no longer enters the next corridor.
  - With 15° allowed, it got to the goal corridor's entrance turned 11.5°, and every vine wedged it there (`out/puzzle/aim_final15/`).
  - With 120 mm cells (`envs/5c_120.json`, 15° jam angle), 8 pushes reach the same entrance and the last push fails the same way (`out/puzzle/5c_120/`).
- **Uniform headings only** (`--no-aim`, `out/puzzle/noaim_h16/`): 5a and 5b solve, with one flagged push in 5b. 5c stops at the first diagonal push.

## Install

```bash
conda env create -f environment.yml
conda activate actvine
```

One env for everything: dvsim and `xnew_rrt/` (`torch`, `cvxpy`, `cvxpylayers`, ...) and master's
jax code (`pbd_vine.py`, `sPAM/`, `kinodynamic/`: `jax`, `flax`, `optax`, `jaxopt`, `ompl`,
`pygame`), plus `opencv-python-headless` and `scikit-image` for `video/`. Activating it makes jax allocate GPU memory only as needed, capped at 20% (the GPU is
shared); `JAX_PLATFORMS=cpu` keeps jax off it entirely. `qpth`, the optional batched GPU QP
solver, is not included because it no longer builds with current setuptools. The older `vine`
env on the lab machine still runs everything except the jax code.

## Run

Everything is headless and writes files, so it works over SSH. Open the outputs from the VS Code
explorer. Run from the repo root.

**One rollout**

```bash
python xnew_rrt/rollout.py                                  # default scene and controls
python xnew_rrt/rollout.py --controls "0:60,3:80,-3:120"    # curvature[1/m]:steps, ...
python xnew_rrt/rollout.py --env envs/env_dynamics.txt --max-bodies 100 --controls "0:200"
python xnew_rrt/rollout.py --plan out/rrt/plan.json         # replay what the RRT found
```

Writes `out/rollout/rollout.gif` (real-time playback) and `rollout_final.png`. With `--html` it also
writes `rollout.html`, a player with a frame slider and arrow-key stepping: open it with the Live
Preview extension, or run `python -m http.server` in the output folder and use the forwarded port.

**RRT**

```bash
python xnew_rrt/sim_rrt.py
python xnew_rrt/sim_rrt.py --env envs/env_dynamics.txt --max-bodies 100 --iters 500 --seed 1
```

While it runs, `out/rrt/tree_live.png` is rewritten every `--viz-every` iterations. Keep it open in
an editor tab; VS Code reloads it when it changes. At the end you get `tree_final.png`,
`tree_stages.png` / `.gif` / `.html`, `tree_log.pkl`, `plan.json` and `best_path.gif`.

The tree drawings show the expansion in stages (`--stages`, default 8): the tip paths added in a
stage in blue and earlier ones in grey, the branch closest to the goal with its vine and boxes,
and the push paths of the boxes. Box poses reached by the tree are a dot per node plus a faint
outline only for poses at least `--box-spread` mm (default 30) from every outline already drawn.
`tree_stages.png` has every stage side by side; the gif and html step through them. Redraw a
finished run with other settings without simulating again:

```bash
python xnew_rrt/tree_stages.py out/batch_tip_block_rrt --stages 6 --box-spread 40
```

**RRT over tip and box positions**

```bash
python xnew_rrt/tip_block_rrt.py                                # box 0 must end in (0.85, 0.60), r 0.06 m
python xnew_rrt/tip_block_rrt.py --box-goal 0.85 0.60 0.06      # the same, explicitly (x y r, m)
python xnew_rrt/tip_block_rrt.py --tip-goal                     # tip goal, boxes still in the metric
```

Same loop and outputs (under `out/tip_block_rrt/`). A sample is a tip position plus a position for
every box, and the distance adds the box displacements (`--box-weight`). The goal is box 0's
goal disc (on another scene give `--box-goal`), so "closest to goal" is about the box. It is the naive version:
box positions are sampled uniformly over the whole workspace, and orientation is ignored.

**Batched RRT**

```bash
python xnew_rrt/batch_tip_block_rrt.py --batch-size 32 --iters 20 --box-goal 0.85 0.60 0.06
```

Each iteration draws `--batch-size` samples, picks a node and a curvature for each, and simulates
all those edges in one batched rollout. `--iters` counts batched iterations. On the default
scene, 20 iterations at batch 32 built a 641-node tree in 59 s; a batched sim step at that size
runs about 7x more rollout-steps per second than batch 1.

**RRT with vine resets as an action (several pushes, one tree)**

`reset_action_rrt.py`, run through `run_envs.py` on the `b2g_retractable` envs. One tree over
(vine, box poses) with two kinds of edge:
- **grow**: the batched RRT's edge, from the node nearest a sample, at a random curvature
  (straight with probability 0.3);
- **reset**: the vine goes back to its stub at the base (retraction itself is not modeled) and the
  boxes stay where they are, at rest.

A node unlocks its reset once its boxes are at least `unlock_mm` (50 mm) from the boxes of every
reset taken so far, the root included, and the reset is added right away. The distance between two
box configurations is the largest, over the boxes, of center shift + half diagonal x turn (turn
modulo the box's symmetry). So each box configuration, at 50 mm resolution, gets one fresh vine:
resets only multiply when the boxes really move. Nothing is thrown away, so a push can start from
any configuration reached before, including after a set-up push that took the box away from the
goal. Samples are (tip, box positions) as in the batched RRT, so the nearest-node search spreads
the effort over box configurations by their Voronoi regions. The tree view marks each reset with
an x at the boxes. With resets off, the same code is the batched RRT with a vectorized
nearest-node search.

**RRT with vine resets (older: one tree per push)**

```bash
python xnew_rrt/reset_rrt.py --base 0.4 0.43 0 --box-goal 1.05 0.40 0.04 --max-episodes 6 \
    --episode-iters 14 --batch-size 32 --patience 5 --min-progress 1
```

A vine only grows, so one vine gives one push. Here the plan is a sequence of episodes, each a
batched RRT from a fresh vine with the boxes where the last push left them.
- The planner ends an episode when its budget runs out, or when box 0 has stopped getting closer
  to its goal for `--patience` iterations (counted only once it has started getting closer).
- It commits the branch that brought box 0 closest, then resets the vine. Retraction is not
  modeled: the reset is instantaneous and the boxes are at rest afterwards.
- `--reuse untouched` starts the next tree with every node whose vine stayed clear of the boxes,
  at both their old and new poses. Such a branch never touched a box, and vine-box contact is
  frictionless, so its motion carries over with the boxes swapped. `--reuse none` (default)
  starts fresh.
- `--straight-prob` (0.3) makes that share of edges grow straight. With uniform curvatures
  almost no branch keeps pushing.

On the command above:

| `--reuse` | Pushes | Box 0 progress (mm) | Total | Time |
|---|---|---|---|---|
| `none` | 6 | 15, 5, 7, 8, 4, 4 | 292 -> 250 mm | 169 s |
| `untouched` | 3 | 15, 3, 1 | 292 -> 274 mm, then stops | 66 s |

- Reuse is worse: its 150-250 carried-over nodes cover the workspace away from the box, and
  nearest-node selection keeps extending them.
- Every push is small for the reason a real vine has: off-center, the tip slides off the box's
  corner and turns it rather than driving it. The sampler is still the naive one
  (tip_block_rrt.py).

Outputs:
- `plan.png`: box 0's path and the vine at the end of each push;
- `best_path.gif`: the whole plan;
- `episode_<k>/`: each episode's tree, as `tree_stages`;
- `plan.json`: per push, the controls and the box poses before and after.

**High-level planner: block pushing on a grid**

```bash
python xnew_rrt/hierarchical_planner.py --example corridor         # also chain, reach, clear
python xnew_rrt/hierarchical_planner.py --map puzzle.txt --max-turns 2 --max-push 3
python xnew_rrt/hierarchical_planner.py --box-goal 1.0625 0.75      # the default scene, 100 mm cells
```

The planner assumes square boxes and walls, and cuts the space into box-sized cells. One action
is one vine deployment:
- The vine starts in a grow zone and grows inward along free cells: optionally at most
  `--max-turns` turns, and never folding back.
- It reaches a cell next to a box and grows into it, pushing the box (and every box lined up
  ahead of it) one cell the way it is heading, or up to `--max-push` cells by growing on
  straight.
- A wall, the grid's edge or the vine's own body past the last box blocks the push.
- The vine then resets.

Goals, in any combination:
- a box must end in a cell;
- cells must end up empty;
- the vine must be able to reach a cell (the corridor to it must be clear).

Grow zones are the free border cells: the ring just inside a wall frame, or cells marked `S`.

The search:
- A* over box positions, with the fewest pushes first, then the shortest vines.
- The heuristic is, per box, one push per axis it still has to move along (more beyond
  `--max-push`), with the max taken over boxes.
- Boxes stuck in a wall corner off their goal are pruned.

Input is an ASCII map (legend in the script's docstring) or a scene rasterized at box size,
aligned to box 0.

The plan (`plan.json`) lists per push:
- the launch cell and heading;
- the cells the vine grows through, which must be free;
- the push direction and distance;
- the boxes before and after.

For scenes, every cell also comes with its center in meters, as subgoals for the dvsim planners.

| Example | Pushes | What it takes |
|---|---|---|
| `corridor` | 6 | push B out of the shaft, then A down the shaft and along the bottom row |
| `chain` | 4 | push three boxes in a row together |
| `reach` | 1 | come around underneath and push A up into a pocket to free the row to G; with `--max-turns 2`, no plan |
| `clear` | 5 | clear a lane, then park A |
| default scene, goal (1.0625, 0.75) | 6 (2 with `--max-push 3`) | push box 0 east, then north |

**Launch-pose search (passive vine)**

```bash
python xnew_rrt/pose_search.py --box-goal 0.85 0.60 0.06 --base-line 0.3 0.25 0.3 0.75 --heading-range 60
```

For a vine with no actuators the plan is just where to put the base, which way to point it and
how long to grow. This tries `--num-poses` launch poses per round in one batch (`'linear'` mode),
scores each by the closest the box got to its goal, and repeats in a window half as wide around
the best. Writes `candidates.png`, `best.gif` and `result.json` under `out/pose_search/`.

**Stiffness mode.** `--stiffness-mode curvature|spam|linear` on `rollout.py` and the RRT scripts.
`curvature` and `spam` give the same motion; `spam` also works out the actuator pressure and rest
length for each commanded curvature. With `linear` the controls do nothing.

**Growth force.** `--grow-force N` (default 5, a placeholder) is how hard a blocked tip pushes, in
newtons. A vine that is stuck stalls, slides along the obstacle, or buckles back on itself,
depending on what gives first. `--grow-force 0` restores the hard constraint, where the tip
always advances at the growth speed.

**Parameter sweep.** `python xnew_rrt/sweep_vine_params.py` reruns the sweep behind the wrapper's
default segment mass and damping (see "What changed").

**Fit to master's vine.** Two steps, in two environments:

```bash
# 1. reference shapes from master's PBD sim (needs jax + flax; e.g. the `aura` env)
git archive master | tar -x -C /tmp/master_src
python xnew_rrt/master_reference.py --master-src /tmp/master_src          # -> xnew_rrt/data/master_ref.npz
python xnew_rrt/master_reference.py --master-src /tmp/master_src --l0-doubling  # old l0 bug, for comparison

# 2. fit (vine env)
python xnew_rrt/fit_to_master.py                   # 5 N growth force
python xnew_rrt/fit_to_master.py --grow-force 0    # hard growth constraint
```

The reference is master's planner configuration (68 mm bodies, sPAM designed by
`sPAM.nns_usage.solve`) grown 900 mm in 10 scenarios: free space at several curvatures, two
curvature switches, into a wall at 45 degrees, along a wall, past a post. `master_reference.py`
runs master's code unchanged from the archive (both bugs below are fixed on master since `befe1e1`). `fit_to_master.py`
regrows each scenario in dvsim with the same control schedule and scores the mean distance
between the centerlines at equal length, plus a penalty on joint oscillation. It searches
(stiffness, damping) with a cross-entropy method, one batch per generation (~3 min), and writes
`fit.json` and `compare.png` (final shapes, master vs dvsim). Rerunning the reference reproduces 9 of the 10 scenarios exactly; in `wall_parallel`
master's vine is pinned against the wall and creeps, and runs differ by a few millimetres there
(floating point order, CPU vs GPU).

Useful flags (all scripts): `--max-bodies` (vine length budget, 18 mm per body), `--grow-rate`,
`--vine-radius`. RRT only: `--steps-per-edge`, `--max-curvature`, `--goal-bias`.

Cost: a step takes about 30 ms at 60 bodies and about 130 ms at 100 bodies on CPU, batch size 1.

**Simulator demos and tests**

```bash
python dvsim/demos/render_demos.py     # gifs next to the script
python dvsim/demos/render_batched.py   # 64 vines in one batched solve
python -m pytest xnew_rrt/test_rrt.py     # needs: pip install pytest (inside the env)
```

## What changed on this branch

Relative to `creating_rrt`.

Simulator fixes:

- `step()` no longer modifies the state it is given. Growth used to rewrite the caller's tensors
  in place while the caller's body count stayed stale, which corrupted every tree node that was
  expanded more than once.
- Movable boxes given with `y1 > y2` (as the env files and the hard-coded scene do) got a negative
  half-height, so the vine passed through them. Corners are now sorted.
- Angles are wrapped where `atan2` meets a stored heading. A vine that curled past 180 degrees, or
  a start heading outside (-pi, pi), used to blow up.
- The first body is placed half a segment along the heading, which is where the base joint
  constraint expects it. It used to start 9 mm off and get yanked into place on the first step.
- `init_state_batched` indexed `bodies` with the wrong loop variable.
- `step(..., return_ok=True)` reports when the QP failed and the step was frozen. Without the flag
  the failure is still silent.
- `render_batched.py` called `init_state_batched` with too few arguments and required `qpth`.

Friction: each movable box has Coulomb friction against the table (`params.obj_friction`,
default 0.5). A box now stays put until it is pushed harder than `mu * m * g`, and a sliding box
decelerates and stops. This replaces `obj_damp = 0.2`, which made every box lose 80% of its
velocity per step whatever its mass. In the same push, a 0.05 kg box moved 98 mm, a 0.2 kg box
45 mm and a 5 kg box 14 mm (270 mm with friction off).

sPAM: the actuator is a constant moment per joint, evaluated once at its design radius, on top
of the vine's linear spring (`dvsim/spam.py`). The version before that (fixed strain, wrinkling
restoring moment, two hand-set gains) and `vine_wrinkle_moment` are gone.

Growth: optional force-limited growth (`params.grow_force`; one slack variable in the QP). The
planning scripts use it by default with a 5 N stall force. `dvsim` itself still defaults to the
hard constraint, which is what DiffVine had.

Vine parameters: the `Sim` wrapper uses segment mass 0.005 kg where `dvsim` defaults to 0.02 kg,
from a sweep of 96 combinations of inertia, mass, stiffness and damping scored on how well a free
vine follows a commanded curvature at 0.3 m/s (with `dvsim`'s defaults each joint oscillates by
37% of the commanded angle). Joint stiffness 0.2 and damping 1.24e-4 (dvsim: 0.3 and 5e-5) are
fitted to `master`'s simulator, see "Fitting to master". None of this is measured.

Batching: `step()` takes per-element parameters, and when a batched QP solve fails it re-solves
element by element and freezes only the ones that fail (before, one infeasible element froze the
whole batch).

Removed: `from_diffvine/` and `reference/` (nothing imported them; the DiffVine fitting scripts
are still in the history of `dysim_fix` under `training/`).

Removed: the second movable-object implementation inside the PBD simulator (`pbd_vine.py`,
`kinodynamic/sst.py`, `render.py` and four callers). Those files are back to their `master`
versions.

Planner:

- Controls now reach the simulator. Before, the sampled pressure and actuator length were written
  to `params.spam_*` while the sim ran in `'linear'` mode, which ignores them, so every rollout
  was the same straight vine.
- An edge is 25 sim steps instead of one (one step is only a few mm of growth).
- Headings are radians everywhere. The start heading was documented as degrees and passed straight
  to the sim.
- The control is stored per body in the node, instead of overwriting one global value for the
  whole vine.
- Fully grown nodes and nodes that keep failing are retired instead of being re-selected forever.
- The distance function no longer adds a signed angle difference (degrees against radians) to
  millimetres. It is tip distance only.
- Tree bookkeeping lives on the tree instance, not on the `Node` class. Tests updated to match.
- Scenes can come from `envs/*.txt`. Seeded sampling, goal bias, CLI flags.
- No JAX import and no model load at import time; no dependence on output folders that are not
  in the repo.

New: `scene.py`, `viz.py`, `rollout.py`, `tip_block_rrt.py`, `batch_tip_block_rrt.py`,
`pose_search.py`, `draw_envs.py`, the `'curvature'` stiffness mode.

## The batched simulator

**Two simulators, two kinds of batching.**
- Master's planner (`kinodynamic/sst.py`) runs `pbd_vine.py`: JAX, soft constraints, a fixed 15
  gradient steps per step. One fixed-size computation, so `jit` + `vmap` run thousands of vines per
  GPU call.
- `dvsim` (torch) has hard constraints: every step solves an optimization problem with the
  contacts and joints as constraints. Until 2026-10-23 that solve was cvxpylayers -> diffcp ->
  Clarabel, one element at a time on a CPU thread pool.

**Now the whole batch steps together, on the GPU.** Opt in with
`SimConfig(qp_backend="TORCH_IPM", device="cuda")`, or `--device cuda` on `run_envs.py` and
`ablation.py`. States stay on the CPU between steps; each batched step moves its inputs to the GPU
and back. The default is still DIFFCP on the CPU.

1. **Device.** `Sim.step_batch` runs the step inside `torch.device("cuda")` with a GPU copy of the
   params. Every bare `torch.zeros` / `eye` / `arange` in dvsim follows the default device, so
   dvsim's files needed no device plumbing.
2. **Jacobians** (`dynamic_vine.proximity_jacobians`). The vine–box gap Jacobians were
   `jacrev` of the whole (bodies x boxes) gap table: one backward pass per output, each over every
   pair, so (bodies x boxes)² work and memory. Gap (j, o) only depends on body j and box o, so one
   small gradient per pair gives exactly the same numbers (max difference 0.0, 1 and 13 boxes) at
   bodies x boxes cost. This also speeds up the CPU path: 4d at B=64 went from 20 to 70
   element-steps/s.
3. **The QP solve** (`dvsim/batched_qp.py`, plain torch, float64):
   - **Friction linearized.** ||(vx, vy, c w)|| becomes two nested polygons, 16 and 9 directions,
     with two extra variables per box. That keeps 96% of the norm in the worst direction; the
     weights are rescaled so it is unbiased on average.
   - **Rows that cannot bind are dropped.** These are rows with a gap over 60 mm (a body would have
     to move that far in one step) and all-zero padding rows. On 1a, 125 rows become about 28.
   - **Equalities eliminated by variable reduction.** A pivoted LU of A' gives x = x0 + Z u, which
     leaves 65 free variables of 184 for one box. Batched LU is 60x faster than batched QR on the
     GPU: 10 ms vs 650 ms at B=1024.
   - **Mehrotra interior point** on the inequality-only problem: one batched Cholesky of a k x k
     matrix per iteration. Each element converges or fails on its own; a failed element comes
     back NaN and is frozen, and the rest of the batch is unaffected.

**Accuracy** (`python xnew_rrt/check_batched_qp.py [--device cuda]`), on 33 QPs recorded from a
push in 1a:

| Check | Max error / max velocity |
|---|---|
| Same linearized QP solved by Clarabel | 1e-6 |
| Exact friction cone (the old path), vine | 0.2% |
| Exact friction cone (the old path), box | 0.75% |
| With vs without dropping far rows | 3e-6 |

Whole rollouts (330 steps, 4 curvatures) end within 2.3 mm of the old path, except the head-on
push. That one is chaotic on the old path too: nudging its curvature by 0.02 1/m moves the box
230 mm.

**Speed** (`python xnew_rrt/bench_batch.py ... --backend TORCH_IPM --device cuda`). A vine about
0.55 m long, element-steps per second. The 4090 was running another user's job at 93% the whole
time. "Best" is the old path's best over B ≤ 64.

| Env (boxes) | diffcp, CPU (best) | GPU, B=64 | GPU, B=256 | GPU, B=1024 |
|---|---|---|---|---|
| 1a (1) | 292 | 519 | 1,768 | 3,415 |
| 4c (2) | 248 | 715 | 2,300 | 4,747 |
| 4d (13) | 70 | 423 | 838 | 15 (memory thrash) |

A GPU step costs about 120 ms at any batch up to about 256. That floor is launch latency: about
47 ms for the forward pass and about 60 ms for the interior point loop, each made of many small
kernels. Past that, the cost grows slowly. So use batches of 256 or more.

**JAX?** (`python xnew_rrt/bench_jax_qp.py`: the same recorded problems, linearized and culled,
on the GPU in float64.)

| Solver | B=256 | B=1024 | Error vs Clarabel |
|---|---|---|---|
| torch `batched_qp` | 69 ms | 198 ms | 3e-6 |
| qpax (JAX interior point, `jit` + `vmap`) | 1,366 ms | 6,405 ms | 4e-7, never reports converged |
| MPAX r2HPDHG (JAX first-order, cvxpylayers' MPAX backend) | 8,035 ms | 33,921 ms | diverges (3e27) |

- qpax factors the full KKT system (about 330 x 330 per iteration) instead of eliminating the
  equalities.
- First-order methods don't handle this ill-conditioned problem; SCS had the same trouble.

So the solver stays in torch. A JAX port of the forward pass could fuse its ~47 ms of launches
into a few kernels. That saves at most about a third of a step at B≤256, and torch can get the
same with CUDA graphs or `torch.compile`. Not worth a port.

**Will the Jacobians become the bottleneck?**
- **With 1–2 boxes, no.** The forward pass is a flat ~50 ms launch latency from B=1 to B=1024,
  while the solve grows with B.
- **With many boxes, yes, through memory rather than flops.** The assembled problem is dense:
  bodies x boxes vine–box rows plus boxes² box–box rows, by about 250 columns. At 13 boxes and
  B=1024 that is about 2.8 GB of float64. At the memory cap the allocator thrashes, which is the
  70 s per step in the table.
- **The fix:** cut rows before assembling, not after. Compute the gaps (cheap), keep the pairs
  under the 60 mm threshold, and build Jacobian rows only for those. Rows then scale with
  contacts, not with every pair. Not done yet.
- **Smaller wins left:**
  - CUDA graphs or `torch.compile` for the forward pass and the interior point loop: the launch
    latency is most of a step at B ≤ 256;
  - keeping states on the GPU for a whole edge instead of copying them back every step.

## Batched kinodynamic planners and the funneling ablation

`xnew_rrt/kino.py` is one batched loop: pick a batch of (node, curvature) pairs, simulate all
those edges in one batched rollout, insert the results. The planners differ in how they pick and
what they keep:

| Planner | Pick | Keep |
|---|---|---|
| `rrt` | nearest node to a uniform (tip, box centers) sample, 10% goal bias | every node |
| `rrt_grid` | same | a node only if its fine cell is empty: SST's pruning with grid cells instead of witnesses, no costs |
| `kpiece` | a coarse cell by KPIECE importance (new, rarely picked, on the frontier, few nodes), then a node in it | every node |
| `kinopax` | a coarse region by Kino-PAX-style weight (expansions from it that reached a new cell, few tries, few nodes) | a node only if its fine cell is empty |
| `*_shape` | as `rrt_grid` / `kinopax` | fine cells also cut two points on the vine's body |

**Cells.** Fine cells cut the tip x, y and every box center at 15 mm and the tip heading at 30°.
Coarse cells cut the same positions at 75 mm, without the heading. Keys are hashed tuples, so many
boxes cost nothing. Resets work in every planner, with the same 50 mm unlock rule as before.

**Why sparsity might matter: the vine funnels** (`python xnew_rrt/funneling.py out/envs`, on the
full-budget CPU trees of 2026-10-23). Take sibling edges, children of one node grown with
curvatures at least 1 1/m apart. When the tip ends in contact (within 5 mm of a wall or box),
42–84% of those pairs end within 2 mm of each other; when it ends free, 1–15% do (4b: 49%). Over
all nodes, 49–88% sit within 2 mm, in tip and boxes, of an earlier node.

**Ablation** (`python xnew_rrt/ablation.py`, then `--summarize`). Every env, every planner, seeds
0–2. Budget: 16,384 edges in batches of 256, on the GPU sim, three processes sharing the 4090.
Each cell is solved/3 runs and, when solved, the median edges and minutes to the goal; when not,
the closest distance in mm.

| Env | rrt | rrt_grid | kpiece | kinopax | rrt_grid_shape | kinopax_shape |
|---|---|---|---|---|---|---|
| 1a | 3/3: 2048, 0.5 min | 3/3: 2048, 0.5 min | 1/3: 1792, 0.4 min | 3/3: 2048, 0.6 min | 3/3: 2048, 0.5 min | 2/3: 2048, 0.5 min |
| 1b | 3/3: 3072, 0.8 min | 3/3: 2048, 0.5 min | 1/3: 2048, 0.6 min | 3/3: 2048, 0.6 min | 3/3: 2048, 0.5 min | 2/3: 2048, 0.6 min |
| 1c | 1/3: 13824, 4.4 min | **3/3: 1792, 0.4 min** | 1/3: 1792, 0.4 min | 2/3: 1792, 0.5 min | 3/3: 2816, 0.7 min | 2/3: 1792, 0.4 min |
| 2a | 3/3: 3072, 0.9 min | 3/3: 3328, 1.0 min | 1/3: 9984, 2.8 min | 3/3: 2560, 0.7 min | 3/3: 2304, 0.6 min | 2/3: 2560, 0.7 min |
| 2b | 1/3: 12288, 3.4 min | 0/3 (518) | 0/3 (515) | 0/3 (219) | 0/3 (211) | 0/3 (481) |
| 3a | 1/3: 7936, 2.5 min | 1/3: 13056, 4.3 min | 0/3 (116) | 1/3: 11776, 4.3 min | 0/3 (109) | 1/3: 15872, 5.3 min |
| 3b | 0/3 (632) | 0/3 (567) | 0/3 (447) | 0/3 (452) | 0/3 (525) | 0/3 (423) |
| 4a | 3/3: 5888, 1.6 min | 3/3: 5888, 1.6 min | 3/3: 6912, 1.8 min | 3/3: 6400, 1.7 min | 3/3: 6400, 1.5 min | 3/3: 6912, 1.2 min |
| 4b | 1/3: 3584, 0.9 min | 0/3 (525) | 1/3: 15360, 5.1 min | 1/3: 3328, 1.1 min | 1/3: 7168, 1.6 min | 1/3: 3584, 0.8 min |
| 4c | **3/3: 4608, 1.3 min** | 0/3 (536) | 2/3: 15872, 4.2 min | 0/3 (536) | 1/3: 4096, 0.8 min | 2/3: 16128, 3.5 min |
| 4d | 3/3: 3328, 2.2 min | 3/3: 3584, 2.5 min | 3/3: 3328, 2.3 min | 3/3: 3328, 2.2 min | 3/3: 3328, 1.4 min | 3/3: 3328, 1.5 min |
| solved | **22/33** | 19/33 | 13/33 | 19/33 | 20/33 | 18/33 |

**What it says.** Three seeds is few, so differences of one or two runs are noise.
- **Pruning pays off where contact funnels the tree.** On 1c (a short push that must also turn
  the block), `rrt_grid` solves 3/3 in about 1,800 edges with about 115 nodes. Dense `rrt` solves
  1/3 and needs 13,824 edges.
- **Pruning on the tip and boxes alone throws away what gaps need.** 4b and 4c need the vine
  threaded through a gap. Two states with the same tip but a different body path count as
  duplicates, and the first one kept is usually the wrong one. `rrt_grid` and `kinopax` drop to
  0/3 on 4c.
- **Adding two body points to the cell partly recovers this:** `kinopax_shape` 2/3 and
  `rrt_grid_shape` 1/3 on 4c, against 3/3 for dense `rrt`.
- **Picking cells without pruning (`kpiece`) is the weakest.** With 256 edges per batch it keeps
  re-picking crowded cells.
- **Pruned trees start slowly with big batches.** From the stub, 256 random curvatures land in
  3–4 fine cells, so most of the first batches are pruned. On open boards that costs as much as
  the pruning saves.
- **Nobody solves 3b, and 2b is solved once (`rrt`).** These need long, aimed pushes, and the
  random-curvature edges rarely make them.

So the hypothesis holds for what the projection captures: duplicates really are duplicates in tip
and box space, and pruning them speeds up contact-heavy pushes. The projection has to include
enough of the vine's body for tasks where the body path matters.

**Next steps:**
- grow the pruning resolution as the tree fills, starting coarse;
- prune by tip, boxes and body shape at the coarser level, keeping more than one node per cell;
- use the funneling directly: when siblings collapse onto one state, stop sampling that parent's
  curvature range.

## Known problems

Not fixed yet, roughly in order of how much they matter.

1. **Friction is only box-on-table.** Contacts (vine against box, box against wall or box) are
   still frictionless, and nothing holds the vine body to the ground: the grown part swings
   freely about the base. `obj_friction` is a guess, not a measurement. The friction term is a
   norm, so the GPU `qpth` path (pure QP) refuses to run with it.
2. **The vine in the sim is about 1000x softer than a real one**, so sPAM pressures are not
   meaningful: holding a 0.5 m bend takes 15 Pa in the sim, against tens of kPa for a real
   actuator. The stiffness cannot simply be raised, because the bending torque is integrated
   explicitly and anything much stiffer is unstable at `dt = 1/90 s`
   ([dvsim/README.md](dvsim/README.md), section 8). Curvatures are still consistent, since the
   same stiffness designs the actuator and simulates the vine.
3. **Fast growth does not track.** The vine follows a commanded curvature at 0.1 and 0.3 m/s. At
   0.5 m/s no parameter combination tried does: joints differ from each other by more than the
   commanded angle.
4. **With force-limited growth the vine folds before it pushes.** On the default scene the box
   (0.2 kg, about 1 N of friction) starts 174 mm from its goal. The launch-pose search got it to
   81 mm with the hard growth constraint, but only to 159 mm with a 5 N stall force and 155 mm
   with 20 N: the soft vine buckles back on itself first. The batched RRT with 5 N left the box
   170 mm away. Pushing needs a stiffer vine (item 2) or a lighter box. The tip+block planners
   also still sample box positions uniformly over the workspace and ignore orientation.
5. **The goal test is generous**: the tip body only has to overlap the goal disc.
6. **Batching is only as parallel as the QP solver.** The Jacobians are vectorised, but
   cvxpylayers hands the batch to diffcp, which solves each QP separately on a CPU thread pool.
   Throughput tops out near 420 rollout-steps/s at batch 128 (47 at batch 1, 12 cores). The
   cached QP layer in `solver.py` is a module global that has to be reset when the scene changes.
7. **The growth force is a constant.** 5 N is a guess. It does not build up while the vine is
   stuck and does not depend on vine length. With the hard constraint (`--grow-force 0`) a fully blocked
   vine makes the QP infeasible; `step()` then freezes that element and the planner drops the edge.
8. **No measured parameters.** Vine mass and inertia are DiffVine's starting values; box masses
   are made up; the sPAM numbers are tube sizes and a 1.5 psi vine pressure. Stiffness and
   damping were fitted to `master`'s simulator, not to a real vine, and the fit pins them down
   only loosely (see "Fitting to master"). `VineParams.opt_params()` and the differentiable
   solve are still there for system identification against real data.
9. **Repo hygiene.** `__pycache__/*.pyc` files are tracked despite `.gitignore`, so running anything
   dirties the working tree. Demo docstrings contain another
   machine's Python path.

## Video tracking

`video/` extracts the vine's centerline and the boxes' poses from an overhead video of a trial
and hands them to the same fit that matched dvsim to master (`xnew_rrt/fit_to_master.py`).

- **Calibration** (`calibrate.py`) is a homography from the image to the board, in the board
  frame of the env files (mm, y up). It is fitted from four corners of known board position,
  either the board's own corners or any other known rectangle (`corners_mm`, e.g. the inner
  corners of the boundary walls). The corners are clicked by hand or found automatically.
  Extra known points, such as obstacle corners, refine the fit and report residuals.
  Optional extras:
  - a checkerboard video for lens distortion;
  - the camera height, to correct parallax: the vine's centerline and the box tops are above
    the board, so an off-center camera sees them pushed outward (about 35 mm for 100 mm of
    height at 700 mm off-axis with the camera 2 m up).
- **Vine** (`track.py`) follows DiffVine's pipeline:
  - The vine is what differs from a reference frame taken before it grows, after blurring and
    morphological cleanup.
  - The part touching the base is skeletonized and turned into the longest path from the base.
  - The path is resampled. The tip end stops at the center of the rounded cap, a radius short of
    the mask's edge, like the sim's last body center.
  - The vine radius is measured from the mask.
- **Boxes** (`track.py`) are tracked by template matching their reference-frame appearance over
  small shifts and turns, starting from a rough pose that is refined on the reference frame.
  - The background the vine is compared to has the boxes painted out, so table uncovered by a
    moving box does not count as vine.
  - Each box's current footprint is removed from the vine's mask.
- **Link to system identification** (`to_reference.py`):
  - writes the tracks in `master_ref.npz`'s format: a frame every 20 mm of growth, the trial's
    walls, base and control schedule, the boxes and their tracked poses, and the free growth
    speed (estimated from the video);
  - `fit_to_master.py --ref` then fits stiffness and damping. For trials with boxes it also
    scores the box centers (`BOX_WEIGHT`), with the box friction as `--obj-friction`.

```bash
python video/calibrate.py grid  trials/push01.json   # frame with a pixel grid: read corners off it
python video/calibrate.py check trials/push01.json   # calib_check.png: walls projected on the frame
python video/track.py trials/push01.json             # out/video/push01/: tracks.npz, overlay.mp4, summary.png
python video/to_reference.py out/video/push01 out/video/push02 --out xnew_rrt/data/video_ref.npz
python xnew_rrt/fit_to_master.py --ref xnew_rrt/data/video_ref.npz --out out/fit_video
```

`video/trial_example.json` lists every trial field. A trial needs:
- the video and the board size;
- the corners (`"auto"`, or pixel coordinates);
- the walls (or an env file);
- the base pose;
- each box's size, mass and rough starting pose;
- for the fit, the control schedule: the curvature each length of vine was built or actuated
  with.

**Synthetic check** (`synthetic.py`): a dvsim push rendered as an oblique, noisy camera view,
with the board corners found automatically. Results:
- centerlines within 1.5 mm (5.8 mm when the vine folds onto itself behind the box);
- box centers within 1.5 mm and 0.5 degrees;
- vine radius 50 mm measured as 49.5-50;
- the fit on the tracks recovers the stiffness that made the video (0.20).

```bash
python video/synthetic.py make --out out/video_synth
python video/track.py out/video_synth/trial.json --out out/video_synth/track
python video/synthetic.py check out/video_synth
```

Things to know before recording:
- **Avoid dead-center pushes for identification.** A vine pushing straight through a box's center
  line buckles off one side or the other depending on a millimeter. In the synthetic test, a
  1 mm calibration offset flipped which way the sim's vine slid, so the error was 110 mm with the
  true parameters. Launch the vine 20 mm or more off the box's center line.
- The silhouette cannot separate overlapping parts of the vine, or the vine from its own shadow.
  For a light vine on a darker table, use `"diff_mode": "brighter"` to drop shadows. Keep hands
  out of view, or at least away from the vine.
- Keep the camera still. Calibration and the reference frame are taken once per video.

### The brick demos (`video/demos/`)

There are 15 videos of the same setup:
- a pegboard with 1 in hole pitch;
- the vine's mount bolted at the top right;
- a clear tube, unactuated, growing straight into a brick at different poses;
- the brick glancing off the vine and shifting 11-133 mm.

`video/trials/make_trials.py` writes one trial json per video. The pipeline, extended for this
setup:

- **Pegboard calibration** (`pegboard_pitch_mm`). Every hole is a calibration point (about 1400
  per video): each hole is indexed by stepping hole to hole from the center. A two-term radial
  distortion is fitted along with the homography.
  - Residuals are 0.5 mm median and 1 mm at the 95th percentile, well under a pixel at
    0.83 px/mm.
  - The distortion comes out the same in every video (k1 = 0.035).
  - The board frame is anchored on the outermost rows and columns that are at least 3/4
    visible: the frame's lip hides the outermost holes differently in each video. Every video
    gets the same 45 x 33-hole frame. The bolted mount lands at the same board position in all
    15 videos, within +-5 mm.
- **Clear tube** (`"diff_mode": "saturation"`). The tube barely changes brightness, but it
  drops the orange board's saturation. The vine is where saturation falls below the reference
  frame's, with the holes smoothed away, the vine stub and mount painted out of the reference,
  and the background slowly following lighting changes away from the vine.
  - The reference frame is taken just before growth starts (`GROWTH_START`). In IMG_3497 the
    light changes while the vine waits.
  - Mask pieces up to `gap_mm` apart are joined, so the vine survives dropouts.
  - Everything behind the base is dropped, which removes the mount.
- **Brick** (`"box_channel": "hue"`, `"pose_mm": "auto"`). It is found and tracked on a map of
  how far each pixel's hue is from the board's: a clean silhouette, without the board's holes,
  which would otherwise pin the template to where it started.
  - Brick size: a standard 203 x 92 x 57 mm brick. Its top face looks 5.5% larger than that,
    which puts the camera about 1.1 m up; `camera_height_mm` corrects the parallax of the brick
    top and the tube's centerline.
  - The brick's mass is unknown; 2.3 kg is assumed. Only mass times friction decides when it
    slides, so the fitted friction is for that mass.
- **Launch heading** (`to_reference.launch_heading`). This is fitted to the whole straight vine
  just before it first touches the brick.
  - It differs by up to 6 degrees from the first 6 cm of the stub, which is what the tracker
    alone measures.
  - It decides which side of the brick the vine passes on. The tip's rolling contact sends any
    off-center hit to one side, so the sim needs the launch heading to a fraction of a degree.

```bash
python video/trials/make_trials.py
for t in video/trials/IMG_*.json; do python video/track.py $t; done        # ~15 s per video on CPU
python video/to_reference.py out/video/IMG_* --out xnew_rrt/data/video_ref.npz
python xnew_rrt/fit_to_master.py --ref xnew_rrt/data/video_ref.npz --vine-radius 0.025 --max-bodies 70 \
    --grow-force 20 --seg-inertia 1e-3 --start 3 5 0.2 --out out/fit_video_gf20
```

**Fit to the 15 videos.** The error is the mean centerline distance at equal vine length, plus
the box-center error and the jitter penalty. Settings: vine radius 25 mm, 1e-3 kg m^2 segment
inertia, brick 2.3 kg. Search: cross-entropy over stiffness, damping and box friction, 12 x 5
candidates.

| Growth stall force | Start (stiffness 3, damping 5, friction 0.2) | Best | Stiffness | Damping | Box friction |
|---|---|---|---|---|---|
| 20 N | 74.6 mm | 46.0 mm | 40 | 17 | 0.097 |
| 40 N | 73.1 mm | 46.5 mm | 36 | 24 | 0.115 |

Per video, 20 N: 11 of 15 within about 50 mm (22-50 mm). The misses:
- IMG_3501, 126 mm: the sim's vine passes the brick on the other side.
- IMG_3522, 95 mm: the real brick slid 133 mm, much further than the sim pushes it.
- IMG_3497, 76 mm and IMG_3518, 54 mm: the real vine rotates further about its base than the
  sim's.

What the fit says:
- **The vine is much stiffer than master's quasi-static vine suggested** (stiffness ~40 vs 0.2).
  It stays straight and pivots at the base, which is what the videos show. Explicit bending at
  that stiffness is only stable with heavier segments (1e-3 vs 1e-5 kg m^2), so the inertia is a
  stand-in for the semi-implicit torque.
- **The stall force is not identified.** Friction holds the brick with mu m g = 0.10 x 2.3 x 9.81
  = 2.2-2.6 N, far below either stall force, so the vine never stalls on it.
- **What a lighter block would identify:** a block heavy enough to stall the vine would pin down
  the stall force. A lighter block at a known mass separates mu from m.
- **Launch-side errors:** vine-box contact is frictionless, matching the rolling tip (low tip
  friction). The remaining side errors come down to the launch heading.

## Fitting to master

Result of `fit_to_master.py` (dvsim 'curvature' mode, 18 mm segments, 0.3 m/s growth, against
master's shapes with the l0 bug fixed). Error is the mean centerline distance over the 10
scenarios, including the oscillation penalty (about 1 mm for a stable vine).

| Growth | Previous defaults (stiffness 0.3, damping 2.0) | Best found |
|---|---|---|
| 5 N stall force | 16.9 mm | 15.4 mm (stiffness 0.20, damping 1.24) |
| hard constraint | 17.2 mm | 16.5 mm (stiffness 0.098, damping 0.50) |

Per scenario, 5 N growth, previous -> fitted: straight 1.7 -> 1.7 mm, 1 1/m 6.8 -> 8.2, 2 1/m
18.6 -> 17.7, -2 1/m 18.0 -> 17.0, 3.33 1/m 24.7 -> 15.2, S-curve 24.3 -> 18.3, late bend 5.7 -> 6.0,
45-degree wall 35.9 -> 36.5, along a wall 10.3 -> 11.1, past a post 14.7 -> 14.5.

- dvsim already bent much like master; the fitted values mainly help the tightest bends. The
  `Sim` wrapper now uses them (`SimConfig.joint_stiffness = 0.2`, `ang_damp = 1.24e-4`).
  Joint oscillation is the same as before with the 5 N growth force.
- Stiffness and damping are barely identifiable from these shapes. In dvsim the joints settle at
  the commanded angle whatever the stiffness, and master's vine is quasi-static, so the shapes
  only say "stable and settles quickly". Any stable pair with stiffness 0.1-0.3 and damping
  0.5-2 lands within about 2 mm.
- Without the oscillation penalty the search goes to stiffness ~0.32, damping 0.27 (13.0 mm).
  There the joints oscillate (85% of the commanded angle at 3.33 1/m) but the averaged shape
  still matches. 0.3 is where dvsim's explicit bending torque stops being stable at this dt.
- What is left is structural. dvsim starts each bend a few centimetres later than master, which
  only a stiffer vine would fix (and that needs the semi-implicit change). Into a wall at 45
  degrees (36 mm) master's vine pivots at the base and stays straight, while dvsim's bends
  where it meets the wall.

## Bugs found on master

Found while generating the reference. Both are fixed on `master` in `befe1e1`; this branch's
`sPAM/` still has the undergrad's copies (see the end of item 2).

1. **The sPAM surrogate does not load.** The only checkpoint committed is
   `sPAM/model_.../checkpoint_100.orbax-checkpoint-tmp-0`, an orbax save that never finished.
   `flax.training.checkpoints.restore_checkpoint` finds no finished checkpoint and returns the
   random initial network without an error, and `get_or_train_model` prints "loaded
   successfully". On a fresh checkout its predictions are off by 0.53 rad in phi and 0.40 in m
   (on its own `data.csv`), `solve` returns negative pressures, and the vine stays straight for
   any commanded curvature below about 3. The fix replaces it with a finished checkpoint, trained
   with master's own training functions and settings on the committed `data.csv` (error 0.021 /
   0.012), and `get_or_train_model` now raises when there is no finished checkpoint. The torch
   port on this branch (`sPAM/model_.../checkpoint.pt`) is fine (0.021 / 0.014).
2. **The designed rest length is doubled.** `sPAM.nns_usage.solve` picks the actuator at rest
   length `l0` but returns `l0 * 2.0`, and `solve_fwd` evaluates the actuator at the length it
   is given. The vine then settles at the wrong curvature: commanding 1, 2 and 3.33 1/m gives
   2.9, 1.9 and 1.0 1/m. Returning `l0` as designed puts the equilibrium within 0.02 1/m of the
   command for 0.5-3.33 1/m. On this branch `sPAM/nns_usage.py` and `sPAM/torch_nns_usage.py`
   still double it; nothing in `dvsim` uses either.

Results produced by master's planner before `befe1e1` depend on how both of these behaved when they were run.

## Original paper

The ActVine README (install for the JAX stack, `kinodynamic/sst.py` usage, troubleshooting) is on
`master`.
