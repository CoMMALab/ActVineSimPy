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
| `sPAM/` | used | sPAM actuator model. `torch_*.py` are the Torch ports (`torch_nns.py` = neural surrogate, `torch_nns_usage.py` = bend radius -> pressure / actuator length). The non-`torch_` files are the JAX originals and need `jax`. |
| `envs/` | used | Environment files (mm). `dynamic_obstacles:` lines are movable boxes: `x1 y1 x2 y2 mass_kg inertia`. `envs/vis/` has a picture of each one (`python xnew_rrt/draw_envs.py`). |
| `pbd_vine.py`, `render.py`, `kinodynamic/`, `geometric/`, `experiments/` | from master | ActVine PBD simulator, pygame renderer, SST planner, biarc RRT*, paper experiments. Identical to `master` on this branch, except `kinodynamic/env_loader.py`, which also parses `dynamic_obstacles:`. Needs `jax`, which the current `vine` env does not have. |
| `cache/`, `pics/`, `solution.png`, `title.gif`, `x_progress_log.txt` | leftovers | Outputs and notes from earlier runs. |

### `dvsim/`: the simulator

[dvsim/README.md](dvsim/README.md) explains the solver in detail: why a step is a QP, how that
relates to SQP, how it is solved and batched, box friction, and how the sPAM model fits in.

| File | Contents |
|---|---|
| `vine.py` | `VineParams`, state layout, growth (`extend`), the constraint functions (`sdf`, `joint_deviation`, `growth_rate`), bending torques (`bending_energy`), and `forward_batched_part`, which evaluates them and their Jacobians for one batch element. |
| `dynamic_vine.py` | Movable boxes: vine-box, box-wall and box-box gap functions, assembly of the full QP (`dynamic_solve`), and `step()`, the single entry point. |
| `solver.py` | The QP solve: cvxpylayers + Clarabel on CPU (differentiable), or `qpth` on GPU if installed. |
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
| `tip_block_rrt.py` | The same RRT over (tip position, box positions), with an optional goal for a box. |
| `batch_tip_block_rrt.py` | That planner with a batch of edges simulated together per iteration. |
| `pose_search.py` | Launch-pose search for a passive vine pushing a box (for the live demo). |
| `rollout.py` | Simulate one rollout from a list of controls (or replay an RRT plan) and render it. |
| `draw_envs.py` | Draw every `envs/*.txt` into `envs/vis/`. |
| `sweep_vine_params.py` | Sweep vine inertia, mass, stiffness and damping; score curvature tracking. |
| `master_reference.py` | Run `master`'s PBD vine (jax) on a set of scenarios and save the shapes to `data/master_ref.npz`. |
| `fit_to_master.py` | Fit dvsim's joint stiffness and damping to those shapes. |
| `viz.py` | Drawing (matplotlib, Agg backend) and gif / html writers. |
| `test_rrt.py` | Unit tests for the tree. |
| `just_sim.py`, `scrap.py` | Older interactive viewer (needs a display) and a cleanup snippet. Superseded by `rollout.py`. |

A tree node is a complete sim state: vine, boxes, and the curvature each existing body was grown
with. The control for an edge is one curvature (1/m) that applies to the part of the vine grown
during that edge; bodies that already exist keep theirs, the way a fabricated vine keeps its
actuators. `sim_rrt.py` samples and measures distance on the tip position only;
`tip_block_rrt.py` adds the box positions to both.

## Install

```bash
conda activate vine
```

The `vine` env on the lab machine has what the live code needs: `torch`, `cvxpy`, `cvxpylayers`,
`diffcp`, `clarabel`, `scipy`, `matplotlib`, `pillow`, `scikit-learn`, `pandas`, `pytorch-minimize`
(`torchmin`), `torchimize`. Optional: `qpth` + CUDA for the batched GPU solve, `pytest` for the tests.
The master-era code (`pbd_vine.py`, `kinodynamic/`, `sPAM/spam.py`, `sPAM/nns.py`) additionally needs
`jax jaxlib jaxopt flax numba pygame ompl`.

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
an editor tab; VS Code reloads it when it changes. It shows every edge as the path the tip actually
took (shaded by tree depth), the branch closest to the goal with its vine shape and box pose, where
the other leaves left the boxes, and the latest sample and new node. At the end you get
`tree_final.png`, `tree_growth.gif`, `plan.json` and `best_path.gif`.

**RRT over tip and box positions**

```bash
python xnew_rrt/tip_block_rrt.py                                # tip goal, boxes in the metric
python xnew_rrt/tip_block_rrt.py --box-goal 0.85 0.60 0.06      # box 0 must end in this disc (x y r, m)
```

Same loop and outputs (under `out/tip_block_rrt/`). A sample is a tip position plus a position for
every box, and the distance adds the box displacements (`--box-weight`). It is the naive version:
box positions are sampled uniformly over the whole workspace, and orientation is ignored.

**Batched RRT**

```bash
python xnew_rrt/batch_tip_block_rrt.py --batch-size 32 --iters 20 --box-goal 0.85 0.60 0.06
```

Each iteration draws `--batch-size` samples, picks a node and a curvature for each, and simulates
all those edges in one batched rollout. `--iters` counts batched iterations. On the default
scene, 20 iterations at batch 32 built a 641-node tree in 59 s; a batched sim step at that size
runs about 7x more rollout-steps per second than batch 1.

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
