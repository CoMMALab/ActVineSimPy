'''
Videos for the paper's supplement, H.264 mp4 (ffmpeg; $FFMPEG or ffmpeg on the PATH).

  record    replay one planning solution on the GPU and keep every sim step (vine and blocks), as paper_figs:
            an RRT result.json split at its resets, or a block puzzle re-run with block_puzzle.py (growth then
            settling, per push) -> out/paper_videos/data/<job>.json
  planning  panels of recorded solutions in a grid, labelled by the problem's number only, played in sim time;
            a panel that finishes first holds its last frame
  val       the validation runs: the sim (contact_figure.py's dense caches; run it first) with the tracked vine
            and blocks drawn over it, at each tracked frame's video time. A panel of several clips (the three 5b
            pushes) plays them one after another.

Run from the repo root:
    xnew_rrt/py.sh xnew_rrt/paper_videos.py record envs/2a.json --result out/bench_overnight/runs/2a/s000/results/2a.result.json \
        --checkpoint checkpoints/spin8_manual.json
    xnew_rrt/py.sh xnew_rrt/paper_videos.py record envs/5b.json --puzzle --checkpoint checkpoints/spin8_manual.json
    xnew_rrt/py.sh xnew_rrt/paper_videos.py planning 2a:2a 2b:2b --out out/paper_videos/2
    xnew_rrt/py.sh xnew_rrt/paper_videos.py val contact --out out/paper_videos/val_contact
'''

import argparse, json, math, os, shlex, subprocess, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "out", "paper_videos", "data")
DT = 1 / 90                     # sim step [s] (dvsim/vine.py)
OTHER_BOX = "#d6a02a"           # blocks other than block 0 (as block_puzzle.draw_plan)
RESET_HOLD_S = 0.6              # an RRT reset: the last frame of the push held this long before the vine is back


# ------------------------------------------------------------------ record (GPU)

def record_rrt(args):
    import torch
    from scene import Sim
    import checkpoint, run_envs, paper_figs
    from reset_action_rrt import RESET
    from reset_rrt import reset_state
    e, scene, cfg, _ = run_envs.load_env(args.env)
    with open(args.result) as f:
        result = json.load(f)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.max_bodies = result.get("sim", {}).get("max_bodies", cfg.max_bodies)
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    sim = Sim(scene, cfg)
    goal_test, goal_distance = run_envs.goal_fns(sim, e)
    steps = result["planner"]["steps_per_edge"]
    for batch in (1, result["planner"]["batch_size"]):       # the planner's batched path if the single replay misses
        state = sim.initial_state()
        frames, hold = [paper_figs.snapshot(sim, state)], round(RESET_HOLD_S / DT)
        for k, (c, _) in enumerate(result["controls"]):
            if c == RESET:
                frames += [frames[-1]] * hold
                state = reset_state(sim, state)
                frames.append(paper_figs.snapshot(sim, state))
                continue
            if batch > 1:
                new, ok = sim.rollout_batch([state] * batch, [float(c)] * batch, steps, stop_fn=goal_test)[0]
            else:
                new, ok = sim.rollout(state, float(c), steps, stop_fn=goal_test if k == len(result["controls"]) - 1 else None)
            frames += [paper_figs.snapshot(sim, s) for s in new]
            state = new[-1] if new else state
            if not ok:
                break
        reached = bool(goal_test(state))
        print(f"{e['name']}: replay (batch {batch}) {'reaches' if reached else 'MISSES'} the goal "
              f"({goal_distance(state) * 1000:.0f} mm), {len(frames)} frames", flush=True)
        if reached:
            break
    return dict(env=e["name"], kind="rrt", source=args.result, checkpoint=args.checkpoint, reached=reached,
                radius_mm=cfg.vine_radius_m * 1000, frames=frames)


def record_plan(args):
    '''Re-run the planner (run_envs.run, rendering off) and keep its own states along the solution branch: for a
    run whose replay misses the goal (the planner's batches are not reproduced by a replay).'''
    import torch
    from scene import Sim
    import checkpoint, kino, run_envs, paper_figs, reset_action_rrt
    e, scene, cfg, planner = run_envs.load_env(args.env)
    checkpoint.apply(cfg, args.checkpoint)
    cfg.device, cfg.qp_backend = "cuda", "TORCH_IPM"
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    sim = Sim(scene, cfg)
    if args.result:                             # the benchmark run's planner settings
        with open(args.result) as f:
            planner.update(json.load(f)["planner"])
    caught = {}
    orig = reset_action_rrt.branch_controls
    def catch(best):
        caught["best"] = best
        return orig(best)
    reset_action_rrt.branch_controls = catch
    result = run_envs.run(e, sim, planner, None, args.seed, 0, render=False)
    states, _ = reset_action_rrt.branch_states(caught["best"])
    states = [s for s in states for _ in range(kino.EDGE_STRIDE)]     # the tree keeps every EDGE_STRIDE-th step
    print(f"{e['name']}: planner {'reaches' if result['reached'] else 'MISSES'} the goal "
          f"({result['goal_distance_mm']:.0f} mm), {len(states)} frames", flush=True)
    return dict(env=e["name"], kind="rrt", source="planner seed %d" % args.seed, checkpoint=args.checkpoint,
                reached=result["reached"], radius_mm=cfg.vine_radius_m * 1000,
                frames=[paper_figs.snapshot(sim, s) for s in states])


def record_puzzle(args):
    import block_puzzle, paper_figs
    captured = {}
    orig = block_puzzle.draw_pushes

    def capture(e, grid, sim, pushes, path, title):
        captured.update(sim=sim, pushes=pushes, title=title)
        return orig(e, grid, sim, pushes, path, title)

    block_puzzle.draw_pushes = capture
    if args.straight:
        import straight_puzzle  # noqa: F401  (straight pushes only, as paper_figs puzzle --straight)
    sys.argv = ["block_puzzle.py", args.env, "--checkpoint", args.checkpoint, "--no-render", "--seed", str(args.seed),
                "--max-attempts", "400", "--mem-fraction", str(args.mem_fraction),
                "--out", os.path.join(DATA, "_puzzle_run")] + shlex.split(args.extra)
    block_puzzle.main()
    sim, frames = captured["sim"], []
    for p in captured["pushes"]:
        frames += [paper_figs.snapshot(sim, s) for s in p["states"] + p["settle_states"]]
    with open(args.env) as f:
        name = json.load(f)["name"]
    print(captured["title"], f"{len(frames)} frames", flush=True)
    return dict(env=name, kind="puzzle", checkpoint=args.checkpoint, title=captured["title"],
                radius_mm=sim.cfg.vine_radius_m * 1000, frames=frames)


def record(args):
    d = record_puzzle(args) if args.puzzle else record_plan(args) if args.plan else record_rrt(args)
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, (args.job or d["env"]) + ".json")
    with open(path, "w") as f:
        json.dump(d, f)
    print("wrote", path)


# ------------------------------------------------------------------ drawing

class Encoder:
    '''Raw RGB frames piped to ffmpeg as H.264 (yuv420p, plays in browsers and slides).'''

    def __init__(self, path, fig, fps):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        self.size = (w - w % 2, h - h % 2)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.p = subprocess.Popen([os.environ.get("FFMPEG", "ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(fps),
                                   "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                                   "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.path, self.n = path, 0

    def add(self, fig, repeat=1):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:self.size[1], :self.size[0], :3]
        b = np.ascontiguousarray(img).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)
        self.n += repeat

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        print(f"wrote {self.path} ({self.n} frames)", flush=True)


def draw_static(ax, d):
    '''The env's board, goal and launch bar, as in paper_figs' overview.'''
    with open(os.path.join(REPO, "envs", d["env"] + ".json")) as f:
        env = json.load(f)
    if d["kind"] == "puzzle":
        import block_puzzle
        block_puzzle.draw_board(ax, env, block_puzzle.Grid(env))
        return
    import paper_figs, run_envs, viz
    from scene import Sim
    _, scene, cfg, _ = run_envs.load_env(os.path.join(REPO, "envs", d["env"] + ".json"))
    cfg.device, cfg.qp_backend = "cpu", "DIFFCP"
    import obstacle_style
    sim = Sim(scene, cfg)
    viz.draw_scene(ax, sim, walls=False)
    obstacle_style.draw_static(ax, sim, lw=1.6, borders=False)
    for t in list(ax.texts):
        t.remove()
    for l in list(ax.lines):
        if l.get_marker() == "^":
            l.remove()
    sx, sy, th = scene.start[0] * 1000, scene.start[1] * 1000, scene.start[2]
    u, w = (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))
    ax.add_patch(Polygon([(sx + a * paper_figs.LAUNCH_L * u[0] + b * paper_figs.LAUNCH_W / 2 * w[0],
                           sy + a * paper_figs.LAUNCH_L * u[1] + b * paper_figs.LAUNCH_W / 2 * w[1])
                          for a, b in ((-1, -1), (0, -1), (0, 1), (-1, 1))], closed=True, facecolor=viz.INK,
                         edgecolor="none", zorder=12))
    xmin, xmax, ymin, ymax = obstacle_style.board_bounds(json.load(open(os.path.join(REPO, "envs", d["env"] + ".json"))))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def draw_state(ax, snap, radius_mm, target=True):
    '''A recorded frame (paper_figs.snapshot): blocks (block 0 in its own color if it is the one to bring to the
    goal), then the vine. Returns the artists, to remove next frame.'''
    import viz
    arts = []
    for k, c in enumerate(snap["boxes"]):
        arts.append(ax.add_patch(Polygon(c[0], closed=True, facecolor=viz.BOX if k == 0 and target else OTHER_BOX,
                                         edgecolor=viz.SURFACE, lw=1.5, alpha=0.9, zorder=7)))
    v = snap["vine"]
    arts.append(ax.add_collection(PatchCollection([Circle(p, radius_mm) for p in v[1:]], facecolor=viz.VINE,
                                                  edgecolor="none", alpha=0.35, zorder=8)))
    arts += ax.plot([p[0] for p in v], [p[1] for p in v], "-", color=viz.VINE, lw=1.8, zorder=9)
    arts.append(ax.add_patch(Circle(v[-1], radius_mm, fill=False, edgecolor=viz.VINE, lw=1.8, zorder=9)))
    return arts


def planning(args):
    import viz
    jobs = [(j.split(":") + [None])[:3] for j in args.jobs]
    ds = []
    for job, _, speed in jobs:
        with open(os.path.join(DATA, job + ".json")) as f:
            ds.append(json.load(f))
        with open(os.path.join(REPO, "envs", ds[-1]["env"] + ".json")) as f:
            ds[-1]["target"] = json.load(f)["goal"]["type"] == "box" or ds[-1]["kind"] == "puzzle"
        ds[-1]["speed"] = float(speed) if speed else args.speed
    rows = args.rows
    cols = math.ceil(len(ds) / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(args.width, args.width), squeeze=False, facecolor="white", dpi=args.dpi)
    for ax, d, (_, label, _) in zip(axes.flat, ds, jobs):
        draw_static(ax, d)
        ax.set_title(label, fontsize=args.fs, pad=4)
        ax.text(0.98, 0.03, f"{d['speed']:g}×", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=args.fs * 0.8, color=viz.INK, zorder=20,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
    for ax in axes.flat[len(ds):]:
        ax.axis("off")
    # the figure's height from the panels' aspect (each row as tall as its tallest panel)
    asp = [[(a.get_ylim()[1] - a.get_ylim()[0]) / (a.get_xlim()[1] - a.get_xlim()[0]) for a in r] for r in axes]
    pw = args.width / cols
    fig.set_size_inches(args.width, sum(max(r) * pw for r in asp) + rows * args.fs / 72 * 2.2)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    # each panel at its own speed (sim seconds per video second); the video lasts as long as the slowest panel
    n_video = max(math.ceil((len(d["frames"]) - 1) * args.fps * DT / d["speed"]) + 1 for d in ds)
    enc = Encoder(args.out + ".mp4", fig, args.fps)
    arts = []
    for i in range(n_video):
        for a in arts:
            a.remove()
        arts = []
        for ax, d in zip(axes.flat, ds):
            t = min(len(d["frames"]) - 1, round(i * d["speed"] / (args.fps * DT)))
            arts += draw_state(ax, d["frames"][t], d["radius_mm"], d["target"])
        enc.add(fig)
    enc.add(fig, repeat=round(args.hold * args.fps))
    enc.close()
    fig.savefig(args.out + "_final.png", dpi=150)


# ------------------------------------------------------------------ validation

VAL = {          # video -> rows; a row is one or more clips (played one after another), drawn real | sim
    "contact": [[("vine_block_06", "contact_envfric_mu2", "contact_val.npz", "contact_envfric_mu2")],
                [("block_block_08", "contact_envfric_mu2", "contact_val.npz", "contact_envfric_mu2")]],
    "wall": [[("block_wall_07", "contact_envfric_mu2", "contact_val.npz", "contact_envfric_mu2")]],
    "multi": [[("long_02", "long_fit", "long.npz", "long_fit")],
              [(f"demo_5b_push{k}", "spin8_manual", "demo_5b.npz", "demo_5b_spin8_manual") for k in (1, 2, 3)]],
}
VAL_LABELS = {"contact": ["vine–block", "block–block"], "wall": ["block–wall"], "multi": ["long horizon", "puzzle 5b"]}
# last tracked frame shown: block_wall_07's last two frames are mis-tracked (the tip's centerline runs up the
# block's edge while the real vine goes on past it), so its clip ends at the last good one
LAST_FRAME = {"block_wall_07": 23}


def sim_trace(sc, cfg, device="cuda"):
    '''Every sim step of fit_sim.run_scenario on one scenario (as fit and scored): [(centerline [mm], box poses)],
    the first before any step. Recorded by wrapping its stepper; it stops where the scoring stops (the last
    tracked length reached).'''
    import fit_sim
    trace = []

    def snap(GS, x):
        st, nb = x["state"][0].cpu().numpy(), int(x["bodies"][0, 0])
        xyt = st[:3 * nb].reshape(nb, 3)
        hl = float(GS.params.half_len)
        tip = xyt[-1, :2] + hl * np.array([math.cos(xyt[-1, 2]), math.sin(xyt[-1, 2])])
        base = x["bases"][0, :2].cpu().numpy()
        trace.append((np.vstack([base[None], xyt[:, :2], tip[None]]), x["obj_pose"][0].cpu().numpy().copy()))

    class Recording(fit_sim.GraphStep):
        def __call__(self, **kw):
            if not trace:
                snap(self, kw)
            out = super().__call__(**kw)
            snap(self, dict(kw, state=out[0], bodies=out[2], obj_pose=out[3]))
            return out

    orig = fit_sim.GraphStep
    fit_sim.GraphStep = Recording
    try:
        fit_sim.run_scenario(sc, cfg, {k: [getattr(cfg, k)] for k in fit_sim.PER_ELEMENT}, device=device,
                             use_graph=device != "cpu")
    finally:
        fit_sim.GraphStep = orig
    return trace


def val_record(args):
    '''The sim trace of every validation clip -> out/paper_videos/data/val_<clip>_<checkpoint>.pkl (GPU).'''
    import pickle, torch
    import val_combo_fig
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    for spec in [c for set_ in VAL.values() for clips in set_ for c in clips]:
        d = val_combo_fig.load(*spec)
        import fit_sim
        cfg = fit_sim.cfg_of(checkpoint_of(spec[1]))
        trace = sim_trace(d["sc"], cfg)
        path = os.path.join(DATA, f"val_{spec[0]}_{spec[1]}.pkl")
        os.makedirs(DATA, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(trace, f)
        print(f"{spec[0]} ({spec[1]}): {len(trace)} sim steps ({len(trace) * DT:.1f} s sim), "
              f"tracked {d['times'][-1][1] - d['times'][0][1]:.1f} s -> {path}", flush=True)


def checkpoint_of(name):
    import checkpoint
    return checkpoint.load(os.path.join(REPO, "checkpoints", name + ".json"))


class RealVideo:
    """The trial's camera video, rectified to the board (contact_figure.draw_real), read forward frame by frame."""

    def __init__(self, d):
        import cv2
        self.d, self.cap, self.idx, self.img = d, cv2.VideoCapture(d["video"]), -1, None

    def at(self, frame):
        import cv2
        import contact_figure as cf
        if frame < self.idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.idx, self.img = frame - 1, None
        while self.idx < frame:
            ok = self.cap.grab()
            self.idx += 1
            if not ok:
                break
            if self.idx == frame:
                raw = self.cap.retrieve()[1]
                self.img = cv2.cvtColor(self.d["cal"].rectify(raw, cf.RES_MM), cv2.COLOR_BGR2RGB)
        return self.img


def draw_sim_side(ax, c, poly, boxes, win, final):
    """The sim state; with final, the last tracked vine and blocks drawn over it (dashed)."""
    import contact_figure as cf
    sc, r = c["d"]["sc"], c["d"]["radius_mm"]
    ax.cla()
    ax.set_facecolor("white")
    cf.draw_walls(ax, sc, c["d"]["name"])
    for pose, o in zip(boxes, sc["objects"]):
        ax.add_patch(Polygon(cf.box_corners(pose, o["size_mm"]), closed=True, facecolor=cf.BOX, alpha=0.85,
                             edgecolor="#9c3d14", lw=0.6, zorder=3))
    ax.add_patch(Polygon(cf.tube(poly, r), closed=True, facecolor=cf.VINE, alpha=0.45, edgecolor=cf.VINE, lw=0.6, zorder=4))
    ax.add_patch(Circle(poly[-1], r, facecolor=cf.VINE, alpha=0.45, edgecolor="none", zorder=4))
    ax.plot(*poly.T, color=cf.VINE, lw=0.8, zorder=5)
    if final:
        k = c["last"]
        ax.plot(*np.asarray(sc["frames"][k]).T, color=cf.REF, lw=1.2, ls=(0, (2.5, 1.5)), zorder=6)
        for pose, o in zip(sc["boxes"][k], sc["objects"]):
            ax.add_patch(Polygon(cf.box_corners(pose, o["size_mm"]), closed=True, fill=False, edgecolor=cf.REF,
                                 lw=1.2, ls=(0, (2.5, 1.5)), zorder=6))
    x, y, th = sc["start"]
    u, w = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    ax.add_patch(Polygon([np.array([x, y]) + a * cf.LAUNCH_L * u + b * cf.LAUNCH_W / 2 * w
                          for a, b in ((-1, -1), (0, -1), (0, 1), (-1, 1))], closed=True, facecolor=cf.REF,
                         edgecolor="none", zorder=7))
    cf.style(ax, win)


def val(args):
    """Each row: the real run (its camera video, rectified) beside the sim (every step, sim_trace), each in its own
    time and stretched to the clip's video length, (tracked duration) / speed, so they start and end together.
    The clip's last tracked vine and blocks are drawn over the sim throughout (dashed)."""
    import pickle
    import contact_figure as cf
    import val_combo_fig
    from fit_to_master import arc_length
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    rows = []
    for clips in VAL[args.set]:
        cs, off = [], 0.0
        for spec in clips:
            d = val_combo_fig.load(*spec)
            with open(os.path.join(DATA, f"val_{spec[0]}_{spec[1]}.pkl"), "rb") as f:
                trace = pickle.load(f)
            last = LAST_FRAME.get(spec[0], len(d["times"]) - 1)
            # the sim up to the last shown frame's length, or until it stops growing (stalled short of it)
            Ls = np.array([arc_length(p)[-1] for p, _ in trace])
            L = min(arc_length(d["sc"]["frames"][last])[-1], Ls.max() - 1.0)
            n = int(np.argmax(Ls >= L)) + 1
            trace = trace[:n]
            (f0, t0), (_, t1) = d["times"][0], d["times"][last]
            tr = d["tracks"]
            c = dict(d=d, trace=trace, last=last, t0=off, D=(t1 - t0) / args.speed, T0=t0, T1=t1,
                     video=RealVideo(d), t_s=np.asarray(tr["t_s"], float), frames=np.asarray(tr["frame"], float))
            cs.append(c)
            off += c["D"] + args.gap
            print(f"{d['name']}: frames 0-{last} of {len(d['times'])}, {n} sim steps, {t1 - t0:.1f} s real -> "
                  f"{c['D']:.1f} s of video ({n / c['D']:.0f} sim states per second)", flush=True)
        pts = []
        for c in cs:
            sc = c["d"]["sc"]
            pts += [np.array(sc["start"][:2])[None]] + list(sc["frames"][:c["last"] + 1]) + [p for p, _ in c["trace"]]
            for B in list(sc["boxes"][:c["last"] + 1]) + [b for _, b in c["trace"]]:
                pts += [cf.box_corners(p, o["size_mm"]) for p, o in zip(B, sc["objects"])]
        P = np.vstack(pts)
        (x0, y0), (x1, y1) = P.min(0) - 50, P.max(0) + 50
        rows.append(dict(clips=cs, T=off - args.gap, win=[x0, x1, y0, y1], asp=(y1 - y0) / (x1 - x0)))

    head, sub, gap, side, mid = args.fs / 72 * 1.9, args.fs / 72 * 1.4, 0.15, 0.1, 0.08
    pw = (args.width - 2 * side - mid) / 2
    foot = args.fs / 72 * 2.4
    H = sum(head + sub + r["asp"] * pw for r in rows) + gap * (len(rows) - 1) + foot
    fig = plt.figure(figsize=(args.width, H), facecolor="white", dpi=args.dpi)
    y = H
    for n, r in enumerate(rows):
        h = r["asp"] * pw
        y -= head
        fig.text(0.5, (y + head * 0.45) / H, VAL_LABELS[args.set][n], ha="center", va="center", fontsize=args.fs)
        y -= sub
        r["real"] = fig.add_axes([side / args.width, (y - h) / H, pw / args.width, h / H])
        r["sim"] = fig.add_axes([(side + pw + mid) / args.width, (y - h) / H, pw / args.width, h / H])
        for x, lab in ((side + pw / 2, "real"), (side + pw + mid + pw / 2, "simulation")):
            fig.text(x / args.width, (y + sub * 0.4) / H, lab, ha="center", va="center", fontsize=args.fs - 3,
                     color="#444444")
        y -= h + gap
    fig.legend(handles=[Patch(facecolor=cf.VINE, alpha=0.45, edgecolor=cf.VINE, label="sim vine"),
                        Patch(facecolor=cf.BOX, alpha=0.85, edgecolor="#9c3d14", label="sim block"),
                        Line2D([], [], color=cf.REF, lw=1.2, ls=(0, (2.5, 1.5)), label="final tracked position"),
                        Patch(facecolor=cf.WALL_FACE, edgecolor="#52514e", hatch="///", label="wall")],
               loc="lower center", ncol=4, fontsize=args.fs - 3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    T = max(r["T"] for r in rows)
    enc = Encoder(args.out + ".mp4", fig, args.fps)
    for tau in list(np.arange(0, T, 1 / args.fps)) + [T]:        # the last frame exactly at the end: every clip done
        for n, r in enumerate(rows):
            c = max((c for c in r["clips"] if c["t0"] <= tau), key=lambda c: c["t0"])
            phi = min(1.0, (tau - c["t0"]) / c["D"])
            t = c["T0"] + phi * (c["T1"] - c["T0"])
            img = c["video"].at(int(round(np.interp(t, c["t_s"], c["frames"]))))
            ax = r["real"]
            ax.cla()
            if img is not None:
                W, Hb = c["d"]["cal"].board_mm
                ax.imshow(img, extent=[0, W, 0, Hb], origin="upper", interpolation="bilinear")
            cf.style(ax, r["win"])
            if len(r["clips"]) > 1:
                ax.text(0.03, 0.97, f"push {r['clips'].index(c) + 1}", transform=ax.transAxes, fontsize=args.fs - 3,
                        ha="left", va="top", color="white", zorder=20,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.55, edgecolor="none"))
            poly, boxes = c["trace"][round(phi * (len(c["trace"]) - 1))]
            draw_sim_side(r["sim"], c, poly, boxes, r["win"], final=True)
            val_combo_fig.bar(r["sim"], r["win"], args.fs - 4)
        enc.add(fig)
    enc.add(fig, repeat=round(args.hold * args.fps))
    enc.close()
    fig.savefig(args.out + "_final.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("env")
    r.add_argument("--result", help="RRT result.json")
    r.add_argument("--puzzle", action="store_true", help="re-run block_puzzle.py instead")
    r.add_argument("--plan", action="store_true", help="re-run the RRT planner (settings from --result) and keep its states")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--extra", default="", help="more block_puzzle.py arguments")
    r.add_argument("--straight", action="store_true", help="puzzle: straight pushes only (straight_puzzle.py)")
    r.add_argument("--job", default=None, help="data file name (default: the env's name)")
    r.add_argument("--mem-fraction", type=float, default=0.45)
    p = sub.add_parser("planning")
    p.add_argument("jobs", nargs="+", help="<job>:<label>[:<speed>]")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--fs", type=float, default=16)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=0.5, help="sim seconds per video second (shown in each panel's corner)")
    p.add_argument("--hold", type=float, default=2.0, help="last frame held [s]")
    p.add_argument("--dpi", type=float, default=160)
    p.add_argument("--out", required=True)
    vr = sub.add_parser("val-record", help="record every sim step of the validation clips (GPU)")
    vr.add_argument("--mem-fraction", type=float, default=0.45)
    v = sub.add_parser("val")
    v.add_argument("set", choices=list(VAL))
    v.add_argument("--width", type=float, default=10.0)
    v.add_argument("--fs", type=float, default=16)
    v.add_argument("--fps", type=int, default=30)
    v.add_argument("--speed", type=float, default=0.5, help="seconds of the real run per video second")
    v.add_argument("--dpi", type=float, default=160)
    v.add_argument("--gap", type=float, default=0.5, help="pause between clips of one panel [s]")
    v.add_argument("--hold", type=float, default=2.0)
    v.add_argument("--out", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    sys.path[:0] = [HERE, REPO]
    {"record": record, "planning": planning, "val": val, "val-record": val_record}[args.cmd](args)


if __name__ == "__main__":
    main()
