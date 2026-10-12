'''
Simulate ONE rollout of the vine (+ movable boxes) and write it out as an animation.

Headless: nothing needs a display. Open the outputs from the VS Code explorer --
  <out>/rollout.gif         the animation (VS Code previews gifs)
  <out>/rollout_final.png   last frame
  <out>/rollout.html        (--html) player with a frame scrubber; right click -> "Open with Live
                            Preview", or `python -m http.server` in <out> and use the forwarded port

Run from the repo root:
    python xnew_rrt/rollout.py                                  # default scene, default controls
    python xnew_rrt/rollout.py --controls "0:60,3:80,-3:120"    # curvature[1/m]:steps, ...
    python xnew_rrt/rollout.py --env envs/env_dynamics.txt --controls "0:150"
    python xnew_rrt/rollout.py --plan out/rrt/plan.json         # replay what the RRT found
'''

import argparse, json, os, shutil, time
from dataclasses import asdict

from scene import STIFFNESS_MODES, Sim, SimConfig, default_scene, load_scene, scene_from_dict  # (puts the repo root on sys.path)
import viz


def parse_controls(text):
    '''"0:60,3:80" -> [(0.0, 60), (3.0, 80)]  (curvature in 1/m, number of sim steps)'''
    controls = []
    for part in text.split(","):
        curvature, steps = part.split(":")
        controls.append((float(curvature), int(steps)))
    return controls


def simulate(sim: Sim, controls):
    '''
    Run the controls one after another from the initial state.
    Returns (states, labels): one entry per sim step (initial state included).
    '''
    cur = sim.initial_state()
    states, labels = [cur], ["initial state"]

    for k, (curvature, num_steps) in enumerate(controls):
        new_states, ok = sim.rollout(cur, curvature, num_steps)
        states += new_states
        labels += [f"control {k + 1}/{len(controls)}: curvature {curvature:+.2f} 1/m"] * len(new_states)
        if new_states:
            cur = new_states[-1]

        if not ok:
            print(f"  sim failed during control {k + 1} (QP infeasible or non-finite state); stopping there")
            break
        if len(new_states) < num_steps:
            print(f"  vine is fully grown ({cur.num_bodies} bodies) during control {k + 1}; stopping there")
            break

    return states, labels


def render_states(sim: Sim, states, labels, out_dir, name="rollout", every=2, html=False, frame_ms=None):
    '''Draw every `every`-th state and assemble <name>.gif / <name>_final.png (/ <name>.html).'''
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"_{name}_frames")
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir)

    dt = float(sim.params.dt)
    if frame_ms is None:
        frame_ms = max(20, int(round(dt * every * 1000)))  # real time

    picked = list(range(0, len(states), every))
    if picked[-1] != len(states) - 1:
        picked.append(len(states) - 1)

    tip_trace, frame_paths = [], []
    next_traced = 0
    for i in picked:
        while next_traced <= i:
            tip_trace.append(viz._tip_mm(states[next_traced]))
            next_traced += 1

        state = states[i]
        fig, ax = viz.new_figure(sim.scene)
        title = (f"{sim.scene.name}  |  step {i}/{len(states) - 1}  t = {i * dt:.2f} s  |  "
                 f"{state.num_bodies} bodies  |  {labels[i]}")
        viz.draw_rollout_frame(ax, sim, state, tip_trace, title)
        path = os.path.join(frame_dir, f"f{i:05d}.png")
        viz.save_figure(fig, path, dpi=70)
        frame_paths.append(path)

    outputs = {"gif": os.path.join(out_dir, f"{name}.gif"),
               "final": os.path.join(out_dir, f"{name}_final.png")}
    viz.write_gif(frame_paths, outputs["gif"], frame_ms=frame_ms)
    shutil.copyfile(frame_paths[-1], outputs["final"])
    if html:
        outputs["html"] = os.path.join(out_dir, f"{name}.html")
        viz.write_html_player(frame_paths, outputs["html"], title=f"{sim.scene.name} {name}", frame_ms=frame_ms)

    shutil.rmtree(frame_dir)
    return outputs


def add_sim_args(parser):
    parser.add_argument("--env", default=None, help="envs/*.txt file (default: built-in table-top scene)")
    parser.add_argument("--max-bodies", type=int, default=SimConfig.max_bodies,
                        help="vine length budget; each body is one 18 mm segment")
    parser.add_argument("--grow-rate", type=float, default=SimConfig.grow_rate_mps, help="growth speed [m/s]")
    parser.add_argument("--vine-radius", type=float, default=SimConfig.vine_radius_m, help="vine radius [m]")
    parser.add_argument("--grow-force", type=float, default=SimConfig.grow_force_n,
                        help="stall force of growth [N]: how hard a blocked tip pushes. 0 = the tip must "
                             "always advance at the growth speed (hard constraint)")
    parser.add_argument("--stiffness-mode", choices=STIFFNESS_MODES, default=SimConfig.stiffness_mode,
                        help="curvature: spring toward the commanded curvature. spam: sPAM actuators designed "
                             "for it (the ActVine model). linear: passive vine, controls are ignored")


def sim_from_args(args):
    scene = load_scene(args.env) if args.env else default_scene()
    cfg = SimConfig(max_bodies=args.max_bodies, grow_rate_mps=args.grow_rate, vine_radius_m=args.vine_radius,
                    stiffness_mode=args.stiffness_mode, grow_force_n=args.grow_force or None)
    return Sim(scene, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sim_args(parser)
    parser.add_argument("--controls", default="0:60,3:80,-3:120",
                        help='"curvature:steps,..." curvature [1/m] of the part grown during those steps')
    parser.add_argument("--plan", default=None, help="plan.json written by sim_rrt.py (overrides --env/--controls)")
    parser.add_argument("--out", default="out/rollout")
    parser.add_argument("--every", type=int, default=2, help="draw every N-th sim step")
    parser.add_argument("--html", action="store_true", help="also write the html player with a scrubber")
    args = parser.parse_args()

    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        sim = Sim(scene_from_dict(plan["scene"]), SimConfig(**plan["sim"]))
        controls = [(c, n) for c, n in plan["controls"]]
    else:
        sim = sim_from_args(args)
        controls = parse_controls(args.controls)

    print(f"Scene '{sim.scene.name}': {len(sim.scene.walls_m)} static boxes, {sim.num_objs} movable; "
          f"sim config {asdict(sim.cfg)}")

    t0 = time.time()
    states, labels = simulate(sim, controls)
    final = states[-1]
    print(f"Simulated {len(states) - 1} steps in {time.time() - t0:.1f} s: {final.num_bodies} bodies, "
          f"tip {tuple(round(v, 3) for v in sim.tip_m(final))} m, "
          f"{sim.goal_distance_m(final):.3f} m from goal center, reached goal: {sim.reached_goal(final)}")
    for k, ((pose, _, _), (pose0, _, _)) in enumerate(zip(sim.object_boxes(final), sim.object_boxes(states[0]))):
        print(f"  box {k}: moved ({pose[0] - pose0[0]:+.1f}, {pose[1] - pose0[1]:+.1f}) mm, "
              f"rotated {pose[2] - pose0[2]:+.3f} rad")

    outputs = render_states(sim, states, labels, args.out, every=args.every, html=args.html)
    for path in outputs.values():
        print("wrote", path)
