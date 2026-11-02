'''
Fit the simulator to tracked trials on the GPU, keep the result as a checkpoint, and compare
checkpoints on held-out trials.

The loss is fit_to_master.py's, on the same reference files (video/to_reference.py): every
scenario is grown again with the trial's base, walls, boxes, growth speed and control schedule,
and scored by the mean centerline distance at equal vine length, the mean box-center error
(BOX_WEIGHT) and a joint-jitter penalty. What is new here:
  - the whole population of one generation runs as one batch on the GPU, each step a replayed
    CUDA graph (batched_rollout.GraphStep, per-element stiffness, damping and box friction);
  - it starts from a checkpoint (checkpoint.py) and writes one, with its training data recorded;
  - `eval` scores any checkpoints on any reference files, per scenario, for comparing models
    trained on different data.
Fitted (--fit, cross-entropy search in log space): any of PER_ELEMENT, by default joint_stiffness,
ang_damp, obj_friction; for the 'wrinkle' bending mode pressure_pa and wrinkle_scale in place of
joint_stiffness. Fixed from the starting checkpoint, or --set: everything else (segment mass and
inertia, box mass and the stall force enter the QP's matrices, shared by the batch).

    python xnew_rrt/fit_sim.py fit --ref xnew_rrt/data/contact_train.npz --init checkpoints/base_pbd_brick.json \
        --name contact
    python xnew_rrt/fit_sim.py fit --ref xnew_rrt/data/contact_train.npz --init checkpoints/contact_from_base.json \
        --name wrinkle --set stiffness_mode=wrinkle obj_friction=0.3 vine_friction=0.3 \
        --fit pressure_pa wrinkle_scale ang_damp vine_friction
    python xnew_rrt/fit_sim.py eval checkpoints/base_pbd_brick.json checkpoints/contact.json \
        --ref xnew_rrt/data/contact_val.npz xnew_rrt/data/contact_train.npz
'''

import argparse, json, math, os, time, warnings

import numpy as np
import torch

from scene import SimConfig, params_to  # (puts the repo root on sys.path)
from dvsim import si
import dvsim.solver as solver
from dvsim.vine import create_state_batched, init_state_batched
from batched_rollout import GraphStep, bucket
from fit_to_master import (load_reference, arc_length, shape_error, MIN_COMPARE_MM, FAIL_LOSS,
                           JITTER_MM_PER_RADPS, TIP_JOINTS, BOX_WEIGHT)
import checkpoint
import launch_fix

FIT = ("joint_stiffness", "ang_damp", "obj_friction")
DAMP_SI = 1.0e-4            # si ang_damp = VineParams.damping * DAMP_SI
# SimConfig field -> (VineParams attribute it sets per batch element, its value from the field's)
PER_ELEMENT = {"joint_stiffness": ("stiffness_val", lambda v: v), "ang_damp": ("damping", lambda v: v / DAMP_SI),
               "obj_friction": ("obj_friction", lambda v: v), "vine_friction": ("vine_friction", lambda v: v),
               "pressure_pa": ("pressure", lambda v: v), "wrinkle_scale": ("wrinkle_scale", lambda v: v),
               "obj_spin": ("obj_spin", lambda v: v), "wall_friction": ("wall_friction", lambda v: v),
               "box_friction": ("box_friction", lambda v: v)}
STD = {"joint_stiffness": 1.0, "ang_damp": 1.0, "obj_friction": 0.7, "vine_friction": 0.7, "pressure_pa": 0.5,
       "wrinkle_scale": 0.7, "obj_spin": 0.7, "wall_friction": 0.7, "box_friction": 0.7}       # initial spread of each log parameter
START = {"vine_friction": 0.3, "wall_friction": 0.3, "box_friction": 0.3}     # where a fit starts a parameter that the checkpoint has at 0
# weight of the boxes' heading error in the box error, as a share of the mean displacement of a box's
# points that the turn causes (0: centers only, as before; --box-angle). Its error in deg is reported anyway
BOX_ANGLE = 0.0


def box_heading_error(sim_pose, ref_pose, sizes):
    '''Per box: |heading difference| [rad], folded by the box's symmetry (90 deg for a square), and the
    mean distance of its points from its center [mm] (0.383 x side for a square).'''
    out = []
    for (w, h), a, b in zip(sizes, sim_pose[:, 2], ref_pose[:, 2]):
        sym = math.pi / 2 if abs(w - h) < 1e-6 else math.pi
        d = (float(a - b) + sym / 2) % sym - sym / 2
        out.append((abs(d), 0.383 * math.sqrt(w * h)))
    return out


def run_scenario(sc, cfg, values, device="cuda", use_graph=True):
    '''
    Grow one scenario for every candidate in one batch. values: {SimConfig field in PER_ELEMENT:
    (n,) values}, one per candidate; the other PER_ELEMENT fields come from cfg.
    Returns per candidate a dict: loss, shape_mm, box_mm, jitter, and the last compared centerline
    and box poses.
    '''
    n = len(next(iter(values.values())))
    B = bucket(n)
    pad = lambda v: np.concatenate([v, np.repeat(v[:1], B - n)])
    values = {k: pad(np.asarray(values[k], float) if k in values else np.full(n, float(getattr(cfg, k))))
              for k in PER_ELEMENT}
    dev = torch.device(device)
    walls_m = [[v * 1e-3 for v in w] for w in sc["walls_mm"]]
    # oriented walls (the slanted panels): [cx, cy, theta_rad, hw, hh], theta not scaled
    walls_m += [[w[0] * 1e-3, w[1] * 1e-3, w[2], w[3] * 1e-3, w[4] * 1e-3] for w in sc.get("walls_obb_mm", [])]
    x0, y0, h0 = sc["start"]
    grow_rate_mps = sc.get("grow_rate_mps") or cfg.grow_rate_mps

    solver.BACKEND = "TORCH_IPM"
    params = si.vine_params_si(max_bodies=cfg.max_bodies, obstacles_m=walls_m or None, grow_rate_mps=grow_rate_mps,
                               radius_m=cfg.vine_radius_m, seg_len_m=cfg.seg_len_m, seg_mass_kg=cfg.seg_mass_kg,
                               seg_inertia_kgm2=cfg.seg_inertia_kgm2, grow_force_n=cfg.grow_force_n)
    params.stiffness_mode = cfg.stiffness_mode
    params.vine_friction = float(values["vine_friction"].max())   # > 0: build the friction rows
    params.wall_friction = float(values["wall_friction"].max())
    params.box_friction = float(values["box_friction"].max())
    objects = sc.get("objects") or []
    if objects:
        aabbs = [[(o["pose"][0] - o["size_mm"][0] / 2) * 1e-3, (o["pose"][1] - o["size_mm"][1] / 2) * 1e-3,
                  (o["pose"][0] + o["size_mm"][0] / 2) * 1e-3, (o["pose"][1] + o["size_mm"][1] / 2) * 1e-3]
                 for o in objects]
        pose0 = si.set_objects_si(params, aabbs, [o["mass_kg"] for o in objects])
        pose0[0, :, 2] = torch.tensor([float(o["pose"][2]) for o in objects])
    else:
        pose0 = torch.zeros(1, 0, 3)
    dparams = params_to(params, dev)

    base = np.array([si.len_to_nd(x0 * 1e-3), si.len_to_nd(y0 * 1e-3), h0])
    state, dstate = create_state_batched(B, cfg.max_bodies)
    bodies = torch.full((B, 1), 2)
    init_state_batched(params, state, bodies, *(torch.full((B, 1), float(v)) for v in (base[2], base[0], base[1])))
    with torch.device(dev):
        x = dict(state=state.to(dev), dstate=dstate.to(dev), bodies=bodies.to(dev).long(),
                 obj_pose=pose0.repeat(B, 1, 1).to(dev), obj_dstate=torch.zeros(B, pose0.shape[1], 3, device=dev))
        bases = torch.tensor(base, dtype=torch.float32, device=dev).repeat(B, 1)
        bend = torch.zeros(B, cfg.max_bodies, device=dev)
        spam = torch.zeros(B, cfg.max_bodies, device=dev)
        per = {attr: torch.tensor(to(values[k]), dtype=torch.float32, device=dev)
               for k, (attr, to) in PER_ELEMENT.items()}
    stepper = GraphStep(dparams, B, dev, use_graph, extra=tuple(per))

    # control schedule in total vine length
    bounds, length = [], sc["initial_body_mm"]
    for curvature, seg_mm in sc["schedule"]:
        length += seg_mm
        bounds.append((length, curvature))
    bend_cpu = torch.zeros(B, cfg.max_bodies)
    phase = np.full(B, -1)

    compared = [i for i, f in enumerate(sc["frames"]) if arc_length(f)[-1] >= MIN_COMPARE_MM]
    ref_frames = [sc["frames"][i] for i in compared]
    ref_boxes = [sc["boxes"][i] for i in compared] if objects else None
    ref_len = torch.tensor([arc_length(f)[-1] for f in ref_frames] + [math.inf], device=dev)
    errors, box_err, box_deg = [[] for _ in range(B)], [[] for _ in range(B)], [[] for _ in range(B)]
    sizes = [o["size_mm"] for o in objects]
    last_poly, last_boxes = [None] * B, [None] * B
    next_frame = torch.zeros(B, dtype=torch.long, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    jit_sum = torch.zeros(B, device=dev)
    jit_n = torch.zeros(B, device=dev)
    hl = float(params.half_len)
    mb = cfg.max_bodies
    idx = torch.arange(mb, device=dev)
    base_xy = torch.tensor(base[:2], dtype=torch.float32, device=dev)

    max_steps = int(1.6 * (bounds[-1][0] - 2 * cfg.seg_len_m * 1e3) / (grow_rate_mps * 1e3 * float(params.dt)))
    for it in range(max_steps):
        st, nb = x["state"], x["bodies"][:, 0]
        xy = torch.stack([st[:, 0::3], st[:, 1::3]], -1)                                  # (B, mb, 2)
        th = st[:, 2::3]
        active = idx[None] < nb[:, None]
        seg = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) * (active[:, 1:])
        L = (xy[:, 0] - base_xy).norm(dim=-1) + seg.sum(1) + hl
        # control phase (schedule boundaries), on the CPU: rare and cheap
        if len(bounds) > 1 or it == 0:
            Lc, nbc = L.cpu().numpy(), nb.cpu().numpy()
            for k in range(B):
                while phase[k] + 1 < len(bounds) and (phase[k] < 0 or Lc[k] >= bounds[phase[k]][0]):
                    phase[k] += 1
                    bend_cpu[k, int(nbc[k]):] = bounds[phase[k]][1] * cfg.seg_len_m
            bend = bend_cpu.to(dev)
        # reference frames reached this step
        due = alive & (L >= ref_len[next_frame])
        if bool(due.any()):
            Lc = L.cpu().numpy()
            stc, nbc, opc = st.cpu().numpy(), nb.cpu().numpy(), x["obj_pose"].cpu().numpy()
            nfc = next_frame.cpu().numpy()
            for k in np.nonzero(due.cpu().numpy())[0]:
                m = int(nbc[k])
                xyt = stc[k, :3 * m].reshape(m, 3)
                tip = xyt[-1, :2] + hl * np.array([math.cos(xyt[-1, 2]), math.sin(xyt[-1, 2])])
                poly = np.vstack([base[None, :2], xyt[:, :2], tip[None]])
                f = int(nfc[k])
                while f < len(ref_frames) and Lc[k] >= arc_length(ref_frames[f])[-1]:
                    errors[k].append(shape_error(ref_frames[f], poly))
                    if objects:
                        turn = box_heading_error(opc[k], ref_boxes[f], sizes)
                        box_err[k].append(float(np.mean(np.linalg.norm(opc[k, :, :2] - ref_boxes[f][:, :2], axis=1)
                                                        + BOX_ANGLE * np.array([d * r for d, r in turn]))))
                        box_deg[k].append(float(np.degrees(np.mean([d for d, _ in turn]))))
                    f += 1
                if f == len(ref_frames):
                    last_poly[k] = poly
                    last_boxes[k] = opc[k].copy() if objects else None
                next_frame[k] = f
        # a vine at the body cap stops growing (its missing frames count as failures); the rest go on
        running = alive & (next_frame < len(ref_frames)) & (nb < mb - 1)
        if not bool(running.any()):
            break
        out = stepper(bases=bases, state=x["state"], dstate=x["dstate"], bodies=x["bodies"], obj_pose=x["obj_pose"],
                      obj_dstate=x["obj_dstate"], bend_target=bend, spam_moment=spam, **per)
        stn, dsn, bon, opn, odn, ok, _ = out
        good = ok & torch.isfinite(stn).all(1) & (stn.abs().amax(1) < 1e5) & torch.isfinite(opn).flatten(1).all(1)
        alive &= good | ~running                                    # (only a stepped element can fail)
        moved = running & good
        for key, v in zip(("state", "dstate", "bodies", "obj_pose", "obj_dstate"), (stn, dsn, bon, opn, odn)):
            m = moved.view(-1, *[1] * (v.dim() - 1))
            x[key] = torch.where(m, v, x[key])
        omega = x["dstate"][:, 2::3]
        n_j = x["bodies"][:, 0] - TIP_JOINTS
        dj = (omega[:, 1:] - omega[:, :-1]).abs()
        mask = (idx[None, :-1] < (n_j - 1)[:, None]).float()
        ok_j = moved & (n_j > 2)
        jit_sum += torch.where(ok_j, (dj * mask).sum(1) / mask.sum(1).clamp_min(1), 0.0)
        jit_n += ok_j.float()

    alive_c = alive.cpu().numpy()
    jit = (jit_sum / jit_n.clamp_min(1)).cpu().numpy()
    out = []
    for k in range(n):
        if not alive_c[k] or not errors[k]:
            out.append(dict(loss=FAIL_LOSS, shape_mm=FAIL_LOSS, box_mm=FAIL_LOSS if objects else 0.0, jitter=0.0,
                            box_deg=180.0 if objects else 0.0, poly=last_poly[k], boxes=last_boxes[k]))
            continue
        missing = len(ref_frames) - len(errors[k])
        shape = (sum(errors[k]) + missing * FAIL_LOSS) / len(ref_frames)
        box = float(np.mean(box_err[k])) if box_err[k] else 0.0
        out.append(dict(loss=float(shape + JITTER_MM_PER_RADPS * jit[k] + BOX_WEIGHT * box), shape_mm=float(shape), box_mm=box,
                        box_deg=float(np.mean(box_deg[k])) if box_deg[k] else 0.0,
                        jitter=float(jit[k]), poly=last_poly[k], boxes=last_boxes[k],
                        reached=len(errors[k]), frames=len(ref_frames), shape_reached_mm=float(np.mean(errors[k]))))
    return out


def evaluate(scenarios, cfg, values, **kw):
    '''{scenario: [per-candidate dict]}'''
    return {name: run_scenario(sc, cfg, values, **kw) for name, sc in scenarios.items()}


def summary(per, i=0):
    rows = {name: v[i] for name, v in per.items()}
    boxed = [r for r in rows.values() if r["box_mm"]]
    return dict(loss=float(np.mean([r["loss"] for r in rows.values()])),
                shape_mm=float(np.mean([r["shape_mm"] for r in rows.values()])),
                box_mm=float(np.mean([r["box_mm"] for r in boxed])) if boxed else 0.0,
                box_deg=float(np.mean([r["box_deg"] for r in boxed])) if boxed else 0.0,
                per_scenario={n: {"loss": float(r["loss"]), "shape_mm": float(r["shape_mm"]), "box_mm": float(r["box_mm"]),
                                  "box_deg": float(r["box_deg"])} for n, r in rows.items()})


def parse_set(pairs):
    out = {}
    for pair in pairs or ():
        k, v = pair.split("=", 1)
        try:
            v = float(v)
        except ValueError:
            pass
        out[k] = v
    return out


def cfg_of(ckpt):
    cfg = SimConfig()
    checkpoint.apply(cfg, ckpt)
    cfg.max_bodies = ckpt["sim"].get("max_bodies", cfg.max_bodies)
    return cfg


def dataset_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def fit(args):
    init = checkpoint.load(args.init)
    init["sim"].update(parse_set(args.set))
    cfg = cfg_of(init)
    fit_names = list(args.fit)
    if args.max_bodies:
        cfg.max_bodies = args.max_bodies
    scenarios = {}
    for ref in args.ref:
        for name, sc in load_reference(ref).items():
            if name.startswith(tuple(args.only)):
                scenarios[name] = sc
    if args.launch_fix:
        launch_fix.apply(scenarios)
    rng = np.random.default_rng(args.seed)
    start = np.log([getattr(cfg, k) or START[k] for k in fit_names])
    mean, std = start.copy(), np.array([STD[k] * args.std for k in fit_names])
    best = (math.inf, None, None)
    history = []
    t0 = time.time()
    for gen in range(args.generations):
        samples = rng.normal(mean, std, size=(args.population, len(fit_names)))
        if gen == 0:
            samples[0] = start
        vals = dict(zip(fit_names, np.exp(samples).T))
        per = evaluate(scenarios, cfg, vals)
        loss = np.mean([[r["loss"] for r in v] for v in per.values()], axis=0)
        order = np.argsort(loss)
        elite = samples[order[:max(2, args.population // 4)]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), 0.05)
        i = order[0]
        if loss[i] < best[0]:
            best = (float(loss[i]), {k: float(v[i]) for k, v in vals.items()}, summary(per, i))
        history.append(dict(generation=gen, loss=loss.tolist(), **{k: v.tolist() for k, v in vals.items()}))
        if gen == 0:
            print(f"start ({args.init}): {loss[0]:.1f}", flush=True)
        print(f"gen {gen}: best {loss[i]:.1f} at " + ", ".join(f"{k} {v[i]:.3g}" for k, v in vals.items())
              + f"  (all-time {best[0]:.1f}, {time.time() - t0:.0f} s)", flush=True)
    loss, fitted, train = best
    sim = dict(init["sim"], **fitted, max_bodies=cfg.max_bodies)
    ckpt = checkpoint.make(args.name, sim, description=args.description or f"fitted from {init['name']} on "
                           + ", ".join(dataset_name(r) for r in args.ref), parent=init["name"],
                           trained_on=[{"ref": r, "scenarios": [n for n in load_reference(r) if n in scenarios]}
                                       for r in args.ref],
                           fit=dict(search="cross-entropy (fit_sim.py)", fitted=fit_names, set=parse_set(args.set), box_angle=BOX_ANGLE,
                                    launch_fix=args.launch_fix,
                                    population=args.population, generations=args.generations, seed=args.seed,
                                    history=history),
                           metrics={"+".join(dataset_name(r) for r in args.ref): train})
    print("wrote", checkpoint.save(ckpt, args.out))


def evaluate_checkpoints(args):
    rows = []
    for path in args.checkpoints:
        ck = checkpoint.load(path)
        cfg = cfg_of(ck)
        if args.max_bodies:
            cfg.max_bodies = max(cfg.max_bodies, args.max_bodies)
        for ref in args.ref:
            scs = {n: sc for n, sc in load_reference(ref).items() if n.startswith(tuple(args.only))}
            if args.launch_fix:
                launch_fix.apply(scs)
            per = evaluate(scs, cfg, {k: [getattr(cfg, k)] for k in PER_ELEMENT})
            m = summary(per)
            key = dataset_name(ref) + "".join(f"[{o}]" for o in args.only if o)
            ck.setdefault("metrics", {})[key] = m
            rows.append((ck["name"], key, m))
            print(f"{ck['name']:24s} {key:18s} loss {m['loss']:7.1f}  shape {m['shape_mm']:6.1f} mm  "
                  f"box {m['box_mm']:6.1f} mm  box turn {m['box_deg']:5.1f} deg", flush=True)
            if args.per_scenario:
                for n, r in m["per_scenario"].items():
                    print(f"    {n:22s} shape {r['shape_mm']:6.1f}  box {r['box_mm']:6.1f}  turn {r['box_deg']:5.1f} deg")
        checkpoint.save(ck, path)
    if args.table:
        names = list(dict.fromkeys(r[0] for r in rows))
        sets = list(dict.fromkeys(r[1] for r in rows))
        lines = ["| Checkpoint | " + " | ".join(f"{s}: shape / box (mm)" for s in sets) + " |",
                 "|---|" + "---|" * len(sets)]
        for n in names:
            cells = [next(f"{m['shape_mm']:.1f} / {m['box_mm']:.1f}" for a, b, m in rows if a == n and b == s)
                     for s in sets]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        with open(args.table, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--ref", nargs="+", required=True, help="reference npz files to train on")
    p.add_argument("--init", default=os.path.join(checkpoint.DIR, "base_pbd_brick.json"))
    p.add_argument("--name", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--out", default=None, help="default checkpoints/<name>.json")
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--fit", nargs="+", default=list(FIT), choices=list(PER_ELEMENT), help="parameters to fit")
    p.add_argument("--set", nargs="*", default=[], help="field=value: fixed settings over the --init checkpoint's")
    p.add_argument("--std", type=float, default=1.0, help="scales the initial spread of every log parameter (STD)")
    p.add_argument("--max-bodies", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--max-bodies", type=int, default=None)
    p.add_argument("--per-scenario", action="store_true")
    p.add_argument("--table", default=None, help="write a markdown comparison table here")
    for sp in sub.choices.values():
        sp.add_argument("--box-angle", type=float, default=0.0,
                        help="weight of the boxes' heading error in the box error (BOX_ANGLE; 0: centers only)")
        sp.add_argument("--launch-fix", action="store_true",
                        help="launch from the vine's own line before contact (launch_fix.py), not the reference's")
        sp.add_argument("--only", nargs="+", default=[""],
                        help="use only the reference scenarios whose names start with one of these (e.g. block_wall)")
    args = parser.parse_args()
    globals()["BOX_ANGLE"] = args.box_angle
    warnings.simplefilter("ignore")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.25)
    fit(args) if args.cmd == "fit" else evaluate_checkpoints(args)


if __name__ == "__main__":
    main()
