'''
Reference rollouts from master's quasi-static PBD vine (the ActVine paper sim), used by
fit_to_master.py to fit dvsim's vine parameters.

Master runs on jax, so this script does NOT run in the `vine` env, and it runs master's code
unchanged from a copy of the branch (the working tree's sPAM/ differs from master):

    git archive master | tar -x -C /tmp/master_src
    <python with jax+flax+jaxopt> xnew_rrt/master_reference.py --master-src /tmp/master_src

It uses the sim settings of master's planner (kinodynamic/sst.py: body 68 mm, radius 50 mm,
grow_rate 20, grow_force 15, stiffness 20, 15 substeps) and master's control: a curvature in
1/m, turned into an sPAM design (pressure, rest length) by sPAM.nns_usage.solve and applied to
every body grown from then on. Writes xnew_rrt/data/master_ref.npz: per scenario the walls,
start, control schedule and the vine centerline (mm) every RECORD_MM of growth.
'''

import argparse, json, os, sys

import numpy as np

RECORD_MM = 20.0

# Before master commit befe1e1, sPAM.nns_usage.solve designed the actuator at rest length l0 but
# returned 2 * l0, and solve_fwd evaluates the actuator at the value it is given, so the vine
# settled at a different curvature than commanded (1 -> 2.9, 2 -> 1.9, 3.33 -> 1.0 1/m).
# --l0-doubling reproduces that on a fixed master.
L0_DOUBLING = 2.0

# name: (walls [x1 y1 x2 y2] mm, start (x, y, heading), [(curvature 1/m, length mm), ...])
SCENARIOS = {
    "free_straight": ([], (0, 0, 0.0), [(0.0, 900)]),
    "free_k1":       ([], (0, 0, 0.0), [(1.0, 900)]),
    "free_k2":       ([], (0, 0, 0.0), [(2.0, 900)]),
    "free_k3.3":     ([], (0, 0, 0.0), [(3.33, 900)]),
    "free_k-2":      ([], (0, 0, 0.0), [(-2.0, 900)]),
    "free_s_curve":  ([], (0, 0, 0.0), [(3.0, 350), (-3.0, 550)]),
    "free_late_bend": ([], (0, 0, 0.0), [(0.0, 400), (2.5, 500)]),
    # grows into a vertical wall at 45 deg and has to slide along it
    "wall_45":       ([[400, -800, 450, 800]], (0, 0, 0.785), [(0.0, 900)]),
    # bends toward a wall it runs parallel to and has to lie along it
    "wall_parallel": ([[-100, 120, 1000, 170]], (0, 0, 0.0), [(3.0, 900)]),
    # bends away from a post it grows past
    "post":          ([[250, -40, 300, 400]], (0, -150, 0.0), [(1.5, 900)]),
}


def vine_polyline(params, cspace, n_bodies, x0, y0, heading0):
    '''Base point plus the end point of every body (mm), as master's render._compute_vine_points.'''
    angles = cspace[:-1][:n_bodies]
    lengths = np.full(n_bodies, params.body_length)
    lengths[n_bodies - 1] = cspace[params.max_bodies]
    glob = heading0 + np.cumsum(angles)
    xs = x0 + np.concatenate([[0.0], np.cumsum(lengths * np.cos(glob))])
    ys = y0 + np.concatenate([[0.0], np.cumsum(lengths * np.sin(glob))])
    return np.stack([xs, ys], axis=1)


def mlp_error(predict, csv_path, n=4000):
    '''Mean abs error of the sPAM surrogate (eps, l0) -> (phi, m) on its own training data.'''
    import pandas as pd
    df = pd.read_csv(csv_path).dropna()
    df = df.sample(n=min(n, len(df)), random_state=0)
    out = np.asarray(predict(np.asarray(df[["eps", "l0"]].values, dtype=np.float32)))
    return np.abs(out - df[["phi", "m"]].values).mean(axis=0)


def load_master_mlp(act_params, get_or_train_model, get_prediction_function):
    '''
    master's sPAM surrogate. Before master commit befe1e1 the only checkpoint was an unfinished
    orbax save (checkpoint_100.orbax-checkpoint-tmp-0), so restore_checkpoint silently returned the random
    initial network. If that happens, retrain it the way get_or_train_model does (same functions,
    100 epochs, lr 5e-2, batch 256, seed) on master's committed data.csv; the result is cached
    next to it as trained_params.npy.
    '''
    import glob, pickle
    import jax, jax.numpy as jnp
    from sPAM import nns

    state, scaling_info, model = get_or_train_model(act_params)
    ckpt_path = os.path.dirname(glob.glob("sPAM/model_*/data.csv")[0])
    csv_path = os.path.join(ckpt_path, "data.csv")
    err = mlp_error(get_prediction_function(state, scaling_info, model), csv_path)
    print(f"master sPAM surrogate as loaded: mean abs error phi {err[0]:.3f}, m {err[1]:.3f}")
    if err.max() < 0.05:   # a working surrogate is ~0.02; the untrained one ~0.5
        return get_prediction_function(state, scaling_info, model)

    cache = os.path.join(ckpt_path, "trained_state.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            params, scaling_info = pickle.load(f)
        state = state.replace(params=params)
    else:
        print("checkpoint did not load; retraining it with master's training code on data.csv")
        import pandas as pd
        df = pd.read_csv(csv_path).dropna()
        inputs = jnp.asarray(df[["eps", "l0"]].values, dtype=jnp.float32)
        outputs = jnp.asarray(df[["phi", "m"]].values, dtype=jnp.float32)
        x_scaled, y_scaled, scaling_info = nns.create_dataset_and_scale(inputs, outputs)
        rng = np.random.default_rng(seed=42)
        idx = np.arange(x_scaled.shape[0]); rng.shuffle(idx)
        train = idx[:int(0.8 * len(idx))]
        x_train, y_train = x_scaled[train], y_scaled[train]
        state = nns.create_train_state(jax.random.PRNGKey(0), model, 5e-2, input_shape=(256, 2))
        for epoch in range(100):
            state = state.replace(metrics=nns.Metrics.empty())
            perm = rng.permutation(len(train))
            for s in range(0, len(train), 256):
                state = nns.train_step(state, x_train[perm][s:s + 256], y_train[perm][s:s + 256])
            if epoch % 20 == 19:
                print(f"  epoch {epoch + 1}: train mse {state.metrics.compute()['mse']:.2e}")
        scaling_info = {k: np.asarray(v) for k, v in scaling_info.items()}
        with open(cache, "wb") as f:
            pickle.dump((jax.device_get(state.params), scaling_info), f)

    predict = get_prediction_function(state, scaling_info, model)
    err = mlp_error(predict, csv_path)
    print(f"master sPAM surrogate retrained: mean abs error phi {err[0]:.4f}, m {err[1]:.4f}")
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-src", required=True, help="a `git archive master` copy of the repo")
    parser.add_argument("--out", default=None,
                        help="default xnew_rrt/data/master_ref.npz (master_ref_l0x2.npz with --l0-doubling)")
    parser.add_argument("--l0-doubling", action="store_true",
                        help="reproduce master's old l0 doubling bug (see L0_DOUBLING)")
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                "master_ref_l0x2.npz" if args.l0_doubling else "master_ref.npz")
    args.out = os.path.abspath(args.out)

    os.chdir(args.master_src)          # sPAM loads its checkpoint by relative path
    sys.path.insert(0, args.master_src)
    import types
    class Stub(types.ModuleType):
        def __getattr__(self, name):
            return None
    # imported by master but only used for drawing / generating sPAM training data
    for stub in ("pygame", "jaxopt", "sklearn", "sklearn.decomposition"):
        if stub not in sys.modules:
            try:
                __import__(stub)
            except ImportError:
                sys.modules[stub] = Stub(stub)
    import jax, jax.numpy as jnp
    from pbd_vine import VineParams, step_vine_batched
    from sPAM.spam import params as act_params
    from sPAM.nns import get_or_train_model, get_prediction_function
    from sPAM.nns_usage import solve, solve_fwd

    predict = load_master_mlp(act_params, get_or_train_model, get_prediction_function)
    design =jax.jit(jax.vmap(solve, in_axes=(None, None, 0)), static_argnames=('predict', 'params'))
    fwd = jax.vmap(lambda r, p, l0: solve_fwd(predict, act_params, r, p, l0), in_axes=(0, 0, 0))

    out = {}
    meta = {}
    for name, (walls, start, schedule) in SCENARIOS.items():
        # master's planner settings (kinodynamic/sst.py); walls far away when there are none
        rects = np.array(walls if walls else [[1e5, 1e5, 1e5 + 1, 1e5 + 1]], dtype=np.float32)
        params = VineParams(max_bodies=70, body_length=68.0, radius=50.0, dt=1 / 10, grow_rate=20.0,
                            grow_force=15.0, stiffness=20.0, damping=50.0, substeps=15, alpha=1e-2,
                            obstacle_rects=rects)
        forward = jax.jit(step_vine_batched, static_argnames=['params', 'x0_list', 'y0_list',
                                                              'heading0_list', 'bend_energy_func'])
        x0, y0, h0 = start
        cspace = np.zeros((1, params.max_bodies + 1), dtype=np.float32)
        cspace[0, -1] = params.body_length
        bodies = np.array([1], dtype=np.int32)
        control = np.zeros((1, params.max_bodies, 2), dtype=np.float32)
        control[:, :, 1] = 0.025 / 2  # master's unactuated default

        grown = 0.0
        frames, lengths = [], []
        next_record = 0.0
        for curvature, seg_mm in schedule:
            if abs(curvature) > 1e-6:
                p, l0 = design(predict, act_params, jnp.array([1.0 / curvature]))
                if args.l0_doubling:
                    l0 = l0 * L0_DOUBLING
                control[0, int(bodies[0]):, 0] = float(p[0])
                control[0, int(bodies[0]):, 1] = float(l0[0])
            else:
                control[0, int(bodies[0]):, 0] = 0.0
                control[0, int(bodies[0]):, 1] = 0.025 / 2
            end = grown + seg_mm
            # a vine that stalls against a wall may never grow the full length: give up after 10x
            # the steps free growth would need
            max_steps = int(10 * seg_mm / (params.grow_rate * params.dt))
            for _ in range(max_steps):
                if grown >= end - 1e-6:
                    break
                if grown >= next_record - 1e-6:
                    frames.append(vine_polyline(params, np.asarray(cspace[0]), int(bodies[0]), x0, y0, h0))
                    lengths.append(grown)
                    next_record += RECORD_MM
                cspace, bodies = forward(params, jnp.asarray(cspace), jnp.asarray(bodies), jnp.asarray(control),
                                         x0, y0, h0, fwd)
                cspace, bodies = np.array(cspace), np.array(bodies)
                total = (int(bodies[0]) - 1) * params.body_length + float(cspace[0, -1])
                grown = total - params.body_length  # growth since the start (starts with one full body)
                if bodies[0] >= params.max_bodies - 1:
                    break
        frames.append(vine_polyline(params, np.asarray(cspace[0]), int(bodies[0]), x0, y0, h0))
        lengths.append(grown)

        for k, f in enumerate(frames):
            out[f"{name}/frame{k}"] = f.astype(np.float32)
        out[f"{name}/grown_mm"] = np.array(lengths, dtype=np.float32)
        meta[name] = {"walls_mm": walls, "start": list(start), "schedule": schedule,
                      "num_frames": len(frames), "initial_body_mm": params.body_length}
        print(f"{name}: {len(frames)} frames, grew {grown:.0f} mm, tip {frames[-1][-1].round(0)}")

    out["meta"] = np.array(json.dumps({"record_mm": RECORD_MM, "l0_doubling": args.l0_doubling,
                                       "scenarios": meta}))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
