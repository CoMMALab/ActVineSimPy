import numpy as np
import pandas as pd
import jaxlib
from jax import numpy as jnp
from jax import jit, vmap, lax

from flax import struct

from pred_phi_eps.use_example import load_trained_model as load_eps_nn
from pred_phi_m.use_example import load_trained_model as load_m_nn
from data.nn_interp import load_nn as load_interp_nn

@struct.dataclass
class ActParams:
    # Actuator design parameters
    Rc: float = 0.005
    R_tube: float = 0.0171807761
    fab_buffer_length: float = 0.005
    film_elasticity_a: float = 0.01
    P_act_max: float = 34473.8  # Pa

    # Vine segment properties
    R_vine: float = 0.0333509183
    P_vine: float = 13789.52     # Pa
    sub_segment_length: float = 0.01
    eps_critical: float = 0.01
    psi: float = 6894.76  # Pa

@jit
def lazy_valid_jit(aa, lc):
    cond1 = jnp.logical_and(lc >= 0.06, lc <= 1.0)
    cond2 = jnp.logical_and(aa > 0.0, aa <= 3.11)
    cond3 = aa <= (3.11 / 0.95) * lc
    return jnp.logical_and(cond1, jnp.logical_and(cond2, cond3))

def lazy_valid(aa, lc):
    if lc < 0.06 or lc > 1:
        return False
    if aa <= 0 or aa > 3.11:
        return False
    if aa > 3.11/0.95 * lc:
        return False
    return True

class sPAM_design_selection:
    # setup
    def remove_constant_columns(self, csv_path, save_cleaned=False, output_path=None):
        df = pd.read_csv(csv_path)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
        df_cleaned = df.drop(columns=constant_cols)

        if save_cleaned:
            if output_path is None:
                output_path = csv_path.replace(".csv", "_cleaned.csv")
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned CSV saved to: {output_path}")

        return df_cleaned

    def __init__(self, df_folder, input_df_mac, output_df_mac, input_df, output_df, wrinkling_model=True):
        # load dfs
        self.vine_segment_df = self.remove_constant_columns(f"{df_folder}/{input_df}.csv")
        self.actuator_design_df = self.remove_constant_columns(f"{df_folder}/{output_df}.csv")
        self.vine_segment_df_mac = self.remove_constant_columns(f"{df_folder}/{input_df_mac}.csv")
        self.actuator_design_df_mac = self.remove_constant_columns(f"{df_folder}/{output_df_mac}.csv")

        # get ordered data ranges
        self.df_length = len(self.vine_segment_df_mac)
        self.vine_segment_lengths = self.vine_segment_df_mac["l_curve"].tolist()
        self.min_vine_segment_length = self.vine_segment_lengths[0]
        self.max_vine_segment_length = self.vine_segment_lengths[-1]
        self.max_arc_angles = self.vine_segment_df_mac["arc_angle"].tolist()

        self.load_and_prepare_data()
        self.phi_eps_nn = load_m_nn()
        self.eps_nn = load_eps_nn()
        self.interp_nn = load_interp_nn()

        self.psi = 6894.76
        self.wrinkling_model = wrinkling_model

    def load_and_prepare_data(self):
        vine_df = self.vine_segment_df
        actuator_df = self.actuator_design_df

        assert len(vine_df) == len(actuator_df), "Mismatch in row count between input and output CSVs"

        # Extract only the required columns
        l_curve = vine_df["l_curve"]
        arc_angle = vine_df["arc_angle"]
        num_actuators = actuator_df["num_actuators"]

        # Combine into one DataFrame (by index)
        df = pd.DataFrame({
            "l_curve": l_curve,
            "arc_angle": arc_angle,
            "num_actuators": num_actuators
        })

        # Sort by vine segment length first, then arc angle
        df = df.sort_values(by=["l_curve", "arc_angle"]).reset_index(drop=True)
        self.interpolate_df = df
        l_vals = np.sort(df["l_curve"].unique())
        a_vals = np.sort(df["arc_angle"].unique())
        grid = np.full((len(l_vals), len(a_vals)), np.nan)

        for _, row in df.iterrows():
            i = np.searchsorted(l_vals, row["l_curve"])
            j = np.searchsorted(a_vals, row["arc_angle"])
            grid[i, j] = row["num_actuators"]

        self.l_vals = l_vals
        self.a_vals = a_vals
        self.num_act_grid = grid
    

    def get_phi_eps_nn(self):
        return self.phi_eps_nn
    
    def get_eps_nn(self):
        return self.phi_eps_nn
    
    def get_vals(self):
        return self.l_vals, self.a_vals, self.num_act_grid


# validity checker (batched)
@jit
def interpolate_num_actuators(l_vals, a_vals, num_act_grid, l_queries, arc_queries):
    lq = jnp.asarray(l_queries)
    aq = jnp.asarray(arc_queries)

    l_idx = jnp.searchsorted(l_vals, lq, side="right")
    a_idx = jnp.searchsorted(a_vals, aq, side="right")

    valid = (l_idx > 0) & (l_idx < len(l_vals)) & (a_idx > 0) & (a_idx < len(a_vals))
    result = jnp.full_like(lq, jnp.nan, dtype=float)

    i0 = l_idx[valid] - 1
    i1 = l_idx[valid]
    j0 = a_idx[valid] - 1
    j1 = a_idx[valid]

    l0 = l_vals[i0]
    l1 = l_vals[i1]
    a0 = a_vals[j0]
    a1 = a_vals[j1]

    f00 = num_act_grid[i0, j0]
    f10 = num_act_grid[i1, j0]
    f01 = num_act_grid[i0, j1]
    f11 = num_act_grid[i1, j1]

    t = (lq[valid] - l0) / (l1 - l0)
    u = (aq[valid] - a0) / (a1 - a0)

    interp_vals = (
        (1 - t) * (1 - u) * f00 +
        t * (1 - u) * f10 +
        (1 - t) * u * f01 +
        t * u * f11
    )

    result = result.at[valid].set(interp_vals)
    return jnp.round(result).astype(int)



@jit
def vine_bending_moment(joint_angles, P_vine, R_vine, eps_critical):
    P = P_vine
    max_moment = P * jnp.pi * R_vine**3

    theta_min = 2.0 * jnp.arcsin(eps_critical)
    theta_min_mask = joint_angles > theta_min

    # Wrinkling-based model
    divisor = jnp.sin(joint_angles / 2.0)
    divisor = jnp.where(divisor == 0.0, 1e-6, divisor)
    gamma_0_wrinkle = jnp.arccos(jnp.clip(2.0 * eps_critical / divisor - 1.0, -1.0, 1.0))

    wrinkle_part = max_moment * (
        (jnp.sin(2.0 * gamma_0_wrinkle) + 2 * jnp.pi - 2 * gamma_0_wrinkle)
        / (4.0 * (jnp.sin(gamma_0_wrinkle) + (jnp.pi - gamma_0_wrinkle) * jnp.cos(gamma_0_wrinkle)))
    )

    # Linear part
    slope = 0.5
    linear_part = max_moment * slope * joint_angles / theta_min

    bending_moment = jnp.where(theta_min_mask, wrinkle_part, linear_part)
    return bending_moment


@jit
def compute_R_act(act_params, phi_rc_array):
    R_acts = act_params.Rc / jnp.cos(phi_rc_array)
    R_acts = jnp.minimum(R_acts, act_params.R_tube)
    return R_acts


@jit
def actuator_tensile_force(Rc, P_act, phi_rc, m):
    Ft = jnp.pi * P_act * Rc**2 * (1.0 - 2.0 * m) / (2.0 * m * jnp.cos(phi_rc)**2)
    return Ft


@jit
def solve_for_R_curve(R_vine, R_act, N, eps_actuator, l_curve, l0):
    N = N[:, None]
    l_curve = l_curve[:, None]

    delta_L = N * l0 * eps_actuator
    eps = delta_L / l_curve
    R_curve = (2.0 * R_vine + R_act) / eps - R_vine
    arc_angle = l_curve / R_curve
    arc_angle = arc_angle % (2.0 * jnp.pi)
    return R_curve, arc_angle


@jit
def solve_for_P_act(act_params, R_act, R_curve, FtperPress):
    joint_angles = act_params.sub_segment_length / R_curve
    M_vine = vine_bending_moment(
        joint_angles,
        act_params.P_vine,
        act_params.R_vine,
        act_params.eps_critical
    )

    P_act_required = M_vine / (2.0 * act_params.R_vine + R_act) / (FtperPress + 1e-8)
    return P_act_required


@jit
def calculate_actuator_length(fab_buffer_length, num_actuators, l_curve):
    return l_curve / num_actuators - fab_buffer_length


@jit
def get_actuator(act_params, aa, lc, phi_eps_nn, interp_nn):
    valid_mask_1 = lazy_valid_jit(aa, lc)
    batch_size = aa.shape[0]
    m_array = jnp.linspace(0.01, 0.5, 1000)  # (1000,)
    m_all = jnp.tile(m_array[None, :], (batch_size, 1))  # (B, 1000)

    # Per-batch num_actuators and actuator length
    num_actuators = interp_nn(lc, aa)               # (B,)
    l0_array = calculate_actuator_length(act_params.fab_buffer_length, num_actuators, lc)         # (B,)

    Rc = act_params.Rc
    R_tube = act_params.R_tube

    l0_all = jnp.repeat(l0_array[:, None], 1000, axis=1)  # (B, 1000)
    inputs = jnp.stack([
        l0_all,
        m_all
    ], axis=-1)  # (B, 1000, 2)
    inputs_flat = inputs.reshape(batch_size * 1000, 2)

    # Run NN and reshape
    phi_eps_outputs = phi_eps_nn(inputs_flat).reshape(batch_size, 1000, 2)
    phi_rc_array = phi_eps_outputs[..., 0]  # (B, 1000)
    eps_array = phi_eps_outputs[..., 1]     # (B, 1000)

    # Compute R_acts from phi_rc
    R_acts = compute_R_act(act_params, phi_rc_array)  # (B, 1000)

    # Mask invalid eps
    valid_eps = eps_array >= 0.0

    P_act = 1.0
    F_tperPress = actuator_tensile_force(act_params.Rc, P_act, phi_rc_array, m_all)

    R_curves, arc_angles = solve_for_R_curve(act_params.R_vine, R_acts, num_actuators, eps_array, lc, l0_array)
    arc_angles = arc_angles % (2 * jnp.pi)

    P_act_required = solve_for_P_act(act_params, R_acts, R_curves, F_tperPress)

    min_P = 0.01 * act_params.psi
    max_P = act_params.P_act_max
    feasible_mask = (P_act_required >= min_P) & (P_act_required <= max_P)

    angle_error = jnp.abs(arc_angles - aa[:, None])
    final_mask = feasible_mask & valid_eps
    angle_error = jnp.where(final_mask, angle_error, jnp.inf)

    best_idx = jnp.argmin(angle_error, axis=1)           # (B,)
    batch_idx = jnp.arange(batch_size)                   # (B,)

    best_P_act = P_act_required[batch_idx, best_idx]
    best_phi_rc = phi_rc_array[batch_idx, best_idx]
    best_m = m_all[batch_idx, best_idx]
    best_R_act = compute_R_act(act_params, best_phi_rc)

    valid_mask_2 = jnp.isfinite(angle_error[batch_idx, best_idx])  # (B,)
    valid_mask = jnp.logical_and(valid_mask_1, valid_mask_2)  # (B,)

    return {
        "design_feasible": valid_mask,        # (B,)
        "l0": l0_array,                       # (B,)
        "P_act": best_P_act,                  # (B,)
        "m": best_m,                          # (B,)
        "phi_rc": best_phi_rc,               # (B,)
        "R_act": best_R_act,                 # (B,)
        "num_actuators": num_actuators,      # (B,)
    }

@jit
def actuator_moment_single(act_params, epsnn, arc_angle, l0, p_act, valid_len, joint_angle_array, sub_segment_length):
    max_len = joint_angle_array.shape[0]
    mask = jnp.arange(max_len) < valid_len
    joint_angles = jnp.where(mask, joint_angle_array, -jnp.inf)

    R_curves = jnp.clip(sub_segment_length / jnp.abs(joint_angles), a_min=1e-8, a_max=1e8)
    eps = (2 * act_params.R_vine + act_params.R_tube) / (R_curves + act_params.R_vine)
    m, phi_rc = epsnn(l0, eps)
    moment = jnp.sign(arc_angle) * calculate_actuator_moment(
        act_params.Rc,
        act_params.R_vine,
        act_params.R_tube,
        p_act,
        m,
        phi_rc,
    )
    return moment

@jit
def compute_R_act_single(Rc, R_tube, phi_rc):
    R_act = Rc / jnp.cos(phi_rc)
    return jnp.minimum(R_act, R_tube)

@jit
def calculate_actuator_moment(Rc, R_vine, R_act, P_act, m, phi_rc):
    F_t = jnp.pi * P_act * Rc**2 * (1.0 - 2.0 * m) / (2.0 * m * jnp.cos(phi_rc)**2)
    return F_t * (2.0 * R_vine + R_act)


@jit
def moment_per_joint(actuator_moment, joint_angle_array, P_vine, R_vine, eps_critical):
    vine_moment = vine_bending_moment(jnp.abs(joint_angle_array), P_vine, R_vine, eps_critical)
    return actuator_moment - vine_moment * jnp.sign(joint_angle_array)


if __name__ == "__main__":
    data_folder = "data"
    input_df_mac = "vine_mac"
    output_df_mac = "act_mac"
    input_df = "vine"
    output_df = "act"

    act_params = ActParams()
    ss = sPAM_design_selection(data_folder, input_df_mac, output_df_mac, input_df, output_df)
    l_vals, a_vals, num_act_grid = ss.get_vals()
    vine_segment_length = 0.9

    arc_angle = np.pi * .1

    num_actuators = interpolate_num_actuators(l_vals, a_vals, num_act_grid, arc_angle, vine_segment_length)

    print("num_actuators", num_actuators)

    # save phi and eps arrays since they run the nn B*1000 times
    phi_eps_nn = ss.get_phi_eps_nn()
    eps_nn = ss.get_eps_nn()
    interp_nn = ss.interp_nn()
    output = get_actuator(act_params, np.abs(arc_angle), vine_segment_length, phi_eps_nn, interp_nn)
    r_act_init = output["r_act"]
    l0 = output["l0"]
    p_act = output["P_act"]
    # once above arrays are calculated for an actuator design you just need to calculate moment using following function as design evolves

    joint_angle_array = 0.01 * np.ones((4,))
    valid_len = 4
    m_array = np.linspace(0.01, 0.5, 1000, endpoint=True)
    actuator_m = actuator_moment_single(m_array, eps_nn, arc_angle, l0, p_act, valid_len, joint_angle_array, act_params.sub_segment_length)

    resultant_moment = moment_per_joint(actuator_m, joint_angle_array, act_params.P_vine, act_params.R_vine, act_params.eps_critical)  
    print("Resultant Moment at Joint:", resultant_moment)