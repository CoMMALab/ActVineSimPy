"""sPAM elastic + actuation bending moment for the dvsim vine (the ActVine design model).

Wraps the trained torch sPAM surrogate. `make_moment_fn()` returns a joint-vectorized
callable moment(turning_radius, p, l0) suitable for use inside dvsim's bending_energy.
"""
import torch

_predict = None
_act_params = None


def _load():
    global _predict, _act_params
    if _predict is None:
        from sPAM.torch_nns import get_or_train_model, get_prediction_function
        from sPAM.spam import params as act_params
        scaling_info, model = get_or_train_model(act_params)
        _predict = get_prediction_function(scaling_info, model)
        _act_params = act_params
    return _predict, _act_params


def make_moment_fn():
    """Joint-vectorized sPAM forward moment: moment(turning_radius, p_act, l0) -> per-joint moment."""
    from sPAM.torch_nns_usage import solve_fwd
    predict, act_params = _load()

    def one(r, p, l0):
        return solve_fwd(predict, act_params, r, p, l0)

    return torch.vmap(one, in_dims=(0, 0, 0))
