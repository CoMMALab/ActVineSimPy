import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from .sqrtm import MatrixSquareRoot

# Built once per process on the first init_layers() call for a given problem shape.
cvxpylayer = None


def init_layers(sol_size, Q_size, p_size, G_size, h_size, A_size, b_size):
    """Set up the differentiable cvxpylayer for the QP (inputs are unbatched sizes).
    No-op if already built -- reset the module global `cvxpylayer` to None to rebuild."""
    global cvxpylayer
    if cvxpylayer is not None:
        return

    next_dstate = cp.Variable(sol_size)
    Q_sqrt = cp.Parameter(Q_size)
    p = cp.Parameter(p_size)
    G = cp.Parameter(G_size)
    h = cp.Parameter(h_size)
    A = cp.Parameter(A_size)
    b = cp.Parameter(b_size)

    objective = cp.Minimize(0.5 * cp.sum_squares(Q_sqrt @ next_dstate) + p @ next_dstate)
    constraints = [A @ next_dstate == b, G @ next_dstate <= h]
    problem = cp.Problem(objective, constraints)

    cvxpylayer = CvxpyLayer(problem, parameters=[Q_sqrt, p, G, h, A, b], variables=[next_dstate])


sqrtm_module = MatrixSquareRoot()


def solve_layers(Q, p, G, h, A, b):
    """Batched QP solve via cvxpylayers/SCS. Returns the primal solution (batch, N) as float32."""
    batch_size = p.shape[0]
    Q_batched = sqrtm_module.apply(Q).unsqueeze(0).expand(batch_size, -1, -1)
    solver_args_scs = {'acceleration_lookback': 40_000, 'verbose': False, 'max_iters': 10_000}
    solution = cvxpylayer(Q_batched, p, G, h, A, b, solver_args=solver_args_scs)
    return solution[0].float()   # diffcp returns float64 under numpy2; keep float32
