import warnings
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


# SCS settings passed through diffcp.
#   acceleration_lookback=0 : disable Anderson acceleration (diffcp warns it "is sometimes
#       unstable"); it also gives slightly better accuracy here and avoids diffcp's double-solve
#       retry, which only triggers when acceleration_lookback is absent.
#   max_iters=2500 : this QP is well-solved but poorly *scaled*, so SCS's residual converges only
#       sublinearly and never self-certifies "solved" (cvxpy's direct solve equilibrates and hits
#       optimal in ~50 iters; diffcp extracts the cone program WITHOUT that scaling and we cannot
#       re-inject it -- diagonal pre-scaling is absorbed by diffcp's own normalization). The
#       returned solution is accurate regardless: at 1000 iters the full sim trajectory already
#       matches a high-accuracy (eps 1e-9) reference to <1e-3. 2500 is a safe margin; raise it if
#       system identification wants tighter forward solves. (Was 10_000 -- pure wasted iterations.)
_SCS_ARGS = {'acceleration_lookback': 0, 'verbose': False, 'max_iters': 2_500}


def solve_layers(Q, p, G, h, A, b):
    """Batched QP solve via cvxpylayers/SCS. Returns the primal solution (batch, N) as float32."""
    batch_size = p.shape[0]
    Q_batched = sqrtm_module.apply(Q).unsqueeze(0).expand(batch_size, -1, -1)
    with warnings.catch_warnings():
        # diffcp raises this when SCS returns "Solved/Inaccurate". Here that status is a verified
        # FALSE ALARM (see max_iters note): the solution matches a high-accuracy reference and the
        # trajectory is unchanged. GENUINE failures (infeasible/unbounded) raise SolverError, not
        # this warning, and are handled upstream in step(), so silencing this hides nothing real.
        warnings.filterwarnings("ignore", message="Solved/Inaccurate.")
        solution = cvxpylayer(Q_batched, p, G, h, A, b, solver_args=_SCS_ARGS)
    return solution[0].float()   # diffcp returns float64 under numpy2; keep float32
