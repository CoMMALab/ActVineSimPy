# How the `dvsim` solver works, and where sPAM fits in

This explains what one simulation step computes, why it is a quadratic program (QP), how that
relates to SQP, how the QP is actually solved, and why the sPAM actuator model is awkward for
it. File and function names refer to this folder.

## 1. The problem a step has to solve

The vine is a chain of rigid segments ("bodies"), each with a pose `(x, y, theta)`. Movable
boxes are extra bodies with their own `(x, y, theta)`. Stack all poses into `q` and all
velocities into `v`.

Over one time step `dt` we want new velocities that obey Newton's law,

```
M (v_new - v_old) = dt * f + (impulses from constraints)
```

where `M` is the mass matrix (masses and rotational inertias on the diagonal) and `f` holds
the forces we can evaluate directly: bending torques and damping. The constraints are:

- **Joints** (equalities): consecutive segments stay connected, and the base stays put.
- **Growth**: the tip link lengthens. Either at exactly the commanded speed (an equality) or
  driven by a force (section 6).
- **Contact** (inequalities): no gap may go negative. Vine against walls, vine against boxes,
  boxes against walls, boxes against each other.

The unknowns are `v_new` and the constraint impulses. The impulses are not free: a contact can
only push, and only when it is touching. That "either there is a gap, or there is a force, not
both" condition is called complementarity, and it is what makes contact hard.

## 2. Why it becomes a QP

There is a classical result (Gauss's principle of least constraint): the velocity that
satisfies Newton's law with ideal constraint forces is the one **closest to the unconstrained
velocity, measured in the mass metric, among all velocities that respect the constraints**.

The unconstrained velocity is `v_free = v_old + dt * M^-1 f`. Expanding
`1/2 (v - v_free)' M (v - v_free)` and dropping the constant gives

```
minimize    1/2 v' M v + p' v          with p = -(M v_old + dt f)
subject to  A v  = b                   joints and growth
            G v <= h                   contacts
```

A quadratic objective with linear constraints is a QP. Its optimality (KKT) conditions are
exactly Newton's law plus the complementarity conditions, and the Lagrange multipliers of the
constraints are the constraint impulses. So the contact forces are never computed explicitly:
they fall out as multipliers.

Because `M` is in the objective, mass matters. A heavy box costs more to move than a light one,
so the optimizer bends the vine around it instead. This is what position-based dynamics (the
simulator on `master`) cannot express: PBD moves positions directly to remove constraint
violations and never sees a mass matrix.

In the code the sign convention for forces is flipped (`forces` holds minus the force), so the
line is `p = forces * dt - dstate @ M`.

### Where the linear constraints come from

The real constraints are nonlinear functions of the poses: the gap between a circle and a
rotated box, the distance between segment ends. Each step linearizes them about the current
state. For a gap function `phi(q)`:

```
phi(q + dt v)  ~  phi(q) + dt * dphi/dq * v  >= 0      ->      -dt * dphi/dq * v <= phi(q)
```

That is one row of `G v <= h`, with `h` the current gap. The Jacobians `dphi/dq` come from
automatic differentiation (`torch.func.jacrev`), which is why a new contact type only needs a
gap function. Rotation comes for free: the gap to a box depends on the box's angle, so its
derivative with respect to that angle is the torque arm.

Joints are handled the same way, as `dt * J v = -deviation`. Putting the current violation on
the right-hand side means any drift left over from the last step is removed in this one.

### The rows, concretely (`dynamic_vine.py:dynamic_solve`)

| Block | Rows | Built from |
|---|---|---|
| objective `M` | 3 per vine body, 3 per box | `create_M`, `create_M_obj` (box inertia from its mass and size) |
| objective `p` | same | previous momentum, bending torque, linear damping |
| objective, friction | one term per box | `object_friction_weights` (section 5) |
| equality: joints | 2 per body | `joint_deviation` and its Jacobian |
| equality: growth | 1 (plus one slack variable in force mode) | `growth_rate` and its Jacobians |
| inequality: vine vs walls | 1 per body | `sdf` |
| inequality: vine vs boxes | bodies x boxes | `proximity_measure` |
| inequality: box vs walls | boxes x walls | `object_env_constraints` |
| inequality: box vs box | boxes x boxes | `object_env_constraints` |

The problem has a fixed size, `max_bodies` segments, whether or not they have grown yet.
Bodies that do not exist yet are pinned with `v = 0` rows so the equality matrix keeps full
rank, and their gap rows are set to "far away".

After the solve, `step()` clamps velocity spikes and integrates `q += dt * v`.

## 3. How this relates to SQP

Sequential quadratic programming is the general method for

```
minimize f(x)   subject to   c(x) = 0,   g(x) <= 0
```

with nonlinear `f`, `c`, `g`. It repeats:

1. At the current guess `x_k`, replace the objective by a quadratic model and the constraints
   by their linearizations. That is a QP.
2. Solve the QP for a step `d`.
3. Move, `x_{k+1} = x_k + d` (usually with a line search), and repeat until nothing changes.

It is Newton's method for constrained problems: each QP is a Newton step on the KKT conditions,
with the bonus that the QP decides by itself which inequalities are active.

One simulation step is exactly one such iteration. The nonlinear problem is "find the velocity
closest to `v_free` such that the poses at the end of the step violate nothing". We linearize
the constraints once, at the start of the step, solve one QP, and move on. We never iterate to
convergence within a step. What makes that acceptable:

- The objective is already exactly quadratic; only the constraints are approximated.
- The step is small (`dt = 1/90 s`), so the linearization is close.
- Errors do not accumulate, because the next step puts the current violation back into `h`
  and `b`.

The price is that fast motion makes the linearization poor (hence the velocity clamps), and
that anything in `f` is frozen for the whole step. That second point limits how stiff the vine
can be (section 8).

## 4. How the QP is solved (`solver.py`)

1. **Stating it.** The QP is written once in `cvxpy` with `M`, `p`, `G`, `h`, `A`, `b` as
   parameters (`init_layers`). The objective is written as `1/2 ||M^(1/2) v||^2`, because
   cvxpy needs the problem to stay convex whatever the parameter values are, and a parameter
   times a variable squared would not pass that check. `sqrtm.py` supplies `M^(1/2)`.
2. **Compiling it.** `cvxpylayers` turns the problem into a standard cone program once. After
   that, every solve only fills in numbers.
3. **Solving it.** `diffcp` passes the cone program to Clarabel, an interior-point solver. It
   keeps all inequalities strictly satisfied and follows a path to the optimum, converging here
   in about 9 iterations.
4. **Batching.** A batch of states gives a batch of QPs. The Jacobians for the whole batch are
   computed together with `vmap`, but `diffcp` solves the QPs one by one on a CPU thread pool.
   Measured on 12 cores with 60 bodies: 47 rollout-steps/s at batch 1, 320 at 32, 422 at 128.
5. **Differentiating it.** The solution of a QP is a differentiable function of its data
   (differentiate the KKT conditions). `diffcp` provides that, so a loss on the simulated
   trajectory can be back-propagated to `m`, `I`, stiffness and damping. That is the mechanism
   DiffVine used for system identification. When nothing asks for gradients the derivative work
   is skipped.
6. **Failing.** With growth as a hard equality, a vine that is boxed in and cannot grow makes
   the QP infeasible. `step()` then re-solves the batch element by element and
   freezes only the elements that fail; `return_ok=True` reports which.

`USE_QPTH` switches to a GPU interior-point solver for large batches. It only handles pure QPs,
so it cannot be used with friction on.

The compiled layer is cached in a module global and depends on the problem's shape, so it has
to be reset (`solver.cvxpylayer = None`) whenever the number of bodies, walls or boxes changes.

## 5. Friction between a box and the table

A box resting on the table is pressed down by its weight `m g`, so the table can resist
sliding with a force of at most `mu * m * g`. In impulse form that adds one term per box to the
objective:

```
dt * mu * m * g * || (vx, vy, c * omega) ||
```

`c` is the mean distance of the box's footprint from its center, which converts spin into an
equivalent sliding speed. Using one norm over all three components means sliding and spinning
share the same friction budget.

The norm has a kink at zero, and that kink is the stick/slip behaviour:

- If the push on the box is smaller than friction can resist, the cheapest velocity is exactly
  zero. The box does not creep.
- If it is larger, the box moves, and friction keeps removing `mu * m * g * dt` of momentum
  per step. A free box sliding at 300 mm/s with `mu = 0.5` loses 54.5 mm/s per step and stops
  dead after six steps.

A heavier box is therefore harder to move twice over: more inertia in `M`, and a higher
friction threshold. With a norm in the objective the problem is strictly a second-order cone
program, not a QP; Clarabel solves both.

`params.obj_friction` is `mu` (default 0.5, not measured). Contacts themselves (vine against
box, box against wall) are still frictionless, and the vine does not rub on the table.

## 6. Growth: a speed or a force

`growth_rate` measures how fast the tip link (last body relative to the one before it) is
lengthening. There are two ways to impose growth.

**Hard (`params.grow_force = None`, as in DiffVine).** One equality row: the tip link lengthens
at exactly `grow_rate`. The solver will apply whatever force that takes. A vine driven into a
wall has no way to satisfy it; a vine driven into a 5 kg box simply shoves the box.

**Force (`params.grow_force` set).** Growth is a force along the tip link that falls off
linearly with speed: zero at the commanded speed, and the stall force `grow_force` when the tip
is stopped. In the QP this is one extra variable `s`, the shortfall in growth speed:

```
growth(v) + s = target speed            cost   1/2 * c * s^2,    c = dt * grow_force / target speed
```

The multiplier of that row, `c * s`, is the growth impulse. Nothing is ever infeasible, and a
blocked vine pushes with a bounded force until something gives. Measured (vine radius 50 mm,
0.3 m/s):

| Case | Hard | 0.5 N | 2 N |
|---|---|---|---|
| free growth for 1 s | 327 mm | 268 mm | 309 mm |
| head-on into a wall | stuck at the wall | stuck at the wall | buckles and runs along the wall |
| 5 kg box in the way (friction 24.5 N) | pushes it 33 mm | - | box does not move, vine goes around |

This is a constant-pressure model: the stall force is fixed. It does not build up over time the
way a pump under PID control would, and it does not depend on vine length.

## 7. The sPAM actuator

### The model

An actuator's moment on its joint is **constant** for a given design. It is evaluated once, at
the contraction the actuator has at its design bend radius, and does not change with the
joint's current angle. What changes with the angle is the vine's own bending moment. The joint
settles where the two cancel.

```
eps     = (2 R_vine + R_act) / (design radius + R_vine)   contraction at the design radius
phi, m  = MLP(eps, |l0|)                                   neural surrogate of the sPAM equations
F_t     = p * pi * R_c^2 * (1 - 2m) / (2m cos^2 phi)       actuator tensile force
M_act   = sign(l0) * F_t * (2 R_vine + R_act)              its moment about the far wall of the vine

joint torque = M_act - k * theta - c * theta_dot          k, c: the vine's linear stiffness and damping
```

`spam.py` has the forward model (`actuation_moment`) and its inverse (`design_actuator`: the
pressure and rest length that give a wanted moment at a wanted radius, choosing the rest
length that needs the least pressure).

This follows the structure of the original sPAM reference code (actuator moment computed once
per design, then `actuator moment - vine moment(joint angle)` per joint), with the vine's
wrinkling moment replaced by the linear spring.

It is **not** what `master`'s `solve_fwd` does. That function uses the same `F_t` and `eps`
expressions, but re-evaluates them at the joint's current radius on every step and subtracts a
constant vine force, so there the actuator varies with the angle and the vine does not.

### Where it enters the solver

`bending_energy` in `vine.py` computes one torque per joint, and in `'spam'` mode that torque
is the line above. It is applied to the two bodies the joint connects and ends up in `p`, like
any other bending torque. sPAM supplies a number per joint and knows nothing about the solver.

Because the actuator moment is constant, it adds no stiffness. The slope of the joint torque
is just `k`, the same as a passive vine, so actuation cannot destabilize the step.

It also makes `'spam'` mode mathematically the same as `'curvature'` mode:
`M_act - k * theta = -k * (theta - M_act / k)`, a spring toward the angle `M_act / k`. The two
give identical trajectories. `'curvature'` skips the MLP and takes the rest angle directly.

### The catch: the vine in the sim is about 1000x too soft

Designing an actuator needs the vine's stiffness `k` in real units. The sim's value is
0.03 N m/rad per joint. The moment needed to hold a 0.5 m bend is then 0.001 N m, and the
actuator pressure that gives it is **15 Pa**. Real sPAMs run at tens of kPa. For comparison,
the vine's maximum bending moment in the reference model is `pi * P * R^3`, about 1.2 N m.

So the pressures this mode reports are not meaningful yet. The curvature it produces is,
because the same `k` is used to design the actuator and to simulate the vine.

## 8. Why the vine has to be soft: the explicit torque limit

The torque in `p` is **explicit**: evaluated at the angle the joint has at the start of the
step, then held for the whole step. For a spring of stiffness `k` on a body of inertia `I`,
that is only stable when the step cannot overshoot, roughly

```
dt^2 * k / I  <  about 1        ->        k  <  about I / dt^2
```

and similarly `dt * c / I < about 1` for damping. With `I = 10` (internal units) and
`dt = 1/90 s` that caps `k` at a few 10^4 and `c` at about 900. The defaults are `k = 30000`
and `c = 50`. The comments in `VineParams` saying "too large is unstable" are this limit.

A real vine is far outside it. With the segment inertia the sim uses (1e-5 kg m^2) and a
realistic stiffness of tens of N m/rad, the joint's natural frequency is over 1000 rad/s, which
a 1/90 s explicit step cannot follow. Physically the vine is quasi-static in bending: it is
always at its equilibrium shape. The sim replaces that with a soft, slow spring.

Consequences:

- Stiffness, damping and inertia cannot simply be set to measured values. Only combinations
  inside the stable region run.
- From motion alone only ratios are identifiable (`k / I`, `c / I`, `M_act / k`), which is why
  system identification gave many equally good answers.
- Within the stable region the choice still matters. A sweep of 96 combinations of inertia,
  mass, stiffness and damping (`xnew_rrt/sweep_vine_params.py`, free vine at 0.3 m/s) scored how
  well the vine follows a commanded curvature. 73 combinations ran to full length. The defaults
  here (inertia 1e-5 kg m^2, mass 0.02 kg, stiffness 0.3, damping 0.5) reach the right mean
  curvature but each joint oscillates by 37% of the commanded angle. Quadrupling the damping and
  quartering the mass brings that to 4%, which is what `xnew_rrt/scene.py` uses. At 0.5 m/s
  nothing tried tracks well.
- To use real stiffness, the bending torque would have to be taken semi-implicitly: add
  `dt^2 * D' K D` to the quadratic term so a stiff joint looks heavy to rotate for that step.
  Same QP, same solver, but the quadratic term then differs per batch element.

## 9. Stiffness modes

`params.stiffness_mode` selects the joint torque:

| Mode | Torque | Control |
|---|---|---|
| `'linear'` | spring toward straight, plus damping | none: a passive vine |
| `'spam'` | the linear spring plus a constant actuator moment (section 7) | `spam_moment` per joint |
| `'curvature'` | the linear spring, toward `bend_target` | `bend_target` per joint; same dynamics as `'spam'` |

`step(..., per_batch={...})` lets the control differ between batch elements.

## 10. What is not physical yet

- No parameter has been measured. Vine mass, inertia, stiffness and damping are DiffVine's
  default starting values; box masses, `obj_friction` and `grow_force` are placeholders.
- `GROW_FACTOR`, `I_FACTOR`, `DAMP_FACTOR`, `STIFF_FACTOR` in `vine.py` are unit fudge factors
  inherited from DiffVine. `si.py` absorbs them so parameters can be given in SI.
- Contact friction (vine against box, box against wall) and vine-on-table friction are missing.
