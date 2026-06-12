######################################################
# vine_sim.py
# Completely rewritten vine simulation using:
#  - JAX for automatic differentiation
#  - configuration space (angles, plus last-segment length)
#  - custom PBD solver
#  - same rendering interface, but driven from new c-space
######################################################
import time
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np

from jax import grad, vmap

# Jax prints like this
# jax.debug.print("penalty_grad {}", penalty_grad)

######################################################
# VineParams: holds environment and vine physical data
######################################################

class VineParams:

  def __init__(self, max_bodies, body_length, radius, dt, grow_rate, grow_force, 
               stiffness, damping, substeps, alpha, obstacle_rects, dynamic_objects = np.empty((0,)), use_tube_obstacle=False):
    self.max_bodies = max_bodies
    self.body_length = body_length
    self.radius = radius
    self.dt = dt
    self.grow_rate = grow_rate
    self.grow_force = grow_force
    self.stiffness = stiffness
    self.damping = damping
    self.substeps = substeps
    self.alpha = alpha
    self.obstacle_rects = obstacle_rects
    self.use_tube_obstacle = use_tube_obstacle  # If True, use hardcoded tube instead of env
    
    # for dynamic objects
    self.dynamic_objects = dynamic_objects   # stores current position

    # JAX-functional paradigm: should be returned by function instead, so get rid of this
    # self.dynamic_obj_cspace = dynamic_objects.copy() # stores current position after being updated by grad descent

    if dynamic_objects.size == 0:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, tuple(map(tuple, obstacle_rects)), use_tube_obstacle))
    else:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, tuple(map(tuple, obstacle_rects)), 
                        tuple(map(tuple, dynamic_objects)), 
                        use_tube_obstacle))

  def _tree_flatten(self):

    if self.dynamic_objects.size == 0:
        children = (self.obstacle_rects,)
    else:
    #    children = (self.obstacle_rects, self.dynamic_objects, self.dynamic_obj_cspace)
       children = (self.obstacle_rects, self.dynamic_objects)


    aux_data = {'max_bodies': self.max_bodies,
                'body_length': self.body_length,
                'radius': self.radius,
                'dt': self.dt,
                'grow_rate': self.grow_rate,
                'grow_force': self.grow_force,
                'stiffness': self.stiffness,
                'damping': self.damping,
                'substeps': self.substeps,
                'alpha': self.alpha,
                'use_tube_obstacle': self.use_tube_obstacle}
    
    return (children, aux_data)

  @classmethod
  def _tree_unflatten(cls, aux_data, children):
    # return cls(*children, **aux_data)

    if len(children) == 2:
        # obstacle_rects, dynamic_objects, dynamic_obj_cspace = children
        obstacle_rects, dynamic_objects = children
    else:
        obstacle_rects = children[0]
        dynamic_objects = np.empty((0,))
        # dynamic_obj_cspace = None
    return cls(
        obstacle_rects=obstacle_rects,
        dynamic_objects=dynamic_objects,
        # dynamic_obj_cspace=dynamic_obj_cspace,
        **aux_data
    )
  
  def __hash__(self):
      return self.hash
  
######################################################
# C-Space: [theta_0, ..., theta_(N-1), last_length]
# Each vine has up to N = max_bodies angles.
#  "bodies[i]" says how many *full segments* are present.
# The final partial segment length is cspace[-1].
######################################################


def cspace_to_positions(params: VineParams, cspace: jnp.ndarray, 
                       n_bodies: int, 
                       x0: float, y0: float, heading0: float):
    """
    Convert a single vine's c-space -> global center coordinates of each body
      cspace has shape (N+1,) but we only use the first n_bodies angles (plus last_length).
    We also incorporate an initial anchor (x0, y0) and heading0 for the first segment.

    Returns:
      coords: shape (n_bodies, 2) = (x_i, y_i) for each full segment
              plus potentially a final partial segment if n_bodies < max_bodies
    """
    angles = cspace[:-1]        # shape (n_bodies,)
    last_len = cspace[params.max_bodies] 

    # Step 1: compute global angles for each segment center
    global_angle_full = heading0 + jnp.cumsum(angles)   # shape (n_bodies,)
    
    # Step 2: compute the center of each segment
    #   For the i-th segment, the center is offset from the anchor by
    #        sum_{k=0..i-1} [ L*cos(global_angle_full[k]), L*sin(global_angle_full[k]) ]
    #   But we can do that more efficiently. We'll build an array of cos/sin, then do a cumsum.

    # Cosines and sines of each segment angle:
    c_ = jnp.cos(global_angle_full)
    s_ = jnp.sin(global_angle_full)

    # Prepare the lengths of each segment
    full_lengths = jnp.full((params.max_bodies,), params.body_length)
    full_lengths = full_lengths.at[n_bodies-1].set(last_len)

    # Now we do a cumulative sum of to get the tip coords of each segment
    tip_x = x0 + jnp.cumsum(full_lengths * c_)
    tip_y = y0 + jnp.cumsum(full_lengths * s_)
        
    # Now we'll compute the center of each segment, by
    # subtracting 0.5*full_lengths from the tip coords
    center_x = tip_x - 0.5 * full_lengths * c_
    center_y = tip_y - 0.5 * full_lengths * s_
    
    # Except, for the very last segment, the center position *is* the tip position
    center_x = center_x.at[n_bodies-1].set(tip_x[n_bodies-1])
    center_y = center_y.at[n_bodies-1].set(tip_y[n_bodies-1])

    # Now, use a mask to zero out the segments past n_bodies
    mask = jnp.arange(params.max_bodies) < n_bodies
    center_x = center_x * mask
    center_y = center_y * mask
    
    # Stack them together
    coords = jnp.stack([center_x, center_y], axis=1)
    
    return coords


######################################################
# Collision / SDF
######################################################
def point_rect_sdf(px, py, rect):
    """
    Signed distance from a point (px,py) to axis-aligned rectangle [rx1,ry1, rx2,ry2].
    If inside, distance is negative.
    Otherwise positive. 
    We'll do the usual approach:
      dx = max( [rx1 - px, 0, px - rx2] ), 
      dy = max( [ry1 - py, 0, py - ry2] ),
      dist = sqrt(dx^2 + dy^2).
    If px in [rx1,rx2], dx=0. If py in [ry1,ry2], dy=0. 
    Then sign is negative if px is strictly inside in both x,y.
    """
    rx1, ry1, rx2, ry2 = rect
    dx = jnp.where(px < rx1, rx1 - px, 0.0)
    dx = jnp.where(px > rx2, px - rx2, dx)
    dy = jnp.where(py < ry1, ry1 - py, 0.0)
    dy = jnp.where(py > ry2, py - ry2, dy)
    dist_out = jnp.sqrt(dx*dx + dy*dy)
    # Check if inside
    inside = jnp.logical_and( (px>=rx1)&(px<=rx2), (py>=ry1)&(py<=ry2))
    # If inside => negative distance, we approximate how negative by min distance to an edge
    # The distance to an edge is min( (px - rx1), (rx2 - px), (py - ry1), (ry2 - py) ), but we can do it carefully
    if_inside_dist = jnp.min(jnp.array([px-rx1, rx2-px, py-ry1, ry2-py]))
    dist_signed = jnp.where(inside, -if_inside_dist, dist_out)
    return dist_signed
    
def tube_sdf(pxy):
    """
    SDF for a tube, 1000 long, 500 high, following a sine wave with a full period over the 1000 (2pi)
    
    And the sdf is the distance from a hypothetical 100 diameter tube around the wall (so positive in the tube, negative outside).
    """
    
    px, py = pxy
    
    def distance_to_arc(ppx, ppy, centerx, centery, rad, min_ang, max_ang):
        """
        Compute the distance from a point (ppx, ppy) to an arc centered at (centerx, centery) with radius rad.
        The arc is a full circle, so we can use the standard distance formula.
        """
        
        dist = jnp.hypot(ppx - centerx, ppy - centery)
        dist = jnp.abs(dist - rad)
        
        ang = jnp.atan2(ppy - centery, ppx - centerx)
        dist = jnp.where((ang < min_ang - 0.1) | (ang > max_ang + 0.1), dist + 9999, dist)
        return dist
    
    
    dist_to_arc0 = distance_to_arc(px, py, 800, 0, 800, jnp.pi, 0)
   
    return 0.5 * (20 - dist_to_arc0) + 5000


def vine_collision_sdf(params: VineParams, body_xy: jnp.ndarray, n_bodies: int):
    """
    body_xy: (n_bodies, 2)
    For each body, find the minimum distance to any rectangle 
    and subtract the vine radius. 
    Return shape (n_bodies,) of SDF values. 
    Negative => inside obstacle.
    """
    def dist_to_all_rects(xy):
        # For each rect, compute distance, then take min
        # shape of rects: (R,4)
        px, py = xy

        # We'll vmap the distance to each rect

        all_rects = None
        if params.dynamic_objects.size != 0:
            all_rects = np.append(params.obstacle_rects, params.dynamic_objects, axis=0)
        else:
            all_rects = params.obstacle_rects

        # dists = vmap(point_rect_sdf, in_axes=(None, None, 0))(px, py, params.obstacle_rects)
        dists = vmap(point_rect_sdf, in_axes=(None, None, 0))(px, py, all_rects)

        min_dist = jnp.min(dists)  # min over all rects
        # Then we subtract radius
        return min_dist - params.radius
    
    if params.use_tube_obstacle:
        sd_vals = vmap(tube_sdf)(body_xy)
    else:
        sd_vals = vmap(dist_to_all_rects)(body_xy)
    
    # Mask out the segments that don't exist
    mask = jnp.arange(params.max_bodies) < n_bodies
    sd_vals = jnp.where(mask, sd_vals, 1e6)
        
    return sd_vals  # shape (n_bodies,)


######################################################
# Bending Force
######################################################
# def bending_torques(params: VineParams, cspace: jnp.ndarray, n_bodies: int):
#     """
#     For each angle, we compute a torque that tries to keep angle=0 (or some ref).
#     Basic model: torque_i = -K * theta_i - D * dtheta_i
#     This code can be extended with user control, advanced curves, etc.

#     We do not have velocity in c-space for simplicity; you can store that if you want.
#     For now, treat "dtheta_i" as small or omit damping, or approximate it.

#     Return shape (n_bodies,) of torques.  The final entry is 0 for partial length.
#     """
#     angles = cspace[:n_bodies]
#     # We just do an easy linear model: T = - K * angle
#     T = -params.stiffness * angles
#     # The partial-segment length dimension doesn't get a torque
#     # So we zero out the torque for any index beyond n_bodies
#     # We'll produce shape = (params.max_bodies,) or so, then slice
#     # but let's keep it shape (n_bodies,) for clarity
#     return T


######################################################
# PBD "Single Solve" for collisions + bending
######################################################
def pbd_solve_once(params: VineParams, 
                   cspace: jnp.ndarray,
                   dynamic_obj_positions: jnp.ndarray, # from last bit of grad descent, like cspace 
                   n_bodies: int,
                   target_len: float,
                   bend_params: jnp.ndarray,
                   x0, y0, heading0, bend_energy_func):
    """
    Perform one iteration of a global PBD solve that tries to:
     - push bodies out of collision
     - apply bending torques 
    in a single linear system.

    cspace shape = (max_bodies+1,)
    n_bodies <= max_bodies

    We do a "closed form" approach or approximate: 
      1. Evaluate collision SDF => how far we are inside (negative => collision).
      2. Evaluate partial derivatives (Jacobian) of body positions wrt each angle.
      3. Use the net force/torque from collisions and bending to solve for delta angles.

    In reality, a thorough PBD or XPBD might have multiple constraints. 
    Here we illustrate a single-lumped approach:
      collisions => normal forces => torque 
      bending => torque 
    We'll do a simplified linear approach: delta angles = -K^-1 * grad(energy).
    """
    
    # Step 1: get current body positions
    # Step 2: compute the collision SDF for each body

    # For collisions, we only care about negative SDF => inside obstacle 
    # We'll define collision penalty = -sd_vals if sd_vals<0, else 0
    # Then the direction is the outward normal from the rectangle. 
    # Strictly we’d do the gradient wrt x,y. 
    # But let's do a quick approximate approach: numeric gradient or small AD snippet.

    # Step 2b: define a function "coll_energy = sum( penalty_i^2 )" 
    #    so that the gradient wrt each angle tries to push out of collision.
    def collision_penalty(q):
        xy_ = cspace_to_positions(params, q, n_bodies, x0, y0, heading0)
        sdfs_ = vine_collision_sdf(params, xy_, n_bodies)
                
        # penalty only for negative
        pen = jnp.where(sdfs_<0.0, -sdfs_, 0.0)
        return jnp.sum(0.5 * pen * pen)  # sum of squared penetration

    # Step 3: bending energy
    def bend_penalty(q):
        # simple: sum( 0.5*K*(angle_i^2) ), ignoring partial segment dimension
        # or do the real function that includes q[:n_bodies]
        angles = q[:-1]
        
        deviation = jnp.abs(angles - target_angles)
        
        # Remeber only the gradient of this function matters --
        # The zero-order values are thrown away in the grad() operation
        
        def logg(x):
            return 0.1 * jnp.log2(10 * jnp.abs(x) + 1)
        # E = params.stiffness * jnp.abs(deviation)
        E = params.stiffness * logg(deviation)

                
        # Somehow works too
        # E = 0.0
        
        # Zero out segments that don't exist
        mask = jnp.arange(params.max_bodies) < n_bodies
        E = E * mask
        
        return jnp.sum(E)
    
    def growth_penalty(q):
        # Penalty for not growing the last segment long enough
        return params.grow_force * jnp.abs(target_len - q[params.max_bodies])
    
    def inertial_penalty(new_dynamic_positions):
        # Penalty for how far dynamic objects have moved from their past positions
        # (measured using Euclidean distance)

        if params.dynamic_objects.size == 0:
            return 1

        # print("Original shape:", params.dynamic_objects)
        # print("Argument shape:", new_dynamic_positions)

        diff = params.dynamic_objects - new_dynamic_positions
        is_zero = jnp.allclose(diff, 0.)    # checks if diff is all-zeros
        safe_diff = jnp.where(is_zero, jnp.ones_like(diff), diff) # replaces all zeros with 1's to allow differentiation
        l = jnp.where(is_zero, 0., jnp.linalg.norm(safe_diff)) 
        
        # jax.debug.print("INERTIAL PENALTY: {x}", x = l)
        return l
    
    # Combine them => total energy
    def total_penalty(cspace, dynamic_obj_cspace):

        return 1.0 * collision_penalty(cspace) + \
               1.0 * growth_penalty(cspace) + \
               1.0 * inertial_penalty(dynamic_obj_cspace)

    # Step 4: compute gradient wrt cspace => this is our "force"
    penalty_grad = grad(total_penalty, argnums=0)(cspace, dynamic_obj_positions)

    # Update next dynamic positions based on gradient
    inertial_grad = grad(total_penalty, argnums=1)(cspace, dynamic_obj_positions)

    turning_radius = jnp.where(jnp.abs(cspace[:-1]) < 1e-3, 0, params.body_length * 1e-3 / cspace[:-1])
    bend_moment = -1 * bend_energy_func(turning_radius, bend_params[:, 0], bend_params[:, 1])
        
    penalty_grad = penalty_grad.at[:-1].add(params.stiffness * bend_moment * 8e0)
    
    # We won't normalize the growth rate gradient, save it
    last_seg_grad = penalty_grad[-1] 
    
    # Scale grads using exponential scale: usually small movements of segments
    # at the root greatly affect the position of the tip. So we will scale down
    # gradients there. This doesn't affect convergence -- with enough substeps 
    # it will still reach the same solution but this tends to speed it up
    # for our problems
    index_from_tip = n_bodies - jnp.arange(params.max_bodies+1)
    scale = 1.0 / (jnp.power(2.0, 0.5 * index_from_tip))
    # In this scale, the tip has (relative) grad scaling of 1, the each
    # additional 10 index positions the grad will be halved
    # TODO this doesn't actually work in practice. I think normalizing the grad
    # already has this exp decay effect because the raw grads scale themselves
    # based on the lever moments
    
    # Zero out gradients after n_bodies
    mask = jnp.arange(params.max_bodies+1) < n_bodies
    penalty_grad = penalty_grad * mask
    
    # Normalize grad magnitude to 1
    # FIXME bandaid solution for stability. The magnitude of the
    # grad does actually mean something and should not be ignored
    # But this will require some work to make it stable
    norm_grad = jnp.linalg.norm(penalty_grad)
    penalty_grad = jnp.where(norm_grad > 1e-8, penalty_grad / norm_grad, penalty_grad)

    # Restore the last segment gradient 
    penalty_grad = penalty_grad.at[-1].set(last_seg_grad)
    
    # Step 5: we want to do a single step: 
    #   cspace_{new} = cspace - alpha * Minv*gE
    # For PBD, we often assume mass/inverse mass are all the same or we do 
    # a direct projection.  We'll do a simple "alpha" that you can tune or 
    # that is dt-based. 
    cspace_new = cspace - params.alpha * penalty_grad
    new_dynamic_positions = dynamic_obj_positions - params.alpha * inertial_grad
    
    return cspace_new, new_dynamic_positions



######################################################
# Growth / Extend
######################################################
def multiply_vine(params: VineParams, cspace: jnp.ndarray, n_bodies: int):
    """
    If it exceeds the nominal body_length, we "promote" it to a new full segment 
    (increase n_bodies by 1) and reset partial length to 0. 
    Return (new_cspace, new_n_bodies).
    """
    # The partial length is cspace[max_bodies] 
    last_len_idx = params.max_bodies
    last_len = cspace[last_len_idx]
    
    # If new_len > body_length => we promote
    def promote_body(_):
        # place a new angle at 0 for the new body, 
        # set partial length to new_len - body_length leftover, but typically 0 
        # or set leftover as well. We'll keep it simple and set leftover=0
        # And increment n_bodies
        q_promoted = cspace.at[n_bodies].set(0.0)  # the new angle
        q_promoted = q_promoted.at[last_len_idx].set(last_len - params.body_length)  # leftover
        return (q_promoted, n_bodies+1)
    
    def no_promote(_):
        # q_no_promote = cspace.at[last_len_idx].add(params.grow_force * params.dt)
        return (cspace, n_bodies)

    (cspace_out, n_bodies_out) = jax.lax.cond(last_len > params.body_length, promote_body, no_promote, None)

    return cspace_out, n_bodies_out


######################################################
# Main "advance" for one simulation step
######################################################

def step_vine(params: VineParams, cspace: jnp.ndarray, dynamic_obj_positions: jnp.ndarray, 
              n_bodies: int,
              bend_params: jnp.ndarray,
              x0, y0, heading0, bend_energy_func):
    """
    Advance a single vine's cspace by 1 step (dt).
    1) Possibly grow 
    2) PBD substeps 
    3) Return updated (cspace, n_bodies)
    """
    
    # TODO
    # Add friction (mainly to make vine slide less when growing directly into a wall)
    # Add a pressure term so the vine grows more slowly when its tip is pressed against a wall
    # Both would help with the vine moving a lot between solves
    
    cspace_grown, n_bodies_grown = multiply_vine(params, cspace, n_bodies)
    
    target_len = cspace[-1] + params.grow_rate * params.dt

    # def body_loop_fun(iter, cspace_in, dynamic_positions_in):
    def body_loop_fun(iter, init_val):
        cspace_in = init_val[0]
        dynamic_positions_in = init_val[1]
        new_cspace, new_dynamic_positions = pbd_solve_once(params, cspace_in, dynamic_positions_in,
                                                           n_bodies_grown, target_len, bend_params, x0, y0, heading0, bend_energy_func)\

        return new_cspace, new_dynamic_positions

    init_val = (cspace_grown, dynamic_obj_positions)    
    cspace_final, final_dynamic_positions = jax.lax.fori_loop(0, params.substeps, body_loop_fun, init_val)
    
    return cspace_final, n_bodies_grown, final_dynamic_positions

######################################################
# Batch stepping
#####################################################
def step_vine_batched(params: VineParams, 
                       cspaces: jnp.ndarray,  # shape (batch, max_bodies+1)
                       batched_dynamic_positions: jnp.ndarray, #shape (batch,)
                       n_bodies_list: jnp.ndarray,  # shape (batch,)
                       bend_params: jnp.ndarray,
                       x0_list: jnp.ndarray,
                       y0_list: jnp.ndarray,
                       heading0_list: jnp.ndarray,
                       bend_energy_func: Callable
                       ):

    # We'll vmap over batch dimension
    new_cspaces, new_n_bodies, new_dynamic_positions = vmap(step_vine, (None, 0, 0, 0, 0, None, None, None, None)) \
                                (params, cspaces, batched_dynamic_positions, n_bodies_list, bend_params, x0_list, y0_list, heading0_list, bend_energy_func)
    # out is ( (batch_cspaces), (batch_nb) )
    return new_cspaces, new_n_bodies, new_dynamic_positions

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0) 

    
# from pbd_render import *
from render import _compute_vine_points, init_vis, draw_dead_state, draw_live_state, render


if __name__ == "__main__":
    # Buckling forcer
    # obstacles = [
    #     [40, -20, 70, 20],
    #     [150, -20, 180, 80],
    #     [10, 60, 450, 80],
    #     [10, -60, 450, -40],
    # ]

    # Fig obs
    obstacles = [
        [-20, 20, 10, 80],
        [40, -40, 70, 30],
        [130, -20, 160, 80],
        [220, -40, 250, 30],
        
        [10, 60, 450, 80],
        [10, -60, 450, -40],
    ]
    
    
    # obstacles = [
    #     # Left wall
    #     [-20, 20, 10, 80],
        
    #     # First obstacle (from bottom to middle)
    #     [80, -40, 110, 20],   
        
    #     # Second obstacle (from top to middle)
    #     [190, 0, 220, 80],
        
    #     # Third obstacle (from bottom to middle)
    #     [300, -40, 330, 20],
        
    #     # Fourth obstacle (from top to middle)
    #     [410, 0, 440, 80],
        
    #     # Fifth obstacle (from bottom to middle)
    #     [520, -40, 550, 20],
        
    #     # Sixth obstacle (from top to middle)
    #     [630, 0, 660, 80],
        
    #     # Boundaries
    #     [0, -60, 700, -40],   # Bottom boundary
    #     [0, 80, 700, 100],    # Top boundary
    # ]
    
    
    # obstacles = [[50, 20, 90, 60], 
    #              [150, 20, 190, 50], 
    #              [300, 20, 340, 60], 
    #              [520, 20, 560, 70],
                 
    #              [0, -60, 800, -40],
    #              [0, 80, 800, 100],
    #              ]
    
    obstacles = jnp.array(obstacles)

    # Fast params
    # params = VineParams(
    #     max_bodies=80,
    #     body_length=12.0,
    #     radius=8.0,
    #     dt=1/10,
    #     grow_rate=30.0,
    #     grow_force=30.0,
    #     stiffness=20.0,
    #     damping=50.0,
    #     substeps=10, 
    #     alpha=1e-2,
    #     obstacle_rects=obstacles
    # )
    
    params = VineParams(
        max_bodies=250,
        body_length=12.0,
        radius=8.0,
        dt=1/10,
        grow_rate=20.0,
        grow_force=5.0,
        stiffness=12.0,
        damping=50.0,
        # Curiously, decreasing substeps helps prevent penetration bugs. But it doesn't fix the root problem
        substeps=15, # FIXME THIS NUMBER CAN BE MUCH SMALLER IF WE DO LANGRANGE PROPERRLY
        alpha=1e-2,
        obstacle_rects=obstacles,
    )


    batch_size = 1
    n_bodies_list = jnp.full((batch_size,), 1, dtype=jnp.int32)

    x0_list = 0
    y0_list = -30
    heading0_list = 0.4
    target_angles = jnp.full((batch_size, params.max_bodies,), -0.0)
    
    cspaces = jnp.zeros((batch_size, params.max_bodies+1))
    cspaces = cspaces.at[:, 0].set(0.0)
    cspaces = cspaces.at[:, params.max_bodies].set(params.body_length)
    
    
    # Jit performance is ever so slightly faster than jit + compile
    forward = jax.jit(step_vine_batched, static_argnames=['params'])

    # step_vines_batched = step_vines_batched.lower(params, 
    #                          cspaces,
    #                          n_bodies_list,
    #                          x0_list, y0_list, heading0_list
    #                          ).compile()

    # init_vis()
    init_vis(figsize=(12,8), obstacles=obstacles)
    
    # time.sleep(8)
    
    times_list = []
    
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for step_i in range(870):
        print('step', step_i)
        
        # Step the vines
        start = time.time()
        
        if step_i == 1200:
            params.stiffness = 40.0 # 50
            params.grow_rate = 0.0
            params.grow_force = 0.0
            params.hash = 2
            print('updated')
            # Recreate the jitted function after parameter updates
            forward = jax.jit(step_vine_batched, static_argnames=['params'])
        
        cspaces, n_bodies_list = \
                forward(
                    params,
                    cspaces, 
                    n_bodies_list, 
                    target_angles,
                    x0_list, 
                    y0_list, 
                    heading0_list)
        
        cspaces.block_until_ready()
        end = time.time()
        times_list.append(end - start)
        
        # Render
        if step_i % 150 == 0:
            # draw_vine_batched(params, cspaces, n_bodies_list, 
            #             x0_list, y0_list, heading0_list, 
            #             color='blue')
        
            # plt.title(f"Step {step_i}")
            # plt.pause(0.01)
            
            draw_live_state(params, cspaces, n_bodies_list, x0_list, y0_list, heading0_list)
            render()
            
            time.sleep(0.1)
            
        # Save the state
        antitip_x, antitip_y, tip_x, tip_y, center_x, center_y, n_bodies = _compute_vine_points(params, cspaces, n_bodies_list, x0_list, y0_list, heading0_list)
        center_x = center_x[0, :n_bodies[0]]
        center_y = center_y[0, :n_bodies[0]]
        
        np.save('vine_center_x.npy', center_x)
        np.save('vine_center_y.npy', center_y)
        
        # If any n_bodies >= max_bodies, we stop
        if jnp.any(n_bodies_list >= params.max_bodies):
            print("Reached max number of bodies - stopping.")
            break
        
    # Remove first 5 times
    times_list = times_list[5:]
    print("Average time per step:", sum(times_list) / len(times_list))
    print('Total steps:', len(times_list))
    print('Steps per body:', len(times_list) / params.max_bodies)
    # plt.show()
