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

import cvxpy as cp
from cvxpylayers.jax import CvxpyLayer
from functools import partial
import torch
cvxpylayer = None


######################################################
# VineParams: holds environment and vine physical data
######################################################

class VineParams:

  def __init__(self, max_bodies, body_length, radius, dt, grow_rate, grow_force, 
               stiffness, damping, substeps, alpha, obstacle_rects, dynamic_objects = np.empty((0,)), 
               use_tube_obstacle=False, dynamic_obj_mass_matrix = np.empty((0,))):
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
    self.dynamic_obj_mass_matrix = dynamic_obj_mass_matrix # shaped: (# of objs,)

    if dynamic_objects.size == 0:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, tuple(map(tuple, obstacle_rects)), 
                        use_tube_obstacle))
    else:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, tuple(map(tuple, obstacle_rects)), 
                        tuple(map(tuple, dynamic_objects)), 
                        use_tube_obstacle, tuple(map(tuple, dynamic_obj_mass_matrix))))

  def _tree_flatten(self):

    if self.dynamic_objects.size == 0:
        children = (self.obstacle_rects,)
    else:
       children = (self.obstacle_rects, self.dynamic_objects, self.dynamic_obj_mass_matrix)

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

    if len(children) == 3:
        obstacle_rects, dynamic_objects, dynamic_obj_mass_matrix = children
    else:
        obstacle_rects = children[0]
        dynamic_objects = np.empty((0,))
        dynamic_obj_mass_matrix = np.empty((0,))

    return cls(
        obstacle_rects=obstacle_rects,
        dynamic_objects=dynamic_objects,
        dynamic_obj_mass_matrix = dynamic_obj_mass_matrix,
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

    # angles = cspace[:-1]        # shape (n_bodies,)
    # last_len = cspace[params.max_bodies] 

    last_len = cspace[params.max_bodies, -1]
    angles = cspace[:-1, -1]

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
# Torch Variants (used in QP solver)
######################################################

def torch_point_rect_sdf(px, py, rect):
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
    dx = torch.where(px < rx1, rx1 - px, 0.0)
    dx = torch.where(px > rx2, px - rx2, dx)
    dy = torch.where(py < ry1, ry1 - py, 0.0)
    dy = torch.where(py > ry2, py - ry2, dy)
    dist_out = torch.sqrt(dx*dx + dy*dy)
    # Check if inside
    inside = torch.logical_and( (px>=rx1)&(px<=rx2), (py>=ry1)&(py<=ry2))
    # If inside => negative distance, we approximate how negative by min distance to an edge
    # The distance to an edge is min( (px - rx1), (rx2 - px), (py - ry1), (ry2 - py) ), but we can do it carefully
    if_inside_dist = torch.min(torch.tensor([px-rx1, rx2-px, py-ry1, ry2-py]))
    dist_signed = torch.where(inside, -if_inside_dist, dist_out)
    return dist_signed


def torch_cspace_to_positions(params: VineParams, cspace: torch.tensor, 
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

    # angles = cspace[:-1]        # shape (n_bodies,)
    # last_len = cspace[params.max_bodies] 

    last_len = cspace[params.max_bodies, -1]
    angles = cspace[:-1, -1]

    # Step 1: compute global angles for each segment center
    global_angle_full = heading0 + torch.cumsum(angles)   # shape (n_bodies,)
    
    # Step 2: compute the center of each segment
    #   For the i-th segment, the center is offset from the anchor by
    #        sum_{k=0..i-1} [ L*cos(global_angle_full[k]), L*sin(global_angle_full[k]) ]
    #   But we can do that more efficiently. We'll build an array of cos/sin, then do a cumsum.

    # Cosines and sines of each segment angle:
    c_ = torch.cos(global_angle_full)
    s_ = torch.sin(global_angle_full)

    # Prepare the lengths of each segment
    full_lengths = torch.full((params.max_bodies,), params.body_length)
    full_lengths = full_lengths.at[n_bodies-1].set(last_len)

    # Now we do a cumulative sum of to get the tip coords of each segment
    tip_x = x0 + torch.cumsum(full_lengths * c_)
    tip_y = y0 + torch.cumsum(full_lengths * s_)
        
    # Now we'll compute the center of each segment, by
    # subtracting 0.5*full_lengths from the tip coords
    center_x = tip_x - 0.5 * full_lengths * c_
    center_y = tip_y - 0.5 * full_lengths * s_
    
    # Except, for the very last segment, the center position *is* the tip position
    center_x = center_x.at[n_bodies-1].set(tip_x[n_bodies-1])
    center_y = center_y.at[n_bodies-1].set(tip_y[n_bodies-1])

    # Now, use a mask to zero out the segments past n_bodies
    mask = torch.arange(params.max_bodies) < n_bodies
    center_x = center_x * mask
    center_y = center_y * mask
    
    # Stack them together
    coords = torch.stack([center_x, center_y], axis=1)
    
    return coords


def torch_vine_collision_sdf(params: VineParams, body_xy: torch.tensor, n_bodies: int):

    def dist_to_all_rects(xy):
        # For each rect, compute distance, then take min
        # shape of rects: (R,4)
        px, py = xy

        # We'll vmap the distance to each rect

        #NOTE: we only compare vine->static sdfs in solver, so no need to include dyn-objs here
        all_rects = params.obstacle_rects

        # dists = vmap(point_rect_sdf, in_axes=(None, None, 0))(px, py, params.obstacle_rects)
        dists = torch.vmap(point_rect_sdf, in_axes=(None, None, 0))(px, py, all_rects)

        min_dist = torch.min(dists)  # min over all rects
        # Then we subtract radius
        return min_dist - params.radius
    
    if params.use_tube_obstacle:
        sd_vals = torch.vmap(tube_sdf)(body_xy)
    else:
        sd_vals = torch.vmap(dist_to_all_rects)(body_xy)
    
    # Mask out the segments that don't exist
    mask = torch.arange(params.max_bodies) < n_bodies
    sd_vals = torch.where(mask, sd_vals, 1e6)
        
    return sd_vals  # shape (n_bodies,)


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
    '''
    NOTE:
    For maximal coord system, cspace is now (n, 3). This needed to be done to incorporate
    physics for dynamic obstacles.
    '''
    
    # Step 1: get current body positions
    # Step 2: compute the collision SDF for each body

    # For collisions, we only care about negative SDF => inside obstacle 
    # We'll define collision penalty = -sd_vals if sd_vals<0, else 0
    # Then the direction is the outward normal from the rectangle. 
    # Strictly we’d do the gradient wrt x,y. 
    # But let's do a quick approximate approach: numeric gradient or small AD snippet.

    # Step 2b: define a function "coll_energy = sum( penalty_i^2 )" 
    #    so that the gradient wrt each angle tries to push out of collision.
    def collision_penalty(q, dynamic_obj_positions):
        xy_ = cspace_to_positions(params, q, n_bodies, x0, y0, heading0)
        sdfs_ = vine_collision_sdf(params, xy_, n_bodies)
                
        # penalty only for negative
        pen = jnp.where(sdfs_<0.0, -sdfs_, 0.0)

        if dynamic_obj_positions.size == 0:
            return jnp.sum(0.5 * pen * pen)
        
        # penalty for dynamic obstacles interacting with each other or static obstacles:
        #NOTE: shape of dynamic_obj_positions: (num objs, 4)

        all_rects = np.append(params.obstacle_rects, params.dynamic_objects, axis=0)

        def obj_collision_penalty(obj_coords: tuple[4]):
            '''
            Uses point_rect_sdf on each corner of the obj 
            Returns sum of the negative distances based from the corners
            '''

            up_left = (obj_coords[0], obj_coords[1])
            down_right = (obj_coords[2], obj_coords[3])
            down_left = (up_left[0], down_right[1])
            up_right = (down_right[0], up_left[1])

            all_corners = [up_left, down_right, down_left, up_right]

            raw_corner_dists = [vmap(point_rect_sdf, (None, None, 0))(corner[0], corner[1], all_rects) 
                                for corner in all_corners]
            # only keep negative values
            real_corner_dists = [jnp.where(dist >= 0, 0.0, dist) for dist in raw_corner_dists]

            # jax.debug.print("REAL CORNER DISTS: {x}", x = type(raw_corner_dists[0]))

            return sum(real_corner_dists)

        all_obj_penalties = [obj_collision_penalty(tuple(obj)) for obj in dynamic_obj_positions]
        total_obj_pen = sum(all_obj_penalties) 

        # jax.debug.print("OBJ COLLISION PEN: {x}", x = total_obj_pen)

        return jnp.sum(0.5 * pen * pen)  # sum of squared penetration

    # Step 3: bending energy
    def bend_penalty(q):
        # simple: sum( 0.5*K*(angle_i^2) ), ignoring partial segment dimension
        # or do the real function that includes q[:n_bodies]
        angles = q[:-1, -1]
        
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
        return params.grow_force * jnp.abs(target_len - q[params.max_bodies, -1])
    
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

        return 1.0 * collision_penalty(cspace, dynamic_obj_cspace) + \
               1.0 * growth_penalty(cspace) + \
               1.0 * inertial_penalty(dynamic_obj_cspace)

    # Step 4: compute gradient wrt cspace => this is our "force"
    penalty_grad = grad(total_penalty, argnums=0)(cspace, dynamic_obj_positions)

    # Update next dynamic positions based on gradient
    inertial_grad = grad(total_penalty, argnums=1)(cspace, dynamic_obj_positions)

    turning_radius = jnp.where(jnp.abs(cspace[:-1, -1]) < 1e-3, 0, params.body_length * 1e-3 / cspace[:-1, -1])
    bend_moment = -1 * bend_energy_func(turning_radius, bend_params[:, 0], bend_params[:, 1])
        
    penalty_grad = penalty_grad.at[:-1, -1].add(params.stiffness * bend_moment * 8e0)
    
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
    penalty_grad = penalty_grad * mask[:, None]
    
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
    last_len = cspace[last_len_idx, -1]
    
    # If new_len > body_length => we promote
    def promote_body(_):
        # place a new angle at 0 for the new body, 
        # set partial length to new_len - body_length leftover, but typically 0 
        # or set leftover as well. We'll keep it simple and set leftover=0
        # And increment n_bodies
        q_promoted = cspace.at[n_bodies, -1].set(0.0)  # the new angle
        q_promoted = q_promoted.at[last_len_idx, -1].set(last_len - params.body_length)  # leftover
        return (q_promoted, n_bodies+1)
    
    def no_promote(_):
        # q_no_promote = cspace.at[last_len_idx].add(params.grow_force * params.dt)
        return (cspace, n_bodies)

    (cspace_out, n_bodies_out) = jax.lax.cond(last_len > params.body_length, promote_body, no_promote, None)

    return cspace_out, n_bodies_out


######################################################
# Splitting Cone Solver: new solver to account for dynamic obstacles
######################################################

def extend_cspace(params: VineParams, cspace: torch.tensor, dstate: torch.tensor, n_bodies:int):
    '''
    Extends the cspace by one, if needed
    (Mostly copied from DiffVine)
    '''

    new_i = n_bodies
    last_i = n_bodies - 1
    penult_i = n_bodies - 2

    #FIXME: attempt to address from DiffVine
    cspace = torch.tensor(cspace)

    # Indices for each one in (n, 3) cspace
    x = 0
    y = 1
    theta = 2

    # Compute position of second last seg
    endingx = cspace[penult_i, x] + params.radius * torch.cos(cspace[penult_i, theta])
    endingy = cspace[penult_i, y] + params.radius * torch.sin(cspace[penult_i, theta])   

    # Compute last body's distance
    last_link_distance = ((cspace[last_i, x] - endingx)**2 + \
                          (cspace[last_i, y] - endingy)**2).sqrt().squeeze(-1)
    
    # x2 to prevent 0-len segments
    extend_needed = last_link_distance > params.radius * 2

    # Compute location of new seg
    last_link_theta = torch.atan2(cspace[last_i, y] - cspace[penult_i, y], 
                                  cspace[last_i, x] - cspace[penult_i, x])
    
    new_seg_x = endingx + params.radius * torch.cos(last_link_theta)
    new_seg_y = endingy + params.radius * torch.sin(last_link_theta)
    new_seg_theta = last_link_theta.squeeze()

    # Copy last body one forward
    cspace[new_i, x] = torch.where(extend_needed, cspace[last_i, x], cspace[new_i, x])
    cspace[new_i, y] = torch.where(extend_needed, cspace[last_i, y], cspace[new_i, y])
    cspace[new_i, theta] = torch.where(extend_needed, cspace[last_i, theta], cspace[new_i, theta])

    # Set the new segment position
    cspace[last_i, x] = torch.where(extend_needed, new_seg_x, cspace[last_i, x])
    cspace[last_i, y] = torch.where(extend_needed, new_seg_y, cspace[last_i, y])
    cspace[last_i, theta] = torch.where(extend_needed, new_seg_theta, cspace[last_i, theta])

    # Initialize d_state
    dstate[new_i, x] = 0
    dstate[new_i, y] = 0
    dstate[new_i, theta] = 0

    # Update n_bodies
    n_bodies = n_bodies + extend_needed

    return cspace, n_bodies


def cspace_sdf_measure(params: VineParams, 
                   cspace: torch.tensor, 
                   n_bodies: int,
                   x0: float, y0:float, heading0: float):
    '''
    Measure of how much vine intersects with static obstacles
    (Dynamic obstacle intersection is handled by another constraint)
    '''

    # Measure of vine colliding with all other obstacles:
    xy_ = torch_cspace_to_positions(params, cspace, n_bodies, x0, y0, heading0)
    sdfs_ = torch_vine_collision_sdf(params, xy_, n_bodies)
    return sdfs_


def dynamic_obj_sdf_measure(params: VineParams, dynamic_obj_positions: torch.tensor):
    '''
    Measure of how much each dynamic obstacle intersects
    any other obstacle (whether dynamic or static)

    #NOTE: assumes dynamic AND static objects to be rectangular
    '''
    
    def obj_to_corners(coords: tuple[4]):
        '''
        Returns x,y's of given object's corners (matching indices => same corner across both arrays)
        '''
        obj_corners = ((coords[0], coords[0]), # top left
                       (coords[1], coords[1]), # bottom right
                       (coords[0], coords[1]), # bottom left
                       (coords[1], coords[0])  # top right
                       )
        xs = obj_corners[:, 0]
        ys = obj_corners[:, 1]
        return xs, ys

    def check_obj_collision(coords: tuple[4], all_rects, is_recursive_call:bool = False):
        '''
        Returns scalar for given dynamic obstacle;
        More negative => more collisions with static objs and/or other dynamic objs
        
        Done by checking how deep each obj's corner is within other objs' sdfs;
        Returns (4 * all_rects,) array, containing negative values where collisions were detected
        '''            

        # Check one way: given obj -> other objs
        xs, ys = obj_to_corners(coords)

        dists = torch.vmap(torch_point_rect_sdf, in_axes=(0, 0, 0))(xs, ys, all_rects)
        dists = dists.flatten()
        dists = torch.where(dists > 0, 0, dists)

        if is_recursive_call:
            return dists 

        # Check other way: other objs -> given obj
        other_dists = torch.vmap(check_obj_collision, in_axes=(0, None, None))(all_rects, coords, True)
        other_dists = other_dists.flatten()
        other_dists = torch.where(other_dists > 0, 0, other_dists)

        return torch.sum(dists) + torch.sum(other_dists)
    
    all_rects = np.append(params.obstacle_rects, dynamic_obj_positions, axis=0)

    dyn_obj_collision_measures = torch.vmap(check_obj_collision, in_axes=(0, None, None))(
        dynamic_obj_positions, all_rects, False
    )

    return dyn_obj_collision_measures


def joint_measure(params: VineParams,
                    c_space: torch.tensor,
                    n_bodies: int,
                    x0: float, y0: float):
    '''
    Joints of the vine should be kept at fixed distance of each other
    '''

    constraints = torch.zeros(n_bodies * 2)
    xs = c_space[:n_bodies, 0]
    ys = c_space[:n_bodies, 1]
    thetas = c_space[:n_bodies, 2]
    
    constraints[0] = (xs[0] - x0) - params.radius * torch.cos(thetas[0])
    constraints[1] = (ys[0] - y0) - params.radius * torch.sin(thetas[0])

    constraints[2::2] = (xs[1:] - xs[:-1]) - params.radius * torch.cos(thetas[1:]) \
                                         - params.radius * torch.cos(thetas[:-1])

    constraints[3::2] = (ys[1:] - ys[:-1]) - params.radius * torch.sin(thetas[1:]) \
                                         - params.radius * torch.sin(thetas[:-1])
    
    #NOTE: is zero_out required here like from DiffVine?    

    return constraints


def proximity_measure(params: VineParams, 
                         cspace: torch.tensor,
                         dynamic_obj_positions: torch.tensor,
                         n_bodies: int,
                         x0: float, y0: float, heading0: float):
    '''
    The distance between any joint to any dynamic obj > 0 to prevent intersection

    # NOTE: assumes that dynamic objects are rectnagular
    '''
    joint_centers = torch_cspace_to_positions(params, cspace, n_bodies, x0, y0, heading0)
    joint_radius = params.radius

    def get_overlap_measure(joint_center_coords: tuple[2], joint_radius: float,
                        dynamic_obj_position: tuple[4]):
        '''
        Checks to see if given joint is in collision with given dynamic object;
        Positive value indicates they're not in collision, 
        negative value indicates otherwise
        '''
        center_x, center_y = joint_center_coords
        rect_left, rect_top, rect_right, rect_bottom = dynamic_obj_position

        # Find point on rectangle closest to joint center
        closest_x = torch.clamp(torch.tensor(center_x), rect_left, rect_right)
        closest_y = torch.clamp(torch.tensor(center_y), rect_bottom, rect_top)

        # Depth of rectangle's penetration:
        dist = torch.sqrt(torch.pow(closest_x - center_x, 2) + torch.pow(closest_y - center_y, 2))
        depth = torch.tensor(joint_radius) - dist

        return depth.item() * -1 # flip sign so that penetration is negative
    
    def overlap_over_all_objects(joint_center_coords: tuple[2], joint_radius: float,
                                 dynamic_obj_positions: torch.tensor):
        '''
        V-mapped get_overlap_measure: get the depths of every dynamic obj
        in respect to the given joint,
        return result shaped (len(dynamic_obj_positions),)
        '''
        joint_depths = torch.vmap(get_overlap_measure, in_axes=(None, None, 0))(
                            joint_center_coords, joint_radius, dynamic_obj_positions)
        return joint_depths

    all_joint_depths = torch.vmap(overlap_over_all_objects, in_axes=(0, None, 0))(
        joint_centers, joint_radius, dynamic_obj_positions
    )

    return all_joint_depths


def growth_measure(params: VineParams, cspace: torch.tensor,
                   dstate: torch.tensor, n_bodies: int):
    '''
    Measure of how much the current growing segment is growing;
    Growth should be constrained such that the current segment should always be growing
    each time step
    '''
    curr_id = n_bodies - 1
    prev_id = n_bodies - 2

    # Current growing segment info:
    curr_x = cspace[curr_id, 0]
    curr_y = cspace[curr_id, 1]
    curr_velocity_x = dstate[curr_id, 0]
    curr_velocity_y = dstate[curr_id, 1]

    # Previous segment info:
    prev_x = cspace[prev_id, 0]
    prev_y = cspace[prev_id, 1]
    prev_velocity_x = dstate[prev_id, 0]
    prev_velocity_y = dstate[prev_id, 1]

    # Derivative of the distance in respect to time:
    growth = ((curr_x - prev_x) * (curr_velocity_x - prev_velocity_x) + 
              (curr_y - prev_y) * (curr_velocity_y - prev_velocity_y)) / \
              torch.sqrt(torch.pow(curr_x - prev_x, 2) + torch.pow(curr_y - prev_y, 2))
    return growth


def get_bending_energy(params: VineParams, cspace: torch.tensor, bend_params: torch.tensor, bend_energy_func: Callable):

    #NOTE: vine stiffness and damping are NOT utilized

    turning_radius = torch.where(torch.abs(cspace[:-1, -1]) < 1e-3, 0, params.body_length * 1e-3 / cspace[:-1, -1])
    bend_moment = -1 * bend_energy_func(turning_radius, bend_params[:, 0], bend_params[:, 1])

    return bend_moment


def get_object_motion(params: VineParams, dstate: torch.tensor, weight: float):
    '''
    Returns (len(dynamic_objects), ) shaped tensor;
    Calculates modified KE for each dynamic object 
    '''
    #NOTE: in forces, each object has (change x, change y, change theta)

    #NOTE: weight is currently random and can be changed
    def compute_KE(weight: float, obj_mass: float, obj_velocities: tuple[3]):
        return torch.tensor([
            weight * obj_mass * obj_velocities[0], # x
            weight * obj_mass * obj_velocities[1], # y
            weight * obj_mass * obj_velocities[2], # theta
        ])

    # Shape: (# dynamic objs, 3)
    return torch.vmap(compute_KE, in_axes=(None, 0, 0))(weight, params.dynamic_obj_mass_matrix, dstate)


def compute_jacobians(params: VineParams, 
                      cspace: torch.tensor,
                      dstate: torch.tensor, # velocity vector; shape: (n_bodies + dynamic_objs, 3)
                      dynamic_obj_positions: torch.tensor,
                      n_bodies: int,
                      x0: float, y0: float, heading0: float,
                      bend_params: torch.tensor, bend_energy_func):
    '''
    Finds jacobians of the contraint equations used by the QP solver
    '''
    #FIXME: convert c_space, dynamic_obj_positions to torch tensors elsewhere in code

    cspace, n_bodies = extend_cspace(params, cspace, dstate, n_bodies)

    cspace_sdj_jac = torch.func.jacrev(partial(cspace_sdf_measure,
                                                    params=params,
                                                    x0=x0, y0=y0, n_bodies=n_bodies, heading0=heading0),
                                            )(c_space=cspace)
    cspace_sdf_now = cspace_sdf_measure(params, cspace, n_bodies, x0, y0, heading0)

    dynamic_sdf_jac = torch.func.jacrev(partial(dynamic_obj_sdf_measure, params))(dynamic_obj_positions)
    dynamic_sdf_now = dynamic_obj_sdf_measure(params, dynamic_obj_positions)

    joint_jac = torch.func.jacrev(partial(joint_measure, params=params, n_bodies=n_bodies,
                                          x0=x0, y0=y0))(cspace=cspace)
    joint_now = joint_measure(params, cspace, n_bodies, x0, y0)

    proximity_jac = torch.func.jacrev(partial(proximity_measure, params=params,
                                              n_bodies=n_bodies, x0=x0, y0=y0, heading0=heading0))(
                                                  cspace=cspace, dynamic_obj_positions=dynamic_obj_positions
                                              )
    proximity_now = proximity_measure(params, cspace, dynamic_obj_positions, n_bodies,
                                      x0, y0, heading0)
    
    growth_jac = torch.func.jacrev(partial(growth_measure, params=params, n_bodies=n_bodies))(
        cspace=cspace, dstate=dstate
    )
    growth_now = growth_measure(params, cspace, dstate, n_bodies)

    # Find bend energy to minimize for the vine:
    bend_energy = get_bending_energy(params, cspace, bend_params, bend_energy_func)

    # Find motion info for dynamic objects:
    
    obj_motion = get_object_motion(params, dynamic_obj_positions)

    #NOTE: forces shape: (cpsace + dyn_obj len, 3)

    forces = torch.zeros(cspace.shape[0] + dynamic_obj_positions.shape[0], 3)
    start_obj_idx = n_bodies + 1

    forces[:n_bodies, 2] += -bend_energy
    forces[:n_bodies - 1, 2] += bend_energy[1:]

    forces[:n_bodies, 0] += dstate[:n_bodies, 0] # for x 
    forces[:n_bodies, 1] += dstate[:n_bodies, 1] # for y

    # Add on movement energy of the objects for minimization        
    #FIXME: add ability to read in mass for each dynamic object

    forces[start_obj_idx:, :] += obj_motion[:, :]

    return n_bodies, forces, cspace_sdj_jac, cspace_sdf_now, dynamic_sdf_jac, dynamic_sdf_now, \
            joint_jac, joint_now, proximity_jac, proximity_now, growth_jac, growth_now

compute_jacobians_batched = torch.func.vmap(compute_jacobians, in_dims = (None, 0, 0, 0, 0, 0, 0, 0, None, None))


def SCS_solve_layers():
    '''
    Batched QP solve
    '''


def SCS_solve(params: VineParams, dstate, forces,
              cspace_sdf_jac, cspace_sdf_now,
              dynamic_sdf_jac, dynamic_sdf_now,
              joint_jac, joint_now,
              proximity_jac, proximity_now,
              growth_jac, growth_now):

    global cvxpylayer

    


    # Initialize layers:
    next_dstate = cp.variable()


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
    
    target_len = cspace[-1, -1] + params.grow_rate * params.dt

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
                       batched_dynamic_positions: jnp.ndarray, #shape (batch, num dybanamic_objs, 4)
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
