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
               use_tube_obstacle=False, 
               dynamic_objs_mass = np.empty((0,)),
               dynamic_objs_inertia = np.empty((0,)),
               mass = torch.tensor([0.02], dtype=torch.float32), # mass for each vine segment
               inertia = torch.tensor([10.0 / 100], dtype=torch.float) # moment of inertia for each vine seg 
               ):
    
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

    self.mass = mass
    self.inertia = inertia
    
    # for dynamic objects
    self.dynamic_objects = dynamic_objects   # stores current position
    self.dynamic_objs_mass = dynamic_objs_mass # shaped: (# of objs,)
    self.dynamic_objs_inertia = dynamic_objs_inertia

    if dynamic_objects.size == 0:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, tuple(map(tuple, obstacle_rects)), 
                        use_tube_obstacle, mass, inertia))
    else:
        self.hash = hash((max_bodies, body_length, radius, dt, grow_rate, grow_force,
                        stiffness, damping, substeps, alpha, 
                        tuple(map(tuple, obstacle_rects)), 
                        tuple(map(tuple, dynamic_objects)), 
                        use_tube_obstacle, 
                        tuple(map(tuple, dynamic_objs_mass)), 
                        tuple(map(tuple, dynamic_objs_inertia)),
                        mass, inertia))

  def _tree_flatten(self):

    if self.dynamic_objects.size == 0:
        children = (self.obstacle_rects,)
    else:
       children = (self.obstacle_rects, self.dynamic_objects, self.dynamic_objs_mass,
                   self.dynamic_objs_inertia)

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
                'use_tube_obstacle': self.use_tube_obstacle,
                'mass': self.mass,
                'inertia': self.inertia}
    
    return (children, aux_data)

  @classmethod
  def _tree_unflatten(cls, aux_data, children):

    if len(children) == 4:
        obstacle_rects, dynamic_objects, dynamic_objs_mass, dynamic_objs_inertia = children
    else:
        obstacle_rects = children[0]
        dynamic_objects = np.empty((0,))
        dynamic_objs_mass = np.empty((0,))
        dynamic_objs_inertia = np.empty((0,))

    return cls(
        obstacle_rects=obstacle_rects,
        dynamic_objects=dynamic_objects,
        dynamic_objs_mass = dynamic_objs_mass,
        dynamic_objs_inertia = dynamic_objs_inertia,
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

def torch_get_from_1D(tensor: torch.tensor, index):
    '''
    Retrieves value from 1D tensor without upsetting jacrev or vmap:
    '''
    idx = torch.arange(tensor.shape[0])
    mask = (idx == index)
    return torch.where(mask, tensor, torch.zeros_like(tensor)).sum()

def torch_set_in_1D(tensor: torch.tensor, index, value):
    '''
    The same as the following: tensor[index] = value, but does so without upsetting jacrev/vmap
    '''

    idx = torch.arange(tensor.shape[0])
    mask = (idx == index)
    return torch.where(mask, value, tensor)


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
    if_inside_dist = torch.min(torch.stack([px-rx1, rx2-px, py-ry1, ry2-py]))
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

    last_len = cspace[params.max_bodies, -1]
    angles = cspace[:-1, -1]

    # Step 1: compute global angles for each segment center
    global_angle_full = heading0 + torch.cumsum(angles, 0)   # shape (n_bodies,)
    
    # Step 2: compute the center of each segment
    #   For the i-th segment, the center is offset from the anchor by
    #        sum_{k=0..i-1} [ L*cos(global_angle_full[k]), L*sin(global_angle_full[k]) ]
    #   But we can do that more efficiently. We'll build an array of cos/sin, then do a cumsum.

    # Cosines and sines of each segment angle:
    c_ = torch.cos(global_angle_full)
    s_ = torch.sin(global_angle_full)

    # Prepare the lengths of each segment
    part_full_lengths = torch.full((params.max_bodies,), params.body_length)
    # full_lengths[n_bodies-1] = last_len
    idx = torch.arange(params.max_bodies)
    full_lengths = torch.where(idx == (n_bodies - 1), last_len, params.body_length)

    # Now we do a cumulative sum of to get the tip coords of each segment
    tip_x = x0 + torch.cumsum(full_lengths * c_, 0)
    tip_y = y0 + torch.cumsum(full_lengths * s_, 0)
        
    # Now we'll compute the center of each segment, by
    # subtracting 0.5*full_lengths from the tip coords
    center_x = tip_x - 0.5 * full_lengths * c_
    center_y = tip_y - 0.5 * full_lengths * s_
    
    # Except, for the very last segment, the center position *is* the tip position
    # center_x[n_bodies-1] = tip_x[n_bodies-1]
    # center_y[n_bodies-1] = tip_y[n_bodies-1]
    #NOTE: everything below is just above, avoiding in-place assignment/retrieval

    idx2 = torch.arange(center_x.shape[0])
    mask = (idx2 == (n_bodies - 1))

    tip_x_val = torch_get_from_1D(tip_x, n_bodies-1)
    tip_y_val = torch_get_from_1D(tip_y, n_bodies-1)

    center_x =  torch.where(mask, tip_x_val, center_x)
    center_y = torch.where(mask, tip_y_val, center_y)

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
        dists = torch.vmap(torch_point_rect_sdf, in_dims=(None, None, 0))(px, py, torch.tensor(all_rects))

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
# Splitting Cone Solver: constraints and minimization measures 
# NOTE: most of the acutal params of really torch tensors (particularly non-static values);
#       use the given types to infer tensor shape
######################################################


def extend_cspace(params: VineParams, cspace: torch.tensor, dstate: torch.tensor, n_bodies:int):
    '''
    Extends the cspace by one, if needed
    (Mostly copied from DiffVine)
    '''

    new_i = n_bodies
    last_i = n_bodies - 1
    penult_i = n_bodies - 2

    # Indices for each one in (n, 3) cspace
    x = 0
    y = 1
    theta = 2

    # Compute position of second last seg
    endingx = cspace[penult_i, x] + params.radius * torch.cos(cspace[penult_i, theta])
    endingy = cspace[penult_i, y] + params.radius * torch.sin(cspace[penult_i, theta])   

    # Compute last body's distance

    part_distance = ((cspace[last_i, x] - endingx)**2 + \
                     (cspace[last_i, y] - endingy)**2).sqrt()
    # print(part_distance)

    if part_distance.dim() > 0:
        # last_link_distance = ((cspace[last_i, x] - endingx)**2 + \
        #                     (cspace[last_i, y] - endingy)**2).sqrt().squeeze(dim=-1)
        last_link_distance = part_distance.squeeze(dim=-1)
    else:
        last_link_distance = part_distance
    
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

    # Initialize corresponding d_state entry
    dstate[new_i, x] = 0
    dstate[new_i, y] = 0
    dstate[new_i, theta] = 0

    # Update n_bodies
    n_bodies = n_bodies + extend_needed

    return cspace, n_bodies


def cspace_sdf_measure(cspace: torch.tensor, 
                   params: VineParams, 
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


def dynamic_obj_sdf_measure(dynamic_obj_positions: torch.tensor, params: VineParams):
    '''
    Measure of how much each dynamic obstacle intersects
    any other obstacle (whether dynamic or static)

    #NOTE: assumes dynamic AND static objects to be rectangular
    '''
    
    def obj_to_corners(coords: torch.tensor # shape: (4,), just like how stored in env file 
                       ):
        '''
        Returns x,y's of given object's corners (matching indices => same corner across both arrays)
        '''

        left = torch_get_from_1D(coords, 0)
        top = torch_get_from_1D(coords, 1)
        right = torch_get_from_1D(coords, 2)
        bottom = torch_get_from_1D(coords, 3)

        xs = torch.stack([left, left, right, right])
        ys = torch.stack([top, bottom, top, bottom])

        return xs, ys

    def check_obj_collision(coords: tuple[4], all_rects, is_recursive_call:bool = False):
        '''
        Returns scalar for given dynamic obstacle;
        More negative => more collisions with static objs and/or other dynamic objs
        
        Done by checking how deep each obj's corner is within other objs' sdfs;
        Returns (4 * all_rects,) array, containing negative values where collisions were detected
        '''            

        def check_one_corner(x, y, all_rects):
            return torch.vmap(torch_point_rect_sdf, in_dims=(None, None, 0))(x, y, all_rects)

        # Check one way: given obj -> other objs
        xs, ys = obj_to_corners(coords)

        # for when doing "other objs -> current obj" (then all_rects is just coords of current obj)
        if all_rects.dim() == 1:
            dists = torch.vmap(torch_point_rect_sdf, in_dims=(0, 0, None))(xs, ys, all_rects)

        # otherwise, all_rects really is a list of objs
        else:
            dists = torch.vmap(check_one_corner, in_dims=(0, 0, None))(xs, ys, all_rects)
        
        dists = dists.flatten()
        dists = torch.where(dists > 0, 0, dists)

        if is_recursive_call:
            return dists 

        # Check other way: other objs -> given obj
        other_dists = torch.vmap(check_obj_collision, in_dims=(0, None, None))(all_rects, coords, True)
        other_dists = other_dists.flatten()
        other_dists = torch.where(other_dists > 0, 0, other_dists)

        return torch.sum(dists) + torch.sum(other_dists)
    
    all_rects = torch.cat([torch.tensor(params.obstacle_rects), dynamic_obj_positions], axis=0)

    dyn_obj_collision_measures = torch.vmap(check_obj_collision, in_dims=(0, None, None))(
        dynamic_obj_positions, all_rects, False
    )

    # shape: (# of dynamic objs,)
    return dyn_obj_collision_measures


def joint_measure(c_space: torch.tensor,
                  params: VineParams,
                  n_bodies: int,
                  x0: float, y0: float):
    '''
    Joints of the vine should be kept at fixed distance of each other
    '''


    # constraints = torch.zeros(n_bodies * 2)
    constraints = torch.zeros(params.max_bodies * 2)
    
    # xs = c_space[:n_bodies, 0]
    # ys = c_space[:n_bodies, 1]
    # thetas = c_space[:n_bodies, 2]

    idx = torch.arange(c_space.shape[0])
    mask = idx < n_bodies

    xs = torch.where(mask, c_space[:, 0], torch.zeros_like(c_space[:, 0]))
    ys = torch.where(mask, c_space[:, 1], torch.zeros_like(c_space[:, 1]))
    thetas = torch.where(mask, c_space[:, 2], torch.zeros_like(c_space[:, 2]))
    
    # constraints[0] = (xs[0] - x0) - params.radius * torch.cos(thetas[0])
    # constraints[1] = (ys[0] - y0) - params.radius * torch.sin(thetas[0])
    torch_set_in_1D(constraints, 0,
                    (torch_get_from_1D(xs, 0) - x0) - params.radius * \
                    torch.cos(torch_get_from_1D(thetas, 0)))
    torch_set_in_1D(constraints, 1,
                    (torch_get_from_1D(ys, 0) - x0) - params.radius * \
                    torch.sin(torch_get_from_1D(thetas, 0)))


    # constraints[2::2] = (xs[1:-1] - xs[:-2]) - params.radius * torch.cos(thetas[1:-1]) \
    #                                      - params.radius * torch.cos(thetas[:-2])

    
    x_diff_values = (xs[1:-1] - xs[:-2]) - params.radius * torch.cos(thetas[1:-1]) - params.radius * torch.cos(thetas[:-2])
    idx = torch.arange(constraints.shape[0])
    target_mask = (idx >= 2) & ((idx - 2) % 2 == 0)
    gather_pos = torch.clamp((idx - 2) // 2, 0, x_diff_values.shape[0] - 1)
    gathered = x_diff_values[gather_pos]

    constraints = torch.where(target_mask, gathered, constraints)

    # constraints[3::2] = (ys[1:-1] - ys[:-2]) - params.radius * torch.sin(thetas[1:-1]) \
    #                                      - params.radius * torch.sin(thetas[:-2])   

    y_diff_values = (ys[1:-1] - ys[:-2]) - params.radius * torch.sin(thetas[1:-1]) - params.radius * torch.sin(thetas[:-2])
    idx = torch.arange(constraints.shape[0])
    target_mask = (idx >= 3) & ((idx - 3) % 2 == 0)
    gather_pos = torch.clamp((idx - 3) // 2, 0, y_diff_values.shape[0] - 1)
    gathered = y_diff_values[gather_pos]

    constraints = torch.where(target_mask, gathered, constraints)

    #NOTE: do we also need to acknowledge special "last body", like in DiffVine?
    # endx = xs[n_bodies - 2] + params.radius * torch.cos(thetas[n_bodies - 2])
    # endy = ys[n_bodies - 2] + params.radius * torch.sin(thetas[n_bodies - 2])

    endx = torch_get_from_1D(xs, n_bodies - 2) + params.radius * torch.cos(torch_get_from_1D(thetas, n_bodies - 2))
    endy = torch_get_from_1D(ys, n_bodies - 2) + params.radius * torch.sin(torch_get_from_1D(thetas, n_bodies - 2))

    # angle_diff = torch.atan2(ys[n_bodies - 1] - endy, xs[n_bodies - 1] - endx - thetas[n_bodies - 1])

    angle_diff = torch.atan2(torch_get_from_1D(ys, n_bodies-1) - endy, 
                             torch_get_from_1D(xs, n_bodies-1) - endx - torch_get_from_1D(thetas, n_bodies-1))

    # constraints[(n_bodies - 1) * 2] = angle_diff   

    torch_set_in_1D(constraints, (n_bodies - 1) * 2, angle_diff) 

    #NOTE: is zero_out required here like from DiffVine?

    #Create mask here to eliminate all calculations that go past N_BODIES

    idx = torch.arange(constraints.shape[0])
    mask = (idx // 2) < n_bodies

    constraints = torch.where(mask, constraints, 0)

    return constraints


def proximity_measure(cspace: torch.tensor,
                      dynamic_obj_positions: torch.tensor,
                      params: VineParams,
                      n_bodies: int,
                      x0: float, y0: float, heading0: float):
    '''
    The distance between any joint to any dynamic obj > 0 to prevent intersection

    # NOTE: assumes that dynamic objects are rectangular
    '''
    joint_centers = torch_cspace_to_positions(params, cspace, n_bodies, x0, y0, heading0)
    joint_radius = params.radius

    def get_overlap_measure(joint_center_coords: tuple[2], joint_radius: float,
                        dynamic_obj_position: tuple[4]):
        '''
        Checks to see if given joint is in collision with given dynamic object;
        Positive value indicates they're not in collision, 
        Negative value indicates otherwise
        '''
        center_x, center_y = joint_center_coords
        rect_left, rect_top, rect_right, rect_bottom = dynamic_obj_position

        # Find point on rectangle closest to joint center
        closest_x = torch.clamp(center_x, rect_left, rect_right)
        closest_y = torch.clamp(center_y, rect_bottom, rect_top)

        # Depth of rectangle's penetration:
        dist = torch.sqrt(torch.pow(closest_x - center_x, 2) + torch.pow(closest_y - center_y, 2))
        depth = torch.tensor(joint_radius) - dist

        return depth * -1 # flip sign so that penetration is negative
    
    def overlap_over_all_objects(joint_center_coords: tuple[2], joint_radius: float,
                                 dynamic_obj_positions: torch.tensor):
        '''
        V-mapped get_overlap_measure: get the depths of every dynamic obj
        in respect to the given joint,
        return result shaped (len(dynamic_obj_positions),)
        '''
        joint_depths = torch.vmap(get_overlap_measure, in_dims=(None, None, 0))(
                            joint_center_coords, joint_radius, dynamic_obj_positions)
        return joint_depths

    all_joint_depths = torch.vmap(overlap_over_all_objects, in_dims=(0, None, None))(
        joint_centers, joint_radius, dynamic_obj_positions
    )

    # shape: (# n_bodies, # dynamic objects)
    return all_joint_depths


def growth_measure(cspace: torch.tensor,
                   dstate: torch.tensor, params: VineParams, n_bodies: int):
    '''
    Measure of how much the current growing segment is growing;
    Growth should be constrained such that the current segment should always be growing
    each time step
    '''

    def torch_get_from_2D(tensor: torch.tensor, row_idx, col_idx):
        '''
        Similar to torch_get_from_1D, but for 2D tensors
        '''
        idx = torch.arange(tensor.shape[0])
        mask = (idx == row_idx)
        return torch.where(mask, tensor[:, col_idx], torch.zeros_like(tensor[:, col_idx])).sum()

    curr_id = n_bodies - 1
    prev_id = n_bodies - 2

    # Current growing segment info:
    # curr_x = cspace[curr_id, 0]
    # curr_y = cspace[curr_id, 1]
    # curr_velocity_x = dstate[curr_id, 0]
    # curr_velocity_y = dstate[curr_id, 1]

    curr_x = torch_get_from_2D(cspace, curr_id, 0)
    curr_y = torch_get_from_2D(cspace, curr_id, 1)
    curr_velocity_x = torch_get_from_2D(dstate, curr_id, 0)
    curr_velocity_y = torch_get_from_2D(dstate, curr_id, 1)

    # Previous segment info:
    # prev_x = cspace[prev_id, 0]
    # prev_y = cspace[prev_id, 1]
    # prev_velocity_x = dstate[prev_id, 0]
    # prev_velocity_y = dstate[prev_id, 1]
    
    prev_x = torch_get_from_2D(cspace, prev_id, 0)
    prev_y = torch_get_from_2D(cspace, prev_id, 1)
    prev_velocity_x = torch_get_from_2D(dstate, prev_id, 0)
    prev_velocity_y = torch_get_from_2D(dstate, prev_id, 1)

    # Derivative of the distance in respect to time:
    growth = ((curr_x - prev_x) * (curr_velocity_x - prev_velocity_x) + 
              (curr_y - prev_y) * (curr_velocity_y - prev_velocity_y)) / \
              torch.sqrt(torch.pow(curr_x - prev_x, 2) + torch.pow(curr_y - prev_y, 2))
    return growth


def get_bending_energy(params: VineParams, cspace: torch.tensor, bend_params: torch.tensor, bend_energy_func: Callable):

    #NOTE: vine stiffness and damping are NOT utilized

    turning_radius = torch.where(torch.abs(cspace[:-1, -1]) < 1e-3, 0, params.body_length * 1e-3 / cspace[:-1, -1])
    bend_moment = -1 * bend_energy_func(turning_radius, bend_params[:, 0], bend_params[:, 1])

    # Pad updates to forces vector works:
    bend_moment = torch.nn.functional.pad(bend_moment, (0, 1 + params.dynamic_objs_mass.shape[0]))    
    return bend_moment


def get_object_motion(params: VineParams, dstate: torch.tensor, weight: float):
    '''
    Returns (len(dynamic_objects), ) shaped tensor;
    Calculates modified KE for each dynamic object 
    '''
    #NOTE: in forces, each object has (change x, change y, change theta)

    #NOTE: weight is currently random and can be changed
    def compute_KE(weight: float, obj_mass: float, obj_velocities: tuple[3]):
       

        obj_mass = obj_mass.squeeze()
    
        return torch.stack([
            # weight * obj_mass * obj_velocities[0], # x
            # weight * obj_mass * obj_velocities[1], # y
            # weight * obj_mass * obj_velocities[2], # theta

            weight * obj_mass * torch_get_from_1D(obj_velocities, 0), # x
            weight * obj_mass * torch_get_from_1D(obj_velocities, 1), # y
            weight * obj_mass * torch_get_from_1D(obj_velocities, 2)  # theta
        ])


    # Pad the mass tensor so that it's the same length as the dstate
    dyn_mass_tensor = torch.tensor(params.dynamic_objs_mass)

    padded_tensor = torch.nn.functional.pad(dyn_mass_tensor, (0, 0, params.max_bodies + 1, 0), value=0.00)
    
    # Shape: (params.max_bodies + 1 + # dynamic objs, 3);
    # note that any parts not corresponding to dynamic objs all be 0
    return torch.vmap(compute_KE, in_dims=(None, 0, 0))(weight, padded_tensor, dstate)


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

    #NOTE: zero_out was not used in some parts, unlike DiffVine; might need to look into this

    cspace, n_bodies = extend_cspace(params, cspace, dstate, n_bodies)

    # print("\nIn Compute Jacobians:")
    # print(cspace.shape)

    cspace_sdf_jac = torch.func.jacrev(partial(cspace_sdf_measure, params=params,
                                                    x0=x0, y0=y0, n_bodies=n_bodies, heading0=heading0)
                                            )(cspace)
    cspace_sdf_now = cspace_sdf_measure(cspace, params, n_bodies, x0, y0, heading0)

    dynamic_sdf_jac = torch.func.jacrev(partial(dynamic_obj_sdf_measure, params=params))(dynamic_obj_positions)
    dynamic_sdf_now = dynamic_obj_sdf_measure(dynamic_obj_positions, params)

    joint_jac = torch.func.jacrev(partial(joint_measure, params=params, n_bodies=n_bodies,
                                          x0=x0, y0=y0))(cspace)
    joint_now = joint_measure(cspace, params, n_bodies, x0, y0)

    proximity_jac = torch.func.jacrev(partial(proximity_measure, params=params,
                                              n_bodies=n_bodies, x0=x0, y0=y0, heading0=heading0))(
                                                  cspace, dynamic_obj_positions
                                              )
    proximity_now = proximity_measure(cspace, dynamic_obj_positions, params, n_bodies,
                                      x0, y0, heading0)
    
    growth_wrt_state, growth_wrt_dstate = torch.func.jacrev(partial(growth_measure, params=params, n_bodies=n_bodies),
                                         argnums=(0, 1))(
        cspace, dstate
    )
    growth_now = growth_measure(cspace, dstate, params, n_bodies)

    # Find bend energy to minimize for the vine:
    bend_energy = get_bending_energy(params, cspace, bend_params, bend_energy_func)

    # Find motion info for dynamic objects:
    
    kinetic_energy_weight = 0.5
    obj_motion = get_object_motion(params, dstate, kinetic_energy_weight)

    #NOTE: forces shape: (cpsace + dyn_obj len, 3)

    forces = torch.zeros(cspace.shape[0] + dynamic_obj_positions.shape[0], 3)
    start_obj_idx = n_bodies + 1

    # forces[:n_bodies, 2] += -bend_energy
    # forces[:n_bodies - 1, 2] += bend_energy[1:]

    # forces[:n_bodies, 0] += dstate[:n_bodies, 0] # for x 
    # forces[:n_bodies, 1] += dstate[:n_bodies, 1] # for y

    idx = torch.arange(forces.shape[0])
    mask_n = (idx < n_bodies)
    mask_n1 = (idx < n_bodies - 1)

    left_shifted_BE = torch.nn.functional.pad(bend_energy[1:], (0,1)) # to drop the first element in BE[1:]

    dstate_x_update = mask_n * dstate[:, 0]
    dstate_y_update = mask_n * dstate[:, 1]
    BE_update = mask_n * -bend_energy + mask_n1 * left_shifted_BE

    updates = torch.stack([dstate_x_update, 
                           dstate_y_update,
                           BE_update], dim=1)

    forces = forces + updates

    # Add on movement energy of the objects for minimization        
    #FIXME: add ability to read in MASS and INERTIA for each dynamic object

    # forces[start_obj_idx:, :] += obj_motion[:, :]
    
    # print(forces.shape, obj_motion.shape)
    forces += obj_motion

    return n_bodies, forces, cspace_sdf_jac, cspace_sdf_now, dynamic_sdf_jac, dynamic_sdf_now, \
            joint_jac, joint_now, proximity_jac, proximity_now, \
            growth_wrt_state, growth_wrt_dstate, growth_now

# compute_jacobians_batched = torch.func.vmap(compute_jacobians, in_dims = (None, 0, 0, 0, None, 0, 0, 0, None, None))


######################################################
# Splitting Cone Solver: the actual solver
######################################################

def create_mass_matrix(params: VineParams, vine_inertia_weight: float, obj_inertia_weight: float):
    '''
    Creates combined mass/inertia matrix (stores mass/inertia for both vine and dynamic objs);
    The idea is the same as in DiffVine:
    Each vine body/dynamic obj's x, y is multiplied by the corresponding mass,
    each of their theta multiplied by the corresponding inertia

    The result is a (max_bodies + 1 + # dynamic_objects, max_bodies + 1 + # dynamic_objects)
    diagonal matrix.
    '''

    vine_mass = torch.tensor(params.mass)
    vine_inertia = torch.tensor(params.inertia)
    max_bodies = params.max_bodies

    objs_mass = torch.tensor(params.dynamic_objs_mass).squeeze(0)
    objs_inertia = torch.tensor(params.dynamic_objs_inertia).squeeze(0)
    num_objs = params.dynamic_objs_mass.size

    # Create mass matrix for vine:
    vine_diag_elements = torch.cat([vine_mass, vine_mass, 
                                    vine_inertia * vine_inertia_weight]).repeat(max_bodies + 1)

    concatentated_tensor = torch.cat([objs_mass, objs_mass, objs_inertia * obj_inertia_weight])
    obj_diag_elements = concatentated_tensor.repeat(num_objs)
    
    all_diag_elements = torch.cat([vine_diag_elements, obj_diag_elements])

    # returns square matrix with all elements in this array down its diagonal:
    return torch.diag(all_diag_elements) 


# Something used within solve_layers:
import scipy.linalg
from torch.autograd import Function

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.detach().cpu().numpy().astype(np.float_)
            gm = grad_output.detach().cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

sqrtm_module = MatrixSquareRoot()

def SCS_solve_layers(mass_matrix, forces_info, inequality_step, inequality_now,
                     equality_step, equality_now):
    '''
    Batched QP solve
    (Newly defined args are for SCS, not cvxpy)
    '''
    global cvxpylayer
    global sqrtm_module

    batch_size = forces_info.shape[0]
    mass_matrix_batched = sqrtm_module.apply(mass_matrix).unsqueeze(0).expand(batch_size, -1, -1)

    solver_args_scs = {'acceleration_lookback': 40_000, 'verbose': False, 'max_iters': 10_000}

    solution = cvxpylayer(mass_matrix_batched, forces_info, inequality_step, inequality_now,
                          equality_step, equality_now, solver_args = solver_args_scs)
    
    return solution[0]


def SCS_solve(params: VineParams, dstate, forces,
              cspace_sdf_jac, cspace_sdf_now,
              dynamic_sdf_jac, dynamic_sdf_now,
              joint_jac, joint_now,
              proximity_jac, proximity_now,
              growth_wrt_state, growth_wrt_dstate, growth_now):
    
    '''
    Also batched: combines batched init_layers with batched solve_layers
    '''

    global cvxpylayer

    # Configure constraints and matrices to be used by the solver

    solution_size = (dstate.shape[0], dstate.shape[1]) # returns next_dstate, so keep the same shape (without batched part)
    dt = params.dt

    vine_inertia_weight = 100
    obj_inertia_weight = 100
    mass_inertia_diag = create_mass_matrix(params, vine_inertia_weight, obj_inertia_weight)

    # Transform force infomation (what to mimimize: bending and KE)

    # Last term: basically multiply every (x, y, theta) with corresponding (mass, mass, inertia)
    # Do this for every vine body and every dynamic object (this also requires some reshaping)
    forces_info = forces * dt - \
                  (dstate.flatten(start_dim=1) @ mass_inertia_diag).reshape(
                      dstate.shape[0], dstate.shape[1], 3)
    
    # Combined Inequality constraints: cspace_sdf, dynamic_sdf, proximity
    cspace_sdf_step = -1 * cspace_sdf_jac * dt
    dynamic_sdf_step = -1 * dynamic_sdf_jac * dt
    proximity_step = -1 * proximity_jac * dt

    # Create diag-matrix so each batch can store info on cspace, dynamic, proximity info

    print("\nIn Solve Step:")
    print(cspace_sdf_step.shape)
    print(cspace_sdf_now.shape)

    print()
    print(dynamic_sdf_step.shape)
    print(dynamic_sdf_now.shape)

    print()
    print(proximity_step.shape)
    print(proximity_now.shape)

    inequality_step = torch.vmap(torch.block_diag, in_dims=(0, 0, 0)) \
                        (cspace_sdf_step, dynamic_sdf_step, proximity_step)
    inequality_now = torch.vmap(torch.block_diag, in_dims=(0, 0, 0)) \
                        (cspace_sdf_now, dynamic_sdf_now, proximity_now)
    
    # Combined Equality constraints: growth, joint
    # NOTE: for now, is copied from DiffVine

    growth_constraint = (
        growth_now.squeeze(1) - 1000 * params.grow_rate -
        torch.bmm(growth_wrt_dstate, dstate.unsqeeze(2)).squeeze(2).squeeze(1)
    )
    growth_coeff = (growth_wrt_state * dt + growth_wrt_dstate)

    equality_step = torch.cat([joint_jac * dt, growth_coeff], dim = 1)
    equality_now = torch.cat([-joint_now, -growth_constraint.unsqueeze(1)], dim = 1)

    # Initialize layers, if need be:
    if cvxpylayer is None:
        next_dstate = cp.Variable(solution_size)

        mass_matrix_sqrt = cp.Parameter(mass_inertia_diag.shape)
        forces_param = cp.Parameter(forces_info.shape[1:])
        inequality_step_param = cp.Parameter(inequality_step.shape[1:])
        inequality_now_param = cp.Parameter(inequality_now.shape[1:])
        equality_step_param = cp.Parameter(equality_step.shape[1:])
        equality_now_param = cp.Parameter(equality_now.shape[1:])

        objective = cp.Minimize(0.5 * cp.sum_squares(mass_matrix_sqrt @ next_dstate) + 
                                forces_param @ next_dstate)
        
        constraints = [equality_step_param @ next_dstate == equality_now_param,
                       inequality_step_param @ next_dstate <= inequality_now_param]
        
        problem = cp.Problem(objective, constraints)
        
        cvxpylayer = CvxpyLayer(problem, parameters = [
            mass_matrix_sqrt, forces_param, inequality_step_param, inequality_now_param,
            equality_step_param, equality_now_param
        ], variables = [next_dstate])

    next_dstate_solution = SCS_solve_layers(mass_inertia_diag, forces_info, 
                                            inequality_step, inequality_now, 
                                            equality_step, equality_now)
    return next_dstate_solution


######################################################
# New simulation advance, using Splitting Cone Solver
######################################################

def SCS_step_vine(params: VineParams, cspace: np.array, dstate: np.array,
                  dynamic_obj_positions: np.array, n_bodies: np.array, 
                  bend_params: np.array, x0: float, y0: float, heading0: float, 
                  bend_energy_func: Callable):
    
    #FIXME: change how batches are handled to resemble DiffVine:
    # 1. compute jacobians: VMAPPED over batches to get info
    # 2. SCS_solve: takes all batched data and returns next_dstate_solution
    # Means that this function DOES NOT NEED TO BE BATCHED

    cspace = torch.tensor(cspace)
    dstate = torch.tensor(dstate)
    n_bodies = torch.tensor(n_bodies)
    dynamic_obj_positions = torch.tensor(dynamic_obj_positions)
    bend_params = torch.tensor(bend_params)

    new_n_bodies, forces, cspace_sdf_jac, cspace_sdf_now, dynamic_sdf_jac, dynamic_sdf_now, \
    joint_jac, joint_now, proximity_jac, proximity_now, \
    growth_wrt_state, growth_wrt_dstate, growth_now = \
    torch.vmap(compute_jacobians, in_dims=(None, 0, 0, 0, 0, None, None, None, 0, None)) \
                             (params, cspace, dstate, dynamic_obj_positions, n_bodies,
                              x0, y0, heading0, bend_params, bend_energy_func)
    
    next_dstate_solution = SCS_solve(params, dstate, forces, cspace_sdf_jac, cspace_sdf_now,
                                     dynamic_sdf_jac, dynamic_sdf_now, joint_jac,
                                     joint_now, proximity_jac, proximity_now,
                                     growth_wrt_state, growth_wrt_dstate, growth_now)
    
    # Use found dstate to find new positions:

    # Potential FIXME: why is detach() being used here in DiffVine?
    # Look into how this approach is differentiable with cvxpylayer:
    # what exactly are we training on?
    new_cspace = cspace + next_dstate_solution[:, :params.max_bodies - 1, :].detach()
    new_dynamic_obj_positions = dynamic_obj_positions + next_dstate_solution[:, params.max_bodies + 1:, :].detach()

    return new_cspace, new_n_bodies, new_dynamic_obj_positions, next_dstate_solution


# def SCS_step_vine_batched(params: VineParams, dstates: torch.tensor, cspaces: torch.tensor, dynamic_positions: torch.tensor,
#                           n_bodies_list: torch.tensor, bend_params: torch.tensor,
#                           x0_list: torch.tensor, y0_list: torch.tensor, heading0_list: torch.tensor,
#                           bend_energy_func: Callable):
#     '''
#     Batched SCS_step_vine
#     '''

#     # Convert to tensors first because state-tree uses numpy arrays:
#     cspaces = torch.tensor(cspaces)
#     dstates = torch.tensor(dstates)
#     dynamic_positions = torch.tensor(dynamic_positions)
#     n_bodies_list = torch.tensor(n_bodies_list)
#     bend_params = torch.tensor(bend_params)
    
#     new_cspaces, new_n_bodies, new_dynamic_positions, next_dstate_solution = \
#                                                 torch.vmap(SCS_step_vine, in_dims=(None, 0, 0, 0, 0, 0, None, None, None, None)) \
#                                                 (params, cspaces, dstates, dynamic_positions, n_bodies_list, bend_params, 
#                                                  x0_list, y0_list, heading0_list, bend_energy_func)
#     return new_cspaces.numpy(), new_n_bodies.numpy(), \
#            new_dynamic_positions.numpy(), next_dstate_solution.numpy()

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
