#--------------------------------------- imports
import argparse
import hashlib
from math import ceil
import os
import random
import time
from typing import List, Tuple
from queue import PriorityQueue

# import jax
# import jax.numpy as jnp
import torch
if torch.cuda.is_available():
    torch.set_default_device('cuda')

from kinodynamic.env_loader import load_box_config
import numpy as np

from render import *
from kinodynamic.max_cover import max_cover
from .vine import VineParams

from geometric.biarc_rrtstar import main as geometric_plan
from kinodynamic.nearest import distance, nearest_neighbor, nearest_neighbor_all

from .dynamic_vine import step as forward

from sPAM.torch_nns_usage import torch_solve as find_actuator_params
from sPAM.spam import params as act_params
from sPAM.torch_nns import get_or_train_model, get_prediction_function

#------------------------------------------ global defs (variable/function defs)

tiebreak_factor = 0.0001
bending_controls_size = 2

find_actuator_params = torch.vmap(find_actuator_params, in_dims=(None, None, 0))

scaling_info, model = get_or_train_model(act_params)
predict = get_prediction_function(scaling_info, model)

#------------------------------------------ SSTparams, DontCompareSecond, and StatesScruct defs
class SSTparams:
    # Contains δBN, δs, min/max 2D sampling
    def __init__(self, 
                 batch_size: int, 
                 δBN: float, 
                 δs: float, 
                 min_x: float, max_x: float, min_y: float, max_y: float, 
                 start, goal, goal_radius,
                 points, point_costs,
                 geo_cost_to_go_weight = 0.2,
                 info = {},
                 do_cost_to_go=True,
                 do_maximal=True,
                 do_set_cover=True,
                 time_to_evolve=100.0,
                 record_every_multiplier=1
                ):

        # Batch size for all SST
        self.batch_size = batch_size
        
        # Scale for how many distance units are equal to one radian of rotation
        # Increase to make SST more strict about needing similar angles
        # Used as metric in nearest neighbors lookup.
        self.rotation_metric_scale = 10
                
        # Radius for best-nearest search, for selecting which node to expand
        self.δBN = δBN 
        # Radius for local best search, for pruning dominated nodes
        self.δs = δs 
        
        # δBN + 2 * δs will be the clearance radius of the plan
        
        # Bounding box for the 2D space
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        self.start = start
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        
        # A list of discrete tip positions and their cost (num curves)
        # to get to the goal
        self.points = points
        self.point_costs = point_costs
        
        # Weightage for the cost-to-go from the geometric solution, used
        # in finding total heuristic cost
        self.geo_cost_to_go_weight = geo_cost_to_go_weight
        
        # Store final states in goal
        # Each entry is a dict {cspace, bodies, bending_control, cost_to_come, cost_total, tip}
        # Yeah params isn't the right place for this but it's good enough
        self.solutions = PriorityQueue()
        
        # Just for info, not used during planning
        self.info = info
       
        self.do_cost_to_go = do_cost_to_go 
        
        # When True; rollout as far as feasible, then use all timesteps as possible
        # new candidates
        # When False; just pick one random timestep 
        self.do_maximal = do_maximal
        
        # When True: Make sure the newly sampled set does not collide
        # When False, all sampled together at once
        self.do_set_cover = do_set_cover
        
        self.time_to_evolve = time_to_evolve
        
        # Fudge factor if you want to record more/less than the heuristic amount
        self.record_every_multiplier = record_every_multiplier
        
        if self.points is not None:
            assert self.points.shape[0] == self.point_costs.shape[0], f"points shape: {self.points.shape}, points_costs shape: {self.point_costs.shape}"


class DontCompareSecond():
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    def __lt__(self, other):
        return self.first < other.first


class StatesStruct:
    '''
    Structure-of-Lists for holding States and Witnesses. Nothing is removed, so indexes to items are stable.
    Expands capacity automatically when needed, like a vector.
    Use the function call (ie isactive()) to get the data, truncated to the last valid item.

    NOTE: on representation
    - dynamic object pose: (init_size, num objs, [x1, y1, x2, y2, x3, y3, x4, y4])
    - dynamic object dstate: (init_size, num objs, [dx, dy, dtheta])
    - vine cspace: (init_size, max_bodies * [x, y, theta])
    - vine dstate: (init_size, max_bodies * [dx, dy, dtheta])
    '''
    
    def __init__(self, max_bodies, num_dynamic_objs):
        self.init_size = 64
        self.max_bodies = max_bodies
        self.num_dynamic_objs = num_dynamic_objs
        
        # --------- States ---------
        self.num_states = 0
        
        self._isactive = np.zeros((self.init_size), dtype=bool)
        self._kil = np.zeros((self.init_size), dtype=bool)

        self._bodies = np.zeros((self.init_size), dtype=np.int32)
        self._times = np.zeros((self.init_size), dtype=np.float32)
        self._bending_controls = np.zeros((self.init_size, max_bodies, bending_controls_size), dtype=np.float32)

        self._c_spaces = np.zeros((self.init_size, max_bodies * 3), dtype=np.float32)
        self._obj_positions = np.zeros((self.init_size, num_dynamic_objs, 4), 
                                           dtype=np.float32)
        
        self._dstates = np.zeros((self.init_size, max_bodies * 3), dtype=np.float32)
        self._obj_dstates = np.zeros((self.init_size, num_dynamic_objs, 3), dtype=np.float32)

        # Heuristic stuff
        self._cost_to_come = np.zeros((self.init_size), dtype=np.int32) # Cost to come
        self._cost_total = np.zeros((self.init_size), dtype=np.int32)
        self._tips = np.zeros((self.init_size, 3), dtype=np.float32)
        
        # Tree stuff
        self._parent_idxs = np.zeros((self.init_size), dtype=np.int32)
        self._num_children = np.zeros((self.init_size), dtype=np.int32)
        
        # --------- Witnesses ---------
        self.num_witnesses = 0
        self._witness_positions = np.zeros((self.init_size, 3), dtype=np.float32)
        # Reps are idxs into the states list
        self._rep_idxs = np.full((self.init_size), -1, dtype=np.int32)
    
    def isactive(self): return self._isactive[:self.num_states]
    def bodies(self): return self._bodies[:self.num_states]
    def times(self): return self._times[:self.num_states]
    def bending_controls(self): return self._bending_controls[:self.num_states]

    def c_spaces(self): return self._c_spaces[:self.num_states]
    def obj_positions(self): return self.obj_positions[:self.num_states]
    
    def dstates(self): return self._dstates[:self.num_states]
    def obj_dstates(self): return self._obj_dstates[:self.num_states]
    
    def cost_to_come(self): return self._cost_to_come[:self.num_states]
    def tip(self): return self._tips[:self.num_states]
    
    def parent_idxs(self): return self._parent_idxs[:self.num_states]
    def num_children(self): return self._num_children[:self.num_states]
    
    def witness_positions(self): return self._witness_positions[:self.num_witnesses]
    def rep_idxs(self): return self._rep_idxs[:self.num_witnesses]
    
    def extend_states(self):
        # Double the size of the arrays
        current_size = self._c_spaces.shape[0]
        
        self._isactive = np.concatenate([self._isactive, np.zeros(current_size, dtype=bool)], axis=0)
        self._kil = np.concatenate([self._kil, np.zeros(current_size, dtype=bool)], axis=0)
        
        
        self._bodies = np.concatenate([self._bodies, np.zeros(current_size, dtype=np.int32)], axis=0)
        self._times = np.concatenate([self._times, np.zeros(current_size, dtype=np.float32)], axis=0)
        self._bending_controls = np.concatenate([self._bending_controls, np.zeros((current_size, self.max_bodies, bending_controls_size), dtype=np.float32)], axis=0)

        self._c_spaces = np.concatenate([self._c_spaces, np.zeros((current_size, self.max_bodies * 3), dtype=np.float32)], axis=0)
        self._obj_positions = np.concatenate([self._obj_positions, np.zeros((current_size, self.num_dynamic_objs, 4), dtype=np.float32)], 
                                                 axis=0)
        self._dstates = np.concatenate([self._dstates, np.zeros((current_size, self.max_bodies * 3), dtype=np.float32)], axix=0)
        self._obj_dstates = np.concatenate([self._obj_dstates, np.zeros((current_size, self.num_dynamic_objs, 3), dtype=np.float32)],
                                           axis=0)

        self._cost_to_come = np.concatenate([self._cost_to_come, np.zeros(current_size, dtype=np.int32)], axis=0)
        self._cost_total = np.concatenate([self._cost_total, np.zeros(current_size, dtype=np.int32)], axis=0)
        self._tips = np.concatenate([self._tips, np.zeros((current_size, 3), dtype=np.float32)], axis=0)
                
        self._parent_idxs = np.concatenate([self._parent_idxs, np.zeros(current_size, dtype=np.int32)], axis=0)
        self._num_children = np.concatenate([self._num_children, np.zeros(current_size, dtype=np.int32)], axis=0)
                
    def add_state(self, isactive, c_space, dstate, obj_positions, obj_dstate, 
                  bodies, time, bending_control, cost_to_come, cost_total, tip, parent_idx, num_children):
        if self.num_states == self._c_spaces.shape[0]:
            self.extend_states()
        
        # Convert to cpu tensor before
        if torch.cuda.is_available():
            tip = tip.cpu().numpy()

        idx = self.num_states
        
        self._isactive[idx] = isactive
        
        
        self._bodies[idx] = bodies
        self._times[idx] = time
        self._bending_controls[idx] = bending_control

        self._c_spaces[idx] = c_space
        self._dstates[idx] = dstate

        if self.num_dynamic_objs != 0:
            self._obj_positions[idx] = obj_positions
            self._obj_dstates[idx] = obj_dstate
                
        self._cost_to_come[idx] = cost_to_come
        self._cost_total[idx] = cost_total
        self._tips[idx] = tip
                
        self._parent_idxs[idx] = parent_idx
        self._num_children[idx] = num_children
        
        self.num_states += 1
        
        return idx
    
    def add_states(self, isactive, c_space, dstate, obj_positions, obj_dstates,
                   bodies, time, bending_control, cost_to_come, cost_total, tip, parent_idx, num_children):
        num_to_add = c_space.shape[0]
        
        assert bodies.shape == (num_to_add,)
        assert time.shape == (num_to_add,)
        assert bending_control.shape == (num_to_add, self.max_bodies, 2)

        assert c_space.shape == (num_to_add, self.max_bodies * 3)
        assert dstate.shape == (num_to_add, self.max_bodies * 3)

        if self.num_dynamic_objs != 0:
            assert obj_positions.shape == (num_to_add, self.num_dynamic_objs, 4)
            assert obj_dstates.shape(num_to_add, self.num_dynamic_objs, 3)
        
        assert cost_to_come.shape == (num_to_add,)
        assert cost_total.shape == (num_to_add,)
        assert tip.shape == (num_to_add, 3)
        
        assert parent_idx.shape == (num_to_add,) 
        
        while self.num_states + num_to_add >= self._c_spaces.shape[0]:
            self.extend_states()
            
        to_add_slice = slice(self.num_states, self.num_states + num_to_add)
        
        self._isactive[to_add_slice] = isactive
        self._kil[to_add_slice] = False
        
        
        self._bodies[to_add_slice] = bodies
        self._times[to_add_slice] = time
        self._bending_controls[to_add_slice] = bending_control

        self._c_spaces[to_add_slice] = c_space
        self._dstates[to_add_slice] = dstate

        if self.num_dynamic_objs != 0:
            self._obj_positions[to_add_slice] = obj_positions
            self._obj_dstates[to_add_slice] = obj_dstates
        
        self._cost_to_come[to_add_slice] = cost_to_come
        self._cost_total[to_add_slice] = cost_total
        self._tips[to_add_slice] = tip
        
        self._parent_idxs[to_add_slice] = parent_idx
        self._num_children[to_add_slice] = num_children
        
        self.num_states += num_to_add
        
        return np.arange(to_add_slice.start, to_add_slice.stop, dtype=np.int32)
    
    def clean_states(self):
        """
        Remove all states with _kil == True
        Then update the _parent_idxs and _rep_idxs indices
        """
        
        keep_states = ~self._kil
        num_states_to_keep = np.sum(keep_states)
        
        # Build a map from old idx -> new idx
        old_to_new = np.full(keep_states.shape[0], dtype=np.int32, fill_value=-1)
        
        old_to_new[keep_states] = np.arange(num_states_to_keep)
        
        # Update the states
        self._isactive = self._isactive[keep_states]
        self._kil = self._kil[keep_states]
        
        
        self._bodies = self._bodies[keep_states]
        self._times = self._times[keep_states]
        self._bending_controls = self._bending_controls[keep_states]

        self._c_spaces = self._c_spaces[keep_states]
        self._dstates = self._dstates[keep_states]        
        self._objs_positions = self._obj_positions[keep_states]
        self._obj_dstates = self._obj_dstates[keep_states]

        self._cost_to_come = self._cost_to_come[keep_states]
        self._cost_total = self._cost_total[keep_states]
        self._tips = self._tips[keep_states]
        
        # Update parent indices
        self._parent_idxs = old_to_new[self._parent_idxs[keep_states]]
        self._num_children = self._num_children[keep_states]
        
        self.num_states = num_states_to_keep
        
        # Witnesses
        # If a rep_idx references a deleted state, then old_to_new will be -1
        self._rep_idxs = old_to_new[self._rep_idxs]
                
    def extend_witnesses(self):
        # Double the size of the arrays
        current_size = self._witness_positions.shape[0]
        
        self._witness_positions = np.concatenate([self._witness_positions, np.zeros((current_size, 3), dtype=np.float32)], axis=0)
        self._rep_idxs = np.concatenate([self._rep_idxs, np.zeros(current_size, dtype=np.int32)], axis=0)
        
    def add_witness(self, tip_pos, rep_idx):
        assert tip_pos.shape == (3,)
        assert rep_idx >= 0
        
        if self.num_witnesses == self._witness_positions.shape[0]:
            self.extend_witnesses()
        
        idx = self.num_witnesses
        self._witness_positions[idx] = tip_pos
        self._rep_idxs[idx] = rep_idx
        
        self.num_witnesses += 1
        
        return idx
    
    def add_witnesses(self, tip_pos, rep_idx):
        num_to_add = len(tip_pos)
        
        assert tip_pos.shape == (num_to_add, 3)
        assert rep_idx.shape == (num_to_add,)
        
        while self.num_witnesses + num_to_add >= self._witness_positions.shape[0]:
            self.extend_witnesses()
            
        to_add_slice = slice(self.num_witnesses, self.num_witnesses + num_to_add)
        
        self._witness_positions[to_add_slice] = tip_pos
        self._rep_idxs[to_add_slice] = rep_idx
        
        self.num_witnesses += num_to_add
        
        return np.arange(to_add_slice.start, to_add_slice.stop, dtype=np.int32)

#------------------------------------------ sst helpers (not including rollout)
'''
NOTE's:
- assumed params.body_length == params.half_len * 2
- for finding the length for the last body for cspace_to_tip,
  simple geometric distance formula was used between x,y's of last and second to last body
'''

def get_last_body_length(cspace: np.ndarray, n_bodies: int):
    '''
    New method for getting length of last body now that cspace is shaped (b, max_bodies * 3)
    NOTE: this is NOT batched, needs to be vmapped
    '''

    last = n_bodies - 1
    prev_last = n_bodies - 2

    last_x = cspace[:, last * 3 + 0]
    prev_x = cspace[:, prev_last * 3 + 0]

    last_y = cspace[:, last * 3 + 1]
    prev_y = cspace[:, prev_last * 3 + 1]

    return torch.sqrt((last_x - prev_x).pow(2) + (last_y - prev_y).pow(2))


def cspace_to_tip(params: VineParams, batch_size, cspace: np.ndarray, 
                       n_bodies: int, 
                       x0: float, y0: float, heading0: float):
    """
    Convert a batch of c-space -> global center coordinates of tip
      cspace has shape (batch, N+1,) but we only use the first n_bodies angles (plus last_length).
    We also incorporate an initial anchor (x0, y0) and heading0 for the first segment.

    Returns:
        Shape (batch, 3) with the tip coordinates (x, y, theta)

    FIXME: before changing this to be compatible with new cspace representation, answer: WHAT DOES THIS EVEN DO?
            (where is it used and does it still need to be used?)
    """
    assert cspace.shape == (batch_size, params.max_bodies * 3), f"cspace shape: {cspace.shape}, batch_size: {batch_size}, max_bodies: {params.max_bodies}"
    assert n_bodies.shape == (batch_size,)
    
    angles = cspace[:, 2::3]        # shape (n_bodies,)
    last_len = torch.vmap(get_last_body_length, in_dims=(0, 0))(cspace, n_bodies)

    assert last_len.shape == (batch_size, 1)
    
    # Step 1: compute global angles for each segment center
    global_angle_full = heading0 + torch.cumsum(torch.tensor(angles), dim=1)
        
    # Step 2: compute the center of each segment
    #   For the i-th segment, the center is offset from the anchor by
    #        sum_{k=0..i-1} [ L*cos(global_angle_full[k]), L*sin(global_angle_full[k]) ]
    #   But we can do that more efficiently. We'll build an array of cos/sin, then do a cumsum.

    # Cosines and sines of each segment angle:
    c_ = torch.cos(global_angle_full)

    s_ = torch.sin(global_angle_full)

    # Prepare the lengths of each segment
    full_lengths = torch.full((batch_size, params.max_bodies), fill_value=params.body_length)

    arange = torch.arange(batch_size)

    full_lengths[arange, n_bodies-1] = torch.tensor(last_len, dtype=torch.float32)
    
    
    # Now we do a cumulative sum of to get the tip coords of each segment
    tip_x = x0 + torch.cumsum(full_lengths * c_, dim=1)

    tip_y = y0 + torch.cumsum(full_lengths * s_, dim=1)
            
    # Return the tip coordinates for each segment as (N, 3)
    arange = torch.arange(batch_size)

    ret = torch.stack([tip_x[arange, n_bodies-1],
                       tip_y[arange, n_bodies-1],
                       global_angle_full[arange, n_bodies-1]
                       ]).transpose(0, 1)
        
    assert ret.shape == (batch_size, 3), f"ret shape: {ret.shape}"
    
    return ret


def length(params: VineParams, cspace: np.ndarray, n_bodies: np.ndarray):
    """
    Compute the total arc length of a batch of vines
    
    Args:
        params: VineParams structure with body_length and other parameters
        cspace: Configuration space tensor of shape (batch_size, max_bodies + 1)
        n_bodies: Array of shape (batch_size,) with the number of bodies for each vine
    
    Returns:
        Array of shape (batch_size,) with the total length of each vine
    """
    
    batch_size = cspace.shape[0]
    assert cspace.shape == (batch_size, params.max_bodies * 3), f"cspace shape: {cspace.shape}, batch_size: {batch_size}, max_bodies: {params.max_bodies}"
    assert n_bodies.shape == (batch_size,)
    
    # Standard bodies (all except the last one) have fixed length
    fixed_length_bodies = (n_bodies - 1) * params.body_length
    
    # Last body has variable length from the cspace
    last_body_length = torch.vmap(get_last_body_length, in_dims=(0, 0))(cspace, n_bodies)
    
    # Total length is the sum
    total_lengths = fixed_length_bodies + last_body_length
    
    assert total_lengths.shape == (batch_size,)
        
    return total_lengths


def length_unbatched(params: VineParams, cspace: np.ndarray, n_bodies: int):
    return length(params, cspace[None, ...], np.array([n_bodies]))[0]


def sample_3D_state(params: SSTparams, batch_size):
    """
    Sample a random 2D state in the range defined by SSTparams.
    Returns (N, 2)
    """
    
    x = np.random.uniform(params.min_x, params.max_x, batch_size)
    y = np.random.uniform(params.min_y, params.max_y, batch_size)
    theta = np.random.uniform(-np.pi, np.pi, batch_size)
    
    return np.stack([x, y, theta], axis=1)


def best_first_selection_sst(
        params: SSTparams,
        active: np.ndarray,
        active_costs: np.ndarray,
        batch_size: int
    ):
    """
    Decides the closest active state to start growing from, returns:
        - The nearest active state if at least one neighbor is within δBN
        - The active state with the minimum cost otherwise

    Args:
        active  : shape (N, 3) array of tip positions of active states
        active_costs : shape (N, 1) array of costs for those states
    Returns:
        shape (B, 3) indices of the selected active states
        shape (B, 3) sampled random states
    """    
    # 1) xrand ← Sample_State(X);
    xrand = sample_3D_state(params, batch_size)
    
    # 2. Xnear ← Near(V, xrand, δBN);
    # Compute the distance from xrand to all active states
    indices, dist2 = nearest_neighbor_all(params, active, xrand)
    
    # Find the (B, K) indices of the neighbors within δBN
    inside_mask = dist2 <= (params.δBN ** 2) # Shape (B, K)
    
    big_val = 1e15
    
    # Get the inside point with the lowest cost
    # If there are no inside points, behave unpredictably
    masked_costs = np.where(inside_mask, active_costs[indices], big_val)
    # Get the index of the active tip (per xrand) that has the minimum cost
    min_idx = masked_costs.argmin(axis=1)
    # Convert the indices to the actual tip positions
    cheapest_inside_point_idx = indices[min_idx]
    
    # Get the closest active state to the random state
    nearest_idx = np.argmin(dist2, axis=1)
    nearest_tip_idx = indices[nearest_idx]
    
    # If at least one neighbor is within δBN, return the nearest active state,
    # else return the active state with the minimum cost
    any_inside = inside_mask.any(axis=1)
    result = np.where(any_inside, cheapest_inside_point_idx, nearest_tip_idx)
    
    assert result.shape == (batch_size,)
    
    return result, xrand


def is_node_locally_the_best_sst(
    params: SSTparams,
    xnew_tips: np.ndarray,     # Shape (B, 3)
    xnew_costs:  np.ndarray,    # Shape (B,)
    witness_tips: np.ndarray,  # Shape (M, 3)
    rep_costs:      np.ndarray     # Shape (M,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given some new states to add, determine if:
      - They are in a new cell (distance > δs), and are added automatically. We later make a witness in its place too.
      - They are in the cell of a witness, and are added if they are better than the witness's current rep

    Args:
      params      : Has .δs as the radius threshold.
      xnew_tips : (B, d) array of newly generated states.
      xnew_costs : (B, 1)   array of costs for each new state.
      witness_tips : (M, d) array of witness states in S, corresponding to reps.
      rep_states    : (M, d) array of existing representatives in S.
      rep_costs     : (M, 1)   array of costs for those reps.

    Returns:
        to_add_fresh_mask : (B,) boolean mask of new states to add as fresh nodes.
        to_add_dominating_states_mask : (B,) boolean mask of new states to add as dominating states.
        to_add_dominating_states_witness_idx : (B,) indices of the witness states that the new states dominate.
    """
    
    assert xnew_tips.shape[0] == xnew_costs.shape[0], f"xnew_tips shape: {xnew_tips.shape}, xnew_costs shape: {xnew_costs.shape}"
    assert witness_tips.shape[0] == rep_costs.shape[0], f"witness_tips shape: {witness_tips.shape}, rep_costs shape: {rep_costs.shape}"
    
    indices, dist2 = nearest_neighbor(params, witness_tips, xnew_tips) # Shape (B)
    assert indices.ndim == 1 and dist2.ndim == 1
    
    # Find the mask of states that are within δs
    within_cell = dist2 <= (params.δs) ** 2
    
    # 2) If distance > δs, we automatically add x_new
    to_add_fresh_mask = ~within_cell
        
    # 3) If distance < δs, only add x_new if x_new_cost < cost of that nearest rep.
    to_add_dominating_states_mask = within_cell & (xnew_costs <= rep_costs[indices])
        
    to_add_dominating_states_witness_idx = np.where(to_add_dominating_states_mask, indices, -1)
    
    assert to_add_dominating_states_mask.shape == to_add_dominating_states_mask.shape
    assert to_add_fresh_mask.shape == (xnew_tips.shape[0],), f"to_add_fresh_mask shape: {to_add_fresh_mask.shape}, xnew_tips shape: {xnew_tips.shape}"
    
    assert to_add_dominating_states_mask.shape == (xnew_tips.shape[0],)
    assert to_add_dominating_states_witness_idx.shape == (xnew_tips.shape[0],)
    
    # Fresh states are disjoint from dominating states
    return to_add_fresh_mask,\
           to_add_dominating_states_mask, to_add_dominating_states_witness_idx


def prune_path_at(old_rep_idx, tree):
    # Prune dead paths. Keep pruning while:
    #   - The old rep is not past the root
    #   - The old rep has no children
    #   - The old rep is not active
    while (old_rep_idx > 0 and old_rep_idx < tree.num_states) and \
            (tree._num_children[old_rep_idx] == 0) and \
            (not tree._isactive[old_rep_idx]):
        
        # Deactivate old_rep, set it to kil
        tree._isactive[old_rep_idx] = False # Kinda redundant
        tree._kil[old_rep_idx] = True
        
        # FIXME Looks real noisy, suppressed
        # erase_sst_parent_edges(tree, old_rep_idx)
        
        # Reduce the child count of old_rep's parent (kill its connection)
        tree._num_children[tree._parent_idxs[old_rep_idx]] -= 1
        # Move up to parent for next iteration
        old_rep_idx = tree._parent_idxs[old_rep_idx]


def geometric_cost_to_go(sst_params: SSTparams, querytips):
    """
    For each tip, find the nearest point from the geometric plan, and return
    the upper bound of curves to get to the goal.
    
    Args:
        querytips : (N, 3) array of tip positions to query.
        sst_params: contains the points and their costs.
    
    Returns:
        (N,) array of costs for each tip.
    """
    
    assert sst_params.points.shape[0] > 0
    
    indices, dist2 = nearest_neighbor_all(sst_params, sst_params.points, querytips)
    
    # Get the index of the nearest point in the geometric plan
    nearest_idx = dist2.argmin(axis=1)
    # Get the cost of that point
    nearest_cost = sst_params.point_costs[nearest_idx]
    
    assert nearest_cost.shape == (querytips.shape[0],)
    
    return nearest_cost * sst_params.geo_cost_to_go_weight

#------------------------------------------ rollout

'''
NOTE's:
- nothing special was done to the obj position/dstate records if num objs == 0:
  should this be a special case, or will the record fill with null entries correctly as is?
'''

def rollout(sst_params, simparams, batch_size, 
            time_to_evolve,
            curr_time, cspace, dstate, obj_positions, obj_dstate,
            bodies, bending_control, 
            init_x, init_y, init_heading,
            ):
    """
    Perform a rollout of the vine simulation for a given number of steps, or until all
    batch elements reach the max_bodies limit. Returns the cspace, bodies, and time at 
    all steps. If a batch element reaches the bodies limit, then the last valid
    cspace and bodies are duplicated for the rest of the steps.
    
    Args:
        Left as an exercise for the reader.
    Returns:

        cspace_record : shape (steps_to_iter, batch_size, max_bodies + 1, 3)
        bodies_record  : shape (steps_to_iter, batch_size)
        time_record    : shape (steps_to_iter, batch_size)

        dynamic_obj_record : shape (steps_to_iter, batch_size, num_dynamic_objs, 4 (for the coords))
    """
    
    # Record every δs distance, to reduce pressure on the set cover
    record_every = int(sst_params.δs // (simparams.grow_rate * simparams.dt)) * sst_params.record_every_multiplier
    steps_to_iter = int(ceil((time_to_evolve) / simparams.dt))        
    
    history_size = steps_to_iter // record_every + 1

    bodies_record = np.zeros((history_size, batch_size), dtype=np.int32)
    time_record = np.zeros((history_size, batch_size), dtype=np.float32)

    cspace_record = np.zeros((history_size, batch_size, simparams.max_bodies * 3), dtype=np.float32)
    dstate_record = np.zeros((history_size, batch_size, simparams.max_bodies * 3), dtype=np.float32)

    obj_position_record = np.zeros((history_size, batch_size, int(simparams.obj_mass.size), 4), dtype=np.float32)
    obj_dstate_record = np.zeros((history_size, batch_size, int(simparams.obj_mass.size), 3), dtype=np.float32)
    
    # Track which batch elements have reached the max_bodies limit,
    # so dont update them anymore
    reached_max = bodies >= simparams.max_bodies - 1
    
    for i in range(steps_to_iter):
        
        next_cspace, next_bodies, next_obj_positions, next_dstate_solution = forward(
            simparams, init_heading, init_x, init_y, cspace, dstate, bodies, bending_control,
            obj_positions, obj_dstate
        )

        # Check if forward() has caused any vine has hit max length
        reached_max = reached_max | (bodies >= simparams.max_bodies - 1)
        
        # Record the cspace and bodies for this step, but if a vine already
        # hit its limit, reuse the last one
        cspace = np.where(reached_max[..., None, None], cspace, next_cspace)
        bodies = np.where(reached_max, bodies, next_bodies)        
        curr_time = curr_time + simparams.dt                
        obj_positions = np.where(reached_max[..., None, None], 
                                         obj_positions, next_obj_positions)
        dstate = np.where(reached_max[..., None, None], dstate, next_dstate_solution)

        if i % record_every == 0:
            # Record the current state
            bodies_record[i // record_every] = bodies
            time_record[i // record_every] = curr_time

            cspace_record[i // record_every] = cspace
            obj_position_record[i // record_every] = obj_positions

            dstate_record[i // record_every] = dstate
            obj_dstate_record[i // record_every] = obj_dstate
            
        # All our vines have hit their limit, stop the rollout
        if np.all(reached_max):
            break
    
    # Assert all bodies are within the max_bodies limit
    assert np.all(bodies < simparams.max_bodies), f"bodies: {bodies}, max_bodies: {simparams.max_bodies}"
    
    last_index_filled = i // record_every

    return time_record[:last_index_filled], \
            bodies_record[:last_index_filled], \
            cspace_record[:last_index_filled], \
            obj_position_record[:last_index_filled], \
            dstate_record[:last_index_filled], \
            obj_dstate_record[:last_index_filled], \
            last_index_filled

#------------------------------------------ actual sst()


#------------------------------------------ sst_star()

    
    decay_factor = 0.8
    sst_iter_0 = 7
    sst_iter = sst_iter_0
    j = 0
    
    # Dimension of the state space (well, not really, but good enough
    # unless you want to set this to infinity)
    d = 3
    # Dimension of the control space
    l = 1
    
    # Initial tree is None, sst will create it
    tree = None
    
    # Callback once to indicate we have started
    if callback is not None:
        callback(None)
    
    while True:
        # Clear the screen
        clear_all_surfaces()
        
        tree, info = sst(sst_params, sim_params, tree, sst_iter, callback)
        
        if 'status' in info and info['status'] == 'callback requested suicide':
            break
        
        # Shrink the δ-robust region towards zero
        sst_params.δs *= decay_factor
        sst_params.δBN *= decay_factor
        
        # Update iters
        j += 1
        sst_iter = sst_iter_0 * (1 + math.log2(j)) * decay_factor ** (-1 * (d + l + 1) * j) 
        
        # Clear the screen
        # clear_all_surfaces()
        
        # Look for reps that no longer fall within δs
        # And deactivate them, also search for prune candidates in their ancestors
        witness_tip = tree.witness_positions()
        witness_rep_idx_ = tree.rep_idxs()
        valid_witness_mask = witness_rep_idx_ >= 0
        
        # Get the witnesses with valid reps
        witness_tip = witness_tip[valid_witness_mask]
        witness_rep_idx = witness_rep_idx_[valid_witness_mask]
        rep_tip = tree._tips[witness_rep_idx]
        
        # Find the ones with distance > δs
        distance_from_witness_to_rep2 = distance(sst_params, witness_tip - rep_tip)
        to_remove_mask = distance_from_witness_to_rep2 > sst_params.δs * sst_params.δs
        
        for rep_idx in witness_rep_idx[to_remove_mask]:
            # Deactivate the rep (prequisite for pruning)
            tree._isactive[rep_idx] = False
                        
            # Prune the old rep, and any valid ancestors
            prune_path_at(rep_idx, tree)
            
        for rep_idx in witness_rep_idx_[~valid_witness_mask]:
            # Deactivate the rep (prequisite for pruning)
            tree._isactive[rep_idx] = False
                        
            # Prune the old rep, and any valid ancestors
            prune_path_at(rep_idx, tree)
        
        # TODO prune_path_at may introduce more invalid nodes
        # we should repeat until there are none
        
        # TODO consider taking the min possible segments K so far,
        # and removing all states with >= k segments (since we know for sure)
        # we can do better
        
        tree.clean_states()

#-------------------------------------------- sst

'''
NOTE's: 
- p, l0 randomization is followed similar to before; is this compatible with new underlying code?

- think about how to render rotation for moving blocks (in this case, they're just squares)
'''

def sst(sst_params: SSTparams, sim_params: VineParams, init_obj_positions, 
        tree, iters=1000, callback=None):

    global find_actuator_params
    find_actuator_params = sim_params.spam_moment_fn

    batch_size = sst_params.batch_size

    init_x = sst_params.start[0]
    init_y = sst_params.start[1]
    init_heading = sst_params.start[2]

    bodies = 1

    cspace = np.zeros((sim_params.max_bodies * 3))
    cspace[0, 0], cspace[0, 1] = init_x, init_y

    bending_control = np.zeros((1, sim_params.max_bodies, 2))
    bending_control[:, :, 0] = 0.0 # pressure
    bending_control[:, :, 1] = 0.025/2 # l0

    tip = cspace_to_tip(sim_params, 1, cspace[None, ...], np.array([bodies]), init_x, init_y, init_heading)

    # Initialize StateTree if needed (using values above)

    if not tree:
        tree = StatesStruct(sim_params.max_bodies, sim_params.obj_mass.size)

        cost_to_go = 0 if sst_params.points is None else geometric_cost_to_go(sst_params, tip).item()

        state0_idx = tree.add_state(isactive=True,
                                    cspace=cspace,
                                    dstate=torch.zeros((sim_params.max_bodies * 3)),
                                    obj_positions=init_obj_positions,
                                    obj_dstate=torch.zeros((sim_params.obj_mass.size, 3)),
                                    bodies=bodies,
                                    bending_control=bending_control,
                                    time=0,
                                    cost_to_come=0,
                                    cost_total= tiebreak_factor * length_unbatched(sim_params, cspace, bodies) + cost_to_go,
                                    tip=tip,
                                    parent_idx=-1,
                                    num_children=0,)
            
        tree.add_witness(np.zeros(3), state0_idx)

    # Draw all witnesses and their rep tips (if existing)
    for wit_idx in range(tree.num_witnesses):
        draw_witness(tree, wit_idx, sst_params.δs)

    # Get all rep_idxs which are not empty
    witness_has_rep_mask = tree.rep_idxs() > 0
    valid_rep_idxs = tree.rep_idxs()[witness_has_rep_mask]
    rep_tips = tree._tips[valid_rep_idxs]

    # Draw all rep tips, then witness_to_rep
    draw_tips(rep_tips, costs=tree._cost_to_come[valid_rep_idxs])

    # SST iteration:

    for sst_iter in range(int(iters)):

        # ------------ Sample random tip positions and their closest active states ------------
        active_states_mask = tree._isactive
        
        # active_states_idx (B,) indexes the items in active_states_mask
        active_states_idx, xrand = best_first_selection_sst(sst_params, tree._tips[active_states_mask], tree._cost_to_come[active_states_mask], batch_size)
        
        # propagate_origin_idx (B,) indexes tree states, are the states we start propagating from
        propagate_origin_idx = np.where(active_states_mask)[0][active_states_idx] 

        # ------------- Monte Carlo propagation of the closest states -------------
        # NOTE: p, l0 randomization is here

        new_bend_angle = np.random.uniform(-3.33, 3.33, batch_size) # Shape (B,)
        new_bend_angle = 1.0 / new_bend_angle

        p, l0 = find_actuator_params(predict, act_params, torch.tensor(new_bend_angle))
        assert np.all(np.isfinite(p)), f"p: {p}, l0: {l0}"

        current_bending_controls = tree._bending_controls[propagate_origin_idx]
        current_bodies = tree._bodies[propagate_origin_idx]

        for idx in range(batch_size):
            current_bending_controls[idx, current_bodies[idx]:, 0] = p[idx]        
            current_bending_controls[idx, current_bodies[idx]:, 1] = l0[idx]

        # Rollout:

        print('Starting rollout...')
        start_time = time.time()

        new_times, new_bodies, new_cspaces, new_obj_positions, \
        new_dstates, new_obj_dstates, last_index_filled = rollout(
            sst_params, sim_params, batch_size, sst_params.time_to_evolve,
            curr_time=tree._times[propagate_origin_idx],
            cspace=tree._c_spaces[propagate_origin_idx],
            dstate=tree._dstates[propagate_origin_idx],
            obj_positions=tree._obj_positions[propagate_origin_idx],
            obj_dstate=tree._obj_dstates[propagate_origin_idx],
            bodies=tree._bodies[propagate_origin_idx],
            bending_control=current_bending_controls,
            init_x=init_x,
            init_y=init_y,
            init_heading=init_heading)
        print('Rollout time:', time.time() - start_time)

        if not sst_params.do_maximal:
            # Sample one timestep to take from per batch
            take_one_idx = np.random.randint(0, steps_to_iter, batch_size)
            batch_indices = np.arange(batch_size)
            
            new_bodies = new_bodies[take_one_idx, batch_indices]
            new_times = new_times[take_one_idx, batch_indices]

            new_cspaces = new_cspaces[take_one_idx, batch_indices]
            new_obj_positions = new_obj_positions[take_one_idx, batch_indices]

            new_dstates = new_dstates[take_one_idx, batch_indices]
            new_obj_dstates = new_obj_dstates[take_one_idx, batch_indices]

            steps_to_iter = 1 

        # Rollout returns a record of position at each timestep, so flatten timestep and batch together                
                
        new_bodies = new_bodies.reshape(-1)
        new_times = new_times.reshape(-1)

        new_cspaces = new_cspaces.reshape(-1, sim_params.max_bodies * 3)
        new_dstates = new_dstates.reshape(-1, tree.dstate_len * 3)
        new_obj_positions = new_obj_positions(-1, sim_params.obj_mass.size, 4)
        new_obj_dstates = new_dstates(-1, sim_params.obj_mass.size, 4)

        assert new_bodies.shape == (steps_to_iter * batch_size,), f"new_bodies shape: {new_bodies.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"
        assert new_times.shape == (steps_to_iter * batch_size,), f"new_times shape: {new_times.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"

        assert new_cspaces.shape == (steps_to_iter * batch_size, sim_params.max_bodies * 3), f"new_cspaces shape: {new_cspaces.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"
        assert new_dstates.shape == (steps_to_iter * batch_size, sim_params.max_bodies * 3), f"new_dstates shape: {new_dstates.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"
        assert new_obj_positions == (steps_to_iter * batch_size, sim_params.obj_mass.size, 4), f"new_dstates shape: {new_obj_positions.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"
        assert new_obj_dstates == (steps_to_iter * batch_size, sim_params.obj_mass.size, 3), f"new_dstates shape: {new_obj_dstates.shape}, steps_to_iter: {steps_to_iter}, batch_size: {batch_size}"

        # Assert that no new cspace is all zeros
        assert np.all(~np.all(new_cspaces == 0, axis=(1,2))), f"new_cspaces shape: {new_cspaces.shape}"

        finite_mask = np.all(np.isfinite(new_cspaces), axis=1)
        assert np.all(finite_mask), f"{np.sum(finite_mask)} finite cspaces out of {new_cspaces.shape[0]}"

        # Get the tip position of the new states
        new_tips = cspace_to_tip(sim_params, new_cspaces.shape[0], new_cspaces, new_bodies, init_x, init_y, init_heading) 

        # Increment the costs of the new states by 1 (since we applied a new control input)
        cost_come = tree._cost_to_come[propagate_origin_idx] + 1

        # Tile the costs to match the shape of cspaces (steps_to_iter * batch_size)
        new_costs_come = np.tile(cost_come, [steps_to_iter])
        propagate_origin_idx = np.tile(propagate_origin_idx, [steps_to_iter])
        current_bending_controls = np.tile(current_bending_controls, [steps_to_iter, 1, 1])
        
        assert new_costs_come.shape == (steps_to_iter * batch_size,)
        assert propagate_origin_idx.shape == (steps_to_iter * batch_size,)
        assert current_bending_controls.shape == (steps_to_iter * batch_size, sim_params.max_bodies, 2)

        # --------- Find non-overlapping subset of states ---------
        if sst_params.do_set_cover:
            print('starting max_cover with', new_tips.shape[0], 'states')
            start_time = time.time()
            non_overlapping_mask = max_cover(sst_params, np.asarray(new_tips))
            print('max_cover time:', time.time() - start_time, 'ended with', non_overlapping_mask.sum(), 'states')
            
            
            new_bodies = new_bodies[non_overlapping_mask]
            new_times = new_times[non_overlapping_mask]
            new_tips = new_tips[non_overlapping_mask]
            new_costs_come = new_costs_come[non_overlapping_mask]
            current_bending_controls = current_bending_controls[non_overlapping_mask]
            propagate_origin_idx = propagate_origin_idx[non_overlapping_mask]

            new_cspaces = new_cspaces[non_overlapping_mask]
            new_dstates = new_dstates[non_overlapping_mask]
            new_obj_positions = new_obj_positions[non_overlapping_mask]
            new_obj_dstates = new_obj_dstates[non_overlapping_mask]
            
        if sst_params.do_cost_to_go:
            new_costs_total = new_costs_come + geometric_cost_to_go(sst_params, new_tips) + \
                            tiebreak_factor * length(sim_params, new_cspaces, new_bodies)
        else:
            new_costs_total = new_costs_come

        draw_dead_state(sim_params, new_cspaces, new_obj_positions, new_bodies, init_x, init_y, init_heading)

        # If any states falls in the goal region, add to the solutions
        # Append all info: bodies, cspaces, costs, bending controls

        initial_num_solutions = sst_params.solutions.qsize()
        in_goal_mask = np.linalg.norm(new_tips[:, 0:2] - sst_params.goal[0:2], axis=1) < sst_params.goal_radius
        for idx in in_goal_mask.nonzero()[0]:
            sst_params.solutions.put(DontCompareSecond(
                new_costs_come[idx].item() + tiebreak_factor * length_unbatched(sim_params, new_cspaces[idx], new_bodies[idx]),
                {
                    'cspace': new_cspaces[idx],
                    'bodies': new_bodies[idx],
                    'bending_control': current_bending_controls[idx],
                    'cost_to_come': new_costs_come[idx],
                    'cost_total': new_costs_total[idx],
                    'tip': new_tips[idx],
                }
            ))

        # Save solutions to file if we passed a multiple of 5
        save_every = 5
        more_than_5 = sst_params.solutions.qsize() >= initial_num_solutions + save_every
        passed_5_mod = sst_params.solutions.qsize() % save_every < initial_num_solutions % save_every
        if more_than_5 or passed_5_mod:
            # Save winning states and sim params
            np.save('cache/solutions.npy', list(sst_params.solutions.queue))
            np.save('cache/winning_sim_params.npy', sim_params)
            np.save('cache/winning_info.npy', sst_params.info)
            print(f'\033[92mSaved solutions ({sst_params.solutions.qsize()}) \033[0m')

        # --------- Add new states to tree ---------
        # Get cost for each witness's rep, or -np.inf if has no rep
        witness_rep_costs = np.where(tree.rep_idxs() > 0, tree._cost_total[tree.rep_idxs()], -np.inf)

        # Fresh states don't touch any existing witness (will make new witnesses for them)
        # Dominating states fall inside an existing witness and has better cost than the witness's rep
        # (will replace the old rep with them)
        # All these masks index into xnew_*
        new_fresh_mask, new_dominating_states_mask, new_dominating_states_witness_idx = \
                    is_node_locally_the_best_sst(sst_params, 
                                                new_tips, 
                                                new_costs_total, 
                                                tree.witness_positions(), 
                                                witness_rep_costs)
        
        draw_tips(new_tips, costs=new_costs_total)


        # Add new witness-creating states to the tree, and record their indexes      
        
        new_fresh_idx = tree.add_states(isactive=True,
                                        c_space=new_cspaces[new_fresh_mask],
                                        dstate=new_dstates[new_fresh_idx],
                                        obj_positions=new_obj_positions[new_fresh_idx],
                                        obj_dstates=new_obj_dstates[new_fresh_idx],
                                        bodies=new_bodies[new_fresh_mask],
                                        time=new_times[new_fresh_mask],
                                        bending_control=current_bending_controls[new_fresh_mask],
                                        # Heuristic stuff
                                        cost_to_come=new_costs_come[new_fresh_mask],
                                        cost_total=new_costs_total[new_fresh_mask],
                                        tip=new_tips[new_fresh_mask],
                                        # Tree stuff
                                        parent_idx=propagate_origin_idx[new_fresh_mask],
                                        num_children=0)
        
        # Add new dominating states to the tree, and record their indexes
        new_dominating_states_idx = tree.add_states(isactive=True,
                                        c_space=new_cspaces[new_dominating_states_mask],
                                        dstate=new_dstates[new_dominating_states_mask],
                                        obj_positions=new_obj_positions[new_dominating_states_mask],
                                        obj_dstates=new_obj_dstates[new_dominating_states_mask],
                                        bodies=new_bodies[new_dominating_states_mask],
                                        time=new_times[new_dominating_states_mask],
                                        bending_control=current_bending_controls[new_dominating_states_mask],
                                        # Heuristic stuff
                                        cost_to_come=new_costs_come[new_dominating_states_mask],
                                        cost_total=new_costs_total[new_dominating_states_mask],
                                        tip=new_tips[new_dominating_states_mask],
                                        # Tree stuff
                                        parent_idx=propagate_origin_idx[new_dominating_states_mask],
                                        num_children=0)

        # Set states to inactive if they have 1 more segments than the best so far
        min_segments = sst_params.solutions.queue[0].second['cost_to_come'] \
            if sst_params.solutions.qsize() > 0 else 9999
        
        too_many_segs_mask = tree.cost_to_come() > min_segments + 1
        tree.isactive()[too_many_segs_mask] = False

        # Update child counter for parents of recently added states
        tree._num_children[propagate_origin_idx[new_fresh_mask]] += 1
        tree._num_children[propagate_origin_idx[new_dominating_states_mask]] += 1

        # For each fresh state, create a new witness and assign state as rep
        new_witness_idx = tree.add_witnesses(new_tips[new_fresh_mask], new_fresh_idx)
        for idx in new_witness_idx:
            draw_witness(tree, idx, sst_params.δs)

        # Overthrow old reps and bring in the new guard of dominating reps
        old_rep_idx = tree._rep_idxs[new_dominating_states_witness_idx]
        
        # Deactivate the old rep
        tree._isactive[old_rep_idx] = False 
        
        # erase_witness_to_rep(tree, the_witness_of_the_hour)
        
        # Set the new rep
        tree._rep_idxs[new_dominating_states_witness_idx[new_dominating_states_mask]] = new_dominating_states_idx
        
        # Prune the old rep, and any valid ancestors
        for idx in old_rep_idx:
            prune_path_at(idx, tree)

        # -------- Print out some stats --------
        num_active_nodes = np.sum(tree.isactive())
        num_total_nodes = tree.num_states
        num_reps = np.sum(tree.rep_idxs() >= 0)
        num_witnesses = tree.num_witnesses

        draw_goal(sst_params.goal, sst_params.goal_radius)
        draw_stats(sst_params, sst_iter, iters, num_active_nodes, num_total_nodes, num_reps, num_witnesses, tree._cost_to_come[tree._isactive])
        render()


        if callback:
            # Serves two functions: let the callback maker know the current best solution,
            # and allow the callback to kill self if needed
            
            if sst_params.solutions.qsize() > 0:
                callback_return = callback(sorted(sst_params.solutions.queue))
            else:
                callback_return = callback(None)
            
            if callback_return is True:
                print('SST: got callback intent to suicide, unaliving')
                return tree, {'status': 'callback requested suicide'}

    return tree, {}

#------------------------------------------ main()
if __name__ == "__main__":
    pass