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
from pbd_vine import VineParams, step_vine_batched, SCS_step_vine

from geometric.biarc_rrtstar import main as geometric_plan
from kinodynamic.nearest import distance, nearest_neighbor, nearest_neighbor_all

from sPAM.spam import paramstype, params as act_params

from sPAM.torch_nns import get_or_train_model, get_prediction_function

from sPAM.torch_nns_usage import torch_solve as find_actuator_params, solve_fwd as actuator_params_fwd_


#------------------------------------------ global defs (variable/function defs)

tiebreak_factor = 0.0001
bending_controls_size = 2

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
        self._obj_positions = np.zeros((self.init_size, num_dynamic_objs, 8), 
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
        self._dynamic_positions = np.concatenate([self._dynamic_positions, np.zeros((current_size, self.num_dynamic_objs, 8), dtype=np.float32)], 
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

        assert c_space.shape == (num_to_add, self.max_bodies + 1, 3)
        assert dstate.shape == (num_to_add, self.dstate_len, 3)

        if self.num_dynamic_objs != 0:
            assert obj_positions.shape == (num_to_add, self.num_dynamic_objs, 8)
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

#------------------------------------------ rollout

#------------------------------------------ actual sst()


#------------------------------------------ sst_star()
def sst_star(sst_params: SSTparams, sim_params: VineParams, callback=None):
    
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



#------------------------------------------ main()
if __name__ == "__main__":
    pass