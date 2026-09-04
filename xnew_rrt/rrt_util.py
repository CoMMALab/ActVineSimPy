'''
Actual file that runs RRT
'''

import torch
from dvsim import si

B = 1 # this RRT implementation is unbatched
MM = lambda x: si.len_to_mm(x)    # internal length -> mm 
ND = lambda x: x / (si.L0 * 1000) # mm -> internal length 

'''
Questions:
1. How will randomly sampling points work? Is that just by randomly selecting p, l_0,
   and then seeing what the state would look like afterwards? Could this potentially
   take a really long time


2. FIXME: How do you take a random step in the sim? 
          Which state do you use as the parent when taking this random step?
          Just the nearest one to the goal? 

          Above kind of changes how the whole structure operates...
'''


#--------------------------------------------------------- Data Structures

class StateInfo:
    '''
    Kind of just a dict that holds all info relevant to the state
    Useful to hold when animating gifs and stuff

    NOTE: because what is saved in here is what is spit out by the sim,
          everything here is in NON-DIM INTERNAL UNITS (when applicable)
    '''
    def __init__(self, state, dstate, bodies, moveable_obj_pose, moveable_obj_dstate,
                 p_control=None, l0_control=None):
        self.state = state
        self.dstate = dstate
        self.bodies = bodies
        self.moveable_obj_pose = moveable_obj_pose
        self.moveable_obj_dstate = moveable_obj_dstate

        self.p_control = p_control
        self.l0_control = l0_control


    def __getitem__(self, key):
        if key == "state":
            return self.state
        elif key == "dstate":
            return self.dstate
        elif key == "bodies":
            return self.bodies
        elif key == "moveable_obj_pose":
            return self.moveable_obj_pose
        elif key == "moveable_obj_dstate":
            return self.moveable_obj_dstate


class Node:
    '''
    Wraps state info so that it can be better
    used as node in the tree
    '''

    counter = 0 # for unique id generation
    all_nodes = [] # for find_nearest_to() in O(n)

    def __init__(self, state_info, parent=None):
        self.parent = parent
        self.info = state_info
        self.children = []

        self.node_id = Node.counter
        Node.counter += 1

        Node.all_nodes.append(self)

    def __eq__(self, value):
        return (self.node_id == value.node_id)

    def __hash__(self):
        return hash(self.node_id)


class RRTTree:
    '''
    Tree from the start state (root) to some goal state.

    TODO: there is not currently any way to store actions/params
          corresponding with each path (how to implement this?)

    TODO: for Nodes, there is no check to make sure that children are unique;
          does this need to be done at all? 
    ''' 

    def __init__(self, start_state, distance_function, goal_test,
                 rest_of_goal_test_args,
                 rest_of_distance_func_args):
        self.root = Node(start_state)
        self.distance_function = distance_function
        self.goal_test = goal_test

        # Packed args for rest of goal test args
        # self.vine_radius = vine_radius
        # self.goal_coords = goal_coords
        # self.goal_radius = goal_radius
        self.goal_test_args = rest_of_goal_test_args

        # Packed args for rest of distance func args
        # self.max_bodies = max_bodies
        # self.num_moveable_objs = num_moveable_objs
        self.distance_func_args = rest_of_distance_func_args

    def find_nearest_to(self, new_state: StateInfo):

        '''
        Returns the nearest state to new_state that is currently in the graph;
        Does so based off of given distance_function.
        '''

        min_distance = float("inf")
        nearest_state = None

        for node in Node.all_nodes:
            dist = self.distance_function(node.info, new_state, 
                                          *self.distance_func_args)

            if dist < min_distance:
                min_distance = dist
                nearest_state = node

        return nearest_state


    def add_edge(self, state_in_graph: Node, new_state: StateInfo):

        '''
        Add new_state to the adj list of state_in_graph'

        Return: None if new_state is not a goal state
                If new_state is a goal_state, returns a path from
                start_state to goal_state
        '''

        new_state_node = Node(new_state, state_in_graph)
        state_in_graph.children.append(new_state_node)

        if (self.goal_test(new_state, *self.goal_test_args)):
            path = []
            curr_node = new_state_node

            while curr_node != self.root:
                path.append(curr_node)
                curr_node = curr_node.parent

            path.append(curr_node) # append root
            path.reverse()
            return path
        
        else:
            return None

            
#----------------------------------------------------------------------- RRT Distance Funcs

def euclidean_distance(state_obj1: StateInfo, state_obj2: StateInfo, 
                 max_bodies, num_moveable_objs,
                 theta_weight = 1, 
                 weight_list = (1, 1, 1, 1, 1)):
    '''
    Default distance function for comparing states returned by sim
    On args:
        - theta_weight: how much to weight diff in theta compared to diff in position
        - weight_list: how much to multiply each measure by (arbitrarily decided)

    NOTE: everything stays in non-dim units, but I don't think that really matters... :p
    '''

    def vmapped_distance(pose1, pose2, theta_weight):
        # Tentative measure for angle diff: 
        # dtheta = torch.atan2(torch.sin(theta1 - theta2), torch.cos(theta1 - theta2))

        # Euclidean distance is used otherwise

        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2

        euclid_dist = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dtheta = torch.atan2(torch.sin(theta2 - theta1), torch.cos(theta2 - theta1))

        return euclid_dist + theta_weight * dtheta

    #NOTE: reshapes are to make things easier for vmap

    vine_state1 = state_obj1["state"].reshape(B * max_bodies, 3)
    vine_state2 = state_obj2["state"].reshape(B * max_bodies, 3)

    vine_dstate1 = state_obj1["dstate"].reshape(B * max_bodies, 3)
    vine_dstate2 = state_obj2["dstate"].reshape(B * max_bodies, 3)

    bodies1 = state_obj1["bodies"].reshape(B * 1)
    bodies2 = state_obj2["bodies"].reshape(B * 1)

    obj_state1 = state_obj1["moveable_obj_pose"].reshape(B * num_moveable_objs, 3)
    obj_state2 = state_obj2["moveable_obj_pose"].reshape(B * num_moveable_objs, 3)

    obj_dstate1 = state_obj1["moveable_obj_dstate"].reshape(B * num_moveable_objs, 3)
    obj_dstate2 = state_obj2["moveable_obj_dstate"].reshape(B * num_moveable_objs, 3)

    # Diff for vine states:
    vine_state_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(vine_state1, vine_state2, theta_weight)
    vine_state_diff = torch.sum(vine_state_diff)

    # Diff for vine dstates:
    vine_dstate_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(vine_dstate1, vine_dstate2, theta_weight)
    vine_dstate_diff = torch.sum(vine_dstate_diff)

    # Diff for bodies:
    bodies_diff = torch.abs(bodies2 - bodies1)

    # Diff for moveable object states:
    obj_state_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(obj_state1, obj_state2, theta_weight)
    obj_state_diff = torch.sum(obj_state_diff)

    # Diff for moveable object dstates:
    obj_dstate_diff = torch.vmap(vmapped_distance, in_dims=(0, 0, None))(obj_dstate1, obj_dstate2, theta_weight)
    obj_dstate_diff = torch.sum(obj_dstate_diff)

    # Return one weighted measure:

    state_w, dstate_w, bodies_w, obj_state_w, obj_dstate_w = weight_list

    return (vine_state_diff * state_w) + \
           (vine_dstate_diff * dstate_w) + \
           (bodies_diff * bodies_w) + \
           (obj_state_diff * obj_state_w) + \
           (obj_dstate_diff * obj_dstate_w)



#----------------------------------------------------------------------- RRT Goal Tests

def last_body_goal_test(state_obj: StateInfo, 
                        max_bodies,
                        vine_radius, # in m, for consistency
                        goal_coords, # (x, y) given in m
                        goal_radius  # given in m
                        ):
    '''
    Simple goal test: if the last body is anywhere within the 
    goal region, then returns True (otherwise returns False).

    NOTE: assumes both the vine body's and the goal region's geometries
          are simple circles.
    '''

    num_bodies = state_obj["bodies"].reshape(B * 1)

    last_body_x, last_body_y, last_body_theta = state_obj["state"].reshape(B * max_bodies, 3)[num_bodies - 1].squeeze()

    # Convert internal dims to meters:
    last_body_x = MM(last_body_x) / 1000; last_body_y = MM(last_body_y) / 1000

    min_dist_before_collision = goal_radius + vine_radius

    distance = torch.sqrt((goal_coords[0] - last_body_x)**2 + (goal_coords[1] - last_body_y)**2)

    return distance < min_dist_before_collision


