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
        elif key == "p_control":
            return self.p_control
        elif key == "l0_control":
            return self.l0_control


class Node:
    '''
    Wraps state info so that it can be better
    used as node in the tree
    '''

    counter = 0 # for unique id generation
    all_nodes = [] # for find_nearest_to() in O(n)

    all_leaves = set() # tracks which nodes don't have children

    def __init__(self, state_info, parent=None):
        self.parent = parent
        self.info = state_info
        self.children = []

        self.node_id = Node.counter
        Node.counter += 1

        Node.all_nodes.append(self)

        Node.all_leaves.add(self)

    def __eq__(self, value):
        return (self.node_id == value.node_id)

    def __hash__(self):
        return hash(self.node_id)


class RRTTree:
    '''
    Tree from the start state (root) to some goal state.
    ''' 

    def __init__(self, start_state, distance_function, goal_test,
                 aux_goal_test_args,
                 aux_distance_func_args):
        
        self.root = Node(start_state)

        self.distance_function = distance_function
        self.goal_test = goal_test

        # Packed args for rest of goal test args
        self.goal_test_args = aux_goal_test_args

        # Packed args for rest of distance func args
        self.distance_func_args = aux_distance_func_args

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


    def add_edge(self, node_in_graph: Node, new_state: StateInfo):

        '''
        Add new_state to the adj list of state_in_graph'

        Return: None if new_state is not a goal state
                If new_state is a goal_state, returns a path from
                start_state to goal_state
        '''

        new_state_node = Node(new_state, node_in_graph)

        # because state_in_graph just got a child, it's no longer a leaf:
        if len(node_in_graph.children) == 0:
            Node.all_leaves.remove(node_in_graph)

        node_in_graph.children.append(new_state_node)


        # TO DEBUG: put prints everywhere to see if the ref really gets dropped/is changed
        #           within the set:
        #   1. nearest_to returning copy, not ref?
        #   2. static structures somehow changing? look into...

        if (self.goal_test(new_state, *self.goal_test_args)):
            path = []
            Node.all_leaves.remove(new_state_node)
            curr_node = new_state_node

            while curr_node != self.root:
                path.append(curr_node)
                curr_node = curr_node.parent

            path.append(curr_node) # append root
            path.reverse()

            return path
        
        else:
            return None


    def get_closest_paths(self, num_closest_paths,
                          measure_quality: callable,
                          aux_quality_func_args):
        '''
        Return a list of paths that is AT MOST "num_closest_paths" long;
        Returns paths in increasing order determined by the measaure_quality function 

        NOTE: at most one of the leaves should be a goal state, since RRT is meant 
              to stop the iteration it finds a goal state;
              For this implementation, the goal state will be popped out of the set of leaves,
              meaning all paths returned here will not end in goal states. 
        '''
        all_leaves = list(Node.all_leaves)
        all_leaves.sort(key = lambda node: measure_quality(node, *aux_quality_func_args))
        
        closest_paths = []

        for i in range(num_closest_paths):

            # if num paths requested > num paths that exist
            if i >= len(all_leaves):
                return closest_paths

            path = []
            curr_node = all_leaves[i]

            while curr_node != self.root:
                path.append(curr_node)
                curr_node = curr_node.parent

            path.append(curr_node) # append root
            path.reverse()
            closest_paths.append(path)

        return closest_paths