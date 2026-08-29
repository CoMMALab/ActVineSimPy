'''
Actual file that runs RRT
'''

import torch

'''
Questions:
1. How will randomly sampling points work? Is that just by randomly selecting p, l_0,
   and then seeing what the state would look like afterwards? Could this potentially
   take a really long time
'''

class StateInfo:
    '''
    Kind of just a dict that holds all info relevant to the state
    Useful to hold when animating gifs and stuff
    '''
    def __init__(self, state, dstate, bodies, moveable_obj_pose, moveable_obj_dstate):
        self.state = state
        self.dstate = dstate
        self.bodies = bodies
        self.moveable_obj_pose = moveable_obj_pose
        self.moveable_obj_dstate = moveable_obj_dstate

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

    def __init__(self, start_state, distance_function, goal_test):
        self.root = Node(start_state)
        self.distance_function = distance_function
        self.goal_test = goal_test


    def find_nearest_to(self, new_state: StateInfo):

        '''
        Returns the nearest state to new_state that is currently in the graph;
        Does so based off of given distance_function.
        '''

        min_distance = float("inf")
        nearest_state = None

        for node in Node.all_nodes:
            dist = self.distance_function(node.info, new_state)

            print(f"Node {node.info["state"]} with dist {dist}")

            if dist < min_distance:
                min_distance = dist
                nearest_state = node
                print(f"Picking node {nearest_state.info["state"]}")

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

        if (self.goal_test(new_state)):
            path = []
            curr_node = new_state_node

            while curr_node != self.root:
                path.append(curr_node)
                curr_node = curr_node.parent

            path.append(curr_node) # append root
            path.reverse()
            return path

            




