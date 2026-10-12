'''
Data structures for the RRT: StateInfo (one sim state), Node and RRTTree.
'''

import torch


#--------------------------------------------------------- Data Structures

class StateInfo:
    '''
    Kind of just a dict that holds all info relevant to the state
    Useful to hold when animating gifs and stuff

    NOTE: because what is saved in here is what is spit out by the sim,
          everything here is in NON-DIM INTERNAL UNITS (when applicable).
          With dvsim.si.L0 = 1 mm that means lengths are in mm, angles in radians.
    '''

    KEYS = ("state", "dstate", "bodies", "moveable_obj_pose", "moveable_obj_dstate",
            "p_control", "l0_control", "moment_control", "bend_target", "base")

    def __init__(self, state, dstate, bodies, moveable_obj_pose, moveable_obj_dstate,
                 p_control=None, l0_control=None, moment_control=None, bend_target=None, base=None):
        self.state = state
        self.dstate = dstate
        self.bodies = bodies
        self.moveable_obj_pose = moveable_obj_pose
        self.moveable_obj_dstate = moveable_obj_dstate

        # The "design" of the vine grown so far, one entry per body (max_bodies,). It belongs to
        # the state because bodies that already exist keep their actuation.
        self.p_control = p_control      # sPAM pressure [Pa]            ('spam' mode)
        self.l0_control = l0_control    # sPAM rest length [m], signed  ('spam' mode)
        self.moment_control = moment_control  # moment those actuators give [N*m] ('spam' mode)
        self.bend_target = bend_target  # rest angle per joint [rad]    ('curvature' mode)

        # (x, y, heading) of the vine's base in internal units; None = the scene's start pose
        self.base = base

    def __getitem__(self, key):
        if key not in self.KEYS:
            raise KeyError(key)
        return getattr(self, key)

    @property
    def num_bodies(self):
        return int(self.bodies.reshape(-1)[0])

    def tip(self):
        '''(x, y, theta) of the last (tip) body, batch element 0.'''
        n = self.num_bodies
        return self.state[0, 3 * (n - 1): 3 * n]


class Node:
    '''
    Wraps state info so that it can be better
    used as node in the tree
    '''

    def __init__(self, node_id, state_info, parent=None, control=None, edge=None):
        self.node_id = node_id
        self.parent = parent
        self.info = state_info
        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1

        self.control = control    # control applied along the edge parent -> self
        self.edge = edge or []    # every sim state along that edge (last one is self.info)
        self.expandable = True    # False once the vine can't grow any further
        self.failures = 0         # rollouts from this node that the sim could not complete

    def __eq__(self, value):
        return isinstance(value, Node) and self.node_id == value.node_id

    def __hash__(self):
        return hash(self.node_id)

    def path_from_root(self):
        path = []
        curr_node = self
        while curr_node is not None:
            path.append(curr_node)
            curr_node = curr_node.parent
        path.reverse()
        return path


class RRTTree:
    '''
    Tree from the start state (root) to some goal state.
    All bookkeeping (nodes, leaves, ids) lives on the tree instance, so several
    trees can exist in one process.
    '''

    def __init__(self, start_state, goal_test, aux_goal_test_args=()):

        self.all_nodes = []      # for find_nearest_to() in O(n)
        self.all_leaves = set()  # tracks which nodes don't have children
        self.goal_node = None

        self.goal_test = goal_test
        self.goal_test_args = aux_goal_test_args  # packed args for rest of goal test args

        self.root = self._new_node(start_state)

    def _new_node(self, state_info, parent=None, control=None, edge=None):
        node = Node(len(self.all_nodes), state_info, parent, control, edge)
        self.all_nodes.append(node)
        self.all_leaves.add(node)
        return node

    def find_nearest_to(self, query, distance_function: callable):
        '''
        Returns the nearest EXPANDABLE node to `query` (None if there is none),
        based off of the given distance_function(state_info, query).
        '''

        min_distance = float("inf")
        nearest_node = None

        for node in self.all_nodes:
            if not node.expandable:
                continue

            dist = float(distance_function(node.info, query))

            if dist < min_distance:
                min_distance = dist
                nearest_node = node

        return nearest_node

    def add_node(self, node_in_graph: Node, new_state: StateInfo, control=None, edge=None):
        '''
        Add new_state as a child of node_in_graph and return the new Node.
        If it passes the goal test it also becomes self.goal_node.
        '''

        new_node = self._new_node(new_state, node_in_graph, control, edge)

        # because node_in_graph just got a child, it's no longer a leaf:
        self.all_leaves.discard(node_in_graph)
        node_in_graph.children.append(new_node)

        if self.goal_test(new_state, *self.goal_test_args):
            self.goal_node = new_node

        return new_node

    def add_edge(self, node_in_graph: Node, new_state: StateInfo, control=None, edge=None):
        '''
        Like add_node(), but returns the path (list of Nodes, root first) if
        new_state is a goal state and None otherwise.
        '''
        new_node = self.add_node(node_in_graph, new_state, control, edge)
        return new_node.path_from_root() if new_node is self.goal_node else None

    def get_closest_nodes(self, num_closest, measure_quality: callable, aux_quality_func_args=(),
                          leaves_only=True):
        '''
        At most `num_closest` nodes, in increasing order of measure_quality(node, *args).
        '''
        candidates = list(self.all_leaves) if leaves_only else list(self.all_nodes)
        candidates.sort(key=lambda node: measure_quality(node, *aux_quality_func_args))
        return candidates[:num_closest]

    def get_closest_paths(self, num_closest_paths, measure_quality: callable,
                          aux_quality_func_args=()):
        '''
        Return a list of paths (root first) that is AT MOST "num_closest_paths" long,
        ending at the leaves that score best under measure_quality.
        '''
        return [node.path_from_root() for node in
                self.get_closest_nodes(num_closest_paths, measure_quality, aux_quality_func_args)]

    def get_num_leaves(self):
        return len(self.all_leaves)

    def print_diagnostics(self):
        '''
        Mostly just for debugging
        '''

        print("-------------- Start Diagnostics --------------")

        print(f"Total number of nodes: {len(self.all_nodes)}")
        print(f"Number of leaves: {len(self.all_leaves)}")
        print(f"Number of children of root: {len(self.root.children)}")
        print(f"Max depth: {max(node.depth for node in self.all_nodes)}")
        print(f"Dead (fully grown) nodes: {sum(not node.expandable for node in self.all_nodes)}")

        print("--------------- End Diagnostics ---------------")
