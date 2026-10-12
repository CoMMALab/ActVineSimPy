"""
Tests for StateInfo, Node and RRTTree (no simulator needed).

Run from the repo root with:
    python -m pytest xnew_rrt/test_rrt.py -v
"""

import os, sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrt_util import StateInfo, Node, RRTTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(value):
    """Minimal StateInfo with a single numeric 'value' tucked into `state`,
    used as a stand-in for real robot state."""
    return StateInfo(state=value, dstate=0, bodies=None,
                     moveable_obj_pose=None, moveable_obj_dstate=None)


def distance_1d(state_info, query):
    return abs(state_info["state"] - query)


def never_goal(state_info):
    return False


def goal_at(target_value):
    def _goal_test(state_info):
        return state_info["state"] == target_value
    return _goal_test


# ---------------------------------------------------------------------------
# StateInfo
# ---------------------------------------------------------------------------

class TestStateInfo:

    def test_getitem_matches_attributes(self):
        s = StateInfo(state=1, dstate=2, bodies=3,
                      moveable_obj_pose=4, moveable_obj_dstate=5, bend_target=6)
        assert s["state"] == 1
        assert s["dstate"] == 2
        assert s["bodies"] == 3
        assert s["moveable_obj_pose"] == 4
        assert s["moveable_obj_dstate"] == 5
        assert s["bend_target"] == 6

    def test_getitem_unknown_key_raises(self):
        with pytest.raises(KeyError):
            make_state(1)["not_a_real_key"]


# ---------------------------------------------------------------------------
# RRTTree
# ---------------------------------------------------------------------------

class TestTree:

    def test_root(self):
        tree = RRTTree(make_state(42), never_goal)
        assert isinstance(tree.root, Node)
        assert tree.root.info["state"] == 42
        assert tree.root.parent is None
        assert tree.root.depth == 0
        assert tree.all_nodes == [tree.root]
        assert tree.all_leaves == {tree.root}

    def test_trees_do_not_share_bookkeeping(self):
        a = RRTTree(make_state(0), never_goal)
        a.add_node(a.root, make_state(1))
        b = RRTTree(make_state(0), never_goal)
        assert len(a.all_nodes) == 2
        assert len(b.all_nodes) == 1
        assert b.root.node_id == 0

    def test_add_node_links_parent_and_child(self):
        tree = RRTTree(make_state(0), never_goal)
        child = tree.add_node(tree.root, make_state(1), control=0.5, edge=["s"])
        assert tree.root.children == [child]
        assert child.parent is tree.root
        assert child.depth == 1
        assert child.control == 0.5 and child.edge == ["s"]

    def test_leaves_are_tracked(self):
        tree = RRTTree(make_state(0), never_goal)
        a = tree.add_node(tree.root, make_state(1))
        b = tree.add_node(tree.root, make_state(2))
        c = tree.add_node(a, make_state(3))
        assert tree.all_leaves == {b, c}
        assert tree.get_num_leaves() == 2

    def test_add_edge_returns_none_unless_goal(self):
        tree = RRTTree(make_state(0), goal_at(2))
        assert tree.add_edge(tree.root, make_state(1)) is None
        path = tree.add_edge(tree.root.children[0], make_state(2))
        assert [node.info["state"] for node in path] == [0, 1, 2]
        assert tree.goal_node is path[-1]

    def test_goal_test_gets_aux_args(self):
        tree = RRTTree(make_state(0), lambda s, target: s["state"] == target, (7,))
        assert tree.add_edge(tree.root, make_state(7)) is not None

    def test_find_nearest(self):
        tree = RRTTree(make_state(0), never_goal)
        for value in (10, 3, -5):
            tree.add_node(tree.root, make_state(value))
        assert tree.find_nearest_to(4, distance_1d).info["state"] == 3

    def test_find_nearest_skips_dead_nodes(self):
        tree = RRTTree(make_state(0), never_goal)
        dead = tree.add_node(tree.root, make_state(3))
        dead.expandable = False
        assert tree.find_nearest_to(4, distance_1d) is tree.root
        tree.root.expandable = False
        assert tree.find_nearest_to(4, distance_1d) is None

    def test_closest_paths_sorted_and_capped(self):
        tree = RRTTree(make_state(0), never_goal)
        for value in (10, 3, -5):
            tree.add_node(tree.root, make_state(value))
        paths = tree.get_closest_paths(2, lambda node, target: abs(node.info["state"] - target), (4,))
        assert [[n.info["state"] for n in p] for p in paths] == [[0, 3], [0, 10]]
        assert len(tree.get_closest_paths(99, lambda node: 0.0)) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
