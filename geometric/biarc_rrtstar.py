from pathlib import Path
import matplotlib.pyplot as plt
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
import numpy as np
from numba import jit, objmode
import random
import geometric.curve_fitting as cf
import geometric.helper as helper
import geometric.globals as G


# cpu_device = jax.devices('cpu')[0]
# tube_sdf = jax.jit(tube_sdf, device=cpu_device)
# res = tube_sdf(jax.numpy.array([1, 1]))
# print('RES', res)
##############################
#   COLLISION CHECKER AND HELPERS
##############################


@jit(nopython=True, fastmath=True, cache=True)
def tube_sdf_np(px, py):
    """
    SDF for a tube, 1000 long, 500 high, following a sine wave with a full period over the 1000 (2pi)
    
    And the sdf is the distance from a hypothetical 100 diameter tube around the wall (so positive in the tube, negative outside).
    """
    
    
    def distance_to_arc(ppx, ppy, centerx, centery, rad, min_ang, max_ang):
        """
        Compute the distance from a point (ppx, ppy) to an arc centered at (centerx, centery) with radius rad.
        The arc is a full circle, so we can use the standard distance formula.
        """
        
        dist = np.hypot(ppx - centerx, ppy - centery)
        dist = np.abs(dist - rad)
        
        ang = np.arctan2(ppy - centery, ppx - centerx)
        dist = np.where((ang < min_ang - 0.1) | (ang > max_ang + 0.1), dist + 9999, dist)
        return dist
    
    
    dist_to_arc0 = distance_to_arc(px, py, 800, 0, 800, np.pi, 0)
   
    return 0.5 * (80 - dist_to_arc0) + 5000

# collision checker
@jit(nopython=True, fastmath=True)
def _isStateValid(x, y, obstacles, obstacle_type):
    # if x < 0 or x > G.bound_x or y < 0 or y > G.bound_y:
    #     return False
    # print()
    # if obstacle_type == "circle":
    #     for (ox, oy, r) in obstacles:
    #         if (x - ox)**2 + (y - oy)**2 <= (r + G.vine_thickness)**2:
    #             return False
    if obstacle_type == "box" or obstacle_type == "tube":
        for i in range(obstacles.shape[0]):
            A = obstacles[i, 0]
            B = obstacles[i, 1]
            if checkBoxCollision(x, y, A[0], A[1], B[0], B[1]):
                return False
    if obstacle_type == "tube":
        if tube_sdf_np(x, y).item() < 0:
            return False

    return True

def isStateValid(state):
    x = state.getX()
    y = state.getY()
    
    obstacles = np.array(G.obstacles)
    obstacle_type = G.obstacle_type
    return _isStateValid(x, y, obstacles, obstacle_type)

# check if 3 points are in a counter clockwise orientation
def ccw(P, Q, R):
    return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])

# true if IN COLLISION
def checkLineCollision(x1, y1, x2, y2):
    C = (x1, y1)
    D = (x2, y2)
    obstacles = np.array(G.obstacles)
    for i in range(obstacles.shape[0]):
        A = obstacles[i, 0]
        B = obstacles[i, 1]
        if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
            return True
    return False

@jit(nopython=True, fastmath=True, cache=True)
def checkBoxCollision(x, y, left, top, right, bottom):
    return left <= x <= right and bottom <= y <= top

# custom goal region so angle doesnt matter
# DOESNT FRICKING WORK SO ITS A PROBOLEM FOR LATER
class customGoalRegion(ob.GoalSampleableRegion):
    def __init__(self, si, goalX, goalY, threshold=1.0):
        super().__init__(si)
        self.goalX = goalX
        self.goalY = goalY
        self.threshold = threshold
        self.spaceInfo_ = si

    def isSatisfied(self, state, distance=None):
        x = state.getX()
        y = state.getY()
        dist = np.hypot(x - self.goalX, y - self.goalY)
        if distance is not None:
            distance[0] = dist
        return (dist <= self.threshold)

    def canSample(self):
        return True

    def maxSampleCount(self):
        return 1

    def sampleGoal(self, state):
        state.setX(self.goalX)
        state.setY(self.goalY)
        yaw = 0.0
        state.setYaw(yaw)


# custom steering
class ArcMotionValidator(ob.MotionValidator):
    def __init__(self, si):
        super().__init__(si)
        self.si_ = si

    def checkMotion(self, source, dest, lastValidFraction=None):
        arc, isvalid = helper.canShortcut(source, dest, self.si_)
        valid = False
        if isvalid:
            valid, _ = checkDiscretizedArc(arc, self.si_)
        if lastValidFraction is not None:
            lastValidFraction.second = 1.0 if valid else 0.0
        return valid

# custom rewiring for rrtstar
class EdgeCountObjective(ob.OptimizationObjective):
    def __init__(self, si):
        super().__init__(si)
        self.si_ = si

    def motionCost(self, s1, s2):
        parent_pose = np.array([s1.getX(), s1.getY(), s1.getYaw()])
        child_pose = np.array([s2.getX(), s2.getY(), s2.getYaw()])

        arc, isvalid = cf.fit_bi_arc(parent_pose, child_pose,
                                     max_arc_length=1000,
                                     max_arc_angle=np.pi)
        if not isvalid or arc is None:
            return ob.Cost(1e6)
        cost = 1
        if G.shrink is not None:
            _, shr = checkDiscretizedArc(arc, self.si_, True)
            if shr:
                cost -= 0.1
        arcLength = arc.arc_lengths[0] + arc.arc_lengths[1]
        tie_factor = 0.01 / ( 0.5 * (G.bound_x + G.bound_y) )
        cost += tie_factor * arcLength
        return ob.Cost(cost)

    def stateCost(self, state):
        # States themselves have no intrinsic cost
        return ob.Cost(0.0)
    
@jit(nopython=True, fastmath=True, cache=True)
def halton(step, last_val):
    """
    Generate the next value in a Halton sequence.
    
    So if your range was 0-1 then you'd start with halton(1.0, -0.5)
    
    Args:
        step: The step size for the sequence.
        last_val: The last value generated in the sequence.
        
    Returns:
        The next step
        The next value in the Halton sequence.
        
    """
    next_val = (last_val + step) % 1.0
        
    # Bool indicating if we wrapped around
    did_wrap = (next_val <= last_val)
    
    # If we wrapped around, then step *= 0.5, else step *= 1
    step *= 1 - 0.5 * did_wrap
    
    # If wrapped, divide the next value by 2
    next_val *= 1 - 0.5 * did_wrap
        
    return step, next_val

# steering function
@jit(nopython=True, fastmath=True, cache=True)
def _checkDiscretizedArc(R1, R2, L1, L2, arc_angle1, arc_angle2, init_pose1, init_pose2, fin_pose1, fin_pose2, 
                         obstacles, obstacle_type, min_r):
    def checkOneArc(R, L, arc_angle, init_pose, fin_pose, min_r):
        if R < min_r:
            return False
        
        if np.abs(L) < 1e-3:
            return True
        
        chord_mid = 0.5 * (init_pose[:2] + fin_pose[:2])
        perp = np.arctan2(-1 * (fin_pose[0] - init_pose[0]),
                               (fin_pose[1] - init_pose[1]))
        d = R * np.cos(arc_angle / 2.0)
        
        C1 = (chord_mid[0] + d * np.cos(perp), chord_mid[1] + d * np.sin(perp))
        C2 = (chord_mid[0] - d * np.cos(perp), chord_mid[1] - d * np.sin(perp))

        center = C2 if arc_angle > 0 else C1
        theta0 = np.arctan2(init_pose[1] - center[1], init_pose[0] - center[0])
        
        # print('Validating arc with theta0:', theta0, 'R:', R)

        # prev_x, prev_y = init_pose[:2]
        
        # shr = None
        
        # Set up halton
        halton_step = 1.0
        halton_val = -0.5
        
        for stp in range(np.abs(L) // 0.1):
            # Get next check angle
            halton_step, halton_val = halton(halton_step, halton_val)
            theta = theta0 + arc_angle * halton_val
                        
            # Convert angle to point
            x = center[0] + R * np.cos(theta)
            y = center[1] + R * np.sin(theta)
            
            # Check point collision
            if not _isStateValid(x, y, obstacles, obstacle_type):
                return False
            
            # if G.obstacle_type == 'line' and checkLineCollision(prev_x, prev_y, x, y):
            #     return False, None
            
            # if shrinkcheck:
            #     for (A, B) in G.unshrunk_ob:
            #         if checkBoxCollision(x, y, A[0], A[1], B[0], B[1]):
            #             shr = True
            #             break
            # prev_x, prev_y = x, y
        return True
    
    if not checkOneArc(R1, L1, arc_angle1, init_pose1, fin_pose1, min_r):
        return False
    if not checkOneArc(R2, L2, arc_angle2, init_pose2, fin_pose2, min_r):
        return False
    return True

def checkDiscretizedArc(biarc, si, shrinkcheck=False):
    R1 = biarc.radii_of_curvature[0]
    L1 = biarc.arc_lengths[0]
    arc_angle1 = biarc.arc_angles[0]
    init_pose1 = biarc.initial_poses[0]
    fin_pose1  = biarc.final_poses[0]
    
    R2 = biarc.radii_of_curvature[1]
    L2 = biarc.arc_lengths[1]
    arc_angle2 = biarc.arc_angles[1]
    init_pose2 = biarc.initial_poses[1]
    fin_pose2  = biarc.final_poses[1]
    
    obstacles = np.array(G.obstacles)
    obstacle_type = G.obstacle_type
    
    return _checkDiscretizedArc(R1, R2, L1, L2, arc_angle1, arc_angle2,
                                init_pose1, init_pose2, fin_pose1, fin_pose2, 
                                obstacles, obstacle_type, G.min_r), None

# smoothing function
def smoothBiArcPath(originalPath, si):
    expanded = helper.expandBiArcPath(originalPath, si)
    N = len(expanded)
    dp = np.array(range(N))
    for i in range(N - 1):
        stateA = expanded[i]
        stateB = expanded[i+1]
        parent_pose = np.array([stateA.getX(), stateA.getY(), stateA.getYaw()])
        child_pose  = np.array([stateB.getX(), stateB.getY(), stateB.getYaw()])
        arc = cf.fit_circular_arc(
            parent_pose, child_pose,
            max_arc_length=1000,
            max_arc_angle=np.pi
        )
        #if arc is not None and isvalid:
        if arc is not None and hasattr(arc, 'arc_lengths'):
            dp[i+1] += arc.arc_lengths
    prev = np.array(range(N)) - 1
    arcs = [None] * N
    dp[0] = 0
    tie_factor = 0.01 / ( 0.5 * (G.bound_x + G.bound_y) )
    for j in range(1, N):
        for i in range(j):
            arc, valid = helper.canShortcut(expanded[i], expanded[j], si)
            if arc is not None:
                col, shr = checkDiscretizedArc(arc, si, G.shrink is not None)
            if valid and col:
                arcLength = arc.arc_lengths[0] + arc.arc_lengths[1]
                cost = dp[i] + 2 + tie_factor * arcLength
                if G.shrink is not None:
                    if shr:
                        cost -= 0.1
                if cost < dp[j]:
                    dp[j] = cost
                    prev[j] = i
                    arcs[j] = arc
    final_path = []
    idx = N-1
    while idx != -1:
        final_path.append(expanded[idx])
        if idx - prev[idx] != 1:
            final_path.append(helper.arrayToState(arcs[idx].final_poses[0], si))
        idx = prev[idx]

    final_path.reverse()
    return helper.statesToPath(final_path, si)


##############################
#   setup
##############################

# setup space information for the planner
def createSpaceInformation():
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0)
    bounds.setHigh(0, G.bound_x)
    bounds.setHigh(1, G.bound_y)
    space.setBounds(bounds)
    
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    si.setMotionValidator(ArcMotionValidator(si))
    si.setup()
    return si

def getBiarcPoints(biarc, step_size=10000):
    """
    Returns a list of (x, y, theta) samples along the two arcs of a BiArc.
    At least 1 point per sub-arc is returned.
    Uses the same geometry as checkDiscretizedArc (but no collision checks).
    """
    points = []
    for i in range(2):
        R = biarc.radii_of_curvature[i]
        L = biarc.arc_lengths[i]
        arc_angle = biarc.arc_angles[i]

        num_steps = max(2, int(np.ceil(L / step_size)))

        init_pose = biarc.initial_poses[i]
        fin_pose  = biarc.final_poses[i]

        chord_mid = 0.5*(init_pose[:2]+fin_pose[:2])
        perp = np.arctan2(-(fin_pose[0]-init_pose[0]),
                          (fin_pose[1]-init_pose[1]))
        d = R*np.cos(arc_angle/2.0)
        C1 = (chord_mid[0]+ d*np.cos(perp), chord_mid[1]+ d*np.sin(perp))
        C2 = (chord_mid[0]- d*np.cos(perp), chord_mid[1]- d*np.sin(perp))

        center = C2 if arc_angle>0 else C1
        theta0 = np.arctan2(init_pose[1]-center[1], init_pose[0]-center[0])
        
        # Include the end pose
        for th in np.linspace(theta0, theta0 + arc_angle, num_steps):
            x = center[0] + R*np.cos(th)
            y = center[1] + R*np.sin(th)
            # Here we keep the same heading convention as the collision-check code:
            # "th" is the angle from the center to the point.
            points.append((x, y, th - np.pi/2))
        
    # Assert first point is the same as the initial pose and
    # last point is the same as the final pose
    assert np.isclose(points[0][0:2], biarc.initial_poses[0][0:2]).all()
    assert np.isclose(points[-1][0:2], biarc.final_poses[1][0:2]).all()
        
    # Plot in matplot lib:
    # red start blue end pose
    # colors = np.linspace(0, 1, len(points))
    # plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors, cmap='RdBu', marker='.')
    
    # plt.axis('equal')
    # plt.pause(0.0001)
        
    return points

def extract_points_and_costs(ob, si, planner):
    """
    Returns:
        all_arcs_points: A list of (x, y, theta) samples from all arcs
        point_costs:     A list of integer costs, parallel to all_arcs_points,
                         where cost is the number of edges from some start
                         to that arc's child.
    """
    pd = ob.PlannerData(si)
    planner.getPlannerData(pd)

    # --------- BFS to assign costs to each vertex ---------
    from collections import deque
    num_vertices = pd.numVertices()
    costs = [None] * num_vertices

    # Initialize queue with all start vertices
    queue = deque()
    for s in range(pd.numStartVertices()):
        start_idx = pd.getStartIndex(s)
        costs[start_idx] = 0
        queue.append(start_idx)

    # BFS
    while queue:
        v = queue.popleft()
        cv = costs[v]  # parent's cost
        edge_list = ou.vectorUint()
        pd.getEdges(v, edge_list)
        for w in edge_list:
            if costs[w] is None:
                costs[w] = cv + 1
                queue.append(w)

    # -------- Collect all arcs (parent->child) along with child's cost --------
    arcs_with_costs = []
    for i in range(num_vertices):
        edge_list = ou.vectorUint()
        pd.getEdges(i, edge_list)

        # Parent state's pose
        stParent = pd.getVertex(i).getState()
        px, py, pt = stParent.getX(), stParent.getY(), stParent.getYaw()
        
        # Get each child's pose and cost
        for child_idx in edge_list:
            stChild = pd.getVertex(child_idx).getState()
            cx, cy, ct = stChild.getX(), stChild.getY(), stChild.getYaw()

            parent_pose = np.array([px, py, pt])
            child_pose  = np.array([cx, cy, ct])

            arc, isvalid = cf.fit_bi_arc(
                parent_pose, 
                child_pose,
                max_arc_length=1000,
                max_arc_angle=np.pi
            )
            if arc is not None and isvalid:
                # We'll store the arc together with the child's "cost"
                arcs_with_costs.append((arc, costs[child_idx]))

    # --------- Sample points along each arc, attaching child's cost --------
    all_arcs_points = []
    point_costs     = []
    for i, (arc, c) in enumerate(arcs_with_costs):
        arc_pts = getBiarcPoints(arc, step_size=200) 
        all_arcs_points.extend(arc_pts)
        point_costs.extend([c] * len(arc_pts))
        
        if i % 1000 == 0:
            print(f"   did {i}/{len(all_arcs_points)}")

    return all_arcs_points, point_costs
                
def main(env, time, thresh, save=None, vine_thickness=0.1, save_points_path=None, plan_goal_to_start=False,
            min_r=0.001):
    '''
    Does the geometic plan and saves points to save_points_path,
    however, points_costs will always be saved to point_costs.npy
    '''
    helper.loadEnvironment(env)
        
    if plan_goal_to_start:
        G.start, G.goal = G.goal, G.start
    
    G.vine_thickness = vine_thickness
    G.min_r = min_r
    start_x, start_y, _ = G.start
    start_angles = np.radians(np.arange(-180, 180, 30))
    si = createSpaceInformation()
    pdef = ob.ProblemDefinition(si)
    
    for angle in start_angles:
        startState = ob.State(si.getStateSpace())
        startState[0] = start_x
        startState[1] = start_y
        startState[2] = angle
        pdef.addStartState(startState)

    goalState = ob.State(si.getStateSpace())
    goalState[0], goalState[1], goalState[2] = G.goal

    # pdef.addStartState(startState)
    # goalRegion = customGoalRegion(si, G.goal[0], G.goal[1], threshold=thresh)
    # pdef.setGoal(goalRegion)
    pdef.setGoalState(goalState, threshold=thresh)


    pdef.setOptimizationObjective(EdgeCountObjective(si))

    planner = og.RRTstar(si)
    # planner = og.PRMstar(si)
    # planner.setRange(1000)
    # planner = og.EITstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    status = planner.solve(time)
    if status:
        print("Found solution!")
    else:   
        print("No solution found!")
        
    # originalPath = helper.clonePath(pdef.getSolutionPath(), si)
    # smoothedPath = smoothBiArcPath(originalPath, si)
    pdef.clearSolutionPaths()
    # pdef.addSolutionPath(originalPath)
    helper.plotRRTAndBothPaths(planner, pdef, si, None, None)
        
    if save is not None:
        helper.saveData(save, planner, pdef, si)
    
    # TODO dont repeat this sampling
    if save_points_path is not None:
        arc_points, point_costs = extract_points_and_costs(ob, si, planner)
        
        # Filter nan costs
        arc_points = np.asarray(arc_points)
        point_costs = np.asarray(point_costs)
        finite_mask = np.isfinite(point_costs)
        arc_points = arc_points[finite_mask]
        point_costs = point_costs[finite_mask]
        
        print(f'Generated {finite_mask.size} points with {finite_mask.sum()} being non-nan')
        
        assert finite_mask.size > 0, 'No valid points generated, not saving'
        
        # save_points_path is something like foo/bar/costs_blah_blah.npy
        # costs_path should be the same foo/bar/costs_blah_blah_costs.npy
        np.save(save_points_path, arc_points)
        costs_path = save_points_path.replace('.npy', '_costs.npy')
        np.save(costs_path, point_costs)
        print(f"Saved points to {save_points_path} and costs to {costs_path}")
        
    # print("original: green, smoothed: black")

# instructions:
# set G.shrink = {factor to multiply by vine thickness}
# to activate shallow penetration collison bias
# put save folder as function input to main to save tree and paths 
if __name__ == "__main__":
    thresh = 0.2
    time = 60.0 * 1
    thickness = 0.1
    env = "envs/env_plus.txt"
    # G.shrink = 3.0
    main(env, time, thresh, save_points_path='cache/points.npy', plan_goal_to_start=True,
    min_r = 0)
    #main(env, time, thresh, save)
