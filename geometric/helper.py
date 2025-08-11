import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
from . import globals as G
from . import curve_fitting as cf

# load from env file
def loadEnvironment(env_path):
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file '{env_path}' not found.")

    G.obstacles.clear()
    G.unshrunk_ob.clear()
    with open(env_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        
        if line.startswith("#"): continue
        
        if line.startswith("bound:"):
            G.bound_x, G.bound_y = map(float, line.split(":")[1].strip().split())
        elif line.startswith("start:"):
            G.start = tuple(map(float, line.split(":")[1].strip().split()))
        elif line.startswith("goal:"):
            G.goal = tuple(map(float, line.split(":")[1].strip().split()))
        elif line.startswith("scale:"):
            G.scale = float(line.split(":")[1].strip())
            pass # Nothing
        elif line.startswith("ob_type"):
            G.obstacle_type = line.split(":")[1].strip()
        elif line.startswith("goal_radius:"):
            pass
        elif line.startswith("obstacles:"):
            continue  # actual obstacle lines follow
        else:
            if line and G.obstacle_type == 'circle': 
                ox, oy, r = map(float, line.split())
                G.obstacles.append((ox, oy, r))
            if line and G.obstacle_type == 'line': 
                x1, y1, x2, y2 = map(float, line.split())
                G.obstacles.append([(x1, y1), (x2, y2)])
            if line and (G.obstacle_type == 'box' or G.obstacle_type == 'tube'):
                x1, y1, x2, y2 = map(float, line.split())
                if x1 == x2 or y1 == y2:
                    print("Env problem")
                else:
                    left = min(x1, x2)
                    right = max(x1, x2)
                    top = max(y1, y2)
                    bottom = min(y1, y2)
                    if G.shrink is None:
                        G.obstacles.append([(left, top), (right, bottom)])
                    else:
                        factor = G.vine_thickness * G.shrink
                        G.unshrunk_ob.append([(left, top), (right, bottom)])
                        if right - left > 2 * factor and top - bottom > 2 * factor:
                            if left != 0:
                                left += factor
                            if right != G.bound_x:
                                right -= factor
                            if top != G.bound_y:
                                top -= factor
                            if bottom != 0:
                                bottom += factor
                            G.obstacles.append([(left, top), (right, bottom)])
                        else:
                            print("box too thin to shrink")

    # G.obstacles = np.asarray(G.obstacles, dtype=np.float32)
    # G.unshrunk_ob = np.asarray(G.unshrunk_ob, dtype=np.float32)
    s = G.scale
    G.obstacles = [ [ (a * s, b * s), (c * s, d * s) ] for (a, b), (c, d) in G.obstacles ]
    
    # print("Obstacles loaded:", G.obstacles)
    # G.unshrunk_ob = G.unshrunk_ob * G.scale
    
    G.start = (G.start[0] * s, G.start[1] * s, G.start[2])
    G.goal = (G.goal[0] * s, G.goal[1] * s, G.goal[2])
    
    # G.goal_radius = G.goal_radius * G.scale
    G.bound_x = G.bound_x * s
    G.bound_y = G.bound_y * s
    
    G.scale = 1.0

##############################
#   plotting functions
##############################

# Convert OMPL's PathGeometric to a simple Python list of (x, y, yaw).
def pathToArray(pathGeometric):
    coords = []
    for i in range(pathGeometric.getStateCount()):
        st = pathGeometric.getState(i)
        coords.append((st.getX(), st.getY(), st.getYaw()))
    return coords

# plot biarcs
def plotBiArcPath(pathGeometric, ax, color="orange", linewidth=2.0):
    coords = pathToArray(pathGeometric)
    if len(coords) < 2:
        return
    for i in range(len(coords) - 1):
        parent_pose = np.array(coords[i])
        child_pose  = np.array(coords[i+1])
        arc, isvalid = cf.fit_bi_arc(
            parent_pose, child_pose,
            max_arc_length=100,
            max_arc_angle=np.pi
        )
        if arc is not None and isvalid:
            cf.plot_bi_arc(arc, ax, color=color, linewidth=linewidth)

# plot single arcs for smoothed path
def plotArcPath(pathGeometric, ax, color="orange", linewidth=2.0):
    coords = pathToArray(pathGeometric)
    for i in range(len(coords) - 1):
        parent_pose = np.array(coords[i])
        child_pose  = np.array(coords[i+1])
        arc = cf.fit_circular_arc(
            parent_pose, child_pose,
            max_arc_length=100,
            max_arc_angle=np.pi
        )
        #if arc is not None and isvalid:
        if arc is not None:
            cf.plot_circular_arc(arc, ax, color=color, linewidth=linewidth)

# check if original and smoothed path are the same
def pathsAreSame(path1, path2, si, tol=1e-5):
    path1 = expandBiArcPath(path1, si)
    if len(path1) != len(path2):
        return False
    for (p1, p2) in zip(path1, path2):
        if not (np.allclose([p1.getX(), p1.getY(), p1.getYaw()], p2, atol=tol)):
            return False
    return True

# plot smoothed path which already has the half nodes
def plotPath(path, ax, color, node_size=50, linewidth=2.0):
    plotArcPath(path, ax, color=color, linewidth=linewidth)
    coords = pathToArray(path)
    node_x = [p[0] for p in coords]
    node_y = [p[1] for p in coords]
    ax.scatter(node_x, node_y, color=color, s=node_size, edgecolor='black', zorder=5)

# plot smoothed path which already has the half nodes
def plotPathWithHalfNodes(path, ax, color, node_size=50, halfnode_size=25, linewidth=2.0):
    plotBiArcPath(path, ax, color=color, linewidth=linewidth)
    node_x, node_y = [], []
    half_x, half_y = [], []
    coords = pathToArray(path)

    for i in range(len(coords) - 1):
        parent_pose = np.array(coords[i])
        child_pose = np.array(coords[i+1])
        arc, isvalid = cf.fit_bi_arc(parent_pose, child_pose,
                                     max_arc_length=100,
                                     max_arc_angle=np.pi)
        if arc is not None and isvalid:
            node_x.append(parent_pose[0])
            node_y.append(parent_pose[1])
            center_pose = arc.final_poses[0]
            half_x.append(center_pose[0])
            half_y.append(center_pose[1])
    node_x.append(coords[-1][0])
    node_y.append(coords[-1][1])
    ax.scatter(node_x, node_y, color=color, s=node_size, edgecolor='black', zorder=5)
    ax.scatter(half_x, half_y, color=color, s=halfnode_size, edgecolor='black', zorder=6)

def plotRRTAndBothPaths(planner, pdef, si, originalPath, smoothedPath):
    pd = ob.PlannerData(si)
    planner.getPlannerData(pd)

    vx, vy = [], []
    for i in range(pd.numVertices()):
        st = pd.getVertex(i).getState()
        vx.append(st.getX())
        vy.append(st.getY())

    # regenerate arces for each edge
    edges = []
    for i in range(pd.numVertices()):
        edge_list = ou.vectorUint()
        pd.getEdges(i, edge_list)
        stParent = pd.getVertex(i).getState()
        px, py, pt = stParent.getX(), stParent.getY(), stParent.getYaw()

        for child_idx in list(edge_list):
            stChild = pd.getVertex(child_idx).getState()
            cx, cy, ct = stChild.getX(), stChild.getY(), stChild.getYaw()

            parent_pose = np.array([px, py, pt])
            child_pose = np.array([cx, cy, ct])
            arc, isvalid = cf.fit_bi_arc(parent_pose, child_pose,
                                         max_arc_length=100,
                                         max_arc_angle=np.pi)
            if arc is not None and isvalid:
                edges.append(arc)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Obstacles
    if G.obstacle_type == 'circle':
        for (ox, oy, r) in G.obstacles:
            circle = plt.Circle((ox, oy), r, color='black', alpha=0.8)
            ax.add_patch(circle)
    elif G.obstacle_type == 'line':
        for (p1, p2) in G.obstacles:
            x_values = [p1[0], p2[0]]
            y_values = [p1[1], p2[1]]
            #ax.scatter(x_values, y_values, color='black')
            ax.plot(x_values, y_values, color='black', linewidth=3, alpha=0.8)
    elif G.obstacle_type == 'box':
        for (p1, p2) in G.obstacles:
            x_min, y_max = p1
            x_max, y_min = p2
            width = x_max - x_min
            height = y_max - y_min
            rect = plt.Rectangle((x_min, y_min), width, height, 
                                 color='black', alpha=0.8)
            ax.add_patch(rect)
        if G.shrink is not None:
            for (p1, p2) in G.unshrunk_ob:
                x_min, y_max = p1
                x_max, y_min = p2
                width = x_max - x_min
                height = y_max - y_min
                rect = plt.Rectangle((x_min, y_min), width, height, 
                                    color='red', alpha=0.5)
                ax.add_patch(rect)

    print('Plotting', len(edges), 'edges and', len(vx), 'vertices')
    
    # Plot tree
    for arc in edges:
        cf.plot_bi_arc(arc, ax, linewidth=0.8, color='gray', alpha=0.5)
    ax.scatter(vx, vy, color='blue', s=15, label='RRT vertices', alpha=0.5)

    # Same Path Check
    # same = pathsAreSame(originalPath, pathToArray(smoothedPath), si)
    # if same:
    #     print("✅ Original and smoothed paths are IDENTICAL.")
    #     print("Length: ", len(pathToArray(originalPath)) * 2 - 1)
    # else:
    #     print("🔄 Original and smoothed paths differ.")
    #     print("Original: ", len(pathToArray(originalPath)) * 2 - 1)
    #     print("Smoothed: ", len(pathToArray(smoothedPath)))

    # plot solution
    # plotPath(smoothedPath, ax, 'black', linewidth = 3)
    # plotPathWithHalfNodes(originalPath, ax, 'yellowgreen')
    # special effects for start and goal nodes
    start = pdef.getStartState(0)
    goal = pdef.getGoal().getState()
    ax.scatter(start.getX(), start.getY(), color='lime', s=100, edgecolors='black', zorder=10, label='Start')
    ax.scatter(goal.getX(), goal.getY(), color='red', s=100, edgecolors='black', zorder=10, label='Goal')

    # Final formatting
    ax.set_xlim(0, G.bound_x)
    ax.set_ylim(0, G.bound_y)
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_title("RRT Solution")
    plt.savefig("solution.png", dpi=150)
    print("Saved figure to solution.png")

##############################
#   PATH SMOOTHING HELPERS
##############################
def clonePath(path, si):
    cloned = og.PathGeometric(si)
    for i in range(path.getStateCount()):
        src = path.getState(i)
        clone = si.allocState()
        clone.setX(src.getX())
        clone.setY(src.getY())
        clone.setYaw(src.getYaw())
        cloned.append(clone)
    return cloned

def arrayToState(arr, si):
    s = si.allocState()
    s.setX(arr[0])
    s.setY(arr[1])
    s.setYaw(arr[2])
    return s

def statesToPath(stateList, si):
    path = og.PathGeometric(si)
    for st in stateList:
        path.append(st)
    return path

def canShortcut(stateA, stateB, si):
    parent_pose = np.array([stateA.getX(), stateA.getY(), stateA.getYaw()])
    child_pose  = np.array([stateB.getX(), stateB.getY(), stateB.getYaw()])
    arc, valid = cf.fit_bi_arc(parent_pose, child_pose,
                               max_arc_length=100,
                               max_arc_angle=np.pi)
    return arc, valid

# add intermediary nodes
def expandBiArcPath(pathGeometric, si):
    states = []
    coords = pathToArray(pathGeometric)
    for idx in range(len(coords) - 1):
        p_parent = np.array(coords[idx])
        p_child  = np.array(coords[idx+1])
        arc, valid = cf.fit_bi_arc(p_parent, p_child,
                                   max_arc_length=9999, max_arc_angle=np.pi)
        if arc is None or not valid:
            raise ValueError("BiArc not feasible for consecutive states!") 
        center_pose = arc.final_poses[0]
        if idx == 0:
            s_parent = arrayToState(p_parent, si)
            states.append(s_parent)
        s_center = arrayToState(center_pose, si)
        states.append(s_center)
        s_child = arrayToState(p_child, si)
        states.append(s_child)
    return states

##############################
#   TREE STORAGE HELPERS
##############################
def treeToArray(planner, si):
    pd = ob.PlannerData(si)
    planner.getPlannerData(pd)
    vertices = []
    edges = []
    for v in range(pd.numVertices()):
        state = pd.getVertex(v).getState().asSE2State()
        x, y, yaw = state.getX(), state.getY(), state.getYaw()
        vertices.append([x, y, yaw])

        for i in range(pd.numEdges(v)):
            edge = pd.getEdge(v, i)
            edges.append([v, edge])
    return np.array(vertices), np.array(edges)

def saveData(planner, si, opath, spath, filename):
    vertices, edges = treeToArray(planner, si)
    op = np.array(pathToArray(opath))
    sp = np.array(pathToArray(spath))
    data_to_save = {
        'original_path': op,
        'smoothed_path': sp,
        'tree_vertices': vertices,
        'tree_edges': edges,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

def readData(filename):
    with open('planning_result.pkl', 'rb') as f:
        data = pickle.load(f)
    original_path = data['original_path']
    smoothed_path = data['smoothed_path']
    tree_vertices = data['tree_vertices']
    tree_edges = data['tree_edges']
    return original_path, smoothed_path, tree_vertices, tree_edges