import math
import time
import numpy as np

from kinodynamic.nearest import nearest_neighbor_all

'''
A greedy algorithm to find the maximum subset of points,
where all points are at least δs apart from each other.
'''

# @njit(cache=True)
# Faster without
def max_cover_pure_part(δs, dist2, num_points, remove_fraction=0.1):
    # Track the valid points as they are executed
    valid_ones = np.ones(num_points, dtype=np.bool_)
    
    for iter in range(10000):
        # print('iter', iter, 'valid_ones', valid_ones.sum())
        # Recompute the intersections, per-COLUMN
        intersections = np.sum(dist2 < δs * δs, axis=0)
        
        # Do not count the intersections of the ones we removed
        intersections[~valid_ones] = -1
        
        # Return if there are no more intersections
        if np.all(intersections[valid_ones] <= 1): # One because it will always intersect itself
            break
        
        # Greedily remove the N points with most intersections
        num_remove = int(np.sum(valid_ones) * remove_fraction)
        num_remove = max(1, num_remove)
        idx_to_remove = np.argpartition(intersections, -num_remove)[-num_remove:]
        valid_ones[idx_to_remove] = False
        
        # Set all the rows with the greedy index to inf, so it won't be counted again
        dist2[idx_to_remove] = np.inf
    
    return valid_ones

def max_cover(params, points):
    """
    Find a subset of points that are all at least params.δs apart from each other.
    Uses a greedy approximate algorithm -- may return a smaller than optimal 
    cover but never an invalid one.
    
    Parameters
    ----------
    params : SSTParams
        Contains rotation distance metric and δs
    points : np.ndarray
        (N, 3) array of points [x, y, theta]
    
    Returns
    -------
    indexes : np.ndarray
        {M,} array of indexes of the points that are kept
    """
    indices, dist2 = nearest_neighbor_all(params, points, points)
    
    return max_cover_pure_part(params.δs, dist2, points.shape[0])

if __name__ == "__main__":
    δs = 20.0
    
    points = np.load('dedup_points.npy') # (N, 3) [x, y, theta]

    print(points.shape)
    
    # Plot using matplotlib a circle of radius 20 around each point
    import matplotlib.pyplot as plt
    
    class Object(object):
        pass
    params = Object()
    params.rotation_metric_scale = 10
    params.δs = δs
    
    start_time = time.time()
    num_trials = 1
    for _ in range(num_trials):
        keep_idx = max_cover(params, points)
    print('Avg set cover time', (time.time() - start_time) / num_trials)
    
    # Plot the old points as blue
    for x, y, theta in points:
        circle = plt.Circle((x, y), δs, color='lightblue', fill=False, linewidth=1)
        plt.gca().add_artist(circle)
        
    # Draw the new points as green
    for x, y, theta in points[keep_idx]:
        circle = plt.Circle((x, y), δs, color='g', fill=False, linewidth=3)
        plt.gca().add_artist(circle)
        
    plt.xlim(-50, 200)
    plt.ylim(-0, 200)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(0.001)
    
    plt.show()

