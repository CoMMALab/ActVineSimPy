
from kinodynamic.rotate_math import wrap_rotation
import numpy as np

'''
nearest neighbor stuff
'''

# from pykeops.numpy import LazyTensor

# def knn_search(ref_points: np.ndarray, query_points: np.ndarray, K: int):
#     """
#     Perform K-nearest neighbors search from query_points to ref_points
#     using brute force with pykeops.

#     Args:
#         ref_points  : shape (N, D), reference dataset
#         query_points: shape (M, D), query points
#         K           : number of neighbors to retrieve

#     Returns:
#         indices : np.array of shape (M, K) with the neighbor indices in ref_points
#         dist2   : np.array of shape (M, K) of squared distances to those neighbors
#     """

#     # We create LazyTensor for (M,1,D) and (1,N,D)
#     # Distances are computed using the symbolic expression squared norm
#     X_j = LazyTensor(ref_points[None, :, :])   # shape: (1, N, D)
#     G_i = LazyTensor(query_points[:, None, :]) # shape: (M, 1, D)
#     D_ij = ((G_i - X_j) ** 2).sum(dim=2)       # shape: (M, N)

#     # argKmin(K, dim=1) returns the K minimal distances (and their indices)
#     indices = D_ij.argKmin(K, dim=1)           # shape: (M, K)
#     dist2 = D_ij[indices]                       # shape: (M, K)

#     return indices, dist2

# def knn_search_radius(ref_points: np.ndarray, query_points: np.ndarray, radius: float, cap: int):
#     """
#     Perform radius search from query_points to ref_points
#     using brute force with pykeops.

#     Args:
#         ref_points  : shape (N, D), reference dataset
#         query_points: shape (M, D), query points
#         radius      : radius of the search
        
#     Returns:
#         indices : np.array of shape (M, K) with the neighbor indices in ref_points
#         dist2   : np.array of shape (M, K) of squared distances to those neighbors
#     """
    
#     # We create LazyTensor for (M,1,D) and (1,N,D)
#     # Distances are computed using the symbolic expression squared norm
#     X_j = LazyTensor(ref_points[None, :, :])   # shape: (1, N, D)
#     G_i = LazyTensor(query_points[:, None, :]) # shape: (M, 1, D)
#     D_ij = ((G_i - X_j) ** 2).sum(dim=2)       # shape: (M, N)

#     mask = D_ij <= radius ** 2
#     D_ij[~mask] = 1e9
    
#     # argKmin(K, dim=1) returns the K minimal distances (and their indices)
#     indices = D_ij.argKmin(K, dim=1)           # shape: (M, K)
#     dist2 = D_ij[indices]                       # shape: (M, K)
    
#     return indices, dist2

def distance(sst_params, diff):
    """
    Return the rotation-scaled squared distance given the difference in delta
    
    Last dim in diff should be size 3 for (x, y, theta)
    """
    # Compute shortest rotation for the theta component [-pi, pi)
    rot_delta = wrap_rotation(diff[..., 2])
    
    # Scale rotation
    rot_delta *= sst_params.rotation_metric_scale
    
    # Compute pairwise distances
    dist2 = diff[..., 0] ** 2 + diff[..., 1] ** 2 + rot_delta ** 2
    
    return dist2    
    
    
def nearest_neighbor(sst_params, ref_points: np.ndarray, query_points: np.ndarray):
    """
    Perform nearest neighbor search from query_points to ref_points
    using brute force with numpy.

    Args:
        ref_points  : shape (N, 3), reference dataset (x, y, theta)
        query_points: shape (M, 3), query points      (x, y, theta)
        
        Thetas may be any radian value and require wrapping

    Returns:
        indices : np.array of shape (M,) with the neighbor indices in ref_points
        dist2   : np.array of shape (M,) of squared distances to those neighbors
    """
    # TODO scale theta with xy
    assert ref_points.shape[1] == query_points.shape[1], f"ref_points shape: {ref_points.shape}, query_points shape: {query_points.shape}"

    # Compute raw differences Shape (M, N, 3)
    diff = query_points[:, None] - ref_points
    
    dist2 = distance(sst_params, diff)
    
    # Find the nearest neighbor indices
    indices = np.argmin(dist2, axis=1)
    
    return indices, np.take_along_axis(dist2, indices[:, None], axis=1).flatten()

def nearest_neighbor_all(sst_params, ref_points: np.ndarray, query_points: np.ndarray):
    """
    Compute all neighbor distances from query_points to ref_points
    using brute force with numpy.

    Args:
        ref_points  : shape (N, 3), reference dataset (x, y, theta)
        query_points: shape (M, 3), query points      (x, y, theta)
        
        Thetas may be any radian value and require wrapping.

    Returns:
        indices : np.array of shape (M, N) with the neighbor indices in ref_points
        dist2   : np.array of shape (M, N) of squared distances to those neighbors
    """
    assert ref_points.shape[1] == query_points.shape[1], f"ref_points shape: {ref_points.shape}, query_points shape: {query_points.shape}"
    
    # Compute raw differences Shape (M, N, 3)
    diff = query_points[:, None] - ref_points
    
    dist2 = distance(sst_params, diff)
    
    # Find all the indices
    indices = np.arange(ref_points.shape[0])
    
    return indices, dist2