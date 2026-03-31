import numpy as np
from numba import njit, prange
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

def group_trajectories(traj_dict, frechet_threshold=100): # threshold should be in pixels, 0.5 is invalid, test required start with 100
    # group trajectories based on frechet distance, using a threshold to determine if they belong to the same group
    traj_ids = list(traj_dict.keys())
    traj_coords = [traj_dict[tid]['trajectory'] for tid in traj_ids]
    
    n = len(traj_coords)
    groups = []
    visited = set()
    
    for i in range(n):
        if traj_ids[i] in visited:
            continue
        group = [traj_ids[i]]
        visited.add(traj_ids[i])
        for j in range(i+1, n):
            if traj_ids[j] in visited:
                continue
            dist = frechet_distance(traj_coords[i], traj_coords[j])
            if dist < frechet_threshold:
                group.append(traj_ids[j])
                visited.add(traj_ids[j])
        groups.append(group)
    
    return groups

@njit("f8(f8[:, ::1], f8[:, ::1])", fastmath=True)
def frechet_distance_old(true_coords, pred_coords):
    n, m = len(true_coords), len(pred_coords)
    dist = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            d = true_coords[i] - pred_coords[j]
            dist[i, j] = np.sqrt(np.sum(d * d))

    cost = np.empty((n, m))
    cost[0, 0] = dist[0, 0]

    for i in range(1, n):
        cost[i, 0] = max(cost[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        cost[0, j] = max(cost[0, j - 1], dist[0, j])

    for i in range(1, n):
        for j in range(1, m):
            prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = max(prev, dist[i, j])

    return cost[n - 1, m - 1]