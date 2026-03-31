import math
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import gaussian_kde

def get_all_points(trajectories):
    """
    Flattens all trajectory point lists into a single NumPy array.

    This function takes a dictionary where each value is a list (or array) of points 
    representing individual trajectories, and returns a single NumPy array containing 
    all points across all trajectories.

    Parameters:
        trajectories (dict): A dictionary where each value is a list or array of (x, y) points.

    Returns:
        np.ndarray: A NumPy array of shape (N, 1, 2) or (N, 2), depending on input, 
                    containing all points from all trajectories.
    """
    all_points = []
    for points in trajectories.values():
        all_points.extend(points)
    return np.array(all_points)

def interpolate_trajectory(trajectory, num_points=100, window_length=20, polyorder=2):
    # Separate the coordinates into x and y arrays
    x_original = np.array([coord[0][0] for coord in trajectory])
    y_original = np.array([coord[0][1] for coord in trajectory])
    extra = np.array([coord[1] for coord in trajectory])
    
    # Apply Savitzky-Golay filter to smooth the x and y coordinates
    x_smoothed = savgol_filter(x_original, window_length, polyorder)
    y_smoothed = savgol_filter(y_original, window_length, polyorder)

    # Generate new time steps with exactly 'num_points' points
    time_new = np.linspace(0, len(trajectory) - 1, num_points)

    # Interpolate the coordinates using linear interpolation after smoothing
    x_new = np.interp(time_new, np.arange(len(x_smoothed)), x_smoothed)
    y_new = np.interp(time_new, np.arange(len(y_smoothed)), y_smoothed)

    # Combine the interpolated x and y coordinates back into a list of tuples
    coords_new = np.array(list(zip(x_new, y_new)))
    out = np.array(list(zip(coords_new, extra)))
    return out

def resample_trajectory(trajectory, xy_smoothing=1000, sr_smoothing=500):
    x = np.array([point[0][0] for point in trajectory])
    y = np.array([point[0][1] for point in trajectory])
    s = np.array([point[1][0] for point in trajectory])
    r = np.array([point[1][1] for point in trajectory])

    # Calculate arc length
    arc_length = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))

    # Generate resampled arc length (linearly spaced)
    num_points = len(trajectory)  # Number of points to resample to
    resampled_arc_length = np.linspace(arc_length[0], arc_length[-1], num_points)

    # Interpolate (with smoothing for xy, optional for s, r)
    spline_x = UnivariateSpline(arc_length, x, s=xy_smoothing)
    spline_y = UnivariateSpline(arc_length, y, s=xy_smoothing)

    spline_s = UnivariateSpline(arc_length, s, s=sr_smoothing)
    spline_r = UnivariateSpline(arc_length, r, s=sr_smoothing)

    x_resampled = spline_x(resampled_arc_length)
    y_resampled = spline_y(resampled_arc_length)
    s_resampled = spline_s(resampled_arc_length)
    r_resampled = spline_r(resampled_arc_length)

    # Combine into output format [[[x, y], [s, r]], ...]
    xy = np.stack([x_resampled, y_resampled], axis=1)
    sr = np.stack([s_resampled, r_resampled], axis=1)
    xysr = np.stack([xy, sr], axis=1)

    return xysr

def smooth_trajectory(trajectory, xy_smoothing=1000, sr_smoothing=1000):
    x = np.array([point[0][0] for point in trajectory])
    y = np.array([point[0][1] for point in trajectory])
    s = np.array([point[1][0] for point in trajectory])
    r = np.array([point[1][1] for point in trajectory])

    # Use index as the independent variable (preserves temporal order)
    t = np.arange(len(trajectory))

    spline_x = UnivariateSpline(t, x, s=xy_smoothing)
    spline_y = UnivariateSpline(t, y, s=xy_smoothing)
    spline_s = UnivariateSpline(t, s, s=sr_smoothing)
    spline_r = UnivariateSpline(t, r, s=sr_smoothing)

    x_smooth = spline_x(t)
    y_smooth = spline_y(t)
    s_smooth = spline_s(t)
    r_smooth = spline_r(t)

    xy = np.stack([x_smooth, y_smooth], axis=1)
    sr = np.stack([s_smooth, r_smooth], axis=1)
    xysr = np.stack([xy, sr], axis=1)

    return xysr

def calculate_smooth_density_profile(trajectory, smoothing_factor=15):
    """
    Calculate and smooth the density profile of a trajectory.
    
    Args:
        trajectory: List of points [[x0,y0], [x1,y1], ...] or similar format
        smoothing_factor: Strength of smoothing (0=no smoothing, higher=more smoothing)
        method: 'spline' or 'gaussian' smoothing method
    
    Returns:
        Tuple of (original_densities, smoothed_densities)
    """
    # Convert to numpy array and extract coordinates
    traj = np.asarray(trajectory)
    if traj.ndim == 3 and traj.shape[1] == 2:  # Handle [[[x,y],[s,r]],...] format
        xy = traj[:,0,:]
    else:
        xy = traj
    
    # Calculate distances between consecutive points
    displacements = np.diff(xy, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    
    # Handle zero distances to avoid division by infinity
    min_nonzero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 1.0
    distances = np.where(distances > 0, distances, min_nonzero_dist)
    
    # Calculate raw density (1/distance)
    raw_density = distances
    
    # Pad the last point by repeating the last density value
    raw_density = np.concatenate([raw_density, [raw_density[-1]]])
    
    # Spline smoothing
    n_points = len(raw_density)
    spline = UnivariateSpline(np.arange(n_points), raw_density, 
                            s=smoothing_factor*n_points, ext='const')
    smooth_density = spline(np.arange(n_points))

    
    # Ensure positivity and normalize to maintain total "mass"
    smooth_density = np.maximum(smooth_density, 1e-6)
    smooth_density = smooth_density * (np.sum(raw_density)/np.sum(smooth_density))

    
    return smooth_density

def get_positions_at_distances(trajectory, input_distances):
    """
    Vectorized version to find positions at multiple distances along a trajectory.
    
    Args:
        trajectory: List of points in format [[[x,y],[s,r]], ...]
        input_distances: Array of target distances along the path
        
    Returns:
        Array of [[x,y],[s,r]] pairs for each input distance
    """
    # Convert to numpy arrays
    xy = np.array([point[0] for point in trajectory])
    sr = np.array([point[1] for point in trajectory])
    
    # Calculate segment distances and cumulative distances
    segment_distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_distances = np.insert(np.cumsum(segment_distances), 0, 0)
    total_length = cum_distances[-1]
    
    cum_input_distances = np.insert(np.cumsum(input_distances), 0, 0)
    
    # Initialize output array
    results = np.empty((len(cum_input_distances), 2, 2))

    # Find segments for each distance (vectorized)
    seg_indices = np.searchsorted(cum_distances, cum_input_distances) - 1
    
    # Clip indices to valid range
    seg_indices = np.clip(seg_indices, 0, len(segment_distances) - 1)
    
    # Calculate ratios (vectorized)
    seg_starts = cum_distances[seg_indices]
    seg_ends = cum_distances[seg_indices + 1]
    seg_lengths = seg_ends - seg_starts
    
    # Handle zero-length segments
    valid_segments = seg_lengths > 0
    ratios = np.zeros_like(cum_input_distances)
    ratios[valid_segments] = ((cum_input_distances - seg_starts) / seg_lengths)[valid_segments]
    # print(f'input_distances: {input_distances}')
    # print(f'ratios: {ratios}')
    # print(f'seg_indices: {seg_indices}')
   
    
    # Vectorized interpolation
    xy0 = xy[seg_indices]
    xy1 = xy[seg_indices + 1]
    sr0 = sr[seg_indices]
    sr1 = sr[seg_indices + 1]
    
    results[:, 0, :] = xy0 + ratios[:, None] * (xy1 - xy0)
    results[:, 1, :] = sr0 + ratios[:, None] * (sr1 - sr0)
    
    # Handle distances beyond path length
    beyond_mask = cum_input_distances >= total_length
    results[beyond_mask, 0, :] = xy[-1]
    results[beyond_mask, 1, :] = sr[-1]
    
    # print(f'cum_distances: {cum_distances}')
    
    return results

def remove_same_points(trajectory, threshold=1e-3):
    """
    Remove points from the trajectory that are closer than the given threshold.

    Args:
        trajectory: np.ndarray of shape (N, 2, 2) or similar.
        threshold: Minimum allowed distance between consecutive points.

    Returns:
        np.ndarray: Filtered trajectory.
    """
    if len(trajectory) == 0:
        return trajectory
    filtered = [trajectory[0]]
    for point in trajectory[1:]:
        prev_point = filtered[-1]
        dist = np.linalg.norm(point[0] - prev_point[0])
        if dist > threshold:
            filtered.append(point)
    return np.array(filtered)

def smooth_density_resample(trajectory, density_smoothing=15, xy_smoothing=1000, sr_smoothing=100):
    """
    Smooth and resample a trajectory while maintaining its density profile.
    
    Args:
        trajectory: List of points [[x0,y0], [x1,y1], ...] or similar format
        smoothness: Strength of smoothing (0=no smoothing, higher=more smoothing)
        xy_smoothing: Smoothing factor for x and y coordinates
        sr_smoothing: Smoothing factor for s and r coordinates
    
    Returns:
        Smoothed and resampled trajectory.
    """
    
    smoothed_trajectory = resample_trajectory(trajectory, xy_smoothing, sr_smoothing)
    density_profile = calculate_smooth_density_profile(trajectory, density_smoothing)
    final_smoothed = get_positions_at_distances(smoothed_trajectory, density_profile)
    final_trajectory = remove_same_points(final_smoothed)
    # print('--------------------------------------------------------------------------------------------------------------')
    # print(f"Smoothed Trajectories: {smoothed_trajectory}")
    # print(f"Density Profile: {density_profile}")
    # print(f"Final Trajectory: {final_trajectory}")
    # print('--------------------------------------------------------------------------------------------------------------')

    return final_trajectory


# Function to sort points in a trajectory
def sort_points(points):
    points = np.array(points)
    sorted_points = [points[0]]
    points = np.delete(points, 0, axis=0)
    while points.size > 0:
        distances = np.linalg.norm(points - sorted_points[-1], axis=1)
        closest_point_index = np.argmin(distances)
        sorted_points.append(points[closest_point_index])
        points = np.delete(points, closest_point_index, axis=0)
    return np.array(sorted_points)

# Function to rotate a rectangle and return its corners ------------------------------return to fix
def rotate_rectangle(point, dir_point, width):
    # length = (math.dist(point, dir_point))*2
    length = (math.dist(point, dir_point))
    delta_x = dir_point[0] - point[0]
    delta_y = dir_point[1] - point[1]
    angle = np.arctan2(delta_y, delta_x)
    perpendicular_angle = angle + np.pi / 2
     # Calculate the coordinates of the rectangle's corners
    half_width = width / 2
    half_height = length / 2

    corners = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])
    rotation_matrix = np.array([
        [np.cos(perpendicular_angle), -np.sin(perpendicular_angle)],
        [np.sin(perpendicular_angle), np.cos(perpendicular_angle)]
    ])
    rotated_corners = corners @ rotation_matrix.T + np.array([point[0], point[1]])

    return rotated_corners

def get_mode_list_by_length_kde(list_of_lists, return_list=False):
    """
    Returns the index (and optionally the list) of the internal list
    whose length is closest to the KDE-estimated mode of all lengths.

    Args:
        list_of_lists (list of list): Input data.
        return_list (bool): If True, return both index and the list.

    Returns:
        int: Index of the internal list closest to mode.
        list (optional): The actual list at that index.
    """
    if not list_of_lists:
        raise ValueError("Input list is empty.")
    elif len(list_of_lists) == 1:
        return list_of_lists[0] if return_list else 0

    lengths = np.array([len(lst) for lst in list_of_lists])

    # Check for zero variance (i.e., all lengths are equal)
    if np.all(lengths == lengths[0]):
        return list_of_lists[0] if return_list else 0

    # KDE estimation
    kde = gaussian_kde(lengths)
    xs = np.linspace(min(lengths) - 1, max(lengths) + 1, 500)
    ys = kde(xs)

    mode_estimate = xs[np.argmax(ys)]
    closest_index = int(np.argmin(np.abs(lengths - mode_estimate)))

    return list_of_lists[closest_index] if return_list else closest_index