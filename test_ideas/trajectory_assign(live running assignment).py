#  intended use:
# First gate trajectories by distance to point, running DTW on all traj every frame is too expensive
# Ideally, we would have more gating metrics, vector dir, maybe speed
# Then, if multiple trjectories pass, we run trajectory matching
# Finally, we should probably stick to the same trajectory for a set period (10/5 frames?) or if distance gets to high, check again.
# KF values should be transferable, just switch the traj --> image map to new traj, needs testing to ensure it doesnt cause issues with the KF.
# This will prevent the need for multiple traj KF instances, hopefully improving usability and computational cost.

def closest_traj(point, trajectories):
    """ Given a point and a list of trajectories, return the closest trajectory to the point"""
    closest_traj = None 
    closest_dist = float('inf') 
    for traj in trajectories: 
        for traj_point in traj: 
            dist = distance(point, traj_point)
            if dist < closest_dist: 
                closest_dist = dist 
                closest_traj = traj 
    return closest_traj

def traj_similarity_DTW(history, trajectories):
    """ Given a track history, compares the similarity to a list of existing trjectories using dtai_distance"""

def traj_similarity_Frechet(history, trajectories):
    """ Given a track history, compares the similarity to a list of existing trjectories using the Frechet distance"""
    return lowest value

def custom_trajectory(history, trajectories, assigned):
    """
    once a trajectory is assigned, we can either using a decaying lateral disp or a polynomail curve to model a custom trajectory that follow the average.
    This would remove the need for the kf to maintain a lateral disp and delta value. and run purely with a arc length and speed.
    Then update the custom trajectory whenever the id diverges too far.
    """
    if assigned is None:
        assigned = traj_similarity_Frechet(history, trajectories)
    
    scale = 0.1 # how fast the custom trajectory converges to the assigned trajectory
    point = history[-1]
    lateral_disp, long_disp, current_index = calculate_disp(point, assigned)
    custom_traj = [point + lateral_disp*(point_index - current_index)*scale for point_index, point in enumerate(assigned)]
    
    return custom_traj
    
    # or define a polynomial curve and write a custom trajectory based on the curve and the current point, this would be more computationally expensive but may provide a smoother trajectory.
        