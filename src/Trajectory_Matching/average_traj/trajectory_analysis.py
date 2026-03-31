from dtaidistance import dtw_barycenter
from ..Utils.utils import is_within
from ..Utils.trajectory_utils import smooth_trajectory

def find_sink_zones(source_zone, source_sink_pairs):
    """Find potential sink zones based on the source zone"""
    potential_sinks = []
    for pair in source_sink_pairs:
        if pair["source"] == source_zone:
            potential_sinks.append(pair["sink"])
    return potential_sinks

# Function to create expected trajectories dictionary
def create_expected_trajectories(clusters, zones):
    """
    clusters needs to be a set of similar trajectories, comes from zone code eariler.
    zones needs to be:
    
        {
            "source_sink_pairs": [{"source": polygon, "sink": polygon}, ...],
            "occ_zones": [Polygon, Polygon, ...]
        }
    """
       
    source_sink_pairs = zones["source_sink_pairs"]
    occ_zones = zones["occ_zones"]
    
    traj_index = 0 # would perfer a better naming convention here, or dont use a dict, but need to assign an id to a track later, dont want to loop through everytime
    avg_trajs = {}
    for cluster in clusters:
        average_traj = dtw_barycenter.dba_loop(cluster)
        local_occ_zones = []
        source_zone = None
        sink_zone = None
        potential_sink_zones = []
        
        for point in average_traj:
            _, occ_zone = is_within(point, occ_zones)
            if occ_zone not in local_occ_zones:
                local_occ_zones.append(occ_zone)
            
            if source_zone is None:
                flag, source_zone = is_within(point, [pair["source"] for pair in source_sink_pairs])
                potential_sink_zones.extend(find_sink_zones(source_zone, source_sink_pairs))
                                     
            if sink_zone is None:
                flag, sink_zone = is_within(point, [pair["sink"] for pair in source_sink_pairs])
                if sink_zone is not None:
                    assert sink_zone in potential_sink_zones, "Sink zone does not match potential sink zones for the source zone, trajectories are likely incorrect"
                
        
        
        avg_trajs[traj_index] = {"average_traj": average_traj, "local_occ_zones": local_occ_zones, "source_zone": source_zone, "sink_zone": sink_zone}
        traj_index += 1
    
    for traj_id, traj_info in avg_trajs.items():
        traj_info["average_traj"] = smooth_trajectory(traj_info["average_traj"])
    
    return avg_trajs
    
        
        
# zones = {"sources": sources, "sinks": sinks, "occ_zones": occ_zones}