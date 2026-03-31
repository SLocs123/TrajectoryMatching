# use:
# filter tracks
# pass complete tracks to trajectory averaging
#  - cluster tracks to group trajectories
#  - average each cluster to get all unique paths
# Pass broken tracks to occlusion detection
#  - cluster start and end point of broken tracks to find common break points
#  - run semantic segmentation to find where road if likely covered by foreground objects
#  - combine segmentation and break points to generate an expected occlusion Zone
# 
# average trajectories are then matched to tracks to generate a longterm expected path, switched if deviation occurs
# occlusion zones activate expected path and speed params to maintain stable motion prediction though poor detection zones
# 
# problems, expected paths may not represent all possible paths - Can switch to normal kf if deveiation occurs and distance gate returns no trajs
#           occlusion zones may not be purely caused by physical occlusion, making it hard to determine if segmentation should be used as a gate or not, if not, poor detectors might casue too many occlusion zones, if used , good zones might get filtered out
#           While this method is more dynamic, it doesn't give entry/exit zones, which are useful when developing a camera link model in MCT

# from labels generate dict, track id as key. internal dict, trajectory, frames, start appearance, end appearance, vecotrs(need to finalise syntax)

# dict{1: {traj: [], xys: [], frames: [], start_appearance: [], end_appearance: []}, 2: {...}, ...}

import numpy as np
from .Utils.filter_tracks import filter_short, filter_end_to_end_disp
from .Utils.io_utils import read_labels_from_txt#, save_json_traj, save_json_zones
from .average_traj.trajectory_analysis import create_expected_trajectories
from .detect_occlusion.auto_occlusion_detect import occlusion_clustering
from .group_trajectories.Generate_track_zones import get_sink_source

class Trajectory_Initialisation:
    def __init__(self, label_path, video_path, video_fps=15, min_time_in_scene=3, img_size=(3840, 2160)):
        _, traj_dict = read_labels_from_txt(label_path)
        self.traj_dict = traj_dict
        self.video_path = video_path
        self.video_fps = video_fps
        self.min_time_in_scene = min_time_in_scene
        self.img_size = img_size
        self.img_diag = np.hypot(img_size[0], img_size[1])
    
    def filter_traj(self):
        self.traj_dict, _ = filter_short(self.traj_dict, self.video_fps*self.min_time_in_scene) # filter small tracks, min_time_in_scene is seconds in scene
        self.traj_dict, _ = filter_end_to_end_disp(self.traj_dict, 0.05*self.img_diag) # filter stationary tracks
        
    def run(self, save_output=False):
        self.filter_traj()
        valid_dict, broken_dict, occ_zones = occlusion_clustering(self.traj_dict, self.video_path) # generate occlusion zones from the broken tracks, can be done in parallel to speed up
        
        zones, clusters = get_sink_source(valid_dict) # get entry and exit zones from valid tracks
        
        zone_dict = {"source/sink_pairs": zones, "occ_zones": occ_zones}
        average_trajs = create_expected_trajectories(clusters, zone_dict) # average common trajectories and apply local occlusion to them
        
        self.traj_dict = average_trajs
        self.zones = zones
        
        # if save_output:
        #     self.save_out("output/")
        return average_trajs, zones
    
    # def save_out(self, save_dir: str):
    #     # If a file path is passed (e.g. ".../out.json"), use its parent directory.
    #     # If a directory path is passed, use it directly.
    #     if save_dir.lower().endswith(".json"):
    #         save_dir = os.path.dirname(save_dir)

    #     # Handle cases like "zones.json" with no parent directory.
    #     if not save_dir:
    #         save_dir = "./output"

    #     os.makedirs(save_dir, exist_ok=True)

    #     save_json_traj(self.traj_dict, os.path.join(save_dir, "traj_dict.json"))
    #     save_json_zones(self.zones, os.path.join(save_dir, "zones.json"))