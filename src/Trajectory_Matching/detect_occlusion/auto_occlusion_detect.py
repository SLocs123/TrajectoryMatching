# Purpose:
# Currently, occlusion is manually labelled, giving easy and accurate occlusion handling
# However, manual label lowers usability and scalability.
# Using segmentation to detect covered roads and existing tracks to find frequently dropped tracks should be enough to auto detect occlusion zones
# This script will runn sem seg and using label tracks to find occlusion.
# Previous broken track code will identify lost tracks, we can then run clustering on the end/start points to find zones, should not pass broken tracks that are too small

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Any
from Trajectory_Matching.Utils.utils import draw_box
from Trajectory_Matching.feature_extractor.config import FeatureExtrator, Reid_config
from Trajectory_Matching.feature_extractor.extractor import Extractor
from Trajectory_Matching.Utils.utils import crop_img

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LinkRecord:
    end_id: str
    start_id: str
    end_point: Point
    start_point: Point


def build_link_records(
    broken_tracks: Dict[str, dict],
    matches: Iterable[dict],
) -> List[LinkRecord]:
    records: List[LinkRecord] = []
    for m in matches:
        end_id = m["end_id"]
        start_id = m["start_id"]

        records.append(
            LinkRecord(
                end_id=end_id,
                start_id=start_id,
                end_point=broken_tracks[end_id]["trajectory"][-1],   # disappears here
                start_point=broken_tracks[start_id]["trajectory"][0], # reappears here
            )
        )
    return records


def occlusion_clustering(traj_dict, video_path):
    """ Atempts to associate start and end points to get an occlusion distance, can then be combines with sementation to build a zone"""
    from Trajectory_Matching.detect_occlusion.track_re_link import _sct_rematch_process
    
    traj_dict = apply_appearance_features(traj_dict, video_path) 
    valid_dict, broken_dict, matched_pairs = _sct_rematch_process(traj_dict) # , apply_relink=False: if you are limited on valid tracks, apply relink to rebuild broken tracks and increase sample size(yet to implement)
    
    # apply clustering to broken tracks to define each zone group of tracks
    # for each zone, draw a box that covers last 3 and first 3 points from all tracks in the cluster
    # then add segmentation, if road is detected in/near the occlusion and a forground object is present, ensure the box covers full area
    # then look a training model to detect occ_zones, compare results, then hybrid method, model takes clusters sementation and track info to predict zones.
    # zone should extend so that it covers the center point when one of the corners reaches the occlusion, meaning prediction can be correct before degradation occurs
    
    
    occ_zones = zones_construction(broken_dict, matched_pairs)
    
    
    return valid_dict, broken_dict, occ_zones

def zones_construction(broken_tracks: Dict[str, dict], matches: List[dict]) -> List[Box]:
    if not matches:
        return []

    records = build_link_records(broken_tracks, matches)

    # Keep your point_clustering call, but pass records instead of ad-hoc dicts.
    # Example expected return: List[List[LinkRecord]]
    clusters = point_clustering(records)

    boxes: List[Box] = []
    for cluster in clusters:
        end_points = [r.end_point for r in cluster]
        start_points = [r.start_point for r in cluster]
        box = draw_box(end_points + start_points)

        end_tracks = [broken_tracks[r.end_id] for r in cluster]
        start_tracks = [broken_tracks[r.start_id] for r in cluster]

        # make boxes polyongs?
        boxes.append(refine_box(box, end_tracks, start_tracks))

    return boxes

def apply_appearance_features(traj_dict, video_path):
    # for each track, extract appearance features using a pretrained model, e.g. resnet50, and store in traj_dict
    # can use the start and end appearance as a simple representation of the track's appearance, or can use a more complex representation if needed

    
    crop_img(traj_dict, video_path) # crop the video frames to get the appearance of the track
    # save_tracks_manifest_simple(traj_dict, out_dir="temp/appearance_features") 
      
    settings = FeatureExtrator(reid_config=Reid_config())
    extractor = Extractor(settings)
    traj_dict = extractor._create_input4embed(traj_dict) # create input for embedding from the image dictionary, and store the features in traj_dict
    return traj_dict

def refine_box(box, end_tracks, start_tracks, tolerance=0.05):
    """
    if the dge of a related track box comes withing tolerance percentage of box size, expand the occlusion box to include that center point. 
    This will ensure that the box covers the entire occlusion zone for center tracking and prevent erroneous detection/tracking inputs when running
    have to run a check on all 4 corners since we cannot assume occlusion is for a specific direction unless decide to use deltas. This should be fine though as leading edge should always be closest
    """
    new_points = []
    for track in end_tracks:
        traj = track['trajectory']
        xys = track['xys']
        for point, xy in zip(reversed(traj), reversed(xys)):            
            if check_box_xy(box, xy, tolerance):
                # first gether all points, then expand to fit.
                new_points.append(point)
                
    for track in start_tracks:
        traj = track['trajectory']
        xys = track['xys']
        for point, xy in zip(traj, xys):
            if check_box_xy(box, xy, tolerance):
                new_points.append(point)
    
    if not new_points:
        return box
    new_box = draw_box(new_points)
    return new_box

def check_box_xy(box, xy, tolerance):
    """checks if any corner of xy is within box expanded by tolerance, returns true if so, false if not"""
    bx1, by1, bx2, by2 = box
    x1, y1, x2, y2 = xy

    width = bx2 - bx1
    height = by2 - by1

    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for x, y in corners:
        if (bx1 - tolerance*width) <= x <= (bx2 + tolerance*width) and (by1 - tolerance*height) <= y <= (by2 + tolerance*height):
            return True
    return False
    
def point_clustering(records: List[LinkRecord], eps=50, min_samples=5) -> List[List[LinkRecord]]:
    """Cluster LinkRecord objects by end_point while preserving full record information.

    Returns a list of clusters, where each cluster is a list of LinkRecord.
    Noise points (DBSCAN label -1) are excluded.
    """
    # Point clustering can be fairly generous, since end points are filtered; if two
    # zones are very close they are often acceptable to treat as one in this stage.
    from sklearn.cluster import DBSCAN
    import numpy as np

    if not records:
        print("No records provided for clustering, no zones can be constructed")
        return []

    feature_points = np.array([r.end_point for r in records], dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feature_points)

    grouped: Dict[int, List[LinkRecord]] = {}
    for record, label in zip(records, labels):
        if label == -1:
            continue
        grouped.setdefault(int(label), []).append(record)

    return [grouped[k] for k in sorted(grouped.keys())]
    

def segmentation(image, boxes):
    """runs semantic segmenation and identifies road and foreground pixels."""
    # should run segmentation on the scene and then monocular depth estimation.
    # from here i can use the existing occ_box to find the occulsion zone depth (by looking at the middle 25% of the box, hopefully targetting the main object)
    # This should give me the occlusion depth. I can then expand the box for objects that are close to the occ depth. 
    # avoid road, grass, pavement, path objects as these are background objects, foreground items like sign, trees, statues, walls, etc will need to be the focus, but i need a way to indentify the types


    # planeRCNN could give an option to define the road plane and then use the depth map to identify objects that deviate from the plane,
    # However, while this owuld be robust for all objects that sit on top of the plane/road
    # it still doesnt defferentiate between foreground and background objects
    # meaning all objects of significant height would be detected and not just objects blocking the road
    # may be best to just stick to box refining as defined above


def save_tracks_manifest_simple(
    traj_dict: Dict[Any, Dict[str, Any]],
    out_dir: str = "temp",
    crop_format: str = "jpg",
) -> str:
    """
    Saves a single JSON file with schema:

    {
      "<track_id>": {
        "frames": [...],
        "trajectory": [[x,y], ...],
        "crop_paths": ["<track_id>/crop_000.jpg", "<track_id>/crop_001.jpg", ...]
      },
      ...
    }

    Also writes crops to:
      temp/<track_id>/crop_000.jpg, crop_001.jpg, ...
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, Any]] = {}

    for track_id, track_info in traj_dict.items():
        tid = str(track_id)

        frames = track_info.get("frames", [])
        trajectory = track_info.get("trajectory", [])
        crops = track_info.get("crops", [])

        # JSON-safe conversions
        frames_json: List[int] = [int(f) for f in frames]
        trajectory_json: List[List[float]] = [[float(x), float(y)] for (x, y) in trajectory]

        track_dir = out_path / tid
        track_dir.mkdir(parents=True, exist_ok=True)

        crop_paths: List[str] = []
        for i, img in enumerate(list(crops)):
            if img is None:
                continue

            crop_name = f"crop_{i:03d}.{crop_format}"
            crop_path = track_dir / crop_name

            ok = cv2.imwrite(str(crop_path), img)
            if not ok:
                raise RuntimeError(f"Failed to write crop image: {crop_path}")

            # Store RELATIVE path (relative to out_dir) so it works when you mount/move the folder
            crop_paths.append(str(Path(tid) / crop_name))   

        manifest[tid] = {
            "frames": frames_json,
            "trajectory": trajectory_json,
            "crop_paths": crop_paths,
        }

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return str(manifest_path)