# filter broken tracks

"""
identifiers:
- intitialise track in middle of scene
- track ends in middle of scene
- track is too short, must find best way to determine, some tracks will be naturally shorter
- track ends close to another track that continues, suggesting error
- track starts at middle of another track
"""
import numpy as np

def find_track_borders(tracks, img_size, buffer_coeff=0.05):
    start_max_x = 0
    start_min_x = np.inf
    start_max_y = 0
    start_min_y = np.inf

    end_max_x = 0
    end_min_x = np.inf
    end_max_y = 0
    end_min_y = np.inf

    for car_id, track in tracks.items():
        start, end = track['trajectory'][0], track['tracjectory'][-1]
        if start[0]>start_max_x:
            start_max_x = start[0]
        if start[0]<start_min_x:
            start_min_x = start[0]
        if start[1]>start_max_y:
            start_max_y = start[1]
        if start[1]<start_min_y:
            start_min_y = start[1]

        if end[0]>end_max_x:
            end_max_x = end[0]
        if end[0]<end_min_x:
            end_min_x = end[0]
        if end[1]>end_max_y:
            end_max_y = end[1]
        if end[1]<end_min_y:
            end_min_y = end[1]
    start_buffer = [start_max_x+img_size[0]*buffer_coeff, start_max_y+img_size[1]*buffer_coeff, start_min_x+img_size[0]*buffer_coeff, start_min_y+img_size*buffer_coeff]
    end_buffer = [end_max_x+img_size[0]*buffer_coeff, end_max_y+img_size[1]*buffer_coeff, end_min_x+img_size[0]*buffer_coeff, end_min_y+img_size*buffer_coeff]        


def start_in_middle(track, image_size, buffer_coeff=0.3):
    x,y = track[0]
    buffer_x = image_size[0] * buffer_coeff
    buffer_y = image_size[1] * buffer_coeff

    if x > buffer_x and x < image_size[0] - buffer_x and y > buffer_y and y < image_size[1] - buffer_y:
        return True
    return False

def end_in_middle(track, image_size, buffer_coeff=0.3):
    x,y = track[-1]
    buffer_x = image_size[0] * buffer_coeff
    buffer_y = image_size[1] * buffer_coeff

    if x > buffer_x and x < image_size[0] - buffer_x and y > buffer_y and y < image_size[1] - buffer_y:
        return True
    return False

def filter_short(tracks, min_length):
    """
    Could use current min_length threshold, but a MAD approach would likely be more generalised.
    def too_short(all_tracks, track, threshold_coeff=1.5):
    """
    filtered_short_tracks = {}
    removed_tracks = {}

    for car_id, track in tracks.items():
        trajectory = track['trajectory']
        if len(trajectory) >= min_length:
            filtered_short_tracks[car_id] = track
        else:
            removed_tracks[car_id] = track

    return filtered_short_tracks, removed_tracks

def filter_end_to_end_disp(tracks, min_disp):
    filtered_disp_tracks = {}
    removed_tracks = {}

    for car_id, track in tracks.items():
        trajectory = track['trajectory']

        displacement = point_displacement(trajectory[0], trajectory[-1])

        if displacement > min_disp:
            filtered_disp_tracks[car_id] = track
        else:
            removed_tracks[car_id] = track

    return filtered_disp_tracks, removed_tracks

def filter_average_displacement(tracks, min_avg_disp):
    filtered_disp_tracks = {}
    removed_tracks = {}

    for car_id, track in tracks.items():
        trajectory = track['trajectory']

        displacements = [
            point_displacement(trajectory[i], trajectory[i - 1])
            for i in range(1, len(trajectory))
        ]

        avg_disp = np.mean(displacements)

        if avg_disp > min_avg_disp:
            filtered_disp_tracks[car_id] = track
        else:
            removed_tracks[car_id] = track

    return filtered_disp_tracks, removed_tracks

def point_displacement(point1, point2):
    """
    Returns Euclidean distance between two 2D points.
    Points can be tuples, lists, or numpy arrays: (x, y)
    """
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return np.hypot(dx, dy)

def end_close_to_other(track, other_tracks, threshold):
    """
    Checks to see if the end point of the track is within the middle 30% of another track,
    suggesting that the track has ended in the middle of the scene
    """
    end_point = track[-1]
    for other_track in other_tracks:
        total_points = len(other_track)
        closest_point, distance, index = get_closest_point(end_point, other_track)
        if distance < threshold and index > total_points * 0.3 and index < total_points * 0.7:
            return True
    return False

def start_close_to_other(track, other_tracks, threshold):
    """
    Checks to see if the start point of the track is within the middle of another track,
    that it is close to. This would suggest that the track has started in the middle of the scene, and is likely broken.
    """
    start_point = track[0]
    for other_track in other_tracks:
        total_points = len(other_track)
        closest_point, distance, index = get_closest_point(start_point, other_track)
        if distance < threshold and index > total_points * 0.3 and index < total_points * 0.7:
            return True
    return False
    
def get_closest_point(point, track):
    """
    Return the closet point along a track, the distance and the index of the point along the track
    """
    closest_point = None
    min_distance = float('inf')
    closest_index = -1
    index = 0
    for track_point in track:
        distance = ((point[0] - track_point[0])**2 + (point[1] - track_point[1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_point = track_point
            closest_index = index
        index += 1
    return closest_point, min_distance, closest_index


# Extra functions:
# Arc length
# num_points (too short copy)
# start to end displacnement (might need tight threshold here to avoid false positives)
# eff = disp / (L + eps) path efficiency (jitter detector) (displacement over arc length, if the track is very jittery, the efficiency will be low, and it may be a broken track, similar to displacement)
# logging metrics can help to reduce the "heavy tail" and get a nicer distribution of track lengths
# logL = log(L + eps)
# logDisp = log(disp + eps)
# logN = log(n_pts)

# This could be changed similar to clustering, where tracks are discarded if they are not similar to any other,
# careful if multiple tracks are broken in the same way, they may be similar to each other but not to any good tracks, so this would need to be carefully designed, and may not be worth it.
# after initial rule-based filtering, could then remove outliers based on track similarity, some track may be unique but still good, 
# clustering need at least 2 tracks anyway, so could work as purely unique would still be removed later.


def filter_broken_tracks(tracks, image_size, min_length, threshold):
    """
    checks a number of broken track gates, if track is too short it is filtered, 
    if it is consered a broken/half track it is added to occlusion group
    """
    filtered_tracks = []
    broken_tracks = []
    short_tracks = []
    for track in tracks:
        start, end, end_close, start_close, too_short = start_in_middle(track, image_size), end_in_middle(track, image_size), end_close_to_other(track, tracks, threshold), start_close_to_other(track, tracks, threshold), too_short(track, min_length)
        if too_short:
            short_tracks.append(track)
            continue
        if start_close and start:
            broken_tracks.append(track)
            continue
        if end_close and end:
            broken_tracks.append(track)
            continue

        # if start and end: # this may be invalid
        #     continue

        filtered_tracks.append(track)
    return filtered_tracks, broken_tracks, short_tracks

def para_tracks(tracks):
    """this function will go through all tracks and look at common metrics
    e.g. arc length, num points, end-end displcement
    

    """


# best method:
# run re-link model, get frame when lost and initialised, plus likelyhood of being same track. From here i can identify occlusion vs sink
# tracks that do not have a likely pair are lost vs one that do are broken
# broken tracks in a common place mean occlusion
# lost tracks in common place means sink, can handle both easily. sink = exit, can be zoned
# the problem here is detecting new occlusion tracks, likely have to compare all tracks to make it work robustly, hard to detinguish between new/old tracks, this will leed to a lot of computational cost

# can we use grouping still? to generate entry zones, sure
# if we run clustering on each point, of DTW alignment, we might be able to find points of convergence in labes, making good entry zones too, might be difficult