# loops through all tracks, will return a list of full tracks , a list of pre_occlusion tracks and a list of post_occlusion tracks
# make sure to filter out short tracks
# afterwards, tracks that start at a common place and then end in middle of the scene can be designated as sinks, occlusion zones found
import numpy as np
from typing import List, Dict, Tuple, Optional

def _sct_rematch_process(tracks, spatial_threshold=600, frame_threshold=45, app_threshold=0.7, threshold=0.5):
    """
    provide a list of ids that show a likely broken track, i.e. a track that appears later. All other end and start points can be consiered true sinks and sources.
    """
    # Select candidate tracklets for merging
    candidate_ids: List[str] = [track_id for track_id in tracks]

    if len(candidate_ids) < 2:
        return tracks.copy(), {}, []  # Not enough tracklets to consider merging

    # Save all candidate tracklets' data for processing
    start_frame: Dict[str, int] = {}
    end_frame: Dict[str, int] = {}
    entry_point: Dict[str, Tuple[float, float]] = {}
    exit_point: Dict[str, Tuple[float, float]] = {}
    start_feat: Dict[str, Optional[np.ndarray]] = {}
    end_feat: Dict[str, Optional[np.ndarray]] = {}
    mean_feat: Dict[str, Optional[np.ndarray]] = {}
    avg_deltas: Dict[str, float] = {}


    for track_id in candidate_ids:
        current_tracklet = tracks[track_id]
        frame_key = 'frames' if 'frames' in current_tracklet else 'frame'
        start_frame[track_id] = current_tracklet[frame_key][0]
        end_frame[track_id] = current_tracklet[frame_key][-1]

        # Entry = earliest det, Exit = latest det (for pixel distance)
        # first_det = current_tracklet['xys'][0]
        # last_det = current_tracklet['xys'][-1]
        entry_point[track_id] = current_tracklet['trajectory'][0]
        exit_point[track_id] = current_tracklet['trajectory'][-1]
        
        # ensure avg_deltas is compatible
        deltas = current_tracklet.get('deltas', [])
        if len(deltas) >= 2:
            window = deltas[-10:-5] if len(deltas) >= 10 else deltas
            avg_delta = float(np.mean(window)) if len(window) > 0 else 1.0
        else:
            avg_delta = 1.0
        avg_deltas[track_id] = max(avg_delta, 1e-6)

        # Features near start/end used for cosine distance
        sfeat = current_tracklet['start_appearance'] # could change this to have all appearances in ['sppearance'][0]
        efeat = current_tracklet['end_appearance']
        # mean_feat = current_tracklet['appearance'] mean_feat[track_id] = current_tracklet.mean_feature
        start_feat[track_id] = sfeat
        end_feat[track_id] = efeat

    # Build candidate pairs with costs
    candidates: List[Tuple[float, float, str, str]] = []  # (cost, pixel_dist, prev_id, next_id)
    for ida in candidate_ids:
        for idb in candidate_ids:
            if ida == idb:
                continue
            # ensure temporal order a -> b could calculate expected difference as well
            if end_frame[ida] >= start_frame[idb]:
                continue

            # spatial proximity constraint use to calculate frame difference
            pixel_dist = _euclidean(exit_point[ida], entry_point[idb])
            if (pixel_dist > spatial_threshold):
                continue
            
            expected_gap = int(pixel_dist / avg_deltas[ida])
            # temporal gap constraint
            max_frame_diff = expected_gap + frame_threshold # calculate based on video fps and desired time buffer, or percentage of expected gap
            frame_gap = start_frame[idb] - end_frame[ida]
            if frame_gap > max_frame_diff:
                continue


            # feature availability
            fa = end_feat.get(ida)
            fb = start_feat.get(idb)
            mfa = mean_feat.get(ida)
            mfb = mean_feat.get(idb)
            if fa is None or fb is None:
                continue

            # maybe add a frame diff score too
            frame_cost = min(abs(expected_gap-frame_gap)/frame_threshold, 1)
            # app_cost = min(_cosine_distance(fa, fb), _cosine_distance(mfa, mfb)) # need to import these functions, remove min if not using mean
            app_cost = _cosine_distance(fa, fb)
            if np.isfinite(app_cost) and app_cost > app_threshold: # define threshold, strict matching?
                continue

            W_APP = 0.7
            W_FRAME = 0.3
            cost = (W_APP * app_cost + W_FRAME * frame_cost) / (W_APP + W_FRAME) # apply weighted average incase 2 similar cars pass the gates

            if np.isfinite(cost) and cost < threshold: # define threshold, strict matching?
                candidates.append((float(cost), float(pixel_dist), ida, idb)) # a is end point before occ, b is start

    if not candidates:
        return tracks.copy(), {}, []

    unique_end_ids = sorted({ida for _, _, ida, _ in candidates})
    unique_start_ids = sorted({idb for _, _, _, idb in candidates})
    end_to_row = {track_id: idx for idx, track_id in enumerate(unique_end_ids)}
    start_to_col = {track_id: idx for idx, track_id in enumerate(unique_start_ids)}

    cost_matrix = np.full((len(unique_end_ids), len(unique_start_ids)), fill_value=1.0, dtype=float)
    for cost, _, ida, idb in candidates:
        cost_matrix[end_to_row[ida], start_to_col[idb]] = cost

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: List[Dict[str, str]] = []
    matched_ids = set()
    for row, col in zip(row_ind, col_ind):
        end_id = unique_end_ids[row]
        start_id = unique_start_ids[col]
        if cost_matrix[row, col] >= threshold:
            continue

        matched_pairs.append({"end_id": end_id, "start_id": start_id})
        matched_ids.add(end_id)
        matched_ids.add(start_id)

    broken_dict = {track_id: tracks[track_id] for track_id in matched_ids}
    valid_dict = {track_id: tracks[track_id] for track_id in tracks if track_id not in matched_ids}
    return valid_dict, broken_dict, matched_pairs
    
    

def _euclidean(pointA, pointB):
    return np.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)

def _cosine_distance(featA, featB):
    if featA is None or featB is None:
        return float('inf')  # or some large number to indicate no similarity
    dot_product = np.dot(featA, featB)
    normA = np.linalg.norm(featA)
    normB = np.linalg.norm(featB)
    if normA == 0 or normB == 0:
        return float('inf')  # avoid division by zero, treat as no similarity
    cosine_similarity = dot_product / (normA * normB)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance