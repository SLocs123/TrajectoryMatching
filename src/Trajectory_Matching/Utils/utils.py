from shapely.geometry import Point, Polygon
import numpy as np

def get_centre(xyxy):
    x1,y1,x2,y2 = xyxy
    x = (x1+x2)/2
    y = (y1+y2)/2
    return (x,y)

def x_to_bbox(x): # conversions originally from SORT code!
    """
    x in form [x,y,s,r]
    Takes a bounding box in the center form [x, y, s, r] and returns it in the form
    [x1, y1, x2, y2] where x1, y1 is the top left and x2, y2 is the bottom right.
    """
    width = np.sqrt(x[2] * x[3])
    height = x[2] / width
    x1= x[0] - width / 2.0
    y1 = x[1] - height / 2.0
    x2 = x[0] + width / 2.0
    y2 = x[1] + height / 2.0
    return np.array([x1, y1, x2, y2])


def bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1, y1, x2, y2] and returns z in the form
    [x, y, s, r] where x, y is the center of the box, s is the scale/area, and r is
    the aspect ratio.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = bbox[0] + width / 2.0
    y = bbox[1] + height / 2.0
    scale = width * height  # scale is just area
    aspectRatio = width / float(height)
    return np.array([x, y, scale, aspectRatio])


def add_item(nested_dict, key1, key2, item, local_occ, deltas):
    """
    Adds an item to a nested dictionary structure.

    Parameters:
        nested_dict (dict): The main dictionary to update.
        key1 (hashable): First-level key.
        key2 (hashable): Second-level key under key1.
        item (Any): The item to store at nested_dict[key1][key2].

    Returns:
        dict: Updated nested dictionary.
    """
    
    if key1 not in nested_dict:
        nested_dict[key1] = {}
    nested_dict[key1][key2] = {'trajectory': item, 'local_occlusion': local_occ, 'deltas': deltas}
    return nested_dict

def show_structure(nested_dict, indent=0):
    """
    Recursively prints the structure and types of values in a nested dictionary.

    Parameters:
        nested_dict (dict): The dictionary to inspect.
        indent (int): Current indentation level (used internally for recursion).
    """
    for key, value in nested_dict.items():
        print(' ' * indent + f'{key} ({type(value).__name__})')
        if isinstance(value, dict):
            show_structure(value, indent + 2)

def get_next_element(seq, index):
    """
    Returns the next element in a sequence if available; otherwise, returns the previous one.

    Parameters:
        seq (list): The list or sequence to access.
        index (int): The current index in the list.

    Returns:
        Any: The next element if possible, otherwise the previous element. If index is 0 and the list has only one element, returns that element.
    """
    if index + 1 < len(seq):
        return seq[index + 1]
    elif index > 0:
        return seq[index - 1]
    else:
        return seq[0]

def is_within(xy, polygons):
    """
    Checks if a 2D point lies within any polygon in a list.

    Parameters:
        xy (tuple): The (x, y) coordinates of the point.
        polygons (list): A list of shapely.geometry.Polygon objects.

    Returns:
        tuple:
            - bool: True if the point is within any polygon, False otherwise.
            - Polygon or None: The containing polygon, or None if not found.
    """
    x, y = xy
    point = Point(x, y)
    for polygon in polygons:
        if polygon.contains(point):
            return True, polygon
    return False, None

def crop_img(traj_dict, video_path, img_size=(3840, 2160)):
    import cv2
    cap = cv2.VideoCapture(video_path)

    for track_id, track_info in traj_dict.items():
        track_info['crops'] = [[], []] # start and end crop
        for i in range(2):
            if i==0:
                index = 0
            else:
                index = -1
            frame = track_info['frames'][index]

            index_cap = cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame} for track {track_id}, please ensure correct video and labels")

            x1, y1, x2, y2 = track_info['xys'][index][0], track_info['xys'][index][1], track_info['xys'][index][2], track_info['xys'][index][3]
        

            # Ensure at least 1x1 crop (avoid empty slice)
            if x2 <= x1: x2 = min(x1 + 1, img_size[0] - 1)
            if y2 <= y1: y2 = min(y1 + 1, img_size[1] - 1)

            crop = img[int(y1):int(y2), int(x1):int(x2)]  # shape (h, w, c), probably RGB

            # Optional: resize to 256x256 if needed for consisten
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)

            # Encode as JPEG (OpenCV expects BGR, so convert if needed)
            crop_bgr = crop[:, :, ::-1] if crop.shape[2] == 3 else crop

            success_crop, crop_jpeg = cv2.imencode('.jpg', crop_bgr)

            traj_dict[track_id]['crops'][index] = crop_jpeg.tobytes() if success_crop else None

    cap.release()
    

def labelstudio_labels_to_yolo(labelstudio_labels_json_path: str, index_video):
    import json
    labels = json.load(open(labelstudio_labels_json_path))[index_video]
    
    # every box stores the frame count of the whole video, so we get it from the first box
    frames_count = labels['annotations'][0]['result'][0]['value']['framesCount'] +1
    yolo_labels = [[] for _ in range(frames_count)]

    # iterate through boxes
    for box in labels['annotations'][0]['result']:
       
        label_numbers = [2 for label in box['value']['labels']]
        # iterate through keypoints (we omit the last keypoint because no interpolation after that)
        for i, keypoint in enumerate(box['value']['sequence'][:-1]):
            start_point = keypoint
            end_point = box['value']['sequence'][i + 1]
            start_frame = start_point['frame']
            end_frame = end_point['frame']

            n_frames_between = end_frame - start_frame
            delta_x = (end_point['x'] - start_point['x']) / n_frames_between
            delta_y = (end_point['y'] - start_point['y']) / n_frames_between
            delta_width = (end_point['width'] - start_point['width']) / n_frames_between
            delta_height = (end_point['height'] - start_point['height']) / n_frames_between

            # In YOLO, x and y are in the center of the box. In Label Studio, x and y are in the corner of the box.
            x = start_point['x'] + start_point['width'] / 2
            y = start_point['y'] + start_point['height'] / 2
            width = start_point['width']
            height = start_point['height']
            car_id = box["id"]            
            
            # iterate through frames between two keypoints
            for frame in range(start_frame, end_frame):
                # Support for multilabel
                yolo_labels = _append_to_yolo_labels(yolo_labels, frame, label_numbers, x, y, width, height, car_id)
                x += delta_x + delta_width / 2
                y += delta_y + delta_height / 2
                width += delta_width
                height += delta_height            
         

        # Handle last keypoint
        yolo_labels = _append_to_yolo_labels(yolo_labels, frame+1, label_numbers, x, y, width, height, car_id)
         
    
    # json output file
    frames_data = {}
    
    # Loop through each frame and its labels
    yolo_labels = yolo_labels[1:]
    for frame, frame_labels in enumerate(yolo_labels):
        # Create a unique ID for each frame, e.g., "frame_1", "frame_2", etc.
        frame_id = frame
        frames_data[frame_id] = []

        # Process each label in the frame
        for label in frame_labels:
            # Extract the values
            car_id, label_type, min_x, min_y, max_x, max_y = label

            # Create a dictionary for the label
            label_data = {
                "car_id": car_id,
                "type": label_type,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y
            }

            # Append the label data to the frame data
            frames_data[frame_id].append(label_data)

    # Write to a JSON file
    #with open('frame_labels.json', 'w') as outfile:
        #json.dump(frames_data, outfile, indent=4)
    
    return frames_data
        

def _append_to_yolo_labels(yolo_labels: list, frame: int, label_numbers: list, x, y, width, height, car_id):

    # we need min_x, min_y, max_x, max_y
    min_x = x - width / 2
    min_y = y - height / 2
    max_x = x + width / 2
    max_y = y + height / 2


    for label_number in label_numbers:
        yolo_labels[frame].append(
            [car_id, label_number, min_x, min_y, max_x, max_y])
        #print(f"appended {[label_number, x, y, width, height]}")
    return yolo_labels


def transform_labelstudio_input(label_studio_json_path, videoindex):
    # input = ".data/labels_CAM-HAZELDELL-126THST.json"
    # videoindex = 44


    #data = json.load(open(label_studio_json_path))
    #json.dump(data, open(input, 'w'), indent=4)

    transformed_json = labelstudio_labels_to_yolo(label_studio_json_path, int(videoindex))

    return transformed_json

def draw_box(points, buffer=0.0):
    """Returns the axis-aligned bounding box (x1, y1, x2, y2) that contains all points.
    points is a list of (x, y) tuples.
    buffer is an extra percentage to expand the box by, e.g. 0.1 means expand by 10% in all directions."""
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_buffer = (x_max - x_min) * buffer
        y_buffer = (y_max - y_min) * buffer
        return (x_min - x_buffer, y_min - y_buffer, x_max + x_buffer, y_max + y_buffer)
    else:
        raise ValueError("No points provided to draw_box")