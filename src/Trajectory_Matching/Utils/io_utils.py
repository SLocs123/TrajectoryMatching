import pandas as pd
import cv2

def get_true_labels(label_data, frame_number, scale_x=38.4, scale_y=21.6):
    """
    Retrieves and rescales bounding box labels for a specific video frame.

    This function assumes that the input labels are normalised to a coordinate system
    and rescales them using fixed scaling factors that correspond to specific camera dimensions.

    Parameters:
        label_data (dict): A dictionary where keys are frame numbers and values are lists of
                           dictionaries with bounding box information (min_x, min_y, max_x, max_y, car_id).
        frame_number (int): The frame number for which to retrieve and rescale labels.
        scale_x (float): Scaling factor for the x-dimension (default is 38.4).
        scale_y (float): Scaling factor for the y-dimension (default is 21.6).

    Returns:
        list: A list of bounding boxes in the format [x1, y1, x2, y2, car_id], scaled to image size.
    """
    if frame_number not in label_data:
        return []

    current_labels = label_data[frame_number]
    scaled_labels = []

    for bbox in current_labels:
        scaled_labels.append([
            bbox["min_x"] * scale_x,
            bbox["min_y"] * scale_y,
            bbox["max_x"] * scale_x,
            bbox["max_y"] * scale_y,
            bbox["car_id"]
        ])

    return scaled_labels

def read_labels_from_txt(txt_path):
    """
    Reads a text file containing vehicle tracking labels and returns them as a dictionary.
    
    Args:
    - txt_path (str): Path to the label text file.
    
    Returns:
    - dict: A dictionary where keys are frame numbers and values are lists of bounding boxes 
            and object IDs in each frame. Each bounding box is represented as [x1, y1, x2, y2, car_id].
    """
    from .utils import get_centre

    frame_dict = {}
    traj_dict = {}
    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            frame = int(parts[0])
            car_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            
            # Ensure the frame exists in the dictionary, then append the label
            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append([x1, y1, x2, y2, car_id])
            if car_id not in traj_dict:
                traj_dict[car_id] = {}
                traj_dict[car_id]['xys'] = []
                traj_dict[car_id]['trajectory'] = []
                traj_dict[car_id]['frames'] = []
                traj_dict[car_id]['start_appearance'] = []
                traj_dict[car_id]['end_appearance'] = []
            traj_dict[car_id]['xys'].append([x1, y1, x2, y2])
            traj_dict[car_id]['trajectory'].append(get_centre([x1, y1, x2, y2]))
            traj_dict[car_id]['frames'].append(frame)

            # traj_dict[car_id][vectors] = [()] # -------! if needed
    return frame_dict, traj_dict

def read_polygons_from_csv(file_path):
    """
    Reads polygons from a CSV file and returns a list of Polygon objects.
    
    Args:
    - file_path (str): Path to the CSV file containing polygon data.
    
    Returns:
    - list: A list of Polygon objects created from the CSV data.
    """
    from shapely.geometry import Polygon
    import ast
    
    df = pd.read_csv(file_path)
    polygons = []

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        points = []
        # Iterate over each column in the row
        for point_str in row:
            if pd.notna(point_str):  # Check if the value is not NaN
                point_list = ast.literal_eval(point_str)  # Convert string to list
                points.append((point_list[0], point_list[1]))
        
        if points:
            polygons.append(Polygon(points))
    
    return polygons

def write_polygons_to_csv(polygons, file_path):
    """
    Writes a list of Polygon objects to a CSV file.
    
    Args:
    - polygons (list): A list of shapely Polygon objects.
    - file_path (str): Path where the CSV file will be saved.
    """
    data = []

    for poly in polygons:
        # Get the exterior coordinates (excluding the repeated last point)
        coords = list(poly.exterior.coords)[:-1]
        # Format each coordinate as a stringified list
        row = [str([x, y]) for x, y in coords]
        data.append(row)

    # Pad rows to equal length if necessary
    max_len = max(len(row) for row in data)
    for row in data:
        row.extend([None] * (max_len - len(row)))

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def read_image_from_path(input_path):
    """
    Reads an image or the first frame of a video from the given file path.
    
    Args:
    - input_path (str): Path to the image or video file.
    
    Returns:
    - numpy.ndarray: The image or the first frame of the video.
    
    Raises:
    - ValueError: If the file format is unsupported.
    """
    # Detect file extension
    import os
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    # Supported video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    if ext in video_extensions:
        # Process as a video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {input_path}.")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Error: Could not read the first frame from video file {input_path}.")
    
    elif ext in image_extensions:
        # Process as an image
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Error: Could not open image file {input_path}.")
    
    else:
        # Raise error for unsupported format
        raise ValueError(f"Unsupported file format: {ext}. Supported formats are: {video_extensions + image_extensions}")
    
    return frame

def save_to_json(data, polygon_zones, filename):
    import json
    from .json_utils import serialise_data, wkt_dumps
    serialized_data = serialise_data(data)
    serialized_data['polygons'] = [wkt_dumps(poly) for poly in polygon_zones]

    with open(filename, 'w') as f:
        json.dump(serialized_data, f, indent=2)