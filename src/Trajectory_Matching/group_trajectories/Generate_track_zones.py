from .Group_traj import group_trajectories
from ..Utils.utils import draw_box
from shapely.geometry import Polygon, box, LineString
from shapely import union, difference, intersection
from shapely.ops import split

def get_sink_source(traj_dict):
    # from Group_traj generate trajectories groups based on frechet similarity
    # for each group, get start and end points and generate a zone that coveers them all, plus a buffer
    # once zones are made, warp overlapping zones to so no overlap and share overlapped area, this allows us to be more generous with original zone generation, but maintain tight zones when close
    # finally, attempt to detect transient/variable zones where a traj tend to switch paths and are less predicatable, might be possible by comparing final zones to frechet clusters
    groups = group_trajectories(traj_dict)
    
    rough_zones = get_rough_boxes(groups)
    final_zones = warp_overlapping_zones(rough_zones)

    # transient_zones = detect_transient_zones(groups, rough_zones)
    
    return final_zones, groups
    
def get_rough_boxes(groups):
    # i may also look to allign them with road or movement direction, would likely need new draw function
    rough_boxes = []
    for group in groups:
        start_points = [traj[0] for traj in group]
        end_points = [traj[-1] for traj in group]
        source_box_coords = draw_box(start_points, buffer=0.3)
        sink_box_coords = draw_box(end_points, buffer=0.3)
        # Convert tuples (x_min, y_min, x_max, y_max) to Polygon objects
        source_box = box(*source_box_coords)
        sink_box = box(*sink_box_coords)
        rough_boxes.append({'source': source_box, 'sink': sink_box})
    return rough_boxes

def warp_overlapping_zones(rough_boxes):
    """Warp overlapping zones so they touch but don't overlap, sharing edges"""
    # Make a copy to avoid modifying original
    boxes = [dict(item) for item in rough_boxes]
    
    # Loop through boxes, check for overlaps and split them
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            # Check source boxes
            if check_union(boxes[i]['source'], boxes[j]['source']):
                new_box1, new_box2 = split_box(boxes[i]['source'], boxes[j]['source'])
                boxes[i]['source'] = new_box1
                boxes[j]['source'] = new_box2
            
            # Check sink boxes
            if check_union(boxes[i]['sink'], boxes[j]['sink']):
                new_box1, new_box2 = split_box(boxes[i]['sink'], boxes[j]['sink'])
                boxes[i]['sink'] = new_box1
                boxes[j]['sink'] = new_box2
    
    return boxes

def check_union(box1, box2):
    """Check if two Polygon boxes overlap or touch"""
    return box1.intersects(box2)

def split_box(box1, box2):
    """
    Split the union of two overlapping boxes by a line perpendicular to the line connecting their centers.
    Returns two Polygon objects that share an edge.
    """    
    # Get centroids
    centre1 = (box1.centroid.x, box1.centroid.y)
    centre2 = (box2.centroid.x, box2.centroid.y)
    
    # Get union bounds
    union_box = union(box1, box2)
    
    perp_line = get_perp_line(centre1, centre2)
    
    parts = list(split(union_box, perp_line).geoms)

    if len(parts) != 2:
        raise ValueError(f"Expected 2 parts, got {len(parts)}")

    poly1, poly2 = parts
    
    # join box1-union with poly1 and box2-union with poly2
    
    box1 = box1.difference(union_box).union(poly1)
    box2 = box2.difference(union_box).union(poly2)
    
    return box1, box2

def get_perp_line(point1, point2):
    # centres
    x1, y1 = point1
    x2, y2 = point2

    # midpoint of the centre line
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # direction of the centre line
    dx = x2 - x1
    dy = y2 - y1

    # perpendicular direction: (-dy, dx) or (dy, -dx)
    pdx = -dy
    pdy = dx

    # choose how long you want the perpendicular line to be
    L = 4000  # large enough to cross your geometry

    perp_line = LineString([
        (mx - pdx * L, my - pdy * L),
        (mx + pdx * L, my + pdy * L),
    ])
    
    return perp_line

def detect_transient_zones(groups, rough_boxes):
    return [] # placeholder, decide if to implement

def draw_aligned_box(points, buffer=0.3):
    # see above, align rectangle with move dir, maybe gives more context, leverage car info? probably not necassary though
    return draw_box(points, buffer)