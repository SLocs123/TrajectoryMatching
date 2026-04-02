from pathlib import Path

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
    final_zones = warp_overlapping_zones(rough_zones, groups)

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

def warp_overlapping_zones(rough_boxes, groups):
    """Warp overlapping zones so they touch but don't overlap, sharing edges"""
    if len(rough_boxes) != len(groups):
        raise ValueError('rough_boxes and groups must have the same length')

    # Make a copy to avoid modifying original
    boxes = [dict(item) for item in rough_boxes]
    
    # Loop through boxes, check for overlaps and split them
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            # Check source boxes
            if check_intersection(boxes[i]['source'], boxes[j]['source']):
                new_box1, new_box2 = split_box(
                    boxes[i]['source'],
                    boxes[j]['source'],
                    groups[i],
                    groups[j],
                    split_kind='source',
                )
                boxes[i]['source'] = new_box1
                boxes[j]['source'] = new_box2
            
            # Check sink boxes
            if check_intersection(boxes[i]['sink'], boxes[j]['sink']):
                new_box1, new_box2 = split_box(
                    boxes[i]['sink'],
                    boxes[j]['sink'],
                    groups[i],
                    groups[j],
                    split_kind='sink',
                )
                boxes[i]['sink'] = new_box1
                boxes[j]['sink'] = new_box2
    
    return boxes

def check_intersection(box1, box2):
    """Check if two Polygon boxes overlap or touch"""
    return box1.intersects(box2)

def split_box(box1, box2, group1, group2, split_kind):
    """
    Split the union of two overlapping boxes by a line perpendicular to the line connecting their centers.
    Returns two Polygon objects that share an edge.
    """    
    # Get centroids
    centre1 = (box1.centroid.x, box1.centroid.y)
    centre2 = (box2.centroid.x, box2.centroid.y)
    
    # Get intersection bounds
    intersection_box = intersection(box1, box2)
    perp_line = get_perp_line(centre1, centre2)
    
    parts = list(split(intersection_box, perp_line).geoms)
    group1_size = len(group1)
    group2_size = len(group2)
    visualize_polygons(
        box1,
        box2,
        intersection_box,
        parts,
        group1,
        group2,
        group1_size,
        group2_size,
    )

    if len(parts) != 2:
        raise ValueError(f"Expected 2 parts, got {len(parts)}")

    poly1, poly2 = parts
    
    # join box1-union with poly1 and box2-union with poly2
    
    box1 = box1.difference(intersection_box).union(poly1)
    box2 = box2.difference(intersection_box).union(poly2)
    
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




def _iter_polygons(geometry):
    if geometry.is_empty:
        return

    if geometry.geom_type == 'Polygon':
        yield geometry
        return

    if hasattr(geometry, 'geoms'):
        for sub_geometry in geometry.geoms:
            yield from _iter_polygons(sub_geometry)

def _plot_geometry(ax, geometry, facecolor, edgecolor, alpha, label=None):
    label_added = False

    for polygon in _iter_polygons(geometry):
        x_coords, y_coords = polygon.exterior.xy
        current_label = label if not label_added else None
        ax.fill(
            x_coords,
            y_coords,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=2,
            label=current_label,
        )
        label_added = True

        for interior in polygon.interiors:
            interior_x, interior_y = interior.xy
            ax.plot(interior_x, interior_y, color=edgecolor, linewidth=1)

def _plot_trajectory(ax, trajectory, color, label):
    if trajectory is None or len(trajectory) == 0:
        return

    x_coords = [point[0] for point in trajectory]
    y_coords = [point[1] for point in trajectory]
    ax.plot(x_coords, y_coords, color=color, linewidth=2.0, alpha=0.95, label=label)
    ax.scatter([x_coords[0]], [y_coords[0]], color=color, s=25)
    ax.scatter([x_coords[-1]], [y_coords[-1]], color=color, s=40, marker='x')

def _plot_group_trajectories(ax, group, color, label):
    if not group:
        return

    label_added = False
    for trajectory in group:
        current_label = label if not label_added else None
        _plot_trajectory(ax, trajectory, color, current_label)
        label_added = True

def visualize_polygons(
    box1,
    box2,
    union_box,
    parts,
    group1=None,
    group2=None,
    group1_size=None,
    group2_size=None,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; skipping polygon visualization')
        return

    background_path = Path.cwd() / 'data' / 'cam04.jpg'
    background_image = None
    background_extent = None
    if background_path.exists():
        background_image = plt.imread(background_path)
        image_height, image_width = background_image.shape[:2]
        background_extent = (0, image_width, image_height, 0)

    output_dir = Path.cwd() / 'data' / 'union_plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_index = 1
    output_path = output_dir / f'split_box_debug_{output_index:03d}.png'
    while output_path.exists():
        output_index += 1
        output_path = output_dir / f'split_box_debug_{output_index:03d}.png'

    figure, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    axes = axes.ravel()

    plot_configs = [
        ('All Objects', [
            (box1, '#4C78A8', '#1F3D5B', 0.35, 'box1'),
            (box2, '#F58518', '#8A4B08', 0.35, 'box2'),
            (union_box, '#54A24B', '#1B5E20', 0.2, 'union'),
            *[(part, '#E45756', '#7F1D1D', 0.3, f'part {index + 1}') for index, part in enumerate(parts)],
        ]),
        ('Unions Only', [
            (union_box, '#54A24B', '#1B5E20', 0.35, 'union'),
        ]),
        ('Boxes Only', [
            (box1, '#4C78A8', '#1F3D5B', 0.35, 'box1'),
            (box2, '#F58518', '#8A4B08', 0.35, 'box2'),
        ]),
        ('Parts Only', [
            *[(part, '#E45756', '#7F1D1D', 0.35, f'part {index + 1}') for index, part in enumerate(parts)],
        ]),
    ]

    geometries = [box1, box2, union_box, *parts]
    min_x = min(geometry.bounds[0] for geometry in geometries)
    min_y = min(geometry.bounds[1] for geometry in geometries)
    max_x = max(geometry.bounds[2] for geometry in geometries)
    max_y = max(geometry.bounds[3] for geometry in geometries)
    pad_x = max((max_x - min_x) * 0.05, 1.0)
    pad_y = max((max_y - min_y) * 0.05, 1.0)

    for ax, (title, layer_configs) in zip(axes, plot_configs):
        if background_image is not None:
            ax.imshow(background_image, extent=background_extent, zorder=0)

        for geometry, facecolor, edgecolor, alpha, label in layer_configs:
            _plot_geometry(ax, geometry, facecolor, edgecolor, alpha, label)

        if title in {'All Objects', 'Boxes Only'}:
            _plot_group_trajectories(ax, group1, '#0B3C5D', 'group1 trajectories')
            _plot_group_trajectories(ax, group2, '#B33C00', 'group2 trajectories')

        label_lines = []
        if group1_size is not None:
            label_lines.append(f'group1 - {group1_size} trajectories')
        if group2_size is not None:
            label_lines.append(f'group2 - {group2_size} trajectories')
        if label_lines:
            ax.text(
                0.02,
                0.98,
                '\n'.join(label_lines),
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=9,
                color='white',
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=4),
            )

        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        if background_image is not None:
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)
        else:
            ax.set_xlim(min_x - pad_x, max_x + pad_x)
            ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique_handles = []
            unique_labels = []
            for handle, legend_label in zip(handles, labels):
                if legend_label not in unique_labels:
                    unique_handles.append(handle)
                    unique_labels.append(legend_label)
            ax.legend(unique_handles, unique_labels)

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)
    print(f'Saved split box debug visualization to {output_path}')