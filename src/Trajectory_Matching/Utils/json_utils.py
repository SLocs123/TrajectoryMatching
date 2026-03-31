import numpy as np
from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
from shapely.geometry.base import BaseGeometry


def serialise_shapely(obj):
    return wkt_dumps(obj)


def deserialise_shapely(wkt_str):
    return wkt_loads(wkt_str)


def _to_json_friendly(value):
    """Recursively convert numpy + shapely types inside dicts/lists/tuples."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, BaseGeometry):
        return wkt_dumps(value)

    if isinstance(value, dict):
        return {k: _to_json_friendly(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_friendly(v) for v in value]

    return value


def serialise_data(data):
    ser_outer = {}
    for input_box, inner_dict in data.items():
        input_key = serialise_shapely(input_box)

        ser_inner = {}
        for output_box, inner_value in inner_dict.items():
            output_key = serialise_shapely(output_box)
            ser_inner[output_key] = _to_json_friendly(inner_value)

        ser_outer[input_key] = ser_inner

    return ser_outer