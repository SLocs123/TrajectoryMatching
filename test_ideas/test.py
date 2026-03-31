"""Small utilities for averaging GPS coordinates.

Provides two approaches:
- `simple_mean`: arithmetic mean for latitude and circular mean for longitude.
- `spherical_mean`: vector (3D) centroid on the unit sphere.

For points within ~100m either approach is fine; for highest local accuracy
you can project to a local ENU plane and average in meters.
"""
import time as _time
import datetime as _dt
import math
from typing import List, Tuple

LatLon = Tuple[float, float]


def mean_angle(deg_list: List[float]) -> float:
	"""Return mean of angles in degrees using atan2 of summed unit vectors."""
	if not deg_list:
		raise ValueError("deg_list must not be empty")
	rads = [math.radians(d) for d in deg_list]
	s = sum(math.sin(a) for a in rads)
	c = sum(math.cos(a) for a in rads)
	return math.degrees(math.atan2(s, c))


def simple_mean(coords: List[LatLon]) -> LatLon:
	"""Simple mean: average latitudes, circular mean for longitudes."""
	if not coords:
		raise ValueError("coords must not be empty")
	lats = [lat for lat, lon in coords]
	lons = [lon for lat, lon in coords]
	return (sum(lats) / len(lats), mean_angle(lons))


def spherical_mean(coords: List[LatLon]) -> LatLon:
	"""Compute geographic midpoint by averaging 3D unit vectors and converting back.

	Robust and generally recommended for global correctness. Returns (lat, lon).
	"""
	if not coords:
		raise ValueError("coords must not be empty")
	x = y = z = 0.0
	for lat, lon in coords:
		phi = math.radians(lat)
		lam = math.radians(lon)
		x += math.cos(phi) * math.cos(lam)
		y += math.cos(phi) * math.sin(lam)
		z += math.sin(phi)
	n = len(coords)
	x /= n; y /= n; z /= n
	lon = math.degrees(math.atan2(y, x))
	hyp = math.hypot(x, y)
	lat = math.degrees(math.atan2(z, hyp))
	return lat, lon

def xy_to_latlon(x: float, y: float, ref_lat: float, ref_lon: float) -> LatLon:
    """Convert local x,y in meters back to lat/lon using the same reference."""
    dlat = y / R
    dlon = x / (R * math.cos(math.radians(ref_lat)))
    lat = ref_lat + math.degrees(dlat)
    lon = ref_lon + math.degrees(dlon)
    return lat, lon

def local_mean(coords: List[LatLon], ref: LatLon | None = None) -> LatLon:
    """Compute mean by projecting to meters, averaging, and converting back.

    `ref` if provided should be (lat, lon) used as the projection origin.
    Otherwise the arithmetic mean of coordinates is used as the reference.
    """
    if not coords:
        raise ValueError("coords must not be empty")
    if ref is None:
        ref = (sum(lat for lat, _ in coords) / len(coords), sum(lon for _, lon in coords) / len(coords))
    ref_lat, ref_lon = ref
    xs = []
    ys = []
    for lat, lon in coords:
        x, y = latlon_to_xy(lat, lon, ref_lat, ref_lon)
        xs.append(x); ys.append(y)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    return xy_to_latlon(mx, my, ref_lat, ref_lon)

def latlon_to_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """Approximate ENU-like x,y in meters using equirectangular approximation.

    x ~ east (meters), y ~ north (meters) relative to reference.
    Good for distances up to a few kilometers.
    """
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = R * dlon * math.cos(math.radians(ref_lat))
    y = R * dlat
    return x, y

def _timestamp(t):
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, _dt.datetime):
        return t.timestamp()
    raise TypeError("timestamp must be float seconds or datetime")

def haversine_m(a: LatLon, b: LatLon) -> float:
    """Haversine distance in meters between two lat/lon points."""
    lat1, lon1 = a; lat2, lon2 = b
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    sa = math.sin(dphi/2.0)
    sb = math.sin(dlambda/2.0)
    hav = sa*sa + math.cos(phi1)*math.cos(phi2)*sb*sb
    return 2 * R * math.asin(min(1.0, math.sqrt(hav)))

def compute_speeds(points: List[tuple]) -> List[tuple]:
    """Given list of (lat, lon, t) where t is seconds or datetime, return list of
    (distance_m, delta_s, speed_m_s) for each consecutive pair.
    """
    if len(points) < 2:
        return []
    out = []
    for (lat1, lon1, t1), (lat2, lon2, t2) in zip(points, points[1:]):
        dt = _timestamp(t2) - _timestamp(t1)
        if dt <= 0:
            out.append((0.0, dt, 0.0))
            continue
        d = haversine_m((lat1, lon1), (lat2, lon2))
        out.append((d, dt, d / dt))
    return out

if __name__ == "__main__":
	# Small demo: nearby points and wrap-around example
	samples = [
		(37.7749, -122.4194),  # SF
		(37.7750, -122.4195),
		(37.7748, -122.4193),
	]

	wrap = [(10.0, 179.0), (10.0, -179.0)]

	print("samples:", samples)
	print("simple_mean ->", simple_mean(samples))
	print("spherical_mean ->", spherical_mean(samples))
	print()
	print("wrap examples:", wrap)
	print("simple_mean ->", simple_mean(wrap))
	print("spherical_mean ->", spherical_mean(wrap))

	# --- Local planar conversion and averaging ---
	R = 6371000.0

	# Demo of local_mean
	print()
	print("local_mean(samples) ->", local_mean(samples))

	# --- Speed calculation (m/s) for timestamped points ---


	# Create a small timestamped track spaced by ~1 second
	now = _time.time()
	timed_samples = [
		(37.7749, -122.4194, now),
		(37.77492, -122.41945, now + 1.0),
		(37.77495, -122.41955, now + 2.5),
	]
	speeds = compute_speeds(timed_samples)
	print()
	print("timed_samples:")
	for p in timed_samples:
		print(p)
	print("dist_m, dt_s, speed_m_s:")
	for row in speeds:
		print(row)

# local mean likely best for the larger sets. potentially enables speed calculation at same time. use m/frame or m/s if available, fps useful

# offline may as well use the most accurate approach, probably wont do any averaging online, should look for speed calculations online (is this one fine?)