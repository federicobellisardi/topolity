#!/usr/bin/env python3
"""DEM extraction and fine-grid graph generation for a city."""
import os
import argparse
import pickle
import csv
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.affinity import translate as shapely_translate, rotate as shapely_rotate, scale as shapely_scale
import json
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree
import gc
import rasterio
from pyproj import Transformer

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import contextily as ctx
plt.rcParams.update({'font.size': 22})

from shapely.wkt import loads as load_wkt
from data_processing import DEMReader
from utils import logger

import time

VARIANT_COUNT = 0
ROOT = "/home/fbellisardi/code/topolity"
DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "data_processed")

_G = None
_GRAPHS_DIR = None
_CENTER_Y = None
_CENTER_X = None
_LAND_MASK = None
_SAFE_LAND_MASK = None   # land_mask pre-buffered by 1e-5° — computed once per city
# Coordinates of nodes that are on land in the ORIGINAL graph.
# Bridge / waterfront nodes that the original graph contains but that fall inside
# a lake polygon are excluded here so that they don't cause every transformation
# to be rejected.  Only a transformation that moves originally-on-land nodes into
# water is rejected.
_ORIG_LAND_NODE_IDS = None  # frozenset — node IDs whose original position is on land
_ORIG_LAND_XS = None        # np.ndarray — original xs of on-land nodes (for generation phase)
_ORIG_LAND_YS = None        # np.ndarray — original ys of on-land nodes
_DEM_TREE = None
_DEM_ALTS = None
_CMAP = None
_LAND_CHECK_MODE = 'sample'
_DEM_MODE = 'tree'
_DEM_SRC = None
_DEM_TRANSFORMER = None


def _sample_on_land_nodes(n_random: int = 45) -> tuple:
    """Return (xs, ys) of a representative sample of originally-on-land nodes.

    Includes the 5 boundary nodes (4 corners + centre of the on-land set) plus
    ``n_random`` uniformly random nodes.  The boundary nodes make the generation-
    phase check much more likely to catch peripheral nodes adjacent to water bodies,
    reducing the gap between the fast pre-filter and the authoritative full check.
    Falls back to (None, None) when the on-land arrays are not yet initialised.
    """
    if _ORIG_LAND_XS is None or len(_ORIG_LAND_XS) == 0:
        return None, None

    xs, ys = _ORIG_LAND_XS, _ORIG_LAND_YS
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2

    boundary_idx: set[int] = set()
    for tx, ty in [(xs.min(), ys.min()), (xs.max(), ys.min()),
                   (xs.min(), ys.max()), (xs.max(), ys.max()), (cx, cy)]:
        boundary_idx.add(int(np.argmin((xs - tx)**2 + (ys - ty)**2)))

    remaining = np.setdiff1d(np.arange(len(xs)), list(boundary_idx))
    rng = np.random.default_rng(seed=42)
    n_r = min(n_random, len(remaining))
    rand_idx = rng.choice(remaining, size=n_r, replace=False) if n_r > 0 else np.array([], dtype=int)
    sample_idx = np.array(sorted(boundary_idx) + rand_idx.tolist())
    return xs[sample_idx], ys[sample_idx]


def _points_in_geom(geom, xs, ys):
    """Vectorised containment check (Shapely 2.x contains_xy).

    xs, ys must be 1-D numpy arrays of the same length.
    Returns True when *every* point is inside `geom`.
    """
    import shapely as _shp
    return bool(_shp.contains_xy(geom, xs, ys).all())

def _init_worker(dem_file, graphs_dir, center_y, center_x, cmap_N=None, dem_mode='tree'):
    """Worker initialiser for mp.Pool.

    Large read-only objects (_G, _DEM_TREE/_DEM_ALTS, _LAND_MASK, _SAFE_LAND_MASK)
    are inherited from the parent process via fork (Linux copy-on-write) — they are
    NOT serialised through this call, which keeps startup fast and memory efficient.

    Rasterio file handles are NOT fork-safe, so raster mode workers open their own.
    """
    global _GRAPHS_DIR, _CENTER_Y, _CENTER_X, _CMAP
    global _DEM_MODE, _DEM_SRC, _DEM_TRANSFORMER
    _GRAPHS_DIR = graphs_dir
    _CENTER_Y = center_y
    _CENTER_X = center_x
    _DEM_MODE = dem_mode
    _CMAP = plt.get_cmap('Set1', int(cmap_N)) if cmap_N is not None else plt.get_cmap('Set1')
    # tree mode: _DEM_TREE and _DEM_ALTS already in parent memory → inherited via fork
    # raster mode: rasterio file handles are not fork-safe; each worker opens its own
    if _DEM_MODE == 'raster' and _DEM_SRC is None:
        _DEM_SRC = rasterio.open(dem_file)
        _DEM_TRANSFORMER = None
        if _DEM_SRC.crs and _DEM_SRC.crs.to_string() != "EPSG:4326":
            _DEM_TRANSFORMER = Transformer.from_crs("EPSG:4326", _DEM_SRC.crs, always_xy=True)


def _resolve_workers(requested):
    """Return the number of workers to use, SLURM-aware."""
    if requested is not None and requested > 0:
        return requested
    for env_var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        val = os.environ.get(env_var)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return mp.cpu_count()


def _nodes_on_land(G, safe_land, mode='sample'):
    """Check that *originally-on-land* graph nodes still lie inside `safe_land`.

    Only nodes whose original position was on land are checked.  Nodes already
    inside a water body in the original embedding (bridges over lakes, waterfront
    roads misclassified by ne_10m_lakes.shp) are intentionally excluded: their
    presence in water existed before any transformation, so they must not cause
    every variant to be rejected.  A transformation is rejected only when it
    moves a formerly-on-land node into water.

    Positions are read directly from G (the variant graph), so the check is
    correct for translations, rotations, and scales alike.

    mode='sample': fast check on a random subset of on-land nodes (≤25).
    mode='full':   strict check on ALL originally-on-land nodes.
    """
    # Fall back to full-graph check when on-land pre-filter is not available.
    if _ORIG_LAND_NODE_IDS is None or len(_ORIG_LAND_NODE_IDS) == 0:
        nodes = list(G.nodes(data=True)) if mode == 'full' else sample_graph_nodes(G)
        xs = np.fromiter((d["x"] for _, d in nodes), dtype=float, count=len(nodes))
        ys = np.fromiter((d["y"] for _, d in nodes), dtype=float, count=len(nodes))
        return _points_in_geom(safe_land, xs, ys)

    land_ids = list(_ORIG_LAND_NODE_IDS)
    if mode == 'sample' and len(land_ids) > 25:
        rng = np.random.default_rng(seed=42)
        land_ids = [land_ids[i] for i in rng.choice(len(land_ids), 25, replace=False)]

    # Read the TRANSFORMED positions of those specific nodes from G_var.
    xs = np.fromiter((G.nodes[n]["x"] for n in land_ids), dtype=float, count=len(land_ids))
    ys = np.fromiter((G.nodes[n]["y"] for n in land_ids), dtype=float, count=len(land_ids))
    return _points_in_geom(safe_land, xs, ys)

def is_dem_complete(path):
    return os.path.exists(path) and os.path.getsize(path) > 1e6

def download_dem(dem_path, bounds, api_key):
    if is_dem_complete(dem_path):
        logger.info(f"DEM already exists: {dem_path}")
        return
    minx, miny, maxx, maxy = bounds
    dx = maxx - minx
    dy = maxy - miny
    bbox = {
        "min_lon": minx - 0.5 * dx,
        "max_lon": maxx + 0.5 * dx,
        "min_lat": miny - 0.5 * dy,
        "max_lat": maxy + 0.5 * dy
    }
    logger.info(f"Downloading DEM for bbox: {bbox}")
    reader = DEMReader(dem_path)
    reader.download_dem(api_key, bbox, dem_file=dem_path)

def load_metropolis_bbox(json_path, city_key):
    with open(json_path) as f:
        data = json.load(f)
    if city_key not in data:
        raise KeyError(city_key)
    raw_coords = data[city_key]           
    swapped_coords = [(lon, lat) for lat, lon in raw_coords]
    return Polygon(swapped_coords)

# OSM geocoding queries for each city.
# Used by --polygon-source osm to download the municipality boundary.
# Polygon is cached to {city}/land_osm/{city}_osm_boundary.geojson after first download.
_OSM_QUERIES: dict[str, str] = {
    "amsterdam":         "Amsterdam, Netherlands",
    "atlanta":           "Atlanta, Georgia, USA",
    "bandung":           "Bandung, West Java, Indonesia",
    "bangkok":           "Bangkok, Thailand",
    "barcelone":         "Barcelona, Spain",
    "berlin":            "Berlin, Germany",
    "bogota":            "Bogotá, Colombia",
    "boston":            "Boston, Massachusetts, USA",
    "bruxelles":         "Brussels, Belgium",
    "buenosaires":       "Buenos Aires, Argentina",
    "caracas":           "Caracas, Venezuela",
    "chicago":           "Chicago, Illinois, USA",
    "dallas":            "Dallas, Texas, USA",
    "detroitwindsor":    "Detroit, Michigan, USA",
    "djakarta":          "Jakarta, Indonesia",
    "dublin":            "Dublin, Ireland",
    "guadalajara":       "Guadalajara, Mexico",
    "hongkong":          "Hong Kong",
    "houston":           "Houston, Texas, USA",
    "istanbul":          "Istanbul, Turkey",
    "kualalumpur":       "Kuala Lumpur, Malaysia",
    "lima":              "Lima, Peru",
    "lisbon":            "Lisbon, Portugal",
    "londres":           "London, United Kingdom",
    "losangeles":        "Los Angeles, California, USA",
    "madrid":            "Madrid, Spain",
    "manchester":        "Manchester, England, UK",
    "manille":           "Manila, Philippines",
    "mexico":            "Mexico City, Mexico",
    "miami":             "Miami, Florida, USA",
    "milan":             "Milan, Italy",
    "montreal":          "Montreal, Quebec, Canada",
    "moscou":            "Moscow, Russia",
    "nagoya":            "Nagoya, Japan",
    "newyork":           "New York City, New York, USA",
    "osaka":             "Osaka, Japan",
    "paris":             "Paris, France",
    "pekin":             "Beijing, China",
    "philadelphie":      "Philadelphia, Pennsylvania, USA",
    "phoenix":           "Phoenix, Arizona, USA",
    "riodejaneiro":      "Rio de Janeiro, Brazil",
    "rome":              "Rome, Italy",
    "saintpetersbourg":  "Saint Petersburg, Russia",
    "sandiego":          "San Diego, California, USA",
    "sanfrancisco":      "San Francisco, California, USA",
    "santiago":          "Santiago, Chile",
    "santodomingo":      "Santo Domingo, Dominican Republic",
    "saopaulo":          "São Paulo, Brazil",
    "seoul":             "Seoul, South Korea",
    "shanghai":          "Shanghai, China",
    "singapour":         "Singapore",
    "stockholm":         "Stockholm, Sweden",
    "sydney":            "Sydney, New South Wales, Australia",
    "taipei":            "Taipei, Taiwan",
    "tokyo":             "Tokyo, Japan",
    "toronto":           "Toronto, Ontario, Canada",
    "vancouver":         "Vancouver, British Columbia, Canada",
    "washington":        "Washington D.C., USA",
}


def load_osm_polygon(city_key: str, cache_dir: str) -> "Polygon | None":
    """Download (and cache) the OSM municipality boundary for a city.

    The polygon is saved as GeoJSON on first call; subsequent calls reload it
    from disk to avoid repeated network requests.

    Returns the polygon in EPSG:4326, or None on failure.
    """
    import json as _json
    from shapely.geometry import shape as _shape

    query = _OSM_QUERIES.get(city_key)
    if query is None:
        logger.warning(f"No OSM query defined for city '{city_key}'")
        return None

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{city_key}_osm_boundary.geojson")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                fc = _json.load(f)
            poly = _shape(fc["features"][0]["geometry"])
            logger.info(f"[{city_key}] OSM boundary loaded from cache: {cache_file}")
            return poly
        except Exception as e:
            logger.warning(f"[{city_key}] Could not load OSM cache ({e}), re-downloading…")

    try:
        import osmnx as ox
        gdf = ox.geocode_to_gdf(query)
        gdf = gdf.to_crs("EPSG:4326")
        poly = gdf.geometry.union_all()
        # Save to cache
        gdf.to_file(cache_file, driver="GeoJSON")
        logger.info(
            f"[{city_key}] OSM boundary downloaded for '{query}', "
            f"area={poly.area:.4f}°²  → {cache_file}"
        )
        return poly
    except Exception as e:
        logger.warning(f"[{city_key}] OSM boundary download failed: {e}")
        return None


def load_fua_polygon(gpkg_path, bbox_poly, city_key=None):
    """Load GHSL FUA geometry intersecting the provided bbox polygon.
    Returns a unified polygon or None if not found/failed.
    """
    try:
        fua_gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        logger.warning(f"Could not read FUA file {gpkg_path}: {e}")
        return None
    try:
        fua_4326 = fua_gdf.to_crs("EPSG:4326")
    except Exception:
        fua_4326 = fua_gdf
    try:
        candidates = fua_4326[fua_4326.intersects(bbox_poly)]
    except Exception as e:
        logger.warning(f"Failed spatial filter on FUA with bbox for {city_key}: {e}")
        return None
    if candidates.empty:
        logger.warning(f"No FUA geometry intersecting bbox for {city_key}")
        return None
    try:
        poly = candidates.geometry.union_all()
    except Exception:
        poly = candidates.unary_union
    return poly

def assign_altitudes(G, dem_reader):
    dem_gdf = dem_reader.get_pixel_centroids()
    dem_coords = np.vstack((dem_gdf.geometry.y.values, dem_gdf.geometry.x.values)).T
    dem_alts = dem_gdf['alt'].values
    tree = cKDTree(dem_coords)

    nodes = list(G.nodes(data=True))
    node_coords = np.array([[data['y'], data['x']] for _, data in nodes])
    dists, idxs = tree.query(node_coords, k=1)
    missing = 0

    for (node, data), alt_idx in zip(nodes, idxs):
        z = float(dem_alts[alt_idx])
        data['z'] = z
        if z == 0:
            missing += 1
    if missing:
        logger.warning(f"{missing} nodes without assigned altitude.")
    return missing

def assign_altitudes_from_tree(G, tree, dem_alts):
    nodes = list(G.nodes(data=True))
    node_coords = np.array([[data['y'], data['x']] for _, data in nodes])
    _, idxs = tree.query(node_coords, k=1)
    missing = 0
    for (node, data), alt_idx in zip(nodes, idxs):
        z = float(dem_alts[int(alt_idx)])
        data['z'] = z
        if z == 0:
            missing += 1
    if missing:
        logger.warning(f"{missing} nodes without assigned altitude.")
    return missing


def assign_altitudes_from_raster(G, dem_src, transformer=None):
    nodes = list(G.nodes(data=True))
    coords = [(data['x'], data['y']) for _, data in nodes]
    if transformer is not None:
        coords = [transformer.transform(lon, lat) for lon, lat in coords]

    zs = [v[0] for v in dem_src.sample(coords)]
    missing = 0
    for (node, data), z in zip(nodes, zs):
        z_val = float(z) if z is not None and np.isfinite(z) else 0.0
        data['z'] = z_val
        if z_val == 0.0:
            missing += 1
    if missing:
        logger.warning(f"{missing} nodes without assigned altitude.")
    return missing

def compute_graph_statistics(G):
    zs = [d.get('z', 0) for _, d in G.nodes(data=True)]
    lengths = [d.get('length', 0) for _, _, d in G.edges(data=True)]
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'z_mean': sum(zs) / len(zs) if zs else 0,
        'z_min': min(zs) if zs else 0,
        'z_max': max(zs) if zs else 0,
        'edge_len_mean': sum(lengths) / len(lengths) if lengths else 0
    }


def validate_stats_file_schema(stats_file, expected_fieldnames):
    """Fail fast when an existing stats CSV has a different header schema."""
    if not os.path.exists(stats_file) or os.path.getsize(stats_file) == 0:
        return

    with open(stats_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header is None:
        return

    normalized = [h.strip() for h in header]
    if normalized != expected_fieldnames:
        raise RuntimeError(
            f"Existing stats schema mismatch in {stats_file}. "
            f"Found columns={normalized}, expected={expected_fieldnames}. "
            "Move/remove the old CSV (or use a new output path) before resuming."
        )

def _apply_geom_transform(G, fn):
    """Apply `fn(coords_2d) -> coords_2d` to all edge geometries in batch (Shapely 2.x).

    Uses shapely.transform which creates NEW geometry objects, so the originals
    in the parent graph (_G) are unaffected even when G is a shallow copy.
    """
    import shapely as _shp
    items = [(d, d["geometry"]) for _, _, _, d in G.edges(keys=True, data=True)
             if d.get("geometry") is not None]
    if not items:
        return
    geoms = np.empty(len(items), dtype=object)
    for i, (_, g) in enumerate(items):
        geoms[i] = g
    new_geoms = _shp.transform(geoms, fn)
    for (d, _), ng in zip(items, new_geoms):
        d["geometry"] = ng


def translate_graph(G, offset):
    ox, oy = float(offset[0]), float(offset[1])
    nodes, data = zip(*G.nodes(data=True))
    n = len(nodes)
    xs = np.fromiter((d['x'] for d in data), float, count=n)
    ys = np.fromiter((d['y'] for d in data), float, count=n)
    xs += ox; ys += oy
    nx.set_node_attributes(G, {nd: {'x': float(x), 'y': float(y)}
                                for nd, x, y in zip(nodes, xs, ys)})
    _apply_geom_transform(G, lambda c: c + np.array([ox, oy]))
    return G


def rotate_graph(G, angle_deg, origin=None):
    nodes, data = zip(*G.nodes(data=True))
    n = len(nodes)
    xs = np.fromiter((d['x'] for d in data), float, count=n)
    ys = np.fromiter((d['y'] for d in data), float, count=n)
    x0, y0 = (origin if origin is not None
               else ((xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2))
    theta = angle_deg * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = xs - x0, ys - y0
    xr = c * dx - s * dy + x0
    yr = s * dx + c * dy + y0
    nx.set_node_attributes(G, {nd: {'x': float(x), 'y': float(y)}
                                for nd, x, y in zip(nodes, xr, yr)})

    def _rot(coords):
        dx2 = coords[:, 0] - x0; dy2 = coords[:, 1] - y0
        return np.column_stack([c * dx2 - s * dy2 + x0, s * dx2 + c * dy2 + y0])

    _apply_geom_transform(G, _rot)
    return G


def scale_graph(G, scale_factor, axis='both', origin=None):
    nodes, data = zip(*G.nodes(data=True))
    n = len(nodes)
    xs = np.fromiter((d['x'] for d in data), float, count=n)
    ys = np.fromiter((d['y'] for d in data), float, count=n)
    x0, y0 = (origin if origin is not None
               else ((xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2))
    xr = x0 + scale_factor * (xs - x0) if axis in ('x', 'both') else xs.copy()
    yr = y0 + scale_factor * (ys - y0) if axis in ('y', 'both') else ys.copy()
    nx.set_node_attributes(G, {nd: {'x': float(x), 'y': float(y)}
                                for nd, x, y in zip(nodes, xr, yr)})
    xfact = float(scale_factor) if axis in ('x', 'both') else 1.0
    yfact = float(scale_factor) if axis in ('y', 'both') else 1.0

    def _scl(coords):
        return np.column_stack([x0 + xfact * (coords[:, 0] - x0),
                                y0 + yfact * (coords[:, 1] - y0)])

    _apply_geom_transform(G, _scl)
    return G

def _metric_transformer_from_graph(G):
    xs = [float(data['x']) for _, data in G.nodes(data=True)]
    ys = [float(data['y']) for _, data in G.nodes(data=True)]
    lon0 = float(np.mean(xs))
    lat0 = float(np.mean(ys))
    proj_str = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)


def _geometry_length_m(geom, transformer):
    coords = np.asarray(geom.coords, dtype=float)
    x_m, y_m = transformer.transform(coords[:, 0], coords[:, 1])
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def recompute_edge_lengths_from_nodes(G):
    transformer = _metric_transformer_from_graph(G)
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is not None:
            d["length"] = _geometry_length_m(geom, transformer)
            continue

        du = G.nodes[u]
        dv = G.nodes[v]
        x1, y1 = float(du["x"]), float(du["y"])
        x2, y2 = float(dv["x"]), float(dv["y"])
        x_m, y_m = transformer.transform([x1, x2], [y1, y2])
        d["length"] = float(np.hypot(x_m[1] - x_m[0], y_m[1] - y_m[0]))

    return G


def graph_geometries_consistent_with_nodes(G, sample_limit=250, tolerance_m=1.0):
    transformer = _metric_transformer_from_graph(G)
    edges = list(G.edges(data=True))
    if not edges:
        return True

    sample_size = min(sample_limit, len(edges))
    sample_indices = np.unique(np.linspace(0, len(edges) - 1, num=sample_size, dtype=int))

    for idx in sample_indices:
        u, v, edge_data = edges[int(idx)]
        geom = edge_data.get("geometry")
        if geom is None:
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start_geom = coords[0]
        end_geom = coords[-1]
        start_node = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        end_node = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))

        start_x, start_y = transformer.transform(start_geom[0], start_geom[1])
        end_x, end_y = transformer.transform(end_geom[0], end_geom[1])
        u_x, u_y = transformer.transform(start_node[0], start_node[1])
        v_x, v_y = transformer.transform(end_node[0], end_node[1])

        direct = max(np.hypot(start_x - u_x, start_y - u_y), np.hypot(end_x - v_x, end_y - v_y))
        swapped = max(np.hypot(start_x - v_x, start_y - v_y), np.hypot(end_x - u_x, end_y - u_y))
        if min(direct, swapped) > tolerance_m:
            return False

    return True



def normalize_rotation_angles(angles_deg):
    normalized = []
    for angle in angles_deg or []:
        a = float(angle) % 360.0
        if a > 180.0:
            a -= 360.0
        normalized.append(a)
    return sorted(set(normalized))

def _format_scale_token(scale_factor):
    return f"{float(scale_factor):.3f}".replace('.', 'p')

def sample_graph_nodes(graph, max_samples=25):
    """
    Sample representative nodes from graph for fast land checking.
    Returns boundary nodes (corners + center) plus random samples.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The graph to sample from
    max_samples : int
        Maximum number of nodes to sample (default 25)
        
    Returns:
    --------
    list : sampled nodes with data [(node_id, data), ...]
    """
    import random
    nodes_list = list(graph.nodes(data=True))
    
    if len(nodes_list) <= max_samples:
        return nodes_list
    
    # Get boundary and center nodes
    xs = [d['x'] for _, d in nodes_list]
    ys = [d['y'] for _, d in nodes_list]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    # Find nodes closest to extremes and center (5 boundary nodes)
    sample_nodes = []
    for target_x, target_y in [(min(xs), min(ys)), (max(xs), min(ys)), 
                                (min(xs), max(ys)), (max(xs), max(ys)),
                                (center_x, center_y)]:
        closest = min(nodes_list, key=lambda n: (n[1]['x']-target_x)**2 + (n[1]['y']-target_y)**2)
        sample_nodes.append(closest)
    
    # Add random samples
    remaining = max_samples - len(sample_nodes)
    if remaining > 0:
        sample_nodes.extend(random.sample(nodes_list, min(remaining, len(nodes_list))))
    
    return sample_nodes

def generate_fine_grid_offsets(step_meters=50, num_points=10, lat_ref=41.0, land_mask=None,
                               graph_bounds=None, seed=None, boundary_polygon=None, graph=None,
                               extra_angles=None, safe_land=None):
    """Generate translation offsets in admissible directions (on land).

    Tests cardinal directions (N/S/E/W) plus any additional angles supplied via
    `extra_angles` (e.g. [30, 45, 135]) and keeps those that keep every sampled
    graph node on land.

    Parameters
    ----------
    extra_angles : list[float] | None
        Additional translation angles in degrees (0 = East, 90 = North, …).
        Combined with the cardinal set [0, 90, 180, 270].
    safe_land : shapely.geometry | None
        Pre-buffered land mask (land_mask.buffer(1e-5)).  Computed on the fly
        when not provided.
    """
    if seed is not None:
        np.random.seed(seed)

    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(lat_ref))
    step_deg_lon = step_meters / meters_per_deg_lon
    step_deg_lat = step_meters / meters_per_deg_lat

    offsets = [((0.0, 0.0), 0.0)]

    if land_mask is None or boundary_polygon is None or graph_bounds is None:
        logger.warning("Missing land_mask or boundary_polygon; using cardinal directions only")
        for i in range(1, num_points + 1):
            offsets.append(((i * step_deg_lon,  0.0), 0.0))
            offsets.append(((-i * step_deg_lon, 0.0), 180.0))
            offsets.append(((0.0,  i * step_deg_lat), 90.0))
            offsets.append(((0.0, -i * step_deg_lat), 270.0))
        return offsets

    # Pre-compute safe_land once (expensive buffer operation)
    if safe_land is None:
        safe_land = land_mask.buffer(1e-5)

    all_angles = sorted(set([0, 90, 180, 270] + list(extra_angles or [])))

    # Use pre-filtered on-land node coordinates (excludes bridges/waterfront).
    # Representative sample: boundary nodes + random pool (see _sample_on_land_nodes).
    node_xs = node_ys = None
    if graph is not None:
        node_xs, node_ys = _sample_on_land_nodes()
        if node_xs is None:
            sample_nodes = sample_graph_nodes(graph)
            node_xs = np.fromiter((d["x"] for _, d in sample_nodes), dtype=float, count=len(sample_nodes))
            node_ys = np.fromiter((d["y"] for _, d in sample_nodes), dtype=float, count=len(sample_nodes))

    for i in range(1, num_points + 1):
        admissible_offsets = []

        for angle_deg in all_angles:
            angle_rad = np.radians(angle_deg)
            ox = i * step_deg_lon * np.cos(angle_rad)
            oy = i * step_deg_lat * np.sin(angle_rad)
            offset = (ox, oy)

            if graph is not None:
                on_land = _points_in_geom(safe_land, node_xs + ox, node_ys + oy)
            else:
                translated = shapely_translate(boundary_polygon, xoff=ox, yoff=oy)
                on_land = safe_land.covers(translated)

            if on_land:
                admissible_offsets.append((offset, float(angle_deg)))

        if admissible_offsets:
            offsets.extend(admissible_offsets)
            logger.info(
                f"Distance +{i * step_meters}m: {len(admissible_offsets)}/{len(all_angles)} "
                f"directions admissible"
            )
        else:
            logger.warning(f"Distance {i * step_meters}m: no admissible directions — stopping early")
            break

    return offsets

def generate_fine_rotations(angles_deg, boundary_polygon=None, land_mask=None, graph=None,
                            safe_land=None):
    """Filter rotation angles that keep sampled graph nodes on land (vectorised)."""
    if land_mask is None or (boundary_polygon is None and graph is None):
        return sorted(set(angles_deg))

    if safe_land is None and land_mask is not None:
        safe_land = land_mask.buffer(1e-5)

    if graph is not None:
        # Boundary-aware sample (see _sample_on_land_nodes)
        sxs, sys_ = _sample_on_land_nodes()
        if sxs is None:
            all_nodes = list(graph.nodes(data=True))
            sxs = np.fromiter((d['x'] for _, d in sample_graph_nodes(graph)), float)
            sys_ = np.fromiter((d['y'] for _, d in sample_graph_nodes(graph)), float)
        all_xs = _ORIG_LAND_XS if _ORIG_LAND_XS is not None else sxs
        all_ys = _ORIG_LAND_YS if _ORIG_LAND_YS is not None else sys_
        origin_x = (all_xs.min() + all_xs.max()) / 2
        origin_y = (all_ys.min() + all_ys.max()) / 2
    else:
        centroid = boundary_polygon.centroid
        origin_x, origin_y = centroid.x, centroid.y

    admissible_angles = []
    for angle in sorted(set(angles_deg)):
        if angle == 0:
            admissible_angles.append(angle)
            continue

        if graph is not None:
            theta = np.radians(angle)
            c, s_a = np.cos(theta), np.sin(theta)
            dx, dy = sxs - origin_x, sys_ - origin_y
            rx = c * dx - s_a * dy + origin_x
            ry = s_a * dx + c * dy + origin_y
            on_land = _points_in_geom(safe_land, rx, ry)
        else:
            rotated = shapely_rotate(boundary_polygon, angle, origin=(origin_x, origin_y))
            on_land = rotated.within(land_mask)

        if on_land:
            admissible_angles.append(angle)
        else:
            logger.info(f"Rotation {angle:+.1f}° rejected: some nodes would be in water")

    logger.info(f"Rotation angles: {len(admissible_angles)} of {len(set(angles_deg))} are admissible")
    return admissible_angles


def generate_fine_scales(scale_factors, axis, land_mask=None, graph=None, safe_land=None):
    """Filter anisotropic scale factors that keep sampled nodes on land (vectorised)."""
    if not scale_factors:
        return []
    unique_scales = sorted(set(float(s) for s in scale_factors))
    if land_mask is None or graph is None:
        return unique_scales

    if safe_land is None:
        safe_land = land_mask.buffer(1e-5)

    # Boundary-aware sample (see _sample_on_land_nodes)
    sxs, sys_ = _sample_on_land_nodes()
    if sxs is None:
        sample_nodes = sample_graph_nodes(graph)
        sxs = np.fromiter((d['x'] for _, d in sample_nodes), float)
        sys_ = np.fromiter((d['y'] for _, d in sample_nodes), float)
    all_xs = _ORIG_LAND_XS if _ORIG_LAND_XS is not None else sxs
    all_ys = _ORIG_LAND_YS if _ORIG_LAND_YS is not None else sys_
    origin_x = (all_xs.min() + all_xs.max()) / 2
    origin_y = (all_ys.min() + all_ys.max()) / 2

    admissible = []
    for scale in unique_scales:
        if scale <= 0:
            logger.warning(f"Scale factor {scale} rejected: must be > 0")
            continue

        dx, dy = sxs - origin_x, sys_ - origin_y
        if axis == 'x':
            tx, ty = origin_x + scale * dx, sys_
        elif axis == 'y':
            tx, ty = sxs, origin_y + scale * dy
        else:
            tx, ty = origin_x + scale * dx, origin_y + scale * dy

        if _points_in_geom(safe_land, tx, ty):
            admissible.append(scale)
        else:
            logger.info(f"Scale {scale:.3f} on axis {axis} rejected: some nodes would be in water")

    logger.info(f"Scale factors on axis {axis}: {len(admissible)} of {len(unique_scales)} are admissible")
    return admissible

def _save_fua_map(city: str, G, fua_poly, land_poly, out_path: str,
                  boundary_label: str = "FUA boundary") -> None:
    """Save a geolocated PNG map of the FUA showing the road network on a basemap.

    Saved once per city at {city_folder}/{city}_fua_map.png.
    The map shows:
      - The FUA polygon boundary (blue outline)
      - The road network (orange edges)
      - A CartoDB Positron basemap via contextily
    """
    try:
        import geopandas as _gpd
        import contextily as _ctx
        from shapely.geometry import LineString as _LS

        # Build edge GeoDataFrame in EPSG:4326
        edge_records = []
        for u, v, data in G.edges(data=True):
            geom = data.get("geometry")
            if geom is None:
                geom = _LS([(G.nodes[u]["x"], G.nodes[u]["y"]),
                             (G.nodes[v]["x"], G.nodes[v]["y"])])
            edge_records.append({"geometry": geom})
        edges_gdf = _gpd.GeoDataFrame(edge_records, geometry="geometry", crs="EPSG:4326")

        # FUA polygon outline
        fua_gdf = _gpd.GeoDataFrame({"geometry": [fua_poly]}, crs="EPSG:4326")
        land_gdf = _gpd.GeoDataFrame({"geometry": [land_poly]}, crs="EPSG:4326")

        # Compute map extent with padding
        minx, miny, maxx, maxy = fua_poly.bounds
        padx = (maxx - minx) * 0.05
        pady = (maxy - miny) * 0.05

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(minx - padx, maxx + padx)
        ax.set_ylim(miny - pady, maxy + pady)

        # Basemap
        try:
            _ctx.add_basemap(ax, crs="EPSG:4326",
                             source=_ctx.providers.CartoDB.Positron,
                             attribution_size=7, zoom="auto")
        except Exception as _e:
            logger.warning(f"[{city}] Basemap failed: {_e}")

        # FUA boundary
        fua_gdf.boundary.plot(ax=ax, color="#1d4ed8", linewidth=2.0,
                              linestyle="--", alpha=0.9, zorder=3, label=boundary_label)

        # Land (clipped to FUA)
        land_gdf.plot(ax=ax, color="#d4a35a", alpha=0.15, zorder=2)

        # Road network
        edges_gdf.plot(ax=ax, color="#f97316", linewidth=0.5, alpha=0.7, zorder=4)

        ax.set_title(f"{city.title()} — {boundary_label}", fontsize=16, pad=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=11, frameon=False, loc="upper left")
        ax.grid(True, color="#94a3b8", alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[{city}] FUA map saved: {out_path}")
    except Exception as exc:
        logger.warning(f"[{city}] Could not save FUA map: {exc}")


def process_folder(folder, base_path, step_meters, num_points, rotation_angles,
                   ns_scale_factors, ew_scale_factors, api_key, workers=None,
                   seed=None, use_fua=False, fua_path=None, resume=False,
                   low_memory=False, dem_mode='auto', extra_translation_angles=None,
                   polygon_source='fua'):
    # ── Polygon-source suffix ─────────────────────────────────────────────────
    # 'fua'  → default paths: graphs_fine_grid/, land/, graphs/
    # 'osm'  → separate paths: graphs_fine_grid_osm/, land_osm/, graphs_osm/
    _sfx = "" if polygon_source == "fua" else f"_{polygon_source}"

    logger.info(f"[{folder}] Starting fine-grid processing (polygon_source={polygon_source})")
    folder_path = os.path.join(base_path, folder)
    csv_file = os.path.join(folder_path, 'data_useful.csv')
    if not os.path.exists(csv_file):
        logger.info(f"[{folder}] No data_useful.csv found, skipping.")
        return

    # Create subdirectory for fine-grid results (source-specific)
    fine_grid_dir = os.path.join(folder_path, f'graphs_fine_grid{_sfx}')
    os.makedirs(fine_grid_dir, exist_ok=True)

    json_bbox = "/data/workspaces/fbellisardi/metropolis.json"
    dem_dir = os.path.join(folder_path, 'dem')
    dem_file = os.path.join(dem_dir, f"{folder}_dem.tif")
    stats_file = os.path.join(fine_grid_dir, 'fine_grid_stats.csv')

    # Load data
    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    logger.info(f"[{folder}] Loaded CSV with {len(gdf)} records")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_y = (miny + maxy) / 2
    center_x = (minx + maxx) / 2

    # Calculate extended bounds for DEM (initial estimate before knowing exact offsets)
    # Use maximum expected offset for DEM download

    max_distance_m = step_meters * num_points
    meters_per_deg = 111000.0
    max_offset_deg = max_distance_m / meters_per_deg
    ext_bounds = [
        minx - max_offset_deg, miny - max_offset_deg,
        maxx + max_offset_deg, maxy + max_offset_deg
    ]

    os.makedirs(dem_dir, exist_ok=True)
    if not is_dem_complete(dem_file):
        download_dem(dem_file, ext_bounds, api_key)

    selected_dem_mode = dem_mode
    if selected_dem_mode == 'auto':
        selected_dem_mode = 'raster' if low_memory else 'tree'
    if selected_dem_mode not in ('tree', 'raster'):
        selected_dem_mode = 'tree'
    logger.info(f"[{folder}] DEM mode: {selected_dem_mode}")

    # ── Load city bbox / FUA BEFORE reading the DEM ───────────────────────────
    # The DEM TIF was downloaded for a large area (CSV bounds + 50 % buffer).
    # We only need altitudes for nodes inside the FUA + max translation offset.
    # Loading the FUA first lets us clip the raster read to that window →
    # rasterio reads 5–20× fewer pixels, dramatically reducing I/O and
    # cKDTree build time.
    city_bbox_poly = load_metropolis_bbox(json_bbox, folder)
    logger.info(f"[{folder}] City bbox polygon loaded")
    poly_for_graph = city_bbox_poly

    if polygon_source == 'osm':
        # ── OSM municipality boundary ──────────────────────────────────────────
        osm_cache_dir = os.path.join(folder_path, 'land_osm')
        osm_poly = load_osm_polygon(folder, osm_cache_dir)
        if osm_poly is not None:
            poly_for_graph = osm_poly
            logger.info(f"[{folder}] Using OSM municipality polygon with bounds {poly_for_graph.bounds}")
        else:
            logger.warning(f"[{folder}] OSM polygon unavailable; falling back to bbox")

    elif use_fua:
        # ── GHSL FUA polygon ───────────────────────────────────────────────────
        default_fua = "/home/fbellisardi/code/topolity/vars/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
        fua_gpkg = fua_path or (default_fua if os.path.exists(default_fua) else None)
        if fua_gpkg is None:
            alt_paths = [
                "/home/fbellisardi/code/topolity/data/extra/ghs_fua_v1/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg",
                "/home/fbellisardi/code/data/extra/ghs_fua_v1/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg",
                "/home/fbellisardi/code/twitter/extra/ghs_fua_v1/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg",
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    fua_gpkg = p
                    break
        if fua_gpkg and os.path.exists(fua_gpkg):
            fua_poly = load_fua_polygon(fua_gpkg, city_bbox_poly, folder)
            if fua_poly is not None:
                poly_for_graph = fua_poly
                logger.info(f"[{folder}] Using FUA polygon with bounds {poly_for_graph.bounds}")
            else:
                logger.warning(f"[{folder}] Falling back to bbox polygon; FUA not applicable")
        else:
            logger.warning(f"[{folder}] FUA gpkg not available; using bbox polygon")

    # DEM read window: FUA bounds + a margin that covers ALL transformations.
    # The TIF was downloaded for a larger area; rasterio reads only this window.
    #
    # Margin must cover:
    #   - Translations: max_offset_deg (step × num_points / 111 km)
    #   - Rotations: FUA_radius × sin(max_angle) — nodes at the boundary can move
    #     by up to this amount when the whole graph rotates
    #   - Scales: FUA_radius × (max_scale - 1)
    #   - Safety buffer: +0.05°
    _fua_b = poly_for_graph.bounds  # (minx, miny, maxx, maxy)
    _fua_radius = 0.5 * np.hypot(_fua_b[2] - _fua_b[0], _fua_b[3] - _fua_b[1])

    _max_rot_deg  = max((abs(a) for a in (rotation_angles or [0])), default=0.0)
    _rot_margin   = _fua_radius * np.sin(np.radians(_max_rot_deg))

    _max_scale    = max(
        max((s for s in (ns_scale_factors or [1.0])), default=1.0),
        max((s for s in (ew_scale_factors or [1.0])), default=1.0),
    )
    _scale_margin = _fua_radius * (_max_scale - 1.0)

    _dem_margin   = max(max_offset_deg, _rot_margin, _scale_margin) + 0.05

    _dem_read_bbox = (
        _fua_b[0] - _dem_margin,
        _fua_b[1] - _dem_margin,
        _fua_b[2] + _dem_margin,
        _fua_b[3] + _dem_margin,
    )
    logger.info(
        f"[{folder}] DEM read margin: {_dem_margin:.3f}° "
        f"(trans={max_offset_deg:.3f}°, rot={_rot_margin:.3f}°, scale={_scale_margin:.3f}°)"
    )
    # Cache filename encodes the bbox so it is invalidated if the FUA changes.
    _bbox_tag = "_".join(f"{v:.2f}" for v in _dem_read_bbox)

    dem_reader = DEMReader(dem_file)
    logger.info(f"[{folder}] DEMReader initialized (read window: {[round(v,3) for v in _dem_read_bbox]})")

    dem_tree = None
    dem_alts = None
    dem_src_main = None
    dem_transformer_main = None
    if selected_dem_mode == 'tree':
        _dem_mtime = os.path.getmtime(dem_file)
        _cache_coords = os.path.join(dem_dir, f"{folder}_dem_coords_{_bbox_tag}.npy")
        _cache_alts   = os.path.join(dem_dir, f"{folder}_dem_alts_{_bbox_tag}.npy")

        _cache_valid = (
            os.path.exists(_cache_coords) and os.path.exists(_cache_alts)
            and os.path.getmtime(_cache_coords) > _dem_mtime
            and os.path.getmtime(_cache_alts) > _dem_mtime
        )

        if _cache_valid:
            logger.info(f"[{folder}] Loading DEM from cache (skip TIF re-read)...")
            dem_coords = np.load(_cache_coords)
            dem_alts   = np.load(_cache_alts)
            logger.info(f"[{folder}] DEM cache loaded: {len(dem_alts):,} pixels")
        else:
            logger.info(f"[{folder}] Building DEM tree from TIF (clipped to FUA window)...")
            dem_gdf = dem_reader.get_pixel_centroids(bbox=_dem_read_bbox)
            dem_coords = np.vstack((dem_gdf.geometry.y.values, dem_gdf.geometry.x.values)).T
            dem_alts = dem_gdf['alt'].values.astype(float)
            del dem_gdf
            np.save(_cache_coords, dem_coords)
            np.save(_cache_alts,   dem_alts)
            logger.info(f"[{folder}] DEM cache saved: {len(dem_alts):,} pixels")

        dem_tree = cKDTree(dem_coords)
        del dem_coords
    else:
        dem_src_main = rasterio.open(dem_file)
        if dem_src_main.crs and dem_src_main.crs.to_string() != "EPSG:4326":
            dem_transformer_main = Transformer.from_crs("EPSG:4326", dem_src_main.crs, always_xy=True)
    gc.collect()

    # Cache DEM globally so forked workers reuse without reloading
    global _DEM_TREE, _DEM_ALTS, _DEM_MODE, _DEM_SRC, _DEM_TRANSFORMER
    _DEM_MODE = selected_dem_mode
    _DEM_TREE, _DEM_ALTS = dem_tree, dem_alts
    _DEM_SRC, _DEM_TRANSFORMER = dem_src_main, dem_transformer_main
    
    raw_dir = "/data/workspaces/fbellisardi/land"
    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land_global = gpd.read_file(shp_file).to_crs("EPSG:4326")
    # NOTE: do NOT union_all() the global land here.  It is only needed for the
    # overlay below; the land_mask for node checks is built locally after clipping
    # to the FUA (orders-of-magnitude smaller polygon → fast buffer, low RAM).

    # Load lakes within the full TRANSFORMATION EXTENT (FUA + dem_margin), not just
    # the FUA bbox. This ensures lakes reachable by translations/rotations/scales are
    # excluded from the land mask (e.g. Lake Ontario south of the Toronto FUA).
    lakes_shp = os.path.join(raw_dir, "ne_10m_lakes.shp")
    _lakes_gdf = None
    if os.path.exists(lakes_shp):
        _lakes_gdf = gpd.read_file(lakes_shp, bbox=_dem_read_bbox).to_crs("EPSG:4326")
        logger.info(f"[{folder}] Loaded {len(_lakes_gdf)} lake(s) within transformation extent")
    else:
        logger.warning(
            f"[{folder}] Lakes shapefile not found at {lakes_shp}; "
            "lake interiors will pass the land check (Toronto, Chicago may get lake translations)"
        )

    # Stale-cache detection: if poly_for_graph bounds changed since last run,
    # the land shapefile and original graph must be rebuilt from scratch.
    land_shp = os.path.join(folder_path, f'land{_sfx}', f'{folder}_clipped_land{_sfx}.shp')
    original_pkl = os.path.join(folder_path, f'graphs{_sfx}', 'graph_original.pkl')
    poly_fingerprint_path = os.path.join(folder_path, f'land{_sfx}', f'{folder}_poly_bounds{_sfx}.json')

    def _load_poly_fingerprint(path):
        try:
            with open(path) as fp:
                return json.load(fp)
        except Exception:
            return {}

    current_bounds = list(poly_for_graph.bounds)
    current_area = float(poly_for_graph.area)
    fp_data = _load_poly_fingerprint(poly_fingerprint_path)
    saved_bounds = fp_data.get('bounds')
    saved_area = fp_data.get('area')

    stale = False
    if saved_bounds is not None:
        bounds_changed = saved_bounds != current_bounds
        # Area comparison catches cities where FUA and bbox share the same outer
        # bounds but differ in shape (e.g. Santiago: FUA < rectangular bbox).
        area_changed = saved_area is not None and abs(saved_area - current_area) > 1e-6
        if bounds_changed or area_changed:
            logger.warning(
                f"[{folder}] poly_for_graph changed "
                f"(saved_bounds={saved_bounds}, current={current_bounds}, "
                f"saved_area={saved_area:.6f}, current_area={current_area:.6f}); "
                f"removing stale land_shp and original graph pkl."
            )
            stale = True
    elif os.path.exists(land_shp):
        # No fingerprint but land_shp exists: legacy state (pre-fingerprint pipeline).
        # Compare land_shp total_bounds against current poly bounds with 0.01° tolerance.
        # NOTE: area comparison is NOT done here to avoid false positives for cities that
        # were correctly built from a non-rectangular FUA (the area of a FUA polygon is
        # always smaller than its bounding-box rectangle, so the check would fire even when
        # nothing has changed). Area comparison is safe only when a fingerprint exists and
        # stored the actual poly_for_graph.area at build time.
        # Exception: Santiago has FUA bounds == bbox bounds; its land_shp must be manually
        # deleted to trigger a rebuild from FUA.
        try:
            legacy_land = gpd.read_file(land_shp)
            lb = list(legacy_land.total_bounds)  # [minx, miny, maxx, maxy]
            cb = current_bounds
            if any(abs(lb[i] - cb[i]) > 0.01 for i in range(4)):
                logger.warning(
                    f"[{folder}] Legacy land_shp bounds {[round(x,4) for x in lb]} "
                    f"differ from current poly bounds {[round(x,4) for x in cb]}; "
                    f"treating as stale."
                )
                stale = True
        except Exception as e:
            logger.warning(f"[{folder}] Could not read legacy land_shp for bounds check: {e}")

    if stale:
        for stale_path in [land_shp, original_pkl]:
            if os.path.exists(stale_path):
                os.remove(stale_path)
                logger.info(f"[{folder}] Removed stale file: {stale_path}")
        # Also clear all fine-grid outputs: they are derived from the old
        # (now-invalidated) polygon and must be fully rebuilt.
        if os.path.isdir(fine_grid_dir):
            n_pkl = 0
            for fname in os.listdir(fine_grid_dir):
                fpath = os.path.join(fine_grid_dir, fname)
                if fname.endswith('.pkl'):
                    os.remove(fpath)
                    n_pkl += 1
                elif fname in ('fine_grid_stats.csv', 'fine_grid_gravitational_work.csv',
                               'fine_grid_gravitational_work.pdf', 'fine_grid_gravitational_work.png'):
                    os.remove(fpath)
                    logger.info(f"[{folder}] Removed stale file: {fname}")
            if n_pkl:
                logger.info(f"[{folder}] Removed {n_pkl} stale fine-grid graph pkl(s)")

    if os.path.exists(land_shp):
        land_gdf = gpd.read_file(land_shp).to_crs('EPSG:4326')
    else:
        bbox_gdf = gpd.GeoDataFrame({'geometry': [poly_for_graph]}, crs="EPSG:4326")
        clipped = gpd.overlay(land_global, bbox_gdf, how='intersection')
        os.makedirs(os.path.join(folder_path, f'land{_sfx}'), exist_ok=True)
        clipped.to_file(land_shp)
        land_gdf = clipped.to_crs('EPSG:4326')
        logger.info(f"[{folder}] Created {land_shp}")
        with open(poly_fingerprint_path, 'w') as fp:
            json.dump({'bounds': current_bounds, 'area': current_area}, fp)
        logger.info(f"[{folder}] Saved poly fingerprint: bounds={current_bounds}, area={current_area:.6f}")

    polygon = land_gdf.geometry.union_all()

    # Build land_mask for admissibility checks — this must cover the full area
    # reachable by ALL transformations (translations, rotations, scales), not just
    # the FUA polygon.  We clip the global land to the transformation extent
    # (_dem_read_bbox) so that:
    #   - Inland cities (e.g. Bogota): land beyond the FUA boundary is recognised
    #     as land, so translations/rotations outside the FUA are not wrongly rejected
    #   - Coastal cities (e.g. Amsterdam): the ocean just outside the FUA remains
    #     water, so invalid translations are correctly rejected
    # Memory cost: clipping to _dem_read_bbox gives a ~1–10 MB polygon vs the
    # ~100–500 MB global union — orders of magnitude cheaper for buffer + checks.
    from shapely.geometry import box as _sg_box
    _extent_gdf = gpd.GeoDataFrame(
        {'geometry': [_sg_box(*_dem_read_bbox)]}, crs="EPSG:4326"
    )
    _land_extended = gpd.overlay(land_global, _extent_gdf, how='intersection')
    land_mask = _land_extended.geometry.union_all()
    del _land_extended, _extent_gdf

    # Free large global GeoDataFrames — no longer needed.
    del land_global, land_gdf
    gc.collect()

    # Subtract lakes within the transformation extent.
    # ne_10m_lakes.shp has a positional accuracy of ~200-500 m at 1:10M scale,
    # while OSM shorelines (used to build the graph) are at sub-metre accuracy.
    # Without a tolerance, OSM waterfront nodes appear "inside" the lake polygon
    # → every transformation is rejected for lakeside cities (Toronto, Chicago).
    # Fix: shrink each lake polygon by _LAKE_SHRINK_DEG before subtracting so that
    # nodes within that margin of the actual shore are treated as land.
    _LAKE_SHRINK_DEG = 0.003   # ≈ 300 m — covers the NaturalEarth resolution gap
    if _lakes_gdf is not None and not _lakes_gdf.empty:
        _local_lake_geom = _lakes_gdf.geometry.union_all()
        if not _local_lake_geom.is_empty:
            _shrunk = _local_lake_geom.buffer(-_LAKE_SHRINK_DEG)
            if not _shrunk.is_empty:
                land_mask = land_mask.difference(_shrunk)
                logger.info(
                    f"[{folder}] Lake mask applied ({len(_lakes_gdf)} lake(s) subtracted, "
                    f"shrunk by {_LAKE_SHRINK_DEG}°≈{int(_LAKE_SHRINK_DEG*111000)}m to handle shoreline resolution)"
                )
            else:
                logger.info(f"[{folder}] Lakes too small after shrink ({_LAKE_SHRINK_DEG}°) — skipped")
        del _lakes_gdf, _local_lake_geom, _shrunk
    elif _lakes_gdf is not None:
        del _lakes_gdf
    gc.collect()
    logger.info(f"[{folder}] Land mask ready (extent: {[round(v,3) for v in _dem_read_bbox]})")

    if os.path.exists(original_pkl):
        with open(original_pkl, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"[{folder}] Loaded existing original graph")
        
        # Ensure altitudes are assigned
        if all(data.get('z', 0) == 0 for _, data in list(G.nodes(data=True))[:10]):
            if selected_dem_mode == 'tree':
                missing = assign_altitudes_from_tree(G, dem_tree, dem_alts)
            else:
                missing = assign_altitudes_from_raster(G, dem_src_main, dem_transformer_main)
            logger.info(f"[{folder}] Re-assigned altitudes, missing: {missing}")
    else:
        logger.info(f"[{folder}] Building original graph from polygon...")
        tol = 25
        G = ox.graph_from_polygon(
            polygon,
            network_type='drive',
            retain_all=True,
            simplify=False,
            truncate_by_edge=True
        )
        G = ox.project_graph(G, to_crs='EPSG:3857')
        G = ox.consolidate_intersections(G, tolerance=tol, rebuild_graph=True, dead_ends=False)
        G = ox.simplify_graph(G)
        G = ox.project_graph(G, to_crs='EPSG:4326')

        if not nx.is_strongly_connected(G):
            cc = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(cc).copy()

        if selected_dem_mode == 'tree':
            missing = assign_altitudes_from_tree(G, dem_tree, dem_alts)
        else:
            missing = assign_altitudes_from_raster(G, dem_src_main, dem_transformer_main)
        logger.info(f"[{folder}] Assigned altitudes, missing: {missing}")

        # Save to original graphs folder (source-specific)
        os.makedirs(os.path.join(folder_path, f'graphs{_sfx}'), exist_ok=True)
        with open(original_pkl, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"[{folder}] Saved original graph")

    # Save domain map (once per source, labelled accordingly)
    _map_label = polygon_source   # "fua" or "osm"
    domain_map_path = os.path.join(folder_path, f'{folder}_{_map_label}_map.png')
    if not os.path.exists(domain_map_path):
        _save_fua_map(folder, G, poly_for_graph, polygon, domain_map_path,
                      boundary_label=_map_label.upper() + " boundary")

    # Pre-compute safe_land once — passed to all generation functions and process_variant workers
    safe_land = land_mask.buffer(1e-5)

    # Identify which original graph nodes are on land.
    # Nodes already inside a water polygon (bridges, waterfront roads mis-classified
    # by ne_10m_lakes.shp) are excluded from land checks so they cannot cause every
    # transformation to be spuriously rejected.
    g_nodes = list(G.nodes(data=True))
    _all_xs = np.fromiter((d["x"] for _, d in g_nodes), dtype=float, count=len(g_nodes))
    _all_ys = np.fromiter((d["y"] for _, d in g_nodes), dtype=float, count=len(g_nodes))
    import shapely as _shp_mod
    _on_land_mask = _shp_mod.contains_xy(safe_land, _all_xs, _all_ys)
    _land_node_ids = frozenset(n for (n, _), ok in zip(g_nodes, _on_land_mask) if ok)
    _land_xs = _all_xs[_on_land_mask]
    _land_ys = _all_ys[_on_land_mask]
    n_water_nodes = int((~_on_land_mask).sum())
    if n_water_nodes > 0:
        logger.info(
            f"[{folder}] {n_water_nodes} original nodes in water body (bridges/waterfront) — "
            f"excluded from land checks; only {len(_land_node_ids):,} on-land nodes checked."
        )
    graph_bounds = (float(_all_xs.min()), float(_all_ys.min()),
                    float(_all_xs.max()), float(_all_ys.max()))

    # !! Set on-land globals NOW — generation functions read them immediately below.
    # (The full globals block at the bottom of process_folder also sets _G, _LAND_MASK
    # etc., but _ORIG_LAND_* must be available before any generation call.)
    global _ORIG_LAND_NODE_IDS, _ORIG_LAND_XS, _ORIG_LAND_YS
    _ORIG_LAND_NODE_IDS = _land_node_ids
    _ORIG_LAND_XS       = _land_xs
    _ORIG_LAND_YS       = _land_ys

    # Generate fine-grid offsets (cardinal + extra_translation_angles, vectorised land checks)
    offsets = generate_fine_grid_offsets(step_meters, num_points, lat_ref=center_y,
                                         land_mask=land_mask, graph_bounds=graph_bounds, seed=seed,
                                         boundary_polygon=poly_for_graph, graph=G,
                                         extra_angles=extra_translation_angles, safe_land=safe_land)
    rotations = generate_fine_rotations(rotation_angles, boundary_polygon=poly_for_graph,
                                        land_mask=land_mask, graph=G, safe_land=safe_land)
    ns_scales = generate_fine_scales(ns_scale_factors, axis='y', land_mask=land_mask, graph=G,
                                     safe_land=safe_land)
    ew_scales = generate_fine_scales(ew_scale_factors, axis='x', land_mask=land_mask, graph=G,
                                     safe_land=safe_land)
    
    logger.info(f"[{folder}] Generated {len(offsets)} translation offsets (including original)")
    logger.info(f"[{folder}] Generated {len(rotations)} rotation angles")
    logger.info(f"[{folder}] Generated {len(ns_scales)} N-S scale factors")
    logger.info(f"[{folder}] Generated {len(ew_scales)} E-W scale factors")

    written_stats = 0

    # Build variant list
    variants = []
    
    # Add original
    variants.append({
        'variant': 'original',
        'type': 'original',
        'offset_x': 0.0,
        'offset_y': 0.0,
        'angle_deg': 0.0
    })
    
    # Add translations
    for idx, ((dx, dy), angle_deg) in enumerate(offsets[1:], start=1):
        # Calculate distance
        distance_x_m = abs(dx) * 111000 * np.cos(np.radians(center_y))
        distance_y_m = abs(dy) * 111000
        distance_m = int(np.sqrt(distance_x_m**2 + distance_y_m**2))
        
        variants.append({
            'variant': f'trans_{distance_m}m_a{int(angle_deg):+04d}',
            'type': 'translate',
            'offset': (dx, dy),
            'offset_x': dx,
            'offset_y': dy,
            'angle_deg': 0.0,
            'translation_angle': angle_deg,
            'translation_distance_m': distance_m
        })
    
    # Add rotations
    for angle in rotations:
        if angle == 0:
            continue  # Already added as original
        variants.append({
            'variant': f'rot_{angle:+.2f}deg'.replace('.', 'p'),
            'type': 'rotate',
            'angle_deg': angle,
            'offset_x': 0.0,
            'offset_y': 0.0
        })

    # Add anisotropic scales (dilations)
    for scale in ns_scales:
        if np.isclose(scale, 1.0):
            continue
        variants.append({
            'variant': f'scale_ns_{_format_scale_token(scale)}',
            'type': 'scale',
            'scale_factor': float(scale),
            'scale_axis': 'y',
            'offset_x': 0.0,
            'offset_y': 0.0,
            'angle_deg': 0.0
        })
    for scale in ew_scales:
        if np.isclose(scale, 1.0):
            continue
        variants.append({
            'variant': f'scale_ew_{_format_scale_token(scale)}',
            'type': 'scale',
            'scale_factor': float(scale),
            'scale_axis': 'x',
            'offset_x': 0.0,
            'offset_y': 0.0,
            'angle_deg': 0.0
        })

    # If resuming, skip variants already processed (both pickle exists and stats row present)
    if resume:
        completed_variants = set()
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        v = row.get('variant')
                        if v:
                            completed_variants.add(v)
            except Exception as e:
                logger.warning(f"[{folder}] Could not read existing stats for resume: {e}")

        variants_to_process = []
        skipped = 0
        for meta in variants:
            variant_name = meta['variant']
            pkl_path = os.path.join(fine_grid_dir, f'graph_{variant_name}.pkl')
            if os.path.exists(pkl_path) and (variant_name in completed_variants):
                skipped += 1
                continue
            variants_to_process.append(meta)

        if skipped > 0:
            logger.info(f"[{folder}] Resume enabled: skipping {skipped} already processed variants")
        variants = variants_to_process

    args_list = [(meta, idx) for idx, meta in enumerate(variants)]

    global VARIANT_COUNT
    VARIANT_COUNT = len(args_list)
    total_start = time.time()
    workers = _resolve_workers(workers)   # SLURM-aware; respects --workers override
    logger.info(f"[{folder}] Processing {VARIANT_COUNT} variants on {workers} core(s)"
                + (" [low-memory: extra GC per variant]" if low_memory else ""))

    # Set large read-only globals BEFORE creating the Pool so forked workers
    # inherit them via Linux copy-on-write without any serialisation cost.
    # NOTE: _ORIG_LAND_NODE_IDS/_XS/_YS already declared global earlier in this
    # function (before the generation calls) — do NOT repeat here (Python 3.13
    # raises SyntaxError for a second global after assignment).
    global _G, _LAND_MASK, _SAFE_LAND_MASK
    _G = G
    _LAND_MASK = land_mask
    _SAFE_LAND_MASK = safe_land
    # _ORIG_LAND_* already set above; redundant but readable reassign:
    _ORIG_LAND_NODE_IDS = _land_node_ids
    _ORIG_LAND_XS = _land_xs
    _ORIG_LAND_YS = _land_ys
    
    cmap_N = len(variants)

    fieldnames = [
        'variant', 'type', 'offset_x', 'offset_y', 'angle_deg',
        'translation_angle', 'translation_distance_m',
        'scale_factor', 'scale_axis',
        'num_nodes', 'num_edges', 'z_mean', 'z_min', 'z_max',
        'edge_len_mean', 'missing_altitude', 'on_land'
    ]

    validate_stats_file_schema(stats_file, fieldnames)

    write_header = not os.path.exists(stats_file)
    with open(stats_file, 'a', newline='') as fstats:
        writer = csv.DictWriter(fstats, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
    
        # initargs no longer include dem_tree / dem_alts: workers inherit them
        # via fork (COW).  Rasterio handles are re-opened per-worker inside
        # _init_worker for raster mode.
        _pool_initargs = (dem_file, fine_grid_dir, center_y, center_x, cmap_N, selected_dem_mode)

        if workers == 1:
            _init_worker(*_pool_initargs)
            for a in args_list:
                stats = process_variant(a)
                if stats is not None:
                    writer.writerow(stats)
                    written_stats += 1
                    fstats.flush()
                if low_memory:
                    gc.collect()
        else:
            with mp.Pool(processes=workers,
                         initializer=_init_worker,
                         initargs=_pool_initargs) as pool:
                for stats in pool.imap_unordered(process_variant, args_list, chunksize=1):
                    if stats is not None:
                        writer.writerow(stats)
                        written_stats += 1
                        fstats.flush()
    
    total_end = time.time()
    logger.info(f"[{folder}] Completed all variants in {total_end - total_start:.2f} seconds")
    
    logger.info(f"[{folder}] Exported {written_stats} stats rows to {stats_file}")

    if dem_src_main is not None:
        dem_src_main.close()

    logger.info(f"[{folder}] Finished fine-grid processing")

def process_variant(args):
    (meta, idx) = args
    variant_start = time.time()
    pid = os.getpid()
    try:
        core = os.sched_getcpu()
    except AttributeError:
        core = 'N/A'
    variant = meta['variant']
    logger.info(f"Processing variant: {variant} ({idx+1}/{VARIANT_COUNT}) in PID {pid} on core {core}")

    pkl_path = os.path.join(_GRAPHS_DIR, f'graph_{variant}.pkl')
    
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                G_var = pickle.load(f)

            if not graph_geometries_consistent_with_nodes(G_var):
                logger.warning(f"Stale edge geometry detected at {pkl_path}, recomputing variant {variant}")
                G_var = None
            else:
                G_var = recompute_edge_lengths_from_nodes(G_var)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(G_var, f)
                on_land = True  # Assume if saved, it was on land
            
        except (EOFError, pickle.UnpicklingError):
            logger.warning(f"Corrupted pickle detected at {pkl_path}, recomputing variant {variant}")
            try:
                os.remove(pkl_path)
            except OSError:
                pass
            G_var = None
    else:
        G_var = None

    if G_var is None:
        # Apply transformation
        if meta['type'] == 'original':
            G_var = _G.copy()
        elif meta['type'] == 'translate':
            G_var = translate_graph(_G.copy(), meta['offset'])
        elif meta['type'] == 'rotate':
            G_var = rotate_graph(_G.copy(), meta['angle_deg'])
        elif meta['type'] == 'scale':
            G_var = scale_graph(_G.copy(), meta.get('scale_factor', 1.0), axis=meta.get('scale_axis', 'both'))
        else:
            G_var = _G.copy()
            
        G_var = recompute_edge_lengths_from_nodes(G_var)
        
        # Check if nodes stay on land using the pre-buffered safe_land mask.
        if meta["type"] == "original":
            nodes_on_land = True
        else:
            nodes_on_land = _nodes_on_land(G_var, _SAFE_LAND_MASK, mode=_LAND_CHECK_MODE)

        if not nodes_on_land:
            logger.warning(f"Variant {variant} has nodes in a water body (lake/coast), skipping.")
            return {
                'variant': variant,
                'type': meta['type'],
                'offset_x': meta.get('offset_x', 0.0),
                'offset_y': meta.get('offset_y', 0.0),
                'angle_deg': meta.get('angle_deg', 0.0),
                'translation_angle': meta.get('translation_angle', 0.0),
                'translation_distance_m': meta.get('translation_distance_m', 0),
                'scale_factor': meta.get('scale_factor', 1.0),
                'scale_axis': meta.get('scale_axis', ''),
                'num_nodes': 0,
                'num_edges': 0,
                'z_mean': 0.0,
                'z_min': 0.0,
                'z_max': 0.0,
                'edge_len_mean': 0.0,
                'missing_altitude': 0,
                'on_land': False
            }
        
        # Assign altitudes
        if _DEM_MODE == 'tree':
            _ = assign_altitudes_from_tree(G_var, _DEM_TREE, _DEM_ALTS)
        else:
            _ = assign_altitudes_from_raster(G_var, _DEM_SRC, _DEM_TRANSFORMER)
        
        # Ensure edge lengths exist
        for u, v, d in G_var.edges(data=True):
            if "length" not in d:
                d["length"] = 1.0
        
        # Save pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(G_var, f)
        
        on_land = True

    # Compute statistics
    stats = compute_graph_statistics(G_var)
    stats['variant'] = variant
    stats['type'] = meta['type']
    stats['offset_x'] = meta.get('offset_x', 0.0)
    stats['offset_y'] = meta.get('offset_y', 0.0)
    stats['angle_deg'] = meta.get('angle_deg', 0.0)
    stats['translation_angle'] = meta.get('translation_angle', 0.0)
    stats['translation_distance_m'] = meta.get('translation_distance_m', 0)
    stats['scale_factor'] = meta.get('scale_factor', 1.0)
    stats['scale_axis'] = meta.get('scale_axis', '')
    stats['missing_altitude'] = sum(1 for _, d in G_var.nodes(data=True) if d.get('z', 0) == 0)
    stats['on_land'] = on_land

    variant_end = time.time()
    logger.info(f"Variant {variant} completed in {variant_end - variant_start:.2f} seconds")
    return stats

def main(api_key=None, example_city=None, cities=None, step_meters=50, num_points=10,
         rotation_angles=None, ns_scale_factors=None, ew_scale_factors=None,
         workers=None, seed=None, use_fua=False, fua_path=None, resume=False,
         base_path=None, low_memory=False, land_check='sample', dem_mode='tree',
         extra_translation_angles=None, polygon_source='fua'):
    config_path = "/home/fbellisardi/code/topolity/tools/conf/conf_extractor.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    api_key = api_key or config.get("api_key")
    example_city = example_city or config.get("city")
    base_path = base_path or DEFAULT_DATA_ROOT
    rotation_angles = normalize_rotation_angles(rotation_angles or [0.0])
    ns_scale_factors = [float(s) for s in (ns_scale_factors or [1.0])]
    ew_scale_factors = [float(s) for s in (ew_scale_factors or [1.0])]
    extra_translation_angles = [float(a) for a in (extra_translation_angles or [])]

    global _LAND_CHECK_MODE
    _LAND_CHECK_MODE = land_check if land_check in ('sample', 'full') else 'sample'

    if cities:
        city_list = [c for c in cities if c]
    elif example_city:
        city_list = [example_city]
    else:
        city_list = [
            city_folder for city_folder in sorted(os.listdir(base_path))
            if os.path.isdir(os.path.join(base_path, city_folder))
        ]

    for city_folder in city_list:
        process_folder(city_folder, base_path, step_meters, num_points,
                       rotation_angles, ns_scale_factors, ew_scale_factors,
                       api_key, workers, seed, use_fua, fua_path,
                       resume=resume, low_memory=low_memory, dem_mode=dem_mode,
                       extra_translation_angles=extra_translation_angles,
                       polygon_source=polygon_source)

    logger.info(f"All fine-grid processing completed.\nFiles in {base_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fine-grained exploration around original graph configuration'
    )
    parser.add_argument('--api_key', type=str,
                        help='OpenTopography API key')
    parser.add_argument('--city', type=str,
                        help='City folder to process')
    parser.add_argument('--cities', type=str, nargs='+',
                        help='List of city folders to process explicitly')
    parser.add_argument('--step-meters', type=float, default=50,
                        help='Distance between consecutive points in meters (default: 50)')
    parser.add_argument('--num-points', type=int, default=10,
                        help='Number of points in each direction (default: 10)')
    parser.add_argument('--rotation-angles', type=float, nargs='+',
                        default=[2, 5, 10, 15, 20],
                        help='Rotation angles in degrees (default: 2 5 10 15 20)')
    parser.add_argument('--ns-scale-factors', type=float, nargs='+',
                        default=[1.0],
                        help='N-S dilation factors (scale on latitude axis; 1.0 keeps original)')
    parser.add_argument('--ew-scale-factors', type=float, nargs='+',
                        default=[1.0],
                        help='E-W dilation factors (scale on longitude axis; 1.0 keeps original)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: all CPUs)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible direction selection')
    parser.add_argument('--use-fua', action='store_true',
                        help='Restrict processing polygon to GHSL FUA intersecting the city bbox')
    parser.add_argument('--fua-path', type=str,
                        help='Path to GHSL FUA .gpkg (optional; defaults to known locations)')
    parser.add_argument('--test', action='store_true',
                        help='Run a quick test with small parameters for validation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip variants already processed (pickle exists and stats row present)')
    parser.add_argument('--base-path', type=str, default=DEFAULT_DATA_ROOT,
                        help=f'Root directory containing the city folders (default: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--low-memory', action='store_true',
                        help='Reduce RAM usage (forces workers=1 and extra garbage collection)')
    parser.add_argument('--land-check', type=str, choices=['sample', 'full'], default='sample',
                        help='Land validation mode for transformed graphs (sample=fast, full=strict)')
    parser.add_argument('--dem-mode', type=str, choices=['auto', 'tree', 'raster'], default='tree',
                        help='DEM assignment mode: tree=reproducible baseline, raster=low-RAM, auto=depends on low-memory')
    parser.add_argument('--extra-translation-angles', type=float, nargs='+', default=[],
                        help='Additional translation angles in degrees to test alongside the cardinal '
                             'directions (0=E, 90=N, 180=W, 270=S). E.g. --extra-translation-angles 30 45 135 225')
    parser.add_argument('--polygon-source', type=str, choices=['fua', 'osm'], default='fua',
                        help='Spatial domain for the graph: '
                             '"fua" (default) = GHSL Functional Urban Area, '
                             '"osm" = OSM municipality boundary (downloaded via OSMnx). '
                             'Results are stored in separate subfolders so both analyses can coexist.')
    args = parser.parse_args()

    # Override with test configuration if --test flag is set
    if args.test:
        args.step_meters = 50
        args.num_points = 3
        args.rotation_angles = [-5, 5]
        args.ns_scale_factors = [0.95, 1.05]
        args.ew_scale_factors = [0.95, 1.05]
        args.workers = 1
        args.seed = 42
        args.use_fua = True
        args.low_memory = True
        logger.info("Running in TEST mode with: step=50m, points=3, rotations=[-5,5], scales=[0.95,1.05], workers=1, seed=42, use_fua=True, low_memory=True")

    main(args.api_key, args.city, args.cities, args.step_meters, args.num_points,
         args.rotation_angles, args.ns_scale_factors, args.ew_scale_factors,
         args.workers, args.seed, args.use_fua, args.fua_path,
         resume=args.resume, base_path=args.base_path,
         low_memory=args.low_memory, land_check=args.land_check,
         dem_mode=args.dem_mode,
         extra_translation_angles=args.extra_translation_angles,
         polygon_source=args.polygon_source)
