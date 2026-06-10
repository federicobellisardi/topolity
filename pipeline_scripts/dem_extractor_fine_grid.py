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
import json
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

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

_G = None
_GRAPHS_DIR = None
_CENTER_Y = None
_CENTER_X = None
_LAND_MASK = None
_DEM_TREE = None
_DEM_ALTS = None
_CMAP = None

def _init_worker(dem_file, graphs_dir, center_y, center_x, cmap_N=None, dem_tree=None, dem_alts=None):
    global _GRAPHS_DIR, _CENTER_Y, _CENTER_X, _DEM_TREE, _DEM_ALTS, _CMAP
    _GRAPHS_DIR = graphs_dir
    _CENTER_Y = center_y
    _CENTER_X = center_x
    if cmap_N is None:
        _CMAP = plt.get_cmap('Set1')
    else:
        _CMAP = plt.get_cmap('Set1', int(cmap_N))
    # If already set (inherited via fork) or explicitly provided, skip reloading
    if _DEM_TREE is None or _DEM_ALTS is None:
        if dem_tree is not None and dem_alts is not None:
            _DEM_TREE = dem_tree
            _DEM_ALTS = dem_alts
        else:
            reader = DEMReader(dem_file)
            dem_gdf = reader.get_pixel_centroids()
            dem_coords = np.vstack((dem_gdf.geometry.y.values, dem_gdf.geometry.x.values)).T
            _DEM_ALTS = dem_gdf['alt'].values.astype(float)
            _DEM_TREE = cKDTree(dem_coords)

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

def translate_graph(G, offset):
    nodes, data = zip(*G.nodes(data=True))
    xs = np.fromiter((d['x'] for d in data), float)
    ys = np.fromiter((d['y'] for d in data), float)
    xs += offset[0]
    ys += offset[1]
    attr = {n:{'x': x, 'y': y} for n, x, y in zip(nodes, xs, ys)}
    nx.set_node_attributes(G, attr)
    return G

def rotate_graph(G, angle_deg, origin=None):
    nodes, data = zip(*G.nodes(data=True))
    xs = np.fromiter((d['x'] for d in data), float)
    ys = np.fromiter((d['y'] for d in data), float)
    if origin is None:
        x0, y0 = (xs.max()+xs.min())/2, (ys.max()+ys.min())/2
    else:
        x0, y0 = origin
    theta = angle_deg * np.pi / 180.0
    c, s = np.cos(theta), np.sin(theta)
    dx = xs - x0
    dy = ys - y0
    xr =  c*dx - s*dy + x0
    yr =  s*dx + c*dy + y0
    attr = {n:{'x': x, 'y': y} for n, x, y in zip(nodes, xr, yr)}
    nx.set_node_attributes(G, attr)
    return G

def scale_graph(G, scale_factor, axis='both', origin=None):
    nodes, data = zip(*G.nodes(data=True))
    xs = np.fromiter((d['x'] for d in data), float)
    ys = np.fromiter((d['y'] for d in data), float)
    if origin is None:
        x0, y0 = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    else:
        x0, y0 = origin
    xr = xs.copy()
    yr = ys.copy()
    if axis in ('x', 'both'):
        xr = x0 + scale_factor * (xs - x0)
    if axis in ('y', 'both'):
        yr = y0 + scale_factor * (ys - y0)
    attr = {n: {'x': x, 'y': y} for n, x, y in zip(nodes, xr, yr)}
    nx.set_node_attributes(G, attr)
    return G

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
                               graph_bounds=None, seed=None, boundary_polygon=None, graph=None):
    """
    Generate translation offsets in admissible cardinal directions (on land).
    For each distance, tests North, South, East, and West and keeps the directions that keep all graph nodes on land.
    
    Parameters:
    -----------
    step_meters : float
        Distance between consecutive points in meters
    num_points : int
        Number of points at different distances from center
    lat_ref : float
        Reference latitude for degree-to-meter conversion
    land_mask : shapely.geometry
        Global land geometry to check if translated nodes stay on land
    graph_bounds : tuple
        (minx, miny, maxx, maxy) of the original graph
    seed : int, optional
        Random seed for reproducibility
    boundary_polygon : shapely.geometry, optional
        FUA or bbox polygon (used as fallback if graph not provided)
    graph : networkx.Graph, optional
        The graph whose nodes will be checked after translation
        
    Returns:
    --------
    list of tuples : ((offset_x_deg, offset_y_deg), angle_deg) with offset in degrees and chosen angle
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Approximate conversion at given latitude
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(lat)
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(lat_ref))
    
    step_deg_lon = step_meters / meters_per_deg_lon
    step_deg_lat = step_meters / meters_per_deg_lat
    
    offsets = [((0.0, 0.0), 0.0)]  # Original position with angle
    
    if land_mask is None or boundary_polygon is None or graph_bounds is None:
        # Fallback to cardinal directions if insufficient information
        logger.warning("Missing land_mask or boundary_polygon; using cardinal directions only")
        for i in range(1, num_points + 1):
            offsets.append((i * step_deg_lon, 0.0))   # East
            offsets.append((-i * step_deg_lon, 0.0))  # West
            offsets.append((0.0, i * step_deg_lat))   # North
            offsets.append((0.0, -i * step_deg_lat))  # South
        return offsets
    
    # Get graph center and corners for testing
    minx, miny, maxx, maxy = graph_bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # Test points: center and corners
    test_points = [
        (center_x, center_y),
        (minx, miny),
        (maxx, miny),
        (minx, maxy),
        (maxx, maxy)
    ]
    
    # Only explore cardinal directions (E, N, W, S) for each step
    from shapely.affinity import translate as shapely_translate
    cardinal_angles = [0, 90, 180, 270]

    for i in range(1, num_points + 1):
        admissible_offsets = []
        sample_nodes = sample_graph_nodes(graph) if graph is not None else None

        for angle_deg in cardinal_angles:
            angle_rad = np.radians(angle_deg)
            distance_deg_lon = i * step_deg_lon * np.cos(angle_rad)
            distance_deg_lat = i * step_deg_lat * np.sin(angle_rad)
            offset = (distance_deg_lon, distance_deg_lat)

            if graph is not None:
                all_nodes_on_land = all(
                    Point(d['x'] + offset[0], d['y'] + offset[1]).within(land_mask)
                    for _, d in sample_nodes
                )
                if all_nodes_on_land:
                    admissible_offsets.append((offset, angle_deg))
            else:
                translated_boundary = shapely_translate(boundary_polygon, xoff=offset[0], yoff=offset[1])
                if translated_boundary.within(land_mask):
                    admissible_offsets.append((offset, angle_deg))

        if admissible_offsets:
            offsets.extend(admissible_offsets)
            logger.info(
                f"Distance +{i * step_meters}m: {len(admissible_offsets)} cardinal directions admissible"
            )
        else:
            logger.warning(f"Distance {i * step_meters}m: No admissible cardinal directions found within boundary")
            break
    
    return offsets

def generate_fine_rotations(angles_deg, boundary_polygon=None, land_mask=None, graph=None):
    """
    Generate small rotation angles around original position, checking that rotated graph nodes stay on land.
    
    Parameters:
    -----------
    angles_deg : list of float
        Rotation angles in degrees (e.g., [-10, -5, -2, -1, 0, 1, 2, 5, 10])
    boundary_polygon : shapely.geometry, optional
        FUA or bbox polygon (used as fallback if graph not provided)
    land_mask : shapely.geometry, optional
        Global land geometry to check against
    graph : networkx.Graph, optional
        The graph whose nodes will be checked after rotation
        
    Returns:
    --------
    list of float : admissible rotation angles in degrees
    """
    from shapely.affinity import rotate as shapely_rotate
    
    if land_mask is None or (boundary_polygon is None and graph is None):
        # No filtering, return all angles
        return sorted(set(angles_deg))
    
    # Get rotation origin (graph center or polygon centroid)
    if graph is not None:
        xs = [d['x'] for _, d in graph.nodes(data=True)]
        ys = [d['y'] for _, d in graph.nodes(data=True)]
        origin_x = (min(xs) + max(xs)) / 2
        origin_y = (min(ys) + max(ys)) / 2
    else:
        centroid = boundary_polygon.centroid
        origin_x, origin_y = centroid.x, centroid.y
    
    admissible_angles = []
    
    for angle in sorted(set(angles_deg)):
        if angle == 0:
            admissible_angles.append(angle)
            continue
        
        # Check if rotated graph nodes stay on land
        if graph is not None:
            sample_nodes = sample_graph_nodes(graph)
            
            # Rotate sampled nodes around origin
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)
            all_nodes_on_land = True
            for _, d in sample_nodes:
                dx = d['x'] - origin_x
                dy = d['y'] - origin_y
                rotated_x = c * dx - s * dy + origin_x
                rotated_y = s * dx + c * dy + origin_y
                if not Point(rotated_x, rotated_y).within(land_mask):
                    all_nodes_on_land = False
                    break
            if all_nodes_on_land:
                admissible_angles.append(angle)
            else:
                logger.info(f"Rotation {angle:+.1f}° rejected: some nodes would be in sea")
        else:
            # Fallback to boundary polygon check
            rotated_boundary = shapely_rotate(boundary_polygon, angle, origin=(origin_x, origin_y))
            if rotated_boundary.within(land_mask):
                admissible_angles.append(angle)
            else:
                logger.info(f"Rotation {angle:+.1f}° rejected: not fully within land")
    
    logger.info(f"Rotation angles: {len(admissible_angles)} of {len(set(angles_deg))} are admissible")
    return admissible_angles

def generate_fine_scales(scale_factors, axis, land_mask=None, graph=None):
    """
    Filter anisotropic scale factors (x=E-W, y=N-S) that keep sampled nodes on land.

    Returns admissible scale factors including 1.0 if requested.
    """
    if not scale_factors:
        return []
    unique_scales = sorted(set(float(s) for s in scale_factors))
    if land_mask is None or graph is None:
        return unique_scales

    xs = [d['x'] for _, d in graph.nodes(data=True)]
    ys = [d['y'] for _, d in graph.nodes(data=True)]
    origin_x = (min(xs) + max(xs)) / 2
    origin_y = (min(ys) + max(ys)) / 2
    sample_nodes = sample_graph_nodes(graph)

    admissible = []
    for scale in unique_scales:
        if scale <= 0:
            logger.warning(f"Scale factor {scale} rejected: must be > 0")
            continue

        all_nodes_on_land = True
        for _, d in sample_nodes:
            dx = d['x'] - origin_x
            dy = d['y'] - origin_y
            if axis == 'x':
                tx = origin_x + scale * dx
                ty = d['y']
            elif axis == 'y':
                tx = d['x']
                ty = origin_y + scale * dy
            else:
                tx = origin_x + scale * dx
                ty = origin_y + scale * dy
            if not Point(tx, ty).within(land_mask):
                all_nodes_on_land = False
                break

        if all_nodes_on_land:
            admissible.append(scale)
        else:
            logger.info(f"Scale {scale:.3f} on axis {axis} rejected: some nodes would be in sea")

    logger.info(f"Scale factors on axis {axis}: {len(admissible)} of {len(unique_scales)} are admissible")
    return admissible

def process_folder(folder, base_path, step_meters, num_points, rotation_angles,
                   ns_scale_factors, ew_scale_factors, api_key, workers=None,
                   seed=None, use_fua=False, fua_path=None, resume=False):
    logger.info(f"[{folder}] Starting fine-grid processing")
    folder_path = os.path.join(base_path, folder)
    csv_file = os.path.join(folder_path, 'data_useful.csv')
    if not os.path.exists(csv_file):
        logger.info(f"[{folder}] No data_useful.csv found, skipping.")
        return

    # Create subdirectory for fine-grid results
    fine_grid_dir = os.path.join(folder_path, 'graphs_fine_grid')
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

    dem_reader = DEMReader(dem_file)
    logger.info(f"[{folder}] DEMReader initialized")
    dem_gdf = dem_reader.get_pixel_centroids()
    dem_coords = np.vstack((dem_gdf.geometry.y.values, dem_gdf.geometry.x.values)).T
    dem_alts = dem_gdf['alt'].values.astype(float)
    dem_tree = cKDTree(dem_coords)
    # Cache DEM globally so forked workers reuse without reloading
    global _DEM_TREE, _DEM_ALTS
    _DEM_TREE, _DEM_ALTS = dem_tree, dem_alts

    # Load city bbox polygon, optionally replace with FUA
    city_bbox_poly = load_metropolis_bbox(json_bbox, folder)
    logger.info(f"[{folder}] City bbox polygon loaded")
    poly_for_graph = city_bbox_poly
    if use_fua:
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
    
    raw_dir = "/data/workspaces/fbellisardi/land"
    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land_global = gpd.read_file(shp_file).to_crs("EPSG:4326")
    land_mask = land_global.geometry.union_all()

    # Check for existing clipped land shapefile
    land_shp = os.path.join(folder_path, 'land', f'{folder}_clipped_land.shp')
    if os.path.exists(land_shp):
        land_gdf = gpd.read_file(land_shp).to_crs('EPSG:4326')
    else:
        bbox_gdf = gpd.GeoDataFrame({'geometry': [poly_for_graph]}, crs="EPSG:4326")
        clipped = gpd.overlay(land_global, bbox_gdf, how='intersection')
        os.makedirs(os.path.join(folder_path, 'land'), exist_ok=True)
        clipped.to_file(land_shp)
        land_gdf = clipped.to_crs('EPSG:4326')
        logger.info(f"[{folder}] Created {land_shp}")
    
    polygon = land_gdf.geometry.union_all()

    # Load or build original graph
    original_pkl = os.path.join(folder_path, 'graphs', 'graph_original.pkl')
    if os.path.exists(original_pkl):
        with open(original_pkl, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"[{folder}] Loaded existing original graph")
        
        # Ensure altitudes are assigned
        if all(data.get('z', 0) == 0 for _, data in list(G.nodes(data=True))[:10]):
            missing = assign_altitudes(G, dem_reader)
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

        missing = assign_altitudes(G, dem_reader)
        logger.info(f"[{folder}] Assigned altitudes, missing: {missing}")

        # Save to original graphs folder if not exists
        os.makedirs(os.path.join(folder_path, 'graphs'), exist_ok=True)
        with open(original_pkl, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"[{folder}] Saved original graph")

    # Get graph bounds for offset generation
    graph_bounds = (
        min(d['x'] for _, d in G.nodes(data=True)),
        min(d['y'] for _, d in G.nodes(data=True)),
        max(d['x'] for _, d in G.nodes(data=True)),
        max(d['y'] for _, d in G.nodes(data=True))
    )

    # Generate fine-grid offsets using graph node checks (more precise than FUA polygon)
    offsets = generate_fine_grid_offsets(step_meters, num_points, lat_ref=center_y, 
                                        land_mask=land_mask, graph_bounds=graph_bounds, seed=seed,
                                        boundary_polygon=poly_for_graph, graph=G)
    rotations = generate_fine_rotations(rotation_angles, boundary_polygon=poly_for_graph, 
                                       land_mask=land_mask, graph=G)
    ns_scales = generate_fine_scales(ns_scale_factors, axis='y', land_mask=land_mask, graph=G)
    ew_scales = generate_fine_scales(ew_scale_factors, axis='x', land_mask=land_mask, graph=G)
    
    logger.info(f"[{folder}] Generated {len(offsets)} translation offsets (including original)")
    logger.info(f"[{folder}] Generated {len(rotations)} rotation angles")
    logger.info(f"[{folder}] Generated {len(ns_scales)} N-S scale factors")
    logger.info(f"[{folder}] Generated {len(ew_scales)} E-W scale factors")

    stats_records = []

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
    workers = workers or mp.cpu_count()
    logger.info(f"[{folder}] Processing {VARIANT_COUNT} variants on {workers} cores")
    
    global _G, _LAND_MASK
    _G = G
    _LAND_MASK = land_mask
    # Precompute which original nodes lie on land; nodes originally on water
    # (e.g., bridges) are excluded from the on-land infeasibility test.
    global _ORIG_ON_LAND
    _ORIG_ON_LAND = {n: Point(d['x'], d['y']).within(_LAND_MASK) for n, d in _G.nodes(data=True)}
    
    results = []
    cmap_N = len(variants)
    
    if workers == 1:
        _init_worker(dem_file, fine_grid_dir, center_y, center_x, cmap_N)
        for a in args_list:
            results.append(process_variant(a))
    else:
        with mp.Pool(processes=workers,
                     initializer=_init_worker,
                     initargs=(dem_file, fine_grid_dir, center_y, center_x, cmap_N)) as pool:
            results = pool.map(process_variant, args_list)
    
    total_end = time.time()
    logger.info(f"[{folder}] Completed all variants in {total_end - total_start:.2f} seconds")
    
    for stats in results:
        if stats is not None:
            stats_records.append(stats)

    # Write stats
    fieldnames = [
        'variant', 'type', 'offset_x', 'offset_y', 'angle_deg',
        'translation_angle', 'translation_distance_m',
        'scale_factor', 'scale_axis',
        'num_nodes', 'num_edges', 'z_mean', 'z_min', 'z_max', 
        'edge_len_mean', 'missing_altitude', 'on_land'
    ]
    
    if stats_records:
        write_header = not os.path.exists(stats_file)
        with open(stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(stats_records)
        logger.info(f"[{folder}] Exported {len(stats_records)} stats rows to {stats_file}")

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
        
        # Check if any node that was originally on land would be moved into water.
        # Nodes that were originally on water (e.g., bridge nodes) are ignored.
        nodes_on_land = True
        for n, d in G_var.nodes(data=True):
            orig_on_land = _ORIG_ON_LAND.get(n, True)
            if not orig_on_land:
                # original node already over water; skip strict check
                continue
            if not Point(d['x'], d['y']).within(_LAND_MASK):
                nodes_on_land = False
                break
        
        if not nodes_on_land:
            logger.warning(f"Variant {variant} has nodes in the sea, skipping.")
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
        missing = assign_altitudes_from_tree(G_var, _DEM_TREE, _DEM_ALTS)
        
        # Ensure edge lengths exist
        for u, v, d in G_var.edges(data=True):
            d.setdefault('length', d.get('length', 1))
        
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
         workers=None, seed=None, use_fua=False, fua_path=None, resume=False):
    config_path = "/home/fbellisardi/code/topolity/tools/conf/conf_extractor.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    api_key = api_key or config.get("api_key")
    example_city = example_city or config.get("city")
    base_path = config.get("base_path", "/data/workspaces/fbellisardi/data_processed")
    rotation_angles = normalize_rotation_angles(rotation_angles or [0.0])
    ns_scale_factors = [float(s) for s in (ns_scale_factors or [1.0])]
    ew_scale_factors = [float(s) for s in (ew_scale_factors or [1.0])]

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
                       api_key, workers, seed, use_fua, fua_path, resume=resume)

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
    args = parser.parse_args()
    
    # Override with test configuration if --test flag is set
    if args.test:
        args.step_meters = 50
        args.num_points = 3
        args.rotation_angles = [-5, 5]
        args.ns_scale_factors = [0.95, 1.05]
        args.ew_scale_factors = [0.95, 1.05]
        args.workers = 3
        args.seed = 42
        args.use_fua = True
        logger.info("Running in TEST mode with: step=50m, points=3, rotations=[-5,5], scales=[0.95,1.05], workers=3, seed=42, use_fua=True")
    
    main(args.api_key, args.city, args.cities, args.step_meters, args.num_points,
         args.rotation_angles, args.ns_scale_factors, args.ew_scale_factors,
         args.workers, args.seed, args.use_fua, args.fua_path, resume=args.resume)
