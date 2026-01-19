"""
author: Federico Bellisardi
execution: runlog -t 96:00 -m 128 -c 8 -j bar python dem_extractor.py --city barcelone
"""
import os
import argparse
import pickle
import csv
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union, transform
from math import cos, sin, pi
import json
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

def get_polygon_coordinates(geom):
    coords = []
    if geom.geom_type == 'Polygon':
        coords.append([(lat, lon) for lon, lat in geom.exterior.coords])
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:
            coords.append([(lat, lon) for lon, lat in polygon.exterior.coords])
    return coords
import matplotlib
matplotlib.use("Agg")
import contextily as ctx
plt.rcParams.update({'font.size': 22})

from shapely.wkt import loads as load_wkt
from data_processing import DEMReader
from utils import logger

import matplotlib.colors as mcolors
import time

VARIANT_COUNT = 0

_G = None
_GRAPHS_DIR = None
_CENTER_Y = None
_CENTER_X = None
_LAND_MASK = None
_DEM_TREE = None
_DEM_ALTS = None
_PRODUCE_MAPS = True
_CMAP = None

def _init_worker(dem_file, graphs_dir, center_y, center_x, produce_maps=True, cmap_N=None):
    global _GRAPHS_DIR, _CENTER_Y, _CENTER_X, _DEM_TREE, _DEM_ALTS, _PRODUCE_MAPS, _CMAP
    _GRAPHS_DIR = graphs_dir
    _CENTER_Y = center_y
    _CENTER_X = center_x
    _PRODUCE_MAPS = bool(produce_maps)
    if cmap_N is None:
        _CMAP = plt.get_cmap('Set1')
    else:
        _CMAP = plt.get_cmap('Set1', int(cmap_N))
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
    try:
        fua_gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        logger.warning(f"Could not read FUA file {gpkg_path}: {e}")
        return None
    try:
        fua_4326 = fua_gdf.to_crs("EPSG:4326")
    except Exception:
        # If CRS is missing, assume it's already 4326
        fua_4326 = fua_gdf
    # Filter FUAs intersecting the bbox polygon
    try:
        candidates = fua_4326[fua_4326.intersects(bbox_poly)]
    except Exception as e:
        logger.warning(f"Failed spatial filter on FUA with bbox for {city_key}: {e}")
        candidates = gpd.GeoDataFrame(columns=fua_4326.columns, geometry="geometry", crs=fua_4326.crs)
    if candidates.empty:
        logger.warning(f"No FUA geometry intersecting bbox for {city_key}; falling back to bbox polygon")
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
        x0, y0 = (xs.max()+xs.min())/2, (ys.max()+ys.min())/2
    else:
        x0, y0 = origin
    xr = xs.copy()
    yr = ys.copy()
    if axis in ('x','both'):
        xr = x0 + scale_factor * (xs - x0)
    if axis in ('y','both'):
        yr = y0 + scale_factor * (ys - y0)
    attr = {n:{'x': x, 'y': y} for n, x, y in zip(nodes, xr, yr)}
    nx.set_node_attributes(G, attr)
    return G

def process_folder(folder, base_path, offsets, rotations, scales, cmap, api_key, workers=None, produce_maps=True, use_fua=False, fua_path=None):
    logger.info(f"[{folder}] Starting processing folder")
    folder_path = os.path.join(base_path, folder)
    csv_file    = os.path.join(folder_path, 'data_useful.csv')
    if not os.path.exists(csv_file):
        logger.info(f"[{folder}] No data_useful.csv found, skipping.")
        return

    graphs_dir = os.path.join(folder_path, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    json_bbox = "/data/workspaces/fbellisardi/metropolis.json"

    dem_dir     = os.path.join(folder_path, 'dem')
    dem_file    = os.path.join(dem_dir, f"{folder}_dem.tif")
    stats_file  = os.path.join(graphs_dir, 'graph_stats.csv')
    expected_pickles = (
        ['graph_original.pkl']
        + [f'graph_translated_{i}.pkl' for i in range(1, len(offsets))]
        + [f'graph_rot_{a}.pkl'        for a in rotations]
        + [f'graph_scale_x_{s}.pkl'    for s in scales]
        + [f'graph_scale_y_{s}.pkl'    for s in scales]
    )
    pickle_files = [os.path.join(graphs_dir, fn) for fn in expected_pickles]


    all_done = (
        is_dem_complete(dem_file)
        and os.path.exists(stats_file)
        and all(os.path.exists(p) for p in pickle_files)
    )
    if all_done:
        logger.info(f"[{folder}] All outputs already exist; skipping.")
        return

    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    logger.info(f"[{folder}] Loaded CSV and created GeoDataFrame with {len(gdf)} records")

    minx, miny, maxx, maxy = gdf.total_bounds
    center_y = (miny + maxy) / 2
    center_x = (minx + maxx) / 2

    min_off_x = min(o[0] for o in offsets)
    min_off_y = min(o[1] for o in offsets)
    max_off_x = max(o[0] for o in offsets)
    max_off_y = max(o[1] for o in offsets)
    ext_bounds = [
        minx + min_off_x, miny + min_off_y,
        maxx + max_off_x, maxy + max_off_y
    ]

    os.makedirs(dem_dir, exist_ok=True)
    if not is_dem_complete(dem_file):
        download_dem(dem_file, ext_bounds, api_key)

    dem_reader = DEMReader(dem_file)
    logger.info(f"[{folder}] DEMReader initialized for {dem_file}")

    city_poly = load_metropolis_bbox(json_bbox, folder)
    logger.info(f"[{folder}] City bbox polygon loaded with bounds {city_poly.bounds}")

    # Optionally replace bbox with GHSL FUA polygon intersecting bbox
    poly_for_graph = city_poly
    if use_fua:
        default_fua = "/home/fbellisardi/code/topolity/vars/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
        fua_gpkg = fua_path or (default_fua if os.path.exists(default_fua) else None)
        if fua_gpkg is None:
            # Try alternative known locations
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
            fua_poly = load_fua_polygon(fua_gpkg, city_poly, folder)
            if fua_poly is not None:
                poly_for_graph = fua_poly
                logger.info(f"[{folder}] Using FUA polygon from {os.path.basename(fua_gpkg)} with bounds {poly_for_graph.bounds}")
            else:
                logger.warning(f"[{folder}] FUA polygon not found; continuing with bbox polygon")
        else:
            logger.warning(f"[{folder}] FUA gpkg not available; continuing with bbox polygon")

    minx, miny, maxx, maxy = poly_for_graph.bounds

    raw_dir = "/data/workspaces/fbellisardi/land"
    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land_global = gpd.read_file(shp_file).to_crs("EPSG:4326")
    land_mask = land_global.geometry.union_all()

    land_shp = os.path.join(folder_path, 'land', f'{folder}_clipped_land.shp')

    bbox_gdf = gpd.GeoDataFrame({'geometry': [poly_for_graph]},crs="EPSG:4326")
    clipped = gpd.overlay(land_global, bbox_gdf, how='intersection')
    os.makedirs(os.path.join(folder_path, 'land'), exist_ok=True)
    clipped.to_file(land_shp)
    logger.info(f"[{folder}] Created {land_shp} by clipping the bbox.")

    land_gdf = gpd.read_file(land_shp).to_crs('EPSG:4326')
    polygon = land_gdf.geometry.union_all()

    graph_original_pkl = os.path.join(graphs_dir, 'graph_original.pkl')
    if os.path.exists(graph_original_pkl):
        with open(graph_original_pkl, 'rb') as f:
            G = pickle.load(f)
        logger.info(f"[{folder}] Loaded existing original graph.")
        if all(data.get('z', 0) == 0 for _, data in G.nodes(data=True)):
           missing = assign_altitudes(G, dem_reader)
           logger.info(f"[{folder}] Re-assigned altitudes to original graph, missing: {missing}")
           with open(graph_original_pkl, 'wb') as f:
               pickle.dump(G, f)
           logger.info(f"[{folder}] Updated original graph pickle with altitudes.")
        else:
            logger.info(f"[{folder}] Original graph already has altitudes assigned, skipping.")
    else:
        tol = 25  # tolerance for OSMnx intersection simplification
        logger.info(f"[{folder}] Building original graph from polygon with tolerance {tol} m...")
        G = ox.graph_from_polygon(
            polygon,
            # custom_filter='["highway"~"motorway|trunk|primary|secondary"]',
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
        logger.info(f"[{folder}] Assigned altitudes to original graph, missing: {missing}")

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes else 0
        density = nx.density(G)
        logger.info(f"[{folder}] Graph stats: nodes={num_nodes}, edges={num_edges}, "
                    f"avg_degree={avg_degree:.2f}, density={density:.2e}")

        logger.info(f"[{folder}] Dijkstra complexity ≈ O(E log V) = O({num_edges} log {num_nodes}).")        

        info_file = os.path.join(graphs_dir, 'original_graph_info.txt')
        with open(info_file, 'w') as fout:
            fout.write(f"num_nodes;{num_nodes}\n")
            fout.write(f"num_edges;{num_edges}\n")
            fout.write(f'tolerance;{tol} m\n')
            fout.write(f"avg_degree;{avg_degree:.2f}\n")
            fout.write(f"density;{density:.2e} [E / [N*(N - 1)]]\n")

            fout.write(f"dijkstra_complexity;O({num_edges} log {num_nodes})\n")
        logger.info(f"[{folder}] Saved graph info to {info_file}")

        with open(graph_original_pkl, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"[{folder}] Built & saved original graph from polygon.")

        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf_edges_4326 = gdf_edges.to_crs(epsg=4326)
        fig, ax = plt.subplots(figsize=(12, 10))
        gdf_edges_4326.plot(ax=ax, linewidth=1, edgecolor='blue')
        ctx.add_basemap(ax,
                        crs='EPSG:4326',
                        source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        minx, miny, maxx, maxy = gdf_edges_4326.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        png_path = os.path.join(graphs_dir, f'graph_map.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[{folder}] Saved static graph map to {png_path}")
    
    stats_records = []

    variants = []
    variants.append({'variant': 'original',
                     'type': 'translate',
                     'offset': (0.0, 0.0),
                     'offset_x': 0.0,
                     'offset_y': 0.0})
    
    for idx, off in enumerate(offsets[1:], start=1):
        name = f'translated_{idx}'
        variants.append({'variant': name,
                         'type': 'translate',
                         'offset': off,
                         'offset_x': off[0],
                         'offset_y': off[1]})
    for angle in rotations:
        name = f'rot_{angle}'
        variants.append({'variant': name,
                         'type': 'rotate',
                         'angle_deg': angle})
    for scale in scales:
        variants.append({'variant': f'scale_x_{scale}',
                         'type': 'scale',
                         'scale_factor': scale,
                         'axis': 'x'})
        variants.append({'variant': f'scale_y_{scale}',
                         'type': 'scale',
                         'scale_factor': scale,
                         'axis': 'y'})

    args_list = []
    for idx, meta in enumerate(variants):
        args_list.append((meta, idx))

    global VARIANT_COUNT
    VARIANT_COUNT = len(args_list)
    total_start = time.time()
    workers = workers or mp.cpu_count()
    logger.info(f"[{folder}] Starting variant processing on {workers} cores")
    global _G, _LAND_MASK
    _G = G
    _LAND_MASK = land_mask
    results = []
    cmap_N = getattr(cmap, 'N', len(variants))
    if workers == 1:
        _init_worker(dem_file, graphs_dir, center_y, center_x, produce_maps, cmap_N)
        global _CMAP
        _CMAP = plt.get_cmap('Set1', cmap_N)
        for a in args_list:
            results.append(process_variant(a))
    else:
        with mp.Pool(processes=workers,
                     initializer=_init_worker,
                     initargs=(dem_file, graphs_dir, center_y, center_x, produce_maps, cmap_N)) as pool:
            results = pool.map(process_variant, args_list)
    total_end = time.time()
    logger.info(f"[{folder}] Completed all variants in {total_end - total_start:.2f} seconds")
    for stats in results:
        if stats is not None:
            stats_records.append(stats)

    fieldnames = [
        'variant','offset_x','offset_y','angle_deg',
        'scale_factor','axis',
        'num_nodes','num_edges','z_mean','z_min','z_max','edge_len_mean','missing_altitude'
    ]
    if stats_records:
        existing = set()
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', newline='') as f:
                    for row in csv.DictReader(f):
                        v = row.get('variant')
                        if v:
                            existing.add(v)
            except Exception as e:
                logger.warning(f"[{folder}] Could not read existing stats for deduplication: {e}")
        to_write = [r for r in stats_records if r.get('variant') not in existing]
        if not to_write:
            logger.info(f"[{folder}] No new stats to append; all variants already present in CSV")
        else:
            write_header = not os.path.exists(stats_file)
            with open(stats_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(to_write)
            logger.info(f"[{folder}] Exported {len(to_write)} new stats rows to {stats_file}")
    try:
        from tools.build_maps_index import process_city as _build_maps_index
        _build_maps_index(base_path, folder, iframes=False)
        logger.info(f"[{folder}] Updated maps index in graphs directory")
    except Exception as e:
        logger.warning(f"[{folder}] Could not update maps index: {e}")

    logger.info(f"[{folder}] Finished processing folder")

def process_variant(args):
    (meta, idx) = args
    variant_start = time.time()
    pid = os.getpid()
    try:
        core = os.sched_getcpu()
    except AttributeError:
        core = 'N/A'
    variant = meta['variant']
    logger.info(f"Processing variant: {variant} ({idx+1}/{VARIANT_COUNT}) in process {pid} on core {core}")
    import pickle, matplotlib.colors as mcolors

    pkl_path = os.path.join(_GRAPHS_DIR, f'graph_{variant}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            G_var = pickle.load(f)
    else:
        if meta['type'] == 'translate':
            G_var = translate_graph(_G.copy(), meta['offset'])
        elif meta['type'] == 'rotate':
            G_var = rotate_graph(_G.copy(), meta['angle_deg'])
        elif meta['type'] == 'scale':
            G_var = scale_graph(
                _G.copy(),
                meta['scale_factor'],
                axis=meta.get('axis','both')
            )
        else:
            G_var = _G.copy()
        if any(not Point(d['x'], d['y']).within(_LAND_MASK) for _, d in G_var.nodes(data=True)):
            logger.warning(f"Variant {variant} has nodes outside the land mask, skipping.")
            return None
        missing = assign_altitudes_from_tree(G_var, _DEM_TREE, _DEM_ALTS)
        for u, v, d in G_var.edges(data=True):
            d.setdefault('length', d.get('length', 1))
        with open(pkl_path, 'wb') as f:
            pickle.dump(G_var, f)
    stats = compute_graph_statistics(G_var)
    stats['variant'] = variant
    stats.update({k: v for k, v in meta.items() if k not in ['variant', 'type', 'offset']})
    stats['missing_altitude'] = sum(1 for _,d in G_var.nodes(data=True) if d.get('z',0)==0)
    # HTML map generation removed for performance - use separate map_generator.py script
    variant_end = time.time()
    logger.info(f"Variant {variant} completed in {variant_end - variant_start:.2f} seconds")
    return stats

def generate_runlog_commands(base_path=None, output_file=None):
    config_path = "/home/fbellisardi/code/topolity/tools/conf/conf_extractor.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if base_path is None:
        base_path = config.get("base_path", "/data/workspaces/fbellisardi/data_processed")
    
    transformations = config.get("transformations", {})
    offsets = transformations.get("offsets", [(0.0, 0.0)])
    rotations = transformations.get("rotations", [0])
    scales = transformations.get("scales", [1.0])
    
    commands = []
    skipped_cities = []
    
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        return
    
    def is_city_completed(city_folder, city_path):
        csv_file = os.path.join(city_path, 'data_useful.csv')
        if not os.path.exists(csv_file):
            return False, "No data_useful.csv found"
        
        graphs_dir = os.path.join(city_path, 'graphs')
        dem_dir = os.path.join(city_path, 'dem')
        dem_file = os.path.join(dem_dir, f"{city_folder}_dem.tif")
        stats_file = os.path.join(graphs_dir, 'graph_stats.csv')
        
        expected_pickles = (
            ['graph_original.pkl']
            + [f'graph_translated_{i}.pkl' for i in range(1, len(offsets))]
            + [f'graph_rot_{a}.pkl'        for a in rotations]
            + [f'graph_scale_x_{s}.pkl'    for s in scales]
            + [f'graph_scale_y_{s}.pkl'    for s in scales]
        )
        pickle_files = [os.path.join(graphs_dir, fn) for fn in expected_pickles]
        
        if not is_dem_complete(dem_file):
            return False, "DEM file missing or incomplete"
        
        if not os.path.exists(stats_file):
            return False, "graph_stats.csv missing"
        
        missing_pickles = [p for p in pickle_files if not os.path.exists(p)]
        if missing_pickles:
            return False, f"Missing {len(missing_pickles)} pickle files"
        
        return True, "All files present"
    
    for city_folder in sorted(os.listdir(base_path)):
        city_path = os.path.join(base_path, city_folder)
        if os.path.isdir(city_path):
            is_completed, reason = is_city_completed(city_folder, city_path)
            
            if is_completed:
                skipped_cities.append(city_folder)
                logger.info(f"Skipping {city_folder}: already completed")
            else:
                csv_file = os.path.join(city_path, 'data_useful.csv')
                if os.path.exists(csv_file):
                    tag = f"dem_{city_folder}"
                    command = f"runlog -t 336:00 -m 450 -c 8 -j {tag} python dem_extractor.py --city {city_folder} --workers 8 --no-maps"
                    commands.append(command)
                    logger.info(f"Generated command for city: {city_folder} (reason: {reason})")
                else:
                    logger.warning(f"Skipping {city_folder}: not a valid city folder (no data_useful.csv)")
    
    if not commands and not skipped_cities:
        logger.warning(f"No valid city folders found in {base_path}")
        return
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated runlog commands for cities that need processing\n")
            f.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Base path: {base_path}\n")
            f.write(f"# Cities to process: {len(commands)}\n")
            f.write(f"# Cities already completed: {len(skipped_cities)}\n\n")
            for cmd in commands:
                f.write(cmd + '\n')
        logger.info(f"Wrote {len(commands)} commands to {output_file}")
    else:
        print("\n# Generated runlog commands for cities that need processing:")
        print("# Usage: Copy and paste these commands or save to a script file")
        print("# Each command will process one city with 8 workers and no map generation")
        print(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Base path: {base_path}")
        print()
        for cmd in commands:
            print(cmd)
        print()
        print(f"# Cities to process: {len(commands)}")
        print(f"# Cities already completed: {len(skipped_cities)}")
        if skipped_cities:
            print(f"# Skipped cities: {', '.join(skipped_cities)}")
        print()
        
        if not commands:
            print("# All cities are already fully processed!")
            print("# No commands generated.")

def main(api_key=None, example_city=None, workers=None, produce_maps=False, use_fua=False, fua_path=None):
    config_path = "/home/fbellisardi/code/topolity/tools/conf/conf_extractor.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = api_key or config.get("api_key")
    example_city = example_city or config.get("city")
    base_path = config.get("base_path", "/data/workspaces/fbellisardi/data_processed")
    
    transformations = config.get("transformations", {})
    offsets = transformations.get("offsets", [(0.0, 0.0)])
    rotations = transformations.get("rotations", [0])
    scales = transformations.get("scales", [1.0])

    cnumber = len(offsets) + len(rotations) + len(scales)
    cmap = plt.get_cmap('Set1', cnumber)

    if example_city:
        process_folder(example_city, base_path, offsets, rotations, scales, cmap, api_key, workers, produce_maps, use_fua, fua_path)
    else:
        for city_folder in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, city_folder)):
                process_folder(city_folder, base_path, offsets, rotations, scales, cmap, api_key, workers, produce_maps, use_fua, fua_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        help='OpenTopography API key (overrides config)')
    parser.add_argument('--city', type=str,
                        help='Example city folder to process (overrides config)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (use 1 to disable multiprocessing)')
    parser.add_argument('--no-maps', action='store_true',
                        help='Disable HTML map generation (now disabled by default for performance - use map_generator.py separately)')
    parser.add_argument('-gc','--generate-commands', action='store_true',
                        help='Generate runlog commands for all cities and exit')
    parser.add_argument('--commands-output', type=str,
                        help='File to write generated commands to (if not specified, prints to stdout)')
    parser.add_argument('--use-fua', action='store_true',
                        help='Restrict processing polygon to GHSL FUA intersecting the city bbox')
    parser.add_argument('--fua-path', type=str,
                        help='Path to GHSL FUA .gpkg (optional; defaults to known locations)')
    args = parser.parse_args()
    
    if args.generate_commands:
        generate_runlog_commands(output_file=args.commands_output)
    else:
        main(args.api_key, args.city, args.workers, produce_maps=not args.no_maps, use_fua=args.use_fua, fua_path=args.fua_path)
