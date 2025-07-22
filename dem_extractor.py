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
import matplotlib
matplotlib.use("Agg")
import contextily as ctx
plt.rcParams.update({'font.size': 22})

from shapely.wkt import loads as load_wkt
from data_processing import DEMReader
from utils import logger

import folium
from folium import FeatureGroup
import matplotlib.colors as mcolors
import time

# global total variants count for logging
VARIANT_COUNT = 0

def is_dem_complete(path):
    """Check if the DEM file exists and is larger than 1MB."""
    return os.path.exists(path) and os.path.getsize(path) > 1e6


def download_dem(dem_path, bounds, api_key):
    """
    Download the DEM using the bounding box with 50% padding.
    bounds: [minx, miny, maxx, maxy] in EPSG:4326
    """
    if is_dem_complete(dem_path):
        logger.info(f"DEM already exists: {dem_path}")
        return
    # Compute 50% padding
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

def assign_altitudes(G, dem_reader):
    """
    Efficiently assign altitude to each node using KDTree nearest-neighbor lookup.
    """
    dem_gdf = dem_reader.get_pixel_centroids()
    # build KDTree on DEM pixel centroids

    dem_coords = np.vstack((dem_gdf.geometry.y.values, dem_gdf.geometry.x.values)).T
    dem_alts = dem_gdf['alt'].values
    tree = cKDTree(dem_coords)

    # get node coordinates
    nodes = list(G.nodes(data=True))
    node_coords = np.array([[data['y'], data['x']] for _, data in nodes])
    # query nearest DEM altitude
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


def compute_graph_statistics(G):
    """Compute basic statistics for the enriched graph."""
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

def process_folder(folder, base_path, offsets, rotations, scales, cmap, api_key):
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
    logger.info(f"[{folder}] City polygon loaded with bounds {city_poly.bounds}")

    folium_map = folium.Map(location=[center_y, center_x], zoom_start=13)

    minx, miny, maxx, maxy = city_poly.bounds

    folium.Polygon(
        locations=[(lon, lat) for lat, lon in city_poly.exterior.coords],
        color='blue', fill=True, fill_opacity=0.1
    ).add_to(folium_map)

    city_map_file = os.path.join(graphs_dir, f'{folder}_city_bbox.html')
    folium_map.save(city_map_file)
    logger.info(f"[{folder}] Saved city bbox map to {city_map_file}")

    raw_dir = "/data/workspaces/fbellisardi/land"
    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land_global = gpd.read_file(shp_file).to_crs("EPSG:4326")
    land_mask = land_global.geometry.union_all()

    land_shp = os.path.join(folder_path, 'land', f'{folder}_clipped_land.shp')

    bbox_gdf = gpd.GeoDataFrame({'geometry': [city_poly]},crs="EPSG:4326")
    clipped = gpd.overlay(land_global, bbox_gdf, how='intersection')
    os.makedirs(os.path.join(folder_path, 'land'), exist_ok=True)
    clipped.to_file(land_shp)
    logger.info(f"[{folder}] Created {land_shp} by clipping the bbox.")

    land_gdf = gpd.read_file(land_shp).to_crs('EPSG:4326')
    polygon = land_gdf.geometry.union_all()

    folium_map2 = folium.Map(location=[center_y, center_x], zoom_start=13)

    folium.Polygon(
        locations=[(lon, lat) for lat, lon in polygon.exterior.coords],
        color='blue', fill=True, fill_opacity=0.1
    ).add_to(folium_map2)

    city_map_file = os.path.join(graphs_dir, f'{folder}_city_bbox_landed.html')
    folium_map2.save(city_map_file)
    logger.info(f"[{folder}] Saved city bbox map to {city_map_file}")

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
    folium_map = folium.Map(location=[center_y, center_x], zoom_start=13)

    variants = []
    for idx, off in enumerate(offsets):
        name = 'original' if idx == 0 else f'translated_{idx}'
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
        args_list.append(
            (G, dem_file, graphs_dir, center_y, center_x,
             land_mask, meta, idx, cmap)
        )
    # set total number of variants for logging
    global VARIANT_COUNT
    VARIANT_COUNT = len(args_list)
    total_start = time.time()
    logger.info(f"[{folder}] Starting variant processing on {mp.cpu_count()} cores")
    with mp.Pool(processes=mp.cpu_count()) as pool:
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
        write_header = not os.path.exists(stats_file)
        with open(stats_file, 'a', newline='') as f:
            # writer = csv.DictWriter(f, fieldnames=stats_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(stats_records)
        logger.info(f"[{folder}] Exported stats to {stats_file}")
    logger.info(f"[{folder}] Finished processing folder")

def process_variant(args):
    (G, dem_file, graphs_dir, center_y, center_x,
     land_mask, meta, idx, cmap) = args
    from data_processing import DEMReader
    variant_start = time.time()
    pid = os.getpid()
    try:
        core = os.sched_getcpu()
    except AttributeError:
        core = 'N/A'
    dem_reader = DEMReader(dem_file)
    variant = meta['variant']
    logger.info(f"Processing variant: {variant} ({idx+1}/{VARIANT_COUNT}) in process {pid} on core {core}")
    import pickle, folium, matplotlib.colors as mcolors

    pkl_path = os.path.join(graphs_dir, f'graph_{variant}.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            G_var = pickle.load(f)
    else:
        if meta['type'] == 'translate':
            G_var = translate_graph(G.copy(), meta['offset'])
        elif meta['type'] == 'rotate':
            G_var = rotate_graph(G.copy(), meta['angle_deg'])
        elif meta['type'] == 'scale':
            G_var = scale_graph(
                G.copy(),
                meta['scale_factor'],
                axis=meta.get('axis','both')
            )
        else:
            G_var = G.copy()
        if any(not Point(d['x'], d['y']).within(land_mask) for _, d in G_var.nodes(data=True)):
            logger.warning(f"Variant {variant} has nodes outside the land mask, skipping.")
            return None
        missing = assign_altitudes(G_var, dem_reader)
        for u, v, d in G_var.edges(data=True):
            d.setdefault('length', d.get('length', 1))
        with open(pkl_path, 'wb') as f:
            pickle.dump(G_var, f)
    stats = compute_graph_statistics(G_var)
    stats['variant'] = variant
    stats.update({k: v for k, v in meta.items() if k not in ['variant', 'type', 'offset']})
    stats['missing_altitude'] = sum(1 for _,d in G_var.nodes(data=True) if d.get('z',0)==0)
    variant_map = folium.Map(location=[center_y, center_x], zoom_start=13)
    color = mcolors.to_hex(cmap(idx))
    for u,v in G_var.edges():
        y1,x1 = G_var.nodes[u]['y'], G_var.nodes[u]['x']
        y2,x2 = G_var.nodes[v]['y'], G_var.nodes[v]['x']
        folium.PolyLine([(y1,x1),(y2,x2)], color=color, weight=2).add_to(variant_map)
    variant_map_file = os.path.join(graphs_dir, f'graph_{variant}_map.html')
    variant_map.save(variant_map_file)
    variant_end = time.time()
    logger.info(f"Variant {variant} completed in {variant_end - variant_start:.2f} seconds")
    return stats

def main(api_key=None, example_city=None):
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
        process_folder(example_city, base_path, offsets, rotations, scales, cmap, api_key)
    else:
        for city_folder in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, city_folder)):
                process_folder(city_folder, base_path, offsets, rotations, scales, cmap, api_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        help='OpenTopography API key (overrides config)')
    parser.add_argument('--city', type=str,
                        help='Example city folder to process (overrides config)')
    args = parser.parse_args()
    main(args.api_key, args.city)
