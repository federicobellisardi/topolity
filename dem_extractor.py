"""
author: Federico Bellisardi
Improved version: DEM download via bbox, graph generation, translations, statistics export, Folium visualization.
"""

import os
import argparse
import pickle
import csv
import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Point
from shapely.ops import unary_union

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from shapely.wkt import loads as load_wkt
from data_processing import DEMReader
from utils import logger

import folium
from folium import FeatureGroup
import matplotlib.colors as mcolors


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


def assign_altitudes(G, dem_reader):
    """
    Efficiently assign altitude to each node using KDTree nearest-neighbor lookup.
    """
    dem_gdf = dem_reader.get_pixel_centroids()
    # build KDTree on DEM pixel centroids
    import numpy as np
    from scipy.spatial import cKDTree
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
    """Apply a translation to graph node coordinates."""
    for _, data in G.nodes(data=True):
        data['x'] += offset[0]
        data['y'] += offset[1]
    return G


def process_folder(folder, base_path, offsets, cmap, api_key):
    """
    Process a single city folder: skip if everything is already done,
    otherwise only run the steps whose outputs are missing.
    """
    folder_path = os.path.join(base_path, folder)
    csv_file    = os.path.join(folder_path, 'data_useful.csv')
    if not os.path.exists(csv_file):
        logger.info(f"[{folder}] No data_useful.csv found, skipping.")
        return

    graphs_dir = os.path.join(folder_path, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    # define all the “final” outputs we expect:
    dem_dir     = os.path.join(folder_path, 'dem')
    dem_file    = os.path.join(dem_dir, f"{folder}_dem.tif")
    stats_file  = os.path.join(graphs_dir, 'graph_stats.csv')
    map_file    = os.path.join(graphs_dir, 'graph_translations_map.html')
    pickle_files = [
        os.path.join(graphs_dir, 'graph_original.pkl')
    ] + [
        os.path.join(graphs_dir, f'graph_translated_{i}.pkl')
        for i in range(1, len(offsets))
    ]

    all_done = (
        is_dem_complete(dem_file)
        and os.path.exists(stats_file)
        and os.path.exists(map_file)
        and all(os.path.exists(p) for p in pickle_files)
    )
    if all_done:
        logger.info(f"[{folder}] All outputs already exist; skipping.")
        return

    df = pd.read_csv(csv_file, sep=';').dropna(subset=['geometry'])
    df['geometry'] = df['geometry'].apply(load_wkt)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

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

    raw_dir = "/data/workspaces/fbellisardi/land"
    shp_file = os.path.join(raw_dir, "ne_10m_land.shp")
    land_global = gpd.read_file(shp_file).to_crs("EPSG:4326")
    land_mask = land_global.geometry.union_all()

    land_shp = os.path.join(folder_path, 'land', f'{folder}_clipped_land.shp')
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
        G = ox.graph_from_polygon(
            polygon,
            network_type='drive',
            retain_all=True,
            simplify=True
        )
        G = ox.project_graph(G, to_crs='EPSG:4326')
        if not nx.is_strongly_connected(G):
            cc = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(cc).copy()

        missing = assign_altitudes(G, dem_reader)
        logger.info(f"[{folder}] Assigned altitudes to original graph, missing: {missing}")

        with open(graph_original_pkl, 'wb') as f:
            pickle.dump(G, f)
        logger.info(f"[{folder}] Built & saved original graph from polygon.")

    stats_records = []
    folium_map = folium.Map(location=[center_y, center_x], zoom_start=13)

    for idx, offset in enumerate(offsets):
        variant = 'original' if idx == 0 else f'translated_{idx}'
        pkl_path = os.path.join(graphs_dir,
                                'graph_original.pkl' if idx == 0 else f'graph_translated_{idx}.pkl')
        stats_needed = True

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                G_variant = pickle.load(f)
            logger.info(f"[{folder}][{variant}] Loaded existing graph pickle.")
            if os.path.exists(stats_file):
                with open(stats_file) as sf:
                    reader = csv.DictReader(sf)
                    stats_needed = variant not in [r['variant'] for r in reader]
        else:
            G_variant = translate_graph(G.copy(), offset)
            drifts_to_sea = False
            for _, data in G_variant.nodes(data=True):
                pt = Point(data['x'], data['y'])
                if not pt.within(land_mask):
                    drifts_to_sea = True
                    break
            if drifts_to_sea:
                logger.warning(f"[{folder}][{variant}] drifts into sea → skipping this variant.")
                continue

            missing = assign_altitudes(G_variant, dem_reader)
            for u, v, data in G_variant.edges(data=True):
                data.setdefault('length', data.get('length', 1))
            with open(pkl_path, 'wb') as f:
                pickle.dump(G_variant, f)
            logger.info(f"[{folder}][{variant}] Created and saved graph pickle.")
            stats_needed = True



        if stats_needed:
            stats = compute_graph_statistics(G_variant)
            stats.update({
                'variant': variant,
                'offset_x': offset[0],
                'offset_y': offset[1],
                'missing_altitude': sum(1 for _,d in G_variant.nodes(data=True) if d.get('z',0)==0)
            })
            stats_records.append(stats)

            color = mcolors.to_hex(cmap(idx))
            layer = FeatureGroup(name=variant)
            for u, v in G_variant.edges():
                y1, x1 = G_variant.nodes[u]['y'], G_variant.nodes[u]['x']
                y2, x2 = G_variant.nodes[v]['y'], G_variant.nodes[v]['x']
                folium.PolyLine([(y1, x1), (y2, x2)], color=color, weight=2).add_to(layer)
            layer.add_to(folium_map)
            logger.info(f"[{folder}][{variant}] Stats computed & map layer added.")

    if stats_records:
        write_header = not os.path.exists(stats_file)
        with open(stats_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_records[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(stats_records)
        logger.info(f"[{folder}] Exported {len(stats_records)} new stat records to {stats_file}")

    if not os.path.exists(map_file):
        folium.LayerControl().add_to(folium_map)
        folium_map.save(map_file)
        logger.info(f"[{folder}] Folium map saved to {map_file}")

def main(api_key, example_city=None):
    base_path = "/data/workspaces/fbellisardi/data_processed"
    offsets = [(0, 0), (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0.05, 0.05), (-0.05, -0.05)]
    cmap = plt.get_cmap('Set1', len(offsets))

    if example_city:
        process_folder(example_city, base_path, offsets, cmap, api_key)
    else:
        for city_folder in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, city_folder)):
                process_folder(city_folder, base_path, offsets, cmap, api_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str,
                        default='da4217e2ba10a0edd8e0269f12c4717c',
                        help='OpenTopography API key')
    parser.add_argument('--city', type=str,
                        help='Example city folder to process, e.g. "barcelone"')
    args = parser.parse_args()
    main(args.api_key, args.city)
