#!/usr/bin/env python3
"""
author: Federico Bellisardi
execution: python wheight.py -c tools/conf/conf_wheight.json

"""

import os
import sys
import json
import argparse
import pickle
import logging
import psutil

import requests
import pandas as pd
import numpy as np
from numpy.linalg import norm

import networkx as nx
import networkit as nk
from networkit import nxadapter, distance

import rasterio
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import box, LineString, Point, Polygon

import folium
import matplotlib.pyplot as plt
import branca.colormap as bcm
import matplotlib.pyplot as plt

import multiprocessing
import time


p = psutil.Process(os.getpid())
p.cpu_percent(None)

# ──────────────────────────────────────────────────────────────────────────────
# Configure logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DEM utilities
# ──────────────────────────────────────────────────────────────────────────────
class DEMReader:
    def __init__(self, dem_file):
        self.dem_file = dem_file
        self.src = None

    def ensure_dem(self, api_key, bbox):
        """Download DEM once; skip if >1MB already exists."""
        if os.path.exists(self.dem_file) and os.path.getsize(self.dem_file) > 1e6:
            logger.info(f"DEM already present ({self.dem_file}), skipping download.")
            return
        logger.info(f"Downloading DEM → {self.dem_file}")
        south = bbox["min_lat"] - (bbox["max_lat"] - bbox["min_lat"])
        north = bbox["max_lat"] + (bbox["max_lat"] - bbox["min_lat"])
        west  = bbox["min_lon"] - (bbox["max_lon"] - bbox["min_lon"])
        east  = bbox["max_lon"] + (bbox["max_lon"] - bbox["min_lon"])
        url = (
            "https://portal.opentopography.org/API/globaldem?"
            f"demtype=SRTMGL3&south={south}&north={north}"
            f"&west={west}&east={east}"
            f"&outputFormat=GTiff&API_Key={api_key}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        with open(self.dem_file, 'wb') as f:
            f.write(resp.content)
        size_mb = os.path.getsize(self.dem_file) / 1e6
        logger.info(f"Saved DEM ({size_mb:.1f} MB)")

    def open(self):
        """Open the GeoTIFF for sampling."""
        logger.info(f"Opening DEM file {self.dem_file}")
        self.src = rasterio.open(self.dem_file)

    def sample(self, coords):
        """
        coords: list of (lon, lat)
        returns: list of elevation floats
        """
        return [val[0] for val in self.src.sample(coords)]

    def close(self):
        if self.src:
            logger.info("Closing DEM dataset")
            self.src.close()


# ──────────────────────────────────────────────────────────────────────────────
# Work evaluator with NetworKit Dijkstra
# ──────────────────────────────────────────────────────────────────────────────
def _work_for_origin(args):
    origin_cell, dest_list, cell_map, node2id, nkG, arc_work, idx, total_origins, start_time = args
    pid = os.getpid()
    proc = multiprocessing.current_process().name
    total = 0.0

    src_node = cell_map.get(origin_cell)
    if src_node is None:
        return 0.0
    sid = node2id[src_node]

    runner = distance.Dijkstra(nkG, sid, True)
    runner.run()

    for dest_cell, flow in dest_list:
        dest_node = cell_map.get(dest_cell)
        if dest_node is None:
            continue
        tid = node2id[dest_node]
        path = runner.getPath(tid)
        if not path:
            continue
        for a_id, b_id in zip(path[:-1], path[1:]):
            total += flow * arc_work.get((a_id, b_id), 0.0)

    if idx % 100 == 0 or idx == total_origins-1:
        elapsed = time.time() - start_time
        eta = (elapsed/(idx+1)) * (total_origins - (idx+1))
        logger.info(f"[{proc} PID {pid}] Progress: {idx+1}/{total_origins} origins | "
                    f"elapsed: {elapsed:.1f}s | ETA: {eta/60:.1f} min")
    return total

class WorkEvaluatorNK:
    def __init__(self, G, dem_reader, ds=10.0, m=1.0, g=1.0):
        """
        G          : NetworkX graph with node['x'], node['y']
        dem_reader : DEMReader (must have .open() called)
        ds         : sampling interval along edges
        m, g       : mass & gravity
        """
        self.G   = G
        self.dem = dem_reader
        self.ds  = ds
        self.m   = m
        self.g   = g

        logger.info("Precomputing edge-work for all edges")
        self.arc_work = {
            (u, v): self._compute_edge_work(u, v)
            for u, v, _ in G.edges(data=True)
        }

        logger.info("Converting NetworkX graph to NetworKit graph")

        self.node2id = {n:i for i,n in enumerate(G.nodes())}
        self.id2node = {i:n for n,i in self.node2id.items()}

        # Precompute uphill work on each original edge
        logger.info("Precomputing edge-work for all edges")
        arc_work_orig = {}
        for u, v, data in G.edges(data=True):
            arc_work_orig[(u, v)] = self._compute_edge_work(u, v)

        # Build a NetworKit graph with contiguous integer IDs
        n = len(self.node2id)
        nkG = nk.Graph(n, weighted=True, directed=G.is_directed())
        for (u, v), w in arc_work_orig.items():
            u_id, v_id = self.node2id[u], self.node2id[v]
            length = G[u][v].get('length', 1.0)
            nkG.addEdge(u_id, v_id, length)

        self.nkG = nkG
        # Build an internal arc_work keyed by (int,int)
        self.arc_work = {
            (self.node2id[u], self.node2id[v]): w
            for (u, v), w in arc_work_orig.items()
        }

        # cache for Dijkstra runners
        self._dijkstra_cache = {}
        logger.info("NetworKit graph built with %d nodes, %d edges",
                    nkG.numberOfNodes(), nkG.numberOfEdges())

    def _compute_edge_work(self, u, v):
        """Sample along the edge and sum positive elevation gains."""
        data = self.G.get_edge_data(u, v)
        geom = None
        if isinstance(data, list):
            for e in data:
                if 'geometry' in e:
                    geom = e['geometry']
                    break
        else:
            geom = data.get('geometry')

        if geom is None:
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            geom = LineString([(x1, y1), (x2, y2)])

        length = geom.length
        n_pts  = max(int(length / self.ds) + 1, 2)
        dists  = np.linspace(0, length, n_pts)
        pts    = [geom.interpolate(d) for d in dists]
        coords = [(pt.x, pt.y) for pt in pts]

        elevs = self.dem.sample(coords)
        w = 0.0
        for h1, h2 in zip(elevs, elevs[1:]):
            if h2 > h1:
                w += self.m * self.g * (h2 - h1)
        return w

    def compute_total_work(self, cell_map, od_df):
        logger.info("Computing total work for %d OD flows", len(od_df))

        od_by_origin = {}
        for origin, dest, flow in od_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, []).append((dest, flow))

        start_time = time.time()
        total_origins = len(od_by_origin)
        args_list = [
            (origin_cell, dest_list, cell_map, self.node2id, self.nkG, self.arc_work, idx, total_origins, start_time)
            for idx, (origin_cell, dest_list) in enumerate(od_by_origin.items())
        ]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(_work_for_origin, args_list)
        total = sum(results)

        logger.info("Total uphill work = %.2f", total)
        return total

# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────


def log_resources(stage: str):
    mem_gb = p.memory_info().rss / (1024**3)
    cpu_pct = p.cpu_percent(interval=None)
    logger.info(f"[{stage}] Memory usage: {mem_gb:.2f} GB | CPU%: {cpu_pct:.1f}")

def load_graphs(graphs_dir):
    logger.info(f"Loading graphs from {graphs_dir}")

    from collections import defaultdict
    stats = (
        pd.read_csv(os.path.join(graphs_dir, 'graph_stats.csv'))
          .rename(columns=lambda c: c.strip())
    )
    graphs = {}

    for _, row in stats.iterrows():
        var = row['variant']
        fn  = 'graph_original.pkl' if var == 'original' else f'graph_{var}.pkl'
        path = os.path.join(graphs_dir, fn)
        with open(path, 'rb') as f:
            G = pickle.load(f)

        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            # logger.info(f"[{var}] collapsing MultiGraph → simple {type(G).__name__}")
            H = nx.DiGraph() if G.is_directed() else nx.Graph()
            for n, data in G.nodes(data=True):
                H.add_node(n, **data)
            grouped = defaultdict(list)
            for u, v, key, data in G.edges(keys=True, data=True):
                grouped[(u, v)].append(data.copy())

            for (u, v), data_list in grouped.items():
                lengths = [d.get('length', np.inf) for d in data_list]
                length = min(lengths)
                H.add_edge(u, v, length=length, all_data=data_list)

            G = H

        total_edges = G.number_of_edges()
        with_len = sum(1 for _, _, d in G.edges(data=True) if 'length' in d)

        if G.is_directed():
            if nx.is_weakly_connected(G):
                # logger.info(f"[{var}] graph is weakly connected")
                pass
            else:
                n_comp = nx.number_weakly_connected_components(G)
                logger.warning(f"[{var}] NOT weakly connected: {n_comp} components")
        else:
            if nx.is_connected(G):
                # logger.info(f"[{var}] graph is connected")
                pass
            else:
                n_comp = nx.number_connected_components(G)
                logger.warning(f"[{var}] NOT connected: {n_comp} components")

        graphs[var] = G
    logger.info(f"Loaded {len(graphs)} graph variants")
    return graphs, stats

def load_cells(path, cells_crs, bbox):
    logger.info("Loading & filtering cells")
    df = pd.read_csv(path).rename(columns={'cell_id':'cell'})
    transf = Transformer.from_crs(cells_crs, 'EPSG:4326', always_xy=True)
    df[['lon_min','lat_min']] = df.apply(
        lambda r: transf.transform(r.x_min, r.y_min),
        axis=1, result_type='expand')
    df[['lon_max','lat_max']] = df.apply(
        lambda r: transf.transform(r.x_max, r.y_max),
        axis=1, result_type='expand')
    poly = Polygon([(lon, lat) for lat, lon in bbox])
    df['keep'] = df.apply(
        lambda r: box(r.lon_min, r.lat_min, r.lon_max, r.lat_max).intersects(poly),
        axis=1)
    df = df[df.keep].copy()
    logger.info(f"  Retained {len(df)} cells in bbox")
    df['cent_lon'] = 0.5*(df.lon_min + df.lon_max)
    df['cent_lat'] = 0.5*(df.lat_min + df.lat_max)
    return df[['cell','cent_lon','cent_lat','lon_min','lat_min','lon_max','lat_max']]

def load_od(path):
    logger.info(f"Loading OD flows from {path}")
    df = pd.read_csv(path).rename(columns={
        'cell_origin':'origin','cell_destination':'dest','count':'flow'
    })
    df.flow = pd.to_numeric(df.flow, errors='coerce').fillna(0)
    df = df[df.flow>0].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} positive flows")
    return df

def map_cells_to_nodes(G, cell_df):
    xs = np.array([data.get('x', data.get('lon')) for _, data in G.nodes(data=True)])
    ys = np.array([data.get('y', data.get('lat')) for _, data in G.nodes(data=True)])
    nodes = np.array([n for n, _ in G.nodes(data=True)])

    mapping = {}
    total = 0
    for _, row in cell_df.iterrows():
        mask = (
            (xs >= row.lon_min) & (xs <= row.lon_max) &
            (ys >= row.lat_min) & (ys <= row.lat_max)
        )
        lst = nodes[mask].tolist()
        if not lst:
            mapping[row.cell] = None
            continue

        cent = np.array([row.cent_lon, row.cent_lat])
        coords = np.array([
            (
                G.nodes[n].get('x', G.nodes[n].get('lon')),
                G.nodes[n].get('y', G.nodes[n].get('lat'))
            )
            for n in lst
        ])
        idx_min = np.argmin(norm(coords - cent, axis=1))
        mapping[row.cell] = lst[idx_min]
        total += 1

    logger.info(f"Mapped {len(mapping)} cells → {total} central nodes")
    return mapping

def plot_work_by_variant(results_df, out_png):
    """
    Explanatory bar plot showing W_WD and W_HOL for each variant side by side.
    """
    variants = results_df['variant']
    wd = results_df['work_wd']
    hol = results_df['work_hol']

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, wd, width, label='Working Day W')
    ax.bar(x + width/2, hol, width, label='Holiday W')

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.set_ylabel('Total uphill work (m·units)')
    ax.set_title('Total uphill work by variant')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved work-by-variant plot → {out_png}")

def plot_variant_differences(results_df, out_png):
    """
    Plot percent difference in W_WD and W_HOL relative to the original network.
    """
    orig = results_df[results_df.variant == 'original'].iloc[0]
    pert = results_df[results_df.variant != 'original'].copy()
    pert['pct_diff_wd']  = (pert.work_wd  - orig.work_wd)  / orig.work_wd  * 100
    pert['pct_diff_hol'] = (pert.work_hol - orig.work_hol) / orig.work_hol * 100

    variants = pert['variant']
    wd_diff  = pert['pct_diff_wd']
    hol_diff = pert['pct_diff_hol']

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, wd_diff, width, label='% Δ W_WD')
    ax.bar(x + width/2, hol_diff, width, label='% Δ W_HOL')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.set_ylabel('Percent difference versus original (%)')
    ax.set_title('Percent change in uphill work by variant')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved percent-difference plot → {out_png}")

def plot_arc_work_distribution(arc_work_dict, out_png):
    """
    Plot histogram of arc_work values.
    """
    vals = list(arc_work_dict.values())
    plt.figure(figsize=(10,5))
    plt.hist(vals, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Uphill work per edge (J)')
    plt.ylabel('Count')
    plt.title('Distribution of uphill work on edges')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved arc-work distribution plot → {out_png}")

def plot_dem_elevation_distribution(dem_reader, out_png):
    """
    Plot histogram of DEM elevation values (masked nodata).
    """
    with rasterio.open(dem_reader.dem_file) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
    plt.figure(figsize=(10,5))
    plt.hist(arr.flatten(), bins=100, color='lightgreen', edgecolor='black')
    plt.xlabel('Elevation (m)')
    plt.ylabel('Count')
    plt.title('Distribution of DEM elevation values')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved DEM elevation distribution plot → {out_png}")


def plot_arc_work_boxplot(arc_work_dict, out_png):
    """
    Boxplot of uphill work per edge.
    """
    vals = list(arc_work_dict.values())
    plt.figure(figsize=(6,8))
    plt.boxplot(vals, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red'))
    plt.ylabel('Uphill work per edge (J)')
    plt.title('Boxplot of uphill work on edges')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved arc-work boxplot → {out_png}")


def plot_cell_node_mapping(cell_df, cell_map, G, out_png):
    """
    Scatter plot of cell centroids and their mapped node.
    """
    centroids = cell_df[['cent_lon','cent_lat']].values
    node_coords = []
    for cell, node in cell_map.items():
        if node is not None:
            data = G.nodes[node]
            lon = data.get('x', data.get('lon'))
            lat = data.get('y', data.get('lat'))
            node_coords.append((lon, lat))
    node_coords = np.array(node_coords)
    plt.figure(figsize=(8,8))
    plt.scatter(centroids[:,0], centroids[:,1], s=10, c='red', label='Cell centroids')
    plt.scatter(node_coords[:,0], node_coords[:,1], s=5, c='blue', label='Mapped nodes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Cell centroids vs mapped node')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    logger.info(f"Saved cell-node mapping plot → {out_png}")


def check_available_variant_types(results_df):
    """
    Check which types of variants are available in the results.
    Returns a dictionary with boolean flags for each type.
    """
    variants = results_df['variant'].tolist()
    
    has_translations = any('translat' in v.lower() for v in variants)
    has_rotations = any('rotat' in v.lower() for v in variants)
    has_scaling = any('scal' in v.lower() for v in variants)
    has_original = 'original' in variants
    
    logger.info(f"Available variant types: translations={has_translations}, "
                f"rotations={has_rotations}, scaling={has_scaling}, original={has_original}")
    
    return {
        'translations': has_translations,
        'rotations': has_rotations, 
        'scaling': has_scaling,
        'original': has_original
    }


def plot_work_parabolic_translations(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    translation_variants = results_df[results_df.variant.str.contains('translat', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for translation plot")
        return
    
    if len(translation_variants) == 0:
        logger.warning("No translation variants found")
        return
    
    # Merge with stats to get transformation parameters
    merged = translation_variants.merge(stats_df, on='variant', how='left')
    
    translated_variants = []
    for _, row in merged.iterrows():
        offset_x = row['offset_x']
        offset_y = row['offset_y']
        sort_key = offset_x + offset_y
        translated_variants.append((sort_key, offset_x, offset_y, row['work_wd'], row['work_hol'], row['variant']))
    
    translated_variants.sort(key=lambda x: x[0])
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    
    all_sort_keys = [sort_key for sort_key, _, _, _, _, _ in translated_variants]
    all_wd = [wd for _, _, _, wd, _, _ in translated_variants]
    all_hol = [hol for _, _, _, _, hol, _ in translated_variants]
    all_offset_x = [offset_x for _, offset_x, _, _, _, _ in translated_variants]
    all_offset_y = [offset_y for _, _, offset_y, _, _, _ in translated_variants]
    all_names = [name for _, _, _, _, _, name in translated_variants]
    
    n_total = len(translated_variants) + 1
    center_idx = len(translated_variants) // 2
    
    # Get original offsets from stats
    orig_stats = stats_df[stats_df.variant == 'original'].iloc[0]
    orig_offset_x = orig_stats['offset_x']
    orig_offset_y = orig_stats['offset_y']
    
    all_sort_keys.insert(center_idx, orig_offset_x + orig_offset_y)
    all_wd.insert(center_idx, orig_wd)
    all_hol.insert(center_idx, orig_hol)
    all_offset_x.insert(center_idx, orig_offset_x)
    all_offset_y.insert(center_idx, orig_offset_y)
    all_names.insert(center_idx, 'Original')
    
    x_positions = list(range(len(all_sort_keys)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append(f'Original\n({orig_offset_x:.2f},{orig_offset_y:.2f})')
        else:
            offset_x = all_offset_x[i]
            offset_y = all_offset_y[i]
            x_labels.append(f'Δx:{offset_x:.2f}\nΔy:{offset_y:.2f}')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(x_positions, wd_values, 'o-', linewidth=2.5, markersize=8, 
            label='Working Day', color='blue', alpha=0.8)
    ax.plot(x_positions, hol_values, 's-', linewidth=2.5, markersize=8, 
            label='Holiday', color='red', alpha=0.8)
    
    orig_idx = center_idx
    ax.plot(x_positions[orig_idx], wd_values[orig_idx], 'o', markersize=14, color='blue', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='blue')
    ax.plot(x_positions[orig_idx], hol_values[orig_idx], 's', markersize=14, color='red', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='red')
    
    ax.axvline(x=x_positions[orig_idx], color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(x_positions[orig_idx], max_y), 
               xytext=(x_positions[orig_idx], max_y + y_range * 0.08),
               ha='center', fontsize=12, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Translation Variants (offset_x, offset_y)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=12, fontweight='bold')
    ax.set_title('Gravitational Work vs Network Translation\n(Original Network at Center)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved translation work plot → {out_png}")


def plot_work_parabolic_rotations(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    rotation_variants = results_df[results_df.variant.str.contains('rotat', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for rotation plot")
        return
    
    if len(rotation_variants) == 0:
        logger.warning("No rotation variants found")
        return
    
    # Merge with stats to get transformation parameters
    merged = rotation_variants.merge(stats_df, on='variant', how='left')
    
    rotated_variants = []
    for _, row in merged.iterrows():
        angle = row['angle_deg']
        if pd.isna(angle):
            continue
        rotated_variants.append((angle, row['work_wd'], row['work_hol'], row['variant']))
    
    rotated_variants.sort(key=lambda x: x[0])
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    
    all_angles = [angle for angle, _, _, _ in rotated_variants]
    all_wd = [wd for _, wd, _, _ in rotated_variants]
    all_hol = [hol for _, _, hol, _ in rotated_variants]
    all_names = [name for _, _, _, name in rotated_variants]
    
    n_total = len(rotated_variants) + 1
    center_idx = len(rotated_variants) // 2
    
    # Get original angle from stats (should be NaN/empty for original)
    orig_stats = stats_df[stats_df.variant == 'original'].iloc[0]
    orig_angle = 0.0  # Original has no rotation
    
    all_angles.insert(center_idx, orig_angle)
    all_wd.insert(center_idx, orig_wd)
    all_hol.insert(center_idx, orig_hol)
    all_names.insert(center_idx, 'Original')
    
    x_positions = list(range(len(all_angles)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append('Original\n(0°)')
        else:
            angle = all_angles[i]
            x_labels.append(f'{angle:.1f}°')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(x_positions, wd_values, 'o-', linewidth=2.5, markersize=8, 
            label='Working Day', color='green', alpha=0.8)
    ax.plot(x_positions, hol_values, 's-', linewidth=2.5, markersize=8, 
            label='Holiday', color='orange', alpha=0.8)
    
    orig_idx = center_idx
    ax.plot(x_positions[orig_idx], wd_values[orig_idx], 'o', markersize=14, color='green', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='green')
    ax.plot(x_positions[orig_idx], hol_values[orig_idx], 's', markersize=14, color='orange', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='orange')
    
    ax.axvline(x=x_positions[orig_idx], color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(x_positions[orig_idx], max_y), 
               xytext=(x_positions[orig_idx], max_y + y_range * 0.08),
               ha='center', fontsize=12, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Rotation Variants (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=12, fontweight='bold')
    ax.set_title('Gravitational Work vs Network Rotation\n(Original Network at Center)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rotation work plot → {out_png}")


def plot_work_parabolic_scaling(results_df, stats_df, out_png):
    orig_row = results_df[results_df.variant == 'original']
    scaling_variants = results_df[results_df.variant.str.contains('scal', case=False, na=False)]
    
    if len(orig_row) == 0:
        logger.warning("No 'original' variant found for scaling plot")
        return
    
    if len(scaling_variants) == 0:
        logger.warning("No scaling variants found")
        return
    
    # Merge with stats to get transformation parameters
    merged = scaling_variants.merge(stats_df, on='variant', how='left')
    
    scaled_variants = []
    for _, row in merged.iterrows():
        scale = row['scale_factor']
        if pd.isna(scale):
            continue
        scaled_variants.append((scale, row['work_wd'], row['work_hol'], row['variant']))
    
    scaled_variants.sort(key=lambda x: x[0])
    
    orig_wd = orig_row.iloc[0]['work_wd']
    orig_hol = orig_row.iloc[0]['work_hol']
    
    all_scales = [scale for scale, _, _, _ in scaled_variants]
    all_wd = [wd for _, wd, _, _ in scaled_variants]
    all_hol = [hol for _, _, hol, _ in scaled_variants]
    all_names = [name for _, _, _, name in scaled_variants]
    
    n_total = len(scaled_variants) + 1
    center_idx = len(scaled_variants) // 2
    
    # Original has scale factor 1.0
    orig_scale = 1.0
    
    all_scales.insert(center_idx, orig_scale)
    all_wd.insert(center_idx, orig_wd)
    all_hol.insert(center_idx, orig_hol)
    all_names.insert(center_idx, 'Original')
    
    x_positions = list(range(len(all_scales)))
    x_labels = []
    
    for i, name in enumerate(all_names):
        if name == 'Original':
            x_labels.append('Original\n(×1.0)')
        else:
            scale = all_scales[i]
            x_labels.append(f'×{scale:.2f}')
    
    x_positions = np.array(x_positions)
    wd_values = np.array(all_wd)
    hol_values = np.array(all_hol)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(x_positions, wd_values, 'o-', linewidth=2.5, markersize=8, 
            label='Working Day', color='purple', alpha=0.8)
    ax.plot(x_positions, hol_values, 's-', linewidth=2.5, markersize=8, 
            label='Holiday', color='brown', alpha=0.8)
    
    orig_idx = center_idx
    ax.plot(x_positions[orig_idx], wd_values[orig_idx], 'o', markersize=14, color='purple', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='purple')
    ax.plot(x_positions[orig_idx], hol_values[orig_idx], 's', markersize=14, color='brown', 
            markerfacecolor='gold', markeredgewidth=3, markeredgecolor='brown')
    
    ax.axvline(x=x_positions[orig_idx], color='black', linestyle='--', alpha=0.6, linewidth=2)
    
    max_y = max(wd_values[orig_idx], hol_values[orig_idx])
    y_range = max(max(wd_values), max(hol_values)) - min(min(wd_values), min(hol_values))
    ax.annotate('Original Network', 
               xy=(x_positions[orig_idx], max_y), 
               xytext=(x_positions[orig_idx], max_y + y_range * 0.08),
               ha='center', fontsize=12, fontweight='bold', color='black',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.8, lw=1.5))
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
    ax.set_xlabel('Scaling Variants (scale factor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Uphill Work (m·units)', fontsize=12, fontweight='bold')
    ax.set_title('Gravitational Work vs Network Scaling\n(Original Network at Center)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    y_min = min(min(wd_values), min(hol_values))
    y_max = max(max(wd_values), max(hol_values))
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scaling work plot → {out_png}")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--conf', default='conf_wheight.json')
    p.add_argument('-city', '--city', help='City name to override conf file')
    args = p.parse_args()
    conf = json.load(open(args.conf))
    
    city = args.city if args.city else conf.get("city")
    if not city:
        logger.error("City not specified in command line or config file")
        sys.exit(1)
    else:
        logger.info(f"Using city: {city}")
    
    bbox_file = os.path.join(os.environ['WORKSPACE'], 'data', 'metropolis.json')
    bbox = json.load(open(bbox_file)).get(city, None)
    if not bbox:
        logger.error(f"No bounding box found for city '{city}' in {bbox_file}")
        sys.exit(1)
    
    base_dir = os.path.join(os.environ['WORKSPACE'], 'topolity', 'data', 'data_processed')
    graphs_dir = os.path.join(base_dir, city, "graphs")
    cells_dir = os.path.join(base_dir, city, f"{city}_basic_model/1000_cells")
    cells_file = os.path.join(cells_dir, "cell_coordinates.csv")
    od_w_file = os.path.join(cells_dir, "od_matrix_working_day.csv")
    od_h_file = os.path.join(cells_dir, "od_matrix_holiday.csv")
    dem_file = os.path.join(base_dir, city, "dem", f"{city}_dem.tif")
      
    cells_crs = conf['cells_crs']
    api_key = conf['api_key']
    ds = conf.get('ds', 10.0)
    m, g = conf.get('m', 1.0), conf.get('g', 1.0)

    # graphs
    graphs, stats = load_graphs(graphs_dir)
    G0     = graphs['original']

    # DEM
    dem = DEMReader(dem_file)
    dem.ensure_dem(api_key, {
        "min_lon": bbox[0][1], "min_lat": bbox[0][0],
        "max_lon": bbox[2][1], "max_lat": bbox[2][0]
    })
    dem.open()

    # cells & OD
    cells   = load_cells(cells_file, cells_crs, bbox)
    od_work = load_od(od_w_file)
    od_hol  = load_od(od_h_file)
    valid   = set(cells.cell)
    od_work = od_work[od_work.origin.isin(valid)&od_work.dest.isin(valid)]
    od_hol  = od_hol[od_hol.origin.isin(valid)&od_hol.dest.isin(valid)]
    logger.info(f"Filtered OD: work={len(od_work)}, hol={len(od_hol)}")
    cell_map = map_cells_to_nodes(G0, cells)

    # compute work
    results = []
    for var, G in graphs.items():
        logger.info(f"=== Variant '{var}' ===")
        log_resources(f"start variant {var}")
        ev = WorkEvaluatorNK(G, dem, ds=ds, m=m, g=g)
        wd = ev.compute_total_work(cell_map, od_work)
        ho = ev.compute_total_work(cell_map, od_hol)
        results.append({'variant':var,'work_wd':wd,'work_hol':ho})
        log_resources(f"end variant {var}")

    # save
    df = pd.DataFrame(results)
    out_csv = os.path.join(graphs_dir, 'gravitational_work_by_variant.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    # explanatory plots
    work_png = os.path.join(graphs_dir, 'work_by_variant.png')
    plot_work_by_variant(df, work_png)

    diff_png = os.path.join(graphs_dir, 'variant_differences.png')
    plot_variant_differences(df, diff_png)

    # parabolic plots for different variant types (only if data is available)
    available_types = check_available_variant_types(df)
    
    if not available_types['original']:
        logger.warning("No 'original' variant found - parabolic plots may not work correctly")
    
    if available_types['translations']:
        translations_png = os.path.join(graphs_dir, 'work_parabolic_translations.png')
        plot_work_parabolic_translations(df, stats, translations_png)
    else:
        logger.info("No translation variants found - skipping translation plot")
    
    if available_types['rotations']:
        rotations_png = os.path.join(graphs_dir, 'work_parabolic_rotations.png')
        plot_work_parabolic_rotations(df, stats, rotations_png)
    else:
        logger.info("No rotation variants found - skipping rotation plot")
    
    if available_types['scaling']:
        scaling_png = os.path.join(graphs_dir, 'work_parabolic_scaling.png')
        plot_work_parabolic_scaling(df, stats, scaling_png)
    else:
        logger.info("No scaling variants found - skipping scaling plot")

    # additional diagnostic plots
    # arc-work distribution for original graph
    ev0 = WorkEvaluatorNK(G0, dem, ds=ds, m=m, g=g)
    aw_png = os.path.join(graphs_dir, 'arc_work_distribution.png')
    plot_arc_work_distribution(ev0.arc_work, aw_png)

    # arc-work boxplot
    aw_box_png = os.path.join(graphs_dir, 'arc_work_boxplot.png')
    plot_arc_work_boxplot(ev0.arc_work, aw_box_png)

    # DEM elevation histogram
    dem_hist_png = os.path.join(graphs_dir, 'dem_elevation_distribution.png')
    plot_dem_elevation_distribution(dem, dem_hist_png)

    # cell-node mapping scatter
    cnm_png = os.path.join(graphs_dir, 'cell_node_mapping.png')
    plot_cell_node_mapping(cells, cell_map, G0, cnm_png)

    dem.close()


if __name__ == '__main__':
    main()
