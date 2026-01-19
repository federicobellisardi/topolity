#!/usr/bin/env python3
"""
Unified gravitational work calculator for graph variants.

Supports both standard graphs (graphs/) and fine-grid graphs (graphs_fine_grid/).
Can use configuration file or command-line arguments.

Author: Federico Bellisardi
Date: 2025-01-14

Files Required:
    /home/fbellisardi/code/topolity/data/data_processed/{CITY}/
    ├── graphs/                              # (opzionale) grafi standard
    │   ├── graph_stats.csv                  # necessario se usi graphs/
    │   ├── graph_original.pkl
    │   └── graph_*.pkl
    ├── graphs_fine_grid/                    # (opzionale) grafi fine grid
    │   └── graph_*.pkl
    ├── {CITY}_basic_model/1000_cells/       # OBBLIGATORIO
    │   ├── cell_coordinates.csv            # ✓ presente
    │   ├── od_matrix_working_day.csv        # ✓ presente
    │   └── od_matrix_holiday.csv            # ✓ presente
    └── dem/                                 # OBBLIGATORIO
        └── {CITY}_dem.tif                   # DEM file

Usage examples:
    # Standard graphs with config file
    python gravitational_work.py -c tools/conf/conf_wheight.json
    
    # Fine-grid graphs with CLI args
    python gravitational_work.py --city santiago --graphs-dir graphs_fine_grid --resume
    
    # With multiprocessing
    python gravitational_work.py --city santiago --workers 8
    
    # With plotting
    python gravitational_work.py --city santiago --plot
"""

import os
import sys
import json
import pickle
import re
import argparse
import logging
import psutil
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import requests
import pandas as pd
import numpy as np
from numpy.linalg import norm

import networkx as nx
import networkit as nk
from networkit import distance

import rasterio
from pyproj import Transformer
from shapely.geometry import box, LineString, Polygon

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import multiprocessing


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Process for resource monitoring
p = psutil.Process(os.getpid())
p.cpu_percent(None)


# ============================================================================
# DEM Management
# ============================================================================

class DEMReader:
    """DEM reader with optional download from OpenTopography."""
    
    def __init__(self, dem_file: Path):
        self.dem_file = Path(dem_file)
        self.src = None

    def ensure_dem(self, api_key: Optional[str], bbox: dict):
        """Download DEM once; skip if >1MB already exists."""
        if self.dem_file.exists() and self.dem_file.stat().st_size > 1e6:
            logger.info(f"DEM already present ({self.dem_file}), skipping download.")
            return
        
        if not api_key:
            logger.warning("No API key provided, cannot download DEM")
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
        
        self.dem_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dem_file, 'wb') as f:
            f.write(resp.content)
        
        size_mb = self.dem_file.stat().st_size / 1e6
        logger.info(f"Saved DEM ({size_mb:.1f} MB)")

    def open(self):
        """Open the GeoTIFF for sampling."""
        if not self.dem_file.exists():
            logger.warning(f"DEM file not found: {self.dem_file}")
            return
        logger.info(f"Opening DEM file {self.dem_file}")
        self.src = rasterio.open(str(self.dem_file))

    def sample(self, coords):
        """coords: list of (lon, lat), returns: list of elevation floats"""
        if self.src is None:
            return [0.0] * len(coords)
        return [val[0] for val in self.src.sample(coords)]

    def close(self):
        if self.src:
            logger.info("Closing DEM dataset")
            self.src.close()


# ============================================================================
# Core Work Computation
# ============================================================================

def compute_edge_work(G: nx.Graph, u, v, dem_reader: DEMReader, 
                     ds: float = 10.0, m: float = 1.0, g: float = 1.0
                     ) -> Tuple[float, List[Dict]]:
    """
    Compute uphill gravitational work for a single edge.
    
    Returns:
        (total_work, segment_results)
    """
    # Get edge data (handle MultiGraph)
    edge_data = G.get_edge_data(u, v)
    
    if isinstance(edge_data, dict):
        data = edge_data
    else:
        # Multiple edges, take first
        data = list(edge_data.values())[0] if edge_data else {}
    
    # Get geometry
    geom = data.get('geometry')
    
    if geom is None:
        # Create straight line from node coordinates
        x1 = G.nodes[u].get('x', G.nodes[u].get('lon'))
        y1 = G.nodes[u].get('y', G.nodes[u].get('lat'))
        x2 = G.nodes[v].get('x', G.nodes[v].get('lon'))
        y2 = G.nodes[v].get('y', G.nodes[v].get('lat'))
        geom = LineString([(x1, y1), (x2, y2)])
    
    # Sample points along edge
    length = geom.length
    n_pts = max(int(length / ds) + 1, 2)
    dists = np.linspace(0, length, n_pts)
    pts = [geom.interpolate(d) for d in dists]
    coords = [(pt.x, pt.y) for pt in pts]
    
    # Sample elevations
    elevs = dem_reader.sample(coords)
    
    # Calculate uphill work per segment
    work = 0.0
    segment_results = []
    for i, (h1, h2) in enumerate(zip(elevs[:-1], elevs[1:])):
        if h2 > h1:
            seg_work = m * g * (h2 - h1)
            work += seg_work
        else:
            seg_work = 0.0
        segment_results.append({
            'start_coord': coords[i],
            'end_coord': coords[i+1],
            'work': seg_work
        })
    
    return work, segment_results


def _work_for_origin(args):
    """Worker function for multiprocessing."""
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


class WorkEvaluator:
    """OD-weighted gravitational work evaluator."""
    
    def __init__(self, G: nx.Graph, dem_reader: DEMReader, 
                 ds: float = 10.0, m: float = 1.0, g: float = 1.0):
        self.G = G
        self.dem = dem_reader
        self.ds = ds
        self.m = m
        self.g = g
        
        # Map nodes to sequential IDs
        self.node2id = {n: i for i, n in enumerate(G.nodes())}
        self.id2node = {i: n for n, i in self.node2id.items()}
        
        # Precompute per-edge uphill work
        logger.info("Precomputing edge work for all edges")
        self.arc_work = {}
        self.arc_work_segments = {}
        
        for u, v in tqdm(G.edges(), desc="Precomputing edge work", leave=False):
            w, segments = compute_edge_work(G, u, v, dem_reader, ds=ds, m=m, g=g)
            self.arc_work[(self.node2id[u], self.node2id[v])] = w
            self.arc_work_segments[(self.node2id[u], self.node2id[v])] = segments
        
        # Build NetworKit graph weighted by geometric length
        logger.info("Converting to NetworKit graph")
        n = len(self.node2id)
        nkG = nk.Graph(n, weighted=True, directed=G.is_directed())
        
        # Collect minimal lengths for (u, v) pairs
        min_len = {}
        for u, v, data in G.edges(data=True):
            length = data.get('length')
            if length is None:
                # Fallback to geometry length
                geom = data.get('geometry')
                if geom is None:
                    x1 = G.nodes[u].get('x', G.nodes[u].get('lon'))
                    y1 = G.nodes[u].get('y', G.nodes[u].get('lat'))
                    x2 = G.nodes[v].get('x', G.nodes[v].get('lon'))
                    y2 = G.nodes[v].get('y', G.nodes[v].get('lat'))
                    if all(c is not None for c in [x1, y1, x2, y2]):
                        geom = LineString([(x1, y1), (x2, y2)])
                length = float(geom.length) if geom is not None else 1.0
            
            key = (u, v)
            min_len[key] = min(length, min_len.get(key, float('inf')))
        
        for (u, v), length in min_len.items():
            u_id, v_id = self.node2id[u], self.node2id[v]
            nkG.addEdge(u_id, v_id, length)
        
        self.nkG = nkG
        logger.info(f"NetworKit graph built with {nkG.numberOfNodes()} nodes, {nkG.numberOfEdges()} edges")
    
    def compute_total_work(self, cell_map: Dict[int, int], od_df: pd.DataFrame, 
                          use_multiprocessing: bool = False) -> float:
        """Sum OD-weighted arc work along shortest paths."""
        logger.info(f"Computing total work for {len(od_df)} OD flows")
        
        # Group destinations by origin
        od_by_origin = {}
        for origin, dest, flow in od_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, []).append((dest, flow))
        
        start_time = time.time()
        total_origins = len(od_by_origin)
        
        if use_multiprocessing:
            # Parallel computation
            args_list = [
                (origin_cell, dest_list, cell_map, self.node2id, self.nkG, 
                 self.arc_work, idx, total_origins, start_time)
                for idx, (origin_cell, dest_list) in enumerate(od_by_origin.items())
            ]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(_work_for_origin, args_list)
            total = sum(results)
        else:
            # Sequential computation
            total = 0.0
            for idx, (origin_cell, dest_list) in enumerate(tqdm(od_by_origin.items(), 
                                                                 desc="OD shortest paths", 
                                                                 leave=False)):
                src_node = cell_map.get(origin_cell)
                if src_node is None:
                    continue
                sid = self.node2id.get(src_node)
                if sid is None:
                    continue
                
                runner = distance.Dijkstra(self.nkG, sid, True)
                runner.run()
                
                for dest_cell, flow in dest_list:
                    dest_node = cell_map.get(dest_cell)
                    if dest_node is None:
                        continue
                    tid = self.node2id.get(dest_node)
                    if tid is None:
                        continue
                    path = runner.getPath(tid)
                    if not path:
                        continue
                    for a_id, b_id in zip(path[:-1], path[1:]):
                        total += flow * self.arc_work.get((a_id, b_id), 0.0)
        
        logger.info(f"Total uphill work = {total:.2f}")
        return total


# ============================================================================
# Graph Loading
# ============================================================================

def parse_filename(filename: str) -> Tuple[str, float]:
    """Parse graph filename to extract variant and parameter."""
    if 'original' in filename:
        return ('original', 0.0)
    
    # Rotation pattern
    rot_match = re.search(r'rot_([+-])(\d+)p(\d+)deg', filename)
    if rot_match:
        sign = 1 if rot_match.group(1) == '+' else -1
        deg = int(rot_match.group(2)) + int(rot_match.group(3)) / 100.0
        deg = sign * deg
        if deg > 180:
            deg = deg - 360
        return ('rotation', deg)
    
    # Translation pattern
    trans_match = re.search(r'trans_(\d+)m_a\+(\d+)', filename)
    if trans_match:
        distance = int(trans_match.group(1))
        angle = int(trans_match.group(2))
        variant_name = f'translation_{angle:03d}'
        return (variant_name, distance)
    
    return ('unknown', 0.0)


def load_graphs_from_stats(graphs_dir: Path) -> Dict[str, nx.Graph]:
    """Load graphs using graph_stats.csv (standard mode)."""
    logger.info(f"Loading graphs from {graphs_dir} using graph_stats.csv")
    
    stats_file = graphs_dir / 'graph_stats.csv'
    if not stats_file.exists():
        raise FileNotFoundError(f"graph_stats.csv not found in {graphs_dir}")
    
    stats = pd.read_csv(stats_file).rename(columns=lambda c: c.strip())
    graphs = {}
    
    for _, row in stats.iterrows():
        var = row['variant']
        fn = 'graph_original.pkl' if var == 'original' else f'graph_{var}.pkl'
        path = graphs_dir / fn
        
        with open(path, 'rb') as f:
            G = pickle.load(f)
        
        # Convert MultiGraph to simple Graph
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            H = nx.DiGraph() if G.is_directed() else nx.Graph()
            for n, data in G.nodes(data=True):
                H.add_node(n, **data)
            
            grouped = defaultdict(list)
            for u, v, key, data in G.edges(keys=True, data=True):
                grouped[(u, v)].append(data.copy())
            
            for (u, v), data_list in grouped.items():
                lengths = [d.get('length', np.inf) for d in data_list]
                length = min(lengths)
                H.add_edge(u, v, length=length)
            
            G = H
        
        graphs[var] = G
    
    logger.info(f"Loaded {len(graphs)} graph variants")
    return graphs


def load_graphs_from_glob(graphs_dir: Path) -> Dict[str, nx.Graph]:
    """Load all graph_*.pkl files (fine-grid mode)."""
    logger.info(f"Loading graphs from {graphs_dir} using glob pattern")
    
    pkl_files = sorted(graphs_dir.glob("graph_*.pkl"))
    logger.info(f"Found {len(pkl_files)} graph files")
    
    graphs = {}
    for pkl_file in tqdm(pkl_files, desc="Loading graphs"):
        try:
            with open(pkl_file, 'rb') as f:
                G = pickle.load(f)
            graphs[pkl_file.name] = G
        except Exception as e:
            logger.error(f"Error loading {pkl_file.name}: {e}")
    
    return graphs


# ============================================================================
# Data Loading
# ============================================================================

def load_cells(path: Path, cells_crs: str = "EPSG:3857", bbox: Optional[dict] = None) -> pd.DataFrame:
    """Load cells and optionally filter by bbox."""
    logger.info("Loading cells")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'cell_id': 'cell'})
    
    transformer = Transformer.from_crs(cells_crs, "EPSG:4326", always_xy=True)
    
    # Determine bbox columns and convert to lon/lat
    if set(['x_min', 'y_min', 'x_max', 'y_max']).issubset(df.columns):
        lon_min, lat_min = transformer.transform(df['x_min'].to_numpy(), df['y_min'].to_numpy())
        lon_max, lat_max = transformer.transform(df['x_max'].to_numpy(), df['y_max'].to_numpy())
        df['lon_min'], df['lat_min'], df['lon_max'], df['lat_max'] = lon_min, lat_min, lon_max, lat_max
    elif set(['lon_min', 'lat_min', 'lon_max', 'lat_max']).issubset(df.columns):
        # Check if values are in meters (EPSG:3857)
        if df['lon_min'].abs().max() > 360:
            lon_min, lat_min = transformer.transform(df['lon_min'].to_numpy(), df['lat_min'].to_numpy())
            lon_max, lat_max = transformer.transform(df['lon_max'].to_numpy(), df['lat_max'].to_numpy())
            df['lon_min'], df['lat_min'], df['lon_max'], df['lat_max'] = lon_min, lat_min, lon_max, lat_max
    elif set(['min_lon', 'min_lat', 'max_lon', 'max_lat']).issubset(df.columns):
        df = df.rename(columns={'min_lon': 'lon_min', 'min_lat': 'lat_min', 
                                'max_lon': 'lon_max', 'max_lat': 'lat_max'})
    else:
        raise KeyError(f"Missing bounding box columns in {path}")
    
    # Filter by bbox if provided
    if bbox:
        poly = Polygon([(lon, lat) for lat, lon in bbox])
        df['keep'] = df.apply(
            lambda r: box(r.lon_min, r.lat_min, r.lon_max, r.lat_max).intersects(poly),
            axis=1)
        df = df[df.keep].copy()
        logger.info(f"  Retained {len(df)} cells in bbox")
    
    df['cent_lon'] = 0.5 * (df['lon_min'] + df['lon_max'])
    df['cent_lat'] = 0.5 * (df['lat_min'] + df['lat_max'])
    
    return df[['cell', 'cent_lon', 'cent_lat', 'lon_min', 'lat_min', 'lon_max', 'lat_max']]


def load_od(path: Path) -> pd.DataFrame:
    """Load OD matrix."""
    logger.info(f"Loading OD flows from {path}")
    df = pd.read_csv(path).rename(columns={
        'cell_origin': 'origin',
        'cell_destination': 'dest',
        'count': 'flow'
    })
    df['flow'] = pd.to_numeric(df['flow'], errors='coerce').fillna(0)
    df = df[df['flow'] > 0].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} positive flows")
    return df


def map_cells_to_nodes(G: nx.Graph, cell_df: pd.DataFrame) -> Dict[int, int]:
    """Map cells to nearest graph nodes."""
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
            (G.nodes[n].get('x', G.nodes[n].get('lon')),
             G.nodes[n].get('y', G.nodes[n].get('lat')))
            for n in lst
        ])
        idx_min = np.argmin(norm(coords - cent, axis=1))
        mapping[row.cell] = lst[idx_min]
        total += 1
    
    logger.info(f"Mapped {len(mapping)} cells → {total} central nodes")
    return mapping


# ============================================================================
# Analysis & Plotting
# ============================================================================

def analyze_graphs(graphs: Dict[str, nx.Graph], dem_reader: DEMReader, 
                  cell_map: Dict[int, int], od_work: pd.DataFrame, od_hol: pd.DataFrame,
                  output_csv: Path, output_dir: Path,
                  ds: float = 10.0, m: float = 1.0, g: float = 1.0,
                  use_multiprocessing: bool = False, resume: bool = False,
                  save_segments: bool = True) -> pd.DataFrame:
    """Analyze all graphs and compute gravitational work."""
    
    # Check for existing results if resuming
    existing_done = set()
    wrote_header = False
    if resume and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            if 'filename' in existing_df.columns:
                existing_done = set(existing_df['filename'].astype(str).tolist())
            elif 'variant' in existing_df.columns:
                existing_done = set(existing_df['variant'].astype(str).tolist())
            wrote_header = True
            logger.info(f"Resume enabled: found {len(existing_done)} results in {output_csv}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV {output_csv}: {e}")
    
    to_process = {k: v for k, v in graphs.items() if (not resume or k not in existing_done)}
    logger.info(f"Analyzing {len(to_process)} graphs (skipped {len(graphs) - len(to_process)} already done)")
    
    new_rows = []
    for variant_key, G in to_process.items():
        logger.info(f"=== Processing '{variant_key}' ===")
        
        # Parse variant info
        if variant_key.endswith('.pkl'):
            variant_type, parameter = parse_filename(variant_key)
            variant_name = variant_key.replace('graph_', '').replace('.pkl', '')
        else:
            variant_type = variant_key
            parameter = 0.0
            variant_name = variant_key
        
        # Create evaluator
        evaluator = WorkEvaluator(G, dem_reader, ds=ds, m=m, g=g)
        
        # Compute work
        wd = evaluator.compute_total_work(cell_map, od_work, use_multiprocessing=use_multiprocessing)
        hol = evaluator.compute_total_work(cell_map, od_hol, use_multiprocessing=use_multiprocessing)
        total_work = wd + hol
        
        row = {
            'variant': variant_key,
            'variant_type': variant_type,
            'parameter': parameter,
            'work_wd': wd,
            'work_hol': hol,
            'total_work': total_work,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
        new_rows.append(row)
        logger.info(f"  WD={wd:,.0f}, HOL={hol:,.0f}, TOTAL={total_work:,.0f}")
        
        # Save segment-level work
        if save_segments:
            segment_rows = []
            for (u, v), segments in evaluator.arc_work_segments.items():
                for seg in segments:
                    segment_rows.append({
                        'u': evaluator.id2node[u],
                        'v': evaluator.id2node[v],
                        'start_x': seg['start_coord'][0],
                        'start_y': seg['start_coord'][1],
                        'end_x': seg['end_coord'][0],
                        'end_y': seg['end_coord'][1],
                        'segment_work': seg['work']
                    })
            seg_work_df = pd.DataFrame(segment_rows)
            seg_work_csv = output_dir / f'arc_work_segments_{variant_name}.csv'
            seg_work_df.to_csv(seg_work_csv, index=False)
            logger.info(f"  Segment work saved → {seg_work_csv.name}")
        
        # Append to CSV incrementally
        try:
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=not wrote_header, index=False)
            wrote_header = True
        except Exception as e:
            logger.warning(f"Failed to append to {output_csv}: {e}")
    
    # Load and return full results
    try:
        df = pd.read_csv(output_csv)
        if 'variant' in df.columns:
            df = df.drop_duplicates(subset=['variant'], keep='last')
        for col in ['parameter', 'total_work', 'work_wd', 'work_hol', 'num_nodes', 'num_edges']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame(new_rows)


def plot_results(df: pd.DataFrame, output_dir: Path, mode: str = 'auto'):
    """Create plots based on data type."""
    
    # Auto-detect mode
    if mode == 'auto':
        if 'rotation' in df['variant_type'].values or any('translation_' in str(v) for v in df['variant_type'].values):
            mode = 'fine_grid'
        else:
            mode = 'standard'
    
    if mode == 'fine_grid':
        plot_fine_grid(df, output_dir)
    else:
        plot_standard(df, output_dir)


def plot_fine_grid(df: pd.DataFrame, output_dir: Path):
    """Create fine-grid analysis plots (3 subplots)."""
    sns.set_style("whitegrid")
    
    df_rotation = df[df['variant_type'] == 'rotation'].copy()
    df_original = df[df['variant_type'] == 'original'].copy()
    
    translation_types = [vt for vt in df['variant_type'].unique() if str(vt).startswith('translation_')]
    
    df_trans_ns = []
    df_trans_we = []
    
    for trans_type in translation_types:
        angle = int(str(trans_type).split('_')[1])
        df_trans = df[df['variant_type'] == trans_type].copy()
        
        if (45 <= angle <= 135) or (225 <= angle <= 315):
            df_trans_ns.append(df_trans)
        else:
            df_trans_we.append(df_trans)
    
    df_trans_ns = pd.concat(df_trans_ns, ignore_index=True) if df_trans_ns else pd.DataFrame()
    df_trans_we = pd.concat(df_trans_we, ignore_index=True) if df_trans_we else pd.DataFrame()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Rotation plot
    ax = axes[0]
    if len(df_rotation) > 0:
        ax.scatter(df_rotation['parameter'], df_rotation['total_work'], 
                  s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)
        
        if len(df_original) > 0:
            ax.scatter([0], df_original['total_work'].values, 
                      s=200, alpha=0.9, c='red', marker='*', 
                      edgecolors='darkred', linewidth=2, 
                      label='Original', zorder=10)
    
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Gravitational Work (J·m)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Translation N-S plot
    ax = axes[1]
    if len(df_trans_ns) > 0:
        for trans_type in df_trans_ns['variant_type'].unique():
            angle = int(str(trans_type).split('_')[1])
            df_subset = df_trans_ns[df_trans_ns['variant_type'] == trans_type]
            
            if 225 <= angle <= 315:
                ax.scatter(-df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'South ({angle}°)')
            else:
                ax.scatter(df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'North ({angle}°)')
    
    if len(df_original) > 0:
        ax.scatter([0], df_original['total_work'].values, 
                  s=200, alpha=0.9, c='red', marker='*', 
                  edgecolors='darkred', linewidth=2, 
                  label='Original', zorder=10)
    
    ax.set_xlabel('Translation Distance (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Gravitational Work (J·m)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Translation W-E plot
    ax = axes[2]
    if len(df_trans_we) > 0:
        for trans_type in df_trans_we['variant_type'].unique():
            angle = int(str(trans_type).split('_')[1])
            df_subset = df_trans_we[df_trans_we['variant_type'] == trans_type]
            
            if 135 <= angle <= 225:
                ax.scatter(-df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'West ({angle}°)')
            else:
                ax.scatter(df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'East ({angle}°)')
    
    if len(df_original) > 0:
        ax.scatter([0], df_original['total_work'].values, 
                  s=200, alpha=0.9, c='red', marker='*', 
                  edgecolors='darkred', linewidth=2, 
                  label='Original', zorder=10)
    
    ax.set_xlabel('Translation Distance (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Gravitational Work (J·m)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "gravitational_work.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved: {output_file}")
    
    output_file_pdf = output_dir / "gravitational_work.pdf"
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved: {output_file_pdf}")
    plt.close()


def plot_standard(df: pd.DataFrame, output_dir: Path):
    """Create standard variant comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by total work
    df_sorted = df.sort_values('total_work')
    
    # Bar plot
    ax1.barh(df_sorted['variant'], df_sorted['total_work'])
    ax1.set_xlabel('Total Gravitational Work')
    ax1.set_ylabel('Variant')
    ax1.set_title('Gravitational Work by Variant')
    
    # Scatter plot (if parameter exists)
    if 'parameter' in df.columns and df['parameter'].notna().any():
        ax2.scatter(df['parameter'], df['total_work'], s=100, alpha=0.7)
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Total Gravitational Work')
        ax2.set_title('Work vs Parameter')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    
    output_file = output_dir / "gravitational_work.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved: {output_file}")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    logger.info("=" * 60)
    logger.info("GRAVITATIONAL WORK SUMMARY")
    logger.info("=" * 60)
    
    df_orig = df[df['variant_type'] == 'original']
    if len(df_orig) > 0:
        orig_work = df_orig['total_work'].values[0]
        logger.info(f"\nOriginal configuration work: {orig_work:,.0f}")
    
    for variant_type in df['variant_type'].unique():
        if variant_type == 'original':
            continue
        
        df_var = df[df['variant_type'] == variant_type]
        logger.info(f"\n{variant_type}:")
        logger.info(f"  Count: {len(df_var)}")
        logger.info(f"  Min work: {df_var['total_work'].min():,.0f} at {df_var.loc[df_var['total_work'].idxmin(), 'parameter']}")
        logger.info(f"  Max work: {df_var['total_work'].max():,.0f} at {df_var.loc[df_var['total_work'].idxmax(), 'parameter']}")
        logger.info(f"  Mean work: {df_var['total_work'].mean():,.0f}")
        
        if len(df_orig) > 0:
            pct_change = ((df_var['total_work'] - orig_work) / orig_work * 100)
            logger.info(f"  Change from original: {pct_change.min():.1f}% to {pct_change.max():.1f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified gravitational work calculator')
    
    # Configuration
    parser.add_argument('-c', '--conf', type=str, help='Config JSON file')
    parser.add_argument('--city', type=str, help='City name')
    
    # Directories
    parser.add_argument('--graphs-dir', type=str, nargs='+', help='Graphs directory name(s) (e.g., "graphs" "graphs_fine_grid")')
    parser.add_argument('--base-dir', type=str, help='Base data directory')
    parser.add_argument('--all-graphs', action='store_true', help='Load from both graphs/ and graphs_fine_grid/ if they exist')
    
    # Processing options
    parser.add_argument('--workers', type=int, help='Use multiprocessing with N workers (0=disabled)')
    parser.add_argument('--resume', action='store_true', help='Skip already computed graphs')
    parser.add_argument('--no-segments', action='store_true', help='Skip segment-level CSV files')
    
    # Physical parameters
    parser.add_argument('--ds', type=float, help='Sampling interval (meters)')
    parser.add_argument('--m', type=float, help='Mass')
    parser.add_argument('--g', type=float, help='Gravity')
    
    # Output
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-mode', type=str, choices=['auto', 'fine_grid', 'standard'], 
                       default='auto', help='Plot style')
    
    args = parser.parse_args()
    
    # Load configuration
    conf = {}
    if args.conf:
        with open(args.conf) as f:
            conf = json.load(f)
    
    # Determine parameters (CLI overrides config)
    city = args.city or conf.get('city')
    if not city:
        logger.error("City not specified")
        sys.exit(1)
    
    # Get bbox
    bbox_file = Path(os.environ.get('WORKSPACE', '/home/fbellisardi/code')) / 'data' / 'metropolis.json'
    if bbox_file.exists():
        with open(bbox_file) as f:
            bbox_data = json.load(f)
            bbox = bbox_data.get(city)
    else:
        bbox = None
    
    # Determine directories
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        workspace = Path(os.environ.get('WORKSPACE', '/home/fbellisardi/code'))
        base_dir = workspace / 'topolity' / 'data' / 'data_processed' / city
    
    # Determine which graph directories to load
    if args.all_graphs:
        graphs_dir_names = []
        for candidate in ['graphs', 'graphs_fine_grid']:
            if (base_dir / candidate).exists():
                graphs_dir_names.append(candidate)
        if not graphs_dir_names:
            logger.error("No graphs directories found (checked 'graphs' and 'graphs_fine_grid')")
            sys.exit(1)
    elif args.graphs_dir:
        graphs_dir_names = args.graphs_dir if isinstance(args.graphs_dir, list) else [args.graphs_dir]
    else:
        graphs_dir_names = [conf.get('graphs_dir', 'graphs')]
    
    graphs_dirs = [base_dir / name for name in graphs_dir_names]
    
    cells_dir = base_dir / f"{city}_basic_model" / "1000_cells"
    cells_file = cells_dir / "cell_coordinates.csv"
    od_w_file = cells_dir / "od_matrix_working_day.csv"
    od_h_file = cells_dir / "od_matrix_holiday.csv"
    
    dem_file = base_dir / "dem" / f"{city}_dem.tif"
    if not dem_file.exists():
        dem_file = base_dir / "land" / f"{city}_dem.tif"
    
    # Parameters
    cells_crs = conf.get('cells_crs', 'EPSG:3857')
    api_key = conf.get('api_key')
    ds = args.ds or conf.get('ds', 10.0)
    m = args.m or conf.get('m', 1.0)
    g = args.g or conf.get('g', 1.0)
    use_multiprocessing = (args.workers or 0) > 0
    
    logger.info(f"City: {city}")
    logger.info(f"Graphs directories: {', '.join(str(d) for d in graphs_dirs)}")
    logger.info(f"Parameters: ds={ds}, m={m}, g={g}")
    logger.info(f"Multiprocessing: {use_multiprocessing}")
    
    # Load graphs from all directories
    all_graphs = {}
    for graphs_dir in graphs_dirs:
        if not graphs_dir.exists():
            logger.warning(f"Graphs directory not found: {graphs_dir}")
            continue
        
        # Auto-detect loading mode
        if (graphs_dir / 'graph_stats.csv').exists():
            graphs = load_graphs_from_stats(graphs_dir)
        else:
            graphs = load_graphs_from_glob(graphs_dir)
        
        # Add directory prefix to avoid name conflicts
        dir_name = graphs_dir.name
        for variant_key, G in graphs.items():
            # Add directory prefix if loading from multiple directories
            if len(graphs_dirs) > 1:
                prefixed_key = f"{dir_name}/{variant_key}"
            else:
                prefixed_key = variant_key
            all_graphs[prefixed_key] = G
    
    if not all_graphs:
        logger.error("No graphs loaded from any directory")
        sys.exit(1)
    
    logger.info(f"Total graphs loaded: {len(all_graphs)}")
    graphs = all_graphs
    
    # Load DEM
    dem = DEMReader(dem_file)
    if bbox and api_key:
        dem.ensure_dem(api_key, {
            "min_lon": bbox[0][1], "min_lat": bbox[0][0],
            "max_lon": bbox[2][1], "max_lat": bbox[2][0]
        })
    dem.open()
    
    # Load cells and OD
    cells = load_cells(cells_file, cells_crs, bbox)
    od_work = load_od(od_w_file)
    od_hol = load_od(od_h_file)
    
    # Filter OD by valid cells
    valid = set(cells.cell)
    od_work = od_work[od_work.origin.isin(valid) & od_work.dest.isin(valid)]
    od_hol = od_hol[od_hol.origin.isin(valid) & od_hol.dest.isin(valid)]
    logger.info(f"Filtered OD: work={len(od_work)}, hol={len(od_hol)}")
    
    # Map cells to nodes (using first/original graph)
    G_ref = next(iter(graphs.values()))
    cell_map = map_cells_to_nodes(G_ref, cells)
    
    # Analyze (output to first directory)
    output_dir = graphs_dirs[0]
    output_csv = output_dir / "gravitational_work_results.csv"
    df = analyze_graphs(
        graphs, dem, cell_map, od_work, od_hol,
        output_csv, output_dir,
        ds=ds, m=m, g=g,
        use_multiprocessing=use_multiprocessing,
        resume=args.resume,
        save_segments=not args.no_segments
    )
    
    logger.info(f"Results saved: {output_csv}")
    
    # Summary
    print_summary(df)
    
    # Plot
    if args.plot:
        plot_results(df, output_dir, mode=args.plot_mode)
    
    # Cleanup
    dem.close()
    
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
