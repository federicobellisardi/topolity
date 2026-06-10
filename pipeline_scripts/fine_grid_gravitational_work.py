#!/usr/bin/env python3
"""OD-weighted gravitational work for fine-grid graph variants."""

import os
import sys
import pickle
import re
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
import networkit as nk
from networkit import distance
from shapely.geometry import LineString
from pyproj import Transformer


# Constants
DS = 10.0    # Sampling interval along edges (meters)
M = 1.0      # Mass (kg)
# Align with wheight.py default physics (m=1.0, g=1.0)
GRAV = 1.0   # Gravity scaling (unitless here, matches wheight.py default)
# CRS for the cells grid (matches wheight.py conf)
CELLS_CRS = "EPSG:3857"


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse graph filename to extract transformation type and parameter.
    
    Returns:
        (variant_name, parameter_value)
    
    Examples:
        graph_original.pkl -> ('original', 0.0)
        graph_rot_+10p00deg.pkl -> ('rotation', 10.0)
        graph_rot_-15p00deg.pkl -> ('rotation', -15.0)
        graph_trans_1500m_a+075.pkl -> ('translation_075', 1500.0)
        graph_trans_2500m_a+255.pkl -> ('translation_255', 2500.0)
    """
    if 'original' in filename:
        return ('original', 0.0)
    
    # Rotation pattern: graph_rot_[+/-]XXpXXdeg.pkl
    rot_match = re.search(r'rot_([+-])(\d+)p(\d+)deg', filename)
    if rot_match:
        sign = 1 if rot_match.group(1) == '+' else -1
        deg = int(rot_match.group(2)) + int(rot_match.group(3)) / 100.0
        deg = sign * deg
        # Convert angles > 180° to negative equivalents for symmetric centering at 0°
        if deg > 180:
            deg = deg - 360
        return ('rotation', deg)
    
    # Translation pattern: graph_trans_XXXXm_a+YYY.pkl
    trans_match = re.search(r'trans_(\d+)m_a\+(\d+)', filename)
    if trans_match:
        distance = int(trans_match.group(1))
        angle = int(trans_match.group(2))
        variant_name = f'translation_{angle:03d}'
        return (variant_name, distance)

    # Scale patterns: graph_scale_ns_1p050.pkl or graph_scale_ew_0p950.pkl
    scale_match = re.search(r'scale_(ns|ew)_([0-9]+p[0-9]+)', filename)
    if scale_match:
        axis = scale_match.group(1)
        factor = float(scale_match.group(2).replace('p', '.'))
        variant_name = f'scale_{axis}'
        return (variant_name, factor)
    
    return ('unknown', 0.0)


def load_dem_reader(dem_file: Path):
    """Load DEM for elevation sampling."""
    try:
        import rasterio
        if dem_file.exists():
            src = rasterio.open(str(dem_file))
            print(f"✓ Loaded DEM: {dem_file}")
            return src
        else:
            print(f"⚠ DEM not found: {dem_file}")
            return None
    except ImportError:
        print("⚠ rasterio not installed, cannot sample elevations")
        return None


def compute_edge_work(G: nx.MultiDiGraph, u, v, dem_src, ds: float = DS) -> Tuple[float, List[Dict]]:
    """
    Compute uphill gravitational work for a single edge.
    
    Args:
        G: NetworkX graph
        u, v: Edge nodes
        dem_src: rasterio DEM source
        ds: Sampling interval in meters
    
    Returns:
        (total_work, segment_results)
        total_work: Total uphill work (m * g * Δh) for this edge
        segment_results: List of dicts with 'start_coord', 'end_coord', 'work' for each segment
    """
    # Get edge data (handle MultiDiGraph)
    edge_data = G.get_edge_data(u, v)
    
    if isinstance(edge_data, dict):
        # Single edge
        data = edge_data
    else:
        # Multiple edges, take first
        data = list(edge_data.values())[0]
    
    # Get geometry
    geom = data.get('geometry')
    
    if geom is None:
        # Create straight line from node coordinates
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        geom = LineString([(x1, y1), (x2, y2)])
    
    # Sample points along edge
    length = geom.length
    n_pts = max(int(length / ds) + 1, 2)
    dists = np.linspace(0, length, n_pts)
    pts = [geom.interpolate(d) for d in dists]
    coords = [(pt.x, pt.y) for pt in pts]
    
    # Sample elevations
    if dem_src is None:
        # Use node elevations if available
        z_u = G.nodes[u].get('z', 0.0)
        z_v = G.nodes[v].get('z', 0.0)
        elevs = np.linspace(z_u, z_v, n_pts)
    else:
        elevs = [val[0] for val in dem_src.sample(coords)]
    
    # Calculate uphill work per segment
    work = 0.0
    segment_results = []
    for i, (h1, h2) in enumerate(zip(elevs[:-1], elevs[1:])):
        if h2 > h1:
            seg_work = M * GRAV * (h2 - h1)
            work += seg_work
        else:
            seg_work = 0.0
        segment_results.append({
            'start_coord': coords[i],
            'end_coord': coords[i+1],
            'work': seg_work
        })
    
    return work, segment_results


class ODWorkEvaluator:
    """OD-weighted gravitational work evaluator (aligned to wheight.py logic)."""
    def __init__(self, G: nx.MultiDiGraph, dem_src, ds: float = DS, m: float = M, g: float = GRAV):
        self.G = G
        self.dem_src = dem_src
        self.ds = ds
        self.m = m
        self.g = g

        # Map nodes to sequential IDs
        self.node2id = {n: i for i, n in enumerate(G.nodes())}
        self.id2node = {i: n for n, i in self.node2id.items()}

        # Precompute per-edge uphill work on the chosen edge per (u, v)
        self.arc_work = {}
        self.arc_work_segments = {}

        # For MultiDiGraph, select a representative edge per (u,v)
        for u, v in tqdm(G.edges(), desc="Precomputing edge work", leave=False):
            w, segments = compute_edge_work(G, u, v, self.dem_src, ds=self.ds)
            self.arc_work[(self.node2id[u], self.node2id[v])] = w
            self.arc_work_segments[(self.node2id[u], self.node2id[v])] = segments

        # Build NetworKit graph weighted by geometric length (as in wheight.py)
        n = len(self.node2id)
        nkG = nk.Graph(n, weighted=True, directed=G.is_directed())

        # Collect minimal lengths for (u, v) pairs
        min_len = {}
        for u, v, data in G.edges(data=True):
            length = data.get('length')
            if length is None:
                # Fallback to geometry length (if available)
                geom = None
                if 'geometry' in data:
                    geom = data['geometry']
                else:
                    # Try to reconstruct straight line
                    x1, y1 = G.nodes[u].get('x', G.nodes[u].get('lon')), G.nodes[u].get('y', G.nodes[u].get('lat'))
                    x2, y2 = G.nodes[v].get('x', G.nodes[v].get('lon')), G.nodes[v].get('y', G.nodes[v].get('lat'))
                    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                        geom = LineString([(x1, y1), (x2, y2)])
                length = float(geom.length) if geom is not None else 1.0

            key = (u, v)
            min_len[key] = min(length, min_len.get(key, float('inf')))

        for (u, v), length in min_len.items():
            u_id, v_id = self.node2id[u], self.node2id[v]
            nkG.addEdge(u_id, v_id, length)

        self.nkG = nkG

    def compute_total_work(self, cell_map: Dict[int, int], od_df: pd.DataFrame) -> float:
        """Sum OD-weighted arc work along shortest paths (by length)."""
        # Group destinations by origin for efficiency
        od_by_origin = {}
        for origin, dest, flow in od_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, []).append((dest, flow))

        total = 0.0
        # For each origin, run Dijkstra once
        for origin_cell, dest_list in tqdm(od_by_origin.items(), desc="OD shortest paths", leave=False):
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
                # Sum uphill work along the path
                for a_id, b_id in zip(path[:-1], path[1:]):
                    total += flow * self.arc_work.get((a_id, b_id), 0.0)

        return total


def load_all_graphs(graphs_dir: Path) -> Dict[str, nx.MultiDiGraph]:
    """Load all pickle graph files from directory."""
    graphs = {}
    
    pkl_files = sorted(graphs_dir.glob("graph_*.pkl"))
    
    print(f"Found {len(pkl_files)} graph files")
    
    for pkl_file in tqdm(pkl_files, desc="Loading graphs"):
        try:
            with open(pkl_file, 'rb') as f:
                G = pickle.load(f)
            
            graphs[pkl_file.name] = G
            
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
    
    return graphs


def analyze_graphs(graphs: Dict[str, nx.MultiDiGraph], dem_src, output_csv: Path, city: str, resume: bool = False) -> pd.DataFrame:
    """
    Compute OD-weighted gravitational work for all graphs (aligned to wheight.py).

    Returns:
        DataFrame with columns: filename, variant_type, parameter, total_work, num_nodes, num_edges
    """
    # Determine already computed files if resuming
    existing_done = set()
    wrote_header = False
    if resume and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            if 'filename' in existing_df.columns:
                existing_done = set(existing_df['filename'].astype(str).tolist())
            wrote_header = True  # CSV already has a header
            print(f"\nResume enabled: found {len(existing_done)} results in {output_csv}")
        except Exception as e:
            print(f"Warning: could not read existing CSV {output_csv}: {e}")
    
    to_process = {fn: G for fn, G in graphs.items() if (not resume or fn not in existing_done)}
    print(f"\nAnalyzing {len(to_process)} graphs (skipped {len(graphs) - len(to_process)} already done)...")
    
    # Incremental append to CSV to persist partial progress
    # Prepare OD data paths
    base_dir = Path(f"/home/fbellisardi/code/topolity/data/data_processed/{city}")
    cells_dir = base_dir / f"{city}_basic_model/1000_cells"
    cells_file = cells_dir / "cell_coordinates.csv"
    od_w_file = cells_dir / "od_matrix_working_day.csv"
    od_h_file = cells_dir / "od_matrix_holiday.csv"

    # Load cells and OD flows (minimal schema without CRS transform)
    def load_cells_minimal(path: Path) -> pd.DataFrame:
        """Load cells file and convert bbox to lon/lat like wheight.py does."""
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'cell_id': 'cell'})

        transformer = Transformer.from_crs(CELLS_CRS, "EPSG:4326", always_xy=True)

        # Determine bbox columns and convert to lon/lat if needed
        if set(['x_min', 'y_min', 'x_max', 'y_max']).issubset(df.columns):
            lon_min, lat_min = transformer.transform(df['x_min'].to_numpy(), df['y_min'].to_numpy())
            lon_max, lat_max = transformer.transform(df['x_max'].to_numpy(), df['y_max'].to_numpy())
            df['lon_min'], df['lat_min'], df['lon_max'], df['lat_max'] = lon_min, lat_min, lon_max, lat_max
        elif set(['lon_min', 'lat_min', 'lon_max', 'lat_max']).issubset(df.columns):
            # If lon/lat columns look like projected meters, convert them
            if df['lon_min'].abs().max() > 360:  # heuristically detect EPSG:3857 meters
                lon_min, lat_min = transformer.transform(df['lon_min'].to_numpy(), df['lat_min'].to_numpy())
                lon_max, lat_max = transformer.transform(df['lon_max'].to_numpy(), df['lat_max'].to_numpy())
                df['lon_min'], df['lat_min'], df['lon_max'], df['lat_max'] = lon_min, lat_min, lon_max, lat_max
        elif set(['min_lon', 'min_lat', 'max_lon', 'max_lat']).issubset(df.columns):
            df = df.rename(columns={'min_lon': 'lon_min', 'min_lat': 'lat_min', 'max_lon': 'lon_max', 'max_lat': 'lat_max'})
        else:
            raise KeyError(f"Missing bounding box columns in {path}; available columns: {list(df.columns)}")

        if 'cent_lon' not in df.columns or 'cent_lat' not in df.columns:
            df['cent_lon'] = 0.5 * (df['lon_min'] + df['lon_max'])
            df['cent_lat'] = 0.5 * (df['lat_min'] + df['lat_max'])

        return df[['cell', 'cent_lon', 'cent_lat', 'lon_min', 'lat_min', 'lon_max', 'lat_max']]

    def load_od_minimal(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path).rename(columns={
            'cell_origin': 'origin',
            'cell_destination': 'dest',
            'count': 'flow'
        })
        df['flow'] = pd.to_numeric(df['flow'], errors='coerce').fillna(0)
        df = df[df['flow'] > 0].reset_index(drop=True)
        return df

    print(f"\nLoading cells and OD flows for city '{city}'")
    cells_df = load_cells_minimal(cells_file)
    od_work = load_od_minimal(od_w_file)
    od_hol = load_od_minimal(od_h_file)

    # Map cells to nearest graph nodes inside bounding box
    def map_cells_to_nodes(G: nx.MultiDiGraph, cell_df: pd.DataFrame) -> Dict[int, int]:
        xs = np.array([data.get('x', data.get('lon')) for _, data in G.nodes(data=True)])
        ys = np.array([data.get('y', data.get('lat')) for _, data in G.nodes(data=True)])
        nodes = np.array([n for n, _ in G.nodes(data=True)])

        mapping = {}
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
            idx_min = np.argmin(np.linalg.norm(coords - cent, axis=1))
            mapping[row.cell] = lst[idx_min]
        return mapping

    # Build cell map using the original graph (closest nodes)
    # Choose original graph if available; otherwise, use any graph
    G_original = None
    for fn, G in graphs.items():
        if 'original' in fn:
            G_original = G
            break
    if G_original is None and graphs:
        G_original = next(iter(graphs.values()))

    cell_map = map_cells_to_nodes(G_original, cells_df)

    new_rows = []
    for filename, G in tqdm(to_process.items(), desc="Computing OD-weighted work"):
        variant_type, parameter = parse_filename(filename)
        evaluator = ODWorkEvaluator(G, dem_src, ds=DS, m=M, g=GRAV)
        wd = evaluator.compute_total_work(cell_map, od_work)
        hol = evaluator.compute_total_work(cell_map, od_hol)
        total_work = wd + hol

        row = {
            'filename': filename,
            'variant_type': variant_type,
            'parameter': parameter,
            'total_work': total_work,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
        new_rows.append(row)
        print(f"  {filename}: WD={wd:,.0f}, HOL={hol:,.0f}, TOTAL={total_work:,.0f}")

        # Save segment-level work to CSV
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
        # Extract variant name from filename (remove graph_ prefix and .pkl suffix)
        variant_name = filename.replace('graph_', '').replace('.pkl', '')
        seg_work_csv = output_csv.parent / f'arc_work_segments_{variant_name}.csv'
        seg_work_df.to_csv(seg_work_csv, index=False)
        print(f"    Segment work saved → {seg_work_csv.name}")

        # Append row to CSV immediately
        try:
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=not wrote_header, index=False)
            wrote_header = True
        except Exception as e:
            print(f"Warning: failed to append to {output_csv}: {e}")
    
    # Load merged dataframe from CSV (ensures de-duplication if needed)
    try:
        merged_df = pd.read_csv(output_csv)
        if 'filename' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['filename'], keep='last')
        # Ensure numeric columns are properly typed
        for col in ['parameter', 'total_work', 'num_nodes', 'num_edges']:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        return merged_df
    except Exception:
        # Fallback: return in-memory results
        df = pd.DataFrame(new_rows)
        return df


def plot_work_vs_parameter(df: pd.DataFrame, output_dir: Path):
    """
    Create scatter plots showing gravitational work vs transformation parameter.
    
    Expected: parabola with minimum at original configuration.
    Note: Work is in Joule-meters (J·m) = force × distance, not Joules.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Separate by transformation type
    df_rotation = df[df['variant_type'] == 'rotation'].copy()
    df_original = df[df['variant_type'] == 'original'].copy()
    
    # Get all translation types dynamically
    translation_types = [vt for vt in df['variant_type'].unique() if vt.startswith('translation_')]
    
    # Separate translations by direction based on actual convention:
    # 0°=East, 90°=North, 180°=West, 270°=South
    df_trans_ns = []  # North-South: angles close to 90° and 270°
    df_trans_we = []  # West-East: angles close to 0° and 180°
    
    for trans_type in translation_types:
        # Extract angle from variant name (e.g., 'translation_075' -> 75)
        angle = int(trans_type.split('_')[1])
        df_trans = df[df['variant_type'] == trans_type].copy()
        
        # Categorize based on angle
        # North-South: 90° ±45° (45-135°) and 270° ±45° (225-315°)
        if (45 <= angle <= 135) or (225 <= angle <= 315):
            df_trans_ns.append(df_trans)
        # West-East: 0° ±45° (315-45°) and 180° ±45° (135-225°)
        else:
            df_trans_we.append(df_trans)
    
    # Combine into single dataframes
    df_trans_ns = pd.concat(df_trans_ns, ignore_index=True) if df_trans_ns else pd.DataFrame()
    df_trans_we = pd.concat(df_trans_we, ignore_index=True) if df_trans_we else pd.DataFrame()
    
    # Create subplots: 3 plots (rotation, translation N-S, translation W-E)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Rotation plot
    ax = axes[0]
    if len(df_rotation) > 0:
        ax.scatter(df_rotation['parameter'], df_rotation['total_work'], 
                  s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)
        
        # Add original point
        if len(df_original) > 0:
            ax.scatter([0], df_original['total_work'].values, 
                      s=200, alpha=0.9, c='red', marker='*', 
                      edgecolors='darkred', linewidth=2, 
                      label='Original', zorder=10)
        
        # Fit parabola only if R² > 0.95
        if len(df_rotation) >= 3 and 1==0:  # Disabled for rotations
            coeffs = np.polyfit(df_rotation['parameter'], df_rotation['total_work'], 2)
            x_fit = np.linspace(df_rotation['parameter'].min(), 
                               df_rotation['parameter'].max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            
            # Calculate R²
            y_mean = df_rotation['total_work'].mean()
            ss_tot = np.sum((df_rotation['total_work'] - y_mean)**2)
            ss_res = np.sum((df_rotation['total_work'] - np.polyval(coeffs, df_rotation['parameter']))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > 0.95:
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.6, 
                       label=f'Parabolic fit (R²={r2:.3f})')
                
                # Find vertex
                vertex_x = -coeffs[1] / (2 * coeffs[0])
                ax.axvline(vertex_x, color='green', linestyle=':', linewidth=1.5, 
                          label=f'Vertex: {vertex_x:.1f}°')
    
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Gravitational Work (J·m)', fontsize=12, fontweight='bold')
    # ax.set_title('Gravitational Work vs Rotation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Translation North-South plot
    ax = axes[1]
    if len(df_trans_ns) > 0:
        for trans_type in df_trans_ns['variant_type'].unique():
            angle = int(trans_type.split('_')[1])
            df_subset = df_trans_ns[df_trans_ns['variant_type'] == trans_type]
            
            # Plot South (270°) on left (negative x) and North (90°) on right (positive x)
            if 225 <= angle <= 315:  # South direction (270°)
                ax.scatter(-df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'South ({angle}°)')
            else:  # North direction (90°)
                ax.scatter(df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'North ({angle}°)')
    
    # Add original point
    if len(df_original) > 0:
        ax.scatter([0], df_original['total_work'].values, 
                  s=200, alpha=0.9, c='red', marker='*', 
                  edgecolors='darkred', linewidth=2, 
                  label='Original', zorder=10)
    
    ax.set_xlabel('Translation Distance (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Gravitational Work (J·m)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Translation West-East plot
    ax = axes[2]
    if len(df_trans_we) > 0:
        for trans_type in df_trans_we['variant_type'].unique():
            angle = int(trans_type.split('_')[1])
            df_subset = df_trans_we[df_trans_we['variant_type'] == trans_type]
            
            # Plot West (180°) on left (negative x) and East (0°) on right (positive x)
            if 135 <= angle <= 225:  # West direction (180°)
                ax.scatter(-df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'West ({angle}°)')
            else:  # East direction (0°)
                ax.scatter(df_subset['parameter'], df_subset['total_work'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'East ({angle}°)')
    
    # Add original point
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
    
    # Save figure
    output_file = output_dir / "fine_grid_gravitational_work.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_file}")
    
    output_file_pdf = output_dir / "fine_grid_gravitational_work.pdf"
    plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {output_file_pdf}")
    

def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("GRAVITATIONAL WORK SUMMARY")
    print("="*60)
    
    # Original
    df_orig = df[df['variant_type'] == 'original']
    if len(df_orig) > 0:
        orig_work = df_orig['total_work'].values[0]
        print(f"\nOriginal configuration work: {orig_work:,.0f} J")
    
    # By type
    for variant_type in df['variant_type'].unique():
        if variant_type == 'original':
            continue
        
        df_var = df[df['variant_type'] == variant_type]
        print(f"\n{variant_type}:")
        print(f"  Count: {len(df_var)}")
        print(f"  Min work: {df_var['total_work'].min():,.0f} J at {df_var.loc[df_var['total_work'].idxmin(), 'parameter']}")
        print(f"  Max work: {df_var['total_work'].max():,.0f} J at {df_var.loc[df_var['total_work'].idxmax(), 'parameter']}")
        print(f"  Mean work: {df_var['total_work'].mean():,.0f} J")
        
        if len(df_orig) > 0:
            pct_change = ((df_var['total_work'] - orig_work) / orig_work * 100)
            print(f"  Change from original: {pct_change.min():.1f}% to {pct_change.max():.1f}%")


def main(city: str, resume: bool = False):
    """Main execution."""
    print("="*60)
    print("FINE GRID GRAVITATIONAL WORK ANALYSIS")
    print("="*60)

    base_dir = Path(f"/home/fbellisardi/code/topolity/data/data_processed/{city}")
    graphs_dir = base_dir / "graphs_fine_grid"
    output_dir = base_dir / "graphs_fine_grid"
    dem_file = base_dir / "dem" / f"{city}_dem.tif"
    if not dem_file.exists():
        alt_dem = base_dir / "land" / f"{city}_dem.tif"
        if alt_dem.exists():
            dem_file = alt_dem
        else:
            print(f"Warning: DEM file not found at {dem_file} or {alt_dem}")

    # Check directories
    if not graphs_dir.exists():
        print(f"Error: Graphs directory not found: {graphs_dir}")
        sys.exit(1)
    
    print(f"\nGraphs directory: {graphs_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load DEM
    dem_src = load_dem_reader(dem_file)
    
    # Load all graphs
    graphs = load_all_graphs(graphs_dir)
    
    if len(graphs) == 0:
        print("No graphs found!")
        sys.exit(1)
    
    # Output CSV path
    output_csv = output_dir / "fine_grid_gravitational_work.csv"
    
    # Analyze graphs with resume support
    df = analyze_graphs(graphs, dem_src, output_csv=output_csv, city=city, resume=resume)
    
    # Ensure de-duplication in memory as well
    if 'filename' in df.columns:
        df = df.drop_duplicates(subset=['filename'], keep='last')
    print(f"\n✓ Results up to date: {output_csv}")
    
    # Print summary
    print_summary(df)
    
    # Create plots
    plot_work_vs_parameter(df, output_dir)
    
    # Close DEM
    if dem_src is not None:
        dem_src.close()
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute gravitational work for fine-grid graph variants')
    parser.add_argument('--city', type=str, default="santiago", help='City name (default: santiago)')
    parser.add_argument('--resume', action='store_true', help='Skip already computed graphs and append new results')
    cli_args = parser.parse_args()
    main(city=cli_args.city, resume=cli_args.resume)
