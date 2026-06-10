#!/usr/bin/env python3
"""OD-weighted gravitational work for fine-grid graph variants."""

import os
import sys
import pickle
import re
import gc
import csv
from pathlib import Path
import argparse
from typing import Dict, Tuple, List, Set
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
import math


# Constants
DS = 10.0    # Sampling interval along edges (meters)

M_PHYS_KG = 1200.0
G_PHYS = 9.81

# CRS for the cells grid (matches wheight.py conf)
CELLS_CRS = "EPSG:3857"

DEBUG_LENGTHS = True
ROOT = Path("/home/fbellisardi/code/topolity")
DEFAULT_DATA_ROOT = ROOT / "data" / "data_processed"


def graph_looks_lonlat(G: nx.MultiDiGraph) -> bool:
    xs = []
    ys = []

    for _, data in list(G.nodes(data=True))[:1000]:
        x = data.get("x", data.get("lon"))
        y = data.get("y", data.get("lat"))
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        return False

    return (
        max(abs(np.array(xs))) <= 180
        and max(abs(np.array(ys))) <= 90
    )


def make_metric_transformer_from_graph(G: nx.MultiDiGraph):
    """
    If graph coordinates are lon/lat, project them to a local metric CRS.
    If they already look metric, return None.
    """
    if not graph_looks_lonlat(G):
        return None

    xs = []
    ys = []

    for _, data in G.nodes(data=True):
        x = data.get("x", data.get("lon"))
        y = data.get("y", data.get("lat"))
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))

    lon0 = float(np.mean(xs))
    lat0 = float(np.mean(ys))

    # Local azimuthal equidistant projection centered on the city
    proj_str = (
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)


def coords_to_metric(coords: np.ndarray, transformer=None) -> np.ndarray:
    """
    Convert Nx2 coordinates to meters if transformer is available.
    Otherwise assume they are already metric.
    """
    coords = np.asarray(coords, dtype=float)

    if transformer is None:
        return coords

    x_m, y_m = transformer.transform(coords[:, 0], coords[:, 1])
    return np.column_stack([x_m, y_m])



def compute_lambda_from_fuel_params(
    consumption_l_per_100km: Tuple[float, float] = (5.0, 8.0),
    energy_mj_per_l: float = 36.0,
    efficiency: float = 0.25,
) -> Dict[str, float]:
    """
    Compute lambda from transport-energy parameters using:
        lambda = eta * E_l * c

    Returns lambda both in MJ/100km and J/m.
    """
    c_min, c_max = consumption_l_per_100km
    c_mean = 0.5 * (c_min + c_max)

    # lambda in MJ/100km
    lambda_min_mj_per_100km = efficiency * energy_mj_per_l * c_min
    lambda_max_mj_per_100km = efficiency * energy_mj_per_l * c_max
    lambda_mean_mj_per_100km = efficiency * energy_mj_per_l * c_mean

    # Convert MJ/100km to J/m: multiply by 10
    lambda_min_j_per_m = lambda_min_mj_per_100km * 10.0
    lambda_max_j_per_m = lambda_max_mj_per_100km * 10.0
    lambda_mean_j_per_m = lambda_mean_mj_per_100km * 10.0

    return {
        "eta": efficiency,
        "E_l_MJ_per_L": energy_mj_per_l,
        "c_min_L_per_100km": c_min,
        "c_max_L_per_100km": c_max,
        "c_mean_L_per_100km": c_mean,
        "lambda_min_MJ_per_100km": lambda_min_mj_per_100km,
        "lambda_max_MJ_per_100km": lambda_max_mj_per_100km,
        "lambda_mean_MJ_per_100km": lambda_mean_mj_per_100km,
        "lambda_min_J_per_m": lambda_min_j_per_m,
        "lambda_max_J_per_m": lambda_max_j_per_m,
        "lambda_mean_J_per_m": lambda_mean_j_per_m,
    }


# Use mean lambda from fuel parameters as default horizontal-cost weight (J/m).
_LAMBDA_DEFAULTS = compute_lambda_from_fuel_params()
HORIZONTAL_COST_WEIGHT = _LAMBDA_DEFAULTS["lambda_mean_J_per_m"]


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
        graph_scale_ns_1p050.pkl -> ('scale_ns', 1.05)
        graph_scale_ew_0p950.pkl -> ('scale_ew', 0.95)
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


def _select_edge_data(G: nx.MultiDiGraph, u, v, data: Dict = None) -> Dict:
    """Select one representative edge-attribute dict for (u, v)."""
    if data is not None:
        return data

    edge_data = G.get_edge_data(u, v)
    if edge_data is None:
        return {}

    # DiGraph-style attrs dict
    if not isinstance(edge_data, dict):
        return {}
    if not edge_data:
        return {}

    first_val = next(iter(edge_data.values()))
    if not isinstance(first_val, dict):
        return edge_data

    # MultiDiGraph keyed dict: choose candidate with minimum length.
    candidates = list(edge_data.values())
    best = None
    best_len = float("inf")
    for cand in candidates:
        c_len = cand.get("length")
        if c_len is None and cand.get("geometry") is not None:
            c_len = float(cand["geometry"].length)
        if c_len is None:
            c_len = float("inf")
        if c_len < best_len:
            best_len = c_len
            best = cand
    return best if best is not None else candidates[0]


def compute_edge_vertical_gain_m(
        G: nx.MultiDiGraph,
        u,
        v,
        dem_src,
        ds: float = DS,
        elevation_source: str = "node",
        edge_data: Dict = None,
        metric_transformer=None,
    ) -> Tuple[float, List[Dict]]:
    """
    Compute positive vertical gain [m] for a single edge.
    """
    data = _select_edge_data(G, u, v, edge_data)

    geom = data.get("geometry")

    if geom is None:
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = LineString([(x1, y1), (x2, y2)])

    raw_coords = np.array(geom.coords, dtype=float)
    metric_transformer = make_metric_transformer_from_graph(G) if metric_transformer is None else metric_transformer
    raw_coords_m = coords_to_metric(raw_coords, metric_transformer)

    dx_raw = np.diff(raw_coords_m[:, 0])
    dy_raw = np.diff(raw_coords_m[:, 1])
    length_2d_m = float(np.sum(np.sqrt(dx_raw**2 + dy_raw**2)))

    n_pts = max(int(length_2d_m / ds) + 1, 2)

    dists = np.linspace(0, float(geom.length), n_pts)
    pts = [geom.interpolate(d) for d in dists]
    coords = [(pt.x, pt.y) for pt in pts]

    z_u = G.nodes[u].get("z", None)
    z_v = G.nodes[v].get("z", None)

    has_node_z = (
        z_u is not None
        and z_v is not None
        and np.isfinite(float(z_u))
        and np.isfinite(float(z_v))
    )

    if elevation_source == "node" and has_node_z:
        elevs = np.linspace(float(z_u), float(z_v), n_pts)
    elif dem_src is not None:
        elevs = [val[0] for val in dem_src.sample(coords)]
    elif has_node_z:
        elevs = np.linspace(float(z_u), float(z_v), n_pts)
    else:
        elevs = np.zeros(n_pts, dtype=float)

    # Clamp elevations to sea level (0 m).  SRTM and similar DEMs sometimes
    # return negative values over water bodies or as artefacts in coastal/delta
    # areas (e.g. Buenos Aires: -86 m; Brussels: -220 m).  Using raw negatives
    # inflates the apparent uphill gain for paths that cross these regions,
    # producing spurious counterexamples.  Since the gravitational work only
    # counts relative *uphill* effort, clamping the datum at 0 is physically
    # conservative and eliminates the artefact without affecting inland cities.
    elevs = np.maximum(0.0, np.asarray(elevs, dtype=float))

    vertical_gain_m = 0.0
    segment_results = []

    for i, (h1, h2) in enumerate(zip(elevs[:-1], elevs[1:])):
        if h2 > h1:
            seg_vertical_gain_m = float(h2 - h1)
            vertical_gain_m += seg_vertical_gain_m
        else:
            seg_vertical_gain_m = 0.0

        segment_results.append({
            "start_coord": coords[i],
            "end_coord": coords[i + 1],
            "vertical_gain_m": seg_vertical_gain_m,
        })

    return vertical_gain_m, segment_results



def compute_edge_geometry_length(
        G: nx.MultiDiGraph,
        u,
        v,
        edge_data: Dict = None,
        metric_transformer=None,
        ds: float = DS,
    ) -> float:
    """
    Return 2D metric length using the same sampling logic used for 3D length.
    This guarantees L3D >= L2D up to numerical precision.
    """
    data = _select_edge_data(G, u, v, edge_data)

    geom = data.get("geometry")
    if geom is None:
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = LineString([(x1, y1), (x2, y2)])

    raw_coords = np.array(geom.coords, dtype=float)
    raw_coords_m = coords_to_metric(raw_coords, metric_transformer)

    dx_raw = np.diff(raw_coords_m[:, 0])
    dy_raw = np.diff(raw_coords_m[:, 1])
    length_2d_m = float(np.sum(np.sqrt(dx_raw**2 + dy_raw**2)))

    n_pts = max(int(length_2d_m / ds) + 1, 2)

    dists = np.linspace(0, float(geom.length), n_pts)
    pts = [geom.interpolate(d) for d in dists]

    coords = np.array([(pt.x, pt.y) for pt in pts], dtype=float)
    coords_m = coords_to_metric(coords, metric_transformer)

    dx = np.diff(coords_m[:, 0])
    dy = np.diff(coords_m[:, 1])

    return float(np.sum(np.sqrt(dx**2 + dy**2)))

def compute_edge_3d_length(
        G: nx.MultiDiGraph,
        u,
        v,
        edge_data: Dict = None,
        dem_src=None,
        ds: float = DS,
        elevation_source: str = "node",
        metric_transformer=None,
    ) -> float:
    """
    Return 3D metric length:

        L_3D = sum sqrt(dx_m^2 + dy_m^2 + dz_m^2)

    where dx_m, dy_m and dz_m are all in meters.
    """
    data = _select_edge_data(G, u, v, edge_data)

    geom = data.get("geometry")
    if geom is None:
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = LineString([(x1, y1), (x2, y2)])

    # First compute metric 2D length
    raw_coords = np.array(geom.coords, dtype=float)
    raw_coords_m = coords_to_metric(raw_coords, metric_transformer)

    dx_raw = np.diff(raw_coords_m[:, 0])
    dy_raw = np.diff(raw_coords_m[:, 1])
    length_2d_m = float(np.sum(np.sqrt(dx_raw**2 + dy_raw**2)))

    # Sample along the original geometry parameter, but choose number of points from metric length
    n_pts = max(int(length_2d_m / ds) + 1, 2)

    # geom.length may be degrees, but interpolation parameter must use geom.length units
    dists = np.linspace(0, float(geom.length), n_pts)
    pts = [geom.interpolate(d) for d in dists]

    coords = np.array([(pt.x, pt.y) for pt in pts], dtype=float)
    coords_m = coords_to_metric(coords, metric_transformer)

    z_u = G.nodes[u].get("z", None)
    z_v = G.nodes[v].get("z", None)

    has_node_z = (
        z_u is not None
        and z_v is not None
        and np.isfinite(float(z_u))
        and np.isfinite(float(z_v))
    )

    if elevation_source == "node" and has_node_z:
        elevs = np.linspace(float(z_u), float(z_v), n_pts)

    elif dem_src is not None:
        # DEM sampling expects original lon/lat if DEM is in EPSG:4326
        elevs = np.array([val[0] for val in dem_src.sample(coords)], dtype=float)

    elif has_node_z:
        elevs = np.linspace(float(z_u), float(z_v), n_pts)

    else:
        elevs = np.zeros(n_pts, dtype=float)

    dx = np.diff(coords_m[:, 0])
    dy = np.diff(coords_m[:, 1])
    dz = np.diff(elevs)

    return float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))


class ODWorkEvaluator:
    """OD-weighted gravitational work evaluator (aligned to wheight.py logic)."""
    def __init__(self, G: nx.MultiDiGraph, dem_src, ds: float = DS,
                 keep_segments: bool = False, elevation_source: str = 'node'):
        self.G = G
        self.dem_src = dem_src
        self.ds = ds
        self.keep_segments = keep_segments
        self.elevation_source = elevation_source
        
        self.metric_transformer = make_metric_transformer_from_graph(G)

        if DEBUG_LENGTHS:
            print("\n[DEBUG] Coordinate system check")
            print(f"  graph_looks_lonlat = {graph_looks_lonlat(G)}")
            print(f"  using_metric_transformer = {self.metric_transformer is not None}")        
        

        # Map nodes to sequential IDs
        self.node2id = {n: i for i, n in enumerate(G.nodes())}
        self.id2node = {i: n for n, i in self.node2id.items()}

        # Precompute per-edge positive vertical gain [m]
        self.arc_vertical_gain_m = {}
        # store both 2D (planar) and 3D lengths for each arc
        self.arc_length_2d = {}
        self.arc_length_3d = {}
        self.arc_vertical_gain_segments = {} if keep_segments else None

        # Select one representative arc per (u, v) to avoid overwriting duplicates in MultiDiGraph.
        seen_arcs = set()
        for u, v in tqdm(G.edges(), desc="Precomputing edge work", leave=False):
            if (u, v) in seen_arcs:
                continue
            seen_arcs.add((u, v))

            selected_data = _select_edge_data(G, u, v)
            
            vertical_gain_m, segments = compute_edge_vertical_gain_m(
                G,
                u,
                v,
                self.dem_src,
                ds=self.ds,
                elevation_source=self.elevation_source,
                edge_data=selected_data,
                metric_transformer=self.metric_transformer,
            )
            self.arc_vertical_gain_m[(self.node2id[u], self.node2id[v])] = vertical_gain_m

            self.arc_length_2d[(self.node2id[u], self.node2id[v])] = compute_edge_geometry_length(
                G,
                u,
                v,
                edge_data=selected_data,
                metric_transformer=self.metric_transformer,
                ds=self.ds,
            )

            self.arc_length_3d[(self.node2id[u], self.node2id[v])] = compute_edge_3d_length(
                G,
                u,
                v,
                edge_data=selected_data,
                dem_src=self.dem_src,
                ds=self.ds,
                elevation_source=self.elevation_source,
                metric_transformer=self.metric_transformer,
            )
                       
            if self.keep_segments: self.arc_vertical_gain_segments[(self.node2id[u], self.node2id[v])] = segments

        if DEBUG_LENGTHS and self.arc_length_2d:
            l2 = np.array(list(self.arc_length_2d.values()), dtype=float)
            l3 = np.array(list(self.arc_length_3d.values()), dtype=float)

            valid = (l2 > 0) & np.isfinite(l2) & np.isfinite(l3)
            ratios = l3[valid] / l2[valid]

            print("\n[DEBUG] Edge length diagnostics")
            print(f"  2D length min/median/max [m]: {np.min(l2[valid]):.3f}, {np.median(l2[valid]):.3f}, {np.max(l2[valid]):.3f}")
            print(f"  3D length min/median/max [m]: {np.min(l3[valid]):.3f}, {np.median(l3[valid]):.3f}, {np.max(l3[valid]):.3f}")
            print(f"  L3D/L2D min/median/max: {np.min(ratios):.6f}, {np.median(ratios):.6f}, {np.max(ratios):.6f}")
            print(f"  suspicious ratios < 0.999999: {np.sum(ratios < 0.999999)} / {len(ratios)}")
            print(f"  suspicious ratios > 2: {np.sum(ratios > 2)} / {len(ratios)}")
            
        # Build two NetworKit graphs:
        # - nkG_2d: Dijkstra on planar/geometric 2D length
        # - nkG_3d: Dijkstra on true 3D terrain-following length
        n = len(self.node2id)

        nkG_2d = nk.Graph(n, weighted=True, directed=G.is_directed())
        nkG_3d = nk.Graph(n, weighted=True, directed=G.is_directed())

        for (u_id, v_id), length_2d in self.arc_length_2d.items():
            length_3d = self.arc_length_3d.get((u_id, v_id), length_2d)

            nkG_2d.addEdge(u_id, v_id, float(length_2d))
            nkG_3d.addEdge(u_id, v_id, float(length_3d))

        self.nkG_2d = nkG_2d
        self.nkG_3d = nkG_3d

        # Backward compatibility
        self.nkG = self.nkG_2d


    def compute_total_lengths_for_graph(
            self,
            nk_graph,
            cell_map: Dict[int, int],
            od_work_df: pd.DataFrame,
            od_hol_df: pd.DataFrame,
        ) -> Tuple[float, float, float]:
        """
        Run Dijkstra on the provided graph and then measure:
        - vertical uphill gain along selected paths
        - 2D path length along selected paths
        - 3D path length along selected paths

        Returns:
            total_vertical_gain, total_length_2d, total_length_3d
        """

        od_by_origin = {}

        for origin, dest, flow in od_work_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[0] += float(flow)

        for origin, dest, flow in od_hol_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[1] += float(flow)

        total_vertical_gain_m = 0.0
        total_length_2d = 0.0
        total_length_3d = 0.0

        for origin_cell, dest_map in tqdm(od_by_origin.items(), desc="OD shortest paths", leave=False):
            src_node = cell_map.get(origin_cell)
            if src_node is None:
                continue

            sid = self.node2id.get(src_node)
            if sid is None:
                continue

            runner = distance.Dijkstra(nk_graph, sid, True)
            runner.run()

            for dest_cell, (flow_wd, flow_hol) in dest_map.items():
                dest_node = cell_map.get(dest_cell)
                if dest_node is None:
                    continue

                tid = self.node2id.get(dest_node)
                if tid is None:
                    continue

                path = runner.getPath(tid)
                if not path:
                    continue

                path_vertical_gain_m = 0.0
                path_length_2d = 0.0
                path_length_3d = 0.0

                for a_id, b_id in zip(path[:-1], path[1:]):
                    path_vertical_gain_m += self.arc_vertical_gain_m.get((a_id, b_id), 0.0)
                    path_length_2d += self.arc_length_2d.get((a_id, b_id), 0.0)
                    path_length_3d += self.arc_length_3d.get((a_id, b_id), 0.0)

                flow_total = flow_wd + flow_hol

                if flow_total:
                    total_vertical_gain_m += flow_total * path_vertical_gain_m
                    total_length_2d += flow_total * path_length_2d
                    total_length_3d += flow_total * path_length_3d

        return total_vertical_gain_m, total_length_2d, total_length_3d

    def compute_total_lengths_2d_and_3d_dijkstra(
        self,
        cell_map: Dict[int, int],
        od_work_df: pd.DataFrame,
        od_hol_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compare Dijkstra paths computed using 2D and 3D edge lengths.
        """

        v_on_2d, l2d_on_2d, l3d_on_2d = self.compute_total_lengths_for_graph(
            self.nkG_2d,
            cell_map,
            od_work_df,
            od_hol_df,
        )

        v_on_3d, l2d_on_3d, l3d_on_3d = self.compute_total_lengths_for_graph(
            self.nkG_3d,
            cell_map,
            od_work_df,
            od_hol_df,
        )

        return {
            "vertical_gain_on_2d_path": v_on_2d,
            "path_length_2d_on_2d_path": l2d_on_2d,
            "path_length_3d_on_2d_path": l3d_on_2d,

            "vertical_gain_on_3d_path": v_on_3d,
            "path_length_2d_on_3d_path": l2d_on_3d,
            "path_length_3d_on_3d_path": l3d_on_3d,
        }

    def compute_total_work(self, cell_map: Dict[int, int], od_df: pd.DataFrame) -> float:
        """Backward-compatible wrapper returning only vertical uphill work."""
        total_wd, _ = self.compute_total_work_pair(cell_map, od_df, pd.DataFrame(columns=od_df.columns))
        return total_wd

    def compute_total_cost_pair(self, cell_map: Dict[int, int],
                                od_work_df: pd.DataFrame,
                                od_hol_df: pd.DataFrame,
                                horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT) -> Tuple[float, float, float]:
        """Compute vertical work, horizontal travel cost and their weighted sum."""
        # Group destinations by origin for efficiency
        od_by_origin = {}

        for origin, dest, flow in od_work_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[0] += float(flow)

        for origin, dest, flow in od_hol_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[1] += float(flow)

        total_vertical = 0.0
        # horizontal totals: keep 2D for backward compatibility
        total_horizontal_2d = 0.0
        # also accumulate 3D distances
        total_horizontal_3d = 0.0

        # For each origin, run Dijkstra once
        for origin_cell, dest_map in tqdm(od_by_origin.items(), desc="OD shortest paths", leave=False):
            src_node = cell_map.get(origin_cell)
            if src_node is None:
                continue
            sid = self.node2id.get(src_node)
            if sid is None:
                continue
            runner = distance.Dijkstra(self.nkG, sid, True)
            runner.run()

            for dest_cell, (flow_wd, flow_hol) in dest_map.items():
                dest_node = cell_map.get(dest_cell)
                if dest_node is None:
                    continue
                tid = self.node2id.get(dest_node)
                if tid is None:
                    continue
                path = runner.getPath(tid)
                if not path:
                    continue
                # Sum uphill work and horizontal travel along the path
                path_vertical = 0.0
                path_length_2d = 0.0
                path_length_3d = 0.0
                for a_id, b_id in zip(path[:-1], path[1:]):
                    path_vertical += self.arc_vertical_gain_m.get((a_id, b_id), 0.0)
                    path_length_2d += self.arc_length_2d.get((a_id, b_id), 0.0)
                    path_length_3d += self.arc_length_3d.get((a_id, b_id), 0.0)

                if flow_wd:
                    total_vertical += flow_wd * path_vertical
                    total_horizontal_2d += flow_wd * path_length_2d
                    total_horizontal_3d += flow_wd * path_length_3d
                if flow_hol:
                    total_vertical += flow_hol * path_vertical
                    total_horizontal_2d += flow_hol * path_length_2d
                    total_horizontal_3d += flow_hol * path_length_3d

        total_combined_2d = total_vertical + horizontal_cost_weight * total_horizontal_2d
        
        vertical_energy_J = M_PHYS_KG * G_PHYS * total_vertical
        horizontal_energy_2d_J = horizontal_cost_weight * total_horizontal_2d

        total_combined_2d = vertical_energy_J + horizontal_energy_2d_J
        
        
        # keep original return semantics (vertical, horizontal, combined) using 2D
        return total_vertical, total_horizontal_2d, total_combined_2d

    def compute_total_cost_pair_all(self, cell_map: Dict[int, int],
                                    od_work_df: pd.DataFrame,
                                    od_hol_df: pd.DataFrame,
                                    horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT) -> Tuple[float, float, float, float, float]:
        """Compute vertical work, horizontal 2D/3D travel totals and both combined costs.

        Returns: (vertical, horizontal_2d, combined_2d, horizontal_3d, combined_3d)
        """
        # Group destinations by origin for efficiency
        od_by_origin = {}

        for origin, dest, flow in od_work_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[0] += float(flow)

        for origin, dest, flow in od_hol_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[1] += float(flow)

        total_vertical = 0.0
        total_horizontal_2d = 0.0
        total_horizontal_3d = 0.0

        # For each origin, run Dijkstra once
        for origin_cell, dest_map in tqdm(od_by_origin.items(), desc="OD shortest paths", leave=False):
            src_node = cell_map.get(origin_cell)
            if src_node is None:
                continue
            sid = self.node2id.get(src_node)
            if sid is None:
                continue
            runner = distance.Dijkstra(self.nkG, sid, True)
            runner.run()

            for dest_cell, (flow_wd, flow_hol) in dest_map.items():
                dest_node = cell_map.get(dest_cell)
                if dest_node is None:
                    continue
                tid = self.node2id.get(dest_node)
                if tid is None:
                    continue
                path = runner.getPath(tid)
                if not path:
                    continue
                # Sum uphill work and both horizontal distances along the path
                path_vertical = 0.0
                path_length_2d = 0.0
                path_length_3d = 0.0
                for a_id, b_id in zip(path[:-1], path[1:]):
                    path_vertical += self.arc_vertical_gain_m.get((a_id, b_id), 0.0)
                    path_length_2d += self.arc_length_2d.get((a_id, b_id), 0.0)
                    path_length_3d += self.arc_length_3d.get((a_id, b_id), 0.0)

                if flow_wd:
                    total_vertical += flow_wd * path_vertical
                    total_horizontal_2d += flow_wd * path_length_2d
                    total_horizontal_3d += flow_wd * path_length_3d
                if flow_hol:
                    total_vertical += flow_hol * path_vertical
                    total_horizontal_2d += flow_hol * path_length_2d
                    total_horizontal_3d += flow_hol * path_length_3d

        vertical_energy_J = M_PHYS_KG * G_PHYS * total_vertical
        horizontal_energy_2d_J = horizontal_cost_weight * total_horizontal_2d
        horizontal_energy_3d_J = horizontal_cost_weight * total_horizontal_3d

        total_combined_2d = vertical_energy_J + horizontal_energy_2d_J
        total_combined_3d = vertical_energy_J + horizontal_energy_3d_J

        return total_vertical, total_horizontal_2d, total_combined_2d, total_horizontal_3d, total_combined_3d

    def compute_total_work_pair(self, cell_map: Dict[int, int],
                                od_work_df: pd.DataFrame,
                                od_hol_df: pd.DataFrame) -> Tuple[float, float]:
        """Compute working-day and holiday work with one Dijkstra pass per origin."""
        od_by_origin = {}

        for origin, dest, flow in od_work_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[0] += float(flow)

        for origin, dest, flow in od_hol_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, {}).setdefault(dest, [0.0, 0.0])[1] += float(flow)

        total_wd = 0.0
        total_hol = 0.0

        for origin_cell, dest_map in tqdm(od_by_origin.items(), desc="OD shortest paths", leave=False):
            src_node = cell_map.get(origin_cell)
            if src_node is None:
                continue
            sid = self.node2id.get(src_node)
            if sid is None:
                continue

            runner = distance.Dijkstra(self.nkG, sid, True)
            runner.run()

            for dest_cell, (flow_wd, flow_hol) in dest_map.items():
                dest_node = cell_map.get(dest_cell)
                if dest_node is None:
                    continue
                tid = self.node2id.get(dest_node)
                if tid is None:
                    continue
                path = runner.getPath(tid)
                if not path:
                    continue

                path_work = 0.0
                for a_id, b_id in zip(path[:-1], path[1:]):
                    path_work += self.arc_vertical_gain_m.get((a_id, b_id), 0.0)

                if flow_wd:
                    total_wd += flow_wd * path_work
                if flow_hol:
                    total_hol += flow_hol * path_work

        return total_wd, total_hol

    def compute_total_cost_components(self, cell_map: Dict[int, int],
                                       od_work_df: pd.DataFrame,
                                       od_hol_df: pd.DataFrame,
                                       horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT) -> Tuple[float, float, float]:
        """Convenience wrapper returning vertical, horizontal and combined cost."""
        return self.compute_total_cost_pair(
            cell_map,
            od_work_df,
            od_hol_df,
            horizontal_cost_weight=horizontal_cost_weight,
        )

    def compute_total_cost_components_all(self, cell_map: Dict[int, int],
                                          od_work_df: pd.DataFrame,
                                          od_hol_df: pd.DataFrame,
                                          horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT) -> Tuple[float, float, float, float, float]:
        """Return vertical, horizontal_2d, combined_2d, horizontal_3d, combined_3d."""
        return self.compute_total_cost_pair_all(
            cell_map,
            od_work_df,
            od_hol_df,
            horizontal_cost_weight=horizontal_cost_weight,
        )


def load_expected_graph_filenames(graphs_dir: Path) -> Set[str]:
    """
    Use fine_grid_stats.csv as the authoritative list of graph variants for the current grid.
    Only original plus on-land variants should be analyzed in step-2.
    """
    stats_csv = graphs_dir / "fine_grid_stats.csv"
    if not stats_csv.exists():
        return set()

    stats_df = pd.read_csv(stats_csv)
    if "variant" not in stats_df.columns or "on_land" not in stats_df.columns:
        return set()

    on_land = stats_df["on_land"].astype(str).str.lower().eq("true")
    variants = stats_df.loc[on_land, "variant"].astype(str).tolist()
    return {f"graph_{variant}.pkl" for variant in variants}


def list_graph_files(graphs_dir: Path) -> List[Path]:
    """List graph pickle files in deterministic order, filtered to the current stats grid when available."""
    pkl_files = sorted(graphs_dir.glob("graph_*.pkl"))
    allowed_filenames = load_expected_graph_filenames(graphs_dir)
    if allowed_filenames:
        before = len(pkl_files)
        pkl_files = [path for path in pkl_files if path.name in allowed_filenames]
        print(f"Found {len(pkl_files)} graph files in current fine_grid_stats.csv (filtered from {before})")
    else:
        print(f"Found {len(pkl_files)} graph files")
    return pkl_files


def analyze_graphs(graph_files: List[Path], dem_src, output_csv: Path, city: str,
                   resume: bool = False, save_segments: bool = False,
                   gc_every: int = 1, horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT,
                   elevation_source: str = 'node') -> pd.DataFrame:
    """
    Compute OD-weighted vertical gain, path lengths, and mobility energy for all graphs and save results to CSV.

    Returns:
        DataFrame with columns: filename, variant_type, parameter, total_work, vertical_work, horizontal_work, num_nodes, num_edges
    """
    # Determine already computed files if resuming
    existing_done = set()
    wrote_header = False
    allowed_filenames = {path.name for path in graph_files}
    new_expected_cols = {
        'filename',
        'total_work',
        'vertical_work',
        'horizontal_work',
        'vertical_gain_on_2d_path_m',
        'path_length_2d_on_2d_path_m',
        'path_length_3d_on_2d_path_m',
        'vertical_gain_on_3d_path_m',
        'path_length_2d_on_3d_path_m',
        'path_length_3d_on_3d_path_m',
        'total_energy_on_2d_path_J',
        'total_energy_on_3d_path_J',
    }
    
    if output_csv.exists():
        if not resume:
            output_csv.unlink()
            wrote_header = False
            print(f"Existing CSV removed because resume=False: {output_csv}")

        else:
            try:
                existing_df = pd.read_csv(output_csv)

                missing = new_expected_cols - set(existing_df.columns)
                if missing:
                    backup = output_csv.with_suffix('.bak.csv')
                    output_csv.replace(backup)
                    print(f"Existing CSV lacked new columns; moved to {backup}")
                    resume = False
                    wrote_header = False
                else:
                    existing_filenames = set(existing_df['filename'].astype(str).tolist())
                    stale_rows = sorted(existing_filenames - allowed_filenames)
                    if stale_rows:
                        backup = output_csv.with_suffix('.stale-grid.bak.csv')
                        output_csv.replace(backup)
                        print(
                            "Existing CSV contains variants outside the current fine_grid_stats grid; "
                            f"moved to {backup}"
                        )
                        resume = False
                        wrote_header = False
                        existing_done = set()
                    else:
                        existing_done = existing_filenames
                        wrote_header = True
                        print(f"\nResume enabled: found {len(existing_done)} results in {output_csv}")

            except Exception as e:
                print(f"Warning: could not read existing CSV {output_csv}: {e}")
                resume = False
                wrote_header = False
                
                
    to_process = [p for p in graph_files if (not resume or p.name not in existing_done)]
    print(f"\nAnalyzing {len(to_process)} graphs (skipped {len(graph_files) - len(to_process)} already done)...")
    
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

    # Build cell map using the original graph (closest nodes), loaded once.
    original_file = next((p for p in graph_files if 'original' in p.name), None)
    if original_file is None and graph_files:
        original_file = graph_files[0]
    if original_file is None:
        return pd.DataFrame()

    with open(original_file, 'rb') as f:
        G_original = pickle.load(f)
    cell_map = map_cells_to_nodes(G_original, cells_df)
    del G_original
    gc.collect()

    new_rows = []
    for idx, pkl_file in enumerate(tqdm(to_process, desc="Computing OD-weighted mobility energy"), start=1):
        filename = pkl_file.name
        try:
            with open(pkl_file, 'rb') as f:
                G = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        variant_type, parameter = parse_filename(filename)
        evaluator = ODWorkEvaluator(
            G,
            dem_src,
            ds=DS,
            keep_segments=save_segments,
            elevation_source=elevation_source,
        )
        
        length_results = evaluator.compute_total_lengths_2d_and_3d_dijkstra(
            cell_map,
            od_work,
            od_hol,
        )
        
        
        if DEBUG_LENGTHS:
            print("\n[DEBUG] OD-weighted path diagnostics")
            print(
                "  2D-path: L3D/L2D =",
                length_results["path_length_3d_on_2d_path"]
                / length_results["path_length_2d_on_2d_path"]
            )
            print(
                "  3D-path: L3D/L2D =",
                length_results["path_length_3d_on_3d_path"]
                / length_results["path_length_2d_on_3d_path"]
            )

        vertical_gain_2dpath_m = length_results["vertical_gain_on_2d_path"]
        horizontal_length_2d_on_2dpath_m = length_results["path_length_2d_on_2d_path"]
        horizontal_length_3d_on_2dpath_m = length_results["path_length_3d_on_2d_path"]

        vertical_gain_3dpath_m = length_results["vertical_gain_on_3d_path"]
        horizontal_length_2d_on_3dpath_m = length_results["path_length_2d_on_3d_path"]
        horizontal_length_3d_on_3dpath_m = length_results["path_length_3d_on_3d_path"]

        vertical_energy_2dpath_J = M_PHYS_KG * G_PHYS * vertical_gain_2dpath_m
        vertical_energy_3dpath_J = M_PHYS_KG * G_PHYS * vertical_gain_3dpath_m

        horizontal_energy_2d_on_2dpath_J = horizontal_cost_weight * horizontal_length_2d_on_2dpath_m
        horizontal_energy_3d_on_2dpath_J = horizontal_cost_weight * horizontal_length_3d_on_2dpath_m

        horizontal_energy_2d_on_3dpath_J = horizontal_cost_weight * horizontal_length_2d_on_3dpath_m
        horizontal_energy_3d_on_3dpath_J = horizontal_cost_weight * horizontal_length_3d_on_3dpath_m

        total_energy_2dpath_J = vertical_energy_2dpath_J + horizontal_energy_2d_on_2dpath_J
        total_energy_3dpath_J = vertical_energy_3dpath_J + horizontal_energy_3d_on_3dpath_J

        row = {
            "filename": filename,
            "variant_type": variant_type,
            "parameter": parameter,

            # Legacy/backward-compatible columns
            "total_work": total_energy_2dpath_J,
            "vertical_work": vertical_gain_2dpath_m,
            "horizontal_work": horizontal_length_2d_on_2dpath_m,

            # Explicit aliases
            "vertical_gain_m": vertical_gain_2dpath_m,
            "horizontal_length_2d_m": horizontal_length_2d_on_2dpath_m,
            "total_energy_J": total_energy_2dpath_J,

            # Dijkstra computed on 2D edge lengths
            "vertical_gain_on_2d_path_m": vertical_gain_2dpath_m,
            "path_length_2d_on_2d_path_m": horizontal_length_2d_on_2dpath_m,
            "path_length_3d_on_2d_path_m": horizontal_length_3d_on_2dpath_m,
            "vertical_energy_on_2d_path_J": vertical_energy_2dpath_J,
            "horizontal_energy_2d_on_2d_path_J": horizontal_energy_2d_on_2dpath_J,
            "horizontal_energy_3d_on_2d_path_J": horizontal_energy_3d_on_2dpath_J,
            "total_energy_on_2d_path_J": total_energy_2dpath_J,

            # Dijkstra computed on 3D edge lengths
            "vertical_gain_on_3d_path_m": vertical_gain_3dpath_m,
            "path_length_2d_on_3d_path_m": horizontal_length_2d_on_3dpath_m,
            "path_length_3d_on_3d_path_m": horizontal_length_3d_on_3dpath_m,
            "vertical_energy_on_3d_path_J": vertical_energy_3dpath_J,
            "horizontal_energy_2d_on_3d_path_J": horizontal_energy_2d_on_3dpath_J,
            "horizontal_energy_3d_on_3d_path_J": horizontal_energy_3d_on_3dpath_J,
            "total_energy_on_3d_path_J": total_energy_3dpath_J,

            # Short aliases
            "total_energy_2d_J": total_energy_2dpath_J,
            "total_energy_3d_J": total_energy_3dpath_J,

            # Graph metadata
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }
        new_rows.append(row)
        print(
            f"  {filename}: "
            f"VGAIN_2D={vertical_gain_2dpath_m:,.0f} m, "
            f"L2D_2D={horizontal_length_2d_on_2dpath_m:,.0f} m, "
            f"L3D_2D={horizontal_length_3d_on_2dpath_m:,.0f} m, "
            f"E_2D={total_energy_2dpath_J:,.0f} J, "
            f"VGAIN_3D={vertical_gain_3dpath_m:,.0f} m, "
            f"L2D_3D={horizontal_length_2d_on_3dpath_m:,.0f} m, "
            f"L3D_3D={horizontal_length_3d_on_3dpath_m:,.0f} m, "
            f"E_3D={total_energy_3dpath_J:,.0f} J"
        )

        if save_segments:
            # Save segment-level work to CSV only when explicitly requested.
            variant_name = filename.replace('graph_', '').replace('.pkl', '')
            seg_work_csv = output_csv.parent / f'arc_work_segments_{variant_name}.csv'
            with open(seg_work_csv, 'w', newline='') as seg_f:
                w = csv.DictWriter(seg_f, fieldnames=["u", "v", "start_x", "start_y", "end_x", "end_y", "vertical_gain_m"])
                w.writeheader()
                for (u, v), segments in (evaluator.arc_vertical_gain_segments or {}).items():
                    u_node = evaluator.id2node[u]
                    v_node = evaluator.id2node[v]
                    for seg in segments:
                        w.writerow({
                            'u': u_node,
                            'v': v_node,
                            'start_x': seg['start_coord'][0],
                            'start_y': seg['start_coord'][1],
                            'end_x': seg['end_coord'][0],
                            'end_y': seg['end_coord'][1],
                            'vertical_gain_m': seg['vertical_gain_m'],
                        })
            print(f"    Segment work saved → {seg_work_csv.name}")

        # Append row to CSV immediately
        try:
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=not wrote_header, index=False)
            wrote_header = True
        except Exception as e:
            print(f"Warning: failed to append to {output_csv}: {e}")

        # Explicit memory cleanup between variants (important for large city graphs).
        del evaluator
        del G
        if idx % max(1, int(gc_every)) == 0:
            gc.collect()
    
    # Load merged dataframe from CSV (ensures de-duplication if needed)
    try:
        merged_df = pd.read_csv(output_csv)
        if 'filename' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['filename'], keep='last')
        # Ensure numeric columns are properly typed
        for col in ['parameter', 'total_work', 'num_nodes', 'num_edges']:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        for col in ['vertical_work', 'horizontal_work']:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        # New 2D/3D columns
        for col in [
            'vertical_gain_on_2d_path_m',
            'path_length_2d_on_2d_path_m',
            'path_length_3d_on_2d_path_m',
            'vertical_gain_on_3d_path_m',
            'path_length_2d_on_3d_path_m',
            'path_length_3d_on_3d_path_m',
            'total_energy_on_2d_path_J',
            'total_energy_on_3d_path_J',
        ]:
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
    Note: total_work is a composite metric: vertical_work + lambda * horizontal_work.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Separate by transformation type
    df_rotation = df[df['variant_type'] == 'rotation'].copy()
    df_original = df[df['variant_type'] == 'original'].copy()
    df_scale_ns = df[df['variant_type'] == 'scale_ns'].copy()
    df_scale_ew = df[df['variant_type'] == 'scale_ew'].copy()

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
    
    # Create subplots: 5 plots (rotation, translation N-S, translation W-E, scale N-S, scale E-W)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    # 1. Rotation plot
    ax = axes[0]
    if len(df_rotation) > 0:
        ax.scatter(df_rotation['parameter'], df_rotation['total_energy_on_3d_path_J'], 
                  s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)
        
        # Add original point
        if len(df_original) > 0:
            ax.scatter([0], df_original['total_energy_on_3d_path_J'].values, 
                      s=200, alpha=0.9, c='red', marker='*', 
                      edgecolors='darkred', linewidth=2, 
                      label='Original', zorder=10)
        
        # Fit parabola only if R² > 0.95
        if len(df_rotation) >= 3 and 1==0:  # Disabled for rotations
            coeffs = np.polyfit(df_rotation['parameter'], df_rotation['total_energy_on_3d_path_J'], 2)
            x_fit = np.linspace(df_rotation['parameter'].min(), 
                               df_rotation['parameter'].max(), 100)
            y_fit = np.polyval(coeffs, x_fit)
            
            # Calculate R²
            y_mean = df_rotation['total_energy_on_3d_path_J'].mean()
            ss_tot = np.sum((df_rotation['total_energy_on_3d_path_J'] - y_mean)**2)
            ss_res = np.sum((df_rotation['total_energy_on_3d_path_J'] - np.polyval(coeffs, df_rotation['parameter']))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            if r2 > 0.95:
                ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.6, 
                       label=f'Parabolic fit (R²={r2:.3f})')
                
                # Find vertex
                vertex_x = -coeffs[1] / (2 * coeffs[0])
                ax.axvline(vertex_x, color='green', linestyle=':', linewidth=1.5, 
                          label=f'Vertex: {vertex_x:.1f}°')
    
    ax.set_xlabel('Rotation Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$W_{TOT}$', fontsize=12, fontweight='bold')
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
                ax.scatter(-df_subset['parameter'], df_subset['total_energy_on_3d_path_J'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'South ({angle}°)')
            else:  # North direction (90°)
                ax.scatter(df_subset['parameter'], df_subset['total_energy_on_3d_path_J'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'North ({angle}°)')
    
    # Add original point
    if len(df_original) > 0:
        ax.scatter([0], df_original['total_energy_on_3d_path_J'].values, 
                  s=200, alpha=0.9, c='red', marker='*', 
                  edgecolors='darkred', linewidth=2, 
                  label='Original', zorder=10)
    
    ax.set_xlabel('Translation Distance (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$W_{TOT}$', fontsize=12, fontweight='bold')
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
                ax.scatter(-df_subset['parameter'], df_subset['total_energy_on_3d_path_J'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'West ({angle}°)')
            else:  # East direction (0°)
                ax.scatter(df_subset['parameter'], df_subset['total_energy_on_3d_path_J'], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'East ({angle}°)')
    
    # Add original point
    if len(df_original) > 0:
        ax.scatter([0], df_original['total_energy_on_3d_path_J'].values, 
                  s=200, alpha=0.9, c='red', marker='*', 
                  edgecolors='darkred', linewidth=2, 
                  label='Original', zorder=10)
    
    ax.set_xlabel('Translation Distance (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$W_{TOT}$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scale North-South plot
    ax = axes[3]
    if len(df_scale_ns) > 0:
        ax.scatter(df_scale_ns['parameter'], df_scale_ns['total_energy_on_3d_path_J'],
                   s=100, alpha=0.7, c='#2ca02c', edgecolors='black', linewidth=1.5,
                   label='Scale NS')
    if len(df_original) > 0:
        ax.scatter([1.0], df_original['total_energy_on_3d_path_J'].values,
                   s=200, alpha=0.9, c='red', marker='*',
                   edgecolors='darkred', linewidth=2,
                   label='Original', zorder=10)
    ax.set_xlabel('NS Scale Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$W_{TOT}$', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Scale East-West plot
    ax = axes[4]
    if len(df_scale_ew) > 0:
        ax.scatter(df_scale_ew['parameter'], df_scale_ew['total_energy_on_3d_path_J'],
                   s=100, alpha=0.7, c='#ff7f0e', edgecolors='black', linewidth=1.5,
                   label='Scale EW')
    if len(df_original) > 0:
        ax.scatter([1.0], df_original['total_energy_on_3d_path_J'].values,
                   s=200, alpha=0.9, c='red', marker='*',
                   edgecolors='darkred', linewidth=2,
                   label='Original', zorder=10)
    ax.set_xlabel('EW Scale Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$W_{TOT}$', fontsize=12, fontweight='bold')
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
    print("MOBILITY ENERGY SUMMARY")
    print("="*60)
    
    # Original
    df_orig = df[df['variant_type'] == 'original']
    if len(df_orig) > 0:
        orig_work = df_orig['total_energy_on_3d_path_J'].values[0]
        print(f"\nOriginal total energy (vertical + horizontal weighted): {orig_work:,.0f}")

    if {'vertical_energy_on_3d_path_J', 'horizontal_energy_3d_on_3d_path_J'}.issubset(df.columns):
        print("\nComponent columns available: vertical_energy_on_3d_path_J, horizontal_energy_3d_on_3d_path_J")
        if len(df_orig) > 0:
            print(f"Original vertical_energy_on_3d_path_J: {df_orig['vertical_energy_on_3d_path_J'].values[0]:,.0f}")
            print(f"Original horizontal_energy_3d_on_3d_path_J: {df_orig['horizontal_energy_3d_on_3d_path_J'].values[0]:,.0f}")
    
    # By type
    for variant_type in df['variant_type'].unique():
        if variant_type == 'original':
            continue
        
        df_var = df[df['variant_type'] == variant_type]
        print(f"\n{variant_type}:")
        print(f"  Count: {len(df_var)}")
        print(f"  Min energy: {df_var['total_energy_on_3d_path_J'].min():,.0f} J at {df_var.loc[df_var['total_energy_on_3d_path_J'].idxmin(), 'parameter']}")
        print(f"  Max energy: {df_var['total_energy_on_3d_path_J'].max():,.0f} J at {df_var.loc[df_var['total_energy_on_3d_path_J'].idxmax(), 'parameter']}")
        print(f"  Mean energy: {df_var['total_energy_on_3d_path_J'].mean():,.0f} J")
        
        if len(df_orig) > 0:
            pct_change = ((df_var['total_energy_on_3d_path_J'] - orig_work) / orig_work * 100)
            print(f"  Change from original: {pct_change.min():.1f}% to {pct_change.max():.1f}%")

    if {'vertical_energy_on_3d_path_J', 'horizontal_energy_3d_on_3d_path_J'}.issubset(df.columns):
        print("\nComponent means by variant:")
        for variant_type in df['variant_type'].unique():
            if variant_type == 'original':
                continue
            df_var = df[df['variant_type'] == variant_type]
            print(f"  {variant_type}: mean vertical={df_var['vertical_energy_on_3d_path_J'].mean():,.0f}, mean horizontal={df_var['horizontal_energy_3d_on_3d_path_J'].mean():,.0f}")


def main(cities: List[str], resume: bool = False, save_segments: bool = False, low_memory: bool = False,
         horizontal_cost_weight: float = HORIZONTAL_COST_WEIGHT,
         elevation_source: str = 'node',
         data_root: Path = DEFAULT_DATA_ROOT,
         polygon_source: str = 'fua'):
    """Main execution for one or multiple cities."""
    if not cities:
        print("Error: No cities specified")
        sys.exit(1)

    # Directory suffix matches the one used by dem_extractor_fine_grid.py
    _sfx = "" if polygon_source == "fua" else f"_{polygon_source}"

    for city in cities:
        print("\n" + "="*60)
        print(f"FINE GRID MOBILITY ENERGY ANALYSIS: {city.upper()}  (source={polygon_source})")
        print("="*60)

        base_dir = Path(data_root) / city
        graphs_dir = base_dir / f"graphs_fine_grid{_sfx}"
        output_dir = base_dir / f"graphs_fine_grid{_sfx}"
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
            continue
        
        print(f"\nGraphs directory: {graphs_dir}")
        print(f"Output directory: {output_dir}")
        
        # Load DEM
        dem_src = load_dem_reader(dem_file)
        
        # List graph files and stream them one by one to keep RAM bounded.
        graph_files = list_graph_files(graphs_dir)
        
        if len(graph_files) == 0:
            print(f"No graphs found for {city}")
            continue
        
        # Output CSV path
        output_csv = output_dir / "fine_grid_gravitational_work.csv"
        
        # Analyze graphs with resume support
        df = analyze_graphs(
            graph_files,
            dem_src,
            output_csv=output_csv,
            city=city,
            resume=resume,
            save_segments=save_segments,
            gc_every=1 if low_memory else 5,
            horizontal_cost_weight=horizontal_cost_weight,
            elevation_source=elevation_source,
        )
        
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
        
        print("✓ Completed for " + city)
    
    print("\n" + "="*60)
    print("✓ ALL ANALYSES COMPLETE")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute gravitational work for fine-grid graph variants')
    parser.add_argument('--city', type=str, default=None, help='Single city name (for backward compatibility)')
    parser.add_argument('--cities', type=str, nargs='+', default=None, help='Multiple city names to process')
    parser.add_argument('--resume', action='store_true', help='Skip already computed graphs and append new results')
    parser.add_argument('--save-segments', action='store_true',
                        help='Export per-segment work CSV for each graph (RAM and disk intensive)')
    parser.add_argument('--low-memory', action='store_true',
                        help='Force extra memory cleanup between graphs')
    parser.add_argument('--horizontal-cost-weight', type=float, default=HORIZONTAL_COST_WEIGHT,
                        help=f'Weight for horizontal travel cost in the composite score (default: lambda_mean={HORIZONTAL_COST_WEIGHT:.1f} J/m)')
    parser.add_argument('--elevation-source', choices=['node', 'dem'], default='node',
                        help='Source for vertical work on each edge: node (use z already assigned in step-1) or dem (sample raster along edge geometry)')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT,
                        help=f'Root directory containing the city folders (default: {DEFAULT_DATA_ROOT})')
    parser.add_argument('--polygon-source', type=str, choices=['fua', 'osm'], default='fua',
                        help='Must match the --polygon-source used in step-1: '
                             '"fua" reads from graphs_fine_grid/, "osm" reads from graphs_fine_grid_osm/')
    cli_args = parser.parse_args()

    # Support both --city (single) and --cities (multiple) for compatibility
    cities_to_process = cli_args.cities if cli_args.cities else ([cli_args.city] if cli_args.city else ["santiago"])
    main(
        cities=cities_to_process,
        resume=cli_args.resume,
        save_segments=cli_args.save_segments,
        low_memory=cli_args.low_memory,
        horizontal_cost_weight=cli_args.horizontal_cost_weight,
        elevation_source=cli_args.elevation_source,
        data_root=cli_args.data_root,
        polygon_source=cli_args.polygon_source,
    )
