#!/usr/bin/env python3
"""
author: Federico Bellisardi
execution: python wheight.py -c tools/conf/conf_wheight.json
"""

import os
import json
import argparse
import pickle
import logging
import psutil

import requests
import pandas as pd
import numpy as np

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
        total = 0.0
        logger.info("Computing total work for %d OD flows", len(od_df))

        od_by_origin = {}
        for origin, dest, flow in od_df.itertuples(index=False, name=None):
            od_by_origin.setdefault(origin, []).append((dest, flow))

        for origin_cell, dest_list in od_by_origin.items():
            src_nodes = cell_map.get(origin_cell, [])
            if not src_nodes:
                continue

            src_ids = [ self.node2id[u] for u in src_nodes ]

            for sid in src_ids:
                # logger.info("Running Dijkstra from source node %d (cell %s)",
                #              sid, origin_cell)
                runner = distance.Dijkstra(self.nkG, sid, True)
                runner.run()

                for dest_cell, flow in dest_list:
                    dst_nodes = cell_map.get(dest_cell, [])
                    if not dst_nodes:
                        continue

                    weight = flow / (len(src_nodes) * len(dst_nodes))
                    for v in dst_nodes:
                        tid = self.node2id[v]
                        path = runner.getPath(tid)
                        if not path:
                            continue
                        for a_id, b_id in zip(path[:-1], path[1:]):
                            total += weight * self.arc_work.get((a_id, b_id), 0.0)

                del runner

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
            logger.info(f"[{var}] collapsing MultiGraph → simple {type(G).__name__}")
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
        logger.info(f"[{var}] edges after collapse: {total_edges}, with 'length': {with_len}")

        if G.is_directed():
            if nx.is_weakly_connected(G):
                logger.info(f"[{var}] graph is weakly connected")
            else:
                n_comp = nx.number_weakly_connected_components(G)
                logger.warning(f"[{var}] NOT weakly connected: {n_comp} components")
        else:
            if nx.is_connected(G):
                logger.info(f"[{var}] graph is connected")
            else:
                n_comp = nx.number_connected_components(G)
                logger.warning(f"[{var}] NOT connected: {n_comp} components")

        graphs[var] = G
    logger.info(f"Loaded {len(graphs)} graph variants")
    return graphs

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
    logger.info(f"  Loaded {len(df)} positive flows")
    return df

def map_cells_to_nodes(G, cell_df):
    xs = []
    ys = []
    for _, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            xs.append(data['x'])
            ys.append(data['y'])
        elif 'lon' in data and 'lat' in data:
            xs.append(data['lon'])
            ys.append(data['lat'])
        else:
            raise KeyError("Nodes in the graph don't have 'x'/'y' nor 'lon'/'lat'")
    xs = np.array(xs)
    ys = np.array(ys)

    nodes = np.array([n for n, _ in G.nodes(data=True)])
    mapping = {}
    total = 0
    for _, row in cell_df.iterrows():
        mask = (
            (xs >= row.lon_min) & (xs <= row.lon_max) &
            (ys >= row.lat_min) & (ys <= row.lat_max)
        )
        lst = nodes[mask].tolist()
        mapping[row.cell] = lst
        total += len(lst)

    logger.info(f"Mapped {len(mapping)} cells → {total} total nodes")
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


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--conf', default='conf_wheight.json')
    args = p.parse_args()
    conf = json.load(open(args.conf))

    bbox       = conf['bbox']
    graphs_dir = conf['graphs_dir']
    cells_file = conf['cells_file']
    od_w_file  = conf['od_work_file']
    od_h_file  = conf['od_holiday_file']
    cells_crs  = conf['cells_crs']
    api_key    = conf['api_key']
    dem_file   = conf['dem_file']
    ds         = conf.get('ds', 10.0)
    m, g       = conf.get('m',1.0), conf.get('g',1.0)

    # graphs
    graphs = load_graphs(graphs_dir)
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
    out_csv = os.path.join(graphs_dir, 'gravitational_work_by_variant.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(graphs_dir,'gravitational_work_by_variant.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    # explanatory plots
    work_png = os.path.join(graphs_dir, 'work_by_variant.png')
    plot_work_by_variant(df, work_png)

    diff_png = os.path.join(graphs_dir, 'variant_differences.png')
    plot_variant_differences(df, diff_png)

    dem.close()


if __name__ == '__main__':
    main()
