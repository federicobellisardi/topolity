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
import branca.colormap as bcm

import multiprocessing
import time


p = psutil.Process(os.getpid())
p.cpu_percent(None)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


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
        self.node2id = {n:i for i,n in enumerate(G.nodes())}
        self.id2node = {i:n for n,i in self.node2id.items()}

        logger.info("Precomputing edge-work for all edges")
        self.arc_work = {}
        self.arc_work_segments = {}
        for u, v, data in G.edges(data=True):
            total_w, segments = self._compute_edge_work(u, v)
            self.arc_work[(self.node2id[u], self.node2id[v])] = total_w
            self.arc_work_segments[(self.node2id[u], self.node2id[v])] = segments

        logger.info("Converting NetworkX graph to NetworKit graph")

        logger.info("Precomputing edge-work for all edges")
        arc_work_orig = {}
        for u, v, data in G.edges(data=True):
            total_w, segments = self._compute_edge_work(u, v)
            arc_work_orig[(u, v)] = total_w


        n = len(self.node2id)
        nkG = nk.Graph(n, weighted=True, directed=G.is_directed())
        for (u, v), w in arc_work_orig.items():
            u_id, v_id = self.node2id[u], self.node2id[v]
            length = G[u][v].get('length', 1.0)
            nkG.addEdge(u_id, v_id, length)

        self.nkG = nkG

        self.arc_work = {
            (self.node2id[u], self.node2id[v]): w
            for (u, v), w in arc_work_orig.items()
        }


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
        segment_results = []
        w = 0.0
        for i, (h1, h2) in enumerate(zip(elevs, elevs[1:])):
            if h2 > h1:
                seg_work = self.m * self.g * (h2 - h1)
                w += seg_work
            else:
                seg_work = 0.0
            segment_results.append({
                'start_coord': coords[i],
                'end_coord': coords[i+1],
                'work': seg_work
            })
        return w, segment_results

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
                pass
            else:
                n_comp = nx.number_weakly_connected_components(G)
                logger.warning(f"[{var}] NOT weakly connected: {n_comp} components")
        else:
            if nx.is_connected(G):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--conf', default='conf_wheight.json')
    p.add_argument('-city', '--city', help='City name to override conf file')
    p.add_argument('-p', '--plot', action='store_true', help='Run plotting script at the end')

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


    graphs, stats = load_graphs(graphs_dir)
    G0     = graphs['original']

    dem = DEMReader(dem_file)
    dem.ensure_dem(api_key, {
        "min_lon": bbox[0][1], "min_lat": bbox[0][0],
        "max_lon": bbox[2][1], "max_lat": bbox[2][0]
    })
    dem.open()


    cells   = load_cells(cells_file, cells_crs, bbox)
    od_work = load_od(od_w_file)
    od_hol  = load_od(od_h_file)
    valid   = set(cells.cell)
    od_work = od_work[od_work.origin.isin(valid)&od_work.dest.isin(valid)]
    od_hol  = od_hol[od_hol.origin.isin(valid)&od_hol.dest.isin(valid)]
    logger.info(f"Filtered OD: work={len(od_work)}, hol={len(od_hol)}")
    cell_map = map_cells_to_nodes(G0, cells)


    results = []
    for var, G in graphs.items():
        logger.info(f"=== Variant '{var}' ===")
        log_resources(f"start variant {var}")
        ev = WorkEvaluatorNK(G, dem, ds=ds, m=m, g=g)
        wd = ev.compute_total_work(cell_map, od_work)
        ho = ev.compute_total_work(cell_map, od_hol)
        results.append({'variant':var,'work_wd':wd,'work_hol':ho})

        rows = []
        for (u, v), segments in ev.arc_work_segments.items():
            for seg in segments:
                rows.append({
                    'u': ev.id2node[u],
                    'v': ev.id2node[v],
                    'start_x': seg['start_coord'][0],
                    'start_y': seg['start_coord'][1],
                    'end_x': seg['end_coord'][0],
                    'end_y': seg['end_coord'][1],
                    'segment_work': seg['work']
                })
        seg_work_df = pd.DataFrame(rows)
        seg_work_csv = os.path.join(graphs_dir, f'arc_work_segments_{var}.csv')
        seg_work_df.to_csv(seg_work_csv, index=False)
        logger.info(f"Arc work per segment saved → {seg_work_csv}")        
        log_resources(f"end variant {var}")


    df = pd.DataFrame(results)
    out_csv = os.path.join(graphs_dir, 'gravitational_work_by_variant.csv')
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved → {out_csv}")

    if args.plot:
        logger.info("Generating plots using external plotting script...")
        plot_script = os.path.join(os.path.dirname(__file__), 'plot_gravitational_work.py')
        plot_cmd = [sys.executable, plot_script, '-c', args.conf]
        if args.city:
            plot_cmd.extend(['-city', args.city])
        
        import subprocess
        try:
            result = subprocess.run(plot_cmd, capture_output=True, text=True, check=True)
            logger.info("Plot generation completed successfully")
            if result.stdout:
                logger.info(f"Plot script output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Plot generation failed: {e}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
    else:
        logger.info("Skipping plot generation as --plot flag not set.")
        logger.info("To generate plots, run 'plot_gravitational_work.py' separately.")

    dem.close()


if __name__ == '__main__':
    main()