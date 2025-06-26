#!/usr/bin/env python3
"""
author: Federico Bellisardi
"""

import os
import argparse
import json
import pickle

import requests
import pandas as pd
import numpy as np
import networkx as nx

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import box, Polygon, LineString, Point
from scipy.spatial import cKDTree

# ----------------------------------
# DEMReader
# ----------------------------------
class DEMReader:
    def __init__(self, dem_file, search_radius=50):
        self.dem_file = dem_file
        self.search_radius = search_radius

    def download_dem(self, api_key, bbox_dict, dem_file=None):
        dem_file = dem_file or self.dem_file
        min_lon = bbox_dict["min_lon"]
        min_lat = bbox_dict["min_lat"]
        max_lon = bbox_dict["max_lon"]
        max_lat = bbox_dict["max_lat"]
        w = max_lon - min_lon
        h = max_lat - min_lat
        south = min_lat - h
        north = max_lat + h
        west  = min_lon - w
        east  = max_lon + w

        url = (
            "https://portal.opentopography.org/API/globaldem?"
            f"demtype=SRTMGL3&south={south}&north={north}"
            f"&west={west}&east={east}"
            f"&outputFormat=GTiff&API_Key={api_key}"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"DEM download failed: {resp.status_code} {resp.text}")

        with open(dem_file, 'wb') as f:
            f.write(resp.content)
        return dem_file

    def get_pixel_centroids(self, bbox=None):
        with rasterio.open(self.dem_file) as src:
            if bbox is not None:
                geom = [box(*bbox)]
                img, transf = mask(src, geom, crop=True)
                elev = img[0]
            else:
                elev = src.read(1, masked=True)
                transf = src.transform

            rows, cols = elev.shape
            lats, lons, alts, pts = [], [], [], []
            for i in range(rows):
                for j in range(cols):
                    val = elev[i, j]
                    if np.ma.is_masked(val):
                        continue
                    lon, lat = rasterio.transform.xy(transf, i, j, offset='center')
                    lons.append(lon)
                    lats.append(lat)
                    alts.append(float(val))
                    pts.append(Point(lon, lat))

        gdf = gpd.GeoDataFrame({
            'lat': lats, 'lon': lons, 'alt': alts, 'geometry': pts
        }, crs=src.crs)
        return gdf

# ----------------------------------
# WorkEvaluator (integrator along each edge)
# ----------------------------------
class WorkEvaluator:
    def __init__(self, G, dem_reader, dh=0.1, m=1.0, g=1.0):
        dem_gdf = dem_reader.get_pixel_centroids()
        coords  = np.vstack((dem_gdf.geometry.y.values,
                             dem_gdf.geometry.x.values)).T
        self.dem_tree = cKDTree(coords)
        self.dem_alts = dem_gdf['alt'].values

        self.G  = G
        self.dh = dh
        self.m  = m
        self.g  = g

    @staticmethod
    def integrator(h1, h2, dh):
        total, h = 0.0, h1
        while h < h2:
            step = dh if h + dh <= h2 else (h2 - h)
            total += step
            h += step
        return total

    @staticmethod
    def gravitational_work(h1, h2, m, g, dh):
        if h2 <= h1:
            return 0.0
        delta_h = WorkEvaluator.integrator(h1, h2, dh)
        return m * g * delta_h

    def edge_work(self, u, v):
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

        n_samp = max(int(geom.length / self.dh), 2)
        dists  = np.linspace(0, geom.length, n_samp)
        pts    = [geom.interpolate(d) for d in dists]

        coords = np.vstack([(pt.y, pt.x) for pt in pts])
        _, idxs = self.dem_tree.query(coords)
        elevs   = self.dem_alts[idxs]

        diffs = np.diff(elevs)
        pos   = diffs[diffs > 0].sum()
        return self.m * self.g * pos

    def compute_total_work(self, cell_map, od_list):
        total = 0.0
        for origin, dest, flow in od_list.itertuples(index=False, name=None):
            src = cell_map.get(origin, [])
            dst = cell_map.get(dest,   [])
            if not src or not dst:
                continue
            weight = flow / (len(src) * len(dst))
            for u in src:
                for v in dst:
                    try:
                        path = nx.shortest_path(self.G, source=u, target=v,
                                                weight='length')
                    except nx.NetworkXNoPath:
                        continue
                    for a, b in zip(path[:-1], path[1:]):
                        total += weight * self.edge_work(a, b)
        return total

# ----------------------------------
# helper functions
# ----------------------------------
def load_graphs(graphs_dir, stats_df):
    graphs = {}
    for _, r in stats_df.iterrows():
        var = r['variant']
        fn  = 'graph_original.pkl' if var=='original' else f'graph_{var}.pkl'
        p   = os.path.join(graphs_dir, fn)
        if not os.path.exists(p):
            print(f"Warning: missing pickle for {var}: {p}")
            continue
        with open(p,'rb') as f:
            graphs[var] = pickle.load(f)
    return graphs

def load_cell_coordinates(path, cells_crs, bbox):
    poly = Polygon([(lon, lat) for lat, lon in bbox])
    df   = pd.read_csv(path)
    req  = {'cell_id','x_min','y_min','x_max','y_max'}
    if not req.issubset(df.columns):
        raise ValueError(f"cell_coordinates.csv must have: {req}")

    transf = Transformer.from_crs(cells_crs, "EPSG:4326", always_xy=True)
    df[['lon_min','lat_min']] = df.apply(
        lambda r: transf.transform(r.x_min, r.y_min),
        axis=1, result_type='expand'
    )
    df[['lon_max','lat_max']] = df.apply(
        lambda r: transf.transform(r.x_max, r.y_max),
        axis=1, result_type='expand'
    )
    df = df.rename(columns={'cell_id':'cell'})
    df['cell_box'] = df.apply(lambda r: box(
        r.lon_min, r.lat_min, r.lon_max, r.lat_max
    ), axis=1)
    df = df[df['cell_box'].apply(lambda b: b.intersects(poly))].copy()
    return df[['cell','lon_min','lat_min','lon_max','lat_max']]

def load_od_list(path):
    df = pd.read_csv(path).rename(columns={
        'cell_origin':'origin','cell_destination':'dest','count':'flow'
    })
    df['origin'] = df.origin.astype(str)
    df['dest']   = df.dest.astype(str)
    df['flow']   = pd.to_numeric(df.flow, errors='coerce').fillna(0.0)
    return df[df.flow>0].reset_index(drop=True)

def map_cells_to_nodes(G, cell_df):
    nodes = np.array([n for n,_ in G.nodes(data=True)])
    lons  = np.array([d['x'] for _,d in G.nodes(data=True)])
    lats  = np.array([d['y'] for _,d in G.nodes(data=True)])
    mapping = {}
    for _,r in cell_df.iterrows():
        mask = ((lons>=r.lon_min)&(lons<=r.lon_max)&
                (lats>=r.lat_min)&(lats<=r.lat_max))
        mapping[r.cell] = nodes[mask].tolist()
    return mapping

def parse_config():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--conf',default='/home/fbellisardi/code/topolity/tools/conf/conf_wheight.json')
    args = p.parse_args()
    conf = json.load(open(args.conf))
    needed = ['bbox','graphs_dir','cells_file',
              'od_work_file','od_holiday_file',
              'cells_crs','api_key','dem_file']
    miss   = [k for k in needed if k not in conf]
    if miss:
        p.error(f"Config missing: {miss}")
    return conf

# ----------------------------------
# main
# ----------------------------------
def main():
    conf = parse_config()

    stats = pd.read_csv(os.path.join(conf['graphs_dir'],'graph_stats.csv'))
    graphs= load_graphs(conf['graphs_dir'], stats)

    demr  = DEMReader(conf['dem_file'])
    bboxd = {
        "min_lon": conf['bbox'][0][1],
        "min_lat": conf['bbox'][0][0],
        "max_lon": conf['bbox'][2][1],
        "max_lat": conf['bbox'][2][0]
    }
    demr.download_dem(conf['api_key'], bboxd, dem_file=conf['dem_file'])

    cells = load_cell_coordinates(conf['cells_file'], conf['cells_crs'], conf['bbox'])
    odw   = load_od_list(conf['od_work_file'])
    odh   = load_od_list(conf['od_holiday_file'])
    valid = set(cells.cell)
    odw   = odw[odw.origin.isin(valid)&odw.dest.isin(valid)]
    odh   = odh[odh.origin.isin(valid)&odh.dest.isin(valid)]

    G0    = graphs['original']
    cmap  = map_cells_to_nodes(G0, cells)

    ev = WorkEvaluator(G0, demr, dh=0.1, m=1, g=1)

    results = []
    for var, Gvar in graphs.items():
        print(f"Processing {var}…")
        ev.G = Gvar
        wwd  = ev.compute_total_work(cmap, odw)
        whol = ev.compute_total_work(cmap, odh)
        results.append({'variant':var,'work_wd':wwd,'work_hol':whol})
        print(f" → WD {wwd:.2f}, HOL {whol:.2f}")

    df = pd.DataFrame(results)
    out = os.path.join(conf['graphs_dir'],'gravitational_work_by_variant.csv')
    df.to_csv(out,index=False)

    orig = df[df.variant=='original'].iloc[0]
    pert = df[df.variant!='original']
    p_wd = ((pert.work_wd - orig.work_wd)/orig.work_wd*100).mean()
    p_hol= ((pert.work_hol-orig.work_hol)/orig.work_hol*100).mean()

    print(f"\nOriginal → WD={orig.work_wd:.1f}, HOL={orig.work_hol:.1f}")
    print(f"Perturbed ↑WD {p_wd:.1f}%, ↑HOL {p_hol:.1f}%")

if __name__=='__main__':
    main()
