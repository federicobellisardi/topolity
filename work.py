#!/usr/bin/env python
"""
author: Federico Bellisardi
Description: Generate a static heatmap (PNG) of movement trajectories weighted by gravitational work.
"""

import os
import ast
import json
import logging
import osmnx as ox
import networkx as nx
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import rasterio
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import LogNorm


from utils import read_conf, haversine

class Work:
    def __init__(self, movements, od_matrix, G, conf, dem):
        self.movements = movements
        self.od_matrix = od_matrix
        self.config = conf
        self.G = G
        self.dem = dem

    def integrator(h1, h2, dh):
        total = 0.0
        h = h1
        while h < h2:
            step = dh if h + dh <= h2 else (h2 - h)
            total += step
            h += step
        return total

    def gravitational_work(h1, h2, m, g, dh):
        if h2 <= h1:
            return 0.0
        delta_h = Work.integrator(h1, h2, dh)
        return m * g * delta_h        

    @staticmethod
    def compute_total_work(polyline, m=1, g=1):
        total = 0.0
        for i in range(len(polyline) - 1):
            _, _, h1 = polyline[i]
            _, _, h2 = polyline[i + 1]
            total += Work.gravitational_work(h1, h2, m, g, dh=0.1)
        return total

    def compute_edge_work(self, od_matrix, movement_df):
        edge_flow = defaultdict(float)
        edge_work = defaultdict(float)

        for _, row in movement_df.iterrows():
            path = row['path']
            polyline = row['polyline_with_alt']
            
            if isinstance(path, str):
                try: path = ast.literal_eval(path)
                except Exception: continue

            if isinstance(polyline, str):
                try: polyline = ast.literal_eval(polyline)
                except Exception: continue

            if len(path) < 2 or len(polyline) < 2: continue

            origin = path[0]
            dest = path[-1]
            try:
                flow = float(od_matrix.at[origin, dest])
            except KeyError:
                flow = 0.0

            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                _, _, h1 = polyline[i]
                _, _, h2 = polyline[i + 1]

                edge_flow[(u, v)] += flow
                work = Work.gravitational_work(h1, h2, m=1, g=1, dh=0.1)
                edge_work[(u, v)] += flow * work


        unique_nodes = set([u for u, _ in edge_flow] + [v for _, v in edge_flow])
        node_coords = {node: (data["x"], data["y"]) for node, data in self.G.nodes(data=True) if node in unique_nodes}

        edge_df = pd.DataFrame([
            {
                "u": u,
                "v": v,
                "flow": edge_flow[(u, v)],
                "work": edge_work[(u, v)],
                "coordinates": (
                    list(self.G[u][v][0]['geometry'].coords)
                    if (u in self.G and v in self.G[u] and 'geometry' in self.G[u][v][0])
                    else [node_coords.get(u), node_coords.get(v)]
                )
            }
            for (u, v) in edge_flow
            if u in node_coords and v in node_coords
        ])

        return edge_df

class WorkPlot():
    def __init__(self, df, G, conf, dem, output):
        self.df = df
        self.conf = conf
        self.dem = dem
        self.output = output
        self.G = G

    def heatmap(self, edge_work_df, tag):
        bbox_dic = self.conf.get("bbox", {}).get(tag, {})
        if not bbox_dic:
            raise ValueError("Bounding box info missing in config.")
        bbox = (bbox_dic["min_lon"], bbox_dic["min_lat"],
                bbox_dic["max_lon"], bbox_dic["max_lat"])

        bbox_geom = [box(*bbox)]
        with rasterio.open(self.dem) as dem:
            if dem.crs.to_string() != 'EPSG:4326':
                from rasterio.warp import transform_bounds
                bbox = transform_bounds('EPSG:4326', dem.crs, *bbox)
                bbox_geom = [box(*bbox)]
            dem_data, dem_transform = mask(dem, bbox_geom, crop=True)
            dem_data = dem_data[0]
            nrows, ncols = dem_data.shape
            extent = [
                dem_transform[2],
                dem_transform[2] + dem_transform[0] * ncols,
                dem_transform[5] + dem_transform[4] * nrows,
                dem_transform[5]
            ]

        work_values = edge_work_df["work"]
        total_work = work_values.sum()
        sci_str = f"{total_work:.2e}"
        mantissa, exp_str = sci_str.split("e")
        exponent = int(exp_str) 

        norm = LogNorm(vmin=work_values[work_values > 0].min(), vmax=work_values.max())
        # norm = Normalize(vmin=edge_work_df["work"].min(), vmax=edge_work_df["work"].max())
        cmap = plt.get_cmap("plasma")

        fig, ax = plt.subplots(figsize=(12, 12))

        dem_cmap = plt.cm.Greys_r
        im = ax.imshow(dem_data, cmap=dem_cmap, extent=extent, alpha=0.35, zorder=1)
        cbar_dem = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, location='right')
        cbar_dem.set_label("DEM Altitude (m)", fontsize=10)

        pos = nx.get_node_attributes(self.G, 'pos')
        if pos:
            nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color='black', style='--', alpha=0.7, width=1, zorder=2)
        else:
            for u, v, data in self.G.edges(data=True):
                if "coordinates" in data:
                    coords = data["coordinates"]
                    try:
                        xs, ys = zip(*coords)
                        ax.plot(xs, ys, color="black", linestyle="--", alpha=0.7, linewidth=1, zorder=2)
                    except Exception:
                        continue

        for _, row in edge_work_df.iterrows():
            coords = row["coordinates"]
            work = row["work"]
            try:
                xs, ys = zip(*coords) 
                ax.plot(xs, ys, color=cmap(norm(work)), linewidth=2, zorder=3)
            except Exception:
                continue

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_work = fig.colorbar(sm, ax=ax, label="Gravitational Work (J)", 
                                fraction=0.03, pad=0.1, location='left')

        ax.set_title(rf"Total work in {tag.capitalize()}: {mantissa} $\times 10^{{{exponent}}}$ J", fontsize=14, pad=12)
        ax.set_xlabel("Longitude (°)", fontsize=11)
        ax.set_ylabel("Latitude (°)", fontsize=11)
        ax.tick_params(labelsize=10)
        plt.tight_layout()

        plt.savefig(self.output, dpi=300, bbox_inches="tight")
        print(f"Static heatmap saved to {self.output}")


def main():
    parser = argparse.ArgumentParser(description="Generate static heatmap of movement trajectories by gravitational work.")
    parser.add_argument("-m", "--movements", required=True, help="CSV file with polyline_with_alt column")
    parser.add_argument("-od", "--od_matrix", required=True, help="CSV file with od")
    parser.add_argument("-c", "--config", required=True, help="Path to configuration JSON with bbox info")
    parser.add_argument("-d", "--dem", required=True, help="Path to DEM .tif file")
    parser.add_argument("-o", "--output", default="work_heatmap.png", help="Output PNG file")
    args = parser.parse_args()

    conf = read_conf(args.config)
    conf = conf.get("data_processing", {}).get("data_source", {}).get("worldpop", {})
    tag = conf.get("tag", "default")

    G = ox.load_graphml('/home/fede/code/topolity/output/tag_bo/graph/bo_graph.gpickle')

    df = pd.read_csv(args.movements, sep=";")
    if 'polyline_with_alt' not in df.columns: raise ValueError("Missing 'polyline_with_alt' column. Run enrichment first.")
    df['polyline_with_alt'] = df['polyline_with_alt'].apply(ast.literal_eval)

    od_matrix = pd.read_csv(args.od_matrix, sep=";", index_col=0)
    od_matrix.columns = od_matrix.columns.astype(int)
    od_matrix.index = od_matrix.index.astype(int)
    od_matrix = 100000000*od_matrix

    work = Work(df, od_matrix, G, conf, args.dem)
    edge_work_df = work.compute_edge_work(od_matrix, df)

    plot = WorkPlot(edge_work_df, G, conf, "/home/fede/code/topolity/data/dem/bo_dem.tif", "test.png")
    plot.heatmap(edge_work_df, tag)   


if __name__ == "__main__":
    main()
