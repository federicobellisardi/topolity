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
import rasterio
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

from utils import read_conf, haversine, integrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def gravitational_work(h1, h2, m=1, g=1, dh=0.1):
    """
    Computes the gravitational work required to go from h1 to h2 with mass m and gravity g.
    Uses step-wise integration if h2 > h1.
    """
    if h2 <= h1:
        return 0.0
    delta_h = integrator(h1, h2, dh)
    return m * g * delta_h


def friction_work(lat1, lon1, lat2, lon2, m=1, g=1, mu=0.1):
    """
    Computes the work against friction from point A to B.
    """
    d = haversine(lat1, lon1, lat2, lon2)
    return mu * m * g * d


def compute_total_work(polyline, m=1, g=1):
    total = 0.0
    for i in range(len(polyline) - 1):
        _, _, h1 = polyline[i]
        _, _, h2 = polyline[i + 1]
        total += gravitational_work(h1, h2, m, g, dh=0.1)
    return total


def create_static_work_heatmap(movements_df, tag, conf, dem_file, output_png):
    # --- Get bounding box from config ---
    bbox_dic = conf.get("bbox", {}).get(tag, {})
    if not bbox_dic:
        raise ValueError("Bounding box info missing in config.")
    bbox = (bbox_dic["min_lon"], bbox_dic["min_lat"],
            bbox_dic["max_lon"], bbox_dic["max_lat"])

    # --- Load road network using osmnx ---
    logging.info("Downloading OSM graph from bounding box...")
    G = ox.graph_from_bbox(bbox, network_type='drive', simplify=True)
    logging.info("Graph loaded with %d nodes and %d edges.", len(G.nodes), len(G.edges))

    # --- Crop DEM to bounding box ---
    bbox_geom = [box(*bbox)]
    with rasterio.open(dem_file) as dem:
        if dem.crs.to_string() != 'EPSG:4326':
            from rasterio.warp import transform_bounds
            bbox = transform_bounds('EPSG:4326', dem.crs, *bbox)
            bbox_geom = [box(*bbox)]
        dem_data, dem_transform = mask(dem, bbox_geom, crop=True)
        dem_data = dem_data[0]
        extent = [
            dem_transform[2],
            dem_transform[2] + dem_transform[0] * dem.width,
            dem_transform[5] + dem_transform[4] * dem.height,
            dem_transform[5]
        ]

    # --- Compute gravitational work for each polyline ---
    total_works = []
    valid_polylines = []
    for _, row in movements_df.iterrows():
        try:
            polyline = row['polyline_with_alt']
            work = compute_total_work(polyline)
            total_works.append(work)
            valid_polylines.append(polyline)
        except Exception:
            continue

    if not total_works:
        raise ValueError("No valid polylines with altitude found.")

    total_work_all = sum(total_works)
    n_trajs = len(valid_polylines)

    # --- Normalize and colormap ---
    norm = Normalize(vmin=min(total_works), vmax=max(total_works))
    cmap = plt.colormaps.get_cmap('plasma')

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # DEM background
    im = ax.imshow(dem_data, cmap='Greys_r', extent=extent, alpha=0.35)
    cbar_dem = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar_dem.set_label("DEM Altitude (m)", fontsize=10)

    # --- Plot OSM graph on background ---
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3, node_color='black', alpha=0.3)

    # --- Plot polylines colored by work ---
    for polyline, work in zip(valid_polylines, total_works):
        lats = [p[0] for p in polyline]
        lons = [p[1] for p in polyline]
        ax.plot(lons, lats, color=cmap(norm(work)), linewidth=1.4, alpha=0.9)

    # Colorbar for work
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_work = plt.colorbar(sm, ax=ax, label="Gravitational Work per Trajectory (J)", fraction=0.03, pad=0.04)

    # Title
    ax.set_title(
        f"Gravitational Work Map – {tag.capitalize()}\n"
        f"Total Work: {total_work_all:.2f} J over {n_trajs} trajectories",
        fontsize=14, pad=12
    )

    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    logging.info(f"Static heatmap saved to {output_png}")
    logging.info(f"Total work: {total_work_all:.2f} J across {n_trajs} trajectories.")


def main():
    parser = argparse.ArgumentParser(description="Generate static heatmap of movement trajectories by gravitational work.")
    parser.add_argument("-m", "--movements", required=True, help="CSV file with polyline_with_alt column")
    parser.add_argument("-c", "--config", required=True, help="Path to configuration JSON with bbox info")
    parser.add_argument("-d", "--dem", required=True, help="Path to DEM .tif file")
    parser.add_argument("-o", "--output", default="work_heatmap.png", help="Output PNG file")
    args = parser.parse_args()

    conf = read_conf(args.config)
    conf = conf.get("data_processing", {}).get("data_source", {}).get("worldpop", {})
    tag = conf.get("tag", "default")

    df = pd.read_csv(args.movements, sep=";")
    if 'polyline_with_alt' not in df.columns: raise ValueError("Missing 'polyline_with_alt' column. Run enrichment first.")
   
    df['polyline_with_alt'] = df['polyline_with_alt'].apply(ast.literal_eval)


    # Generate map
    create_static_work_heatmap(df, tag, conf, args.dem, args.output)


if __name__ == "__main__":
    main()
