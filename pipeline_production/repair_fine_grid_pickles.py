#!/usr/bin/env python3
"""Repair fine-grid graph pickles to ensure valid edge geometries and lengths."""


from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.affinity import rotate as shapely_rotate, scale as shapely_scale, translate as shapely_translate
from shapely.geometry import LineString


ROOT = Path("/home/fbellisardi/code/topolity")
DEFAULT_DATA_ROOT = ROOT / "data" / "data_processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", required=True, help="City folder names to repair.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing city folders.")
    return parser.parse_args()


def load_graph(path: Path) -> nx.MultiDiGraph:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_graph(path: Path, graph: nx.MultiDiGraph) -> None:
    with path.open("wb") as handle:
        pickle.dump(graph, handle)


def bbox_center(G: nx.MultiDiGraph) -> tuple[float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    return float((xs.min() + xs.max()) / 2.0), float((ys.min() + ys.max()) / 2.0)


def metric_transformer(G: nx.MultiDiGraph) -> Transformer:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    lon0 = float(xs.mean())
    lat0 = float(ys.mean())
    proj_str = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)


def geometry_length_m(geom: LineString, transformer: Transformer) -> float:
    coords = np.asarray(geom.coords, dtype=float)
    x_m, y_m = transformer.transform(coords[:, 0], coords[:, 1])
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    return float(np.sum(np.sqrt(dx**2 + dy**2)))


def recompute_lengths(G: nx.MultiDiGraph) -> None:
    transformer = metric_transformer(G)
    for u, v, _, edge_data in G.edges(keys=True, data=True):
        geom = edge_data.get("geometry")
        if isinstance(geom, LineString):
            edge_data["length"] = geometry_length_m(geom, transformer)
        else:
            x1 = float(G.nodes[u]["x"])
            y1 = float(G.nodes[u]["y"])
            x2 = float(G.nodes[v]["x"])
            y2 = float(G.nodes[v]["y"])
            x_m, y_m = transformer.transform([x1, x2], [y1, y2])
            edge_data["length"] = float(np.hypot(x_m[1] - x_m[0], y_m[1] - y_m[0]))


def original_edge_geometry(G_original: nx.MultiDiGraph, u, v, k) -> LineString:
    edge_data = G_original.get_edge_data(u, v, k)
    if edge_data is None:
        u_node = G_original.nodes[u]
        v_node = G_original.nodes[v]
        return LineString([(float(u_node["x"]), float(u_node["y"])), (float(v_node["x"]), float(v_node["y"]))])
    geom = edge_data.get("geometry")
    if isinstance(geom, LineString):
        return geom
    u_node = G_original.nodes[u]
    v_node = G_original.nodes[v]
    return LineString([(float(u_node["x"]), float(u_node["y"])), (float(v_node["x"]), float(v_node["y"]))])


def apply_transform_to_geometry(geom: LineString, row: pd.Series, origin: tuple[float, float]) -> LineString:
    variant_type = str(row["type"])
    if variant_type == "translate":
        return shapely_translate(geom, xoff=float(row["offset_x"]), yoff=float(row["offset_y"]))
    if variant_type == "rotate":
        return shapely_rotate(geom, float(row["angle_deg"]), origin=origin, use_radians=False)
    if variant_type == "scale":
        axis = str(row.get("scale_axis", "")).lower()
        xfact = float(row["scale_factor"]) if axis == "x" else 1.0
        yfact = float(row["scale_factor"]) if axis == "y" else 1.0
        return shapely_scale(geom, xfact=xfact, yfact=yfact, origin=origin)
    return geom


def repair_city(city: str, data_root: Path) -> None:
    graphs_dir = data_root / city / "graphs_fine_grid"
    stats_csv = graphs_dir / "fine_grid_stats.csv"
    original_pkl = graphs_dir / "graph_original.pkl"

    stats_df = pd.read_csv(stats_csv)
    stats_df["variant"] = stats_df["variant"].astype(str)
    stats_df["on_land_bool"] = stats_df["on_land"].astype(str).str.lower().eq("true")
    valid_rows = stats_df.loc[stats_df["on_land_bool"]].copy()

    G_original = load_graph(original_pkl)
    origin = bbox_center(G_original)

    repaired = 0
    for row in valid_rows.itertuples(index=False):
        variant = str(row.variant)
        pkl_path = graphs_dir / f"graph_{variant}.pkl"
        if not pkl_path.exists():
            continue

        if variant == "original":
            G_variant = load_graph(pkl_path)
            recompute_lengths(G_variant)
            save_graph(pkl_path, G_variant)
            repaired += 1
            continue

        G_variant = load_graph(pkl_path)
        row_series = pd.Series(row._asdict())

        for u, v, k, edge_data in G_variant.edges(keys=True, data=True):
            base_geom = original_edge_geometry(G_original, u, v, k)
            edge_data["geometry"] = apply_transform_to_geometry(base_geom, row_series, origin)

        recompute_lengths(G_variant)
        save_graph(pkl_path, G_variant)
        repaired += 1

    print(f"{city}: repaired {repaired} pickle files in {graphs_dir}")


def main() -> None:
    args = parse_args()
    for city in args.cities:
        repair_city(city, args.data_root)


if __name__ == "__main__":
    main()
