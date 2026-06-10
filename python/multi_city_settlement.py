#!/usr/bin/env python3
"""Multi-city terrain-aware densification analysis using FUA boundaries."""


from __future__ import annotations

import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from matplotlib.patches import Patch
from rasterio.mask import mask
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from shapely.geometry import box, LineString
from tqdm.auto import tqdm



CITIES = [
    "milan",
    "barcelone",
    "toronto",
    "chicago",
    "amsterdam",
    "bandung",
    "bruxelles",
    "bogota",
]

DATA_ROOT = Path("/home/fbellisardi/code/topolity/data/data_processed")
ALT_DATA_ROOT = Path("/home/fbellisardi/code/data/data_processed")

OUTPUT_ROOT = Path("/home/fbellisardi/code/topolity/output/terrain_aware_densification")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FUA_GPKG = Path(
    "/home/fbellisardi/code/data/extra/ghs_fua_v1/"
    "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
)

WORLDPOP_ROOT = Path("/home/fbellisardi/code/topolity/data/worldpop/raw/2020")

WORLDPOP_BY_CITY = {
    "milan": "ita_ppp_2020.tif",
    "barcelone": "esp_ppp_2020.tif",
    "bruxelles": "bel_ppp_2020.tif",
    "amsterdam": "nld_ppp_2020.tif",
    "toronto": "can_ppp_2020.tif",
    "chicago": "usa_ppp_2020.tif",
    "bogota": "col_ppp_2020.tif",
    "bandung": "idn_ppp_2020.tif",
}

CITY_NAME_ALIASES = {
    "barcelone": "barcelona",
    "bruxelles": "brussels",
    "bogota": "bogota",
    "milan": "milano",
}

NEW_RESIDENTS = 25_000
TRIPS_PER_PERSON_PER_DAY = 2.0

D_0 = 25_000.0
ALPHA = 1.0

DS = 10.0

M_PHYS_KG = 1200.0
G_PHYS = 9.81

MIN_POPULATION_CELL = 10
MAX_CANDIDATE_CELLS = 120
GRAVITY_DESTINATION_THRESHOLD = 0.001

N_FAVORABLE = 3
N_UNFAVORABLE = 2

HORIZONTAL_COMPARABILITY_MODE = True
HORIZONTAL_REFERENCE = "median"
HORIZONTAL_TOLERANCES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
MIN_COMPARABLE_CELLS = max(20, N_FAVORABLE + N_UNFAVORABLE)

# Minimum distance between selected cells.
# If too restrictive, the code relaxes it automatically.
MIN_SELECTED_DISTANCE_M = 5_000
MIN_SELECTED_DISTANCE_RELAXATION = [1.0, 0.75, 0.50, 0.25, 0.0]

ZONE_COLORS = [
    "#1b9e77",
    "#7570b3",
    "#66a61e",
    "#d95f02",
    "#e7298a",
]

MAP_BASEMAP_PROVIDER = ctx.providers.CartoDB.PositronNoLabels
MAP_DEM_DOWNSAMPLE = 4
MAP_CONTOUR_LEVELS = 20
MAP_CONTOUR_LABEL_EVERY = 5
MAP_CONTOUR_ALPHA = 0.22
MAP_CONTOUR_LINEWIDTH = 0.25

FONT_TITLE = 26
FONT_LABEL = 24
FONT_TICK = 20
FONT_LEGEND = 18
FONT_BAR_TEXT = 18
FONT_MAP_NUMBER = 22
FONT_CONTOUR_LABEL = 12


MAKE_MAPS = True
MAKE_CHARTS = True



def compute_lambda_from_fuel_params(
    consumption_l_per_100km=(5.0, 8.0),
    energy_mj_per_l=36.0,
    efficiency=0.25,
):
    c_min, c_max = consumption_l_per_100km
    c_mean = 0.5 * (c_min + c_max)

    lambda_mean_mj_per_100km = efficiency * energy_mj_per_l * c_mean
    lambda_mean_j_per_m = lambda_mean_mj_per_100km * 10.0

    return lambda_mean_j_per_m


HORIZONTAL_COST_WEIGHT = compute_lambda_from_fuel_params()



def city_base_dir(city: str) -> Path:
    p1 = DATA_ROOT / city
    p2 = ALT_DATA_ROOT / city

    if p1.exists():
        return p1
    if p2.exists():
        return p2

    return p1


def city_paths(city: str) -> dict:
    base = city_base_dir(city)

    return {
        "base": base,
        "cells": base / f"{city}_basic_model" / "1000_cells" / "cell_coordinates.csv",
        "graph": base / "graphs_fine_grid" / "graph_original.pkl",
        "dem": base / "dem" / f"{city}_dem.tif",
        "worldpop": WORLDPOP_ROOT / WORLDPOP_BY_CITY[city],
        "output": OUTPUT_ROOT / city,
    }


def normalize_city_name(x: str) -> str:
    return (
        str(x)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("á", "a")
        .replace("à", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ú", "u")
        .strip()
    )



def load_city_fua(city: str) -> gpd.GeoDataFrame:
    if not FUA_GPKG.exists():
        raise FileNotFoundError(f"FUA file not found: {FUA_GPKG}")

    fua = gpd.read_file(FUA_GPKG)

    city_query = CITY_NAME_ALIASES.get(city, city)
    city_norm = normalize_city_name(city_query)

    name_cols = [
        c for c in fua.columns
        if any(k in c.lower() for k in ["name", "city", "fua", "uc"])
    ]

    if not name_cols:
        raise ValueError(f"No name-like columns found in FUA file: {list(fua.columns)}")

    mask = np.zeros(len(fua), dtype=bool)

    for col in name_cols:
        vals = fua[col].astype(str).map(normalize_city_name)
        mask |= vals.str.contains(city_norm, na=False)

    matches = fua[mask].copy()

    if matches.empty:
        raise ValueError(
            f"No FUA match found for city='{city}' normalized='{city_norm}'. "
            f"Searched columns: {name_cols}"
        )

    metric = matches.to_crs(3857)
    matches["area_tmp"] = metric.geometry.area.values

    selected = matches.sort_values("area_tmp", ascending=False).head(1)
    selected = selected.drop(columns=["area_tmp"])
    selected = selected.to_crs("EPSG:4326")

    print(f"  FUA match for {city}:")
    print(selected[name_cols].iloc[0].to_dict())

    return selected



def load_cells(cells_file: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(cells_file)

    df["centroid_x"] = (df["x_min"] + df["x_max"]) / 2
    df["centroid_y"] = (df["y_min"] + df["y_max"]) / 2

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            box(row.x_min, row.y_min, row.x_max, row.y_max)
            for _, row in df.iterrows()
        ],
        crs="EPSG:3857",
    )

    gdf["centroid"] = gdf.geometry.centroid
    return gdf


def load_graph(graph_file: Path):
    with open(graph_file, "rb") as f:
        return pickle.load(f)


def graph_nodes_gdf(G):
    rows = []

    for node_id in G.nodes():
        nd = G.nodes[node_id]
        rows.append(
            {
                "node_id": node_id,
                "x": nd.get("x", np.nan),
                "y": nd.get("y", np.nan),
            }
        )

    nodes_df = pd.DataFrame(rows)

    nodes_gdf = gpd.GeoDataFrame(
        nodes_df,
        geometry=gpd.points_from_xy(nodes_df["x"], nodes_df["y"]),
        crs="EPSG:4326",
    )

    nodes_wm = nodes_gdf.to_crs("EPSG:3857")
    return nodes_df, nodes_gdf, nodes_wm



def extract_population_to_cells(
    cells_gdf: gpd.GeoDataFrame,
    worldpop_file: Path,
    fua_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if not worldpop_file.exists():
        raise FileNotFoundError(f"WorldPop file not found: {worldpop_file}")

    with rasterio.open(worldpop_file) as src:
        pop_crs = src.crs

        cells_pop = cells_gdf.to_crs(pop_crs)
        fua_pop = fua_gdf.to_crs(pop_crs)

        try:
            fua_geom = fua_pop.geometry.union_all()
        except AttributeError:
            fua_geom = fua_pop.geometry.unary_union

        cells_pop["in_fua"] = cells_pop.geometry.intersects(fua_geom)
        cells_to_process = cells_pop[cells_pop["in_fua"]].copy()

        print(f"  Cells inside FUA: {len(cells_to_process):,} / {len(cells_gdf):,}")

        populations = []

        for idx, row in tqdm(
            cells_to_process.iterrows(),
            total=len(cells_to_process),
            desc="Extracting population inside FUA",
        ):
            try:
                out_image, _ = mask(src, [row.geometry], crop=True, nodata=0)
                pop = float(out_image.sum())
                populations.append((row.name, max(0.0, pop)))
            except Exception:
                populations.append((row.name, 0.0))

    cells_gdf = cells_gdf.copy()
    cells_gdf["population"] = 0.0

    for idx, pop in populations:
        cells_gdf.loc[idx, "population"] = pop

    cells_gdf = cells_gdf[cells_gdf["population"] > MIN_POPULATION_CELL].copy()
    cells_gdf.reset_index(drop=True, inplace=True)

    if cells_gdf.empty:
        raise ValueError("No populated cells inside FUA after filtering.")

    print(f"  Populated cells inside FUA: {len(cells_gdf):,}")
    print(f"  Total population inside FUA: {cells_gdf['population'].sum():,.0f}")

    return cells_gdf



def assign_nearest_nodes(cells_gdf, nodes_df, nodes_wm):
    node_coords = np.array([[geom.x, geom.y] for geom in nodes_wm.geometry])
    tree = KDTree(node_coords)

    cell_coords = np.array(
        [[row.centroid_x, row.centroid_y] for _, row in cells_gdf.iterrows()]
    )

    distances, indices = tree.query(cell_coords, k=1)

    cells_gdf = cells_gdf.copy()
    cells_gdf["nearest_node"] = nodes_df.iloc[indices]["node_id"].values
    cells_gdf["node_distance"] = distances

    print(f"  Mean distance cell → nearest node: {distances.mean():.1f} m")
    print(f"  Max distance cell → nearest node: {distances.max():.1f} m")

    return cells_gdf


def get_edge_data(G, u, v, key=None):
    data = G.get_edge_data(u, v)

    if data is None:
        return {}

    if key is not None and isinstance(data, dict) and key in data:
        return data[key]

    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            return first_val

    return data


def compute_edge_vertical_gain_m(G, u, v, key, dem_src, ds=10.0):
    edge_data = get_edge_data(G, u, v, key)

    geom = edge_data.get("geometry", None)

    if geom is None:
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = LineString([(x1, y1), (x2, y2)])

    length = float(geom.length)
    n_pts = max(int(length / ds) + 1, 2)
    dists = np.linspace(0, length, n_pts)

    pts = [geom.interpolate(d) for d in dists]
    coords = [(pt.x, pt.y) for pt in pts]

    elevs = np.array([val[0] for val in dem_src.sample(coords)], dtype=float)

    vertical_gain_m = 0.0

    for h1, h2 in zip(elevs[:-1], elevs[1:]):
        if np.isfinite(h1) and np.isfinite(h2) and h2 > h1:
            vertical_gain_m += float(h2 - h1)

    return vertical_gain_m


def precompute_edge_vertical_gain(G, dem_file: Path) -> dict:
    edge_vertical_gain_m = {}

    with rasterio.open(dem_file) as dem_src:
        if getattr(G, "is_multigraph", lambda: False)():
            iterator = G.edges(keys=True)
            for u, v, key in tqdm(
                iterator,
                desc="Precomputing edge vertical gain",
                total=G.number_of_edges(),
            ):
                edge_vertical_gain_m[(u, v, key)] = compute_edge_vertical_gain_m(
                    G, u, v, key, dem_src, ds=DS
                )
        else:
            iterator = G.edges()
            for u, v in tqdm(
                iterator,
                desc="Precomputing edge vertical gain",
                total=G.number_of_edges(),
            ):
                edge_vertical_gain_m[(u, v, 0)] = compute_edge_vertical_gain_m(
                    G, u, v, 0, dem_src, ds=DS
                )

    return edge_vertical_gain_m


def edge_length_and_vertical_gain(G, edge_vertical_gain_m, u, v):
    data = G.get_edge_data(u, v)

    if data is None:
        return 0.0, 0.0

    if isinstance(data, dict):
        first_val = next(iter(data.values()))
        if isinstance(first_val, dict):
            key = list(data.keys())[0]
            edge_data = data[key]
        else:
            key = 0
            edge_data = data
    else:
        key = 0
        edge_data = {}

    length_m = edge_data.get("length", 0.0)
    vertical_gain_m = edge_vertical_gain_m.get((u, v, key), 0.0)

    return float(length_m), float(vertical_gain_m)


def path_components(G, edge_vertical_gain_m, source, target):
    path = nx.shortest_path(G, source=source, target=target, weight="length")

    length_m = 0.0
    vertical_gain_m = 0.0

    for u, v in zip(path[:-1], path[1:]):
        le, vg = edge_length_and_vertical_gain(G, edge_vertical_gain_m, u, v)
        length_m += le
        vertical_gain_m += vg

    return length_m, vertical_gain_m



def select_candidate_cells(cells_gdf: gpd.GeoDataFrame, max_candidates: int) -> gpd.GeoDataFrame:
    cells = cells_gdf.copy()

    center_x = np.average(cells["centroid_x"], weights=cells["population"])
    center_y = np.average(cells["centroid_y"], weights=cells["population"])

    cells["dist_to_center"] = np.sqrt(
        (cells["centroid_x"] - center_x) ** 2
        + (cells["centroid_y"] - center_y) ** 2
    )

    q_pop = cells["population"].quantile(0.65)
    q_near = cells["dist_to_center"].quantile(0.50)
    q_far = cells["dist_to_center"].quantile(0.75)

    high_pop = cells[cells["population"] >= q_pop]
    central = cells[cells["dist_to_center"] <= q_near]
    peripheral = cells[cells["dist_to_center"] >= q_far]

    selected = pd.concat(
        [
            high_pop.nlargest(max_candidates // 2, "population"),
            central.nlargest(max_candidates // 4, "population"),
            peripheral.nlargest(max_candidates // 4, "population"),
        ]
    ).drop_duplicates(subset=["cell_id"])

    if len(selected) > max_candidates:
        selected = selected.nlargest(max_candidates, "population")

    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=cells_gdf.crs)
    selected.reset_index(drop=True, inplace=True)

    return selected



def compute_densification_cost_for_cell(
    candidate_cell,
    cells_gdf,
    distance_matrix,
    G,
    edge_vertical_gain_m,
):
    populations = cells_gdf["population"].values.copy()

    test_cell_idx = cells_gdf[cells_gdf["cell_id"] == candidate_cell["cell_id"]].index[0]
    populations[test_cell_idx] += NEW_RESIDENTS

    distances_from_test = distance_matrix[test_cell_idx, :]

    attractions = populations**ALPHA * np.exp(-distances_from_test / D_0)
    attractions[test_cell_idx] = 0.0

    total_attraction = attractions.sum()

    if total_attraction <= 0:
        return None

    trip_probabilities = attractions / total_attraction
    total_trips = NEW_RESIDENTS * TRIPS_PER_PERSON_PER_DAY
    trips_to_destinations = total_trips * trip_probabilities

    significant_destinations = np.where(
        trip_probabilities > GRAVITY_DESTINATION_THRESHOLD
    )[0]

    test_node = candidate_cell["nearest_node"]

    total_energy_J = 0.0
    total_vertical_energy_J = 0.0
    total_vertical_gain_m = 0.0
    total_horizontal_distance_m = 0.0
    total_horizontal_energy_J = 0.0

    successful_routes = 0
    failed_routes = 0
    total_route_distance_m = 0.0
    weighted_distance_m = 0.0

    for dest_idx in significant_destinations:
        dest_cell = cells_gdf.iloc[dest_idx]
        dest_node = dest_cell["nearest_node"]
        n_trips = trips_to_destinations[dest_idx]

        if n_trips < 1:
            continue

        try:
            length_out_m, vertical_out_m = path_components(
                G, edge_vertical_gain_m, test_node, dest_node
            )

            length_ret_m, vertical_ret_m = path_components(
                G, edge_vertical_gain_m, dest_node, test_node
            )

            vertical_gain_per_trip_m = vertical_out_m + vertical_ret_m
            horizontal_distance_per_trip_m = length_out_m + length_ret_m

            vertical_energy_per_trip_J = M_PHYS_KG * G_PHYS * vertical_gain_per_trip_m
            horizontal_energy_per_trip_J = HORIZONTAL_COST_WEIGHT * horizontal_distance_per_trip_m
            total_energy_per_trip_J = vertical_energy_per_trip_J + horizontal_energy_per_trip_J

            total_vertical_gain_m += n_trips * vertical_gain_per_trip_m
            total_vertical_energy_J += n_trips * vertical_energy_per_trip_J
            total_horizontal_distance_m += n_trips * horizontal_distance_per_trip_m
            total_horizontal_energy_J += n_trips * horizontal_energy_per_trip_J
            total_energy_J += n_trips * total_energy_per_trip_J

            total_route_distance_m += length_out_m
            weighted_distance_m += length_out_m * n_trips

            successful_routes += 1

        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            failed_routes += 1
            continue

    if successful_routes == 0:
        return None

    avg_route_distance_m = total_route_distance_m / successful_routes
    avg_weighted_distance_m = weighted_distance_m / total_trips if total_trips > 0 else np.nan

    return {
        "cell_id": candidate_cell["cell_id"],
        "centroid_x": candidate_cell["centroid_x"],
        "centroid_y": candidate_cell["centroid_y"],
        "baseline_population": candidate_cell["population"],
        "new_residents_added": NEW_RESIDENTS,
        "total_trips": total_trips,
        "successful_routes": successful_routes,
        "failed_routes": failed_routes,
        "avg_route_distance_m": avg_route_distance_m,
        "avg_weighted_distance_m": avg_weighted_distance_m,
        "vertical_gain_m": total_vertical_gain_m,
        "vertical_work_joules": total_vertical_energy_J,
        "horizontal_distance_weighted_m": total_horizontal_distance_m,
        "horizontal_cost_joules": total_horizontal_energy_J,
        "total_work_joules": total_energy_J,
        "work_per_resident": total_energy_J / NEW_RESIDENTS,
        "work_per_trip": total_energy_J / total_trips,
        "vertical_per_resident": total_vertical_energy_J / NEW_RESIDENTS,
        "horizontal_per_resident": total_horizontal_energy_J / NEW_RESIDENTS,
        "vertical_share_pct": total_vertical_energy_J / total_energy_J * 100,
        "horizontal_share_pct": total_horizontal_energy_J / total_energy_J * 100,
    }



def _is_far_enough(row, selected_rows, min_dist_m: float) -> bool:
    if min_dist_m <= 0 or not selected_rows:
        return True

    x = float(row["centroid_x"])
    y = float(row["centroid_y"])

    for s in selected_rows:
        dx = x - float(s["centroid_x"])
        dy = y - float(s["centroid_y"])
        if np.sqrt(dx * dx + dy * dy) < min_dist_m:
            return False

    return True


def _greedy_select_spaced(
    df: pd.DataFrame,
    n: int,
    sort_col: str,
    ascending: bool,
    already_selected: list[dict],
    min_dist_m: float,
) -> list[dict]:
    selected = []
    pool = df.sort_values(sort_col, ascending=ascending).copy()

    for _, row in pool.iterrows():
        row_dict = row.to_dict()
        if _is_far_enough(row_dict, already_selected + selected, min_dist_m):
            selected.append(row_dict)
        if len(selected) >= n:
            break

    return selected


def select_favorable_unfavorable(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select favorable/unfavorable cells at comparable horizontal accessibility.

    Favorable:
        lowest vertical energy.

    Unfavorable:
        highest vertical energy.

    Extra constraints:
        selected cells should be spatially separated so that map circles do not overlap.
    """
    df = results_df.copy()

    if not HORIZONTAL_COMPARABILITY_MODE:
        comparable_df = df.copy()
        h_ref = np.nan
        selected_tolerance = np.nan
    else:
        if HORIZONTAL_REFERENCE == "best":
            h_ref = df.loc[df["total_work_joules"].idxmin(), "horizontal_cost_joules"]
        else:
            h_ref = df["horizontal_cost_joules"].median()

        comparable_df = pd.DataFrame()
        selected_tolerance = None

        for tol in HORIZONTAL_TOLERANCES:
            tmp = df[
                (df["horizontal_cost_joules"] >= h_ref * (1 - tol))
                & (df["horizontal_cost_joules"] <= h_ref * (1 + tol))
            ].copy()

            if len(tmp) >= MIN_COMPARABLE_CELLS:
                comparable_df = tmp
                selected_tolerance = tol
                break

        if comparable_df.empty:
            comparable_df = df.copy()
            selected_tolerance = np.nan

    ranking_col = "vertical_work_joules"

    selected_rows = []
    used_distance = None

    for relax in MIN_SELECTED_DISTANCE_RELAXATION:
        min_dist = MIN_SELECTED_DISTANCE_M * relax

        favorable_rows = _greedy_select_spaced(
            comparable_df,
            n=N_FAVORABLE,
            sort_col=ranking_col,
            ascending=True,
            already_selected=[],
            min_dist_m=min_dist,
        )

        unfavorable_rows = _greedy_select_spaced(
            comparable_df,
            n=N_UNFAVORABLE,
            sort_col=ranking_col,
            ascending=False,
            already_selected=favorable_rows,
            min_dist_m=min_dist,
        )

        if len(favorable_rows) == N_FAVORABLE and len(unfavorable_rows) == N_UNFAVORABLE:
            selected_rows = favorable_rows + unfavorable_rows
            used_distance = min_dist
            break

    if not selected_rows:
        favorable_rows = _greedy_select_spaced(
            comparable_df,
            n=N_FAVORABLE,
            sort_col=ranking_col,
            ascending=True,
            already_selected=[],
            min_dist_m=0.0,
        )

        unfavorable_rows = _greedy_select_spaced(
            comparable_df,
            n=N_UNFAVORABLE,
            sort_col=ranking_col,
            ascending=False,
            already_selected=favorable_rows,
            min_dist_m=0.0,
        )

        selected_rows = favorable_rows + unfavorable_rows
        used_distance = 0.0

    selected = pd.DataFrame(selected_rows)

    selected["scenario_type"] = (
        ["favorable"] * N_FAVORABLE
        + ["unfavorable"] * N_UNFAVORABLE
    )

    selected = selected.sort_values(ranking_col, ascending=True).reset_index(drop=True)

    selected["rank_total"] = np.arange(1, len(selected) + 1)
    selected["zone_label"] = [
        f"Favorable {i + 1}" if i < N_FAVORABLE else f"Unfavorable {i - N_FAVORABLE + 1}"
        for i in range(len(selected))
    ]
    selected["zone_color"] = ZONE_COLORS[: len(selected)]

    selected["selection_ranking_col"] = ranking_col
    selected["horizontal_reference_joules"] = h_ref
    selected["horizontal_tolerance_used"] = selected_tolerance
    selected["min_selected_distance_used_m"] = used_distance
    selected["horizontal_cost_relative_to_ref_pct"] = (
        (selected["horizontal_cost_joules"] / h_ref - 1.0) * 100
        if np.isfinite(h_ref) and h_ref != 0
        else np.nan
    )

    print("\n[selection diagnostic]")
    print(f"  mode: horizontal-comparable vertical ranking")
    print(f"  ranking column: {ranking_col}")
    print(f"  horizontal reference: {h_ref:,.0f} J")
    print(f"  tolerance used: {selected_tolerance}")
    print(f"  comparable cells: {len(comparable_df)} / {len(df)}")
    print(f"  min selected distance used: {used_distance:,.0f} m")
    print(f"  vertical range selected: {selected['vertical_work_joules'].min()/1e9:.2f}–{selected['vertical_work_joules'].max()/1e9:.2f} GJ")

    return selected



def add_dem_contours_to_ax(
    ax,
    city: str,
    dem_file: Path,
    levels_n: int = MAP_CONTOUR_LEVELS,
    downsample: int = MAP_DEM_DOWNSAMPLE,
):
    with rasterio.open(dem_file) as src:
        dem = src.read(
            1,
            out_shape=(src.height // downsample, src.width // downsample),
            resampling=Resampling.bilinear,
        ).astype("float32")

        bounds = src.bounds
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan

    dem[~np.isfinite(dem)] = np.nan

    if np.all(~np.isfinite(dem)):
        print(f"[{city}] DEM contour skipped: invalid DEM")
        return

    levels = np.linspace(
        np.nanpercentile(dem, 2),
        np.nanpercentile(dem, 98),
        levels_n,
    )

    contours = ax.contour(
        dem,
        levels=levels,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        colors="black",
        linewidths=MAP_CONTOUR_LINEWIDTH,
        alpha=MAP_CONTOUR_ALPHA,
        zorder=12,
    )

    ax.clabel(
        contours,
        contours.levels[::MAP_CONTOUR_LABEL_EVERY],
        inline=True,
        fontsize=FONT_CONTOUR_LABEL,
        fmt="%.0f m",
    )

def make_city_plots(city, cells_gdf, selected_df, output_dir, dem_file: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cells = cells_gdf[cells_gdf["cell_id"].isin(selected_df["cell_id"])].copy()

    selected_cells = selected_cells.merge(
        selected_df[
            [
                "cell_id",
                "scenario_type",
                "zone_label",
                "zone_color",
                "total_work_joules",
                "vertical_work_joules",
                "horizontal_cost_joules",
                "work_per_resident",
                "vertical_share_pct",
                "horizontal_share_pct",
                "rank_total",
            ]
        ],
        on="cell_id",
        how="left",
    )

    cells_ll = cells_gdf.to_crs("EPSG:4326")
    selected_ll = selected_cells.to_crs("EPSG:4326")

    if MAKE_MAPS:
        fig, ax = plt.subplots(figsize=(12, 10))

        cells_ll.plot(ax=ax, facecolor="#e0e0e0", edgecolor="none", alpha=0.25)

        for _, row in selected_ll.iterrows():
            color = row["zone_color"]
            centroid = row.geometry.centroid

            gpd.GeoSeries([row.geometry], crs=selected_ll.crs).plot(
                ax=ax,
                facecolor=color,
                edgecolor="black",
                alpha=0.45,
                linewidth=2,
                zorder=10,
            )

            ax.scatter(
                centroid.x,
                centroid.y,
                s=850,
                color=color,
                edgecolor="black",
                linewidth=2,
                zorder=20,
            )

            ax.text(
                centroid.x,
                centroid.y,
                str(int(row["rank_total"])),
                ha="center",
                va="center",
                fontsize=FONT_MAP_NUMBER,
                weight="bold",
                color="black",
                zorder=30,
            )

        try:
            ctx.add_basemap(
                ax,
                crs=cells_ll.crs,
                source=MAP_BASEMAP_PROVIDER,
                alpha=1,
                attribution=False,
                zorder=1,
            )
        except Exception as e:
            print(f"[{city}] Could not add basemap: {e}")
        # Focus the map on the analyzed area (selected cells). Use selected extent
        # with a padding fraction so the map does not include large empty regions.
        try:
            if not selected_ll.empty:
                minx, miny, maxx, maxy = selected_ll.total_bounds
                # add 25% padding of the width/height (fallback to small value)
                pad_x = (maxx - minx) * 0.25 if (maxx - minx) > 0 else 0.01
                pad_y = (maxy - miny) * 0.25 if (maxy - miny) > 0 else 0.01
                ax.set_xlim(minx - pad_x, maxx + pad_x)
                ax.set_ylim(miny - pad_y, maxy + pad_y)
            else:
                # fallback to full cells extent with smaller padding
                minx, miny, maxx, maxy = cells_ll.total_bounds
                pad_x = (maxx - minx) * 0.10 if (maxx - minx) > 0 else 0.01
                pad_y = (maxy - miny) * 0.10 if (maxy - miny) > 0 else 0.01
                ax.set_xlim(minx - pad_x, maxx + pad_x)
                ax.set_ylim(miny - pad_y, maxy + pad_y)
        except Exception as e:
            print(f"[{city}] Could not set focused extent: {e}")

        # try:
        #     add_dem_contours_to_ax(
        #         ax=ax,
        #         city=city,
        #         dem_file=dem_file,
        #     )
        # except Exception as e:
        #     print(f"[{city}] Could not add DEM contours: {e}")

        legend_handles = [
            Patch(
                facecolor=row["zone_color"],
                edgecolor="black",
                label=f"{int(row['rank_total'])}. {row['zone_label']}",
            )
            for _, row in selected_df.iterrows()
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=FONT_LEGEND)

        ax.set_title(f"{city.title()}")
        ax.set_xlabel("Longitude", fontsize=FONT_LABEL)
        ax.set_ylabel("Latitude", fontsize=FONT_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linestyle="--")

        fig.tight_layout()
        fig.savefig(output_dir / f"{city}_selected_densification_map.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{city}_selected_densification_map.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if MAKE_CHARTS:
        plot_df = selected_df.copy().reset_index(drop=True)

        x = np.arange(len(plot_df))
        colors = plot_df["zone_color"].tolist()

        # --------------------------------------------------------
        # Main chart: vertical energy only
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))

        vertical_gj = plot_df["vertical_work_joules"] / 1e9

        ax.bar(
            x,
            vertical_gj,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.90,
        )

        ymax = max(vertical_gj.max(), 1e-9)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            h = row["vertical_work_joules"] / 1e9
            ax.text(
                i,
                h + 0.03 * ymax,
                f"{h:.2f} GJ\n{row['vertical_per_resident'] / 1e6:.2f} MJ/res",
                ha="center",
                va="bottom",
                fontsize=FONT_BAR_TEXT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["zone_label"], rotation=20, fontsize=FONT_TICK)
        ax.set_ylabel("Vertical mobility energy (GJ)", fontsize=FONT_LABEL)
        ax.set_title(f"{city.title()}")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / f"{city}_densification_vertical_energy.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{city}_densification_vertical_energy.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------------------
        # Total chart: same colors as map
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))

        total_gj = plot_df["total_work_joules"] / 1e9

        ax.bar(
            x,
            total_gj,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.90,
        )

        ymax = max(total_gj.max(), 1e-9)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            h = row["total_work_joules"] / 1e9
            ax.text(
                i,
                h + 0.02 * ymax,
                f"{h:.2f} GJ\n{row['work_per_resident'] / 1e6:.1f} MJ/res",
                ha="center",
                va="bottom",
                fontsize=FONT_BAR_TEXT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["zone_label"], rotation=20, fontsize=FONT_TICK)
        ax.set_ylabel("Total additional mobility energy (GJ)", fontsize=FONT_LABEL)
        ax.set_title(f"{city.title()}", fontsize=FONT_TITLE)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / f"{city}_densification_total_energy.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{city}_densification_total_energy.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------------------
        # Component chart: same zone color, solid vertical + transparent horizontal
        # --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))

        vertical_gj = plot_df["vertical_work_joules"] / 1e9
        horizontal_gj = plot_df["horizontal_cost_joules"] / 1e9

        ax.bar(
            x,
            vertical_gj,
            color=colors,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95,
            label="Vertical",
        )

        ax.bar(
            x,
            horizontal_gj,
            bottom=vertical_gj,
            color=colors,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.35,
            label="Horizontal",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["zone_label"], rotation=20, fontsize=FONT_TICK)
        ax.set_ylabel("Additional mobility energy (GJ)", fontsize=FONT_LABEL)
        ax.set_title(f"{city.title()}", fontsize=FONT_TITLE)
        ax.legend(
            handles=[
                Patch(facecolor="gray", edgecolor="black", alpha=0.95, label="Altitudinal component"),
                Patch(facecolor="gray", edgecolor="black", alpha=0.35, label="Longitudinal component"),
            ],
            loc="upper left",
            fontsize=FONT_LEGEND,
        )
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(output_dir / f"{city}_densification_components.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{city}_densification_components.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)



def run_city(city: str):
    print("\n" + "=" * 100)
    print(f"RUNNING CITY: {city.upper()}")
    print("=" * 100)

    paths = city_paths(city)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["cells", "graph", "dem", "worldpop"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"[{city}] Missing {key}: {paths[key]}")

    fua_gdf = load_city_fua(city)

    cells = load_cells(paths["cells"])
    cells = extract_population_to_cells(cells, paths["worldpop"], fua_gdf)

    G = load_graph(paths["graph"])
    nodes_df, nodes_ll, nodes_wm = graph_nodes_gdf(G)

    cells = assign_nearest_nodes(cells, nodes_df, nodes_wm)

    edge_vertical_gain_m = precompute_edge_vertical_gain(G, paths["dem"])

    candidate_cells = select_candidate_cells(cells, MAX_CANDIDATE_CELLS)

    print(f"[{city}] Populated cells inside FUA: {len(cells):,}")
    print(f"[{city}] Candidate cells evaluated: {len(candidate_cells):,}")

    cell_centroids = np.array(
        [[row.centroid_x, row.centroid_y] for _, row in cells.iterrows()]
    )

    distance_matrix = cdist(cell_centroids, cell_centroids, metric="euclidean")

    results = []

    for _, candidate in tqdm(
        candidate_cells.iterrows(),
        total=len(candidate_cells),
        desc=f"{city} candidate cells",
    ):
        res = compute_densification_cost_for_cell(
            candidate,
            cells,
            distance_matrix,
            G,
            edge_vertical_gain_m,
        )

        if res is not None:
            results.append(res)

    if not results:
        raise RuntimeError(f"[{city}] No valid candidate results.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_work_joules")

    best = results_df["total_work_joules"].min()
    results_df["work_increase_vs_best"] = results_df["total_work_joules"] - best
    results_df["work_increase_pct"] = (
        results_df["total_work_joules"] / best - 1.0
    ) * 100

    selected_df = select_favorable_unfavorable(results_df)

    results_csv = output_dir / f"{city}_all_candidate_densification_results.csv"
    selected_csv = output_dir / f"{city}_selected_favorable_unfavorable_cells.csv"

    results_df.to_csv(results_csv, index=False, sep=";")
    selected_df.to_csv(selected_csv, index=False, sep=";")

    selected_gdf = cells[cells["cell_id"].isin(selected_df["cell_id"])].copy()
    selected_gdf = selected_gdf.merge(selected_df, on="cell_id", how="left")

    if "centroid" in selected_gdf.columns:
        selected_gdf = selected_gdf.drop(columns=["centroid"])

    selected_gdf.to_file(
        output_dir / f"{city}_selected_favorable_unfavorable_cells.gpkg",
        driver="GPKG",
    )

    make_city_plots(city, cells, selected_df, output_dir, paths["dem"])

    print(f"[{city}] Best comparable-horizontal cell: {selected_df.iloc[0]['cell_id']}")
    print(f"[{city}] Vertical energy: {selected_df.iloc[0]['vertical_work_joules'] / 1e9:.2f} GJ")
    print(f"[{city}] Total energy: {selected_df.iloc[0]['total_work_joules'] / 1e9:.2f} GJ")
    print(f"[{city}] Horizontal deviation from reference: {selected_df.iloc[0]['horizontal_cost_relative_to_ref_pct']:.2f}%")
    print(f"[{city}] Saved to: {output_dir}")

    selected_df["city"] = city
    results_df["city"] = city

    return results_df, selected_df



def main():
    print("=" * 100)
    print("MULTI-CITY TERRAIN-AWARE DENSIFICATION ANALYSIS USING FUA")
    print("=" * 100)
    print(f"New residents per test cell: {NEW_RESIDENTS:,}")
    print(f"M_PHYS_KG: {M_PHYS_KG}")
    print(f"G_PHYS: {G_PHYS}")
    print(f"Horizontal lambda: {HORIZONTAL_COST_WEIGHT:.2f} J/m")
    print(f"FUA file: {FUA_GPKG}")
    print(f"Output root: {OUTPUT_ROOT}")

    all_results = []
    all_selected = []

    for city in CITIES:
        try:
            results_df, selected_df = run_city(city)
            all_results.append(results_df)
            all_selected.append(selected_df)
            # break  # TEMP: run only the first city for now
        except Exception as e:
            warnings.warn(f"[{city}] failed: {e}")

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(
            OUTPUT_ROOT / "all_cities_candidate_densification_results.csv",
            index=False,
            sep=";",
        )

    if all_selected:
        all_selected_df = pd.concat(all_selected, ignore_index=True)
        all_selected_df.to_csv(
            OUTPUT_ROOT / "all_cities_selected_favorable_unfavorable_cells.csv",
            index=False,
            sep=";",
        )

        print("\nSelected favorable/unfavorable cells:")
        print(
            all_selected_df[
                [
                    "city",
                    "zone_label",
                    "scenario_type",
                    "cell_id",
                    "selection_ranking_col",
                    "horizontal_tolerance_used",
                    "min_selected_distance_used_m",
                    "horizontal_cost_relative_to_ref_pct",
                    "total_work_joules",
                    "vertical_work_joules",
                    "horizontal_cost_joules",
                    "vertical_share_pct",
                    "horizontal_share_pct",
                    "work_increase_pct",
                ]
            ]
        )

    print("\nDone.")
    print(f"All files saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()