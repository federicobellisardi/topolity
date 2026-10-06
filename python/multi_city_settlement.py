#!/usr/bin/env python3
"""Multi-city terrain-aware densification analysis using FUA boundaries.

Pipeline for one city
---------------------
1. FUA boundary -> local UTM CRS. Every distance below is in true metres.
2. Grid cells + WorldPop -> populated cells inside the FUA.
3. Road graph: every edge is densified every DS metres in the UTM plane and the
   DEM (reprojected to UTM, bilinear) is sampled bilinearly along it, giving
     - planar length              L_e^2D
     - terrain-following length   L_e = sum sqrt(dx^2 + dy^2 + dh^2)   (Eq. 3)
     - positive elevation gain    Dh_e^+ = sum max(0, h_{i+1} - h_i)
4. d_0 is fitted on the empirical working-day OD flows with the same
   production-constrained gravity model used for the synthetic demand (Eq. 4):
   Poisson pseudo-maximum-likelihood with origin fixed effects, which is
   equivalent to the multinomial likelihood of Eq. 4 and includes zero-flow
   pairs through the normalisation over all destinations.
5. For every candidate cell, the new residents make one round trip per day
   (outbound + return). Destinations follow Eq. 4; trips are routed by
   minimising the 3-D length L_e, and each round trip costs
       W = lambda * (L_out + L_ret) + m g (Dh_out^+ + Dh_ret^+)        (Eq. 1, 5)
   All destinations are routed (GRAVITY_DESTINATION_THRESHOLD = 0); if a
   threshold is set, the remaining probabilities are renormalised so the totals
   always refer to the full demand N_trips = Delta P round trips.
6. Favourable / unfavourable cells are selected among candidates with
   comparable horizontal cost.
"""

from __future__ import annotations

import argparse
import json
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
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from shapely.geometry import box
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

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

DISPLAY_NAMES = {
    "milan": "Milan",
    "barcelone": "Barcelona",
    "toronto": "Toronto",
    "chicago": "Chicago",
    "amsterdam": "Amsterdam",
    "bandung": "Bandung",
    "bruxelles": "Brussels",
    "bogota": "Bogotá",
}

DATA_ROOT = Path("/home/fbellisardi/code/topolity/data/data_processed")
ALT_DATA_ROOT = Path("/home/fbellisardi/code/data/data_processed")

OUTPUT_ROOT = Path("/home/fbellisardi/code/topolity/output/terrain_aware_densification")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FUA_GPKG = Path(
    "/home/fbellisardi/code/data/extra/ghs_fua_v1/"
    "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
)

# NOTE: "<iso>_ppp_2020.tif" is the WorldPop *unconstrained* product.
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

# Demand
NEW_RESIDENTS = 25_000
ROUND_TRIPS_PER_PERSON_PER_DAY = 1.0   # one round trip: e.g. home->work + work->home
ALPHA = 1.0                            # destination-population exponent (fixed)
GRAVITY_DESTINATION_THRESHOLD = 0.0    # 0 = route every destination

# Geometry / terrain
DS = 10.0                 # sampling step along edges [m]
DEM_MIN_VALID = -500.0    # anything below is treated as a DEM void
SRTM_VOID = -32768

# Routing and horizontal cost both use the terrain-following length (Eq. 3)
ROUTING_WEIGHT = "length_3d"
HORIZONTAL_LENGTH_ATTR = "length_3d"

# Energetics
M_PHYS_KG = 1200.0
G_PHYS = 9.81

# Candidate cells and selection
MIN_POPULATION_CELL = 10
MAX_CANDIDATE_CELLS = 120
N_FAVORABLE = 3
N_UNFAVORABLE = 2

HORIZONTAL_COMPARABILITY_MODE = True
HORIZONTAL_REFERENCE = "median"
HORIZONTAL_TOLERANCES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
MIN_COMPARABLE_CELLS = max(20, N_FAVORABLE + N_UNFAVORABLE)

MIN_SELECTED_DISTANCE_M = 5_000   # true metres (UTM)
MIN_SELECTED_DISTANCE_RELAXATION = [1.0, 0.75, 0.50, 0.25, 0.0]

# Plotting
ZONE_COLORS = ["#1b9e77", "#7570b3", "#66a61e", "#d95f02", "#e7298a"]
# CARTO Positron (no labels), as credited in the paper's Data Availability
# statement. CARTO raster basemaps require an API key passed as ?key=...;
# without it every tile is watermarked "API KEY REQUIRED".
CONF_FILE = Path("/home/fbellisardi/code/topolity/conf/conf.json")
CARTO_STYLE = "light_nolabels"


def carto_basemap_url(style: str = CARTO_STYLE) -> str:
    url = f"https://basemaps.cartocdn.com/rastertiles/{style}/{{z}}/{{x}}/{{y}}.png"
    try:
        with open(CONF_FILE) as f:
            key = json.load(f)["api_keys"]["carto"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"[basemap] CARTO API key not found in {CONF_FILE} ({e!r}); "
              "tiles will be watermarked.")
        return url
    return f"{url}?key={key}"


MAP_BASEMAP_PROVIDER = carto_basemap_url()
FONT_TITLE = 26
FONT_LABEL = 24
FONT_TICK = 24
FONT_LEGEND = 18
FONT_BAR_TEXT = 18
FONT_MAP_NUMBER = 22
MAKE_MAPS = True
MAKE_CHARTS = True
MAKE_COMBINED_FIGURE = True   # Fig. 7: map (a) above components (b)


def compute_lambda_from_fuel_params(
    consumption_l_per_100km=(5.0, 8.0),
    energy_mj_per_l=36.0,
    efficiency=0.25,
):
    """lambda = eta * E_l * c, in J/m (mid-range consumption)."""
    c_mean = 0.5 * (consumption_l_per_100km[0] + consumption_l_per_100km[1])
    lambda_mj_per_100km = efficiency * energy_mj_per_l * c_mean
    return lambda_mj_per_100km * 10.0  # MJ/100km -> J/m


LAMBDA_J_PER_M = compute_lambda_from_fuel_params()  # = 585 J/m


# =============================================================================
# Paths and FUA
# =============================================================================

def city_base_dir(city: str) -> Path:
    for root in (DATA_ROOT, ALT_DATA_ROOT):
        if (root / city).exists():
            return root / city
    return DATA_ROOT / city


def city_paths(city: str) -> dict:
    base = city_base_dir(city)
    return {
        "base": base,
        "cells": base / f"{city}_basic_model" / "1000_cells" / "cell_coordinates.csv",
        "od_working": base / f"{city}_basic_model" / "1000_cells" / "od_matrix_working_day.csv",
        "graph": base / "graphs_fine_grid" / "graph_original.pkl",
        "dem": base / "dem" / f"{city}_dem.tif",
        "worldpop": WORLDPOP_ROOT / WORLDPOP_BY_CITY[city],
        "output": OUTPUT_ROOT / city,
    }


def normalize_city_name(x: str) -> str:
    out = str(x).lower().replace("_", " ").replace("-", " ")
    for a, b in [("á", "a"), ("à", "a"), ("é", "e"), ("è", "e"),
                 ("í", "i"), ("ó", "o"), ("ò", "o"), ("ú", "u")]:
        out = out.replace(a, b)
    return out.strip()


def load_city_fua(city: str) -> gpd.GeoDataFrame:
    if not FUA_GPKG.exists():
        raise FileNotFoundError(f"FUA file not found: {FUA_GPKG}")

    fua = gpd.read_file(FUA_GPKG)
    city_norm = normalize_city_name(CITY_NAME_ALIASES.get(city, city))

    name_cols = [c for c in fua.columns
                 if any(k in c.lower() for k in ["name", "city", "fua", "uc"])]
    if not name_cols:
        raise ValueError(f"No name-like columns in FUA file: {list(fua.columns)}")

    sel = np.zeros(len(fua), dtype=bool)
    for col in name_cols:
        sel |= fua[col].astype(str).map(normalize_city_name).str.contains(city_norm, na=False)

    matches = fua[sel].copy()
    if matches.empty:
        raise ValueError(f"No FUA match for '{city}' (normalized '{city_norm}')")

    # largest match, area measured in an equal-area-ish metric CRS
    matches["area_tmp"] = matches.to_crs(matches.estimate_utm_crs()).geometry.area.values
    selected = matches.sort_values("area_tmp", ascending=False).head(1).drop(columns=["area_tmp"])
    selected = selected.to_crs("EPSG:4326")

    print(f"  FUA match: {selected[name_cols].iloc[0].to_dict()}")
    return selected


# =============================================================================
# Cells and population
# =============================================================================

def load_cells(cells_file: Path, metric_crs) -> gpd.GeoDataFrame:
    """Grid cells are stored in EPSG:3857; they are reprojected to the metric CRS
    so that centroids and distances are in true metres."""
    df = pd.read_csv(cells_file)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[box(r.x_min, r.y_min, r.x_max, r.y_max) for r in df.itertuples()],
        crs="EPSG:3857",
    ).to_crs(metric_crs)

    cent = gdf.geometry.centroid
    gdf["centroid_x"] = cent.x.values
    gdf["centroid_y"] = cent.y.values

    side = np.sqrt(gdf.geometry.area)
    print(f"  Cell ground side length: mean {side.mean():.0f} m "
          f"(min {side.min():.0f}, max {side.max():.0f})")
    return gdf


def extract_population_to_cells(cells_gdf, worldpop_file: Path, fua_gdf) -> gpd.GeoDataFrame:
    if not worldpop_file.exists():
        raise FileNotFoundError(f"WorldPop file not found: {worldpop_file}")

    with rasterio.open(worldpop_file) as src:
        cells_pop = cells_gdf.to_crs(src.crs)
        fua_pop = fua_gdf.to_crs(src.crs)
        try:
            fua_geom = fua_pop.geometry.union_all()
        except AttributeError:
            fua_geom = fua_pop.geometry.unary_union

        inside = cells_pop[cells_pop.geometry.intersects(fua_geom)]
        print(f"  Cells inside FUA: {len(inside):,} / {len(cells_gdf):,}")

        populations = {}
        for idx, row in tqdm(inside.iterrows(), total=len(inside), desc="Population per cell"):
            try:
                out_image, _ = mask(src, [row.geometry], crop=True, nodata=0)
                populations[idx] = max(0.0, float(out_image.sum()))
            except Exception:
                populations[idx] = 0.0

    cells = cells_gdf.copy()
    cells["population"] = pd.Series(populations).reindex(cells.index).fillna(0.0).values
    cells = cells[cells["population"] > MIN_POPULATION_CELL].reset_index(drop=True)

    if cells.empty:
        raise ValueError("No populated cells inside FUA after filtering.")

    print(f"  Populated cells: {len(cells):,}; population: {cells['population'].sum():,.0f}")
    return cells


# =============================================================================
# DEM in metric CRS
# =============================================================================

class MetricDEM:
    """DEM reprojected (bilinear) to the metric CRS and sampled bilinearly."""

    def __init__(self, dem_file: Path, metric_crs):
        with rasterio.open(dem_file) as src:
            src_nodata = src.nodata if src.nodata is not None else SRTM_VOID
            transform, width, height = calculate_default_transform(
                src.crs, metric_crs, src.width, src.height, *src.bounds
            )
            arr = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=metric_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        arr[~np.isfinite(arr) | (arr < DEM_MIN_VALID)] = np.nan
        self.arr = arr
        self.inv = ~transform
        print(f"  DEM reprojected: {width}x{height} px, "
              f"{abs(transform.a):.1f} m x {abs(transform.e):.1f} m")

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        inv = self.inv
        # affine gives pixel-corner coordinates; map_coordinates uses pixel centres
        col = inv.a * x + inv.b * y + inv.c - 0.5
        row = inv.d * x + inv.e * y + inv.f - 0.5
        return map_coordinates(self.arr, [row, col], order=1, mode="constant",
                               cval=np.nan, prefilter=False)


# =============================================================================
# Road graph: 3-D length and uphill gain per edge
# =============================================================================

def load_graph(graph_file: Path):
    with open(graph_file, "rb") as f:
        return pickle.load(f)


def iter_edges(G):
    if G.is_multigraph():
        return list(G.edges(keys=True, data=True))
    return [(u, v, None, d) for u, v, d in G.edges(data=True)]


def annotate_edges(G, graph_crs, metric_crs, dem: MetricDEM, ds: float = DS) -> None:
    """Add length_2d, length_3d and dh_up [m] to every edge, in place.

    Each edge polyline is transformed to the metric CRS, resampled every `ds`
    metres along its length, and the DEM is sampled at those points. The edge
    geometry is oriented u -> v before sampling, because uphill gain depends
    on the direction of travel.
    """
    to_m = Transformer.from_crs(graph_crs, metric_crs, always_xy=True)
    edges = iter_edges(G)

    coords = []
    for u, v, _, data in edges:
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        geom = data.get("geometry")
        if geom is None:
            c = np.array([[xu, yu], [xv, yv]], dtype=float)
        else:
            c = np.asarray(geom.coords, dtype=float)[:, :2]
            if np.hypot(c[0, 0] - xu, c[0, 1] - yu) > np.hypot(c[-1, 0] - xu, c[-1, 1] - yu):
                c = c[::-1]
        coords.append(c)

    sizes = np.array([len(c) for c in coords])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.concatenate(coords)
    mx, my = to_m.transform(flat[:, 0], flat[:, 1])

    n_void_edges = 0
    for i, (_, _, _, data) in enumerate(tqdm(edges, desc="Edges: 3-D length and uphill gain")):
        x = mx[offsets[i]:offsets[i + 1]]
        y = my[offsets[i]:offsets[i + 1]]
        seg = np.hypot(np.diff(x), np.diff(y))
        L2 = float(seg.sum())

        if L2 <= 0:
            data["length_2d"] = data["length_3d"] = data["dh_up"] = 0.0
            continue

        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(np.ceil(L2 / ds)) + 1, 2)
        s = np.linspace(0.0, L2, n)
        h = dem.sample(np.interp(s, cum, x), np.interp(s, cum, y))

        dh = np.diff(h)
        bad = ~np.isfinite(dh)
        if bad.any():
            n_void_edges += 1
            dh[bad] = 0.0

        step = L2 / (n - 1)
        data["length_2d"] = L2
        data["length_3d"] = float(np.sqrt(step ** 2 + dh ** 2).sum())
        data["dh_up"] = float(np.clip(dh, 0.0, None).sum())

    print(f"  Edges annotated: {len(edges):,} (with DEM voids: {n_void_edges:,})")


def graph_node_coords(G, graph_crs, metric_crs):
    nodes = list(G.nodes())
    x = np.array([G.nodes[n]["x"] for n in nodes], dtype=float)
    y = np.array([G.nodes[n]["y"] for n in nodes], dtype=float)
    mx, my = Transformer.from_crs(graph_crs, metric_crs, always_xy=True).transform(x, y)
    return nodes, np.column_stack([mx, my])


def assign_nearest_nodes(cells, nodes, node_xy) -> gpd.GeoDataFrame:
    dist, idx = KDTree(node_xy).query(cells[["centroid_x", "centroid_y"]].to_numpy(), k=1)
    cells = cells.copy()
    cells["nearest_node"] = [nodes[i] for i in idx]
    cells["node_distance"] = dist
    print(f"  Cell centroid -> nearest node: mean {dist.mean():.0f} m, max {dist.max():.0f} m")
    return cells


def edge_cost_terms(G, u, v):
    """(horizontal length, uphill gain) of the edge u->v actually used by
    Dijkstra, i.e. the parallel edge with minimum routing weight."""
    data = G.get_edge_data(u, v)
    if G.is_multigraph():
        data = min(data.values(), key=lambda d: d.get(ROUTING_WEIGHT, np.inf))
    return data[HORIZONTAL_LENGTH_ATTR], data["dh_up"]


def tree_sums(pred: dict, source, targets, edge_fn) -> dict:
    """Cumulative (length, uphill gain) from `source` to each target along the
    shortest-path tree `pred`, memoised so shared prefixes are summed once.
    edge_fn(parent, child) returns the costs of the tree edge parent->child."""
    cache = {source: (0.0, 0.0)}
    out = {}
    for t in targets:
        if t in out:
            continue
        if t not in pred:
            out[t] = None
            continue
        stack, n = [], t
        while n not in cache:
            stack.append(n)
            n = pred[n][0]
        L, H = cache[n]
        while stack:
            m = stack.pop()
            l, h = edge_fn(pred[m][0], m)
            L += l
            H += h
            cache[m] = (L, H)
        out[t] = cache[t]
    return out


# =============================================================================
# Gravity model: fit of d_0 on empirical flows
# =============================================================================

def fit_d0_production_constrained(od_file: Path, cells, D: np.ndarray, alpha: float = ALPHA) -> dict:
    """Fit d_0 of Eq. 4 on empirical working-day OD flows.

    Poisson PML with origin fixed effects; the fixed effects are concentrated
    out analytically, which leaves the multinomial likelihood of the
    production-constrained model
        p_OD = P_D^alpha exp(-d_OD/d_0) / sum_{K != O} P_K^alpha exp(-d_OK/d_0).
    Zero-flow pairs enter through the normalisation over all destinations.
    Standard errors are cluster-robust by origin (sandwich estimator).
    """
    od = pd.read_csv(od_file)
    index = {cid: i for i, cid in enumerate(cells["cell_id"].values)}
    o = od["cell_origin"].map(index)
    d = od["cell_destination"].map(index)
    ok = (o.notna() & d.notna()).to_numpy()

    o = o[ok].astype(int).to_numpy()
    d = d[ok].astype(int).to_numpy()
    F = od.loc[ok, "count"].to_numpy(dtype=float)
    keep = (o != d) & (F > 0)
    o, d, F = o[keep], d[keep], F[keep]
    if len(F) == 0:
        raise ValueError("No usable OD flows between populated cells.")

    logP = alpha * np.log(cells["population"].to_numpy(dtype=float))
    origins, row = np.unique(o, return_inverse=True)
    Dsub = D[origins]
    self_mask = np.zeros(Dsub.shape, dtype=bool)
    self_mask[np.arange(len(origins)), origins] = True

    Fo = np.bincount(row, weights=F, minlength=len(origins))
    FD = np.bincount(row, weights=F * D[o, d], minlength=len(origins))
    const = float(np.sum(F * logP[d]))

    def logits(beta):
        z = logP[None, :] - beta * Dsub
        z[self_mask] = -np.inf
        return z

    def neg_ll(log_d0):
        beta = np.exp(-log_d0)
        lse = logsumexp(logits(beta), axis=1)
        return -(const - beta * FD.sum() - np.sum(Fo * lse))

    lo, hi = np.log(200.0), np.log(200_000.0)
    res = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    if not res.success or abs(res.x - lo) < 1e-3 or abs(res.x - hi) < 1e-3:
        raise ValueError(f"d_0 fit did not converge inside bounds (log d_0 = {res.x:.3f}).")

    beta = float(np.exp(-res.x))
    z = logits(beta)
    p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    mean_d = (p * Dsub).sum(axis=1)
    var_d = (p * Dsub ** 2).sum(axis=1) - mean_d ** 2
    score = -FD + Fo * mean_d                 # d ll_O / d beta
    hess = -float(np.sum(Fo * var_d))         # d2 ll / d beta2
    se_beta = float(np.sqrt(np.sum(score ** 2)) / abs(hess))

    return {
        "d_0": 1.0 / beta,
        "d_0_se": se_beta / beta ** 2,
        "n_pairs_positive": int(len(F)),
        "n_origins": int(len(origins)),
        "n_trips": float(F.sum()),
        "z_beta": beta / se_beta,
    }


# =============================================================================
# Densification cost of one candidate cell
# =============================================================================

def select_candidate_cells(cells, max_candidates: int) -> gpd.GeoDataFrame:
    c = cells.copy()
    cx = np.average(c["centroid_x"], weights=c["population"])
    cy = np.average(c["centroid_y"], weights=c["population"])
    c["dist_to_center"] = np.hypot(c["centroid_x"] - cx, c["centroid_y"] - cy)

    high_pop = c[c["population"] >= c["population"].quantile(0.65)]
    central = c[c["dist_to_center"] <= c["dist_to_center"].quantile(0.50)]
    peripheral = c[c["dist_to_center"] >= c["dist_to_center"].quantile(0.75)]

    sel = pd.concat([
        high_pop.nlargest(max_candidates // 2, "population"),
        central.nlargest(max_candidates // 4, "population"),
        peripheral.nlargest(max_candidates // 4, "population"),
    ]).drop_duplicates(subset=["cell_id"])

    if len(sel) > max_candidates:
        sel = sel.nlargest(max_candidates, "population")
    return sel  # keeps the original index of `cells`


def compute_densification_cost_for_cell(cand_idx: int, cells, D, G, G_rev, d_0):
    populations = cells["population"].to_numpy(dtype=float)
    attraction = populations ** ALPHA * np.exp(-D[cand_idx] / d_0)
    attraction[cand_idx] = 0.0
    if attraction.sum() <= 0:
        return None
    p = attraction / attraction.sum()

    dests = np.flatnonzero(p > GRAVITY_DESTINATION_THRESHOLD)
    dest_nodes = cells["nearest_node"].to_numpy()[dests]
    source = cells.at[cand_idx, "nearest_node"]

    pred_out, _ = nx.dijkstra_predecessor_and_distance(G, source, weight=ROUTING_WEIGHT)
    pred_ret, _ = nx.dijkstra_predecessor_and_distance(G_rev, source, weight=ROUTING_WEIGHT)

    out = tree_sums(pred_out, source, dest_nodes, lambda a, b: edge_cost_terms(G, a, b))
    # in the reversed graph a->b corresponds to the original edge b->a
    ret = tree_sums(pred_ret, source, dest_nodes, lambda a, b: edge_cost_terms(G, b, a))

    p_routed = 0.0
    len_w = 0.0
    gain_w = 0.0
    failed = 0
    for j, node in zip(dests, dest_nodes):
        a, b = out.get(node), ret.get(node)
        if a is None or b is None:
            failed += 1
            continue
        p_routed += p[j]
        len_w += p[j] * (a[0] + b[0])
        gain_w += p[j] * (a[1] + b[1])

    if p_routed <= 0:
        return None

    n_round_trips = NEW_RESIDENTS * ROUND_TRIPS_PER_PERSON_PER_DAY
    mean_rt_length = len_w / p_routed          # per round trip [m]
    mean_rt_gain = gain_w / p_routed           # per round trip [m]

    W_hor = LAMBDA_J_PER_M * n_round_trips * mean_rt_length
    W_alt = M_PHYS_KG * G_PHYS * n_round_trips * mean_rt_gain
    W_tot = W_hor + W_alt

    return {
        "cell_id": cells.at[cand_idx, "cell_id"],
        "centroid_x": cells.at[cand_idx, "centroid_x"],
        "centroid_y": cells.at[cand_idx, "centroid_y"],
        "baseline_population": cells.at[cand_idx, "population"],
        "new_residents_added": NEW_RESIDENTS,
        "round_trips": n_round_trips,
        "n_destinations_routed": int(len(dests) - failed),
        "n_destinations_failed": int(failed),
        "prob_share_above_threshold": float(p[dests].sum()),
        "prob_share_routed": float(p_routed),
        "mean_round_trip_length_m": mean_rt_length,
        "mean_round_trip_uphill_m": mean_rt_gain,
        "vertical_work_joules": W_alt,
        "horizontal_cost_joules": W_hor,
        "total_work_joules": W_tot,
        "work_per_resident": W_tot / NEW_RESIDENTS,
        "vertical_per_resident": W_alt / NEW_RESIDENTS,
        "horizontal_per_resident": W_hor / NEW_RESIDENTS,
        "vertical_share_pct": 100 * W_alt / W_tot,
        "horizontal_share_pct": 100 * W_hor / W_tot,
    }


# =============================================================================
# Selection of favourable / unfavourable cells
# =============================================================================

def _greedy_select_spaced(df, n, ascending, already, min_dist_m):
    chosen = []
    for _, row in df.sort_values("vertical_work_joules", ascending=ascending).iterrows():
        r = row.to_dict()
        if min_dist_m > 0 and any(
            np.hypot(r["centroid_x"] - s["centroid_x"], r["centroid_y"] - s["centroid_y"]) < min_dist_m
            for s in already + chosen
        ):
            continue
        chosen.append(r)
        if len(chosen) >= n:
            break
    return chosen


def select_favorable_unfavorable(results_df: pd.DataFrame) -> pd.DataFrame:
    """Lowest / highest altitudinal work among candidates whose horizontal cost
    is within a tolerance of the median, with a minimum spacing between cells."""
    df = results_df.copy()
    h_ref, tol_used, comparable = np.nan, np.nan, df

    if HORIZONTAL_COMPARABILITY_MODE:
        h = df["horizontal_cost_joules"]
        h_ref = (df.loc[df["total_work_joules"].idxmin(), "horizontal_cost_joules"]
                 if HORIZONTAL_REFERENCE == "best" else h.median())
        for tol in HORIZONTAL_TOLERANCES:
            tmp = df[(h >= h_ref * (1 - tol)) & (h <= h_ref * (1 + tol))]
            if len(tmp) >= MIN_COMPARABLE_CELLS:
                comparable, tol_used = tmp, tol
                break

    selected_rows, used_distance = [], 0.0
    for relax in MIN_SELECTED_DISTANCE_RELAXATION:
        min_dist = MIN_SELECTED_DISTANCE_M * relax
        fav = _greedy_select_spaced(comparable, N_FAVORABLE, True, [], min_dist)
        unf = _greedy_select_spaced(comparable, N_UNFAVORABLE, False, fav, min_dist)
        if len(fav) == N_FAVORABLE and len(unf) == N_UNFAVORABLE:
            selected_rows, used_distance = fav + unf, min_dist
            break

    sel = pd.DataFrame(selected_rows)
    sel["scenario_type"] = ["favorable"] * N_FAVORABLE + ["unfavorable"] * N_UNFAVORABLE
    sel = sel.sort_values("vertical_work_joules").reset_index(drop=True)
    sel["rank_total"] = np.arange(1, len(sel) + 1)
    sel["zone_label"] = [f"Favorable {i + 1}" if i < N_FAVORABLE
                         else f"Unfavorable {i - N_FAVORABLE + 1}" for i in range(len(sel))]
    sel["zone_color"] = ZONE_COLORS[:len(sel)]
    sel["horizontal_reference_joules"] = h_ref
    sel["horizontal_tolerance_used"] = tol_used
    sel["min_selected_distance_used_m"] = used_distance
    sel["horizontal_cost_relative_to_ref_pct"] = (
        (sel["horizontal_cost_joules"] / h_ref - 1.0) * 100 if np.isfinite(h_ref) else np.nan
    )

    print("\n[selection diagnostic]")
    print(f"  horizontal reference (median): {h_ref / 1e9:,.2f} GJ")
    print(f"  tolerance used: {tol_used}")
    print(f"  comparable cells: {len(comparable)} / {len(df)}")
    print(f"  min selected distance used: {used_distance:,.0f} m")
    return sel


# =============================================================================
# Plots
# =============================================================================

def make_city_plots(city, cells, selected_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = DISPLAY_NAMES.get(city, city.title())

    if MAKE_MAPS:
        sel_cells = cells[cells["cell_id"].isin(selected_df["cell_id"])].merge(
            selected_df[["cell_id", "zone_color", "rank_total"]], on="cell_id")
        cells_ll = cells.to_crs("EPSG:4326")
        sel_ll = sel_cells.to_crs("EPSG:4326")

        fig, ax = plt.subplots(figsize=(12, 10))
        cells_ll.plot(ax=ax, facecolor="#e0e0e0", edgecolor="none", alpha=0.25)
        for _, row in sel_ll.iterrows():
            gpd.GeoSeries([row.geometry], crs=sel_ll.crs).plot(
                ax=ax, facecolor=row["zone_color"], edgecolor="black",
                alpha=0.45, linewidth=2, zorder=10)
            c = row.geometry.centroid
            ax.scatter(c.x, c.y, s=850, color=row["zone_color"], edgecolor="black",
                       linewidth=2, zorder=20)
            ax.text(c.x, c.y, str(int(row["rank_total"])), ha="center", va="center",
                    fontsize=FONT_MAP_NUMBER, weight="bold", zorder=30)

        minx, miny, maxx, maxy = sel_ll.total_bounds
        pad_x = (maxx - minx) * 0.25 or 0.01
        pad_y = (maxy - miny) * 0.25 or 0.01
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        try:
            ctx.add_basemap(ax, crs=cells_ll.crs, source=MAP_BASEMAP_PROVIDER,
                            attribution=False, zorder=1)
        except Exception as e:
            print(f"[{city}] Could not add basemap: {e}")

        ax.legend(handles=[Patch(facecolor=r["zone_color"], edgecolor="black",
                                 label=f"{int(r['rank_total'])}. {r['zone_label']}")
                           for _, r in selected_df.iterrows()],
                  loc="lower right", frameon=True, fontsize=FONT_LEGEND)
        ax.set_title(name, fontsize=FONT_TITLE)
        ax.set_xlabel("Longitude", fontsize=FONT_LABEL)
        ax.set_ylabel("Latitude", fontsize=FONT_LABEL)
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"{city}_selected_densification_map.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    if MAKE_CHARTS:
        df = selected_df.reset_index(drop=True)
        x = np.arange(len(df))
        colors = df["zone_color"].tolist()
        v_gj = df["vertical_work_joules"] / 1e9
        h_gj = df["horizontal_cost_joules"] / 1e9
        t_gj = df["total_work_joules"] / 1e9

        def finish(ax, ylabel, fname, title=True):
            ax.set_xticks(x)
            ax.set_xticklabels(df["zone_label"], rotation=20, fontsize=FONT_TICK)
            ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            if title:
                ax.set_title(name, fontsize=FONT_TITLE)
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.figure.tight_layout()
            for ext in ("png", "pdf"):
                ax.figure.savefig(output_dir / f"{city}_{fname}.{ext}", dpi=300, bbox_inches="tight")
            plt.close(ax.figure)

        # Fig. 7b: altitudinal (solid) + horizontal (transparent)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(x, v_gj, color=colors, edgecolor="black", linewidth=1.2, alpha=0.95)
        ax.bar(x, h_gj, bottom=v_gj, color=colors, edgecolor="black", linewidth=1.2, alpha=0.35)
        ax.legend(handles=[
            Patch(facecolor="gray", edgecolor="black", alpha=0.95, label="Altitudinal"),
            Patch(facecolor="gray", edgecolor="black", alpha=0.35, label="Horizontal"),
        ], loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0,
            fontsize=FONT_LEGEND)
        finish(ax, "Additional mobility energy (GJ)", "densification_components", title=False)

        # Total energy
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(x, t_gj, color=colors, edgecolor="black", linewidth=1.5, alpha=0.9)
        for i, r in df.iterrows():
            ax.text(i, t_gj[i] * 1.02, f"{t_gj[i]:.1f} GJ\n{r['work_per_resident'] / 1e6:.1f} MJ/res",
                    ha="center", va="bottom", fontsize=FONT_BAR_TEXT)
        finish(ax, "Total additional mobility energy (GJ)", "densification_total_energy")

        # Altitudinal energy only
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(x, v_gj, color=colors, edgecolor="black", linewidth=1.5, alpha=0.9)
        for i, r in df.iterrows():
            ax.text(i, v_gj[i] * 1.03, f"{v_gj[i]:.1f} GJ\n{r['vertical_per_resident'] / 1e6:.2f} MJ/res",
                    ha="center", va="bottom", fontsize=FONT_BAR_TEXT)
        finish(ax, "Altitudinal mobility energy (GJ)", "densification_vertical_energy")



def make_combined_figure(city, cells, selected_df, output_dir: Path):
    """Paper Fig. 7: (a) map of the selected cells, (b) energy components,
    stacked vertically at single-column width."""
    name = DISPLAY_NAMES.get(city, city.title())
    df = selected_df.reset_index(drop=True)
    colors = df["zone_color"].tolist()

    sel = (cells[cells["cell_id"].isin(df["cell_id"])]
           .merge(df[["cell_id", "zone_color", "rank_total"]], on="cell_id")
           .to_crs("EPSG:3857"))

    fig = plt.figure(figsize=(3.4, 4.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.32)
    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ---- (a) map --------------------------------------------------------
    for _, row in sel.iterrows():
        c = row.geometry.centroid
        ax_map.text(
            c.x, c.y, str(int(row["rank_total"])),
            ha="center", va="center", fontsize=6.5, weight="bold", zorder=30, clip_on=True,
            bbox=dict(boxstyle="square,pad=0.3", facecolor=row["zone_color"],
                      edgecolor="black", linewidth=0.6, alpha=0.85),
        )

    # Extent: selected cells + padding, enlarged along one axis so that the
    # map fills the panel with an undistorted (equal) aspect ratio.
    minx, miny, maxx, maxy = sel.total_bounds
    pad = max(0.20 * max(maxx - minx, maxy - miny), 2_000.0)
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
    w, h = (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad
    bbox = ax_map.get_position()
    fig_w, fig_h = fig.get_size_inches()
    box_ratio = (bbox.height * fig_h) / (bbox.width * fig_w)
    if h / w < box_ratio:
        h = w * box_ratio
    else:
        w = h / box_ratio
    ax_map.set_xlim(cx - w / 2, cx + w / 2)
    ax_map.set_ylim(cy - h / 2, cy + h / 2)
    ax_map.set_aspect("equal", adjustable="box")
    try:
        ctx.add_basemap(ax_map, crs="EPSG:3857", source=MAP_BASEMAP_PROVIDER,
                        attribution=False, zorder=1)
    except Exception as e:
        print(f"[{city}] Could not add basemap: {e}")

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.text(0.03, 0.97, name, transform=ax_map.transAxes, ha="left", va="top",
                fontsize=11, weight="bold", zorder=40)
    ax_map.legend(
        handles=[Patch(facecolor=r["zone_color"], edgecolor="black", linewidth=0.6,
                       label=f"{int(r['rank_total'])}. {r['zone_label']}")
                 for _, r in df.iterrows()],
        loc="lower right", fontsize=5.5, frameon=True, handlelength=1.6,
        borderpad=0.4, labelspacing=0.3,
    )

    # ---- (b) components -------------------------------------------------
    x = np.arange(len(df))
    v_gj = df["vertical_work_joules"].to_numpy() / 1e9
    h_gj = df["horizontal_cost_joules"].to_numpy() / 1e9
    ax_bar.bar(x, v_gj, color=colors, edgecolor="black", linewidth=0.6, alpha=0.95)
    ax_bar.bar(x, h_gj, bottom=v_gj, color=colors, edgecolor="black", linewidth=0.6, alpha=0.35)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(df["zone_label"], rotation=20, ha="right",
                           rotation_mode="anchor", fontsize=6.5)
    ax_bar.set_ylabel("Additional mobility energy (GJ)", fontsize=6.5)
    ax_bar.tick_params(axis="y", labelsize=6.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(
        handles=[Patch(facecolor="gray", edgecolor="black", alpha=0.95, label="Altitudinal"),
                 Patch(facecolor="gray", edgecolor="black", alpha=0.35, label="Horizontal")],
        loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0,
        fontsize=6, frameon=True,
    )

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{city}_fig7_densification.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def run_city(city: str):
    print(f"\nRUNNING CITY: {city.upper()}")
    paths = city_paths(city)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["cells", "od_working", "graph", "dem", "worldpop"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"[{city}] Missing {key}: {paths[key]}")

    fua = load_city_fua(city)
    metric_crs = fua.estimate_utm_crs()
    print(f"  Metric CRS: {metric_crs.to_string()}")

    cells = load_cells(paths["cells"], metric_crs)
    cells = extract_population_to_cells(cells, paths["worldpop"], fua)

    G = load_graph(paths["graph"])
    graph_crs = G.graph.get("crs", "EPSG:4326")
    dem = MetricDEM(paths["dem"], metric_crs)
    annotate_edges(G, graph_crs, metric_crs, dem)
    G_rev = G.reverse(copy=False)

    nodes, node_xy = graph_node_coords(G, graph_crs, metric_crs)
    cells = assign_nearest_nodes(cells, nodes, node_xy)

    D = cdist(cells[["centroid_x", "centroid_y"]].to_numpy(),
              cells[["centroid_x", "centroid_y"]].to_numpy())

    fit = fit_d0_production_constrained(paths["od_working"], cells, D)
    d_0 = fit["d_0"]
    print(f"[{city}] d_0 = {d_0 / 1000:.2f} km (robust s.e. {fit['d_0_se'] / 1000:.2f} km, "
          f"z = {fit['z_beta']:.1f}; {fit['n_pairs_positive']:,} positive OD pairs, "
          f"{fit['n_origins']:,} origins, {fit['n_trips']:,.0f} trips)")

    candidates = select_candidate_cells(cells, MAX_CANDIDATE_CELLS)
    print(f"[{city}] Candidate cells evaluated: {len(candidates):,}")

    results = []
    for cand_idx in tqdm(candidates.index, desc=f"{city} candidate cells"):
        res = compute_densification_cost_for_cell(cand_idx, cells, D, G, G_rev, d_0)
        if res is not None:
            results.append(res)
    if not results:
        raise RuntimeError(f"[{city}] No valid candidate results.")

    results_df = pd.DataFrame(results).sort_values("total_work_joules")
    best = results_df["total_work_joules"].min()
    results_df["work_increase_pct"] = (results_df["total_work_joules"] / best - 1.0) * 100

    print(f"[{city}] Gravity mass above threshold: "
          f"{results_df['prob_share_above_threshold'].min():.3f}–"
          f"{results_df['prob_share_above_threshold'].max():.3f} (renormalised)")

    selected_df = select_favorable_unfavorable(results_df)

    for df in (results_df, selected_df):
        df["city"] = city
        df["d_0_m"] = d_0

    results_df.to_csv(output_dir / f"{city}_all_candidate_densification_results.csv", index=False, sep=";")
    selected_df.to_csv(output_dir / f"{city}_selected_favorable_unfavorable_cells.csv", index=False, sep=";")

    sel_gdf = cells[cells["cell_id"].isin(selected_df["cell_id"])].merge(
        selected_df.drop(columns=["centroid_x", "centroid_y"]), on="cell_id", how="left")
    sel_gdf.to_file(output_dir / f"{city}_selected_favorable_unfavorable_cells.gpkg", driver="GPKG")

    make_city_plots(city, cells, selected_df, output_dir)
    if MAKE_COMBINED_FIGURE:
        make_combined_figure(city, cells, selected_df, output_dir)

    print(f"\n[{city}] Selected cells:")
    print(selected_df[["zone_label", "cell_id", "horizontal_cost_relative_to_ref_pct",
                       "horizontal_cost_joules", "vertical_work_joules", "total_work_joules"]]
          .assign(horizontal_cost_joules=lambda t: t["horizontal_cost_joules"] / 1e9,
                  vertical_work_joules=lambda t: t["vertical_work_joules"] / 1e9,
                  total_work_joules=lambda t: t["total_work_joules"] / 1e9)
          .rename(columns={"horizontal_cost_joules": "W_hor [GJ]",
                           "vertical_work_joules": "W_alt [GJ]",
                           "total_work_joules": "W_tot [GJ]"})
          .to_string(index=False, float_format="%.2f"))
    print(f"[{city}] Saved to: {output_dir}")

    return results_df, selected_df


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-city terrain-aware densification analysis")
    parser.add_argument("--city", choices=CITIES, help="Run a single city")
    return parser.parse_args()


def main():
    args = parse_args()
    cities = [args.city] if args.city else CITIES

    print("MULTI-CITY TERRAIN-AWARE DENSIFICATION ANALYSIS USING FUA")
    print(f"New residents per test cell: {NEW_RESIDENTS:,}")
    print(f"Round trips per resident per day: {ROUND_TRIPS_PER_PERSON_PER_DAY}")
    print(f"lambda = {LAMBDA_J_PER_M:.1f} J/m, m = {M_PHYS_KG} kg, g = {G_PHYS} m/s^2")
    print(f"Routing weight: {ROUTING_WEIGHT}; DEM sampling step: {DS} m")

    all_results, all_selected = [], []
    for city in cities:
        try:
            r, s = run_city(city)
            all_results.append(r)
            all_selected.append(s)
        except Exception as e:
            warnings.warn(f"[{city}] failed: {e}")

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            OUTPUT_ROOT / "all_cities_candidate_densification_results.csv", index=False, sep=";")
    if all_selected:
        pd.concat(all_selected, ignore_index=True).to_csv(
            OUTPUT_ROOT / "all_cities_selected_favorable_unfavorable_cells.csv", index=False, sep=";")

    print(f"\nDone. All files saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()