#!/usr/bin/env python3
"""Origin-destination flows of Barcelona and Madrid from the MITMA open mobility study (SI Note 9).

The Spanish Ministry of Transport publishes daily trip matrices between districts derived from mobile
phone records ("Estudio de movilidad con Big Data", basic study, district zoning, 2022 onward).
The files carry no transport mode; modes are assigned later from the travel surveys (config files).

This script turns the district matrices into flows between the 1 km cells used by the multimodal
analyses, in the same format as the cell OD files of the main pipeline:

1. average daily trips T_DE over the selected weekdays, all hours, purposes and traveller profiles,
   for districts D, E that intersect the bounding box of the city network;
2. population weights w_cD = P(c and D) / P(D) from WorldPop 2020 (100 m pixels, assigned by their
   centre), counting only cells that contain at least one network node; the share of a district that
   lies outside the network is dropped;
3. cell flows F_cc' = sum_DE w_cD T_DE w_c'E, i.e. trip ends are spread over the populated cells of
   each district in proportion to population, independently at the two ends;
4. the dense product has several million pairs, so it is thinned without bias: pairs with flow
   above a threshold tau are kept, smaller ones are kept with probability F/tau and given flow tau
   (tau is chosen so that about `--max-pairs` pairs remain; seeded). Truncating the smallest flows
   instead would drop mostly long trips and bias the distance distribution.

Two variants are written: all trips (MITMA counts trips of 0.5 km and more), and trips of 2 km and
more, i.e. without the 0.5-2 km distance band, where walking dominates (a coarse proxy of motorised
trips; MITMA has no mode).

Output (new directory, nothing is overwritten):
    data/data_processed/<city>/<city>_mitma_model/1000_cells/od_matrix_mitma_weekday{,_ge2km}.csv
    (cell_origin, cell_destination, count) and mitma_od_meta{,_ge2km}.json.

Usage:
    python python/mitma_od.py                      # both cities, October 2022 weekdays
    python python/mitma_od.py --cities madrid --max-pairs 1000000
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import sparse
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MITMA_DIR = DATA / "green_mobility" / "od_raw"
DISTRICTS = MITMA_DIR / "zonificacion_distritos.shp"
WORLDPOP = DATA / "worldpop" / "raw" / "2020" / "esp_ppp_2020.tif"
CELLS_CRS = "EPSG:3857"
CELL_SIZE_M = 1000.0
CITIES = ("barcelone", "madrid")
# Weekdays of October 2022 without 12 Oct (national holiday) and 31 Oct (bridge before 1 Nov).
EXCLUDED_DAYS = {date(2022, 10, 12), date(2022, 10, 31)}


def october_2022_weekdays() -> list[date]:
    days = [date(2022, 10, 1) + timedelta(i) for i in range(31)]
    return [d for d in days if d.weekday() < 5 and d not in EXCLUDED_DAYS]


def city_dir(city: str) -> Path:
    return DATA / "data_processed" / city


def output_dir(city: str) -> Path:
    return city_dir(city) / f"{city}_mitma_model" / "1000_cells"


def network_nodes(city: str) -> tuple[np.ndarray, np.ndarray]:
    with open(city_dir(city) / "graphs_fine_grid" / "graph_original.pkl", "rb") as f:
        G = pickle.load(f)
    lon = np.fromiter((d["x"] for _, d in G.nodes(data=True)), float)
    lat = np.fromiter((d["y"] for _, d in G.nodes(data=True)), float)
    return lon, lat


class CellGrid:
    """The regular 1 km grid of cell_coordinates.csv (EPSG:3857 bounds)."""

    def __init__(self, cells: pd.DataFrame):
        self.x0, self.y0 = cells["x_min"].min(), cells["y_min"].min()
        ix = np.rint((cells["x_min"] - self.x0) / CELL_SIZE_M).astype(int)
        iy = np.rint((cells["y_min"] - self.y0) / CELL_SIZE_M).astype(int)
        self.key = {(i, j): c for i, j, c in zip(ix, iy, cells["cell_id"])}

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.floor((x - self.x0) / CELL_SIZE_M).astype(int)
        iy = np.floor((y - self.y0) / CELL_SIZE_M).astype(int)
        return np.array([self.key.get((i, j)) for i, j in zip(ix, iy)], dtype=object)


def district_cell_weights(city: str, districts: gpd.GeoDataFrame):
    """Sparse weights W (district x cell) with W[D, c] = P(c and D) / P(D), cells with network nodes only."""
    cells = pd.read_csv(city_dir(city) / f"{city}_basic_model" / "1000_cells" / "cell_coordinates.csv")
    grid = CellGrid(cells)
    lon, lat = network_nodes(city)
    to_3857 = Transformer.from_crs("EPSG:4326", CELLS_CRS, always_xy=True)
    node_cells = set(grid.lookup(*to_3857.transform(lon, lat))) - {None}

    with rasterio.open(WORLDPOP) as src:
        d = districts.to_crs(src.crs)
        win = from_bounds(*d.total_bounds, transform=src.transform).round_offsets().round_lengths()
        pop = src.read(1, window=win).astype(float)
        tr = src.window_transform(win)
        pop[(pop < 0) | ~np.isfinite(pop) | (pop == src.nodata)] = 0.0
    dist_idx = rasterize(((g, i + 1) for i, g in enumerate(d.geometry)), out_shape=pop.shape, transform=tr,
                         fill=0, dtype="int32")
    rr, cc = np.nonzero((dist_idx > 0) & (pop > 0))
    xs, ys = rasterio.transform.xy(tr, rr, cc, offset="center")
    px_cell = grid.lookup(*to_3857.transform(np.asarray(xs), np.asarray(ys)))
    df = pd.DataFrame({"district": dist_idx[rr, cc] - 1, "cell": px_cell, "pop": pop[rr, cc]})
    P_D = df.groupby("district")["pop"].sum()
    df = df[df["cell"].isin(node_cells)]
    pc_ = df.groupby(["district", "cell"], as_index=False)["pop"].sum()
    cell_ids = np.array(sorted(pc_["cell"].unique()))
    col = {c: i for i, c in enumerate(cell_ids)}
    w = pc_["pop"].to_numpy() / P_D.loc[pc_["district"]].to_numpy()
    W = sparse.csr_matrix((w, (pc_["district"].to_numpy(), pc_["cell"].map(col).to_numpy())),
                          shape=(len(d), len(cell_ids)))
    coverage = pd.Series(np.asarray(W.sum(axis=1)).ravel(), index=districts["ID"].to_numpy())
    return W, cell_ids, coverage, P_D


VARIANTS = {"": None, "_ge2km": {"0.5-2"}}     # file suffix -> excluded MITMA distance bands


def read_day(path: Path, ids: set[str]) -> pd.DataFrame:
    """Trips between the selected districts in one daily file by distance band, summed over the rest."""
    opts = pacsv.ConvertOptions(include_columns=["origen", "destino", "distancia", "viajes"],
                                column_types={"origen": pa.string(), "destino": pa.string(),
                                              "distancia": pa.string(), "viajes": pa.float64()})
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="|"), convert_options=opts)
    vs = pa.array(sorted(ids))
    t = t.filter(pc.and_(pc.is_in(t["origen"], value_set=vs), pc.is_in(t["destino"], value_set=vs)))
    return t.to_pandas().groupby(["origen", "destino", "distancia"], as_index=False)["viajes"].sum()


def disaggregate(T: sparse.spmatrix, W: sparse.spmatrix) -> sparse.coo_matrix:
    """Cell flows F = W^T T W (district trips spread over cells by population, independently at both ends)."""
    return (W.T @ T @ W).tocoo()


def thinning_threshold(f: np.ndarray, max_pairs: int) -> float:
    """Smallest tau whose expected number of kept pairs, #(f >= tau) + sum_{f < tau} f / tau, is <= max_pairs."""
    if len(f) <= max_pairs:
        return 0.0
    lo, hi = 0.0, float(f.max())
    for _ in range(100):
        tau = 0.5 * (lo + hi)
        n = np.count_nonzero(f >= tau) + f[f < tau].sum() / tau
        lo, hi = (tau, hi) if n > max_pairs else (lo, tau)
    return hi


def thin(F: sparse.coo_matrix, max_pairs: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unbiased thinning: every pair keeps its expected flow, E[kept flow] = F."""
    f = F.data
    tau = thinning_threshold(f, max_pairs)
    if tau == 0.0:
        return F.row, F.col, f
    small = f < tau
    keep = ~small | (np.random.default_rng(seed).random(len(f)) < f / tau)
    return F.row[keep], F.col[keep], np.where(small, tau, f)[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--max-pairs", type=int, default=800_000, help="approximate number of cell pairs kept")
    ap.add_argument("--max-days", type=int, default=None, help="use only the first N days (testing)")
    args = ap.parse_args()

    days = october_2022_weekdays()[:args.max_days]
    files = [MITMA_DIR / f"{d:%Y-%m}" / f"{d:%Y%m%d}_Viajes_distritos.csv.gz" for d in days]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing MITMA files: {missing}")

    all_d = gpd.read_file(DISTRICTS)
    sel = {}
    for city in args.cities:
        lon, lat = network_nodes(city)
        bb = gpd.GeoSeries([box(lon.min(), lat.min(), lon.max(), lat.max())], crs="EPSG:4326").to_crs(all_d.crs)
        sel[city] = all_d[all_d.intersects(bb.iloc[0])].reset_index(drop=True)
        print(f"[{city}] {len(sel[city])} districts intersect the network bounding box")

    ids = set().union(*(set(s["ID"]) for s in sel.values()))
    daily = []
    for f in files:
        daily.append(read_day(f, ids))
        print(f"  read {f.name}: {daily[-1]['viajes'].sum():,.0f} trips between selected districts")
    trips = pd.concat(daily).groupby(["origen", "destino", "distancia"], as_index=False)["viajes"].sum()
    trips["viajes"] /= len(files)
    band_share = (trips.groupby("distancia")["viajes"].sum() / trips["viajes"].sum()).round(4).to_dict()

    for city, suffix in ((c, s) for c in args.cities for s in VARIANTS):
        d = sel[city]
        index = {z: i for i, z in enumerate(d["ID"])}
        t = trips[trips["origen"].isin(index) & trips["destino"].isin(index)]
        if VARIANTS[suffix]:
            t = t[~t["distancia"].isin(VARIANTS[suffix])]
        t = t.groupby(["origen", "destino"], as_index=False)["viajes"].sum()
        T = sparse.csr_matrix((t["viajes"], (t["origen"].map(index), t["destino"].map(index))),
                              shape=(len(d), len(d)))
        W, cell_ids, coverage, _ = district_cell_weights(city, d)
        F = disaggregate(T, W)
        o, de, f = thin(F, args.max_pairs)
        out = output_dir(city)
        out.mkdir(parents=True, exist_ok=True)
        od = pd.DataFrame({"cell_origin": cell_ids[o], "cell_destination": cell_ids[de], "count": f})
        od.to_csv(out / f"od_matrix_mitma_weekday{suffix}.csv", index=False)
        meta = {
            "source": "MITMA, Estudio de movilidad con Big Data, basic study, Viajes_distritos (open data)",
            "days": [str(x) for x in days], "excluded_distance_bands": sorted(VARIANTS[suffix] or []),
            "distance_band_shares_all_selected_districts": band_share, "max_pairs": args.max_pairs,
            "n_districts": len(d), "n_cells_with_nodes_and_population": len(cell_ids),
            "district_trips_per_day": float(T.sum()),
            "cell_trips_per_day": float(F.data.sum()), "cell_trips_after_thinning": float(f.sum()),
            "retained_share_of_district_trips": float(F.data.sum() / T.sum()),
            "n_cell_pairs_dense": int(F.nnz), "n_cell_pairs_kept": int(len(f)),
            "n_origin_cells_kept": int(len(np.unique(o))),
            "intra_district_share_of_trips": float(T.diagonal().sum() / T.sum()),
            "district_population_share_in_network_cells": {"median": float(coverage.median()),
                                                           "min": float(coverage.min())},
        }
        (out / f"mitma_od_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
        print(f"[{city}{suffix}] {json.dumps({k: v for k, v in meta.items() if k != 'days'})}")


if __name__ == "__main__":
    main()
