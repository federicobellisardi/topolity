#!/usr/bin/env python3
"""Audit of the constrained cities: why every transformation of their network is rejected.

Step 1 (pipeline_production/dem_extractor_fine_grid.py, --land-check full) rejects a transformed
embedding when any node that lies on land in the observed embedding is moved into water. The land mask
is Natural Earth 1:10M land (ne_10m_land) clipped to the transformation extent, minus Natural Earth
lakes (ne_10m_lakes) shrunk by 0.003 deg, buffered by 1e-5 deg; nodes already in water in the observed
embedding (bridges, piers) are not checked. This script rebuilds the same mask and, for every candidate
transformation of the step-1 grid (40 translations, 14 rotations, 14 expansions), counts the on-land
nodes that end in water and whether that water is sea or a lake. It also checks that step 1 finished
(stats table and graphs present), so that a city is not reported as constrained because of a failed run.

Output: reports/reviewer1/tables/constrained_audit_candidates.csv (one row per city and candidate) and
reports/reviewer1/tables/constrained_audit_summary.csv (one row per city).

Usage:
    python python/constrained_audit.py                 # cities classified constrained by the SI builder
    python python/constrained_audit.py --cities sydney chicago
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
sys.path.insert(0, str(ROOT / "supplementary" / "python"))
from terrain import MetricFrame, rotation_fn, scaling_fn, translation_fn  # noqa: E402

DATA = ROOT / "data" / "data_processed"
NE_LAND = ROOT / "data" / "land" / "ne_10m_land.shp"
NE_LAKES = ROOT / "data" / "land" / "ne_10m_lakes.shp"
GSHHG = ROOT / "data" / "coast" / "gshhg" / "GSHHS_shp" / "f"     # full resolution: L1 land, L2 lakes, L3 islands
MASK = "ne"                                                        # "ne" (step 1) or "gshhg"
OUT = ROOT / "reports" / "reviewer1" / "tables"
LAKE_SHRINK_DEG = 0.003
COAST_TOL_DEG = 0.0               # optional coastline buffer (deg); 0 = mask of step 1
SAFE_BUFFER_DEG = 1e-5
EXTENT_MARGIN_DEG = 0.6          # covers every node position reachable by the grid below
DISTANCES = np.arange(250, 2501, 250)
ANGLES = {0: "east", 90: "north", 180: "west", 270: "south"}
ROTATIONS = [-40, -30, -20, -15, -10, -5, -1, 1, 5, 10, 15, 20, 30, 40]
EXPANSIONS = [1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20]
CONTRACTIONS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]
EXTRA_CITIES = {"madrid", "santiago"}          # oblique translations and contractions (submit_rerun_v2.sh)
EXTRA_ANGLES = {30: "30deg", 45: "45deg", 135: "135deg", 225: "225deg", 315: "315deg"}
from study_config import EXCLUDED_CITIES as EXCLUDED  # noqa: E402


def candidates(extra: bool = False):
    angles = {**ANGLES, **(EXTRA_ANGLES if extra else {})}
    for d in DISTANCES:
        for a, name in angles.items():
            t = np.radians(a)
            yield (f"trans_{d}m_a+{a:03d}", "translation", name, d,
                   translation_fn(d * np.cos(t), d * np.sin(t)))
    for r in ROTATIONS:
        yield f"rot_{r:+d}deg", "rotation", "", r, rotation_fn(r)
    for s in EXPANSIONS + (CONTRACTIONS if extra else []):
        for ax, name in (("x", "E-W"), ("y", "N-S")):
            yield f"scale_{'ew' if ax == 'x' else 'ns'}_{s:.2f}", "expansion", name, s, scaling_fn(s, ax)


def masks_gshhg(bb):
    """GSHHG full resolution: land (L1) minus lakes (L2) plus islands in lakes (L3)."""
    rd = lambda lv: gpd.read_file(GSHHG / f"GSHHS_f_L{lv}.shp", bbox=bb).to_crs(4326)  # noqa: E731
    land = rd(1).clip(box(*bb)).union_all()
    lakes = rd(2)
    lake = lakes.union_all() if not lakes.empty else shapely.Polygon()
    if not lake.is_empty:
        land = land.difference(lake)
        isl = rd(3)
        if not isl.empty:
            land = land.union(isl.union_all())
    return land.buffer(SAFE_BUFFER_DEG), lake


def masks(lon, lat, coast_tol: float = COAST_TOL_DEG):
    bb = (lon.min() - EXTENT_MARGIN_DEG, lat.min() - EXTENT_MARGIN_DEG,
          lon.max() + EXTENT_MARGIN_DEG, lat.max() + EXTENT_MARGIN_DEG)
    if MASK == "gshhg":
        return masks_gshhg(bb)
    land = gpd.read_file(NE_LAND, bbox=bb).to_crs(4326).clip(box(*bb)).union_all()
    if coast_tol > 0:
        land = land.buffer(coast_tol)
    lakes = gpd.read_file(NE_LAKES, bbox=bb).to_crs(4326)
    lake = lakes.union_all().buffer(-LAKE_SHRINK_DEG) if not lakes.empty else shapely.Polygon()
    if not lake.is_empty:
        land = land.difference(lake)
    return land.buffer(SAFE_BUFFER_DEG), lake


def audit_city(city: str, coast_tol: float = COAST_TOL_DEG) -> tuple[pd.DataFrame, dict]:
    base = DATA / city / "graphs_fine_grid"
    with open(base / "graph_original.pkl", "rb") as f:
        G = pickle.load(f)
    lon = np.array([d["x"] for _, d in G.nodes(data=True)])
    lat = np.array([d["y"] for _, d in G.nodes(data=True)])
    safe, lake = masks(lon, lat, coast_tol)
    on = shapely.contains_xy(safe, lon, lat)
    frame = MetricFrame.from_graph(G)
    x0, y0 = frame.to_metric(lon[on], lat[on])
    rows = []
    for name, fam, direction, par, fn in candidates(city in EXTRA_CITIES):
        xm, ym = fn(x0, y0)
        lo, la = frame.to_lonlat(xm, ym)
        wet = ~shapely.contains_xy(safe, lo, la)
        in_lake = shapely.contains_xy(lake, lo[wet], la[wet]) if (wet.any() and not lake.is_empty) else np.zeros(0, bool)
        rows.append({"city": city, "candidate": name, "family": fam, "direction": direction, "parameter": par,
                     "n_nodes_in_water": int(wet.sum()), "pct_nodes_in_water": 100 * wet.mean(),
                     "n_in_lake": int(in_lake.sum()), "n_in_sea": int(wet.sum() - in_lake.sum())})
    df = pd.DataFrame(rows)
    stats = pd.read_csv(base / "fine_grid_stats.csv") if (base / "fine_grid_stats.csv").exists() else pd.DataFrame()
    graphs = sorted(p.name for p in base.glob("graph_*.pkl"))
    adm = df[df["n_nodes_in_water"] == 0]
    have = {p.removeprefix("graph_").removesuffix(".pkl") for p in graphs}
    new = [c for c in adm["candidate"] if not _matches(c, have)]
    summary = {
        "city": city, "n_nodes": len(lon), "n_nodes_checked": int(on.sum()),
        "n_nodes_in_water_observed": int((~on).sum()),
        "step1_stats_rows": len(stats), "step1_graphs": len(graphs),
        "step1_admissible": int(stats["on_land"].astype(str).eq("True").sum()) - 1 if len(stats) else -1,
        "audit_admissible": len(adm), "audit_admissible_list": ";".join(adm["candidate"]),
        "n_new_admissible": len(new), "new_admissible_list": ";".join(new),
        "min_pct_in_water": df["pct_nodes_in_water"].min(),
        "best_candidate": df.loc[df["pct_nodes_in_water"].idxmin(), "candidate"],
        "share_water_in_sea": df["n_in_sea"].sum() / max(df["n_nodes_in_water"].sum(), 1),
    }
    for fam in ("translation", "rotation", "expansion"):
        f = df[df["family"] == fam]
        summary[f"min_pct_{fam}"] = f["pct_nodes_in_water"].min()
    t = df[df["family"] == "translation"]
    for d in ANGLES.values():
        summary[f"min_pct_trans_{d}"] = t.loc[t["direction"] == d, "pct_nodes_in_water"].min()
    return df, summary


def _matches(candidate: str, have: set[str]) -> bool:
    """Is the audit candidate among the existing step-1 graphs (names differ in number format)?"""
    import re
    m = re.fullmatch(r"rot_([+-]\d+)deg", candidate)
    if m:
        return f"rot_{int(m.group(1)):+d}p00deg" in have
    m = re.fullmatch(r"scale_(ns|ew)_(\d)\.(\d+)", candidate)
    if m:
        return any(re.fullmatch(rf"scale_{m.group(1)}_{m.group(2)}p{m.group(3)}0*", h) for h in have)
    return candidate in have


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help="every city of the study, not only the constrained ones")
    ap.add_argument("--coast-tol", type=float, default=COAST_TOL_DEG, help="0 reproduces the strict first v2 mask")
    ap.add_argument("--tag", default="", help="suffix of the output tables")
    ap.add_argument("--mask", default="ne", choices=["ne", "gshhg"], help="land mask (ne = step 1)")
    args = ap.parse_args()
    global MASK
    MASK = args.mask
    cities = args.cities
    if args.all:
        cities = sorted(p.name for p in DATA.iterdir()
                        if (p / "graphs_fine_grid" / "graph_original.pkl").exists() and p.name not in EXCLUDED)
    if cities is None:
        import build_gravitational_supplementary as bgs
        s, _ = bgs.build_summary()
        cities = sorted(s.loc[s["status"] == "CONSTRAINED", "city"])
    OUT.mkdir(parents=True, exist_ok=True)
    allc, summ = [], []
    for c in cities:
        df, sm = audit_city(c, args.coast_tol)
        allc.append(df)
        summ.append(sm)
        print(f"{c:18s} admissible(audit)={sm['audit_admissible']:3d} step1={sm['step1_admissible']:3d} "
              f"new={sm['n_new_admissible']:3d} "
              f"min water {sm['min_pct_in_water']:.2f}% ({sm['best_candidate']}), sea share {sm['share_water_in_sea']:.2f}")
    pd.concat(allc).to_csv(OUT / f"constrained_audit_candidates{args.tag}.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / f"constrained_audit_summary{args.tag}.csv", index=False)
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
