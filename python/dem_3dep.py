#!/usr/bin/env python3
"""Bare-earth DEM from USGS 3DEP (1/3 arc-second, ~10 m) for the DEM sensitivity test of Chicago and Dallas.

The pipeline uses SRTM GL3 (3 arc-second, ~90 m), a radar surface model that includes part of the height
of buildings and vegetation: in the Chicago Loop it reads 240-256 m where the ground is at 180-185 m.
3DEP is a bare-earth terrain model (mostly lidar) published by the U.S. Geological Survey (public domain).

The script downloads the 1x1 degree 3DEP tiles that cover every node position reachable by the step-1
grid (a circle around the network centre with radius = farthest node + 2.6 km, which contains all
translations up to 2.5 km and all rotations), mosaics them and writes a clipped, compressed GeoTIFF next to
the SRTM DEM. Nothing existing is overwritten.

Output: data/data_processed/<city>/dem/<city>_dem_3dep13.tif
Usage:  python python/dem_3dep.py --cities chicago dallas
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
from terrain import MetricFrame  # noqa: E402

DATA = ROOT / "data" / "data_processed"
RAW = ROOT / "data" / "dem_3dep" / "raw"
URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/{t}/USGS_13_{t}.tif"
TAG = "3dep13"
MARGIN_M = 2600.0


def reachable_bounds(city: str) -> tuple[float, float, float, float]:
    with open(DATA / city / "graphs_fine_grid" / "graph_original.pkl", "rb") as f:
        G = pickle.load(f)
    lon = np.array([d["x"] for _, d in G.nodes(data=True)])
    lat = np.array([d["y"] for _, d in G.nodes(data=True)])
    fr = MetricFrame.from_graph(G)
    x, y = fr.to_metric(lon, lat)
    r = float(np.hypot(x, y).max()) + MARGIN_M
    t = np.linspace(0, 2 * np.pi, 361)
    clon, clat = fr.to_lonlat(r * np.cos(t), r * np.sin(t))
    return float(clon.min()), float(clat.min()), float(clon.max()), float(clat.max())


def tiles(bounds) -> list[str]:
    w, s, e, n = bounds
    out = []
    for top in range(math.floor(s) + 1, math.ceil(n) + 1):          # tile nYY covers lat YY-1..YY
        for left in range(math.floor(w), math.ceil(e)):              # tile wXXX covers lon -XXX..-XXX+1
            out.append(f"n{top:02d}w{-left:03d}")
    return out


def download(tile: str) -> Path | None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"USGS_13_{tile}.tif"
    if path.exists() and path.stat().st_size > 1e6:
        return path
    r = requests.get(URL.format(t=tile), stream=True, timeout=120)
    if r.status_code == 404:
        print(f"  {tile}: not available (open water or outside coverage)")
        return None
    r.raise_for_status()
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 22):
            f.write(chunk)
    tmp.replace(path)
    print(f"  {tile}: {path.stat().st_size / 1e6:.0f} MB")
    return path


def build(city: str) -> Path:
    out = DATA / city / "dem" / f"{city}_dem_{TAG}.tif"
    if out.exists():
        print(f"[{city}] {out} exists, kept")
        return out
    b = reachable_bounds(city)
    ts = tiles(b)
    print(f"[{city}] reachable bounds {np.round(b, 3)}; tiles {ts}")
    paths = [p for p in (download(t) for t in ts) if p is not None]
    srcs = [rasterio.open(p) for p in paths]
    arr, transform = merge(srcs, bounds=b, nodata=-999999.0)
    prof = srcs[0].profile.copy()
    # 3DEP is in NAD83 (EPSG:4269); the step-2 reader expects EPSG:4326. The NAD83-WGS84 offset is ~1 m in
    # the conterminous US, far below the 10 m resolution, so the grid is tagged as EPSG:4326.
    prof.update(crs=rasterio.crs.CRS.from_epsg(4326))
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=transform, count=1, dtype="float32",
                nodata=-999999.0, compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512,
                BIGTIFF="YES")
    for s in srcs:
        s.close()
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr[0].astype("float32"), 1)
    tmp.replace(out)
    print(f"[{city}] wrote {out} ({arr.shape[2]} x {arr.shape[1]} px)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cities", nargs="+", default=["chicago", "dallas"])
    args = ap.parse_args()
    for c in args.cities:
        build(c)


if __name__ == "__main__":
    main()
