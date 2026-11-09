#!/usr/bin/env python3
"""Sensitivity of the city classification to DEM noise (SI test).

Sampling a ~90 m SRTM DEM every 10 m and summing only the positive steps also
accumulates the DEM's vertical noise. This script recomputes step 2 for one
city with the DEM smoothed by a Gaussian of `sigma` pixels (NaN-aware, voids
kept as voids) and with the raw DEM, using the same O/D draws, and then
classifies the city with the SI rules (python/vehicle_ratio_sweep.py).

The step-2 code is used unchanged: only its DEM sampler is replaced in memory.

    python python/dem_noise_sensitivity.py --city bruxelles            # compute
    python python/dem_noise_sensitivity.py --summarize                  # table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
sys.path.insert(0, str(ROOT / "python"))
import fine_grid_gravitational_work as fg  # noqa: E402
from terrain import GeoDEM  # noqa: E402
from vehicle_ratio_sweep import R_PAPER, U_COL, classify_city, load_city  # noqa: E402

OUTPUT_DIR = ROOT / "output" / "dem_noise_sensitivity"
SIGMAS_PX = [0.0, 1.0, 2.0]
OD_SAMPLES = 1      # same draws (seed) for every sigma: paired comparison


def smoothed_dem_class(sigma_px: float):
    class SmoothedGeoDEM(GeoDEM):
        def __init__(self, dem_file, bounds, margin_deg=0.05):
            super().__init__(dem_file, bounds, margin_deg)
            if sigma_px > 0:
                valid = np.isfinite(self.arr)
                num = gaussian_filter(np.where(valid, self.arr, 0.0).astype(float), sigma_px)
                den = gaussian_filter(valid.astype(float), sigma_px)
                out = np.full_like(num, np.nan)
                out[valid] = num[valid] / den[valid]
                self.arr = out.astype("float32")
    return SmoothedGeoDEM


def csv_path(city, sigma):
    return OUTPUT_DIR / city / f"work_sigma{sigma:g}.csv"


def compute(city: str, sigmas):
    fg.DEBUG_LENGTHS = False
    base = fg.DEFAULT_DATA_ROOT / city
    graph_files = fg.list_graph_files(base / "graphs_fine_grid")
    dem_file = base / "dem" / f"{city}_dem.tif"
    for sigma in sigmas:
        out = csv_path(city, sigma)
        out.parent.mkdir(parents=True, exist_ok=True)
        fg.GeoDEM = smoothed_dem_class(sigma)
        print(f"\n[{city}] sigma = {sigma:g} px -> {out}")
        fg.analyze_graphs(graph_files, dem_file, out, city, resume=True,
                          od_samples=OD_SAMPLES, od_seed=fg.OD_SEED)


def summarize():
    rows = []
    for city_dir in sorted(p for p in OUTPUT_DIR.iterdir() if p.is_dir()):
        raw = csv_path(city_dir.name, 0.0)
        if not raw.exists():
            continue
        U_raw = load_city(raw).set_index("filename")[U_COL]
        for sigma in SIGMAS_PX:
            f = csv_path(city_dir.name, sigma)
            if not f.exists():
                continue
            df = load_city(f)
            res = classify_city(df, {})
            U = df.set_index("filename")[U_COL]
            common = U.index.intersection(U_raw.index)
            rows.append({"city": city_dir.name, "sigma_px": sigma, "n_embeddings": len(df),
                         "status": res.get("status"), "delta_paper_pct": res.get("delta_paper_pct"),
                         "best_variant": res.get("best_variant_paper"),
                         "r_min_lo": res.get("r_min_lo"), "r_min_hi": res.get("r_min_hi"),
                         "uphill_vs_raw_original": float(U.get("graph_original.pkl", np.nan)
                                                         / U_raw.get("graph_original.pkl", np.nan)),
                         "uphill_vs_raw_median": float((U[common] / U_raw[common]).median())})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"r of the paper: {R_PAPER:.2f}")
    print(out.to_string(index=False, float_format="%.3f"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city")
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS_PX)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    elif args.city:
        compute(args.city, args.sigmas)
    else:
        ap.error("give --city or --summarize")


if __name__ == "__main__":
    main()
