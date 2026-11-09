#!/usr/bin/env python3
"""Robustness of the city classification to the vehicle type (reviewer request).

For every embedding k the energy of the paper is E_k = m g U_k + lambda L_k,
with U_k the OD-weighted uphill gain and L_k the OD-weighted 3-D path length.
Routes are shortest 3-D paths, so U_k and L_k do not depend on the vehicle and

    E_k / lambda = r U_k + L_k,      r = m g (1 - eps) / lambda,

where eps is the share of the descent energy recovered by regenerative braking
(round trips are closed loops, so descent = ascent on average). The ranking of
the embeddings therefore depends on the vehicle only through r, and occupancy
and drivetrain efficiency cancel.

For each city this script computes the exact interval of r >= 0 over which the
observed embedding is the minimum, and compares it with the r range of each
vehicle type (python/config/barcelona_multimodal_params.json). Nothing is
recomputed on the networks: it only reads the step-2 output tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_theory as rt  # noqa: E402
from barcelona_multimodal_robustness import CONFIG_FILE, ratio_range  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "data_processed"
OUTPUT_ROOT = ROOT / "output" / "vehicle_ratio_sweep"
U_COL = "vertical_gain_on_3d_path_m"
L_COL = "path_length_3d_on_3d_path_m"
ORIGINAL = "graph_original.pkl"
G_PHYS = 9.81
R_PAPER = 1200.0 * G_PHYS / 585.0          # m = 1200 kg, lambda = 585 J/m
NEAR_LOWEST = 0.01                         # "near-lowest": within 1 % of the best
# Classification rules: python/study_config.py (re-exported here; other scripts import them from this module)
from study_config import (CONSTRAINED_OVERRIDE, EXCLUDED_CITIES, MIN_TRANSLATION_M, RTOL,  # noqa: E402,F401
                          TIE_TOL, dem_source, step2_table, study_cities)
VEHICLES = ["car", "car_ev", "moto", "bus", "metro", "train", "tram"]


def main_ensemble_mask(filenames: pd.Series) -> pd.Series:
    """True for embeddings of the main analysis: no contractions (SI Note 8), no translations < 200 m."""
    name = filenames.astype(str).str.replace("graph_", "", regex=False).str.replace(".pkl", "", regex=False)
    scale = pd.to_numeric(name.str.extract(r"^scale_(?:ns|ew)_(\d+p\d+)$")[0].str.replace("p", "."),
                          errors="coerce")
    dist = pd.to_numeric(name.str.extract(r"^trans_(\d+)m_")[0], errors="coerce")
    return ~(scale < 1.0) & ~(dist < MIN_TRANSLATION_M)


def load_city(table: Path) -> pd.DataFrame:
    """Embeddings of the main analysis: no contractions (SI Note 8), no translations < 200 m."""
    df = pd.read_csv(table).drop_duplicates("filename", keep="last")
    keep = main_ensemble_mask(df["filename"])
    return df.loc[keep, ["filename", U_COL, L_COL]].reset_index(drop=True)


def excess_over_best(r, U_o, L_o, U, L):
    """Regret (E_0 - min_c E_c) / min_c E_c at ratio r (E = L + r U); <= RTOL counts as minimum."""
    return rt.regret(L_o + r * U_o, L + r * U)


def classify_city(df: pd.DataFrame, ranges: dict, constrained_override=False) -> dict:
    o = df[df["filename"] == ORIGINAL]
    var = df[df["filename"] != ORIGINAL]
    out = {"n_variants": len(var), "constrained_override": constrained_override}
    if o.empty or var.empty:
        out["status"] = "constrained" if not o.empty else "missing_original"
        return out
    U_o, L_o = float(o[U_COL].iloc[0]), float(o[L_COL].iloc[0])
    U, L = var[U_COL].to_numpy(float), var[L_COL].to_numpy(float)

    interval = rt.minimizer_interval(L_o, U_o, L, U, rtol=RTOL)
    strict = rt.minimizer_interval(L_o, U_o, L, U)
    out["r_min_lo"], out["r_min_hi"] = interval if interval else (np.nan, np.nan)
    out["r_min_lo_strict"], out["r_min_hi_strict"] = strict if strict else (np.nan, np.nan)
    E_paper = L + R_PAPER * U
    out["status_unconstrained"] = ("original" if rt.rank_of(L_o + R_PAPER * U_o, E_paper, RTOL) == 1
                                   else "displaced_lower")
    out["status"] = "constrained" if constrained_override else out["status_unconstrained"]
    best = var.iloc[int(np.argmin(E_paper))]
    out["best_variant_paper"] = best["filename"].replace("graph_", "").replace(".pkl", "")
    out["delta_paper_pct"] = 100 * float(np.min(E_paper) / (L_o + R_PAPER * U_o) - 1)
    out["rank_paper"] = rt.rank_of(L_o + R_PAPER * U_o, E_paper, RTOL)
    out["n_breakpoints_all_vehicles"] = len(rt.ranking_breakpoints(
        np.r_[L_o, L], np.r_[U_o, U], *ranges.get("all_vehicles", (0, 0, 0))[::2]))

    for v, (r_lo, _, r_hi) in ranges.items():
        status, _ = rt.classify_on_interval(L_o, U_o, L, U, r_lo, r_hi, rtol=RTOL)
        out[f"{v}_status"] = status
        out[f"{v}_frac_range_original_min"] = (
            0.0 if interval is None else
            max(0.0, min(r_hi, interval[1]) - max(r_lo, interval[0])) / (r_hi - r_lo))
        grid = np.linspace(r_lo, r_hi, 201)
        ex = np.array([excess_over_best(r, U_o, L_o, U, L) for r in grid])
        out[f"{v}_near_lowest_whole_range"] = bool(np.all(ex <= NEAR_LOWEST))
        out[f"{v}_worst_regret_pct"] = float(100 * ex.max())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-dir", default="graphs_fine_grid",
                    help="per-city folder with fine_grid_gravitational_work.csv (e.g. graphs_fine_grid_v1)")
    ap.add_argument("--exclude", nargs="*", default=[], help="in addition to study_config.EXCLUDED_CITIES")
    args = ap.parse_args()

    cfg = json.loads(CONFIG_FILE.read_text())
    ranges = {v: ratio_range(cfg["modes"][v]) for v in VEHICLES}
    r_all = (min(r[0] for r in ranges.values()), max(r[2] for r in ranges.values()))
    ranges["all_vehicles"] = (r_all[0], R_PAPER, r_all[1])

    rows = []
    skip = set(EXCLUDED_CITIES) | set(args.exclude)
    for city_dir in sorted(p for p in DATA_ROOT.iterdir() if (p / "data_useful.csv").exists()):
        if city_dir.name in skip:
            continue
        table = step2_table(city_dir.name, args.grid_dir)
        if not table.exists():
            rows.append({"city": city_dir.name, "status": "no_results"})
            continue
        rows.append({"city": city_dir.name, "dem": dem_source(city_dir.name) if args.grid_dir == "graphs_fine_grid"
                     else "srtm",
                     **classify_city(load_city(table), ranges, city_dir.name in CONSTRAINED_OVERRIDE)})
    res = pd.DataFrame(rows)

    out_dir = OUTPUT_ROOT / args.grid_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "city_ratio_robustness.csv", index=False)
    pd.DataFrame([{"vehicle": v, "r_low": a, "r_central": b, "r_high": c} for v, (a, b, c) in ranges.items()]) \
        .to_csv(out_dir / "vehicle_ratio_ranges.csv", index=False)

    tested = res[res["status"].isin(["original", "displaced_lower"])]
    print(f"r of the paper (m = 1200 kg, lambda = 585 J/m): {R_PAPER:.2f}")
    print(f"Cities: {len(res)}; with transformed embeddings: {len(tested)}; "
          f"original lowest at the paper r: {(tested['status'] == 'original').sum()}")
    print("\nVehicle      r range        original lowest over whole range   near-lowest (1%) over whole range")
    for v, (a, _, c) in ranges.items():
        n_min = int((tested[f"{v}_status"] == "robust_minimum").sum())
        n_near = int(tested[f"{v}_near_lowest_whole_range"].astype(bool).sum())
        print(f"{v:12s} [{a:6.1f}, {c:6.1f}]   {n_min:3d} / {len(tested)}"
              f"                          {n_near:3d} / {len(tested)}")
    changed = tested[(tested["status"] == "original") & (tested["all_vehicles_status"] != "robust_minimum")]
    if not changed.empty:
        print("\nCities whose classification changes somewhere in the vehicle range:")
        print(changed[["city", "r_min_lo", "r_min_hi", "all_vehicles_frac_range_original_min",
                       "all_vehicles_worst_regret_pct"]].to_string(index=False, float_format="%.3f"))
    disp = tested[tested["status"] == "displaced_lower"]
    if not disp.empty:
        print("\nDisplaced-lower cities at the paper r:")
        print(disp[["city", "best_variant_paper", "delta_paper_pct", "r_min_lo", "r_min_hi",
                    "all_vehicles_status", "all_vehicles_frac_range_original_min"]]
              .to_string(index=False, float_format="%.3f"))
    ov = res[res["constrained_override"].astype(bool) & res["status_unconstrained"].notna()]
    if not ov.empty:
        print("\nConstrained by hand (SI override), classification without the override:")
        print(ov[["city", "n_variants", "status_unconstrained", "delta_paper_pct", "r_min_lo", "r_min_hi"]]
              .to_string(index=False, float_format="%.3f"))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
