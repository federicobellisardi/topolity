#!/usr/bin/env python3
"""Flow-weighted net rise Z_c and certification of the city ranking with regenerative braking.

With recovery of a fraction eps of the descent energy, the energy of a
configuration is lambda D + m g [(1 - eps) H + eps Z], where

    Z_c = sum_q F_q [z(d_q) - z(o_q)]

is the flow-weighted net rise between route endpoints (reports/reviewer1/
MODE_INVARIANCE_THEOREM.md, Sec. 3). The one-parameter reduction r = m g (1-eps)/lambda
used by vehicle_ratio_sweep.py is exact only if Z_c = 0.

Z_c is recomputed here with exactly the O/D nodes, draws and flows of step 2
(sample_od_nodes with the same K and seed, working-day + holiday flows), and
endpoint elevations sampled from the DEM as in step 2. No routing is needed.
For every vehicle with eps > 0 the observed city is then certified with the
exact two-parameter vertex test of ranking_theory.certified_minimum_with_recovery.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
sys.path.insert(0, str(ROOT / "python"))
import fine_grid_gravitational_work as fg  # noqa: E402
import ranking_theory as rt  # noqa: E402
from terrain import GeoDEM, Z_MIN_VALID_M  # noqa: E402
from vehicle_ratio_sweep import (CONFIG_FILE, CONSTRAINED_OVERRIDE, L_COL, ORIGINAL, RTOL,  # noqa: E402
                                 U_COL, load_city)

OUTPUT_DIR = ROOT / "output" / "net_rise"
REGEN_VEHICLES = ["car_ev", "bus", "metro", "train", "tram"]


def load_cells(path: Path) -> pd.DataFrame:
    """Same conversion as load_cells_minimal in step 2 (EPSG:3857 bounds -> lon/lat)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"cell_id": "cell"})
    if not {"x_min", "y_min", "x_max", "y_max"}.issubset(df.columns):
        raise KeyError(f"unexpected cell columns in {path}: {list(df.columns)}")
    tr = Transformer.from_crs(fg.CELLS_CRS, "EPSG:4326", always_xy=True)
    df["lon_min"], df["lat_min"] = tr.transform(df["x_min"].to_numpy(), df["y_min"].to_numpy())
    df["lon_max"], df["lat_max"] = tr.transform(df["x_max"].to_numpy(), df["y_max"].to_numpy())
    return df[["cell", "lon_min", "lat_min", "lon_max", "lat_max"]]


def load_flows(cells_dir: Path) -> pd.DataFrame:
    """Working-day + holiday flows summed per (origin, dest), as in step 2."""
    frames = []
    for name in ("od_matrix_working_day.csv", "od_matrix_holiday.csv"):
        df = pd.read_csv(cells_dir / name).rename(
            columns={"cell_origin": "origin", "cell_destination": "dest", "count": "flow"})
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0)
        frames.append(df[df["flow"] > 0][["origin", "dest", "flow"]])
    return pd.concat(frames).groupby(["origin", "dest"], as_index=False)["flow"].sum()


def node_elevations(G, dem_file: Path) -> dict:
    nodes = list(G.nodes())
    lon = np.array([G.nodes[n]["x"] for n in nodes])
    lat = np.array([G.nodes[n]["y"] for n in nodes])
    z = GeoDEM(dem_file, fg.graph_bounds(G)).sample(lon, lat)
    z[~np.isfinite(z) | (z < Z_MIN_VALID_M)] = np.nan
    return dict(zip(nodes, z))


def city_net_rise(city: str, grid_dir: str, od_samples: int, od_seed: int) -> pd.DataFrame:
    base = fg.DEFAULT_DATA_ROOT / city
    graph_files = fg.list_graph_files(base / grid_dir)
    cells = load_cells(base / f"{city}_basic_model" / "1000_cells" / "cell_coordinates.csv")
    flows = load_flows(base / f"{city}_basic_model" / "1000_cells")
    original = next(p for p in graph_files if p.name == ORIGINAL)
    with open(original, "rb") as f:
        samples = fg.sample_od_nodes(pickle.load(f), cells, od_samples, od_seed)
    from study_config import dem_source
    tag = dem_source(city)                   # 3DEP for Chicago and Dallas, as their step-2 tables
    dem_file = base / "dem" / (f"{city}_dem.tif" if tag == "srtm" else f"{city}_dem_{tag}.tif")

    o_cells, d_cells, F = flows["origin"].to_numpy(), flows["dest"].to_numpy(), flows["flow"].to_numpy(float)
    rows = []
    for gf in graph_files:
        with open(gf, "rb") as f:
            G = pickle.load(f)
        z = node_elevations(G, dem_file)
        Zk, Fk, n_nan = [], [], 0
        for o_map, d_map in samples:
            zo = np.array([z.get(o_map.get(c), np.nan) if o_map.get(c) is not None else np.nan for c in o_cells])
            zd = np.array([z.get(d_map.get(c), np.nan) if d_map.get(c) is not None else np.nan for c in d_cells])
            ok = np.isfinite(zo) & np.isfinite(zd)
            n_nan += int((~ok).sum())
            Zk.append(float(np.sum(F[ok] * (zd[ok] - zo[ok]))))
            Fk.append(float(F[ok].sum()))
        rows.append({"filename": gf.name, "net_rise_Z_m": float(np.mean(Zk)),
                     "net_rise_Z_sd_m": float(np.std(Zk)), "flow_used": float(np.mean(Fk)),
                     "pairs_missing_elevation": n_nan})
        del G
    return pd.DataFrame(rows)


def certify(city: str, z_df: pd.DataFrame, work_csv: Path, cfg: dict) -> dict:
    df = load_city(work_csv).merge(z_df[["filename", "net_rise_Z_m"]], on="filename", how="left")
    o, var = df[df["filename"] == ORIGINAL], df[df["filename"] != ORIGINAL]
    out = {"city": city, "n_variants": len(var), "constrained_override": city in CONSTRAINED_OVERRIDE}
    if o.empty or var.empty or df["net_rise_Z_m"].isna().any():
        out["status"] = "constrained" if not o.empty and var.empty else "incomplete"
        return out
    D0, H0, Z0 = (float(o[c].iloc[0]) for c in (L_COL, U_COL, "net_rise_Z_m"))
    D, H, Z = (var[c].to_numpy(float) for c in (L_COL, U_COL, "net_rise_Z_m"))
    out.update({"H0_m": H0, "Z0_m": Z0, "Z0_over_H0": Z0 / H0,
                "max_abs_dZ_over_H0": float(np.max(np.abs(Z - Z0)) / H0),
                "max_abs_dH_over_H0": float(np.max(np.abs(H - H0)) / H0)})
    for v in REGEN_VEHICLES:
        mc = cfg["modes"][v]
        V = rt.regen_box_vertices(mc["mass_kg"][::2], mc["lambda_J_per_m"][::2], mc["regen_eff"][::2])
        out[f"{v}_certified_with_Z"] = rt.certified_minimum_with_recovery(D0, H0, Z0, D, H, Z, V, RTOL)
        out[f"{v}_certified_Z0"] = rt.certified_minimum_with_recovery(
            D0, H0, 0.0, D, H, np.zeros_like(Z), V, RTOL)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cities", nargs="*", help="default: every city with step-2 results")
    ap.add_argument("--grid-dir", default="graphs_fine_grid")
    ap.add_argument("--exclude", nargs="*", default=[], help="in addition to study_config.EXCLUDED_CITIES")
    args = ap.parse_args()
    cfg = json.loads(CONFIG_FILE.read_text())
    out_dir = OUTPUT_DIR / args.grid_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from study_config import EXCLUDED_CITIES, step2_table, study_cities
    cities = args.cities or study_cities(args.grid_dir)
    rows = []
    for city in cities:
        if city in args.exclude or city in EXCLUDED_CITIES:
            continue
        work_csv = step2_table(city, args.grid_dir)
        if not work_csv.exists():
            continue
        z_csv = out_dir / f"{city}_net_rise.csv"
        if z_csv.exists():
            z_df = pd.read_csv(z_csv)
        else:
            run = pd.read_csv(work_csv)
            k = int(run["od_samples"].iloc[0]) if "od_samples" in run else fg.OD_SAMPLES
            seed = int(run["od_seed"].iloc[0]) if "od_seed" in run else fg.OD_SEED
            print(f"[{city}] computing Z_c (K={k}, seed={seed})", flush=True)
            z_df = city_net_rise(city, args.grid_dir, k, seed)
            z_df.to_csv(z_csv, index=False)
        rows.append(certify(city, z_df, work_csv, cfg))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "regen_certification.csv", index=False)

    tested = res[res.get("Z0_over_H0").notna()] if "Z0_over_H0" in res else res.iloc[0:0]
    tested = tested[~tested["constrained_override"].astype(bool)]
    print(f"\nCities with variants: {len(tested)}")
    if len(tested):
        print(f"|Z_0| / H_0: median {tested['Z0_over_H0'].abs().median():.4f}, "
              f"max {tested['Z0_over_H0'].abs().max():.4f}")
        print(f"max_c |Z_c - Z_0| / H_0: median {tested['max_abs_dZ_over_H0'].median():.4f}, "
              f"max {tested['max_abs_dZ_over_H0'].max():.4f}  "
              f"(compare max_c |H_c - H_0| / H_0: median {tested['max_abs_dH_over_H0'].median():.4f})")
        for v in REGEN_VEHICLES:
            print(f"{v:7s} certified minimum: with measured Z {int(tested[f'{v}_certified_with_Z'].sum()):3d}"
                  f" / {len(tested)}   assuming Z = 0 {int(tested[f'{v}_certified_Z0'].sum()):3d} / {len(tested)}")
        changed = np.any(tested[[f"{v}_certified_with_Z" for v in REGEN_VEHICLES]].to_numpy()
                         != tested[[f"{v}_certified_Z0" for v in REGEN_VEHICLES]].to_numpy(), axis=1)
        print("cities where Z changes the certification:", list(tested["city"][changed]) or "none")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
