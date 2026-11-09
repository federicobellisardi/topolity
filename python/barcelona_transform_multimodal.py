#!/usr/bin/env python3
"""Transformation experiment with several transport modes (SI test; Barcelona, Madrid).

Is the observed embedding of a city still the lowest-energy one among its
transformed embeddings (translations, rotations, anisotropic scalings) when trips
are made by combustion cars, electric cars, motorcycles, buses or rail, alone or
in the observed modal mix?

For every embedding produced by step 1 (graphs_fine_grid) and the same O/D
nodes, draws (K, seed) and flows as step 2:

* road: each OD is routed on the 3-D length graph of step 2; per OD we keep the
  3-D length L, uphill H+ and downhill H- = H+ - (z_d - z_o). Trips are one way,
  so recovery is applied explicitly: e = (m g (H+ - eps H-) + lambda L) / occupancy.
* rail: the OSM stop graph of rail_network.py is moved with the same metric
  transformation as the road network and its tracks re-profiled on the DEM
  (every 10 m on the terrain, uniform grade in tunnels and on bridges); trips
  board at the stop nearest to the origin node and alight at the stop nearest to
  the destination node. ODs with no usable rail option (access > max_access,
  same stop, no path) take the bus.
* bus: the OSM bus lines (python/bus_network.py) are moved with the city and their
  tracks re-profiled on the DEM; a trip boards at the bus stop nearest to its origin
  node and alights at the one nearest to its destination. ODs without a usable
  bus option use the road path.

Energies are linear in the vehicle parameters, so only F-weighted sums are stored
per embedding and every scenario / Monte Carlo draw recombines them.

Usage (one embedding per Slurm array task, then one merge):
    python barcelona_transform_multimodal.py --city madrid --list
    python barcelona_transform_multimodal.py --city madrid --task 17
    python barcelona_transform_multimodal.py --city madrid --merge
Without --task/--merge all missing embeddings are computed in sequence, then merged.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from networkit import distance
from pyproj import Transformer
from scipy.spatial import KDTree
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
sys.path.insert(0, str(ROOT / "python"))
import fine_grid_gravitational_work as fg  # noqa: E402
import bus_network as bn  # noqa: E402
import multi_city_settlement as mcs  # noqa: E402
import ranking_theory as rt  # noqa: E402
import rail_network as rn  # noqa: E402
from barcelona_multimodal_robustness import (CONFIG_FILE, PARAMS, RAIL_MODES, draw_params,  # noqa: E402
                                             central_params, modal_shares)
from net_rise import load_cells, load_flows, node_elevations  # noqa: E402
from terrain import GeoDEM, MetricFrame, rotation_fn, scaling_fn, translation_fn  # noqa: E402
from vehicle_ratio_sweep import RTOL  # noqa: E402

GRID_DIR = "graphs_fine_grid"
CONFIGS = {"barcelone": CONFIG_FILE, "madrid": ROOT / "python" / "config" / "madrid_multimodal_params.json"}
OUTPUT_ROOT = ROOT / "output" / "transform_multimodal"          # OD flows: geolocated records (cells)
OD_SOURCES = ("twitter", "mitma", "mitma_ge2km")


def output_root(od: str) -> Path:
    return OUTPUT_ROOT if od == "twitter" else ROOT / "output" / f"transform_multimodal_{od}"
OUTPUT_DIR = ROOT / "output" / "barcelona_transform_multimodal"   # v1 run (rail sampled at stops only)
ROAD_MODES = ["car", "car_ev", "moto", "bus"]
TUNNEL_FACTORS = (0.0, 2.0)          # metro gain on non-terrain track: none / doubled
ORIGINAL = "graph_original.pkl"


def family_of(filename: str) -> str:
    """original / trans / rot / scale_up (expansion) / scale_down (contraction)."""
    name = filename.replace("graph_", "").replace(".pkl", "")
    if name == "original":
        return "original"
    m = re.fullmatch(r"scale_(?:ew|ns)_(\d+)p(\d+)", name)
    if m:
        return "scale_up" if float(f"{m.group(1)}.{m.group(2)}") > 1.0 else "scale_down"
    return name.split("_")[0]


def transformation_of(filename: str):
    """Metric transformation of step 1 for a variant file name."""
    name = filename.replace("graph_", "").replace(".pkl", "")
    if name == "original":
        return lambda x, y: (x, y)
    m = re.fullmatch(r"trans_(\d+)m_a([+-]\d+)", name)
    if m:
        d, a = float(m.group(1)), np.radians(float(m.group(2)))
        return translation_fn(d * np.cos(a), d * np.sin(a))
    m = re.fullmatch(r"rot_([+-]\d+)p(\d+)deg", name)
    if m:
        return rotation_fn(float(f"{m.group(1)}.{m.group(2)}"))
    m = re.fullmatch(r"scale_(ew|ns)_(\d+)p(\d+)", name)
    if m:
        return scaling_fn(float(f"{m.group(2)}.{m.group(3)}"), "x" if m.group(1) == "ew" else "y")
    raise NotImplementedError(f"transformation not recognised: {name}")


def embedding_sums(G, R, stop_nodes, stop_xy, B, bus_nodes, bus_xy, samples, flows, z, access_thresholds,
                   metric_crs, dem):
    """F-weighted sums of (L, H+, H-) per mode family, averaged over the O/D draws.

    Bus legs use the bus network (B) when both ends have a stop within the access
    distance and the two stops differ, otherwise the road path ('busroad').
    Rail trips without a usable rail option fall back to the bus in the same way.
    """
    ev = fg.ODWorkEvaluator(G, dem, ds=fg.DS, elevation_source="dem")
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    nodes = list(G.nodes())
    nx_, ny_ = to_m.transform([G.nodes[n]["x"] for n in nodes], [G.nodes[n]["y"] for n in nodes])
    acc_d, acc_i = KDTree(stop_xy).query(np.column_stack([nx_, ny_]))
    node_stop = {n: (stop_nodes[i], d) for n, i, d in zip(nodes, acc_i, acc_d)}
    router = bn.BusRouter(B)
    bacc_d, bacc_i = KDTree(bus_xy).query(np.column_stack([nx_, ny_]))
    node_bstop = {n: (router.index[bus_nodes[i]], d) for n, i, d in zip(nodes, bacc_i, bacc_d)}

    by_origin = {}
    for o, d, f in flows.itertuples(index=False, name=None):
        by_origin.setdefault(o, []).append((d, f))

    keys = ["F", "road_L", "road_Hu", "road_Hd"]
    for a in access_thresholds:
        keys += [f"a{a}_rail_F", f"a{a}_bus_F"]
        keys += [f"a{a}_{m}_{q}" for m in RAIL_MODES + ["busnet", "busroad", "rfbnet", "rfbroad"]
                 for q in ("L", "Hu", "Hd")]
        keys += [f"a{a}_{m}_{q}" for m in RAIL_MODES for q in ("Hut", "Hdt")]
    acc = {k: 0.0 for k in keys}
    rail_cache, bus_cache = {}, {}

    def add(prefix, f, vals):
        acc[f"{prefix}_L"] += f * vals[0]
        acc[f"{prefix}_Hu"] += f * vals[1]
        acc[f"{prefix}_Hd"] += f * vals[2]
        if len(vals) > 3:                     # rail: gains on non-terrain track
            acc[f"{prefix}_Hut"] += f * vals[3]
            acc[f"{prefix}_Hdt"] += f * vals[4]
    for o_map, d_map in samples:
        for o_cell, dests in by_origin.items():
            src = o_map.get(o_cell)
            sid = ev.node2id.get(src)
            if sid is None:
                continue
            run = distance.Dijkstra(ev.nkG_3d, sid, True)
            run.run()
            s_o, a_o = node_stop[src]
            if s_o not in rail_cache:
                rail_cache[s_o] = rn.rail_source_sums(R, s_o)
            b_o, ba_o = node_bstop[src]
            if b_o not in bus_cache:
                bus_cache[b_o] = router.sums(b_o)
            for d_cell, f in dests:
                dst = d_map.get(d_cell)
                tid = ev.node2id.get(dst)
                if tid is None or tid == sid:
                    continue
                path = run.getPath(tid)
                if not path:
                    continue
                L = sum(ev.arc_length_3d[(a, b)] for a, b in zip(path[:-1], path[1:]))
                Hu = sum(ev.arc_vertical_gain_m[(a, b)] for a, b in zip(path[:-1], path[1:]))
                Hd = Hu - (z[dst] - z[src])
                acc["F"] += f
                acc["road_L"] += f * L
                acc["road_Hu"] += f * Hu
                acc["road_Hd"] += f * Hd
                s_d, a_d = node_stop[dst]
                rail = rail_cache[s_o].get(s_d) if s_d != s_o else None
                b_d, ba_d = node_bstop[dst]
                bus = bus_cache[b_o][:, b_d] if b_d != b_o else None
                if bus is not None and not np.isfinite(bus[0]):
                    bus = None
                road = (L, Hu, Hd)
                for a in access_thresholds:
                    bus_ok = bus is not None and ba_o <= a and ba_d <= a
                    if bus_ok:
                        acc[f"a{a}_bus_F"] += f
                    add(f"a{a}_busnet" if bus_ok else f"a{a}_busroad", f, bus if bus_ok else road)
                    if rail is not None and a_o <= a and a_d <= a:
                        acc[f"a{a}_rail_F"] += f
                        for m in RAIL_MODES:
                            add(f"a{a}_{m}", f, rail[m])
                    else:
                        add(f"a{a}_rfbnet" if bus_ok else f"a{a}_rfbroad", f, bus if bus_ok else road)
    k = len(samples)
    return {key: val / k for key, val in acc.items()}


def leg_energy(par, L, Hu, Hd):
    """Per-passenger energy of F-weighted sums for one mode (one-way trips, explicit recovery)."""
    return (par["mass_kg"] * rt.G * (Hu - par["regen_eff"] * Hd) + par["lambda_J_per_m"] * L) / par["occupancy"]


def scenario_energy(row, par, shares, a, tunnel_factor=1.0):
    """Per-trip energy of one embedding: dict mode -> energy, plus 'rail' and 'mix'.

    tunnel_factor multiplies the metro gains on non-terrain track (sensitivity test)."""
    F = row["F"]
    def leg(mode, prefix):
        Hu, Hd = row[f"{prefix}_Hu"], row[f"{prefix}_Hd"]
        if mode == "metro" and tunnel_factor != 1.0:
            Hu = Hu + (tunnel_factor - 1.0) * row[f"{prefix}_Hut"]
            Hd = Hd + (tunnel_factor - 1.0) * row[f"{prefix}_Hdt"]
        return leg_energy(par[mode], row[f"{prefix}_L"], Hu, Hd)

    e = {m: leg(m, "road") / F for m in ROAD_MODES if m != "bus"}
    e["bus"] = (leg("bus", f"a{a}_busnet") + leg("bus", f"a{a}_busroad")) / F
    rail = sum(leg(m, f"a{a}_{m}") for m in RAIL_MODES)
    rail += leg("bus", f"a{a}_rfbnet") + leg("bus", f"a{a}_rfbroad")
    e["rail"] = rail / F
    e["mix"] = sum(shares.get(m, 0.0) * e[m] for m in ROAD_MODES) + shares["rail"] * e["rail"]
    return e


def observed_status(E: pd.Series):
    """(is minimum within the SI tolerance, regret %, rank, best embedding, margin %).

    margin = (min_c E_c - E_0) / E_0: positive when the observed embedding is cheaper."""
    E0, Ec = E[ORIGINAL], E.drop(ORIGINAL)
    return (rt.rank_of(E0, Ec.to_numpy(), RTOL) == 1, 100 * rt.regret(E0, Ec.to_numpy()),
            rt.rank_of(E0, Ec.to_numpy(), RTOL), Ec.idxmin(), 100 * (Ec.min() - E0) / E0)


def load_od_flows(city: str, od: str) -> pd.DataFrame:
    """Cell flows (origin, dest, flow): geolocated records as step 2, or MITMA weekdays (mitma_od.py)."""
    base = fg.DEFAULT_DATA_ROOT / city
    if od == "twitter":
        return load_flows(base / f"{city}_basic_model" / "1000_cells")
    suffix = od.removeprefix("mitma")
    df = pd.read_csv(base / f"{city}_mitma_model" / "1000_cells" / f"od_matrix_mitma_weekday{suffix}.csv").rename(
        columns={"cell_origin": "origin", "cell_destination": "dest", "count": "flow"})
    return df[df["flow"] > 0][["origin", "dest", "flow"]]


def city_setup(city: str, od: str = "twitter"):
    """Everything shared by the embeddings of one city (built identically in every task)."""
    cfg = json.loads(CONFIGS[city].read_text())
    base = fg.DEFAULT_DATA_ROOT / city
    graph_files = fg.list_graph_files(base / GRID_DIR)
    work = pd.read_csv(base / GRID_DIR / "fine_grid_gravitational_work.csv")
    k, seed = int(work["od_samples"].iloc[0]), int(work["od_seed"].iloc[0])
    cells = load_cells(base / f"{city}_basic_model" / "1000_cells" / "cell_coordinates.csv")
    flows = load_od_flows(city, od)
    with open(base / GRID_DIR / ORIGINAL, "rb") as f:
        G0 = pickle.load(f)
    xs = [d["x"] for _, d in G0.nodes(data=True)]
    ys = [d["y"] for _, d in G0.nodes(data=True)]
    bbox = (min(ys), min(xs), max(ys), max(xs))
    bounds = (bbox[1], bbox[0], bbox[3], bbox[2])
    metric_crs = mcs.load_city_fua(city).estimate_utm_crs()
    net = cfg["rail_network"]
    R0, rstats = rn.build_rail_graph(rn.download_rail_routes(bbox, base / "transit" / "osm_rail_routes_v2.json", net["include_networks"]),
                                     net["include_networks"], metric_crs, bounds=bounds)
    B0, bstats = bn.build_bus_graph(bn.download_bus_routes(bbox, base / "transit" / "osm_bus_routes.json"),
                                    metric_crs, net["transfer_max_m"], bounds=bounds)
    n0 = list(G0.nodes())
    return {"city": city, "cfg": cfg, "base": base, "graph_files": graph_files,
            "samples": fg.sample_od_nodes(G0, cells, k, seed), "flows": flows,
            "frame": MetricFrame.from_graph(G0), "metric_crs": metric_crs,
            "R0": R0, "B0": B0, "rstats": rstats, "bstats": bstats,
            "access": [int(a) for a in net["max_access_m"]], "transfer": net["transfer_max_m"],
            "n0": n0, "x0": np.array([G0.nodes[n]["x"] for n in n0]),
            "y0": np.array([G0.nodes[n]["y"] for n in n0]),
            "dem_file": base / "dem" / f"{city}_dem.tif"}


def compute_embedding(ctx, gf: Path) -> dict:
    """F-weighted sums of one embedding; transit networks moved exactly as the road network."""
    with open(gf, "rb") as f:
        G = pickle.load(f)
    fn = transformation_of(gf.name)
    xt, yt = ctx["frame"].apply(ctx["x0"], ctx["y0"], fn)
    err = max(np.max(np.abs(xt - [G.nodes[n]["x"] for n in ctx["n0"]])),
              np.max(np.abs(yt - [G.nodes[n]["y"] for n in ctx["n0"]])))
    if err > 1e-7:
        raise RuntimeError(f"{gf.name}: transformation mismatch {err:.2e} deg")
    dem = GeoDEM(ctx["dem_file"], fg.graph_bounds(G), margin_deg=0.05)
    R, stops, stop_xy = rn.profile_rail_graph(ctx["R0"], dem, ctx["metric_crs"], ctx["transfer"], ctx["frame"], fn)
    B, bus_nodes, bus_xy = bn.profile_bus_graph(ctx["B0"], dem, ctx["metric_crs"], ctx["transfer"],
                                                ctx["frame"], fn)
    z = node_elevations(G, ctx["dem_file"])
    z = {n: (v if np.isfinite(v) else 0.0) for n, v in z.items()}
    return {"filename": gf.name, "transform_mismatch_deg": float(err),
            **embedding_sums(G, R, stops, stop_xy, B, bus_nodes, bus_xy, ctx["samples"], ctx["flows"],
                             z, ctx["access"], ctx["metric_crs"], dem)}


def scenarios(S: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Observed-minimum status by mode, transformation family and scenario."""
    shares = modal_shares(cfg)
    access = [int(a) for a in cfg["rail_network"]["max_access_m"]]
    modes = ROAD_MODES + ["rail", "mix"]
    fam = pd.Series({f: family_of(f) for f in S.index})
    # "main": the ensemble of the paper (contractions are a separate sensitivity test there)
    groups = {"all": list(S.index), "main": list(fam.index[fam != "scale_down"])}
    groups.update({g: [ORIGINAL] + list(fam.index[fam == g]) for g in sorted(set(fam) - {"original"})})

    def status_rows(E_by_file, extra):
        rows = []
        for mode in modes:
            E = pd.Series({f: e[mode] for f, e in E_by_file.items()})
            for g, files in groups.items():
                is_min, reg, rank, best, margin = observed_status(E[files])
                rows.append({**extra, "family": g, "n_alternatives": len(files) - 1, "mode": mode,
                             "observed_minimum": is_min, "regret_pct": reg, "margin_pct": margin,
                             "rank": rank, "best_alternative": best})
        return rows

    out = []
    central = central_params(cfg)
    for k_t in (1.0,) + TUNNEL_FACTORS:
        for a in access:
            E = {f: scenario_energy(r, central, shares, a, k_t) for f, r in S.iterrows()}
            out += status_rows(E, {"stat": "central", "access_m": a, "tunnel_factor": k_t})
    # Exact worst case over the parameter box (no distribution assumed): E_c - E_0 is separable by
    # vehicle and multilinear in its parameters, so the minimum is reached at vertices of the box.
    boxes = {v: {k: (cfg["modes"][v][k][0], cfg["modes"][v][k][2]) for k in PARAMS} for v in cfg["modes"]}
    alts = [f for f in S.index if f != ORIGINAL]
    for a in access:
        def diff(par):
            E = {f: scenario_energy(S.loc[f], par, shares, a) for f in [ORIGINAL] + alts}
            return np.array([[E[f][m] - E[ORIGINAL][m] for m in modes] for f in alts])
        worst, _ = rt.separable_box_minimum(diff, central, boxes)
        e0 = scenario_energy(S.loc[ORIGINAL], central, shares, a)
        W = pd.DataFrame(worst, index=alts, columns=modes)
        for mode in modes:
            for g, files in groups.items():
                w = W.loc[[f for f in files if f != ORIGINAL], mode]
                out.append({"stat": "worst_case", "access_m": a, "tunnel_factor": 1.0, "family": g,
                            "n_alternatives": len(w), "mode": mode, "observed_minimum": bool((w >= 0).all()),
                            "margin_pct": 100 * w.min() / e0[mode], "n_cheaper": int((w < 0).sum()),
                            "best_alternative": w.idxmin()})

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])
    draws = []
    for i in tqdm(range(cfg["monte_carlo"]["n_draws"]), desc="Monte Carlo"):
        par = draw_params(cfg, rng)
        a = access[rng.integers(len(access))]
        E = {f: scenario_energy(r, par, shares, a) for f, r in S.iterrows()}
        draws += status_rows(E, {"draw": i, "access_m": a})
    D = pd.DataFrame(draws)
    mc = D.groupby(["family", "mode"]).agg(observed_minimum=("observed_minimum", "mean"),
                                           regret_pct=("regret_pct", "max"), margin_pct=("margin_pct", "min"),
                                           n_alternatives=("n_alternatives", "first")).reset_index()
    mc["stat"], mc["tunnel_factor"] = "monte_carlo", 1.0
    return pd.concat([pd.DataFrame(out), mc], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--city", default="barcelone", choices=sorted(CONFIGS))
    ap.add_argument("--task", type=int, default=None,
                    help="index of the embedding to compute (default: $SLURM_ARRAY_TASK_ID if --array-task)")
    ap.add_argument("--array-task", action="store_true", help="take --task from $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--list", action="store_true", help="print the embeddings with their task index")
    ap.add_argument("--merge", action="store_true", help="merge per-embedding sums and evaluate scenarios")
    ap.add_argument("--od", default="twitter", choices=OD_SOURCES,
                    help="OD flows: geolocated records (as step 2) or MITMA open data, all trips or >= 2 km "
                         "(Supplementary Note 9)")
    args = ap.parse_args()
    out_dir = output_root(args.od) / args.city
    sums_dir = out_dir / "sums"
    sums_dir.mkdir(parents=True, exist_ok=True)
    base = fg.DEFAULT_DATA_ROOT / args.city
    graph_files = fg.list_graph_files(base / GRID_DIR)

    if args.list:
        for i, gf in enumerate(graph_files):
            print(i, gf.name)
        return
    task = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.array_task else args.task
    if not args.merge:
        todo = [graph_files[task]] if task is not None else graph_files
        todo = [gf for gf in todo if not (sums_dir / f"{gf.stem}.json").exists()]
        if todo:
            ctx = city_setup(args.city, args.od)
            (out_dir / "network_stats.json").write_text(
                json.dumps({"rail": ctx["rstats"], "bus": ctx["bstats"]}, indent=2, default=float))
            for gf in tqdm(todo, desc="embeddings"):
                row = compute_embedding(ctx, gf)
                tmp = sums_dir / f"{gf.stem}.json.tmp"
                tmp.write_text(json.dumps(row, default=float))
                tmp.replace(sums_dir / f"{gf.stem}.json")
        if task is not None:
            return

    missing = [gf.name for gf in graph_files if not (sums_dir / f"{gf.stem}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} embeddings missing, e.g. {missing[:3]}")
    S = pd.DataFrame([json.loads((sums_dir / f"{gf.stem}.json").read_text()) for gf in graph_files])
    S.to_csv(out_dir / "embedding_sums.csv", index=False)
    res = scenarios(S.set_index("filename"), json.loads(CONFIGS[args.city].read_text()))
    res.to_csv(out_dir / "observed_minimum_by_mode.csv", index=False)
    show = res[(res["family"] == "main") & ((res["stat"] == "monte_carlo") |
                                          ((res["access_m"] == 1500) & (res["tunnel_factor"] == 1.0)))]
    print(show[["stat", "mode", "observed_minimum", "margin_pct", "best_alternative"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {out_dir}")

if __name__ == "__main__":
    main()
