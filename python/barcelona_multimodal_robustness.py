#!/usr/bin/env python3
"""Barcelona densification: robustness of the cell ranking to a multimodal demand (SI test).

The paper ranks candidate cells for 25 000 new residents by the energy of their
round trips by private car. Here the same round trips are split over car, moto,
bus and rail (metro, Rodalies/FGC, tram) with the fixed EMEF 2025 modal split,
and energies are computed per passenger:

    e_m = (m_m g (1 - eps_m) H + lambda_m L) / occ_m

A round trip is a closed loop, so total descent equals total ascent H and
regenerative braking only rescales the vertical term by (1 - eps_m).

* Road modes (car, moto, bus) use the road paths of the paper (3-D length routing).
* Rail uses the stop-to-stop graph of rail_network.py (OpenStreetMap route
  relations): consecutive stops are joined along the tracks, whose profile is
  sampled from the DEM every 10 m where the track is on the terrain and is a
  uniform grade inside tunnels and on bridges; stops of different routes within
  `transfer_max_m` are joined by walking transfers. Each trip boards at the stop
  nearest to the origin cell centroid and alights at the stop nearest to the
  destination centroid. Trips with no rail option (access > max_access_m, same
  stop, no path) take the bus instead.

All per-cell energies are linear in the vehicle parameters, so the Monte Carlo
over parameter ranges only recombines per-cell sums computed once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_city_settlement as mcs  # noqa: E402
import rail_network as rn  # noqa: E402
import ranking_theory as rt  # noqa: E402
from rail_network import RAIL_MODES, ROUTE_MODE, USER_AGENT, WALK_FACTOR, _norm  # noqa: F401,E402
from terrain import GeoDEM  # noqa: E402

CITY = "barcelone"
CONFIG_FILE = Path(__file__).resolve().parent / "config" / "barcelona_multimodal_params.json"
OUTPUT_DIR = mcs.OUTPUT_ROOT.parent / "barcelona_multimodal_robustness"
# OD source of the gravity-model scale d_0: "twitter" (geolocated records, as Fig. 7), "mitma" (Spanish
# Ministry of Transport open data, all trips >= 0.5 km) or "mitma_ge2km" (MITMA trips >= 2 km); see
# python/mitma_od.py (Supplementary Note 9).
OD_SOURCES = ("twitter", "mitma", "mitma_ge2km")
OD_SOURCE = "twitter"


def set_od_source(od: str) -> None:
    """Select the OD source; outputs of the MITMA run go to a separate directory."""
    global OD_SOURCE, OUTPUT_DIR
    if od not in OD_SOURCES:
        raise ValueError(f"unknown OD source {od!r}")
    OD_SOURCE = od
    OUTPUT_DIR = mcs.OUTPUT_ROOT.parent / ("barcelona_multimodal_robustness" + ("" if od == "twitter" else f"_{od}"))


def od_file(city: str) -> Path:
    """Cell OD file (cell_origin, cell_destination, count) of the selected source."""
    if OD_SOURCE.startswith("mitma"):
        suffix = OD_SOURCE.removeprefix("mitma")
        return mcs.city_base_dir(city) / f"{city}_mitma_model" / "1000_cells" / f"od_matrix_mitma_weekday{suffix}.csv"
    return mcs.city_paths(city)["od_working"]
TRANSIT_DIR = mcs.city_base_dir(CITY) / "transit"
OSM_CACHE = TRANSIT_DIR / "osm_rail_routes.json"          # v1 cache (untagged ways), map only
RAIL_CACHE = TRANSIT_DIR / "osm_rail_routes_v2.json"       # tagged ways: tunnels and bridges

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ROAD_MODES = ["car", "car_ev", "moto", "bus"]
SINGLE_MODES = ROAD_MODES + ["rail"]
RATIO_GRID = np.linspace(0.0, 200.0, 801)   # r = m g (1 - eps) / lambda
PARAMS = ["mass_kg", "lambda_J_per_m", "occupancy", "regen_eff"]
TUNNEL_FACTORS = (0.0, 2.0)                  # metro gain inside tunnels: none / doubled


# =============================================================================
# Per-cell sums (computed once)
# =============================================================================

def cell_features(cand_idx, cells, D, G, G_rev, d_0, R, stop_nodes, stop_xy, access_thresholds,
                  rail_cache, router=None, bus_xy=None, bus_cache=None):
    """p-weighted mean round-trip L and H per mode family, normalised to the routed demand.

    Bus (if `router` is given) uses the bus lines: outbound and return are routed separately on
    the directed bus graph; ODs without a stop within the access distance use the road path.
    Rail trips without a usable rail option fall back to the bus in the same way.
    """
    populations = cells["population"].to_numpy(dtype=float)
    attraction = populations ** mcs.ALPHA * np.exp(-D[cand_idx] / d_0)
    attraction[cand_idx] = 0.0
    if attraction.sum() <= 0:
        return None
    p = attraction / attraction.sum()
    dests = np.flatnonzero(p > mcs.GRAVITY_DESTINATION_THRESHOLD)
    dest_nodes = cells["nearest_node"].to_numpy()[dests]
    source = cells.at[cand_idx, "nearest_node"]

    pred_out, _ = nx.dijkstra_predecessor_and_distance(G, source, weight=mcs.ROUTING_WEIGHT)
    pred_ret, _ = nx.dijkstra_predecessor_and_distance(G_rev, source, weight=mcs.ROUTING_WEIGHT)
    out = mcs.tree_sums(pred_out, source, dest_nodes, lambda a, b: mcs.edge_cost_terms(G, a, b))
    ret = mcs.tree_sums(pred_ret, source, dest_nodes, lambda a, b: mcs.edge_cost_terms(G, b, a))

    access_tree = KDTree(stop_xy)
    acc_o, i_o = access_tree.query(cells.loc[cand_idx, ["centroid_x", "centroid_y"]].to_numpy(dtype=float))
    acc_d, i_d = access_tree.query(cells.loc[cells.index[dests], ["centroid_x", "centroid_y"]].to_numpy(dtype=float))
    s_o = stop_nodes[i_o]
    if s_o not in rail_cache:
        rail_cache[s_o] = rn.rail_source_sums(R, s_o)
    rail_from_o = rail_cache[s_o]

    def bus_sums(i):
        if i not in bus_cache:
            bus_cache[i] = router.sums(i)
        return bus_cache[i]

    if router is not None:
        bus_tree = KDTree(bus_xy)
        bacc_o, b_o = bus_tree.query(cells.loc[cand_idx, ["centroid_x", "centroid_y"]].to_numpy(dtype=float))
        bacc_d, b_d = bus_tree.query(cells.loc[cells.index[dests], ["centroid_x", "centroid_y"]].to_numpy(dtype=float))

    rows = []
    for k, (j, node) in enumerate(zip(dests, dest_nodes)):
        a, b = out.get(node), ret.get(node)
        if a is None or b is None:
            continue
        s_d = stop_nodes[i_d[k]]
        rail = rail_from_o.get(s_d) if s_d != s_o else None
        bus = None
        if router is not None and b_d[k] != b_o:
            out_b, ret_b = bus_sums(b_o)[:, b_d[k]], bus_sums(b_d[k])[:, b_o]
            if np.isfinite(out_b[0]) and np.isfinite(ret_b[0]):
                bus = (float(out_b[0] + ret_b[0]), float(out_b[1] + ret_b[1]), float(bacc_d[k]))
        rows.append((p[j], a[0] + b[0], a[1] + b[1], acc_d[k], rail, bus))
    if not rows:
        return None

    w = np.array([r[0] for r in rows])
    w = w / w.sum()
    L = np.array([r[1] for r in rows])
    H = np.array([r[2] for r in rows])
    feats = {"road_L": float(w @ L), "road_H": float(w @ H)}

    for a_max in access_thresholds:
        tag = f"a{int(a_max)}"
        ok = np.array([r[4] is not None and acc_o <= a_max and r[3] <= a_max for r in rows])
        feats[f"{tag}_rail_share"] = float(w[ok].sum())
        bus_ok = np.array([r[5] is not None and bacc_o <= a_max and r[5][2] <= a_max for r in rows]) \
            if router is not None else np.zeros(len(rows), bool)
        bL = np.array([r[5][0] if okb else 0.0 for r, okb in zip(rows, bus_ok)])
        bH = np.array([r[5][1] if okb else 0.0 for r, okb in zip(rows, bus_ok)])
        feats[f"{tag}_bus_share"] = float(w[bus_ok].sum())
        feats[f"{tag}_busnet_L"], feats[f"{tag}_busnet_H"] = float(w @ bL), float(w @ bH)
        feats[f"{tag}_busroad_L"] = float(w[~bus_ok] @ L[~bus_ok])
        feats[f"{tag}_busroad_H"] = float(w[~bus_ok] @ H[~bus_ok])
        fb_net, fb_road = ~ok & bus_ok, ~ok & ~bus_ok      # rail unavailable -> bus network / road
        feats[f"{tag}_rfbnet_L"], feats[f"{tag}_rfbnet_H"] = float(w[fb_net] @ bL[fb_net]), float(w[fb_net] @ bH[fb_net])
        feats[f"{tag}_rfbroad_L"], feats[f"{tag}_rfbroad_H"] = float(w[fb_road] @ L[fb_road]), float(w[fb_road] @ H[fb_road])
        for m in RAIL_MODES:
            # return on the reverse path: same length, its uphill is the outbound downhill
            lm = np.array([2 * r[4][m][0] if okk else 0.0 for r, okk in zip(rows, ok)])
            hm = np.array([r[4][m][1] + r[4][m][2] if okk else 0.0 for r, okk in zip(rows, ok)])
            ht = np.array([r[4][m][3] + r[4][m][4] if okk else 0.0 for r, okk in zip(rows, ok)])
            feats[f"{tag}_{m}_L"] = float(w @ lm)
            feats[f"{tag}_{m}_H"] = float(w @ hm)
            feats[f"{tag}_{m}_Ht"] = float(w @ ht)       # part of H in tunnels / on bridges
        feats[f"{tag}_access_o_m"] = float(acc_o)

    feats["cell_id"] = cells.at[cand_idx, "cell_id"]
    feats["centroid_x"] = cells.at[cand_idx, "centroid_x"]
    feats["centroid_y"] = cells.at[cand_idx, "centroid_y"]
    feats["cand_idx"] = cand_idx
    return feats


# =============================================================================
# Energies
# =============================================================================

def mode_energy(par, L, H):
    """Per-passenger round-trip energy [J] for one mode (H = round-trip uphill)."""
    return (par["mass_kg"] * mcs.G_PHYS * (1.0 - par["regen_eff"]) * H
            + par["lambda_J_per_m"] * L) / par["occupancy"]


def mixed_energy(df, par, shares, tag, tunnel_factor=1.0):
    """Mean per-resident round-trip energy [J] under the fixed modal split."""
    e = sum(shares.get(m, 0.0) * single_mode_energy(df, par, m, tag) for m in ROAD_MODES)
    return e + shares.get("rail", 0.0) * rail_energy(df, par, tag, tunnel_factor)


def _leg(df, par, mode, prefix):
    return mode_energy(par[mode], df[f"{prefix}_L"].to_numpy(), df[f"{prefix}_H"].to_numpy())


def rail_energy(df, par, tag, tunnel_factor=1.0):
    """Per-resident energy if every trip took rail (bus where rail is not available).

    tunnel_factor multiplies the metro gain on non-terrain track (sensitivity to the
    uniform-grade assumption inside tunnels)."""
    def h(m):
        H = df[f"{tag}_{m}_H"].to_numpy()
        if m == "metro" and tunnel_factor != 1.0:
            H = H + (tunnel_factor - 1.0) * df[f"{tag}_{m}_Ht"].to_numpy()
        return H
    rail = sum(mode_energy(par[m], df[f"{tag}_{m}_L"].to_numpy(), h(m)) for m in RAIL_MODES)
    return rail + _leg(df, par, "bus", f"{tag}_rfbnet") + _leg(df, par, "bus", f"{tag}_rfbroad")


def single_mode_energy(df, par, mode, tag):
    """Per-resident energy if every trip used `mode` (road mode, or 'rail')."""
    if mode == "rail":
        return rail_energy(df, par, tag)
    if mode == "bus":
        return _leg(df, par, "bus", f"{tag}_busnet") + _leg(df, par, "bus", f"{tag}_busroad")
    return mode_energy(par[mode], df["road_L"].to_numpy(), df["road_H"].to_numpy())


def ratio_range(mode_cfg):
    """(low, central, high) of r = m g (1 - eps) / lambda over the parameter box."""
    m, lam, eps = (mode_cfg[k] for k in ("mass_kg", "lambda_J_per_m", "regen_eff"))
    r = lambda mi, ei, li: mi * mcs.G_PHYS * (1.0 - ei) / li  # noqa: E731
    return r(m[0], eps[2], lam[2]), r(m[1], eps[1], lam[1]), r(m[2], eps[0], lam[0])


def central_params(cfg):
    return {m: {k: v[k][1] for k in PARAMS} for m, v in cfg["modes"].items()}


def draw_params(cfg, rng):
    return {m: {k: rng.uniform(v[k][0], v[k][2]) for k in PARAMS} for m, v in cfg["modes"].items()}


def modal_shares(cfg):
    """Mode shares renormalised over the modelled modes ('trips_thousands' or 'trips_percent')."""
    ms = cfg["modal_split"]
    trips = ms["trips_thousands"] if "trips_thousands" in ms else ms["trips_percent"]
    tot = sum(trips.values())
    return {m: t / tot for m, t in trips.items()}


# =============================================================================
# Robustness metrics
# =============================================================================

def ordering_metrics(E_ref, E, fav_idx, unf_idx, comparable=None, k=10):
    """Agreement between the car-only ranking E_ref and E.

    Over all candidates the ranking is dominated by trip length, so the
    informative statistic is `spearman_comparable`: the rank correlation among
    the cells with comparable horizontal cost, where only terrain differs.
    """
    rho = spearmanr(E_ref, E).statistic
    tau = kendalltau(E_ref, E).statistic
    top_ref, top = set(np.argsort(E_ref)[:k]), set(np.argsort(E)[:k])
    bot_ref, bot = set(np.argsort(E_ref)[-k:]), set(np.argsort(E)[-k:])
    pairs = [(f, u) for f in fav_idx for u in unf_idx]
    return {
        "spearman": rho,
        "spearman_comparable": (spearmanr(E_ref[comparable], E[comparable]).statistic
                                if comparable is not None else np.nan),
        "kendall": tau,
        f"top{k}_overlap": len(top_ref & top) / k,
        f"bottom{k}_overlap": len(bot_ref & bot) / k,
        "fav_below_unf_pairs": np.mean([E[f] < E[u] for f, u in pairs]),
        "all_fav_below_all_unf": float(max(E[fav_idx]) < min(E[unf_idx])),
    }


def worst_case_order(df, cfg, shares, fav_idx, unf_idx, access_thresholds, tunnel_factors=(1.0,)):
    """Exact worst case of the Fig. 7 order over the parameter box (no distribution assumed).

    For every unfavourable/favourable pair, E_u - E_f is separable by vehicle and multilinear in its
    parameters, so its minimum over the box is reached at vertices (ranking_theory.separable_box_minimum).
    The order holds for every parameter set iff all pair minima are >= 0.
    """
    central = central_params(cfg)
    boxes = {v: {k: (cfg["modes"][v][k][0], cfg["modes"][v][k][2]) for k in PARAMS} for v in cfg["modes"]}
    pairs = [(u, f) for u in unf_idx for f in fav_idx]
    rows = []
    for k_t in tunnel_factors:
        for scen in SINGLE_MODES + ["mix"]:
            for a in access_thresholds:
                tag = f"a{int(a)}"

                def energy(par):
                    if scen == "mix":
                        return mixed_energy(df, par, shares, tag, k_t)
                    if scen == "rail":
                        return rail_energy(df, par, tag, k_t)
                    return single_mode_energy(df, par, scen, tag)

                def diff(par):
                    E = energy(par)
                    return np.array([E[u] - E[f] for u, f in pairs])
                worst, _ = rt.separable_box_minimum(diff, central, boxes)
                E0 = energy(central)
                rows.append({"mode": scen, "tunnel_factor": k_t, "access_m": a,
                             "order_holds_everywhere": bool((worst >= 0).all()),
                             "worst_pair_margin_pct": float(min(100 * w / E0[f] for w, (u, f) in zip(worst, pairs))),
                             "n_pairs_failing": int((worst < 0).sum())})
    return pd.DataFrame(rows)


def worst_case_from_saved(cfg):
    """Worst-case Fig. 7 order from the saved per-cell sums (no rerouting)."""
    df = pd.read_csv(OUTPUT_DIR / "candidate_cells_multimodal.csv", sep=";")
    sel = pd.read_csv(OUTPUT_DIR / "selected_cells_car_baseline.csv", sep=";")
    pos = {cid: i for i, cid in enumerate(df["cell_id"])}
    fav = [pos[c] for c in sel.loc[sel["scenario_type"] == "favorable", "cell_id"]]
    unf = [pos[c] for c in sel.loc[sel["scenario_type"] == "unfavorable", "cell_id"]]
    res = worst_case_order(df, cfg, modal_shares(cfg), fav, unf, cfg["rail_network"]["max_access_m"],
                           (1.0,) + TUNNEL_FACTORS)
    res.to_csv(OUTPUT_DIR / "robustness_worst_case.csv", index=False, sep=";")
    print(res.groupby(["tunnel_factor", "mode"]).agg(order_everywhere=("order_holds_everywhere", "all"),
                                                       worst_margin_pct=("worst_pair_margin_pct", "min"),
                                                       pairs_failing=("n_pairs_failing", "max")).round(2).to_string())


def main():
    cfg = json.loads(CONFIG_FILE.read_text())
    shares = modal_shares(cfg)
    net_cfg = cfg["rail_network"]
    access_thresholds = net_cfg["max_access_m"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Modal split (motorised, renormalised):", {m: round(s, 4) for m, s in shares.items()})

    paths = mcs.city_paths(CITY)
    fua = mcs.load_city_fua(CITY)
    metric_crs = fua.estimate_utm_crs()
    cells = mcs.extract_population_to_cells(mcs.load_cells(paths["cells"], metric_crs), paths["worldpop"], fua)

    G = mcs.load_graph(paths["graph"])
    graph_crs = G.graph.get("crs", "EPSG:4326")
    mcs.annotate_edges(G, graph_crs, metric_crs, mcs.MetricDEM(paths["dem"], metric_crs))
    G_rev = G.reverse(copy=False)
    nodes, node_xy = mcs.graph_node_coords(G, graph_crs, metric_crs)
    cells = mcs.assign_nearest_nodes(cells, nodes, node_xy)
    D = cdist(cells[["centroid_x", "centroid_y"]].to_numpy(), cells[["centroid_x", "centroid_y"]].to_numpy())
    fit = mcs.fit_d0_production_constrained(od_file(CITY), cells, D)
    d_0 = fit["d_0"]
    print(f"OD source {OD_SOURCE}: d_0 = {d_0 / 1000:.2f} +/- {fit['d_0_se'] / 1000:.2f} km "
          f"({fit['n_pairs_positive']:,} positive pairs)")

    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    bbox = (min(ys), min(xs), max(ys), max(xs))
    bounds = (bbox[1], bbox[0], bbox[3], bbox[2])
    R0, rstats = rn.build_rail_graph(rn.download_rail_routes(bbox, RAIL_CACHE), net_cfg["include_networks"],
                                     metric_crs, bounds=bounds)
    R, stop_nodes, stop_xy = rn.profile_rail_graph(
        R0, GeoDEM(paths["dem"], bounds, margin_deg=0.05), metric_crs, net_cfg["transfer_max_m"])
    print("Rail graph:", rstats)
    import bus_network as bn
    B0, bstats = bn.build_bus_graph(bn.download_bus_routes(bbox), metric_crs, net_cfg["transfer_max_m"],
                                    bounds=(bbox[1], bbox[0], bbox[3], bbox[2]))
    B, bus_nodes, bus_xy = bn.profile_bus_graph(
        B0, GeoDEM(paths["dem"], (bbox[1], bbox[0], bbox[3], bbox[2]), margin_deg=0.05),
        metric_crs, net_cfg["transfer_max_m"])
    router = bn.BusRouter(B)
    assert router.nodes == bus_nodes          # same B.nodes() order: KDTree index == router index
    print("Bus graph:", bstats)

    candidates = mcs.select_candidate_cells(cells, mcs.MAX_CANDIDATE_CELLS)
    rail_cache, bus_cache = {}, {}
    feats = [f for f in (cell_features(c, cells, D, G, G_rev, d_0, R, stop_nodes, stop_xy,
                                       access_thresholds, rail_cache, router, bus_xy, bus_cache)
                         for c in tqdm(candidates.index, desc="candidate cells")) if f is not None]
    df = pd.DataFrame(feats).reset_index(drop=True)

    # Paper baseline: car only, per vehicle, m = 1200 kg, lambda = 585 J/m
    n_rt = mcs.NEW_RESIDENTS * mcs.ROUND_TRIPS_PER_PERSON_PER_DAY
    df["vertical_work_joules"] = mcs.M_PHYS_KG * mcs.G_PHYS * n_rt * df["road_H"]
    df["horizontal_cost_joules"] = mcs.LAMBDA_J_PER_M * n_rt * df["road_L"]
    df["total_work_joules"] = df["vertical_work_joules"] + df["horizontal_cost_joules"]
    sel = mcs.select_favorable_unfavorable(df)
    pos = {cid: i for i, cid in enumerate(df["cell_id"])}
    fav_idx = [pos[c] for c in sel.loc[sel["scenario_type"] == "favorable", "cell_id"]]
    unf_idx = [pos[c] for c in sel.loc[sel["scenario_type"] == "unfavorable", "cell_id"]]
    E_ref = df["total_work_joules"].to_numpy()
    h = df["horizontal_cost_joules"]
    tol = sel["horizontal_tolerance_used"].iloc[0]
    comparable = ((h - h.median()).abs() <= tol * h.median()).to_numpy()
    df["horizontally_comparable"] = comparable
    print(f"Horizontally comparable cells (+/-{tol:.0%} of median): {comparable.sum()}; "
          f"car-only vs horizontal-only rank correlation among them: "
          f"{spearmanr(E_ref[comparable], df['road_L'][comparable]).statistic:.3f}")

    central = central_params(cfg)
    tag_c = f"a{int(access_thresholds[1])}"
    df["E_mix_central_J_per_resident"] = mixed_energy(df, central, shares, tag_c)

    rows = []
    for a in access_thresholds:
        rows.append({"scenario": "central", "access_m": a,
                     **ordering_metrics(E_ref, mixed_energy(df, central, shares, f"a{int(a)}"), fav_idx, unf_idx,
                                         comparable)})

    rng = np.random.default_rng(cfg["monte_carlo"]["seed"])
    mc = []
    for _ in tqdm(range(cfg["monte_carlo"]["n_draws"]), desc="Monte Carlo"):
        a = access_thresholds[rng.integers(len(access_thresholds))]
        E = mixed_energy(df, draw_params(cfg, rng), shares, f"a{int(a)}")
        mc.append({"access_m": a, **ordering_metrics(E_ref, E, fav_idx, unf_idx, comparable)})
    mc = pd.DataFrame(mc)

    # Every trip by a single mode: central parameters and Monte Carlo over its box
    by_mode = []
    for mode in SINGLE_MODES:
        E = single_mode_energy(df, central, mode, tag_c)
        by_mode.append({"mode": mode, "stat": "central", **ordering_metrics(E_ref, E, fav_idx, unf_idx, comparable)})
        draws = pd.DataFrame([
            ordering_metrics(E_ref, single_mode_energy(df, draw_params(cfg, rng), mode,
                                                       f"a{int(access_thresholds[rng.integers(len(access_thresholds))])}"),
                             fav_idx, unf_idx, comparable)
            for _ in range(cfg["monte_carlo"]["n_draws"] // 4)])
        for qq, lab in ((0.05, "q05"), (0.95, "q95")):
            by_mode.append({"mode": mode, "stat": lab, **draws.quantile(qq).to_dict()})
        by_mode.append({"mode": mode, "stat": "min", **draws.min().to_dict()})
        by_mode.append({"mode": mode, "stat": "frac", **draws.mean().to_dict()})   # mean = fraction for 0/1
    # Sensitivity to the metro profile inside tunnels (gain on non-terrain track x 0 or x 2).
    # Drawn after all other scenarios, so the earlier random sequence is unchanged.
    for k_t in TUNNEL_FACTORS:
        for mode, fun in (("rail", rail_energy), ("mix", None)):
            def energy(par, tag):
                return (rail_energy(df, par, tag, k_t) if fun is not None
                        else mixed_energy(df, par, shares, tag, k_t))
            lab = f"{mode}_tunnel_x{k_t:g}"
            by_mode.append({"mode": lab, "stat": "central",
                            **ordering_metrics(E_ref, energy(central, tag_c), fav_idx, unf_idx, comparable)})
            draws = pd.DataFrame([
                ordering_metrics(E_ref, energy(draw_params(cfg, rng),
                                               f"a{int(access_thresholds[rng.integers(len(access_thresholds))])}"),
                                 fav_idx, unf_idx, comparable)
                for _ in range(cfg["monte_carlo"]["n_draws"] // 4)])
            for qq, q_lab in ((0.05, "q05"), (0.95, "q95")):
                by_mode.append({"mode": lab, "stat": q_lab, **draws.quantile(qq).to_dict()})
            by_mode.append({"mode": lab, "stat": "min", **draws.min().to_dict()})
            by_mode.append({"mode": lab, "stat": "frac", **draws.mean().to_dict()})
    by_mode = pd.DataFrame(by_mode)

    # Road vehicles: ranking depends only on r = m g (1 - eps) / lambda
    sweep = pd.DataFrame([{"r": r, **ordering_metrics(E_ref, r * df["road_H"].to_numpy() + df["road_L"].to_numpy(),
                                                       fav_idx, unf_idx, comparable)} for r in RATIO_GRID])
    ranges = pd.DataFrame([{"mode": m, **dict(zip(("r_low", "r_central", "r_high"), ratio_range(cfg["modes"][m])))}
                           for m in ROAD_MODES + RAIL_MODES])
    flips = sweep.loc[sweep["all_fav_below_all_unf"] < 1, "r"]
    r_empirical = (ranges["r_low"].min(), ranges["r_high"].max())
    in_range = sweep[(sweep["r"] >= r_empirical[0]) & (sweep["r"] <= r_empirical[1])]

    summary = pd.DataFrame(rows).drop_duplicates()
    q = mc.drop(columns="access_m").quantile([0.05, 0.5, 0.95]).T
    q["mean"] = mc.drop(columns="access_m").mean()

    df.to_csv(OUTPUT_DIR / "candidate_cells_multimodal.csv", index=False, sep=";")
    sel.to_csv(OUTPUT_DIR / "selected_cells_car_baseline.csv", index=False, sep=";")
    summary.to_csv(OUTPUT_DIR / "robustness_central.csv", index=False, sep=";")
    mc.to_csv(OUTPUT_DIR / "robustness_monte_carlo_draws.csv", index=False, sep=";")
    q.to_csv(OUTPUT_DIR / "robustness_monte_carlo_summary.csv", sep=";")
    by_mode.to_csv(OUTPUT_DIR / "robustness_by_mode.csv", index=False, sep=";")
    sweep.to_csv(OUTPUT_DIR / "robustness_ratio_sweep.csv", index=False, sep=";")
    ranges.to_csv(OUTPUT_DIR / "mode_ratio_ranges.csv", index=False, sep=";")
    (OUTPUT_DIR / "rail_graph_stats.json").write_text(json.dumps(
        {**rstats, "modal_split": shares, "d_0_m": d_0, "d_0_se_m": fit["d_0_se"], "od_source": OD_SOURCE,
         "bus_graph": bstats,
         **{f"rail_share_mean_{a}": float(df[f"a{int(a)}_rail_share"].mean()) for a in access_thresholds},
         **{f"bus_share_mean_{a}": float(df[f"a{int(a)}_bus_share"].mean()) for a in access_thresholds}},
        indent=2, default=float))

    print("\nCentral parameters:\n", summary.to_string(index=False, float_format="%.3f"))
    print("\nMonte Carlo (5% / 50% / 95% / mean):\n", q.to_string(float_format="%.3f"))
    print("\nSingle mode (all trips by one mode):\n",
          by_mode[["mode", "stat", "spearman_comparable", "fav_below_unf_pairs", "all_fav_below_all_unf"]]
          .to_string(index=False, float_format="%.3f"))
    print("\nr = m g (1 - eps) / lambda per mode:\n", ranges.to_string(index=False, float_format="%.1f"))
    print(f"\nRatio sweep: empirical range r in [{r_empirical[0]:.1f}, {r_empirical[1]:.1f}]; "
          f"min spearman_comparable there = {in_range['spearman_comparable'].min():.3f}; "
          f"Fig. 7 ordering holds for all r in range: {bool(in_range['all_fav_below_all_unf'].all())}; "
          f"r where it fails: {'none in [0, 200]' if flips.empty else f'{flips.min():.2f}..{flips.max():.2f}'}; "
          f"r = 0 (no terrain): spearman_comparable = {sweep['spearman_comparable'].iloc[0]:.3f}")
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    if "--od" in sys.argv:
        set_od_source(sys.argv[sys.argv.index("--od") + 1])
    if "--worst-case" in sys.argv:
        worst_case_from_saved(json.loads(CONFIG_FILE.read_text()))
    else:
        main()
