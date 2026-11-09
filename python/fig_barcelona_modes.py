#!/usr/bin/env python3
"""SI figure: Barcelona networks used for the transport-mode test and energy profiles by mode.

(a) road network (cars, motorcycles), bus lines and rail network
    (metro, commuter rail, tram) from the OSM relations used in the analysis;
(b-d) energy of each translated embedding relative to the observed one, for
    rail (bus where rail is unavailable), road vehicles in their EMEF shares,
    and the full EMEF mix. Central parameters, access distance 1.5 km.
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from barcelona_multimodal_robustness import (CONFIG_FILE, RAIL_CACHE, ROUTE_MODE, _norm,  # noqa: E402
                                             central_params, modal_shares)
from barcelona_transform_multimodal import OUTPUT_ROOT, ROAD_MODES, output_root, scenario_energy  # noqa: E402
import bus_network as bn  # noqa: E402
from paper_style import AQUA, BLUE, GRID, INK, INK2, ORANGE, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data/data_processed/barcelone/graphs_fine_grid/graph_original.pkl"
OUT = ROOT / "paper_v2/supp_images/transport_modes/barcelona_modes"
OD = sys.argv[sys.argv.index("--od") + 1] if "--od" in sys.argv else "twitter"
if OD != "twitter":
    OUTPUT_ROOT, OUT = output_root(OD), OUT.with_name(f"{OUT.name}_{OD}")
UTM = "EPSG:25831"
ACCESS = 1500
ROAD_GREY = "#d3d1c9"
RAIL_COLORS = {"metro": BLUE, "train": ORANGE, "tram": AQUA}
RAIL_LABELS = {"metro": "metro", "train": "commuter rail (Rodalies, FGC)", "tram": "tram"}
DIRECTIONS = {90: ("north", "o"), 180: ("west", "s")}


def road_segments(to_m):
    with open(GRAPH, "rb") as f:
        G = pickle.load(f)
    segs = []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        xy = (np.asarray(geom.coords) if geom is not None else
              np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]))
        x, y = to_m.transform(xy[:, 0], xy[:, 1])
        segs.append(np.column_stack([x, y]))
    return segs


def rail_segments(to_m, include):
    osm = json.loads(RAIL_CACHE.read_text())
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in osm["elements"] if e["type"] == "node"}
    ways = {e["id"]: e["nodes"] for e in osm["elements"] if e["type"] == "way"}
    segs = {m: [] for m in RAIL_COLORS}
    for rel in (e for e in osm["elements"] if e["type"] == "relation"):
        tags = rel.get("tags", {})
        mode = ROUTE_MODE.get(tags.get("route"))
        if mode is None or _norm(tags.get("network")) not in include:
            continue
        for m in rel["members"]:
            if m["type"] == "way" and m["ref"] in ways:
                ll = np.array([nodes[n] for n in ways[m["ref"]] if n in nodes])
                if len(ll) > 1:
                    x, y = to_m.transform(ll[:, 0], ll[:, 1])
                    segs[mode].append(np.column_stack([x, y]))
    return segs


def bus_segments(to_m, bbox):
    """Bus line tracks between consecutive stops, as used in the analysis (python/bus_network.py)."""
    osm = bn.download_bus_routes(bbox)
    B0, _ = bn.build_bus_graph(osm, UTM, 300, bounds=(bbox[1], bbox[0], bbox[3], bbox[2]))
    segs = []
    for _, _, d in B0.edges(data=True):
        x, y = to_m.transform(d["poly"][:, 0], d["poly"][:, 1])
        segs.append(np.column_stack([x, y]))
    return segs


def profiles(cfg):
    S = pd.read_csv(OUTPUT_ROOT / "barcelone" / "embedding_sums.csv").set_index("filename")
    par, shares = central_params(cfg), modal_shares(cfg)
    road_tot = sum(shares[m] for m in ROAD_MODES)
    rows = []
    for fname, r in S.iterrows():
        e = scenario_energy(r, par, shares, ACCESS)
        rows.append({"filename": fname, "rail": e["rail"], "mix": e["mix"],
                     "road": sum(shares[m] * e[m] for m in ROAD_MODES) / road_tot})
    E = pd.DataFrame(rows).set_index("filename")
    rel = 100 * (E / E.loc["graph_original.pkl"] - 1)
    m = rel.index.str.extract(r"trans_(\d+)m_a\+(\d+)", expand=True)
    rel["distance_m"] = pd.to_numeric(m[0]).fillna(0).to_numpy()
    rel["angle"] = pd.to_numeric(m[1]).to_numpy()
    return rel


def main():
    cfg = json.loads(CONFIG_FILE.read_text())
    include = {_norm(n) for n in cfg["rail_network"]["include_networks"]}
    to_m = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    rel = profiles(cfg)

    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.28)
    ax_map = fig.add_subplot(gs[:, 0])
    road = road_segments(to_m)
    ax_map.add_collection(LineCollection(road, colors=ROAD_GREY, linewidths=0.15, zorder=1))
    with open(GRAPH, "rb") as f:
        G0 = pickle.load(f)
    lon = [d["x"] for _, d in G0.nodes(data=True)]
    lat = [d["y"] for _, d in G0.nodes(data=True)]
    ax_map.add_collection(LineCollection(bus_segments(to_m, (min(lat), min(lon), max(lat), max(lon))),
                                         colors=INK2, linewidths=0.3, zorder=2))
    for mode, segs in rail_segments(to_m, include).items():
        ax_map.add_collection(LineCollection(segs, colors=RAIL_COLORS[mode], linewidths=1.0, zorder=3))
    # view limited to the road network: rail stops outside it are never the nearest to an O/D node
    allxy = np.vstack(road)
    pad = 2_000
    ax_map.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax_map.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    x0, x1 = ax_map.get_xlim()
    y0, _ = ax_map.get_ylim()
    ax_map.plot([x1 - 12_000, x1 - 2_000], [y0 + 2_000, y0 + 2_000], color=INK, lw=1.2)
    ax_map.text(x1 - 7_000, y0 + 2_800, "10 km", ha="center", va="bottom", fontsize=6.5, color=INK)
    handles = [Line2D([], [], color=ROAD_GREY, lw=1.6, label="road network (car, motorcycle)"),
               Line2D([], [], color=INK2, lw=0.8, label="bus lines")]
    handles += [Line2D([], [], color=RAIL_COLORS[m], lw=1.6, label=RAIL_LABELS[m]) for m in RAIL_COLORS]
    ax_map.legend(handles=handles, frameon=False, loc="upper left", fontsize=6.5, ncol=1,
                  bbox_to_anchor=(0, 0.02), handlelength=1.6, columnspacing=1.0)
    ax_map.set_title("a", loc="left", fontweight="bold", fontsize=8)

    panels = [("rail", "b", "rail (bus where no rail stop is within reach)"),
              ("road", "c", "road vehicles (car, EV, motorcycle; bus on its lines)"),
              ("mix", "d", "all modes (EMEF 2025 mix)")]
    axes = []
    for i, (col, letter, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.axhline(0, color=INK2, lw=0.6)
        for ang, (lab, mk) in DIRECTIONS.items():
            d = rel[rel["angle"] == ang].sort_values("distance_m")
            ax.plot(d["distance_m"], d[col], ls="none", marker=mk, ms=3.5, mfc="white" if mk == "s" else INK2,
                    mec=INK2, mew=0.7, label=f"{lab}ward translation")
        ax.plot([0], [0], marker="*", ms=7, color="#e34948", ls="none", label="observed", zorder=3)
        ax.grid(False, axis="x")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_title(f"{title}", loc="left", fontsize=7, color=INK)
        ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
        ax.set_ylabel(r"$\Delta E$ (%)")
        if i < 2:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("translation distance (m)")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.55))
    save(fig, OUT)
    n = len(rel) - 1
    print(f"embeddings: {n} translated; min dE: " +
          ", ".join(f"{c} {rel.drop('graph_original.pkl')[c].min():.2f}%" for c in ("rail", "road", "mix")))


if __name__ == "__main__":
    main()
