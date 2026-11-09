#!/usr/bin/env python3
"""SI figures for Madrid (multimodal transformation test), companions of the Barcelona ones.

madrid_ratio: excess energy of every alternative of the main ensemble (translations, rotations,
              expansions) relative to the observed embedding as a function of r (road network).
madrid_modes: (a) road, bus and rail networks; (b-j) energy of each alternative relative to the
              observed one, by family (columns) and mode group (rows). Central parameters, 1.5 km.
"""

import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bus_network as bn  # noqa: E402
import rail_network as rn  # noqa: E402
import ranking_theory as rt  # noqa: E402
from barcelona_multimodal_robustness import central_params, modal_shares, ratio_range  # noqa: E402
from barcelona_transform_multimodal import (CONFIGS, ROAD_MODES, family_of, output_root,  # noqa: E402
                                            scenario_energy)
from paper_style import AQUA, BAND, BLUE, GRID, INK, INK2, MUTED, ORANGE, save  # noqa: E402
from vehicle_ratio_sweep import L_COL, ORIGINAL, R_PAPER, RTOL, U_COL, load_city  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/data_processed/madrid"
OUT = ROOT / "paper_v2/supp_images/transport_modes"
UTM = "EPSG:25830"
ACCESS = 1500
ROAD_GREY = "#d3d1c9"
RAIL_COLORS = {"metro": BLUE, "train": ORANGE, "tram": AQUA}
RAIL_LABELS = {"metro": "metro", "train": "commuter rail (Cercanías)", "tram": "light rail, tram"}
FAM_COLORS = {"trans": MUTED, "rot": ORANGE, "scale_up": AQUA}
FAM_LABELS = {"trans": "translations", "rot": "rotations", "scale_up": "expansions"}
OD = sys.argv[sys.argv.index("--od") + 1] if "--od" in sys.argv else "twitter"
SUMS_ROOT = output_root(OD)
SUFFIX = "" if OD == "twitter" else f"_{OD}"
DIRS = {0: "east", 30: "east-north-east", 45: "north-east", 90: "north", 135: "north-west", 180: "west",
        225: "south-west", 270: "south", 315: "south-east"}


def pretty(fname):
    n = fname.replace("graph_", "").replace(".pkl", "")
    m = re.fullmatch(r"trans_(\d+)m_a\+(\d+)", n)
    if m:
        return f"{m.group(1)} m {DIRS.get(int(m.group(2)), m.group(2) + ' deg')}"
    m = re.fullmatch(r"rot_([+-]\d+)p(\d+)deg", n)
    if m:
        return f"rotation {float(m.group(1) + '.' + m.group(2)):+g} deg"
    m = re.fullmatch(r"scale_(ew|ns)_(\d+)p(\d+)", n)
    return f"{m.group(1).upper()} scaling {m.group(2)}.{m.group(3)}" if m else n


def parse(fname):
    n = fname.replace("graph_", "").replace(".pkl", "")
    m = re.fullmatch(r"trans_(\d+)m_a([+-]\d+)", n)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"rot_([+-]\d+)p(\d+)deg", n)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.fullmatch(r"scale_(ew|ns)_(\d+)p(\d+)", n)
    return float(f"{m.group(2)}.{m.group(3)}") if m else 0.0


def ratio_figure(cfg):
    if OD != "twitter":   # road sums of the MITMA run (3-D length and climbing of the road paths)
        from vehicle_ratio_sweep import main_ensemble_mask
        S = pd.read_csv(SUMS_ROOT / "madrid" / "embedding_sums.csv")
        df = S[main_ensemble_mask(S["filename"])].assign(**{L_COL: S["road_L"] / S["F"], U_COL: S["road_Hu"] / S["F"]})
    else:
        df = load_city(BASE / "graphs_fine_grid/fine_grid_gravitational_work.csv")
    df = df[[family_of(f) != "scale_down" for f in df.filename]]
    o = df[df.filename == ORIGINAL].iloc[0]
    var = df[df.filename != ORIGINAL]
    rr = pd.DataFrame([ratio_range(cfg["modes"][m]) for m in cfg["modes"]])
    lo, hi = rr[0].min(), rr[2].max()
    iv = rt.minimizer_interval(o[L_COL], o[U_COL], var[L_COL], var[U_COL], rtol=RTOL)
    assert iv is not None and iv[0] <= lo and hi <= iv[1], "observed embedding not minimal on the range"
    r = np.linspace(0, 100, 400)
    E0 = o[L_COL] + r * o[U_COL]
    ex = 100 * ((var[L_COL].to_numpy()[:, None] + np.outer(var[U_COL], r)) / E0 - 1)
    fam = var.filename.map(family_of).to_numpy()
    best = int(np.argmin(ex[:, np.searchsorted(r, R_PAPER)]))

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 0.97, "range of r of all vehicles", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK2, fontsize=6.5)
    ax.axhline(0, color=INK2, lw=0.8)
    for f in ("trans", "rot", "scale_up"):
        for k in np.flatnonzero(fam == f):
            if k != best:
                ax.plot(r, ex[k], color=FAM_COLORS[f], lw=0.5, alpha=0.6)
    ax.plot(r, ex[best], color=BLUE, lw=1.4)
    ax.text(97, -0.35, "closest alternative: " + pretty(var.filename.iloc[best]), color=BLUE, fontsize=6.5,
            ha="right", va="top")
    ax.axvline(R_PAPER, color=INK2, lw=0.6, ls="--")
    ax.text(R_PAPER + 1, 0.60, r"$r_0$ (paper)", transform=ax.get_xaxis_transform(), color=INK2, fontsize=6.5)
    ax.legend(handles=[Line2D([], [], color=FAM_COLORS[f], lw=1.0, label=FAM_LABELS[f]) for f in FAM_COLORS],
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=-1.6)
    ax.grid(False, axis="x")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_xlabel(r"$r = mg(1-\varepsilon)/\lambda$ (dimensionless)")
    ax.set_ylabel("energy relative to observed (%)")
    save(fig, OUT / f"madrid_ratio{SUFFIX}")
    print(f"ratio: R0 = [{iv[0]:.2f}, {iv[1]:.2f}], band [{lo:.1f}, {hi:.1f}], "
          f"min excess over band {ex[:, (r >= lo) & (r <= hi)].min():.2f} %")


def profiles(cfg):
    S = pd.read_csv(SUMS_ROOT / "madrid" / "embedding_sums.csv").set_index("filename")
    par, shares = central_params(cfg), modal_shares(cfg)
    road_tot = sum(shares[m] for m in ROAD_MODES)
    rows = []
    for f, r in S.iterrows():
        e = scenario_energy(r, par, shares, ACCESS)
        rows.append({"filename": f, "rail": e["rail"], "mix": e["mix"],
                     "road": sum(shares[m] * e[m] for m in ROAD_MODES) / road_tot})
    E = pd.DataFrame(rows).set_index("filename")
    rel = 100 * (E / E.loc[ORIGINAL] - 1)
    rel["family"] = [family_of(f) for f in rel.index]
    rel["x"] = [parse(f) for f in rel.index]
    rel["axis"] = ["ew" if "_ew_" in f else "ns" if "_ns_" in f else "" for f in rel.index]
    # translations along the two cardinal axes, signed (angle from east, counter-clockwise):
    # north-south: +north / -south; east-west: +east / -west. Oblique directions get no signed axis.
    ang = [int(m.group(1)) if (m := re.search(r"_a([+-]\d+)\.pkl$", f)) else None for f in rel.index]
    sign = {90: ("ns", 1), 270: ("ns", -1), 0: ("ew", 1), 180: ("ew", -1)}
    rel["card"] = [sign[a][0] if a in sign else "" for a in ang]
    rel["signed_km"] = [sign[a][1] * x / 1000 if a in sign else np.nan for a, x in zip(ang, rel["x"])]
    rel.loc[ORIGINAL, ["card", "signed_km"]] = ["origin", 0.0]
    return rel[rel.family != "scale_down"]


def segments(G, to_m, cfg, bbox):
    road = []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        xy = (np.asarray(geom.coords) if geom is not None else
              np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]))
        road.append(np.column_stack(to_m.transform(xy[:, 0], xy[:, 1])))
    bounds = (bbox[1], bbox[0], bbox[3], bbox[2])
    B0, _ = bn.build_bus_graph(json.loads((BASE / "transit/osm_bus_routes.json").read_text()), UTM, 300, bounds=bounds)
    bus = [np.column_stack(to_m.transform(d["poly"][:, 0], d["poly"][:, 1])) for _, _, d in B0.edges(data=True)]
    R0, _ = rn.build_rail_graph(json.loads((BASE / "transit/osm_rail_routes_v2.json").read_text()),
                                cfg["rail_network"]["include_networks"], UTM, bounds=bounds)
    rail = {m: [] for m in RAIL_COLORS}
    for _, _, d in R0.edges(data=True):
        rail[d["mode"]].append(np.column_stack(to_m.transform(d["poly"][:, 0], d["poly"][:, 1])))
    return road, bus, rail


def modes_figure(cfg):
    rel = profiles(cfg)
    with open(BASE / "graphs_fine_grid" / ORIGINAL, "rb") as f:
        G = pickle.load(f)
    lon = [d["x"] for _, d in G.nodes(data=True)]
    lat = [d["y"] for _, d in G.nodes(data=True)]
    to_m = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    road, bus, rail = segments(G, to_m, cfg, (min(lat), min(lon), max(lat), max(lon)))

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(3, 6, width_ratios=[1.3, 0.2, 0.8, 0.8, 0.8, 0.62], hspace=0.62, wspace=0.2)
    ax = fig.add_subplot(gs[:, 0])
    ax.add_collection(LineCollection(road, colors=ROAD_GREY, linewidths=0.15, zorder=1))
    ax.add_collection(LineCollection(bus, colors=INK2, linewidths=0.25, zorder=2))
    for m, segs in rail.items():
        ax.add_collection(LineCollection(segs, colors=RAIL_COLORS[m], linewidths=1.0, zorder=3))
    allxy = np.vstack(road)
    ax.set_xlim(allxy[:, 0].min() - 2000, allxy[:, 0].max() + 2000)
    ax.set_ylim(allxy[:, 1].min() - 2000, allxy[:, 1].max() + 2000)
    ax.set_aspect("equal")
    ax.set_axis_off()
    x0, x1 = ax.get_xlim()
    y0, _ = ax.get_ylim()
    ax.plot([x1 - 12_000, x1 - 2_000], [y0 + 2_000, y0 + 2_000], color=INK, lw=1.2)
    ax.text(x1 - 7_000, y0 + 2_800, "10 km", ha="center", va="bottom", fontsize=6.5, color=INK)
    h = [Line2D([], [], color=ROAD_GREY, lw=1.6, label="road network (car, motorcycle)"),
         Line2D([], [], color=INK2, lw=0.8, label="bus lines")]
    h += [Line2D([], [], color=RAIL_COLORS[m], lw=1.6, label=RAIL_LABELS[m]) for m in RAIL_COLORS]
    ax.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.0), fontsize=6.5)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8)

    rows = [("rail", "rail (bus where no rail stop is within reach)"), ("road", "road vehicles"),
            ("mix", "all modes (ESM2024 mix)")]
    cols = [("ns", "translation (km)\nsouth  \u2190  \u2192  north"),
            ("ew", "translation (km)\nwest  \u2190  \u2192  east"),
            ("rot", "rotation angle (\u00b0)\nclockwise  \u2190  \u2192  counter-cl."),
            ("scale_up", "expansion factor")]
    letters = iter("bcdefghijklm")
    star = dict(marker="*", ms=6, color="#e34948", ls="none", zorder=4)
    for i, (mode, title) in enumerate(rows):
        ymax = rel[mode].max() * 1.08
        for j, (col, xlab) in enumerate(cols):
            a = fig.add_subplot(gs[i, j + 2])
            a.axhline(0, color=INK2, lw=0.6)
            if col in ("ns", "ew"):
                d = rel[rel.card.isin([col, "origin"])].sort_values("signed_km")
                a.plot(d.signed_km, d[mode], color=INK2, lw=0.6, marker="o", ms=2.4, mfc=INK2, mec=INK2)
                a.plot([0], [0], **star)
                a.set_xlim(-2.8, 2.8)
                a.set_xticks([-2, 0, 2])
            elif col == "rot":
                d = rel[(rel.family == "rot") | (rel.index == ORIGINAL)].copy()
                d.loc[ORIGINAL, "x"] = 0.0
                d = d.sort_values("x")
                a.plot(d.x, d[mode], color=INK2, lw=0.6, marker="o", ms=2.4, mfc=INK2, mec=INK2)
                a.plot([0], [0], **star)
                a.set_xticks([-30, 0, 30])
            else:
                for axis, mk in (("ew", "o"), ("ns", "s")):
                    e = pd.concat([rel.loc[[ORIGINAL]].assign(x=1.0), rel[(rel.family == "scale_up") & (rel.axis == axis)]])
                    e = e.sort_values("x")
                    a.plot(e.x, e[mode], color=INK2, lw=0.6, ls="-", marker=mk, ms=2.8,
                           mfc="white" if mk == "s" else INK2, mec=INK2, mew=0.6,
                           label="E\u2013W" if axis == "ew" else "N\u2013S")
                a.plot([1.0], [0], **star)
                a.set_xticks([1.0, 1.1, 1.2])
                a.set_xlim(0.98, 1.215)
            a.set_ylim(-0.5, ymax)
            a.grid(False, axis="x")
            a.grid(axis="y", color=GRID, lw=0.4)
            if j:
                a.tick_params(labelleft=False)
            else:
                a.set_ylabel(r"$\Delta E$ (%)")
                a.set_title(title, loc="left", fontsize=7, color=INK, pad=12)
            if i == 2:
                a.set_xlabel(xlab, fontsize=6)
            else:
                a.tick_params(labelbottom=False)
            a.text(-0.08 if j else -0.34, 1.06, next(letters), transform=a.transAxes, fontweight="bold", fontsize=8)
            if i == 0 and j == 3:
                a.legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.2, handletextpad=0.3, borderaxespad=0.0)
    save(fig, OUT / f"madrid_modes{SUFFIX}")
    alt = rel.drop(ORIGINAL)
    print("modes: min dE " + ", ".join(f"{c} {alt[c].min():.2f}%" for c in ("rail", "road", "mix")))


if __name__ == "__main__":
    cfg = json.loads(CONFIGS["madrid"].read_text())
    ratio_figure(cfg)
    modes_figure(cfg)
