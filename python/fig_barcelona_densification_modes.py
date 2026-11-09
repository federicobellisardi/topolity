#!/usr/bin/env python3
"""SI figure: Barcelona densification by transport mode.

(a-f) Candidate cells on the road network; the 23 cells with comparable horizontal cost (+/-5 % of the
      median, the pool of Fig. 7) are coloured by the energy per new resident relative to their mean, for
      each scenario (all trips by one mode, or the observed EMEF 2025 mix); the five cells of Fig. 7 are
      numbered. Central parameters, access distance 1.5 km.
(g)   Rank of the five Fig. 7 cells among the 23 comparable cells in each scenario.
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcelona_multimodal_robustness as b  # noqa: E402
import multi_city_settlement as mcs  # noqa: E402
from paper_style import GRID, INK, INK2, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_v2/supp_images/transport_modes/barcelona_densification_modes"
TAG = "a1500"
CELL_M = 750.0
# colours of Fig. 7 (main text)
ZONE_COLORS = {"Favorable 1": "#1b9e77", "Favorable 2": "#7570b3", "Favorable 3": "#66a61e",
               "Unfavorable 1": "#d95f02", "Unfavorable 2": "#e7298a"}
SCENARIOS = [("car", "combustion car"), ("car_ev", "electric car"), ("moto", "motorcycle"),
             ("bus", "bus (bus network)"), ("rail", "rail (bus where no rail stop is within reach)"), ("mix", "EMEF 2025 modal mix")]


def energies(df, cfg):
    par, shares = b.central_params(cfg), b.modal_shares(cfg)
    out = {}
    for key, _ in SCENARIOS:
        out[key] = (b.mixed_energy(df, par, shares, TAG) if key == "mix"
                    else b.single_mode_energy(df, par, key, TAG))
    return pd.DataFrame(out, index=df.index)


def road_segments(metric_crs):
    with open(ROOT / "data/data_processed/barcelone/graphs_fine_grid/graph_original.pkl", "rb") as f:
        G = pickle.load(f)
    to_m = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    segs = []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        xy = (np.asarray(geom.coords) if geom is not None else
              np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]))
        segs.append(np.column_stack(to_m.transform(xy[:, 0], xy[:, 1])))
    return segs


def main():
    global OUT
    od = sys.argv[sys.argv.index("--od") + 1] if "--od" in sys.argv else "twitter"
    if od != "twitter":
        b.set_od_source(od)
        OUT = OUT.with_name(f"{OUT.name}_{od}")
    cfg = json.loads(b.CONFIG_FILE.read_text())
    df = pd.read_csv(b.OUTPUT_DIR / "candidate_cells_multimodal.csv", sep=";")
    sel = pd.read_csv(b.OUTPUT_DIR / "selected_cells_car_baseline.csv", sep=";")
    lab = dict(zip(sel.cell_id, sel.zone_label))
    E = energies(df, cfg)
    comp = df["horizontally_comparable"].to_numpy()
    rel = 100 * (E[comp] / E[comp].mean() - 1)
    lim = float(np.ceil(np.abs(rel.to_numpy()).max()))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")
    road = road_segments(mcs.load_city_fua("barcelone").estimate_utm_crs())

    x, y = df.centroid_x.to_numpy(), df.centroid_y.to_numpy()
    pad = 1_500                                  # view: the comparable cells (all five Fig. 7 cells among them)
    cx, cy = x[comp], y[comp]
    xlim, ylim = (cx.min() - pad, cx.max() + pad), (cy.min() - pad, cy.max() + pad)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.55], hspace=0.16, wspace=0.04)
    letters = iter("abcdefg")
    for k, (key, title) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.add_collection(LineCollection(road, colors="#d9d7d0", linewidths=0.25, zorder=1))
        others = [Rectangle((xi - CELL_M / 2, yi - CELL_M / 2), CELL_M, CELL_M) for xi, yi, c in zip(x, y, comp) if not c]
        ax.add_collection(PatchCollection(others, facecolor="#c9c7c0", edgecolor="none", zorder=2))
        cells = [Rectangle((xi - CELL_M / 2, yi - CELL_M / 2), CELL_M, CELL_M) for xi, yi in zip(x[comp], y[comp])]
        pc = PatchCollection(cells, cmap=cmap, norm=norm, edgecolor=INK2, linewidth=0.3, zorder=3)
        pc.set_array(rel[key].to_numpy())
        ax.add_collection(pc)
        for _, r in sel.iterrows():
            i = int(np.flatnonzero(df.cell_id.to_numpy() == r.cell_id)[0])
            n = r.zone_label.split()[0][0] + r.zone_label.split()[1]          # F1..F3, U1, U2
            num = {"F1": "1", "F2": "2", "F3": "3", "U1": "4", "U2": "5"}[n]
            ax.add_patch(Rectangle((x[i] - CELL_M / 2, y[i] - CELL_M / 2), CELL_M, CELL_M, fill=False,
                                   edgecolor=ZONE_COLORS[r.zone_label], linewidth=2.0, zorder=4))
            ax.text(x[i] + CELL_M * 0.65, y[i] + CELL_M * 0.65, num, fontsize=7.5, fontweight="bold",
                    color=ZONE_COLORS[r.zone_label], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        ax.text(0.02, 0.97, next(letters), transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")
        if k == 0:
            ax.plot([xlim[1] - 2_500, xlim[1] - 500], [ylim[0] + 500] * 2, color=INK, lw=1.0)
            ax.text(xlim[1] - 1_500, ylim[0] + 700, "2 km", fontsize=6, ha="center", color=INK)
    cax = fig.add_axes([0.92, 0.42, 0.012, 0.42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("energy per new resident\nrelative to the mean of the\n23 comparable cells (%)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[2, :])
    ranks = E[comp].rank(axis=0).astype(int)
    xs = np.arange(len(SCENARIOS))
    for _, r in sel.sort_values("rank_total").iterrows():
        i = int(np.flatnonzero(df.cell_id.to_numpy() == r.cell_id)[0])
        rk = ranks.loc[i, [k for k, _ in SCENARIOS]].to_numpy()
        ax.plot(xs, rk, color=ZONE_COLORS[r.zone_label], lw=1.2, marker="o", ms=4, label=r.zone_label)
    ax.set_xticks(xs, ["combustion\ncar", "electric\ncar", "motorcycle", "bus", "rail", "EMEF 2025\nmix"], fontsize=6.5)
    ax.set_ylim(comp.sum() + 0.8, 0.2)
    ax.set_yticks([1, 6, 12, 18, 23])
    ax.set_ylabel("rank among the\n23 comparable cells\n(1 = least energy)", fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.text(-0.06, 1.05, next(letters), transform=ax.transAxes, fontweight="bold", fontsize=8)
    save(fig, OUT)
    print("ranks of Fig. 7 cells:\n", ranks.loc[[int(np.flatnonzero(df.cell_id.to_numpy() == c)[0]) for c in sel.cell_id]]
          .set_axis([lab[c] for c in sel.cell_id]).to_string())


if __name__ == "__main__":
    main()
