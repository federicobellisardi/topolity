#!/usr/bin/env python3
"""Main-text Fig. 6 from the v2 step-2 results: Bogota, observed network and lowest-energy displaced
configuration on the DEM, and Delta W_TOT along the westward translations.
The Cerros Orientales reserve is the OSM relation 13780896 (Reserva Forestal Bosque Oriental de Bogota,
source IDECA / POT Bogota), cached in data/data_processed/bogota/extra/."""

import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import save  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/data_processed/bogota"
OUT = ROOT / "paper_v2/images/fig6"


def segments(G):
    segs = []
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        segs.append(np.asarray(geom.coords) if geom is not None else
                    np.array([[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]))
    return segs


def reserve():
    osm = json.loads((BASE / "extra/osm_cerros_orientales_13780896.json").read_text())
    lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in osm["elements"][0]["members"] if m["type"] == "way" and m.get("role", "outer") in ("outer", "")]
    return unary_union(list(polygonize(unary_union(lines))))


def main():
    W = pd.read_csv(BASE / "graphs_fine_grid/fine_grid_gravitational_work.csv").drop_duplicates("filename")
    W = W.set_index("filename")["total_energy_on_3d_path_J"]  # W_Tot on 3-D shortest paths (Methods)
    w0 = W["graph_original.pkl"]
    west = {int(m.group(1)): w for f, w in W.items() if (m := re.fullmatch(r"graph_trans_(\d+)m_a\+180\.pkl", f))}
    xs = np.array([0] + sorted(west))
    dw = 100 * (np.array([w0] + [west[k] for k in sorted(west)]) / w0 - 1)
    best_f = W.idxmin()
    m = re.fullmatch(r"graph_trans_(\d+)m_a\+180\.pkl", best_f)
    assert m, f"lowest-energy configuration is not a westward translation: {best_f}"
    best = int(m.group(1))
    print(f"best: {best_f}  dW = {100 * (W[best_f] / w0 - 1):+.2f}%")

    G0 = pickle.load(open(BASE / "graphs_fine_grid/graph_original.pkl", "rb"))
    G1 = pickle.load(open(BASE / f"graphs_fine_grid/{best_f}", "rb"))
    s0, s1 = segments(G0), segments(G1)
    allxy = np.vstack(s0 + s1)
    pad = 0.02
    ext = (allxy[:, 0].min() - pad, allxy[:, 0].max() + pad, allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    with rasterio.open(BASE / "dem/bogota_dem.tif") as src:
        win = rasterio.windows.from_bounds(ext[0], ext[2], ext[1], ext[3], src.transform)
        dem = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(float)
    dem[dem < -30] = np.nan
    res = reserve()

    fig = plt.figure(figsize=(12, 12.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1.1, 1], hspace=0.12, wspace=0.08)
    for j, (segs, col, lab, title, tcol) in enumerate([(s0, "#ff7f0e", "Original network", "Original", "#ff3a1a"),
                                                        (s1, "#1f4fe0", "Displaced network", "Displaced", "#0000cc")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(dem, extent=ext, origin="upper", cmap="gist_earth", aspect="auto")
        for g in getattr(res, "geoms", [res]):
            x, y = g.exterior.xy
            ax.fill(x, y, facecolor="#7fe0a8", edgecolor="#1b7f4a", alpha=0.45, lw=1.0, zorder=2)
        ax.add_collection(LineCollection(segs, colors=col, linewidths=0.5, zorder=3))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=40, color=tcol)
        ax.text(-0.08, 1.02, "ab"[j], transform=ax.transAxes, fontsize=34, fontweight="bold")
        h = [Line2D([], [], color=col, lw=2, label=lab), Patch(facecolor="#7fe0a8", edgecolor="#1b7f4a", label="Cerros Orientales")]
        ax.legend(handles=h, loc="lower right", fontsize=15, framealpha=0.9)
        if j == 1:
            ax.text(0.03, 0.96, f"{best:,} m westward".replace(",", " "), transform=ax.transAxes, fontsize=17,
                    va="top", bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("Elevation (m)", fontsize=17); cb.ax.tick_params(labelsize=14)

    ax = fig.add_subplot(gs[1, :2])
    ax.axhline(0, color="0.5", lw=1, ls=":")
    ax.axvline(0, color="#ff7f0e", lw=2.5, ls="--", label="Empirical position")
    ax.plot(xs, dw, color="#1f4fe0", lw=3, marker="o", ms=11, mfc="white", mec="#1f4fe0", mew=2)
    ax.plot([best], [dw[xs == best][0]], "o", ms=14, color="#1f4fe0", label=f"Minimum at {best} m")
    # label below the minimum, clear of the line (Reviewer 2: the label overlapped the line)
    ax.annotate(f"{dw[xs == best][0]:.2f}%", (best, dw[xs == best][0]), xytext=(0, -16),
                textcoords="offset points", fontsize=20, color="#1f4fe0", ha="center", va="top")
    ax.set_ylim(bottom=dw.min() - 0.22 * (dw.max() - dw.min()))
    ax.set_xticks(xs)
    ax.set_xlabel("Westward displacement (m)", fontsize=22)
    ax.set_ylabel(r"$\Delta W_{\mathrm{Tot}}$ (%)", fontsize=22)
    ax.tick_params(labelsize=17)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=19, frameon=False, loc="center right")
    ax.text(-0.08, 1.0, "c", transform=ax.transAxes, fontsize=34, fontweight="bold")
    save(fig, OUT)
    print("dW% by westward distance:", dict(zip(xs.tolist(), np.round(dw, 2).tolist())))


if __name__ == "__main__":
    main()
