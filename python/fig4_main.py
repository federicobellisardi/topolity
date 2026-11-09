#!/usr/bin/env python3
"""Main-text Fig. 4 from the v2 step-2 results: W_TOT of Madrid and Santiago under translations
(north-south and east-west, signed), rotations and north-south / east-west expansions.
Same layout as the submitted figure; contractions are not part of the main ensemble."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import save  # noqa: E402,F401  (rcParams)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_v2/images/fig4"
BLUE, ORANGE, GREEN, STAR = "#1f77b4", "#ff7f0e", "#2ca02c", "red"
CITIES = [("madrid", "Madrid"), ("santiago", "Santiago")]


def load(city):
    d = pd.read_csv(ROOT / f"data/data_processed/{city}/graphs_fine_grid/fine_grid_gravitational_work.csv")
    d = d.drop_duplicates("filename").set_index("filename")["total_energy_on_3d_path_J"]  # W_Tot on 3-D shortest paths (Methods)
    rows = []
    for f, w in d.items():
        n = f.replace("graph_", "").replace(".pkl", "")
        if n == "original":
            rows.append(("orig", 0.0, "", w)); continue
        m = re.fullmatch(r"trans_(\d+)m_a([+-]\d+)", n)
        if m:
            dist, ang = float(m.group(1)), int(m.group(2))
            key = {90: ("ns", 1, "N"), 270: ("ns", -1, "S"), 0: ("ew", 1, "E"), 180: ("ew", -1, "W")}.get(ang)
            if key:
                rows.append((key[0], key[1] * dist, key[2], w))
            continue
        m = re.fullmatch(r"rot_([+-]\d+)p(\d+)deg", n)
        if m:
            rows.append(("rot", float(f"{m.group(1)}.{m.group(2)}"), "", w)); continue
        m = re.fullmatch(r"scale_(ew|ns)_(\d+)p(\d+)", n)
        if m and float(f"{m.group(2)}.{m.group(3)}") > 1.0:
            rows.append((f"sc_{m.group(1)}", float(f"{m.group(2)}.{m.group(3)}"), "", w))
    return pd.DataFrame(rows, columns=["kind", "x", "dir", "W"])


def panel(ax, d, w0, kind, xlab, colors):
    sub = d[d.kind == kind]
    if kind in ("ns", "ew"):
        for lab, col in colors.items():
            s = sub[sub.dir == lab]
            ax.scatter(s.x, s.W, s=150, color=col, edgecolor="k", lw=1.6, label=lab, zorder=3)
    else:
        ax.scatter(sub.x, sub.W, s=150, color=colors, edgecolor="k", lw=1.6, zorder=3)
    x0 = 1.0 if kind.startswith("sc") else 0.0
    ax.scatter([x0], [w0], marker="*", s=750, color=STAR, edgecolor="darkred", lw=1.6, label="O", zorder=4)
    ax.set_xlabel(xlab, fontsize=26)
    ax.set_ylabel(r"$W_{\mathrm{Tot}}$ (J)", fontsize=26)
    ax.tick_params(labelsize=19)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#cccccc")
    ax.yaxis.get_offset_text().set_fontsize(19)
    if kind in ("ns", "ew"):
        ax.legend(fontsize=21, frameon=True, loc="best", markerscale=1.0)


def main():
    fig = plt.figure(figsize=(15.3, 27.6))
    outer = fig.add_gridspec(2, 1, hspace=0.17, top=0.955, bottom=0.03, left=0.09, right=0.98)
    for k, (city, name) in enumerate(CITIES):
        d = load(city)
        w0 = d.loc[d.kind == "orig", "W"].iloc[0]
        g = outer[k].subgridspec(3, 2, hspace=0.5, wspace=0.24)
        ax0 = fig.add_subplot(g[0, 0])
        panel(ax0, d, w0, "ns", "Translation Distance (m)", {"N": BLUE, "S": ORANGE})
        panel(fig.add_subplot(g[0, 1]), d, w0, "ew", "Translation Distance (m)", {"E": BLUE, "W": ORANGE})
        panel(fig.add_subplot(g[1, :]), d, w0, "rot", "Rotation Angle (deg)", BLUE)
        panel(fig.add_subplot(g[2, 0]), d, w0, "sc_ns", "NS Scale Factor", GREEN)
        panel(fig.add_subplot(g[2, 1]), d, w0, "sc_ew", "EW Scale Factor", ORANGE)
        y = ax0.get_position().y1 + 0.018
        fig.text(0.53, y, name, ha="center", fontsize=60, fontweight="bold", color="#0000aa", stretch="condensed")
        fig.text(0.01, y, "ab"[k], fontsize=72, fontweight="bold")
        alt = d[d.kind != "orig"]
        print(f"{name}: observed {w0:.4e} J; best alternative {alt.W.min():.4e} ({100*(alt.W.min()/w0-1):+.2f}%); "
              f"{len(alt)} alternatives shown")
    save(fig, OUT)


if __name__ == "__main__":
    main()
