#!/usr/bin/env python3
"""SI Note 8 figure (shrink.pdf) from the v2 step-2 results: W_TOT of Madrid (a) and Santiago (b)
under north-south and east-west contractions (0.80-0.98). Same layout as the submitted figure."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_v2/supp_images/shrink"
CITIES = [("madrid", "Madrid"), ("santiago", "Santiago")]
COLORS = {"ns": "#2ca25f", "ew": "#ff7f0e"}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.8))
    fig.subplots_adjust(hspace=0.42, wspace=0.22, top=0.93)
    for i, (city, name) in enumerate(CITIES):
        d = pd.read_csv(ROOT / f"data/data_processed/{city}/graphs_fine_grid/fine_grid_gravitational_work.csv")
        d = d.drop_duplicates("filename").set_index("filename")["total_energy_on_3d_path_J"]  # W_Tot on 3-D shortest paths (Methods)
        w0 = d["graph_original.pkl"]
        for j, ax_ in enumerate(("ns", "ew")):
            pts = sorted((float(f"{m.group(1)}.{m.group(2)}"), w) for f, w in d.items()
                         if (m := re.fullmatch(rf"graph_scale_{ax_}_(\d)p(\d+)\.pkl", f))
                         and float(f"{m.group(1)}.{m.group(2)}") < 1.0)
            ax = axes[i, j]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=80, alpha=0.8, c=COLORS[ax_])
            ax.scatter([1.0], [w0], s=250, c="red", marker="*", edgecolor="darkred", linewidth=1.5, zorder=5,
                       label="Original")
            ax.set_xlabel(f"{ax_.upper()} Scale Factor", fontsize=20)
            ax.set_ylabel(r"$W_{TOT}$ [J]", fontsize=20)
            ax.tick_params(axis="both", labelsize=18)
            f = ScalarFormatter(useMathText=True)
            f.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(f)
            ax.yaxis.get_offset_text().set_fontsize(16)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=16, frameon=True, loc="upper left")
        y = axes[i, 0].get_position().y1 + 0.03
        fig.text(0.5, y, name, ha="center", fontsize=40, fontweight="bold", color="#0000bb", stretch="condensed")
        fig.text(0.01, y, "ab"[i], fontsize=44, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
