#!/usr/bin/env python3
"""Distance-invariance audit (Reviewer #1, todo.rev1.md Phase 2).

Question: do the distance terms used in the energy stay constant when the city embedding changes?

From the step-2 sums (graphs_fine_grid/fine_grid_gravitational_work.csv), for every city and
transformation c, relative to the observed embedding 0:
  * dD2_2d : planar length of planar shortest paths (routes independent of terrain). For rigid
             motions this must be invariant: a check of the metric-frame transformations.
  * dD     : 3-D length of the 3-D shortest paths, the distance term of the paper's energy.
  * dH     : uphill gain of the 3-D shortest paths.
  * share_D: share of |dW| due to the distance term at the paper's parameters,
             |lambda dD| / (|lambda dD| + |m g dH|).
Output: reports/reviewer1/tables/distance_invariance_audit.csv (per transformation),
        reports/reviewer1/tables/distance_invariance_summary.csv (per class), and a figure.

Usage: python distance_invariance_audit.py [--exclude city1 city2 ...]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import AQUA, BLUE, GRID, INK2, MUTED, ORANGE, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/data_processed"
OUT = ROOT / "reports/reviewer1"
LAMBDA, MG = 585.0, 1200.0 * 9.81
from study_config import EXCLUDED_CITIES as EXCLUDED_ALWAYS, step2_table, study_cities  # noqa: E402


def klass(f):
    n = f.replace("graph_", "").replace(".pkl", "")
    if n.startswith("trans"):
        return "translation"
    if n.startswith("rot"):
        return "rotation"
    m = re.fullmatch(r"scale_(?:ew|ns)_(\d+)p(\d+)", n)
    if m:
        return "expansion" if float(f"{m.group(1)}.{m.group(2)}") > 1 else "contraction"
    return "original"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    rows = []
    for city in study_cities(exclude=args.exclude):
        csv = step2_table(city)
        if city in EXCLUDED_ALWAYS or city in args.exclude:
            continue
        d = pd.read_csv(csv).drop_duplicates("filename").set_index("filename")
        if "graph_original.pkl" not in d.index:
            continue
        o = d.loc["graph_original.pkl"]
        for f, r in d.drop("graph_original.pkl").iterrows():
            dD = r.path_length_3d_on_3d_path_m - o.path_length_3d_on_3d_path_m
            dH = r.vertical_gain_on_3d_path_m - o.vertical_gain_on_3d_path_m
            rows.append({
                "city": city, "filename": f, "class": klass(f),
                "dD2d_2dpath_pct": 100 * (r.path_length_2d_on_2d_path_m / o.path_length_2d_on_2d_path_m - 1),
                "dD_pct": 100 * dD / o.path_length_3d_on_3d_path_m,
                "dH_pct": 100 * dH / o.vertical_gain_on_3d_path_m,
                "dW_pct": 100 * (LAMBDA * dD + MG * dH) / (LAMBDA * o.path_length_3d_on_3d_path_m + MG * o.vertical_gain_on_3d_path_m),
                "share_D": abs(LAMBDA * dD) / (abs(LAMBDA * dD) + abs(MG * dH)) if (dD or dH) else np.nan,
                "rerouting_3d_vs_2d_length_pct": 100 * (r.path_length_2d_on_3d_path_m / r.path_length_2d_on_2d_path_m - 1),
            })
    A = pd.DataFrame(rows)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "tables/distance_invariance_audit.csv", index=False)
    q = lambda s: f"{s.quantile(0.5):.3g} [{s.quantile(0.05):.3g}, {s.quantile(0.95):.3g}]"  # noqa: E731
    S = A.groupby("class").agg(
        n=("filename", "size"), cities=("city", "nunique"),
        max_abs_dD2d_2dpath_pct=("dD2d_2dpath_pct", lambda s: s.abs().max()),
        dD_pct=("dD_pct", q), dH_pct=("dH_pct", q), dW_pct=("dW_pct", q),
        median_share_D=("share_D", "median"),
        frac_dD_positive=("dD_pct", lambda s: (s > 0).mean()))
    S.to_csv(OUT / "tables/distance_invariance_summary.csv")
    print(S.to_string())

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    colors = {"translation": MUTED, "rotation": ORANGE, "expansion": AQUA, "contraction": BLUE}
    for k, c in colors.items():
        s = A[A["class"] == k]
        ax.scatter(s.dH_pct, s.dD_pct, s=4, color=c, alpha=0.6, lw=0, label=f"{k} ({len(s)})")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlabel(r"change in uphill gain $\Delta H/H_0$ (%)")
    ax.set_ylabel(r"change in route length $\Delta D/D_0$ (%)")
    ax.set_xscale("symlog", linthresh=1)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=6, markerscale=2.5, loc="upper left")
    save(fig, OUT / "figures/distance_invariance")


if __name__ == "__main__":
    main()
