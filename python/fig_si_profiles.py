#!/usr/bin/env python3
"""SI figure: compact energy profiles of the cities whose observed embedding is the minimum.

Replaces the one-figure-per-city profiles of the SI (Reviewer 2: too much blank space). One row per city,
four fixed columns: translation along N-S, translation along E-W, rotation, rescaling (filled: E-W axis,
open: N-S axis). Each panel shows Delta W = 100 (W_c / W_0 - 1) for the admissible embeddings of that
family; the star is the observed embedding. Families without admissible embeddings are marked "none".
Same data and ensemble as the rest of the SI (3-D energy, no contractions, no translations < 200 m),
read with build_gravitational_supplementary.read_work_df.

Usage:
    python python/fig_si_profiles.py                       # all 'Original' cities, two parts
    python python/fig_si_profiles.py --cities bangkok berlin --per-part 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "supplementary" / "python"))
sys.path.insert(0, str(ROOT / "python"))
import build_gravitational_supplementary as bgs  # noqa: E402
from paper_style import BLUE, GRID, INK2, MUTED, save  # noqa: E402

OUT = ROOT / "paper_v2" / "supp_images" / "figures" / "supporting_profiles" / "profiles_grid"
MAIN_TEXT_CITIES = {"madrid", "santiago"}       # profiles shown in main Fig. 4


def default_cities() -> list[str]:
    """Cities classified 'Original' (observed embedding is the minimum) by the SI builder, minus Fig. 4."""
    summary, _ = bgs.build_summary()
    ok = set(summary.loc[summary["status"] == "OK", "city"]) - MAIN_TEXT_CITIES
    incomplete = sorted(c for c in ok if not bgs.step2_complete(c))
    if incomplete:
        print(f"step 2 incomplete, not shown: {', '.join(incomplete)}")
    return sorted(ok - set(incomplete))


COLUMNS = [("ns", "translation N–S (km)\nsouth ← → north"), ("ew", "translation E–W (km)\nwest ← → east"),
           ("rot", "rotation (°)\nclockwise ← → counter-cl."), ("scale", "rescaling factor\n● E–W   ○ N–S")]
RED = "#d62728"


def panel_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """x and Delta W (%) of the admissible embeddings of every family."""
    w0 = df.loc[df["family"] == "original", "energy_plot"].iloc[0]
    d = df.assign(dW=100 * (df["energy_plot"] / w0 - 1))
    tr = d[d["family"] == "translation"]
    out = {
        "ns": tr[tr["translation_axis"] == "ns"].assign(x=lambda t: t["signed_translation"] / 1000),
        "ew": tr[tr["translation_axis"] == "ew"].assign(x=lambda t: t["signed_translation"] / 1000),
        "rot": d[d["family"] == "rotation"].assign(
            x=lambda t: t["filename"].str.extract(r"rot_([+-]\d+)p(\d+)deg")
            .astype(float).pipe(lambda m: m[0] + np.sign(m[0]) * m[1] / 100)),
    }
    sc = d[d["family"] == "scale"].copy()
    m = sc["filename"].str.extract(r"scale_(ns|ew)_(\d+)p(\d+)")
    sc["x"] = (m[1] + "." + m[2]).astype(float)
    sc["axis"] = m[0]
    out["scale"] = sc
    return out


def draw_part(cities: list[str], out: Path) -> None:
    n = len(cities)
    fig, axes = plt.subplots(n, 4, figsize=(7.0, 0.82 * n + 0.9), sharex="col", squeeze=False,
                             gridspec_kw={"hspace": 0.28, "wspace": 0.28})
    origin_x = {"ns": 0.0, "ew": 0.0, "rot": 0.0, "scale": 1.0}
    for i, city in enumerate(cities):
        data = panel_data(bgs.read_work_df(city))
        for j, (key, label) in enumerate(COLUMNS):
            ax = axes[i, j]
            p = data[key]
            ax.axhline(0, color=INK2, lw=0.5)
            ax.grid(axis="y", color=GRID, lw=0.3)
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            if p.empty:
                ax.text(0.5, 0.5, "none admissible", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=5.5)
                ax.set_yticks([])
                ax.tick_params(axis="x", length=0, labelbottom=i == n - 1)
                for s in ("left", "bottom"):
                    ax.spines[s].set_visible(i == n - 1 and s == "bottom")
            else:
                if key == "scale":
                    for axis, mk in (("ew", "o"), ("ns", "o")):
                        q = p[p["axis"] == axis].sort_values("x")
                        ax.plot(q["x"], q["dW"], mk, ms=2.6, color=INK2, mfc=INK2 if axis == "ew" else "white",
                                mew=0.6)
                else:
                    q = p.sort_values("x")
                    ax.plot(q["x"], q["dW"], "o", ms=2.6, color=BLUE, mew=0)
                lo, hi = min(0.0, p["dW"].min()), max(0.0, p["dW"].max())
                pad = 0.12 * (hi - lo if hi > lo else 1.0)
                ax.set_ylim(lo - pad, hi + pad)
                ax.plot(origin_x[key], 0, "*", ms=5.5, color=RED, mew=0, zorder=5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(bgs.city_display_name(city), fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=14)
            if i == n - 1:
                ax.set_xlabel(label, fontsize=6)
    fig.text(0.015, 0.5, r"$\Delta W_{\mathrm{Tot}}$ (%)", rotation=90, va="center", fontsize=6.5)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.995, bottom=0.9 / (0.82 * n + 0.9))
    save(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cities", nargs="+", default=None, help="default: 'Original' cities of the builder")
    ap.add_argument("--per-part", type=int, default=11)
    args = ap.parse_args()
    bgs.configure_matplotlib()
    plt.rcParams.update({"figure.figsize": (7, 5)})
    cities = sorted(args.cities or default_cities(), key=bgs.city_display_name)
    parts = [cities[k:k + args.per_part] for k in range(0, len(cities), args.per_part)]
    for k, part in enumerate(parts, 1):
        draw_part(part, OUT.with_name(f"{OUT.name}_{k}"))
        print(f"part {k}: {', '.join(bgs.city_display_name(c) for c in part)}")


if __name__ == "__main__":
    main()
