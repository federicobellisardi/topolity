"""Main-text Fig. 3: concentration of segment-level uphill work.

(a) Lorenz curves of the uphill work per road segment for Barcelona, Rome, Madrid, Boston and Santiago.
(b) Barcelona: mean uphill work of the segments in hexagonal cells, with log colour scale and colour-bar
    ticks at the quantiles (Q0, Q50, Q75, Q100) of the positive segment values; cells whose segments are
    at or above the 70th percentile are redrawn on a coarser grid with dark-red outlines.

Recreated on 2026-11-09 from the exploratory notebooks that produced the submitted figure
(.bak/notebooks/old/GW_distribution.ipynb and segment_work_intensity_map.ipynb, function make_map_hexbin).
Input: data/data_processed/<city>/graphs/arc_work_segments_original.csv (column ``segment_work``), written
by the first-round (v1) pipeline, which interpolated elevations between edge endpoints. The current
pipeline (``--save-segments``) exports only the per-segment vertical gain, not the flow-weighted work, so
a v2 version of this figure needs a new export.

The script prints the Gini index and the share of the work carried by the top 10% of segments, the values
quoted in the Results.

Usage:
    python python/fig3_main.py                      # -> paper_v2/images/fig3_regenerated.{pdf,png}
    python python/fig3_main.py --out paper_v2/images/fig3   # overwrite the paper figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import contextily as ctx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data_processed"
CITIES = {"barcelone": "Barcelona", "rome": "Rome", "madrid": "Madrid", "boston": "Boston",
          "santiago": "Santiago"}
MAP_CITY = "barcelone"
COLORS = ["#08519c", "#3182bd", "#6baed6", "#9e9ac8", "#756bb1", "#54278f", "#bd0026", "#f03b20"]
CMAP = LinearSegmentedColormap.from_list("no_yellow", COLORS, N=256)
MM = 1 / 25.4
CONF_FILE = ROOT / "conf" / "conf.json"   # holds api_keys.carto (not versioned)


def carto_basemap_url(style: str = "light_all") -> str:
    """CARTO Positron tiles (with labels, as in the submitted figure); the key avoids watermarked tiles."""
    url = f"https://basemaps.cartocdn.com/rastertiles/{style}/{{z}}/{{x}}/{{y}}.png"
    try:
        key = json.loads(CONF_FILE.read_text())["api_keys"]["carto"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"[basemap] CARTO API key not found in {CONF_FILE} ({e!r}); tiles will be watermarked.")
        return url
    return f"{url}?key={key}"


def load(city: str) -> pd.DataFrame:
    return pd.read_csv(DATA / city / "graphs" / "arc_work_segments_original.csv")


def lorenz(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve (x: share of segments, y: share of work), segments sorted by increasing work."""
    s = np.sort(w)
    y = np.r_[0.0, np.cumsum(s) / s.sum()]
    x = np.linspace(0.0, 1.0, len(s) + 1)
    # keep the points where the value changes (the curve is linear in between)
    keep = np.r_[True, np.diff(np.r_[s, np.inf]) != 0]
    return x[keep], y[keep]


def gini(w: np.ndarray) -> float:
    x, y = lorenz(w)
    return 1.0 - 2.0 * np.trapezoid(y, x)


def top_share(w: np.ndarray, frac: float = 0.10) -> float:
    s = np.sort(w)[::-1]
    return s[: int(round(frac * len(s)))].sum() / s.sum()


def panel_lorenz(ax, data: dict[str, np.ndarray]) -> None:
    ax.plot([0, 1], [0, 1], ls="--", color="0.4", lw=0.8)
    for city, w in data.items():
        x, y = lorenz(w)
        ax.plot(x, y, lw=1.0, label=CITIES[city])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative share of segments")
    ax.set_ylabel("Cumulative share of total work")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.6, columnspacing=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_map(ax, cax, df: pd.DataFrame, gridsize: int, highlight_pct: float, zoom: int) -> None:
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = t.transform(((df.start_x + df.end_x) / 2).values, ((df.start_y + df.end_y) / 2).values)
    v = df.segment_work.values.astype(float)
    pos = v[v > 0]
    norm = LogNorm(vmin=float(np.percentile(pos, 1)), vmax=float(pos.max()))

    pad = 0.02
    ax.set_xlim(x.min() - pad * np.ptp(x), x.max() + pad * np.ptp(x))
    ax.set_ylim(y.min() - pad * np.ptp(y), y.max() + pad * np.ptp(y))
    ax.set_aspect("equal")
    ctx.add_basemap(ax, crs=3857, source=carto_basemap_url(), zoom=zoom, attribution=False)
    ax.hexbin(x, y, C=v, gridsize=gridsize, cmap=CMAP, norm=norm, reduce_C_function=np.mean,
              alpha=0.7, edgecolors="none", mincnt=1, zorder=2)
    hi = v >= np.percentile(v, highlight_pct)
    ax.hexbin(x[hi], y[hi], C=v[hi], gridsize=int(gridsize * 0.7), cmap=CMAP, norm=norm,
              reduce_C_function=np.mean, alpha=0.9, edgecolors="darkred", linewidths=0.3, mincnt=1, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])

    q = np.percentile(pos, [0, 25, 50, 75, 100])
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_ticks(q)
    cb.set_ticklabels([f"Q{p} ({int(val)})" if p != 25 else "" for p, val in zip((0, 25, 50, 75, 100), q)])
    cb.minorticks_off()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "paper_v2" / "images" / "fig3_regenerated"),
                    help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--gridsize", type=int, default=100, help="hexagons across the map (base layer)")
    ap.add_argument("--highlight-pct", type=float, default=70.0)
    ap.add_argument("--zoom", type=int, default=12, help="basemap tile zoom")
    args = ap.parse_args()

    frames = {c: load(c) for c in CITIES}
    data = {c: f.segment_work.values.astype(float) for c, f in frames.items()}
    for c, w in data.items():
        print(f"{CITIES[c]:10s} segments={len(w):7d}  zero-work={np.mean(w <= 0):.1%}  "
              f"top10%={top_share(w):.1%}  Gini={gini(w):.3f}")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(85 * MM, 112 * MM))
    ax_a = fig.add_axes([0.13, 0.60, 0.84, 0.37])
    ax_b = fig.add_axes([0.02, 0.02, 0.80, 0.48])
    cax = fig.add_axes([0.83, 0.02, 0.03, 0.48])
    panel_lorenz(ax_a, data)
    panel_map(ax_b, cax, frames[MAP_CITY], args.gridsize, args.highlight_pct, args.zoom)
    # align the colour bar with the map after the equal-aspect adjustment
    fig.canvas.draw()
    pb = ax_b.get_position()
    cax.set_position([pb.x1 + 0.01, pb.y0, 0.03, pb.height])
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        p = ax.get_position()
        fig.text(0.0, p.y1 + 0.005, letter, fontsize=10, weight="bold", va="bottom")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    print(f"written {out}.pdf/.png")


if __name__ == "__main__":
    main()
