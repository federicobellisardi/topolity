#!/usr/bin/env python3
"""Main Fig. 2: (a) location of the analysed cities, (b-d) DEM and road network of Barcelona, Madrid, Santiago.

Vector PDF kept light enough for Illustrator:
  - world land: Natural Earth 1:10M, simplified to 0.05 deg, drawn as one compound path;
  - city positions: centre of each city's DEM extent (no geocoding), for every study city
    (study_config.study_cities, i.e. without the excluded cities);
  - road networks: each street once (both directions merged), geometry simplified to about 15 m, drawn as one
    compound path per city;
  - DEM: raster image (decimated to at most MAX_DEM_PX per side) inside the vector PDF.

Output: paper_v2/images/fig2.{pdf,png}

Usage: python python/fig2_main.py [--out DIR]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from rasterio.windows import from_bounds
from shapely.geometry import LineString, MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from study_config import study_cities  # noqa: E402

DATA = ROOT / "data" / "data_processed"
LAND = ROOT / "data" / "land" / "ne_10m_land.shp"
LAKES = ROOT / "data" / "land" / "ne_10m_lakes.shp"
OUT_DIR = ROOT / "paper_v2" / "images"

SEA, LAND_C, COAST = "#EEF2F7", "#EDE8D0", "#BDBDBD"
DOT = "#2D2D2D"
ROAD = "#F26B6B"
SHOWCASE = {"barcelone": ("Barcelona", "X", "#8B2D91"), "santiago": ("Santiago", "D", "#FF7F0E"),
            "madrid": ("Madrid", "s", "#1F77B4")}
PANELS = [("barcelone", "Barcelona"), ("madrid", "Madrid"), ("santiago", "Santiago")]
MAX_DEM_PX = 700
SIMPLIFY_DEG = 0.00014          # about 15 m
LAND_SIMPLIFY_DEG = 0.05


def compound_path(lines) -> MPath:
    """One matplotlib Path (MOVETO/LINETO) for many polylines: a single object in the PDF."""
    verts, codes = [], []
    for xy in lines:
        if len(xy) < 2:
            continue
        verts.append(xy)
        c = np.full(len(xy), MPath.LINETO, dtype=np.uint8)
        c[0] = MPath.MOVETO
        codes.append(c)
    return MPath(np.concatenate(verts), np.concatenate(codes))


def polygon_path(geoms) -> MPath:
    """Compound path of polygon rings (filled with the even-odd rule of matplotlib paths)."""
    verts, codes = [], []
    for g in geoms:
        polys = g.geoms if isinstance(g, MultiPolygon) else [g]
        for p in polys:
            if not isinstance(p, Polygon) or p.is_empty:
                continue
            for ring in [p.exterior, *p.interiors]:
                xy = np.asarray(ring.coords)
                if len(xy) < 4:
                    continue
                c = np.full(len(xy), MPath.LINETO, dtype=np.uint8)
                c[0], c[-1] = MPath.MOVETO, MPath.CLOSEPOLY
                verts.append(xy)
                codes.append(c)
    return MPath(np.concatenate(verts), np.concatenate(codes))


def city_centres(cities):
    out = {}
    for c in cities:
        f = DATA / c / "dem" / f"{c}_dem.tif"
        if not f.exists():
            print(f"no DEM for {c}, not on the map")
            continue
        with rasterio.open(f) as src:
            b = src.bounds
        out[c] = ((b.left + b.right) / 2, (b.bottom + b.top) / 2)
    return out


def road_lines(city):
    with open(DATA / city / "graphs_fine_grid" / "graph_original.pkl", "rb") as f:
        G = pickle.load(f)
    seen, lines = set(), []
    for u, v, d in G.edges(data=True):
        key = (min(u, v), max(u, v), d.get("osmid") if not isinstance(d.get("osmid"), list) else tuple(d["osmid"]))
        if key in seen:
            continue
        seen.add(key)
        g = d.get("geometry")
        if g is None:
            g = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        lines.append(np.asarray(g.simplify(SIMPLIFY_DEG, preserve_topology=False).coords))
    xs = np.array([G.nodes[n]["x"] for n in G.nodes])
    ys = np.array([G.nodes[n]["y"] for n in G.nodes])
    return lines, (xs.min(), xs.max(), ys.min(), ys.max())


def dem_window(city, bounds, margin=0.04):
    x0, x1, y0, y1 = bounds
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy
    with rasterio.open(DATA / city / "dem" / f"{city}_dem.tif") as src:
        w = from_bounds(x0, y0, x1, y1, src.transform)
        step = max(1, int(np.ceil(max(w.width, w.height) / MAX_DEM_PX)))
        out_shape = (int(round(w.height / step)), int(round(w.width / step)))
        z = src.read(1, window=w, out_shape=out_shape, resampling=rasterio.enums.Resampling.average,
                     boundless=True, fill_value=src.nodata if src.nodata is not None else -32768).astype(float)
        nod = src.nodata
    if nod is not None:
        z[z == nod] = np.nan
    z[z <= 0] = np.nan                          # sea (Barcelona); no inland city reaches 0 m
    return z, (x0, x1, y0, y1)


def draw_world(ax, centres):
    land = gpd.read_file(LAND).geometry.simplify(LAND_SIMPLIFY_DEG)
    ax.set_facecolor(SEA)
    ax.add_patch(PathPatch(polygon_path(land), facecolor=LAND_C, edgecolor=COAST, lw=0.1))
    if LAKES.exists():
        lk = gpd.read_file(LAKES)
        lk = lk[lk.to_crs("EPSG:6933").area > 2e10].geometry.simplify(LAND_SIMPLIFY_DEG)   # large lakes only
        ax.add_patch(PathPatch(polygon_path(lk), facecolor=SEA, edgecolor=COAST, lw=0.1))
    others = np.array([xy for c, xy in centres.items() if c not in SHOWCASE])
    ax.scatter(others[:, 0], others[:, 1], s=5, color=DOT, edgecolors="white", linewidths=0.15, zorder=4)
    handles = []
    for c, (lab, mk, col) in SHOWCASE.items():
        if c in centres:
            ax.scatter(*centres[c], s=13, marker=mk, color=col, edgecolors="white", linewidths=0.25, zorder=5)
            handles.append(Line2D([0], [0], marker=mk, color="none", markerfacecolor=col,
                                  markeredgecolor="white", markersize=3.5, label=lab))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=DOT, markeredgecolor="white",
                          markersize=3, label="other cities"))
    leg = ax.legend(handles=handles, loc="lower left", fontsize=4.5, framealpha=0.92, edgecolor="#CCCCCC",
                    borderpad=0.4, labelspacing=0.35, handletextpad=0.2, borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.3)
    ax.set_xlim(-130, 160)
    ax.set_ylim(-48, 68)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def city_layer(city):
    lines, bounds = road_lines(city)
    z, ext = dem_window(city, bounds)
    x0, x1, y0, y1 = ext
    aspect = (x1 - x0) * np.cos(np.radians((y0 + y1) / 2)) / (y1 - y0)      # width / height on the ground
    return lines, z, ext, aspect


def draw_city(fig, ax, cax, layer, title, cmap):
    lines, z, (x0, x1, y0, y1), _ = layer
    lat0 = (y0 + y1) / 2
    ax.set_facecolor("#DCE6F0")
    im = ax.imshow(z, extent=(x0, x1, y0, y1), origin="upper", cmap=cmap, interpolation="nearest",
                   vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.add_patch(PathPatch(compound_path(lines), facecolor="none", edgecolor=ROAD, lw=0.06, alpha=0.95))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.3)
    ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top", color="white", fontsize=6,
            fontweight="bold")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Elevation (m)", fontsize=4.5, labelpad=1)
    cb.ax.tick_params(labelsize=4.2, length=1.2, width=0.3, pad=0.8)
    cb.outline.set_linewidth(0.3)
    return len(lines), sum(len(x) for x in lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    cities = study_cities()
    centres = city_centres(cities)
    print(f"{len(centres)} cities on the map")
    cmap = LinearSegmentedColormap.from_list("dem", plt.get_cmap("gist_earth")(np.linspace(0.08, 0.98, 256)))
    cmap.set_bad("#DCE6F0")

    layers = [city_layer(c) for c, _ in PANELS]
    W = 8.5 / 2.54                            # figure width (in): the column width of the paper
    lm, gap, cbw, cbspace = 0.10, 0.07, 0.045, 0.36   # left margin, gap, colorbar width, colorbar + labels
    h = (W - lm - 3 * (cbspace + gap)) / sum(l[3] for l in layers)    # common height of the three maps
    world_h = W * (68 + 48) / 290.0
    top_pad, mid, bot = 0.02, 0.12, 0.02
    H = top_pad + world_h + mid + h + bot
    fig = plt.figure(figsize=(W, H))
    axa = fig.add_axes([0, (bot + h + mid) / H, 1, world_h / H])
    draw_world(axa, centres)
    axa.text(0.006, 0.985, "a", transform=axa.transAxes, fontsize=8, fontweight="bold", va="top")
    x = lm
    for k, ((city, title), layer) in enumerate(zip(PANELS, layers)):
        w = layer[3] * h
        ax = fig.add_axes([x / W, bot / H, w / W, h / H])
        cax = fig.add_axes([(x + w + 0.025) / W, bot / H, cbw / W, h / H])
        n, p = draw_city(fig, ax, cax, layer, title, cmap)
        print(f"{title}: {n} streets, {p} points")
        fig.text((x - 0.015) / W, (bot + h + 0.01) / H, "bcd"[k], fontsize=8, fontweight="bold", ha="right",
                 va="bottom")
        x += w + cbspace + gap
    args.out.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out / "fig2.pdf", args.out / "fig2.png"
    fig.savefig(pdf, dpi=300, facecolor="white")
    fig.savefig(png, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"saved {pdf} ({pdf.stat().st_size / 1e6:.1f} MB), {png}")


if __name__ == "__main__":
    main()
