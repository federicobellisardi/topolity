#!/usr/bin/env python3
"""Generate paper figures for all cities."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import sys
from pathlib import Path
import csv as _csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT         = Path("/home/fbellisardi/code/topolity")
DATA_ROOT    = ROOT / "data" / "data_processed"
PIPELINE_DIR = ROOT / "pipeline_production"
PYTHON_DIR   = ROOT / "python"
DPI = 300

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PYTHON_DIR))




def _sfx(polygon_source: str) -> str:
    """Return directory suffix: '' for fua, '_osm' for osm."""
    return "" if polygon_source == "fua" else f"_{polygon_source}"


def _fine_grid_dir(city: str, polygon_source: str) -> Path:
    return DATA_ROOT / city / f"graphs_fine_grid{_sfx(polygon_source)}"


def _graph_pkl(city: str, polygon_source: str) -> Path:
    return DATA_ROOT / city / f"graphs{_sfx(polygon_source)}" / "graph_original.pkl"


def _domain_map_src(city: str, polygon_source: str) -> Path:
    """Path to the domain-map PNG saved by step1."""
    return DATA_ROOT / city / f"{city}_{polygon_source}_map.png"


def _city_out(city: str, output_root: Path) -> Path:
    p = output_root / city
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cities_with_step2(polygon_source: str) -> list[str]:
    return sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir()
        and (_fine_grid_dir(d.name, polygon_source) / "fine_grid_gravitational_work.csv").exists()
    )




def copy_fig4(city: str, out_dir: Path, polygon_source: str) -> None:
    fgdir = _fine_grid_dir(city, polygon_source)
    found = False
    for ext in ("png", "pdf"):
        src = fgdir / f"fine_grid_gravitational_work.{ext}"
        if src.exists():
            shutil.copy2(src, out_dir / f"fig4_{city}.{ext}")
            found = True
    print(f"  saved: fig4_{city}.png" if found else f"  [{city}] fig4 not found")




def make_fig5(city: str, out_dir: Path, polygon_source: str) -> None:
    import pandas as pd
    from generate_combined_work_figures import (
        load_all_results,
        plot_combined_work_comparison,
        plot_boxplot_by_transformation,
    )

    sfx = _sfx(polygon_source)
    source_filter = "fine_grid" if sfx == "" else f"fine_grid{sfx}"

    # load_all_results looks for fine_grid_gravitational_work.csv inside
    # BASE_DIR / city / "graphs_fine_grid" — patch BASE_DIR if needed
    try:
        if sfx:
            # OSM: read CSV directly and normalise to the format expected by plot functions
            fgdir = _fine_grid_dir(city, polygon_source)
            work_csv = fgdir / "fine_grid_gravitational_work.csv"
            if not work_csv.exists():
                print(f"  [{city}] no OSM work CSV, skipping fig5")
                return
            df = pd.read_csv(work_csv)

            # Derive 'variant' from filename — strip "graph_" prefix so that
            # generate_combined_work_figures finds df["variant"] == "original"
            fn_col = next((c for c in ("filename", "variant_type") if c in df.columns), None)
            if fn_col:
                def _clean_variant(s: str) -> str:
                    stem = Path(s).stem if s.endswith(".pkl") else s
                    return stem[len("graph_"):] if stem.startswith("graph_") else stem
                df["variant"] = df[fn_col].apply(_clean_variant)
            else:
                print(f"  [{city}] no filename column in OSM CSV, skipping fig5")
                return

            # 'transformation_type': map "translation_NNN" → "translation", "scale_ew/ns" → "scale"
            def _classify_type(vt: str) -> str:
                v = str(vt).lower()
                if "original" in v: return "original"
                if v.startswith("rot"): return "rotation"
                if v.startswith("trans"): return "translation"
                if v.startswith("scale"): return "scale"
                return "other"
            df["transformation_type"] = df.get("variant_type", df["variant"]).apply(_classify_type)

            # parameter column
            if "parameter" not in df.columns:
                df["parameter"] = None

            # generate_combined_work_figures uses "total_work" column
            if "total_work" not in df.columns:
                for ecol in ("total_energy_3d_J", "total_energy_J", "total_energy_on_3d_path_J"):
                    if ecol in df.columns:
                        df["total_work"] = df[ecol]
                        break
        else:
            df = load_all_results(city, source_filter="fine_grid", exclude_scale=False)
    except Exception as e:
        print(f"  [{city}] could not load results: {e}")
        return

    for col in ("total_energy_3d_J", "total_energy_on_3d_path_J",
                "total_work_3d", "total_energy_J", "total_work"):
        if col in df.columns:
            df["energy_plot"] = df[col]
            break
    if "energy_plot" not in df.columns:
        print(f"  [{city}] no energy column, skipping fig5")
        return

    for fn, label in [(plot_combined_work_comparison, "fig5_bar"),
                      (plot_boxplot_by_transformation, "fig5_box")]:
        out_png = out_dir / f"{label}_{city}.png"
        try:
            fn(df.copy(), out_png)
            print(f"  saved: {label}_{city}.png")
        except Exception as e:
            print(f"  [{city}] {label} failed: {e}")

    del df
    gc.collect()




def copy_domain_map(city: str, out_dir: Path, polygon_source: str) -> None:
    src = _domain_map_src(city, polygon_source)
    if src.exists():
        shutil.copy2(src, out_dir / f"domain_map_{city}.png")
        print(f"  saved: domain_map_{city}.png  ({polygon_source})")
    else:
        print(f"  [{city}] domain map ({polygon_source}) not found — step1 output missing")




def make_fig2_dem(city: str, out_dir: Path, polygon_source: str) -> None:
    """DEM map using the correct (FUA or OSM) graph and domain boundary."""
    import numpy as np
    import geopandas as gpd
    import rasterio
    import rasterio.plot
    from shapely.geometry import LineString, box as sg_box, shape as sg_shape
    from matplotlib.patches import Rectangle

    dem_file = DATA_ROOT / city / "dem" / f"{city}_dem.tif"
    pkl_file = _graph_pkl(city, polygon_source)

    if not dem_file.exists():
        print(f"  [{city}] DEM missing, skipping fig2")
        return
    if not pkl_file.exists():
        # Fallback: try graphs_fine_grid/graph_original.pkl
        pkl_file = DATA_ROOT / city / "graphs_fine_grid" / "graph_original.pkl"
    if not pkl_file.exists():
        print(f"  [{city}] graph pkl missing ({polygon_source}), skipping fig2")
        return

    print(f"  [{city}] building DEM map ({polygon_source})...")

    with open(pkl_file, "rb") as f:
        G = pickle.load(f)

    edges = [
        data.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                             (G.nodes[v]["x"], G.nodes[v]["y"])])
        for u, v, data in G.edges(data=True)
    ]
    edges_gdf = gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326")

    # Load domain boundary for drawing the bounding box
    domain_poly = None
    if polygon_source == "osm":
        osm_boundary = DATA_ROOT / city / "land_osm" / f"{city}_osm_boundary.geojson"
        if osm_boundary.exists():
            with open(osm_boundary) as f:
                fc = json.load(f)
            domain_poly = sg_shape(fc["features"][0]["geometry"])
    else:
        fp_file = DATA_ROOT / city / "land" / f"{city}_poly_bounds.json"
        if fp_file.exists():
            with open(fp_file) as f:
                b = json.load(f).get("bounds")
            if b:
                domain_poly = sg_box(*b)

    node_xs = np.array([d["x"] for _, d in G.nodes(data=True)])
    node_ys = np.array([d["y"] for _, d in G.nodes(data=True)])
    pad_x = (node_xs.max() - node_xs.min()) * 0.08
    pad_y = (node_ys.max() - node_ys.min()) * 0.08

    from rasterio.mask import mask as rio_mask
    with rasterio.open(dem_file) as src:
        try:
            clip_box = sg_box(node_xs.min()-pad_x, node_ys.min()-pad_y,
                              node_xs.max()+pad_x, node_ys.max()+pad_y)
            clipped, tf = rio_mask(src, [clip_box], crop=True)
            elevation = clipped[0].astype(float)
            nodata = src.nodata
            if nodata is not None:
                elevation[elevation == nodata] = float("nan")
        except Exception:
            elevation = src.read(1).astype(float)
            tf = src.transform


    import numpy as np

    elevation = np.ma.masked_invalid(elevation)

    cmap = plt.get_cmap("gist_earth").copy()
    cmap.set_bad("#dbeaf2")   # colore uniforme per mare / NoData
    fig, ax = plt.subplots(figsize=(8, 7))
    # gist_earth: dark-blue (sea-level) → cyan → green → yellow → brown
    # Matches Fig. 2b-d in the paper.
    ax.set_facecolor("#dbeaf2")

    rasterio.plot.show(elevation, transform=tf, ax=ax, cmap=cmap, alpha=0.92, zorder=1)
    if ax.images:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)

        cbar = plt.colorbar(ax.images[0], cax=cax)
        cbar.set_label("Elevation (m)", fontsize=26)
        cbar.ax.tick_params(labelsize=26)



    edges_gdf.plot(ax=ax, color="#FF6E6E", linewidth=0.4, alpha=0.7, zorder=3)

    # Draw domain boundary
    # if domain_poly is not None:
    #     b = domain_poly.bounds  # (minx, miny, maxx, maxy)
    #     ax.add_patch(Rectangle(
    #         (b[0], b[1]), b[2]-b[0], b[3]-b[1],
    #         linewidth=2, edgecolor="black", facecolor="none", zorder=5,
    #         label=f"{polygon_source.upper()} boundary"
    #     ))

    # ax.set_title(f"{city.title()} — DEM & road network ({polygon_source.upper()})",
    #              fontsize=13, pad=6)
    # ax.set_xlabel("Longitude", fontsize=22)
    # ax.set_ylabel("Latitude", fontsize=22)
    ax.tick_params(labelsize=26)
    ax.tick_params(axis='x', labelrotation=45)

    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(
            fontsize=18,
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.95
        )

        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_linewidth(1.5)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_{city}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: fig2_{city}.png")
    del G, edges, edges_gdf, elevation
    gc.collect()




def make_world_map(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import importlib
        wm = importlib.import_module("world_cities_map")
        wm.plot_world_with_cities(show=False, output_path=out_dir / "fig2a_world_map.png")
        print("  saved: fig2a_world_map.png")
    except Exception as e:
        print(f"  [world_map] failed: {e}")




def process_city(city: str, output_root: Path, polygon_source: str,
                 skip_dem: bool) -> None:
    print(f"\n── {city} ({polygon_source}) ──")
    out = _city_out(city, output_root)

    # copy_fig4(city, out, polygon_source)
    # make_fig5(city, out, polygon_source)
    # copy_domain_map(city, out, polygon_source)

    if not skip_dem:
        try:
            make_fig2_dem(city, out, polygon_source)
        except Exception as e:
            print(f"  [{city}] DEM map failed: {e}")

    gc.collect()




def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures per city")
    parser.add_argument("--city", nargs="+", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Process all cities with step-2 output")
    parser.add_argument("--polygon-source", choices=["fua", "osm"], default="fua",
                        help="Which analysis to plot: 'fua' (default) or 'osm'. "
                             "Must match the --polygon-source used in the pipeline.")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "figures",
                        help="Root output directory (default: topolity/figures/)")
    parser.add_argument("--skip-dem", action="store_true",
                        help="Skip the DEM map (faster, no rasterio read)")
    parser.add_argument("--world-map-only", action="store_true",
                        help="Generate only the world map (Fig. 2a)")
    args = parser.parse_args()

    polygon_source = args.polygon_source
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Global figures (not source-specific)
    print("\n── Global figures ──")
    # make_world_map(args.output_root / "global")

    if args.world_map_only:
        return

    cities = args.city or (_cities_with_step2(polygon_source) if args.all else [])
    if not cities:
        print("No cities specified. Use --city or --all.")
        return

    print(f"\nProcessing {len(cities)} cities → {args.output_root}  (source={polygon_source})")
    for city in cities:
        process_city(city, args.output_root, polygon_source, args.skip_dem)

    print(f"\n✓ All figures saved to {args.output_root}")


if __name__ == "__main__":
    main()
