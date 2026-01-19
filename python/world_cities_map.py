#!/usr/bin/env python3
"""
Plot a world map and mark specific cities with points.

Cities plotted:
- Rome
- Madrid
- Paris
- Boston
- Atlanta
- Santiago de Chile
- Barcelona

Usage:
  python world_cities_map.py [--show] [--output PATH]

By default, saves to ../output/world_cities_map.png relative to this file.
"""

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os


def get_default_output_path() -> Path:
    # This file is in topolity/python/, so the repo root is two levels up from here
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "world_cities_map.png"


def cities_geodataframe() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with the target cities in WGS84 (EPSG:4326).

    Geometry coordinates must be (lon, lat).
    """
    cities: List[Tuple[str, float, float]] = [
        ("Rome", 12.4964, 41.9028),
        ("Madrid", -3.7038, 40.4168),
        ("Paris", 2.3522, 48.8566),
        ("Boston", -71.0589, 42.3601),
        ("Atlanta", -84.3880, 33.7490),
        ("Santiago de Chile", -70.6693, -33.4489),
        ("Barcelona", 2.1686, 41.3874),
    ]

    gdf = gpd.GeoDataFrame(
        {
            "name": [c[0] for c in cities],
            "geometry": [Point(c[1], c[2]) for c in cities],
        },
        crs="EPSG:4326",
    )
    return gdf


def _cities_bounds(cities: gpd.GeoDataFrame, pad_lon: float = 10.0, pad_lat: float = 10.0):
    """Compute longitude and latitude bounds from city points with padding.

    Returns (lon_min, lon_max, lat_min, lat_max) clamped to valid ranges.
    """
    lons = cities.geometry.x
    lats = cities.geometry.y
    lon_min = max(-180.0, float(lons.min() - pad_lon))
    lon_max = min(180.0, float(lons.max() + pad_lon))
    lat_min = max(-90.0, float(lats.min() - pad_lat))
    lat_max = min(90.0, float(lats.max() + pad_lat))
    return lon_min, lon_max, lat_min, lat_max


def _find_local_natural_earth_zip() -> Path | None:
    """Try to locate a local Natural Earth shapefile ZIP in this workspace.

    Looks for geo_flow/old/ne_10m_land.zip relative to the workspace root.
    Returns None if not found.
    """
    # This file: topolity/python/world_cities_map.py
    # repo_root -> /home/.../topolity; workspace_root -> parent of repo_root (/home/.../code)
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    candidates = [
        workspace_root / "geo_flow" / "old" / "ne_10m_land.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_world_geodataframe() -> gpd.GeoDataFrame | None:
    """Load a world geometry GeoDataFrame using robust fallbacks.

    Tries, in order:
    1) Local Natural Earth zip under geo_flow/old/ne_10m_land.zip
    2) Remote Natural Earth 110m admin-0 countries (S3) via zip+https
    Returns None if all attempts fail.
    """
    # 1) Local zip
    local_zip = _find_local_natural_earth_zip()
    if local_zip is not None:
        for url in (f"zip://{local_zip}", str(local_zip)):
            try:
                gdf = gpd.read_file(url)
                if gdf.crs is None:
                    gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
                else:
                    gdf = gdf.to_crs("EPSG:4326")
                return gdf
            except Exception:
                pass

    # 2) Remote Natural Earth admin 0 countries (110m)
    ne_urls = [
        # Primary S3 path used by Natural Earth
        "zip+https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
        # Fallback: land polygons
        "zip+https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip",
    ]
    for url in ne_urls:
        try:
            gdf = gpd.read_file(url)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
            else:
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception:
            continue

    return None


def _plot_with_cartopy(cities: gpd.GeoDataFrame):
    """Plot using Cartopy as a fallback if GeoDataFrame world not available.

    Returns (fig, ax) if successful, else (None, None).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        return None, None

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f1e6c8")
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linestyle=":", linewidth=0.5)

    # Plot cities with distinct markers and colors, add legend
    marker_cycle = ["o", "s", "^", "D", "X", "P", "*"]
    color_cycle = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "darkmagenta", "teal"]
    for (name, geom), mk, col in zip(cities[["name", "geometry"]].itertuples(index=False), marker_cycle, color_cycle):
        ax.scatter(geom.x, geom.y, color=col, s=100, marker=mk, label=name, transform=ccrs.PlateCarree(), edgecolors="white", linewidths=0.5)
    ax.legend(loc="lower left", frameon=True)

    # Crop longitudes around cities; keep a generous latitude span
    lon_min, lon_max, lat_min, lat_max = _cities_bounds(cities, pad_lon=10.0, pad_lat=10.0)
    ax.set_extent([lon_min, lon_max, max(lat_min, -60.0), min(lat_max, 70.0)], crs=ccrs.PlateCarree())
    return fig, ax


def plot_world_with_cities(show: bool = False, output_path: Path | None = None) -> Path:
    """Plot the world and the selected cities.

    Args:
        show: If True, display the plot window (if environment supports it).
        output_path: Where to save the PNG. If None, uses default output path.

    Returns:
        The path where the figure was saved.
    """
    cities = cities_geodataframe()

    world = _load_world_geodataframe()
    fig = None
    ax = None

    if world is not None:
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        # Ocean background
        ax.set_facecolor("#e6f2ff")
        # Plot world boundaries
        world.plot(ax=ax, color="#f1e6c8", edgecolor="#888888", linewidth=0.4)
        # Plot cities with distinct markers and colors, add legend
        marker_cycle = ["o", "s", "^", "D", "X", "P", "*"]
        color_cycle = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "darkmagenta", "teal"]
        for (name, geom), mk, col in zip(cities[["name", "geometry"]].itertuples(index=False), marker_cycle, color_cycle):
            ax.scatter(geom.x, geom.y, color=col, s=60, marker=mk, label=name, edgecolors="white", linewidths=0.5)
        ax.legend(loc="lower left", frameon=True)
        ax.set_axis_off()

        # Crop longitude around cities
        lon_min, lon_max, _, _ = _cities_bounds(cities, pad_lon=10.0, pad_lat=10.0)
        ax.set_xlim(lon_min, lon_max)
    else:
        # Try cartopy-based fallback
        fig, ax = _plot_with_cartopy(cities)
        if fig is None:
            # Last resort: blank axes with points (no basemap)
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            ax.set_facecolor("#e6f2ff")
            marker_cycle = ["o", "s", "^", "D", "X", "P", "*"]
            color_cycle = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "darkmagenta", "teal"]
            for (name, geom), mk, col in zip(cities[["name", "geometry"]].itertuples(index=False), marker_cycle, color_cycle):
                ax.scatter(geom.x, geom.y, color=col, s=60, marker=mk, label=name, edgecolors="white", linewidths=0.5)
            ax.legend(loc="lower left", frameon=True)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            # Crop longitude around cities
            lon_min, lon_max, _, _ = _cities_bounds(cities, pad_lon=10.0, pad_lat=10.0)
            ax.set_xlim(lon_min, lon_max)

    out_path = output_path or get_default_output_path()
    fig.savefig(out_path, dpi=200)
    
    # Also save PDF version
    out_path_pdf = out_path.with_suffix('.pdf')
    fig.savefig(out_path_pdf, dpi=200)

    if show:
        try:
            plt.show()
        except Exception:
            # In headless environments, showing may fail; ignore gracefully.
            pass
    else:
        plt.close(fig)

    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot a world map and mark selected cities.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving (may not work in headless environments)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (PNG). Defaults to ../output/world_cities_map.png",
    )
    args = parser.parse_args()

    out_path = plot_world_with_cities(
        show=args.show,
        output_path=Path(args.output).expanduser().resolve() if args.output else None,
    )
    out_path_pdf = out_path.with_suffix('.pdf')
    print(f"Saved map to: {out_path}")
    print(f"Saved map to: {out_path_pdf}")


if __name__ == "__main__":
    main()
