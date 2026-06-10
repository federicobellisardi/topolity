#!/usr/bin/env python3
"""World map of analysed cities for the gravitational-morphology paper."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SELECTED_CITIES = [
    "Barcelona",      # showcase
    "Santiago",       # showcase
    "Madrid",         # showcase
    "Amsterdam",
    "Atlanta",
    "Bandung",
    "Berlin",
    "Bogotá",
    "Brussels",
    "Buenos Aires",
    "Caracas",
    "Chicago",
    "Dallas",
    "Detroit",
    "Guadalajara",
    "Lima",
    "Mexico City",
    "Milan",
    "Moscow",
    "Paris",
    "Beijing",
    "Phoenix",
    "Rome",
    "São Paulo",
    "Toronto",
]

SHOWCASE = {
    "barcelone": ("Barcelona", "X", "#8B2D91", 360),
    "barcelona": ("Barcelona", "X", "#8B2D91", 360),
    "santiago":  ("Santiago",  "D", "#FF7F0E", 360),
    "madrid":    ("Madrid",    "s", "#1F77B4", 360),
}

NAME_MAPPING = {
    "atlanta":          "Atlanta",
    "barcelone":        "Barcelona",
    "barcelona":        "Barcelona",
    "berlin":           "Berlin",
    "bandung":          "Bandung",
    "bogota":           "Bogotá",
    "bruxelles":        "Brussels",
    "buenosaires":      "Buenos Aires",
    "caracas":          "Caracas",
    "chicago":          "Chicago",
    "dallas":           "Dallas",
    "detroitwindsor":   "Detroit",
    "guadalajara":      "Guadalajara",
    "lima":             "Lima",
    "madrid":           "Madrid",
    "mexico":           "Mexico City",
    "milan":            "Milan",
    "moscou":           "Moscow",
    "moscow":           "Moscow",
    "paris":            "Paris",
    "pekin":            "Beijing",
    "beijing":          "Beijing",
    "phoenix":          "Phoenix",
    "rome":             "Rome",
    "santiago":         "Santiago",
    "saopaulo":         "São Paulo",
    "singapour":        "Singapore",
    "toronto":          "Toronto",
}

GEOCODE_HINTS = {
    "Atlanta":      "USA",
    "Barcelona":    "Spain",
    "Bandung":      "Indonesia",
    "Beijing":      "China",
    "Berlin":       "Germany",
    "Bogotá":       "Colombia",
    "Brussels":     "Belgium",
    "Buenos Aires": "Argentina",
    "Caracas":      "Venezuela",
    "Chicago":      "USA",
    "Dallas":       "USA",
    "Detroit":      "USA",
    "Guadalajara":  "Mexico",
    "Lima":         "Peru",
    "Madrid":       "Spain",
    "Mexico City":  "Mexico",
    "Milan":        "Italy",
    "Moscow":       "Russia",
    "Paris":        "France",
    "Phoenix":      "USA",
    "Rome":         "Italy",
    "Santiago":     "Chile",
    "São Paulo":    "Brazil",
    "Toronto":      "Canada",
    "Amsterdam":    "Netherlands",
}

DOT_COLOR = "#2D2D2D"


def get_default_output_path() -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "world_cities_map.png"


def map_name_english(name: str) -> str:
    n = name.strip().lower()
    for k, v in NAME_MAPPING.items():
        if k == n:
            return v
    for k, v in NAME_MAPPING.items():
        if k in n:
            return v
    return name.replace("_", " ").replace("-", " ").title()


def cities_geodataframe() -> gpd.GeoDataFrame:
    import time
    import requests
    from urllib.parse import urlencode

    dp = Path("/home/fbellisardi/code/topolity/data/data_processed")
    if not dp.exists():
        return gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

    selected_set = set(SELECTED_CITIES)
    rows = []

    def geocode(q: str):
        url = "https://nominatim.openstreetmap.org/search?" + urlencode(
            {"q": q, "format": "json", "limit": 1}
        )
        try:
            resp = requests.get(url, headers={"User-Agent": "topolity/1.0"}, timeout=10)
            data = resp.json()
            if data:
                return float(data[0]["lon"]), float(data[0]["lat"])
        except Exception:
            pass
        return None

    seen = set()
    for child in sorted(dp.iterdir()):
        if not child.is_dir():
            continue
        disp = map_name_english(child.name)
        if disp not in selected_set or disp in seen:
            continue
        seen.add(disp)
        hint = GEOCODE_HINTS.get(disp)
        coord = geocode(f"{disp}, {hint}") if hint else None
        if hint:
            time.sleep(1.0)
        if coord is None:
            coord = geocode(disp)
            time.sleep(1.0)
        if coord is None:
            continue
        rows.append({"name": disp, "folder": child.name, "geometry": Point(*coord)})

    if not rows:
        return gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _load_world() -> gpd.GeoDataFrame | None:
    local = Path("/data/workspaces/fbellisardi/land/ne_10m_land.shp")
    if local.exists():
        try:
            return gpd.read_file(local).to_crs("EPSG:4326")
        except Exception:
            pass
    for url in [
        "zip+https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
        "zip+https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip",
    ]:
        try:
            return gpd.read_file(url).to_crs("EPSG:4326")
        except Exception:
            continue
    return None


def _cities_bounds(cities, pad_lon=12.0, pad_lat=10.0):
    lons = cities.geometry.x
    lats = cities.geometry.y
    return (
        max(-180.0, float(lons.min()) - pad_lon),
        min(180.0,  float(lons.max()) + pad_lon),
        max(-90.0,  float(lats.min()) - pad_lat),
        min(90.0,   float(lats.max()) + pad_lat),
    )


def plot_world_with_cities(show: bool = False, output_path: Path | None = None) -> Path:
    cities = cities_geodataframe()
    world = _load_world()

    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#C8DDF0")

    if world is not None:
        world.plot(ax=ax, color="#EDE8D0", edgecolor="#BBBBBB", linewidth=0.35, zorder=1)
    else:
        ax.set_facecolor("#E8ECD0")

    showcase_handles = []

    for _, row in cities.iterrows():
        folder = str(row.get("folder", "")).lower()
        lon, lat = row.geometry.x, row.geometry.y

        if folder in SHOWCASE:
            disp, mk, col, sz = SHOWCASE[folder]
            ax.scatter(lon, lat, s=sz, marker=mk, color=col,
                       edgecolors="white", linewidths=1.2, zorder=5)
            if not any(h.get_label() == disp for h in showcase_handles):
                showcase_handles.append(
                    Line2D([0], [0], marker=mk, color="none",
                           markerfacecolor=col, markeredgecolor="white",
                           markeredgewidth=1.0, markersize=13, label=disp)
                )
        else:
            ax.scatter(lon, lat, s=260, marker="o", color=DOT_COLOR,
                       edgecolors="white", linewidths=0.6, alpha=0.85, zorder=4)

    _order = ["Barcelona", "Santiago", "Madrid"]
    showcase_handles.sort(
        key=lambda h: _order.index(h.get_label()) if h.get_label() in _order else 99
    )
    if showcase_handles:
        ax.legend(
            handles=showcase_handles,
            loc="lower left",
            fontsize=22,
            markerscale=1.2,
            labelspacing=0.9,
            borderpad=0.9,
            handletextpad=0.7,
            framealpha=0.92,
            edgecolor="#CCCCCC",
            fancybox=True,
        )

    if not cities.empty:
        lon_min, lon_max, lat_min, lat_max = _cities_bounds(cities)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    ax.set_axis_off()

    out = output_path or get_default_output_path()
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    out = plot_world_with_cities(
        show=args.show,
        output_path=Path(args.output).expanduser().resolve() if args.output else None,
    )
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
