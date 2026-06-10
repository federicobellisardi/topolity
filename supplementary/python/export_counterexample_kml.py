#!/usr/bin/env python3
"""
Export diagnostic KML files for counterexample cities in the supplementary analysis.

The script focuses on visual debugging:
- original network footprint
- best transformed footprint
- optional worst transformed footprint
- clipped land polygon used by the pipeline
- original / transformed centroids and bounding boxes
- translation candidate markers from fine_grid_stats.csv
- rejected variants (on_land=False)
- optional transformed edge geometries stored in the pickle, to spot stale geometry

The KML is intentionally lightweight enough to open in Google Earth or QGIS.
For large cities, network edges are evenly sampled unless `--max-edges 0` is used.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import unicodedata
import xml.sax.saxutils as xml_utils
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon


ROOT = Path("/home/fbellisardi/code/topolity")
DATA_DIR = ROOT / "data" / "data_processed"
SUPP_DIR = ROOT / "supplementary"
TABLE_DIR = SUPP_DIR / "output" / "tables"
OUTPUT_DIR = SUPP_DIR / "output" / "kml" / "counterexample_diagnostics"

DEFAULT_CITIES = [
    "Amsterdam",
    "Bandung",
    "Bogota",
    "Bruxelles",
    "Chicago",
    "Toronto",
    "Buenos Aires",
]

CITY_ALIASES = {
    "amsterdam": "amsterdam",
    "bandung": "bandung",
    "bogota": "bogota",
    "bogotá": "bogota",
    "bruxelles": "bruxelles",
    "brussels": "bruxelles",
    "chicago": "chicago",
    "toronto": "toronto",
    "buenosaires": "buenosaires",
    "buenosaieres": "buenosaires",
    "buenosares": "buenosaires",
}

KML_STYLE_BLOCK = """
<Style id="land-poly">
  <LineStyle><color>ff2f855a</color><width>1.3</width></LineStyle>
  <PolyStyle><color>332f855a</color></PolyStyle>
</Style>
<Style id="original-line">
  <LineStyle><color>ff1473f9</color><width>1.5</width></LineStyle>
</Style>
<Style id="best-line">
  <LineStyle><color>ffd84e1d</color><width>1.7</width></LineStyle>
</Style>
<Style id="worst-line">
  <LineStyle><color>ff111827</color><width>1.7</width></LineStyle>
</Style>
<Style id="stored-line">
  <LineStyle><color>ff8b5cf6</color><width>1.2</width></LineStyle>
</Style>
<Style id="bbox-original">
  <LineStyle><color>ff1473f9</color><width>1.8</width></LineStyle>
  <PolyStyle><color>001473f9</color></PolyStyle>
</Style>
<Style id="bbox-best">
  <LineStyle><color>ffd84e1d</color><width>1.8</width></LineStyle>
  <PolyStyle><color>00d84e1d</color></PolyStyle>
</Style>
<Style id="bbox-worst">
  <LineStyle><color>ff111827</color><width>1.8</width></LineStyle>
  <PolyStyle><color>00111827</color></PolyStyle>
</Style>
<Style id="bbox-land">
  <LineStyle><color>ff2f855a</color><width>1.3</width></LineStyle>
  <PolyStyle><color>002f855a</color></PolyStyle>
</Style>
<Style id="centroid-original">
  <IconStyle>
    <color>ff1473f9</color>
    <scale>1.0</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon>
  </IconStyle>
</Style>
<Style id="centroid-best">
  <IconStyle>
    <color>ffd84e1d</color>
    <scale>1.0</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon>
  </IconStyle>
</Style>
<Style id="centroid-worst">
  <IconStyle>
    <color>ff111827</color>
    <scale>1.0</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon>
  </IconStyle>
</Style>
<Style id="variant-valid">
  <IconStyle>
    <color>ff0f766e</color>
    <scale>0.8</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
  </IconStyle>
</Style>
<Style id="variant-invalid">
  <IconStyle>
    <color>ff1f1fff</color>
    <scale>0.9</scale>
    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/forbidden.png</href></Icon>
  </IconStyle>
</Style>
<Style id="shift-line">
  <LineStyle><color>ff475569</color><width>1.8</width></LineStyle>
</Style>
"""


@dataclass
class SummaryRow:
    city: str
    best_filename: str
    best_family: str
    best_parameter: float
    delta_pct: float
    n_land_false: int
    frac_land_false: float


@dataclass
class EdgeMismatchSummary:
    n_compared: int
    mean_m: float
    median_m: float
    max_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=DEFAULT_CITIES,
        help="City names or dataset keys. Default: the seven cities discussed in the request.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=25000,
        help="Maximum number of edges exported per network layer. Use 0 for full export.",
    )
    parser.add_argument(
        "--land-simplify",
        type=float,
        default=0.0,
        help="Optional simplification tolerance in degrees for land polygons.",
    )
    parser.add_argument(
        "--include-stored-edge-geometry",
        action="store_true",
        help="Also export the transformed edge geometry as stored in the pickle.",
    )
    parser.add_argument(
        "--focus-best-only",
        action="store_true",
        help="Export only the original and lowest-energy transformed configuration, omitting worst-variant layers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for KML files. Default: {OUTPUT_DIR}",
    )
    return parser.parse_args()


def normalize_city_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-z0-9]+", "", ascii_only.lower())
    return CITY_ALIASES.get(compact, compact)


def city_display_name(city_key: str) -> str:
    raw = {
        "amsterdam": "Amsterdam",
        "bandung": "Bandung",
        "bogota": "Bogota",
        "bruxelles": "Bruxelles",
        "chicago": "Chicago",
        "toronto": "Toronto",
        "buenosaires": "Buenos Aires",
    }
    return raw.get(city_key, city_key.title())


def load_summary_table() -> pd.DataFrame:
    path = TABLE_DIR / "city_hypothesis_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary table: {path}")
    return pd.read_csv(path)


def select_summary_rows(summary_df: pd.DataFrame, cities: list[str]) -> list[SummaryRow]:
    selected: list[SummaryRow] = []
    for value in cities:
        city_key = normalize_city_token(value)
        city_rows = summary_df.loc[summary_df["city"] == city_key]
        if city_rows.empty:
            raise KeyError(f"City not found in summary table: {value!r} -> {city_key!r}")
        row = city_rows.iloc[0]
        selected.append(
            SummaryRow(
                city=city_key,
                best_filename=str(row["best_filename"]),
                best_family=str(row["best_family"]),
                best_parameter=float(row["best_parameter"]),
                delta_pct=float(row["delta_pct"]),
                n_land_false=int(row["n_land_false"]),
                frac_land_false=float(row["frac_land_false"]),
            )
        )
    return selected


def graph_dir(city: str) -> Path:
    return DATA_DIR / city / "graphs_fine_grid"


def graph_path(city: str, filename: str) -> Path:
    return graph_dir(city) / filename


def stats_path(city: str) -> Path:
    return graph_dir(city) / "fine_grid_stats.csv"


def work_path(city: str) -> Path:
    return graph_dir(city) / "fine_grid_gravitational_work.csv"


def land_path(city: str) -> Path:
    return DATA_DIR / city / "land" / f"{city}_clipped_land.shp"


def load_graph(city: str, filename: str) -> nx.MultiDiGraph:
    with graph_path(city, filename).open("rb") as handle:
        return pickle.load(handle)


def graph_centroid(G: nx.MultiDiGraph) -> tuple[float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    return float(xs.mean()), float(ys.mean())


def graph_bbox(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_polygon(bounds: tuple[float, float, float, float]) -> Polygon:
    minx, miny, maxx, maxy = bounds
    return Polygon(
        [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy),
            (minx, miny),
        ]
    )


def evenly_sample_indices(n_items: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=int)
    return np.unique(np.linspace(0, n_items - 1, num=max_items, dtype=int))


def iter_graph_edges(
    G: nx.MultiDiGraph,
    *,
    mode: str,
    max_edges: int,
) -> list[LineString]:
    edge_records = list(G.edges(data=True))
    keep = evenly_sample_indices(len(edge_records), max_edges)
    lines: list[LineString] = []

    for idx in keep:
        u, v, data = edge_records[int(idx)]
        if mode == "stored":
            geom = data.get("geometry")
            if geom is None:
                continue
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(list(geom.geoms))
            continue

        u_data = G.nodes[u]
        v_data = G.nodes[v]
        lines.append(
            LineString(
                [
                    (float(u_data["x"]), float(u_data["y"])),
                    (float(v_data["x"]), float(v_data["y"])),
                ]
            )
        )

    return lines


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def describe_bearing(dx: float, dy: float) -> str:
    components: list[str] = []
    if abs(dy) > 1e-12:
        components.append("north" if dy > 0 else "south")
    if abs(dx) > 1e-12:
        components.append("east" if dx > 0 else "west")
    return "-".join(components) if components else "none"


def classify_stats_family(row: pd.Series) -> str:
    row_type = str(row.get("type", "")).lower()
    variant = str(row.get("variant", "")).lower()
    if row_type == "original" or variant == "original":
        return "original"
    if row_type.startswith("translate") or variant.startswith("trans_"):
        return "translation"
    if row_type.startswith("rotate") or variant.startswith("rot_"):
        return "rotation"
    if row_type.startswith("scale") or variant.startswith("scale_"):
        return "scale"
    return "other"


def load_variant_tables(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = pd.read_csv(stats_path(city)).copy()
    stats_df["variant"] = stats_df["variant"].astype(str)
    stats_df["family"] = stats_df.apply(classify_stats_family, axis=1)
    stats_df["on_land_bool"] = stats_df["on_land"].astype(str).str.lower().eq("true")

    work_df = pd.read_csv(work_path(city)).copy()
    work_df["variant"] = work_df["filename"].astype(str).str.replace(r"^graph_", "", regex=True).str.replace(".pkl", "", regex=False)
    return stats_df, work_df


def select_best_and_worst_transformed(work_df: pd.DataFrame, stats_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    valid_variants = set(
        stats_df.loc[
            (stats_df["variant"].astype(str) != "original") & stats_df["on_land_bool"],
            "variant",
        ].astype(str)
    )
    transformed = work_df.loc[
        (work_df["filename"].astype(str) != "graph_original.pkl")
        & work_df["variant"].astype(str).isin(valid_variants)
    ].copy()
    if transformed.empty:
        raise ValueError("No valid transformed variants from current fine_grid_stats.csv are available in fine_grid_gravitational_work.csv")
    best_row = transformed.loc[transformed["total_energy_3d_J"].idxmin()]
    worst_row = transformed.loc[transformed["total_energy_3d_J"].idxmax()]
    return best_row, worst_row


def build_variant_markers(
    city: str,
    original_centroid: tuple[float, float],
    stats_df: pd.DataFrame,
    work_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    merged = stats_df.merge(
        work_df[["variant", "filename", "total_energy_3d_J", "parameter", "variant_type"]],
        on="variant",
        how="left",
    )
    valid_markers: list[str] = []
    invalid_markers: list[str] = []

    for row in merged.itertuples(index=False):
        family = str(row.family)
        lon, lat = original_centroid
        if family == "translation":
            lon = float(original_centroid[0] + float(row.offset_x))
            lat = float(original_centroid[1] + float(row.offset_y))

        parts = [
            f"city={city_display_name(city)}",
            f"variant={row.variant}",
            f"family={family}",
            f"on_land={bool(row.on_land_bool)}",
        ]
        if not pd.isna(getattr(row, "translation_distance_m", np.nan)):
            parts.append(f"translation_distance_m={float(row.translation_distance_m):.0f}")
        if not pd.isna(getattr(row, "translation_angle", np.nan)):
            parts.append(f"translation_angle={float(row.translation_angle):.0f}")
        if not pd.isna(getattr(row, "angle_deg", np.nan)) and family == "rotation":
            parts.append(f"rotation_deg={float(row.angle_deg):.2f}")
        if not pd.isna(getattr(row, "scale_factor", np.nan)) and family == "scale":
            parts.append(f"scale_factor={float(row.scale_factor):.3f}")
        if isinstance(getattr(row, "filename", None), str):
            parts.append(f"filename={row.filename}")
        if not pd.isna(getattr(row, "total_energy_3d_J", np.nan)):
            parts.append(f"energy_3d_J={float(row.total_energy_3d_J):.3f}")

        placemark = point_placemark(
            name=row.variant,
            lon=lon,
            lat=lat,
            style_url="variant-valid" if bool(row.on_land_bool) else "variant-invalid",
            description="\n".join(parts),
        )
        if bool(row.on_land_bool):
            valid_markers.append(placemark)
        else:
            invalid_markers.append(placemark)

    return valid_markers, invalid_markers


def compare_stored_vs_node_geometry(
    G: nx.MultiDiGraph,
    *,
    sample_size: int = 250,
) -> EdgeMismatchSummary:
    edges = list(G.edges(data=True))
    keep = evenly_sample_indices(len(edges), sample_size)
    distances_m: list[float] = []

    for idx in keep:
        u, v, data = edges[int(idx)]
        geom = data.get("geometry")
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start_geom = coords[0]
        end_geom = coords[-1]
        start_node = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        end_node = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))

        direct = max(
            haversine_m(start_geom[0], start_geom[1], start_node[0], start_node[1]),
            haversine_m(end_geom[0], end_geom[1], end_node[0], end_node[1]),
        )
        swapped = max(
            haversine_m(start_geom[0], start_geom[1], end_node[0], end_node[1]),
            haversine_m(end_geom[0], end_geom[1], start_node[0], start_node[1]),
        )
        distances_m.append(min(direct, swapped))

    if not distances_m:
        return EdgeMismatchSummary(0, float("nan"), float("nan"), float("nan"))

    arr = np.asarray(distances_m, dtype=float)
    return EdgeMismatchSummary(
        n_compared=int(arr.size),
        mean_m=float(arr.mean()),
        median_m=float(np.median(arr)),
        max_m=float(arr.max()),
    )


def escape(value: str) -> str:
    return xml_utils.escape(value, entities={'"': "&quot;"})


def format_coord(lon: float, lat: float, alt: float = 0.0) -> str:
    return f"{lon:.8f},{lat:.8f},{alt:.2f}"


def coords_from_lines(lines: Iterable[LineString]) -> str:
    chunks: list[str] = []
    for line in lines:
        coords = " ".join(format_coord(float(x), float(y)) for x, y in line.coords)
        chunks.append(f"<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>")
    return "".join(chunks)


def polygon_outer_kml(poly: Polygon) -> str:
    exterior = " ".join(format_coord(float(x), float(y)) for x, y in poly.exterior.coords)
    return (
        "<Polygon><tessellate>1</tessellate>"
        "<outerBoundaryIs><LinearRing><coordinates>"
        f"{exterior}"
        "</coordinates></LinearRing></outerBoundaryIs>"
        "</Polygon>"
    )


def polygons_to_kml(geom: Polygon | MultiPolygon) -> str:
    if isinstance(geom, Polygon):
        return polygon_outer_kml(geom)
    if isinstance(geom, MultiPolygon):
        return "".join(polygon_outer_kml(poly) for poly in geom.geoms)
    return ""


def folder(name: str, content: str) -> str:
    return f"<Folder><name>{escape(name)}</name>{content}</Folder>"


def point_placemark(name: str, lon: float, lat: float, style_url: str, description: str = "") -> str:
    desc_block = f"<description><![CDATA[{description}]]></description>" if description else ""
    return (
        "<Placemark>"
        f"<name>{escape(name)}</name>"
        f"{desc_block}"
        f"<styleUrl>#{style_url}</styleUrl>"
        "<Point>"
        f"<coordinates>{format_coord(lon, lat)}</coordinates>"
        "</Point>"
        "</Placemark>"
    )


def line_placemark(name: str, lines: list[LineString], style_url: str, description: str = "") -> str:
    desc_block = f"<description><![CDATA[{description}]]></description>" if description else ""
    return (
        "<Placemark>"
        f"<name>{escape(name)}</name>"
        f"{desc_block}"
        f"<styleUrl>#{style_url}</styleUrl>"
        "<MultiGeometry>"
        f"{coords_from_lines(lines)}"
        "</MultiGeometry>"
        "</Placemark>"
    )


def polygon_placemark(name: str, geom: Polygon | MultiPolygon, style_url: str, description: str = "") -> str:
    desc_block = f"<description><![CDATA[{description}]]></description>" if description else ""
    return (
        "<Placemark>"
        f"<name>{escape(name)}</name>"
        f"{desc_block}"
        f"<styleUrl>#{style_url}</styleUrl>"
        "<MultiGeometry>"
        f"{polygons_to_kml(geom)}"
        "</MultiGeometry>"
        "</Placemark>"
    )


def diagnostic_description(
    summary: SummaryRow,
    original_centroid: tuple[float, float],
    best_centroid: tuple[float, float],
    worst_centroid: tuple[float, float] | None,
    best_row: pd.Series,
    worst_row: pd.Series,
    mismatch: EdgeMismatchSummary,
    original_bounds: tuple[float, float, float, float],
    best_bounds: tuple[float, float, float, float],
    worst_bounds: tuple[float, float, float, float] | None,
    land_bounds: tuple[float, float, float, float],
    stats_df: pd.DataFrame,
) -> str:
    dx = best_centroid[0] - original_centroid[0]
    dy = best_centroid[1] - original_centroid[1]
    shift_km = haversine_m(original_centroid[0], original_centroid[1], best_centroid[0], best_centroid[1]) / 1000.0
    valid_stats = int(stats_df["on_land_bool"].sum())
    invalid_stats = int((~stats_df["on_land_bool"]).sum())
    valid_translations = stats_df.loc[
        (stats_df["family"] == "translation") & stats_df["on_land_bool"],
        ["translation_distance_m", "translation_angle"],
    ].sort_values(["translation_distance_m", "translation_angle"])

    direction_labels = []
    for angle in sorted(valid_translations["translation_angle"].dropna().unique()):
        angle_int = int(round(float(angle))) % 360
        angle_map = {0: "east", 90: "north", 180: "west", 270: "south"}
        direction_labels.append(angle_map.get(angle_int, str(angle_int)))

    lines = [
        f"city={city_display_name(summary.city)}",
        f"best_filename={summary.best_filename}",
        f"best_family={summary.best_family}",
        f"best_parameter={summary.best_parameter:.2f}",
        f"delta_pct={summary.delta_pct:.4f}",
        f"n_land_false={summary.n_land_false}",
        f"frac_land_false={summary.frac_land_false:.4f}",
        f"original_centroid={original_centroid[0]:.6f},{original_centroid[1]:.6f}",
        f"best_centroid={best_centroid[0]:.6f},{best_centroid[1]:.6f}",
        f"centroid_shift_km={shift_km:.3f}",
        f"centroid_shift_direction={describe_bearing(dx, dy)}",
        f"valid_variants={valid_stats}",
        f"invalid_variants={invalid_stats}",
        f"valid_translation_directions={','.join(direction_labels) if direction_labels else 'none'}",
        (
            "stored_vs_node_edge_mismatch_m="
            f"mean:{mismatch.mean_m:.1f},median:{mismatch.median_m:.1f},max:{mismatch.max_m:.1f},n:{mismatch.n_compared}"
            if mismatch.n_compared
            else "stored_vs_node_edge_mismatch_m=not_available"
        ),
        (
            "note: large stored_vs_node mismatch usually means transformed node coordinates were updated "
            "but edge geometries in the pickle still reflect the original embedding."
        ),
        f"original_bounds={','.join(f'{value:.6f}' for value in original_bounds)}",
        f"best_bounds={','.join(f'{value:.6f}' for value in best_bounds)}",
        f"land_bounds={','.join(f'{value:.6f}' for value in land_bounds)}",
    ]
    if worst_centroid is not None and worst_bounds is not None:
        lines[5:5] = [
            f"worst_filename={str(worst_row['filename'])}",
            f"worst_family={str(worst_row['variant_type'])}",
            f"worst_parameter={float(worst_row['parameter']):.2f}",
            f"worst_total_energy_3d_J={float(worst_row['total_energy_3d_J']):.3f}",
        ]
        insert_at = lines.index(f"best_centroid={best_centroid[0]:.6f},{best_centroid[1]:.6f}") + 1
        lines.insert(insert_at, f"worst_centroid={worst_centroid[0]:.6f},{worst_centroid[1]:.6f}")
        lines.insert(lines.index(f"land_bounds={','.join(f'{value:.6f}' for value in land_bounds)}"), f"worst_bounds={','.join(f'{value:.6f}' for value in worst_bounds)}")
    return "\n".join(lines)


def export_city_kml(
    summary: SummaryRow,
    *,
    max_edges: int,
    land_simplify: float,
    include_stored_edge_geometry: bool,
    focus_best_only: bool,
    output_dir: Path,
) -> Path:
    city = summary.city
    original_graph = load_graph(city, "graph_original.pkl")
    stats_df, work_df = load_variant_tables(city)
    best_row, worst_row = select_best_and_worst_transformed(work_df, stats_df)
    best_graph = load_graph(city, str(best_row["filename"]))
    worst_graph = None if focus_best_only else load_graph(city, str(worst_row["filename"]))

    land_gdf = gpd.read_file(land_path(city))
    if land_gdf.crs is None:
        land_gdf = land_gdf.set_crs("EPSG:4326")
    else:
        land_gdf = land_gdf.to_crs("EPSG:4326")

    land_geom = land_gdf.union_all()
    if land_simplify > 0:
        land_geom = land_geom.simplify(land_simplify, preserve_topology=True)

    original_centroid = graph_centroid(original_graph)
    best_centroid = graph_centroid(best_graph)
    worst_centroid = None if worst_graph is None else graph_centroid(worst_graph)
    original_bounds = graph_bbox(original_graph)
    best_bounds = graph_bbox(best_graph)
    worst_bounds = None if worst_graph is None else graph_bbox(worst_graph)
    land_bounds = tuple(float(value) for value in land_gdf.total_bounds)

    original_lines = iter_graph_edges(original_graph, mode="node", max_edges=max_edges)
    best_lines = iter_graph_edges(best_graph, mode="node", max_edges=max_edges)
    worst_lines = [] if worst_graph is None else iter_graph_edges(worst_graph, mode="node", max_edges=max_edges)
    stored_lines_best = iter_graph_edges(best_graph, mode="stored", max_edges=max_edges) if include_stored_edge_geometry else []
    stored_lines_worst = (
        iter_graph_edges(worst_graph, mode="stored", max_edges=max_edges)
        if include_stored_edge_geometry and worst_graph is not None
        else []
    )
    mismatch = compare_stored_vs_node_geometry(best_graph)

    valid_markers, invalid_markers = build_variant_markers(city, original_centroid, stats_df, work_df)

    description = diagnostic_description(
        summary,
        original_centroid,
        best_centroid,
        worst_centroid,
        best_row,
        worst_row,
        mismatch,
        original_bounds,
        best_bounds,
        worst_bounds,
        land_bounds,
        stats_df,
    )

    shift_line = [LineString([original_centroid, best_centroid])]
    bbox_original = bbox_polygon(original_bounds)
    bbox_best = bbox_polygon(best_bounds)
    bbox_worst = None if worst_bounds is None else bbox_polygon(worst_bounds)
    bbox_land = bbox_polygon(land_bounds)

    network_items = [
        line_placemark(
            name=f"{city_display_name(city)} original network (node-reconstructed)",
            lines=original_lines,
            style_url="original-line",
            description=f"sampled_edges={len(original_lines)}",
        ),
        line_placemark(
            name=f"{city_display_name(city)} best transformed network (node-reconstructed)",
            lines=best_lines,
            style_url="best-line",
            description=f"sampled_edges={len(best_lines)}\nfilename={summary.best_filename}",
        ),
    ]
    if worst_graph is not None:
        network_items.append(
            line_placemark(
                name=f"{city_display_name(city)} worst transformed network (node-reconstructed)",
                lines=worst_lines,
                style_url="worst-line",
                description=(
                    f"sampled_edges={len(worst_lines)}\n"
                    f"filename={str(worst_row['filename'])}\n"
                    f"variant_type={str(worst_row['variant_type'])}\n"
                    f"parameter={float(worst_row['parameter']):.2f}\n"
                    f"total_energy_3d_J={float(worst_row['total_energy_3d_J']):.3f}"
                ),
            )
        )
    if stored_lines_best:
        network_items.append(
            line_placemark(
                name=f"{city_display_name(city)} best transformed stored edge geometry",
                lines=stored_lines_best,
                style_url="stored-line",
                description=(
                    f"sampled_edges={len(stored_lines_best)}\n"
                    "This layer uses the geometry stored inside the transformed pickle."
                ),
            )
        )
    if stored_lines_worst:
        network_items.append(
            line_placemark(
                name=f"{city_display_name(city)} worst transformed stored edge geometry",
                lines=stored_lines_worst,
                style_url="stored-line",
                description=(
                    f"sampled_edges={len(stored_lines_worst)}\n"
                    "This layer uses the geometry stored inside the transformed pickle."
                ),
            )
        )

    network_folder = folder(
        "Networks",
        "".join(network_items),
    )

    context_items = [
        polygon_placemark(
            name=f"{city_display_name(city)} clipped land",
            geom=land_geom,
            style_url="land-poly",
        ),
        polygon_placemark(
            name=f"{city_display_name(city)} original network bbox",
            geom=bbox_original,
            style_url="bbox-original",
        ),
        polygon_placemark(
            name=f"{city_display_name(city)} best network bbox",
            geom=bbox_best,
            style_url="bbox-best",
        ),
        polygon_placemark(
            name=f"{city_display_name(city)} land bbox",
            geom=bbox_land,
            style_url="bbox-land",
        ),
    ]
    if bbox_worst is not None:
        context_items.append(
            polygon_placemark(
                name=f"{city_display_name(city)} worst network bbox",
                geom=bbox_worst,
                style_url="bbox-worst",
            )
        )

    context_folder = folder(
        "Context",
        "".join(context_items),
    )

    centroid_items = [
        point_placemark(
            name="original centroid",
            lon=original_centroid[0],
            lat=original_centroid[1],
            style_url="centroid-original",
            description=description,
        ),
        point_placemark(
            name="best centroid",
            lon=best_centroid[0],
            lat=best_centroid[1],
            style_url="centroid-best",
            description=description,
        ),
        line_placemark(
            name="centroid shift",
            lines=shift_line,
            style_url="shift-line",
            description=description,
        ),
    ]
    if worst_centroid is not None:
        centroid_items.append(
            point_placemark(
                name="worst centroid",
                lon=worst_centroid[0],
                lat=worst_centroid[1],
                style_url="centroid-worst",
                description=description,
            )
        )

    centroid_folder = folder(
        "Centroids",
        "".join(centroid_items),
    )

    variant_folder = folder("Valid variants", "".join(valid_markers))
    rejected_folder = folder("Rejected variants", "".join(invalid_markers))

    document_name = f"{city}_counterexample_diagnostic"
    kml_text = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<kml xmlns=\"http://www.opengis.net/kml/2.2\">"
        "<Document>"
        f"<name>{escape(document_name)}</name>"
        f"<description><![CDATA[{description}]]></description>"
        f"{KML_STYLE_BLOCK}"
        f"{network_folder}{context_folder}{centroid_folder}{variant_folder}{rejected_folder}"
        "</Document>"
        "</kml>"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{city}_counterexample_diagnostic.kml"
    out_path.write_text(kml_text, encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    summary_df = load_summary_table()
    selected = select_summary_rows(summary_df, args.cities)

    print(f"Writing KML files to: {args.output_dir}")
    for summary in selected:
        out_path = export_city_kml(
            summary,
            max_edges=args.max_edges,
            land_simplify=args.land_simplify,
            include_stored_edge_geometry=args.include_stored_edge_geometry,
            focus_best_only=args.focus_best_only,
            output_dir=args.output_dir,
        )
        print(f"- {city_display_name(summary.city)} -> {out_path}")


if __name__ == "__main__":
    main()
