#!/usr/bin/env python3
"""Audit the fine-grid two-step pipeline for a subset of cities."""


from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon


ROOT = Path("/home/fbellisardi/code/topolity")
DATA_DIR = ROOT / "data" / "data_processed"
PIPELINE_DIR = ROOT / "pipeline_production"
OUTPUT_DIR = ROOT / "supplementary" / "output" / "pipeline_audit"

STEP1_LAUNCHER = PIPELINE_DIR / "runlog_step1_all_cities.sh"
STEP2_LAUNCHER = PIPELINE_DIR / "runlog_step2_all_cities.sh"
STEP1_SCRIPT = PIPELINE_DIR / "dem_extractor_fine_grid.py"
STEP2_SCRIPT = PIPELINE_DIR / "fine_grid_gravitational_work.py"

DEFAULT_CITIES = ["Buenos Aires", "Toronto"]
CITY_ALIASES = {
    "buenosaires": "buenosaires",
    "buenosaieres": "buenosaires",
    "toronto": "toronto",
    "amsterdam": "amsterdam",
    "bandung": "bandung",
    "bogota": "bogota",
    "bogotá": "bogota",
    "bruxelles": "bruxelles",
    "brussels": "bruxelles",
    "chicago": "chicago",
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES, help="Cities to audit.")
    parser.add_argument("--sample-nodes", type=int, default=1000, help="Sample size for node-level checks.")
    parser.add_argument("--sample-edges", type=int, default=250, help="Sample size for edge-geometry checks.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_DIR / "fine_grid_pipeline_audit.json",
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def normalize_city_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-z0-9]+", "", ascii_only.lower())
    return CITY_ALIASES.get(compact, compact)


def city_label(city: str) -> str:
    labels = {
        "buenosaires": "Buenos Aires",
        "toronto": "Toronto",
        "amsterdam": "Amsterdam",
        "bandung": "Bandung",
        "bogota": "Bogota",
        "bruxelles": "Bruxelles",
        "chicago": "Chicago",
    }
    return labels.get(city, city.title())


def evenly_sample_indices(n_items: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=int)
    return np.unique(np.linspace(0, n_items - 1, num=max_items, dtype=int))


def graph_dir(city: str) -> Path:
    return DATA_DIR / city / "graphs_fine_grid"


def stats_path(city: str) -> Path:
    return graph_dir(city) / "fine_grid_stats.csv"


def work_path(city: str) -> Path:
    return graph_dir(city) / "fine_grid_gravitational_work.csv"


def land_path(city: str) -> Path:
    return DATA_DIR / city / "land" / f"{city}_clipped_land.shp"


def load_graph(path: Path) -> nx.MultiDiGraph:
    with path.open("rb") as handle:
        return pickle.load(handle)


def graph_bounds(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def graph_centroid(G: nx.MultiDiGraph) -> tuple[float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((float(data["y"]) for _, data in G.nodes(data=True)), dtype=float)
    return float(xs.mean()), float(ys.mean())


def metric_transformer_for_graph(G: nx.MultiDiGraph) -> Transformer:
    lon0, lat0 = graph_centroid(G)
    proj_str = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)


def project_point(transformer: Transformer, lon: float, lat: float) -> tuple[float, float]:
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def point_distance_m(transformer: Transformer, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = project_point(transformer, a[0], a[1])
    bx, by = project_point(transformer, b[0], b[1])
    return math.hypot(ax - bx, ay - by)


def rotate_xy(x: float, y: float, origin_x: float, origin_y: float, angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    dx = x - origin_x
    dy = y - origin_y
    return c * dx - s * dy + origin_x, s * dx + c * dy + origin_y


def scale_xy(x: float, y: float, origin_x: float, origin_y: float, scale_factor: float, axis: str) -> tuple[float, float]:
    out_x = x
    out_y = y
    if axis in {"x", "both"}:
        out_x = origin_x + scale_factor * (x - origin_x)
    if axis in {"y", "both"}:
        out_y = origin_y + scale_factor * (y - origin_y)
    return out_x, out_y


def normalize_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    df = stats_df.copy()
    df["variant"] = df["variant"].astype(str)
    df["on_land_bool"] = df["on_land"].astype(str).str.lower().eq("true")
    return df


def normalize_work(work_df: pd.DataFrame) -> pd.DataFrame:
    df = work_df.copy()
    df["filename"] = df["filename"].astype(str)
    df["variant"] = df["filename"].str.replace(r"^graph_", "", regex=True).str.replace(".pkl", "", regex=False)
    return df


def launcher_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    results.append(
        CheckResult(
            name="step1_launcher_exists",
            status="PASS" if STEP1_LAUNCHER.exists() else "FAIL",
            detail=str(STEP1_LAUNCHER),
        )
    )
    results.append(
        CheckResult(
            name="step2_launcher_exists",
            status="PASS" if STEP2_LAUNCHER.exists() else "FAIL",
            detail=str(STEP2_LAUNCHER),
        )
    )

    step1_text = STEP1_LAUNCHER.read_text(encoding="utf-8") if STEP1_LAUNCHER.exists() else ""
    step2_text = STEP2_LAUNCHER.read_text(encoding="utf-8") if STEP2_LAUNCHER.exists() else ""

    results.append(
        CheckResult(
            name="step1_script_reference",
            status="PASS" if STEP1_SCRIPT.name in step1_text and STEP1_SCRIPT.exists() else "FAIL",
            detail=str(STEP1_SCRIPT),
        )
    )
    results.append(
        CheckResult(
            name="step2_script_reference",
            status="PASS" if STEP2_SCRIPT.name in step2_text and STEP2_SCRIPT.exists() else "FAIL",
            detail=str(STEP2_SCRIPT),
        )
    )

    uses_resume_step1 = "--resume" in step1_text
    uses_resume_step2 = "--resume" in step2_text
    results.append(
        CheckResult(
            name="launchers_use_resume",
            status="PASS" if uses_resume_step1 and uses_resume_step2 else "WARN",
            detail=f"step1_resume={uses_resume_step1}, step2_resume={uses_resume_step2}",
        )
    )

    uses_node_elevation = "--elevation-source" not in step2_text
    results.append(
        CheckResult(
            name="step2_elevation_source_default_node",
            status="PASS" if uses_node_elevation else "WARN",
            detail="runlog_step2 does not override fine_grid_gravitational_work.py default elevation_source=node",
        )
    )
    return results


def work_best_worst(work_df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    transformed = work_df.loc[work_df["filename"] != "graph_original.pkl"].copy()
    if transformed.empty:
        raise ValueError("No transformed variants found in work CSV.")
    best = transformed.loc[transformed["total_energy_3d_J"].idxmin()]
    worst = transformed.loc[transformed["total_energy_3d_J"].idxmax()]
    return best.to_dict(), worst.to_dict()


def translation_directions(stats_df: pd.DataFrame) -> dict[str, list[str]]:
    mapping = {0: "east", 90: "north", 180: "west", 270: "south"}
    valid_angles = sorted(stats_df.loc[(stats_df["type"] == "translate") & stats_df["on_land_bool"], "translation_angle"].dropna().unique())
    invalid_angles = sorted(stats_df.loc[(stats_df["type"] == "translate") & (~stats_df["on_land_bool"]), "translation_angle"].dropna().unique())
    return {
        "valid": [mapping.get(int(round(float(angle))) % 360, str(angle)) for angle in valid_angles],
        "invalid": [mapping.get(int(round(float(angle))) % 360, str(angle)) for angle in invalid_angles],
    }


def compare_transform_to_expected(
    original_graph: nx.MultiDiGraph,
    transformed_graph: nx.MultiDiGraph,
    stats_row: pd.Series,
    sample_nodes: int,
) -> dict[str, Any]:
    original_nodes = list(original_graph.nodes(data=True))
    keep = evenly_sample_indices(len(original_nodes), sample_nodes)
    bounds = graph_bounds(original_graph)
    origin_x, origin_y = bbox_center(bounds)
    transformer = metric_transformer_for_graph(original_graph)

    errors_m: list[float] = []
    family = str(stats_row["type"])

    for idx in keep:
        node_id, data0 = original_nodes[int(idx)]
        data1 = transformed_graph.nodes[node_id]
        x0 = float(data0["x"])
        y0 = float(data0["y"])

        if family == "translate":
            exp_x = x0 + float(stats_row["offset_x"])
            exp_y = y0 + float(stats_row["offset_y"])
        elif family == "rotate":
            exp_x, exp_y = rotate_xy(x0, y0, origin_x, origin_y, float(stats_row["angle_deg"]))
        elif family == "scale":
            exp_x, exp_y = scale_xy(
                x0,
                y0,
                origin_x,
                origin_y,
                float(stats_row["scale_factor"]),
                "x" if str(stats_row.get("scale_axis", "")).lower() == "x" else "y",
            )
        else:
            exp_x, exp_y = x0, y0

        got_x = float(data1["x"])
        got_y = float(data1["y"])
        errors_m.append(point_distance_m(transformer, (exp_x, exp_y), (got_x, got_y)))

    arr = np.asarray(errors_m, dtype=float)
    return {
        "checked_nodes": int(arr.size),
        "mean_error_m": float(arr.mean()) if arr.size else float("nan"),
        "median_error_m": float(np.median(arr)) if arr.size else float("nan"),
        "max_error_m": float(arr.max()) if arr.size else float("nan"),
    }


def compare_stored_geometry_to_nodes(G: nx.MultiDiGraph, sample_edges: int) -> dict[str, Any]:
    edges = list(G.edges(data=True))
    keep = evenly_sample_indices(len(edges), sample_edges)
    transformer = metric_transformer_for_graph(G)
    errors_m: list[float] = []

    for idx in keep:
        u, v, data = edges[int(idx)]
        geom = data.get("geometry")
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start_geom = (float(coords[0][0]), float(coords[0][1]))
        end_geom = (float(coords[-1][0]), float(coords[-1][1]))
        start_node = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        end_node = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))

        direct = max(
            point_distance_m(transformer, start_geom, start_node),
            point_distance_m(transformer, end_geom, end_node),
        )
        swapped = max(
            point_distance_m(transformer, start_geom, end_node),
            point_distance_m(transformer, end_geom, start_node),
        )
        errors_m.append(min(direct, swapped))

    if not errors_m:
        return {"checked_edges": 0, "mean_error_m": float("nan"), "median_error_m": float("nan"), "max_error_m": float("nan")}

    arr = np.asarray(errors_m, dtype=float)
    return {
        "checked_edges": int(arr.size),
        "mean_error_m": float(arr.mean()),
        "median_error_m": float(np.median(arr)),
        "max_error_m": float(arr.max()),
    }


def centroid_rotation_diagnostic(original_graph: nx.MultiDiGraph, transformed_graph: nx.MultiDiGraph, angle_deg: float) -> dict[str, Any]:
    original_cent = graph_centroid(original_graph)
    transformed_cent = graph_centroid(transformed_graph)
    origin = bbox_center(graph_bounds(original_graph))
    expected_cent = rotate_xy(original_cent[0], original_cent[1], origin[0], origin[1], angle_deg)
    transformer = metric_transformer_for_graph(original_graph)
    return {
        "rotation_origin_bbox_center": {"x": origin[0], "y": origin[1]},
        "original_centroid": {"x": original_cent[0], "y": original_cent[1]},
        "expected_rotated_centroid": {"x": expected_cent[0], "y": expected_cent[1]},
        "actual_centroid": {"x": transformed_cent[0], "y": transformed_cent[1]},
        "expected_vs_actual_error_m": point_distance_m(transformer, expected_cent, transformed_cent),
        "note": "A large centroid shift is not a bug by itself if the graph is rotated around the bbox center rather than around its centroid.",
    }


def water_boundary_proximity(city: str, original_graph: nx.MultiDiGraph) -> dict[str, Any]:
    land_gdf = gpd.read_file(land_path(city))
    land_gdf = land_gdf.to_crs("EPSG:4326") if land_gdf.crs is not None else land_gdf.set_crs("EPSG:4326")
    land_geom = land_gdf.union_all()

    bounds = graph_bounds(original_graph)
    bbox_poly = Polygon(
        [
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
            (bounds[0], bounds[1]),
        ]
    )
    centroid = Point(*graph_centroid(original_graph))

    transformer = metric_transformer_for_graph(original_graph)
    project = lambda geom: gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(
        f"+proj=aeqd +lat_0={graph_centroid(original_graph)[1]} +lon_0={graph_centroid(original_graph)[0]} +datum=WGS84 +units=m +no_defs"
    ).iloc[0]

    land_boundary_m = float(project(land_geom.boundary).distance(project(bbox_poly)))
    centroid_to_boundary_m = float(project(land_geom.boundary).distance(project(centroid)))

    return {
        "graph_bbox_to_land_boundary_m": land_boundary_m,
        "graph_centroid_to_land_boundary_m": centroid_to_boundary_m,
        "note": (
            "This is a coarse indicator based on the clipped land polygon. "
            "It can reflect coastline, estuary, lake edge, or the clipping boundary itself; "
            "it does not reliably capture narrow rivers or canals."
        ),
        "possible_near_water": bool(land_boundary_m < 5000.0 or centroid_to_boundary_m < 10000.0),
    }


def city_audit(city: str, sample_nodes: int, sample_edges: int) -> dict[str, Any]:
    city_dir = DATA_DIR / city
    gdir = graph_dir(city)
    original_pkl = gdir / "graph_original.pkl"
    stats_csv = stats_path(city)
    work_csv = work_path(city)

    checks: list[CheckResult] = []
    for name, path in [
        ("city_dir", city_dir),
        ("graphs_fine_grid_dir", gdir),
        ("graph_original", original_pkl),
        ("fine_grid_stats_csv", stats_csv),
        ("fine_grid_gravitational_work_csv", work_csv),
        ("clipped_land_shp", land_path(city)),
    ]:
        checks.append(CheckResult(name=name, status="PASS" if path.exists() else "FAIL", detail=str(path)))

    if any(result.status == "FAIL" for result in checks):
        return {
            "city": city,
            "city_label": city_label(city),
            "checks": [asdict(item) for item in checks],
            "status": "FAIL",
            "summary": "Missing required inputs.",
        }

    stats_df = normalize_stats(pd.read_csv(stats_csv))
    work_df = normalize_work(pd.read_csv(work_csv))
    original_graph = load_graph(original_pkl)

    checks.append(
        CheckResult(
            name="stats_rows_nonzero",
            status="PASS" if len(stats_df) > 0 else "FAIL",
            detail=f"rows={len(stats_df)}",
        )
    )
    checks.append(
        CheckResult(
            name="work_rows_nonzero",
            status="PASS" if len(work_df) > 0 else "FAIL",
            detail=f"rows={len(work_df)}",
        )
    )

    valid_stats = stats_df.loc[stats_df["on_land_bool"]].copy()
    invalid_stats = stats_df.loc[~stats_df["on_land_bool"]].copy()
    graph_pickles = sorted(gdir.glob("graph_*.pkl"))
    graph_filenames = {path.name for path in graph_pickles}

    expected_valid_pickles = {f"graph_{variant}.pkl" for variant in valid_stats["variant"]}
    missing_valid_pickles = sorted(expected_valid_pickles - graph_filenames)

    checks.append(
        CheckResult(
            name="valid_stats_have_pickles",
            status="PASS" if not missing_valid_pickles else "FAIL",
            detail=f"missing={missing_valid_pickles[:10]}",
        )
    )

    work_filenames = set(work_df["filename"].astype(str))
    missing_work_for_pickles = sorted((graph_filenames - {"graph_original.pkl"}) - work_filenames)
    extra_work_without_pickle = sorted(work_filenames - graph_filenames)
    checks.append(
        CheckResult(
            name="work_matches_pickles",
            status="PASS" if not missing_work_for_pickles and not extra_work_without_pickle else "FAIL",
            detail=f"missing_work={missing_work_for_pickles[:10]}, extra_work={extra_work_without_pickle[:10]}",
        )
    )

    stats_variants = set(stats_df["variant"].astype(str))
    work_variants = set(work_df["variant"].astype(str))
    extra_work_variants = sorted(work_variants - stats_variants)
    checks.append(
        CheckResult(
            name="work_variants_belong_to_current_stats_grid",
            status="PASS" if not extra_work_variants else "FAIL",
            detail=f"extra_variants={extra_work_variants[:20]}",
        )
    )

    best_variant, worst_variant = work_best_worst(work_df)

    best_variant_name = str(best_variant["variant"])
    best_graph = load_graph(gdir / str(best_variant["filename"]))
    best_stats_row = valid_stats.loc[valid_stats["variant"] == best_variant_name]
    if best_stats_row.empty:
        checks.append(
            CheckResult(
                name="best_variant_present_in_stats",
                status="FAIL",
                detail=best_variant_name,
            )
        )
        best_transform_check = {}
        best_geometry_check = {}
        toronto_rotation_check = {}
    else:
        best_stats_row = best_stats_row.iloc[0]
        best_transform_check = compare_transform_to_expected(original_graph, best_graph, best_stats_row, sample_nodes)
        best_geometry_check = compare_stored_geometry_to_nodes(best_graph, sample_edges)
        checks.append(
            CheckResult(
                name="best_variant_node_transform_matches_formula",
                status="PASS" if best_transform_check.get("max_error_m", 1e9) < 1.0 else "FAIL",
                detail=json.dumps(best_transform_check),
            )
        )

        geometry_status = "PASS"
        family = str(best_stats_row["type"])
        if family in {"rotate", "scale"} and best_geometry_check.get("median_error_m", 0.0) > 50.0:
            geometry_status = "FAIL"
        elif family == "translate" and best_geometry_check.get("median_error_m", 0.0) > 50.0:
            geometry_status = "WARN"
        checks.append(
            CheckResult(
                name="best_variant_edge_geometry_matches_nodes",
                status=geometry_status,
                detail=json.dumps(best_geometry_check),
            )
        )

        toronto_rotation_check = {}
        if str(best_stats_row["type"]) == "rotate":
            toronto_rotation_check = centroid_rotation_diagnostic(
                original_graph,
                best_graph,
                float(best_stats_row["angle_deg"]),
            )
            checks.append(
                CheckResult(
                    name="rotation_centroid_shift_explained_by_bbox_pivot",
                    status="PASS" if toronto_rotation_check["expected_vs_actual_error_m"] < 1.0 else "FAIL",
                    detail=json.dumps(toronto_rotation_check),
                )
            )

    worst_graph = load_graph(gdir / str(worst_variant["filename"]))
    worst_stats_row = valid_stats.loc[valid_stats["variant"] == str(worst_variant["variant"])]
    if worst_stats_row.empty:
        worst_transform_check = {}
    else:
        worst_transform_check = compare_transform_to_expected(original_graph, worst_graph, worst_stats_row.iloc[0], sample_nodes)
        checks.append(
            CheckResult(
                name="worst_variant_node_transform_matches_formula",
                status="PASS" if worst_transform_check.get("max_error_m", 1e9) < 1.0 else "FAIL",
                detail=json.dumps(worst_transform_check),
            )
        )

    water_check = water_boundary_proximity(city, original_graph)
    checks.append(
        CheckResult(
            name="near_land_boundary_or_water_indicator",
            status="WARN" if water_check["possible_near_water"] else "PASS",
            detail=json.dumps(water_check),
        )
    )

    directions = translation_directions(stats_df)
    summary_lines = [
        f"best={best_variant['filename']} ({best_variant['variant_type']}, parameter={float(best_variant['parameter']):.2f}, delta-like energy={float(best_variant['total_energy_3d_J']):.3f})",
        f"worst={worst_variant['filename']} ({worst_variant['variant_type']}, parameter={float(worst_variant['parameter']):.2f}, energy={float(worst_variant['total_energy_3d_J']):.3f})",
        f"translation_valid={directions['valid']}",
        f"translation_invalid={directions['invalid']}",
    ]

    return {
        "city": city,
        "city_label": city_label(city),
        "checks": [asdict(item) for item in checks],
        "best_variant": best_variant,
        "worst_variant": worst_variant,
        "best_transform_check": best_transform_check,
        "worst_transform_check": worst_transform_check,
        "best_geometry_check": best_geometry_check,
        "rotation_centroid_check": toronto_rotation_check,
        "translation_directions": directions,
        "water_boundary_check": water_check,
        "n_valid_variants": int(valid_stats.shape[0]),
        "n_invalid_variants": int(invalid_stats.shape[0]),
        "summary": " | ".join(summary_lines),
        "status": "FAIL" if any(item.status == "FAIL" for item in checks) else ("WARN" if any(item.status == "WARN" for item in checks) else "PASS"),
    }


def main() -> None:
    args = parse_args()
    cities = [normalize_city_token(city) for city in args.cities]

    launcher = [asdict(item) for item in launcher_checks()]
    city_reports = [city_audit(city, args.sample_nodes, args.sample_edges) for city in cities]

    report = {
        "root": str(ROOT),
        "launchers": launcher,
        "cities": city_reports,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Audit written to: {args.output_json}")
    print("")
    print("Launcher checks:")
    for item in launcher:
        print(f"- {item['status']:>4} {item['name']}: {item['detail']}")

    print("")
    for city_report in city_reports:
        print(f"{city_report['city_label']}: {city_report['status']}")
        print(f"  {city_report['summary']}")
        for check in city_report["checks"]:
            print(f"  - {check['status']:>4} {check['name']}: {check['detail']}")
        print("")


if __name__ == "__main__":
    main()
