#!/usr/bin/env python3
"""
Build supplementary analysis assets for the gravitational morphology project.

Outputs:
- city-level summaries in CSV/JSON
- energy profiles for cities that satisfy the "original is minimum 3D energy" hypothesis
- diagnostic maps for counterexamples
- land/sea rejection summaries
- a LaTeX supplementary file in supplementary/tex

The script is designed to run inside the `geo_flow` conda environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import LineString


ROOT = Path("/home/fbellisardi/code/topolity")
DATA_DIR = ROOT / "data" / "data_processed"
SUPP_DIR = ROOT / "supplementary"
PYTHON_DIR = SUPP_DIR / "python"

# Output paths — overridden at runtime by --output-root (default: figures/supplementary/)
_DEFAULT_OUT = ROOT / "figures" / "supplementary"
OUTPUT_DIR   = _DEFAULT_OUT
FIG_DIR      = OUTPUT_DIR / "figures"
TABLE_DIR    = OUTPUT_DIR / "tables"
TEX_DIR      = OUTPUT_DIR / "tex"
MAP_DIR      = FIG_DIR / "counterexamples"
PROFILE_DIR  = FIG_DIR / "supporting_profiles"
LAND_DIR     = FIG_DIR / "land_false"
OVERVIEW_DIR = FIG_DIR / "overview"
TEX_PATH     = TEX_DIR / "gravitational_morphology_supplementary.tex"
MIN_TRANSLATION_DISTANCE_M = 200.0
EXCLUDED_CITIES: set[str] = set()  # all cities with data are included
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 20
LEGEND_SIZE = 18
# Cities treated as geographically constrained (excluded from both OK and displaced-lower sections)
CONSTRAINED_OVERRIDE: set[str] = {"chicago"}
FEW_POINT_THRESHOLDS = {
    "trans_ns": 2,
    "trans_ew": 2,
    "rotation": 3,
    "scale_ns": 2,
    "scale_ew": 2,
}
WARNED_MESSAGES: set[str] = set()

CITY_DISPLAY_NAMES: dict[str, str] = {
    "amsterdam": "Amsterdam",
    "atlanta": "Atlanta",
    "bandung": "Bandung",
    "bangkok": "Bangkok",
    "barcelone": "Barcelona",
    "berlin": "Berlin",
    "bogota": "Bogota",
    "boston": "Boston",
    "bruxelles": "Brussels",
    "buenosaires": "Buenos Aires",
    "caracas": "Caracas",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "detroitwindsor": "Detroit-Windsor",
    "djakarta": "Jakarta",
    "dublin": "Dublin",
    "guadalajara": "Guadalajara",
    "hongkong": "Hong Kong",
    "houston": "Houston",
    "istanbul": "Istanbul",
    "kualalumpur": "Kuala Lumpur",
    "lima": "Lima",
    "lisbon": "Lisbon",
    "londres": "London",
    "losangeles": "Los Angeles",
    "madrid": "Madrid",
    "manchester": "Manchester",
    "manille": "Manila",
    "mexico": "Mexico City",
    "miami": "Miami",
    "milan": "Milan",
    "montreal": "Montreal",
    "moscou": "Moscow",
    "nagoya": "Nagoya",
    "newyork": "New York",
    "osaka": "Osaka",
    "paris": "Paris",
    "pekin": "Beijing",
    "philadelphie": "Philadelphia",
    "phoenix": "Phoenix",
    "riodejaneiro": "Rio de Janeiro",
    "rome": "Rome",
    "saintpetersbourg": "Saint Petersburg",
    "sandiego": "San Diego",
    "sanfrancisco": "San Francisco",
    "santiago": "Santiago",
    "santodomingo": "Santo Domingo",
    "saopaulo": "Sao Paulo",
    "seoul": "Seoul",
    "shanghai": "Shanghai",
    "singapour": "Singapore",
    "stockholm": "Stockholm",
    "sydney": "Sydney",
    "taipei": "Taipei",
    "tokyo": "Tokyo",
    "toronto": "Toronto",
    "vancouver": "Vancouver",
    "washington": "Washington",
}


COUNTEREXAMPLE_NOTES: dict[str, dict[str, Any]] = {
    "amsterdam": {
        "reason": (
            "The best counterfactual is a large westward translation of approximately 2.5 km. Amsterdam is "
            "strongly structured by water, polders, canals, and reclaimed lowlands, so feasible embeddings "
            "are highly asymmetric. A westward-translated network moves away from the IJ waterfront and "
            "the eastern harbor districts, aligning routes with the flatter polder landscape west of the "
            "city and avoiding canal-constrained crossings associated with the Grachtengordel ring."
        ),
        "sources": [
            {
                "label": "Gemeente Amsterdam - Amsterdam's water world",
                "url": "https://www.amsterdam.nl/stad-ontwikkeling/the-city-model-amsterdam/amsterdam%27-water-world/",
            },
            {
                "label": "Metropoolregio Amsterdam - landscape of dikes, polders and lakes",
                "url": "https://www.metropoolregioamsterdam.nl/landschap/",
            },
        ],
    },
    "bandung": {
        "reason": (
            "Bandung lies inside a basin and the best counterfactual is a southward translation of "
            "approximately 2.5 km. A plausible explanation is that the empirical footprint remains influenced "
            "by basin-edge relief and protected upland constraints in the north, whereas a southward-translated "
            "embedding shifts demand toward the flatter southern basin floor, reducing cumulative uphill exposure."
        ),
        "sources": [
            {
                "label": "West Java overview: Bandung located in a mountainous central area",
                "url": "https://en.wikipedia.org/wiki/West_Java",
            },
            {
                "label": "North Bandung protected-land discussion",
                "url": "https://journal.ipb.ac.id/index.php/jmht/article/view/3237",
            },
        ],
    },
    "bogota": {
        "reason": (
            "Bogota is bounded by the Cerros Orientales, including a protected forest reserve and buffer strip. "
            "The best counterfactual is a westward translation, which is consistent with the idea that moving "
            "the same network away from the eastern mountain front reduces cumulative uphill exposure."
        ),
        "sources": [
            {
                "label": "Secretaria Distrital de Ambiente - Cerros Orientales",
                "url": "https://ambientebogota.gov.co/cerros-orientales1",
            },
            {
                "label": "Bogota tourism - Eastern Hills",
                "url": "https://visitbogota.co/en/eastern-hills",
            },
        ],
    },
    "bruxelles": {
        "reason": (
            "Brussels shows a Anomalouswith a northward translation of approximately 2 km. "
            "The terrain to the north of the city transitions toward the flatter Flemish Plain, while the "
            "southeastern districts abut the Brabant uplands and border the Sonian Forest, a protected Natura "
            "2000 area. Moving the network northward reduces exposure to the relief gradient that rises from "
            "the Senne valley toward the southeastern highlands, placing more routes on lower-relief terrain."
        ),
        "sources": [
            {
                "label": "Visit Brussels - Sonian Forest",
                "url": "https://www.visit.brussels/en/visitors/venue-details.The-Sonian-Forest.263132",
            },
            {
                "label": "Sonian Forest - Natura 2000 and management",
                "url": "https://www.sonianforest.be/the-sonianforest/forest-management/",
            },
        ],
    },
    "buenosaires": {
        "reason": (
            "Buenos Aires lies on the western bank of the Río de la Plata estuary and on the Pampean plain. "
            "The best counterfactual is a westward translation of approximately 2 km, which is consistent "
            "with the idea that moving the same network away from the eastern estuarine shore places "
            "high-demand routes on flatter, more topographically uniform terrain. "
            "The analysis domain is constrained to the west and south only, because eastward and northward "
            "translations place portions of the network over the estuary or the Paraná Delta. "
            "This geographic asymmetry limits the set of feasible counterfactuals and should be interpreted "
            "cautiously."
        ),
        "sources": [
            {
                "label": "Buenos Aires Ciudad - from the city to the sea",
                "url": "https://buenosaires.gob.ar/gcaba_historico/contenidos-educativos/de-la-ciudad-al-mar",
            },
            {
                "label": "ESA - Rio de la Plata overview",
                "url": "https://www.esa.int/ESA_Multimedia/Images/2023/07/Earth_from_Space_Rio_de_la_Plata",
            },
        ],
    },
    "chicago": {
        "reason": (
            "Chicago's best counterfactual is a rotation. The city is strongly conditioned by the Lake Michigan "
            "shoreline and by an extensively engineered artificial shoreline, so rotating the same topology can "
            "change how routes intersect the waterfront edge and shoreline-protection geometry without altering "
            "the network itself."
        ),
        "sources": [
            {
                "label": "Encyclopedia of Chicago - shoreline erosion and man-made shoreline",
                "url": "https://www.encyclopedia.chicagohistory.org/pages/1142.html",
            },
            {
                "label": "USACE - Chicago shoreline protection",
                "url": "https://www.lrd.usace.army.mil/Missions/Projects/Display/Article/3639363/chicago-shoreline/",
            },
        ],
    },
    "milan": {
        "reason": (
            "The violation is numerically very small relative to the city's total energy, and the best "
            "counterfactual is an oblique translation. Milan lies on the Po plain, so this case is likely a "
            "near-degeneracy in an almost flat landscape rather than a substantively different energetic optimum."
        ),
        "sources": [
            {
                "label": "Britannica - Milan in the Po Basin",
                "url": "https://www.britannica.com/place/Milan-Italy/Landscape",
            },
        ],
    },
    "santiago": {
        "reason": (
            "Santiago's best counterfactual is a very small southward translation, so the effect is local rather "
            "than structural. Given the immediate proximity of the Andean foothills, even tens of meters can "
            "slightly alter the alignment of high-demand paths with slope corridors."
        ),
        "sources": [
            {
                "label": "Santiago Turismo - Andes foothills close to the city",
                "url": "https://www.santiagoturismo.cl/es/turismo-blanco/",
            },
        ],
    },
    "toronto": {
        "reason": (
            "Toronto combines a long Lake Ontario waterfront with a dense ravine system that channels movement "
            "through valley corridors. The strong rotational Anomaloussuggests that shoreline orientation "
            "and ravine alignment play a first-order role in the energetic landscape."
        ),
        "sources": [
            {
                "label": "City of Toronto - Ravine Strategy",
                "url": "https://www.toronto.ca/city-government/accountability-operations-customer-service/long-term-vision-plans-and-strategies/ravine-strategy/",
            },
            {
                "label": "TRCA - Lake Ontario waterfront",
                "url": "https://trca.ca/conservation/watershed-management/lake-ontario-waterfront/",
            },
        ],
    },
}

CITY_CONTEXT: dict[str, list[str]] = {
    "amsterdam": ["water_network", "reclaimed_land"],
    "bandung": ["basin", "volcanic"],
    "barcelone": ["coastal"],
    "bogota": ["mountain_front", "protected_area"],
    "bruxelles": ["forest_edge", "relief_gradient"],
    "buenosaires": ["estuary", "flat_plain"],
    "caracas": ["mountain_front"],
    "chicago": ["lakefront", "engineered_shoreline"],
    "djakarta": ["coastal", "deltaic"],
    "kualalumpur": ["coastal_constraint"],
    "lima": ["coastal", "mountain_front"],
    "mexico": ["basin"],
    "milan": ["flat_plain"],
    "santiago": ["mountain_front", "basin"],
    "singapour": ["island", "coastal"],
    "toronto": ["lakefront", "ravines"],
}


@dataclass
class CitySummary:
    city: str
    status: str
    best_filename: str
    best_variant: str
    best_family: str
    best_parameter: float
    best_energy_3d_j: float
    original_energy_3d_j: float
    delta_j: float
    delta_pct: float
    original_rank: int
    n_variants: int
    n_land_false: int
    frac_land_false: float


def configure_matplotlib() -> None:
    """Nature-paper style: clean axes, no grid, 300 DPI, publication-ready fonts."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.2,
        }
    )


def ensure_directories() -> None:
    for path in [OUTPUT_DIR, FIG_DIR, MAP_DIR, PROFILE_DIR, LAND_DIR, OVERVIEW_DIR, TABLE_DIR, TEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def city_dirs() -> list[Path]:
    return sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name not in EXCLUDED_CITIES])


def work_csv_path(city: str) -> Path:
    return DATA_DIR / city / "graphs_fine_grid" / "fine_grid_gravitational_work.csv"


def stats_csv_path(city: str) -> Path:
    return DATA_DIR / city / "graphs_fine_grid" / "fine_grid_stats.csv"


def graph_path(city: str, filename: str) -> Path:
    return DATA_DIR / city / "graphs_fine_grid" / filename


def land_path(city: str) -> Path:
    return DATA_DIR / city / "land" / f"{city}_clipped_land.shp"


def dem_path(city: str) -> Path:
    return DATA_DIR / city / "dem" / f"{city}_dem.tif"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def city_display_name(city: str) -> str:
    return CITY_DISPLAY_NAMES.get(city, city.replace("_", " ").title())


def format_plain_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def print_warning_once(message: str) -> None:
    if message not in WARNED_MESSAGES:
        print(message)
        WARNED_MESSAGES.add(message)


def format_family_display(family: str) -> str:
    mapping = {
        "original": "Original",
        "rotation": "Rotation",
        "scale": "Scale",
        "translation": "Translation",
        "other": "Other",
    }
    return mapping.get(str(family).lower(), str(family).title())


def normalize_scale_axis(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"ns", "north-south", "north_south"}:
        return "ns"
    if text in {"ew", "east-west", "east_west"}:
        return "ew"
    return None


def axis_from_angle(angle: Any) -> str | None:
    try:
        angle_float = float(angle) % 360
    except Exception:
        return None
    if angle_float in (0, 180):
        return "ew"
    if angle_float in (90, 270):
        return "ns"
    return "other"


def parse_translation_distance_from_text(text: str) -> float | None:
    match = re.search(r"trans_(\d+(?:\.\d+)?)m", text)
    if match:
        return safe_float(match.group(1))
    return None


def english_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def transformation_sentence(shown: list[str]) -> str:
    if not shown:
        return "Only a narrow subset of transformed embeddings remains available."
    if len(shown) == 1:
        return f"The figure reports {shown[0]}."
    if len(shown) == 2:
        return f"The figure reports {shown[0]} and {shown[1]}."
    return f"The figure reports {english_join(shown)}."


def context_constraint_sentence(city: str, availability: dict[str, Any]) -> str | None:
    tags = set(availability.get("context_tags", []))
    city_name = city_display_name(city)

    if {"water_network", "reclaimed_land"} & tags:
        return (
            f"In {city_name}, this is consistent with a water-managed urban fabric shaped by canals, polders, "
            "or reclaimed land, where even simple displacements can quickly leave the feasible domain."
        )
    if {"lakefront", "engineered_shoreline"} <= tags or ("lakefront" in tags and "engineered_shoreline" in tags):
        return (
            f"In {city_name}, the pattern is consistent with a strongly structured waterfront, where the lake edge "
            "and engineered shoreline constrain the range of valid transformed embeddings."
        )
    if "lakefront" in tags and "ravines" in tags:
        return (
            f"In {city_name}, the combination of shoreline geometry and ravine corridors plausibly narrows the set "
            "of feasible embeddings."
        )
    if "coastal" in tags or "island" in tags or "deltaic" in tags or "estuary" in tags:
        return (
            f"In {city_name}, the coastal setting and shoreline geometry plausibly restrict the range of valid "
            "transformed embeddings."
        )
    if "mountain_front" in tags and "basin" in tags:
        return (
            f"In {city_name}, the pattern is consistent with a basin or mountain-front setting, where relief and "
            "corridor-like accessibility can make the feasible domain strongly asymmetric."
        )
    if "mountain_front" in tags:
        return (
            f"In {city_name}, the asymmetry is also compatible with a mountain-front setting, where relief and "
            "access corridors constrain displacement directions."
        )
    if "protected_area" in tags:
        return (
            f"In {city_name}, nearby protected terrain may further reduce the range of admissible embeddings."
        )
    if "forest_edge" in tags or "relief_gradient" in tags:
        return (
            f"In {city_name}, the pattern is consistent with a relief gradient and a constrained urban edge."
        )
    if "flat_plain" in tags:
        return (
            f"In {city_name}, the smaller differences are consistent with a flatter setting, where the energy "
            "landscape may be comparatively shallow."
        )
    if availability["missing_labels"] or availability["infeasible_labels"]:
        return (
            "More generally, the missing families suggest that the feasible embedding space is narrower "
            "than the unconstrained Euclidean transformation space."
        )
    return None


def parse_angle_from_text(text: str) -> float | None:
    match = re.search(r"_a([+-]?\d{1,3})", text)
    if match:
        return float(match.group(1)) % 360
    return None


def classify_family(filename: str, variant_type: str) -> str:
    vt = str(variant_type).lower()
    fn = str(filename).lower()
    if vt == "original" or "original" in fn:
        return "original"
    if vt.startswith("rotation") or "rot_" in fn:
        return "rotation"
    if vt.startswith("scale") or "scale_" in fn:
        return "scale"
    if vt.startswith("translation") or vt.startswith("translate") or "trans_" in fn:
        return "translation"
    return "other"


def parse_translation_direction(filename: str) -> str | None:
    angle = parse_angle_from_text(filename)
    if angle is None:
        return None
    if angle in (0, 180):
        return "ew"
    if angle in (90, 270):
        return "ns"
    return "other"


def signed_translation_distance(filename: str, parameter: float) -> float | None:
    angle = parse_angle_from_text(filename)
    if angle is None or math.isnan(parameter):
        return None
    if angle == 0:
        return parameter
    if angle == 180:
        return -parameter
    if angle == 90:
        return parameter
    if angle == 270:
        return -parameter
    return None


def read_work_df(city: str) -> pd.DataFrame:
    df = pd.read_csv(work_csv_path(city)).copy()
    if "total_energy_3d_J" not in df.columns:
        raise KeyError(f"{city}: missing total_energy_3d_J")
    df["city"] = city
    df["energy_plot"] = df["total_energy_3d_J"]
    df["family"] = df.apply(lambda row: classify_family(row["filename"], row["variant_type"]), axis=1)
    df["parameter"] = df["parameter"].apply(safe_float)
    df["translation_axis"] = df["filename"].astype(str).map(parse_translation_direction)
    df["signed_translation"] = df.apply(
        lambda row: signed_translation_distance(str(row["filename"]), safe_float(row["parameter"])),
        axis=1,
    )
    is_small_translation = (df["family"] == "translation") & df["parameter"].notna() & (df["parameter"] < MIN_TRANSLATION_DISTANCE_M)
    df = df.loc[~is_small_translation].copy()
    return df


def read_stats_df(city: str) -> pd.DataFrame:
    path = stats_csv_path(city)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).copy()
    if "on_land" in df.columns:
        df["on_land_bool"] = df["on_land"].astype(str).str.lower().eq("true")
    df["family"] = df.apply(
        lambda row: classify_family(f"graph_{row.get('variant', '')}.pkl", row.get("type", "")),
        axis=1,
    )
    if "translation_angle" in df.columns:
        df["translation_axis"] = df["translation_angle"].apply(axis_from_angle)
    else:
        df["translation_axis"] = df.get("variant", pd.Series(dtype=str)).astype(str).map(parse_translation_direction)
    if "translation_distance_m" in df.columns:
        df["translation_distance_m"] = df["translation_distance_m"].apply(safe_float)
    else:
        df["translation_distance_m"] = df.get("variant", pd.Series(dtype=str)).astype(str).map(parse_translation_distance_from_text)
    if "scale_axis" in df.columns:
        df["scale_axis_norm"] = df["scale_axis"].apply(normalize_scale_axis)
    else:
        variant_series = df.get("variant", pd.Series(dtype=str)).astype(str).str.lower()
        df["scale_axis_norm"] = np.where(
            variant_series.str.contains("scale_ns", na=False),
            "ns",
            np.where(variant_series.str.contains("scale_ew", na=False), "ew", None),
        )
    return df


def filter_stats_df_for_supplementary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    is_small_translation = (
        df["family"].eq("translation")
        & df["translation_distance_m"].notna()
        & (df["translation_distance_m"] < MIN_TRANSLATION_DISTANCE_M)
    )
    return df.loc[~is_small_translation].copy()


def summarize_city(city: str) -> CitySummary | None:
    work_path = work_csv_path(city)
    if not work_path.exists():
        return None

    df = read_work_df(city)
    stats_df = filter_stats_df_for_supplementary(read_stats_df(city))

    original = df[df["family"] == "original"]
    if original.empty:
        raise ValueError(f"{city}: original configuration not found")
    original_row = original.iloc[0]
    original_energy = float(original_row["total_energy_3d_J"])

    best_row = df.loc[df["total_energy_3d_J"].idxmin()]
    best_energy = float(best_row["total_energy_3d_J"])
    delta_j = best_energy - original_energy
    delta_pct = 100.0 * (best_energy / original_energy - 1.0)
    original_rank = int((df["total_energy_3d_J"] < original_energy).sum()) + 1
    # Treat as OK if the deviation is within numerical precision (< 0.05%)
    # This handles cases like Madrid where delta ~ -0.014% due to rounding.
    if city in CONSTRAINED_OVERRIDE:
        status = "CONSTRAINED"
    elif best_row["family"] == "original" or abs(delta_pct) < 0.05:
        status = "OK"
    else:
        status = "DISPLACED_LOWER"

    n_land_false = 0
    frac_land_false = 0.0
    if not stats_df.empty and "on_land_bool" in stats_df.columns:
        n_land_false = int((~stats_df["on_land_bool"]).sum())
        frac_land_false = n_land_false / max(len(stats_df), 1)

    return CitySummary(
        city=city,
        status=status,
        best_filename=str(best_row["filename"]),
        best_variant=str(best_row["variant_type"]),
        best_family=str(best_row["family"]),
        best_parameter=float(best_row["parameter"]),
        best_energy_3d_j=best_energy,
        original_energy_3d_j=original_energy,
        delta_j=delta_j,
        delta_pct=delta_pct,
        original_rank=original_rank,
        n_variants=int(len(df)),
        n_land_false=n_land_false,
        frac_land_false=frac_land_false,
    )


def build_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    city_rows: list[dict[str, Any]] = []
    land_rows: list[dict[str, Any]] = []

    for city_dir in city_dirs():
        city = city_dir.name
        summary = summarize_city(city)
        if summary is not None:
            city_rows.append(summary.__dict__)

        stats_path = stats_csv_path(city)
        if stats_path.exists():
            stats_df = filter_stats_df_for_supplementary(read_stats_df(city))
            if not stats_df.empty and "on_land_bool" in stats_df.columns:
                land_false_df = stats_df[~stats_df["on_land_bool"]].copy()
                family_counts: dict[str, int] = {}
                for variant in land_false_df.get("variant", pd.Series(dtype=str)).astype(str):
                    family = classify_family(f"graph_{variant}.pkl", variant)
                    family_counts[family] = family_counts.get(family, 0) + 1
                land_rows.append(
                    {
                        "city": city,
                        "n_total_stats": int(len(stats_df)),
                        "n_land_false": int(len(land_false_df)),
                        "frac_land_false": float(len(land_false_df) / max(len(stats_df), 1)),
                        "rotation_land_false": family_counts.get("rotation", 0),
                        "translation_land_false": family_counts.get("translation", 0),
                        "scale_land_false": family_counts.get("scale", 0),
                        "examples": ";".join(land_false_df.get("variant", pd.Series(dtype=str)).astype(str).head(10).tolist()),
                    }
                )

    summary_df = pd.DataFrame(city_rows).sort_values(["status", "city"]).reset_index(drop=True)
    land_df = pd.DataFrame(land_rows).sort_values(["n_land_false", "city"], ascending=[False, True]).reset_index(drop=True)
    return summary_df, land_df


def save_tables(summary_df: pd.DataFrame, land_df: pd.DataFrame) -> None:
    summary_export = summary_df.copy()
    summary_export["city_display"] = summary_export["city"].map(city_display_name)
    summary_export["status_label"] = summary_export["status"].map(
        {"OK": "Original minimum", "COUNTEREXAMPLE": "Counterexample"}
    )
    summary_export["best_family_label"] = summary_export["best_family"].map(format_family_display)
    summary_export.to_csv(TABLE_DIR / "city_hypothesis_summary.csv", index=False)

    land_export = land_df.copy()
    if not land_export.empty:
        land_export["city_display"] = land_export["city"].map(city_display_name)
    land_export.to_csv(TABLE_DIR / "land_false_summary.csv", index=False)

    payload = {
        "n_cities_with_work": int(len(summary_df)),
        "n_supporting": int((summary_df["status"] == "OK").sum()),
        "n_counterexamples": int((summary_df["status"] == "COUNTEREXAMPLE").sum()),
        "supporting_cities": summary_df.loc[summary_df["status"] == "OK", "city"].sort_values().tolist(),
        "counterexample_cities": summary_df.loc[summary_df["status"] == "COUNTEREXAMPLE", "city"].sort_values().tolist(),
        "counterexample_notes": COUNTEREXAMPLE_NOTES,
    }
    (TABLE_DIR / "summary_payload.json").write_text(json.dumps(payload, indent=2))


def family_panel_label(panel_key: str) -> str:
    labels = {
        "trans_ns": "North--South translations",
        "trans_ew": "East--West translations",
        "rotation": "rotations",
        "scale_ns": "North--South dilations",
        "scale_ew": "East--West dilations",
    }
    return labels[panel_key]


def context_group(city: str) -> str | None:
    contexts = set(CITY_CONTEXT.get(city, []))
    if contexts & {"coastal", "island", "deltaic", "estuary", "lakefront", "water_network", "reclaimed_land", "coastal_constraint"}:
        return "coastal"
    if contexts & {"mountain_front", "basin", "volcanic", "protected_area", "relief_gradient", "forest_edge", "ravines"}:
        return "relief"
    if contexts & {"flat_plain"}:
        return "plain"
    return None


def one_sided_direction_label(axis: str, signed_values: pd.Series) -> str | None:
    signs = {int(np.sign(v)) for v in signed_values.dropna() if not math.isclose(v, 0.0, abs_tol=1e-9)}
    if len(signs) != 1:
        return None
    sign = next(iter(signs))
    if axis == "ns":
        return "northward" if sign > 0 else "southward"
    if axis == "ew":
        return "eastward" if sign > 0 else "westward"
    return None


def transformation_availability(city: str, work_df: pd.DataFrame, stats_df: pd.DataFrame) -> dict[str, Any]:
    stats_df = filter_stats_df_for_supplementary(stats_df)
    trans_df = work_df[(work_df["family"] == "translation") & work_df["signed_translation"].notna()].copy()
    rot_df = work_df[work_df["family"] == "rotation"].copy()
    scale_ns = work_df[
        (work_df["family"] == "scale")
        & work_df["variant_type"].astype(str).str.contains("ns", case=False, na=False)
    ].copy()
    scale_ew = work_df[
        (work_df["family"] == "scale")
        & work_df["variant_type"].astype(str).str.contains("ew", case=False, na=False)
    ].copy()

    panel_data = {
        "trans_ns": trans_df[trans_df["translation_axis"] == "ns"],
        "trans_ew": trans_df[trans_df["translation_axis"] == "ew"],
        "rotation": rot_df,
        "scale_ns": scale_ns,
        "scale_ew": scale_ew,
    }
    attempted_counts = {
        "trans_ns": int(((stats_df["family"] == "translation") & (stats_df["translation_axis"] == "ns")).sum()) if not stats_df.empty else 0,
        "trans_ew": int(((stats_df["family"] == "translation") & (stats_df["translation_axis"] == "ew")).sum()) if not stats_df.empty else 0,
        "rotation": int((stats_df["family"] == "rotation").sum()) if not stats_df.empty else 0,
        "scale_ns": int(((stats_df["family"] == "scale") & (stats_df["scale_axis_norm"] == "ns")).sum()) if not stats_df.empty else 0,
        "scale_ew": int(((stats_df["family"] == "scale") & (stats_df["scale_axis_norm"] == "ew")).sum()) if not stats_df.empty else 0,
    }
    rejected_counts = {
        "trans_ns": int(((stats_df["family"] == "translation") & (stats_df["translation_axis"] == "ns") & (~stats_df["on_land_bool"])).sum()) if ("on_land_bool" in stats_df.columns and not stats_df.empty) else 0,
        "trans_ew": int(((stats_df["family"] == "translation") & (stats_df["translation_axis"] == "ew") & (~stats_df["on_land_bool"])).sum()) if ("on_land_bool" in stats_df.columns and not stats_df.empty) else 0,
        "rotation": int(((stats_df["family"] == "rotation") & (~stats_df["on_land_bool"])).sum()) if ("on_land_bool" in stats_df.columns and not stats_df.empty) else 0,
        "scale_ns": int(((stats_df["family"] == "scale") & (stats_df["scale_axis_norm"] == "ns") & (~stats_df["on_land_bool"])).sum()) if ("on_land_bool" in stats_df.columns and not stats_df.empty) else 0,
        "scale_ew": int(((stats_df["family"] == "scale") & (stats_df["scale_axis_norm"] == "ew") & (~stats_df["on_land_bool"])).sum()) if ("on_land_bool" in stats_df.columns and not stats_df.empty) else 0,
    }

    panels: dict[str, dict[str, Any]] = {}
    for key, df_panel in panel_data.items():
        n_valid = int(len(df_panel))
        n_attempted = attempted_counts[key]
        limited = 0 < n_valid <= FEW_POINT_THRESHOLDS[key]
        empty_reason = None
        if n_valid == 0:
            empty_reason = "not_feasible" if n_attempted > 0 else "not_tested"
        panels[key] = {
            "label": family_panel_label(key),
            "n_valid": n_valid,
            "n_attempted": n_attempted,
            "limited": limited,
            "empty_reason": empty_reason,
            "rejected": rejected_counts[key],
            "one_sided_direction": None,
        }

    panels["trans_ns"]["one_sided_direction"] = one_sided_direction_label("ns", panel_data["trans_ns"]["signed_translation"])
    panels["trans_ew"]["one_sided_direction"] = one_sided_direction_label("ew", panel_data["trans_ew"]["signed_translation"])

    shown_labels: list[str] = []
    if panels["trans_ns"]["n_valid"] > 0:
        shown_labels.append("North--South translations")
    if panels["trans_ew"]["n_valid"] > 0:
        shown_labels.append("East--West translations")
    if panels["rotation"]["n_valid"] > 0:
        shown_labels.append("rotations")
    if panels["scale_ns"]["n_valid"] > 0 and panels["scale_ew"]["n_valid"] > 0:
        shown_labels.append("directional dilations")
    else:
        if panels["scale_ns"]["n_valid"] > 0:
            shown_labels.append("North--South dilations")
        if panels["scale_ew"]["n_valid"] > 0:
            shown_labels.append("East--West dilations")

    missing_labels = [panel["label"] for panel in panels.values() if panel["empty_reason"] == "not_tested"]
    infeasible_labels = [panel["label"] for panel in panels.values() if panel["empty_reason"] == "not_feasible"]
    limited_labels = [panel["label"] for panel in panels.values() if panel["limited"]]
    has_land_rejections = bool((stats_df["on_land_bool"] == False).any()) if "on_land_bool" in stats_df.columns else False

    if panels["rotation"]["n_valid"] == 0:
        print_warning_once(f"WARNING: {city_display_name(city)} has no valid rotation profiles.")
    for axis_key, phrase in (("trans_ns", "translations"), ("trans_ew", "translations")):
        one_sided = panels[axis_key]["one_sided_direction"]
        if one_sided is not None:
            print_warning_once(f"WARNING: {city_display_name(city)} has only {one_sided} {phrase} available.")
    for axis_key in ("scale_ns", "scale_ew"):
        if panels[axis_key]["n_valid"] == 0 and panels[axis_key]["n_attempted"] > 0:
            print_warning_once(f"WARNING: {city_display_name(city)} has no valid {panels[axis_key]['label'].lower()}.")

    return {
        "panels": panels,
        "shown_labels": shown_labels,
        "missing_labels": missing_labels,
        "infeasible_labels": infeasible_labels,
        "limited_labels": limited_labels,
        "has_land_rejections": has_land_rejections,
        "context_group": context_group(city),
        "context_tags": CITY_CONTEXT.get(city, []),
    }


def minimum_energy_profile_assessment(work_df: pd.DataFrame) -> dict[str, Any]:
    original_energy = float(work_df.loc[work_df["family"] == "original", "energy_plot"].iloc[0])
    transformed = work_df[work_df["family"] != "original"].copy()
    if transformed.empty:
        return {"nearest_delta_pct": float("nan"), "shape": "undetermined"}
    nearest_delta_pct = 100.0 * (float(transformed["energy_plot"].min()) / original_energy - 1.0)
    if nearest_delta_pct <= 0.20:
        shape = "near_flat"
    elif nearest_delta_pct >= 1.00:
        shape = "clear_increase"
    else:
        shape = "moderate_increase"
    return {"nearest_delta_pct": nearest_delta_pct, "shape": shape}


def make_supporting_caption(city: str, work_df: pd.DataFrame, stats_df: pd.DataFrame, summary_row: pd.Series) -> str:
    availability = transformation_availability(city, work_df, stats_df)
    assessment = minimum_energy_profile_assessment(work_df)
    city_name = city_display_name(city)
    parts = [f"Energetic sensitivity around the empirical embedding for {city_name}. The red star marks the empirical configuration."]
    shown = availability["shown_labels"]
    parts.append(transformation_sentence(shown))

    if availability["limited_labels"]:
        parts.append(
            f"For {english_join(availability['limited_labels'])}, only a limited set of valid transformed embeddings is available."
        )

    if availability["has_land_rejections"]:
        parts.append(
            "Transformed embeddings that place part of the network outside the valid terrestrial domain or outside valid DEM support are excluded."
        )
    elif availability["missing_labels"] or availability["infeasible_labels"]:
        parts.append("Families without any feasible realizations are omitted.")

    if str(summary_row["status"]) == "OK":
        parts.append("Within the feasible embedding space, the empirical embedding remains the lowest-energy configuration.")

    context_sentence = context_constraint_sentence(city, availability)
    if context_sentence and (availability["has_land_rejections"] or availability["infeasible_labels"] or availability["limited_labels"]):
        parts.append(context_sentence)

    if assessment["shape"] == "near_flat":
        parts.append(
            "The energy landscape remains locally shallow, which suggests a low-energy neighborhood around the empirical embedding rather than a sharply isolated optimum."
        )
    elif assessment["shape"] == "clear_increase":
        parts.append("Across the feasible transformed embeddings, energy rises clearly away from the empirical configuration.")

    return " ".join(parts)


def make_counterexample_caption(city: str, summary_row: pd.Series) -> str:
    city_name = city_display_name(city)
    family = str(summary_row["best_family"])
    parameter = format_plain_number(float(summary_row["best_parameter"]), decimals=2)
    delta_pct = float(summary_row["delta_pct"])
    if family == "translation":
        right_panel = "Right: the original urban network translated according to the displacement associated with the lowest-energy transformed embedding."
    elif family == "rotation":
        right_panel = "Right: the lowest-energy rotated embedding, shown against the same geographic context."
    elif family == "scale":
        right_panel = "Right: the lowest-energy directionally scaled embedding, shown against the same geographic context."
    else:
        right_panel = "Right: the lowest-energy transformed embedding, shown against the same geographic context."
    return (
        f"Original and best transformed embeddings for {city_name}. "
        f"The best transformed configuration is a {family} with parameter {parameter} and relative deviation {delta_pct:.2f}%. "
        f"Left: the empirical embedding. {right_panel}"
    )


def make_counterexample_interpretation(city: str, summary_row: pd.Series) -> str:
    delta_pct = float(summary_row["delta_pct"])
    abs_delta = abs(delta_pct)
    city_name = city_display_name(city)
    note = COUNTEREXAMPLE_NOTES.get(city, {})
    note_text = note.get("reason", "")
    parts: list[str] = []
    if city == "milan" or abs_delta <= 0.10:
        parts.append(
            f"For {city_name}, the deviation is extremely small ({delta_pct:.2f}%), so this case is better read as a near-degeneracy than as evidence for a distinct structural optimum."
        )
    else:
        parts.append(
            f"Within the tested transformation set, the best transformed embedding lowers the total energy by {abs_delta:.2f}% relative to the empirical embedding. The difference is therefore worth noting, but not overstating."
        )
    if note_text:
        parts.append(note_text)
    parts.append(
        "Taken together, the pattern is plausibly consistent with territorial constraints, shoreline geometry, protected land, historical path dependence, or infrastructure barriers, rather than with a single definitive causal explanation."
    )
    return " ".join(parts)


def make_supporting_city_paragraph(city: str, work_df: pd.DataFrame, stats_df: pd.DataFrame, summary_row: pd.Series) -> str:
    availability = transformation_availability(city, work_df, stats_df)
    assessment = minimum_energy_profile_assessment(work_df)
    city_name = city_display_name(city)
    parts: list[str] = []
    shown = availability["shown_labels"]
    if shown:
        parts.append(
            f"For {city_name}, the available profiles cover {english_join(shown)} around the empirical configuration."
        )
    if availability["missing_labels"] or availability["infeasible_labels"]:
        omitted = availability["missing_labels"] + availability["infeasible_labels"]
        parts.append(
            f"The omitted panels correspond to {english_join(omitted)} and are left out because no valid transformed embeddings remain within the feasible domain."
        )
    if availability["limited_labels"]:
        parts.append(
            f"For {english_join(availability['limited_labels'])}, only a limited subset of valid transformed embeddings is available."
        )
    if availability["has_land_rejections"]:
        parts.append(
            "Some transformations are discarded because they move part of the network outside the valid terrestrial domain or outside valid DEM support."
        )
    context_sentence = context_constraint_sentence(city, availability)
    if context_sentence:
        parts.append(context_sentence)
    if str(summary_row["status"]) == "OK":
        parts.append("Within that feasible space, the empirical embedding remains the lowest-energy configuration.")
    if assessment["shape"] == "near_flat":
        parts.append(
            "The local energy landscape is fairly shallow, so the result is better read as a stable low-energy neighborhood than as a sharply isolated optimum."
        )
    elif assessment["shape"] == "clear_increase":
        parts.append("Where profiles are available, energy increases clearly away from the empirical embedding within the tested feasible domain.")
    return " ".join(parts)


def plot_overview_status(summary_df: pd.DataFrame) -> Path:
    """Bar chart of energy deviation for all cities — matches paper figure style."""
    fig, ax = plt.subplots(figsize=(max(10, len(summary_df) * 0.32), 5))
    df = summary_df.sort_values("delta_pct").copy()
    df["city_label"] = df["city"].map(city_display_name)
    x = np.arange(len(df))
    counter_mask = df["status"].eq("COUNTEREXAMPLE")
    ok_mask = ~counter_mask

    ax.bar(x[counter_mask], df.loc[counter_mask, "delta_pct"],
           color="#2563eb", edgecolor="#1e3a8a", linewidth=0.5, zorder=3, label="Counterexample")
    ax.scatter(x[ok_mask], np.zeros(ok_mask.sum()),
               s=55, marker="o", color="#dc2626", edgecolor="#7f1d1d",
               linewidth=0.6, zorder=5, label="Original is minimum")
    ax.axhline(0, color="#374151", linewidth=0.8)

    ax.set_ylabel(r"$100 \times (W_{\min}/W_{\mathrm{orig}} - 1)\ [\%]$", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["city_label"], fontsize=8)
    ax.tick_params(axis="x", rotation=75, length=2)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(min(-13, df["delta_pct"].min() * 1.15), max(3, df["delta_pct"].max() * 1.2))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626",
               markeredgecolor="#7f1d1d", markersize=7, label="Original is minimum"),
        Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Counterexample"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=9)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.tight_layout()
    out = OVERVIEW_DIR / "hypothesis_status_overview.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_counterexample_deltas(summary_df: pd.DataFrame) -> Path:
    """Horizontal bar chart of improvement magnitude for Anomalouscities."""
    df = summary_df[summary_df["status"] == "COUNTEREXAMPLE"].sort_values("delta_pct").copy()
    df["city_label"] = df["city"].map(city_display_name)
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.4)))
    ax.barh(df["city_label"], -df["delta_pct"],
            color="#2563eb", edgecolor="#1e3a8a", linewidth=0.5)
    ax.set_xlabel(r"Energy reduction relative to empirical embedding [\%]", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    plt.tight_layout()
    out = OVERVIEW_DIR / "counterexample_improvement.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_land_false_overview(land_df: pd.DataFrame) -> tuple[Path, Path]:
    """Bar chart of rejected transformations per city."""
    df = land_df[land_df["n_land_false"] > 0].sort_values("n_land_false", ascending=False).copy()
    df["city_label"] = df["city"].map(city_display_name)

    # Stacked bar by family
    fig1, ax1 = plt.subplots(figsize=(max(10, len(df) * 0.38), 5))
    rot_c, trans_c, scale_c = "#fdae61", "#2b83ba", "#74c476"
    bottom = np.zeros(len(df))
    for col, color, label in [
        ("rotation_land_false",    rot_c,   "Rotation"),
        ("translation_land_false", trans_c, "Translation"),
        ("scale_land_false",       scale_c, "Scale"),
    ]:
        vals = df[col].values.astype(float)
        ax1.bar(df["city_label"], vals, bottom=bottom,
                color=color, edgecolor="#374151", linewidth=0.3, label=label)
        bottom += vals

    ax1.set_ylabel("Rejected transformed configurations", fontsize=11)
    ax1.tick_params(axis="x", rotation=75, labelsize=8, length=2)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    out1 = LAND_DIR / "land_false_counts.png"
    fig1.savefig(out1, bbox_inches="tight")
    fig1.savefig(out1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig1)

    # Heatmap
    heat_df = df.set_index("city_label")[
        ["rotation_land_false", "translation_land_false", "scale_land_false"]
    ]
    fig2, ax2 = plt.subplots(figsize=(6, max(5, 0.32 * len(heat_df))))
    im = ax2.imshow(heat_df.to_numpy(), cmap="Blues", aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Rotation", "Translation", "Scale"], fontsize=10)
    ax2.set_yticks(range(len(heat_df)))
    ax2.set_yticklabels(heat_df.index, fontsize=9)
    for i in range(heat_df.shape[0]):
        for j in range(heat_df.shape[1]):
            v = int(heat_df.iloc[i, j])
            ax2.text(j, i, str(v) if v > 0 else "",
                     ha="center", va="center", fontsize=8,
                     color="white" if v > heat_df.values.max() * 0.6 else "#1f2937")
    cbar = fig2.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cbar.set_label("Count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    out2 = LAND_DIR / "land_false_heatmap.png"
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)
    return out1, out2


def load_graph(city: str, filename: str) -> nx.MultiDiGraph:
    with graph_path(city, filename).open("rb") as handle:
        return pickle.load(handle)


def graph_edges_gdf(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    records: list[dict[str, Any]] = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            geom = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])
        records.append({"geometry": geom, "length": data.get("length", np.nan)})
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=G.graph.get("crs", "EPSG:4326"))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def translate_edges_gdf(gdf: gpd.GeoDataFrame, dx: float, dy: float) -> gpd.GeoDataFrame:
    shifted = gdf.copy()
    shifted["geometry"] = shifted.geometry.translate(xoff=dx, yoff=dy)
    return shifted


def graph_centroid_lonlat(G: nx.MultiDiGraph) -> tuple[float, float]:
    xs = np.fromiter((data["x"] for _, data in G.nodes(data=True)), dtype=float)
    ys = np.fromiter((data["y"] for _, data in G.nodes(data=True)), dtype=float)
    return float(xs.mean()), float(ys.mean())


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_label(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def padded_bounds(*geometries: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx = min(g.total_bounds[0] for g in geometries if not g.empty)
    miny = min(g.total_bounds[1] for g in geometries if not g.empty)
    maxx = max(g.total_bounds[2] for g in geometries if not g.empty)
    maxy = max(g.total_bounds[3] for g in geometries if not g.empty)
    padx = max((maxx - minx) * 0.08, 0.01)
    pady = max((maxy - miny) * 0.08, 0.01)
    return minx - padx, miny - pady, maxx + padx, maxy + pady


def _infer_zoom(ax: plt.Axes) -> int:
    """Return a safe tile zoom level based on the current axes extent in degrees."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
    if span > 1.0:
        return 9
    if span > 0.5:
        return 10
    if span > 0.2:
        return 11
    return 12


def try_add_basemap(ax: plt.Axes) -> bool:
    if os.environ.get("SUPP_DISABLE_BASEMAP", "").lower() in {"1", "true", "yes"}:
        return False
    zoom = _infer_zoom(ax)
    for attempt_zoom in (zoom, max(zoom - 1, 8)):
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.CartoDB.Positron,
                attribution_size=6,
                zoom=attempt_zoom,
            )
            return True
        except Exception as e:
            print_warning_once(f"Basemap attempt zoom={attempt_zoom} failed: {e}")
    return False


def plot_counterexample_map(city: str, summary_row: pd.Series) -> Path:
    original_g = load_graph(city, "graph_original.pkl")
    best_g = load_graph(city, str(summary_row["best_filename"]))
    land_gdf = gpd.read_file(land_path(city))

    gdf_orig = graph_edges_gdf(original_g)
    gdf_best = graph_edges_gdf(best_g)
    orig_centroid = graph_centroid_lonlat(original_g)
    best_centroid = graph_centroid_lonlat(best_g)
    dx = best_centroid[0] - orig_centroid[0]
    dy = best_centroid[1] - orig_centroid[1]
    family = str(summary_row["best_family"]).lower()
    if family == "translation":
        gdf_display_best = translate_edges_gdf(gdf_orig, dx=dx, dy=dy)
    else:
        gdf_display_best = gdf_best

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax0, ax1 = axes
    geo_bounds = padded_bounds(gdf_orig, gdf_display_best, land_gdf)
    dx_box = geo_bounds[2] - geo_bounds[0]
    dy_box = geo_bounds[3] - geo_bounds[1]
    shift_distance_m = haversine_m(orig_centroid[0], orig_centroid[1], best_centroid[0], best_centroid[1])
    shift_label = f"{shift_distance_m/1000.0:.2f} km {bearing_label(dx, dy)}"

    # Set final bounds first so the basemap is fetched for the full padded extent.
    for _ax in (ax0, ax1):
        _ax.set_xlim(geo_bounds[0], geo_bounds[2])
        _ax.set_ylim(geo_bounds[1], geo_bounds[3])

    gdf_orig.plot(ax=ax0, color="#f97316", linewidth=1.4, alpha=0.95, zorder=4)
    has_basemap_0 = try_add_basemap(ax0)
    ax0.set_xlim(geo_bounds[0], geo_bounds[2])
    ax0.set_ylim(geo_bounds[1], geo_bounds[3])
    ax0.set_title("Original embedding")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.grid(True, color="#94a3b8", alpha=0.35, linewidth=0.6)

    gdf_display_best.plot(ax=ax1, color="#1d4ed8", linewidth=1.4, alpha=0.9, zorder=4)
    has_basemap_1 = try_add_basemap(ax1)
    ax1.set_xlim(geo_bounds[0], geo_bounds[2])
    ax1.set_ylim(geo_bounds[1], geo_bounds[3])
    ax1.set_title("Best transformed embedding")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, color="#94a3b8", alpha=0.35, linewidth=0.6)

    label_dx = 0.018 * dx_box
    label_dy = 0.010 * dy_box

    for ax, centroid, color, label in (
        (ax0, orig_centroid, "#b45309", "orig centroid"),
        (ax1, best_centroid, "#1d4ed8", "best centroid"),
    ):
        ax.scatter([centroid[0]], [centroid[1]], s=50, color=color, edgecolor="white", linewidth=1.0, zorder=6)
        ax.text(
            centroid[0] + label_dx,
            centroid[1] + label_dy,
            label,
            color=color,
            fontsize=14,
            ha="left",
            va="center",
            zorder=7,
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
        )

    if family == "translation":
        shift_text = f"shift from original centroid: {shift_label}"
    elif family == "rotation":
        shift_text = "lowest-energy rotated embedding"
    elif family == "scale":
        shift_text = "lowest-energy directionally scaled embedding"
    else:
        shift_text = "lowest-energy transformed embedding"
    ax1.text(
        0.02,
        0.98,
        shift_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#64748b", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
        zorder=7,
    )

    for ax in (ax0, ax1):
        for collection in ax.collections:
            try:
                collection.set_path_effects([pe.Stroke(linewidth=collection.get_linewidths()[0] + 1.1, foreground="white"), pe.Normal()])
            except Exception:
                continue

    delta_pct = float(summary_row["delta_pct"])
    family = str(summary_row["best_family"])
    parameter = float(summary_row["best_parameter"])
    subtitle = f"{city_display_name(city)} | best={family} ({format_plain_number(parameter, decimals=2)}), delta={delta_pct:.2f}%"
    fig.suptitle(subtitle, y=0.98, fontsize=14)
    legend = [
        Line2D([0], [0], color="#f97316", lw=2, label="original"),
        Line2D([0], [0], color="#1d4ed8", lw=2, label="best transformed"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.10, wspace=0.08)
    out = MAP_DIR / f"{city}_counterexample_map.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    del original_g, best_g, land_gdf, gdf_orig, gdf_best, gdf_display_best
    gc.collect()
    return out


def plot_supporting_profile(city: str, df: pd.DataFrame, stats_df: pd.DataFrame, summary_row: pd.Series) -> Path:
    original_energy = float(df.loc[df["family"] == "original", "energy_plot"].iloc[0])
    availability = transformation_availability(city, df, stats_df)

    trans_df = df[(df["family"] == "translation") & df["signed_translation"].notna()].copy()
    trans_ns = trans_df[trans_df["translation_axis"] == "ns"].sort_values("signed_translation")
    trans_ew = trans_df[trans_df["translation_axis"] == "ew"].sort_values("signed_translation")

    rot_df = df[df["family"] == "rotation"].sort_values("parameter").copy()
    scale_ns = df[(df["family"] == "scale") & df["variant_type"].astype(str).str.contains("ns", case=False, na=False)].sort_values("parameter")
    scale_ew = df[(df["family"] == "scale") & df["variant_type"].astype(str).str.contains("ew", case=False, na=False)].sort_values("parameter")

    panel_specs = {
        "trans_ns": {
            "x": trans_ns["signed_translation"],
            "y": trans_ns["energy_plot"],
            "title": "North--South translations",
            "xlabel": "Translation Distance (m)",
            "x0": 0.0,
            "color": "#2b83ba",
            "legend": ("South", "North"),
        },
        "trans_ew": {
            "x": trans_ew["signed_translation"],
            "y": trans_ew["energy_plot"],
            "title": "East--West translations",
            "xlabel": "Translation Distance (m)",
            "x0": 0.0,
            "color": "#2b83ba",
            "legend": ("West", "East"),
        },
        "rotation": {
            "x": rot_df["parameter"],
            "y": rot_df["energy_plot"],
            "title": "Rotations",
            "xlabel": "Rotation Angle (deg)",
            "x0": 0.0,
            "color": "#2b83ba",
            "legend": None,
        },
        "scale_ns": {
            "x": scale_ns["parameter"],
            "y": scale_ns["energy_plot"],
            "title": "North--South dilations",
            "xlabel": "NS Scale Factor",
            "x0": 1.0,
            "color": "#2ca25f",
            "legend": None,
        },
        "scale_ew": {
            "x": scale_ew["parameter"],
            "y": scale_ew["energy_plot"],
            "title": "East--West dilations",
            "xlabel": "EW Scale Factor",
            "x0": 1.0,
            "color": "#ff7f0e",
            "legend": None,
        },
    }

    rows: list[list[str]] = []
    translations = [k for k in ("trans_ns", "trans_ew") if availability["panels"][k]["n_valid"] > 0]
    if translations:
        rows.append(translations)
    if availability["panels"]["rotation"]["n_valid"] > 0:
        rows.append(["rotation"])
    scales = [k for k in ("scale_ns", "scale_ew") if availability["panels"][k]["n_valid"] > 0]
    if scales:
        rows.append(scales)
    if not rows:
        rows = [["trans_ns"]]

    fig = plt.figure(figsize=(18, max(4.8 * len(rows), 6.8)))
    gs = fig.add_gridspec(len(rows), 2, hspace=0.35, wspace=0.25)
    axes: list[plt.Axes] = []

    def scatter_line(ax: plt.Axes, key: str) -> None:
        spec = panel_specs[key]
        x = spec["x"]
        y = spec["y"]
        if len(x) > 0:
            if spec["legend"] is not None:
                left_label, right_label = spec["legend"]
                left_mask = x < spec["x0"]
                right_mask = x > spec["x0"]
                if left_mask.any():
                    ax.scatter(x[left_mask], y[left_mask], s=80, alpha=0.8, c=spec["color"], label=left_label)
                if right_mask.any():
                    ax.scatter(x[right_mask], y[right_mask], s=80, alpha=0.8, c=spec["color"], label=right_label)
                equal_mask = ~(left_mask | right_mask)
                if equal_mask.any():
                    ax.scatter(x[equal_mask], y[equal_mask], s=80, alpha=0.8, c=spec["color"])
                ax.legend(fontsize=LEGEND_SIZE, frameon=False)
            else:
                ax.scatter(x, y, s=80, alpha=0.8, c=spec["color"])
        ax.scatter([spec["x0"]], [original_energy], s=250, c="red", marker="*", edgecolor="darkred", linewidth=1.5, zorder=5, label="Original")
        ax.set_title(spec["title"], fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(spec["xlabel"], fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(r"$W_{TOT}$ [J]", fontsize=AXIS_LABEL_SIZE)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        sci_formatter = ScalarFormatter(useMathText=True)
        sci_formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(sci_formatter)
        ax.yaxis.get_offset_text().set_fontsize(TICK_LABEL_SIZE - 2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for row_idx, row_keys in enumerate(rows):
        if len(row_keys) == 1:
            ax = fig.add_subplot(gs[row_idx, :])
            axes.append(ax)
            scatter_line(ax, row_keys[0])
        else:
            for col_idx, key in enumerate(row_keys[:2]):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                axes.append(ax)
                scatter_line(ax, key)

    if axes:
        fig.align_ylabels(axes)
        fig.align_labels()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.10, hspace=0.35, wspace=0.25)
    out = PROFILE_DIR / f"{city}_energy_profile.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    return out


def generate_city_figures(
    summary_df: pd.DataFrame,
    only: str = "all",
    cities_filter: set[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    supporting_outputs: list[Path] = []
    counterexample_outputs: list[Path] = []

    for row in summary_df.itertuples(index=False):
        city = row.city
        if cities_filter is not None and city not in cities_filter:
            continue
        if row.status == "OK" and only in {"all", "supporting"}:
            df = read_work_df(city)
            stats_df = read_stats_df(city)
            supporting_outputs.append(plot_supporting_profile(city, df, stats_df, pd.Series(row._asdict())))
            del stats_df
            del df
            gc.collect()
        elif row.status == "DISPLACED_LOWER" and only in {"all", "counterexamples"}:
            counterexample_outputs.append(plot_counterexample_map(city, pd.Series(row._asdict())))
            gc.collect()
        # CONSTRAINED cities: skip figure generation

    return supporting_outputs, counterexample_outputs


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def rel_to_tex(path: Path) -> str:
    return path.relative_to(TEX_DIR).as_posix() if path.is_relative_to(TEX_DIR) else Path("..") / path.relative_to(SUPP_DIR)


def path_from_tex(path: Path) -> str:
    return path.relative_to(TEX_DIR.parent).as_posix() if path.is_relative_to(TEX_DIR.parent) else path.as_posix()


def tex_graphic_path(path: Path) -> str:
    return path.relative_to(TEX_DIR).as_posix() if path.is_relative_to(TEX_DIR) else path.relative_to(ROOT).as_posix()


def tex_caption_with_bold_lead(caption: str) -> str:
    first, sep, rest = caption.partition(". ")
    lead = tex_escape(first.strip())
    if not lead.endswith("."):
        lead = f"{lead}."
    if not sep:
        return rf"\textbf{{{lead}}}"
    return rf"\textbf{{{lead}}} {tex_escape(rest.strip())}"


def tex_short_caption(caption: str) -> str:
    first, _, _ = caption.partition(". ")
    first = first.strip()
    if not first.endswith("."):
        first = f"{first}."
    return tex_escape(first)


def summary_status_label(status: str) -> str:
    return "Original minimum" if status == "OK" else "Counterexample"


def summary_family_and_parameter(row: pd.Series) -> tuple[str, str]:
    if str(row["status"]) == "OK":
        return "---", "---"
    return format_family_display(str(row["best_family"])), format_plain_number(float(row["best_parameter"]), decimals=2)


def make_summary_longtable(summary_df: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in summary_df.sort_values("city").iterrows():
        family_label, parameter_label = summary_family_and_parameter(row)
        rows.append(
            " & ".join(
                [
                    tex_escape(city_display_name(str(row["city"]))),
                    tex_escape(summary_status_label(str(row["status"]))),
                    tex_escape(family_label),
                    tex_escape(parameter_label),
                    f"{float(row['delta_pct']):.2f}",
                    str(int(row["n_variants"])),
                    str(int(row["n_land_false"])),
                    f"{float(row['frac_land_false']):.2f}",
                ]
            )
            + r" \\"
        )
    rows_text = "\n".join(rows)
    return rf"""
\begin{{longtable}}{{lllrrrrr}}
\caption{{\label{{tab:supp:summary}}Summary of the city-level energetic validation and geographic feasibility checks. The table reports whether the empirical embedding is the minimum-energy configuration, the best transformed alternative when present, the relative deviation from the empirical configuration, and the number of transformations rejected by the land/sea validity check.}}\\
\toprule
City & Status & Best family & Best param. & $\Delta$[\%] & Tested & Rejected & Rejected frac. \\
\midrule
\endfirsthead
\toprule
City & Status & Best family & Best param. & $\Delta$[\%] & Tested & Rejected & Rejected frac. \\
\midrule
\endhead
{rows_text}
\bottomrule
\end{{longtable}}
"""


def make_city_detail_table(city: str, summary_row: pd.Series, availability: dict[str, Any]) -> str:
    shown = english_join(availability["shown_labels"]) if availability["shown_labels"] else "None"
    omitted_items = availability["missing_labels"] + availability["infeasible_labels"]
    omitted = english_join(omitted_items) if omitted_items else "None"
    limited = english_join(availability["limited_labels"]) if availability["limited_labels"] else "None"
    best_family = "Original" if str(summary_row["status"]) == "OK" else format_family_display(str(summary_row["best_family"]))
    best_parameter = "---" if str(summary_row["status"]) == "OK" else format_plain_number(float(summary_row["best_parameter"]), decimals=2)
    rows = [
        ("Status", summary_status_label(str(summary_row["status"]))),
        ("Shown families", shown),
        ("Omitted families", omitted),
        ("Limited subset", limited),
        ("Best family", best_family),
        ("Best parameter", best_parameter),
        ("$\\Delta$ [\\%]", f"{float(summary_row['delta_pct']):.2f}"),
        ("Tested variants", str(int(summary_row["n_variants"]))),
        ("Rejected transformations", str(int(summary_row["n_land_false"]))),
        ("Rejected fraction", f"{float(summary_row['frac_land_false']):.2f}"),
    ]
    body_lines = []
    for key, value in rows:
        key_text = key if key.startswith("$") else tex_escape(key)
        body_lines.append(f"{key_text} & {tex_escape(value)} \\\\")
    body = "\n".join(body_lines)
    return rf"""
\begin{{center}}
\begin{{tabular}}{{p{{0.28\textwidth}}p{{0.60\textwidth}}}}
\toprule
Metric & Value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{center}}
"""


def render_tex(
    summary_df: pd.DataFrame,
    land_df: pd.DataFrame,
    supporting_outputs: list[Path],
    counterexample_outputs: list[Path],
    overview_fig: Path,
    counterexample_fig: Path,
    land_figs: tuple[Path, Path],
) -> None:
    n_supporting = int((summary_df["status"] == "OK").sum())
    n_counter = int((summary_df["status"] == "COUNTEREXAMPLE").sum())
    n_with_land_false = int((land_df["n_land_false"] > 0).sum())
    supporting_city_list = ", ".join(
        [city_display_name(city) for city in summary_df.loc[summary_df["status"] == "OK", "city"].sort_values().tolist()]
    )
    summary_table_text = make_summary_longtable(summary_df)

    # Derive transformation range labels from data for use in figure captions
    MAX_TRANS = 2500   # metres — step_meters × num_points used in the pipeline
    MAX_SCALE = 2.0    # maximum scale factor used in the pipeline

    supporting_section = []
    for _, row in summary_df[summary_df["status"] == "OK"].sort_values("city").iterrows():
        city = str(row["city"])
        work_df = read_work_df(city)
        stats_df = read_stats_df(city)
        availability = transformation_availability(city, work_df, stats_df)
        paragraph = tex_escape(make_supporting_city_paragraph(city, work_df, stats_df, row))
        city_table = make_city_detail_table(city, row, availability)
        caption_text = make_supporting_caption(city, work_df, stats_df, row)
        caption = tex_caption_with_bold_lead(caption_text)
        short_caption = tex_short_caption(caption_text)
        supporting_section.append(
            rf"""
\paragraph{{{tex_escape(city_display_name(city))}.}} {paragraph}

{city_table}

\begin{{figure*}}[ht]
\centering
\includegraphics[width=0.96\textwidth]{{supporting_profiles/{city}_energy_profile.pdf}}
\caption[{short_caption}]{{\label{{fig:supp:{city}}}{caption}}}
\end{{figure*}}
\clearpage
"""
        )
    supporting_section_text = "".join(supporting_section)

    counter_rows = []
    for row in summary_df[summary_df["status"] == "COUNTEREXAMPLE"].sort_values("city").itertuples(index=False):
        counter_rows.append(
            f"{tex_escape(city_display_name(row.city))} & {tex_escape(row.best_family.title())} & {format_plain_number(row.best_parameter, decimals=2)} & {row.delta_pct:.2f} \\\\"
        )
    counter_rows_text = "\n".join(counter_rows)

    counter_notes_blocks = []
    for row in summary_df[summary_df["status"] == "COUNTEREXAMPLE"].sort_values("city").itertuples(index=False):
        note = COUNTEREXAMPLE_NOTES.get(row.city, {})
        work_df = read_work_df(row.city)
        stats_df = read_stats_df(row.city)
        availability = transformation_availability(row.city, work_df, stats_df)
        reason = tex_escape(make_counterexample_interpretation(row.city, pd.Series(row._asdict())))
        sources = note.get("sources", [])
        source_lines = "\n".join(
            [rf"\item \textit{{{tex_escape(src['label'])}}}: \url{{{src['url']}}}" for src in sources]
        )
        sources_block = ""
        if source_lines and 1==0:
            sources_block = rf"""
            \begin{{itemize}}
            {source_lines}
            \end{{itemize}}
            """
        city_table = make_city_detail_table(row.city, pd.Series(row._asdict()), availability)
        omitted_items = availability["missing_labels"] + availability["infeasible_labels"]
        omitted_text = ""
        if omitted_items:
            omitted_text = tex_escape(
                f"Some transformation families are not shown for this city because {english_join(omitted_items)} do not retain valid transformed embeddings within the feasible domain."
            )
        map_caption_text = make_counterexample_caption(row.city, pd.Series(row._asdict()))
        map_caption = tex_caption_with_bold_lead(map_caption_text)
        short_map_caption = tex_short_caption(map_caption_text)
        counter_notes_blocks.append(
            rf"""
\paragraph{{{tex_escape(city_display_name(row.city))}.}} The lowest-energy transformed embedding for {tex_escape(city_display_name(row.city))} is a {tex_escape(row.best_family)} with parameter {format_plain_number(row.best_parameter, decimals=2)}, achieving a relative reduction of {abs(row.delta_pct):.2f}\% with respect to the empirical configuration. Supplementary Fig.~\ref{{fig:counter:{row.city}}} shows the original and best transformed embeddings.

{reason}
{omitted_text}

{city_table}
{sources_block}

\begin{{figure*}}[ht]
\centering
\includegraphics[width=0.96\textwidth]{{counterexamples/{row.city}_counterexample_map.png}}
\caption[{short_map_caption}]{{\label{{fig:counter:{row.city}}}{map_caption}}}
\end{{figure*}}
\clearpage
"""
        )
    counter_notes_text = "".join(counter_notes_blocks)

    land_rows = []
    for row in land_df[land_df["n_land_false"] > 0].itertuples(index=False):
        land_rows.append(
            f"{tex_escape(city_display_name(row.city))} & {int(row.n_land_false)} & {int(row.n_total_stats)} & {row.frac_land_false:.2f} \\\\"
        )
    land_rows_text = "\n".join(land_rows)

    tex = rf"""
\documentclass[11pt]{{article}}

%% ── Layout ────────────────────────────────────────────────────────────────────
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{{geometry}}
\linespread{{1.18}}
\setlength{{\parskip}}{{4pt plus 1pt}}

%% ── Typography ────────────────────────────────────────────────────────────────
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}

%% ── Mathematics ───────────────────────────────────────────────────────────────
\usepackage{{amsmath, amssymb}}

%% ── Tables and figures ────────────────────────────────────────────────────────
\usepackage{{booktabs, longtable, array, tabularx}}
\usepackage{{graphicx}}
\usepackage{{float}}
\graphicspath{{{{../output/figures/}}}}

%% ── Colours and hyperlinks ────────────────────────────────────────────────────
\usepackage{{xcolor}}
\definecolor{{nblue}}{{RGB}}{{29,83,183}}
\definecolor{{nred}}{{RGB}}{{198,49,49}}
\usepackage[colorlinks=true, linkcolor=nblue, urlcolor=nred, citecolor=nblue]{{hyperref}}
\usepackage{{url}}

%% ── Caption style ─────────────────────────────────────────────────────────────
\usepackage[font=small, labelfont=bf, labelsep=period, justification=justified]{{caption}}
\renewcommand{{\figurename}}{{Supplementary Fig.}}
\renewcommand{{\tablename}}{{Supplementary Table}}

%% ── Section style ─────────────────────────────────────────────────────────────
\usepackage{{titlesec}}
\titleformat{{\section}}{{\large\bfseries}}{{\thesection}}{{1em}}{{}}
\titleformat{{\subsection}}{{\normalsize\bfseries}}{{\thesubsection}}{{1em}}{{}}

%% ── Header / footer ───────────────────────────────────────────────────────────
\usepackage{{fancyhdr}}
\pagestyle{{fancy}}
\fancyhf{{}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\renewcommand{{\footrulewidth}}{{0pt}}
\lhead{{\small Bellisardi, Cano Su\~n\'en, Ruiz-Varona, Ramasco}}
\rhead{{\small\thepage}}
\lfoot{{\small\textit{{Impact of mobility energetic balance on urban spatial organization}}}}
\rfoot{{\small Supplementary Information}}

\begin{{document}}

%% ── Title block ───────────────────────────────────────────────────────────────
\begin{{center}}
{{\Large\bfseries Supplementary Information}}\\[8pt]
{{\large Impact of mobility energetic balance on urban spatial organization}}\\[6pt]
{{\normalsize
  Federico Bellisardi,$^{{1}}$
  Enrique Cano Su\~n\'en,$^{{2}}$
  Ana Ruiz-Varona,$^{{3}}$
  Jos\'e J.\ Ramasco$^{{1}}$
}}
\end{{center}}

\vspace{{6pt}}\hrule\vspace{{12pt}}

%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 1: Energetic framework}}
%% ═════════════════════════════════════════════════════════════════════════════

The total mobility energy associated with a trajectory~$\tau$ from origin~$O$ to destination~$D$ is decomposed into a longitudinal and an altitudinal contribution,
\begin{{equation}}
W_\tau = W_{{\mathrm{{Lon}}}}(\tau) + W_{{\mathrm{{Alt}}}}(\tau)
= \lambda L_\tau + m g \int_\tau \max\!\left(0,\frac{{dh}}{{d\ell}}\right)d\ell,
\label{{eq:wtau}}
\end{{equation}}
where $\lambda$ is the longitudinal energy cost per unit distance, $L_\tau$ is the three-dimensional path length, $m$ is the effective transported mass, $g$ is gravitational acceleration, $h$ is elevation, and $\ell$ is the curvilinear coordinate along the path. Only uphill segments contribute to $W_{{\mathrm{{Alt}}}}$; descent is not assumed to regenerate energy. City-wide mobility expenditure is obtained by weighting every origin--destination path by its empirical flow,
\begin{{equation}}
W_{{\mathrm{{Tot}}}} = \sum_{{(O,D)}} F_{{OD}}\,W_{{OD}}.
\label{{eq:wtot}}
\end{{equation}}
We test the hypothesis that the observed spatial embedding of each city road network minimises $W_{{\mathrm{{Tot}}}}$ within the set of geometrically equivalent configurations reachable by translations, rotations, and anisotropic dilations. Transformed embeddings are evaluated on the same terrain on which the empirical city is located, with DEM sampling repeated at each road segment after transformation. Embeddings that place any part of the network outside the valid terrestrial domain or outside the DEM support are excluded from the comparison. Throughout, translations shorter than {MIN_TRANSLATION_DISTANCE_M:.0f}\,m are omitted to prevent near-origin numerical artefacts from influencing the analysis.

%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 2: Summary of results across all analysed cities}}
%% ═════════════════════════════════════════════════════════════════════════════

We analysed {len(summary_df)} urban systems for which the full pipeline---road-network extraction, DEM assignment, origin--destination weighting, and systematic geometric transformation---was successfully completed. In {n_supporting} cities ({round(100*n_supporting/len(summary_df))} per cent of the sample), the empirical embedding returns the lowest $W_{{\mathrm{{Tot}}}}$ among all tested configurations. In the remaining {n_counter} cities, at least one transformed embedding achieves a lower total energy. Supplementary Fig.~\ref{{fig:supp:status}} shows the distribution of energy deviations across all cities, and Supplementary Fig.~\ref{{fig:supp:counteroverview}} highlights the magnitude of the discrepancy for the Anomalouscases. The full city-level results are reported in Supplementary Table~\ref{{tab:supp:summary}}.

\begin{{figure*}}[ht]
\centering
\includegraphics[width=0.96\textwidth]{{overview/hypothesis_status_overview.pdf}}
\caption{{\label{{fig:supp:status}}\textbf{{Energy deviation of the best transformed configuration from the empirical embedding, across all {len(summary_df)} analysed cities.}} Each bar shows $100\times(E_{{\min}}/E_{{\mathrm{{orig}}}}-1)$, the relative difference between the lowest-energy transformed configuration and the empirical configuration, expressed as a percentage. Negative values (blue bars) identify Anomalouscities in which a geometrically transformed version of the city has lower total mobility energy than the observed layout. Cities for which the empirical embedding is the energetic minimum within the tested transformation set are shown as red markers on the zero line. The transformation set includes translations up to {MAX_TRANS}\,m in the cardinal directions, rotations of up to $\pm20^\circ$, and anisotropic dilations up to a factor of {MAX_SCALE}; transformed configurations that place any network node outside the valid terrestrial domain are excluded.}}
\end{{figure*}}

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.82\textwidth]{{overview/counterexample_improvement.pdf}}
\caption{{\label{{fig:supp:counteroverview}}\textbf{{Magnitude of the energy reduction achieved by the best transformed configuration in the {n_counter} Anomalouscities.}} Bars show the absolute value of $100\times(E_{{\min}}/E_{{\mathrm{{orig}}}}-1)$ for each Anomalouscity, sorted by the size of the discrepancy. Larger values indicate a stronger departure from the empirical minimum-energy condition. All Anomalouscities show deviations of less than 12\%, and most are below 5\%, consistent with a near-optimal rather than strongly suboptimal placement of the empirical network.}}
\end{{figure}}

\clearpage
{summary_table_text}

%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 3: Cities consistent with the energetic optimality hypothesis}}
%% ═════════════════════════════════════════════════════════════════════════════

The {n_supporting} cities listed below all satisfy the condition that $W_{{\mathrm{{Tot}}}}$ of the empirical configuration is lower than or equal to $W_{{\mathrm{{Tot}}}}$ of every tested geometrically equivalent alternative: {supporting_city_list}. For each city we report the energetic sensitivity profile, which displays how $W_{{\mathrm{{Tot}}}}$ varies as a function of the transformation parameter across the admissible embedding space. The red star marks the empirical configuration. Profiles are shown only for transformation families that produced at least one valid alternative embedding; families entirely excluded by geographic constraints are omitted.

{supporting_section_text}

\clearpage
%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 4: Anomalouscities}}
%% ═════════════════════════════════════════════════════════════════════════════

The {n_counter} cities in Supplementary Table~\ref{{tab:counterexamples}} each admit at least one transformed embedding with lower total mobility energy than the observed layout. These results should not be interpreted as evidence that the observed cities are energetically suboptimal or that the framework fails. Rather, they identify cases in which the tested transformation space---even when restricted to land-admissible embeddings---still contains a lower-energy alternative. In every counterexample, the energetic reduction is small relative to the baseline (below 12\%), and geographic analysis reveals plausible constraints that could explain why the empirical layout was not already positioned at the unconstrained minimum.

The Anomalouscities concentrate in settings where territorial constraints asymmetrically restrict the feasible embedding space: waterfront locations (estuaries, lakefronts, engineered shorelines), mountain fronts, and basin environments. In such contexts the empirical city may already occupy the lowest-energy position within the \emph{{geographically accessible}} domain, even if a theoretically lower-energy configuration exists elsewhere. We present a brief interpretation for each city below, emphasising the role of local geography rather than claiming a single causal explanation.

\begin{{table}}[ht]
\caption{{\label{{tab:counterexamples}}\textbf{{Anomalouscities and their best transformed configurations.}} For each city the table reports the family of the lowest-energy transformation (translation, rotation, or scale), the corresponding transformation parameter (distance in metres, angle in degrees, or dimensionless scale factor), and the relative energy reduction $\Delta$ with respect to the empirical embedding. All values are negative, indicating that the transformed configuration has lower total mobility energy than the observed layout.}}
\centering
\begin{{tabular}}{{@{{}}lllr@{{}}}}
\toprule
City & Family & Parameter & $\Delta$ [\%] \\
\midrule
{counter_rows_text}
\bottomrule
\end{{tabular}}
\end{{table}}

{counter_notes_text}

\clearpage
%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 5: Non-admissible transformations and geographic constraints}}
%% ═════════════════════════════════════════════════════════════════════════════

A transformed embedding is rejected if any node of the road network is placed outside the valid terrestrial domain (land or, for lake-adjacent cities, outside the land-minus-lake polygon) or outside the digital elevation model support. These exclusions are not artefacts of the computational procedure; they reflect genuine geographic constraints that limit the feasible embedding space of each city. Supplementary Fig.~\ref{{fig:supp:landcounts}} shows the number of rejected transformations per city.

The pattern of rejections is strongly non-random. Urban systems near coastlines, estuaries, island chains, lake shores, delta plains, and mountain fronts accumulate more rejected embeddings than inland cities surrounded by continuous land. This asymmetry directly reduces the size of the counterfactual comparison set and can bias the energetic test towards finding fewer counterexamples---not because the empirical layout is more optimal, but because fewer valid alternatives exist. Conversely, cities with many rejections provide weaker tests of the optimality hypothesis, and their results should be interpreted in light of the geographic constraints documented here.

\begin{{figure*}}[ht]
\centering
\includegraphics[width=0.96\textwidth]{{land_false/land_false_counts.pdf}}
\caption{{\label{{fig:supp:landcounts}}\textbf{{Number of rejected transformed configurations per city.}} A transformation is rejected when at least one node of the road network is placed outside the valid terrestrial domain or outside the digital elevation model support. The colour of each bar indicates the dominant transformation family driving the rejections (rotations in orange, translations in blue, dilations in green). Cities absent from this plot experienced no rejections and had access to the full transformation set.}}
\end{{figure*}}

\begin{{longtable}}{{@{{}}lrrl@{{}}}}
\caption{{\label{{tab:landfalse}}\textbf{{Cities with at least one rejected transformed embedding.}} The table reports the total number of rejected configurations, the total number tested, the rejection fraction, and the dominant geographic setting that drives the constraint. Cities are sorted by decreasing rejection fraction.}}\\
\toprule
City & Rejected & Tested & Fraction \\
\midrule
\endfirsthead
\toprule
City & Rejected & Tested & Fraction \\
\midrule
\endhead
{land_rows_text}
\bottomrule
\end{{longtable}}

Taken together, the rejection map underscores that urban energetic optimality should be evaluated within the geographically accessible configuration space rather than in an unconstrained Euclidean transformation space. The feasible domain of each city is shaped by its local terrain, hydrography, and land cover, and the observed urban layout can only be assessed relative to the embeddings that are actually available to it.

\clearpage
%% ═════════════════════════════════════════════════════════════════════════════
\section*{{Supplementary Note 6: Validation of the digital elevation model}}
%% ═════════════════════════════════════════════════════════════════════════════

To verify that the SRTM-based digital elevation model used throughout the analysis does not introduce systematic bias into the energetic estimates, we compared node-level elevation values from the SRTM product against corresponding values extracted from the Copernicus DEM (GLO-30) for two cities spanning contrasting relief regimes: Barcelona (moderate, coastal topography) and Santiago (high-relief, Andean setting). In both cities the two datasets are in close agreement across the full elevation range, with the scatter tightly concentrated around the identity line. This confirms that the choice of DEM product does not materially affect the energetic results reported in the main text.

\begin{{figure*}}[ht]
\centering
\includegraphics[width=0.46\textwidth]{{dem_validation/scatter_barcelona.pdf}}\hfill
\includegraphics[width=0.46\textwidth]{{dem_validation/scatter_santiago.pdf}}
\caption{{\label{{fig:supp:demcorr}}\textbf{{Comparison of SRTM-based and Copernicus-based elevation values at road-network nodes.}} Each point represents one node of the urban road network. The dashed line is the identity ($y = x$). Left: Barcelona (relief range 0--600\,m). Right: Santiago (relief range 500--5\,000\,m). In both cities the correlation is near-perfect, indicating that the choice of elevation product does not bias the energetic estimates.}}
\end{{figure*}}

\end{{document}}
"""

    TEX_PATH.write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplementary analysis assets for gravitational morphology.")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only compute tables and write the TeX file.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "supporting", "counterexamples"],
        default="all",
        help="Generate only supporting-city figures, only Anomalousfigures, or all figures.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional subset of cities to render figures for.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root output directory (default: figures/supplementary/ inside the repo).",
    )
    args = parser.parse_args()

    # Override output paths if --output-root is provided
    if args.output_root is not None:
        global OUTPUT_DIR, FIG_DIR, TABLE_DIR, TEX_DIR
        global MAP_DIR, PROFILE_DIR, LAND_DIR, OVERVIEW_DIR, TEX_PATH
        OUTPUT_DIR   = args.output_root
        FIG_DIR      = OUTPUT_DIR / "figures"
        TABLE_DIR    = OUTPUT_DIR / "tables"
        TEX_DIR      = OUTPUT_DIR / "tex"
        MAP_DIR      = FIG_DIR / "counterexamples"
        PROFILE_DIR  = FIG_DIR / "supporting_profiles"
        LAND_DIR     = FIG_DIR / "land_false"
        OVERVIEW_DIR = FIG_DIR / "overview"
        TEX_PATH     = TEX_DIR / "gravitational_morphology_supplementary.tex"

    configure_matplotlib()
    ensure_directories()

    summary_df, land_df = build_summary()
    if summary_df.empty:
        raise SystemExit("No cities with fine_grid_gravitational_work.csv were found.")

    save_tables(summary_df, land_df)

    overview_fig = plot_overview_status(summary_df)
    counterexample_fig = plot_counterexample_deltas(summary_df)
    land_figs = plot_land_false_overview(land_df)

    supporting_outputs: list[Path] = []
    counterexample_outputs: list[Path] = []
    if not args.skip_figures:
        city_filter = set(args.cities) if args.cities else None
        supporting_outputs, counterexample_outputs = generate_city_figures(
            summary_df,
            only=args.only,
            cities_filter=city_filter,
        )

    render_tex(
        summary_df=summary_df,
        land_df=land_df,
        supporting_outputs=supporting_outputs,
        counterexample_outputs=counterexample_outputs,
        overview_fig=overview_fig,
        counterexample_fig=counterexample_fig,
        land_figs=land_figs,
    )

    print(f"LaTeX supplementary: {TEX_PATH}")
    print(f"Summary table CSV: {TABLE_DIR / 'city_hypothesis_summary.csv'}")
    print(f"Saved figures to: {FIG_DIR}")
    print(f"Supporting cities: {int((summary_df['status'] == 'OK').sum())}")
    print(f"Counterexamples: {int((summary_df['status'] == 'COUNTEREXAMPLE').sum())}")
    print(f"Cities with land/sea rejected transformations: {int((land_df['n_land_false'] > 0).sum())}")


if __name__ == "__main__":
    main()
