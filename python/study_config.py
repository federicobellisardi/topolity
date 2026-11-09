"""Single source of the study-wide choices used by every classification and SI script.

Change a rule here and every script that counts or classifies cities (vehicle_ratio_sweep, lower_energy,
net_rise, distance_invariance_audit, exception_analysis, the SI builder) follows it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "data_processed"
GRID_DIR = "graphs_fine_grid"
STEP2_TABLE = "fine_grid_gravitational_work.csv"
ORIGINAL = "graph_original.pkl"

# Removed from the study: geolocated Twitter/X activity in mainland China is dominated by foreign visitors.
EXCLUDED_CITIES = frozenset({"pekin", "shanghai"})

# Main transformation ensemble and tie rule (Methods; SI Notes 1-3).
MIN_TRANSLATION_M = 200.0              # translations shorter than this are not part of the ensemble
TIE_TOL = 0.0005                       # |Delta| < 0.05 % counts as the observed embedding being the minimum
RTOL = TIE_TOL / (1 - TIE_TOL)         # same rule written as E_0 <= (1 + RTOL) E_c

# Cities classified as constrained by hand, with the reason given in the SI. Empty since 2026-11-06: Chicago
# (formerly here) is evaluated with 3DEP and has 18 admissible translations, so it is classified by the data.
CONSTRAINED_OVERRIDE = frozenset()

# Cities whose step 2 uses the USGS 3DEP bare-earth DEM instead of SRTM: SRTM records building heights in
# their downtowns (Chicago Loop 240-256 m vs 181 m ground; downtown Dallas 170 m vs 131 m). Computed with
# python/step2_alt_dem.py; the 3DEP table replaces the SRTM one as soon as its merge exists.
ALT_DEM_CITIES = {"chicago": "3dep13", "dallas": "3dep13"}


def alt_dem_table(city: str) -> Path | None:
    tag = ALT_DEM_CITIES.get(city)
    if tag is None:
        return None
    return ROOT / "output" / "step2_alt_dem" / city / tag / f"fine_grid_gravitational_work_{tag}.csv"


def step2_table(city: str, grid_dir: str = GRID_DIR) -> Path:
    """Step-2 table of a city for the paper: the 3DEP table for ALT_DEM_CITIES once available, else SRTM.

    Other grid directories (e.g. graphs_fine_grid_v1) always return their own table.
    """
    if grid_dir == GRID_DIR:
        alt = alt_dem_table(city)
        if alt is not None and alt.exists():
            return alt
    return DATA_ROOT / city / grid_dir / STEP2_TABLE


def dem_source(city: str) -> str:
    alt = alt_dem_table(city)
    return ALT_DEM_CITIES[city] if alt is not None and alt.exists() else "srtm"


def study_cities(grid_dir: str = GRID_DIR, exclude=()) -> list[str]:
    """Cities of the study with a step-2 table, excluding EXCLUDED_CITIES and `exclude`."""
    skip = set(EXCLUDED_CITIES) | set(exclude)
    return sorted(p.name for p in DATA_ROOT.iterdir()
                  if p.is_dir() and p.name not in skip and step2_table(p.name, grid_dir).exists())
