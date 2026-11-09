"""The city classification of the paper (Supplementary Table 1) is reproduced from the result tables.

The SI builder (supplementary/python/build_gravitational_supplementary.py) classifies every city from its step-2
result table with the study-wide rules of python/study_config.py (excluded cities, tie tolerance, transformation
ensemble, 3DEP for Chicago and Dallas). This test checks that the result is the one reported in the paper:
26 cities where the observed configuration is the lowest-energy one, 4 with a lower-energy displaced configuration
(with their transformation and energy change) and 26 constrained cities.

The result tables are not distributed with the code (available from the authors on request); without them the
test is skipped.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "supplementary" / "python"))

import study_config as sc  # noqa: E402

ORIGINAL = {
    "bangkok", "barcelone", "berlin", "caracas", "chicago", "dallas", "djakarta", "guadalajara", "kualalumpur",
    "lima", "londres", "madrid", "manchester", "manille", "mexico", "milan", "montreal", "moscou", "paris",
    "phoenix", "rome", "saintpetersbourg", "santiago", "santodomingo", "saopaulo", "singapour",
}
CONSTRAINED = {
    "amsterdam", "atlanta", "boston", "detroitwindsor", "dublin", "hongkong", "houston", "istanbul", "lisbon",
    "losangeles", "miami", "nagoya", "newyork", "osaka", "philadelphie", "riodejaneiro", "sandiego",
    "sanfrancisco", "seoul", "stockholm", "sydney", "taipei", "tokyo", "toronto", "vancouver", "washington",
}
# city: (family, direction as azimuth from east, displacement in m, energy change in %)
DISPLACED_LOWER = {
    "bogota": ("translation", "180", 1250.0, -4.24),        # west
    "buenosaires": ("translation", "180", 2500.0, -1.60),   # west
    "bandung": ("translation", "270", 2250.0, -0.99),       # south
    "bruxelles": ("translation", "090", 2250.0, -0.45),     # north
}


def _tables_available() -> bool:
    return all(sc.step2_table(c).exists() for c in ORIGINAL | CONSTRAINED | set(DISPLACED_LOWER))


pytestmark = pytest.mark.skipif(
    not _tables_available(),
    reason="result tables not available (they are not distributed with the code; available from the authors)")


@pytest.fixture(scope="module")
def summary():
    import build_gravitational_supplementary as bgs
    s, _ = bgs.build_summary()
    return s.set_index("city")


def test_study_has_56_cities(summary):
    assert len(summary) == 56
    assert not set(summary.index) & set(sc.EXCLUDED_CITIES)


def test_original_cities(summary):
    assert set(summary.index[summary["status"] == "OK"]) == ORIGINAL


def test_constrained_cities(summary):
    assert set(summary.index[summary["status"] == "CONSTRAINED"]) == CONSTRAINED


def test_displaced_lower_cities_and_values(summary):
    lower = summary[summary["status"] == "DISPLACED_LOWER"]
    assert set(lower.index) == set(DISPLACED_LOWER)
    for city, (family, direction, distance, delta) in DISPLACED_LOWER.items():
        row = lower.loc[city]
        assert row["best_family"] == family
        assert row["best_variant"] == f"{family}_{direction}"
        assert row["best_parameter"] == pytest.approx(distance)
        assert round(float(row["delta_pct"]), 2) == pytest.approx(delta)


def test_alternative_dem_used_for_chicago_and_dallas():
    for city in ("chicago", "dallas"):
        assert sc.dem_source(city) == "3dep13"
