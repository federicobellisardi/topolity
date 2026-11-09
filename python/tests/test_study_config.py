"""Every classification script must take its study-wide rules from python/study_config.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "supplementary" / "python"))

import study_config as sc  # noqa: E402


def test_rules_are_shared_objects():
    import build_gravitational_supplementary as bgs
    import vehicle_ratio_sweep as vrs
    assert vrs.TIE_TOL is sc.TIE_TOL and vrs.RTOL is sc.RTOL
    assert vrs.MIN_TRANSLATION_M == sc.MIN_TRANSLATION_M == bgs.MIN_TRANSLATION_DISTANCE_M
    assert vrs.CONSTRAINED_OVERRIDE is sc.CONSTRAINED_OVERRIDE is bgs.CONSTRAINED_OVERRIDE
    assert bgs.EXCLUDED_CITIES is sc.EXCLUDED_CITIES
    assert bgs.work_csv_path("barcelone") == sc.step2_table("barcelone")


def test_step2_table_uses_alt_dem_only_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "DATA_ROOT", tmp_path / "data")
    srtm = tmp_path / "data" / "dallas" / "graphs_fine_grid" / sc.STEP2_TABLE
    assert sc.step2_table("dallas") == srtm and sc.dem_source("dallas") == "srtm"
    alt = sc.alt_dem_table("dallas")
    alt.parent.mkdir(parents=True)
    alt.write_text("filename\n")
    assert sc.step2_table("dallas") == alt and sc.dem_source("dallas") == "3dep13"
    assert sc.step2_table("dallas", "graphs_fine_grid_v1").name == sc.STEP2_TABLE    # other grids untouched
    assert sc.step2_table("barcelone") == tmp_path / "data" / "barcelone" / "graphs_fine_grid" / sc.STEP2_TABLE
