"""Tests for python/barcelona_multimodal_robustness.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import barcelona_multimodal_robustness as bmr  # noqa: E402
import multi_city_settlement as mcs  # noqa: E402


# Rail graph construction and profiles are tested in test_rail_network.py.


def test_tunnel_factor_scales_only_metro_tunnel_gain():
    df = pd.DataFrame({"a1500_metro_L": [1000.0], "a1500_metro_H": [30.0], "a1500_metro_Ht": [20.0],
                       "a1500_train_L": [500.0], "a1500_train_H": [10.0], "a1500_train_Ht": [5.0],
                       "a1500_tram_L": [0.0], "a1500_tram_H": [0.0], "a1500_tram_Ht": [0.0],
                       **{f"a1500_{p}_{q}": [0.0] for p in ("rfbnet", "rfbroad") for q in ("L", "H")}})
    par = {m: {"mass_kg": 1.0, "lambda_J_per_m": 0.0, "occupancy": 1.0, "regen_eff": 0.0}
           for m in bmr.RAIL_MODES + ["bus"]}
    g = mcs.G_PHYS
    assert bmr.rail_energy(df, par, "a1500")[0] == pytest.approx(g * 40.0)
    assert bmr.rail_energy(df, par, "a1500", 0.0)[0] == pytest.approx(g * 20.0)   # metro 30 - 20, train 10
    assert bmr.rail_energy(df, par, "a1500", 2.0)[0] == pytest.approx(g * 60.0)   # metro 30 + 20, train 10


def test_mode_energy_regen_and_occupancy():
    par = {"mass_kg": 1000.0, "lambda_J_per_m": 500.0, "occupancy": 2.0, "regen_eff": 0.5}
    e = bmr.mode_energy(par, L=1000.0, H=10.0)
    assert e == pytest.approx((1000 * mcs.G_PHYS * 0.5 * 10 + 500 * 1000) / 2)


def test_car_only_split_reproduces_paper_ranking():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"road_L": rng.uniform(8e3, 2e4, n), "road_H": rng.uniform(50, 300, n)})
    for m in bmr.RAIL_MODES:
        df[f"a1500_{m}_L"] = rng.uniform(0, 1e4, n)
        df[f"a1500_{m}_H"] = rng.uniform(0, 100, n)
    for pre in ("busnet", "busroad", "rfbnet", "rfbroad"):
        df[f"a1500_{pre}_L"] = rng.uniform(0, 1e4, n)
        df[f"a1500_{pre}_H"] = rng.uniform(0, 100, n)
    par = {m: {"mass_kg": 1.0, "lambda_J_per_m": 1.0, "occupancy": 1.0, "regen_eff": 0.0}
           for m in bmr.ROAD_MODES + bmr.RAIL_MODES}
    par["car"] = {"mass_kg": mcs.M_PHYS_KG, "lambda_J_per_m": mcs.LAMBDA_J_PER_M,
                  "occupancy": 1.0, "regen_eff": 0.0}
    shares = {"car": 1.0, "moto": 0.0, "bus": 0.0, "rail": 0.0}
    E = bmr.mixed_energy(df, par, shares, "a1500")
    E_paper = mcs.M_PHYS_KG * mcs.G_PHYS * df["road_H"] + mcs.LAMBDA_J_PER_M * df["road_L"]
    np.testing.assert_allclose(E, E_paper)
    m = bmr.ordering_metrics(E_paper.to_numpy(), E, [0, 1], [2, 3])
    assert m["spearman"] == pytest.approx(1.0)


def test_modal_split_from_emef():
    cfg = {"modal_split": {"trips_thousands": {"car": 5886.8, "moto": 758.1, "bus": 1361.8, "rail": 2135.5}}}
    s = bmr.modal_shares(cfg)
    assert sum(s.values()) == pytest.approx(1.0)
    assert s["car"] == pytest.approx(0.5804, abs=1e-4)
    assert s["rail"] == pytest.approx(0.2106, abs=1e-4)


def test_ratio_range_brackets_central_value():
    cfg = {"mass_kg": [1000, 1200, 1800], "lambda_J_per_m": [400, 585, 800], "regen_eff": [0.0, 0.0, 0.0]}
    lo, c, hi = bmr.ratio_range(cfg)
    assert c == pytest.approx(1200 * mcs.G_PHYS / 585)
    assert lo == pytest.approx(1000 * mcs.G_PHYS / 800) and hi == pytest.approx(1800 * mcs.G_PHYS / 400)


def test_single_road_mode_ranking_depends_only_on_ratio():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"road_L": rng.uniform(8e3, 2e4, 20), "road_H": rng.uniform(50, 300, 20)})
    a = {"car": {"mass_kg": 1200.0, "lambda_J_per_m": 585.0, "occupancy": 1.19, "regen_eff": 0.0}}
    b = {"car": {"mass_kg": 2400.0, "lambda_J_per_m": 1170.0, "occupancy": 3.0, "regen_eff": 0.0}}
    ea, eb = bmr.single_mode_energy(df, a, "car", "a1500"), bmr.single_mode_energy(df, b, "car", "a1500")
    np.testing.assert_array_equal(np.argsort(ea), np.argsort(eb))


def test_modal_split_from_percentages():
    cfg = {"modal_split": {"trips_percent": {"car": 30.0, "bus": 10.0, "rail": 10.0}}}
    assert bmr.modal_shares(cfg) == pytest.approx({"car": 0.6, "bus": 0.2, "rail": 0.2})
