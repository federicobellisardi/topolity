"""Tests for python/multimodal_breakdown.py (breakdown thresholds of Supplementary Note 9)."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import multimodal_breakdown as mb  # noqa: E402
import ranking_theory as rt  # noqa: E402


def test_bisect_finds_threshold_of_monotone_failure():
    s_star = mb.bisect(lambda s: s >= 2.6, tol=1e-6)
    assert s_star == pytest.approx(2.6, abs=1e-5)


def test_bisect_returns_inf_when_never_failing_and_lo_when_failing_at_once():
    assert mb.bisect(lambda s: False) == np.inf
    assert mb.bisect(lambda s: True) == 1.0


def test_scaled_multiplies_energy_only_of_listed_modes():
    par = {"bus": {"occupancy": 13.6, "mass_kg": 15000.0},
           "car": {"occupancy": 1.19, "mass_kg": 1200.0}}
    out = mb.scaled(par, ["bus"], 2.0)
    assert out["bus"]["occupancy"] == pytest.approx(6.8)   # energy per passenger x2
    assert out["bus"]["mass_kg"] == 15000.0
    assert out["car"] == par["car"]
    assert par["bus"]["occupancy"] == 13.6                  # input not modified


def test_proportional_lambda_tie_and_boxes():
    cfg = {"modes": {"v": {"mass_kg": [100.0, 200.0, 300.0], "lambda_J_per_m": [5.0, 10.0, 20.0],
                           "occupancy": [1.0, 2.0, 3.0], "regen_eff": [0.0, 0.1, 0.2]}}}
    central = {"v": {k: v[1] for k, v in cfg["modes"]["v"].items()}}
    boxes, tie = mb.proportional_lambda_boxes(cfg, central)
    assert set(boxes["v"]) == {"mass_kg", "occupancy", "regen_eff"}     # lambda is not free
    tied = tie({"v": {**central["v"], "mass_kg": 300.0}})
    assert tied["v"]["lambda_J_per_m"] == pytest.approx(15.0)          # 10 * 300 / 200


def test_tied_worst_case_is_exact_and_not_below_unconstrained():
    """With lambda tied to mass the energy stays multilinear in (m, n, eps); the vertex minimum must equal
    a brute-force minimum over a grid and can only be higher than the unconstrained worst case."""
    cfg = {"modes": {"v": {"mass_kg": [100.0, 200.0, 300.0], "lambda_J_per_m": [5.0, 10.0, 20.0],
                           "occupancy": [1.0, 2.0, 3.0], "regen_eff": [0.0, 0.1, 0.2]}}}
    central = {"v": {k: v[1] for k, v in cfg["modes"]["v"].items()}}
    L = np.array([10.0, 12.0])        # cell 0 shorter, cell 1 flatter
    H = np.array([3.0, 1.0])

    def energy(p):
        q = p["v"]
        return (q["mass_kg"] * 9.81 * (1 - q["regen_eff"]) * H + q["lambda_J_per_m"] * L) / q["occupancy"]

    boxes, tie = mb.proportional_lambda_boxes(cfg, central)
    diff_t = lambda p: np.array([energy(tie(p))[1] - energy(tie(p))[0]])  # noqa: E731
    worst_t, _ = rt.separable_box_minimum(diff_t, central, boxes)
    grid = [np.linspace(*boxes["v"][k], 7) for k in boxes["v"]]
    brute = min(diff_t({"v": {**central["v"], **dict(zip(boxes["v"], x))}})[0] for x in itertools.product(*grid))
    assert worst_t[0] == pytest.approx(brute, rel=1e-9)

    full = {"v": {k: (v[0], v[2]) for k, v in cfg["modes"]["v"].items()}}
    worst_u, _ = rt.separable_box_minimum(lambda p: np.array([energy(p)[1] - energy(p)[0]]), central, full)
    assert worst_t[0] >= worst_u[0] - 1e-9


REPORTED = Path(__file__).resolve().parents[2] / "reports/reviewer1/tables/multimodal_breakdown.csv"


@pytest.mark.skipif(not (mb.bmr.OUTPUT_DIR / "candidate_cells_multimodal.csv").exists() or not REPORTED.exists(),
                    reason="saved densification sums or reported table not available")
def test_reported_barcelona_thresholds_are_reproduced():
    """Regression: the values quoted in Supplementary Note 9 come from the current config and sums."""
    import pandas as pd
    rep = pd.read_csv(REPORTED)
    rows = pd.DataFrame(mb.barcelona_densification(json.loads(mb.bmr.CONFIG_FILE.read_text())))
    for case in ("bcn_densification_mix_PT_scale", "bcn_densification_mix_worst_lambda_prop_m"):
        a = rep[rep.case == case].sort_values("access_m")["value"].to_numpy()
        b = rows[rows.case == case].sort_values("access_m")["value"].to_numpy()
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)
