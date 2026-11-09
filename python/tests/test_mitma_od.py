"""Tests for python/mitma_od.py (MITMA district flows spread over 1 km cells, SI Note 9)."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mitma_od as mo  # noqa: E402


def test_weekdays_exclude_holiday_and_bridge():
    days = mo.october_2022_weekdays()
    assert len(days) == 19
    assert date(2022, 10, 12) not in days and date(2022, 10, 31) not in days
    assert all(d.weekday() < 5 for d in days)


def test_disaggregate_conserves_trips_and_splits_by_population():
    T = sparse.csr_matrix(np.array([[10.0, 30.0], [0.0, 5.0]]))       # districts A, B
    W = sparse.csr_matrix(np.array([[0.25, 0.75, 0.0], [0.0, 0.0, 1.0]]))  # A -> cells 0, 1; B -> cell 2
    F = mo.disaggregate(T, W).toarray()
    assert F.sum() == pytest.approx(T.sum())
    assert F[0, 2] == pytest.approx(0.25 * 30.0)
    assert F[1, 1] == pytest.approx(0.75 * 0.75 * 10.0)
    assert F[2, 0] == 0.0


def test_disaggregate_drops_population_outside_the_network():
    T = sparse.csr_matrix(np.array([[8.0]]))
    W = sparse.csr_matrix(np.array([[0.5]]))         # half of the district has no network cell
    assert mo.disaggregate(T, W).sum() == pytest.approx(0.5 * 0.5 * 8.0)


def test_thinning_keeps_large_flows_and_is_unbiased():
    rng = np.random.default_rng(0)
    f = rng.lognormal(0, 2, 20000)
    F = sparse.coo_matrix((f, (np.arange(len(f)), np.arange(len(f)))), shape=(len(f), len(f)))
    tau = mo.thinning_threshold(f, 2000)
    o, d, g = mo.thin(F, 2000, seed=1)
    assert abs(len(g) - 2000) < 150                      # about max_pairs kept
    assert set(np.flatnonzero(f >= tau)) <= set(o)       # every large flow kept, unchanged
    assert g.sum() == pytest.approx(f.sum(), rel=0.05)   # total preserved in expectation
    # unbiased by size class: the small flows keep their total in expectation as well
    small = f < tau
    kept_small = np.isin(np.arange(len(f)), o) & small
    assert tau * kept_small.sum() == pytest.approx(f[small].sum(), rel=0.1)


def test_thinning_noop_below_max_pairs():
    F = sparse.coo_matrix((np.array([1.0, 2.0]), (np.array([0, 1]), np.array([1, 0]))), shape=(2, 2))
    o, d, g = mo.thin(F, 10)
    assert g.tolist() == [1.0, 2.0]


def test_cell_grid_lookup():
    cells = pd.DataFrame({"cell_id": ["0_0", "1_0", "0_1"], "x_min": [0.0, 1000.0, 0.0],
                          "y_min": [0.0, 0.0, 1000.0]})
    g = mo.CellGrid(cells)
    out = g.lookup(np.array([10.0, 1500.0, 500.0, 5000.0]), np.array([10.0, 999.0, 1200.0, 0.0]))
    assert out.tolist() == ["0_0", "1_0", "0_1", None]
