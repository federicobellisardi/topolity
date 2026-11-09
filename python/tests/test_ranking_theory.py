"""Unit tests for python/ranking_theory.py (Phase 1 of todo.rev1.md)."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ranking_theory as rt  # noqa: E402

R_GRID = np.linspace(0.0, 200.0, 20001)


def brute_min_set(D0, H0, D, H, rtol=0.0):
    e0 = D0 + R_GRID * H0
    ec = np.asarray(D)[:, None] + np.outer(H, R_GRID)
    return np.all(e0[None, :] <= ec * (1 + rtol), axis=0)


# --- units and basic objects ---------------------------------------------------

def test_ratio_is_dimensionless_and_matches_paper():
    # m g [N] / lambda [J/m = N]
    assert rt.ratio(1200.0, 585.0) == pytest.approx(20.123, abs=1e-3)
    assert rt.ratio(1800.0, 460.0, regen_eff=0.5) == pytest.approx(1800 * 9.81 * 0.5 / 460)


def test_crossing_of_the_illustrative_pair():
    # Fig. 1 of the working draft: (D, H) = (110, 7) and (100, 9) cross at r* = 5
    assert rt.crossing(110, 7, 100, 9) == pytest.approx(5.0)
    assert rt.crossing(110, 7, 100, 7) is None


# --- Theorem 2.2 and dominance -------------------------------------------------

def test_interval_test_equals_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        Da, Db = rng.uniform(90, 110, 2)
        Ha, Hb = rng.uniform(0, 10, 2)
        lo, hi = np.sort(rng.uniform(0, 100, 2))
        r = np.linspace(lo, hi, 2001)
        brute = bool(np.all(Da + r * Ha <= Db + r * Hb))
        assert rt.no_more_costly_on_interval(Da, Ha, Db, Hb, lo, hi) == brute


def test_dominance_and_its_converse():
    assert rt.dominates(100, 5, 101, 5)
    assert rt.no_more_costly_on_interval(100, 5, 101, 5, 0, 1e9)
    # converse: without dominance some r >= 0 reverses the order
    for (Da, Ha, Db, Hb) in [(101, 4, 100, 5), (99, 6, 100, 5)]:
        assert not rt.dominates(Da, Ha, Db, Hb)
        assert not rt.no_more_costly_on_interval(Da, Ha, Db, Hb, 0, 1e6)


# --- observed-city minimizer interval -------------------------------------------

def test_minimizer_interval_equals_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(200):
        D0, H0 = rng.uniform(95, 105), rng.uniform(2, 8)
        k = rng.integers(1, 8)
        D, H = rng.uniform(95, 110, k), rng.uniform(0, 10, k)
        iv = rt.minimizer_interval(D0, H0, D, H)
        brute = brute_min_set(D0, H0, D, H)
        inside = np.zeros_like(brute) if iv is None else (R_GRID >= iv[0]) & (R_GRID <= iv[1])
        # disagreements only within one grid step of an interval endpoint
        bad = R_GRID[inside != brute]
        assert bad.size == 0 or (iv is not None and np.all(
            np.minimum(abs(bad - iv[0]), abs(bad - iv[1])) <= R_GRID[1]))


def test_ties_parallel_lines_and_duplicates():
    # duplicate of the observed configuration: tie, never strictly better
    assert rt.minimizer_interval(100, 5, [100], [5]) == (0.0, np.inf)
    # parallel and cheaper: never minimal; parallel and dearer: always minimal
    assert rt.minimizer_interval(100, 5, [99], [5]) is None
    assert rt.minimizer_interval(100, 5, [101], [5]) == (0.0, np.inf)
    # tolerance turns a 0.04 % deficit into a tie (SI rule: |delta| < 0.05 %)
    assert rt.minimizer_interval(100.04, 5, [100.0], [5.0]) is None
    assert rt.minimizer_interval(100.04, 5, [100.0], [5.0], rtol=0.0005) is not None


def test_classification_on_a_vehicle_range():
    # observed shorter but hillier than the alternative: minimal only for r <= 5
    assert rt.classify_on_interval(100, 9, [110], [7], 1, 4)[0] == "robust_minimum"
    assert rt.classify_on_interval(100, 9, [110], [7], 6, 76)[0] == "never_minimum"
    status, iv = rt.classify_on_interval(100, 9, [110], [7], 1, 76)
    assert status == "switches" and iv == pytest.approx((0.0, 5.0))


# --- invariance of the ranking under (m, lambda) at fixed r ----------------------

def test_ranking_depends_on_parameters_only_through_r():
    rng = np.random.default_rng(2)
    D, H = rng.uniform(90, 110, 12), rng.uniform(0, 10, 12)
    base = np.argsort(1200 * 9.81 * H + 585 * D)
    for scale, occ in [(2.0, 1.0), (0.3, 1.19), (5.0, 13.6)]:
        m, lam = 1200 * scale / occ, 585 * scale / occ
        assert np.array_equal(np.argsort(m * 9.81 * H + lam * D), base)


# --- full ranking intervals and adjacency ---------------------------------------

def test_order_is_constant_between_breakpoints():
    rng = np.random.default_rng(3)
    D, H = rng.uniform(90, 110, 6), rng.uniform(0, 10, 6)
    for lo, hi, order in rt.ranking_intervals(D, H, 0, 100):
        for r in np.linspace(lo, hi, 9)[1:-1]:
            assert list(np.argsort(rt.energy(D, H, r), kind="stable")) == order


def test_adjacent_pairs_certify_the_full_order():
    rng = np.random.default_rng(4)
    for _ in range(100):
        D, H = rng.uniform(90, 110, 5), rng.uniform(0, 10, 5)
        lo, hi = 5.0, 40.0
        order = np.argsort(rt.energy(D, H, lo))
        adjacent = all(rt.no_more_costly_on_interval(D[a], H[a], D[b], H[b], lo, hi)
                       for a, b in zip(order[:-1], order[1:]))
        every_pair = all(rt.no_more_costly_on_interval(D[a], H[a], D[b], H[b], lo, hi)
                         for a, b in itertools.combinations(order, 2))
        assert adjacent == every_pair


# --- mixtures on shared routes (Section 9) ---------------------------------------

def test_mixture_ratio_lies_between_mode_ratios_and_reproduces_the_mixed_ranking():
    rng = np.random.default_rng(5)
    D, H = rng.uniform(90, 110, 8), rng.uniform(0, 10, 8)
    for _ in range(200):
        n = rng.integers(2, 6)
        p = rng.dirichlet(np.ones(n))
        m, lam, eps = rng.uniform(100, 2e5, n), rng.uniform(100, 6e4, n), rng.uniform(0, 0.7, n)
        r_s = rt.ratio(m, lam, eps)
        r_eff = rt.mixture_ratio(p, m, lam, eps)
        assert r_s.min() - 1e-9 <= r_eff <= r_s.max() + 1e-9
        W_mix = sum(p[s] * (lam[s] * D + m[s] * rt.G * (1 - eps[s]) * H) for s in range(n))
        np.testing.assert_allclose(W_mix, np.sum(p * lam) * rt.energy(D, H, r_eff), rtol=1e-12)


def test_counterexample_modes_with_equal_ratio_but_different_routes_flip_the_ranking():
    # Two modes with the SAME r but different paths: road (D, H) and rail (D', H').
    r = 20.0
    road = {"a": (100.0, 5.0), "b": (105.0, 5.0)}     # road prefers a
    rail = {"a": (120.0, 6.0), "b": (100.0, 5.0)}     # rail prefers b
    E = lambda d, p: p * rt.energy(*road[d], r) + (1 - p) * rt.energy(*rail[d], r)  # noqa: E731
    assert E("a", 1.0) < E("b", 1.0)                   # all road
    assert E("a", 0.0) > E("b", 0.0)                   # all rail
    # no single (D, H) per configuration can represent both share scenarios at one r


# --- Theorem 5.1: arbitrary OD-specific shares ----------------------------------

def test_worst_case_over_shares_matches_the_illustration_and_brute_force():
    dC = np.array([[10.0, 5.0], [8.0, -2.0]])
    assert rt.worst_case_over_shares(dC, [1, 1]) == pytest.approx(3.0)
    grid = np.linspace(0, 1, 101)
    brute = min(p1 * 10 + (1 - p1) * 5 + p2 * 8 + (1 - p2) * (-2) for p1 in grid for p2 in grid)
    assert brute == pytest.approx(3.0)
    # infeasible mode (NaN) is skipped
    assert rt.worst_case_over_shares([[10.0, np.nan], [8.0, -2.0]], [1, 1]) == pytest.approx(8.0)


# --- recovery with non-zero net rise (Proposition 3) -----------------------------

def test_recovery_reduces_to_r_when_net_rise_is_zero():
    D, H = np.array([100.0, 104.0]), np.array([9.0, 7.0])
    a, b = rt.regen_coefficients(1800.0, 460.0, 0.5)
    assert a == pytest.approx(rt.ratio(1800.0, 460.0, 0.5))
    np.testing.assert_allclose(D + a * H + b * np.zeros(2), rt.energy(D, H, a))


def test_vertex_certificate_matches_brute_force_over_the_box():
    rng = np.random.default_rng(6)
    m_r, l_r, e_r = (1500.0, 2200.0), (310.0, 610.0), (0.3, 0.7)
    V = rt.regen_box_vertices(m_r, l_r, e_r)
    for _ in range(200):
        D0, H0, Z0 = rng.uniform(95, 105), rng.uniform(2, 8), rng.uniform(-3, 3)
        D, H, Z = rng.uniform(95, 110, 4), rng.uniform(0, 10, 4), rng.uniform(-3, 3, 4)
        cert = rt.certified_minimum_with_recovery(D0, H0, Z0, D, H, Z, V)
        ok = all(D0 + a * H0 + b * Z0 <= np.min(D + a * H + b * Z)
                 for m in np.linspace(*m_r, 6) for lam in np.linspace(*l_r, 6) for e in np.linspace(*e_r, 6)
                 for a, b in [rt.regen_coefficients(m, lam, e)])
        assert not cert or ok          # certificate is sufficient


def test_separable_box_minimum_matches_brute_force():
    """Energy-like differences (m g (dHu - eps dHd) + lambda dL) / occ, two vehicles, three alternatives."""
    rng = np.random.default_rng(3)
    dHu, dHd, dL = rng.normal(size=(3, 2, 3))      # vehicle x alternative
    boxes = {"a": {"m": (1.0, 3.0), "eps": (0.0, 0.6), "lam": (0.5, 2.0), "occ": (1.0, 4.0)},
             "b": {"m": (2.0, 5.0), "eps": (0.1, 0.4), "lam": (1.0, 3.0), "occ": (2.0, 6.0)}}
    central = {v: {k: float(np.mean(b)) for k, b in box.items()} for v, box in boxes.items()}

    def diff(par):
        return sum((p["m"] * 9.81 * (dHu[i] - p["eps"] * dHd[i]) + p["lam"] * dL[i]) / p["occ"]
                   for i, p in enumerate((par["a"], par["b"])))
    exact, _ = rt.separable_box_minimum(diff, central, boxes)
    grid = {v: list(itertools.product(*(np.linspace(*b, 4) for b in box.values()))) for v, box in boxes.items()}
    brute = np.full(3, np.inf)
    keys = {v: list(box) for v, box in boxes.items()}
    for xa in grid["a"]:
        for xb in grid["b"]:
            brute = np.minimum(brute, diff({"a": dict(zip(keys["a"], xa)), "b": dict(zip(keys["b"], xb))}))
    np.testing.assert_allclose(exact, brute, rtol=1e-10)     # the grid contains the vertices
    assert np.all(exact <= diff(central) + 1e-12)
