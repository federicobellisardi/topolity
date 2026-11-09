"""Exact ranking criteria for the linear energy model (reports/reviewer1/MODE_INVARIANCE_THEOREM.md).

Every configuration c is summarised by its trip-weighted path length D_c [m]
and uphill gain H_c [m]. For a vehicle with parameters (m, lambda, eps) the
normalised energy is

    E_c(r) = W_c / lambda = D_c + r H_c,      r = m g (1 - eps) / lambda  (dimensionless),

so every comparison is between straight lines in r. All functions below are
exact (no sampling of r) and treat ties with an explicit relative tolerance.
"""

from __future__ import annotations

import numpy as np

G = 9.81


def ratio(mass_kg, lambda_J_per_m, regen_eff=0.0, g=G):
    """r = m g (1 - eps) / lambda, dimensionless (N / N)."""
    return mass_kg * g * (1.0 - regen_eff) / lambda_J_per_m


def energy(D, H, r):
    """Normalised energy E(r) = D + r H (same units as D)."""
    return np.asarray(D, float) + r * np.asarray(H, float)


def crossing(Da, Ha, Db, Hb):
    """r* where E_a(r) = E_b(r); None if the lines are parallel (H_a == H_b)."""
    if Ha == Hb:
        return None
    return -(Da - Db) / (Ha - Hb)


def no_more_costly_on_interval(Da, Ha, Db, Hb, r_lo, r_hi, rtol=0.0):
    """Theorem 2.2: E_a(r) <= E_b(r) (1 + rtol) for every r in [r_lo, r_hi].

    E_a - E_b is affine in r, so it is enough to check the two endpoints.
    """
    if not 0.0 <= r_lo <= r_hi:
        raise ValueError("need 0 <= r_lo <= r_hi")
    for r in (r_lo, r_hi):
        if Da + r * Ha > (Db + r * Hb) * (1.0 + rtol):
            return False
    return True


def dominates(Da, Ha, Db, Hb):
    """Componentwise dominance: D_a <= D_b and H_a <= H_b, i.e. E_a <= E_b for every r >= 0."""
    return Da <= Db and Ha <= Hb


def minimizer_interval(D0, H0, D, H, rtol=0.0):
    """Exact set {r >= 0 : E_0(r) <= E_c(r)(1 + rtol) for all c}, as (lo, hi) or None.

    Each alternative gives one half-line constraint a_c r <= b_c with
    a_c = H0 - (1 + rtol) H_c and b_c = (1 + rtol) D_c - D0; the feasible set is
    their intersection with [0, inf), always an interval (possibly empty).
    """
    D, H = np.asarray(D, float), np.asarray(H, float)
    a = H0 - (1.0 + rtol) * H
    b = (1.0 + rtol) * D - D0
    lo, hi = 0.0, np.inf
    pos, neg, zero = a > 0, a < 0, a == 0
    if np.any(b[zero] < 0):
        return None
    if pos.any():
        hi = float(np.min(b[pos] / a[pos]))
    if neg.any():
        lo = max(lo, float(np.max(b[neg] / a[neg])))
    return (lo, hi) if lo <= hi else None


def classify_on_interval(D0, H0, D, H, r_lo, r_hi, rtol=0.0):
    """'robust_minimum' if E_0 is minimal on all of [r_lo, r_hi]; 'never_minimum' if on none of it;
    otherwise 'switches' (the interval of r where it is minimal is returned too)."""
    iv = minimizer_interval(D0, H0, D, H, rtol)
    if iv is not None and iv[0] <= r_lo and r_hi <= iv[1]:
        return "robust_minimum", iv
    if iv is None or iv[1] < r_lo or iv[0] > r_hi:
        return "never_minimum", iv
    return "switches", iv


def rank_of(E0, E, rtol=0.0):
    """Observed rank: 1 + number of alternatives strictly cheaper beyond the tolerance."""
    return 1 + int(np.sum(np.asarray(E, float) * (1.0 + rtol) < E0))


def regret(E0, E):
    """(E_0 - min(E_0, E_c)) / min(E_0, E_c) >= 0; zero iff the observed configuration is minimal."""
    best = min(E0, float(np.min(E))) if len(E) else E0
    return (E0 - best) / best


def ranking_breakpoints(D, H, r_lo=0.0, r_hi=np.inf):
    """Sorted distinct pairwise crossings inside (r_lo, r_hi).

    Between consecutive breakpoints the full order of the configurations is
    constant (ties only at the breakpoints themselves).
    """
    D, H = np.asarray(D, float), np.asarray(H, float)
    i, j = np.triu_indices(len(D), k=1)
    dH = H[i] - H[j]
    ok = dH != 0
    r = -(D[i][ok] - D[j][ok]) / dH[ok]
    r = r[(r > r_lo) & (r < r_hi)]
    return np.unique(r)


def ranking_intervals(D, H, r_lo=0.0, r_hi=np.inf, labels=None):
    """List of (lo, hi, order) with the order (cheapest first) constant on (lo, hi).

    The order inside each interval is evaluated at its midpoint; equal energies
    are broken by label only for presentation.
    """
    D, H = np.asarray(D, float), np.asarray(H, float)
    labels = list(range(len(D))) if labels is None else list(labels)
    edges = [r_lo, *ranking_breakpoints(D, H, r_lo, r_hi), r_hi]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = lo + 1.0 if np.isinf(hi) else 0.5 * (lo + hi)
        e = energy(D, H, mid)
        order = [labels[k] for k in sorted(range(len(D)), key=lambda k: (e[k], str(labels[k])))]
        out.append((lo, hi, order))
    return out


def mixture_ratio(shares, masses, lambdas, regen=None, g=G):
    """r_eff of a common-share mixture on shared routes (Section 9).

    r_eff = g sum_s p_s m_s (1 - eps_s) / sum_s p_s lambda_s = sum_s w_s r_s with
    w_s = p_s lambda_s / sum_j p_j lambda_j, hence min_s r_s <= r_eff <= max_s r_s.
    `masses` and `lambdas` must use the same normalisation (per vehicle, or per
    passenger after dividing both by occupancy).
    """
    p, m, lam = (np.asarray(x, float) for x in (shares, masses, lambdas))
    eps = np.zeros_like(m) if regen is None else np.asarray(regen, float)
    if np.any(p < 0) or np.any(m <= 0) or np.any(lam <= 0):
        raise ValueError("shares must be >= 0, masses and lambdas > 0")
    return g * float(np.sum(p * m * (1.0 - eps))) / float(np.sum(p * lam))


def worst_case_over_shares(delta_cost, flows):
    """Theorem 5.1: min over OD-specific share simplices of sum_q F_q sum_s p_qs dC_qs.

    delta_cost: (n_od, n_modes) array of C_c - C_0 per OD and mode (NaN = mode not
    feasible for that OD); flows: (n_od,). The minimum puts all mass on the
    cheapest feasible mode of each OD.
    """
    dc = np.asarray(delta_cost, float)
    if np.any(np.all(np.isnan(dc), axis=1)):
        raise ValueError("every OD needs at least one feasible mode")
    return float(np.sum(np.asarray(flows, float) * np.nanmin(dc, axis=1)))


def regen_coefficients(mass_kg, lambda_J_per_m, regen_eff, g=G):
    """(a, b) of E = D + a H + b Z with recovery: a = m g (1 - eps)/lambda, b = m g eps/lambda."""
    k = mass_kg * g / lambda_J_per_m
    return k * (1.0 - regen_eff), k * regen_eff


def regen_box_vertices(mass_range, lambda_range, regen_range, g=G):
    """(a, b) at the 8 corners of a (m, lambda, eps) box; their convex hull contains the image of the box
    because a and b are monotone in each parameter."""
    return np.array([regen_coefficients(m, lam, e, g)
                     for m in mass_range for lam in lambda_range for e in regen_range])


def certified_minimum_with_recovery(D0, H0, Z0, D, H, Z, ab_vertices, rtol=0.0):
    """Sufficient test that E_0 <= (1 + rtol) E_c for every c and every (a, b) in the convex hull of
    `ab_vertices`, with E = D + a H + b Z. E_0 - (1 + rtol) E_c is affine in (a, b), so checking the
    vertices is exact for the hull (and conservative for any subset of it)."""
    D, H, Z = (np.asarray(x, float) for x in (D, H, Z))
    for a, b in ab_vertices:
        if np.any(D0 + a * H0 + b * Z0 > (1.0 + rtol) * (D + a * H + b * Z)):
            return False
    return True


def separable_box_minimum(diff, central, boxes):
    """Exact minimum over a parameter box of a difference that is separable by vehicle.

    diff(par) -> array: e.g. E_c - E_0 for every alternative c, where par = {vehicle: {param: value}}.
    The function must be a sum of per-vehicle terms, each multilinear in that vehicle's parameters
    (true for (m g (H+ - eps H-) + lambda L) / occupancy, which is multilinear in m, eps, lambda and
    1/occupancy). A multilinear function on a box attains its minimum at a vertex, and the per-vehicle
    terms are independent, so
        min diff = diff(central) + sum_v min_{vertices of v} [diff(central with v at the vertex) - diff(central)].
    boxes: {vehicle: {param: (low, high)}}. Returns (minimum array, {vehicle: argmin vertex index array}).
    Cost: sum_v 2^(number of parameters of v) evaluations instead of their product.
    """
    import itertools
    d0 = np.asarray(diff(central), dtype=float)
    total = d0.copy()
    arg = {}
    for v, box in boxes.items():
        names = list(box)
        verts = list(itertools.product(*(box[k] for k in names)))
        vals = np.array([np.asarray(diff({**central, v: {**central[v], **dict(zip(names, x))}}), float) - d0
                         for x in verts])
        total += vals.min(axis=0)
        arg[v] = vals.argmin(axis=0)
    return total, arg
