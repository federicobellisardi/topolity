#!/usr/bin/env python3
"""Breakdown thresholds of the two failing multimodal worst cases (Supplementary Note 9).

Reads only saved per-cell / per-embedding sums (no rerouting):
  * Barcelona densification (output/barcelona_multimodal_robustness/candidate_cells_multimodal.csv):
    - smallest factor s by which the per-passenger energy of all public transport (bus, metro,
      train, tram) must be multiplied, other parameters central, for the Fig. 7 order with the
      observed modal mix to fail; implied mean bus load and public-transport/car energy per
      passenger-km at that s;
    - worst case over the box with lambda proportional to mass for every vehicle.
  * Madrid transformation (output/transform_multimodal/madrid/embedding_sums.csv), rail-only
    scenario, main family (translations, rotations, expansions):
    - smallest factor on the metro per-passenger energy (other parameters central) at which an
      alternative becomes cheaper; same for commuter rail alone;
    - worst case with lambda proportional to mass and the metro described per car
      (mass and load per car; train length cancels out of the per-passenger energy).

Usage: python multimodal_breakdown.py [--bcn-config PATH] [--mad-config PATH]
Writes reports/reviewer1/tables/multimodal_breakdown.csv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barcelona_multimodal_robustness as bmr  # noqa: E402
import barcelona_transform_multimodal as btm  # noqa: E402
import multi_city_settlement as mcs  # noqa: E402
import ranking_theory as rt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PT_MODES = ["bus"] + bmr.RAIL_MODES
S_MAX = 20.0


def scaled(par, modes, s):
    """Per-passenger energy of `modes` multiplied by s (occupancy / s)."""
    return {m: ({**p, "occupancy": p["occupancy"] / s} if m in modes else p) for m, p in par.items()}


def bisect(fails, lo=1.0, hi=S_MAX, tol=1e-3):
    """Smallest s in [lo, hi] with fails(s) True (fails monotone); inf if fails(hi) is False."""
    if fails(lo):
        return lo
    if not fails(hi):
        return np.inf
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fails(mid) else (mid, hi)
    return hi


def proportional_lambda_boxes(cfg, central):
    """Boxes without lambda; lambda = lambda_central * m / m_central is imposed inside the energy."""
    boxes = {v: {k: (cfg["modes"][v][k][0], cfg["modes"][v][k][2]) for k in ("mass_kg", "occupancy", "regen_eff")}
             for v in cfg["modes"]}
    ratio = {v: central[v]["lambda_J_per_m"] / central[v]["mass_kg"] for v in cfg["modes"]}

    def tie(par):
        return {v: {**p, "lambda_J_per_m": ratio[v] * p["mass_kg"]} for v, p in par.items()}
    return boxes, tie


def barcelona_densification(cfg, out_dir=bmr.OUTPUT_DIR):
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "candidate_cells_multimodal.csv", sep=";")
    sel = pd.read_csv(out_dir / "selected_cells_car_baseline.csv", sep=";")
    pos = {cid: i for i, cid in enumerate(df["cell_id"])}
    fav = [pos[c] for c in sel.loc[sel["scenario_type"] == "favorable", "cell_id"]]
    unf = [pos[c] for c in sel.loc[sel["scenario_type"] == "unfavorable", "cell_id"]]
    pairs = [(u, f) for u in unf for f in fav]
    shares = bmr.modal_shares(cfg)
    central = bmr.central_params(cfg)
    rows = []
    for a in cfg["rail_network"]["max_access_m"]:
        tag = f"a{int(a)}"

        def order_fails(s):
            E = bmr.mixed_energy(df, scaled(central, PT_MODES, s), shares, tag)
            return max(E[fav]) >= min(E[unf])
        s_star = bisect(order_fails)
        # Energy per passenger-km at s*, flow-weighted over the five selected cells
        idx = fav + unf
        par = scaled(central, PT_MODES, s_star)
        e_bus = bmr.single_mode_energy(df, par, "bus", tag)[idx].sum()
        L_bus = (df[f"{tag}_busnet_L"] + df[f"{tag}_busroad_L"]).to_numpy()[idx].sum()
        e_rail = bmr.rail_energy(df, par, tag)[idx].sum()
        L_rail = sum(df[f"{tag}_{m}_L"].to_numpy()[idx].sum() for m in bmr.RAIL_MODES) + \
            (df[f"{tag}_rfbnet_L"] + df[f"{tag}_rfbroad_L"]).to_numpy()[idx].sum()
        e_car = bmr.single_mode_energy(df, central, "car", tag)[idx].sum()
        L_car = df["road_L"].to_numpy()[idx].sum()
        w_bus, w_rail = shares["bus"], shares["rail"]
        pt_per_m = (w_bus * e_bus + w_rail * e_rail) / (w_bus * L_bus + w_rail * L_rail)
        rows.append({"case": "bcn_densification_mix_PT_scale", "access_m": a, "value": s_star,
                     "bus_load_at_threshold": central["bus"]["occupancy"] / s_star,
                     "PT_over_car_energy_per_pkm": pt_per_m / (e_car / L_car)})

        boxes, tie = proportional_lambda_boxes(cfg, central)

        def diff(p):
            E = bmr.mixed_energy(df, tie(p), shares, tag)
            return np.array([E[u] - E[f] for u, f in pairs])
        worst, _ = rt.separable_box_minimum(diff, central, boxes)
        E0 = bmr.mixed_energy(df, central, shares, tag)
        rows.append({"case": "bcn_densification_mix_worst_lambda_prop_m", "access_m": a,
                     "value": float(min(100 * w / E0[f] for w, (u, f) in zip(worst, pairs))),
                     "n_pairs_failing": int((worst < 0).sum())})
    return rows


def madrid_transformation(cfg):
    S = pd.read_csv(btm.OUTPUT_ROOT / "madrid" / "embedding_sums.csv").set_index("filename")
    fam = pd.Series({f: btm.family_of(f) for f in S.index})
    files = [btm.ORIGINAL] + [f for f in S.index if fam[f] not in ("original", "scale_down")]
    S = S.loc[files]
    alts = files[1:]
    shares = bmr.modal_shares(cfg)
    central = bmr.central_params(cfg)
    rows = []
    for a in [int(x) for x in cfg["rail_network"]["max_access_m"]]:
        def rail_E(par):
            return pd.Series({f: btm.scenario_energy(S.loc[f], par, shares, a)["rail"] for f in files})

        for mode in ("metro", "train"):
            def fails(s, mode=mode):
                E = rail_E(scaled(central, [mode], s))
                return E[alts].min() < E[btm.ORIGINAL]
            up = bisect(fails)
            down = bisect(lambda s: fails(1.0 / s))
            rows.append({"case": f"mad_rail_{mode}_scale_up", "access_m": a, "value": up})
            rows.append({"case": f"mad_rail_{mode}_scale_down_inverse", "access_m": a, "value": down})

        # lambda proportional to mass, metro per car (mass and load per car, Barcelona values)
        cars_c = 5.0
        cfg_car = json.loads(json.dumps(cfg))
        mc = cfg_car["modes"]["metro"]
        mc["mass_kg"] = [v / n for v, n in zip(mc["mass_kg"], (4, 5, 6))]
        mc["occupancy"] = [v / n for v, n in zip(mc["occupancy"], (4, 5, 6))]
        mc["lambda_J_per_m"] = [v / cars_c for v in mc["lambda_J_per_m"]]
        cen = bmr.central_params(cfg_car)
        boxes, tie = proportional_lambda_boxes(cfg_car, cen)

        def diff(p):
            E = rail_E(tie(p))
            return (E[alts] - E[btm.ORIGINAL]).to_numpy()
        worst, _ = rt.separable_box_minimum(diff, cen, boxes)
        E0 = rail_E(cen)[btm.ORIGINAL]
        rows.append({"case": "mad_rail_worst_lambda_prop_m_per_car", "access_m": a,
                     "value": 100 * worst.min() / E0, "n_cheaper": int((worst < 0).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bcn-config", default=str(bmr.CONFIG_FILE))
    ap.add_argument("--mad-config", default=str(btm.CONFIGS["madrid"]))
    ap.add_argument("--dens-dir", default=str(bmr.OUTPUT_DIR))
    ap.add_argument("--out", default=str(ROOT / "reports/reviewer1/tables/multimodal_breakdown.csv"))
    args = ap.parse_args()
    rows = barcelona_densification(json.loads(Path(args.bcn_config).read_text()), args.dens_dir)
    rows += madrid_transformation(json.loads(Path(args.mad_config).read_text()))
    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
