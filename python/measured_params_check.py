#!/usr/bin/env python3
"""SI Note 9, Supplementary Table 3: margins of the observed embedding at measured vehicle parameters.

The table gives low-central-high ranges; the exact worst case over these ranges is already in the SI. This
script evaluates, for Barcelona and Madrid (MITMA trips >= 2 km), the margin of the closest alternative
(translations, rotations, expansions) when every parameter with a measured value takes that value and the
others stay at their central value. Measured values and their sources are in MEASURED below; the values
must lie inside the ranges of the configs, so the worst case over the ranges also covers them.

Output: reports/reviewer1/tables/measured_params_margins.csv

Usage: python python/measured_params_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import barcelona_multimodal_robustness as bmr  # noqa: E402
import barcelona_transform_multimodal as btm  # noqa: E402

OD = "mitma_ge2km"
PASSENGER_KG = 70.0
OUT = ROOT / "reports" / "reviewer1" / "tables" / "measured_params_margins.csv"

# Measured values (None = no measured value found; the central value is kept).
#  car: EU new petrol and diesel cars registered in 2021, OBFCM on-board data (European Commission,
#       SWD(2024) 59, Tables 9 and 11): mass in running order 1407 kg (incl. 75 kg driver), real-world
#       fuel use 7.44 L/100 km -> lambda = 0.25 * 36 MJ/L * 7.44 L/100 km = 670 J/m (paper convention).
#  car_ev: EV Database real-world mean 191 Wh/km x 0.85 = 584 J/m; mass: electric cars 34% heavier than
#       combustion cars in 2023 (ICCT 2024) -> about 1.9 t.
#  bus: TMB 2021 fleet energy 6.4 kWh per bus-km x 0.25 = 5760 J/m; load 13.6 (TMB 2024).
#  metro (Barcelona): 157 t five-car train + 120 passengers; 2.184 kWh/car-km x 5 x 0.85 = 33.4 kJ/m;
#       120 passengers per train (TMB 2024).  Madrid: 1.99 kWh/car-km x 5 x 0.85 = 30.5 kJ/m.
#  train: Civia 465, 156.6-157.3 t in running order (157 t); 15.0-15.9 kWh per train-km x 0.85 -> 47.3 kJ/m.
#  tram: Citadis 302, 40 t + 30 passengers; 5.0 kWh per tram-km x 0.85 = 15.3 kJ/m; 30 passengers (TRAM).
#  occupancy of cars and motorcycles: EMEF 2025 (Barcelona) and EDM2018 (Madrid), as in the configs.
COMMON = {
    "car": {"mass_kg": 1407.0, "lambda_J_per_m": 670.0},
    "car_ev": {"mass_kg": 1900.0, "lambda_J_per_m": 584.0},
    "bus": {"lambda_J_per_m": 5760.0, "occupancy": 13.6},
    "train": {"mass_kg": 157000.0, "lambda_J_per_m": 47300.0},
    "tram": {"mass_kg": 40000.0 + 30 * PASSENGER_KG, "lambda_J_per_m": 15300.0, "occupancy": 30.0},
}
MEASURED = {
    "barcelone": {**COMMON, "metro": {"mass_kg": 157000.0 + 120 * PASSENGER_KG, "lambda_J_per_m": 33400.0,
                                      "occupancy": 120.0}},
    "madrid": {**COMMON, "metro": {"lambda_J_per_m": 30500.0}},
}
MODES = ["car", "car_ev", "moto", "bus", "rail", "mix"]


def params(cfg, city, measured: bool):
    par = bmr.central_params(cfg)
    if measured:
        for v, kv in MEASURED[city].items():
            for k, x in kv.items():
                lo, _, hi = cfg["modes"][v][k]
                assert lo <= x <= hi, f"{city} {v} {k}={x} outside [{lo}, {hi}]"
                par[v][k] = x
    return par


def margins(city):
    cfg = json.loads(btm.CONFIGS[city].read_text())
    S = pd.read_csv(btm.output_root(OD) / city / "embedding_sums.csv").set_index("filename")
    keep = [f for f in S.index if btm.family_of(f) not in ("scale_down",)]
    S = S.loc[keep]
    shares = bmr.modal_shares(cfg)
    rows = []
    for a in [int(x) for x in cfg["rail_network"]["max_access_m"]]:
        for label, measured in (("central", False), ("measured", True)):
            par = params(cfg, city, measured)
            E = pd.DataFrame({f: btm.scenario_energy(S.loc[f], par, shares, a) for f in S.index}).T
            for m in MODES:
                is_min, _, _, best, margin = btm.observed_status(E[m])
                rows.append({"city": city, "access_m": a, "params": label, "mode": m,
                             "observed_is_min": bool(is_min), "closest": best, "margin_pct": margin})
    for v in cfg["modes"]:                         # r = m g (1 - eps) / lambda at measured values
        p = params(cfg, city, True)[v]
        rows.append({"city": city, "params": "measured", "mode": v,
                     "r": p["mass_kg"] * 9.81 * (1 - p["regen_eff"]) / p["lambda_J_per_m"]})
    return rows


def main():
    T = pd.DataFrame([r for c in MEASURED for r in margins(c)])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT, index=False)
    M = T.dropna(subset=["margin_pct"])
    print(M.pivot_table(index=["city", "access_m", "mode"], columns="params", values="margin_pct").round(2)
          .to_string())
    print("\nobserved embedding is the minimum in every case:", bool(M["observed_is_min"].all()))
    print(T.dropna(subset=["r"])[["city", "mode", "r"]].round(1).to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
