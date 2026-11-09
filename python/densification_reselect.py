#!/usr/bin/env python3
"""Fig. 7 cell selection with fixed, published criteria (reviewer point on relaxed criteria).

multi_city_settlement.py relaxes the horizontal tolerance (5 % -> 75 %) and the
minimum separation (5 km -> 0) until it finds 3 favourable + 2 unfavourable
cells. Here the selection is redone from its saved candidate table with fixed
criteria, with no relaxation:

* comparable cells: horizontal cost within +/- tol of the median (Methods: 10 %);
* favourable / unfavourable: lowest / highest altitudinal work among them;
* every selected pair at least `sep` metres apart.

A small grid of (tol, sep) values reports whether the conclusion depends on
these choices. If a combination cannot provide 3 + 2 cells it is reported as
infeasible, not relaxed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_city_settlement as mcs  # noqa: E402

DEFAULT = (0.05, 3750.0)                        # criteria actually used for Fig. 7 (to be stated in Methods)
TOLERANCES = [0.05, 0.10]
SEPARATIONS_M = [3750.0, 5000.0]
OUTPUT_DIR = mcs.OUTPUT_ROOT.parent / "densification_fixed_selection"


def select_fixed(results: pd.DataFrame, tol: float, sep: float):
    h = results["horizontal_cost_joules"]
    comparable = results[(h - h.median()).abs() <= tol * h.median()]
    fav = mcs._greedy_select_spaced(comparable, mcs.N_FAVORABLE, True, [], sep)
    unf = mcs._greedy_select_spaced(comparable, mcs.N_UNFAVORABLE, False, fav, sep)
    feasible = len(fav) == mcs.N_FAVORABLE and len(unf) == mcs.N_UNFAVORABLE
    return comparable, pd.DataFrame(fav), pd.DataFrame(unf), feasible


def summarize(city, results, tol, sep):
    comparable, fav, unf, feasible = select_fixed(results, tol, sep)
    row = {"city": city, "tolerance": tol, "min_separation_m": sep,
           "n_candidates": len(results), "n_comparable": len(comparable), "feasible": feasible}
    if not feasible:
        return row, None
    best = fav["total_work_joules"].min()
    row.update({
        "fav_cells": " ".join(fav["cell_id"]), "unf_cells": " ".join(unf["cell_id"]),
        "alt_fav_min_GJ": fav["vertical_work_joules"].min() / 1e9,
        "alt_unf_max_GJ": unf["vertical_work_joules"].max() / 1e9,
        "unf_total_excess_min_pct": 100 * (unf["total_work_joules"].min() / best - 1),
        "unf_total_excess_max_pct": 100 * (unf["total_work_joules"].max() / best - 1),
        "hor_spread_selected_pct": 100 * (lambda x: (x.max() - x.min()) / x.median())(
            pd.concat([fav, unf])["horizontal_cost_joules"]),
        "all_fav_below_all_unf_total": bool(fav["total_work_joules"].max() < unf["total_work_joules"].min()),
    })
    sel = pd.concat([fav.assign(scenario_type="favorable"), unf.assign(scenario_type="unfavorable")])
    return row, sel.assign(tolerance=tol, min_separation_m=sep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=mcs.OUTPUT_ROOT,
                    help="folder with <city>/<city>_all_candidate_densification_results.csv")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, selections = [], []
    for city in mcs.CITIES:
        f = args.results_root / city / f"{city}_all_candidate_densification_results.csv"
        if not f.exists():
            print(f"[{city}] no results yet ({f})")
            continue
        results = pd.read_csv(f, sep=";")
        for tol in TOLERANCES:
            for sep in SEPARATIONS_M:
                row, sel = summarize(city, results, tol, sep)
                rows.append(row)
                if sel is not None:
                    selections.append(sel.assign(city=city))
    summary = pd.DataFrame(rows)

    # stability of the selected cells with respect to the default criteria
    def cells(r):
        return set(f"{r.get('fav_cells', '')} {r.get('unf_cells', '')}".split()) if r["feasible"] else set()
    ref = {r["city"]: cells(r) for _, r in summary.iterrows()
           if (r["tolerance"], r["min_separation_m"]) == DEFAULT}
    summary["jaccard_vs_default"] = [
        len(cells(r) & ref.get(r["city"], set())) / max(len(cells(r) | ref.get(r["city"], set())), 1)
        if r["feasible"] and ref.get(r["city"]) else np.nan for _, r in summary.iterrows()]

    summary.to_csv(OUTPUT_DIR / "fixed_selection_summary.csv", index=False)
    if selections:
        pd.concat(selections).to_csv(OUTPUT_DIR / "fixed_selection_cells.csv", index=False)
    cols = ["city", "tolerance", "min_separation_m", "n_comparable", "feasible", "alt_fav_min_GJ", "alt_unf_max_GJ",
            "unf_total_excess_min_pct", "unf_total_excess_max_pct", "all_fav_below_all_unf_total",
            "jaccard_vs_default"]
    print(summary[[c for c in cols if c in summary]].to_string(index=False, float_format="%.2f"))
    print(f"\nDefault criteria: tolerance {DEFAULT[0]:.0%}, separation {DEFAULT[1]:.0f} m. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
