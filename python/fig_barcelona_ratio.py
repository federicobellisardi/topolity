#!/usr/bin/env python3
"""SI figure: Barcelona, relative energy of every transformed embedding as a function of r."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_theory as rt  # noqa: E402
from paper_style import BAND, BLUE, GRID, INK2, MUTED, save  # noqa: E402
from vehicle_ratio_sweep import L_COL, ORIGINAL, R_PAPER, RTOL, U_COL, load_city  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/data_processed/barcelone/graphs_fine_grid/fine_grid_gravitational_work.csv"
RANGES = ROOT / "output/vehicle_ratio_sweep/graphs_fine_grid/vehicle_ratio_ranges.csv"
OUT = ROOT / "paper_v2/supp_images/transport_modes/barcelona_ratio"
OD = sys.argv[sys.argv.index("--od") + 1] if "--od" in sys.argv else "twitter"


def load_mitma():
    """Road sums of the MITMA run (3-D length and climbing of the road paths), same ensemble."""
    from vehicle_ratio_sweep import main_ensemble_mask
    S = pd.read_csv(ROOT / f"output/transform_multimodal_{OD}/barcelone/embedding_sums.csv")
    S = S[main_ensemble_mask(S["filename"])]
    return S.assign(**{L_COL: S["road_L"] / S["F"], U_COL: S["road_Hu"] / S["F"]})


def pretty(fname):
    import re
    m = re.match(r"graph_trans_(\d+)m_a([+-]\d+)\.pkl", fname)
    if not m:
        return fname
    return f"{m.group(1)} m " + {0: "east", 90: "north", 180: "west", 270: "south"}.get(int(m.group(2)), m.group(2))


def main():
    mitma = OD != "twitter"
    df = load_mitma() if mitma else load_city(TABLE)
    o = df[df["filename"] == ORIGINAL].iloc[0]
    var = df[df["filename"] != ORIGINAL]
    lo, hi = pd.read_csv(RANGES).set_index("vehicle").loc["all_vehicles", ["r_low", "r_high"]]
    iv = rt.minimizer_interval(o[L_COL], o[U_COL], var[L_COL], var[U_COL], rtol=RTOL)
    assert iv is not None and iv[0] <= lo and hi <= iv[1], "observed embedding not minimal on the range"

    r = np.linspace(0, 90, 400)
    E0 = o[L_COL] + r * o[U_COL]
    excess = 100 * ((var[L_COL].to_numpy()[:, None] + np.outer(var[U_COL], r)) / E0 - 1)
    best = np.argmin(excess[:, np.searchsorted(r, R_PAPER)])

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 0.97, "range of r of all vehicles", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=INK2, fontsize=6.5)
    ax.axhline(0, color=INK2, lw=0.8)
    for k in range(len(excess)):
        if k != best:
            ax.plot(r, excess[k], color=MUTED, lw=0.6, alpha=0.7)
    ax.plot(r, excess[best], color=BLUE, lw=1.4)
    ax.text(88, excess[best][-1] + 1.5, "closest alternative:\n" + pretty(var["filename"].iloc[best]),
            color=BLUE, fontsize=6.5, ha="right", va="bottom")
    ax.axvline(R_PAPER, color=INK2, lw=0.6, ls="--")
    ax.text(R_PAPER + 1, 0.60, r"$r_0$ (paper)", transform=ax.get_xaxis_transform(), color=INK2, fontsize=6.5)
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=min(-0.5, excess.min() - 0.2))
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_xlabel(r"$r = mg(1-\varepsilon)/\lambda$ (dimensionless)")
    ax.set_ylabel("energy relative to observed (%)")
    save(fig, OUT.with_name(f"{OUT.name}_{OD}") if mitma else OUT)
    print(f"observed minimal for r in [{iv[0]:.2f}, {iv[1]:.2f}]; vehicle range [{lo:.1f}, {hi:.1f}]; "
          f"min excess over range {excess[:, (r >= lo) & (r <= hi)].min():.2f} %")


if __name__ == "__main__":
    main()
