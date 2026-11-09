"""The main transformation ensemble and the tie rule are shared by every classification script."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vehicle_ratio_sweep as vrs  # noqa: E402


def test_main_ensemble_excludes_contractions_and_short_translations():
    names = pd.Series(["graph_original.pkl", "graph_scale_ns_1p05.pkl", "graph_scale_ew_0p920.pkl",
                       "graph_scale_ns_0p80.pkl", "graph_trans_250m_a+090.pkl", "graph_trans_100m_a+090.pkl",
                       "graph_rot_+10p00deg.pkl"])
    keep = vrs.main_ensemble_mask(names).tolist()
    assert keep == [True, True, False, False, True, False, True]


def test_tie_rule_is_005_percent():
    assert vrs.TIE_TOL == 0.0005
    e0 = 100.0
    assert e0 <= (1 + vrs.RTOL) * 99.951      # 0.049 % below: tie, observed still minimum
    assert not e0 <= (1 + vrs.RTOL) * 99.9    # 0.1 % below: displaced lower
