#!/usr/bin/env python3
"""Step 2 with an alternative DEM (DEM sensitivity test; Chicago and Dallas with USGS 3DEP bare earth).

Runs the unchanged step-2 code (pipeline_production/fine_grid_gravitational_work.analyze_graphs) on the
existing step-1 graphs of a city, with the same OD flows, OD draws (K, seed) and energy parameters, but
sampling elevations from another DEM (python/dem_3dep.py). Results go to a separate directory; the SRTM
results of the paper are not touched.

One Slurm task per embedding: task 0 computes the observed embedding; every other task copies the task-0
table and computes one transformed embedding with resume, so the observed embedding is computed once and
the OD draws are always taken on the observed graph.

Usage:
    python python/step2_alt_dem.py --city chicago --list
    python python/step2_alt_dem.py --city chicago --task 0          # observed embedding (first)
    python python/step2_alt_dem.py --city chicago --task 7
    python python/step2_alt_dem.py --city chicago --merge           # table + comparison with SRTM
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline_production"))
sys.path.insert(0, str(ROOT / "python"))
import fine_grid_gravitational_work as fg  # noqa: E402
from vehicle_ratio_sweep import main_ensemble_mask  # noqa: E402

DATA = ROOT / "data" / "data_processed"
OUT_ROOT = ROOT / "output" / "step2_alt_dem"
ORIGINAL = "graph_original.pkl"
E = "total_energy_on_3d_path_J"


def graph_list(city: str) -> list[Path]:
    files = fg.list_graph_files(DATA / city / "graphs_fine_grid")
    orig = [p for p in files if p.name == ORIGINAL]
    if not orig:
        raise SystemExit(f"{city}: {ORIGINAL} not found")
    return orig + [p for p in files if p.name != ORIGINAL]


def od_params(city: str) -> tuple[int, int]:
    ref = pd.read_csv(DATA / city / "graphs_fine_grid" / "fine_grid_gravitational_work.csv",
                      usecols=["od_samples", "od_seed"])
    return int(ref["od_samples"].iloc[0]), int(ref["od_seed"].iloc[0])


def run_task(city: str, tag: str, task: int) -> None:
    dem = DATA / city / "dem" / f"{city}_dem_{tag}.tif"
    if not dem.exists():
        raise SystemExit(f"missing DEM {dem} (python/dem_3dep.py)")
    files = graph_list(city)
    out = OUT_ROOT / city / tag
    out.mkdir(parents=True, exist_ok=True)
    k, seed = od_params(city)
    csv = out / f"task_{task:03d}.csv"
    if task == 0:
        todo = files[:1]
    else:
        base = out / "task_000.csv"
        if not base.exists():
            raise SystemExit(f"{base} missing: run task 0 first")
        if not csv.exists():
            shutil.copy(base, csv)
        todo = [files[0], files[task]]
    fg.analyze_graphs(todo, dem, output_csv=csv, city=city, resume=True, gc_every=1,
                      data_root=DATA, od_samples=k, od_seed=seed)


def merge(city: str, tag: str) -> None:
    out = OUT_ROOT / city / tag
    parts = sorted(out.glob("task_*.csv"))
    alt = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates("filename", keep="last")
    alt.to_csv(out / f"fine_grid_gravitational_work_{tag}.csv", index=False)
    ref = pd.read_csv(DATA / city / "graphs_fine_grid" / "fine_grid_gravitational_work.csv")
    ref = ref.drop_duplicates("filename", keep="last")
    rows = []
    for name, df in (("srtm", ref), (tag, alt)):
        df = df[main_ensemble_mask(df["filename"])]
        e0 = df.loc[df["filename"] == ORIGINAL, E].iloc[0]
        h0 = df.loc[df["filename"] == ORIGINAL, "vertical_gain_on_3d_path_m"].iloc[0]
        rows.append(df.assign(dem=name, dW_pct=100 * (df[E] / e0 - 1),
                              dH_pct=100 * (df["vertical_gain_on_3d_path_m"] / h0 - 1))
                    [["dem", "filename", "dW_pct", "dH_pct", "vertical_gain_on_3d_path_m"]])
    cmp_ = pd.concat(rows).pivot(index="filename", columns="dem", values=["dW_pct", "dH_pct"])
    cmp_.to_csv(out / "comparison_with_srtm.csv")
    done = set(alt["filename"])
    print(cmp_[cmp_.index.isin(done)].round(3).sort_values(("dW_pct", tag)).to_string())
    for name in ("srtm", tag):
        col = cmp_[("dW_pct", name)].loc[[f for f in cmp_.index if f in done and f != ORIGINAL]]
        print(f"{name}: min dW {col.min():.3f}% ({col.idxmin()}), cheaper than observed: {(col < -0.05).sum()}"
              f" of {len(col)}")
    print(f"embeddings with {tag}: {len(done)} of {len(graph_list(city))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--city", required=True)
    ap.add_argument("--dem-tag", default="3dep13")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--array-task", action="store_true", help="take --task from $SLURM_ARRAY_TASK_ID")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.list:
        for i, p in enumerate(graph_list(args.city)):
            print(i, p.name)
        return
    if args.merge:
        merge(args.city, args.dem_tag)
        return
    task = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.array_task else args.task
    if task is None:
        raise SystemExit("give --task, --array-task, --list or --merge")
    run_task(args.city, args.dem_tag, task)


if __name__ == "__main__":
    main()
