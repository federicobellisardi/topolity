# Reproducing the results of the paper

This file lists, for every result of the paper, the script that produces it, its inputs and its outputs.
All paths are relative to the repository root. Figure outputs are written to `paper_v2/` (the manuscript
folder, not distributed with the code).

All results come from the corrected (v2) pipeline, complete for the 56 cities of the study.

## 1. Environment

The environment is locked: `requirements-lock.txt` pins every package imported by the analysis code and
all its dependencies (45 packages) to the versions used for the results; `environment.yml` adds Python
3.13.2 from conda-forge.

```bash
conda env create -f environment.yml
conda activate topolity
export PYTHONNOUSERSITE=1                 # ignore packages installed with pip --user
export OMP_NUM_THREADS=2                  # the scripts use numpy/scipy threads
```

`requirements.txt` lists only the direct dependencies.

Tests (about 5 s):

```bash
python -m pytest -q python/tests                          # ranking theory, transit networks, multimodal, rules
(cd pipeline_production && python -m pytest -q tests)     # CRS frames, 10 m sampling, DEM validity
```

`python/tests/test_paper_classification.py` recomputes the classification of the 56 cities from their
result tables and checks it against Supplementary Table 1 (categories, displaced configurations, energy
changes). The result tables are not distributed with the code (available from the authors on request);
without them this test is skipped.

## 2. Data

Inputs are stored under `data/` at the repository root and are not version-controlled. All scripts locate
the repository from their own position, so no path needs editing; if the data live elsewhere, make `data`
a symbolic link to them (`ln -s /path/to/data data`).

| Input | Location | Origin | Shareable |
|---|---|---|---|
| Street networks | downloaded by step 1 | OpenStreetMap (ODbL) | yes |
| DEM | `data/data_processed/<city>/dem/<city>_dem.tif` | SRTM | yes |
| Population | `data/worldpop/raw/2020/` | WorldPop 2020 | yes |
| FUA boundaries, city boxes | `data/metropolis.json`, FUA layer | GHSL / repository | yes |
| OD flows (1 km cells) | `data/data_processed/<city>/<city>_basic_model/1000_cells/` | geolocated records under a data-sharing agreement | **no** (aggregated matrices on request) |
| Barcelona working-day OD (gravity fit) | `multi_city_settlement.city_paths("barcelone")["od_working"]` | same | **no** |
| Transit networks | `data/data_processed/<city>/transit/osm_rail_routes_v2.json`, `osm_bus_routes.json` | OpenStreetMap route relations via Overpass | yes (cached; do not re-download, Overpass is unreliable) |
| Transit parameters | `python/config/{barcelona,madrid}_multimodal_params.json`, `data/parameters/transport_parameters_sources.csv` | operator reports, cited in the files and in `paper_v2/data_sources_round2.bib` | yes |

## 3. Pipeline: steps 1 and 2 (all cities, cluster)

The Slurm scripts in `pipeline_production/` read the Python interpreter (`PY`) and the submission helper
(`RUNLOG`) from `pipeline_production/local.env` (copy `local.env.example`); they find the repository from
their own position or, inside a job, from the submission directory (`TOPOLITY_ROOT` overrides both).

| Step | Script | Output per city |
|---|---|---|
| 1. Graphs and transformed embeddings (translations, rotations, scalings in a local metric frame; DEM sampled every 10 m) | `pipeline_production/dem_extractor_fine_grid.py` | `data/data_processed/<city>/graphs_fine_grid/graph_*.pkl` |
| 2. OD-weighted work on 3-D shortest paths (K = 5 seeded OD draws within cells) | `pipeline_production/fine_grid_gravitational_work.py` | `.../graphs_fine_grid/fine_grid_gravitational_work.csv` |

The revision run is submitted with `pipeline_production/submit_rerun_v2.sh` (Slurm; `DRY_RUN=1` by default
prints the commands; the grid of transformations is defined at the top of the script). Step 2 takes up to
about 120 h for the largest cities. Previous (v1) results are archived as `graphs_fine_grid_v1/`.

Single city, e.g.:

```bash
python pipeline_production/dem_extractor_fine_grid.py --city santiago --step-meters 250 --num-points 10 \
    --rotation-angles -40 -30 -20 -15 -10 -5 -1 1 5 10 15 20 30 40 --use-fua --land-check full --resume
python pipeline_production/fine_grid_gravitational_work.py --city santiago --resume
```

## 4. All-city analyses (read the step-2 tables; minutes)

Study-wide rules live in one module, `python/study_config.py`, read by every classification script and by the
SI builder: cities excluded from the study, the minimum translation, the tie rule (0.05 %), the constrained
override, and the cities whose step 2 uses the USGS 3DEP bare-earth DEM instead of SRTM (Chicago, Dallas;
their 3DEP tables are produced by `python/step2_alt_dem.py` and replace the SRTM ones once merged).

Run after all step-2 jobs have finished. Archive the existing output directory first.

| Result | Command | Output |
|---|---|---|
| City classification, Fig. 5, exception panels, SI Note 4 | `python python/lower_energy.py` | `output/lower_energy_analysis/` |
| Robustness to the vehicle ratio r (all cities, road vehicles) | `python python/vehicle_ratio_sweep.py --grid-dir graphs_fine_grid` | `output/vehicle_ratio_sweep/graphs_fine_grid/` |
| Regenerative braking, net rise Z_c | `python python/net_rise.py` | `output/net_rise/graphs_fine_grid/` |
| Distance-invariance audit | `python python/distance_invariance_audit.py` | `reports/reviewer1/tables/distance_invariance_*.csv`, `reports/reviewer1/figures/distance_invariance.*` |
| DEM-noise sensitivity (SI) | `python python/dem_noise_sensitivity.py --city <city>`; then `--summarize` | see script header |
| Constrained cities: water body, nodes in water per direction (Supplementary Table 2) | `python python/constrained_audit.py --all` | `reports/reviewer1/tables/constrained_audit_*.csv` |
| USGS 3DEP bare-earth DEM for Chicago and Dallas | `python python/dem_3dep.py --city <city>` | `data/data_processed/<city>/dem/<city>_dem_3dep13.tif` |
| Step 2 with 3DEP (cluster; then `--merge`) | `CITY=<city> bash pipeline_production/submit_step2_alt_dem.sh` | `output/step2_alt_dem/<city>/3dep13/` |
| Fig. 2 (cities, DEMs and road networks) | `python python/fig2_main.py` | `paper_v2/images/fig2.*` |
| Fig. 4 (Madrid, Santiago) | `python python/fig4_main.py` | `paper_v2/images/fig4.*` |
| Fig. 6 (Bogotá) | `python python/fig6_main.py` | `paper_v2/images/fig6.*` |
| SI contraction profiles (`shrink.pdf`) | `python python/fig_shrink_si.py` | `paper_v2/supp_images/shrink.*` |
| SI per-city profiles (Supplementary Figs 2-3) | `python python/fig_si_profiles.py --per-part 12` | `paper_v2/supp_images/figures/supporting_profiles/profiles_grid_{1,2}.*` |

`vehicle_ratio_sweep.py` and `net_rise.py` exclude Beijing and Shanghai by default (removed from the study).
`net_rise.py` caches Z_c per city in its output directory; Z_c depends only on the graphs, so delete the
cache of a city only if its graphs change.

## 5. Densification (Fig. 7) and transport modes (SI Note 9)

| Result | Command | Output |
|---|---|---|
| Fig. 7 (car), candidate cells, gravity-model fit | `python python/multi_city_settlement.py` (city `barcelone`) | `output/terrain_aware_densification/barcelone/` |
| Fig. 7 (paper file, redrawn from the saved selection) | `python python/fig7_main.py` | `paper_v2/images/fig7.{pdf,png}` |
| Fixed-criteria reselection check | `python python/densification_reselect.py` | see script header |
| Transformations by mode, per embedding (cluster, one task per embedding) | `pipeline_production/submit_transform_multimodal.sh` (`CITY=barcelone` or `madrid`) | `output/transform_multimodal/<city>/sums/*.json` |
| Transformations by mode, merge and worst case (seconds) | `python python/barcelona_transform_multimodal.py --city <city> --merge` | `output/transform_multimodal/<city>/observed_minimum_by_mode.csv` |
| Densification by mode (rerouting; about 30 min) | `python python/barcelona_multimodal_robustness.py` | `output/barcelona_multimodal_robustness/` |
| Densification worst case only (from saved sums) | `python python/barcelona_multimodal_robustness.py --worst-case` | `.../robustness_worst_case.csv` |
| MITMA open OD flows for SI Note 9 (district trips -> 1 km cells; all trips and >= 2 km) | `python python/mitma_od.py` | `data/data_processed/<city>/<city>_mitma_model/1000_cells/` |
| Transformations by mode with MITMA OD (SI Note 9) | `OD=mitma_ge2km CITY=<city> bash pipeline_production/submit_transform_multimodal.sh`, then `--merge --od mitma_ge2km`; figures with `fig_barcelona_modes.py --od mitma_ge2km`, `fig_madrid.py --od mitma_ge2km` | `output/transform_multimodal_mitma_ge2km/<city>/` |
| Margins at the measured vehicle parameters (Supplementary Table 3) | `python python/measured_params_check.py` | `reports/reviewer1/tables/measured_params_margins.csv` |
| Breakdown thresholds of the failing worst cases | `python python/multimodal_breakdown.py` | `reports/reviewer1/tables/multimodal_breakdown.csv` |
| SI figures `barcelona_ratio`, `barcelona_modes`, `madrid_ratio`, `madrid_modes`, `barcelona_densification_modes` | `python python/fig_barcelona_ratio.py`, `fig_barcelona_modes.py`, `fig_madrid.py`, `fig_barcelona_densification_modes.py` | `paper_v2/supp_images/transport_modes/` |

Vehicle parameters enter only the merge and the worst-case steps. After a parameter change, edit the
config and rerun the merge, `barcelona_multimodal_robustness.py`, `multimodal_breakdown.py` and the figure
scripts; the per-embedding sums do not need to be recomputed.

## 6. Other figures

| Figure | How it is produced |
|---|---|
| SI energy profiles (`figures/supporting_profiles/`), counterexample maps (`figures/counterexamples/`), `land_false_counts`, overview figure (`figures/overview/`) | `python supplementary/python/build_gravitational_supplementary.py` (`--only supporting|counterexamples`, `--cities ...`, `--output-root <dir>`; default output `figures/supplementary/`). Rerun with the final classification and copy the figures to `paper_v2/supp_images/figures/`. |
| Fig. 1 (concept) | does not depend on the corrected pipeline |
| Fig. 3 (Lorenz curves, Barcelona work map) | depends on the corrected pipeline; generating script not in the repository |
| `supp_osm_comparison`, DEM-validation scatter plots | round-1 figures, unchanged; generating script not identified |

## Known limitations

- Old code, superseded manuscript versions and backups are archived in `.bak/` (not versioned), with their
  original relative paths.

- No path needs editing: scripts locate the repository from their own position and read the data from
  `data/` at its root (a symbolic link is fine).
- The OD flows cannot be redistributed, so steps 2, 4 and 5 cannot be rerun from public data alone; the
  result tables are available from the authors on request. The transport-mode test of SI Note 9 uses the
  open data of the Spanish Ministry of Transport (`python/mitma_od.py`).
