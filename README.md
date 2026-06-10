# Topolity

Code for the paper *"Mobility energetic balance frames spatial configuration of cities"* (Bellisardi et al.).

The core hypothesis is that real cities tend to minimize the total gravitational work required for mobility: the original street network embedding sits at — or very close to — the energy minimum among all rigid translations, rotations, and scalings of the same graph.

## Repository structure

```
pipeline_scripts/       legacy pipeline scripts (exploration)
pipeline_production/    production pipeline used for the paper results
python/                 analysis and figure scripts
notebooks/              exploratory notebooks
tools/                  utilities (bbox extraction, map rendering)
```

## Pipeline overview

The analysis runs in two steps per city:

**Step 1 — graph generation** (`pipeline_production/dem_extractor_fine_grid.py`)

Downloads the OSM street network, generates transformed variants (translations, rotations, NS/EW scalings), assigns DEM elevation to each node, and flags sea-covered cells.

**Step 2 — energy computation** (`pipeline_production/fine_grid_gravitational_work.py`)

For each graph variant computes the OD-weighted gravitational work along shortest paths, using the population grid and a gravity model for trip distribution.

**Post-processing** (`pipeline_production/generate_combined_work_figures.py`, `python/exception_analysis.py`)

Aggregates results, identifies counterexamples, and generates paper figures.

## Quickstart

```bash
conda activate geo_flow

# single city, step 1
python pipeline_production/dem_extractor_fine_grid.py \
    --city santiago --step-meters 500 --num-points 5 \
    --rotation-angles -20 -15 -10 -5 5 10 15 20 \
    --workers 4 --seed 42 --use-fua --land-check full --resume

# single city, step 2
python pipeline_production/fine_grid_gravitational_work.py \
    --city santiago --workers 4 --resume

# all cities (cluster)
bash pipeline_production/runlog_step1_all_cities.sh
bash pipeline_production/runlog_step2_all_cities.sh
```

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: `osmnx`, `networkx`, `networkit`, `geopandas`, `rasterio`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `tqdm`.

## Data

Raw data (OSM graphs, DEM files, WorldPop rasters, OD matrices) are stored outside the repository under `data/data_processed/<city>/` and are not version-controlled.

## Analysis scripts

| Script | Description |
|--------|-------------|
| `python/exception_analysis.py` | Identify cities where a lower-energy variant exists |
| `python/lower_energy.py` | Detailed analysis of counterexample cities |
| `python/world_cities_map.py` | World map figure for the paper |
| `python/multi_city_settlement.py` | Terrain-aware densification analysis across cities |
| `python/comparative_settlement_analysis.py` | Single-city settlement comparison |

## Authors

Federico Bellisardi — bellisardi@gmail.com - fbellisardi@ifisc.uib-csic.es
