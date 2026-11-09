# Topolity

Code for the paper *"Mobility energetic balance frames spatial configuration of cities"* (Bellisardi,
Cano Suñén, Ruiz-Varona and Ramasco).

For each city, the observed road network is compared with translated, rotated and rescaled copies of itself
on the same terrain, under the same origin–destination demand. The comparison uses a simplified measure of
movement energy with a horizontal (route length) and an altitudinal (uphill gain) component.

`README_REPRODUCE.md` lists, for every result of the paper, the script that produces it.

## Repository structure

```
pipeline_production/    steps 1 and 2 of the pipeline (graphs and transformed embeddings; energies) and
                        the Slurm scripts used to run them on a cluster
python/                 all-city analyses, densification, transport-mode test (SI Note 9), figures, tests
python/config/          vehicle and modal-split parameters with their sources (Barcelona, Madrid)
supplementary/python/   builder of the Supplementary tables and figures
tools/python/           input preparation (FUA boxes, OD matrices)
notebooks/              exploratory notebooks
```

Study-wide rules (excluded cities, transformation ensemble, tie rule, elevation model per city) are defined
once in `python/study_config.py`.

## Pipeline

**Step 1** (`pipeline_production/dem_extractor_fine_grid.py`): road network of the functional urban area,
transformed embeddings in a local metric frame, land check of every node, elevations sampled every 10 m.

**Step 2** (`pipeline_production/fine_grid_gravitational_work.py`): OD-weighted energy on shortest paths
for every embedding, with origins and destinations drawn within cells (five seeded realisations).

```bash
python pipeline_production/dem_extractor_fine_grid.py --city santiago --step-meters 250 --num-points 10 \
    --rotation-angles -40 -30 -20 -15 -10 -5 -1 1 5 10 15 20 30 40 --use-fua --land-check full --resume
python pipeline_production/fine_grid_gravitational_work.py --city santiago --resume
```

## Environment and tests

```bash
conda env create -f environment.yml          # locked versions in requirements-lock.txt
conda activate topolity
export PYTHONNOUSERSITE=1
python -m pytest -q python/tests
(cd pipeline_production && python -m pytest -q tests)
```

## Data

Inputs are read from `data/` at the repository root (not versioned; a symbolic link is fine). Road
networks (OpenStreetMap), elevation models (SRTM, USGS 3DEP), population (WorldPop), functional urban areas
(GHSL) and the mobility data of the Spanish Ministry of Transport are public. The origin–destination
matrices derived from geolocated records were obtained under a data-sharing agreement and cannot be
redistributed; aggregated matrices and the result tables are available from the authors on request.

Step 1 downloads missing SRTM tiles from OpenTopography; set your own key with
`export OPENTOPOGRAPHY_API_KEY=...` or `--api_key`. Keys and local settings are never versioned
(`conf/`, `tools/conf/`, `pipeline_production/local.env`).

## Contact

Federico Bellisardi — fbellisardi@ifisc.uib-csic.es
