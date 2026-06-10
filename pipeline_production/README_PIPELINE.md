# Topolity Pipeline Production

Guida operativa completa per la pipeline fine-grid:

1. Step 1: generazione grafi trasformati + quote DEM
2. Step 2: calcolo lavoro gravitazionale OD-weighted
3. Post-processing: generazione delle 2 figure finali del notebook combined work

La pipeline e pensata per cluster (runlog) e include controlli di robustezza per evitare job inutili.

## Struttura Script

- `dem_extractor_fine_grid.py`: Step 1 (traslazioni, rotazioni, dilatazioni NS/EW, controlli terra/mare, quote)
- `fine_grid_gravitational_work.py`: Step 2 (work calculation su grafi fine-grid, resume, low-memory)
- `runlog_step1_all_cities.sh`: submit batch Step 1, una citta per job
- `runlog_step2_all_cities.sh`: submit batch Step 2, una citta per job
- `generate_combined_work_figures.py`: genera le ultime 2 figure di `combined_work_analysis.ipynb`
- `run_pipeline.sh`: orchestrazione locale/sequenziale (utile per test controllati)
- `validate_pipeline.sh`: validazione rapida file/paths/sintassi
- `quickstart.sh`: esempio di avvio veloce

## Prerequisiti

- Ambiente conda `geo_flow`
- Dati citta in `data/data_processed/<city>/`
- Per ogni citta valida: presenza di `data_useful.csv`
- Dipendenze Python installate (`requirements.txt` del progetto)

Comando ambiente:

```bash
cd /home/fbellisardi/code/topolity
conda activate geo_flow
```

## Pipeline End-to-End

### 1) Step 1 - Generazione grafi trasformati

Esempio singola citta:

```bash
python pipeline_production/dem_extractor_fine_grid.py \
  --city santiago \
  --step-meters 500 \
  --num-points 5 \
  --rotation-angles -20 -15 -10 -5 5 10 15 20 \
  --ns-scale-factors 1.02 1.05 1.08 1.12 \
  --ew-scale-factors 1.02 1.05 1.08 1.12 \
  --workers 1 \
  --seed 42 \
  --use-fua \
  --land-check full \
  --dem-mode tree \
  --low-memory \
  --resume
```

Output atteso per citta:

- `data/data_processed/<city>/graphs_fine_grid/graph_original.pkl`
- `data/data_processed/<city>/graphs_fine_grid/graph_*.pkl` (varianti)
- `data/data_processed/<city>/graphs_fine_grid/fine_grid_stats.csv`

### 2) Step 2 - Lavoro gravitazionale

Esempio singola citta:

```bash
python pipeline_production/fine_grid_gravitational_work.py \
  --city santiago \
  --low-memory \
  --resume
```

Esempio multi-citta:

```bash
python pipeline_production/fine_grid_gravitational_work.py \
  --cities santiago madrid barcelona \
  --low-memory \
  --resume
```

Output atteso:

- `data/data_processed/<city>/graphs_fine_grid/fine_grid_gravitational_work.csv`

### 3) Figure finali (ultime 2 del notebook)

Script dedicato:

```bash
python pipeline_production/generate_combined_work_figures.py --auto-from-step2
```

Comportamento:

- processa le citta che hanno gia `fine_grid_gravitational_work.csv`
- genera per ogni citta:
  - `<city>_combined_work_comparison.png` e `.pdf`
  - `<city>_boxplot_by_transformation.png` e `.pdf`
- output default in:
  - `paper/images/figPenultima/test/<city>/`

Opzioni utili:

```bash
# citta esplicite
python pipeline_production/generate_combined_work_figures.py \
  --cities santiago madrid \
  --source-filter graphs

# includere anche scale
python pipeline_production/generate_combined_work_figures.py \
  --auto-from-step2 \
  --include-scale
```

## Esecuzione su Cluster con runlog

### Step 1 batch

Dry-run (default):

```bash
cd /home/fbellisardi/code/topolity
bash pipeline_production/runlog_step1_all_cities.sh
```

Submit reale:

```bash
DRY_RUN=0 bash pipeline_production/runlog_step1_all_cities.sh
```

Variabili principali (override via env):

- `TIME_LIMIT`, `MEM_GB`, `CPUS`, `WORKERS`
- `STEP_METERS`, `NUM_POINTS`, `ROT_ANGLES`
- `NS_SCALES`, `EW_SCALES`
- `LOW_MEMORY`, `USE_FUA`, `LAND_CHECK`, `DEM_MODE`
- `SKIP_IF_STATS`

### Step 2 batch

Dry-run (default):

```bash
cd /home/fbellisardi/code/topolity
bash pipeline_production/runlog_step2_all_cities.sh
```

Submit reale:

```bash
DRY_RUN=0 bash pipeline_production/runlog_step2_all_cities.sh
```

Controllo readiness integrato in `runlog_step2_all_cities.sh`:

- richiede `fine_grid_stats.csv` (se `REQUIRE_STATS=1`)
- richiede almeno N grafi trasformati `graph_*.pkl` escluso `graph_original.pkl`
  - attivato con `REQUIRE_STEP1_COMPLETE=1` (default)
  - soglia con `MIN_TRANSFORMED_PICKLES` (default `1`)

Esempio piu restrittivo:

```bash
DRY_RUN=0 REQUIRE_STEP1_COMPLETE=1 MIN_TRANSFORMED_PICKLES=5 \
  bash pipeline_production/runlog_step2_all_cities.sh
```

Altri flag utili:

- `SKIP_IF_WORK_CSV=1`: salta citta gia elaborate
- `LOW_MEMORY=1`: profilo memoria ridotta
- `SAVE_SEGMENTS=1`: salva anche output segmenti

## Validazione Rapida

```bash
cd /home/fbellisardi/code/topolity/pipeline_production
bash validate_pipeline.sh
```

## Sequenza Consigliata

1. `bash pipeline_production/runlog_step1_all_cities.sh` (dry-run)
2. `DRY_RUN=0 bash pipeline_production/runlog_step1_all_cities.sh`
3. attesa/completamento step1
4. `bash pipeline_production/runlog_step2_all_cities.sh` (dry-run)
5. `DRY_RUN=0 bash pipeline_production/runlog_step2_all_cities.sh`
6. `python pipeline_production/generate_combined_work_figures.py --auto-from-step2`

## Note Operative

- Coastal cities possono produrre poche varianti valide (controllo terra/mare).
- In caso di OOM: aumentare `MEM_GB`, mantenere `LOW_MEMORY=1`, ridurre `WORKERS`.
- Per continuita e ripartenza, usare sempre `--resume` dove disponibile.
- Per riproducibilita rispetto a baseline, usare `--dem-mode tree`.
