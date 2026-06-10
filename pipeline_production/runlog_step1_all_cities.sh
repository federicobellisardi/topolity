#!/usr/bin/env bash
set -euo pipefail

# Launch Step-1 (graph extraction/transforms) on all cities in data/data_processed using runlog.
# Safe defaults for coastal cities and low RAM.

ROOT="/home/fbellisardi/code/topolity"
DATA_ROOT="$ROOT/data/data_processed"
SCRIPT="$ROOT/pipeline_production/dem_extractor_fine_grid.py"

# Tunable defaults (override via env vars)
TIME_LIMIT="${TIME_LIMIT:-300:00}"
MEM_GB="${MEM_GB:-128}"
CPUS="${CPUS:-4}"
WORKERS="${WORKERS:-1}"
STEP_METERS="${STEP_METERS:-500}"
NUM_POINTS="${NUM_POINTS:-5}"
ROT_ANGLES="${ROT_ANGLES:- -20 -15 -10 -5 5 10 15 20}"
NS_SCALES="${NS_SCALES:-1.02 1.05 1.08 1.12}"
EW_SCALES="${EW_SCALES:-1.02 1.05 1.08 1.12}"
LAND_CHECK="${LAND_CHECK:-full}"   # sample|full
LOW_MEMORY="${LOW_MEMORY:-1}"      # 1=enabled
USE_FUA="${USE_FUA:-1}"            # 1=enabled
DEM_MODE="${DEM_MODE:-tree}"       # auto|tree|raster (tree keeps baseline reproducibility)
DRY_RUN="${DRY_RUN:-1}"            # 1=print only, 0=submit
SKIP_IF_STATS="${SKIP_IF_STATS:-0}" # 1=skip cities with fine_grid_stats.csv already present

# Keep numerical libs from oversubscribing memory.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Root: $ROOT"
echo "Data root: $DATA_ROOT"
echo "Script: $SCRIPT"
echo "Resources: t=$TIME_LIMIT mem=${MEM_GB}GB cpus=$CPUS workers=$WORKERS"
echo "Transforms: step=$STEP_METERS points=$NUM_POINTS rotations=[$ROT_ANGLES] ns=[$NS_SCALES] ew=[$EW_SCALES]"
echo "Safety: land_check=$LAND_CHECK low_memory=$LOW_MEMORY use_fua=$USE_FUA dem_mode=$DEM_MODE"
echo "Skip if stats exists: $SKIP_IF_STATS"
echo "Dry run: $DRY_RUN"

action_submit() {
  local city="$1"
  local city_dir="$DATA_ROOT/$city"
  local data_useful="$city_dir/data_useful.csv"
  local stats="$city_dir/graphs_fine_grid/fine_grid_stats.csv"

  # Skip folders that are not valid cities.
  [[ -f "$data_useful" ]] || return 0

  # With dilations introduced later, many cities may need new variants even if stats exists.
  # Default behavior is to submit and rely on --resume for incremental completion.
  if [[ "$SKIP_IF_STATS" == "1" && -f "$stats" ]]; then
    echo "[skip] $city: fine_grid_stats.csv already exists (SKIP_IF_STATS=1)"
    return 0
  fi

  local cmd="python -u $SCRIPT"
  cmd+=" --base-path $DATA_ROOT"
  cmd+=" --city $city"
  cmd+=" --step-meters $STEP_METERS --num-points $NUM_POINTS"
  cmd+=" --rotation-angles $ROT_ANGLES"
  cmd+=" --ns-scale-factors $NS_SCALES --ew-scale-factors $EW_SCALES"
  cmd+=" --workers $WORKERS --seed 42 --resume"
  cmd+=" --dem-mode $DEM_MODE"

  if [[ "$LOW_MEMORY" == "1" ]]; then
    cmd+=" --low-memory"
  fi
  if [[ "$USE_FUA" == "1" ]]; then
    cmd+=" --use-fua"
  fi
  cmd+=" --land-check $LAND_CHECK"

  local tag="fg_${city}"
  local runlog_cmd="runlog -t $TIME_LIMIT -m $MEM_GB -c $CPUS -j $tag $cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$runlog_cmd"
  else
    echo "[submit] $city"
    eval "$runlog_cmd"
  fi
}

while IFS= read -r city; do
  action_submit "$city"
done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

echo "Done."
