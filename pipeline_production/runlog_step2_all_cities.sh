#!/usr/bin/env bash
set -euo pipefail

# Launch Step-2 (gravitational work) one city per runlog job.
# This mirrors step-1 sharding and keeps RAM bounded.

ROOT="/home/fbellisardi/code/topolity"
DATA_ROOT="$ROOT/data/data_processed"
SCRIPT="$ROOT/pipeline_production/fine_grid_gravitational_work.py"

# Tunable defaults (override via env vars)
TIME_LIMIT="${TIME_LIMIT:-120:00}"
MEM_GB="${MEM_GB:-128}"
CPUS="${CPUS:-4}"
LOW_MEMORY="${LOW_MEMORY:-1}"
SAVE_SEGMENTS="${SAVE_SEGMENTS:-0}"
DRY_RUN="${DRY_RUN:-1}"                     # 1=print only, 0=submit
SKIP_IF_WORK_CSV="${SKIP_IF_WORK_CSV:-0}"   # 1=skip city if fine_grid_gravitational_work.csv exists
REQUIRE_STATS="${REQUIRE_STATS:-1}"         # 1=submit only if fine_grid_stats.csv exists
REQUIRE_STEP1_COMPLETE="${REQUIRE_STEP1_COMPLETE:-1}"   # 1=require transformed pickles from step1
MIN_TRANSFORMED_PICKLES="${MIN_TRANSFORMED_PICKLES:-1}" # minimum transformed graph_*.pkl (excluding original)

# Avoid BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Root: $ROOT"
echo "Data root: $DATA_ROOT"
echo "Script: $SCRIPT"
echo "Resources: t=$TIME_LIMIT mem=${MEM_GB}GB cpus=$CPUS"
echo "Options: low_memory=$LOW_MEMORY save_segments=$SAVE_SEGMENTS"
echo "Filters: skip_if_work_csv=$SKIP_IF_WORK_CSV require_stats=$REQUIRE_STATS require_step1_complete=$REQUIRE_STEP1_COMPLETE min_transformed_pickles=$MIN_TRANSFORMED_PICKLES"
echo "Dry run: $DRY_RUN"

action_submit() {
  local city="$1"
  local city_dir="$DATA_ROOT/$city"
  local data_useful="$city_dir/data_useful.csv"
  local stats="$city_dir/graphs_fine_grid/fine_grid_stats.csv"
  local fine_grid_dir="$city_dir/graphs_fine_grid"
  local out_csv="$city_dir/graphs_fine_grid/fine_grid_gravitational_work.csv"

  # Valid city folder
  [[ -f "$data_useful" ]] || return 0

  # Step-2 should typically run after step-1 stats exist.
  if [[ "$REQUIRE_STATS" == "1" && ! -f "$stats" ]]; then
    echo "[skip] $city: missing fine_grid_stats.csv (step-1 not ready)"
    return 0
  fi

  if [[ "$REQUIRE_STEP1_COMPLETE" == "1" ]]; then
    local transformed_count
    transformed_count=$(find "$fine_grid_dir" -maxdepth 1 -type f -name 'graph_*.pkl' ! -name 'graph_original.pkl' | wc -l | tr -d ' ')
    if [[ "$transformed_count" -lt "$MIN_TRANSFORMED_PICKLES" ]]; then
      echo "[skip] $city: step-1 not complete (transformed graphs=$transformed_count, required>=$MIN_TRANSFORMED_PICKLES)"
      return 0
    fi
  fi

  if [[ "$SKIP_IF_WORK_CSV" == "1" && -f "$out_csv" ]]; then
    echo "[skip] $city: fine_grid_gravitational_work.csv already exists"
    return 0
  fi

  local cmd="python $SCRIPT"
  cmd+=" --data-root $DATA_ROOT"
  cmd+=" --city $city --resume"

  if [[ "$LOW_MEMORY" == "1" ]]; then
    cmd+=" --low-memory"
  fi
  if [[ "$SAVE_SEGMENTS" == "1" ]]; then
    cmd+=" --save-segments"
  fi

  local tag="fgw_${city}"
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
