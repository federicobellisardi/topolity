#!/bin/bash
# Pipeline orchestration script for topolity fine-grid analysis
# Usage: ./run_pipeline.sh [cities...]
# Example: ./run_pipeline.sh santiago madrid barcelona

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs/pipeline_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# Default cities if none provided
CITIES="${@:-santiago}"
WORKERS="${PIPELINE_WORKERS:-1}"
LOW_MEMORY="${PIPELINE_LOW_MEMORY:-1}"
SAVE_SEGMENTS="${PIPELINE_SAVE_SEGMENTS:-0}"

# Limit BLAS/OpenMP threads to avoid memory oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "====================================================================="
echo "TOPOLITY FINE-GRID PIPELINE"
echo "====================================================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Cities: $CITIES"
echo "Workers: $WORKERS"
echo "Low-memory profile: $LOW_MEMORY"
echo "Log directory: $LOG_DIR"
echo "====================================================================="

# Activate conda environment
echo "[1/2] Activating conda environment..."
if ! conda activate geo_flow 2>/dev/null; then
    echo "Warning: Could not activate geo_flow, assuming it's in PATH"
fi

# Step 1: Generate transformed graphs for each city
echo ""
echo "====================================================================="
echo "[STEP 1] Generating transformed graphs"
echo "====================================================================="

for city in $CITIES; do
    echo ""
    echo "Processing city: $city"
    city_log="${LOG_DIR}/step1_graphs_${city}.log"
    
    python "${SCRIPT_DIR}/dem_extractor_fine_grid.py" \
        --city "$city" \
        --step-meters 500 \
        --num-points 5 \
        --rotation-angles -20 -15 -10 -5 5 10 15 20 \
        --ns-scale-factors 0.95 1.05 \
        --ew-scale-factors 0.95 1.05 \
        --workers "$WORKERS" \
        --seed 42 \
        --resume \
        $( [[ "$LOW_MEMORY" == "1" ]] && echo "--low-memory" ) \
        2>&1 | tee "$city_log"
    
    echo "✓ Graphs generated for $city (log: $city_log)"
done

# Step 2: Compute gravitational work for all cities
echo ""
echo "====================================================================="
echo "[STEP 2] Computing gravitational work for all cities"
echo "====================================================================="

cities_arg=""
for city in $CITIES; do
    cities_arg="$cities_arg $city"
done

work_log="${LOG_DIR}/step2_work.log"
python "${SCRIPT_DIR}/fine_grid_gravitational_work.py" \
    --cities $cities_arg \
    --resume \
    $( [[ "$LOW_MEMORY" == "1" ]] && echo "--low-memory" ) \
    $( [[ "$SAVE_SEGMENTS" == "1" ]] && echo "--save-segments" ) \
    2>&1 | tee "$work_log"

echo "✓ Gravitational work computed (log: $work_log)"

# Summary
echo ""
echo "====================================================================="
echo "PIPELINE COMPLETE"
echo "====================================================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output logs: $LOG_DIR"
echo ""
echo "Results locations (per city):"
for city in $CITIES; do
    result_dir="${PROJECT_ROOT}/data/data_processed/${city}/graphs_fine_grid"
    if [ -d "$result_dir" ]; then
        echo "  $city:"
        echo "    Grafi: ${result_dir}/graph_*.pkl"
        echo "    Stats: ${result_dir}/fine_grid_stats.csv"
        echo "    Lavoro: ${result_dir}/fine_grid_gravitational_work.csv"
    fi
done
echo "====================================================================="
