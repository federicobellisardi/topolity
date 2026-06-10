#!/bin/bash
# Quick validation test for topolity pipeline
# Checks configuration, paths, and imports without running full processing

set -e

PROJECT_ROOT="/home/fbellisardi/code/topolity"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline_production"

echo "====================================================================="
echo "TOPOLITY PIPELINE - QUICK VALIDATION TEST"
echo "====================================================================="

# Test 1: Check required files exist
echo ""
echo "[TEST 1] Checking required files..."
required_files=(
    "dem_extractor_fine_grid.py"
    "fine_grid_gravitational_work.py"
    "gravitational_work.py"
    "utils.py"
    "data_processing.py"
    "run_pipeline.sh"
    "README_PIPELINE.md"
)

for file in "${required_files[@]}"; do
    if [ -f "${PIPELINE_DIR}/${file}" ]; then
        size=$(ls -lh "${PIPELINE_DIR}/${file}" | awk '{print $5}')
        echo "  ✓ ${file} (${size})"
    else
        echo "  ✗ ${file} MISSING!"
        exit 1
    fi
done

# Test 2: Check Python syntax
echo ""
echo "[TEST 2] Validating Python syntax..."
cd "$PIPELINE_DIR"
for pyfile in *.py; do
    if python3 -m py_compile "$pyfile" 2>/dev/null; then
        echo "  ✓ $pyfile"
    else
        echo "  ✗ $pyfile SYNTAX ERROR!"
        exit 1
    fi
done

# Test 3: Check data directories exist
echo ""
echo "[TEST 3] Checking data directories..."
data_root="${PROJECT_ROOT}/data/data_processed"
if [ -d "$data_root" ]; then
    echo "  ✓ Data root exists: $data_root"
    # List available cities
    cities=$(ls -d "$data_root"/*/ 2>/dev/null | xargs -I {} basename {} | tr '\n' ' ')
    if [ -n "$cities" ]; then
        echo "  ✓ Available cities: $cities"
    else
        echo "  ⚠ No city folders found"
    fi
else
    echo "  ✗ Data root missing: $data_root"
fi

# Test 4: Configuration
echo ""
echo "[TEST 4] Checking configuration..."
conf_file="${PROJECT_ROOT}/tools/conf/conf_extractor.json"
if [ -f "$conf_file" ]; then
    echo "  ✓ Config file exists: $conf_file"
else
    echo "  ⚠ Config file not found (optional)"
fi

# Test 5: DEM files
echo ""
echo "[TEST 5] Checking DEM availability..."
dem_root="${PROJECT_ROOT}/data/data_processed"
if [ -d "$dem_root" ]; then
    dem_count=$(find "$dem_root" -name "*_dem.tif" 2>/dev/null | wc -l)
    if [ "$dem_count" -gt 0 ]; then
        echo "  ✓ Found $dem_count DEM files"
        find "$dem_root" -name "*_dem.tif" -exec ls -lh {} \; | awk '{print "    " $5 "\t" $9}' | head -5
    else
        echo "  ⚠ No DEM files found (will be downloaded on first run)"
    fi
fi

# Test 6: Graph directories
echo ""
echo "[TEST 6] Checking graphs directories..."
graphs_dir=$(find "$dem_root" -type d -name "graphs_fine_grid" 2>/dev/null | head -1)
if [ -n "$graphs_dir" ]; then
    graph_count=$(ls "$graphs_dir"/graph_*.pkl 2>/dev/null | wc -l)
    echo "  ✓ Found graphs_fine_grid directory with $graph_count graphs"
else
    echo "  ⚠ No graphs_fine_grid directories found (will be created on first run)"
fi

# Final summary
echo ""
echo "====================================================================="
echo "✓ VALIDATION COMPLETE"
echo "====================================================================="
echo ""
echo "To run the full pipeline, use:"
echo "  cd ${PIPELINE_DIR}"
echo "  ./run_pipeline.sh santiago  # for single city"
echo "  ./run_pipeline.sh santiago madrid barcelona  # for multiple cities"
echo ""
echo "Or run individual steps:"
echo "  conda run -n geo_flow python dem_extractor_fine_grid.py --city santiago --test"
echo "  conda run -n geo_flow python fine_grid_gravitational_work.py --city santiago"
echo ""
